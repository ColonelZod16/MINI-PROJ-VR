import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy import ndimage as ndi
import torch
import torch.nn as nn
from torchvision import transforms as T


TARGET_CLASSES = {
    "long sleeve top",
    "short sleeve top",
    "shorts",
    "skirt",
    "trousers",
}
ITEM_KEYS = ["item1", "item2", "item3", "item4"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
CLASS_NAMES = [
    "long sleeve top",
    "short sleeve top",
    "shorts",
    "skirt",
    "trousers",
]
CLASS_TO_ID = {name: idx + 1 for idx, name in enumerate(CLASS_NAMES)}
COCO_CAT_ID = {name: idx + 1 for idx, name in enumerate(CLASS_NAMES)}


def normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("_", " ")


def rasterize_polygons(width: int, height: int, polygons: List[List[float]]) -> np.ndarray:
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in polygons:
        if isinstance(poly, list) and len(poly) >= 6:
            points = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly) - 1, 2)]
            draw.polygon(points, fill=1)
    return np.array(mask_img, dtype=np.uint8)


def resolve_unet_checkpoint_path(provided: Optional[str]) -> Path:
    if provided:
        candidate = Path(provided)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"U-Net checkpoint not found: {candidate}")

    candidates = [
        Path("best_model.pth"),
        Path("Runs") / "unet" / "best_model.pth",
        Path("Runs") / "UNet" / "best_model.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find U-Net best_model.pth. Checked: {checked}")


def resolve_yolo_summary_path(provided: Optional[str]) -> Optional[Path]:
    if provided:
        path = Path(provided)
        if path.exists():
            return path
        raise FileNotFoundError(f"YOLO summary not found: {path}")

    candidates = [
        Path("Runs") / "task2" / "top5_inference" / "top5_inference_summary.json",
        Path("Report") / "task2_yolo" / "task2_yolo_validation_summary.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_device(provided: Optional[str]) -> str:
    if provided and str(provided).strip():
        return str(provided).strip()
    return "cuda" if torch.cuda.is_available() else "cpu"


def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module."):
            new_key = new_key[len("module.") :]
        if new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod.") :]
        cleaned[new_key] = value
    return cleaned


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 6):
        super().__init__()
        f = [64, 128, 256, 512]
        self.enc1 = DoubleConv(in_channels, f[0])
        self.enc2 = DoubleConv(f[0], f[1])
        self.enc3 = DoubleConv(f[1], f[2])
        self.enc4 = DoubleConv(f[2], f[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(f[3], f[3] * 2)
        self.up4 = nn.ConvTranspose2d(f[3] * 2, f[3], 2, 2)
        self.dec4 = DoubleConv(f[3] * 2, f[3])
        self.up3 = nn.ConvTranspose2d(f[3], f[2], 2, 2)
        self.dec3 = DoubleConv(f[2] * 2, f[2])
        self.up2 = nn.ConvTranspose2d(f[2], f[1], 2, 2)
        self.dec2 = DoubleConv(f[1] * 2, f[1])
        self.up1 = nn.ConvTranspose2d(f[1], f[0], 2, 2)
        self.dec1 = DoubleConv(f[0] * 2, f[0])
        self.out = nn.Conv2d(f[0], num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class SmallUNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bridge = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.out = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        b = self.bridge(self.pool2(d2))
        u2 = self.up2(b)
        c2 = self.dec2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(c2)
        c1 = self.dec1(torch.cat([u1, d1], dim=1))
        return self.out(c1)


def detect_checkpoint_family(state_dict: Dict[str, torch.Tensor]) -> str:
    keys = list(state_dict.keys())
    if any(key.startswith("down1.") for key in keys):
        return "small_unet"
    if any(key.startswith("enc1.") for key in keys):
        return "simple_unet"
    if any(key.startswith("encoder.") for key in keys) or any(key.startswith("segmentation_head.") for key in keys):
        return "smp_unet"
    return "unknown"


def detect_smp_encoder_hint(state_dict: Dict[str, torch.Tensor]) -> Optional[str]:
    keys = set(state_dict.keys())
    if "encoder.layer4.2.conv1.weight" in keys or any(key.startswith("encoder.layer3.5") for key in keys):
        return "resnet34"
    if "encoder.layer4.1.conv1.weight" in keys:
        return "resnet18"
    return None


def infer_num_classes(state_dict: Dict[str, torch.Tensor], fallback: int = 6) -> int:
    candidates = [
        "segmentation_head.0.weight",
        "out.weight",
        "final_conv.weight",
    ]
    for key in candidates:
        if key in state_dict and state_dict[key].ndim >= 1:
            return int(state_dict[key].shape[0])
    return fallback


def build_smp_unet(encoder_name: str, num_classes: int) -> nn.Module:
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=num_classes,
    )


def load_unet_model(ckpt_path: Path, device: torch.device) -> Tuple[nn.Module, dict]:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format for U-Net")

    state_dict = normalize_state_dict_keys(state_dict)
    family = detect_checkpoint_family(state_dict)
    num_classes = infer_num_classes(state_dict)

    errors = []

    if family == "small_unet":
        model = SmallUNet(num_classes=num_classes)
        model.load_state_dict(state_dict, strict=True)
        return model.to(device).eval(), {"family": family, "num_classes": num_classes}

    if family == "simple_unet":
        model = SimpleUNet(in_channels=3, num_classes=num_classes)
        model.load_state_dict(state_dict, strict=True)
        return model.to(device).eval(), {"family": family, "num_classes": num_classes}

    if family == "smp_unet":
        try:
            import segmentation_models_pytorch  # noqa: F401
        except Exception as exc:
            raise RuntimeError(
                "Checkpoint appears to be SMP U-Net but segmentation-models-pytorch is unavailable"
            ) from exc

        encoder_hint = detect_smp_encoder_hint(state_dict)
        candidates = []
        if encoder_hint:
            candidates.append(encoder_hint)
        for e in ["resnet34", "resnet18", "resnet50"]:
            if e not in candidates:
                candidates.append(e)

        for encoder_name in candidates:
            try:
                model = build_smp_unet(encoder_name=encoder_name, num_classes=num_classes)
                model.load_state_dict(state_dict, strict=True)
                return model.to(device).eval(), {
                    "family": family,
                    "encoder": encoder_name,
                    "num_classes": num_classes,
                }
            except Exception as exc:
                errors.append(f"{encoder_name}: {exc}")

    raise RuntimeError("Could not reconstruct U-Net from checkpoint. Attempts: " + " | ".join(errors))


def annotation_target_classes(anno_path: Path) -> Set[str]:
    with open(anno_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    present = set()
    for key in ITEM_KEYS:
        item = data.get(key)
        if not isinstance(item, dict):
            continue
        cat = normalize_name(item.get("category_name", ""))
        if cat in TARGET_CLASSES:
            present.add(cat)
    return present


def parse_gt_annotation(
    ann_path: Path,
    width: int,
    height: int,
) -> Tuple[List[dict], Dict[str, np.ndarray], Dict[str, int]]:
    with open(ann_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    instances = []
    class_masks = {name: np.zeros((height, width), dtype=np.uint8) for name in CLASS_NAMES}
    class_presence = {name: 0 for name in CLASS_NAMES}

    for key in ITEM_KEYS:
        item = data.get(key)
        if not isinstance(item, dict):
            continue

        category_name = normalize_name(item.get("category_name", ""))
        if category_name not in CLASS_TO_ID:
            continue

        bb = item.get("bounding_box")
        if not bb or len(bb) < 4:
            continue

        x1, y1, x2, y2 = [float(v) for v in bb[:4]]
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(width), x2), min(float(height), y2)
        if x2 <= x1 or y2 <= y1:
            continue

        polygons = item.get("segmentation") or []
        inst_mask = rasterize_polygons(width, height, polygons)
        class_masks[category_name] = np.maximum(class_masks[category_name], inst_mask)
        class_presence[category_name] = 1

        instances.append(
            {
                "category_name": category_name,
                "bbox_xyxy": [x1, y1, x2, y2],
                "segmentation": polygons,
            }
        )

    return instances, class_masks, class_presence


def compute_roc_auc(y_true: List[int], y_score: List[float]) -> Tuple[List[float], List[float], List[float], float]:
    y_true_np = np.array(y_true, dtype=np.int32)
    y_score_np = np.array(y_score, dtype=np.float32)

    positives = int((y_true_np == 1).sum())
    negatives = int((y_true_np == 0).sum())

    if positives == 0 or negatives == 0:
        return [0.0, 1.0], [0.0, 1.0], [float("inf"), 0.0], float("nan")

    order = np.argsort(-y_score_np)
    y_sorted = y_true_np[order]
    s_sorted = y_score_np[order]

    tp = 0
    fp = 0
    tpr = [0.0]
    fpr = [0.0]
    thresholds = [float("inf")]

    for idx in range(len(y_sorted)):
        if y_sorted[idx] == 1:
            tp += 1
        else:
            fp += 1

        is_last = idx == len(y_sorted) - 1
        score_changed = (not is_last) and (s_sorted[idx] != s_sorted[idx + 1])
        if is_last or score_changed:
            tpr.append(tp / positives)
            fpr.append(fp / negatives)
            thresholds.append(float(s_sorted[idx]))

    auc = float(np.trapezoid(np.array(tpr, dtype=np.float32), np.array(fpr, dtype=np.float32)))
    return fpr, tpr, thresholds, auc


def best_f1_from_scores(y_true: List[int], y_score: List[float]) -> Tuple[float, float, float, float]:
    y_true_np = np.array(y_true, dtype=np.int32)
    y_score_np = np.array(y_score, dtype=np.float32)

    candidates = np.unique(y_score_np)
    candidates = np.concatenate([np.array([0.0, 1.0], dtype=np.float32), candidates])

    best_thr = 0.5
    best_f1 = -1.0
    best_p = 0.0
    best_r = 0.0

    for thr in candidates:
        y_pred = (y_score_np >= thr).astype(np.int32)
        tp = int(np.logical_and(y_pred == 1, y_true_np == 1).sum())
        fp = int(np.logical_and(y_pred == 1, y_true_np == 0).sum())
        fn = int(np.logical_and(y_pred == 0, y_true_np == 1).sum())

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        if f1 > best_f1:
            best_f1 = float(f1)
            best_thr = float(thr)
            best_p = float(precision)
            best_r = float(recall)

    return best_thr, best_p, best_r, best_f1


def save_roc_plots(roc_data: Dict[str, dict], out_dir: Path, title_prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for class_name in CLASS_NAMES:
        r = roc_data[class_name]
        label = f"{class_name} (AUC={r['auc']:.3f})" if not np.isnan(r["auc"]) else f"{class_name} (AUC=n/a)"
        plt.plot(r["fpr"], r["tpr"], linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title_prefix} ROC Curves (Validation Top-5)")
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "roc_all_classes.png", dpi=180)
    plt.close()

    for class_name in CLASS_NAMES:
        r = roc_data[class_name]
        auc_text = f"{r['auc']:.4f}" if not np.isnan(r["auc"]) else "n/a"
        plt.figure(figsize=(6, 5))
        plt.plot(r["fpr"], r["tpr"], linewidth=2, label=f"AUC={auc_text}")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC - {class_name}")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"roc_{class_name.replace(' ', '_')}.png", dpi=180)
        plt.close()


def collect_target_samples(images_dir: Path, annos_dir: Path, max_images: Optional[int]) -> List[Dict]:
    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    samples = []
    for image_path in image_paths:
        ann_path = annos_dir / f"{image_path.stem}.json"
        if not ann_path.exists():
            continue

        present = annotation_target_classes(ann_path)
        if not present:
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        samples.append(
            {
                "image_id": len(samples) + 1,
                "image_path": image_path,
                "ann_path": ann_path,
                "target_classes_present": sorted(list(present)),
                "width": width,
                "height": height,
            }
        )

        if max_images is not None and len(samples) >= max_images:
            break

    return samples


def preprocess_image(image_path: Path, img_size: int) -> torch.Tensor:
    tf = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    with Image.open(image_path) as image:
        image_rgb = image.convert("RGB")
        tensor = tf(image_rgb)
    return tensor.unsqueeze(0)


def extract_unet_instances(
    class_map: np.ndarray,
    prob_map_per_class: Dict[str, np.ndarray],
    min_component_area: int,
) -> Tuple[List[dict], Dict[str, np.ndarray], Dict[str, float]]:
    instances = []
    pred_class_masks = {name: np.zeros_like(class_map, dtype=np.uint8) for name in CLASS_NAMES}
    pred_max_score = {name: 0.0 for name in CLASS_NAMES}

    for class_name, class_id in CLASS_TO_ID.items():
        binary = (class_map == class_id).astype(np.uint8)
        if int(binary.sum()) == 0:
            continue

        labeled, n_components = ndi.label(binary)
        for comp_id in range(1, n_components + 1):
            comp_mask = labeled == comp_id
            area = int(comp_mask.sum())
            if area < min_component_area:
                continue

            rows = np.where(comp_mask.any(axis=1))[0]
            cols = np.where(comp_mask.any(axis=0))[0]
            if len(rows) == 0 or len(cols) == 0:
                continue

            y1, y2 = int(rows[0]), int(rows[-1])
            x1, x2 = int(cols[0]), int(cols[-1])

            score = float(prob_map_per_class[class_name][comp_mask].mean()) if area > 0 else 0.0
            pred_max_score[class_name] = max(pred_max_score[class_name], score)

            pred_mask_u8 = comp_mask.astype(np.uint8)
            pred_class_masks[class_name] = np.maximum(pred_class_masks[class_name], pred_mask_u8)

            instances.append(
                {
                    "category_name": class_name,
                    "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "mask": pred_mask_u8,
                    "score": score,
                }
            )

    return instances, pred_class_masks, pred_max_score


def run_unet_inference_and_evaluate(
    model: nn.Module,
    samples: List[Dict],
    img_size: int,
    device: torch.device,
    min_component_area: int,
) -> Tuple[List[Dict], Dict[str, dict]]:
    output_rows: List[Dict] = []
    seg_stats = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASS_NAMES}
    det_y_true = {name: [] for name in CLASS_NAMES}
    det_y_score = {name: [] for name in CLASS_NAMES}

    bbox_preds = []
    seg_preds = []

    model.eval()
    with torch.no_grad():
        for sample in samples:
            image_path = sample["image_path"]
            width = sample["width"]
            height = sample["height"]

            x = preprocess_image(image_path, img_size=img_size).to(device)
            logits = model(x)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]

            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1, keepdim=True).float()

            pred_up = torch.nn.functional.interpolate(pred, size=(height, width), mode="nearest")
            class_map = pred_up[0, 0].cpu().numpy().astype(np.int32)

            probs_up = torch.nn.functional.interpolate(probs, size=(height, width), mode="bilinear", align_corners=False)
            probs_np = probs_up[0].cpu().numpy()
            prob_map_per_class = {
                class_name: probs_np[class_id] for class_name, class_id in CLASS_TO_ID.items()
            }

            pred_instances, pred_class_masks, pred_max_score = extract_unet_instances(
                class_map=class_map,
                prob_map_per_class=prob_map_per_class,
                min_component_area=min_component_area,
            )

            for inst in pred_instances:
                class_name = inst["category_name"]
                x1, y1, x2, y2 = inst["bbox_xyxy"]
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w <= 0 or h <= 0:
                    continue

                score = float(inst["score"])
                bbox_preds.append(
                    {
                        "image_id": sample["image_id"],
                        "category_id": COCO_CAT_ID[class_name],
                        "bbox": [x1, y1, w, h],
                        "score": score,
                    }
                )

                pred_mask = inst["mask"]
                seg_rle = None
                if int(pred_mask.sum()) > 0:
                    pred_mask_fortran = np.asfortranarray(pred_mask)
                    seg_rle = mask_utils.encode(pred_mask_fortran)
                    seg_rle["counts"] = seg_rle["counts"].decode("utf-8")

                seg_preds.append(
                    {
                        "image_id": sample["image_id"],
                        "category_id": COCO_CAT_ID[class_name],
                        "segmentation": seg_rle,
                        "score": score,
                    }
                )

            _, gt_class_masks, gt_presence = parse_gt_annotation(sample["ann_path"], width, height)

            for class_name in CLASS_NAMES:
                gt_mask = gt_class_masks[class_name] > 0
                pd_mask = pred_class_masks[class_name] > 0

                tp = int(np.logical_and(pd_mask, gt_mask).sum())
                fp = int(np.logical_and(pd_mask, np.logical_not(gt_mask)).sum())
                fn = int(np.logical_and(np.logical_not(pd_mask), gt_mask).sum())

                seg_stats[class_name]["tp"] += tp
                seg_stats[class_name]["fp"] += fp
                seg_stats[class_name]["fn"] += fn

                det_y_true[class_name].append(int(gt_presence[class_name]))
                det_y_score[class_name].append(float(pred_max_score[class_name]))

            output_rows.append(
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "annotation_path": str(sample["ann_path"]),
                    "target_classes_present": sample["target_classes_present"],
                    "num_detections": len(pred_instances),
                    "detections": [
                        {
                            "class_name": inst["category_name"],
                            "score": float(inst["score"]),
                            "bbox_xyxy": [float(v) for v in inst["bbox_xyxy"]],
                        }
                        for inst in pred_instances
                    ],
                }
            )

    metrics = {
        "seg_stats": seg_stats,
        "det_y_true": det_y_true,
        "det_y_score": det_y_score,
        "bbox_preds": bbox_preds,
        "seg_preds": seg_preds,
    }
    return output_rows, metrics


def evaluate_and_save_metrics(samples: List[Dict], metrics_data: Dict[str, dict], output_dir: Path) -> Dict:
    seg_stats = metrics_data["seg_stats"]
    det_y_true = metrics_data["det_y_true"]
    det_y_score = metrics_data["det_y_score"]
    bbox_preds = metrics_data["bbox_preds"]
    seg_preds = metrics_data["seg_preds"]

    gt_coco = {
        "info": {"description": "Top-5 validation GT (U-Net evaluator)"},
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(CLASS_NAMES)],
    }

    ann_id = 1
    for sample in samples:
        gt_coco["images"].append(
            {
                "id": sample["image_id"],
                "file_name": sample["image_path"].name,
                "width": sample["width"],
                "height": sample["height"],
            }
        )

        gt_instances, _, _ = parse_gt_annotation(sample["ann_path"], sample["width"], sample["height"])
        for inst in gt_instances:
            x1, y1, x2, y2 = inst["bbox_xyxy"]
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue

            segs = []
            for poly in inst["segmentation"]:
                if isinstance(poly, list) and len(poly) >= 6:
                    segs.append([float(v) for v in poly])

            gt_coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": sample["image_id"],
                    "category_id": COCO_CAT_ID[inst["category_name"]],
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "segmentation": segs,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    gt_coco_path = output_dir / "gt_coco_top5.json"
    with open(gt_coco_path, "w", encoding="utf-8") as file:
        json.dump(gt_coco, file)

    bbox_pred_path = output_dir / "pred_bbox_coco_top5.json"
    seg_pred_path = output_dir / "pred_seg_coco_top5.json"
    with open(bbox_pred_path, "w", encoding="utf-8") as file:
        json.dump(bbox_preds, file)
    with open(seg_pred_path, "w", encoding="utf-8") as file:
        json.dump(seg_preds, file)

    coco_gt_api = COCO(str(gt_coco_path))

    bbox_eval_summary = {"mAP_50_95": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}
    if len(bbox_preds) > 0:
        coco_dt_bbox = coco_gt_api.loadRes(str(bbox_pred_path))
        coco_eval_bbox = COCOeval(coco_gt_api, coco_dt_bbox, "bbox")
        coco_eval_bbox.evaluate()
        coco_eval_bbox.accumulate()
        coco_eval_bbox.summarize()
        bbox_eval_summary = {
            "mAP_50_95": float(coco_eval_bbox.stats[0]),
            "mAP_50": float(coco_eval_bbox.stats[1]),
            "mAP_75": float(coco_eval_bbox.stats[2]),
        }

    seg_eval_summary = {"mAP_50_95": 0.0, "mAP_50": 0.0, "mAP_75": 0.0}
    seg_eval_error = None
    seg_preds_non_empty = [p for p in seg_preds if p["segmentation"]]
    if len(seg_preds_non_empty) > 0:
        seg_non_empty_path = output_dir / "pred_seg_coco_top5_non_empty.json"
        with open(seg_non_empty_path, "w", encoding="utf-8") as file:
            json.dump(seg_preds_non_empty, file)
        try:
            coco_dt_seg = coco_gt_api.loadRes(str(seg_non_empty_path))
            coco_eval_seg = COCOeval(coco_gt_api, coco_dt_seg, "segm")
            coco_eval_seg.evaluate()
            coco_eval_seg.accumulate()
            coco_eval_seg.summarize()
            seg_eval_summary = {
                "mAP_50_95": float(coco_eval_seg.stats[0]),
                "mAP_50": float(coco_eval_seg.stats[1]),
                "mAP_75": float(coco_eval_seg.stats[2]),
            }
        except Exception as exc:
            seg_eval_error = str(exc)

    seg_per_class = {}
    iou_values = []
    dice_values = []
    for class_name in CLASS_NAMES:
        tp = seg_stats[class_name]["tp"]
        fp = seg_stats[class_name]["fp"]
        fn = seg_stats[class_name]["fn"]

        iou = tp / (tp + fp + fn + 1e-8)
        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
        seg_per_class[class_name] = {
            "iou": float(iou),
            "dice": float(dice),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }
        iou_values.append(iou)
        dice_values.append(dice)

    seg_macro = {
        "mIoU_macro": float(np.mean(iou_values)),
        "dice_macro": float(np.mean(dice_values)),
    }

    roc_data = {}
    det_per_class = {}
    roc_rows = []

    for class_name in CLASS_NAMES:
        y_true = det_y_true[class_name]
        y_score = det_y_score[class_name]

        fpr, tpr, thresholds, auc = compute_roc_auc(y_true, y_score)
        best_thr, best_p, best_r, best_f1 = best_f1_from_scores(y_true, y_score)

        roc_data[class_name] = {
            "fpr": fpr,
            "tpr": tpr,
            "thresholds": thresholds,
            "auc": auc,
        }
        det_per_class[class_name] = {
            "auc": None if np.isnan(auc) else float(auc),
            "best_threshold": float(best_thr),
            "precision": float(best_p),
            "recall": float(best_r),
            "f1": float(best_f1),
        }

        for idx, (fpr_v, tpr_v, thr_v) in enumerate(zip(fpr, tpr, thresholds)):
            roc_rows.append(
                {
                    "class_name": class_name,
                    "point_index": idx,
                    "fpr": float(fpr_v),
                    "tpr": float(tpr_v),
                    "threshold": float(thr_v),
                }
            )

    auc_values = [v["auc"] for v in det_per_class.values() if v["auc"] is not None]
    f1_values = [v["f1"] for v in det_per_class.values()]
    det_macro = {
        "auc_macro": float(np.mean(auc_values)) if auc_values else None,
        "f1_macro": float(np.mean(f1_values)) if f1_values else None,
    }

    roc_csv_path = output_dir / "top5_detection_roc_points.csv"
    with open(roc_csv_path, "w", encoding="utf-8") as file:
        file.write("class_name,point_index,fpr,tpr,threshold\n")
        for row in roc_rows:
            file.write(
                f"{row['class_name']},{row['point_index']},{row['fpr']},{row['tpr']},{row['threshold']}\n"
            )

    roc_plot_dir = output_dir / "top5_roc_plots"
    save_roc_plots(roc_data, roc_plot_dir, title_prefix="U-Net Detection")

    metrics_summary = {
        "segmentation": {
            "per_class": seg_per_class,
            "macro": seg_macro,
            "coco_segm": seg_eval_summary,
            "coco_segm_error": seg_eval_error,
        },
        "detection": {
            "coco_bbox": bbox_eval_summary,
            "macro": det_macro,
            "per_class": det_per_class,
        },
        "artifacts": {
            "gt_coco": str(gt_coco_path),
            "pred_bbox": str(bbox_pred_path),
            "pred_seg": str(seg_pred_path),
            "roc_points": str(roc_csv_path),
            "roc_plot_dir": str(roc_plot_dir),
        },
    }

    report_txt_path = output_dir / "top5_inference_metrics_report.txt"
    with open(report_txt_path, "w", encoding="utf-8") as file:
        file.write("=" * 90 + "\n")
        file.write("Top-5 Validation Inference Metrics Report (U-Net)\n")
        file.write("=" * 90 + "\n")
        file.write(f"Images evaluated: {len(samples)}\n\n")

        file.write("[Segmentation]\n")
        file.write(f"Macro mIoU:  {seg_macro['mIoU_macro']:.6f}\n")
        file.write(f"Macro Dice:  {seg_macro['dice_macro']:.6f}\n")
        file.write(f"COCO Seg mAP@[0.5:0.95]: {seg_eval_summary['mAP_50_95']:.6f}\n")
        file.write(f"COCO Seg mAP@0.5:        {seg_eval_summary['mAP_50']:.6f}\n\n")
        file.write(f"{'Class':<22} {'IoU':>10} {'Dice':>10}\n")
        for class_name in CLASS_NAMES:
            c = seg_per_class[class_name]
            file.write(f"{class_name:<22} {c['iou']:>10.6f} {c['dice']:>10.6f}\n")

        file.write("\n[Detection]\n")
        file.write(f"COCO BBox mAP@[0.5:0.95]: {bbox_eval_summary['mAP_50_95']:.6f}\n")
        file.write(f"COCO BBox mAP@0.5:        {bbox_eval_summary['mAP_50']:.6f}\n")
        file.write(f"COCO BBox mAP@0.75:       {bbox_eval_summary['mAP_75']:.6f}\n")
        if det_macro["auc_macro"] is None:
            file.write("Macro AUC (detection):      n/a\n")
        else:
            file.write(f"Macro AUC (detection):      {det_macro['auc_macro']:.6f}\n")
        if det_macro["f1_macro"] is None:
            file.write("Macro F1 (detection):       n/a\n\n")
        else:
            file.write(f"Macro F1 (detection):       {det_macro['f1_macro']:.6f}\n\n")
        file.write(f"{'Class':<22} {'AUC':>10} {'F1':>10} {'P':>10} {'R':>10} {'Thr':>10}\n")
        for class_name in CLASS_NAMES:
            c = det_per_class[class_name]
            auc_text = "n/a" if c["auc"] is None else f"{c['auc']:.6f}"
            file.write(
                f"{class_name:<22} {auc_text:>10} {c['f1']:>10.6f} {c['precision']:>10.6f} "
                f"{c['recall']:>10.6f} {c['best_threshold']:>10.6f}\n"
            )

    metrics_summary["artifacts"]["metrics_report_txt"] = str(report_txt_path)
    return metrics_summary


def _extract_metrics_block(summary: Dict) -> Dict:
    root = summary.get("metrics", summary)

    segmentation = root.get("segmentation", {})
    detection = root.get("detection", {})

    seg_macro = segmentation.get("macro", {})
    det_macro = detection.get("macro", {})
    det_coco = detection.get("coco_bbox", {})
    seg_coco = segmentation.get("coco_segm", {})

    return {
        "bbox_map_50_95": det_coco.get("mAP_50_95"),
        "bbox_map_50": det_coco.get("mAP_50"),
        "seg_map_50_95": seg_coco.get("mAP_50_95"),
        "seg_map_50": seg_coco.get("mAP_50"),
        "seg_miou_macro": seg_macro.get("mIoU_macro"),
        "seg_dice_macro": seg_macro.get("dice_macro"),
        "det_auc_macro": det_macro.get("auc_macro"),
        "det_f1_macro": det_macro.get("f1_macro"),
    }


def _safe_diff(a, b):
    if a is None or b is None:
        return None
    return float(a) - float(b)


def build_unet_vs_yolo_comparison(
    unet_summary_path: Path,
    yolo_summary_path: Optional[Path],
    output_dir: Path,
) -> Optional[Path]:
    if yolo_summary_path is None or not yolo_summary_path.exists():
        return None

    with open(unet_summary_path, "r", encoding="utf-8") as file:
        unet_summary = json.load(file)
    with open(yolo_summary_path, "r", encoding="utf-8") as file:
        yolo_summary = json.load(file)

    unet_metrics = _extract_metrics_block(unet_summary)
    yolo_metrics = _extract_metrics_block(yolo_summary)

    delta = {
        key: _safe_diff(unet_metrics.get(key), yolo_metrics.get(key))
        for key in sorted(set(unet_metrics.keys()) | set(yolo_metrics.keys()))
    }

    comparison = {
        "unet_summary": str(unet_summary_path),
        "yolo_summary": str(yolo_summary_path),
        "unet": unet_metrics,
        "yolo": yolo_metrics,
        "delta_unet_minus_yolo": delta,
    }

    comparison_json = output_dir / "unet_vs_yolo_comparison.json"
    with open(comparison_json, "w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)

    ordered_keys = [
        "bbox_map_50_95",
        "bbox_map_50",
        "seg_map_50_95",
        "seg_map_50",
        "seg_miou_macro",
        "seg_dice_macro",
        "det_auc_macro",
        "det_f1_macro",
    ]
    comparison_txt = output_dir / "unet_vs_yolo_comparison.txt"
    with open(comparison_txt, "w", encoding="utf-8") as file:
        file.write("=" * 96 + "\n")
        file.write("U-Net vs YOLO (Top-5 Validation)\n")
        file.write("=" * 96 + "\n")
        file.write(f"U-Net summary: {unet_summary_path}\n")
        file.write(f"YOLO summary : {yolo_summary_path}\n\n")
        file.write(f"{'Metric':<24} {'U-Net':>12} {'YOLO':>12} {'Δ(U-Net-YOLO)':>16}\n")
        for key in ordered_keys:
            u = unet_metrics.get(key)
            y = yolo_metrics.get(key)
            d = delta.get(key)
            u_txt = "n/a" if u is None else f"{float(u):.6f}"
            y_txt = "n/a" if y is None else f"{float(y):.6f}"
            d_txt = "n/a" if d is None else f"{float(d):+.6f}"
            file.write(f"{key:<24} {u_txt:>12} {y_txt:>12} {d_txt:>16}\n")

    return comparison_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run U-Net best_model.pth validation on top-5 samples and compare with YOLO"
    )
    parser.add_argument("--model_path", default=None, help="Path to U-Net checkpoint (.pth)")
    parser.add_argument("--images_dir", default="validation/validation/image")
    parser.add_argument("--annos_dir", default="validation/validation/annos")
    parser.add_argument("--output_dir", default="Report/task2_unet_top5")
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--min_component_area", type=int, default=300)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--yolo_summary", default=None, help="Optional YOLO summary JSON path")
    args = parser.parse_args()

    ckpt_path = resolve_unet_checkpoint_path(args.model_path)
    images_dir = Path(args.images_dir)
    annos_dir = Path(args.annos_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not annos_dir.exists():
        raise FileNotFoundError(f"Annotations dir not found: {annos_dir}")

    device_name = resolve_device(args.device)
    device = torch.device(device_name)

    print(f"[INFO] U-Net checkpoint: {ckpt_path}")
    print(f"[INFO] Validation images: {images_dir}")
    print(f"[INFO] Validation annos : {annos_dir}")
    print(f"[INFO] Device          : {device}")

    model, model_meta = load_unet_model(ckpt_path, device)
    print(f"[INFO] Loaded model meta: {model_meta}")

    samples = collect_target_samples(images_dir, annos_dir, args.max_images)
    if not samples:
        raise RuntimeError("No validation images matched top-5 target classes")
    print(f"[INFO] Filtered top-5 validation images: {len(samples)}")

    outputs, metrics_data = run_unet_inference_and_evaluate(
        model=model,
        samples=samples,
        img_size=args.img_size,
        device=device,
        min_component_area=args.min_component_area,
    )

    metrics_summary = evaluate_and_save_metrics(samples, metrics_data, output_dir)

    predictions_path = output_dir / "top5_validation_predictions.json"
    with open(predictions_path, "w", encoding="utf-8") as file:
        json.dump(outputs, file, indent=2)

    filtered_list_path = output_dir / "top5_filtered_images.txt"
    with open(filtered_list_path, "w", encoding="utf-8") as file:
        for sample in samples:
            file.write(f"{sample['image_path']}\n")

    class_counts: Dict[str, int] = {name: 0 for name in sorted(TARGET_CLASSES)}
    for sample in samples:
        for class_name in sample["target_classes_present"]:
            class_counts[class_name] += 1

    summary = {
        "model_path": str(ckpt_path),
        "model_meta": model_meta,
        "images_dir": str(images_dir),
        "annos_dir": str(annos_dir),
        "output_dir": str(output_dir),
        "target_classes": sorted(TARGET_CLASSES),
        "num_filtered_images": len(samples),
        "num_predicted_images": len(outputs),
        "class_image_counts": class_counts,
        "metrics": metrics_summary,
        "artifacts": {
            "predictions_json": str(predictions_path),
            "filtered_images_txt": str(filtered_list_path),
        },
    }

    summary_path = output_dir / "top5_inference_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    yolo_summary_path = resolve_yolo_summary_path(args.yolo_summary)
    comparison_path = build_unet_vs_yolo_comparison(summary_path, yolo_summary_path, output_dir)

    print("[DONE] U-Net Top-5 validation complete")
    print(f"Summary     : {summary_path}")
    print(f"Metrics txt : {output_dir / 'top5_inference_metrics_report.txt'}")
    print(f"ROC plots   : {output_dir / 'top5_roc_plots'}")
    if comparison_path:
        print(f"Comparison  : {comparison_path}")
    else:
        print("Comparison  : skipped (YOLO summary not found)")


if __name__ == "__main__":
    main()
