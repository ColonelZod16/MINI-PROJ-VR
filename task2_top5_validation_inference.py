import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from ultralytics import YOLO
import torch


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
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
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


def polygon_xy_to_flat(poly_xy: np.ndarray) -> List[float]:
    if poly_xy is None or len(poly_xy) < 3:
        return []
    return [float(v) for point in poly_xy for v in point]


def resolve_model_path(provided: Optional[str]) -> Path:
    if provided:
        candidate = Path(provided)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Model not found: {candidate}")

    candidates = [
        Path("Runs") / "task2" / "best.pt",
        Path("Runs") / "Task2" / "best.pt",
        Path("Runs") / "task2" / "train" / "weights" / "best.pt",
        Path("Runs") / "Task2" / "train" / "weights" / "best.pt",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find model. Checked: {checked}")


def resolve_device(provided: Optional[str]) -> str:
    if provided and str(provided).strip():
        return str(provided).strip()
    return "0" if torch.cuda.is_available() else "cpu"


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

    auc = float(np.trapz(np.array(tpr, dtype=np.float32), np.array(fpr, dtype=np.float32)))
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


def save_roc_plots(roc_data: Dict[str, dict], out_dir: Path) -> None:
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
    plt.title("YOLO Detection ROC Curves (Validation Top-5)")
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

        with Image.open(image_path) as img:
            width, height = img.size

        present = annotation_target_classes(ann_path)
        if not present:
            continue

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


def run_inference_and_evaluate(
    model: YOLO,
    samples: List[Dict],
    imgsz: int,
    conf: float,
    iou: float,
    batch: int,
    device: str,
) -> Tuple[List[Dict], Dict[str, dict]]:
    image_paths = [str(s["image_path"]) for s in samples]
    chunk_size = max(1, int(batch))

    output_rows: List[Dict] = []
    seg_stats = {name: {"tp": 0, "fp": 0, "fn": 0} for name in CLASS_NAMES}
    det_y_true = {name: [] for name in CLASS_NAMES}
    det_y_score = {name: [] for name in CLASS_NAMES}

    bbox_preds = []
    seg_preds = []

    for start in range(0, len(image_paths), chunk_size):
        chunk = image_paths[start : start + chunk_size]
        chunk_samples = samples[start : start + chunk_size]
        preds = model.predict(
            source=chunk,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            batch=min(batch, len(chunk)),
            verbose=False,
            stream=True,
        )

        for result_idx, result in enumerate(preds):
            if result_idx >= len(chunk_samples):
                continue
            sample = chunk_samples[result_idx]
            image_name = sample["image_path"].name

            width = sample["width"]
            height = sample["height"]

            pred_class_masks = {name: np.zeros((height, width), dtype=np.uint8) for name in CLASS_NAMES}
            pred_max_score = {name: 0.0 for name in CLASS_NAMES}

            detections = []
            boxes = result.boxes
            masks = result.masks

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                cls = boxes.cls.detach().cpu().numpy().astype(int)
                scores = boxes.conf.detach().cpu().numpy()
                polygons = masks.xy if masks is not None and masks.xy is not None else None
                mask_data = masks.data.detach().cpu().numpy() if masks is not None and masks.data is not None else None

                for i in range(len(cls)):
                    class_id = int(cls[i])
                    class_name_raw = str(result.names.get(class_id, class_id))
                    class_name = normalize_name(class_name_raw)
                    if class_name not in CLASS_TO_ID:
                        continue

                    x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                    x1 = max(0.0, min(x1, float(width)))
                    y1 = max(0.0, min(y1, float(height)))
                    x2 = max(0.0, min(x2, float(width)))
                    y2 = max(0.0, min(y2, float(height)))
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    if w <= 0 or h <= 0:
                        continue

                    score = float(scores[i])
                    pred_max_score[class_name] = max(pred_max_score[class_name], score)

                    seg_poly = []
                    seg_rle = None
                    if polygons is not None and i < len(polygons):
                        poly = polygons[i]
                        if poly is not None and len(poly) >= 3:
                            flat = polygon_xy_to_flat(poly)
                            if len(flat) >= 6:
                                seg_poly = [flat]

                    if mask_data is not None and i < len(mask_data):
                        raw_mask = (mask_data[i] > 0.5).astype(np.uint8)
                        mask_img = Image.fromarray(raw_mask * 255)
                        mask_img = mask_img.resize((width, height), Image.NEAREST)
                        pred_mask = (np.array(mask_img, dtype=np.uint8) > 127).astype(np.uint8)
                    elif seg_poly:
                        pred_mask = rasterize_polygons(width, height, seg_poly)
                    else:
                        pred_mask = np.zeros((height, width), dtype=np.uint8)

                    pred_class_masks[class_name] = np.maximum(pred_class_masks[class_name], pred_mask)

                    if int(pred_mask.sum()) > 0:
                        pred_mask_fortran = np.asfortranarray(pred_mask)
                        seg_rle = mask_utils.encode(pred_mask_fortran)
                        seg_rle["counts"] = seg_rle["counts"].decode("utf-8")

                    bbox_preds.append(
                        {
                            "image_id": sample["image_id"],
                            "category_id": COCO_CAT_ID[class_name],
                            "bbox": [x1, y1, w, h],
                            "score": score,
                        }
                    )

                    seg_preds.append(
                        {
                            "image_id": sample["image_id"],
                            "category_id": COCO_CAT_ID[class_name],
                            "segmentation": seg_rle,
                            "score": score,
                        }
                    )

                    detections.append(
                        {
                            "class_id": class_id,
                            "class_name": class_name,
                            "score": score,
                            "bbox_xyxy": [x1, y1, x2, y2],
                            "segmentation_polygon_xy": seg_poly,
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
                    "image_name": image_name,
                    "image_path": str(sample["image_path"]),
                    "annotation_path": str(sample["ann_path"]),
                    "target_classes_present": sample["target_classes_present"],
                    "num_detections": len(detections),
                    "detections": detections,
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
        "info": {"description": "Top-5 validation GT"},
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
    save_roc_plots(roc_data, roc_plot_dir)

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
        file.write("Top-5 Validation Inference Metrics Report\n")
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Task2 YOLO inference on validation images containing top-5 classes only"
    )
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--images_dir", default="validation/validation/image")
    parser.add_argument("--annos_dir", default="validation/validation/annos")
    parser.add_argument("--output_dir", default="Runs/task2/top5_inference")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path)
    images_dir = Path(args.images_dir)
    annos_dir = Path(args.annos_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not annos_dir.exists():
        raise FileNotFoundError(f"Annotations dir not found: {annos_dir}")

    print(f"[INFO] Model: {model_path}")
    print(f"[INFO] Images dir: {images_dir}")
    print(f"[INFO] Annos dir: {annos_dir}")

    samples = collect_target_samples(images_dir, annos_dir, args.max_images)
    if not samples:
        raise RuntimeError("No validation images matched target classes")

    print(f"[INFO] Filtered images (top-5 categories): {len(samples)}")

    model = YOLO(str(model_path))
    device = resolve_device(args.device)
    outputs, metrics_data = run_inference_and_evaluate(
        model=model,
        samples=samples,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        batch=args.batch,
        device=device,
    )

    metrics_summary = evaluate_and_save_metrics(samples, metrics_data, output_dir)

    predictions_path = output_dir / "top5_validation_predictions.json"
    with open(predictions_path, "w", encoding="utf-8") as file:
        json.dump(outputs, file, indent=2)

    filtered_list_path = output_dir / "top5_filtered_images.txt"
    with open(filtered_list_path, "w", encoding="utf-8") as file:
        for s in samples:
            file.write(f"{s['image_path']}\n")

    class_counts: Dict[str, int] = {name: 0 for name in sorted(TARGET_CLASSES)}
    for s in samples:
        for class_name in s["target_classes_present"]:
            class_counts[class_name] += 1

    summary = {
        "model_path": str(model_path),
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

    print("[DONE] Top-5 validation inference complete")
    print(f"Predictions: {predictions_path}")
    print(f"Summary    : {summary_path}")
    print(f"Image list : {filtered_list_path}")
    print(f"Metrics txt: {output_dir / 'top5_inference_metrics_report.txt'}")


if __name__ == "__main__":
    main()
