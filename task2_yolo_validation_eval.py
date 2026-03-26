import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
import torch


CLASS_NAMES = [
    "long sleeve top",
    "short sleeve top",
    "shorts",
    "skirt",
    "trousers",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
COCO_CAT_ID = {name: idx + 1 for idx, name in enumerate(CLASS_NAMES)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
ITEM_KEYS = ["item1", "item2", "item3", "item4"]


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
    raise FileNotFoundError(f"Could not find Task2 YOLO model. Checked: {checked}")


def resolve_device(provided: Optional[str]) -> str:
    if provided and str(provided).strip():
        return str(provided).strip()
    return "0" if torch.cuda.is_available() else "cpu"


def normalize_category_name(name: str) -> str:
    return str(name).strip().lower().replace("_", " ")


def category_folder_name(name: str) -> str:
    return normalize_category_name(name).replace(" ", "_")


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


def parse_gt_annotation(ann_path: Path, width: int, height: int) -> Tuple[List[dict], Dict[str, np.ndarray], Dict[str, int]]:
    with open(ann_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    instances = []
    class_masks = {name: np.zeros((height, width), dtype=np.uint8) for name in CLASS_NAMES}
    class_presence = {name: 0 for name in CLASS_NAMES}

    for key in ITEM_KEYS:
        item = data.get(key)
        if not isinstance(item, dict):
            continue

        category_name = normalize_category_name(item.get("category_name", ""))
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


def extract_annotation_categories(ann_path: Path) -> List[str]:
    with open(ann_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    categories = []
    for key in ITEM_KEYS:
        item = data.get(key)
        if not isinstance(item, dict):
            continue
        category_name = normalize_category_name(item.get("category_name", ""))
        if category_name:
            categories.append(category_name)
    return categories


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
    plt.title("YOLO Detection ROC Curves (Validation)")
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
        out_name = f"roc_{category_folder_name(class_name)}.png"
        plt.savefig(out_dir / out_name, dpi=180)
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Task2 YOLO validation inference + metrics")
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--images_dir", default="validation/validation/image")
    parser.add_argument("--annos_dir", default="validation/validation/annos")
    parser.add_argument("--report_dir", default="Report/task2_yolo")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.6)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    model_path = resolve_model_path(args.model_path)
    device = resolve_device(args.device)
    images_dir = Path(args.images_dir)
    annos_dir = Path(args.annos_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")
    if not annos_dir.exists():
        raise FileNotFoundError(f"Annotations dir not found: {annos_dir}")

    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if args.max_images is not None:
        image_paths = image_paths[: max(0, int(args.max_images))]

    if not image_paths:
        raise RuntimeError("No validation images found")

    samples = []
    for idx, image_path in enumerate(image_paths, start=1):
        ann_path = annos_dir / f"{image_path.stem}.json"
        if not ann_path.exists():
            continue

        with Image.open(image_path) as img:
            width, height = img.size

        samples.append(
            {
                "image_id": idx,
                "image_path": image_path,
                "ann_path": ann_path,
                "width": width,
                "height": height,
            }
        )

    if not samples:
        raise RuntimeError("No image/annotation pairs found")

    print(f"[INFO] Samples for evaluation: {len(samples)}")

    gt_coco = {
        "info": {"description": "Task2 validation GT"},
        "images": [],
        "annotations": [],
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(CLASS_NAMES)],
    }

    ann_id = 1
    total_gt_instances = 0
    seen_annotation_categories = set()
    for s in samples:
        gt_coco["images"].append(
            {
                "id": s["image_id"],
                "file_name": s["image_path"].name,
                "width": s["width"],
                "height": s["height"],
            }
        )

        gt_instances, _, _ = parse_gt_annotation(s["ann_path"], s["width"], s["height"])
        total_gt_instances += len(gt_instances)
        seen_annotation_categories.update(extract_annotation_categories(s["ann_path"]))
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
                    "image_id": s["image_id"],
                    "category_id": COCO_CAT_ID[inst["category_name"]],
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "segmentation": segs,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    if total_gt_instances == 0:
        found = ", ".join(sorted(seen_annotation_categories)) if seen_annotation_categories else "none"
        expected = ", ".join(CLASS_NAMES)
        raise RuntimeError(
            "No ground-truth instances were found for the evaluator target classes. "
            f"Expected classes: [{expected}]. "
            f"Found annotation categories in selected split: [{found}]. "
            "Choose a validation subset containing the target 5 classes."
        )

    gt_coco_path = report_dir / "gt_coco.json"
    with open(gt_coco_path, "w", encoding="utf-8") as file:
        json.dump(gt_coco, file)

    seg_stats = {
        name: {"tp": 0, "fp": 0, "fn": 0}
        for name in CLASS_NAMES
    }

    det_y_true = {name: [] for name in CLASS_NAMES}
    det_y_score = {name: [] for name in CLASS_NAMES}

    bbox_preds = []
    seg_preds = []

    model = YOLO(str(model_path))
    inference_paths = [str(s["image_path"]) for s in samples]

    print("[INFO] Running YOLO inference...")
    chunk_size = max(1, int(args.batch))
    for start in range(0, len(inference_paths), chunk_size):
        chunk_paths = inference_paths[start : start + chunk_size]
        chunk_samples = samples[start : start + chunk_size]
        results = model.predict(
            source=chunk_paths,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=device,
            batch=min(args.batch, len(chunk_paths)),
            verbose=False,
            stream=True,
        )

        for s, result in zip(chunk_samples, results):

            width = s["width"]
            height = s["height"]

            pred_class_masks = {name: np.zeros((height, width), dtype=np.uint8) for name in CLASS_NAMES}
            pred_max_score = {name: 0.0 for name in CLASS_NAMES}

            boxes = result.boxes
            masks = result.masks

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                cls = boxes.cls.detach().cpu().numpy().astype(int)
                conf = boxes.conf.detach().cpu().numpy()
                masks_xy = masks.xy if masks is not None and masks.xy is not None else None

                for i in range(len(cls)):
                    class_idx = int(cls[i])
                    if class_idx < 0 or class_idx >= len(CLASS_NAMES):
                        continue

                    class_name = CLASS_NAMES[class_idx]
                    score = float(conf[i])
                    pred_max_score[class_name] = max(pred_max_score[class_name], score)

                    x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                    w = max(0.0, x2 - x1)
                    h = max(0.0, y2 - y1)
                    if w <= 0 or h <= 0:
                        continue

                    bbox_preds.append(
                        {
                            "image_id": s["image_id"],
                            "category_id": class_idx + 1,
                            "bbox": [x1, y1, w, h],
                            "score": score,
                        }
                    )

                    seg_rle = None
                    if masks_xy is not None and i < len(masks_xy):
                        flat = polygon_xy_to_flat(masks_xy[i])
                        if len(flat) >= 6:
                            seg_poly = [flat]

                            pred_mask = rasterize_polygons(width, height, seg_poly)
                            pred_class_masks[class_name] = np.maximum(pred_class_masks[class_name], pred_mask)

                            if pred_mask.any():
                                rle = maskUtils.encode(np.asfortranarray(pred_mask.astype(np.uint8)))
                                seg_rle = {
                                    "size": [int(v) for v in rle["size"]],
                                    "counts": rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else str(rle["counts"]),
                                }

                    seg_preds.append(
                        {
                            "image_id": s["image_id"],
                            "category_id": class_idx + 1,
                            "segmentation": seg_rle,
                            "score": score,
                        }
                    )

            # Parse GT for this image on-the-fly (avoids huge RAM usage)
            _, gt_class_masks, gt_presence = parse_gt_annotation(s["ann_path"], width, height)

            # Update segmentation TP/FP/FN and detection ROC arrays
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

    bbox_pred_path = report_dir / "pred_bbox_coco.json"
    seg_pred_path = report_dir / "pred_seg_coco.json"

    with open(bbox_pred_path, "w", encoding="utf-8") as file:
        json.dump(bbox_preds, file)
    with open(seg_pred_path, "w", encoding="utf-8") as file:
        json.dump(seg_preds, file)

    print("[INFO] Running COCO evaluation...")
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
    seg_preds_non_empty = [p for p in seg_preds if p["segmentation"] is not None]
    if len(seg_preds_non_empty) > 0:
        seg_pred_non_empty_path = report_dir / "pred_seg_coco_non_empty.json"
        with open(seg_pred_non_empty_path, "w", encoding="utf-8") as file:
            json.dump(seg_preds_non_empty, file)
        coco_dt_seg = coco_gt_api.loadRes(str(seg_pred_non_empty_path))
        coco_eval_seg = COCOeval(coco_gt_api, coco_dt_seg, "segm")
        coco_eval_seg.evaluate()
        coco_eval_seg.accumulate()
        coco_eval_seg.summarize()
        seg_eval_summary = {
            "mAP_50_95": float(coco_eval_seg.stats[0]),
            "mAP_50": float(coco_eval_seg.stats[1]),
            "mAP_75": float(coco_eval_seg.stats[2]),
        }

    # Segmentation metrics
    seg_per_class = {}
    miou_values = []
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
        miou_values.append(iou)
        dice_values.append(dice)

    seg_macro = {
        "mIoU_macro": float(np.mean(miou_values)),
        "dice_macro": float(np.mean(dice_values)),
    }

    # Detection ROC/AUC/F1 per class
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

        for idx_point, (fpr_v, tpr_v, thr_v) in enumerate(zip(fpr, tpr, thresholds)):
            roc_rows.append(
                {
                    "class_name": class_name,
                    "point_index": idx_point,
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

    # Save ROC CSV and plots
    roc_csv_path = report_dir / "detection_roc_points.csv"
    with open(roc_csv_path, "w", encoding="utf-8") as file:
        file.write("class_name,point_index,fpr,tpr,threshold\n")
        for row in roc_rows:
            file.write(
                f"{row['class_name']},{row['point_index']},{row['fpr']},{row['tpr']},{row['threshold']}\n"
            )

    plot_dir = report_dir / "roc_plots"
    save_roc_plots(roc_data, plot_dir)

    summary = {
        "model_path": str(model_path),
        "images_dir": str(images_dir),
        "annos_dir": str(annos_dir),
        "num_images_evaluated": len(samples),
        "segmentation": {
            "per_class": seg_per_class,
            "macro": seg_macro,
            "coco_segm": seg_eval_summary,
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
            "roc_plot_dir": str(plot_dir),
        },
    }

    summary_json_path = report_dir / "task2_yolo_validation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    summary_txt_path = report_dir / "task2_yolo_validation_report.txt"
    with open(summary_txt_path, "w", encoding="utf-8") as file:
        file.write("=" * 84 + "\n")
        file.write("Task2 YOLO Validation Report\n")
        file.write("=" * 84 + "\n")
        file.write(f"Model: {model_path}\n")
        file.write(f"Images evaluated: {len(samples)}\n\n")

        file.write("[Segmentation]\n")
        file.write(f"Macro mIoU:  {seg_macro['mIoU_macro']:.6f}\n")
        file.write(f"Macro Dice:  {seg_macro['dice_macro']:.6f}\n")
        file.write(f"COCO Seg mAP@[0.5:0.95]: {seg_eval_summary['mAP_50_95']:.6f}\n")
        file.write(f"COCO Seg mAP@0.5:        {seg_eval_summary['mAP_50']:.6f}\n\n")

        file.write("Per-class Segmentation\n")
        file.write(f"{'Class':<22} {'IoU':>10} {'Dice':>10}\n")
        for class_name in CLASS_NAMES:
            c = seg_per_class[class_name]
            file.write(f"{class_name:<22} {c['iou']:>10.6f} {c['dice']:>10.6f}\n")

        file.write("\n[Detection]\n")
        file.write(f"COCO BBox mAP@[0.5:0.95]: {bbox_eval_summary['mAP_50_95']:.6f}\n")
        file.write(f"COCO BBox mAP@0.5:        {bbox_eval_summary['mAP_50']:.6f}\n")
        file.write(f"COCO BBox mAP@0.75:       {bbox_eval_summary['mAP_75']:.6f}\n\n")
        if det_macro["auc_macro"] is None:
            file.write("Macro AUC (detection):      n/a\n")
        else:
            file.write(f"Macro AUC (detection):      {det_macro['auc_macro']:.6f}\n")
        if det_macro["f1_macro"] is None:
            file.write("Macro F1 (detection):       n/a\n\n")
        else:
            file.write(f"Macro F1 (detection):       {det_macro['f1_macro']:.6f}\n\n")

        file.write("Per-class Detection (ROC/AUC/F1)\n")
        file.write(f"{'Class':<22} {'AUC':>10} {'F1':>10} {'P':>10} {'R':>10} {'Thr':>10}\n")
        for class_name in CLASS_NAMES:
            c = det_per_class[class_name]
            auc_text = "n/a" if c["auc"] is None else f"{c['auc']:.6f}"
            file.write(
                f"{class_name:<22} {auc_text:>10} {c['f1']:>10.6f} {c['precision']:>10.6f} "
                f"{c['recall']:>10.6f} {c['best_threshold']:>10.6f}\n"
            )

    print("\n[DONE] Task2 YOLO validation evaluation complete")
    print(f"Report JSON: {summary_json_path}")
    print(f"Report TXT : {summary_txt_path}")
    print(f"ROC plots  : {plot_dir}")


if __name__ == "__main__":
    main()
