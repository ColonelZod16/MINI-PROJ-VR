"""
validation_inference.py
======================
Utility to run trained multi-label classifiers on a folder of validation images.
Supports EfficientNet and MobileNetV3 checkpoints produced by train.py.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from models import build_model


CATEGORY_MAP = {
    "long sleeve top": 0,
    "short sleeve top": 1,
    "shorts": 2,
    "skirt": 3,
    "trousers": 4,
}
CLASS_NAMES = [name for name, _ in sorted(CATEGORY_MAP.items(), key=lambda item: item[1])]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
ITEM_KEYS = ["item1", "item2", "item3", "item4"]


def _find_first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _resolve_run_dir(arch_name: str) -> Path:
    candidates = [
        Path("Runs") / "Smaller_Dataset" / arch_name,
        Path("Runs") / arch_name,
    ]
    run_dir = _find_first_existing(candidates)
    if run_dir is None:
        raise FileNotFoundError(
            f"Could not find run directory for {arch_name}. Checked: {[str(path) for path in candidates]}"
        )
    return run_dir


def _load_results_metadata(results_path: Path) -> dict:
    if not results_path.exists():
        return {}
    with open(results_path, "r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_thresholds(results_data: dict, fallback_threshold: float, num_classes: int) -> np.ndarray:
    threshold_block = results_data.get("thresholds", {}) if isinstance(results_data, dict) else {}
    tuned_thresholds = threshold_block.get("tuned")

    if isinstance(tuned_thresholds, list) and len(tuned_thresholds) == num_classes:
        return np.array(tuned_thresholds, dtype=np.float32)

    default_threshold = threshold_block.get("default", fallback_threshold)
    return np.full((num_classes,), float(default_threshold), dtype=np.float32)


def _resolve_img_size(results_data: dict, fallback_size: int) -> int:
    cfg = results_data.get("config", {}) if isinstance(results_data, dict) else {}
    value = cfg.get("img_size", fallback_size)
    return int(value)


def _resolve_dropout(results_data: dict, fallback_dropout: float) -> float:
    cfg = results_data.get("config", {}) if isinstance(results_data, dict) else {}
    value = cfg.get("dropout", fallback_dropout)
    return float(value)


def _resolve_validation_images_dir(provided_dir: Optional[str]) -> Path:
    if provided_dir:
        images_dir = Path(provided_dir)
        if not images_dir.exists():
            raise FileNotFoundError(f"Validation images directory not found: {images_dir}")
        return images_dir

    candidates = [
        Path("validation") / "validation" / "image",
        Path("validation") / "image",
    ]

    images_dir = _find_first_existing(candidates)
    if images_dir is None:
        raise FileNotFoundError(
            f"Could not find validation image folder. Checked: {[str(path) for path in candidates]}"
        )
    return images_dir


def _resolve_validation_annos_dir(provided_dir: Optional[str]) -> Optional[Path]:
    if provided_dir:
        annos_dir = Path(provided_dir)
        if not annos_dir.exists():
            raise FileNotFoundError(f"Validation annotation directory not found: {annos_dir}")
        return annos_dir

    candidates = [
        Path("validation") / "validation" / "annos",
        Path("validation") / "annos",
    ]

    return _find_first_existing(candidates)


def _labels_from_annotation(ann_path: Path) -> np.ndarray:
    target = np.zeros((len(CLASS_NAMES),), dtype=np.int32)
    if not ann_path.exists():
        return target

    with open(ann_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    for item_key in ITEM_KEYS:
        item = data.get(item_key)
        if not isinstance(item, dict):
            continue
        category_name = str(item.get("category_name", "")).strip().lower()
        if category_name in CATEGORY_MAP:
            target[CATEGORY_MAP[category_name]] = 1

    return target


def _precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, int, int, int]:
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return float(precision), float(recall), float(f1), tp, fp, fn


def _roc_curve_and_auc(y_true: np.ndarray, y_score: np.ndarray):
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())

    if positives == 0 or negatives == 0:
        return [0.0, 1.0], [0.0, 1.0], [float("inf"), 0.0], None

    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    s_sorted = y_score[order]

    tpr = [0.0]
    fpr = [0.0]
    thresholds = [float("inf")]

    tp = 0
    fp = 0

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


def _write_metrics_report(
    report_path: Path,
    model_name: str,
    n_images: int,
    n_eval: int,
    per_class: Dict[str, Dict[str, float]],
    macro_f1: float,
    micro_f1: float,
    exact_match_accuracy: float,
    label_accuracy: float,
    hamming_loss: float,
) -> None:
    lines = []
    lines.append("=" * 74)
    lines.append(f"Validation Metrics Report - {model_name}")
    lines.append("=" * 74)
    lines.append(f"Total predicted images      : {n_images}")
    lines.append(f"Images with GT annotations  : {n_eval}")
    lines.append("")
    lines.append("Overall Metrics")
    lines.append("-" * 74)
    lines.append(f"Exact-match Accuracy        : {exact_match_accuracy:.6f}")
    lines.append(f"Label-wise Accuracy         : {label_accuracy:.6f}")
    lines.append(f"Hamming Loss                : {hamming_loss:.6f}")
    lines.append(f"Macro F1                    : {macro_f1:.6f}")
    lines.append(f"Micro F1                    : {micro_f1:.6f}")
    lines.append("")
    lines.append("Per-class Metrics")
    lines.append("-" * 74)
    lines.append(f"{'Class':<22} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC_AUC':>10}")
    for class_name in CLASS_NAMES:
        stats = per_class[class_name]
        auc_value = stats["roc_auc"]
        auc_text = f"{auc_value:.6f}" if isinstance(auc_value, float) else "n/a"
        lines.append(
            f"{class_name:<22} {stats['precision']:>10.6f} {stats['recall']:>10.6f} {stats['f1']:>10.6f} {auc_text:>10}"
        )
    lines.append("=" * 74)

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _load_model(
    arch: str,
    variant: str,
    dropout: float,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model(
        arch=arch,
        variant=variant,
        num_classes=len(CLASS_NAMES),
        dropout=dropout,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def run_validation_inference(
    arch: str,
    variant: str,
    output_csv_name: str,
    validation_images_dir: Optional[str] = None,
    validation_annos_dir: Optional[str] = None,
    threshold: float = 0.65,
    max_images: Optional[int] = None,
    output_txt_name: str = "validation_metrics.txt",
    roc_csv_name: str = "validation_roc_points.csv",
    report_dir: str = "Report",
    batch_size: int = 64,
) -> Path:
    arch_name = f"efficientnet_{variant}" if arch == "efficientnet" else arch
    run_dir = _resolve_run_dir(arch_name)

    checkpoint_path = run_dir / "best_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    results_path = run_dir / "results.json"
    results_data = _load_results_metadata(results_path)

    img_size = _resolve_img_size(results_data, fallback_size=260)
    dropout = _resolve_dropout(results_data, fallback_dropout=0.6)
    thresholds = _resolve_thresholds(
        results_data=results_data,
        fallback_threshold=threshold,
        num_classes=len(CLASS_NAMES),
    )

    image_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    images_dir = _resolve_validation_images_dir(validation_images_dir)
    annos_dir = _resolve_validation_annos_dir(validation_annos_dir)
    image_paths = sorted([
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ])

    if max_images is not None:
        image_paths = image_paths[: max(0, int(max_images))]

    if not image_paths:
        raise RuntimeError(f"No images found in {images_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(
        arch=arch,
        variant=variant,
        dropout=dropout,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    report_root = Path(report_dir)
    report_model_dir = report_root / arch_name
    report_model_dir.mkdir(parents=True, exist_ok=True)

    output_rows: List[Dict[str, object]] = []
    eval_probs: List[np.ndarray] = []
    eval_preds: List[np.ndarray] = []
    eval_targets: List[np.ndarray] = []

    bs = max(1, int(batch_size))

    with torch.no_grad():
        for start_idx in range(0, len(image_paths), bs):
            batch_paths = image_paths[start_idx:start_idx + bs]
            batch_tensors = []

            for image_path in batch_paths:
                image = Image.open(image_path).convert("RGB")
                batch_tensors.append(image_transform(image))

            batch_tensor = torch.stack(batch_tensors, dim=0).to(device)
            logits = model(batch_tensor)
            probs_batch = torch.sigmoid(logits).detach().cpu().numpy()

            for image_path, probs in zip(batch_paths, probs_batch):
                binary_preds = (probs >= thresholds).astype(np.int32)

                labels = [CLASS_NAMES[idx] for idx, value in enumerate(binary_preds) if value == 1]

                row: Dict[str, object] = {
                    "image_name": image_path.name,
                    "predicted_labels": "|".join(labels) if labels else "none",
                }

                for index, class_name in enumerate(CLASS_NAMES):
                    class_key = class_name.replace(" ", "_")
                    row[f"prob_{class_key}"] = float(probs[index])
                    row[f"pred_{class_key}"] = int(binary_preds[index])
                    row[f"thr_{class_key}"] = float(thresholds[index])

                if annos_dir is not None:
                    ann_path = annos_dir / f"{image_path.stem}.json"
                    target = _labels_from_annotation(ann_path)
                    row["has_ground_truth"] = int(ann_path.exists())

                    if ann_path.exists():
                        eval_probs.append(probs.astype(np.float32))
                        eval_preds.append(binary_preds.astype(np.int32))
                        eval_targets.append(target)
                else:
                    row["has_ground_truth"] = 0

                output_rows.append(row)

    output_path = report_model_dir / output_csv_name
    field_names = list(output_rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(output_rows)

    print("=" * 68)
    print(f"Model           : {arch_name}")
    print(f"Device          : {device}")
    print(f"Run dir         : {run_dir}")
    print(f"Report dir      : {report_model_dir}")
    print(f"Images dir      : {images_dir}")
    print(f"Annos dir       : {annos_dir if annos_dir is not None else 'not found'}")
    print(f"Images tested   : {len(output_rows)}")
    print(f"Output CSV      : {output_path}")
    print("=" * 68)

    label_counts = {name: 0 for name in CLASS_NAMES}
    for row in output_rows:
        for class_name in CLASS_NAMES:
            pred_key = f"pred_{class_name.replace(' ', '_')}"
            label_counts[class_name] += int(row[pred_key])

    print("Predicted-positive counts:")
    for class_name in CLASS_NAMES:
        print(f"  {class_name:<22} {label_counts[class_name]:>6}")

    if eval_targets:
        probs_np = np.stack(eval_probs, axis=0)
        preds_np = np.stack(eval_preds, axis=0)
        targets_np = np.stack(eval_targets, axis=0)

        per_class_metrics: Dict[str, Dict[str, float]] = {}
        roc_rows: List[Dict[str, object]] = []

        macro_f1_values = []
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for class_idx, class_name in enumerate(CLASS_NAMES):
            y_true = targets_np[:, class_idx]
            y_pred = preds_np[:, class_idx]
            y_prob = probs_np[:, class_idx]

            precision, recall, f1, tp, fp, fn = _precision_recall_f1(y_true, y_pred)
            fpr_points, tpr_points, roc_thresholds, auc = _roc_curve_and_auc(y_true, y_prob)

            per_class_metrics[class_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "roc_auc": auc,
            }

            macro_f1_values.append(f1)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            for point_idx in range(len(fpr_points)):
                roc_rows.append({
                    "class_name": class_name,
                    "point_index": point_idx,
                    "fpr": float(fpr_points[point_idx]),
                    "tpr": float(tpr_points[point_idx]),
                    "threshold": float(roc_thresholds[point_idx]),
                })

        macro_f1 = float(np.mean(macro_f1_values))
        micro_precision = total_tp / (total_tp + total_fp + 1e-8)
        micro_recall = total_tp / (total_tp + total_fn + 1e-8)
        micro_f1 = float(2 * micro_precision * micro_recall / (micro_precision + micro_recall + 1e-8))

        exact_match_accuracy = float(np.mean(np.all(preds_np == targets_np, axis=1)))
        hamming_loss = float(np.mean(preds_np != targets_np))
        label_accuracy = float(1.0 - hamming_loss)

        metrics_report_path = report_model_dir / output_txt_name
        _write_metrics_report(
            report_path=metrics_report_path,
            model_name=arch_name,
            n_images=len(output_rows),
            n_eval=targets_np.shape[0],
            per_class=per_class_metrics,
            macro_f1=macro_f1,
            micro_f1=micro_f1,
            exact_match_accuracy=exact_match_accuracy,
            label_accuracy=label_accuracy,
            hamming_loss=hamming_loss,
        )

        roc_csv_path = report_model_dir / roc_csv_name
        with open(roc_csv_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["class_name", "point_index", "fpr", "tpr", "threshold"])
            writer.writeheader()
            writer.writerows(roc_rows)

        print("\nEvaluation metrics:")
        print(f"  Metrics report : {metrics_report_path}")
        print(f"  ROC points CSV : {roc_csv_path}")
        print(f"  Exact-match Acc: {exact_match_accuracy:.4f}")
        print(f"  Label Accuracy : {label_accuracy:.4f}")
        print(f"  Macro F1       : {macro_f1:.4f}")
        print(f"  Micro F1       : {micro_f1:.4f}")
    else:
        print("\nNo matching annotation JSON files were found; metrics report was not generated.")

    return output_path
