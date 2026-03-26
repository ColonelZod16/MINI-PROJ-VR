"""
train.py
========
Training script for multi-label fashion classification.
Trains all 3 required architectures:
  - ResNet-50
  - EfficientNet-B0 / B2
  - MobileNetV3-Large

2-Phase Training Strategy:
  Phase 1 (epochs 1-10)  : Backbone frozen, only head trains at lr=1e-3
  Phase 2 (epochs 11+)   : Full backbone unfrozen with differential LR
                            backbone=1e-5 (slow), head=1e-4 (faster)

Run one architecture at a time:
  python train.py --arch resnet50
  python train.py --arch efficientnet --variant b0
  python train.py --arch efficientnet --variant b2
  python train.py --arch mobilenetv3

Or run all back-to-back:
  python train.py --run_all
"""

import os
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

from models  import build_model
from dataset import get_dataloaders
from metrics import MultiLabelMetrics


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DEFAULT_CONFIG = {
    "processed_dir":  "Pruned_Clothing_Dataset_30k",
    "images_dir":     r"Pruned_Clothing_Dataset_30k",
    "output_dir":     r"Runs/Smaller_Dataset",

    # Training
    "epochs":           60,
    "batch_size":       48,
    "img_size":         260,
    "lr":               3e-4,    # Phase 1 head-only LR
    "lr_backbone":      1e-5,    # Phase 2 backbone LR
    "lr_head":          1e-4,    # Phase 2 head LR
    "phase2_epoch":     8,      # epoch at which backbone unfreezes
    "weight_decay":     1e-4,
    "dropout":          0.6,
    "num_workers":      4,

    # Early stopping
    "patience":         10,

    # Mixed precision — big speedup on RTX 4070 Super
    "use_amp":          True,

    # Pos weight for BCE loss
    "use_pos_weight":   True,
    "pos_weight_power": 0.5,

    # Sampling / decision thresholds
    "use_weighted_sampler": True,
    "threshold":        0.65,
    "tune_thresholds":  False,
}

CATEGORY_MAP = {
    "long sleeve top":  0,
    "short sleeve top": 1,
    "shorts":           2,
    "skirt":            3,
    "trousers":         4,
}

CLASS_NAMES = [k for k, _ in sorted(CATEGORY_MAP.items(), key=lambda x: x[1])]


# ─────────────────────────────────────────────
# BACKBONE ACCESSOR
# Returns the feature extractor submodule for freezing/unfreezing.
# Each architecture stores it under a different attribute name.
# ─────────────────────────────────────────────
def get_backbone_params(model: nn.Module, arch: str):
    """Return (backbone_params, head_params) for differential LR."""
    if arch in ("efficientnet", "resnet50"):
        return model.features.parameters(), model.classifier.parameters()
    elif arch == "mobilenetv3":
        return model.features.parameters(), model.classifier.parameters()
    else:
        # Fallback: treat all params as head
        return [], model.parameters()


def freeze_backbone(model: nn.Module, arch: str):
    if hasattr(model, "features"):
        for param in model.features.parameters():
            param.requires_grad = False
        print("  [Phase 1] Backbone frozen — training head only")
    elif hasattr(model, "backbone"):
        for param in model.backbone.parameters():
            param.requires_grad = False
        print("  [Phase 1] Backbone frozen — training head only")


def unfreeze_all(model: nn.Module):
    for param in model.parameters():
        param.requires_grad = True
    print("  [Phase 2] All layers unfrozen")


# ─────────────────────────────────────────────
# LOSS
# ─────────────────────────────────────────────
def build_loss(
    processed_dir:    str,
    device:           torch.device,
    use_pos_weight:   bool  = True,
    pos_weight_power: float = 0.5,
) -> nn.BCEWithLogitsLoss:
    if not use_pos_weight:
        print("[Loss] BCEWithLogitsLoss  (pos_weight disabled)")
        return nn.BCEWithLogitsLoss()

    weights_path = Path(processed_dir) / "class_weights.json"

    if weights_path.exists():
        with open(weights_path) as f:
            data = json.load(f)
        cw = data.get("class_weights", {})
        pw = torch.ones(len(CATEGORY_MAP))
        for name, idx in CATEGORY_MAP.items():
            pw[idx] = float(cw.get(name, 1.0)) ** float(pos_weight_power)
        print(f"[Loss] BCEWithLogitsLoss  pos_weight = {[round(x,4) for x in pw.tolist()]}")
        return nn.BCEWithLogitsLoss(pos_weight=pw.to(device))
    else:
        print("[Loss] BCEWithLogitsLoss  (class_weights.json not found — no pos_weight)")
        return nn.BCEWithLogitsLoss()


# ─────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────
def train_one_epoch(
    model, loader, criterion, optimizer,
    scaler, device, use_amp: bool, epoch: int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for i, (images, targets, _) in enumerate(loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if (i + 1) % 100 == 0 or (i + 1) == n_batches:
            print(f"  Epoch {epoch}  [{i+1:>4}/{n_batches}]  "
                  f"loss={total_loss/(i+1):.4f}", end="\r")
    print()
    return total_loss / n_batches


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model, loader, criterion, device,
    use_amp: bool, metrics: MultiLabelMetrics,
) -> dict:
    model.eval()
    metrics.reset()
    total_loss = 0.0

    for images, targets, _ in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, targets)

        total_loss += loss.item()
        metrics.update(logits, targets)

    results         = metrics.compute()
    results["loss"] = total_loss / len(loader)
    return results


@torch.no_grad()
def collect_probs_targets(model, loader, device, use_amp: bool):
    model.eval()
    all_probs, all_targets = [], []

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        with autocast(enabled=use_amp):
            logits = model(images)
        all_probs.append(torch.sigmoid(logits).detach().cpu())
        all_targets.append(targets.detach().cpu().float())

    return (
        torch.cat(all_probs,   dim=0).numpy(),
        torch.cat(all_targets, dim=0).numpy(),
    )


# ─────────────────────────────────────────────
# THRESHOLD TUNING
# ─────────────────────────────────────────────
def tune_thresholds_by_f1(
    probs:         np.ndarray,
    targets:       np.ndarray,
    class_names:   List[str],
    search_values: np.ndarray,
) -> List[float]:
    print("\n[Threshold Tuning] Optimizing per-class thresholds on VAL set …")
    tuned = []

    for idx, name in enumerate(class_names):
        y_true = targets[:, idx]
        y_prob = probs[:, idx]
        best_thr, best_f1 = 0.5, -1.0

        for thr in search_values:
            preds = (y_prob >= thr).astype(np.float32)
            tp = ((preds == 1) & (y_true == 1)).sum()
            fp = ((preds == 1) & (y_true == 0)).sum()
            fn = ((preds == 0) & (y_true == 1)).sum()
            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = 2 * prec * rec / (prec + rec + 1e-8)
            if f1 > best_f1:
                best_f1  = f1
                best_thr = float(thr)

        tuned.append(best_thr)
        print(f"  {name:<22}  threshold={best_thr:.2f}  val_f1={best_f1:.4f}")

    return tuned


# ─────────────────────────────────────────────
# MAIN TRAIN FUNCTION
# ─────────────────────────────────────────────
def train(cfg: dict, arch: str, variant: str = "b0"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch_name = f"efficientnet_{variant}" if arch == "efficientnet" else arch

    print(f"\n{'='*60}")
    print(f"  Training : {arch_name.upper()}")
    print(f"  Device   : {device}")
    print(f"  Strategy : Phase1(freeze,lr={cfg['lr']}) → "
          f"Phase2(unfreeze,backbone={cfg['lr_backbone']},head={cfg['lr_head']})"
          f" at epoch {cfg['phase2_epoch']}")
    print(f"{'='*60}")

    run_dir = Path(cfg["output_dir"]) / arch_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataloaders ───────────────────────────────────────────────────
    train_loader, val_loader, test_loader = get_dataloaders(
        processed_dir        = cfg["processed_dir"],
        images_dir           = cfg["images_dir"],
        category_map         = CATEGORY_MAP,
        batch_size           = cfg["batch_size"],
        img_size             = cfg["img_size"],
        num_workers          = cfg["num_workers"],
        use_weighted_sampler = cfg["use_weighted_sampler"],
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(
        arch=arch, num_classes=len(CATEGORY_MAP),
        dropout=cfg["dropout"], variant=variant,
    ).to(device)

    # ── Loss ──────────────────────────────────────────────────────────
    criterion = build_loss(
        cfg["processed_dir"], device,
        use_pos_weight   = cfg["use_pos_weight"],
        pos_weight_power = cfg["pos_weight_power"],
    )

    # ── Phase 1 setup: freeze backbone ────────────────────────────────
    freeze_backbone(model, arch)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["lr"], weight_decay=cfg["weight_decay"],
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max  = cfg["phase2_epoch"] - 1,
        eta_min= 1e-6,
    )

    scaler  = GradScaler(enabled=cfg["use_amp"])
    metrics = MultiLabelMetrics(len(CATEGORY_MAP), CLASS_NAMES, threshold=cfg["threshold"])

    best_map     = 0.0
    patience_ctr = 0
    history      = []
    phase        = 1

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # ── Switch to Phase 2 ─────────────────────────────────────────
        if epoch == cfg["phase2_epoch"] and phase == 1:
            phase = 2
            print(f"\n{'─'*60}")
            print(f"  Switching to Phase 2 at epoch {epoch}")
            unfreeze_all(model)
            optimizer = AdamW([
                {"params": model.features.parameters(),
                 "lr": cfg["lr_backbone"]},
                {"params": model.classifier.parameters(),
                 "lr": cfg["lr_head"]},
            ], weight_decay=cfg["weight_decay"])
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max  = cfg["epochs"] - cfg["phase2_epoch"] + 1,
                eta_min= 1e-7,
            )
            print(f"  backbone lr={cfg['lr_backbone']}  head lr={cfg['lr_head']}")
            print(f"{'─'*60}\n")

        # ── Train + Evaluate ──────────────────────────────────────────
        train_loss  = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, cfg["use_amp"], epoch,
        )
        val_results = evaluate(
            model, val_loader, criterion, device, cfg["use_amp"], metrics,
        )
        scheduler.step()

        elapsed = time.time() - t0
        val_map = val_results["mAP"]
        val_f1  = val_results["macro_f1"]

        print(f"  [Phase {phase}] Epoch {epoch:>3}/{cfg['epochs']}  "
              f"train_loss={train_loss:.4f}  "
              f"val_loss={val_results['loss']:.4f}  "
              f"val_mAP={val_map:.4f}  "
              f"val_F1={val_f1:.4f}  "
              f"({elapsed:.0f}s)")

        history.append({"epoch": epoch, "phase": phase,
                        "train_loss": train_loss, **val_results})

        # ── Save best checkpoint ──────────────────────────────────────
        if val_map > best_map:
            best_map     = val_map
            patience_ctr = 0
            torch.save({
                "epoch":        epoch,
                "phase":        phase,
                "arch":         arch_name,
                "state_dict":   model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "val_mAP":      val_map,
                "category_map": CATEGORY_MAP,
            }, run_dir / "best_model.pth")
            print(f"  ✓ Best checkpoint saved  (val_mAP={best_map:.4f})")
        else:
            patience_ctr += 1
            # Only count patience in Phase 2 — Phase 1 is just warmup
            if phase == 2 and patience_ctr >= cfg["patience"]:
                print(f"\n  Early stopping: no val_mAP improvement for "
                      f"{cfg['patience']} epochs")
                break

    # ── Load best model for test evaluation ───────────────────────────
    print(f"\n[Test] Loading best checkpoint (val_mAP={best_map:.4f}) …")
    ckpt = torch.load(run_dir / "best_model.pth", map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    # ── Optional: tune per-class thresholds on val ───────────────────
    tuned_thresholds = None
    if cfg["tune_thresholds"]:
        val_probs, val_targets = collect_probs_targets(
            model, val_loader, device, cfg["use_amp"]
        )
        tuned_thresholds = tune_thresholds_by_f1(
            probs         = val_probs,
            targets       = val_targets,
            class_names   = CLASS_NAMES,
            search_values = np.arange(0.30, 0.86, 0.05),
        )
        metrics.threshold = np.array(tuned_thresholds, dtype=np.float32)
        print(f"\n[Threshold Tuning] Applied to TEST: {tuned_thresholds}")

    # ── Test evaluation ───────────────────────────────────────────────
    test_results = evaluate(
        model, test_loader, criterion, device, cfg["use_amp"], metrics,
    )
    print(f"\n{'='*60}")
    print(f"  TEST RESULTS — {arch_name.upper()}")
    print(f"{'='*60}")
    metrics.pretty_print(test_results)

    # ── Save results to JSON ──────────────────────────────────────────
    with open(run_dir / "results.json", "w") as f:
        json.dump({
            "arch":         arch_name,
            "best_val_mAP": best_map,
            "test":         test_results,
            "history":      history,
            "config":       cfg,
            "thresholds": {
                "default": cfg["threshold"],
                "tuned":   tuned_thresholds,
            },
        }, f, indent=2)
    print(f"[Saved] {run_dir / 'results.json'}")

    return test_results


# ─────────────────────────────────────────────
# COMPARE ALL RESULTS
# ─────────────────────────────────────────────
def compare_results(output_dir: str):
    print(f"\n{'='*65}")
    print("  ARCHITECTURE COMPARISON (Test Set)")
    print(f"{'='*65}")
    print(f"  {'Architecture':<22} {'mAP':>6}  {'MacroF1':>8}  "
          f"{'MicroF1':>8}  {'Hamming':>8}")
    print(f"{'─'*65}")

    for arch_dir in sorted(Path(output_dir).iterdir()):
        rpath = arch_dir / "results.json"
        if not rpath.exists():
            continue
        with open(rpath) as f:
            data = json.load(f)
        t = data["test"]
        print(f"  {data['arch']:<22} "
              f"{t['mAP']:>6.4f}  "
              f"{t['macro_f1']:>8.4f}  "
              f"{t['micro_f1']:>8.4f}  "
              f"{t['hamming_loss']:>8.4f}")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch",     default="efficientnet",
                        choices=["resnet50", "efficientnet", "mobilenetv3"])
    parser.add_argument("--variant",  default="b2", choices=["b0", "b2"])
    parser.add_argument("--run_all",  action="store_true",
                        help="Train all architectures back-to-back")

    # Paths
    parser.add_argument("--processed_dir", default=DEFAULT_CONFIG["processed_dir"])
    parser.add_argument("--images_dir",    default=DEFAULT_CONFIG["images_dir"])
    parser.add_argument("--output_dir",    default=DEFAULT_CONFIG["output_dir"])

    # Training hyperparams
    parser.add_argument("--epochs",       type=int,   default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--batch_size",   type=int,   default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr",           type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--lr_backbone",  type=float, default=DEFAULT_CONFIG["lr_backbone"])
    parser.add_argument("--lr_head",      type=float, default=DEFAULT_CONFIG["lr_head"])
    parser.add_argument("--phase2_epoch", type=int,   default=DEFAULT_CONFIG["phase2_epoch"])
    parser.add_argument("--patience",     type=int,   default=DEFAULT_CONFIG["patience"])
    parser.add_argument("--dropout",      type=float, default=DEFAULT_CONFIG["dropout"])

    # Threshold
    parser.add_argument("--threshold",      type=float, default=DEFAULT_CONFIG["threshold"])
    parser.add_argument("--tune_thresholds", action="store_true")

    # Flags
    parser.add_argument("--no_amp",              action="store_true")
    parser.add_argument("--no_weighted_sampler", action="store_true")
    parser.add_argument("--no_pos_weight",       action="store_true")
    parser.add_argument("--pos_weight_power",    type=float,
                        default=DEFAULT_CONFIG["pos_weight_power"])

    args = parser.parse_args()

    cfg = {**DEFAULT_CONFIG}
    cfg.update({
        "processed_dir":        args.processed_dir,
        "images_dir":           args.images_dir,
        "output_dir":           args.output_dir,
        "epochs":               args.epochs,
        "batch_size":           args.batch_size,
        "lr":                   args.lr,
        "lr_backbone":          args.lr_backbone,
        "lr_head":              args.lr_head,
        "phase2_epoch":         args.phase2_epoch,
        "patience":             args.patience,
        "dropout":              args.dropout,
        "threshold":            args.threshold,
        "tune_thresholds":      args.tune_thresholds,
        "use_amp":              not args.no_amp,
        "use_weighted_sampler": not args.no_weighted_sampler,
        "use_pos_weight":       not args.no_pos_weight,
        "pos_weight_power":     args.pos_weight_power,
    })

    if args.run_all:
        for arch, variant in [
            ("resnet50",     "b0"),
            ("efficientnet", "b0"),
            ("efficientnet", "b2"),
            ("mobilenetv3",  "b0"),
        ]:
            train(cfg, arch=arch, variant=variant)
        compare_results(cfg["output_dir"])
    else:
        train(cfg, arch=args.arch, variant=args.variant)