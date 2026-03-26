"""
metrics.py
==========
Multi-label classification metrics:
  - Per-class Precision, Recall, F1
  - Mean Average Precision (mAP)
  - Macro / Micro F1
  - Hamming Loss
  - Exact Match Ratio
"""

import numpy as np
import torch
from typing import Dict, List


class MultiLabelMetrics:
    """
    Accumulates predictions over an entire split, then computes all metrics.

    Usage
    -----
        m = MultiLabelMetrics(num_classes=5, class_names=[...])
        for images, targets, _ in val_loader:
            logits = model(images)
            m.update(logits, targets)
        results = m.compute()
        m.pretty_print(results)
        m.reset()
    """

    def __init__(self, num_classes: int, class_names: List[str], threshold: float = 0.65):
        self.num_classes = num_classes
        self.class_names = class_names
        self.threshold   = threshold
        self.reset()

    def reset(self):
        self._probs   = []
        self._targets = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        self._probs.append(torch.sigmoid(logits).detach().cpu())
        self._targets.append(targets.detach().cpu().float())

    def compute(self) -> Dict[str, float]:
        probs   = torch.cat(self._probs,   dim=0).numpy()   # [N, C]
        targets = torch.cat(self._targets, dim=0).numpy()   # [N, C]
        preds   = (probs >= self.threshold).astype(np.float32)

        results        = {}
        per_class_f1   = []
        per_class_ap   = []

        for c, name in enumerate(self.class_names):
            tp = ((preds[:, c] == 1) & (targets[:, c] == 1)).sum()
            fp = ((preds[:, c] == 1) & (targets[:, c] == 0)).sum()
            fn = ((preds[:, c] == 0) & (targets[:, c] == 1)).sum()

            prec = tp / (tp + fp + 1e-8)
            rec  = tp / (tp + fn + 1e-8)
            f1   = 2 * prec * rec / (prec + rec + 1e-8)

            results[f"{name}/precision"] = float(prec)
            results[f"{name}/recall"]    = float(rec)
            results[f"{name}/f1"]        = float(f1)

            ap = self._average_precision(targets[:, c], probs[:, c])
            results[f"{name}/ap"] = ap
            per_class_f1.append(f1)
            per_class_ap.append(ap)

        # Macro
        results["macro_f1"] = float(np.mean(per_class_f1))
        results["mAP"]      = float(np.mean(per_class_ap))

        # Micro
        tp_all = ((preds == 1) & (targets == 1)).sum()
        fp_all = ((preds == 1) & (targets == 0)).sum()
        fn_all = ((preds == 0) & (targets == 1)).sum()
        mp = tp_all / (tp_all + fp_all + 1e-8)
        mr = tp_all / (tp_all + fn_all + 1e-8)
        results["micro_f1"]     = float(2 * mp * mr / (mp + mr + 1e-8))
        results["hamming_loss"] = float(np.mean(preds != targets))
        results["exact_match"]  = float(np.mean(np.all(preds == targets, axis=1)))

        return results

    @staticmethod
    def _average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
        """Compute AP via precision-recall curve (trapezoidal)."""
        sorted_idx = np.argsort(-y_score)
        y_true     = y_true[sorted_idx]
        tp_cumsum  = np.cumsum(y_true)
        n_pos      = y_true.sum()
        if n_pos == 0:
            return 0.0
        precision  = tp_cumsum / (np.arange(len(y_true)) + 1)
        recall     = tp_cumsum / n_pos
        # Prepend (0,0) for trapezoid
        precision  = np.concatenate([[0], precision])
        recall     = np.concatenate([[0], recall])
        return float(np.trapz(precision, recall))

    def pretty_print(self, results: Dict[str, float]):
        w = 55
        print(f"\n{'─'*w}")
        print(f"  {'Metric':<30}  {'Value':>10}")
        print(f"{'─'*w}")
        for key in ["mAP", "macro_f1", "micro_f1", "hamming_loss", "exact_match"]:
            print(f"  {key:<30}  {results[key]:>10.4f}")
        print(f"{'─'*w}")
        print(f"  {'Class':<22} {'AP':>6}  {'F1':>6}  {'P':>6}  {'R':>6}")
        print(f"{'─'*w}")
        for name in self.class_names:
            print(
                f"  {name:<22} "
                f"{results.get(f'{name}/ap', 0):>6.3f}  "
                f"{results.get(f'{name}/f1', 0):>6.3f}  "
                f"{results.get(f'{name}/precision', 0):>6.3f}  "
                f"{results.get(f'{name}/recall', 0):>6.3f}"
            )
        print(f"{'─'*w}\n")
