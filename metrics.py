from __future__ import annotations

import numpy as np
import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, ignore_index: int = 255) -> None:
        pred = predictions.detach().cpu().numpy().reshape(-1)
        target = targets.detach().cpu().numpy().reshape(-1)
        valid = (target != ignore_index) & (target >= 0) & (target < self.num_classes)
        valid &= (pred >= 0) & (pred < self.num_classes)
        indices = self.num_classes * target[valid] + pred[valid]
        self.matrix += np.bincount(indices, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self) -> dict[str, object]:
        matrix = self.matrix.astype(np.float64)
        true_count = matrix.sum(axis=1)
        pred_count = matrix.sum(axis=0)
        correct = np.diag(matrix)
        union = true_count + pred_count - correct
        iou = np.divide(correct, union, out=np.full_like(correct, np.nan), where=union > 0)
        dice_denominator = true_count + pred_count
        dice = np.divide(2 * correct, dice_denominator, out=np.full_like(correct, np.nan), where=dice_denominator > 0)
        class_accuracy = np.divide(correct, true_count, out=np.full_like(correct, np.nan), where=true_count > 0)
        total = matrix.sum()
        return {
            "confusion_matrix": self.matrix.copy(),
            "per_class_iou": iou,
            "per_class_dice": dice,
            "mean_iou": float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0,
            "mean_iou_no_background": float(np.nanmean(iou[1:])) if np.any(~np.isnan(iou[1:])) else 0.0,
            "mean_dice": float(np.nanmean(dice)) if np.any(~np.isnan(dice)) else 0.0,
            "pixel_accuracy": float(correct.sum() / total) if total else 0.0,
            "mean_class_accuracy": float(np.nanmean(class_accuracy)) if np.any(~np.isnan(class_accuracy)) else 0.0,
        }
