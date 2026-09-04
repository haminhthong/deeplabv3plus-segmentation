from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from voc_meta import VOC_CLASSES


class SegmentationMetrics:
    def __init__(self, num_classes: int):
        if num_classes <= 0:
            raise ValueError("Số lớp phải lớn hơn 0")
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, predictions: torch.Tensor, targets: torch.Tensor, ignore_index: int = 255) -> None:
        if predictions.shape != targets.shape:
            raise ValueError(
                "Dự đoán và nhãn thật phải cùng kích thước: "
                f"{tuple(predictions.shape)} != {tuple(targets.shape)}"
            )
        pred = predictions.detach().cpu().numpy().reshape(-1)
        target = targets.detach().cpu().numpy().reshape(-1)
        valid = (target != ignore_index) & (target >= 0) & (target < self.num_classes)
        valid &= (pred >= 0) & (pred < self.num_classes)
        indices = self.num_classes * target[valid] + pred[valid]
        self.matrix += np.bincount(indices, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self, ignore_index: int = 255) -> dict[str, Any]:
        matrix = self.matrix.astype(np.float64)
        true_count = matrix.sum(axis=1)
        pred_count = matrix.sum(axis=0)
        correct = np.diag(matrix)
        union = true_count + pred_count - correct

        iou = np.divide(correct, union, out=np.full_like(correct, np.nan), where=union > 0)
        dice_denominator = true_count + pred_count
        dice = np.divide(
            2 * correct,
            dice_denominator,
            out=np.full_like(correct, np.nan),
            where=dice_denominator > 0,
        )
        class_accuracy = np.divide(
            correct,
            true_count,
            out=np.full_like(correct, np.nan),
            where=true_count > 0,
        )
        total = matrix.sum()

        mean_iou_all = float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0
        mean_iou_no_bg = float(np.nanmean(iou[1:])) if np.any(~np.isnan(iou[1:])) else 0.0
        mean_dice_all = float(np.nanmean(dice)) if np.any(~np.isnan(dice)) else 0.0
        mean_dice_no_bg = float(np.nanmean(dice[1:])) if np.any(~np.isnan(dice[1:])) else 0.0

        return {
            "confusion_matrix": self.matrix.copy(),
            "per_class_iou": iou,
            "per_class_dice": dice,
            "per_class_pixels": true_count.astype(np.int64),
            "present_classes_count": int(np.sum(true_count > 0)),
            "mean_iou_all": mean_iou_all,
            "mean_iou_no_background": mean_iou_no_bg,
            "mean_iou": mean_iou_all,
            "mean_dice_all": mean_dice_all,
            "mean_dice_no_background": mean_dice_no_bg,
            "mean_dice": mean_dice_all,
            "pixel_accuracy": float(correct.sum() / total) if total else 0.0,
            "mean_class_accuracy": float(np.nanmean(class_accuracy)) if np.any(~np.isnan(class_accuracy)) else 0.0,
            "ignore_index": ignore_index,
            "background_included": True,
            "absent_class_policy": "exclude",
        }


def save_metrics(metrics: dict[str, Any], json_path: str | Path, csv_path: str | Path | None = None) -> None:
    """Lưu metric thành JSON hợp lệ và tùy chọn xuất CSV per-class."""
    json_path = Path(json_path)
    output: dict[str, Any] = {}

    per_class_iou = metrics.get("per_class_iou")
    per_class_dice = metrics.get("per_class_dice")
    per_class_pixels = metrics.get("per_class_pixels")

    classes_report = []
    if isinstance(per_class_iou, np.ndarray):
        for class_id in range(len(per_class_iou)):
            c_name = VOC_CLASSES[class_id] if class_id < len(VOC_CLASSES) else f"Class {class_id}"
            iou_val = None if np.isnan(per_class_iou[class_id]) else float(per_class_iou[class_id])
            dice_val = None if (per_class_dice is None or np.isnan(per_class_dice[class_id])) else float(per_class_dice[class_id])
            px_val = int(per_class_pixels[class_id]) if per_class_pixels is not None else 0

            classes_report.append({
                "class_id": class_id,
                "class_name": c_name,
                "iou": iou_val,
                "dice": dice_val,
                "pixels": px_val,
            })

    for key, value in metrics.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            output[key] = value.tolist()
        elif isinstance(value, np.ndarray):
            output[key] = [None if np.isnan(item) else float(item) for item in value]
        else:
            output[key] = value

    if classes_report:
        output["classes"] = classes_report

    json_path.write_text(json.dumps(output, indent=2, allow_nan=False, ensure_ascii=False), encoding="utf-8")

    if csv_path is not None and classes_report:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "iou", "dice", "pixels"])
            writer.writeheader()
            for row in classes_report:
                writer.writerow(row)
