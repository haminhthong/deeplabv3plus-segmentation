from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from voc_meta import VOC_CLASSES


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """Trích xuất đường biên của mặt nạ nhị phân 2D bằng phép co hình thái học (morphological erosion)."""
    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    p = np.pad(mask, ((1, 1), ((1, 1))), mode="constant", constant_values=False)
    eroded = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return mask & (~eroded)


def dilate_boundary(boundary: np.ndarray, radius: int = 2) -> np.ndarray:
    """Giãn nở đường biên nhị phân theo bán kính pixel tolerance."""
    if radius <= 0 or not boundary.any():
        return boundary
    current = boundary.copy()
    for _ in range(radius):
        p = np.pad(current, ((1, 1), ((1, 1))), mode="constant", constant_values=False)
        current = p[1:-1, 1:-1] | p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:]
    return current


def compute_boundary_f1_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    num_classes: int,
    radius: int = 2,
    ignore_index: int = 255,
) -> dict[int, float]:
    """Tính Boundary F1 (BF-score) cho từng lớp ngữ nghĩa trên một ảnh (hoặc batch)."""
    scores: dict[int, float] = {}
    valid_mask = gt_mask != ignore_index

    for c in range(num_classes):
        pred_c = (pred_mask == c) & valid_mask
        gt_c = (gt_mask == c) & valid_mask

        gt_count = int(gt_c.sum())
        pred_count = int(pred_c.sum())

        if gt_count == 0 and pred_count == 0:
            continue
        if gt_count == 0 or pred_count == 0:
            scores[c] = 0.0
            continue

        b_pred = extract_boundary(pred_c)
        b_gt = extract_boundary(gt_c)

        n_pred = int(b_pred.sum())
        n_gt = int(b_gt.sum())

        if n_pred == 0 and n_gt == 0:
            scores[c] = 1.0
            continue
        if n_pred == 0 or n_gt == 0:
            scores[c] = 0.0
            continue

        b_gt_dilated = dilate_boundary(b_gt, radius=radius)
        b_pred_dilated = dilate_boundary(b_pred, radius=radius)

        precision = float((b_pred & b_gt_dilated).sum()) / n_pred
        recall = float((b_gt & b_pred_dilated).sum()) / n_gt

        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        scores[c] = float(np.clip(f1, 0.0, 1.0))

    return scores


def calculate_region_size_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> dict[str, Any]:
    """Phân khúc đánh giá mIoU theo kích thước vùng đối tượng Ground Truth:
    - Small: < 32^2 (1,024 px)
    - Medium: 32^2 đến 96^2 (1,024 - 9,216 px)
    - Large: >= 96^2 (>= 9,216 px)
    """
    valid = targets != ignore_index
    small_ious: list[float] = []
    medium_ious: list[float] = []
    large_ious: list[float] = []

    # Xét các lớp đối tượng tiền cảnh (foreground classes >= 1)
    for c in range(1, num_classes):
        gt_c = (targets == c) & valid
        pred_c = (predictions == c) & valid

        area = int(gt_c.sum())
        if area == 0:
            continue

        intersection = int((gt_c & pred_c).sum())
        union = int((gt_c | pred_c).sum())
        iou = float(intersection / union) if union > 0 else 0.0

        if area < 1024:
            small_ious.append(iou)
        elif area < 9216:
            medium_ious.append(iou)
        else:
            large_ious.append(iou)

    return {
        "small_region_miou": float(np.mean(small_ious)) if small_ious else None,
        "medium_region_miou": float(np.mean(medium_ious)) if medium_ious else None,
        "large_region_miou": float(np.mean(large_ious)) if large_ious else None,
        "small_regions_count": len(small_ious),
        "medium_regions_count": len(medium_ious),
        "large_regions_count": len(large_ious),
    }


def extract_confusion_analysis(
    matrix: np.ndarray,
    num_classes: int,
    top_k_pairs: int = 5,
) -> dict[str, Any]:
    """Trích xuất Best 5, Worst 5 classes và Top confusion pairs từ Confusion Matrix."""
    mat = matrix.astype(np.float64)
    true_count = mat.sum(axis=1)
    pred_count = mat.sum(axis=0)
    correct = np.diag(mat)
    union = true_count + pred_count - correct

    iou = np.divide(
        correct,
        union,
        out=np.full_like(correct, np.nan, dtype=np.float64),
        where=union > 0,
    )

    # Đánh giá trên các lớp có xuất hiện trong GT
    valid_classes = [c for c in range(num_classes) if true_count[c] > 0]

    # Best 5 & Worst 5 (ưu tiên tiền cảnh nếu có)
    fg_valid = [c for c in valid_classes if c > 0]
    eval_classes = fg_valid if fg_valid else valid_classes

    sorted_by_iou = sorted(
        eval_classes,
        key=lambda c: (iou[c] if not np.isnan(iou[c]) else -1.0),
        reverse=True,
    )

    best_5 = [
        {
            "class_id": c,
            "class_name": VOC_CLASSES[c] if c < len(VOC_CLASSES) else f"Class {c}",
            "iou": float(iou[c]),
            "pixels": int(true_count[c]),
        }
        for c in sorted_by_iou[:5]
    ]

    worst_5 = [
        {
            "class_id": c,
            "class_name": VOC_CLASSES[c] if c < len(VOC_CLASSES) else f"Class {c}",
            "iou": float(iou[c]),
            "pixels": int(true_count[c]),
        }
        for c in sorted_by_iou[::-1][:5]
    ]

    # Confusion Pairs (cặp lớp nhầm lẫn nhiều nhất off-diagonal)
    pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and matrix[i, j] > 0 and true_count[i] > 0:
                pairs.append({
                    "true_class_id": i,
                    "true_class_name": VOC_CLASSES[i] if i < len(VOC_CLASSES) else f"Class {i}",
                    "pred_class_id": j,
                    "pred_class_name": VOC_CLASSES[j] if j < len(VOC_CLASSES) else f"Class {j}",
                    "confused_pixels": int(matrix[i, j]),
                    "percent_of_true": float((matrix[i, j] / true_count[i]) * 100.0),
                })

    pairs.sort(key=lambda p: p["confused_pixels"], reverse=True)
    top_pairs = pairs[:top_k_pairs]

    return {
        "best_classes": best_5,
        "worst_classes": worst_5,
        "top_confusion_pairs": top_pairs,
    }


class SegmentationMetrics:
    def __init__(self, num_classes: int):
        if num_classes <= 0:
            raise ValueError("Số lớp phải lớn hơn 0")
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self.boundary_scores: dict[int, list[float]] = {c: [] for c in range(num_classes)}
        self.region_records: list[dict[str, Any]] = []

    def update(
        self,
        predictions: torch.Tensor | np.ndarray,
        targets: torch.Tensor | np.ndarray,
        ignore_index: int = 255,
        compute_boundary: bool = False,
    ) -> None:
        if isinstance(predictions, torch.Tensor):
            pred_arr = predictions.detach().cpu().numpy()
        else:
            pred_arr = np.asarray(predictions)

        if isinstance(targets, torch.Tensor):
            target_arr = targets.detach().cpu().numpy()
        else:
            target_arr = np.asarray(targets)

        if pred_arr.shape != target_arr.shape:
            raise ValueError(
                "Dự đoán và nhãn thật phải cùng kích thước: "
                f"{tuple(pred_arr.shape)} != {tuple(target_arr.shape)}"
            )

        pred_flat = pred_arr.reshape(-1)
        target_flat = target_arr.reshape(-1)
        valid = (target_flat != ignore_index) & (target_flat >= 0) & (target_flat < self.num_classes)
        valid &= (pred_flat >= 0) & (pred_flat < self.num_classes)
        indices = self.num_classes * target_flat[valid] + pred_flat[valid]
        self.matrix += np.bincount(indices, minlength=self.num_classes**2).reshape(
            self.num_classes, self.num_classes
        )

        if compute_boundary:
            # Nếu truyền vào batch 3D (B, H, W) hoặc 2D (H, W)
            if pred_arr.ndim == 3:
                for b in range(pred_arr.shape[0]):
                    sample_b_scores = compute_boundary_f1_score(
                        pred_arr[b], target_arr[b], self.num_classes, radius=2, ignore_index=ignore_index
                    )
                    for c, val in sample_b_scores.items():
                        self.boundary_scores[c].append(val)
                    self.region_records.append(
                        calculate_region_size_metrics(pred_arr[b], target_arr[b], self.num_classes, ignore_index)
                    )
            elif pred_arr.ndim == 2:
                sample_b_scores = compute_boundary_f1_score(
                    pred_arr, target_arr, self.num_classes, radius=2, ignore_index=ignore_index
                )
                for c, val in sample_b_scores.items():
                    self.boundary_scores[c].append(val)
                self.region_records.append(
                    calculate_region_size_metrics(pred_arr, target_arr, self.num_classes, ignore_index)
                )

    def compute(self, ignore_index: int = 255) -> dict[str, Any]:
        matrix = self.matrix.astype(np.float64)
        true_count = matrix.sum(axis=1)
        pred_count = matrix.sum(axis=0)
        correct = np.diag(matrix)
        union = true_count + pred_count - correct

        iou = np.divide(correct, union, out=np.full_like(correct, np.nan, dtype=np.float64), where=union > 0)
        dice_denominator = true_count + pred_count
        dice = np.divide(
            2 * correct,
            dice_denominator,
            out=np.full_like(correct, np.nan, dtype=np.float64),
            where=dice_denominator > 0,
        )
        class_accuracy = np.divide(
            correct,
            true_count,
            out=np.full_like(correct, np.nan, dtype=np.float64),
            where=true_count > 0,
        )
        total = matrix.sum()

        mean_iou_all = float(np.nanmean(iou)) if np.any(~np.isnan(iou)) else 0.0
        mean_iou_no_bg = float(np.nanmean(iou[1:])) if np.any(~np.isnan(iou[1:])) else 0.0
        mean_dice_all = float(np.nanmean(dice)) if np.any(~np.isnan(dice)) else 0.0
        mean_dice_no_bg = float(np.nanmean(dice[1:])) if np.any(~np.isnan(dice[1:])) else 0.0

        error_analysis = extract_confusion_analysis(self.matrix, self.num_classes)

        # Boundary F1 aggregation
        per_class_boundary = np.full(self.num_classes, np.nan, dtype=np.float64)
        for c in range(self.num_classes):
            if self.boundary_scores[c]:
                per_class_boundary[c] = float(np.mean(self.boundary_scores[c]))

        mean_bf1_all = float(np.nanmean(per_class_boundary)) if np.any(~np.isnan(per_class_boundary)) else None
        mean_bf1_no_bg = float(np.nanmean(per_class_boundary[1:])) if np.any(~np.isnan(per_class_boundary[1:])) else None

        # Region size aggregation
        small_list = [r["small_region_miou"] for r in self.region_records if r.get("small_region_miou") is not None]
        med_list = [r["medium_region_miou"] for r in self.region_records if r.get("medium_region_miou") is not None]
        large_list = [r["large_region_miou"] for r in self.region_records if r.get("large_region_miou") is not None]

        region_slices = {
            "small_region_miou": float(np.mean(small_list)) if small_list else None,
            "medium_region_miou": float(np.mean(med_list)) if med_list else None,
            "large_region_miou": float(np.mean(large_list)) if large_list else None,
        }

        return {
            "confusion_matrix": self.matrix.copy(),
            "per_class_iou": iou,
            "per_class_dice": dice,
            "per_class_boundary_f1": per_class_boundary,
            "per_class_pixels": true_count.astype(np.int64),
            "present_classes_count": int(np.sum(true_count > 0)),
            # Headline metrics
            "mean_iou_all": mean_iou_all,
            "mean_iou_no_background": mean_iou_no_bg,
            "mean_iou": mean_iou_all,
            # Supporting metrics
            "mean_dice_all": mean_dice_all,
            "mean_dice_no_background": mean_dice_no_bg,
            "mean_dice": mean_dice_all,
            "pixel_accuracy": float(correct.sum() / total) if total else 0.0,
            "mean_class_accuracy": float(np.nanmean(class_accuracy)) if np.any(~np.isnan(class_accuracy)) else 0.0,
            # Structural & Boundary metrics
            "boundary_f1_all": mean_bf1_all,
            "boundary_f1_no_background": mean_bf1_no_bg,
            # Error analysis & region slicing
            "region_size_slices": region_slices,
            "best_classes": error_analysis["best_classes"],
            "worst_classes": error_analysis["worst_classes"],
            "top_confusion_pairs": error_analysis["top_confusion_pairs"],
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
    per_class_boundary = metrics.get("per_class_boundary_f1")
    per_class_pixels = metrics.get("per_class_pixels")

    classes_report = []
    if isinstance(per_class_iou, np.ndarray):
        for class_id in range(len(per_class_iou)):
            c_name = VOC_CLASSES[class_id] if class_id < len(VOC_CLASSES) else f"Class {class_id}"
            iou_val = None if np.isnan(per_class_iou[class_id]) else float(per_class_iou[class_id])
            dice_val = None if (per_class_dice is None or np.isnan(per_class_dice[class_id])) else float(per_class_dice[class_id])
            bf1_val = None if (per_class_boundary is None or np.isnan(per_class_boundary[class_id])) else float(per_class_boundary[class_id])
            px_val = int(per_class_pixels[class_id]) if per_class_pixels is not None else 0

            classes_report.append({
                "class_id": class_id,
                "class_name": c_name,
                "iou": iou_val,
                "dice": dice_val,
                "boundary_f1": bf1_val,
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

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2, allow_nan=False, ensure_ascii=False), encoding="utf-8")

    if csv_path is not None and classes_report:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["class_id", "class_name", "iou", "dice", "boundary_f1", "pixels"])
            writer.writeheader()
            for row in classes_report:
                writer.writerow(row)
