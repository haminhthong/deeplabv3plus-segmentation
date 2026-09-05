from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from metrics import (
    SegmentationMetrics,
    calculate_region_size_metrics,
    compute_boundary_f1_score,
    extract_boundary,
    extract_confusion_analysis,
    save_metrics,
)


def test_perfect_prediction():
    prediction = torch.tensor([[0, 1], [1, 0]])
    target = prediction.clone()

    metric = SegmentationMetrics(num_classes=2)
    metric.update(prediction, target)
    result = metric.compute()

    assert result["mean_iou"] == 1.0
    assert result["mean_dice"] == 1.0
    assert result["pixel_accuracy"] == 1.0


def test_ignore_index_255():
    prediction = torch.tensor([[0, 1], [1, 0]])
    target = torch.tensor([[0, 1], [1, 255]])  # 255 là pixel bị bỏ qua

    metric = SegmentationMetrics(num_classes=2)
    metric.update(prediction, target, ignore_index=255)
    result = metric.compute(ignore_index=255)

    assert result["mean_iou"] == 1.0
    assert result["pixel_accuracy"] == 1.0


def test_absent_class_handling():
    # Chỉ có lớp 0 xuất hiện trong target
    prediction = torch.tensor([[0, 0], [0, 0]])
    target = torch.tensor([[0, 0], [0, 0]])

    metric = SegmentationMetrics(num_classes=3)
    metric.update(prediction, target)
    result = metric.compute()

    # Lớp 0 có IoU=1.0, Lớp 1 và 2 vắng mặt (NaN), nanmean chỉ tính trên lớp 0
    assert result["mean_iou_all"] == 1.0
    assert result["present_classes_count"] == 1
    assert np.isnan(result["per_class_iou"][1])


def test_shape_mismatch():
    metric = SegmentationMetrics(num_classes=2)
    with pytest.raises(ValueError, match="cùng kích thước"):
        metric.update(torch.zeros((2, 2)), torch.zeros((3, 3)))


def test_boundary_f1_score():
    # Tạo mask hình vuông 10x10 ở tâm
    gt = np.zeros((30, 30), dtype=np.int64)
    gt[10:20, 10:20] = 1
    pred_perfect = gt.copy()

    scores_perfect = compute_boundary_f1_score(pred_perfect, gt, num_classes=2, radius=2)
    assert scores_perfect[1] == 1.0

    # Lệch 1 pixel: BF1 vẫn cao do trong tolerance radius=2
    pred_shift = np.zeros((30, 30), dtype=np.int64)
    pred_shift[11:21, 11:21] = 1
    scores_shift = compute_boundary_f1_score(pred_shift, gt, num_classes=2, radius=2)
    assert scores_shift[1] > 0.8

    # Cách xa hoàn toàn ngoài tolerance radius=2 -> BF1 = 0
    pred_far = np.zeros((30, 30), dtype=np.int64)
    pred_far[0:5, 0:5] = 1
    scores_far = compute_boundary_f1_score(pred_far, gt, num_classes=2, radius=2)
    assert scores_far[1] == 0.0


def test_extract_boundary():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:4, 1:4] = True
    # Điểm tâm (2, 2) có đủ 4 lân cận -> không phải biên
    b = extract_boundary(mask)
    assert not b[2, 2]
    assert b[1, 2]
    assert b[3, 2]


def test_region_size_metrics():
    gt = np.zeros((100, 100), dtype=np.int64)
    # Vùng nhỏ (< 1024 px): 20x20 = 400 px
    gt[10:30, 10:30] = 1
    pred = gt.copy()

    res = calculate_region_size_metrics(pred, gt, num_classes=2)
    assert res["small_region_miou"] == 1.0
    assert res["small_regions_count"] == 1


def test_confusion_analysis():
    # Giả lập ma trận nhầm lẫn 3 lớp:
    # Lớp 0: background
    # Lớp 1: person (bị nhầm 10 pixel sang 0)
    # Lớp 2: dog
    matrix = np.array([
        [100, 0, 0],
        [10, 50, 0],
        [0, 5, 80],
    ], dtype=np.int64)

    analysis = extract_confusion_analysis(matrix, num_classes=3)
    assert len(analysis["best_classes"]) > 0
    assert len(analysis["top_confusion_pairs"]) > 0
    # Cặp nhầm lẫn lớn nhất là 1 -> 0 (10 pixel)
    top_pair = analysis["top_confusion_pairs"][0]
    assert top_pair["true_class_id"] == 1
    assert top_pair["pred_class_id"] == 0
    assert top_pair["confused_pixels"] == 10


def test_save_metrics(tmp_path: Path):
    metric = SegmentationMetrics(num_classes=2)
    metric.update(torch.tensor([[0, 1]]), torch.tensor([[0, 1]]), compute_boundary=True)
    res = metric.compute()

    json_file = tmp_path / "metrics.json"
    csv_file = tmp_path / "metrics.csv"
    save_metrics(res, json_file, csv_file)

    assert json_file.exists()
    assert csv_file.exists()

    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert data["mean_iou"] == 1.0
    assert "classes" in data
    assert "best_classes" in data
    assert "top_confusion_pairs" in data
