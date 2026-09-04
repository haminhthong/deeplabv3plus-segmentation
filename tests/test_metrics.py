from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from metrics import SegmentationMetrics, save_metrics


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


def test_save_metrics(tmp_path: Path):
    metric = SegmentationMetrics(num_classes=2)
    metric.update(torch.tensor([[0, 1]]), torch.tensor([[0, 1]]))
    res = metric.compute()

    json_file = tmp_path / "metrics.json"
    csv_file = tmp_path / "metrics.csv"
    save_metrics(res, json_file, csv_file)

    assert json_file.exists()
    assert csv_file.exists()

    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert data["mean_iou"] == 1.0
    assert "classes" in data
