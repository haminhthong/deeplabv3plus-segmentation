import numpy as np
import torch

from metrics import SegmentationMetrics


def test_perfect_prediction_ignores_void_pixels():
    metric = SegmentationMetrics(3)
    metric.update(torch.tensor([[0, 1], [2, 0]]), torch.tensor([[0, 1], [2, 255]]))
    result = metric.compute()
    assert result["mean_iou"] == 1.0
    assert result["mean_dice"] == 1.0
    assert result["pixel_accuracy"] == 1.0
    np.testing.assert_allclose(result["per_class_iou"], [1.0, 1.0, 1.0])


def test_known_binary_confusion_matrix():
    metric = SegmentationMetrics(2)
    metric.update(torch.tensor([0, 1, 1, 1]), torch.tensor([0, 0, 1, 1]))
    result = metric.compute()
    np.testing.assert_array_equal(result["confusion_matrix"], [[1, 1], [0, 2]])
    np.testing.assert_allclose(result["per_class_iou"], [0.5, 2 / 3])
