from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from inference import build_model, overlay_mask, predict_original_size, prepare_image


class DummySegmentationModel(nn.Module):
    def __init__(self, num_classes: int = 21):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        return torch.zeros((b, self.num_classes, h, w), dtype=torch.float32)


@pytest.mark.parametrize(
    "width, height",
    [
        (640, 480),  # Ảnh ngang
        (480, 640),  # Ảnh dọc
        (500, 500),  # Ảnh vuông
        (32, 32),    # Ảnh rất nhỏ
    ],
)
def test_predict_original_size_returns_correct_shape(width: int, height: int):
    model = DummySegmentationModel(num_classes=21)
    image = Image.new("RGB", (width, height), color="red")
    device = torch.device("cpu")

    pred_mask = predict_original_size(model, image, image_size=320, device=device)

    assert pred_mask.shape == (height, width)
    assert pred_mask.dtype == np.int64


def test_prepare_image_invalid_size():
    image = Image.new("RGB", (100, 100))
    with pytest.raises(ValueError, match="lớn hơn 0"):
        prepare_image(image, 0)


def test_overlay_mask_shape_and_range():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.ones((100, 100, 3), dtype=np.uint8) * 255
    res = overlay_mask(img, mask, alpha=0.5)

    assert res.shape == (100, 100, 3)
    assert res.dtype == np.uint8
    assert res[0, 0, 0] == 127


def test_build_model_baseline():
    deeplab = build_model("resnet50", None, 21, "deeplabv3plus")
    unet = build_model("resnet50", None, 21, "unet")
    fcn = build_model("resnet50", None, 21, "fcn")

    assert deeplab is not None
    assert unet is not None
    assert fcn is not None

    with pytest.raises(ValueError, match="không được hỗ trợ"):
        build_model("resnet50", None, 21, "invalid_arch")
