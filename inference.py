from __future__ import annotations

from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from config import NUM_CLASSES

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def load_checkpoint_model(path: str | Path, device: torch.device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    state_dict = metadata.get("model_state_dict", checkpoint)
    encoder = metadata.get("encoder", "resnet50")
    num_classes = int(metadata.get("num_classes", NUM_CLASSES))
    model = smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=None,
        classes=num_classes,
        activation=None,
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, metadata


def prepare_image(image: Image.Image, image_size: int):
    if image_size <= 0:
        raise ValueError("Kích thước ảnh đầu vào phải lớn hơn 0")
    original_size = image.size
    scale = min(image_size / image.width, image_size / image.height)
    resized_w, resized_h = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    resized = TF.resize(image, (resized_h, resized_w), interpolation=transforms.InterpolationMode.BILINEAR)
    left, top = (image_size - resized_w) // 2, (image_size - resized_h) // 2
    padded = TF.pad(resized, [left, top, image_size - resized_w - left, image_size - resized_h - top], fill=0)
    tensor = TF.normalize(TF.to_tensor(padded), MEAN, STD)
    return tensor, (left, top, resized_w, resized_h), original_size


@torch.inference_mode()
def predict_original_size(model, image: Image.Image, image_size: int, device: torch.device) -> np.ndarray:
    tensor, (left, top, resized_w, resized_h), (original_w, original_h) = prepare_image(image, image_size)
    logits = model(tensor.unsqueeze(0).to(device))
    logits = logits[:, :, top : top + resized_h, left : left + resized_w]
    logits = F.interpolate(logits, size=(original_h, original_w), mode="bilinear", align_corners=False)
    return logits.argmax(1).squeeze(0).cpu().numpy().astype(np.int64)
