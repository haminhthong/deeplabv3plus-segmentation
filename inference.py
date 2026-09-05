from __future__ import annotations

from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from config import IMAGE_MEAN, IMAGE_STD, NUM_CLASSES
from dataset_voc import calculate_letterbox_geometry


def build_model(
    encoder: str = "resnet50",
    encoder_weights: str | None = "imagenet",
    num_classes: int = NUM_CLASSES,
    architecture: str = "deeplabv3plus",
):
    """Khởi tạo mô hình phân đoạn ảnh theo cấu hình thống nhất.

    Các kiến trúc ứng viên được hỗ trợ:
    - deeplabv3plus: DeepLabV3+ với Atrous Spatial Pyramid Pooling (ASPP).
    - unet: U-Net với các kết nối tắt (skip connections).
    - fpn: Feature Pyramid Network cho multi-scale representation.
    """
    arch = architecture.lower()
    if arch in ("deeplabv3plus", "deeplabv3+"):
        return smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None,
        )
    elif arch == "unet":
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None,
        )
    elif arch == "fpn":
        return smp.FPN(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None,
        )
    else:
        raise ValueError(
            f"Kiến trúc không được hỗ trợ: {architecture}. "
            "Lựa chọn hợp lệ: deeplabv3plus, unet, fpn"
        )


def load_checkpoint_model(path: str | Path, device: torch.device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    metadata = checkpoint if isinstance(checkpoint, dict) else {}
    state_dict = metadata.get("model_state_dict", checkpoint)
    encoder = metadata.get("encoder", "resnet50")
    architecture = metadata.get("architecture", "deeplabv3plus")
    num_classes = int(metadata.get("num_classes", NUM_CLASSES))
    model = build_model(encoder, None, num_classes, architecture)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model, metadata


def prepare_image(image: Image.Image, image_size: int):
    if image_size <= 0:
        raise ValueError("Kích thước ảnh đầu vào phải lớn hơn 0")
    _, resized_w, resized_h, pad_left, pad_top, pad_right, pad_bottom = calculate_letterbox_geometry(
        image.width, image.height, image_size, image_size
    )
    resized = TF.resize(image, (resized_h, resized_w), interpolation=transforms.InterpolationMode.BILINEAR)
    padded = TF.pad(resized, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
    tensor = TF.normalize(TF.to_tensor(padded), IMAGE_MEAN, IMAGE_STD)
    return tensor, (pad_left, pad_top, resized_w, resized_h), image.size


def overlay_mask(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Phủ mặt nạ màu lên ảnh RGB."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    output = image.astype(np.float32) * (1 - alpha) + mask.astype(np.float32) * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


@torch.inference_mode()
def predict_with_uncertainty(
    model: torch.nn.Module,
    image: Image.Image,
    image_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Thực hiện suy luận tại độ phân giải gốc và kết xuất bản đồ độ bất định/độ tin cậy.

    Quy trình:
    1. Letterbox ảnh gốc về target image_size
    2. Model forward pass thu được logits
    3. Cắt bỏ vùng padding do letterbox tạo ra
    4. Nội suy logits song tuyến (bilinear) về kích thước ảnh gốc (original_h, original_w)
    5. Softmax trên logits để thu được phân bố xác suất per-pixel
    6. Tính Hard mask (argmax), Max-probability map và Normalized Entropy map

    LƯU Ý KỸ THUẬT:
    Bản đồ này phản ánh độ phân vân/bất định (uncertainty / reliability map) của phân bố Softmax,
    không xem là xác suất Bayes đã hiệu chuẩn (calibrated confidence) trừ khi đã qua calibration.
    """
    was_training = model.training
    if was_training:
        model.eval()
    try:
        tensor, (left, top, resized_w, resized_h), (original_w, original_h) = prepare_image(image, image_size)
        logits = model(tensor.unsqueeze(0).to(device))
        logits = logits[:, :, top : top + resized_h, left : left + resized_w]
        logits = F.interpolate(logits, size=(original_h, original_w), mode="bilinear", align_corners=False)

        probs = F.softmax(logits, dim=1).squeeze(0)  # [C, H, W]
        hard_mask = probs.argmax(dim=0).cpu().numpy().astype(np.int64)
        max_prob = probs.max(dim=0).values.cpu().numpy().astype(np.float32)

        # Normalized Entropy: H = - sum(p * log(p + eps)) / log(num_classes)
        num_classes = probs.shape[0]
        eps = 1e-7
        entropy = -(probs * torch.log(probs + eps)).sum(dim=0)
        norm_factor = float(np.log(max(num_classes, 2)))
        normalized_entropy = (entropy / norm_factor).clamp(0.0, 1.0).cpu().numpy().astype(np.float32)

        return {
            "hard_mask": hard_mask,
            "max_prob_map": max_prob,
            "entropy_map": normalized_entropy,
            "softmax_probs": probs.cpu().numpy().astype(np.float32),
        }
    finally:
        if was_training:
            model.train()


@torch.inference_mode()
def predict_original_size(model, image: Image.Image, image_size: int, device: torch.device) -> np.ndarray:
    """Suy luận trả về hard mask tại kích thước gốc của ảnh (tối ưu tốc độ khi chỉ cần nhãn)."""
    was_training = model.training
    if was_training:
        model.eval()
    try:
        tensor, (left, top, resized_w, resized_h), (original_w, original_h) = prepare_image(image, image_size)
        logits = model(tensor.unsqueeze(0).to(device))
        logits = logits[:, :, top : top + resized_h, left : left + resized_w]
        logits = F.interpolate(logits, size=(original_h, original_w), mode="bilinear", align_corners=False)
        return logits.argmax(1).squeeze(0).cpu().numpy().astype(np.int64)
    finally:
        if was_training:
            model.train()
