"""Dataset và Data Augmentation cho Pascal VOC Semantic Segmentation.

QUY ƯỚC TIỀN XỬ LÝ (TRANSFORM CONTRACT):
- Training pipeline: Sử dụng joint augmentation ngẫu nhiên (scale 0.75-1.5, đệm ngẫu nhiên không thiên lệch góc,
  crop về target (h, w), lật ngang, affine nhẹ, color jitter chỉ trên ảnh RGB).
- Validation & Serving pipeline: Sử dụng deterministic letterbox (giữ nguyên tỷ lệ khung hình, đệm đều vào giữa
  về target (h, w), không crop mất thông tin).
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IGNORE_INDEX, IMAGE_MEAN, IMAGE_STD, VOC_ROOT


def calculate_letterbox_geometry(
    width: int,
    height: int,
    target_width: int,
    target_height: int,
) -> tuple[float, int, int, int, int, int, int]:
    """Tính toán thông số hình học resize và đệm (letterbox) dùng chung giữa training và inference."""
    if width <= 0 or height <= 0 or target_width <= 0 or target_height <= 0:
        raise ValueError("Chiều cao và chiều rộng phải lớn hơn 0")
    scale = min(target_width / width, target_height / height)
    new_w = max(1, round(width * scale))
    new_h = max(1, round(height * scale))
    pad_left = (target_width - new_w) // 2
    pad_top = (target_height - new_h) // 2
    pad_right = target_width - new_w - pad_left
    pad_bottom = target_height - new_h - pad_top
    return scale, new_w, new_h, pad_left, pad_top, pad_right, pad_bottom


def read_split_ids(
    root: Path | str,
    split: str,
    split_dir: Path | str | None = None,
    split_type: str = "benchmark",
) -> list[str]:
    """Đọc danh sách mã ảnh và kiểm tra tính hợp lệ dữ liệu.

    Ưu tiên tìm split file:
    1. split_dir chỉ định tường minh.
    2. root/splits/{split_type}/{split}.txt
    3. root/ImageSets/Segmentation/{split}.txt (VOC gốc)
    4. root/splits/{split}.txt
    5. splits/{split_type}/{split}.txt (workspace splits)
    6. splits/{split}.txt (legacy workspace splits)
    """
    root = Path(root)

    split_file = None
    if split_dir is not None:
        cand = Path(split_dir) / f"{split}.txt"
        if cand.is_file():
            split_file = cand

    if split_file is None:
        cand_sub = root / "splits" / split_type / f"{split}.txt"
        cand_voc = root / "ImageSets" / "Segmentation" / f"{split}.txt"
        cand_flat = root / "splits" / f"{split}.txt"
        cand_ws_sub = Path("splits") / split_type / f"{split}.txt"
        cand_ws_flat = Path("splits") / f"{split}.txt"

        if cand_sub.is_file():
            split_file = cand_sub
        elif cand_voc.is_file():
            split_file = cand_voc
        elif cand_flat.is_file():
            split_file = cand_flat
        elif (root == Path(".") or root == VOC_ROOT or root == Path("data")) and cand_ws_sub.is_file():
            split_file = cand_ws_sub
        elif (root == Path(".") or root == VOC_ROOT or root == Path("data")) and cand_ws_flat.is_file():
            split_file = cand_ws_flat

    if split_file is None or not split_file.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy tệp chia dữ liệu cho split='{split}' (split_type='{split_type}') tại {root} hoặc splits/"
        )

    ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"Tệp chia dữ liệu không chứa mã ảnh: {split_file}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Tệp chia dữ liệu chứa mã ảnh trùng lặp: {split_file}")
    return ids


def validate_voc_dataset(
    root: Path | str,
    split_dir: Path | str | None = None,
    split_type: str = "benchmark",
) -> None:
    """Kiểm tra split, file ảnh/mặt nạ và rò rỉ giữa train/val/test."""
    root = Path(root)
    splits_to_check = []
    for s in ["train", "val", "test"]:
        try:
            ids = read_split_ids(root, s, split_dir=split_dir, split_type=split_type)
            splits_to_check.append((s, ids))
        except FileNotFoundError:
            pass

    if not splits_to_check:
        raise FileNotFoundError("Không tìm thấy tệp chia dữ liệu nào")

    for i in range(len(splits_to_check)):
        for j in range(i + 1, len(splits_to_check)):
            s1_name, s1_ids = splits_to_check[i]
            s2_name, s2_ids = splits_to_check[j]
            overlap = set(s1_ids).intersection(s2_ids)
            if overlap:
                raise ValueError(f"Split {s1_name} và {s2_name} bị trùng {len(overlap)} ảnh")

    missing = []
    all_ids = set()
    for _, ids in splits_to_check:
        all_ids.update(ids)

    for image_id in all_ids:
        for path in (
            root / "JPEGImages" / f"{image_id}.jpg",
            root / "SegmentationClass" / f"{image_id}.png",
        ):
            if not path.is_file():
                missing.append(path)
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"Thiếu {len(missing)} file dữ liệu, ví dụ:\n{preview}")


class JointTransform:
    """Deterministic letterbox transform dùng cho validation và inference."""

    def __init__(self, h: int, w: int) -> None:
        self.h = h
        self.w = w
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(IMAGE_MEAN, IMAGE_STD)

    def __call__(self, image: Image.Image, mask: Image.Image):
        image, mask = resize_and_pad(image, mask, self.h, self.w)
        image_t = self.normalize(self.to_tensor(image))
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_t, mask_t


class TrainJointTransform:
    """Joint augmentations cho quá trình huấn luyện:
    - Random scale [0.75, 1.5]
    - Unbiased padding: phân bố lề ngẫu nhiên khi kích thước thu nhỏ, tránh thiên lệch góc trên-trái
    - Random crop về target (h, w)
    - Random horizontal flip (p=0.5)
    - Random affine: xoay +/- 10 độ, tịnh tiến +/- 5%, scale +/- 10%
    - Mild Color Jitter (chỉ áp dụng trên ảnh RGB)
    - Mask luôn nội suy NEAREST và điền IGNORE_INDEX=255
    """

    def __init__(self, h: int, w: int) -> None:
        self.h = h
        self.w = w
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.normalize = transforms.Normalize(IMAGE_MEAN, IMAGE_STD)

    def __call__(self, image: Image.Image, mask: Image.Image):
        scale = random.uniform(0.75, 1.5)
        scaled_h = max(1, round(image.height * scale))
        scaled_w = max(1, round(image.width * scale))
        image = TF.resize(image, (scaled_h, scaled_w), interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (scaled_h, scaled_w), interpolation=transforms.InterpolationMode.NEAREST)

        pad_total_w = max(0, self.w - scaled_w)
        pad_total_h = max(0, self.h - scaled_h)
        if pad_total_w > 0 or pad_total_h > 0:
            # Random/unbiased padding: không neo cứng về góc trên-trái
            pad_left = random.randint(0, pad_total_w) if pad_total_w > 0 else 0
            pad_right = pad_total_w - pad_left
            pad_top = random.randint(0, pad_total_h) if pad_total_h > 0 else 0
            pad_bottom = pad_total_h - pad_top
            image = TF.pad(image, [pad_left, pad_top, pad_right, pad_bottom], fill=0)
            mask = TF.pad(mask, [pad_left, pad_top, pad_right, pad_bottom], fill=IGNORE_INDEX)

        top, left, _, _ = transforms.RandomCrop.get_params(image, (self.h, self.w))
        image = TF.crop(image, top, left, self.h, self.w)
        mask = TF.crop(mask, top, left, self.h, self.w)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() > 0.5:
            angle = random.uniform(-10.0, 10.0)
            translate = [
                int(random.uniform(-0.05, 0.05) * self.w),
                int(random.uniform(-0.05, 0.05) * self.h),
            ]
            scale = random.uniform(0.9, 1.1)
            image = TF.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=0,
                interpolation=transforms.InterpolationMode.BILINEAR,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=0,
                interpolation=transforms.InterpolationMode.NEAREST,
                fill=IGNORE_INDEX,
            )

        if random.random() > 0.5:
            image = self.color_jitter(image)
        image_t = self.normalize(TF.to_tensor(image))
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_t, mask_t


def get_val_transforms(h: int, w: int):
    return JointTransform(h, w)


def get_train_transforms(h: int, w: int):
    return TrainJointTransform(h, w)


def resize_and_pad(image: Image.Image, mask: Image.Image, h: int, w: int):
    """Đổi kích thước theo đúng tỷ lệ letterbox rồi đệm tới kích thước yêu cầu."""
    if image.size != mask.size:
        raise ValueError(f"Ảnh và mặt nạ phải cùng kích thước: {image.size} != {mask.size}")
    _, new_w, new_h, pad_left, pad_top, pad_right, pad_bottom = calculate_letterbox_geometry(
        image.width, image.height, w, h
    )
    image = TF.resize(image, (new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)
    mask = TF.resize(mask, (new_h, new_w), interpolation=transforms.InterpolationMode.NEAREST)
    padding = [pad_left, pad_top, pad_right, pad_bottom]
    return TF.pad(image, padding, fill=0), TF.pad(mask, padding, fill=IGNORE_INDEX)


class VOCSegmentationDataset(Dataset):
    def __init__(
        self,
        root: Path | str,
        split: str = "val",
        joint_transform=None,
        split_dir: Path | str | None = None,
        split_type: str = "benchmark",
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.joint_transform = joint_transform
        self.split_dir = split_dir
        self.split_type = split_type

        self.jpeg_dir = self.root / "JPEGImages"
        self.mask_dir = self.root / "SegmentationClass"
        self.ids = read_split_ids(self.root, split, split_dir=split_dir, split_type=split_type)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        with Image.open(self.jpeg_dir / f"{sid}.jpg") as source:
            image = source.convert("RGB")
        with Image.open(self.mask_dir / f"{sid}.png") as source:
            mask = source.copy()

        if self.joint_transform is not None:
            image_t, mask_t = self.joint_transform(image, mask)
        else:
            image_t = transforms.ToTensor()(image)
            mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image_t, mask_t
