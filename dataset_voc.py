from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IGNORE_INDEX, IMAGE_MEAN, IMAGE_STD


def read_split_ids(root: Path | str, split: str) -> list[str]:
    """Đọc danh sách mã ảnh và kiểm tra dữ liệu cơ bản."""
    root = Path(root)
    split_file = root / "ImageSets" / "Segmentation" / f"{split}.txt"
    if not split_file.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp chia dữ liệu: {split_file}")
    ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"Tệp chia dữ liệu không chứa mã ảnh: {split_file}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Tệp chia dữ liệu chứa mã ảnh trùng lặp: {split_file}")
    return ids


def validate_voc_dataset(root: Path | str) -> None:
    """Kiểm tra split, file ảnh/mặt nạ và rò rỉ giữa train với val."""
    root = Path(root)
    train_ids = read_split_ids(root, "train")
    val_ids = read_split_ids(root, "val")
    overlap = set(train_ids).intersection(val_ids)
    if overlap:
        raise ValueError(f"Train và val bị trùng {len(overlap)} ảnh")

    missing = []
    for image_id in train_ids + val_ids:
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
    def __init__(self, h: int, w: int) -> None:
        self.h = h
        self.w = w
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.normalize = transforms.Normalize(IMAGE_MEAN, IMAGE_STD)

    def __call__(self, image: Image.Image, mask: Image.Image):
        # Thay đổi tỷ lệ ngẫu nhiên rồi cắt hoặc đệm để không làm méo đối tượng.
        scale = random.uniform(0.75, 1.5)
        scaled_h = max(1, round(image.height * scale))
        scaled_w = max(1, round(image.width * scale))
        image = TF.resize(image, (scaled_h, scaled_w), interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (scaled_h, scaled_w), interpolation=transforms.InterpolationMode.NEAREST)
        pad_right = max(0, self.w - scaled_w)
        pad_bottom = max(0, self.h - scaled_h)
        if pad_right or pad_bottom:
            image = TF.pad(image, [0, 0, pad_right, pad_bottom], fill=0)
            mask = TF.pad(mask, [0, 0, pad_right, pad_bottom], fill=IGNORE_INDEX)
        top, left, _, _ = transforms.RandomCrop.get_params(image, (self.h, self.w))
        image = TF.crop(image, top, left, self.h, self.w)
        mask = TF.crop(mask, top, left, self.h, self.w)

        # Lật ngang ảnh và mặt nạ cùng lúc.
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Xoay, tịnh tiến và thu phóng nhẹ để tăng độ đa dạng của dữ liệu.
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
            # Nhãn 255 là vùng không được tính vào hàm mất mát và chỉ số đánh giá.
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=0,
                interpolation=transforms.InterpolationMode.NEAREST,
                fill=IGNORE_INDEX,
            )

        # Chỉ thay đổi màu ảnh vì mặt nạ chứa mã lớp.
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
    """Đổi kích thước theo đúng tỷ lệ rồi đệm tới kích thước yêu cầu."""
    if h <= 0 or w <= 0:
        raise ValueError("Chiều cao và chiều rộng phải lớn hơn 0")
    if image.size != mask.size:
        raise ValueError(f"Ảnh và mặt nạ phải cùng kích thước: {image.size} != {mask.size}")
    scale = min(w / image.width, h / image.height)
    new_w = max(1, round(image.width * scale))
    new_h = max(1, round(image.height * scale))
    image = TF.resize(image, (new_h, new_w), interpolation=transforms.InterpolationMode.BILINEAR)
    mask = TF.resize(mask, (new_h, new_w), interpolation=transforms.InterpolationMode.NEAREST)
    pad_left = (w - new_w) // 2
    pad_top = (h - new_h) // 2
    padding = [pad_left, pad_top, w - new_w - pad_left, h - new_h - pad_top]
    return TF.pad(image, padding, fill=0), TF.pad(mask, padding, fill=IGNORE_INDEX)


class VOCSegmentationDataset(Dataset):
    def __init__(self, root: Path | str, split: str = "val", joint_transform=None) -> None:
        self.root = Path(root)
        self.split = split
        self.joint_transform = joint_transform

        self.jpeg_dir = self.root / "JPEGImages"
        self.mask_dir = self.root / "SegmentationClass"
        self.ids = read_split_ids(self.root, split)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        image = Image.open(self.jpeg_dir / f"{sid}.jpg").convert("RGB")
        mask = Image.open(self.mask_dir / f"{sid}.png")

        if self.joint_transform is not None:
            image_t, mask_t = self.joint_transform(image, mask)
        else:
            image_t = transforms.ToTensor()(image)
            mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image_t, mask_t
