from __future__ import annotations

from pathlib import Path

import random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class JointTransform:
    def __init__(self, h: int, w: int) -> None:
        self.h = h
        self.w = w
        self.img_resize = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.BILINEAR)
        self.mask_resize = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __call__(self, image: Image.Image, mask: Image.Image):
        image = self.img_resize(image)
        mask = self.mask_resize(mask)
        image_t = self.normalize(self.to_tensor(image))
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_t, mask_t


class TrainJointTransform:
    def __init__(self, h: int, w: int) -> None:
        self.h = h
        self.w = w
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __call__(self, image: Image.Image, mask: Image.Image):
        # Thay đổi kích thước (Resize)
        image = TF.resize(image, (self.h, self.w), interpolation=transforms.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.h, self.w), interpolation=transforms.InterpolationMode.NEAREST)

        # Lật ngang ngẫu nhiên
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Biến đổi Affine ngẫu nhiên (Xoay, Tịnh tiến, Thu phóng)
        if random.random() > 0.5:
            # Thu hẹp độ nhiễu để tránh làm mờ hình / mất cấu trúc (Scale 0.9-1.1, góc nhỏ)
            angle = random.uniform(-10.0, 10.0)
            translate = [int(random.uniform(-0.05, 0.05) * self.w), int(random.uniform(-0.05, 0.05) * self.h)]
            scale = random.uniform(0.9, 1.1)
            # 255 là chỉ số bỏ qua (ignore index) cho nhãn nền của VOC
            image = TF.affine(image, angle=angle, translate=translate, scale=scale, shear=0, interpolation=transforms.InterpolationMode.BILINEAR)
            mask = TF.affine(mask, angle=angle, translate=translate, scale=scale, shear=0, interpolation=transforms.InterpolationMode.NEAREST, fill=255)

        # Thay đổi màu sắc ngẫu nhiên (chỉ áp dụng cho ảnh)
        if random.random() > 0.5:
            image = self.color_jitter(image)
        
        # Chuyển thành Tensor
        image_t = self.normalize(TF.to_tensor(image))
        mask_t = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image_t, mask_t


def get_val_transforms(h: int, w: int):
    return JointTransform(h, w)


def get_train_transforms(h: int, w: int):
    return TrainJointTransform(h, w)


class VOCSegmentationDataset(Dataset):
    def __init__(self, root: Path | str, split: str = "val", joint_transform=None) -> None:
        self.root = Path(root)
        self.split = split
        self.joint_transform = joint_transform

        self.jpeg_dir = self.root / "JPEGImages"
        self.mask_dir = self.root / "SegmentationClass"
        split_file = self.root / "ImageSets" / "Segmentation" / f"{split}.txt"
        self.ids = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]

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

