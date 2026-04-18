from __future__ import annotations

import numpy as np

VOC_CLASSES = [
    "Background (Nền)",
    "Aeroplane (Máy bay)",
    "Bicycle (Xe đạp)",
    "Bird (Chim)",
    "Boat (Thuyền)",
    "Bottle (Chai lọ)",
    "Bus (Xe buýt)",
    "Car (Ô tô)",
    "Cat (Mèo)",
    "Chair (Ghế)",
    "Cow (Bò)",
    "Diningtable (Bàn ăn)",
    "Dog (Chó)",
    "Horse (Ngựa)",
    "Motorbike (Xe máy)",
    "Person (Người)",
    "Pottedplant (Cây chậu)",
    "Sheep (Cừu)",
    "Sofa (Sofa)",
    "Train (Tàu hỏa)",
    "Tvmonitor (Tivi/Màn hình)",
]

VOC_CLASS_NAMES = {idx: name for idx, name in enumerate(VOC_CLASSES)}

VOC_COLORMAP = np.array(
    [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ],
    dtype=np.uint8,
)


def mask_to_color_rgb(mask: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """
    Chuyển đổi một mask số nguyên 2D (H, W) thành một ảnh RGB (H, W, 3).
    Các pixel thuộc `ignore_index` được ánh xạ thành màu trắng [255, 255, 255].
    """
    valid = mask < len(VOC_COLORMAP)
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    rgb[valid] = VOC_COLORMAP[mask[valid]]
    if ignore_index is not None:
        rgb[mask == ignore_index] = [255, 255, 255]
    return rgb

