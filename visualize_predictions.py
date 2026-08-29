"""Trực quan hóa ảnh, nhãn thật, mặt nạ dự đoán và ảnh phủ màu."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from config import CHECKPOINT_PATH, IGNORE_INDEX, VOC_ROOT, configure_console
from dataset_voc import VOCSegmentationDataset
from inference import load_checkpoint_model, overlay_mask, predict_original_size
from voc_meta import mask_to_color_rgb


def main():
    configure_console()
    parser = argparse.ArgumentParser(description="Trực quan hóa dự đoán trên Pascal VOC")
    parser.add_argument("--data-root", type=Path, default=VOC_ROOT)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=CHECKPOINT_PATH,
    )
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument(
        "--indices",
        type=int,
        nargs="*",
        default=[0, 1, 2, 3, 4],
        help="Chỉ số mẫu cần trực quan hóa",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs") / "viz")
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Mặc định dùng kích thước ghi trong checkpoint",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_checkpoint_model(args.checkpoint, device)
    ckpt_image_size = int(metadata.get("image_size", 320))
    image_size = args.image_size if args.image_size is not None else ckpt_image_size
    if image_size <= 0:
        parser.error("--image-size phải lớn hơn 0")

    ds = VOCSegmentationDataset(
        root=args.data_root,
        split=args.split,
        joint_transform=None,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for idx in args.indices:
        if not 0 <= idx < len(ds):
            print(f"Bỏ qua chỉ số ngoài phạm vi: {idx}")
            continue
        sid = ds.ids[idx]
        raw_img = Image.open(ds.jpeg_dir / f"{sid}.jpg").convert("RGB")
        gt = np.array(Image.open(ds.mask_dir / f"{sid}.png"), dtype=np.int64)
        pred = predict_original_size(model, raw_img, image_size, device)
        overlay = overlay_mask(
            np.asarray(raw_img),
            mask_to_color_rgb(pred, IGNORE_INDEX),
            alpha=0.45,
        )

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(raw_img)
        axes[0].set_title("Ảnh gốc")
        axes[1].imshow(mask_to_color_rgb(gt, IGNORE_INDEX))
        axes[1].set_title("Nhãn thật")
        axes[2].imshow(mask_to_color_rgb(pred, IGNORE_INDEX))
        axes[2].set_title("Dự đoán")
        axes[3].imshow(overlay)
        axes[3].set_title("Ảnh phủ màu")
        for ax in axes:
            ax.axis("off")
        out = args.out_dir / f"{sid}_viz.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Đã lưu: {out}")


if __name__ == "__main__":
    main()
