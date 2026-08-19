"""
Trực quan hóa ảnh gốc, nhãn thật, mask dự đoán và lớp phủ overlay (bảng màu Pascal VOC).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from config import IGNORE_INDEX, NUM_CLASSES, VOC_ROOT
from dataset_voc import VOCSegmentationDataset
from inference import load_checkpoint_model, predict_original_size
from voc_meta import mask_to_color_rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=VOC_ROOT)
    ap.add_argument("--checkpoint", type=Path, default=Path("outputs") / "deeplabv3plus_voc_best.pth")
    ap.add_argument("--split", type=str, default="val", choices=("train", "val"))
    ap.add_argument("--indices", type=int, nargs="*", default=[0, 1, 2, 3, 4], help="Chỉ số mẫu trong split")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs") / "viz")
    ap.add_argument("--val-h", type=int, default=None, help="Mặc định lấy theo image_size trong checkpoint")
    ap.add_argument("--val-w", type=int, default=None, help="Mặc định lấy theo image_size trong checkpoint")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_checkpoint_model(args.checkpoint, device)
    ckpt_image_size = int(metadata.get("image_size", 320))
    vis_h = args.val_h if args.val_h is not None else ckpt_image_size
    vis_w = args.val_w if args.val_w is not None else ckpt_image_size

    ds = VOCSegmentationDataset(
        root=args.data_root,
        split=args.split,
        joint_transform=None,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for idx in args.indices:
        if idx >= len(ds):
            continue
        sid = ds.ids[idx]
        raw_img = Image.open(ds.jpeg_dir / f"{sid}.jpg").convert("RGB")
        gt = np.array(Image.open(ds.mask_dir / f"{sid}.png"), dtype=np.int64)
        pred = predict_original_size(model, raw_img, max(vis_h, vis_w), device)
        overlay = (np.array(raw_img) * 0.55 + mask_to_color_rgb(pred, IGNORE_INDEX) * 0.45).clip(0, 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(raw_img)
        axes[0].set_title("Image")
        axes[1].imshow(mask_to_color_rgb(gt, IGNORE_INDEX))
        axes[1].set_title("Ground truth")
        axes[2].imshow(mask_to_color_rgb(pred, IGNORE_INDEX))
        axes[2].set_title("Prediction")
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay")
        for ax in axes:
            ax.axis("off")
        out = args.out_dir / f"{sid}_viz.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
