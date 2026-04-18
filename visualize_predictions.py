"""
Trực quan hóa ảnh gốc, nhãn thật, mask dự đoán và lớp phủ overlay (bảng màu Pascal VOC).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
from PIL import Image

from config import IGNORE_INDEX, NUM_CLASSES, VOC_ROOT
from dataset_voc import VOCSegmentationDataset, get_val_transforms
from voc_meta import mask_to_color_rgb


def load_model(checkpoint: Path | None, device: torch.device):
    encoder_name = "resnet50"
    encoder_weights = "imagenet"
    image_size = 320
    state = None
    if checkpoint is not None and checkpoint.is_file():
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict):
            encoder_name = state.get("encoder", encoder_name)
            encoder_weights = state.get("encoder_weights", encoder_weights)
            image_size = int(state.get("image_size", image_size))

    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights if checkpoint is None else None,
        classes=NUM_CLASSES,
        activation=None,
    )

    if checkpoint is not None and checkpoint.is_file() and state is not None:
        sd = state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state
        model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model, image_size


@torch.no_grad()
def predict(model, image_tensor: torch.Tensor, device) -> np.ndarray:
    x = image_tensor.unsqueeze(0).to(device)
    logits = model(x)
    pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.int64)
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=VOC_ROOT)
    ap.add_argument("--checkpoint", type=Path, default=Path("outputs") / "deeplabv3plus_voc_best.pth")
    ap.add_argument("--split", type=str, default="val", choices=("train", "val"))
    ap.add_argument("--indices", type=int, nargs="*", default=[0, 1, 2, 3, 4], help="Chỉ số mẫu trong split")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs") / "viz")
    ap.add_argument("--val-h", type=int, default=None, help="Mặc định lấy theo image_size trong checkpoint")
    ap.add_argument("--val-w", type=int, default=None, help="Mặc định lấy theo image_size trong checkpoint")
    ap.add_argument("--no-train-weights", action="store_true", help="Không tải trọng số huấn luyện (No train weights)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = None if args.no_train_weights else args.checkpoint
    model, ckpt_image_size = load_model(ckpt, device)
    vis_h = args.val_h if args.val_h is not None else ckpt_image_size
    vis_w = args.val_w if args.val_w is not None else ckpt_image_size

    ds = VOCSegmentationDataset(
        root=args.data_root,
        split=args.split,
        joint_transform=get_val_transforms(vis_h, vis_w),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for idx in args.indices:
        if idx >= len(ds):
            continue
        img_t, mask_t = ds[idx]
        pred = predict(model, img_t, device)

        sid = ds.ids[idx]
        raw_img = Image.open(ds.jpeg_dir / f"{sid}.jpg").convert("RGB")
        raw_img = raw_img.resize((vis_w, vis_h), Image.BILINEAR)

        gt = mask_t.numpy()
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
