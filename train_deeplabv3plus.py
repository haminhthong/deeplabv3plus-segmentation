import argparse
import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import IGNORE_INDEX as VOC_IGNORE_INDEX
from config import NUM_CLASSES as VOC_NUM_CLASSES
from config import VOC_ROOT
from dataset_voc import VOCSegmentationDataset, get_train_transforms, get_val_transforms
from metrics import SegmentationMetrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, num_classes, ignore_index, train_mode=True, scaler=None):
    if train_mode:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    metrics = SegmentationMetrics(num_classes)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.set_grad_enabled(train_mode), torch.autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(images)
            loss = criterion(logits, masks)

            if train_mode:
                optimizer.zero_grad()
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

        preds = torch.argmax(logits, dim=1)
        metrics.update(preds, masks, ignore_index)
        epoch_loss += loss.item()

    avg_loss = epoch_loss / max(len(loader), 1)
    return avg_loss, metrics.compute()


def build_model(num_classes: int, encoder: str, encoder_weights: str):
    return smp.DeepLabV3Plus(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=num_classes,
        activation=None,
    )


class CombinedLoss(nn.Module):
    def __init__(self, ignore_index):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        # DiceLoss bỏ qua vùng void 255; background của Pascal VOC là class 0.
        self.dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index)

    def forward(self, logits, masks):
        return self.ce(logits, masks) + 0.5 * self.dice(logits, masks)


def main():
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ on Pascal VOC segmentation dataset")
    parser.add_argument("--data-root", type=str, default=str(VOC_ROOT))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--encoder", type=str, default="resnet50")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--resume", type=Path, default=None, help="Tiếp tục từ checkpoint")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping; 0 là tắt")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "deeplabv3plus_voc_best.pth"

    train_dataset = VOCSegmentationDataset(
        data_root,
        split="train",
        joint_transform=get_train_transforms(args.image_size, args.image_size),
    )
    val_dataset = VOCSegmentationDataset(
        data_root,
        split="val",
        joint_transform=get_val_transforms(args.image_size, args.image_size),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = build_model(VOC_NUM_CLASSES, args.encoder, args.encoder_weights).to(device)
    criterion = CombinedLoss(ignore_index=VOC_IGNORE_INDEX)
    
    # Giảm weight decay để tránh cản trở quá trình học (5e-2 có thể là quá lớn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Trình tự động giảm Learning Rate theo hình Cosine (Cosine Annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    active_scaler = scaler if scaler.is_enabled() else None

    best_miou = -1.0
    start_epoch = 1
    stale_epochs = 0
    history_lines = ["epoch,train_loss,train_miou,val_loss,val_miou"]
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if active_scaler is not None and checkpoint.get("scaler_state_dict"):
            active_scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_miou = float(checkpoint.get("best_val_miou", -1.0))
        log_path = output_dir / "train_log.csv"
        if log_path.exists():
            history_lines = log_path.read_text(encoding="utf-8").splitlines()

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, VOC_NUM_CLASSES, VOC_IGNORE_INDEX, True, active_scaler
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, device, VOC_NUM_CLASSES, VOC_IGNORE_INDEX, train_mode=False
        )
        train_miou, val_miou = train_metrics["mean_iou"], val_metrics["mean_iou"]

        print(
            f"Epoch {epoch:02d}/{args.epochs} | LR={scheduler.get_last_lr()[0]:.2e} | "
            f"train_loss={train_loss:.4f}, train_mIoU={train_miou:.4f} | "
            f"val_loss={val_loss:.4f}, val_mIoU={val_miou:.4f}"
        )
        history_lines.append(f"{epoch},{train_loss:.6f},{train_miou:.6f},{val_loss:.6f},{val_miou:.6f}")
        
        # Cập nhật scheduler
        scheduler.step()

        if val_miou > best_miou:
            best_miou = val_miou
            stale_epochs = 0
            try:
                git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            except (OSError, subprocess.SubprocessError):
                git_sha = None
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": active_scaler.state_dict() if active_scaler else None,
                    "encoder": args.encoder,
                    "encoder_weights": args.encoder_weights,
                    "image_size": args.image_size,
                    "best_val_miou": best_miou,
                    "loss": "cross_entropy + 0.5 * dice",
                    "num_classes": VOC_NUM_CLASSES,
                    "ignore_index": VOC_IGNORE_INDEX,
                    "class_mapping": "Pascal VOC 2012",
                    "train_args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    "git_commit": git_sha,
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint -> {ckpt_path}")
            serializable = {k: v for k, v in val_metrics.items() if k != "confusion_matrix"}
            for key in ("per_class_iou", "per_class_dice"):
                serializable[key] = [None if np.isnan(value) else float(value) for value in serializable[key]]
            (output_dir / "best_metrics.json").write_text(
                json.dumps(serializable, indent=2, allow_nan=False), encoding="utf-8"
            )
        else:
            stale_epochs += 1

        (output_dir / "train_log.csv").write_text("\n".join(history_lines))
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} epochs without improvement.")
            break

    print(f"Training finished. Best val mIoU={best_miou:.4f}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
    
