from __future__ import annotations

import argparse
import logging
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import segmentation_models_pytorch as smp
import torch
from segmentation_models_pytorch.losses import DiceLoss
from torch import nn
from torch.utils.data import DataLoader

from config import (
    IGNORE_INDEX,
    IMAGE_SIZE,
    NUM_CLASSES,
    OUTPUT_DIR,
    VOC_ROOT,
    configure_console,
)
from dataset_voc import (
    VOCSegmentationDataset,
    get_train_transforms,
    get_val_transforms,
    validate_voc_dataset,
)
from inference import build_model
from metrics import SegmentationMetrics, save_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Đã bật chế độ deterministic cho cuDNN")
    else:
        torch.backends.cudnn.benchmark = True


class CombinedLoss(nn.Module):
    def __init__(self, ignore_index: int):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(mode="multiclass", ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        return self.ce(logits, masks) + 0.5 * self.dice(logits, masks)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình phân đoạn ảnh trên Pascal VOC")
    parser.add_argument("--data-root", type=str, default=str(VOC_ROOT))
    parser.add_argument("--architecture", type=str, default="deeplabv3plus", choices=["deeplabv3plus", "unet", "fcn"])
    parser.add_argument("--encoder", type=str, default="resnet50")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", help="Bật chế độ deterministic")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--resume", type=Path, default=None, help="Tiếp tục từ checkpoint")
    parser.add_argument("--patience", type=int, default=0, help="Dừng sớm (early stopping); 0 là tắt")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs phải lớn hơn 0")
    if args.batch_size <= 0:
        parser.error("--batch-size phải lớn hơn 0")
    if args.image_size <= 0:
        parser.error("--image-size phải lớn hơn 0")
    if args.lr <= 0:
        parser.error("--lr phải lớn hơn 0")
    if args.patience < 0:
        parser.error("--patience không được âm")
    return args


def create_dataloaders(args: argparse.Namespace, device: torch.device):
    data_root = Path(args.data_root)
    validate_voc_dataset(data_root)

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
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    return train_loader, val_loader


def create_training_components(args: argparse.Namespace, device: torch.device, resume_checkpoint: dict[str, Any] | None):
    initial_weights = None if resume_checkpoint is not None else args.encoder_weights
    model = build_model(args.encoder, initial_weights, NUM_CLASSES, args.architecture).to(device)
    criterion = CombinedLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    active_scaler = scaler if scaler.is_enabled() else None
    return model, criterion, optimizer, scheduler, active_scaler


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
    train_mode: bool = True,
    scaler: torch.amp.GradScaler | None = None,
):
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
                optimizer.zero_grad(set_to_none=True)
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
    return avg_loss, metrics.compute(ignore_index=ignore_index)


def save_checkpoint(
    ckpt_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    active_scaler: Any,
    best_miou: float,
    args: argparse.Namespace,
    val_metrics: dict[str, Any],
) -> None:
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        git_sha = None

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": active_scaler.state_dict() if active_scaler else None,
            "architecture": args.architecture,
            "encoder": args.encoder,
            "encoder_weights": args.encoder_weights,
            "image_size": args.image_size,
            "best_val_miou": best_miou,
            "loss": "cross_entropy + 0.5 * dice",
            "num_classes": NUM_CLASSES,
            "ignore_index": IGNORE_INDEX,
            "class_mapping": "Pascal VOC 2012",
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "smp_version": smp.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "seed": args.seed,
            "deterministic": args.deterministic,
            "train_args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "git_commit": git_sha,
        },
        ckpt_path,
    )
    logger.info("Đã lưu checkpoint tốt nhất tại: %s", ckpt_path)
    save_metrics(val_metrics, ckpt_path.parent / "best_metrics.json", ckpt_path.parent / "per_class_metrics.csv")


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Huấn luyện kiến trúc: %s (%s) trên thiết bị: %s", args.architecture, args.encoder, device)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / f"{args.architecture}_{args.encoder}_voc_best.pth"

    resume_checkpoint = None
    if args.resume:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=True)

    train_loader, val_loader = create_dataloaders(args, device)
    model, criterion, optimizer, scheduler, active_scaler = create_training_components(args, device, resume_checkpoint)

    best_miou = -1.0
    start_epoch = 1
    stale_epochs = 0
    history_lines = ["epoch,train_loss,train_miou,val_loss,val_miou"]

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        if "optimizer_state_dict" in resume_checkpoint:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
            scheduler.T_max = args.epochs
        if active_scaler is not None and resume_checkpoint.get("scaler_state_dict"):
            active_scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_miou = float(resume_checkpoint.get("best_val_miou", -1.0))
        log_path = output_dir / "train_log.csv"
        if log_path.exists():
            history_lines = log_path.read_text(encoding="utf-8").splitlines()

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            NUM_CLASSES,
            IGNORE_INDEX,
            train_mode=True,
            scaler=active_scaler,
        )
        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, device, NUM_CLASSES, IGNORE_INDEX, train_mode=False
        )
        train_miou, val_miou = train_metrics["mean_iou"], val_metrics["mean_iou"]

        logger.info(
            "Epoch %02d/%d | LR=%.2e | train_loss=%.4f, train_mIoU=%.4f | val_loss=%.4f, val_mIoU=%.4f",
            epoch,
            args.epochs,
            scheduler.get_last_lr()[0],
            train_loss,
            train_miou,
            val_loss,
            val_miou,
        )
        history_lines.append(f"{epoch},{train_loss:.6f},{train_miou:.6f},{val_loss:.6f},{val_miou:.6f}")
        scheduler.step()

        if val_miou > best_miou:
            best_miou = val_miou
            stale_epochs = 0
            save_checkpoint(ckpt_path, epoch, model, optimizer, scheduler, active_scaler, best_miou, args, val_metrics)
        else:
            stale_epochs += 1

        (output_dir / "train_log.csv").write_text("\n".join(history_lines), encoding="utf-8")
        if args.patience > 0 and stale_epochs >= args.patience:
            logger.info("Dừng sớm (early stopping) sau %d epoch không cải thiện.", stale_epochs)
            break

    logger.info("Huấn luyện hoàn tất. Validation mIoU tốt nhất: %.4f", best_miou)


def main() -> None:
    configure_console()
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
