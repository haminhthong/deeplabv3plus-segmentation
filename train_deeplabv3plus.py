import argparse
import os
import random
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ConfusionMatrix:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor, ignore_index: int):
        valid = targets != ignore_index
        a = targets[valid].cpu().numpy()
        b = preds[valid].cpu().numpy()
        n = self.num_classes
        k = (a >= 0) & (a < n)
        inds = n * a[k] + b[k]
        self.mat += np.bincount(inds, minlength=n**2).reshape(n, n)

    def compute(self) -> float:
        h = self.mat.sum(1)
        w = self.mat.sum(0)
        diag = np.diag(self.mat)
        union = h + w - diag
        valid_classes = union > 0
        if not np.any(valid_classes):
            return 0.0
        ious = diag[valid_classes] / union[valid_classes]
        return float(np.mean(ious))


def run_epoch(model, loader, criterion, optimizer, device, num_classes, ignore_index, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()

    epoch_loss = 0.0
    conf_mat = ConfusionMatrix(num_classes)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.set_grad_enabled(train_mode):
            logits = model(images)
            loss = criterion(logits, masks)

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        preds = torch.argmax(logits, dim=1)
        conf_mat.update(preds, masks, ignore_index)
        epoch_loss += loss.item()

    avg_loss = epoch_loss / max(len(loader), 1)
    avg_miou = conf_mat.compute()
    return avg_loss, avg_miou


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
        # smp.losses.DiceLoss hỗ trợ ignore_index để tránh tính loss trên vùng nhãn nền (255)
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

    best_miou = -1.0
    history_lines = ["epoch,train_loss,train_miou,val_loss,val_miou"]

    for epoch in range(1, args.epochs + 1):
        train_loss, train_miou = run_epoch(
            model, train_loader, criterion, optimizer, device, VOC_NUM_CLASSES, VOC_IGNORE_INDEX, train_mode=True
        )
        val_loss, val_miou = run_epoch(
            model, val_loader, criterion, optimizer, device, VOC_NUM_CLASSES, VOC_IGNORE_INDEX, train_mode=False
        )

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
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder": args.encoder,
                    "encoder_weights": args.encoder_weights,
                    "image_size": args.image_size,
                    "best_val_miou": best_miou,
                },
                ckpt_path,
            )
            print(f"Saved best checkpoint -> {ckpt_path}")

        (output_dir / "train_log.csv").write_text("\n".join(history_lines))

    print(f"Training finished. Best val mIoU={best_miou:.4f}")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main()
    
