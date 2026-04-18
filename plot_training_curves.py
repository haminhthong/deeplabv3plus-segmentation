import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_training_log(csv_path: Path):
    epochs = []
    train_loss = []
    train_miou = []
    val_loss = []
    val_miou = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_miou.append(float(row["train_miou"]))
            val_loss.append(float(row["val_loss"]))
            val_miou.append(float(row["val_miou"]))

    return epochs, train_loss, train_miou, val_loss, val_miou


def main():
    parser = argparse.ArgumentParser(description="Plot training curves from train_log.csv")
    parser.add_argument("--log-path", type=str, default="outputs/train_log.csv")
    parser.add_argument("--output-path", type=str, default="outputs/training_curves.png")
    parser.add_argument("--show", action="store_true", help="Hiển thị cửa sổ đồ thị (có thể làm chậm nếu chạy từ xa)")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        raise FileNotFoundError(f"Cannot find log file: {log_path}")

    epochs, train_loss, train_miou, val_loss, val_miou = read_training_log(log_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Đồ thị Loss
    axes[0].plot(epochs, train_loss, marker="o", label="Train Loss")
    axes[0].plot(epochs, val_loss, marker="o", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curve")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Đồ thị mIoU
    axes[1].plot(epochs, train_miou, marker="o", label="Train mIoU")
    axes[1].plot(epochs, val_miou, marker="o", label="Val mIoU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("mIoU")
    axes[1].set_title("mIoU Curve")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot -> {out_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()

