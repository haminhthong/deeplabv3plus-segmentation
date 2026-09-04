import argparse
import csv
from pathlib import Path

from matplotlib.figure import Figure

from config import configure_console


def read_training_log(csv_path: Path):
    epochs = []
    train_loss = []
    train_miou = []
    val_loss = []
    val_miou = []

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required_columns = {"epoch", "train_loss", "train_miou", "val_loss", "val_miou"}
        missing_columns = required_columns.difference(reader.fieldnames or [])
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"Tệp nhật ký thiếu các cột: {missing}")
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_miou.append(float(row["train_miou"]))
            val_loss.append(float(row["val_loss"]))
            val_miou.append(float(row["val_miou"]))

    return epochs, train_loss, train_miou, val_loss, val_miou


def create_training_figure(log_path: Path):
    """Tạo biểu đồ loss và mIoU từ nhật ký huấn luyện."""
    epochs, train_loss, train_miou, val_loss, val_miou = read_training_log(log_path)
    figure = Figure(figsize=(12, 4))
    axes = figure.subplots(1, 2)

    axes[0].plot(epochs, train_loss, marker="o", label="Huấn luyện")
    axes[0].plot(epochs, val_loss, marker="o", label="Xác thực")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Hàm mất mát")

    axes[1].plot(epochs, train_miou, marker="o", label="Huấn luyện")
    axes[1].plot(epochs, val_miou, marker="o", label="Xác thực")
    axes[1].set(xlabel="Epoch", ylabel="mIoU", title="mIoU")

    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()
    return figure


def main():
    configure_console()
    parser = argparse.ArgumentParser(description="Vẽ đồ thị từ tệp train_log.csv")
    parser.add_argument("--log-path", type=str, default="outputs/train_log.csv")
    parser.add_argument("--output-path", type=str, default="outputs/training_curves.png")
    args = parser.parse_args()

    log_path = Path(args.log_path)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not log_path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp nhật ký: {log_path}")

    figure = create_training_figure(log_path)
    figure.savefig(out_path, dpi=150)
    print(f"Đã lưu đồ thị: {out_path}")

if __name__ == "__main__":
    main()

