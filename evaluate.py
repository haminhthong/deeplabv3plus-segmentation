"""Đánh giá checkpoint mô hình phân đoạn ảnh trên tập dữ liệu Pascal VOC."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import (
    CHECKPOINT_PATH,
    IGNORE_INDEX,
    NUM_CLASSES,
    OUTPUT_DIR,
    VOC_ROOT,
    configure_console,
)
from dataset_voc import VOCSegmentationDataset, get_val_transforms
from inference import load_checkpoint_model
from metrics import SegmentationMetrics, save_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


@torch.inference_mode()
def evaluate(model, loader, device: torch.device) -> dict[str, object]:
    metrics = SegmentationMetrics(NUM_CLASSES)
    elapsed = 0.0
    image_count = 0

    # Warm-up nếu dùng GPU
    if device.type == "cuda" and len(loader) > 0:
        dummy_input = next(iter(loader))[0][:1].to(device)
        for _ in range(5):
            _ = model(dummy_input)
        torch.cuda.synchronize()

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        predictions = model(images).argmax(1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        image_count += len(images)
        metrics.update(predictions, masks, IGNORE_INDEX)

    result = metrics.compute(ignore_index=IGNORE_INDEX)
    result["images"] = image_count
    result["total_time_seconds"] = elapsed
    result["latency_ms_per_image"] = 1000 * elapsed / max(image_count, 1)
    result["fps"] = image_count / max(elapsed, 1e-6)
    return result


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="Đánh giá checkpoint mô hình trên Pascal VOC")
    parser.add_argument("--data-root", type=Path, default=VOC_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "evaluation.json")
    parser.add_argument("--csv-output", type=Path, default=OUTPUT_DIR / "per_class_metrics.csv")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size phải lớn hơn 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Sử dụng thiết bị: %s", device)

    model, metadata = load_checkpoint_model(args.checkpoint, device)
    image_size = int(metadata.get("image_size", 320))
    logger.info("Đã tải checkpoint từ: %s (Kích thước ảnh: %d)", args.checkpoint, image_size)

    dataset = VOCSegmentationDataset(
        args.data_root,
        args.split,
        get_val_transforms(image_size, image_size),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    result = evaluate(model, loader, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_metrics(result, args.output, args.csv_output)

    logger.info("mIoU (tất cả các lớp): %.4f", result["mean_iou_all"])
    logger.info("mIoU (không tính background): %.4f", result["mean_iou_no_background"])
    logger.info("Dice trung bình: %.4f", result["mean_dice_all"])
    logger.info("Độ chính xác pixel: %.4f", result["pixel_accuracy"])
    logger.info("Độ trễ trung bình: %.2f ms/ảnh (%.1f FPS)", result["latency_ms_per_image"], result["fps"])
    logger.info("Đã lưu JSON kết quả: %s", args.output)
    logger.info("Đã lưu CSV chi tiết theo lớp: %s", args.csv_output)


if __name__ == "__main__":
    main()
