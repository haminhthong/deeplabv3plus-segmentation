"""Đánh giá checkpoint DeepLabV3+ trên một tập Pascal VOC."""

from __future__ import annotations

import argparse
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


@torch.inference_mode()
def evaluate(model, loader, device: torch.device) -> dict[str, object]:
    metrics = SegmentationMetrics(NUM_CLASSES)
    elapsed = 0.0
    image_count = 0

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

    result = metrics.compute()
    result["images"] = image_count
    result["latency_ms_per_image"] = 1000 * elapsed / image_count
    result["fps"] = image_count / elapsed
    return result


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="Đánh giá checkpoint trên Pascal VOC")
    parser.add_argument("--data-root", type=Path, default=VOC_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "evaluation.json")
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size phải lớn hơn 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_checkpoint_model(args.checkpoint, device)
    image_size = int(metadata.get("image_size", 320))
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
    save_metrics(result, args.output)
    print(f"mIoU: {result['mean_iou']:.4f}")
    print(f"Dice trung bình: {result['mean_dice']:.4f}")
    print(f"Độ chính xác pixel: {result['pixel_accuracy']:.4f}")
    print(f"Độ trễ: {result['latency_ms_per_image']:.2f} ms/ảnh")
    print(f"Đã lưu kết quả: {args.output}")


if __name__ == "__main__":
    main()
