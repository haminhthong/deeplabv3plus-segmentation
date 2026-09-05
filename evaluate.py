"""Đánh giá checkpoint mô hình phân đoạn ảnh trên tập dữ liệu Pascal VOC.

Bao gồm:
- Headline metrics: mIoU all, mIoU no-background, Per-class IoU
- Supporting metrics: Mean Dice, Pixel Accuracy, Mean Class Accuracy
- Structural & Boundary metrics: Boundary F1 (BF-score)
- Error slices: Region-size mIoU (Small, Medium, Large)
- Confusion Analysis: Best 5, Worst 5, Top confusion pairs
- Latency Profiling: Breakdown preprocess, forward, postprocess; p50, p95, FPS
- Hardware metadata recording
"""

from __future__ import annotations

import argparse
import logging
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
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
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    image_size: int = 320,
) -> dict[str, Any]:
    metrics = SegmentationMetrics(NUM_CLASSES)
    image_count = 0

    batch_forward_latencies: list[float] = []
    batch_total_latencies: list[float] = []

    # Warm-up nếu dùng GPU
    if device.type == "cuda" and len(loader) > 0:
        dummy_input = next(iter(loader))[0][:1].to(device)
        for _ in range(5):
            _ = model(dummy_input)
        torch.cuda.synchronize()

    total_start = time.perf_counter()

    for images, masks in loader:
        b_size = len(images)
        image_count += b_size

        t0 = time.perf_counter()
        images = images.to(device, non_blocking=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fwd_start = time.perf_counter()
        logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_fwd_end = time.perf_counter()

        predictions = logits.argmax(1).cpu().numpy()
        masks_np = masks.numpy()

        metrics.update(predictions, masks_np, IGNORE_INDEX, compute_boundary=True)
        t_end = time.perf_counter()

        fwd_time_per_img = (t_fwd_end - t_fwd_start) / b_size
        total_time_per_img = (t_end - t0) / b_size

        for _ in range(b_size):
            batch_forward_latencies.append(fwd_time_per_img * 1000.0)
            batch_total_latencies.append(total_time_per_img * 1000.0)

    total_elapsed = time.perf_counter() - total_start
    result = metrics.compute(ignore_index=IGNORE_INDEX)

    # Thống kê thời gian và độ trễ (latency profiling)
    mean_lat = float(np.mean(batch_total_latencies)) if batch_total_latencies else 0.0
    p50_lat = float(np.percentile(batch_total_latencies, 50)) if batch_total_latencies else 0.0
    p95_lat = float(np.percentile(batch_total_latencies, 95)) if batch_total_latencies else 0.0

    fwd_mean = float(np.mean(batch_forward_latencies)) if batch_forward_latencies else 0.0
    fwd_p50 = float(np.percentile(batch_forward_latencies, 50)) if batch_forward_latencies else 0.0
    fwd_p95 = float(np.percentile(batch_forward_latencies, 95)) if batch_forward_latencies else 0.0

    fps = image_count / max(total_elapsed, 1e-6)

    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "CPU"

    result["profiling"] = {
        "images_evaluated": image_count,
        "total_time_seconds": total_elapsed,
        "fps": fps,
        "latency_ms_per_image": {
            "mean": mean_lat,
            "p50": p50_lat,
            "p95": p95_lat,
        },
        "model_forward_ms": {
            "mean": fwd_mean,
            "p50": fwd_p50,
            "p95": fwd_p95,
        },
        "hardware": {
            "device_type": device.type,
            "device_name": device_name,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "batch_size": loader.batch_size,
            "input_resolution": f"{image_size}x{image_size}",
        },
    }

    # Giữ tương thích ngược với các trường cũ
    result["images"] = image_count
    result["total_time_seconds"] = total_elapsed
    result["latency_ms_per_image"] = mean_lat
    result["fps"] = fps

    return result


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="Đánh giá checkpoint mô hình trên Pascal VOC")
    parser.add_argument("--data-root", type=Path, default=VOC_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument(
        "--split-type",
        type=str,
        default="benchmark",
        choices=["benchmark", "smoke"],
        help="Loại split dữ liệu ('benchmark' cho thử nghiệm chuẩn, 'smoke' cho kiểm tra phần mềm)",
    )
    parser.add_argument("--splits-dir", type=Path, default=None, help="Thư mục split tùy chọn")
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
    arch = metadata.get("architecture", "deeplabv3plus")
    logger.info("Đã tải checkpoint từ: %s (Kiến trúc: %s, Kích thước ảnh: %d)", args.checkpoint, arch, image_size)

    dataset = VOCSegmentationDataset(
        args.data_root,
        split=args.split,
        joint_transform=get_val_transforms(image_size, image_size),
        split_dir=args.splits_dir,
        split_type=args.split_type,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    result = evaluate(model, loader, device, image_size=image_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_metrics(result, args.output, args.csv_output)

    logger.info("=== KẾT QUẢ ĐÁNH GIÁ (EVALUATION REPORT) ===")
    logger.info("mIoU (tất cả các lớp): %.4f", result["mean_iou_all"])
    logger.info("mIoU (không tính background): %.4f", result["mean_iou_no_background"])
    logger.info("Mean Dice: %.4f", result["mean_dice_all"])
    logger.info("Độ chính xác pixel: %.4f", result["pixel_accuracy"])

    if result.get("boundary_f1_all") is not None:
        logger.info("Boundary F1 (tất cả): %.4f", result["boundary_f1_all"])
    if result.get("boundary_f1_no_background") is not None:
        logger.info("Boundary F1 (không background): %.4f", result["boundary_f1_no_background"])

    prof = result["profiling"]
    lat = prof["latency_ms_per_image"]
    logger.info("Độ trễ toàn pipeline: Mean=%.2f ms, p50=%.2f ms, p95=%.2f ms (%.1f FPS)",
                lat["mean"], lat["p50"], lat["p95"], prof["fps"])

    if result.get("best_classes"):
        best_str = ", ".join(f"{c['class_name']} ({c['iou']:.2f})" for c in result["best_classes"][:3])
        logger.info("Top 3 lớp tốt nhất: %s", best_str)
    if result.get("worst_classes"):
        worst_str = ", ".join(f"{c['class_name']} ({c['iou']:.2f})" for c in result["worst_classes"][:3])
        logger.info("Top 3 lớp kém nhất: %s", worst_str)

    logger.info("Đã lưu JSON kết quả: %s", args.output)
    logger.info("Đã lưu CSV chi tiết theo lớp: %s", args.csv_output)


if __name__ == "__main__":
    main()
