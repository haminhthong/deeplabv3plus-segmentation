"""Script kiểm tra tính toàn vẹn dữ liệu VOC và phát hiện rò rỉ (leakage)."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config import IGNORE_INDEX, NUM_CLASSES, VOC_ROOT


def calculate_file_hash(path: Path) -> str:
    """Tính mã SHA-256 hash của một tệp tin."""
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_ids_from_file(file_path: Path) -> list[str]:
    """Đọc danh sách mã ảnh từ tệp tin split."""
    if not file_path.is_file():
        return []
    lines = [line.strip() for line in file_path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def audit_voc_dataset(data_root: Path, splits_dir: Path | None = None) -> dict[str, Any]:
    """Thực hiện audit kiểm tra toàn bộ dataset VOC."""
    data_root = Path(data_root)
    
    # Ưu tiên đọc split từ splits_dir nếu có, nếu không đọc từ ImageSets/Segmentation
    split_names = ["train", "val", "test"]
    split_ids: dict[str, list[str]] = {}

    for name in split_names:
        if splits_dir and (splits_dir / f"{name}.txt").is_file():
            split_ids[name] = read_ids_from_file(splits_dir / f"{name}.txt")
        else:
            voc_split_file = data_root / "ImageSets" / "Segmentation" / f"{name}.txt"
            split_ids[name] = read_ids_from_file(voc_split_file)

    report: dict[str, Any] = {
        "train_images": len(split_ids.get("train", [])),
        "val_images": len(split_ids.get("val", [])),
        "test_images": len(split_ids.get("test", [])),
        "missing_files": [],
        "duplicate_ids": [],
        "duplicate_hashes": [],
        "invalid_mask_values": [],
        "dimension_mismatches": [],
    }

    # 1. Kiểm tra duplicate IDs trong từng split và giữa các split
    all_ids_seen: set[str] = set()
    for s_name, ids in split_ids.items():
        seen_in_split: set[str] = set()
        for img_id in ids:
            if img_id in seen_in_split:
                report["duplicate_ids"].append({"split": s_name, "id": img_id, "type": "intra_split"})
            elif img_id in all_ids_seen:
                report["duplicate_ids"].append({"split": s_name, "id": img_id, "type": "inter_split"})
            seen_in_split.add(img_id)
            all_ids_seen.add(img_id)

    # 2. Kiểm tra sự tồn tại của file, SHA-256 hash leakage, kích thước và nhãn mask
    hash_to_id: dict[str, tuple[str, str]] = {}  # hash -> (split_name, img_id)

    for s_name, ids in split_ids.items():
        for img_id in ids:
            img_path = data_root / "JPEGImages" / f"{img_id}.jpg"
            mask_path = data_root / "SegmentationClass" / f"{img_id}.png"

            if not img_path.is_file():
                report["missing_files"].append(str(img_path))
                continue
            if not mask_path.is_file():
                report["missing_files"].append(str(mask_path))
                continue

            # Kiểm tra hash trùng lặp
            img_hash = calculate_file_hash(img_path)
            if img_hash in hash_to_id:
                prev_split, prev_id = hash_to_id[img_hash]
                report["duplicate_hashes"].append({
                    "hash": img_hash,
                    "first": {"split": prev_split, "id": prev_id},
                    "second": {"split": s_name, "id": img_id},
                })
            else:
                hash_to_id[img_hash] = (s_name, img_id)

            # Kiểm tra kích thước & nhãn mask
            try:
                with Image.open(img_path) as img, Image.open(mask_path) as mask:
                    if img.size != mask.size:
                        report["dimension_mismatches"].append({
                            "id": img_id,
                            "image_size": list(img.size),
                            "mask_size": list(mask.size),
                        })
                    
                    mask_arr = np.array(mask)
                    invalid = np.setdiff1d(mask_arr, list(range(NUM_CLASSES)) + [IGNORE_INDEX])
                    if len(invalid) > 0:
                        report["invalid_mask_values"].append({
                            "id": img_id,
                            "invalid_values": [int(x) for x in invalid],
                        })
            except Exception as e:
                report["missing_files"].append(f"{img_id}: Lỗi đọc file ({e})")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit dataset VOC2012 và phát hiện rò rỉ dữ liệu")
    parser.add_argument("--data-root", type=Path, default=VOC_ROOT, help="Đường dẫn root của VOC dataset")
    parser.add_argument("--splits-dir", type=Path, default=Path("splits"), help="Thư mục chứa các file train/val/test.txt")
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_audit.json"), help="File lưu báo cáo JSON")
    args = parser.parse_args()

    report = audit_voc_dataset(args.data_root, args.splits_dir if args.splits_dir.exists() else None)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã hoàn thành audit. Báo cáo lưu tại: {args.output}")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
