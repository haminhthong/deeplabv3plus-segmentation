"""Script kiểm tra tính toàn vẹn dữ liệu VOC, phân bố lớp và phát hiện rò rỉ (leakage)."""

from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from config import IGNORE_INDEX, NUM_CLASSES, VOC_ROOT, configure_console
from voc_meta import VOC_CLASSES


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


def audit_voc_dataset(
    data_root: Path,
    splits_dir: Path | None = None,
    split_type: str = "benchmark",
) -> dict[str, Any]:
    """Thực hiện audit kiểm tra toàn bộ dataset VOC, bao gồm rò rỉ và phân bố lớp."""
    data_root = Path(data_root)

    # Ưu tiên đọc split từ splits_dir nếu có, hoặc splits/{split_type}, nếu không đọc từ ImageSets/Segmentation
    split_names = ["train", "val", "test"]
    split_ids: dict[str, list[str]] = {}
    split_files: dict[str, Path] = {}

    for name in split_names:
        cand_dir = splits_dir / f"{name}.txt" if splits_dir else None
        cand_type = Path("splits") / split_type / f"{name}.txt"
        cand_flat = Path("splits") / f"{name}.txt"
        cand_voc = data_root / "ImageSets" / "Segmentation" / f"{name}.txt"

        if cand_dir and cand_dir.is_file():
            split_ids[name] = read_ids_from_file(cand_dir)
            split_files[name] = cand_dir
        elif cand_type.is_file():
            split_ids[name] = read_ids_from_file(cand_type)
            split_files[name] = cand_type
        elif cand_voc.is_file():
            split_ids[name] = read_ids_from_file(cand_voc)
            split_files[name] = cand_voc
        elif cand_flat.is_file():
            split_ids[name] = read_ids_from_file(cand_flat)
            split_files[name] = cand_flat
        else:
            split_ids[name] = []

    report: dict[str, Any] = {
        "train_images": len(split_ids.get("train", [])),
        "val_images": len(split_ids.get("val", [])),
        "test_images": len(split_ids.get("test", [])),
        "split_files_sha256": {
            s: calculate_file_hash(p) for s, p in split_files.items() if p.is_file()
        },
        "missing_files": [],
        "duplicate_ids": [],
        "duplicate_hashes": [],
        "invalid_mask_values": [],
        "dimension_mismatches": [],
        "class_distribution": {},
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

    # 2. Kiểm tra sự tồn tại của file, SHA-256 hash leakage, kích thước, nhãn mask và phân bố lớp
    hash_to_id: dict[str, tuple[str, str]] = {}  # hash -> (split_name, img_id)

    for s_name, ids in split_ids.items():
        split_pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        split_image_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
        total_valid_pixels = 0

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

                    # Thống kê phân bố lớp
                    valid_pixels = mask_arr[mask_arr != IGNORE_INDEX]
                    if len(valid_pixels) > 0:
                        counts = np.bincount(valid_pixels, minlength=NUM_CLASSES)
                        split_pixel_counts[:len(counts)] += counts[:NUM_CLASSES]
                        total_valid_pixels += int(counts[:NUM_CLASSES].sum())

                        present_classes_in_img = np.unique(valid_pixels)
                        for c in present_classes_in_img:
                            if c < NUM_CLASSES:
                                split_image_counts[c] += 1

            except Exception as e:
                report["missing_files"].append(f"{img_id}: Lỗi đọc file ({e})")

        bg_pixels = int(split_pixel_counts[0])
        fg_pixels = int(split_pixel_counts[1:].sum())
        fg_bg_ratio = float(fg_pixels / max(bg_pixels, 1))

        report["class_distribution"][s_name] = {
            "total_images": len(ids),
            "total_pixels": total_valid_pixels,
            "background_pixels": bg_pixels,
            "foreground_pixels": fg_pixels,
            "foreground_to_background_ratio": fg_bg_ratio,
            "present_classes_count": int(np.sum(split_pixel_counts > 0)),
            "per_class_pixels": {
                VOC_CLASSES[c]: int(split_pixel_counts[c]) for c in range(NUM_CLASSES)
            },
            "per_class_image_occurrences": {
                VOC_CLASSES[c]: int(split_image_counts[c]) for c in range(NUM_CLASSES)
            },
        }

    return report


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="Audit dataset VOC2012 và phát hiện rò rỉ dữ liệu")
    parser.add_argument("--data-root", type=Path, default=VOC_ROOT, help="Đường dẫn root của VOC dataset")
    parser.add_argument(
        "--split-type",
        type=str,
        default="benchmark",
        choices=["benchmark", "smoke"],
        help="Loại split để audit ('benchmark' hoặc 'smoke')",
    )
    parser.add_argument("--splits-dir", type=Path, default=None, help="Thư mục chứa các file train/val/test.txt")
    parser.add_argument("--output", type=Path, default=Path("outputs/dataset_audit.json"), help="File lưu báo cáo JSON")
    parser.add_argument("--generate-manifest", type=Path, default=None, help="Đường dẫn xuất split_manifest.json")
    args = parser.parse_args()

    splits_dir = args.splits_dir
    if splits_dir is None:
        cand = Path("splits") / args.split_type
        if cand.exists():
            splits_dir = cand

    report = audit_voc_dataset(args.data_root, splits_dir=splits_dir, split_type=args.split_type)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Đã hoàn thành audit. Báo cáo lưu tại: {args.output}")

    if args.generate_manifest:
        manifest = {
            "benchmark_protocol": "Pascal VOC 2012 Split Manifest",
            "split_type": args.split_type,
            "report_summary": {
                "train_images": report["train_images"],
                "val_images": report["val_images"],
                "test_images": report["test_images"],
                "split_files_sha256": report["split_files_sha256"],
                "class_distribution": report["class_distribution"],
            },
        }
        args.generate_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.generate_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Đã lưu split manifest tại: {args.generate_manifest}")

    print(json.dumps({
        "train_images": report["train_images"],
        "val_images": report["val_images"],
        "test_images": report["test_images"],
        "duplicate_ids": len(report["duplicate_ids"]),
        "duplicate_hashes": len(report["duplicate_hashes"]),
        "missing_files": len(report["missing_files"]),
    }, indent=2))


if __name__ == "__main__":
    main()
