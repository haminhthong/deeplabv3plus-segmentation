from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from dataset_voc import (
    calculate_letterbox_geometry,
    read_split_ids,
    resize_and_pad,
    validate_voc_dataset,
)


def test_letterbox_geometry_calculation():
    # Ảnh 640x480 resize về 320x320
    scale, new_w, new_h, pad_left, pad_top, pad_right, pad_bottom = calculate_letterbox_geometry(640, 480, 320, 320)
    assert scale == pytest.approx(320 / 640)
    assert new_w == 320
    assert new_h == 240
    assert pad_left == 0
    assert pad_right == 0
    assert pad_top == 40
    assert pad_bottom == 40


def test_letterbox_geometry_invalid_dimensions():
    with pytest.raises(ValueError):
        calculate_letterbox_geometry(0, 480, 320, 320)


def test_read_split_ids_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        read_split_ids(tmp_path, "nonexistent")


def test_read_split_ids_empty(tmp_path: Path):
    split_dir = tmp_path / "ImageSets" / "Segmentation"
    split_dir.mkdir(parents=True)
    (split_dir / "val.txt").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="không chứa mã ảnh"):
        read_split_ids(tmp_path, "val")


def test_read_split_ids_duplicates(tmp_path: Path):
    split_dir = tmp_path / "ImageSets" / "Segmentation"
    split_dir.mkdir(parents=True)
    (split_dir / "train.txt").write_text("2007_000032\n2007_000032\n", encoding="utf-8")

    with pytest.raises(ValueError, match="trùng lặp"):
        read_split_ids(tmp_path, "train")


def test_validate_dataset_train_val_leakage(tmp_path: Path):
    split_dir = tmp_path / "ImageSets" / "Segmentation"
    split_dir.mkdir(parents=True)
    (split_dir / "train.txt").write_text("2007_000032\n", encoding="utf-8")
    (split_dir / "val.txt").write_text("2007_000032\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bị trùng"):
        validate_voc_dataset(tmp_path)


def test_validate_dataset_missing_files(tmp_path: Path):
    split_dir = tmp_path / "ImageSets" / "Segmentation"
    split_dir.mkdir(parents=True)
    (split_dir / "val.txt").write_text("2007_000033\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Thiếu"):
        validate_voc_dataset(tmp_path)


def test_resize_and_pad_mismatched_size():
    img = Image.new("RGB", (100, 100))
    mask = Image.new("L", (100, 200))
    with pytest.raises(ValueError, match="cùng kích thước"):
        resize_and_pad(img, mask, 320, 320)
