"""Script tải checkpoint mô hình DeepLabV3+ từ GitHub Releases / Hugging Face và xác minh SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from urllib.request import urlretrieve

from config import CHECKPOINT_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CHECKPOINT_URL = "https://github.com/example/deeplabv3plus-segmentation/releases/download/v1.0.0/deeplabv3plus_voc_best.pth"
EXPECTED_SHA256 = ""  # Nhập chuỗi SHA-256 khi phát hành release chính thức


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_checkpoint(url: str, output_path: Path, expected_sha256: str = "") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Đang tải checkpoint từ: %s", url)
    urlretrieve(url, output_path)
    logger.info("Tải xuống thành công: %s", output_path)

    actual_sha256 = calculate_sha256(output_path)
    logger.info("SHA-256 hash của file: %s", actual_sha256)

    if expected_sha256:
        if actual_sha256.lower() != expected_sha256.lower():
            output_path.unlink(missing_ok=True)
            raise ValueError(f"Xác minh SHA-256 thất bại! Kỳ vọng: {expected_sha256}, Thực tế: {actual_sha256}")
        logger.info("Xác minh SHA-256 thành công!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tải checkpoint mô hình đã huấn luyện")
    parser.add_argument("--url", type=str, default=DEFAULT_CHECKPOINT_URL, help="URL tải checkpoint")
    parser.add_argument("--output", type=Path, default=CHECKPOINT_PATH, help="Đường dẫn file lưu")
    parser.add_argument("--sha256", type=str, default=EXPECTED_SHA256, help="SHA-256 checksum mong muốn")
    args = parser.parse_args()

    download_checkpoint(args.url, args.output, args.sha256)


if __name__ == "__main__":
    main()
