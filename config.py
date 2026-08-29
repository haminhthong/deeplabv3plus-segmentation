import sys
from pathlib import Path

NUM_CLASSES = 21
IGNORE_INDEX = 255
IMAGE_SIZE = 320
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
VOC_ROOT = Path("data") / "VOC2012_train_val" / "VOC2012_train_val"
OUTPUT_DIR = Path("outputs")
CHECKPOINT_PATH = OUTPUT_DIR / "deeplabv3plus_voc_best.pth"


def configure_console() -> None:
    """Cho phép terminal Windows hiển thị thông báo tiếng Việt."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
