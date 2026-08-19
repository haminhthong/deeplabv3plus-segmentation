import numpy as np
from PIL import Image

from dataset_voc import resize_and_pad


def test_resize_and_pad_preserves_content_aspect_ratio_and_void_padding():
    image = Image.new("RGB", (200, 100), "red")
    mask = Image.fromarray(np.ones((100, 200), dtype=np.uint8))
    image_out, mask_out = resize_and_pad(image, mask, 320, 320)
    assert image_out.size == (320, 320)
    assert mask_out.size == (320, 320)
    values, counts = np.unique(np.asarray(mask_out), return_counts=True)
    assert dict(zip(values, counts)) == {1: 320 * 160, 255: 320 * 160}
