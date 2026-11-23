"""
Image loading. Reads JPEG/PNG and the HEIC files an iPhone produces (via pillow-heif),
applies the EXIF orientation and returns RGB uint8 arrays.

Author: Nguyen Duc Nam Khanh
"""

import os
from typing import List

import numpy as np
import pillow_heif
from PIL import Image, ImageOps

pillow_heif.register_heif_opener()

IMAGE_EXTENSIONS = (".heic", ".heif", ".jpg", ".jpeg", ".png", ".bmp")


def list_images(folder: str) -> List[str]:
    """Sorted paths of all images in `folder`."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Image folder not found: {folder}")
    names = sorted(n for n in os.listdir(folder) if n.lower().endswith(IMAGE_EXTENSIONS))
    return [os.path.join(folder, n) for n in names]


def load_rgb(path: str) -> np.ndarray:
    """RGB uint8 image (H, W, 3), upright according to its EXIF orientation."""
    with Image.open(path) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))


def camera_model(path: str) -> str:
    """'<Make> <Model>' from the EXIF data, or 'unknown'."""
    with Image.open(path) as img:
        exif = img.getexif()
    name = " ".join(str(exif.get(tag, "")).strip() for tag in (271, 272)).strip()  # 271 Make, 272 Model
    return name or "unknown"
