from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageOps

from .types import ImageArray


def load_image(data: bytes) -> ImageArray:
    img = Image.open(BytesIO(data))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def load_image_with_alpha(data: bytes) -> tuple[ImageArray, ImageArray | None]:
    img = Image.open(BytesIO(data))
    img = ImageOps.exif_transpose(img)
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    if has_alpha:
        rgba = np.asarray(img.convert("RGBA"), dtype=np.uint8)
        return rgba[..., :3], rgba[..., 3]
    return np.asarray(img.convert("RGB"), dtype=np.uint8), None


def save_image(array: ImageArray, fmt: str = "PNG") -> bytes:
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    img = Image.fromarray(array, mode="RGB")
    buf = BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def resize_max_edge(array: ImageArray, max_edge: int) -> ImageArray:
    """Downscale so the longest edge is at most max_edge; no-op if already smaller."""
    h, w = array.shape[:2]
    longest = max(h, w)
    if longest <= max_edge:
        return array
    scale = max_edge / longest
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    img = Image.fromarray(array).resize((new_w, new_h), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)
