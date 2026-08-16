from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image


def decode_mask(data_url: str | None, target_hw: tuple[int, int]) -> np.ndarray | None:
    """Decode a canvas.toDataURL() PNG into a boolean mask matching target_hw, or
    None if no mask was supplied / it failed to decode."""
    if not data_url:
        return None
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_url)
        img = Image.open(BytesIO(raw)).convert("L")
    except Exception:
        return None

    target_h, target_w = target_hw
    if img.size != (target_w, target_h):
        img = img.resize((target_w, target_h), Image.NEAREST)
    return np.asarray(img) > 127
