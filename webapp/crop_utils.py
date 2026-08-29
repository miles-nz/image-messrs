from __future__ import annotations

import numpy as np

from .store import CropRect


def apply_crop(image: np.ndarray, crop: CropRect) -> np.ndarray:
    """Slices image to the region described by crop's normalized (0..1)
    fractions, clamped to valid bounds and never producing an empty slice."""
    h, w = image.shape[:2]
    x0 = max(0, min(w - 1, round(crop.x * w)))
    y0 = max(0, min(h - 1, round(crop.y * h)))
    x1 = max(x0 + 1, min(w, round((crop.x + crop.width) * w)))
    y1 = max(y0 + 1, min(h, round((crop.y + crop.height) * h)))
    return image[y0:y1, x0:x1]
