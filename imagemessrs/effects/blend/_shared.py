from __future__ import annotations

import cv2
import numpy as np


def resize_to_match(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    if image.shape[:2] == (th, tw):
        return image
    return cv2.resize(image, (tw, th), interpolation=cv2.INTER_LANCZOS4)
