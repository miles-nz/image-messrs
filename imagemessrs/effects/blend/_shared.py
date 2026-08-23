from __future__ import annotations

import cv2
import numpy as np


def resize_to_match(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Stretch image to exactly target_hw, distorting its aspect ratio if it differs."""
    th, tw = target_hw
    if image.shape[:2] == (th, tw):
        return image
    return cv2.resize(image, (tw, th), interpolation=cv2.INTER_LANCZOS4)


def resize_cover(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Scale image to fill target_hw preserving aspect ratio, center-cropping the overflow."""
    th, tw = target_hw
    h, w = image.shape[:2]
    if (h, w) == (th, tw):
        return image
    scale = max(tw / w, th / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    y0 = max(0, (new_h - th) // 2)
    x0 = max(0, (new_w - tw) // 2)
    cropped = resized[y0 : y0 + th, x0 : x0 + tw]
    if cropped.shape[:2] != (th, tw):
        pad_h, pad_w = th - cropped.shape[0], tw - cropped.shape[1]
        cropped = cv2.copyMakeBorder(cropped, 0, max(0, pad_h), 0, max(0, pad_w), cv2.BORDER_REPLICATE)
        cropped = cropped[:th, :tw]
    return cropped


def resize_contain(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Scale image to fit within target_hw preserving aspect ratio, edge-padding the rest."""
    th, tw = target_hw
    h, w = image.shape[:2]
    if (h, w) == (th, tw):
        return image
    scale = min(tw / w, th / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    pad_top = (th - new_h) // 2
    pad_bottom = th - new_h - pad_top
    pad_left = (tw - new_w) // 2
    pad_right = tw - new_w - pad_left
    return cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REPLICATE)
