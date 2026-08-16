from __future__ import annotations

import cv2
import numpy as np

from ...core.registry import register_effect
from ...core.types import ImageArray, ParamSpec
from ._shared import resize_to_match


def _grayscale(image: np.ndarray) -> np.ndarray:
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


@register_effect(
    name="energy_warp",
    label="Energy Warp",
    category="blend",
    multi_image=True,
    description="Uses image A's edges as a displacement map to push and pull image B's pixels around, warping B along A's contours.",
    about={
        "what": "Computes an edge map from image A and uses it as a displacement field to push and pull image B's pixels around, warping B along A's contours.",
        "how_to_use": "Upload a second image, then raise Warp Strength to pull B further along A's edges; increase Edge Blur if you want broader, smoother warps instead of pixel-level jitter.",
        "used_for": "Making one image's structure visibly distort another - a liquify/displacement-map effect driven by real image content rather than a hand-drawn mask.",
        "examples": "Displacement mapping - using one image's luminance or edges to distort another - is a long-standing technique in music-video and VFX work for liquid, melting, or “heat haze” transitions, and is a core building block in tools like After Effects' Displacement Map effect.",
    },
    params=[
        ParamSpec(
            name="strength", kind="float", default=0.1, min=0.0, max=2.0, step=0.01, label="Warp Strength",
            description="How far image B's pixels get displaced along image A's edge gradients. 0 leaves image B unwarped; higher values pull it further, producing a more distorted, liquified look near strong edges.",
        ),
        ParamSpec(
            name="blur_ksize", kind="int", default=3, min=1, max=31, step=2, label="Edge Map Blur",
            description="Blur applied to image A's edge map before computing the displacement. Higher values smooth out fine detail, producing broader, softer warps instead of pixel-level jitter.",
        ),
    ],
)
def apply(image_a: ImageArray, image_b: ImageArray, strength: float = 0.1, blur_ksize: int = 3) -> ImageArray:
    h, w = image_a.shape[:2]
    b = resize_to_match(image_b, (h, w))

    gray_a = _grayscale(image_a).astype(np.float32)
    k = max(1, int(blur_ksize) | 1)
    if k > 1:
        gray_a = cv2.GaussianBlur(gray_a, (k, k), 0)

    gx = cv2.Sobel(gray_a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray_a, cv2.CV_32F, 0, 1, ksize=3)

    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = (grid_x + gx * float(strength)).astype(np.float32)
    map_y = (grid_y + gy * float(strength)).astype(np.float32)

    return cv2.remap(b, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
