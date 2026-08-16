from __future__ import annotations

import cv2
import numpy as np

from ...core.registry import register_effect
from ...core.types import ImageArray, ParamSpec
from ._shared import resize_to_match

_CLONE_MODES = {"normal": cv2.NORMAL_CLONE, "mixed": cv2.MIXED_CLONE}


def _default_mask(shape: tuple[int, int], margin: int) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    m = max(1, margin)
    mask[m : h - m, m : w - m] = 255
    return mask


@register_effect(
    name="poisson_blend",
    label="Poisson (Seamless) Blend",
    category="blend",
    multi_image=True,
    accepts_mask=True,
    description="Seamlessly clones a region of image B into image A using Poisson blending, matching gradients at the boundary so the seam disappears (or, in Mixed mode, blends texture from both images).",
    about={
        "what": "Clones a region of image B into image A by matching gradients at the boundary (Poisson blending) rather than just pasting pixels, so the seam between the two images disappears.",
        "how_to_use": "Upload a second image, paint (or leave the default) the region of B to clone in, then choose Normal for a seamlessly lit clone or Mixed to keep whichever image has stronger local detail at each point for a more chaotic, texture-preserving blend.",
        "used_for": "Seamlessly compositing an object or region from one photo into another - the same underlying technique behind Photoshop's Healing Brush and “seamless cloning” tools.",
        "examples": "Poisson blending comes from the 2003 SIGGRAPH paper “Poisson Image Editing” by Pérez, Gangnet, and Blake, and is the mathematical basis for the gradient-domain cloning/healing tools now standard in photo editors.",
    },
    params=[
        ParamSpec(
            name="mode", kind="choice", default="normal", choices=["normal", "mixed"],
            description="Normal blends image B's region on top, matching lighting/color at the seam. Mixed instead keeps whichever image has stronger local detail at each point, preserving texture from both but looking more chaotic.",
        ),
        ParamSpec(
            name="mask", kind="mask", default=None, label="Region of image B to blend in",
            description="Paint the region of image B to clone into image A. Leave blank to use a centered rectangle covering most of image B.",
        ),
    ],
)
def apply(
    image_a: ImageArray, image_b: ImageArray, mode: str = "normal", mask: np.ndarray | None = None
) -> ImageArray:
    h, w = image_a.shape[:2]
    src = resize_to_match(image_b, (h, w))

    if mask is None:
        mask_arr = _default_mask((h, w), margin=max(2, min(h, w) // 20))
    else:
        mask_resized = (
            mask if mask.shape[:2] == (h, w) else cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        )
        mask_arr = np.where(mask_resized > 0, 255, 0).astype(np.uint8)

    center = (w // 2, h // 2)
    flags = _CLONE_MODES.get(mode, cv2.NORMAL_CLONE)

    try:
        return cv2.seamlessClone(src, image_a, mask_arr, center, flags)
    except cv2.error:
        return image_a.copy()
