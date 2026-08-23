from __future__ import annotations

import cv2
import numpy as np

from ...core.registry import register_effect
from ...core.types import ImageArray, ParamSpec
from ..seam_carve import resize_seam_carve
from ._shared import resize_contain, resize_cover, resize_to_match

_CLONE_MODES = {"normal": cv2.NORMAL_CLONE, "mixed": cv2.MIXED_CLONE}
_FIT_MODES = {
    "cover": resize_cover,
    "contain": resize_contain,
    "stretch": resize_to_match,
    "seam": resize_seam_carve,
}


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
        "how_to_use": "Upload a second image, paint (or leave the default) the region of B to clone in, then choose Normal for a seamlessly lit clone or Mixed to keep whichever image has stronger local detail at each point for a more chaotic, texture-preserving blend. If A and B have different aspect ratios, Canvas picks whose frame the output keeps, and Aspect Fit controls how the other image is resized into it instead of being stretched.",
        "used_for": "Seamlessly compositing an object or region from one photo into another - the same underlying technique behind Photoshop's Healing Brush and “seamless cloning” tools.",
        "examples": "Poisson blending comes from the 2003 SIGGRAPH paper “Poisson Image Editing” by Pérez, Gangnet, and Blake, and is the mathematical basis for the gradient-domain cloning/healing tools now standard in photo editors.",
    },
    params=[
        ParamSpec(
            name="mode", kind="choice", default="normal", choices=["normal", "mixed"],
            description="Normal blends image B's region on top, matching lighting/color at the seam. Mixed instead keeps whichever image has stronger local detail at each point, preserving texture from both but looking more chaotic.",
        ),
        ParamSpec(
            name="canvas", kind="choice", default="image_a", choices=["image_a", "image_b"],
            label="Canvas",
            description="Which image's dimensions the output keeps. Image A (default) keeps A's frame and fits B into it. Image B keeps B's frame instead - the roles swap, and A gets fit into B's frame using the Aspect Fit setting below.",
        ),
        ParamSpec(
            name="fit", kind="choice", default="cover", choices=["cover", "contain", "stretch", "seam"],
            label="Aspect Fit",
            description="How to reconcile the two images' aspect ratios when they differ - applied to whichever image isn't the chosen Canvas. Cover scales it to fill the frame and crops the overflow (no distortion, may lose some edges). Contain scales it to fit entirely within the frame and pads the rest by extending edge pixels (no distortion, no cropping). Stretch resizes it to exactly match the canvas's dimensions, distorting it if the ratios differ. Seam fits it like Contain, then grows it the rest of the way using seam carving's content-aware duplication instead of a padded border - no cropping or stretching, but can look increasingly warped the larger the aspect-ratio gap.",
        ),
        ParamSpec(
            name="mask", kind="mask", default=None, label="Region of image B to blend in",
            description="Paint the region of image B to clone into image A. Leave blank to use a centered rectangle covering most of image B.",
        ),
    ],
)
def apply(
    image_a: ImageArray,
    image_b: ImageArray,
    mode: str = "normal",
    canvas: str = "image_a",
    fit: str = "cover",
    mask: np.ndarray | None = None,
) -> ImageArray:
    fit_fn = _FIT_MODES.get(fit, resize_cover)
    if canvas == "image_b":
        h, w = image_b.shape[:2]
        dst = fit_fn(image_a, (h, w))
        src = image_b
    else:
        h, w = image_a.shape[:2]
        dst = image_a
        src = fit_fn(image_b, (h, w))

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
        return cv2.seamlessClone(src, dst, mask_arr, center, flags)
    except cv2.error:
        return dst.copy()
