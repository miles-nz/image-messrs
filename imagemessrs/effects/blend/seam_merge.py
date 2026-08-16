from __future__ import annotations

import numpy as np

from ...core.registry import register_effect
from ...core.types import ImageArray, ParamSpec
from ..seam_carve import _find_vertical_seam
from ._shared import resize_to_match


def _grayscale(image: np.ndarray) -> np.ndarray:
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


@register_effect(
    name="seam_merge",
    label="Seam Merge (Grow Into)",
    category="blend",
    multi_image=True,
    description="Blends two images by sweeping a boundary across the frame that follows the seam where the two images are most visually similar, so image B appears to 'grow into' image A.",
    about={
        "what": "Sweeps a boundary across the frame that follows the seam where the two images are most visually similar, so image B appears to “grow into” image A rather than simply fading in.",
        "how_to_use": "Upload a second image, choose whether the boundary sweeps vertically or horizontally, then move Growth to reveal more of B; raise Feather Width to soften the boundary from a hard edge into a smoother gradient.",
        "used_for": "A more organic alternative to a cross-fade or hard split-screen - good for transitions and composites where you want the boundary itself to feel content-aware rather than a straight line.",
        "examples": "This effect adapts the same seam-finding idea behind seam carving to image compositing, following the tradition of seam-based image stitching used in panorama software to hide the join between overlapping photos.",
    },
    params=[
        ParamSpec(
            name="axis", kind="choice", default="vertical", choices=["vertical", "horizontal"],
            description="Whether the growth boundary sweeps left-to-right (vertical seam) or top-to-bottom (horizontal seam).",
        ),
        ParamSpec(
            name="progress",
            kind="float",
            default=0.5,
            min=0.0,
            max=1.0,
            step=0.01,
            label="Growth Progress (0=image A, 1=image B)",
            description="How far image B has grown into image A. 0 is entirely image A, 1 is entirely image B, and values in between reveal a wiggly boundary that follows the seam of least visual difference between the two images.",
        ),
        ParamSpec(
            name="feather", kind="int", default=3, min=0, max=50, step=1, label="Feather Width (px)",
            description="Width in pixels of the soft blend right at the boundary. 0 gives a hard, pixel-sharp edge; higher values feather the transition into a smoother gradient.",
        ),
    ],
)
def apply(
    image_a: ImageArray,
    image_b: ImageArray,
    axis: str = "vertical",
    progress: float = 0.5,
    feather: int = 3,
) -> ImageArray:
    h, w = image_a.shape[:2]
    b = resize_to_match(image_b, (h, w))

    horizontal = axis == "horizontal"
    a_work = np.transpose(image_a, (1, 0, 2)) if horizontal else image_a
    b_work = np.transpose(b, (1, 0, 2)) if horizontal else b
    wh, ww = a_work.shape[:2]

    # Seam through the region where the two images are most similar: the
    # least visually disruptive place for the A/B boundary to pass through.
    diff_energy = np.abs(_grayscale(a_work) - _grayscale(b_work))
    seam = _find_vertical_seam(diff_energy)

    # Boundary sweeps left (0) -> right (ww) as progress goes 0 -> 1, with the
    # seam adding shape-following wiggle that fades out at both endpoints so
    # progress=0 is exactly image A and progress=1 is exactly image B.
    p = float(np.clip(progress, 0.0, 1.0))
    jitter = seam.astype(np.float64) - seam.mean()
    wiggle = jitter * (4 * p * (1 - p))
    boundary = np.clip(p * ww + wiggle, 0, ww)

    feather_px = max(0, int(feather))
    out = np.empty_like(a_work)
    col_idx = np.arange(ww)
    for y in range(wh):
        b_col = boundary[y]
        if feather_px == 0:
            grown = col_idx < b_col  # already "grown into" by image B
            out[y, grown] = b_work[y, grown]
            out[y, ~grown] = a_work[y, ~grown]
        else:
            weight_b = np.clip((b_col - col_idx + feather_px) / (2 * feather_px), 0.0, 1.0)
            row_a = a_work[y].astype(np.float32)
            row_b = b_work[y].astype(np.float32)
            out[y] = (row_a * (1 - weight_b[:, None]) + row_b * weight_b[:, None]).astype(np.uint8)

    return np.transpose(out, (1, 0, 2)).copy() if horizontal else out
