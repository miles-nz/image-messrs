from __future__ import annotations

import cv2
import numpy as np

from ...core.registry import register_effect
from ...core.types import ImageArray, ParamSpec
from ._shared import resize_to_match


@register_effect(
    name="optical_flow_blend",
    label="Optical Flow Blend",
    category="blend",
    multi_image=True,
    description="Computes motion vectors from image A to image B, warps image A along those vectors, then cross-fades into image B - a smooth morph rather than a simple cross-fade.",
    about={
        "what": "Computes motion vectors describing how content moves from image A to image B, warps A along those vectors, then cross-fades into B - producing a smooth morph rather than a flat dissolve.",
        "how_to_use": "Upload a second image, then move Blend Factor from 0 (A) to 1 (B) to scrub through the morph; raise Flow Strength to exaggerate the computed motion for a more distorted, overdriven transition.",
        "used_for": "Morphing between two similar images (portraits, poses, similar compositions) with motion-aware warping instead of a flat cross-fade.",
        "examples": "Motion-based image morphing became famous in mainstream media through music videos of the early 1990s - Michael Jackson's “Black or White” (1991) is the best-known example that introduced face-morphing effects to a mass audience - and the technique remains a staple of transition effects in motion graphics today.",
    },
    params=[
        ParamSpec(
            name="blend_factor", kind="float", default=0.5, min=0.0, max=1.0, step=0.01, label="Blend Factor (0=A, 1=B)",
            description="How far to morph from image A (0) to image B (1). Also scales how much of the computed motion warp gets applied - at 0 there's no warp or blend at all.",
        ),
        ParamSpec(
            name="flow_strength",
            kind="float",
            default=1.0,
            min=0.0,
            max=3.0,
            step=0.05,
            label="Flow Strength (>1 = overdrive/warped)",
            description="Multiplier on the computed motion displacement. 1 is the 'natural' warp strength; values above 1 exaggerate the motion for a more distorted, overdriven morph.",
        ),
    ],
)
def apply(
    image_a: ImageArray, image_b: ImageArray, blend_factor: float = 0.5, flow_strength: float = 1.0
) -> ImageArray:
    h, w = image_a.shape[:2]
    b = resize_to_match(image_b, (h, w))

    gray_a = cv2.cvtColor(image_a, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray_a, gray_b, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    t = float(np.clip(blend_factor, 0.0, 1.0))
    displacement = flow * t * float(flow_strength)

    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = (grid_x + displacement[..., 0]).astype(np.float32)
    map_y = (grid_y + displacement[..., 1]).astype(np.float32)
    warped_a = cv2.remap(image_a, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    blended = (1 - t) * warped_a.astype(np.float32) + t * b.astype(np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)
