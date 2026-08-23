from __future__ import annotations

import numpy as np

from ...core.registry import register_effect
from ...core.types import ImageArray, ParamSpec
from ._shared import resize_to_match


@register_effect(
    name="difference_blend",
    label="Difference / Double Exposure",
    category="blend",
    multi_image=True,
    description="Combines two images with a Difference or Double Exposure blend mode - Difference highlights where they differ, Double Exposure adds their light together like two exposures on one frame of film.",
    about={
        "what": "Two simple per-pixel blend modes between image A and image B: Difference takes the absolute value of A minus B, so identical areas go black and differing areas glow by how much they differ. Double Exposure uses a screen blend (light adds, dark disappears), the way two exposures overlapping on the same piece of film combine.",
        "how_to_use": "Upload a second image, pick a Mode, then use Blend Factor to crossfade from the untouched image A (0) to the fully blended result (1).",
        "used_for": "Difference mode is a quick way to spot or highlight changes between two similar images. Double Exposure mode recreates the classic analog photography technique of exposing two images onto one frame - bright areas of both images show through, dark areas recede.",
        "examples": "Difference mode is the same blend mode found in Photoshop and other editors under 'Difference', commonly used for change detection or glitchy 'ghosting' effects. Double Exposure mode mimics the in-camera multiple-exposure technique popularized by toy film cameras like the Diana and Holga, where two shots share one frame and their light adds together.",
    },
    params=[
        ParamSpec(
            name="mode", kind="choice", default="double_exposure", choices=["double_exposure", "difference"],
            description="Double Exposure screen-blends the two images together (light adds, dark disappears), like two overlapping film exposures. Difference takes the absolute per-pixel difference, so identical areas go black and differing areas glow.",
        ),
        ParamSpec(
            name="blend_factor", kind="float", default=1.0, min=0.0, max=1.0, step=0.01, label="Blend Factor (0=A, 1=blended)",
            description="Crossfades from the untouched image A (0) to the fully blended result (1).",
        ),
    ],
)
def apply(
    image_a: ImageArray, image_b: ImageArray, mode: str = "double_exposure", blend_factor: float = 1.0
) -> ImageArray:
    h, w = image_a.shape[:2]
    b = resize_to_match(image_b, (h, w))

    a = image_a.astype(np.float32)
    bb = b.astype(np.float32)

    if mode == "difference":
        blended = np.abs(a - bb)
    else:
        blended = 255.0 - (255.0 - a) * (255.0 - bb) / 255.0

    t = float(np.clip(blend_factor, 0.0, 1.0))
    out = a * (1 - t) + blended * t
    return np.clip(out, 0, 255).astype(np.uint8)
