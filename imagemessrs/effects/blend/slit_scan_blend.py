from __future__ import annotations

import numpy as np

from ...core.registry import register_effect
from ...core.types import ImageArray, ParamSpec
from ._shared import resize_to_match


@register_effect(
    name="slit_scan_blend",
    label="Slit-Scan Blend",
    category="blend",
    multi_image=True,
    description="Blends image A and image B with a repeating band pattern that sweeps across the frame - each column or row reads a different point in the A/B mix, like a slit-scan camera exposing a strip of film over time.",
    about={
        "what": "Instead of one uniform crossfade, each column (or row) of the output gets its own A/B mix weight, following a repeating wave pattern across the frame - so bands of image A and image B alternate across the image rather than a single hard boundary between them.",
        "how_to_use": "Upload a second image, choose whether the bands run across Columns or Rows, then raise Frequency for more, narrower bands. Progress scrolls the whole banding pattern across the frame like a scan head sweeping over time. Raise Sharpness to turn the soft gradient bands into harder-edged stripes.",
        "used_for": "A rippling, banded alternative to a straight crossfade or wipe - good for combining two images into a single frame that reads as neither a clean composite nor a simple blend.",
        "examples": "Slit-scan photography builds an image strip-by-strip over time rather than in one exposure - a technique used for the 'stargate' sequence in 2001: A Space Odyssey and for the stretched, time-smeared look of finish-line photography. This effect borrows the strip-by-strip read pattern and applies it to blending two still images instead of scanning a single scene over time.",
    },
    params=[
        ParamSpec(
            name="axis", kind="choice", default="columns", choices=["columns", "rows"],
            description="Whether the alternating A/B bands run vertically (varying by Column) or horizontally (varying by Row).",
        ),
        ParamSpec(
            name="frequency", kind="float", default=2.0, min=0.0, max=30.0, step=0.1,
            description="Number of A/B band cycles across the frame. 0 gives a single uniform blend with no banding; higher values give more, narrower bands.",
        ),
        ParamSpec(
            name="progress", kind="float", default=0.0, min=0.0, max=1.0, step=0.01,
            description="Scrolls the banding pattern across the frame, like a scan head sweeping over time. A full 0-to-1 sweep returns the pattern to its starting position.",
        ),
        ParamSpec(
            name="sharpness", kind="float", default=0.0, min=0.0, max=1.0, step=0.05,
            description="Steepens the soft gradient bands into harder-edged stripes. 0 keeps a smooth sine gradient; 1 gives sharp, mostly-flat stripes.",
        ),
    ],
)
def apply(
    image_a: ImageArray,
    image_b: ImageArray,
    axis: str = "columns",
    frequency: float = 2.0,
    progress: float = 0.0,
    sharpness: float = 0.0,
) -> ImageArray:
    h, w = image_a.shape[:2]
    b = resize_to_match(image_b, (h, w))

    if axis == "rows":
        coord = np.arange(h, dtype=np.float32)[:, None, None]
        n = h
    else:
        coord = np.arange(w, dtype=np.float32)[None, :, None]
        n = w

    phase = 2 * np.pi * float(progress)
    weight_b = 0.5 + 0.5 * np.sin(2 * np.pi * float(frequency) * coord / max(n, 1) - phase)

    if sharpness > 0:
        k = 1.0 + float(sharpness) * 20.0
        weight_b = 1.0 / (1.0 + np.exp(-k * (weight_b - 0.5)))

    a = image_a.astype(np.float32)
    bb = b.astype(np.float32)
    out = a * (1 - weight_b) + bb * weight_b
    return np.clip(out, 0, 255).astype(np.uint8)
