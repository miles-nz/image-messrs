from __future__ import annotations

import cv2
import numpy as np

from ...core.registry import register_effect
from ...core.types import ImageArray, ParamSpec

_TINTS: dict[str, tuple[float, float, float]] = {
    "white": (1.0, 1.0, 1.0),
    "warm": (1.0, 0.7, 0.4),
    "cool": (0.5, 0.75, 1.0),
}


@register_effect(
    name="bloom",
    label="Bloom / Glow",
    category="color",
    description="Makes bright areas of the image glow by blurring just the highlights and adding that soft light back on top - a standalone bloom effect independent of any vintage camera profile.",
    about={
        "what": "Isolates the brightest parts of the image (above a threshold), blurs just that isolated highlight layer into a soft glow, and adds it back on top of the original - so bright areas bleed light into their surroundings instead of staying sharply contained.",
        "how_to_use": "Lower Threshold to make more of the image count as 'bright' and contribute to the glow. Raise Radius for a broader, softer glow and Intensity for a stronger one. Tint colors the glow itself - white for a neutral glow, warm for a sunny/practical-light feel, cool for a moonlit/screen-light feel.",
        "used_for": "Adding a soft, glowing quality to bright highlights, light sources, or overexposed areas - useful on its own for a dreamy or overexposed look, or layered with other effects for extra polish.",
        "examples": "Bloom is the same highlight-glow effect used throughout photography lens design (bright light scattering slightly inside the lens) and in video game/render post-processing pipelines, where it's one of the most common 'make bright things glow' post-effects. It's a generalized, standalone version of the halation baked into this app's vintage camera film profiles.",
    },
    params=[
        ParamSpec(
            name="threshold", kind="float", default=0.7, min=0.0, max=1.0, step=0.01,
            description="Normalized brightness (0=black, 1=white) above which pixels contribute to the glow. Lower this to make more of the image bloom.",
        ),
        ParamSpec(
            name="radius", kind="float", default=8.0, min=0.0, max=50.0, step=0.5,
            description="Blur radius of the glow, in pixels. Higher values spread the glow further from its source.",
        ),
        ParamSpec(
            name="intensity", kind="float", default=0.6, min=0.0, max=2.0, step=0.05,
            description="Strength of the glow added back on top of the original image. 0 disables the effect entirely.",
        ),
        ParamSpec(
            name="tint", kind="choice", default="white", choices=list(_TINTS.keys()),
            description="Color cast of the glow itself: white for neutral, warm for a sunny/practical-light feel, cool for a moonlit/screen-light feel.",
        ),
    ],
)
def apply(
    image: ImageArray,
    threshold: float = 0.7,
    radius: float = 8.0,
    intensity: float = 0.6,
    tint: str = "white",
) -> ImageArray:
    if intensity <= 0:
        return image.copy()

    img = image.astype(np.float32)
    luma = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]) / 255.0
    mask = np.clip((luma - float(threshold)) / max(1e-6, 1 - float(threshold)), 0, None)
    bright = img * mask[..., None]

    sigma = max(0.0, float(radius))
    glow = cv2.GaussianBlur(bright, (0, 0), sigmaX=sigma) if sigma > 0 else bright

    tint_arr = np.array(_TINTS.get(tint, _TINTS["white"]), dtype=np.float32)
    out = img + glow * tint_arr * float(intensity)
    return np.clip(out, 0, 255).astype(np.uint8)
