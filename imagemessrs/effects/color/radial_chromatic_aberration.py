from __future__ import annotations

import cv2
import numpy as np

from ...core.registry import register_effect
from ...core.types import ImageArray, ParamSpec


@register_effect(
    name="radial_chromatic_aberration",
    label="Radial Chromatic Aberration",
    category="color",
    description="Stretches the red channel outward and the blue channel inward around the image center, growing stronger toward the edges - the way a real camera lens fringes color at the frame's periphery. Distinct from Channel Shift's flat, uniform pixel offset.",
    about={
        "what": "Scales the red channel slightly outward and the blue channel slightly inward around the image's center point, with the amount of displacement growing from the center toward the edges - unlike Channel Shift, which offsets each channel by the same flat amount everywhere in the frame.",
        "how_to_use": "Raise Intensity for stronger color fringing. Edge Start pushes the effect out so it only appears near the edges, leaving the center untouched. Falloff controls how sharply the effect ramps up as it approaches the edge - higher values keep more of the frame clean before fringing kicks in.",
        "used_for": "Recreating the lateral chromatic aberration real camera lenses produce - color fringing that's worst at the corners and edges of the frame and near-absent at the center, unlike a uniform color-channel misalignment.",
        "examples": "This is the same radial lens artifact used inside this app's Vintage Camera Profile pipeline, pulled out here as its own standalone, fully adjustable effect so it can be applied to any image without going through a full film/camera emulation.",
    },
    params=[
        ParamSpec(
            name="intensity", kind="float", default=1.0, min=0.0, max=5.0, step=0.1,
            description="Strength of the color fringing. 0 disables the effect entirely.",
        ),
        ParamSpec(
            name="edge_start", kind="float", default=0.0, min=0.0, max=0.9, step=0.05, label="Edge Start",
            description="Normalized distance from the center (0=center, 1=corner) where the effect starts to appear. Raise this to keep the center of the image completely clean.",
        ),
        ParamSpec(
            name="falloff", kind="float", default=2.0, min=0.5, max=6.0, step=0.1,
            description="Exponent controlling how sharply the effect ramps up between Edge Start and the corners. Higher values keep more of the frame clean before fringing kicks in strongly.",
        ),
    ],
)
def apply(
    image: ImageArray,
    intensity: float = 1.0,
    edge_start: float = 0.0,
    falloff: float = 2.0,
) -> ImageArray:
    if intensity <= 0:
        return image.copy()

    h, w = image.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy

    corner_dist = float(np.sqrt(cx**2 + cy**2)) or 1.0
    r = np.sqrt(dx**2 + dy**2) / corner_dist
    edge_start = float(np.clip(edge_start, 0.0, 0.99))
    growth = np.clip((r - edge_start) / max(1e-6, 1 - edge_start), 0, 1) ** float(falloff)

    scale = 0.02 * float(intensity)
    out = np.empty_like(image)
    channel_factors = (growth * scale, np.zeros_like(growth), -growth * scale)
    for i, factor in enumerate(channel_factors):
        map_x = (cx + dx * (1 + factor)).astype(np.float32)
        map_y = (cy + dy * (1 + factor)).astype(np.float32)
        out[..., i] = cv2.remap(image[..., i], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out
