from __future__ import annotations

import cv2
import numpy as np

from ...core.registry import register_effect
from ...core.types import ImageArray, ParamSpec
from ._shared import resize_to_match


@register_effect(
    name="displacement_map_blend",
    label="Displacement Map Blend",
    category="blend",
    multi_image=True,
    description="Uses image B's brightness as a displacement map, pushing image A's pixels along a fixed direction by an amount proportional to how bright or dark B is at that point - then optionally reveals B underneath.",
    about={
        "what": "Reads image B's per-pixel brightness as a push amount: pixels brighter than mid-gray get pushed one way along a chosen direction, darker pixels get pushed the other way, and mid-gray pixels stay put. This warps image A using image B's raw luminance values directly, rather than Energy Warp's approach of warping along B's edge gradients.",
        "how_to_use": "Upload a second image, raise Strength to push pixels further, and set Direction to control which way brightness pushes them. Turn on Invert Direction to flip which end (bright or dark) pushes forward. Raise Blend Factor to fade image B in on top of the warped result, from 0 (pure warped A) to 1 (pure B).",
        "used_for": "A more literal, classic 'displacement map' effect than Energy Warp - useful when you want a flat image (like a gradient, cloud texture, or noise pattern) to push another image around in one consistent direction, rather than warping along detected edges.",
        "examples": "This is the direct luminance-driven counterpart to Energy Warp's edge-driven approach - the same displacement-mapping technique found in After Effects' Displacement Map effect and in classic 'liquid distortion' filters, where a control image's brightness values drive how much (and which way) another image's pixels get pushed.",
    },
    params=[
        ParamSpec(
            name="strength", kind="float", default=15.0, min=0.0, max=100.0, step=1.0,
            description="Maximum displacement, in pixels, applied where image B is fully bright or fully dark. 0 leaves image A unwarped.",
        ),
        ParamSpec(
            name="angle", kind="float", default=0.0, min=0.0, max=360.0, step=1.0, label="Direction (degrees)",
            description="Direction pixels get pushed, in degrees (0 = right, 90 = down, 180 = left, 270 = up).",
        ),
        ParamSpec(
            name="invert", kind="bool", default=False, label="Invert Direction",
            description="Flips which end of the brightness range (bright or dark) pushes forward along Direction versus backward.",
        ),
        ParamSpec(
            name="blend_factor", kind="float", default=0.0, min=0.0, max=1.0, step=0.01, label="Blend Factor (0=warped A, 1=B)",
            description="Fades image B in on top of the warped result. 0 shows only the warped image A, 1 shows only image B.",
        ),
    ],
)
def apply(
    image_a: ImageArray,
    image_b: ImageArray,
    strength: float = 15.0,
    angle: float = 0.0,
    invert: bool = False,
    blend_factor: float = 0.0,
) -> ImageArray:
    h, w = image_a.shape[:2]
    b = resize_to_match(image_b, (h, w))

    luma = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
    norm = (luma.astype(np.float32) - 127.5) / 127.5
    if invert:
        norm = -norm

    theta = np.deg2rad(float(angle))
    dx = norm * float(strength) * np.cos(theta)
    dy = norm * float(strength) * np.sin(theta)

    grid_y, grid_x = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)
    warped_a = cv2.remap(image_a, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    t = float(np.clip(blend_factor, 0.0, 1.0))
    out = warped_a.astype(np.float32) * (1 - t) + b.astype(np.float32) * t
    return np.clip(out, 0, 255).astype(np.uint8)
