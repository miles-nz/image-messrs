from __future__ import annotations

import cv2
import numpy as np

from ...core.registry import register_effect
from ...core.types import ImageArray, ParamSpec

_BORDER_MODES = {
    "reflect": cv2.BORDER_REFLECT,
    "replicate": cv2.BORDER_REPLICATE,
    "black": cv2.BORDER_CONSTANT,
}


@register_effect(
    name="lens_distort",
    label="Lens Distortion",
    category="distort",
    description="Bulges or pinches the image radially around its center, like a convex (barrel/fisheye) or concave (pincushion) camera lens.",
    about={
        "what": "Warps the image outward from its center (positive Strength) or inward toward it (negative Strength), following a radial power-law curve - the same family of curve used by simple lens-distortion and 'spherize' filters. Positive values bulge and magnify the center like a wide-angle or fisheye lens; negative values pinch the center inward like a telephoto or concave lens.",
        "how_to_use": "Raise Strength for a barrel/fisheye bulge, or lower it below 0 for a pincushion pinch. Edge Radius controls how far from the center the distortion reaches before fading out - smaller values concentrate the warp near the middle. Turn on Fisheye to exaggerate the same warp into a much stronger wide-angle bulge. Edge Fill controls what appears in any gaps revealed at the image's border.",
        "used_for": "Adding a physical, lens-like warp to an image - a bulging 'through the peephole' or fisheye look, or a pinched, concave distortion - as a standalone creative effect rather than a camera calibration correction.",
        "examples": "Barrel and pincushion distortion are named for how a real lens bends a rectangular grid - barrel distortion bows the sides outward like a barrel, pincushion distortion pulls them inward like a pincushion. Fisheye lenses push barrel distortion to its extreme for the dramatic ultra-wide-angle look seen in action-camera and skateboarding photography.",
    },
    params=[
        ParamSpec(
            name="strength", kind="float", default=0.4, min=-0.95, max=0.95, step=0.01,
            description="Positive values bulge the image outward from the center (barrel/fisheye); negative values pinch it inward (pincushion). 0 leaves the image unchanged.",
        ),
        ParamSpec(
            name="radius", kind="float", default=1.0, min=0.2, max=2.0, step=0.05, label="Edge Radius",
            description="How far from the center, in units of half the image's shorter side, the distortion reaches before tapering off. Smaller values concentrate the warp near the middle of the frame.",
        ),
        ParamSpec(
            name="fisheye", kind="bool", default=False, label="Fisheye",
            description="Exaggerates Strength into a much more extreme wide-angle bulge, for a full fisheye-lens look.",
        ),
        ParamSpec(
            name="edge_mode", kind="choice", default="reflect", choices=["reflect", "replicate", "black"], label="Edge Fill",
            description="What fills any gaps revealed at the image's border by the warp: Reflect mirrors nearby content, Replicate stretches the edge pixels, Black fills with solid black.",
        ),
    ],
)
def apply(
    image: ImageArray,
    strength: float = 0.4,
    radius: float = 1.0,
    fisheye: bool = False,
    edge_mode: str = "reflect",
) -> ImageArray:
    if strength == 0:
        return image.copy()

    h, w = image.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    scale = max(1.0, min(cx, cy) * max(0.2, float(radius)))

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    r = np.sqrt(dx**2 + dy**2) / scale
    r_safe = np.maximum(r, 1.0 / scale)

    s = float(np.clip(strength, -0.95, 0.95))
    if fisheye:
        s = float(np.clip(s * 2.2, -0.97, 0.97))
    exponent = 1.0 - s

    factor = r_safe ** (exponent - 1.0)
    map_x = (cx + dx * factor).astype(np.float32)
    map_y = (cy + dy * factor).astype(np.float32)

    border = _BORDER_MODES.get(edge_mode, cv2.BORDER_REFLECT)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=border)
