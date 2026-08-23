from __future__ import annotations

import cv2
import numpy as np

from ..core.registry import register_effect
from ..core.types import ImageArray, ParamSpec


@register_effect(
    name="wave_displace",
    label="Wave Displacement",
    category="glitch",
    description="Warps the whole image with smooth sinusoidal waves - horizontal position rippling with vertical position and vice versa - for a liquid, heat-haze, or analog-video 'wobbulator' distortion.",
    about={
        "what": "Displaces every pixel by a smooth sine wave: horizontal position shifts based on a wave running down the image's height, and vertical position shifts based on a wave running across its width. Unlike Block Displacement or Scanline Jitter's blocky/row-based movement, this produces a continuous, liquid ripple across the whole image.",
        "how_to_use": "Raise Amplitude X and Amplitude Y to make the ripple stronger, and Frequency X / Frequency Y to make the waves tighter (more ripples) or broader (fewer, wider ripples). Nudge Wave Phase to shift where the ripple pattern sits without changing its shape - useful for finding a pleasing frame from an otherwise deterministic wave.",
        "used_for": "A smooth, continuous distortion for liquid/melting looks, heat-haze effects, or an analog-video 'wobbulator' feel - distinct from this app's other glitch effects, which all work in discrete blocks, rows, or pixels rather than a continuous field.",
        "examples": "This is the same idea behind Photoshop's Ripple/Wave filters and the analog video synthesizers (like Nam June Paik's wobbulator-based work) that physically distorted a CRT's scan pattern with sine waves - a technique later absorbed into VJ and glitch-art visual culture.",
    },
    params=[
        ParamSpec(
            name="amplitude_x", kind="float", default=10.0, min=0.0, max=100.0, step=1.0, label="Amplitude X (px)",
            description="Peak horizontal displacement, in pixels, driven by the wave running down the image's height.",
        ),
        ParamSpec(
            name="amplitude_y", kind="float", default=10.0, min=0.0, max=100.0, step=1.0, label="Amplitude Y (px)",
            description="Peak vertical displacement, in pixels, driven by the wave running across the image's width.",
        ),
        ParamSpec(
            name="frequency_x", kind="float", default=3.0, min=0.1, max=30.0, step=0.1, label="Frequency X (cycles)",
            description="Number of wave cycles across the image's width, controlling the vertical displacement wave.",
        ),
        ParamSpec(
            name="frequency_y", kind="float", default=3.0, min=0.1, max=30.0, step=0.1, label="Frequency Y (cycles)",
            description="Number of wave cycles down the image's height, controlling the horizontal displacement wave.",
        ),
        ParamSpec(
            name="phase", kind="float", default=0.0, min=0.0, max=6.2832, step=0.01, label="Wave Phase",
            description="Shifts where the wave pattern starts, in radians, without changing its amplitude or frequency - lets you pick a different-looking frame from the same deterministic wave.",
        ),
    ],
)
def apply(
    image: ImageArray,
    amplitude_x: float = 10.0,
    amplitude_y: float = 10.0,
    frequency_x: float = 3.0,
    frequency_y: float = 3.0,
    phase: float = 0.0,
) -> ImageArray:
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    map_x = xx + float(amplitude_x) * np.sin(2 * np.pi * float(frequency_y) * yy / max(h, 1) + float(phase))
    map_y = yy + float(amplitude_y) * np.sin(2 * np.pi * float(frequency_x) * xx / max(w, 1) + float(phase))

    return cv2.remap(
        image, map_x.astype(np.float32), map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )
