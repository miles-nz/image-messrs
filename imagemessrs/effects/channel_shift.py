from __future__ import annotations

import numpy as np

from ..core.registry import register_effect
from ..core.types import ImageArray, ParamSpec


def _shift_channel(channel: np.ndarray, dx: int, dy: int, wrap: bool) -> np.ndarray:
    shifted = np.roll(channel, shift=(dy, dx), axis=(0, 1))
    if wrap:
        return shifted
    if dy > 0:
        shifted[:dy, :] = 0
    elif dy < 0:
        shifted[dy:, :] = 0
    if dx > 0:
        shifted[:, :dx] = 0
    elif dx < 0:
        shifted[:, dx:] = 0
    return shifted


@register_effect(
    name="channel_shift",
    label="Channel Shift",
    category="glitch",
    description="Offsets the red, green, and blue channels independently, creating chromatic-aberration-style color fringing.",
    about={
        "what": "Shifts the red, green, and blue color channels independently in x and y, so they no longer line up - producing color fringing along edges.",
        "how_to_use": "Nudge one channel (say Red X) a few pixels at a time and watch edges split into color fringes; combine offsets on more than one channel for a more chaotic look. Turn on Wrap to keep pixels pushed off one edge instead of leaving black gaps.",
        "used_for": "Recreating the look of chromatic aberration or misaligned color-separation printing - a fast way to add a glitchy, VHS/retro-tech or lens-distortion feel to a photo.",
        "examples": "This mimics real chromatic aberration from imperfect camera lenses and the RGB misregistration of old analog color TV and print processes - an aesthetic widely embraced in glitch art and vaporwave-era design.",
    },
    params=[
        ParamSpec(
            name="red_dx", kind="int", default=0, min=-50, max=50, step=1, label="Red X",
            description="Horizontal shift of the red channel, in pixels. Positive moves it right, negative left.",
        ),
        ParamSpec(
            name="red_dy", kind="int", default=0, min=-50, max=50, step=1, label="Red Y",
            description="Vertical shift of the red channel, in pixels. Positive moves it down, negative up.",
        ),
        ParamSpec(
            name="green_dx", kind="int", default=0, min=-50, max=50, step=1, label="Green X",
            description="Horizontal shift of the green channel, in pixels. Positive moves it right, negative left.",
        ),
        ParamSpec(
            name="green_dy", kind="int", default=0, min=-50, max=50, step=1, label="Green Y",
            description="Vertical shift of the green channel, in pixels. Positive moves it down, negative up.",
        ),
        ParamSpec(
            name="blue_dx", kind="int", default=0, min=-50, max=50, step=1, label="Blue X",
            description="Horizontal shift of the blue channel, in pixels. Positive moves it right, negative left.",
        ),
        ParamSpec(
            name="blue_dy", kind="int", default=0, min=-50, max=50, step=1, label="Blue Y",
            description="Vertical shift of the blue channel, in pixels. Positive moves it down, negative up.",
        ),
        ParamSpec(
            name="wrap", kind="bool", default=False, label="Wrap Edges",
            description="When on, pixels shifted off one edge wrap around to the opposite edge instead of leaving a black gap.",
        ),
    ],
)
def apply(
    image: ImageArray,
    red_dx: int = 0,
    red_dy: int = 0,
    green_dx: int = 0,
    green_dy: int = 0,
    blue_dx: int = 0,
    blue_dy: int = 0,
    wrap: bool = False,
) -> ImageArray:
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    r = _shift_channel(r, int(red_dx), int(red_dy), wrap)
    g = _shift_channel(g, int(green_dx), int(green_dy), wrap)
    b = _shift_channel(b, int(blue_dx), int(blue_dy), wrap)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)
