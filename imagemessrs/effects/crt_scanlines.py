from __future__ import annotations

import numpy as np

from ..core.registry import register_effect
from ..core.types import ImageArray, ParamSpec


@register_effect(
    name="crt_scanlines",
    label="CRT Scanlines",
    category="glitch",
    description="Darkens alternating rows and fringes the red/blue channels slightly out of register, mimicking a CRT's visible scanlines and RGB phosphor triads.",
    about={
        "what": "Two effects layered together: periodic rows are darkened to simulate the visible gaps between a CRT's scan lines, and the red and blue channels are nudged one pixel apart to simulate the red/green/blue phosphor triads of a shadow mask or aperture grille.",
        "how_to_use": "Set Scanline Spacing to how many pixel-rows apart the dark lines fall (2 gives the classic alternating-row look), and Scanline Darkness for how visible they are. Raise Subpixel Fringe for more visible RGB triad separation, and Brightness Boost to brighten the un-darkened rows back up if the scanlines make the image feel too dim overall.",
        "used_for": "Recreating the look of a CRT television or monitor - a staple of retro-gaming screenshots, VHS/analog-TV aesthetic edits, and pixel-art presentation.",
        "examples": "This mimics the physical structure of a cathode-ray tube display: the visible gaps between interlaced scan lines, and the red/green/blue phosphor dots or stripes (the shadow mask or aperture grille) that make up each pixel - the same visual texture emulated by 'CRT filter' shaders in retro game emulators.",
    },
    params=[
        ParamSpec(
            name="line_spacing", kind="int", default=2, min=1, max=8, step=1, label="Scanline Spacing (px)",
            description="How many pixel-rows apart the darkened scanlines fall. 2 darkens every other row, the classic look; higher values give sparser, more spread-out lines.",
        ),
        ParamSpec(
            name="line_darkness", kind="float", default=0.35, min=0.0, max=1.0, step=0.05, label="Scanline Darkness",
            description="How much darker the scanline rows are than the rest of the image. 0 disables scanlines entirely, 1 makes them fully black.",
        ),
        ParamSpec(
            name="subpixel_strength", kind="float", default=0.3, min=0.0, max=1.0, step=0.05, label="Subpixel Fringe",
            description="How strongly the red and blue channels are nudged apart to simulate RGB phosphor triads. 0 disables the fringing.",
        ),
        ParamSpec(
            name="brightness_boost", kind="float", default=0.1, min=0.0, max=1.0, step=0.05, label="Brightness Boost",
            description="Brightens the non-scanline rows to compensate for the overall dimming effect of the scanlines. 0 leaves them at their original brightness.",
        ),
    ],
)
def apply(
    image: ImageArray,
    line_spacing: int = 2,
    line_darkness: float = 0.35,
    subpixel_strength: float = 0.3,
    brightness_boost: float = 0.1,
) -> ImageArray:
    h, w = image.shape[:2]
    out = image.astype(np.float32)

    if subpixel_strength > 0:
        r_shifted = np.roll(out[..., 0], -1, axis=1)
        b_shifted = np.roll(out[..., 2], 1, axis=1)
        s = float(subpixel_strength)
        out[..., 0] = out[..., 0] * (1 - s) + r_shifted * s
        out[..., 2] = out[..., 2] * (1 - s) + b_shifted * s

    spacing = max(1, int(line_spacing))
    row_idx = np.arange(h)
    is_scanline = (row_idx % spacing) == 0
    mult = np.ones(h, dtype=np.float32)
    mult[is_scanline] = 1.0 - float(line_darkness)
    if brightness_boost > 0:
        mult[~is_scanline] *= 1.0 + float(brightness_boost)
    out *= mult[:, None, None]

    return np.clip(out, 0, 255).astype(np.uint8)
