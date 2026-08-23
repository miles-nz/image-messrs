from __future__ import annotations

import numpy as np

from ..core.registry import register_effect
from ..core.types import ImageArray, ParamSpec


@register_effect(
    name="scanline_jitter",
    label="Scanline Jitter",
    category="glitch",
    description="Shifts each row of the image horizontally by a smooth wave plus random per-row jitter, with an extra offset on alternating rows - the classic wobble and interlace tear of a mistracking VHS tape.",
    about={
        "what": "Displaces each horizontal row of pixels independently: a smooth sine wave shifts rows in a rolling wobble, random per-row jitter adds noise on top, and an extra offset applied only to alternating rows splits the image into two misaligned interlaced fields.",
        "how_to_use": "Start with Wave Amplitude and Wave Frequency to dial in a slow rolling wobble, then add Row Jitter for a noisier, more erratic wiggle. Raise Interlace Offset to pull alternating rows apart into a torn, comb-like look. Turn on Wrap Edges to keep pixels pushed off one side instead of leaving black gaps.",
        "used_for": "Recreating VHS tracking errors, analog signal wobble, and interlaced-video field tearing - a staple of lo-fi/VHS aesthetic edits and glitch art.",
        "examples": "This mimics the horizontal sync wobble of a damaged or mistracking VHS tape, and the comb-like tearing that shows up when interlaced video (two alternating fields per frame) is captured or deinterlaced incorrectly - both widely referenced in retro/analog-glitch visual styles.",
    },
    params=[
        ParamSpec(
            name="amplitude", kind="float", default=8.0, min=0.0, max=100.0, step=1.0, label="Wave Amplitude (px)",
            description="Peak horizontal displacement of the smooth rolling wave, in pixels. 0 disables the wave component.",
        ),
        ParamSpec(
            name="frequency", kind="float", default=3.0, min=0.1, max=30.0, step=0.1, label="Wave Frequency (cycles)",
            description="Number of full wave cycles from the top of the image to the bottom. Higher values give a tighter, more rippled wobble.",
        ),
        ParamSpec(
            name="jitter", kind="float", default=2.0, min=0.0, max=50.0, step=0.5, label="Row Jitter (px)",
            description="Maximum random horizontal offset added independently to each row, on top of the smooth wave. 0 disables random jitter, leaving only the wave.",
        ),
        ParamSpec(
            name="interlace_offset", kind="int", default=6, min=-50, max=50, step=1, label="Interlace Offset (px)",
            description="Constant extra horizontal offset applied only to every other row, splitting the image into two misaligned interlaced fields. 0 disables interlace tearing.",
        ),
        ParamSpec(
            name="wrap", kind="bool", default=True, label="Wrap Edges",
            description="When on, pixels shifted off one edge of a row wrap around to the opposite edge instead of leaving a black gap.",
        ),
        ParamSpec(
            name="seed", kind="int", default=0, min=0, max=999999, step=1,
            description="Random seed for the per-row jitter. The same seed with the same other settings always reproduces the exact same jitter pattern.",
        ),
    ],
)
def apply(
    image: ImageArray,
    amplitude: float = 8.0,
    frequency: float = 3.0,
    jitter: float = 2.0,
    interlace_offset: int = 6,
    wrap: bool = True,
    seed: int = 0,
) -> ImageArray:
    h, w = image.shape[:2]
    rows = np.arange(h)

    dx = float(amplitude) * np.sin(2 * np.pi * float(frequency) * rows / max(h, 1))
    if jitter > 0:
        rng = np.random.default_rng(int(seed))
        dx = dx + rng.uniform(-float(jitter), float(jitter), size=h)
    if interlace_offset:
        dx[1::2] += float(interlace_offset)
    dx = np.round(dx).astype(np.int64)

    if not np.any(dx):
        return image.copy()

    col_idx = np.arange(w)[None, :] - dx[:, None]
    if wrap:
        col_idx = col_idx % w
        valid = None
    else:
        valid = (col_idx >= 0) & (col_idx < w)
        col_idx = np.clip(col_idx, 0, w - 1)

    gather_idx = np.repeat(col_idx[..., None], image.shape[2], axis=2)
    out = np.take_along_axis(image, gather_idx, axis=1)
    if valid is not None:
        out = np.where(valid[..., None], out, 0)
    return out.astype(np.uint8)
