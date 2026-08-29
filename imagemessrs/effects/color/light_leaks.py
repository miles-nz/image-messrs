from __future__ import annotations

import numpy as np

from ...core.registry import register_effect
from ...core.types import ImageArray, ParamSpec

_POSITIONS: dict[str, tuple[float, float]] = {
    "top-left": (0.0, 0.0),
    "top-right": (0.0, 1.0),
    "bottom-left": (1.0, 0.0),
    "bottom-right": (1.0, 1.0),
    "top": (0.0, 0.5),
    "bottom": (1.0, 0.5),
    "left": (0.5, 0.0),
    "right": (0.5, 1.0),
}

_DEFAULT_COLOR = "#ff8c3c"  # amber
_DEFAULT_EDGE_COLOR = "#ff3c28"  # deep red
_FALLBACK_TINT = (255.0, 140.0, 60.0)

_CORE_PRESETS = ["#fff4d6", "#ffe08a", "#ff8c3c", "#ff6a3c"]
_EDGE_PRESETS = ["#ff3c28", "#ff2f6e", "#c81ee0", "#7a2bff"]

_N_BLOBS = 3


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    s = str(hex_color).strip().lstrip("#")
    if len(s) != 6:
        return _FALLBACK_TINT
    try:
        return (float(int(s[0:2], 16)), float(int(s[2:4], 16)), float(int(s[4:6], 16)))
    except ValueError:
        return _FALLBACK_TINT


@register_effect(
    name="light_leaks",
    label="Light Leaks",
    category="color",
    description="Adds a procedural warm glow bleeding in from an edge or corner of the frame, screen-blended on top - the light-struck streaks of a film camera with a tiny gap in its seals.",
    about={
        "what": "Generates a cluster of soft, randomly offset glowing blobs entering from a chosen edge or corner of the frame and screen-blends them on top of the image, so the glow adds light without ever darkening anything underneath. The glow isn't a single flat tint - it blends from a hot Core Color at each blob's center out to a cooler Edge Color at its fringes, the way real light leaks shift in color across their spread.",
        "how_to_use": "Pick a Position for where the leak enters (or Random to let the Seed decide), then raise Intensity for a stronger glow and Size for a bigger, more spread-out leak. Pick a Core Color and Edge Color from the presets or the swatch for the leak's hot center and fringe tones - warm ambers fading to reds and magentas read as classic film light leaks - and adjust Seed to get a different, still-reproducible blob arrangement.",
        "used_for": "Recreating the warm, streaky light leaks that show up on film shot in a camera with imperfect light seals - a popular stylistic flourish in film-emulation and lo-fi photo editing.",
        "examples": "Real light leaks happen when stray light sneaks past a film camera's seals or through a damaged canister and exposes the edge of the film - an accident that toy cameras like the Holga became known for, and that photo apps have emulated as a deliberate stylistic overlay ever since.",
    },
    params=[
        ParamSpec(
            name="position", kind="choice", default="top-right", choices=list(_POSITIONS.keys()) + ["random"],
            description="Which edge or corner the light leak enters from. Random picks one based on Seed.",
        ),
        ParamSpec(
            name="intensity", kind="float", default=0.6, min=0.0, max=1.5, step=0.05,
            description="Strength of the glow. 0 disables the effect entirely.",
        ),
        ParamSpec(
            name="size", kind="float", default=1.0, min=0.2, max=3.0, step=0.1,
            description="Relative size and spread of the leak. Larger values bleed further into the frame.",
        ),
        ParamSpec(
            name="color", kind="color", default=_DEFAULT_COLOR, label="Core Color", choices=_CORE_PRESETS,
            description="Color at the hot center of each glowing blob. Pick a preset or any color from the swatch.",
        ),
        ParamSpec(
            name="edge_color", kind="color", default=_DEFAULT_EDGE_COLOR, label="Edge Color", choices=_EDGE_PRESETS,
            description="Color the glow fades toward at each blob's fringes. Pick a preset or any color from the swatch.",
        ),
        ParamSpec(
            name="seed", kind="int", default=0, min=0, max=999999, step=1,
            description="Random seed controlling the leak's blob shape and position. The same seed with the same other settings always reproduces the exact same leak.",
        ),
    ],
)
def apply(
    image: ImageArray,
    position: str = "top-right",
    intensity: float = 0.6,
    size: float = 1.0,
    color: str = _DEFAULT_COLOR,
    edge_color: str = _DEFAULT_EDGE_COLOR,
    seed: int = 0,
) -> ImageArray:
    if intensity <= 0:
        return image.copy()

    h, w = image.shape[:2]
    rng = np.random.default_rng(int(seed))

    pos_key = str(rng.choice(list(_POSITIONS.keys()))) if position == "random" else position
    cy_frac, cx_frac = _POSITIONS.get(pos_key, _POSITIONS["top-right"])
    cy, cx = cy_frac * h, cx_frac * w

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    diag = float(np.sqrt(h**2 + w**2))
    base_radius = max(1.0, diag * 0.5 * float(size))

    leak = np.zeros((h, w), dtype=np.float32)
    for _ in range(_N_BLOBS):
        offset_y = rng.uniform(-0.15, 0.15) * h
        offset_x = rng.uniform(-0.15, 0.15) * w
        radius = base_radius * rng.uniform(0.5, 1.1)
        d2 = (yy - (cy + offset_y)) ** 2 + (xx - (cx + offset_x)) ** 2
        leak += np.exp(-d2 / (2 * radius**2)) * rng.uniform(0.6, 1.0)

    peak = float(leak.max())
    if peak > 0:
        leak = leak / peak

    core_tint = np.array(_hex_to_rgb(color), dtype=np.float32)
    edge_tint = np.array(_hex_to_rgb(edge_color), dtype=np.float32)
    t = leak[..., None]
    tint_field = core_tint[None, None, :] * t + edge_tint[None, None, :] * (1 - t)
    glow = tint_field * t * float(intensity)
    glow = np.clip(glow, 0, 255)

    img = image.astype(np.float32)
    out = 255.0 - (255.0 - img) * (255.0 - glow) / 255.0
    return np.clip(out, 0, 255).astype(np.uint8)
