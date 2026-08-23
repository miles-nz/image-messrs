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

_COLORS: dict[str, tuple[float, float, float]] = {
    "amber": (255.0, 140.0, 60.0),
    "red": (255.0, 60.0, 40.0),
    "magenta": (255.0, 50.0, 150.0),
}

_N_BLOBS = 3


@register_effect(
    name="light_leaks",
    label="Light Leaks",
    category="color",
    description="Adds a procedural warm glow bleeding in from an edge or corner of the frame, screen-blended on top - the light-struck streaks of a film camera with a tiny gap in its seals.",
    about={
        "what": "Generates a cluster of soft, randomly offset glowing blobs entering from a chosen edge or corner of the frame and screen-blends them on top of the image, so the glow adds light without ever darkening anything underneath.",
        "how_to_use": "Pick a Position for where the leak enters (or Random to let the Seed decide), then raise Intensity for a stronger glow and Size for a bigger, more spread-out leak. Change the Color to shift the leak's tone, and adjust Seed to get a different, still-reproducible blob arrangement.",
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
            name="color", kind="choice", default="amber", choices=list(_COLORS.keys()),
            description="Color tone of the light leak.",
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
    color: str = "amber",
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

    tint = np.array(_COLORS.get(color, _COLORS["amber"]), dtype=np.float32)
    glow = leak[..., None] * tint[None, None, :] * float(intensity)
    glow = np.clip(glow, 0, 255)

    img = image.astype(np.float32)
    out = 255.0 - (255.0 - img) * (255.0 - glow) / 255.0
    return np.clip(out, 0, 255).astype(np.uint8)
