from __future__ import annotations

import cv2
import numpy as np

from ...core.registry import register_effect
from ...core.types import ImageArray, ParamSpec

_MAX_DUST = 8000


@register_effect(
    name="dust_scratches",
    label="Dust & Scratches",
    category="color",
    description="Overlays procedural dust specks and vertical scratch lines, the way damaged or poorly stored film looks when scanned - independent of any vintage camera profile.",
    about={
        "what": "Scatters small bright and dark specks across the frame to simulate dust and debris, and draws a handful of thin, slightly wavering vertical lines to simulate scratches - both blended on top of the image rather than baked into a specific film profile.",
        "how_to_use": "Raise Dust Density for more specks and Dust Intensity for how visible they are. Raise Scratch Count for more scratch lines and Scratch Intensity for how visible they are. Adjust Seed to get a different, still-reproducible arrangement of dust and scratches.",
        "used_for": "Adding the physical wear-and-tear look of scanned analog film to a digital image - useful on its own, or layered after a vintage camera profile for extra grunge.",
        "examples": "Dust and scratches are the most common defects in scanned film, caused by debris on the film surface and physical abrasion from repeated handling or aging - restoration tools like Kodak's Digital ICE exist specifically to remove them, while this effect deliberately adds them back for an authentically worn analog look.",
    },
    params=[
        ParamSpec(
            name="dust_density", kind="float", default=0.002, min=0.0, max=0.02, step=0.0005, label="Dust Density",
            description="Fraction of the frame covered by dust specks. 0 disables dust entirely.",
        ),
        ParamSpec(
            name="dust_intensity", kind="float", default=0.7, min=0.0, max=1.0, step=0.05, label="Dust Intensity",
            description="How visible the dust specks are.",
        ),
        ParamSpec(
            name="scratch_count", kind="int", default=3, min=0, max=20, step=1, label="Scratch Count",
            description="Number of vertical scratch lines drawn across the frame. 0 disables scratches entirely.",
        ),
        ParamSpec(
            name="scratch_intensity", kind="float", default=0.5, min=0.0, max=1.0, step=0.05, label="Scratch Intensity",
            description="How visible the scratch lines are.",
        ),
        ParamSpec(
            name="seed", kind="int", default=0, min=0, max=999999, step=1,
            description="Random seed. The same seed with the same other settings always reproduces the exact same dust and scratch pattern.",
        ),
    ],
)
def apply(
    image: ImageArray,
    dust_density: float = 0.002,
    dust_intensity: float = 0.7,
    scratch_count: int = 3,
    scratch_intensity: float = 0.5,
    seed: int = 0,
) -> ImageArray:
    h, w = image.shape[:2]
    rng = np.random.default_rng(int(seed))
    base = image.copy()

    n_dust = min(_MAX_DUST, int(h * w * max(0.0, float(dust_density))))
    if n_dust > 0 and dust_intensity > 0:
        overlay = base.copy()
        xs = rng.integers(0, w, size=n_dust)
        ys = rng.integers(0, h, size=n_dust)
        radii = rng.integers(1, 3, size=n_dust)
        bright = rng.random(n_dust) < 0.8
        for x, y, r, is_bright in zip(xs, ys, radii, bright):
            val = 255 if is_bright else 0
            cv2.circle(overlay, (int(x), int(y)), int(r), (val, val, val), -1, lineType=cv2.LINE_AA)
        a = float(dust_intensity)
        base = np.clip(base.astype(np.float32) * (1 - a) + overlay.astype(np.float32) * a, 0, 255).astype(np.uint8)

    n_scratches = max(0, int(scratch_count))
    if n_scratches > 0 and scratch_intensity > 0:
        pre_scratch = base.copy()
        overlay = base.copy()
        n_pts = 10
        for _ in range(n_scratches):
            x0 = rng.uniform(0, w)
            drift = rng.uniform(-0.05, 0.05) * w
            val = 255 if rng.random() < 0.7 else 0
            pts = np.array(
                [
                    [x0 + drift * (t / n_pts) + rng.normal(0, 1.5), t / n_pts * h]
                    for t in range(n_pts + 1)
                ],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], isClosed=False, color=(val, val, val), thickness=1, lineType=cv2.LINE_AA)
        a = float(scratch_intensity)
        base = np.clip(
            pre_scratch.astype(np.float32) * (1 - a) + overlay.astype(np.float32) * a, 0, 255
        ).astype(np.uint8)

    return base
