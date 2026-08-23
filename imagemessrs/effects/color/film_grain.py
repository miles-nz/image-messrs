from __future__ import annotations

import cv2
import numpy as np

from ...core.registry import register_effect
from ...core.types import ImageArray, ParamSpec

_NOISE_SCALE = 255 * 0.15


def _clumped_noise(rng: np.random.Generator, shape: tuple[int, ...], size: float) -> np.ndarray:
    noise = rng.normal(0, 1, size=shape).astype(np.float32)
    if size > 0:
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=size)
        std = float(noise.std())
        if std > 1e-6:
            noise /= std
    return noise


@register_effect(
    name="film_grain",
    label="Film Grain",
    category="color",
    description="Adds photographic grain texture - either monochrome luminance grain like silver-halide film, or independent per-channel color noise like a noisy digital sensor - independent of any vintage camera profile.",
    about={
        "what": "Generates textured random noise and adds it on top of the image. Monochrome mode applies the same noise value to all three color channels at each point (grayscale grain, like light-sensitive silver-halide crystals in film). Color mode generates independent noise per channel (colored speckling, like a noisy digital sensor).",
        "how_to_use": "Raise Intensity for more visible grain, and Grain Size to make the noise clump into larger, coarser grains instead of fine speckle. Toggle Monochrome off for colored, digital-sensor-style noise instead of classic film grain. Adjust Seed for a different, still-reproducible texture.",
        "used_for": "Adding photographic texture to an otherwise clean digital image - useful on its own for a grainy, tactile look, or layered after a vintage camera profile for extra texture beyond what's baked into that profile.",
        "examples": "Monochrome grain recreates the look of light-sensitive silver-halide crystals in physical film stock - finer in slower, lower-ISO films and coarser in faster, higher-ISO ones. Color mode instead recreates the colored noise pattern of a small or under-exposed digital camera sensor. This is a standalone version of the grain and sensor-noise baked into this app's Vintage Camera Profile pipeline, for use without a full film/camera emulation.",
    },
    params=[
        ParamSpec(
            name="intensity", kind="float", default=0.3, min=0.0, max=1.0, step=0.01,
            description="Strength of the grain. 0 disables the effect entirely.",
        ),
        ParamSpec(
            name="size", kind="float", default=1.0, min=0.0, max=8.0, step=0.1, label="Grain Size",
            description="Blur radius applied to the noise before adding it, controlling how large and clumpy each grain looks. 0 gives the finest, most pixel-level speckle.",
        ),
        ParamSpec(
            name="monochrome", kind="bool", default=True, label="Monochrome",
            description="When on, the same noise value applies to all three channels (grayscale film-like grain). When off, each channel gets independent noise (colored, digital-sensor-like noise).",
        ),
        ParamSpec(
            name="seed", kind="int", default=0, min=0, max=999999, step=1,
            description="Random seed. The same seed with the same other settings always reproduces the exact same grain texture.",
        ),
    ],
)
def apply(
    image: ImageArray,
    intensity: float = 0.3,
    size: float = 1.0,
    monochrome: bool = True,
    seed: int = 0,
) -> ImageArray:
    if intensity <= 0:
        return image.copy()

    h, w = image.shape[:2]
    rng = np.random.default_rng(int(seed))
    img = image.astype(np.float32)

    if monochrome:
        noise = _clumped_noise(rng, (h, w), float(size))[..., None]
    else:
        noise = _clumped_noise(rng, (h, w, 3), float(size))

    out = img + noise * float(intensity) * _NOISE_SCALE
    return np.clip(out, 0, 255).astype(np.uint8)
