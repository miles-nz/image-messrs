from __future__ import annotations

import numpy as np

from ..core.registry import register_effect
from ..core.types import ImageArray, ParamSpec


@register_effect(
    name="bit_plane_noise",
    label="Bit-Plane Noise",
    category="glitch",
    description="Corrupts a single bit of significance across every pixel/channel - low bits give near-invisible dithery grain, high bits blow color values apart into dramatic false-color blocks.",
    about={
        "what": "Every 8-bit color value is really 8 layered bit planes, from the least-significant bit (barely changes the value) up to the most-significant bit (flipping it can swap a value between near-black and near-white). This effect targets one bit plane at a time and flips, zeroes, or randomizes it across a chosen fraction of pixel/channel values.",
        "how_to_use": "Raise Bit Plane to pick which layer of significance gets hit - low values (0-2) give subtle grain, high values (6-7) give violent false-color blockiness. Intensity controls what fraction of pixel/channel values are affected. Mode chooses whether the bit gets flipped (toggled), zeroed (cleared), or randomized (set to a fresh coin-flip) at each affected position.",
        "used_for": "A controllable, bit-level alternative to random byte corruption - useful when you want either barely-visible sensor-like grain (low bit planes) or blocky, saturated color-corruption glitches (high bit planes) rather than fully random noise.",
        "examples": "Bit-plane slicing is a classic digital image processing and steganography technique for isolating how much visual information lives in each bit of significance; deliberately corrupting a single plane (rather than XOR-ing a random byte, as Byte Corruption's Raw Pixels mode does) produces a more uniform, tunable kind of damage that databending and glitch artists use for a more 'structured' corrupted look.",
    },
    params=[
        ParamSpec(
            name="bit_index", kind="int", default=6, min=0, max=7, step=1, label="Bit Plane (0=subtle, 7=extreme)",
            description="Which bit of significance to target, 0 (least significant, subtle) through 7 (most significant, can swap a value between near-black and near-white).",
        ),
        ParamSpec(
            name="mode", kind="choice", default="flip", choices=["flip", "zero", "randomize"],
            description="Flip toggles the targeted bit at each affected position. Zero clears it to 0. Randomize sets it to a fresh random 0/1, independent of the original value.",
        ),
        ParamSpec(
            name="intensity", kind="float", default=0.1, min=0.0, max=1.0, step=0.01,
            description="Fraction of pixel/channel values affected. 0 leaves the image untouched, 1 hits every value.",
        ),
        ParamSpec(
            name="seed", kind="int", default=0, min=0, max=999999, step=1,
            description="Random seed. The same seed with the same other settings always reproduces the exact same corruption pattern.",
        ),
    ],
)
def apply(
    image: ImageArray,
    bit_index: int = 6,
    mode: str = "flip",
    intensity: float = 0.1,
    seed: int = 0,
) -> ImageArray:
    if intensity <= 0:
        return image.copy()

    bit = int(np.clip(int(bit_index), 0, 7))
    bit_mask = np.uint8(1 << bit)
    inverse_mask = np.uint8(~bit_mask & 0xFF)

    rng = np.random.default_rng(int(seed))
    flat = image.reshape(-1).copy()
    hit = rng.random(flat.size) < float(intensity)

    if mode == "zero":
        flat[hit] &= inverse_mask
    elif mode == "randomize":
        random_bits = (rng.integers(0, 2, size=int(hit.sum())).astype(np.uint8)) * bit_mask
        flat[hit] = (flat[hit] & inverse_mask) | random_bits
    else:  # "flip"
        flat[hit] ^= bit_mask

    return flat.reshape(image.shape)
