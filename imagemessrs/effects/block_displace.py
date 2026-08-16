from __future__ import annotations

import numpy as np

from ..core.registry import register_effect
from ..core.types import ImageArray, ParamSpec


@register_effect(
    name="block_displace",
    label="Block Displacement",
    category="glitch",
    description="Chops the image into a grid of blocks, then randomly swaps and jitters them - like corrupted JPEG macroblocks.",
    about={
        "what": "Chops the image into a grid of blocks, then randomly swaps blocks with each other and jitters their contents within their own cell - mimicking the look of corrupted JPEG macroblocks.",
        "how_to_use": "Start with Block Size to set the grain of the effect, then bring up Swap Probability to shuffle blocks around and Jitter Strength to nudge each block's contents off-register. Dial in a Seed you like and keep it fixed while you tweak the others, since changing the seed reshuffles the whole pattern.",
        "used_for": "A quick, controllable way to add a “digital corruption” or databending look to any image without touching actual file bytes - useful for album art, glitch-aesthetic social posts, and simulating compression damage on demand.",
        "examples": "This is a stylized take on the mosaic block artifacts that show up when video/JPEG compression breaks down - an aesthetic the glitch art movement has mined since the 2000s, with artists like Rosa Menkman writing extensively about compression artifacts as a visual language (“Vernacular of File Formats”).",
    },
    params=[
        ParamSpec(
            name="block_size", kind="int", default=16, min=4, max=128, step=4,
            description="Size of each grid block in pixels. Smaller blocks give finer, noisier glitching; larger blocks give chunkier, more obviously 'blocky' displacement.",
        ),
        ParamSpec(
            name="swap_probability", kind="float", default=0.3, min=0.0, max=1.0, step=0.05,
            description="Fraction of blocks that get randomly swapped with another block elsewhere in the grid. 0 = no swapping, 1 = as many blocks as possible get shuffled.",
        ),
        ParamSpec(
            name="jitter_strength", kind="float", default=0.3, min=0.0, max=1.0, step=0.05, label="Jitter Strength",
            description="How far (as a fraction of block size) each block's contents shift within its own cell. 0 = no jitter, 1 = a block's contents can shift by its full size.",
        ),
        ParamSpec(
            name="seed", kind="int", default=0, min=0, max=999999, step=1,
            description="Random seed. The same seed with the same other settings always reproduces the exact same glitch pattern, so you can dial one in and come back to it.",
        ),
    ],
)
def apply(
    image: ImageArray,
    block_size: int = 16,
    swap_probability: float = 0.3,
    jitter_strength: float = 0.3,
    seed: int = 0,
) -> ImageArray:
    block_size = max(1, int(block_size))
    rng = np.random.default_rng(int(seed))
    out = image.copy()
    h, w = image.shape[:2]
    rows, cols = h // block_size, w // block_size
    if rows == 0 or cols == 0:
        return out

    indices = [(r, c) for r in range(rows) for c in range(cols)]
    rng.shuffle(indices)
    n_swap = int(len(indices) * swap_probability)
    n_swap -= n_swap % 2
    for k in range(0, n_swap, 2):
        r1, c1 = indices[k]
        r2, c2 = indices[k + 1]
        y1, x1 = r1 * block_size, c1 * block_size
        y2, x2 = r2 * block_size, c2 * block_size
        b1 = out[y1 : y1 + block_size, x1 : x1 + block_size].copy()
        b2 = out[y2 : y2 + block_size, x2 : x2 + block_size].copy()
        out[y1 : y1 + block_size, x1 : x1 + block_size] = b2
        out[y2 : y2 + block_size, x2 : x2 + block_size] = b1

    max_offset = int(block_size * jitter_strength)
    if max_offset > 0:
        for r in range(rows):
            for c in range(cols):
                y, x = r * block_size, c * block_size
                dx = int(rng.integers(-max_offset, max_offset + 1))
                dy = int(rng.integers(-max_offset, max_offset + 1))
                if dx == 0 and dy == 0:
                    continue
                block = out[y : y + block_size, x : x + block_size]
                out[y : y + block_size, x : x + block_size] = np.roll(block, shift=(dy, dx), axis=(0, 1))

    return out
