from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError

from ..core.registry import register_effect
from ..core.types import ImageArray, ParamSpec

_SOS_MARKER = b"\xff\xda"
_EOI_MARKER = b"\xff\xd9"


def _corrupt_raw_pixels(image: ImageArray, intensity: float, seed: int) -> ImageArray:
    rng = np.random.default_rng(seed)
    flat = image.reshape(-1).copy()
    n_corrupt = int(len(flat) * intensity)
    if n_corrupt == 0:
        return image
    positions = rng.choice(len(flat), size=n_corrupt, replace=False)
    bit_flips = rng.integers(1, 256, size=n_corrupt, dtype=np.uint8)
    flat[positions] ^= bit_flips
    return flat.reshape(image.shape)


def _find_scan_data_bounds(jpeg_bytes: bytes) -> tuple[int, int]:
    sos_index = jpeg_bytes.find(_SOS_MARKER)
    if sos_index == -1:
        return 0, 0
    length = int.from_bytes(jpeg_bytes[sos_index + 2 : sos_index + 4], "big")
    start = sos_index + 2 + length
    eoi_index = jpeg_bytes.rfind(_EOI_MARKER)
    end = eoi_index if eoi_index != -1 else len(jpeg_bytes)
    return start, end


def _corrupt_jpeg_bytes(image: ImageArray, intensity: float, seed: int, quality: int = 90) -> ImageArray:
    rng = np.random.default_rng(seed)
    buf = BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=quality)
    original = buf.getvalue()

    start, end = _find_scan_data_bounds(original)
    eligible = list(range(start, end)) if end > start else []
    n_corrupt = int(len(eligible) * intensity)
    if not eligible or n_corrupt == 0:
        return image

    scale = 1.0
    for _ in range(4):
        data = bytearray(original)
        count = max(1, int(n_corrupt * scale))
        positions = rng.choice(eligible, size=min(count, len(eligible)), replace=False)
        for pos in positions:
            data[pos] = int(rng.integers(0, 256))
        try:
            decoded = Image.open(BytesIO(bytes(data)))
            decoded.load()
            return np.asarray(decoded.convert("RGB"), dtype=np.uint8)
        except (UnidentifiedImageError, OSError):
            scale /= 2
    return image


@register_effect(
    name="byte_corrupt",
    label="Byte Corruption",
    category="glitch",
    description="Directly mangles image bytes for authentic-looking corruption artifacts, in either the compressed JPEG stream or the raw pixel data.",
    about={
        "what": "Directly mangles image bytes to simulate real data corruption - either flipping bits in the raw pixel data (fine, static-like noise) or corrupting the compressed JPEG bitstream itself (blocky smears and color bleeding as the decoder misinterprets damaged data).",
        "how_to_use": "Pick a mode first - JPEG Bytes for the classic “corrupted download” look, Raw Pixels for finer static-like noise - then raise Intensity gradually; in JPEG mode small increases can have an outsized effect since one flipped byte can smear a whole block. Lock in a Seed once you find damage you like.",
        "used_for": "Simulating file corruption, transmission errors, or degraded media - a staple move for glitch art, “corrupted file” aesthetics, and lo-fi/vaporwave visuals.",
        "examples": "Directly editing compressed file bytes (databending) is a technique glitch artists have used since the mid-2000s; it's closely related to the “corrupted JPEG” images that circulated widely online and became their own recognizable genre of glitch art.",
    },
    params=[
        ParamSpec(
            name="mode", kind="choice", default="jpeg_bytes", choices=["jpeg_bytes", "raw_pixels"],
            description="JPEG Bytes corrupts the compressed JPEG data stream - glitchy blocks and color bleeding, the classic 'corrupted JPEG' look; falls back to the original image if corruption makes the file undecodable. Raw Pixels corrupts pixel values directly - finer noise/static, always decodable.",
        ),
        ParamSpec(
            name="intensity", kind="float", default=0.02, min=0.0, max=0.3, step=0.005,
            description="Fraction of eligible bytes/pixels that get corrupted. Higher values mean more damage - in JPEG Bytes mode even small increases can have an outsized effect, since one flipped byte can smear a whole block.",
        ),
        ParamSpec(
            name="seed", kind="int", default=0, min=0, max=999999, step=1,
            description="Random seed. The same seed with the same other settings always reproduces the exact same corruption pattern.",
        ),
    ],
)
def apply(image: ImageArray, mode: str = "jpeg_bytes", intensity: float = 0.02, seed: int = 0) -> ImageArray:
    if mode == "raw_pixels":
        return _corrupt_raw_pixels(image, float(intensity), int(seed))
    return _corrupt_jpeg_bytes(image, float(intensity), int(seed))
