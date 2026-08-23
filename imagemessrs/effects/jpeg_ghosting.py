from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from ..core.registry import register_effect
from ..core.types import ImageArray, ParamSpec


def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
    buf = BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=max(1, min(95, int(quality))))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


def _boost(image: np.ndarray) -> np.ndarray:
    """Nudges contrast and saturation up slightly, the way each re-share on a
    lossy platform tends to sharpen/punch up an image a little before
    re-compressing it - the "deep fried meme" feedback loop."""
    contrasted = np.clip((image.astype(np.float32) - 128.0) * 1.08 + 128.0, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(contrasted, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.15, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


@register_effect(
    name="jpeg_ghosting",
    label="JPEG Ghosting",
    category="glitch",
    description="Re-encodes the image as a low-quality JPEG over and over, each generation compounding the last - the accumulated blocking and color bleed of an image re-shared and re-saved many times.",
    about={
        "what": "Repeatedly re-encodes the image as JPEG at a low quality setting, each pass building on the compression artifacts of the last generation instead of starting fresh from the original - the way a photo actually degrades after being downloaded, re-saved, and re-uploaded across several platforms.",
        "how_to_use": "Raise Generations to compound more passes of damage, and lower JPEG Quality to make each pass more destructive. Quality Decay makes later generations progressively worse than earlier ones, mimicking a chain of re-shares each a little more compressed than the last. Turn on Deep Fry to also punch up contrast and saturation each generation, for the exaggerated over-processed meme look.",
        "used_for": "Simulating the 'generation loss' look of an image passed through many rounds of screenshotting, re-saving, and re-uploading - a recognizable aesthetic in internet meme culture and glitch art alike.",
        "examples": "Distinct from Byte Corruption's single-pass byte mangling - this compounds ordinary JPEG re-compression the way real images degrade when repeatedly re-shared, the same phenomenon behind the 'deep fried meme' genre that intentionally exaggerates this decay for comic effect.",
    },
    params=[
        ParamSpec(
            name="generations", kind="int", default=5, min=1, max=50, step=1,
            description="How many times the image is re-encoded and re-decoded as JPEG, each pass compounding the previous generation's artifacts.",
        ),
        ParamSpec(
            name="quality", kind="int", default=15, min=1, max=95, step=1, label="JPEG Quality",
            description="JPEG quality used for each re-encode pass. Lower values introduce more blocking and color bleed per generation.",
        ),
        ParamSpec(
            name="quality_decay", kind="float", default=0.0, min=0.0, max=0.5, step=0.01, label="Quality Decay",
            description="Fraction by which quality drops after each generation, so later passes are more destructive than earlier ones. 0 keeps every generation at the same quality.",
        ),
        ParamSpec(
            name="deep_fry", kind="bool", default=False, label="Deep Fry",
            description="Also boosts contrast and saturation slightly before each re-encode pass, for the exaggerated over-processed 'deep fried meme' look.",
        ),
    ],
)
def apply(
    image: ImageArray,
    generations: int = 5,
    quality: int = 15,
    quality_decay: float = 0.0,
    deep_fry: bool = False,
) -> ImageArray:
    out = image
    q = float(quality)
    for _ in range(max(0, int(generations))):
        if deep_fry:
            out = _boost(out)
        out = _jpeg_roundtrip(out, int(round(q)))
        q = max(3.0, q * (1 - float(quality_decay)))
    return out
