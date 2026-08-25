from __future__ import annotations

import numpy as np

from ..core.registry import register_effect
from ..core.types import ImageArray, ParamSpec


def _sort_key(pixels: np.ndarray, sort_by: str) -> np.ndarray:
    r = pixels[:, 0].astype(np.float32)
    g = pixels[:, 1].astype(np.float32)
    b = pixels[:, 2].astype(np.float32)
    if sort_by == "hue":
        maxc = np.max(pixels, axis=1).astype(np.float32)
        minc = np.min(pixels, axis=1).astype(np.float32)
        delta = maxc - minc
        hue = np.zeros_like(maxc)
        has_delta = delta != 0
        is_r = has_delta & (maxc == r)
        is_g = has_delta & (maxc == g) & ~is_r
        is_b = has_delta & (maxc == b) & ~is_r & ~is_g
        safe_delta = np.where(delta == 0, 1, delta)
        hue[is_r] = (60 * ((g[is_r] - b[is_r]) / safe_delta[is_r]) + 360) % 360
        hue[is_g] = (60 * ((b[is_g] - r[is_g]) / safe_delta[is_g]) + 120) % 360
        hue[is_b] = (60 * ((r[is_b] - g[is_b]) / safe_delta[is_b]) + 240) % 360
        return hue
    if sort_by == "saturation":
        maxc = np.max(pixels, axis=1).astype(np.float32)
        minc = np.min(pixels, axis=1).astype(np.float32)
        return np.where(maxc == 0, 0, (maxc - minc) / np.maximum(maxc, 1))
    return 0.299 * r + 0.587 * g + 0.114 * b  # brightness


def _sort_line(
    line: np.ndarray,
    sort_by: str,
    threshold_low: float,
    threshold_high: float,
    reverse: bool,
    protect_line: np.ndarray | None = None,
) -> np.ndarray:
    brightness = (
        0.299 * line[:, 0].astype(np.float32)
        + 0.587 * line[:, 1].astype(np.float32)
        + 0.114 * line[:, 2].astype(np.float32)
    )
    mask = (brightness >= threshold_low) & (brightness <= threshold_high)
    if protect_line is not None:
        mask = mask & ~protect_line

    out = line.copy()
    n = len(line)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        run = line[i:j]
        if len(run) > 1:
            keys = _sort_key(run, sort_by)
            order = np.argsort(keys, kind="stable")
            if reverse:
                order = order[::-1]
            out[i:j] = run[order]
        i = j
    return out


@register_effect(
    name="pixel_sort",
    label="Pixel Sort",
    category="glitch",
    accepts_mask=True,
    description="Sorts runs of pixels along each row or column by brightness, hue, or saturation. Pixels outside the threshold range act as anchors and stay put, creating streaks between them.",
    about={
        "what": "Sorts runs of pixels along each row or column by brightness, hue, or saturation, leaving pixels outside a threshold range untouched so they act as anchors that break the sorted runs into streaks.",
        "how_to_use": "Choose a Direction (rows or columns) and what to sort by, then narrow or widen Threshold Low/High to control how much of the image gets swept into streaks - a narrow band picks out just bright highlights or dark shadows to smear.",
        "used_for": "One of the most recognizable glitch-art effects - turns skies, water, or gradients into flowing streaks while leaving high-contrast edges intact.",
        "examples": "Pixel sorting was popularized around 2010 by artist and programmer Kim Asendorf, who published early pixel-sorting scripts; it went on to become one of the signature techniques of the glitch art community and shows up widely in music visuals and digital art.",
    },
    params=[
        ParamSpec(
            name="axis", kind="choice", default="columns", choices=["rows", "columns"],
            description="Whether pixels are sorted along each row (streaks run horizontally) or each column (streaks run vertically).",
        ),
        ParamSpec(
            name="sort_by", kind="choice", default="brightness", choices=["brightness", "hue", "saturation"],
            description="The pixel value used to order each sorted run: brightness (luminance), hue (color angle), or saturation (color intensity).",
        ),
        ParamSpec(
            name="threshold_low", kind="float", default=50.0, min=0, max=255, step=1,
            description="Pixels darker than this are left untouched and act as anchors that break up the sorted runs. Lower this to sort more of the image.",
        ),
        ParamSpec(
            name="threshold_high", kind="float", default=200.0, min=0, max=255, step=1,
            description="Pixels brighter than this are left untouched. Raise this to sort more of the image - only the brightness range between Threshold Low and Threshold High actually gets sorted.",
        ),
        ParamSpec(
            name="reverse", kind="bool", default=False,
            description="Sort each run high-to-low instead of low-to-high.",
        ),
        ParamSpec(
            name="protect_mask", kind="mask", default=None, label="Protected Region",
            description="Paint over an area (e.g. a face or subject) to keep its pixels from being swept into sorted streaks.",
        ),
    ],
)
def apply(
    image: ImageArray,
    axis: str = "columns",
    sort_by: str = "brightness",
    threshold_low: float = 50.0,
    threshold_high: float = 200.0,
    reverse: bool = False,
    protect_mask: np.ndarray | None = None,
) -> ImageArray:
    work = image if axis == "rows" else np.transpose(image, (1, 0, 2))
    mask_work = None
    if protect_mask is not None:
        mask_work = protect_mask.astype(bool) if axis == "rows" else protect_mask.astype(bool).T
    out = np.empty_like(work)
    for idx in range(work.shape[0]):
        protect_line = mask_work[idx] if mask_work is not None else None
        out[idx] = _sort_line(work[idx], sort_by, threshold_low, threshold_high, reverse, protect_line)
    return out if axis == "rows" else np.transpose(out, (1, 0, 2)).copy()
