"""
Effect function convention:

    def apply(image: np.ndarray, **params) -> np.ndarray

Blend effects (category="blend", multi_image=True) take two images:

    def apply(image_a: np.ndarray, image_b: np.ndarray, **params) -> np.ndarray

Effects must not mutate their input array in place, and must return uint8 RGB
arrays of the same dtype convention as the input (shape may differ, e.g. seam
carving).
"""

from __future__ import annotations

import re
from typing import Any

from ..core.types import ParamSpec

_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def coerce_params(params: list[ParamSpec], raw: dict[str, Any]) -> dict[str, Any]:
    """Coerce raw (e.g. form/query string) values to the types declared by params, filling defaults."""
    out: dict[str, Any] = {}
    for spec in params:
        value = raw.get(spec.name, None)
        is_empty = value is None or (isinstance(value, str) and value == "")
        if is_empty:
            out[spec.name] = spec.default
            continue
        if spec.kind == "float":
            out[spec.name] = float(value)
        elif spec.kind == "int":
            out[spec.name] = int(float(value))
        elif spec.kind == "bool":
            out[spec.name] = value in (True, "true", "True", "1", "on", 1)
        elif spec.kind == "choice":
            out[spec.name] = str(value)
        elif spec.kind == "color":
            candidate = str(value).strip()
            out[spec.name] = candidate if _HEX_COLOR_RE.match(candidate) else spec.default
        else:  # "mask" or anything else passed through as-is
            out[spec.name] = value
    return out
