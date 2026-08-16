from __future__ import annotations

from .core.registry import get_effect
from .core.types import ImageArray
from .effects.base import coerce_params


def run_pipeline(image: ImageArray, steps: list[tuple[str, dict]]) -> ImageArray:
    """Fold a list of (effect_name, raw_params) through the registry in order."""
    result = image
    for name, raw_params in steps:
        effect = get_effect(name)
        params = coerce_params(effect.params, raw_params)
        result = effect.fn(result, **params)
    return result
