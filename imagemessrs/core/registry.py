from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

from .types import ImageArray, ParamSpec

EffectCategory = Literal["seam_carve", "glitch", "blend", "video", "color", "distort"]


@dataclass
class Effect:
    name: str
    label: str
    category: EffectCategory
    fn: Callable[..., ImageArray]
    params: list[ParamSpec] = field(default_factory=list)
    multi_image: bool = False  # True for blend effects: fn(image_a, image_b, **params)
    accepts_mask: bool = False
    description: str = ""
    about: dict = field(default_factory=dict)


EFFECT_REGISTRY: dict[str, Effect] = {}


def register_effect(
    *,
    name: str,
    label: str,
    category: EffectCategory,
    params: list[ParamSpec] | None = None,
    multi_image: bool = False,
    accepts_mask: bool = False,
    description: str = "",
    about: dict | None = None,
) -> Callable[[Callable[..., ImageArray]], Callable[..., ImageArray]]:
    def decorator(fn: Callable[..., ImageArray]) -> Callable[..., ImageArray]:
        EFFECT_REGISTRY[name] = Effect(
            name=name,
            label=label,
            category=category,
            fn=fn,
            params=params or [],
            multi_image=multi_image,
            accepts_mask=accepts_mask,
            description=description,
            about=about or {},
        )
        return fn

    return decorator


def get_effect(name: str) -> Effect:
    try:
        return EFFECT_REGISTRY[name]
    except KeyError:
        raise KeyError(f"No effect registered under name {name!r}. Known: {sorted(EFFECT_REGISTRY)}") from None


def list_effects(category: EffectCategory | None = None) -> list[Effect]:
    effects = EFFECT_REGISTRY.values()
    if category:
        effects = [e for e in effects if e.category == category]
    return sorted(effects, key=lambda e: (e.multi_image, e.name))
