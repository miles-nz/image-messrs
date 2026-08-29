from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np

ImageArray = np.ndarray  # uint8, HxWxC, RGB

ParamKind = Literal["float", "int", "bool", "choice", "color", "mask"]


@dataclass
class ParamSpec:
    name: str
    kind: ParamKind
    default: Any
    label: str = ""
    description: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.name.replace("_", " ").title()


@dataclass
class EffectResult:
    image: ImageArray
    meta: dict = field(default_factory=dict)
