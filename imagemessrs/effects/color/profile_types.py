from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Point = tuple[float, float]


@dataclass
class ToneCurve:
    r: list[Point] = field(default_factory=lambda: [(0, 0), (255, 255)])
    g: list[Point] = field(default_factory=lambda: [(0, 0), (255, 255)])
    b: list[Point] = field(default_factory=lambda: [(0, 0), (255, 255)])


@dataclass
class Saturation:
    global_mult: float = 1.0
    curve: list[Point] | None = None  # (luminance 0-255, multiplier), optional


@dataclass
class SplitToning:
    shadow_color: tuple[int, int, int] = (0, 0, 0)
    shadow_strength: float = 0.0
    highlight_color: tuple[int, int, int] = (255, 255, 255)
    highlight_strength: float = 0.0


@dataclass
class Vignette:
    strength: float = 0.0  # 0 = none, 1 = corners near black
    radius: float = 0.6  # normalized radius (0-1) where falloff starts
    falloff: float = 2.0  # exponent, higher = harder edge


@dataclass
class ChromaticAberration:
    intensity: float = 0.0  # radial displacement scale, 0 = none


@dataclass
class Grain:
    intensity: float = 0.0  # 0-1, additive luminance noise strength
    size: float = 1.0  # blur sigma controlling grain clump size


@dataclass
class Halation:
    intensity: float = 0.0  # 0 = none
    threshold: float = 0.85  # normalized (0-1) luminance threshold for bloom source
    radius: float = 6.0  # blur sigma


@dataclass
class SensorNoise:
    intensity: float = 0.0  # 0-1, luma noise
    chroma: float = 0.0  # 0-1, chroma noise (coarser, colored blotches)


@dataclass
class DynamicRange:
    black_point: float = 0.0  # 0-255
    white_point: float = 255.0  # 0-255


@dataclass
class CameraProfile:
    id: str
    label: str
    type: Literal["film", "digital"]
    description: str = ""
    tone_curve: ToneCurve = field(default_factory=ToneCurve)
    saturation: Saturation = field(default_factory=Saturation)
    split_toning: SplitToning = field(default_factory=SplitToning)
    vignette: Vignette = field(default_factory=Vignette)
    chromatic_aberration: ChromaticAberration = field(default_factory=ChromaticAberration)
    grain: Grain = field(default_factory=Grain)
    halation: Halation = field(default_factory=Halation)
    sensor_noise: SensorNoise = field(default_factory=SensorNoise)
    dynamic_range: DynamicRange = field(default_factory=DynamicRange)
    jpeg_quality: int | None = None  # None = skip JPEG round-trip


@dataclass
class SourceCorrection:
    undo_sharpen: float = 0.0  # 0-1, blends toward a blurred copy to counter oversharpening halos
    undo_hdr: float = 0.0  # 0-1, pulls contrast in around mid-gray to flatten aggressive tone-mapping
    undo_noise_reduction: float = 0.0  # 0-1, adds back texture removed by multi-frame noise reduction
    undo_saturation: float = 0.0  # 0-1, reduces punchy default saturation


@dataclass
class SourceCameraProfile:
    id: str
    label: str
    description: str = ""
    exif_aliases: list[str] = field(default_factory=list)
    correction: SourceCorrection = field(default_factory=SourceCorrection)
