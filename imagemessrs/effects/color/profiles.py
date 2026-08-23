from __future__ import annotations

import json
import logging
from pathlib import Path

from .profile_types import (
    CameraProfile,
    ChromaticAberration,
    DynamicRange,
    Grain,
    Halation,
    Saturation,
    SensorNoise,
    SplitToning,
    ToneCurve,
    Vignette,
)

logger = logging.getLogger(__name__)
PROFILES_DIR = Path(__file__).parent / "profiles"


def _points(d: dict, key: str, default: list[list[float]]) -> list[tuple[float, float]]:
    raw = d.get(key, default)
    return [(float(p[0]), float(p[1])) for p in raw]


def _profile_from_dict(d: dict) -> CameraProfile:
    tc = d.get("tone_curve", {})
    sat = d.get("saturation", {})
    st = d.get("split_toning", {})
    vg = d.get("vignette", {})
    ca = d.get("chromatic_aberration", {})
    gr = d.get("grain", {})
    ha = d.get("halation", {})
    sn = d.get("sensor_noise", {})
    dr = d.get("dynamic_range", {})
    return CameraProfile(
        id=d["id"],
        label=d["label"],
        type=d["type"],
        description=d.get("description", ""),
        tone_curve=ToneCurve(
            r=_points(tc, "r", [[0, 0], [255, 255]]),
            g=_points(tc, "g", [[0, 0], [255, 255]]),
            b=_points(tc, "b", [[0, 0], [255, 255]]),
        ),
        saturation=Saturation(
            global_mult=sat.get("global_mult", 1.0),
            curve=_points(sat, "curve", []) or None,
        ),
        split_toning=SplitToning(
            shadow_color=tuple(st.get("shadow_color", (0, 0, 0))),
            shadow_strength=st.get("shadow_strength", 0.0),
            highlight_color=tuple(st.get("highlight_color", (255, 255, 255))),
            highlight_strength=st.get("highlight_strength", 0.0),
        ),
        vignette=Vignette(**{k: vg[k] for k in ("strength", "radius", "falloff") if k in vg}),
        chromatic_aberration=ChromaticAberration(intensity=ca.get("intensity", 0.0)),
        grain=Grain(intensity=gr.get("intensity", 0.0), size=gr.get("size", 1.0)),
        halation=Halation(**{k: ha[k] for k in ("intensity", "threshold", "radius") if k in ha}),
        sensor_noise=SensorNoise(intensity=sn.get("intensity", 0.0), chroma=sn.get("chroma", 0.0)),
        dynamic_range=DynamicRange(
            black_point=dr.get("black_point", 0.0),
            white_point=dr.get("white_point", 255.0),
        ),
        jpeg_quality=d.get("jpeg_quality"),
    )


def load_profiles(directory: Path = PROFILES_DIR) -> dict[str, CameraProfile]:
    """Load every *.json in `directory` into a dict keyed by profile id.

    A malformed file is logged and skipped rather than raising, since this
    runs at import time and one bad preset shouldn't take down the app.
    """
    profiles: dict[str, CameraProfile] = {}
    for path in sorted(Path(directory).glob("*.json")):
        try:
            data = json.loads(path.read_text())
            profile = _profile_from_dict(data)
        except Exception:
            logger.warning("Skipping invalid vintage profile %s", path, exc_info=True)
            continue
        if profile.id in profiles:
            logger.warning("Duplicate vintage profile id %r in %s, overwriting", profile.id, path)
        profiles[profile.id] = profile
    return profiles


PROFILES: dict[str, CameraProfile] = load_profiles()

# Sentinel "no target look" profile - every field left at its neutral default,
# so the pipeline reduces to whatever the source camera correction alone does.
# Deliberately not a JSON file in profiles/, since its dropdown position (always
# last) shouldn't be subject to the alphabetical sort applied to real profiles.
NONE_PROFILE = CameraProfile(
    id="none",
    label="None",
    type="digital",
    description="No target camera/film look - passes the image through with only the source camera correction applied.",
)


def get_profile(profile_id: str) -> CameraProfile:
    if profile_id == "none":
        return NONE_PROFILE
    if profile_id in PROFILES:
        return PROFILES[profile_id]
    # Fall back to the first available profile rather than raising, so a
    # stale dropdown value (e.g. a removed profile) degrades gracefully.
    return next(iter(PROFILES.values()))
