from __future__ import annotations

import json
import logging
from pathlib import Path

from .profile_types import SourceCameraProfile, SourceCorrection

logger = logging.getLogger(__name__)
SOURCE_CAMERAS_DIR = Path(__file__).parent / "source_cameras"

_DEFAULT_ID = "unknown"


def _source_camera_from_dict(d: dict) -> SourceCameraProfile:
    corr = d.get("correction", {})
    return SourceCameraProfile(
        id=d["id"],
        label=d["label"],
        description=d.get("description", ""),
        exif_aliases=[a.lower() for a in d.get("exif_aliases", [])],
        correction=SourceCorrection(
            undo_sharpen=corr.get("undo_sharpen", 0.0),
            undo_hdr=corr.get("undo_hdr", 0.0),
            undo_noise_reduction=corr.get("undo_noise_reduction", 0.0),
            undo_saturation=corr.get("undo_saturation", 0.0),
        ),
    )


def load_source_cameras(directory: Path = SOURCE_CAMERAS_DIR) -> dict[str, SourceCameraProfile]:
    """Load every *.json in `directory` into a dict keyed by camera id.

    A malformed file is logged and skipped rather than raising, since this
    runs at import time and one bad preset shouldn't take down the app.
    """
    cameras: dict[str, SourceCameraProfile] = {}
    for path in sorted(Path(directory).glob("*.json")):
        try:
            data = json.loads(path.read_text())
            camera = _source_camera_from_dict(data)
        except Exception:
            logger.warning("Skipping invalid source camera profile %s", path, exc_info=True)
            continue
        if camera.id in cameras:
            logger.warning("Duplicate source camera id %r in %s, overwriting", camera.id, path)
        cameras[camera.id] = camera
    return cameras


SOURCE_CAMERAS: dict[str, SourceCameraProfile] = load_source_cameras()


def get_source_camera(camera_id: str) -> SourceCameraProfile:
    if camera_id in SOURCE_CAMERAS:
        return SOURCE_CAMERAS[camera_id]
    if _DEFAULT_ID in SOURCE_CAMERAS:
        return SOURCE_CAMERAS[_DEFAULT_ID]
    return next(iter(SOURCE_CAMERAS.values()))


def match_source_camera_from_exif(make: str | None, model: str | None) -> str | None:
    make_l = (make or "").lower()
    model_l = (model or "").lower()
    if not make_l and not model_l:
        return None

    best_id: str | None = None
    best_len = 0
    best_in_model = False
    for camera in SOURCE_CAMERAS.values():
        for alias in camera.exif_aliases:
            if not alias:
                continue
            in_model = alias in model_l
            if not in_model and alias not in make_l:
                continue
            # A match in the specific Model field beats one found only in the
            # generic Make field, e.g. a real Instax camera reports Make
            # "Fujifilm" (also a generic mirrorless alias) and Model "INSTAX
            # ..." - the model-level match should win regardless of alias
            # length. Ties within the same tier fall back to longest alias.
            better_tier = in_model and not best_in_model
            same_tier_longer = in_model == best_in_model and len(alias) > best_len
            if best_id is None or better_tier or same_tier_longer:
                best_id, best_len, best_in_model = camera.id, len(alias), in_model
    return best_id
