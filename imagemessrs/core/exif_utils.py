from __future__ import annotations

from io import BytesIO

from PIL import Image

_TAG_MAKE = 271
_TAG_MODEL = 272


def extract_camera_info(data: bytes) -> tuple[str | None, str | None]:
    """Best-effort Make/Model EXIF extraction. Never raises - returns
    (None, None) on any failure or absent tags."""
    try:
        img = Image.open(BytesIO(data))
        exif = img.getexif()
    except Exception:
        return None, None
    if not exif:
        return None, None
    make = exif.get(_TAG_MAKE)
    model = exif.get(_TAG_MODEL)
    make = make.strip() if isinstance(make, str) else None
    model = model.strip() if isinstance(model, str) else None
    return make or None, model or None
