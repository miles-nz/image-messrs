from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from .video_store import OUTPUTS_DIR, UPLOADS_DIR

MIN_TTL_HOURS = 1  # floor, so a misconfigured/blank STORAGE_TTL_HOURS can't wipe everything on the next startup
MAX_AGE_SECONDS = max(MIN_TTL_HOURS, int(os.environ.get("STORAGE_TTL_HOURS", "24"))) * 3600
SWEEP_INTERVAL_SECONDS = 3600


def sweep_directory(directory: Path, max_age_seconds: float, now: float | None = None) -> None:
    """Delete files in `directory` older than `max_age_seconds`. Skips .gitkeep."""
    if not directory.exists():
        return
    cutoff = (now if now is not None else time.time()) - max_age_seconds
    for path in directory.iterdir():
        if path.name == ".gitkeep" or not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except FileNotFoundError:
            pass


def _sweep_loop() -> None:
    while True:
        time.sleep(SWEEP_INTERVAL_SECONDS)
        sweep_directory(UPLOADS_DIR, MAX_AGE_SECONDS)
        sweep_directory(OUTPUTS_DIR, MAX_AGE_SECONDS)


def start_cleanup_thread() -> None:
    """Sweep stale uploads/outputs now, then repeat on an hourly timer for
    the life of the process - sessions are in-memory only (see store.py /
    video_store.py), so leftover files past MAX_AGE_SECONDS are orphaned."""
    sweep_directory(UPLOADS_DIR, MAX_AGE_SECONDS)
    sweep_directory(OUTPUTS_DIR, MAX_AGE_SECONDS)
    threading.Thread(target=_sweep_loop, daemon=True).start()
