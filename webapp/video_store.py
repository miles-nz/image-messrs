from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

BASE_STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage"
UPLOADS_DIR = BASE_STORAGE_DIR / "uploads"
OUTPUTS_DIR = BASE_STORAGE_DIR / "outputs"


@dataclass
class VideoSession:
    id: str
    original_path: Path
    motion_path: Path | None = None


class VideoStore:
    def __init__(self) -> None:
        self._sessions: dict[str, VideoSession] = {}
        self._lock = Lock()
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    def create(self, file_bytes: bytes, suffix: str = ".mp4") -> str:
        session_id = uuid.uuid4().hex
        path = UPLOADS_DIR / f"{session_id}{suffix}"
        path.write_bytes(file_bytes)
        with self._lock:
            self._sessions[session_id] = VideoSession(id=session_id, original_path=path)
        return session_id

    def get(self, session_id: str) -> VideoSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def set_motion_clip(self, session_id: str, file_bytes: bytes, suffix: str = ".mp4") -> None:
        path = UPLOADS_DIR / f"{session_id}_motion{suffix}"
        path.write_bytes(file_bytes)
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.motion_path = path

    def set_original(self, session_id: str, path: Path) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.original_path = path


VIDEO_STORE = VideoStore()
