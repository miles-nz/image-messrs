from __future__ import annotations

import uuid
from dataclasses import dataclass
from threading import Lock

import numpy as np


@dataclass
class ImageSession:
    id: str
    original: np.ndarray
    original_b: np.ndarray | None = None


class ImageStore:
    """In-process, in-memory session store. No DB for v1 - sessions live only
    as long as the dev server process does."""

    def __init__(self) -> None:
        self._sessions: dict[str, ImageSession] = {}
        self._lock = Lock()

    def create(self, original: np.ndarray) -> str:
        session_id = uuid.uuid4().hex
        with self._lock:
            self._sessions[session_id] = ImageSession(id=session_id, original=original)
        return session_id

    def get(self, session_id: str) -> ImageSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def set_image_a(self, session_id: str, image_a: np.ndarray) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.original = image_a

    def set_image_b(self, session_id: str, image_b: np.ndarray) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.original_b = image_b


IMAGE_STORE = ImageStore()
