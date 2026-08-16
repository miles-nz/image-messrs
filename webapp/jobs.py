from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class Job:
    id: str
    status: str = "pending"  # pending, running, done, error
    progress: dict = field(default_factory=dict)
    result_path: Path | None = None
    error: str | None = None


class JobTracker:
    """In-process background-thread job runner. No queue/worker pool for
    v1 - fine for a single local user running one video job at a time."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, fn: Callable[[Callable[[dict], None]], Path]) -> str:
        job_id = uuid.uuid4().hex
        job = Job(id=job_id, status="running")
        with self._lock:
            self._jobs[job_id] = job

        def on_progress(data: dict) -> None:
            with self._lock:
                job.progress = data

        def runner() -> None:
            try:
                result_path = fn(on_progress)
                with self._lock:
                    job.result_path = result_path
                    job.status = "done"
            except Exception as exc:
                with self._lock:
                    job.status = "error"
                    job.error = f"{exc}\n{traceback.format_exc(limit=3)}"

        threading.Thread(target=runner, daemon=True).start()
        return job_id

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


JOB_TRACKER = JobTracker()
