from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Callable

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"


class FFmpegError(RuntimeError):
    pass


def probe(path: str | Path) -> dict:
    cmd = [FFPROBE, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(result.stderr)
    return json.loads(result.stdout)


def run(args: list[str], on_progress: Callable[[dict], None] | None = None) -> None:
    """Run ffmpeg with the given args (input/output flags included). If
    on_progress is given, ffmpeg's -progress key=value stream is parsed and
    the accumulated progress dict is passed to the callback after each tick."""
    base_cmd = [FFMPEG, "-y", *args]

    if on_progress is None:
        result = subprocess.run(base_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise FFmpegError(result.stderr)
        return

    cmd = base_cmd + ["-progress", "pipe:1", "-nostats"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    progress_data: dict[str, str] = {}
    assert process.stdout is not None
    for line in process.stdout:
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        progress_data[key] = value
        if key == "progress":
            on_progress(dict(progress_data))

    stderr = process.stderr.read() if process.stderr else ""
    process.wait()
    if process.returncode != 0:
        raise FFmpegError(stderr)
