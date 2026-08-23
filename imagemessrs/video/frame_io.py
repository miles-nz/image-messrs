from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np

from . import ffmpeg_wrapper as ffmpeg


def has_audio(path: str | Path) -> bool:
    try:
        info = ffmpeg.probe(path)
    except ffmpeg.FFmpegError:
        return False
    return any(s.get("codec_type") == "audio" for s in info.get("streams", []))


def remux_audio(original: str | Path, video_only: str | Path, output: str | Path) -> None:
    """Copies the audio track from `original` onto the (silent) `video_only`
    render. If `original` has no audio, just moves `video_only` into place."""
    if not has_audio(original):
        Path(video_only).replace(output)
        return
    try:
        ffmpeg.run(
            [
                "-i", str(video_only),
                "-i", str(original),
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                str(output),
            ]
        )
    finally:
        if Path(video_only).exists():
            Path(video_only).unlink()


def extract_frames(path: str | Path) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def get_fps(path: str | Path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return float(fps)


def get_frame_count(path: str | Path) -> int:
    cap = cv2.VideoCapture(str(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def write_frames(frames: Iterable[np.ndarray], path: str | Path, fps: float) -> None:
    frames_iter = iter(frames)
    first = next(frames_iter, None)
    if first is None:
        raise ValueError("no frames to write")

    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    try:
        writer.write(cv2.cvtColor(first, cv2.COLOR_RGB2BGR))
        for frame in frames_iter:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
