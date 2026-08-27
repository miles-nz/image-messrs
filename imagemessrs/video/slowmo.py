from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .frame_io import extract_frames, get_fps, write_frames


def _warp(frame: np.ndarray, flow: np.ndarray, t: float) -> np.ndarray:
    h, w = frame.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = grid_x + flow[..., 0] * t
    map_y = grid_y + flow[..., 1] * t
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _interpolate(prev: np.ndarray, frame: np.ndarray, t: float, method: str) -> np.ndarray:
    """Synthesizes the frame at fractional timestep `t` (0..1) between `prev`
    and `frame`. `duplicate` just holds `prev` - no new information, the
    naive "lower fps" baseline this technique exists to beat. `blend` is a
    plain cross-dissolve. `optical_flow` estimates motion both directions
    with Farneback dense flow, warps each frame toward `t` along that
    motion, then blends the two warped results - genuinely new in-between
    motion instead of a double-exposure."""
    if method == "duplicate":
        return prev
    if method == "blend":
        return cv2.addWeighted(prev, 1 - t, frame, t, 0)

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    next_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    flow_fwd = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_bwd = cv2.calcOpticalFlowFarneback(next_gray, prev_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    warped_from_prev = _warp(prev, flow_fwd, t)
    warped_from_next = _warp(frame, flow_bwd, 1 - t)
    return cv2.addWeighted(warped_from_prev, 1 - t, warped_from_next, t, 0)


def _combine_group(group: list[np.ndarray], method: str) -> np.ndarray:
    """Collapses a group of consecutive source frames into one output frame
    (the inverse of `_interpolate`'s job). `drop` just keeps the last frame
    - the naive "lower fps" baseline, motion in between is simply gone.
    `blend` averages every frame in the group. `optical_flow` warps every
    frame but the last onto the last frame's motion position before
    averaging, so fast motion collapses into a motion-blur-like combination
    instead of a flat multi-exposure ghost."""
    anchor = group[-1]
    if method == "drop" or len(group) == 1:
        return anchor
    if method == "blend":
        acc = np.zeros_like(anchor, dtype=np.float32)
        for f in group:
            acc += f.astype(np.float32)
        return np.clip(acc / len(group), 0, 255).astype(np.uint8)

    anchor_gray = cv2.cvtColor(anchor, cv2.COLOR_RGB2GRAY)
    acc = anchor.astype(np.float32)
    for f in group[:-1]:
        f_gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(f_gray, anchor_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        acc += _warp(f, flow, 1.0).astype(np.float32)
    return np.clip(acc / len(group), 0, 255).astype(np.uint8)


def _write_video_only(frames, fps: float, output_path: str) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        video_only = str(Path(tmp_dir) / "video_only.mp4")
        write_frames(frames, video_only, fps)
        Path(video_only).replace(output_path)


def slow_motion(
    input_path: str,
    output_path: str,
    speed_factor: float = 0.5,
    method: str = "optical_flow",
    on_progress: Callable[[dict], None] | None = None,
) -> None:
    """Slows a clip down by synthesizing new in-between frames rather than
    just retiming the existing ones, so motion stays smooth instead of
    juddering. `speed_factor` of 0.5 plays at half speed (2x slower);
    non-integer factors (e.g. 0.4) round to a fixed number of inserted
    frames per gap, so actual output speed is only approximate.

    Audio is dropped: this is the first technique whose output duration
    differs from its input, and `remux_audio`/`write_frames` assume video
    and audio stay the same length.
    """
    effect_speed = float(np.clip(speed_factor, 0.05, 0.95))
    fps = get_fps(input_path)
    inserted_per_gap = max(0, round(1 / effect_speed) - 1)

    def frames():
        prev = None
        for i, frame in enumerate(extract_frames(input_path)):
            if prev is not None:
                for k in range(1, inserted_per_gap + 1):
                    t = k / (inserted_per_gap + 1)
                    yield _interpolate(prev, frame, t, method)
            yield frame
            prev = frame
            if on_progress is not None:
                on_progress({"frame": str(i + 1)})

    _write_video_only(frames(), fps, output_path)


def speed_up(
    input_path: str,
    output_path: str,
    speed_factor: float = 2.0,
    method: str = "optical_flow",
    on_progress: Callable[[dict], None] | None = None,
) -> None:
    """Speeds a clip up by collapsing groups of consecutive source frames
    into single output frames, rather than just keeping every Nth frame and
    throwing the rest away. `speed_factor` of 2.0 plays twice as fast; it
    rounds to a whole number of source frames per output frame, so actual
    output speed is only approximate.

    Audio is dropped, same reasoning as `slow_motion`: this shortens the
    clip, and `remux_audio`/`write_frames` assume video and audio stay the
    same length.
    """
    group_size = max(2, round(float(speed_factor)))
    fps = get_fps(input_path)

    def frames():
        group: list[np.ndarray] = []
        for i, frame in enumerate(extract_frames(input_path)):
            group.append(frame)
            if len(group) == group_size:
                yield _combine_group(group, method)
                group = []
            if on_progress is not None:
                on_progress({"frame": str(i + 1)})
        if group:
            yield _combine_group(group, method)

    _write_video_only(frames(), fps, output_path)
