from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

from ..core.registry import get_effect
from .frame_io import extract_frames, get_fps, get_frame_count, remux_audio, write_frames
from .param_sweep import compute_sweep_values


def apply_frame_effect(
    input_path: str,
    output_path: str,
    effect_name: str,
    params: dict[str, Any],
    on_progress: Callable[[dict], None] | None = None,
) -> None:
    """Runs a single-image effect from the image effect registry over every
    frame of a video independently - the same still-image transform applied
    frame by frame, with no motion or temporal awareness between frames.

    `params` must already be coerced to the effect's declared param types
    (see imagemessrs.effects.base.coerce_params) - this function does not
    coerce raw form values itself.
    """
    effect = get_effect(effect_name)
    if effect.multi_image:
        raise ValueError(f"{effect_name!r} needs a second image and can't run as a per-frame video effect")

    fps = get_fps(input_path)

    def processed_frames():
        for i, frame in enumerate(extract_frames(input_path)):
            if on_progress is not None:
                on_progress({"frame": str(i + 1)})
            yield effect.fn(frame, **params)

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_only = str(Path(tmp_dir) / "video_only.mp4")
        write_frames(processed_frames(), video_only, fps)
        remux_audio(input_path, video_only, output_path)


def apply_frame_effect_with_sweep(
    input_path: str,
    output_path: str,
    effect_name: str,
    base_params: dict[str, Any],
    sweep_param: str,
    start: float,
    end: float,
    ping_pong: bool = False,
    on_progress: Callable[[dict], None] | None = None,
) -> None:
    """Like `apply_frame_effect`, but sweeps one of the effect's own params
    linearly across the video's existing frames - `start` at the first frame,
    `end` at the last - instead of holding every param fixed.

    `base_params` must already be coerced to the effect's declared param
    types; `sweep_param`'s value in it is overridden per frame.
    """
    effect = get_effect(effect_name)
    if effect.multi_image:
        raise ValueError(f"{effect_name!r} needs a second image and can't run as a per-frame video effect")

    param_spec = next((p for p in effect.params if p.name == sweep_param), None)
    if param_spec is None:
        raise ValueError(f"{effect_name!r} has no param named {sweep_param!r}")
    if param_spec.kind not in ("int", "float"):
        raise ValueError(f"param {sweep_param!r} is not numeric and can't be swept")

    fps = get_fps(input_path)
    num_frames = get_frame_count(input_path)
    values = compute_sweep_values(float(start), float(end), num_frames, ping_pong)
    if param_spec.kind == "int":
        values = [round(v) for v in values]

    def processed_frames():
        for i, (frame, value) in enumerate(zip(extract_frames(input_path), values)):
            if on_progress is not None:
                on_progress({"frame": str(i + 1)})
            params = dict(base_params)
            params[sweep_param] = value
            yield effect.fn(frame, **params)

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_only = str(Path(tmp_dir) / "video_only.mp4")
        write_frames(processed_frames(), video_only, fps)
        remux_audio(input_path, video_only, output_path)
