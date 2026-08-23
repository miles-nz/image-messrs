from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable

from ..core.registry import get_effect
from .frame_io import extract_frames, get_fps, remux_audio, write_frames


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
