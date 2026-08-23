from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ..core.registry import get_effect
from ..core.types import ImageArray
from .frame_io import write_frames


def compute_sweep_values(start: float, end: float, num_frames: int, ping_pong: bool) -> list[float]:
    """`num_frames` values sweeping from `start` to `end`.

    Ramp (ping_pong=False) is a plain start->end line, inclusive of both
    endpoints - meant to be watched once, not looped.

    Ping-pong instead samples a triangle wave over [0, 1) - value 0 at frame
    0, rising to 1 at the midpoint, falling back towards (but never
    re-touching) 0 - so the clip loops seamlessly with no held duplicate
    frame at either the turnaround or the wrap from last frame back to first.
    """
    num_frames = max(1, int(num_frames))
    if num_frames == 1:
        return [float(start)]
    if not ping_pong:
        return np.linspace(start, end, num_frames).tolist()

    t = np.arange(num_frames) / num_frames
    triangle = np.where(t < 0.5, t / 0.5, (1 - t) / 0.5)
    return (start + (end - start) * triangle).tolist()


def apply_param_sweep(
    image: ImageArray,
    output_path: str,
    effect_name: str,
    base_params: dict[str, Any],
    sweep_param: str,
    start: float,
    end: float,
    duration: float,
    fps: float,
    ping_pong: bool = False,
    seed_param: str | None = None,
    image_b: ImageArray | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> None:
    """Animates a still image by sweeping one of an effect's own params over
    time: calls the effect once per output frame with that one param varied,
    everything else held at `base_params`, and encodes the frames as a video.

    For a multi-image (blend) effect, `image_b` stays fixed across every
    frame while `sweep_param` (e.g. a blend's progress/blend_factor) moves.
    """
    effect = get_effect(effect_name)
    if effect.multi_image and image_b is None:
        raise ValueError(f"{effect_name!r} needs a second image and can't be animated without one")

    param_spec = next((p for p in effect.params if p.name == sweep_param), None)
    if param_spec is None:
        raise ValueError(f"{effect_name!r} has no param named {sweep_param!r}")
    if param_spec.kind not in ("int", "float"):
        raise ValueError(f"param {sweep_param!r} is not numeric and can't be swept")

    if seed_param is not None:
        seed_spec = next((p for p in effect.params if p.name == seed_param), None)
        if seed_spec is None or seed_spec.kind != "int":
            raise ValueError(f"{effect_name!r} has no int param named {seed_param!r} to vary")
        if seed_param == sweep_param:
            raise ValueError("seed_param can't be the same param as sweep_param")

    num_frames = max(2, round(float(duration) * float(fps)))
    values = compute_sweep_values(float(start), float(end), num_frames, ping_pong)
    if param_spec.kind == "int":
        values = [round(v) for v in values]

    base_seed = int(base_params.get(seed_param, 0)) if seed_param else 0

    def frames():
        for i, v in enumerate(values):
            params = dict(base_params)
            params[sweep_param] = v
            if seed_param is not None:
                params[seed_param] = base_seed + i
            if on_progress is not None:
                on_progress({"frame": str(i + 1)})
            if effect.multi_image:
                yield effect.fn(image, image_b, **params)
            else:
                yield effect.fn(image, **params)

    write_frames(frames(), output_path, float(fps))
