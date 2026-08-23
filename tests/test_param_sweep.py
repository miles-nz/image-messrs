import numpy as np
import pytest

import imagemessrs.effects  # noqa: F401  (populates the effect registry)
from imagemessrs.video import param_sweep
from imagemessrs.video.frame_io import extract_frames, get_frame_count


def test_compute_sweep_values_ramp_hits_both_endpoints():
    values = param_sweep.compute_sweep_values(0, 10, 6, ping_pong=False)
    assert values[0] == 0
    assert values[-1] == 10
    assert values == sorted(values)


def test_compute_sweep_values_ping_pong_returns_to_start_without_duplicate():
    values = param_sweep.compute_sweep_values(0, 10, 8, ping_pong=True)
    assert values[0] == 0
    assert max(values) == pytest.approx(10, abs=1e-6)
    # Peak sits mid-sequence, and the sequence doesn't hold at the peak or
    # the start for two frames in a row (no stutter frame).
    assert values.count(max(values)) == 1
    assert values.count(min(values)) == 1


def test_apply_param_sweep_frame_count(gradient_image, tmp_path):
    output = str(tmp_path / "out.mp4")
    param_sweep.apply_param_sweep(
        gradient_image, output, "channel_shift", {}, "blue_dx", 0, 20, duration=1.0, fps=6
    )
    assert get_frame_count(output) == 6


def test_apply_param_sweep_changes_across_frames(gradient_image, tmp_path):
    output = str(tmp_path / "out.mp4")
    param_sweep.apply_param_sweep(
        gradient_image, output, "channel_shift", {}, "blue_dx", 0, 20, duration=1.0, fps=6
    )
    frames = list(extract_frames(output))
    assert not np.array_equal(frames[0], frames[-1])


def test_apply_param_sweep_unknown_param_raises(gradient_image, tmp_path):
    output = str(tmp_path / "out.mp4")
    with pytest.raises(ValueError):
        param_sweep.apply_param_sweep(
            gradient_image, output, "channel_shift", {}, "not_a_real_param", 0, 1, duration=1.0, fps=6
        )


def test_apply_param_sweep_non_numeric_param_raises(gradient_image, tmp_path):
    output = str(tmp_path / "out.mp4")
    with pytest.raises(ValueError):
        param_sweep.apply_param_sweep(
            gradient_image, output, "channel_shift", {}, "wrap", 0, 1, duration=1.0, fps=6
        )


def test_apply_param_sweep_multi_image_requires_image_b(gradient_image, tmp_path):
    output = str(tmp_path / "out.mp4")
    with pytest.raises(ValueError):
        param_sweep.apply_param_sweep(
            gradient_image, output, "difference_blend", {}, "blend_factor", 0, 1, duration=1.0, fps=6
        )


def test_apply_param_sweep_multi_image_runs_with_image_b(gradient_image, checkerboard_image, tmp_path):
    output = str(tmp_path / "out.mp4")
    param_sweep.apply_param_sweep(
        gradient_image,
        output,
        "difference_blend",
        {"mode": "difference"},
        "blend_factor",
        0,
        1,
        duration=1.0,
        fps=6,
        image_b=checkerboard_image,
    )
    frames = list(extract_frames(output))
    assert len(frames) == 6
    # blend_factor=0 -> pure image A, modulo lossy mp4v re-encoding (not bit-exact).
    mean_abs_diff = np.abs(frames[0].astype(np.int16) - gradient_image.astype(np.int16)).mean()
    assert mean_abs_diff < 5.0


def test_apply_param_sweep_int_param_rounds_values(gradient_image, tmp_path):
    output = str(tmp_path / "out.mp4")
    # red_dx is an int param; non-integer sweep steps must still run cleanly.
    param_sweep.apply_param_sweep(
        gradient_image, output, "channel_shift", {}, "red_dx", 0, 7, duration=1.0, fps=5
    )
    assert get_frame_count(output) == 5


def test_apply_param_sweep_seed_param_varies_noise(gradient_image, tmp_path):
    output = str(tmp_path / "out.mp4")
    param_sweep.apply_param_sweep(
        gradient_image,
        output,
        "byte_corrupt",
        {"mode": "raw_pixels", "intensity": 0.05},
        "intensity",
        0.05,
        0.05,
        duration=1.0,
        fps=4,
        seed_param="seed",
    )
    frames = list(extract_frames(output))
    # Same intensity every frame, but seed increments -> different noise each frame.
    assert not np.array_equal(frames[0], frames[1])


def test_apply_param_sweep_invalid_seed_param_raises(gradient_image, tmp_path):
    output = str(tmp_path / "out.mp4")
    with pytest.raises(ValueError):
        param_sweep.apply_param_sweep(
            gradient_image, output, "channel_shift", {}, "blue_dx", 0, 10, duration=1.0, fps=6, seed_param="wrap"
        )


def test_apply_param_sweep_reports_progress(gradient_image, tmp_path):
    output = str(tmp_path / "out.mp4")
    seen = []
    param_sweep.apply_param_sweep(
        gradient_image,
        output,
        "channel_shift",
        {},
        "blue_dx",
        0,
        10,
        duration=1.0,
        fps=5,
        on_progress=lambda p: seen.append(p["frame"]),
    )
    assert seen == ["1", "2", "3", "4", "5"]
