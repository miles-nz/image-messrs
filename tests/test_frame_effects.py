import numpy as np
import pytest

import imagemessrs.effects  # noqa: F401  (populates the effect registry)
from imagemessrs.video import frame_effects
from imagemessrs.video.frame_io import extract_frames, get_frame_count

pytestmark = pytest.mark.slow  # shells out to ffmpeg; needs it on PATH


def test_apply_frame_effect_preserves_frame_count(sample_video_path, tmp_path):
    output = str(tmp_path / "out.mp4")
    frame_effects.apply_frame_effect(sample_video_path, output, "channel_shift", {"blue_dx": 5})
    assert get_frame_count(output) == get_frame_count(sample_video_path)


def test_apply_frame_effect_changes_content(sample_video_path, tmp_path):
    output = str(tmp_path / "out.mp4")
    frame_effects.apply_frame_effect(sample_video_path, output, "channel_shift", {"blue_dx": 10, "red_dx": -10})
    original_frames = list(extract_frames(sample_video_path))
    result_frames = list(extract_frames(output))
    assert len(result_frames) == len(original_frames)
    assert not np.array_equal(original_frames[0], result_frames[0])


def test_apply_frame_effect_rejects_multi_image_effect(sample_video_path, tmp_path):
    output = str(tmp_path / "out.mp4")
    with pytest.raises(ValueError):
        frame_effects.apply_frame_effect(sample_video_path, output, "poisson_blend", {})


def test_apply_frame_effect_reports_progress(sample_video_path, tmp_path):
    output = str(tmp_path / "out.mp4")
    seen_frames = []
    frame_effects.apply_frame_effect(
        sample_video_path, output, "channel_shift", {"blue_dx": 2}, on_progress=lambda p: seen_frames.append(p["frame"])
    )
    expected_count = get_frame_count(sample_video_path)
    assert seen_frames == [str(i) for i in range(1, expected_count + 1)]
