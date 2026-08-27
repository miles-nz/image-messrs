import numpy as np
import pytest

from imagemessrs.video import slowmo
from imagemessrs.video.frame_io import extract_frames, get_frame_count

pytestmark = pytest.mark.slow  # shells out to ffmpeg; needs it on PATH


@pytest.mark.parametrize("method", ["blend", "optical_flow", "duplicate"])
def test_slow_motion_produces_more_frames_than_input(sample_video_path, tmp_path, method):
    output = str(tmp_path / "slow.mp4")
    slowmo.slow_motion(sample_video_path, output, speed_factor=0.5, method=method)
    assert get_frame_count(output) > get_frame_count(sample_video_path)


def test_slow_motion_frame_count_matches_speed_factor(sample_video_path, tmp_path):
    output = str(tmp_path / "slow.mp4")
    slowmo.slow_motion(sample_video_path, output, speed_factor=0.5, method="blend")
    input_frames = get_frame_count(sample_video_path)
    # speed_factor=0.5 -> 1 inserted frame per gap -> roughly double the frames
    assert get_frame_count(output) == input_frames * 2 - 1


def test_slow_motion_near_full_speed_is_near_passthrough(sample_video_path, tmp_path):
    output = str(tmp_path / "slow.mp4")
    slowmo.slow_motion(sample_video_path, output, speed_factor=0.9, method="blend")
    assert get_frame_count(output) == get_frame_count(sample_video_path)


def test_slow_motion_reports_progress_per_input_frame(sample_video_path, tmp_path):
    output = str(tmp_path / "slow.mp4")
    seen_frames = []
    slowmo.slow_motion(
        sample_video_path, output, speed_factor=0.5, method="blend", on_progress=lambda p: seen_frames.append(p["frame"])
    )
    expected_count = get_frame_count(sample_video_path)
    assert seen_frames == [str(i) for i in range(1, expected_count + 1)]


def test_slow_motion_duplicate_holds_the_previous_frame(sample_video_path, tmp_path):
    output = str(tmp_path / "slow.mp4")
    slowmo.slow_motion(sample_video_path, output, speed_factor=0.5, method="duplicate")
    original_frames = list(extract_frames(sample_video_path))
    result_frames = list(extract_frames(output))
    # result[1] is the synthesized frame between original[0] and original[1] -
    # duplicate mode should hold original[0] exactly (modulo mp4v re-encode rounding)
    mean_abs_diff = np.abs(result_frames[1].astype(np.int16) - original_frames[0].astype(np.int16)).mean()
    assert mean_abs_diff < 5.0


def test_slow_motion_interpolated_frame_lies_between_originals(sample_video_path, tmp_path):
    output = str(tmp_path / "slow.mp4")
    slowmo.slow_motion(sample_video_path, output, speed_factor=0.5, method="blend")
    original_frames = list(extract_frames(sample_video_path))
    result_frames = list(extract_frames(output))
    # result[1] is the synthesized frame between original[0] and original[1]
    synthesized = result_frames[1].astype(np.int16)
    before = original_frames[0].astype(np.int16)
    after = original_frames[1].astype(np.int16)
    assert not np.array_equal(synthesized, before)
    assert not np.array_equal(synthesized, after)


@pytest.mark.parametrize("method", ["blend", "optical_flow", "drop"])
def test_speed_up_produces_fewer_frames_than_input(sample_video_path, tmp_path, method):
    output = str(tmp_path / "fast.mp4")
    slowmo.speed_up(sample_video_path, output, speed_factor=2.0, method=method)
    assert get_frame_count(output) < get_frame_count(sample_video_path)


def test_speed_up_frame_count_matches_speed_factor(sample_video_path, tmp_path):
    output = str(tmp_path / "fast.mp4")
    slowmo.speed_up(sample_video_path, output, speed_factor=2.0, method="blend")
    input_frames = get_frame_count(sample_video_path)
    # speed_factor=2.0 -> groups of 2 source frames per output frame
    assert get_frame_count(output) == -(-input_frames // 2)  # ceil division


def test_speed_up_reports_progress_per_input_frame(sample_video_path, tmp_path):
    output = str(tmp_path / "fast.mp4")
    seen_frames = []
    slowmo.speed_up(
        sample_video_path, output, speed_factor=2.0, method="blend", on_progress=lambda p: seen_frames.append(p["frame"])
    )
    expected_count = get_frame_count(sample_video_path)
    assert seen_frames == [str(i) for i in range(1, expected_count + 1)]


def test_speed_up_drop_keeps_the_last_frame_of_each_group(sample_video_path, tmp_path):
    output = str(tmp_path / "fast.mp4")
    slowmo.speed_up(sample_video_path, output, speed_factor=2.0, method="drop")
    original_frames = list(extract_frames(sample_video_path))
    result_frames = list(extract_frames(output))
    # first output frame is the last frame of the first group of 2 -> original index 1
    mean_abs_diff = np.abs(result_frames[0].astype(np.int16) - original_frames[1].astype(np.int16)).mean()
    assert mean_abs_diff < 5.0


def test_speed_up_blend_combines_the_group_rather_than_dropping(sample_video_path, tmp_path):
    dropped = str(tmp_path / "dropped.mp4")
    blended = str(tmp_path / "blended.mp4")
    slowmo.speed_up(sample_video_path, dropped, speed_factor=2.0, method="drop")
    slowmo.speed_up(sample_video_path, blended, speed_factor=2.0, method="blend")
    dropped_frames = list(extract_frames(dropped))
    blended_frames = list(extract_frames(blended))
    assert not np.array_equal(dropped_frames[0], blended_frames[0])
