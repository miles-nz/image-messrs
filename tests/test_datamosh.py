import numpy as np
import pytest

from imagemessrs.video import datamosh
from imagemessrs.video.frame_io import extract_frames, get_frame_count

pytestmark = pytest.mark.slow  # shells out to ffmpeg; needs it on PATH


def test_frame_blend_preserves_frame_count(sample_video_path, tmp_path):
    output = str(tmp_path / "blended.mp4")
    datamosh.frame_blend(sample_video_path, output, trail_length=3, decay=0.6)
    assert get_frame_count(output) == get_frame_count(sample_video_path)


def test_frame_blend_changes_content(sample_video_path, tmp_path):
    output = str(tmp_path / "blended.mp4")
    datamosh.frame_blend(sample_video_path, output, trail_length=4, decay=0.7)
    original_frames = list(extract_frames(sample_video_path))
    blended_frames = list(extract_frames(output))
    assert len(blended_frames) == len(original_frames)
    assert not np.array_equal(original_frames[-1], blended_frames[-1])


def test_frame_blend_trail_length_one_is_near_identity(sample_video_path, tmp_path):
    output = str(tmp_path / "blended.mp4")
    datamosh.frame_blend(sample_video_path, output, trail_length=1, decay=0.6)
    original_frames = list(extract_frames(sample_video_path))
    blended_frames = list(extract_frames(output))
    # trail_length=1 means no history to blend with -> output ~= input frames,
    # modulo lossy mp4v re-encoding round-trip (not bit-exact)
    for orig, blended in zip(original_frames, blended_frames):
        mean_abs_diff = np.abs(orig.astype(np.int16) - blended.astype(np.int16)).mean()
        assert mean_abs_diff < 5.0


def test_feedback_loop_runs_and_produces_playable_output(sample_video_path, tmp_path):
    output = str(tmp_path / "feedback.mp4")
    datamosh.feedback_loop(sample_video_path, output, iterations=2, quality=30)
    assert get_frame_count(output) > 0


def test_iframe_smear_produces_playable_output_longer_than_base(sample_video_path, sample_video_path_2, tmp_path):
    output = str(tmp_path / "smeared.mp4")
    datamosh.iframe_smear(sample_video_path, sample_video_path_2, output, gop_size=300, qscale=4)
    base_frames = get_frame_count(sample_video_path)
    smeared_frames = get_frame_count(output)
    assert smeared_frames > base_frames  # base frames + motion clip's P-frames (I-frame stripped)


def test_iframe_smear_raises_on_motionless_clip_with_no_frames(sample_video_path, tmp_path):
    empty_path = str(tmp_path / "empty.m4v")
    open(empty_path, "wb").close()
    with pytest.raises(Exception):
        datamosh.iframe_smear(sample_video_path, empty_path, str(tmp_path / "out.mp4"))
