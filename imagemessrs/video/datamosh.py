from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from . import ffmpeg_wrapper as ffmpeg
from .frame_io import extract_frames, get_fps, remux_audio, write_frames


def frame_blend(input_path: str, output_path: str, trail_length: int = 3, decay: float = 0.6) -> None:
    """Motion-trail effect: each output frame is a weighted blend of the
    current frame and the previous `trail_length` frames, decaying by `decay`
    per step back in time."""
    fps = get_fps(input_path)
    trail_length = max(1, int(trail_length))
    decay = float(np.clip(decay, 0.0, 0.95))

    history: list[np.ndarray] = []

    def blended_frames():
        for frame in extract_frames(input_path):
            history.append(frame.astype(np.float32))
            if len(history) > trail_length:
                history.pop(0)
            n = len(history)
            weights = np.array([decay ** (n - 1 - i) for i in range(n)], dtype=np.float32)
            weights /= weights.sum()
            acc = np.zeros_like(history[0])
            for w, past_frame in zip(weights, history):
                acc += w * past_frame
            yield np.clip(acc, 0, 255).astype(np.uint8)

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_only = str(Path(tmp_dir) / "video_only.mp4")
        write_frames(blended_frames(), video_only, fps)
        remux_audio(input_path, video_only, output_path)


_VOP_START_CODE = b"\x00\x00\x01\xb6"


def _find_vops(data: bytes) -> list[tuple[int, int]]:
    """Locate MPEG-4 Part 2 VOP start codes and their coding type
    (0=I, 1=P, 2=B, 3=S), read from the 2 high bits of the byte after the
    4-byte start code."""
    positions = []
    i = 0
    while True:
        idx = data.find(_VOP_START_CODE, i)
        if idx == -1:
            break
        coding_type = (data[idx + 4] >> 6) & 0b11
        positions.append((idx, coding_type))
        i = idx + 4
    return positions


def _transcode_to_raw_mpeg4(input_path: str, output_path: str, gop_size: int, qscale: int) -> None:
    ffmpeg.run(
        [
            "-i", input_path,
            "-c:v", "mpeg4",
            "-g", str(gop_size),
            "-bf", "0",
            "-qscale:v", str(qscale),
            "-an",
            "-f", "m4v",
            output_path,
        ]
    )


def iframe_smear(base_path: str, motion_path: str, output_path: str, gop_size: int = 300, qscale: int = 4) -> None:
    """EXPERIMENTAL true I-frame-removal datamosh.

    Transcodes both clips to MPEG-4 Part 2 elementary streams with a long GOP
    and no B-frames (one I-frame at the very start, everything else P-frames),
    strips motion_path's own leading I-frame, and splices its P-frames
    directly after base_path's bitstream. A decoder then applies
    motion_path's motion-vector deltas against base_path's last decoded
    frame instead of motion_path's real reference, producing the classic
    "melt into the next scene" datamosh look.

    This manipulates the bitstream directly rather than going through a
    normal encode pipeline, so results vary by content and are not
    guaranteed to decode identically in every player. Verify playback
    (e.g. ffplay/VLC) rather than trusting a clean ffmpeg exit code alone.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        base_raw = str(Path(tmp_dir) / "base.m4v")
        motion_raw = str(Path(tmp_dir) / "motion.m4v")
        _transcode_to_raw_mpeg4(base_path, base_raw, gop_size, qscale)
        _transcode_to_raw_mpeg4(motion_path, motion_raw, gop_size, qscale)

        base_data = Path(base_raw).read_bytes()
        motion_data = Path(motion_raw).read_bytes()

        motion_vops = _find_vops(motion_data)
        if not motion_vops:
            raise ValueError("motion clip has no decodable frames")

        first_non_i = next((pos for pos, coding_type in motion_vops if coding_type != 0), None)
        motion_p_only = motion_data[first_non_i:] if first_non_i is not None else b""

        moshed_path = str(Path(tmp_dir) / "moshed.m4v")
        Path(moshed_path).write_bytes(base_data + motion_p_only)

        fps = get_fps(base_path)
        try:
            ffmpeg.run(["-f", "m4v", "-r", str(fps), "-i", moshed_path, "-c:v", "copy", output_path])
        except ffmpeg.FFmpegError as exc:
            raise ffmpeg.FFmpegError(f"datamosh splice failed to re-mux (experimental technique): {exc}") from exc


def feedback_loop(input_path: str, output_path: str, iterations: int = 3, quality: int = 28) -> None:
    """Repeated low-bitrate re-encode passes, each degrading/smearing the
    previous a bit more (classic 'VHS generation loss' feedback effect). Pure
    ffmpeg re-encode calls, no bitstream editing."""
    iterations = max(1, int(iterations))
    quality = int(np.clip(quality, 2, 51))

    with tempfile.TemporaryDirectory() as tmp_dir:
        current = str(input_path)
        for i in range(iterations):
            next_path = str(Path(tmp_dir) / f"pass_{i}.mp4")
            ffmpeg.run(
                [
                    "-i", current,
                    "-c:v", "libx264",
                    "-crf", str(quality),
                    "-preset", "veryfast",
                    "-c:a", "copy",
                    next_path,
                ]
            )
            current = next_path
        Path(current).replace(output_path)
