import cv2
import numpy as np
import pytest


@pytest.fixture
def gradient_image() -> np.ndarray:
    """32x48 RGB image, horizontal gradient 0->255, identical across channels."""
    h, w = 32, 48
    row = np.linspace(0, 255, w, dtype=np.uint8)
    gradient = np.tile(row, (h, 1))
    return np.stack([gradient, gradient, gradient], axis=-1).astype(np.uint8)


@pytest.fixture
def checkerboard_image() -> np.ndarray:
    """32x32 RGB checkerboard, 4px squares, black/white."""
    h, w, square = 32, 32, 4
    yy, xx = np.indices((h, w))
    pattern = ((yy // square) + (xx // square)) % 2
    img = np.where(pattern[..., None] == 1, 255, 0).astype(np.uint8)
    return np.repeat(img, 3, axis=-1)


@pytest.fixture
def sample_video_path(tmp_path) -> str:
    """8-frame, 64x48, silent synthetic mp4 with a moving square."""
    path = tmp_path / "sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 48))
    for i in range(8):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = (i * 30) % 256
        x = (i * 6) % 50
        frame[10:20, x : x + 10] = [255, 255, 255]
        writer.write(frame)
    writer.release()
    return str(path)


@pytest.fixture
def sample_video_path_2(tmp_path) -> str:
    """A second, differently-moving 8-frame silent synthetic mp4 (same size)."""
    path = tmp_path / "sample2.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 48))
    for i in range(8):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 2] = (i * 25) % 256
        y = (i * 5) % 30
        frame[y : y + 8, 40:55] = [0, 255, 0]
        writer.write(frame)
    writer.release()
    return str(path)
