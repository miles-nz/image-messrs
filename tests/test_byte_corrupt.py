import numpy as np

import imagemessrs.effects  # noqa: F401
from imagemessrs.pipeline import run_pipeline


def test_byte_corrupt_raw_pixels_shape_and_dtype(gradient_image):
    result = run_pipeline(
        gradient_image, [("byte_corrupt", {"mode": "raw_pixels", "intensity": 0.05, "seed": 1})]
    )
    assert result.shape == gradient_image.shape
    assert result.dtype == np.uint8


def test_byte_corrupt_raw_pixels_changes_image(gradient_image):
    result = run_pipeline(
        gradient_image, [("byte_corrupt", {"mode": "raw_pixels", "intensity": 0.05, "seed": 1})]
    )
    assert not np.array_equal(result, gradient_image)


def test_byte_corrupt_raw_pixels_deterministic(gradient_image):
    params = {"mode": "raw_pixels", "intensity": 0.05, "seed": 7}
    r1 = run_pipeline(gradient_image, [("byte_corrupt", params)])
    r2 = run_pipeline(gradient_image, [("byte_corrupt", params)])
    assert np.array_equal(r1, r2)


def test_byte_corrupt_raw_pixels_zero_intensity_is_noop(gradient_image):
    result = run_pipeline(gradient_image, [("byte_corrupt", {"mode": "raw_pixels", "intensity": 0.0, "seed": 1})])
    assert np.array_equal(result, gradient_image)


def test_byte_corrupt_jpeg_bytes_returns_valid_image(checkerboard_image):
    result = run_pipeline(
        checkerboard_image, [("byte_corrupt", {"mode": "jpeg_bytes", "intensity": 0.02, "seed": 3})]
    )
    assert result.shape[:2] == checkerboard_image.shape[:2]
    assert result.dtype == np.uint8
