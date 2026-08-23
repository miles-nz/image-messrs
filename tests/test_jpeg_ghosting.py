import numpy as np

import imagemessrs.effects  # noqa: F401
from imagemessrs.pipeline import run_pipeline


def test_jpeg_ghosting_shape_and_dtype(gradient_image):
    result = run_pipeline(gradient_image, [("jpeg_ghosting", {"generations": 3, "quality": 20})])
    assert result.shape == gradient_image.shape
    assert result.dtype == np.uint8


def test_jpeg_ghosting_changes_image(gradient_image):
    result = run_pipeline(gradient_image, [("jpeg_ghosting", {"generations": 4, "quality": 10})])
    assert not np.array_equal(result, gradient_image)


def test_jpeg_ghosting_deterministic(gradient_image):
    params = {"generations": 3, "quality": 15, "quality_decay": 0.1}
    r1 = run_pipeline(gradient_image, [("jpeg_ghosting", params)])
    r2 = run_pipeline(gradient_image, [("jpeg_ghosting", params)])
    assert np.array_equal(r1, r2)


def test_jpeg_ghosting_zero_generations_is_noop(gradient_image):
    result = run_pipeline(gradient_image, [("jpeg_ghosting", {"generations": 0, "quality": 10})])
    assert np.array_equal(result, gradient_image)


def test_jpeg_ghosting_deep_fry_runs(checkerboard_image):
    result = run_pipeline(
        checkerboard_image, [("jpeg_ghosting", {"generations": 2, "quality": 20, "deep_fry": True})]
    )
    assert result.shape == checkerboard_image.shape
    assert result.dtype == np.uint8
