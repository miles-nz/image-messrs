import numpy as np

import imagemessrs.effects  # noqa: F401
from imagemessrs.pipeline import run_pipeline


def test_lens_distort_shape_and_dtype(checkerboard_image):
    result = run_pipeline(checkerboard_image, [("lens_distort", {"strength": 0.4, "radius": 1.0})])
    assert result.shape == checkerboard_image.shape
    assert result.dtype == np.uint8


def test_lens_distort_changes_image_barrel(checkerboard_image):
    result = run_pipeline(checkerboard_image, [("lens_distort", {"strength": 0.5})])
    assert not np.array_equal(result, checkerboard_image)


def test_lens_distort_changes_image_pincushion(checkerboard_image):
    result = run_pipeline(checkerboard_image, [("lens_distort", {"strength": -0.5})])
    assert not np.array_equal(result, checkerboard_image)


def test_lens_distort_fisheye_runs(checkerboard_image):
    result = run_pipeline(checkerboard_image, [("lens_distort", {"strength": 0.6, "fisheye": True})])
    assert result.shape == checkerboard_image.shape
    assert result.dtype == np.uint8


def test_lens_distort_zero_strength_is_noop(checkerboard_image):
    result = run_pipeline(checkerboard_image, [("lens_distort", {"strength": 0.0})])
    assert np.array_equal(result, checkerboard_image)
