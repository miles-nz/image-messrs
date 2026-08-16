import numpy as np

import imagemessrs.effects  # noqa: F401
from imagemessrs.pipeline import run_pipeline


def test_pixel_sort_shape_and_dtype(checkerboard_image):
    result = run_pipeline(
        checkerboard_image,
        [("pixel_sort", {"axis": "rows", "threshold_low": 0, "threshold_high": 255})],
    )
    assert result.shape == checkerboard_image.shape
    assert result.dtype == np.uint8


def test_pixel_sort_changes_image(checkerboard_image):
    result = run_pipeline(
        checkerboard_image,
        [("pixel_sort", {"axis": "rows", "threshold_low": 0, "threshold_high": 255})],
    )
    assert not np.array_equal(result, checkerboard_image)


def test_pixel_sort_deterministic(checkerboard_image):
    params = {"axis": "columns", "sort_by": "brightness", "threshold_low": 0, "threshold_high": 255}
    r1 = run_pipeline(checkerboard_image, [("pixel_sort", params)])
    r2 = run_pipeline(checkerboard_image, [("pixel_sort", params)])
    assert np.array_equal(r1, r2)


def test_pixel_sort_narrow_threshold_is_noop(gradient_image):
    result = run_pipeline(
        gradient_image,
        [("pixel_sort", {"axis": "rows", "threshold_low": 1000, "threshold_high": 2000})],
    )
    assert np.array_equal(result, gradient_image)
