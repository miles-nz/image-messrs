import numpy as np

import imagemessrs.effects  # noqa: F401
from imagemessrs.pipeline import run_pipeline


def test_crt_scanlines_shape_and_dtype(checkerboard_image):
    result = run_pipeline(
        checkerboard_image, [("crt_scanlines", {"line_spacing": 2, "line_darkness": 0.4, "subpixel_strength": 0.3})]
    )
    assert result.shape == checkerboard_image.shape
    assert result.dtype == np.uint8


def test_crt_scanlines_changes_image(checkerboard_image):
    result = run_pipeline(
        checkerboard_image, [("crt_scanlines", {"line_spacing": 2, "line_darkness": 0.5, "subpixel_strength": 0.5})]
    )
    assert not np.array_equal(result, checkerboard_image)


def test_crt_scanlines_deterministic(checkerboard_image):
    params = {"line_spacing": 3, "line_darkness": 0.3, "subpixel_strength": 0.2, "brightness_boost": 0.1}
    r1 = run_pipeline(checkerboard_image, [("crt_scanlines", params)])
    r2 = run_pipeline(checkerboard_image, [("crt_scanlines", params)])
    assert np.array_equal(r1, r2)


def test_crt_scanlines_zero_params_is_noop(checkerboard_image):
    params = {"line_spacing": 2, "line_darkness": 0.0, "subpixel_strength": 0.0, "brightness_boost": 0.0}
    result = run_pipeline(checkerboard_image, [("crt_scanlines", params)])
    assert np.array_equal(result, checkerboard_image)
