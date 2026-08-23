import numpy as np

import imagemessrs.effects  # noqa: F401
from imagemessrs.pipeline import run_pipeline


def test_scanline_jitter_shape_and_dtype(checkerboard_image):
    result = run_pipeline(
        checkerboard_image,
        [("scanline_jitter", {"amplitude": 5, "frequency": 2, "jitter": 2, "interlace_offset": 3, "seed": 1})],
    )
    assert result.shape == checkerboard_image.shape
    assert result.dtype == np.uint8


def test_scanline_jitter_changes_image(checkerboard_image):
    result = run_pipeline(
        checkerboard_image,
        [("scanline_jitter", {"amplitude": 5, "frequency": 2, "jitter": 2, "interlace_offset": 3, "seed": 1})],
    )
    assert not np.array_equal(result, checkerboard_image)


def test_scanline_jitter_deterministic(checkerboard_image):
    params = {"amplitude": 4, "frequency": 3, "jitter": 3, "interlace_offset": 2, "seed": 7}
    r1 = run_pipeline(checkerboard_image, [("scanline_jitter", params)])
    r2 = run_pipeline(checkerboard_image, [("scanline_jitter", params)])
    assert np.array_equal(r1, r2)


def test_scanline_jitter_zero_params_is_noop(checkerboard_image):
    params = {"amplitude": 0, "frequency": 3, "jitter": 0, "interlace_offset": 0, "seed": 0}
    result = run_pipeline(checkerboard_image, [("scanline_jitter", params)])
    assert np.array_equal(result, checkerboard_image)
