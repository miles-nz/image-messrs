import numpy as np

import imagemessrs.effects  # noqa: F401
from imagemessrs.pipeline import run_pipeline


def test_block_displace_shape_and_dtype(checkerboard_image):
    result = run_pipeline(
        checkerboard_image,
        [("block_displace", {"block_size": 4, "swap_probability": 0.8, "jitter_strength": 0.5, "seed": 1})],
    )
    assert result.shape == checkerboard_image.shape
    assert result.dtype == np.uint8


def test_block_displace_changes_image(checkerboard_image):
    result = run_pipeline(
        checkerboard_image,
        [("block_displace", {"block_size": 4, "swap_probability": 0.8, "jitter_strength": 0.5, "seed": 1})],
    )
    assert not np.array_equal(result, checkerboard_image)


def test_block_displace_deterministic(checkerboard_image):
    params = {"block_size": 4, "swap_probability": 0.5, "jitter_strength": 0.4, "seed": 42}
    r1 = run_pipeline(checkerboard_image, [("block_displace", params)])
    r2 = run_pipeline(checkerboard_image, [("block_displace", params)])
    assert np.array_equal(r1, r2)


def test_block_displace_zero_params_is_noop(checkerboard_image):
    params = {"block_size": 4, "swap_probability": 0.0, "jitter_strength": 0.0, "seed": 0}
    result = run_pipeline(checkerboard_image, [("block_displace", params)])
    assert np.array_equal(result, checkerboard_image)
