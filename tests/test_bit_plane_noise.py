import numpy as np

import imagemessrs.effects  # noqa: F401
from imagemessrs.pipeline import run_pipeline


def test_bit_plane_noise_shape_and_dtype(gradient_image):
    result = run_pipeline(gradient_image, [("bit_plane_noise", {"bit_index": 6, "intensity": 0.3, "seed": 1})])
    assert result.shape == gradient_image.shape
    assert result.dtype == np.uint8


def test_bit_plane_noise_changes_image(gradient_image):
    result = run_pipeline(gradient_image, [("bit_plane_noise", {"bit_index": 6, "intensity": 0.5, "seed": 1})])
    assert not np.array_equal(result, gradient_image)


def test_bit_plane_noise_deterministic(gradient_image):
    params = {"bit_index": 4, "mode": "randomize", "intensity": 0.3, "seed": 9}
    r1 = run_pipeline(gradient_image, [("bit_plane_noise", params)])
    r2 = run_pipeline(gradient_image, [("bit_plane_noise", params)])
    assert np.array_equal(r1, r2)


def test_bit_plane_noise_zero_intensity_is_noop(gradient_image):
    result = run_pipeline(gradient_image, [("bit_plane_noise", {"intensity": 0.0, "seed": 1})])
    assert np.array_equal(result, gradient_image)


def test_bit_plane_noise_zero_mode_clears_bit(checkerboard_image):
    result = run_pipeline(
        checkerboard_image, [("bit_plane_noise", {"bit_index": 0, "mode": "zero", "intensity": 1.0, "seed": 1})]
    )
    assert np.all(result.reshape(-1) & 1 == 0)
