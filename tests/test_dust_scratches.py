import numpy as np

import imagemessrs.effects  # noqa: F401
from imagemessrs.pipeline import run_pipeline


def test_dust_scratches_shape_and_dtype(gradient_image):
    result = run_pipeline(
        gradient_image, [("dust_scratches", {"dust_density": 0.01, "scratch_count": 5, "seed": 1})]
    )
    assert result.shape == gradient_image.shape
    assert result.dtype == np.uint8


def test_dust_scratches_changes_image(gradient_image):
    result = run_pipeline(
        gradient_image, [("dust_scratches", {"dust_density": 0.02, "scratch_count": 8, "seed": 1})]
    )
    assert not np.array_equal(result, gradient_image)


def test_dust_scratches_deterministic(gradient_image):
    params = {"dust_density": 0.01, "dust_intensity": 0.6, "scratch_count": 4, "scratch_intensity": 0.5, "seed": 11}
    r1 = run_pipeline(gradient_image, [("dust_scratches", params)])
    r2 = run_pipeline(gradient_image, [("dust_scratches", params)])
    assert np.array_equal(r1, r2)


def test_dust_scratches_zero_params_is_noop(gradient_image):
    params = {"dust_density": 0.0, "scratch_count": 0, "seed": 1}
    result = run_pipeline(gradient_image, [("dust_scratches", params)])
    assert np.array_equal(result, gradient_image)
