import numpy as np

import imagemessrs.effects  # noqa: F401
from imagemessrs.pipeline import run_pipeline


def test_light_leaks_shape_and_dtype(gradient_image):
    result = run_pipeline(gradient_image, [("light_leaks", {"intensity": 0.6, "seed": 1})])
    assert result.shape == gradient_image.shape
    assert result.dtype == np.uint8


def test_light_leaks_changes_image(gradient_image):
    result = run_pipeline(gradient_image, [("light_leaks", {"intensity": 0.8, "seed": 1})])
    assert not np.array_equal(result, gradient_image)


def test_light_leaks_deterministic(gradient_image):
    params = {"position": "bottom-left", "intensity": 0.7, "size": 1.2, "color": "red", "seed": 5}
    r1 = run_pipeline(gradient_image, [("light_leaks", params)])
    r2 = run_pipeline(gradient_image, [("light_leaks", params)])
    assert np.array_equal(r1, r2)


def test_light_leaks_zero_intensity_is_noop(gradient_image):
    result = run_pipeline(gradient_image, [("light_leaks", {"intensity": 0.0, "seed": 1})])
    assert np.array_equal(result, gradient_image)


def test_light_leaks_random_position_runs(gradient_image):
    result = run_pipeline(gradient_image, [("light_leaks", {"position": "random", "intensity": 0.5, "seed": 3})])
    assert result.shape == gradient_image.shape
