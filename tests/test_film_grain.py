import numpy as np

import imagemessrs.effects  # noqa: F401
from imagemessrs.pipeline import run_pipeline


def test_film_grain_shape_and_dtype(gradient_image):
    result = run_pipeline(gradient_image, [("film_grain", {"intensity": 0.4, "size": 1.0, "seed": 1})])
    assert result.shape == gradient_image.shape
    assert result.dtype == np.uint8


def test_film_grain_changes_image(gradient_image):
    result = run_pipeline(gradient_image, [("film_grain", {"intensity": 0.5, "seed": 1})])
    assert not np.array_equal(result, gradient_image)


def test_film_grain_deterministic(gradient_image):
    params = {"intensity": 0.4, "size": 2.0, "monochrome": False, "seed": 8}
    r1 = run_pipeline(gradient_image, [("film_grain", params)])
    r2 = run_pipeline(gradient_image, [("film_grain", params)])
    assert np.array_equal(r1, r2)


def test_film_grain_zero_intensity_is_noop(gradient_image):
    result = run_pipeline(gradient_image, [("film_grain", {"intensity": 0.0, "seed": 1})])
    assert np.array_equal(result, gradient_image)


def test_film_grain_color_mode_runs(gradient_image):
    result = run_pipeline(gradient_image, [("film_grain", {"intensity": 0.3, "monochrome": False, "seed": 2})])
    assert result.shape == gradient_image.shape
