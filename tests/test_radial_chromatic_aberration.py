import numpy as np

import imagemessrs.effects  # noqa: F401
from imagemessrs.pipeline import run_pipeline


def test_radial_chromatic_aberration_shape_and_dtype(gradient_image):
    result = run_pipeline(gradient_image, [("radial_chromatic_aberration", {"intensity": 1.0})])
    assert result.shape == gradient_image.shape
    assert result.dtype == np.uint8


def test_radial_chromatic_aberration_changes_image(gradient_image):
    result = run_pipeline(gradient_image, [("radial_chromatic_aberration", {"intensity": 2.0})])
    assert not np.array_equal(result, gradient_image)


def test_radial_chromatic_aberration_zero_intensity_is_noop(gradient_image):
    result = run_pipeline(gradient_image, [("radial_chromatic_aberration", {"intensity": 0.0})])
    assert np.array_equal(result, gradient_image)


def test_radial_chromatic_aberration_edge_start_runs(gradient_image):
    result = run_pipeline(
        gradient_image, [("radial_chromatic_aberration", {"intensity": 1.5, "edge_start": 0.5, "falloff": 3.0})]
    )
    assert result.shape == gradient_image.shape
