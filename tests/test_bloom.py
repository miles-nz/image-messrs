import numpy as np

import imagemessrs.effects  # noqa: F401
from imagemessrs.pipeline import run_pipeline


def test_bloom_shape_and_dtype(gradient_image):
    result = run_pipeline(gradient_image, [("bloom", {"threshold": 0.3, "radius": 4, "intensity": 0.6})])
    assert result.shape == gradient_image.shape
    assert result.dtype == np.uint8


def test_bloom_changes_image(gradient_image):
    result = run_pipeline(gradient_image, [("bloom", {"threshold": 0.2, "radius": 4, "intensity": 0.8})])
    assert not np.array_equal(result, gradient_image)


def test_bloom_zero_intensity_is_noop(gradient_image):
    result = run_pipeline(gradient_image, [("bloom", {"intensity": 0.0})])
    assert np.array_equal(result, gradient_image)


def test_bloom_tints_run(gradient_image):
    for tint in ("white", "warm", "cool"):
        result = run_pipeline(gradient_image, [("bloom", {"intensity": 0.5, "tint": tint})])
        assert result.shape == gradient_image.shape
