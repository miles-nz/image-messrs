import numpy as np

import imagemessrs.effects  # noqa: F401  (populates registry)
from imagemessrs.core.registry import get_effect
from imagemessrs.pipeline import run_pipeline


def test_channel_shift_registered():
    effect = get_effect("channel_shift")
    assert effect.category == "glitch"
    assert effect.name == "channel_shift"


def test_channel_shift_changes_image(gradient_image):
    result = run_pipeline(gradient_image, [("channel_shift", {"blue_dx": 5})])
    assert result.shape == gradient_image.shape
    assert result.dtype == np.uint8
    assert not np.array_equal(result, gradient_image)


def test_channel_shift_identity_with_zero_offsets(checkerboard_image):
    zero_params = {
        "red_dx": 0, "red_dy": 0,
        "green_dx": 0, "green_dy": 0,
        "blue_dx": 0, "blue_dy": 0,
    }
    result = run_pipeline(checkerboard_image, [("channel_shift", zero_params)])
    assert np.array_equal(result, checkerboard_image)


def test_channel_shift_deterministic(gradient_image):
    params = {"blue_dx": 3, "blue_dy": -2}
    r1 = run_pipeline(gradient_image, [("channel_shift", params)])
    r2 = run_pipeline(gradient_image, [("channel_shift", params)])
    assert np.array_equal(r1, r2)
