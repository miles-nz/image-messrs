import numpy as np

import imagemessrs.effects  # noqa: F401
from imagemessrs.pipeline import run_pipeline


def test_wave_displace_shape_and_dtype(checkerboard_image):
    result = run_pipeline(
        checkerboard_image, [("wave_displace", {"amplitude_x": 5, "amplitude_y": 5, "frequency_x": 2, "frequency_y": 2})]
    )
    assert result.shape == checkerboard_image.shape
    assert result.dtype == np.uint8


def test_wave_displace_changes_image(checkerboard_image):
    result = run_pipeline(
        checkerboard_image, [("wave_displace", {"amplitude_x": 6, "amplitude_y": 6, "frequency_x": 3, "frequency_y": 3})]
    )
    assert not np.array_equal(result, checkerboard_image)


def test_wave_displace_deterministic(checkerboard_image):
    params = {"amplitude_x": 4, "amplitude_y": 4, "frequency_x": 2, "frequency_y": 2, "phase": 1.0}
    r1 = run_pipeline(checkerboard_image, [("wave_displace", params)])
    r2 = run_pipeline(checkerboard_image, [("wave_displace", params)])
    assert np.array_equal(r1, r2)


def test_wave_displace_zero_amplitude_is_noop(checkerboard_image):
    params = {"amplitude_x": 0, "amplitude_y": 0, "frequency_x": 3, "frequency_y": 3}
    result = run_pipeline(checkerboard_image, [("wave_displace", params)])
    assert np.array_equal(result, checkerboard_image)
