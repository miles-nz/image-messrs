import numpy as np

import imagemessrs.effects  # noqa: F401
from imagemessrs.core.registry import get_effect
from imagemessrs.effects.base import coerce_params


def _run_blend(name: str, image_a, image_b, **params):
    effect = get_effect(name)
    assert effect.multi_image
    coerced = coerce_params(effect.params, params)
    return effect.fn(image_a, image_b, **coerced)


def test_optical_flow_blend_shape_and_dtype(gradient_image, checkerboard_image):
    result = _run_blend("optical_flow_blend", gradient_image, checkerboard_image, blend_factor=0.5)
    assert result.shape == gradient_image.shape
    assert result.dtype == np.uint8


def test_optical_flow_blend_zero_factor_is_noop(gradient_image, checkerboard_image):
    result = _run_blend("optical_flow_blend", gradient_image, checkerboard_image, blend_factor=0.0, flow_strength=1.0)
    assert np.array_equal(result, gradient_image)


def test_optical_flow_blend_deterministic(gradient_image, checkerboard_image):
    params = {"blend_factor": 0.4, "flow_strength": 1.0}
    r1 = _run_blend("optical_flow_blend", gradient_image, checkerboard_image, **params)
    r2 = _run_blend("optical_flow_blend", gradient_image, checkerboard_image, **params)
    assert np.array_equal(r1, r2)


def test_poisson_blend_shape_and_dtype(gradient_image, checkerboard_image):
    result = _run_blend("poisson_blend", gradient_image, checkerboard_image, mode="normal")
    assert result.shape == gradient_image.shape
    assert result.dtype == np.uint8


def test_poisson_blend_mixed_mode_runs(gradient_image, checkerboard_image):
    result = _run_blend("poisson_blend", gradient_image, checkerboard_image, mode="mixed")
    assert result.shape == gradient_image.shape


def test_poisson_blend_fit_modes_run(gradient_image, checkerboard_image):
    for fit in ("cover", "contain", "stretch", "seam"):
        result = _run_blend("poisson_blend", gradient_image, checkerboard_image, mode="normal", fit=fit)
        assert result.shape == gradient_image.shape


def test_poisson_blend_canvas_defaults_to_image_a_shape(gradient_image, checkerboard_image):
    result = _run_blend("poisson_blend", gradient_image, checkerboard_image, mode="normal", canvas="image_a")
    assert result.shape == gradient_image.shape


def test_poisson_blend_canvas_b_uses_image_b_shape(gradient_image, checkerboard_image):
    result = _run_blend("poisson_blend", gradient_image, checkerboard_image, mode="normal", canvas="image_b")
    assert result.shape == checkerboard_image.shape
    assert result.dtype == np.uint8


def test_seam_merge_shape_and_dtype(gradient_image, checkerboard_image):
    result = _run_blend("seam_merge", gradient_image, checkerboard_image, progress=0.5, feather=2)
    assert result.shape == gradient_image.shape
    assert result.dtype == np.uint8


def test_seam_merge_progress_zero_is_all_image_a(gradient_image, checkerboard_image):
    result = _run_blend("seam_merge", gradient_image, checkerboard_image, progress=0.0, feather=0)
    assert np.array_equal(result, gradient_image)


def test_seam_merge_progress_one_is_all_image_b(gradient_image, checkerboard_image):
    from imagemessrs.effects.blend._shared import resize_to_match

    result = _run_blend("seam_merge", gradient_image, checkerboard_image, progress=1.0, feather=0)
    expected_b = resize_to_match(checkerboard_image, gradient_image.shape[:2])
    assert np.array_equal(result, expected_b)


def test_energy_warp_shape_and_dtype(gradient_image, checkerboard_image):
    result = _run_blend("energy_warp", gradient_image, checkerboard_image, strength=0.2)
    assert result.shape == gradient_image.shape
    assert result.dtype == np.uint8


def test_energy_warp_zero_strength_is_noop(checkerboard_image):
    result = _run_blend("energy_warp", checkerboard_image, checkerboard_image, strength=0.0)
    assert np.array_equal(result, checkerboard_image)


def test_difference_blend_shape_and_dtype(gradient_image, checkerboard_image):
    result = _run_blend("difference_blend", gradient_image, checkerboard_image, mode="double_exposure")
    assert result.shape == gradient_image.shape
    assert result.dtype == np.uint8


def test_difference_blend_difference_mode_changes_image(gradient_image, checkerboard_image):
    result = _run_blend("difference_blend", gradient_image, checkerboard_image, mode="difference", blend_factor=1.0)
    assert not np.array_equal(result, gradient_image)


def test_difference_blend_zero_factor_is_noop(gradient_image, checkerboard_image):
    result = _run_blend("difference_blend", gradient_image, checkerboard_image, mode="difference", blend_factor=0.0)
    assert np.array_equal(result, gradient_image)


def test_slit_scan_blend_shape_and_dtype(gradient_image, checkerboard_image):
    result = _run_blend("slit_scan_blend", gradient_image, checkerboard_image, frequency=3.0)
    assert result.shape == gradient_image.shape
    assert result.dtype == np.uint8


def test_slit_scan_blend_changes_image(gradient_image, checkerboard_image):
    result = _run_blend("slit_scan_blend", gradient_image, checkerboard_image, frequency=4.0, progress=0.1)
    assert not np.array_equal(result, gradient_image)


def test_slit_scan_blend_zero_frequency_at_trough_is_all_a(gradient_image, checkerboard_image):
    # frequency=0 -> constant weight; progress=0.25 puts sin(-phase) at -1, weight_b=0 everywhere.
    result = _run_blend("slit_scan_blend", gradient_image, checkerboard_image, frequency=0.0, progress=0.25)
    assert np.array_equal(result, gradient_image)


def test_slit_scan_blend_deterministic(gradient_image, checkerboard_image):
    params = {"axis": "rows", "frequency": 2.0, "progress": 0.3, "sharpness": 0.5}
    r1 = _run_blend("slit_scan_blend", gradient_image, checkerboard_image, **params)
    r2 = _run_blend("slit_scan_blend", gradient_image, checkerboard_image, **params)
    assert np.array_equal(r1, r2)


def test_displacement_map_blend_shape_and_dtype(gradient_image, checkerboard_image):
    result = _run_blend("displacement_map_blend", gradient_image, checkerboard_image, strength=15.0)
    assert result.shape == gradient_image.shape
    assert result.dtype == np.uint8


def test_displacement_map_blend_changes_image(gradient_image, checkerboard_image):
    result = _run_blend("displacement_map_blend", gradient_image, checkerboard_image, strength=20.0)
    assert not np.array_equal(result, gradient_image)


def test_displacement_map_blend_zero_strength_is_noop(gradient_image, checkerboard_image):
    result = _run_blend(
        "displacement_map_blend", gradient_image, checkerboard_image, strength=0.0, blend_factor=0.0
    )
    assert np.array_equal(result, gradient_image)
