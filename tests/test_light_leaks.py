import numpy as np

import imagemessrs.effects  # noqa: F401
from imagemessrs.core.registry import get_effect
from imagemessrs.effects.base import coerce_params
from imagemessrs.pipeline import run_pipeline


def test_light_leaks_shape_and_dtype(gradient_image):
    result = run_pipeline(gradient_image, [("light_leaks", {"intensity": 0.6, "seed": 1})])
    assert result.shape == gradient_image.shape
    assert result.dtype == np.uint8


def test_light_leaks_changes_image(gradient_image):
    result = run_pipeline(gradient_image, [("light_leaks", {"intensity": 0.8, "seed": 1})])
    assert not np.array_equal(result, gradient_image)


def test_light_leaks_deterministic(gradient_image):
    params = {"position": "bottom-left", "intensity": 0.7, "size": 1.2, "color": "#ff3c28", "seed": 5}
    r1 = run_pipeline(gradient_image, [("light_leaks", params)])
    r2 = run_pipeline(gradient_image, [("light_leaks", params)])
    assert np.array_equal(r1, r2)


def test_light_leaks_zero_intensity_is_noop(gradient_image):
    result = run_pipeline(gradient_image, [("light_leaks", {"intensity": 0.0, "seed": 1})])
    assert np.array_equal(result, gradient_image)


def test_light_leaks_random_position_runs(gradient_image):
    result = run_pipeline(gradient_image, [("light_leaks", {"position": "random", "intensity": 0.5, "seed": 3})])
    assert result.shape == gradient_image.shape


def test_light_leaks_custom_hex_color_differs_from_default(gradient_image):
    default = run_pipeline(gradient_image, [("light_leaks", {"intensity": 0.8, "seed": 1})])
    blue = run_pipeline(gradient_image, [("light_leaks", {"intensity": 0.8, "seed": 1, "color": "#3060ff"})])
    assert not np.array_equal(default, blue)


def test_light_leaks_invalid_hex_falls_back_to_default():
    effect = get_effect("light_leaks")
    color_spec = next(p for p in effect.params if p.name == "color")
    params = coerce_params(effect.params, {"color": "not-a-color"})
    assert params["color"] == color_spec.default


def test_light_leaks_invalid_edge_hex_falls_back_to_default():
    effect = get_effect("light_leaks")
    edge_spec = next(p for p in effect.params if p.name == "edge_color")
    params = coerce_params(effect.params, {"edge_color": "not-a-color"})
    assert params["edge_color"] == edge_spec.default


def test_light_leaks_custom_edge_color_differs_from_default(gradient_image):
    default = run_pipeline(gradient_image, [("light_leaks", {"intensity": 0.8, "seed": 1})])
    custom = run_pipeline(
        gradient_image, [("light_leaks", {"intensity": 0.8, "seed": 1, "edge_color": "#3060ff"})]
    )
    assert not np.array_equal(default, custom)


def test_light_leaks_dual_tone_differs_from_flat_tint(gradient_image):
    dual_tone = run_pipeline(
        gradient_image, [("light_leaks", {"intensity": 0.8, "seed": 1, "color": "#fff4d6", "edge_color": "#7a2bff"})]
    )
    flat_tint = run_pipeline(
        gradient_image, [("light_leaks", {"intensity": 0.8, "seed": 1, "color": "#fff4d6", "edge_color": "#fff4d6"})]
    )
    assert not np.array_equal(dual_tone, flat_tint)
