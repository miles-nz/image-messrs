import cv2
import numpy as np
import pytest

import imagemessrs.effects  # noqa: F401
from imagemessrs.core.exif_utils import extract_camera_info
from imagemessrs.core.io_utils import save_image
from imagemessrs.core.registry import get_effect
from imagemessrs.effects.base import coerce_params
from imagemessrs.effects.color.profiles import get_profile, load_profiles
from imagemessrs.effects.color.source_cameras import (
    get_source_camera,
    load_source_cameras,
    match_source_camera_from_exif,
)


def _run(name: str, image, **params):
    effect = get_effect(name)
    coerced = coerce_params(effect.params, params)
    return effect.fn(image, **coerced)


def test_registration():
    effect = get_effect("vintage_camera_profile")
    assert effect.category == "color"
    by_name = {p.name: p for p in effect.params}
    assert by_name["profile"].kind == "choice"
    assert len(by_name["profile"].choices) > 0
    assert by_name["source_camera"].kind == "choice"
    assert len(by_name["source_camera"].choices) > 0
    assert by_name["strength"].kind == "float"
    assert by_name["seed"].kind == "int"
    assert by_name["skin_tone_protection"].kind == "float"
    assert by_name["skin_tone_protection"].default == 0.0
    assert by_name["skin_tone"].kind == "choice"
    assert by_name["skin_tone"].default == "reddish_pink"
    assert "reddish_pink" in by_name["skin_tone"].choices
    assert len(by_name["skin_tone"].choices) > 1


def test_none_profile_is_last_and_not_default():
    effect = get_effect("vintage_camera_profile")
    by_name = {p.name: p for p in effect.params}
    assert by_name["profile"].choices[-1] == "none"
    assert by_name["profile"].default != "none"


@pytest.mark.parametrize("profile", ["kodachrome_64", "kodak_dc4800_1999"])
def test_shape_and_dtype_preserved(gradient_image, profile):
    result = _run("vintage_camera_profile", gradient_image, profile=profile, strength=1.0)
    assert result.shape == gradient_image.shape
    assert result.dtype == np.uint8


def test_strength_zero_is_noop(gradient_image):
    result = _run("vintage_camera_profile", gradient_image, profile="kodachrome_64", strength=0.0)
    assert np.array_equal(result, gradient_image)


def test_strength_one_changes_image(checkerboard_image):
    result = _run("vintage_camera_profile", checkerboard_image, profile="kodachrome_64", strength=1.0)
    assert not np.array_equal(result, checkerboard_image)


def test_deterministic_same_seed(gradient_image):
    params = {"profile": "holga_120n", "strength": 1.0, "seed": 42}
    r1 = _run("vintage_camera_profile", gradient_image, **params)
    r2 = _run("vintage_camera_profile", gradient_image, **params)
    assert np.array_equal(r1, r2)


def test_different_seeds_diverge(gradient_image):
    r1 = _run("vintage_camera_profile", gradient_image, profile="holga_120n", strength=1.0, seed=1)
    r2 = _run("vintage_camera_profile", gradient_image, profile="holga_120n", strength=1.0, seed=2)
    assert not np.array_equal(r1, r2)


def _mean_saturation(image: np.ndarray) -> float:
    return float(cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[..., 1].mean().astype(np.float64))


def test_skin_tone_protection_reduces_saturation_shift_on_reddish_hue():
    skin_patch = np.full((16, 16, 3), [216, 172, 152], dtype=np.uint8)  # pinkish skin tone
    orig_sat = _mean_saturation(skin_patch)

    unprotected = _run(
        "vintage_camera_profile", skin_patch,
        source_camera="unknown", profile="fuji_velvia", strength=1.0, skin_tone_protection=0.0,
    )
    protected = _run(
        "vintage_camera_profile", skin_patch,
        source_camera="unknown", profile="fuji_velvia", strength=1.0, skin_tone_protection=1.0,
    )

    unprotected_shift = abs(_mean_saturation(unprotected) - orig_sat)
    protected_shift = abs(_mean_saturation(protected) - orig_sat)
    assert protected_shift < unprotected_shift


def test_skin_tone_protection_barely_affects_non_skin_hue():
    blue_patch = np.full((16, 16, 3), [80, 110, 220], dtype=np.uint8)  # saturated blue, far from skin hue

    unprotected = _run(
        "vintage_camera_profile", blue_patch,
        source_camera="unknown", profile="fuji_velvia", strength=1.0, skin_tone_protection=0.0,
    )
    protected = _run(
        "vintage_camera_profile", blue_patch,
        source_camera="unknown", profile="fuji_velvia", strength=1.0, skin_tone_protection=1.0,
    )

    assert abs(_mean_saturation(protected) - _mean_saturation(unprotected)) < 2.0


def _hsv_patch(hue: int, sat: int, val: int, size: int = 16) -> np.ndarray:
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[..., 0], hsv[..., 1], hsv[..., 2] = hue, sat, val
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def test_skin_tone_choice_targets_the_selected_hue_band():
    # Hue 20 (cv2 0-179 scale) sits in golden_olive's band, far from reddish_pink's.
    olive_patch = _hsv_patch(hue=20, sat=140, val=190)
    orig_sat = _mean_saturation(olive_patch)

    protected_as_pink = _run(
        "vintage_camera_profile", olive_patch,
        source_camera="unknown", profile="fuji_velvia", strength=1.0,
        skin_tone="reddish_pink", skin_tone_protection=1.0,
    )
    protected_as_olive = _run(
        "vintage_camera_profile", olive_patch,
        source_camera="unknown", profile="fuji_velvia", strength=1.0,
        skin_tone="golden_olive", skin_tone_protection=1.0,
    )

    pink_shift = abs(_mean_saturation(protected_as_pink) - orig_sat)
    olive_shift = abs(_mean_saturation(protected_as_olive) - orig_sat)
    assert olive_shift < pink_shift


def test_none_profile_with_unknown_source_is_noop(gradient_image):
    result = _run(
        "vintage_camera_profile", gradient_image,
        source_camera="unknown", profile="none", strength=1.0,
    )
    assert np.array_equal(result, gradient_image)


def test_none_profile_still_applies_source_correction(checkerboard_image):
    result = _run(
        "vintage_camera_profile", checkerboard_image,
        source_camera="iphone_pro", profile="none", strength=1.0,
    )
    assert not np.array_equal(result, checkerboard_image)


def test_source_camera_correction_changes_output(checkerboard_image):
    baseline = _run(
        "vintage_camera_profile", checkerboard_image,
        source_camera="unknown", profile="kodachrome_64", strength=1.0, seed=0,
    )
    corrected = _run(
        "vintage_camera_profile", checkerboard_image,
        source_camera="iphone_pro", profile="kodachrome_64", strength=1.0, seed=0,
    )
    assert not np.array_equal(baseline, corrected)


def test_load_profiles_valid_and_malformed(tmp_path):
    (tmp_path / "minimal.json").write_text(
        '{"id": "minimal", "label": "Minimal", "type": "film"}'
    )
    (tmp_path / "broken.json").write_text("{not valid json")

    profiles = load_profiles(tmp_path)

    assert set(profiles.keys()) == {"minimal"}
    minimal = profiles["minimal"]
    assert minimal.tone_curve.r == [(0.0, 0.0), (255.0, 255.0)]
    assert minimal.saturation.global_mult == 1.0
    assert minimal.vignette.strength == 0.0
    assert minimal.jpeg_quality is None


def test_get_profile_unknown_falls_back():
    profile = get_profile("does_not_exist")
    assert profile is not None


def test_load_source_cameras_valid_and_malformed(tmp_path):
    (tmp_path / "minimal.json").write_text('{"id": "minimal", "label": "Minimal"}')
    (tmp_path / "broken.json").write_text("{not valid json")

    cameras = load_source_cameras(tmp_path)

    assert set(cameras.keys()) == {"minimal"}
    minimal = cameras["minimal"]
    assert minimal.correction.undo_sharpen == 0.0
    assert minimal.correction.undo_hdr == 0.0
    assert minimal.exif_aliases == []


def test_get_source_camera_unknown_falls_back():
    camera = get_source_camera("does_not_exist")
    assert camera is not None


def test_match_source_camera_from_exif():
    assert match_source_camera_from_exif("Apple", "iPhone 16 Pro") == "iphone_pro"
    assert match_source_camera_from_exif("Kodak", "Charmera") == "kodak_charmera"
    assert match_source_camera_from_exif(None, None) is None
    assert match_source_camera_from_exif("Fictional Corp", "Model X") is None


def test_match_source_camera_from_exif_film_and_instant():
    assert match_source_camera_from_exif("Fujifilm", "INSTAX Mini Evo") == "fujifilm_instax"
    assert match_source_camera_from_exif("Polaroid", "Now+") == "polaroid_instant"
    # Real Fujifilm digital cameras should still match the generic mirrorless
    # bucket, not the Instax preset, since "instax" (not bare "fujifilm") is
    # what the Instax profile keys off.
    assert match_source_camera_from_exif("FUJIFILM", "X100V") == "dslr_mirrorless"


@pytest.mark.parametrize(
    "camera_id", ["fujifilm_instax", "polaroid_instant", "disposable_35mm", "film_slr_35mm"]
)
def test_film_instant_source_cameras_load_and_change_output(checkerboard_image, camera_id):
    camera = get_source_camera(camera_id)
    assert camera.id == camera_id

    baseline = _run(
        "vintage_camera_profile", checkerboard_image,
        source_camera="unknown", profile="kodachrome_64", strength=1.0, seed=0,
    )
    with_source = _run(
        "vintage_camera_profile", checkerboard_image,
        source_camera=camera_id, profile="kodachrome_64", strength=1.0, seed=0,
    )
    assert not np.array_equal(baseline, with_source)


def test_extract_camera_info_no_exif():
    png_bytes = save_image(np.zeros((4, 4, 3), dtype=np.uint8), fmt="PNG")
    make, model = extract_camera_info(png_bytes)
    assert (make, model) == (None, None)


def test_extract_camera_info_invalid_bytes():
    make, model = extract_camera_info(b"not an image")
    assert (make, model) == (None, None)
