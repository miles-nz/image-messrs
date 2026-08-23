from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from ...core.registry import register_effect
from ...core.types import ImageArray, ParamSpec
from .profile_types import (
    ChromaticAberration,
    DynamicRange,
    Grain,
    Halation,
    Saturation,
    SensorNoise,
    SourceCorrection,
    SplitToning,
    ToneCurve,
    Vignette,
)
from .profiles import PROFILES, get_profile
from .source_cameras import SOURCE_CAMERAS, get_source_camera


def _apply_tone_curve(img: np.ndarray, curve: ToneCurve) -> np.ndarray:
    out = img.copy()
    for i, pts in enumerate((curve.r, curve.g, curve.b)):
        xp, fp = zip(*pts)
        out[..., i] = np.interp(img[..., i], xp, fp)
    return out


# Named skin-tone hue bands, each a (center, sigma) pair in OpenCV's 0-179 hue
# space. These approximate where different skin tones cluster on the hue
# wheel - real skin varies mostly in value/saturation rather than hue, so
# treat these as a practical starting point to pick from and fine-tune via
# Skin Tone Protection strength, not a precise colorimetric match.
_SKIN_TONE_PRESETS: dict[str, tuple[float, float]] = {
    "reddish_pink": (6.0, 12.0),
    "peachy_tan": (13.0, 13.0),
    "golden_olive": (20.0, 14.0),
    "deep_brown": (15.0, 17.0),
}
_DEFAULT_SKIN_TONE = "reddish_pink"


def _skin_tone_weight(hue: np.ndarray, center: float, sigma: float) -> np.ndarray:
    """Per-pixel 0-1 "how skin-tone-like is this hue" weight, via a Gaussian
    bump on the circular hue wheel (handles wraparound near hue 0/179)."""
    diff = np.abs(hue - center)
    diff = np.minimum(diff, 180 - diff)
    return np.exp(-0.5 * (diff / sigma) ** 2)


def _apply_saturation(
    img: np.ndarray, sat: Saturation, skin_protect: float = 0.0, skin_tone: str = _DEFAULT_SKIN_TONE,
) -> np.ndarray:
    if sat.global_mult == 1.0 and sat.curve is None and skin_protect <= 0:
        return img
    hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    mult = sat.global_mult
    if sat.curve:
        xp, fp = zip(*sat.curve)
        mult = mult * np.interp(hsv[..., 2], xp, fp)
    if skin_protect > 0:
        # Pull the multiplier back toward 1 (no change) for skin-tone hues,
        # proportional to protection strength - so a punchy profile's
        # saturation boost lands on the rest of the image but leaves skin
        # closer to its original saturation instead of pushing it redder.
        center, sigma = _SKIN_TONE_PRESETS.get(skin_tone, _SKIN_TONE_PRESETS[_DEFAULT_SKIN_TONE])
        weight = _skin_tone_weight(hsv[..., 0], center, sigma)
        mult = 1 + (mult - 1) * (1 - skin_protect * weight)
    hsv[..., 1] = np.clip(hsv[..., 1] * mult, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)


def _apply_source_correction(img: np.ndarray, corr: SourceCorrection, rng: np.random.Generator) -> np.ndarray:
    """Corrects for the *source* camera's own processing (sharpening halos, HDR
    tone-mapping, noise reduction, saturation boost) before the target vintage
    look is layered on top, so the grade lands on a more neutral starting point."""
    out = img
    if corr.undo_sharpen > 0:
        blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=1.5)
        out = out * (1 - corr.undo_sharpen) + blurred * corr.undo_sharpen
    if corr.undo_hdr > 0:
        out = 128 + (out - 128) * (1 - 0.5 * corr.undo_hdr)
    if corr.undo_noise_reduction > 0:
        h, w = out.shape[:2]
        texture = rng.normal(0, 1, size=(h, w)).astype(np.float32)
        out = out + (texture * corr.undo_noise_reduction * 255 * 0.03)[..., None]
    if corr.undo_saturation > 0:
        out = _apply_saturation(out, Saturation(global_mult=1 - corr.undo_saturation))
    return out


def _apply_split_toning(img: np.ndarray, st: SplitToning) -> np.ndarray:
    if st.shadow_strength == 0 and st.highlight_strength == 0:
        return img
    luma = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]) / 255.0
    shadow_w = (1 - luma) * st.shadow_strength
    highlight_w = luma * st.highlight_strength
    out = img.copy()
    for i in range(3):
        out[..., i] += shadow_w * (st.shadow_color[i] - img[..., i])
        out[..., i] += highlight_w * (st.highlight_color[i] - img[..., i])
    return out


def _apply_vignette(img: np.ndarray, vg: Vignette) -> np.ndarray:
    if vg.strength <= 0:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2, w / 2
    r = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2) / np.sqrt(2)
    falloff = np.clip((r - vg.radius) / max(1e-6, 1 - vg.radius), 0, 1) ** vg.falloff
    mult = 1 - vg.strength * falloff
    return img * mult[..., None]


def _apply_radial_chromatic_aberration(img: np.ndarray, ca: ChromaticAberration) -> np.ndarray:
    """Radial lens-style fringing, distinct from channel_shift.py's flat pixel offset:
    displacement grows with distance from the image center, per channel."""
    if ca.intensity <= 0:
        return img
    h, w = img.shape[:2]
    cy, cx = h / 2, w / 2
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    src_u8 = np.clip(img, 0, 255).astype(np.uint8)
    out = np.empty_like(src_u8)
    # Red scaled outward, blue scaled inward, green untouched - classic lateral CA fringe.
    channel_factors = (ca.intensity * 0.02, 0.0, -ca.intensity * 0.02)
    for i, factor in enumerate(channel_factors):
        map_x = (cx + dx * (1 + factor)).astype(np.float32)
        map_y = (cy + dy * (1 + factor)).astype(np.float32)
        out[..., i] = cv2.remap(src_u8[..., i], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.astype(np.float32)


def _apply_halation(img: np.ndarray, ha: Halation) -> np.ndarray:
    if ha.intensity <= 0:
        return img
    luma = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    mask = np.clip(luma / 255.0 - ha.threshold, 0, None) / max(1e-6, 1 - ha.threshold)
    bright = img * mask[..., None]
    bloom = cv2.GaussianBlur(bright, (0, 0), sigmaX=ha.radius)
    tint = np.array([1.0, 0.55, 0.3], dtype=np.float32)  # warm red/orange halation glow
    return img + bloom * tint * ha.intensity


def _apply_grain(img: np.ndarray, gr: Grain, rng: np.random.Generator) -> np.ndarray:
    if gr.intensity <= 0:
        return img
    h, w = img.shape[:2]
    noise = rng.normal(0, 1, size=(h, w)).astype(np.float32)
    if gr.size > 0:
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=gr.size)
        noise /= max(float(noise.std()), 1e-6)
    return img + (noise * gr.intensity * 255 * 0.15)[..., None]


def _apply_sensor_noise(img: np.ndarray, sn: SensorNoise, rng: np.random.Generator) -> np.ndarray:
    if sn.intensity <= 0 and sn.chroma <= 0:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    if sn.intensity > 0:
        luma_noise = rng.normal(0, sn.intensity * 255 * 0.2, size=(h, w)).astype(np.float32)
        out += luma_noise[..., None]
    if sn.chroma > 0:
        chroma_noise = rng.normal(0, sn.chroma * 255 * 0.2, size=(h, w, 3)).astype(np.float32)
        chroma_noise = cv2.GaussianBlur(chroma_noise, (0, 0), sigmaX=2.0)
        out += chroma_noise
    return out


def _apply_dynamic_range(img: np.ndarray, dr: DynamicRange) -> np.ndarray:
    if dr.black_point <= 0 and dr.white_point >= 255:
        return img
    span = max(1e-6, dr.white_point - dr.black_point)
    return (img - dr.black_point) / span * 255


def _jpeg_roundtrip(img_u8: np.ndarray, quality: int) -> np.ndarray:
    buf = BytesIO()
    Image.fromarray(img_u8).save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.uint8)


_REAL_PROFILE_CHOICES = sorted(PROFILES.keys())
# "none" (no target look, source correction only) is always last and never the
# default - it's an opt-in preview mode, not a starting point.
_PROFILE_CHOICES = _REAL_PROFILE_CHOICES + ["none"] if _REAL_PROFILE_CHOICES else ["none"]
_DEFAULT_PROFILE = _PROFILE_CHOICES[0]

_SOURCE_CHOICES = sorted(SOURCE_CAMERAS.keys()) or ["unknown"]
_DEFAULT_SOURCE = "unknown" if "unknown" in SOURCE_CAMERAS else _SOURCE_CHOICES[0]


@register_effect(
    name="vintage_camera_profile",
    label="Vintage Camera Profile",
    category="color",
    description="Emulates how a specific vintage camera or film stock would have rendered this photo: color science, lens artifacts, grain/halation, and (for early digital cameras) sensor noise and JPEG compression character.",
    about={
        "what": "Two independent camera choices: Source Camera corrects for what actually took this digital photo (e.g. undoing a phone's oversharpening, HDR tone-mapping, noise reduction, and saturation boost), and Camera / Film applies the target vintage look - a tone curve, color-cast character, lens vignette and chromatic aberration, film grain and halation, and (for digital-era profiles) sensor noise, dynamic-range clipping, and JPEG compression character.",
        "how_to_use": "If you know what device actually shot the photo, set Source Camera first (auto-suggested from EXIF when available) so the vintage grade starts from a more neutral base. Then pick a Camera / Film target look and dial Intensity to blend between the untouched original and the full effect. Adjust Grain Seed to get a different, still-reproducible grain/noise/texture pattern.",
        "used_for": "Recreating the specific look of a vintage camera or film stock, with more authentic results when the source device's own computational processing is corrected for first rather than graded directly on top of it.",
        "examples": "An iPhone photo has Smart HDR and sharpening baked in; setting Source Camera to iPhone flattens and softens that before Kodachrome's punchy reds and cool shadows, Portra's soft skin tones, a Polaroid SX-70's warm vignette, or a Holga's light leaks are applied. If a punchy profile pushes your skin tone into an unnaturally oversaturated look, pick the closest Skin Tone and raise Skin Tone Protection to hold that hue range back from the saturation boost.",
    },
    params=[
        ParamSpec(
            name="source_camera", kind="choice", default=_DEFAULT_SOURCE, choices=_SOURCE_CHOICES,
            label="Source Camera",
            description="What actually took this photo - corrects for that device's own processing (sharpening, HDR, noise reduction, saturation) before the vintage look is applied.",
        ),
        ParamSpec(
            name="profile", kind="choice", default=_DEFAULT_PROFILE, choices=_PROFILE_CHOICES,
            label="Camera / Film", description="Which curated camera/film look to emulate.",
        ),
        ParamSpec(
            name="strength", kind="float", default=1.0, min=0.0, max=1.0, step=0.01,
            label="Intensity",
            description="Blends the emulated look with the original photo; 0 leaves it unchanged, 1 is the full effect.",
        ),
        ParamSpec(
            name="skin_tone", kind="choice", default=_DEFAULT_SKIN_TONE, choices=list(_SKIN_TONE_PRESETS.keys()),
            label="Skin Tone",
            description="Which hue range Skin Tone Protection should target. Only matters when Skin Tone Protection is above 0.",
        ),
        ParamSpec(
            name="skin_tone_protection", kind="float", default=0.0, min=0.0, max=1.0, step=0.05,
            label="Skin Tone Protection",
            description="Holds the chosen Skin Tone's hue range back from the profile's saturation change, so punchy profiles don't push skin into an oversaturated, unnatural look. 0 = off, 1 = that hue's saturation left untouched.",
        ),
        ParamSpec(
            name="seed", kind="int", default=0, min=0, max=999999, step=1,
            label="Grain Seed",
            description="Random seed for grain/sensor/texture noise; the same seed reproduces the same texture.",
        ),
    ],
)
def apply(
    image: ImageArray,
    source_camera: str = _DEFAULT_SOURCE,
    profile: str = _DEFAULT_PROFILE,
    strength: float = 1.0,
    skin_tone: str = _DEFAULT_SKIN_TONE,
    skin_tone_protection: float = 0.0,
    seed: int = 0,
) -> ImageArray:
    if strength <= 0:
        return image.copy()

    src = get_source_camera(source_camera)
    prof = get_profile(profile)
    rng = np.random.default_rng(int(seed))
    out = image.astype(np.float32)

    out = _apply_source_correction(out, src.correction, rng)
    out = _apply_radial_chromatic_aberration(out, prof.chromatic_aberration)
    out = _apply_vignette(out, prof.vignette)
    out = _apply_tone_curve(out, prof.tone_curve)
    out = _apply_saturation(out, prof.saturation, skin_protect=skin_tone_protection, skin_tone=skin_tone)
    out = _apply_split_toning(out, prof.split_toning)
    out = _apply_halation(out, prof.halation)
    out = _apply_grain(out, prof.grain, rng)
    out = _apply_sensor_noise(out, prof.sensor_noise, rng)
    out = _apply_dynamic_range(out, prof.dynamic_range)
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)

    if prof.jpeg_quality is not None:
        out_u8 = _jpeg_roundtrip(out_u8, prof.jpeg_quality)

    if strength < 1.0:
        blended = image.astype(np.float32) * (1 - strength) + out_u8.astype(np.float32) * strength
        out_u8 = np.clip(blended, 0, 255).astype(np.uint8)

    return out_u8
