from __future__ import annotations

import numpy as np

from ...core.registry import register_effect
from ...core.types import ImageArray, ParamSpec

_MAX_DUST = 8000
_DUST_PATCH_MIN_RADIUS = 2
_DUST_PATCH_MAX_RADIUS = 26  # must stay >= ~3 * the "dust_size" param's max so the biggest specks aren't clipped
# The long-tailed speck-size distribution (see `apply`) is defined relative to
# the "dust_size" param rather than as fixed sigmas, so the whole distribution
# scales with the slider instead of just its ceiling.
_DUST_SIGMA_MIN_RATIO = 0.125
_DUST_SIGMA_EXP_SCALE_RATIO = 0.125
_SCRATCH_HALF_WIDTH = 0.35
_SCRATCH_PATCH_PAD = 2

# Downscaling by any amount to make a preview softens fine features roughly
# one antialiased preview-pixel's worth, independent of exactly how much
# downscaling happens - simple size/intensity scaling alone (see
# `resolution_scale` below) shrinks and dims specks proportionally to the
# preview's scale, but real downsampling also blurs their edges, which
# suppresses their peak brightness far more than proportional dimming does.
# Calibrated against a full-resolution bake downsized for display; ramps to
# 0 as resolution_scale -> 1 (no downscaling, i.e. the full-res pass itself).
_PREVIEW_ANTIALIAS_BLUR = 0.4


def _splat_dust(
    h: int,
    w: int,
    xs: np.ndarray,
    ys: np.ndarray,
    sigmas: np.ndarray,
    vals: np.ndarray,
    dot_alphas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Stamp true Gaussian blobs for each dust speck instead of rasterized circles.

    cv2.circle's antialiasing at radius ~1px renders specks as small diamonds
    or plus-shapes rather than round dots, however the shape is softened
    afterwards. Computing an actual Gaussian falloff per speck and
    max-compositing overlapping specks (so one occludes rather than blends
    into another) gives genuinely soft, round, irregular-looking specks.
    """
    alpha_layer = np.zeros((h, w), dtype=np.float32)
    val_layer = np.zeros((h, w), dtype=np.float32)
    if sigmas.size == 0:
        return alpha_layer, val_layer
    # Window sized to the largest speck present so its Gaussian tail isn't
    # clipped into a hard-edged circle - most specks are far smaller than
    # this and just use the inner part of the same window.
    k = int(np.clip(np.ceil(3.0 * float(sigmas.max())), _DUST_PATCH_MIN_RADIUS, _DUST_PATCH_MAX_RADIUS))
    grid = np.arange(-k, k + 1)
    gy, gx = np.meshgrid(grid, grid, indexing="ij")
    for x, y, sigma, val, da in zip(xs, ys, sigmas, vals, dot_alphas):
        cx, cy = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - cx, y - cy
        sx0, sx1 = max(cx - k, 0), min(cx + k + 1, w)
        sy0, sy1 = max(cy - k, 0), min(cy + k + 1, h)
        if sx0 >= sx1 or sy0 >= sy1:
            continue
        lx0, lx1 = sx0 - (cx - k), sx1 - (cx - k)
        ly0, ly1 = sy0 - (cy - k), sy1 - (cy - k)
        dx = gx[ly0:ly1, lx0:lx1] - fx
        dy = gy[ly0:ly1, lx0:lx1] - fy
        patch = np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma)) * da
        region = alpha_layer[sy0:sy1, sx0:sx1]
        mask = patch > region
        alpha_layer[sy0:sy1, sx0:sx1] = np.where(mask, patch, region)
        val_layer[sy0:sy1, sx0:sx1] = np.where(mask, val, val_layer[sy0:sy1, sx0:sx1])
    return alpha_layer, val_layer


def _splat_segment(
    alpha_layer: np.ndarray,
    val_layer: np.ndarray,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    half_width: float,
    val: float,
    seg_alpha: float,
    h: int,
    w: int,
) -> None:
    """Stamp one scratch segment as an analytic, sub-pixel-wide line.

    cv2.line can't draw thinner than 1px, which reads as noticeably wide on
    a typical preview-sized image. Scoring each pixel by its perpendicular
    distance to the segment (clamped to its endpoints) and falling off with
    a sub-pixel Gaussian gives a properly anti-aliased hairline instead.
    """
    pad = _SCRATCH_PATCH_PAD
    sx0 = max(int(np.floor(min(x0, x1))) - pad, 0)
    sx1 = min(int(np.ceil(max(x0, x1))) + pad + 1, w)
    sy0 = max(int(np.floor(min(y0, y1))) - pad, 0)
    sy1 = min(int(np.ceil(max(y0, y1))) + pad + 1, h)
    if sx0 >= sx1 or sy0 >= sy1:
        return
    px, py = np.meshgrid(np.arange(sx0, sx1), np.arange(sy0, sy1))
    dx, dy = x1 - x0, y1 - y0
    length2 = dx * dx + dy * dy
    if length2 < 1e-9:
        t = np.zeros(px.shape, dtype=np.float32)
    else:
        t = np.clip(((px - x0) * dx + (py - y0) * dy) / length2, 0.0, 1.0)
    dist2 = (px - (x0 + t * dx)) ** 2 + (py - (y0 + t * dy)) ** 2
    patch = np.exp(-dist2 / (2 * half_width * half_width)) * seg_alpha
    region = alpha_layer[sy0:sy1, sx0:sx1]
    mask = patch > region
    alpha_layer[sy0:sy1, sx0:sx1] = np.where(mask, patch, region)
    val_layer[sy0:sy1, sx0:sx1] = np.where(mask, val, val_layer[sy0:sy1, sx0:sx1])


@register_effect(
    name="dust_scratches",
    label="Dust & Scratches",
    category="color",
    description="Overlays procedural dust specks and vertical scratch lines, the way damaged or poorly stored film looks when scanned - independent of any vintage camera profile.",
    about={
        "what": "Scatters small bright and dark specks across the frame to simulate dust and debris, and draws a handful of thin, slightly wavering vertical lines to simulate scratches - both blended on top of the image rather than baked into a specific film profile.",
        "how_to_use": "Raise Dust Density for more specks, Dust Size for how big they can get, and Dust Intensity for how visible they are. Raise Scratch Count for more scratch lines and Scratch Intensity for how visible they are. Adjust Seed to get a different, still-reproducible arrangement of dust and scratches.",
        "used_for": "Adding the physical wear-and-tear look of scanned analog film to a digital image - useful on its own, or layered after a vintage camera profile for extra grunge.",
        "examples": "Dust and scratches are the most common defects in scanned film, caused by debris on the film surface and physical abrasion from repeated handling or aging - restoration tools like Kodak's Digital ICE exist specifically to remove them, while this effect deliberately adds them back for an authentically worn analog look.",
    },
    params=[
        ParamSpec(
            name="dust_density", kind="float", default=0.002, min=0.0, max=0.1, step=0.0005, label="Dust Density",
            description="Fraction of the frame covered by dust specks. 0 disables dust entirely.",
        ),
        ParamSpec(
            name="dust_intensity", kind="float", default=0.7, min=0.0, max=1.0, step=0.05, label="Dust Intensity",
            description="How visible the dust specks are.",
        ),
        ParamSpec(
            name="dust_size", kind="float", default=2.5, min=0.0, max=8.0, step=0.1, label="Dust Size",
            description="Ceiling on dust speck radius, in pixels at full resolution. Real dust follows a long-tailed size distribution, so most specks land well under this ceiling with only a shrinking few large ones near it. 0 gives uniformly tiny, barely visible flecks.",
        ),
        ParamSpec(
            name="scratch_count", kind="int", default=3, min=0, max=20, step=1, label="Scratch Count",
            description="Number of vertical scratch lines drawn across the frame. 0 disables scratches entirely.",
        ),
        ParamSpec(
            name="scratch_intensity", kind="float", default=0.5, min=0.0, max=1.0, step=0.05, label="Scratch Intensity",
            description="How visible the scratch lines are.",
        ),
        ParamSpec(
            name="seed", kind="int", default=0, min=0, max=999999, step=1,
            description="Random seed. The same seed with the same other settings always reproduces the exact same dust and scratch pattern.",
        ),
    ],
)
def apply(
    image: ImageArray,
    dust_density: float = 0.002,
    dust_intensity: float = 0.7,
    dust_size: float = 2.5,
    scratch_count: int = 3,
    scratch_intensity: float = 0.5,
    seed: int = 0,
    resolution_scale: float = 1.0,
) -> ImageArray:
    h, w = image.shape[:2]
    rng = np.random.default_rng(int(seed))
    base = image.astype(np.float32)

    # Speck/scratch size and the intensity params are tuned for full-resolution
    # output, same as film_grain's "size"/"intensity" - see the comment there.
    # On a downscaled preview both need scaling by the same factor, or specks
    # and scratches read as far bigger and louder than they'll actually be
    # once baked at full resolution and downsized for display.
    size_scale = float(resolution_scale)
    # Added in quadrature (Gaussian blurs compose that way) so it softens
    # specks/scratches without double-counting the size_scale shrink above.
    extra_blur = _PREVIEW_ANTIALIAS_BLUR * (1.0 - size_scale)

    n_dust = min(_MAX_DUST, int(h * w * max(0.0, float(dust_density))))
    if n_dust > 0 and dust_intensity > 0:
        # Each speck gets its own size and opacity instead of identical
        # hard-edged full-opacity dots - real dust and debris varies in size
        # and how much light it blocks.
        xs = rng.uniform(0, w, size=n_dust)
        ys = rng.uniform(0, h, size=n_dust)
        # Real dust/debris size follows a long-tailed distribution: most
        # specks are fine, film-grain-sized flecks, with a shrinking few
        # larger particles of debris - not one uniform size.
        dust_sigma_max = max(0.0, float(dust_size))
        dust_sigma_min = _DUST_SIGMA_MIN_RATIO * dust_sigma_max
        dust_sigma_exp_scale = max(_DUST_SIGMA_EXP_SCALE_RATIO * dust_sigma_max, 1e-6)
        sigmas = np.clip(
            dust_sigma_min + rng.exponential(scale=dust_sigma_exp_scale, size=n_dust),
            dust_sigma_min, dust_sigma_max,
        ) * size_scale
        # Floor well above 0 - _splat_dust divides by sigma**2, and dust_size=0
        # would otherwise produce an exact 0 here at full resolution (where
        # extra_blur is also 0).
        sigmas = np.sqrt(sigmas**2 + extra_blur**2)
        sigmas = np.maximum(sigmas, 1e-3)
        bright = rng.random(n_dust) < 0.8
        vals = np.where(bright, rng.uniform(190, 255, n_dust), rng.uniform(0, 55, n_dust))
        dot_alphas = rng.uniform(0.35, 1.0, n_dust)
        alpha_f, val_f = _splat_dust(h, w, xs, ys, sigmas, vals, dot_alphas)
        a = (alpha_f * float(dust_intensity) * size_scale)[..., None]
        base = base * (1 - a) + val_f[..., None] * a

    n_scratches = max(0, int(scratch_count))
    if n_scratches > 0 and scratch_intensity > 0:
        # Built from several short segments with random gaps and per-segment opacity,
        # rather than one smooth full-height line - real scratches wander irregularly
        # and rarely stay equally visible along their whole length.
        alpha_layer = np.zeros((h, w), dtype=np.float32)
        val_layer = np.zeros((h, w), dtype=np.float32)
        n_pts = 14
        scratch_half_width = np.sqrt((_SCRATCH_HALF_WIDTH * size_scale) ** 2 + extra_blur**2)
        for _ in range(n_scratches):
            x0 = rng.uniform(0, w)
            val = 255 if rng.random() < 0.75 else 0
            drift = np.clip(np.cumsum(rng.normal(0, 0.9, n_pts)), -0.05 * w, 0.05 * w)
            ys_pts = np.linspace(0, h, n_pts + 1)
            xs_pts = np.concatenate([[x0], x0 + drift])
            keep = rng.random(n_pts) < 0.82
            # A real scratch is rarely fully opaque along its whole length -
            # capping well below 1.0 keeps even "full intensity" from
            # reading as a bold, high-contrast line.
            seg_alphas = rng.uniform(0.2, 0.6, n_pts)
            for i in range(n_pts):
                if not keep[i]:
                    continue
                _splat_segment(
                    alpha_layer, val_layer,
                    xs_pts[i], ys_pts[i], xs_pts[i + 1], ys_pts[i + 1],
                    scratch_half_width, val, seg_alphas[i], h, w,
                )
        a = (alpha_layer * float(scratch_intensity) * size_scale)[..., None]
        base = base * (1 - a) + val_layer[..., None] * a

    return np.clip(base, 0, 255).astype(np.uint8)
