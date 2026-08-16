from __future__ import annotations

import cv2
import numpy as np
from numba import njit

from ..core.registry import register_effect
from ..core.types import ImageArray, ParamSpec

_PROTECT_BIAS = 1e6


def _grayscale(image: np.ndarray) -> np.ndarray:
    return 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]


def _sobel_energy(image: np.ndarray) -> np.ndarray:
    gray = _grayscale(image).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt(gx**2 + gy**2)


@njit(cache=True)
def _dp_forward_energy_table(cost_left, cost_up, cost_right, row0):
    """Row-sequential DP recurrence for forward energy, JIT-compiled since it's
    inherently sequential (row y depends on row y-1) and can't be vectorized
    across rows with plain numpy. Returns the cumulative-cost table and, per
    cell, which of left(0)/up(1)/right(2) produced the minimum - same
    add-then-compare order and first-option-wins tie-break as the original
    numpy loop (np.argmin over [left, up, right] stacked in that order)."""
    h, w = cost_up.shape
    cum = np.empty((h, w), dtype=np.float64)
    choice = np.zeros((h, w), dtype=np.int64)
    for x in range(w):
        cum[0, x] = row0[x]
    for y in range(1, h):
        for x in range(w):
            prev_left = np.inf if x == 0 else cum[y - 1, x - 1]
            prev_up = cum[y - 1, x]
            prev_right = np.inf if x == w - 1 else cum[y - 1, x + 1]
            opt_left = prev_left + cost_left[y, x]
            opt_up = prev_up + cost_up[y, x]
            opt_right = prev_right + cost_right[y, x]
            best = opt_left
            c = 0
            if opt_up < best:
                best = opt_up
                c = 1
            if opt_right < best:
                best = opt_right
                c = 2
            cum[y, x] = best
            choice[y, x] = c
    return cum, choice


def _forward_energy(image: np.ndarray) -> np.ndarray:
    """Approximation of Avidan & Shamir forward energy: cost a seam step would
    introduce, used as a per-pixel energy map fed into the same DP seam finder
    as Sobel energy (rather than folding the cost directly into the DP)."""
    gray = _grayscale(image).astype(np.float64)
    up = np.roll(gray, 1, axis=0)
    left = np.roll(gray, 1, axis=1)
    right = np.roll(gray, -1, axis=1)

    cost_up = np.abs(right - left)
    cost_left = cost_up + np.abs(up - left)
    cost_right = cost_up + np.abs(up - right)

    _, choice = _dp_forward_energy_table(cost_left, cost_up, cost_right, cost_up[0])

    stacked_costs = np.stack([cost_left, cost_up, cost_right], axis=0)
    step_energy = np.take_along_axis(stacked_costs, choice[None, :, :], axis=0)[0]
    step_energy[0] = cost_up[0]
    return step_energy


_ENERGY_FUNCS = {"sobel": _sobel_energy, "forward": _forward_energy}


@njit(cache=True)
def _dp_seam_cost_table(energy):
    """Row-sequential seam-cost DP, JIT-compiled for the same reason as
    _dp_forward_energy_table above. Compares raw predecessor costs (left/up/
    right) first and adds this row's energy only after picking the minimum -
    same operation order as the original numpy loop - so results are
    bit-identical, not just numerically close. backtrack values are -1/0/+1,
    the relative x offset into row y-1."""
    h, w = energy.shape
    cost = np.empty((h, w), dtype=np.float64)
    backtrack = np.zeros((h, w), dtype=np.int64)
    for x in range(w):
        cost[0, x] = energy[0, x]
    for y in range(1, h):
        for x in range(w):
            left_val = np.inf if x == 0 else cost[y - 1, x - 1]
            up_val = cost[y - 1, x]
            right_val = np.inf if x == w - 1 else cost[y - 1, x + 1]
            best = left_val
            choice = 0
            if up_val < best:
                best = up_val
                choice = 1
            if right_val < best:
                best = right_val
                choice = 2
            cost[y, x] = energy[y, x] + best
            backtrack[y, x] = choice - 1
    return cost, backtrack


def _find_vertical_seam(energy: np.ndarray) -> np.ndarray:
    h, w = energy.shape
    cost, backtrack = _dp_seam_cost_table(energy)

    seam = np.zeros(h, dtype=np.int64)
    x = int(np.argmin(cost[-1]))
    seam[-1] = x
    for y in range(h - 2, -1, -1):
        x = int(np.clip(x + backtrack[y + 1, x], 0, w - 1))
        seam[y] = x
    return seam


@njit(cache=True)
def _backtrack_all_columns(backtrack: np.ndarray) -> np.ndarray:
    """Backtrack from every possible last-row starting column at once, in one
    jitted O(h*w) pass. seam_matrix[:, x0] is the exact seam _find_vertical_seam
    would produce if it started at column x0 in the last row - same clip/index
    arithmetic as its scalar backtrack loop, just computed for all columns
    together so batch selection below never needs a per-candidate Python loop."""
    h, w = backtrack.shape
    seam_matrix = np.empty((h, w), dtype=np.int64)
    for x0 in range(w):
        seam_matrix[h - 1, x0] = x0
    for y in range(h - 2, -1, -1):
        for x0 in range(w):
            prev_x = seam_matrix[y + 1, x0]
            nx = prev_x + backtrack[y + 1, prev_x]
            if nx < 0:
                nx = 0
            elif nx > w - 1:
                nx = w - 1
            seam_matrix[y, x0] = nx
    return seam_matrix


_MIN_SEAM_SPACING = 2  # exclusion zone (in columns) kept between seams accepted into the same batch


@njit(cache=True)
def _select_seam_batch(
    cost_last_row: np.ndarray, seam_matrix: np.ndarray, k: int, min_spacing: int = _MIN_SEAM_SPACING
) -> np.ndarray:
    """Pick up to k non-overlapping seams out of one DP pass instead of paying
    for a fresh DP pass per seam. Repeatedly takes the lowest-cost remaining
    candidate (one per last-row starting column; ties toward the lower column,
    matching np.argmin's leftmost-wins behavior since this scan uses strict
    '<'), accepting it only if it doesn't fall within min_spacing columns of an
    already-accepted seam in any shared row - each candidate column is
    considered at most once (accepted or rejected), so this can't loop forever.
    The single cheapest candidate is always accepted regardless of spacing, so
    this always returns at least one seam. At k=1 this returns exactly the same
    seam _find_vertical_seam would - same tie-break, same backtrack - so
    energy_refresh_interval=1 (the default) is unaffected by batching."""
    h, w = seam_matrix.shape
    available = np.ones(w, dtype=np.bool_)
    occupied = np.zeros((h, w), dtype=np.bool_)
    accepted = np.empty((k, h), dtype=np.int64)
    n_accepted = 0
    while n_accepted < k:
        best_val = np.inf
        best_x0 = -1
        for x0 in range(w):
            if available[x0] and cost_last_row[x0] < best_val:
                best_val = cost_last_row[x0]
                best_x0 = x0
        if best_x0 == -1:
            break
        available[best_x0] = False

        collides = False
        for y in range(h):
            xc = seam_matrix[y, best_x0]
            lo = max(xc - (min_spacing - 1), 0)
            hi = min(xc + (min_spacing - 1), w - 1)
            for xx in range(lo, hi + 1):
                if occupied[y, xx]:
                    collides = True
                    break
            if collides:
                break
        if collides:
            continue

        for y in range(h):
            xc = seam_matrix[y, best_x0]
            accepted[n_accepted, y] = xc
            lo = max(xc - (min_spacing - 1), 0)
            hi = min(xc + (min_spacing - 1), w - 1)
            for xx in range(lo, hi + 1):
                occupied[y, xx] = True
        n_accepted += 1

    return accepted[:n_accepted]


def _remove_seams(array: np.ndarray, seams: np.ndarray) -> np.ndarray:
    """Remove k seams (shape (k, h)) from array in one vectorized pass."""
    h, w = array.shape[:2]
    k = seams.shape[0]
    keep = np.ones((h, w), dtype=bool)
    rows = np.repeat(np.arange(h), k)
    cols = seams.T.reshape(-1)
    keep[rows, cols] = False
    new_w = w - k
    if array.ndim == 2:
        return array[keep].reshape(h, new_w)
    return array[keep].reshape(h, new_w, array.shape[2])


def _insert_seams(array: np.ndarray, seams: np.ndarray) -> np.ndarray:
    """Insert k seams (shape (k, h)) into array in one vectorized pass. Each
    inserted pixel is the average of its original left neighbor and itself,
    same as the single-seam case; multiple insertions in the same row are
    resolved simultaneously against the original (pre-batch) column indices,
    which is why _select_seam_batch enforces spacing between accepted seams."""
    h, w = array.shape[:2]
    k = seams.shape[0]
    cols_sorted = np.sort(seams.T, axis=1)  # (h, k), ascending per row
    ranks = np.arange(k)[None, :]
    insert_pos = cols_sorted + ranks + 1  # (h, k)

    orig_cols = np.arange(w)[None, :]
    rank_before = (cols_sorted[:, None, :] < orig_cols[:, :, None]).sum(axis=2)  # (h, w)
    orig_new_pos = orig_cols + rank_before  # (h, w)

    new_w = w + k
    out_shape = (h, new_w) if array.ndim == 2 else (h, new_w, array.shape[2])
    out = np.empty(out_shape, dtype=array.dtype)
    rows_idx = np.arange(h)[:, None]
    out[rows_idx, orig_new_pos] = array

    left_cols = np.clip(cols_sorted - 1, 0, w - 1)
    left_vals = array[rows_idx, left_cols].astype(np.float64)
    cur_vals = array[rows_idx, cols_sorted].astype(np.float64)
    avg_vals = ((left_vals + cur_vals) / 2).astype(array.dtype)
    out[rows_idx, insert_pos] = avg_vals
    return out


@register_effect(
    name="seam_carve",
    label="Seam Carving",
    category="seam_carve",
    accepts_mask=True,
    description="Removes or duplicates low-energy seams to reshape the image while mostly protecting visually important content - content-aware resizing, or pushed further, deliberate warped/melty distortion.",
    about={
        "what": "Repeatedly finds and removes (or duplicates) the lowest-energy seam - a connected path of pixels running top-to-bottom or left-to-right - to reshape the image while trying to preserve visually important content.",
        "how_to_use": "Pick an axis and whether to Shrink or Enlarge, then push Seam Count up gradually; small counts do a clean content-aware resize, while pushing well past the image's own width/height forces increasingly warped, melty distortion. Paint a mask to protect a subject (like a face) from being carved through.",
        "used_for": "Content-aware resizing that avoids naive cropping or stretching - and, pushed to extremes, a deliberate “melting”/liquid distortion effect.",
        "examples": "Seam carving was introduced in the 2007 SIGGRAPH paper “Seam Carving for Content-Aware Image Resizing” by Shai Avidan and Ariel Shamir, and later shipped in Photoshop as “Content-Aware Scale.” Artists have since pushed the same algorithm past its intended use, exploiting it as a glitch technique to melt and warp images instead of resizing them cleanly.",
    },
    params=[
        ParamSpec(
            name="axis",
            kind="choice",
            default="vertical",
            choices=["vertical", "horizontal"],
            label="Axis (vertical=width, horizontal=height)",
            description="Vertical seams run top-to-bottom and are removed/added to change the image's width. Horizontal seams run left-to-right and change its height.",
        ),
        ParamSpec(
            name="mode", kind="choice", default="shrink", choices=["shrink", "enlarge"],
            description="Shrink removes seams to make the image smaller. Enlarge duplicates seams to make it bigger - pushed far, this stretches and smears content.",
        ),
        ParamSpec(
            name="seam_count", kind="int", default=50, min=1, max=2000, step=1, label="Seam Count",
            description="How many seams to remove or insert. Small values do a subtle content-aware resize; pushing this well past the image's width/height forces increasingly warped, melty results as an intended glitch effect.",
        ),
        ParamSpec(
            name="energy_mode", kind="choice", default="sobel", choices=["sobel", "forward"],
            description="Sobel is a fast edge-detection energy map - cheap, and its imperfections are part of the glitch look. Forward energy accounts for the cost each removal introduces, giving cleaner resizing with fewer jagged artifacts.",
        ),
        ParamSpec(
            name="energy_refresh_interval",
            kind="int",
            default=1,
            min=1,
            max=50,
            label="Energy Refresh Interval (higher = melty/overdrive)",
            description="How often the energy map is recalculated from the actual pixels, in seams. 1 recalculates every time (technically correct). Higher values reuse a stale, outdated map for several seams in a row - that staleness is what produces the warped/melty overdrive look.",
        ),
        ParamSpec(
            name="protect_mask", kind="mask", default=None, label="Protected Region",
            description="Paint the region seams should avoid cutting through (e.g. a face or subject). Leave blank to let seams go anywhere.",
        ),
    ],
)
def apply(
    image: ImageArray,
    axis: str = "vertical",
    mode: str = "shrink",
    seam_count: int = 50,
    energy_mode: str = "sobel",
    energy_refresh_interval: int = 1,
    protect_mask: np.ndarray | None = None,
) -> ImageArray:
    horizontal = axis == "horizontal"
    work = np.transpose(image, (1, 0, 2)).copy() if horizontal else image.copy()

    mask = None
    if protect_mask is not None:
        mask = np.transpose(protect_mask) if horizontal else protect_mask
        mask = mask.astype(bool)

    energy_fn = _ENERGY_FUNCS[energy_mode]
    refresh_interval = max(1, int(energy_refresh_interval))
    seam_count = max(1, int(seam_count))

    seams_done = 0
    while seams_done < seam_count:
        width = work.shape[1]
        if mode == "shrink" and width <= 2:
            break

        # One DP pass covers a whole "refresh window" - up to refresh_interval
        # seams are extracted from it below, instead of one DP pass per seam.
        energy = energy_fn(work)
        if mask is not None:
            energy = energy + mask.astype(np.float64) * _PROTECT_BIAS

        batch_k = min(refresh_interval, seam_count - seams_done)
        if mode == "shrink":
            batch_k = min(batch_k, width - 2)  # never remove past the width>2 floor
            if batch_k <= 0:
                break

        cost, backtrack = _dp_seam_cost_table(energy)
        seam_matrix = _backtrack_all_columns(backtrack)
        seams = _select_seam_batch(cost[-1], seam_matrix, batch_k)

        if mode == "shrink":
            work = _remove_seams(work, seams)
            if mask is not None:
                mask = _remove_seams(mask, seams)
        else:
            work = _insert_seams(work, seams)
            if mask is not None:
                mask = _insert_seams(mask, seams)

        seams_done += seams.shape[0]

    return np.transpose(work, (1, 0, 2)).copy() if horizontal else work
