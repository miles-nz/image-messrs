import numpy as np

import imagemessrs.effects  # noqa: F401
from imagemessrs.pipeline import run_pipeline


def _alternating_rows_image() -> np.ndarray:
    """5x5 image where column 2 is constant (128) on every row while every
    other column alternates 0/255 row-to-row. Column 2 is unambiguously the
    lowest-energy column in the interior."""
    rows = []
    for r in range(5):
        rows.append([0, 255, 128, 255, 0] if r % 2 == 0 else [255, 0, 128, 0, 255])
    return np.array([[[v, v, v] for v in row] for row in rows], dtype=np.uint8)


def test_seam_carve_removes_hand_computable_column():
    img = _alternating_rows_image()
    mask = np.zeros((5, 5), dtype=bool)
    mask[:, 0] = True
    mask[:, 4] = True  # protect both border columns, which are 0-energy Sobel artifacts

    result = run_pipeline(
        img,
        [
            (
                "seam_carve",
                {
                    "axis": "vertical",
                    "mode": "shrink",
                    "seam_count": 1,
                    "energy_mode": "sobel",
                    "protect_mask": mask,
                },
            )
        ],
    )
    assert result.shape == (5, 4, 3)
    expected = np.delete(img, 2, axis=1)
    assert np.array_equal(result, expected)


def test_seam_carve_shrink_reduces_width(gradient_image):
    result = run_pipeline(gradient_image, [("seam_carve", {"axis": "vertical", "mode": "shrink", "seam_count": 5})])
    assert result.shape == (gradient_image.shape[0], gradient_image.shape[1] - 5, 3)


def test_seam_carve_shrink_reduces_height(gradient_image):
    result = run_pipeline(gradient_image, [("seam_carve", {"axis": "horizontal", "mode": "shrink", "seam_count": 5})])
    assert result.shape == (gradient_image.shape[0] - 5, gradient_image.shape[1], 3)


def test_seam_carve_enlarge_increases_width(gradient_image):
    result = run_pipeline(gradient_image, [("seam_carve", {"axis": "vertical", "mode": "enlarge", "seam_count": 5})])
    assert result.shape == (gradient_image.shape[0], gradient_image.shape[1] + 5, 3)


def test_seam_carve_overdrive_seam_count_past_width_does_not_crash(checkerboard_image):
    width = checkerboard_image.shape[1]
    result = run_pipeline(
        checkerboard_image,
        [("seam_carve", {"axis": "vertical", "mode": "shrink", "seam_count": width * 2, "energy_refresh_interval": 5})],
    )
    assert result.shape[1] >= 2
    assert result.shape[0] == checkerboard_image.shape[0]


def test_seam_carve_forward_energy_runs(gradient_image):
    result = run_pipeline(
        gradient_image, [("seam_carve", {"axis": "vertical", "mode": "shrink", "seam_count": 3, "energy_mode": "forward"})]
    )
    assert result.shape == (gradient_image.shape[0], gradient_image.shape[1] - 3, 3)


def test_seam_carve_deterministic(gradient_image):
    params = {"axis": "vertical", "mode": "shrink", "seam_count": 4}
    r1 = run_pipeline(gradient_image, [("seam_carve", params)])
    r2 = run_pipeline(gradient_image, [("seam_carve", params)])
    assert np.array_equal(r1, r2)
