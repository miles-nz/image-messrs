import numpy as np

from imagemessrs.core.io_utils import load_image, resize_max_edge, save_image


def test_save_load_round_trip_png(gradient_image):
    data = save_image(gradient_image, fmt="PNG")
    loaded = load_image(data)
    assert loaded.shape == gradient_image.shape
    assert np.array_equal(loaded, gradient_image)


def test_resize_max_edge_downscales(checkerboard_image):
    resized = resize_max_edge(checkerboard_image, max_edge=16)
    assert max(resized.shape[:2]) == 16


def test_resize_max_edge_noop_when_smaller(checkerboard_image):
    resized = resize_max_edge(checkerboard_image, max_edge=1000)
    assert resized.shape == checkerboard_image.shape
