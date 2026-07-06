"""Tests for the perspective (bird's-eye) transform."""

import numpy as np

from lane_detector.perspective import PerspectiveTransform


def test_warp_preserves_shape(frame_size, blank_frame):
    pt = PerspectiveTransform(frame_size)
    warped = pt.warp(blank_frame)
    assert warped.shape == blank_frame.shape


def test_warp_unwarp_round_trip_is_close(frame_size):
    """Unwarping a warped image should approximately recover the original."""
    import cv2

    pt = PerspectiveTransform(frame_size)
    w, h = frame_size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.rectangle(img, (int(w * 0.3), int(h * 0.7)), (int(w * 0.7), h), (255, 255, 255), -1)

    recovered = pt.unwarp(pt.warp(img))
    # compare only the lower-central road region, which the transform covers
    y0, y1 = int(h * 0.75), h
    x0, x1 = int(w * 0.35), int(w * 0.65)
    a = img[y0:y1, x0:x1].astype(float)
    b = recovered[y0:y1, x0:x1].astype(float)
    mae = np.abs(a - b).mean()
    assert mae < 70  # lenient: perspective interpolation loss is expected


def test_matrices_are_inverses(frame_size):
    pt = PerspectiveTransform(frame_size)
    product = pt.M @ pt.M_inv
    identity = np.eye(3)
    # perspective inverse carries ~1e-5 numerical error; normalise then compare
    assert np.allclose(product / product[2, 2], identity, atol=1e-3)


def test_draw_src_region_does_not_mutate_input(frame_size, blank_frame):
    pt = PerspectiveTransform(frame_size)
    before = blank_frame.copy()
    pt.draw_src_region(blank_frame)
    assert np.array_equal(before, blank_frame)
