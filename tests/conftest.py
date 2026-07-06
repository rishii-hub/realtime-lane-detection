"""Shared fixtures for the test suite."""

import numpy as np
import pytest

FRAME_SIZE = (640, 360)


@pytest.fixture
def frame_size():
    return FRAME_SIZE


@pytest.fixture
def blank_frame():
    """A black BGR frame."""
    w, h = FRAME_SIZE
    return np.zeros((h, w, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_lane_frame():
    """
    A synthetic road: dark background with two bright near-vertical lane
    markings converging toward the horizon. Deterministic, so tests don't
    depend on the bundled video.
    """
    import cv2

    w, h = FRAME_SIZE
    frame = np.full((h, w, 3), 40, dtype=np.uint8)  # grey tarmac
    # left and right lane lines converging upward
    cv2.line(frame, (int(w * 0.20), h), (int(w * 0.44), int(h * 0.62)), (255, 255, 255), 6)
    cv2.line(frame, (int(w * 0.80), h), (int(w * 0.56), int(h * 0.62)), (255, 255, 255), 6)
    return frame
