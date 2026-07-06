"""Integration tests for the LaneDetector orchestrator."""

import os

import numpy as np
import pytest

from lane_detector import LaneDetector

DEMO_VIDEO = os.path.join(os.path.dirname(os.path.dirname(__file__)), "test3.mp4")


def test_process_returns_same_size_bgr(synthetic_lane_frame, frame_size):
    det = LaneDetector(frame_size=frame_size)
    out = det.process(synthetic_lane_frame)
    assert out.shape == (frame_size[1], frame_size[0], 3)
    assert out.dtype == np.uint8


def test_metrics_schema(synthetic_lane_frame, frame_size):
    det = LaneDetector(frame_size=frame_size)
    det.process(synthetic_lane_frame)
    m = det.metrics()
    assert set(m.keys()) == {"curvature_m", "offset_m", "status", "fps", "confidence"}
    assert 0.0 <= m["confidence"] <= 1.0


@pytest.mark.parametrize("mode", ["final", "threshold", "birdseye", "roi"])
def test_view_modes_render(synthetic_lane_frame, frame_size, mode):
    det = LaneDetector(frame_size=frame_size)
    out = det.process(synthetic_lane_frame, view_mode=mode)
    assert out.shape == (frame_size[1], frame_size[0], 3)


@pytest.mark.skipif(not os.path.exists(DEMO_VIDEO), reason="demo video not present")
def test_high_detection_rate_on_demo_clip(frame_size):
    """The pipeline should lock onto lanes for the large majority of frames."""
    import cv2

    det = LaneDetector(frame_size=frame_size)
    cap = cv2.VideoCapture(DEMO_VIDEO)
    frames, locked = 0, 0
    while frames < 300:
        ret, frame = cap.read()
        if not ret:
            break
        det.process(frame)
        frames += 1
        if det.status not in ("SEARCHING", "INITIALISING"):
            locked += 1
    cap.release()
    assert frames > 0
    assert locked / frames > 0.8  # expect a strong lock rate
