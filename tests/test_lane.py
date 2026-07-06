"""Tests for lane geometry, classification, slope filtering and smoothing."""

from __future__ import annotations

import numpy as np

from app.config import DetectionConfig
from app.lane import Lane, LaneEstimator, LaneFrame, LaneSide, mean_slope


def _segment(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    return np.array([[x1, y1, x2, y2]], dtype=np.int32)


def test_lane_points_and_array() -> None:
    lane = Lane(LaneSide.LEFT, 10, 480, 200, 300)
    assert lane.points == ((10, 480), (200, 300))
    assert lane.as_array().tolist() == [10, 480, 200, 300]


def test_lane_frame_pair_detection() -> None:
    left = Lane(LaneSide.LEFT, 100, 480, 250, 300)
    right = Lane(LaneSide.RIGHT, 540, 480, 390, 300)
    frame = LaneFrame(left=left, right=right)
    assert frame.has_pair
    assert len(frame.lanes) == 2


def test_deviation_centered_is_small() -> None:
    left = Lane(LaneSide.LEFT, 200, 480, 260, 300)
    right = Lane(LaneSide.RIGHT, 440, 480, 380, 300)
    frame = LaneFrame(left=left, right=right)
    # lane center = 320, vehicle center = 320 → deviation 0.
    assert frame.deviation_px(640) == 0


def test_deviation_none_without_pair() -> None:
    frame = LaneFrame(left=Lane(LaneSide.LEFT, 100, 480, 250, 300))
    assert frame.deviation_px(640) is None


def test_estimator_classifies_two_lanes() -> None:
    estimator = LaneEstimator(DetectionConfig())
    segments = [
        _segment(120, 480, 280, 290),  # left (negative slope)
        _segment(520, 480, 360, 290),  # right (positive slope)
    ]
    frame = estimator.estimate(segments, (480, 640))
    assert frame.has_pair
    assert frame.left is not None and frame.left.side is LaneSide.LEFT
    assert frame.right is not None and frame.right.side is LaneSide.RIGHT


def test_estimator_filters_shallow_slopes() -> None:
    estimator = LaneEstimator(DetectionConfig())
    # Near-horizontal segment: slope well below min_slope, must be dropped.
    frame = estimator.estimate([_segment(0, 240, 640, 250)], (480, 640))
    assert not frame.lanes


def test_estimator_handles_no_segments() -> None:
    estimator = LaneEstimator(DetectionConfig())
    frame = estimator.estimate(None, (480, 640))
    assert frame.lanes == []


def test_smoothing_bridges_dropout() -> None:
    estimator = LaneEstimator(DetectionConfig())
    segments = [_segment(120, 480, 280, 290), _segment(520, 480, 360, 290)]
    estimator.estimate(segments, (480, 640))
    # Next frame detects nothing — smoothing should still emit lanes from history.
    bridged = estimator.estimate(None, (480, 640))
    assert bridged.has_pair


def test_reset_clears_history() -> None:
    estimator = LaneEstimator(DetectionConfig())
    estimator.estimate([_segment(120, 480, 280, 290)], (480, 640))
    estimator.reset()
    assert estimator.estimate(None, (480, 640)).lanes == []


def test_mean_slope() -> None:
    assert mean_slope([]) == 0.0
    assert mean_slope([(1.0, 0.0), (3.0, 0.0)]) == 2.0
