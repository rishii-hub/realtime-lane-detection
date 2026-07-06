"""Integration tests for the end-to-end detector and metrics tracker."""

from __future__ import annotations

from app.detector import DetectionResult, LaneDetector
from app.metrics import MetricsTracker


def test_process_returns_result(pipeline_config, synthetic_lane_frame) -> None:
    detector = LaneDetector(pipeline_config)
    result = detector.process(synthetic_lane_frame)
    assert isinstance(result, DetectionResult)
    assert result.annotated.shape == synthetic_lane_frame.shape
    assert result.edges.ndim == 2


def test_process_detects_synthetic_lanes(pipeline_config, synthetic_lane_frame) -> None:
    detector = LaneDetector(pipeline_config)
    result = detector.process(synthetic_lane_frame)
    # The synthetic 'V' should yield at least one boundary.
    assert result.lanes.lanes


def test_process_blank_frame_is_safe(pipeline_config, blank_frame) -> None:
    detector = LaneDetector(pipeline_config)
    result = detector.process(blank_frame)
    assert result.lanes.lanes == []
    assert result.metrics.frame_count == 1


def test_reset_clears_metrics(pipeline_config, blank_frame) -> None:
    detector = LaneDetector(pipeline_config)
    detector.process(blank_frame)
    detector.reset()
    assert detector.metrics.frame_count == 0


def test_metrics_tracker_records_fps() -> None:
    tracker = MetricsTracker(window=5)
    metrics = tracker.record(elapsed_s=0.02)  # 50 FPS
    assert metrics.fps == 50.0
    assert metrics.frame_count == 1
    assert metrics.latency_ms == 20.0


def test_metrics_tracker_zero_elapsed_is_safe() -> None:
    tracker = MetricsTracker()
    metrics = tracker.record(elapsed_s=0.0)
    assert metrics.fps == 0.0
