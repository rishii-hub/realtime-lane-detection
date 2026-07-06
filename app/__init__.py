"""Real-Time Lane Detection.

A lightweight, real-time lane detection pipeline built on classical computer
vision (Canny edge detection + Hough transform) with temporal smoothing.

The public API intentionally mirrors the module layout so that consumers can
import exactly what they need::

    from app import LaneDetector, PipelineConfig, VideoSource

Example
-------
>>> from app import LaneDetector, PipelineConfig
>>> detector = LaneDetector(PipelineConfig())
>>> result = detector.process(frame)  # doctest: +SKIP
"""

from __future__ import annotations

from app.camera import VideoSource
from app.config import CameraConfig, DetectionConfig, PipelineConfig, VisualizationConfig
from app.detector import DetectionResult, LaneDetector
from app.lane import Lane, LaneFrame
from app.metrics import FrameMetrics, MetricsTracker

__all__ = [
    "CameraConfig",
    "DetectionConfig",
    "DetectionResult",
    "FrameMetrics",
    "Lane",
    "LaneDetector",
    "LaneFrame",
    "MetricsTracker",
    "PipelineConfig",
    "VideoSource",
    "VisualizationConfig",
]

__version__ = "1.0.0"
