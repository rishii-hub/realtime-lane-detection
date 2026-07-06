"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from app.config import DetectionConfig, PipelineConfig


@pytest.fixture()
def detection_config() -> DetectionConfig:
    return DetectionConfig()


@pytest.fixture()
def pipeline_config() -> PipelineConfig:
    return PipelineConfig()


@pytest.fixture()
def blank_frame() -> np.ndarray:
    """A 480x640 black BGR frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture()
def synthetic_lane_frame() -> np.ndarray:
    """A black frame with two bright lane-like lines forming a 'V'."""
    import cv2

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Left boundary (negative slope) and right boundary (positive slope).
    cv2.line(frame, (120, 480), (280, 290), (255, 255, 255), 8)
    cv2.line(frame, (520, 480), (360, 290), (255, 255, 255), 8)
    return frame
