"""Tests for configuration construction and validation."""

from __future__ import annotations

import pytest

from app.config import (
    CameraConfig,
    DetectionConfig,
    PipelineConfig,
    VisualizationConfig,
)


def test_defaults_are_valid() -> None:
    config = PipelineConfig()
    assert isinstance(config.camera, CameraConfig)
    assert isinstance(config.detection, DetectionConfig)
    assert isinstance(config.visualization, VisualizationConfig)


def test_from_mapping_ignores_unknown_keys() -> None:
    config = PipelineConfig.from_mapping(
        {"camera": {"width": 1280, "height": 720}, "unknown": {"x": 1}}
    )
    assert config.camera.width == 1280
    assert config.camera.height == 720


def test_to_dict_roundtrip() -> None:
    original = PipelineConfig()
    restored = PipelineConfig.from_mapping(original.to_dict())
    assert restored == original


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"height": -1},
        {"fps": 0},
    ],
)
def test_camera_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CameraConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gaussian_kernel": 4},  # even kernel
        {"canny_low": 200, "canny_high": 100},  # inverted thresholds
        {"roi_horizon": 1.5},  # out of range
        {"min_slope": 3.0, "max_slope": 2.0},  # inverted slopes
        {"smoothing_window": 0},  # too small
    ],
)
def test_detection_rejects_invalid_values(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DetectionConfig(**kwargs)


def test_from_yaml_missing_file_returns_defaults(tmp_path) -> None:
    config = PipelineConfig.from_yaml(tmp_path / "nope.yaml")
    assert config == PipelineConfig()
