"""Typed, validated configuration for the lane detection pipeline.

All tunable parameters live here as immutable :class:`dataclasses.dataclass`
objects. Configuration can be constructed programmatically or loaded from a
YAML file via :meth:`PipelineConfig.from_yaml`, which keeps experiment tracking
and reproducibility trivial.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CameraConfig:
    """Capture settings for the video source."""

    width: int = 640
    height: int = 480
    fps: int = 30
    max_processing_width: int = 800

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Camera dimensions must be positive.")
        if self.fps <= 0:
            raise ValueError("Camera FPS must be positive.")


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """Parameters that govern the classical CV detection stages."""

    # Pre-processing
    gaussian_kernel: int = 5
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8

    # Canny edge detection
    canny_low: int = 50
    canny_high: int = 150

    # Region of interest (fractions of frame dimensions)
    roi_bottom_width: float = 0.90
    roi_top_width: float = 0.20
    roi_horizon: float = 0.60

    # Hough transform
    hough_rho: int = 2
    hough_threshold: int = 40
    hough_min_line_length: int = 30
    hough_max_line_gap: int = 100

    # Slope filtering (radians-free heuristic thresholds)
    min_slope: float = 0.4
    max_slope: float = 2.5

    # Temporal smoothing
    smoothing_window: int = 5

    def __post_init__(self) -> None:
        if self.gaussian_kernel % 2 == 0:
            raise ValueError("gaussian_kernel must be odd.")
        if self.canny_low >= self.canny_high:
            raise ValueError("canny_low must be smaller than canny_high.")
        if not 0.0 < self.roi_horizon < 1.0:
            raise ValueError("roi_horizon must be a fraction in (0, 1).")
        if self.min_slope >= self.max_slope:
            raise ValueError("min_slope must be smaller than max_slope.")
        if self.smoothing_window < 1:
            raise ValueError("smoothing_window must be >= 1.")


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    """Rendering options for the annotated output frame."""

    lane_color: tuple[int, int, int] = (0, 255, 0)
    single_lane_color: tuple[int, int, int] = (0, 255, 255)
    warning_color: tuple[int, int, int] = (0, 165, 255)
    lane_thickness: int = 10
    fill_alpha: float = 0.30
    deviation_threshold_px: int = 50
    show_hud: bool = True


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Top-level configuration aggregating every stage of the pipeline."""

    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # ------------------------------------------------------------------ #
    # Serialization helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> PipelineConfig:
        """Build a config from a nested mapping, ignoring unknown keys."""
        return cls(
            camera=CameraConfig(**data.get("camera", {})),
            detection=DetectionConfig(**data.get("detection", {})),
            visualization=VisualizationConfig(**data.get("visualization", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PipelineConfig:
        """Load a config from a YAML file.

        Falls back to defaults with a warning if the file is missing so that
        the pipeline is always runnable out of the box.
        """
        path = Path(path)
        if not path.exists():
            logger.warning("Config file %s not found; using defaults.", path)
            return cls()

        import yaml  # imported lazily to keep the core dependency-light

        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        logger.info("Loaded pipeline configuration from %s", path)
        return cls.from_mapping(data)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable representation."""
        return asdict(self)
