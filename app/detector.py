"""The lane detection pipeline orchestrator.

:class:`LaneDetector` wires together the imaging stages (pre-processing, edge
detection, region masking, Hough transform) with the geometric estimator and
the renderer. It is deliberately free of any I/O so that it can be unit-tested
on synthetic frames and reused from both the CLI and notebooks.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from app.config import PipelineConfig
from app.lane import LaneEstimator, LaneFrame
from app.metrics import FrameMetrics, MetricsTracker
from app.utils import scale_to_width
from app.visualization import LaneRenderer

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Everything produced by a single :meth:`LaneDetector.process` call."""

    frame: np.ndarray
    edges: np.ndarray
    lanes: LaneFrame
    metrics: FrameMetrics
    annotated: np.ndarray


class LaneDetector:
    """End-to-end classical lane detection pipeline."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self._config = config or PipelineConfig()
        self._estimator = LaneEstimator(self._config.detection)
        self._renderer = LaneRenderer(self._config.visualization)
        self._metrics = MetricsTracker()

    @property
    def config(self) -> PipelineConfig:
        return self._config

    @property
    def metrics(self) -> MetricsTracker:
        return self._metrics

    def reset(self) -> None:
        """Reset temporal smoothing and metrics (e.g. on source change)."""
        self._estimator.reset()
        self._metrics.reset()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def process(self, frame: np.ndarray) -> DetectionResult:
        """Run the full pipeline on a single BGR frame."""
        start = time.perf_counter()

        frame = self._preprocess(frame)
        edges = self._detect_edges(frame)
        roi = self._apply_roi(edges)
        segments = self._hough_lines(roi)
        lanes = self._estimator.estimate(segments, frame.shape)

        elapsed = time.perf_counter() - start
        metrics = self._metrics.record(elapsed, lanes.deviation_px(frame.shape[1]))
        annotated = self._renderer.render(frame, lanes, metrics)

        return DetectionResult(
            frame=frame, edges=edges, lanes=lanes, metrics=metrics, annotated=annotated
        )

    # ------------------------------------------------------------------ #
    # Imaging stages
    # ------------------------------------------------------------------ #
    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Downscale wide frames to keep processing real-time."""
        scale = scale_to_width(frame.shape, self._config.camera.max_processing_width)
        if scale < 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        return frame

    def _detect_edges(self, frame: np.ndarray) -> np.ndarray:
        """Grayscale → blur → CLAHE → Canny."""
        cfg = self._config.detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (cfg.gaussian_kernel, cfg.gaussian_kernel), 0)
        clahe = cv2.createCLAHE(
            clipLimit=cfg.clahe_clip_limit,
            tileGridSize=(cfg.clahe_grid_size, cfg.clahe_grid_size),
        )
        enhanced = clahe.apply(blur)
        return cv2.Canny(enhanced, cfg.canny_low, cfg.canny_high)

    def _apply_roi(self, edges: np.ndarray) -> np.ndarray:
        """Mask everything outside a trapezoidal region of interest."""
        cfg = self._config.detection
        height, width = edges.shape[:2]
        bottom_margin = (1 - cfg.roi_bottom_width) / 2
        top_margin = (1 - cfg.roi_top_width) / 2
        polygon = np.array(
            [
                [
                    (int(width * bottom_margin), height),
                    (int(width * (1 - bottom_margin)), height),
                    (int(width * (1 - top_margin)), int(height * cfg.roi_horizon)),
                    (int(width * top_margin), int(height * cfg.roi_horizon)),
                ]
            ],
            dtype=np.int32,
        )
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)

    def _hough_lines(self, roi: np.ndarray) -> list[np.ndarray] | None:
        """Probabilistic Hough transform returning raw segments."""
        cfg = self._config.detection
        lines = cv2.HoughLinesP(
            roi,
            rho=cfg.hough_rho,
            theta=np.pi / 180,
            threshold=cfg.hough_threshold,
            minLineLength=cfg.hough_min_line_length,
            maxLineGap=cfg.hough_max_line_gap,
        )
        return list(lines) if lines is not None else None
