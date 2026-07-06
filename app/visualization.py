"""Rendering of lane overlays and the heads-up display (HUD)."""

from __future__ import annotations

import cv2
import numpy as np

from app.config import VisualizationConfig
from app.lane import LaneFrame
from app.metrics import FrameMetrics

_FONT = cv2.FONT_HERSHEY_SIMPLEX


class LaneRenderer:
    """Draw lane estimates and metrics onto a BGR frame."""

    def __init__(self, config: VisualizationConfig) -> None:
        self._config = config

    def render(
        self,
        frame: np.ndarray,
        lanes: LaneFrame,
        metrics: FrameMetrics | None = None,
    ) -> np.ndarray:
        """Return an annotated copy of ``frame``."""
        output = frame.copy()
        output = self._draw_lanes(output, lanes)
        if self._config.show_hud and metrics is not None:
            output = self._draw_hud(output, metrics)
        return output

    # ------------------------------------------------------------------ #
    # Lane overlay
    # ------------------------------------------------------------------ #
    def _draw_lanes(self, frame: np.ndarray, lanes: LaneFrame) -> np.ndarray:
        cfg = self._config

        if lanes.has_pair:
            frame = self._fill_lane_region(frame, lanes)
            for lane in lanes.lanes:
                cv2.line(frame, *lane.points, cfg.lane_color, cfg.lane_thickness)
            self._draw_deviation(frame, lanes)
        elif lanes.lanes:
            lane = lanes.lanes[0]
            cv2.line(frame, *lane.points, cfg.single_lane_color, cfg.lane_thickness)
            self._label(frame, "Single lane detected", cfg.single_lane_color)
        else:
            self._label(frame, "No lanes detected", cfg.warning_color)

        return frame

    def _fill_lane_region(self, frame: np.ndarray, lanes: LaneFrame) -> np.ndarray:
        assert lanes.left is not None and lanes.right is not None
        left, right = lanes.left, lanes.right
        polygon = np.array(
            [
                [
                    (left.x1, left.y1),
                    (left.x2, left.y2),
                    (right.x2, right.y2),
                    (right.x1, right.y1),
                ]
            ],
            dtype=np.int32,
        )
        overlay = frame.copy()
        cv2.fillPoly(overlay, polygon, self._config.lane_color)
        alpha = self._config.fill_alpha
        return cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    def _draw_deviation(self, frame: np.ndarray, lanes: LaneFrame) -> None:
        deviation = lanes.deviation_px(frame.shape[1])
        if deviation is None:
            return
        cfg = self._config
        within_tolerance = abs(deviation) < cfg.deviation_threshold_px
        color = cfg.lane_color if within_tolerance else cfg.warning_color
        direction = "RIGHT" if deviation > 0 else "LEFT"
        self._label(frame, f"Offset: {abs(deviation)}px {direction}", color)

    # ------------------------------------------------------------------ #
    # HUD
    # ------------------------------------------------------------------ #
    def _draw_hud(self, frame: np.ndarray, metrics: FrameMetrics) -> np.ndarray:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (280, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        color = self._config.lane_color
        cv2.putText(frame, f"FPS: {metrics.avg_fps:.1f}", (10, 30), _FONT, 0.7, color, 2)
        cv2.putText(frame, f"Latency: {metrics.latency_ms:.1f}ms", (10, 60), _FONT, 0.6, color, 2)
        cv2.putText(frame, f"Frames: {metrics.frame_count}", (10, 90), _FONT, 0.6, color, 2)
        cv2.putText(
            frame,
            "Q:Quit  D:Debug  E:Edges  P:Pause  S:Save",
            (10, frame.shape[0] - 10),
            _FONT,
            0.5,
            (255, 255, 255),
            1,
        )
        return frame

    @staticmethod
    def _label(frame: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
        cv2.putText(frame, text, (10, 150), _FONT, 0.6, color, 2)

    @staticmethod
    def stack_edges(frame: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Return ``frame`` with the edge map stacked horizontally beside it."""
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return np.hstack((frame, edges_bgr))
