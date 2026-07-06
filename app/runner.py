"""Interactive display loop tying the detector to a live video window."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2

from app.camera import VideoSource
from app.config import PipelineConfig
from app.detector import LaneDetector
from app.utils import ensure_dir
from app.visualization import LaneRenderer

logger = logging.getLogger(__name__)

_WINDOW = "Real-Time Lane Detection"


class DetectionApp:
    """Drive :class:`LaneDetector` against a :class:`VideoSource` with an OpenCV UI.

    Keyboard controls
    -----------------
    ``Q`` quit · ``P`` pause/resume · ``E`` toggle edge view · ``S`` save frame.
    """

    def __init__(
        self,
        source: int | str = 0,
        config: PipelineConfig | None = None,
        capture_dir: str | Path = "captures",
    ) -> None:
        self._config = config or PipelineConfig()
        self._source = source
        self._detector = LaneDetector(self._config)
        self._renderer = LaneRenderer(self._config.visualization)
        self._capture_dir = Path(capture_dir)
        self._paused = False
        self._show_edges = False

    def run(self) -> None:
        """Open the source and process frames until the user quits."""
        logger.info("Starting lane detection. Press Q to quit.")
        last_output = None
        with VideoSource(self._source, self._config.camera) as source:
            while True:
                if not self._paused:
                    frame = source.read()
                    if frame is None:
                        logger.info("End of stream reached.")
                        break
                    result = self._detector.process(frame)
                    output = result.annotated
                    if self._show_edges:
                        output = self._renderer.stack_edges(output, result.edges)
                    last_output = output

                if last_output is not None:
                    cv2.imshow(_WINDOW, last_output)

                if not self._handle_key(last_output):
                    break

        cv2.destroyAllWindows()
        self._log_summary()

    # ------------------------------------------------------------------ #
    # Input handling
    # ------------------------------------------------------------------ #
    def _handle_key(self, frame) -> bool:
        """Process a single keypress. Returns ``False`` to stop the loop."""
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            return False
        if key == ord("p"):
            self._paused = not self._paused
            logger.info("Paused" if self._paused else "Resumed")
        elif key == ord("e"):
            self._show_edges = not self._show_edges
            logger.info("Edge view %s", "on" if self._show_edges else "off")
        elif key == ord("s") and frame is not None:
            self._save_frame(frame)
        return True

    def _save_frame(self, frame) -> None:
        ensure_dir(self._capture_dir)
        path = self._capture_dir / f"lane_capture_{int(time.time())}.jpg"
        cv2.imwrite(str(path), frame)
        logger.info("Saved capture to %s", path)

    def _log_summary(self) -> None:
        tracker = self._detector.metrics
        logger.info(
            "Session complete — %d frames, %.2f avg FPS",
            tracker.frame_count,
            tracker.avg_fps,
        )
