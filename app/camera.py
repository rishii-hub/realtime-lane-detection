"""Video capture abstraction.

Wrapping :class:`cv2.VideoCapture` behind a small context-manager keeps resource
handling explicit and makes the capture source trivially swappable (webcam,
video file, or IP camera stream).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from types import TracebackType

import cv2
import numpy as np

from app.config import CameraConfig

logger = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when a video source cannot be opened or read."""


class VideoSource:
    """A context-managed frame source backed by OpenCV.

    Parameters
    ----------
    source:
        ``0`` for the default webcam, a filesystem path to a video, or a URL for
        an IP camera stream (e.g. ``http://192.168.1.100:8080/video``).
    config:
        Capture resolution and FPS hints.
    """

    def __init__(self, source: int | str = 0, config: CameraConfig | None = None) -> None:
        self._source = source
        self._config = config or CameraConfig()
        self._capture: cv2.VideoCapture | None = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def open(self) -> VideoSource:
        """Open the underlying capture device and apply capture hints."""
        capture = cv2.VideoCapture(self._source)
        if not capture.isOpened():
            raise CameraError(f"Could not open video source: {self._source!r}")

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        capture.set(cv2.CAP_PROP_FPS, self._config.fps)

        self._capture = capture
        logger.info(
            "Opened source %r at %dx%d",
            self._source,
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        return self

    def read(self) -> np.ndarray | None:
        """Read the next frame, or ``None`` at end-of-stream."""
        if self._capture is None:
            raise CameraError("VideoSource.read() called before open().")
        ok, frame = self._capture.read()
        return frame if ok else None

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames until the stream is exhausted."""
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame

    def release(self) -> None:
        """Release the underlying capture device."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("Released video source %r", self._source)

    # ------------------------------------------------------------------ #
    # Context manager protocol
    # ------------------------------------------------------------------ #
    def __enter__(self) -> VideoSource:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.release()
