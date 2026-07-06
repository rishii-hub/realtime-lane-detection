"""Runtime performance metrics for the detection pipeline."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from app.utils import moving_average


@dataclass(frozen=True, slots=True)
class FrameMetrics:
    """A snapshot of performance metrics for a single frame."""

    fps: float
    avg_fps: float
    latency_ms: float
    frame_count: int
    deviation_px: int | None = None


class MetricsTracker:
    """Accumulate per-frame timing information and expose smoothed metrics."""

    def __init__(self, window: int = 30) -> None:
        self._fps_history: deque[float] = deque(maxlen=window)
        self._frame_count = 0
        self._last_latency_ms = 0.0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def avg_fps(self) -> float:
        return moving_average(list(self._fps_history))

    def record(self, elapsed_s: float, deviation_px: int | None = None) -> FrameMetrics:
        """Record a processed frame and return its :class:`FrameMetrics`."""
        self._frame_count += 1
        self._last_latency_ms = elapsed_s * 1000.0
        fps = 1.0 / elapsed_s if elapsed_s > 0 else 0.0
        self._fps_history.append(fps)
        return FrameMetrics(
            fps=fps,
            avg_fps=self.avg_fps,
            latency_ms=self._last_latency_ms,
            frame_count=self._frame_count,
            deviation_px=deviation_px,
        )

    def reset(self) -> None:
        """Clear all accumulated metrics."""
        self._fps_history.clear()
        self._frame_count = 0
        self._last_latency_ms = 0.0
