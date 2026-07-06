"""Lane geometry primitives and estimation logic.

This module isolates the *math* of lane detection from the *imaging* stages.
Everything here operates on Hough line segments and produces clean, typed
:class:`Lane` objects, which keeps :mod:`app.detector` focused on orchestration.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from app.config import DetectionConfig


class LaneSide(str, Enum):
    """Which side of the road a lane boundary belongs to."""

    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True, slots=True)
class Lane:
    """A single lane boundary as a line segment in image coordinates."""

    side: LaneSide
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def points(self) -> tuple[tuple[int, int], tuple[int, int]]:
        """Return the two endpoints as ``((x1, y1), (x2, y2))``."""
        return ((self.x1, self.y1), (self.x2, self.y2))

    def as_array(self) -> np.ndarray:
        """Return the segment as a flat ``[x1, y1, x2, y2]`` array."""
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.int32)


@dataclass(frozen=True, slots=True)
class LaneFrame:
    """The lane estimate for a single processed frame."""

    left: Lane | None = None
    right: Lane | None = None

    @property
    def lanes(self) -> list[Lane]:
        """Return the detected lanes as an ordered list (left, then right)."""
        return [lane for lane in (self.left, self.right) if lane is not None]

    @property
    def has_pair(self) -> bool:
        """``True`` when both boundaries were detected."""
        return self.left is not None and self.right is not None

    def deviation_px(self, frame_width: int) -> int | None:
        """Signed pixel offset of the vehicle from the lane centre.

        Positive values mean the vehicle is right of centre. Requires both lane
        boundaries to be present.
        """
        if not self.has_pair:
            return None
        assert self.left is not None and self.right is not None
        lane_center = (self.left.x1 + self.right.x1) // 2
        vehicle_center = frame_width // 2
        return vehicle_center - lane_center


class LaneEstimator:
    """Convert raw Hough segments into a smoothed :class:`LaneFrame`.

    The estimator is stateful: it keeps a short rolling history of past lanes so
    that momentary detection dropouts do not cause the overlay to flicker.
    """

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config
        self._left_history: deque[np.ndarray] = deque(maxlen=config.smoothing_window)
        self._right_history: deque[np.ndarray] = deque(maxlen=config.smoothing_window)

    def reset(self) -> None:
        """Clear the temporal smoothing buffers."""
        self._left_history.clear()
        self._right_history.clear()

    # ------------------------------------------------------------------ #
    # Estimation
    # ------------------------------------------------------------------ #
    def estimate(
        self, segments: Sequence[np.ndarray] | None, frame_shape: tuple[int, int]
    ) -> LaneFrame:
        """Return a smoothed lane estimate from raw Hough segments."""
        height, width = frame_shape[:2]
        left_fit, right_fit = self._classify_segments(segments, width)

        left = self._fit_to_lane(left_fit, LaneSide.LEFT, height, width)
        right = self._fit_to_lane(right_fit, LaneSide.RIGHT, height, width)
        return self._smooth(LaneFrame(left=left, right=right), height, width)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _classify_segments(
        self, segments: Sequence[np.ndarray] | None, width: int
    ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        """Split segments into left/right (slope, intercept) candidates."""
        left: list[tuple[float, float]] = []
        right: list[tuple[float, float]] = []
        if segments is None:
            return left, right

        cfg = self._config
        for segment in segments:
            x1, y1, x2, y2 = np.asarray(segment).reshape(4)
            if x2 == x1:  # perfectly vertical: undefined slope
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            if not (cfg.min_slope <= abs(slope) <= cfg.max_slope):
                continue

            # Negative slope on the left half → left boundary; mirror for right.
            if slope < 0 and x1 < width * 0.55:
                left.append((slope, intercept))
            elif slope > 0 and x1 > width * 0.45:
                right.append((slope, intercept))

        return left, right

    def _fit_to_lane(
        self,
        fits: list[tuple[float, float]],
        side: LaneSide,
        height: int,
        width: int,
    ) -> Lane | None:
        """Collapse candidate fits into a single validated lane segment."""
        if not fits:
            return None

        slope, intercept = np.median(np.asarray(fits), axis=0)
        if slope == 0:
            return None

        y1 = height
        y2 = int(height * self._config.roi_horizon)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        # Reject fits that project wildly outside the frame.
        if not (-width < x1 < width * 2 and -width < x2 < width * 2):
            return None

        return Lane(side=side, x1=x1, y1=y1, x2=x2, y2=y2)

    def _smooth(self, frame: LaneFrame, height: int, width: int) -> LaneFrame:
        """Blend the current estimate with recent history for stability."""
        left = self._smooth_side(frame.left, self._left_history, LaneSide.LEFT)
        right = self._smooth_side(frame.right, self._right_history, LaneSide.RIGHT)
        return LaneFrame(left=left, right=right)

    @staticmethod
    def _smooth_side(lane: Lane | None, history: deque[np.ndarray], side: LaneSide) -> Lane | None:
        """Average a single side against its rolling history."""
        if lane is not None:
            history.append(lane.as_array())
        if not history:
            return None
        mean = np.mean(np.asarray(history), axis=0).astype(int)
        return Lane(side=side, x1=int(mean[0]), y1=int(mean[1]), x2=int(mean[2]), y2=int(mean[3]))


def mean_slope(fits: Iterable[tuple[float, float]]) -> float:
    """Return the mean slope of a collection of ``(slope, intercept)`` fits.

    Exposed as a small pure helper primarily so it can be unit-tested in
    isolation from OpenCV.
    """
    fits = list(fits)
    if not fits:
        return 0.0
    return float(np.mean([slope for slope, _ in fits]))
