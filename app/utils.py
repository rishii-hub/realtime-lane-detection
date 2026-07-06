"""Small, dependency-light utilities shared across the pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a clean, consistent root logger for CLI usage."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def clamp(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to the inclusive range ``[low, high]``."""
    return max(low, min(high, value))


def scale_to_width(shape: tuple[int, int], max_width: int) -> float:
    """Return the scale factor needed to fit ``shape`` within ``max_width``.

    Returns ``1.0`` when the frame is already narrow enough, so callers can skip
    the resize entirely.
    """
    _, width = shape[:2]
    if width <= max_width:
        return 1.0
    return max_width / width


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if needed and return it as a ``Path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def moving_average(values: list[float]) -> float:
    """Return the arithmetic mean of ``values`` (0.0 for an empty list)."""
    if not values:
        return 0.0
    return float(np.mean(values))
