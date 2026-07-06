"""Tests for utility helpers."""

from __future__ import annotations

import pytest

from app.utils import clamp, ensure_dir, moving_average, scale_to_width


@pytest.mark.parametrize(
    ("value", "low", "high", "expected"),
    [(5, 0, 10, 5), (-1, 0, 10, 0), (11, 0, 10, 10), (0.5, 0.0, 1.0, 0.5)],
)
def test_clamp(value, low, high, expected) -> None:
    assert clamp(value, low, high) == expected


def test_scale_to_width_no_resize_needed() -> None:
    assert scale_to_width((480, 640), 800) == 1.0


def test_scale_to_width_downscales() -> None:
    assert scale_to_width((1080, 1920), 800) == pytest.approx(800 / 1920)


def test_moving_average() -> None:
    assert moving_average([]) == 0.0
    assert moving_average([2.0, 4.0]) == 3.0


def test_ensure_dir_creates(tmp_path) -> None:
    target = tmp_path / "a" / "b"
    result = ensure_dir(target)
    assert result.exists() and result.is_dir()
