"""Tests for sliding-window lane fitting and metric computation."""

import numpy as np

from lane_detector.lane_fit import search_around_prior, sliding_window_fit


def _straight_lane_mask(w=640, h=360, left_x=200, right_x=440):
    """Bird's-eye binary mask with two vertical lane lines."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, left_x - 4 : left_x + 4] = 255
    mask[:, right_x - 4 : right_x + 4] = 255
    return mask


def test_sliding_window_detects_two_lanes():
    result = sliding_window_fit(_straight_lane_mask())
    assert result.detected
    assert result.left_fit is not None
    assert result.right_fit is not None


def test_detected_lane_width_is_reasonable():
    result = sliding_window_fit(_straight_lane_mask(left_x=200, right_x=440))
    assert 200 < result.lane_width_px < 300


def test_straight_lane_has_large_curvature():
    """Perfectly straight lines => near-infinite radius (very large number)."""
    result = sliding_window_fit(_straight_lane_mask())
    assert result.curvature_m is not None
    assert result.curvature_m > 2000


def test_offset_near_zero_for_centered_lane():
    # lane centred around x=320 in a 640-wide frame => ~0 offset
    result = sliding_window_fit(_straight_lane_mask(left_x=220, right_x=420))
    assert abs(result.offset_m) < 0.5


def test_empty_mask_is_not_detected():
    empty = np.zeros((360, 640), dtype=np.uint8)
    result = sliding_window_fit(empty)
    assert not result.detected


def test_search_around_prior_tracks_previous_fit():
    mask = _straight_lane_mask()
    first = sliding_window_fit(mask)
    second = search_around_prior(mask, first.left_fit, first.right_fit)
    assert second.detected
    # fits should be close since the mask is unchanged
    assert np.allclose(first.left_fit, second.left_fit, atol=1.0)
