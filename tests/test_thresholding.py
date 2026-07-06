"""Tests for lane-pixel thresholding."""

import numpy as np

from lane_detector import thresholding as th


def test_combined_threshold_is_binary(synthetic_lane_frame):
    mask = th.combined_threshold(synthetic_lane_frame)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_combined_threshold_matches_frame_shape(synthetic_lane_frame):
    mask = th.combined_threshold(synthetic_lane_frame)
    assert mask.shape == synthetic_lane_frame.shape[:2]


def test_white_markings_are_detected(synthetic_lane_frame):
    mask = th.combined_threshold(synthetic_lane_frame)
    # the painted lane lines should light up a non-trivial number of pixels
    assert (mask > 0).sum() > 500


def test_blank_frame_has_few_pixels(blank_frame):
    mask = th.combined_threshold(blank_frame)
    # a pure black frame should not hallucinate lane pixels
    assert (mask > 0).sum() < 50


def test_white_mask_thresholds_on_lightness(synthetic_lane_frame):
    mask = th.white_mask(synthetic_lane_frame)
    assert set(np.unique(mask)).issubset({0, 255})
    assert (mask > 0).sum() > 0
