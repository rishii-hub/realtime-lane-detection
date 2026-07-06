"""
Sliding-window lane fitting on the bird's-eye binary mask.

This is what lets the detector follow *curves*: instead of forcing a single
straight line (Hough), we collect lane pixels in vertical windows and fit a
2nd-degree polynomial  x = a*y^2 + b*y + c  to each lane. From those polynomials
we derive the real-world radius of curvature and the vehicle's offset from lane
centre.
"""

import numpy as np

# Real-world scaling (metres per pixel) for curvature/offset in physical units.
# Approximate values for a warped ~ lane-width view; tune per camera.
YM_PER_PIX = 30 / 720
XM_PER_PIX = 3.7 / 700


class LaneFitResult:
    def __init__(self):
        self.left_fit = None  # polynomial coeffs (pixel space)
        self.right_fit = None
        self.left_x = None  # plotted x per y-row
        self.right_x = None
        self.ploty = None
        self.curvature_m = None  # radius of curvature (metres)
        self.offset_m = None  # + = vehicle right of centre
        self.detected = False
        self.lane_width_px = None


def _polyfit_lane(ys, xs, height):
    """Fit x = f(y) and evaluate over the full column height."""
    if len(ys) < 200:  # too few pixels to trust a fit
        return None, None, None
    fit = np.polyfit(ys, xs, 2)
    ploty = np.linspace(0, height - 1, height)
    plotx = fit[0] * ploty**2 + fit[1] * ploty + fit[2]
    return fit, plotx, ploty


def sliding_window_fit(binary_warped, n_windows=9, margin=60, minpix=40):
    """Full sliding-window search (used on the first frame or after a reset)."""
    height, width = binary_warped.shape
    result = LaneFitResult()

    # Histogram of the bottom half locates the two lane bases.
    histogram = np.sum(binary_warped[height // 2 :, :], axis=0)
    midpoint = width // 2
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    window_height = height // n_windows
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    left_current, right_current = left_base, right_base
    left_inds, right_inds = [], []

    for window in range(n_windows):
        y_low = height - (window + 1) * window_height
        y_high = height - window * window_height

        xl_low, xl_high = left_current - margin, left_current + margin
        xr_low, xr_high = right_current - margin, right_current + margin

        good_left = (
            (nonzero_y >= y_low)
            & (nonzero_y < y_high)
            & (nonzero_x >= xl_low)
            & (nonzero_x < xl_high)
        ).nonzero()[0]
        good_right = (
            (nonzero_y >= y_low)
            & (nonzero_y < y_high)
            & (nonzero_x >= xr_low)
            & (nonzero_x < xr_high)
        ).nonzero()[0]

        left_inds.append(good_left)
        right_inds.append(good_right)

        if len(good_left) > minpix:
            left_current = int(np.mean(nonzero_x[good_left]))
        if len(good_right) > minpix:
            right_current = int(np.mean(nonzero_x[good_right]))

    left_inds = np.concatenate(left_inds) if left_inds else np.array([], int)
    right_inds = np.concatenate(right_inds) if right_inds else np.array([], int)

    return _finalise_fit(
        result,
        binary_warped,
        nonzero_x[left_inds],
        nonzero_y[left_inds],
        nonzero_x[right_inds],
        nonzero_y[right_inds],
    )


def search_around_prior(binary_warped, left_fit, right_fit, margin=60):
    """Fast path: search near the previous frame's polynomials."""
    height, width = binary_warped.shape
    result = LaneFitResult()
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    left_curve = left_fit[0] * nonzero_y**2 + left_fit[1] * nonzero_y + left_fit[2]
    right_curve = right_fit[0] * nonzero_y**2 + right_fit[1] * nonzero_y + right_fit[2]

    left_inds = (np.abs(nonzero_x - left_curve) < margin).nonzero()[0]
    right_inds = (np.abs(nonzero_x - right_curve) < margin).nonzero()[0]

    return _finalise_fit(
        result,
        binary_warped,
        nonzero_x[left_inds],
        nonzero_y[left_inds],
        nonzero_x[right_inds],
        nonzero_y[right_inds],
    )


def _finalise_fit(result, binary_warped, lx, ly, rx, ry):
    height, width = binary_warped.shape

    left_fit, left_plotx, ploty = _polyfit_lane(ly, lx, height)
    right_fit, right_plotx, _ = _polyfit_lane(ry, rx, height)

    result.left_fit, result.right_fit = left_fit, right_fit
    result.left_x, result.right_x = left_plotx, right_plotx
    result.ploty = ploty

    if left_fit is not None and right_fit is not None:
        result.detected = True
        result.lane_width_px = float(np.mean(right_plotx - left_plotx))
        result.curvature_m = _curvature(ly, lx, ry, rx, height)
        result.offset_m = _offset(left_plotx, right_plotx, width)

    return result


def _curvature(ly, lx, ry, rx, height):
    """Radius of curvature in metres, evaluated at the vehicle (bottom row)."""
    y_eval = height * YM_PER_PIX
    try:
        left_fit_m = np.polyfit(ly * YM_PER_PIX, lx * XM_PER_PIX, 2)
        right_fit_m = np.polyfit(ry * YM_PER_PIX, rx * XM_PER_PIX, 2)
        left_r = ((1 + (2 * left_fit_m[0] * y_eval + left_fit_m[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_m[0]
        )
        right_r = ((1 + (2 * right_fit_m[0] * y_eval + right_fit_m[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_m[0]
        )
        return float((left_r + right_r) / 2)
    except (TypeError, ValueError):
        return None


def _offset(left_plotx, right_plotx, width):
    """Vehicle offset from lane centre in metres (+ = right of centre)."""
    lane_center = (left_plotx[-1] + right_plotx[-1]) / 2
    vehicle_center = width / 2
    return float((vehicle_center - lane_center) * XM_PER_PIX)
