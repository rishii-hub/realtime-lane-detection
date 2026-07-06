"""
LaneDetector: orchestrates the full pipeline for one frame.

    frame
      -> threshold (colour + gradient)
      -> perspective warp (bird's-eye)
      -> sliding-window / prior-search polynomial fit
      -> sanity check + temporal smoothing
      -> unwarp lane polygon back onto the frame
      -> annotate (curvature, offset, departure status, FPS)

The class keeps a small amount of state (previous fit, smoothing buffers) so
consecutive frames use the fast search-around-prior path and jitter is damped.
"""

import time
from collections import deque

import cv2
import numpy as np

from .lane_fit import search_around_prior, sliding_window_fit
from .perspective import PerspectiveTransform
from .thresholding import combined_threshold


class LaneDetector:
    def __init__(self, frame_size=(640, 360), smooth_n=8):
        self.frame_size = frame_size
        self.perspective = PerspectiveTransform(frame_size)

        self.left_buffer = deque(maxlen=smooth_n)
        self.right_buffer = deque(maxlen=smooth_n)
        self.prev_left_fit = None
        self.prev_right_fit = None
        self.frames_since_reset = 0

        # rolling metrics
        self.curvature_m = None
        self.offset_m = None
        self.status = "INITIALISING"
        self.fps_history = deque(maxlen=30)
        self.confidence = 0.0

    # ---- sanity checks -------------------------------------------------
    def _plausible(self, fit_result):
        """Reject fits with implausible lane width or crossing lines."""
        if not fit_result.detected:
            return False
        w = fit_result.lane_width_px
        if w is None or w < 150 or w > 700:
            return False
        # lines should not cross
        if np.any(fit_result.right_x - fit_result.left_x < 50):
            return False
        return True

    def _smooth(self, fit_result):
        self.left_buffer.append(fit_result.left_fit)
        self.right_buffer.append(fit_result.right_fit)
        left = np.mean(self.left_buffer, axis=0)
        right = np.mean(self.right_buffer, axis=0)
        return left, right

    # ---- main entry ----------------------------------------------------
    def process(self, frame, view_mode="final"):
        t0 = time.time()
        frame = cv2.resize(frame, self.frame_size)

        binary = combined_threshold(frame)
        warped = self.perspective.warp(binary)

        # choose search strategy
        if self.prev_left_fit is not None and self.frames_since_reset < 30:
            fit = search_around_prior(warped, self.prev_left_fit, self.prev_right_fit)
            if not self._plausible(fit):
                fit = sliding_window_fit(warped)
                self.frames_since_reset = 0
        else:
            fit = sliding_window_fit(warped)
            self.frames_since_reset = 0

        if self._plausible(fit):
            self.prev_left_fit, self.prev_right_fit = fit.left_fit, fit.right_fit
            self.frames_since_reset += 1
            left_s, right_s = self._smooth(fit)
            self.curvature_m = fit.curvature_m
            self.offset_m = fit.offset_m
            self.confidence = min(1.0, len(self.left_buffer) / self.left_buffer.maxlen)
            self._update_status()
            output = self._render(frame, left_s, right_s, fit.ploty)
        else:
            self.confidence = max(0.0, self.confidence - 0.2)
            self.status = "SEARCHING"
            output = frame.copy()

        # view modes for the frontend's debug toggles
        if view_mode == "threshold":
            output = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        elif view_mode == "birdseye":
            output = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
        elif view_mode == "roi":
            output = self.perspective.draw_src_region(output)

        dt = time.time() - t0
        if dt > 0:
            self.fps_history.append(1.0 / dt)
        output = self._draw_hud(output)
        return output

    def _update_status(self):
        if self.offset_m is None:
            self.status = "SEARCHING"
        elif abs(self.offset_m) > 0.55:
            side = "RIGHT" if self.offset_m > 0 else "LEFT"
            self.status = f"LANE DEPARTURE {side}"
        else:
            self.status = "CENTERED"

    # ---- rendering -----------------------------------------------------
    def _render(self, frame, left_fit, right_fit, ploty):
        overlay = np.zeros_like(frame)
        left_x = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_x = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        pts_left = np.array([np.transpose(np.vstack([left_x, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_x, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        departure = self.status.startswith("LANE DEPARTURE")
        fill_color = (0, 80, 255) if departure else (0, 220, 0)
        cv2.fillPoly(overlay, np.int32([pts]), fill_color)

        # lane boundary lines
        cv2.polylines(overlay, np.int32([pts_left]), False, (255, 120, 0), 12)
        cv2.polylines(overlay, np.int32([pts_right]), False, (0, 120, 255), 12)

        unwarped = self.perspective.unwarp(overlay)
        return cv2.addWeighted(frame, 1.0, unwarped, 0.4, 0)

    def _draw_hud(self, image):
        h, w = image.shape[:2]
        panel = image.copy()
        cv2.rectangle(panel, (0, 0), (w, 78), (0, 0, 0), -1)
        image = cv2.addWeighted(image, 0.55, panel, 0.45, 0)

        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        curv = f"{self.curvature_m:,.0f} m" if self.curvature_m else "--"
        off = f"{self.offset_m:+.2f} m" if self.offset_m is not None else "--"

        status_color = {
            "CENTERED": (0, 220, 0),
            "SEARCHING": (0, 200, 255),
            "INITIALISING": (200, 200, 200),
        }.get(self.status, (0, 80, 255))

        cv2.putText(
            image,
            f"Curvature: {curv}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Offset: {off}",
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"FPS: {avg_fps:.1f}",
            (12, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            self.status,
            (w - 250, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            status_color,
            2,
            cv2.LINE_AA,
        )
        return image

    def metrics(self):
        """Machine-readable state for the frontend dashboard."""
        return {
            "curvature_m": round(self.curvature_m, 1) if self.curvature_m else None,
            "offset_m": round(self.offset_m, 3) if self.offset_m is not None else None,
            "status": self.status,
            "fps": round(float(np.mean(self.fps_history)), 1) if self.fps_history else 0.0,
            "confidence": round(self.confidence, 2),
        }
