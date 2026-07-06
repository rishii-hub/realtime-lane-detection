"""
Perspective (bird's-eye) transform.

Fitting a polynomial to lane pixels is far more reliable in a top-down view
where lane lines become roughly parallel and vertical. We define a trapezoid
on the road plane (source) and map it to a rectangle (destination).

Source points are expressed as fractions of frame size so the same config
works regardless of input resolution.
"""

import cv2
import numpy as np


class PerspectiveTransform:
    def __init__(self, frame_size):
        """frame_size: (width, height)"""
        self.width, self.height = frame_size
        w, h = self.width, self.height

        # Trapezoid on the road plane, tuned for a forward-facing dashcam
        # where the horizon sits near the vertical middle of the frame.
        self.src = np.float32(
            [
                [w * 0.43, h * 0.62],  # top-left
                [w * 0.58, h * 0.62],  # top-right
                [w * 0.92, h * 0.95],  # bottom-right
                [w * 0.10, h * 0.95],  # bottom-left
            ]
        )

        # Destination rectangle (bird's-eye). Margins keep curved lanes
        # inside the frame as they bend.
        margin = w * 0.22
        self.dst = np.float32(
            [
                [margin, 0],
                [w - margin, 0],
                [w - margin, h],
                [margin, h],
            ]
        )

        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def warp(self, image):
        return cv2.warpPerspective(image, self.M, (self.width, self.height), flags=cv2.INTER_LINEAR)

    def unwarp(self, image):
        return cv2.warpPerspective(
            image, self.M_inv, (self.width, self.height), flags=cv2.INTER_LINEAR
        )

    def draw_src_region(self, image, color=(0, 200, 255)):
        """Overlay the source trapezoid for debugging/tuning."""
        pts = self.src.astype(np.int32).reshape((-1, 1, 2))
        out = image.copy()
        cv2.polylines(out, [pts], True, color, 2)
        return out
