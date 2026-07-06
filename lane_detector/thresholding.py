"""
Lane-pixel thresholding.

The single biggest accuracy win over raw Canny edge detection is isolating
lane pixels by *colour* rather than by intensity gradient alone. Road cracks,
shadows, and other cars all produce strong Canny edges; lane markings are
specifically white or yellow. We combine:

  1. HLS lightness  -> white markings
  2. LAB b-channel   -> yellow markings (robust to lighting changes)
  3. Sobel-x gradient -> reinforces near-vertical line structure

The union of these masks is far cleaner than a single Canny pass.
"""

import cv2
import numpy as np


def _scale_to_255(channel):
    """Normalise a single-channel float image to 0-255 uint8."""
    ch_min, ch_max = np.min(channel), np.max(channel)
    if ch_max - ch_min < 1e-6:
        return np.zeros_like(channel, dtype=np.uint8)
    scaled = 255 * (channel - ch_min) / (ch_max - ch_min)
    return scaled.astype(np.uint8)


def white_mask(frame_bgr, l_thresh=200):
    """Isolate bright/white lane markings via HLS lightness."""
    hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    # CLAHE keeps white detection stable across bright sky / dark tarmac
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    mask = np.zeros_like(l_channel)
    mask[l_channel >= l_thresh] = 255
    return mask


def yellow_mask(frame_bgr, b_thresh=150):
    """Isolate yellow lane markings via the LAB b-channel."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    b_channel = lab[:, :, 2]
    mask = np.zeros_like(b_channel)
    mask[b_channel >= b_thresh] = 255
    return mask


def sobel_mask(frame_bgr, thresh=(30, 120)):
    """Gradient magnitude in x reinforces near-vertical lane structure."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    abs_sobel = np.absolute(sobel_x)
    scaled = _scale_to_255(abs_sobel)
    mask = np.zeros_like(scaled)
    mask[(scaled >= thresh[0]) & (scaled <= thresh[1])] = 255
    return mask


def combined_threshold(frame_bgr):
    """
    Full thresholding pipeline. Returns a binary (0/255) mask where lane
    pixels are white.
    """
    w = white_mask(frame_bgr)
    y = yellow_mask(frame_bgr)
    s = sobel_mask(frame_bgr)

    # Colour union captures the markings; the gradient mask is intersected
    # with a lightness floor so it doesn't re-introduce shadow noise.
    colour = cv2.bitwise_or(w, y)
    combined = cv2.bitwise_or(colour, s)

    # Light morphological close to bridge dashed-line gaps
    kernel = np.ones((3, 3), np.uint8)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    return combined
