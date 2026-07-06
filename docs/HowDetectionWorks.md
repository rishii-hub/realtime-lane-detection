# How Detection Works

A friendly, from-first-principles tour of the classical computer-vision
techniques behind the pipeline. No deep learning required — just geometry and
gradients.

## 1. Canny Edge Detection

Lane markings are bright lines on a darker road: sharp changes in brightness.
The [Canny edge detector](https://en.wikipedia.org/wiki/Canny_edge_detector)
finds exactly these changes in four steps:

1. **Noise reduction** — a Gaussian blur smooths the image so tiny fluctuations
   don't register as edges.
2. **Gradient computation** — Sobel operators estimate the intensity gradient
   (magnitude and direction) at every pixel.
3. **Non-maximum suppression** — thin the thick gradient ridges down to
   single-pixel-wide edges.
4. **Hysteresis thresholding** — two thresholds (`canny_low`, `canny_high`)
   classify edges as strong, weak, or noise; weak edges are kept only if they
   connect to strong ones.

> **Why CLAHE first?** Real roads have shadows, glare, and worn paint. Contrast
> Limited Adaptive Histogram Equalisation locally stretches contrast so faint
> markings still produce a strong gradient.

## 2. Region of Interest (ROI) Masking

After edge detection we still have edges from trees, cars, signs, and the
horizon. But lanes only appear in a predictable trapezoid ahead of the vehicle.

We build a binary mask of that trapezoid and `bitwise_and` it with the edge map,
zeroing out everything outside. Because the trapezoid is defined as *fractions*
of the frame, it works at any resolution:

```
(0.05·W, H) ────────────── (0.95·W, H)      ← bottom, full width
        \                    /
         \                  /
   (0.40·W, 0.6·H) ── (0.60·W, 0.6·H)        ← horizon line
```

## 3. Hough Transform

We now have edge *pixels*, but we want *lines*. The
[Hough transform](https://en.wikipedia.org/wiki/Hough_transform) is a voting
scheme: every edge pixel votes for all the lines that could pass through it, in
polar `(ρ, θ)` space. Lines that collect enough votes are real.

We use the **probabilistic** variant (`HoughLinesP`), which:

- is faster (samples edge points instead of using all of them), and
- returns concrete segment endpoints `(x1, y1, x2, y2)` — perfect for drawing.

Key knobs (in `DetectionConfig`):

| Parameter          | Meaning                                              |
| ------------------ | ---------------------------------------------------- |
| `hough_threshold`  | Minimum votes for a line to count                    |
| `min_line_length`  | Reject segments shorter than this                    |
| `max_line_gap`     | Bridge collinear segments separated by small gaps    |

## 4. Slope Filtering & Line Averaging

Hough returns many short, noisy segments. We turn them into two clean lanes:

**Slope filtering.** For a segment, `slope = (y2 - y1) / (x2 - x1)`. We discard:

- **near-horizontal** segments (`|slope| < min_slope`) — usually shadows or cars,
- **near-vertical** segments (`|slope| > max_slope`) — usually poles or noise.

**Classification.** In image coordinates *y grows downward*, so:

- the **left** lane has a **negative** slope and lives on the left half,
- the **right** lane has a **positive** slope and lives on the right half.

**Median averaging.** Each group is collapsed into a single `(slope, intercept)`
using the **median**, which ignores outliers far better than the mean. We then
project that line between the bottom of the frame and the horizon.

## 5. Temporal Smoothing

A single frame's estimate can jitter or briefly vanish. We keep a short rolling
buffer (`smoothing_window` frames) per side and render the **average**. Benefits:

- **Stability** — the overlay stops flickering.
- **Robustness** — if a frame detects nothing, we fall back to recent history so
  the lane doesn't disappear for a split second.

This is a lightweight, causal filter — no future frames needed, so it stays
real-time.

## Putting it together

```
edges → ROI → Hough → filter slopes → classify L/R → median fit → smooth → draw
```

That's the entire algorithm. It's fast, interpretable, and has zero training
data — which is exactly why classical CV is still a great teaching tool and a
solid baseline for lane-keeping assistance.
