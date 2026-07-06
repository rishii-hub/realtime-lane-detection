# The detection pipeline

Every frame goes through the same five stages. This document explains what each
one does and, more importantly, *why* — because the "why" is what separates this
from a naive edge detector.

![Pipeline stages](pipeline.png)

## 1. Thresholding — find lane *pixels*, not just edges

`lane_detector/thresholding.py`

A naive detector runs Canny and treats every strong edge as a lane candidate.
The problem: shadows, tar seams, cracks, and other vehicles all produce strong
edges too. Lane markings have a more specific property — they are **white or
yellow**. So we threshold on colour:

- **White markings** via the **lightness (L) channel of HLS**, with CLAHE
  (contrast-limited adaptive histogram equalisation) so detection stays stable
  when part of the road is in bright sun and part in shade.
- **Yellow markings** via the **b-channel of LAB**, which separates
  yellow-vs-blue and is robust to overall brightness changes.
- A **Sobel-x gradient** term reinforces near-vertical line structure.

The union of these, lightly morphologically closed to bridge dashed-line gaps,
gives a clean binary mask where lane pixels are white.

## 2. Perspective warp — get a bird's-eye view

`lane_detector/perspective.py`

Fitting a curve to lane pixels is far easier from directly above, where lane
lines become roughly parallel and vertical. We define a trapezoid on the road
plane (source) and map it to a rectangle (destination) with
`cv2.getPerspectiveTransform`, then `warpPerspective` the binary mask into that
top-down space. The inverse matrix is kept so we can project the result back
onto the driver's view later.

Source points are expressed as **fractions of frame size**, so the same
configuration works at any input resolution.

## 3. Sliding-window search — collect pixels per lane

`lane_detector/lane_fit.py`

On the first frame (or after a lost track) we locate the two lanes with a
**histogram of the bottom half** of the warped mask: the two tallest peaks are
the lane bases. From each base we slide a stack of windows upward, re-centring
each window on the mean x of the pixels it captures. This walks the window up a
curve instead of assuming a straight line.

On subsequent frames we skip the histogram and simply **search around the
previous frame's polynomial** — cheaper and more stable.

## 4. Polynomial fit — follow the curve

For each lane we fit a **2nd-degree polynomial** `x = a·y² + b·y + c` to the
collected pixels. The quadratic term is what lets the detected lane bend with
the road. A straight line is just the special case where `a ≈ 0`.

## 5. Metrics + rendering

From the polynomials we compute, in real-world units:

- **Radius of curvature** — evaluated at the vehicle (bottom of the frame),
  after rescaling pixels to metres. A straight road yields a very large radius;
  a tight curve yields a few hundred metres.
- **Vehicle offset** — the horizontal distance between the lane centre and the
  image centre, signed so positive means the car sits right of centre. When the
  magnitude exceeds a threshold, the status becomes a **lane-departure** warning.

Finally the filled lane polygon is **unwarped** back to the driver's perspective
and blended onto the original frame, and a HUD prints the live numbers.

## Robustness details

- **Sanity checks.** A fit is rejected if the lane width is implausible or the
  two lines cross. A rejected frame falls back to a fresh sliding-window search.
- **Temporal smoothing.** Accepted fits are averaged over a short rolling buffer,
  so the overlay doesn't jitter frame to frame.

## Limitations

This is classical CV, not a learned model — fast and interpretable, but it
assumes visible markings and reasonable lighting. Heavy rain, snow cover, or
missing lane paint will degrade it. The perspective trapezoid and metres-per-
pixel scaling are tuned for a forward-facing dashcam; a different mounting needs
those constants re-tuned.
