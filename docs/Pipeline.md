# The Detection Pipeline

The pipeline turns a raw camera frame into an annotated lane overlay in a fixed
sequence of stages. Each stage is a small, testable method on `LaneDetector`.

```
 ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐
 │  Capture │─▶ │ Pre-proc  │─▶ │   Edges  │─▶ │    ROI    │─▶ │  Hough   │─▶ │  Estimate  │─▶ │  Render  │
 │  frame   │   │ downscale │   │ Canny+CL │   │ trapezoid │   │ segments │   │ + smooth   │   │ overlay  │
 └──────────┘   └───────────┘   └──────────┘   └───────────┘   └──────────┘   └────────────┘   └──────────┘
```

## Stage 1 — Capture

`VideoSource` reads BGR frames from a webcam, video file, or IP camera. Capture
resolution and FPS hints come from `CameraConfig`.

## Stage 2 — Pre-processing

Wide frames are linearly downscaled so the widest dimension is at most
`camera.max_processing_width` (default 800 px). This keeps the pipeline
real-time on modest hardware without materially hurting accuracy.

## Stage 3 — Edge detection

```
grayscale → Gaussian blur → CLAHE → Canny
```

- **Grayscale** collapses colour; lane edges are intensity discontinuities.
- **Gaussian blur** (5×5) suppresses sensor noise that would create spurious edges.
- **CLAHE** (Contrast Limited Adaptive Histogram Equalisation) locally boosts
  contrast so faint lane markings survive shadows and glare.
- **Canny** produces a clean binary edge map.

See [HowDetectionWorks.md](HowDetectionWorks.md) for the theory.

## Stage 4 — Region of interest (ROI)

Most of the frame (sky, dashboard, roadside) contains no useful lane
information. A trapezoidal mask keeps only the drivable area ahead of the
vehicle. The trapezoid is defined as fractions of the frame in `DetectionConfig`
so it adapts to any resolution.

## Stage 5 — Hough transform

`cv2.HoughLinesP` converts edge pixels into straight line **segments**. The
probabilistic variant is faster than the standard transform and returns concrete
endpoints, which is exactly what we need for drawing.

## Stage 6 — Estimation (geometry)

`LaneEstimator` turns dozens of noisy segments into two clean boundaries:

1. **Slope + intercept** are computed for each segment.
2. **Slope filtering** discards near-horizontal and near-vertical clutter.
3. **Left/right classification** uses slope sign and horizontal position.
4. **Median fitting** collapses each group into one robust line (median resists
   outliers far better than mean).
5. **Temporal smoothing** blends the result with a short rolling history so the
   overlay stays stable frame-to-frame and survives momentary dropouts.

## Stage 7 — Rendering

`LaneRenderer` fills the detected lane region, draws the boundary lines,
computes the vehicle's lateral offset from lane centre, and overlays a HUD with
FPS, latency, and frame count.

## Output

The stages are bundled into an immutable `DetectionResult`:

```python
@dataclass(frozen=True)
class DetectionResult:
    frame: np.ndarray       # pre-processed input
    edges: np.ndarray       # Canny edge map
    lanes: LaneFrame        # smoothed left/right boundaries
    metrics: FrameMetrics   # FPS / latency / frame count
    annotated: np.ndarray   # final overlay for display
```
