# Performance

The pipeline is designed to run comfortably in real time on a CPU — no GPU, no
neural network, no cloud. This page explains where the time goes and how to make
it faster.

## Reference numbers

Measured on a mid-range laptop CPU (Intel i5, 640×480 input). Treat these as
ballpark figures — your mileage will vary with resolution and hardware.

| Stage                | Typical time | Share  |
| -------------------- | ------------ | ------ |
| Pre-processing       | ~0.4 ms      | 4 %    |
| Edge detection       | ~3.1 ms      | 33 %   |
| ROI masking          | ~0.5 ms      | 5 %    |
| Hough transform      | ~4.2 ms      | 45 %   |
| Estimation + smooth  | ~0.6 ms      | 6 %    |
| Rendering            | ~0.7 ms      | 7 %    |
| **Total**            | **~9.5 ms**  | 100 %  |

That's roughly **90–120 FPS** of headroom on the algorithm itself; real-world
throughput is usually bounded by camera capture and display, not compute.

## Why it's fast

- **Classical CV, not deep learning** — every operation is a well-optimised
  OpenCV kernel running on the CPU.
- **Aggressive ROI** — masking discards ~70 % of edge pixels before the
  (expensive) Hough transform ever sees them.
- **Downscaling** — wide frames are shrunk to `max_processing_width` up front.
- **Probabilistic Hough** — samples edge points instead of exhaustively
  transforming all of them.

## Measuring it yourself

`MetricsTracker` records per-frame timing; the HUD shows a rolling average FPS
and latency. Programmatically:

```python
from app import LaneDetector, PipelineConfig

detector = LaneDetector(PipelineConfig())
result = detector.process(frame)
print(result.metrics.fps, result.metrics.latency_ms)
```

## Tuning for speed vs. quality

| Want…            | Do this                                                        |
| ---------------- | ------------------------------------------------------------- |
| **More FPS**     | Lower `max_processing_width`; raise `hough_threshold`.         |
| **Smoother**     | Increase `smoothing_window` (adds a few ms of latency).       |
| **More accuracy**| Increase capture resolution; widen the Canny band.            |

## Scaling roadmap

The current bottleneck (Hough) and the straight-line model are the two obvious
next targets:

- Replace straight-line averaging with a **polynomial fit** over a bird's-eye
  (perspective-warped) view to handle curves.
- Port the hot path to **WebAssembly** so the same algorithm runs in the
  dashboard with no server round-trip.
