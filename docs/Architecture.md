# Architecture

This document explains how the codebase is organised and why. The guiding
principle is **separation of concerns**: each module does one thing, has a
narrow interface, and can be tested in isolation.

## High-level view

```
                ┌─────────────────────────────────────────────┐
                │                  app (package)               │
                │                                              │
  video ──▶ VideoSource ──▶ LaneDetector ──▶ DetectionResult ──┼──▶ UI / file
                │              │    │                           │
                │              │    ├─ LaneEstimator (geometry) │
                │              │    ├─ LaneRenderer  (drawing)   │
                │              │    └─ MetricsTracker (timing)   │
                │              │                                 │
                │        PipelineConfig (typed settings)        │
                └─────────────────────────────────────────────┘
```

## Modules

| Module                 | Responsibility                                                        |
| ---------------------- | --------------------------------------------------------------------- |
| `app/config.py`        | Immutable, validated dataclasses for every tunable parameter.         |
| `app/camera.py`        | Context-managed `VideoSource` wrapping `cv2.VideoCapture`.            |
| `app/detector.py`      | Orchestrates the imaging stages → produces a `DetectionResult`.       |
| `app/lane.py`          | Pure geometry: classification, slope filtering, fitting, smoothing.   |
| `app/visualization.py` | Renders lane overlays and the heads-up display.                       |
| `app/metrics.py`       | Tracks FPS, latency, and frame counts.                                |
| `app/runner.py`        | Interactive OpenCV window + keyboard controls.                        |
| `app/utils.py`         | Small shared helpers (logging, clamping, scaling).                    |
| `app/__main__.py`      | `argparse` CLI entry point (`python -m app`).                         |

## Design decisions

### 1. Configuration as data, not constants

Every magic number lives in a frozen dataclass in `config.py` with validation in
`__post_init__`. This makes experiments reproducible (dump to dict / load from
YAML) and impossible to misconfigure silently.

### 2. Geometry is decoupled from imaging

`LaneEstimator` never touches OpenCV — it operates purely on line segments and
frame dimensions. That means the trickiest logic (slope filtering, temporal
smoothing) is unit-tested with plain NumPy arrays, no camera required.

### 3. The detector is I/O-free

`LaneDetector.process()` takes a frame and returns a `DetectionResult`. It reads
nothing and writes nothing. All I/O (windows, files, cameras) lives in
`runner.py` and `camera.py`, so the core is trivially embeddable in notebooks,
tests, or a future web service.

### 4. Immutability by default

Config objects and geometry primitives (`Lane`, `LaneFrame`, `DetectionResult`)
are frozen dataclasses. State that *must* mutate (smoothing history, metrics) is
confined to a couple of clearly-named stateful classes.

## Data flow

1. `VideoSource.read()` yields a BGR frame.
2. `LaneDetector.process()`:
   - `_preprocess` downscales wide frames.
   - `_detect_edges` runs grayscale → blur → CLAHE → Canny.
   - `_apply_roi` masks a trapezoidal region.
   - `_hough_lines` extracts probabilistic Hough segments.
   - `LaneEstimator.estimate` classifies, fits, and smooths them.
   - `MetricsTracker.record` timestamps the frame.
   - `LaneRenderer.render` draws the annotated output.
3. The `DetectionResult` bundles frame, edges, lanes, metrics, and annotation.

See [Pipeline.md](Pipeline.md) for the stage-by-stage image-processing detail.
