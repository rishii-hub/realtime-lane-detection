# Architecture

LaneVision is split into three cleanly separated layers: a **detection package**
(pure computer vision, no I/O), a **thin web/CLI layer** that feeds frames in and
renders results out, and a **React dashboard** that talks to the web layer over
HTTP.

```
┌────────────────────────────────────────────────────────────────┐
│  frontend/  (React + TypeScript, Vite)                          │
│  • MJPEG <img> for the live feed                                │
│  • polls /api/metrics for telemetry                             │
│  • POSTs view-mode / source / upload changes                    │
└───────────────▲───────────────────────────┬────────────────────┘
                │ HTTP / MJPEG              │ control
┌───────────────┴───────────────────────────▼────────────────────┐
│  app.py  (FastAPI)                     cli.py  (desktop/headless)│
│  • owns the VideoCapture + LaneDetector                          │
│  • streams annotated JPEGs, exposes JSON telemetry               │
└───────────────────────────▲────────────────────────────────────┘
                            │ frame in / annotated frame out
┌───────────────────────────┴────────────────────────────────────┐
│  lane_detector/  (pure CV, framework-agnostic)                  │
│                                                                 │
│   thresholding.py  colour + gradient → binary lane mask         │
│   perspective.py   bird's-eye warp / unwarp                     │
│   lane_fit.py      sliding-window search, polyfit, curvature    │
│   detector.py      orchestration, smoothing, rendering, HUD     │
└─────────────────────────────────────────────────────────────────┘
```

## Why this separation

- **The `lane_detector` package has no web or GUI dependencies.** It takes a
  NumPy BGR frame and returns a NumPy BGR frame. That makes it trivial to test
  (see `tests/`), reuse from the CLI, or drop into a different host later.
- **`app.py` and `cli.py` are interchangeable front-doors.** Both construct a
  `LaneDetector` and pump frames through it; neither contains any CV logic.
- **The React app never touches Python.** It only knows the HTTP contract:
  an MJPEG stream and a small JSON telemetry object.

## State and threading

The detector keeps a little state between frames (the previous polynomial fit
and short smoothing buffers) so it can use the fast "search around the previous
fit" path and damp jitter. In the web server this state lives inside a single
`StreamState` guarded by a lock, because the MJPEG generator and the control
endpoints run on different threads.

## The HTTP contract

| Endpoint         | Method | Purpose                                        |
| ---------------- | ------ | ---------------------------------------------- |
| `/api/stream`    | GET    | `multipart/x-mixed-replace` MJPEG of results   |
| `/api/metrics`   | GET    | JSON: curvature, offset, status, fps, conf.    |
| `/api/view`      | POST   | Switch view mode (final/threshold/birdseye/roi)|
| `/api/source`    | POST   | Switch to demo clip or webcam                  |
| `/api/upload`    | POST   | Upload a video file to process                 |

See [PIPELINE.md](PIPELINE.md) for how a single frame becomes a detection.
