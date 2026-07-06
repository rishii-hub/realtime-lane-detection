# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0]

### Added
- **Curve-following detection.** New pipeline: colour + gradient thresholding →
  bird's-eye perspective warp → sliding-window search → 2nd-degree polynomial
  fit. Tracks curved lanes instead of only straight lines.
- Real-world **radius of curvature** and **vehicle offset** metrics, plus a
  lane-departure status.
- **React + TypeScript dashboard** (Vite) with a live MJPEG feed, telemetry
  panel, a lane-position gauge, view-mode toggles, and video upload.
- **FastAPI backend** streaming annotated frames and serving JSON telemetry.
- Full test suite (`pytest`), linting/formatting config (`ruff`, `black`,
  `eslint`), CI workflow, and community health files.

### Changed
- Restructured the single-file script into a `lane_detector/` package with
  separated threshold / perspective / fit / detector modules.

### Preserved
- The original v1 straight-line detector is kept under `legacy/` for reference.

## [1.0.0]

### Added
- Initial single-file detector: Canny edge detection + Hough transform +
  temporal smoothing, visualised with an OpenCV window.
