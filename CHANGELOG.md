# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- Perspective (bird's-eye) transform for curved-lane fitting
- Polynomial lane fitting to replace straight-line Hough averaging
- WebAssembly build of the pipeline for in-browser inference

## [1.0.0] - 2026-07-06

### Added

- Modular Python package (`app/`) with a clean separation of concerns:
  configuration, camera I/O, detection, geometry, rendering, and metrics.
- Typed, validated configuration via dataclasses with YAML loading.
- Temporal smoothing with rolling history to eliminate overlay flicker.
- Full `pytest` suite covering config, geometry, utilities, and the pipeline.
- Modern React + Vite + TypeScript + Tailwind + Framer Motion dashboard.
- Comprehensive documentation set under `docs/`.
- CI workflows for linting, type-checking, and testing.
- Community health files: contributing guide, code of conduct, security policy,
  issue/PR templates.

### Changed

- Refactored the original single-file script into a maintainable package.
- Replaced blocking `input()` prompts with a proper `argparse` CLI (`python -m app`).

[Unreleased]: https://github.com/rishi/realtime-lane-detection/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/rishi/realtime-lane-detection/releases/tag/v1.0.0
