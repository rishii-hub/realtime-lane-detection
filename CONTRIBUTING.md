# Contributing to LaneVision

Thanks for your interest in improving LaneVision. This is a computer-vision
portfolio project, but contributions of any size are welcome.

## Getting set up

```bash
git clone https://github.com/rishii-hub/realtime-lane-detection.git
cd realtime-lane-detection

# Python side
pip install -e ".[dev]"

# Frontend side
cd frontend && npm install && cd ..
```

## Before you open a pull request

Run the full quality gate locally — CI runs the same checks:

```bash
make test      # pytest
make lint      # ruff + eslint
make format    # black + ruff --fix (then review the diff)
```

All of these must pass. New behaviour should come with a test.

## Project layout

| Path              | What lives there                                        |
| ----------------- | ------------------------------------------------------- |
| `lane_detector/`  | The detection pipeline (threshold, warp, fit, render)   |
| `app.py`          | FastAPI server (MJPEG stream + telemetry)               |
| `cli.py`          | Desktop / headless runner                               |
| `frontend/`       | React + TypeScript dashboard                            |
| `tests/`          | pytest suite                                            |
| `docs/`           | Architecture and pipeline notes                         |

## Coding conventions

- **Python**: formatted with `black` (line length 100), linted with `ruff`,
  type hints on public functions, docstrings explaining *why* not just *what*.
- **TypeScript**: `strict` mode, no `any`, components in PascalCase files with
  a co-located `.module.css`.
- Keep functions small and single-purpose; prefer clarity over cleverness.

## Commit messages

Use present-tense, imperative summaries (e.g. "Add curvature sanity check").
Conventional-commit prefixes (`feat:`, `fix:`, `docs:`, `test:`) are welcome
but not required.

## Reporting bugs / requesting features

Open an issue using the templates under `.github/ISSUE_TEMPLATE/`. For bugs,
include your OS, Python version, and a sample frame or clip if you can.
