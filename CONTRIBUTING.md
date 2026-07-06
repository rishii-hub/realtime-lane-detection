# Contributing to Real-Time Lane Detection

First off — thank you for taking the time to contribute! 🎉 This project aims to
be a friendly, well-documented reference for classical computer-vision lane
detection, and contributions of every size are welcome.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Commit Convention](#commit-convention)
- [Pull Request Process](#pull-request-process)

## Code of Conduct

This project and everyone participating in it is governed by our
[Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold it.

## Ways to Contribute

- 🐛 **Report bugs** using the bug report issue template.
- 💡 **Suggest features** using the feature request template.
- 📖 **Improve documentation** — even fixing a typo helps.
- 🧪 **Add tests** to increase coverage.
- ⚡ **Tune the pipeline** — better ROI heuristics, smoothing, or performance.

## Development Setup

```bash
# 1. Fork & clone
git clone https://github.com/<you>/realtime-lane-detection.git
cd realtime-lane-detection

# 2. Install with dev tooling
make install-dev          # or: pip install -e ".[dev]"

# 3. Install the git hooks
pre-commit install

# 4. Run the quality gates
make check                # lint + typecheck + tests
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Project Structure

```
app/         Python package (pipeline, config, rendering)
frontend/    React + Vite + TypeScript dashboard
tests/       pytest suite
docs/        Deep-dive documentation
examples/    Runnable usage examples
```

See [docs/Architecture.md](docs/Architecture.md) for a detailed tour.

## Coding Standards

- **Python** follows [PEP 8](https://peps.python.org/pep-0008/), enforced by
  `black` + `ruff`. Public functions need type hints and docstrings.
- **TypeScript** is formatted with the project ESLint/Prettier config.
- Keep functions small and single-purpose; prefer composition over long methods.
- Every behavioural change should come with a test.

Run everything before pushing:

```bash
make format      # auto-fix formatting
make check       # lint, typecheck, test
```

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add adaptive ROI based on horizon estimation
fix: guard against zero-slope lane fits
docs: expand Hough transform explanation
test: cover smoothing dropout bridging
```

## Pull Request Process

1. Create a topic branch: `git checkout -b feat/my-feature`.
2. Make your change with tests and docs.
3. Ensure `make check` passes and pre-commit is clean.
4. Open a PR using the template and link any related issues.
5. A maintainer will review — please be responsive to feedback.

Thanks again for helping make this project better! 💚
