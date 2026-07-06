# ---------------------------------------------------------------------------
# Developer convenience commands. Run `make help` to see everything.
# ---------------------------------------------------------------------------
.DEFAULT_GOAL := help
PYTHON ?= python

.PHONY: help install install-dev run test lint format typecheck check clean frontend

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev: ## Install the package with dev extras
	$(PYTHON) -m pip install -e ".[dev]"

run: ## Run lane detection (default webcam). Override: make run SOURCE=samples/highway_drive.mp4
	$(PYTHON) -m app --source $(or $(SOURCE),0)

test: ## Run the test suite with coverage
	$(PYTHON) -m pytest --cov=app --cov-report=term-missing

lint: ## Lint with ruff
	$(PYTHON) -m ruff check app tests

format: ## Auto-format with black + ruff
	$(PYTHON) -m black app tests examples
	$(PYTHON) -m ruff check --fix app tests

typecheck: ## Static type-check with mypy
	$(PYTHON) -m mypy app

check: lint typecheck test ## Run all quality gates

frontend: ## Start the dashboard dev server
	cd frontend && npm install && npm run dev

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
