# LaneVision — developer tasks
.PHONY: help install install-dev run cli bench frontend build test lint format clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install runtime dependencies
	pip install -r requirements.txt

install-dev: ## Install runtime + dev dependencies
	pip install -e ".[dev]"

run: ## Start the web dashboard (http://localhost:8000)
	python app.py

cli: ## Run the desktop/CLI detector on the demo clip
	python cli.py --source demo

bench: ## Headless benchmark (detection rate + fps)
	python cli.py --source demo --benchmark

frontend: ## Start the Vite dev server (proxies to the backend)
	cd frontend && npm run dev

build: ## Build the frontend into static/dist
	cd frontend && npm install && npm run build

test: ## Run the Python test suite
	pytest

lint: ## Lint Python (ruff) and frontend (eslint)
	ruff check lane_detector app.py cli.py tests
	cd frontend && npm run lint

format: ## Auto-format Python with black + ruff
	black lane_detector app.py cli.py tests
	ruff check --fix lane_detector app.py cli.py tests

clean: ## Remove caches and build artifacts
	rm -rf __pycache__ */__pycache__ .pytest_cache .ruff_cache uploads
	rm -rf frontend/node_modules frontend/dist static/dist
