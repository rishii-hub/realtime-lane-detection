"""Minimal example: run lane detection on the default webcam.

Run with::

    python examples/run_webcam.py
"""

from __future__ import annotations

from app.config import PipelineConfig
from app.runner import DetectionApp
from app.utils import configure_logging


def main() -> None:
    configure_logging()
    DetectionApp(source=0, config=PipelineConfig()).run()


if __name__ == "__main__":
    main()
