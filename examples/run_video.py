"""Example: process the bundled sample highway clip.

Run with::

    python examples/run_video.py
"""

from __future__ import annotations

from pathlib import Path

from app.config import PipelineConfig
from app.runner import DetectionApp
from app.utils import configure_logging

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "highway_drive.mp4"


def main() -> None:
    configure_logging()
    if not SAMPLE.exists():
        raise SystemExit(f"Sample clip not found at {SAMPLE}")
    DetectionApp(source=str(SAMPLE), config=PipelineConfig()).run()


if __name__ == "__main__":
    main()
