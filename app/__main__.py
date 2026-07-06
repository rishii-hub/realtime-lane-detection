"""Command-line entry point: ``python -m app``.

Examples
--------
Run against the default webcam::

    python -m app

Process a video file with a custom config::

    python -m app --source test3.mp4 --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys

from app import __version__
from app.config import PipelineConfig
from app.runner import DetectionApp
from app.utils import configure_logging


def _parse_source(raw: str) -> int | str:
    """Interpret ``0`` as a webcam index and anything else as a path/URL."""
    return int(raw) if raw.isdigit() else raw


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="app",
        description="Real-time lane detection using classical computer vision.",
    )
    parser.add_argument(
        "-s",
        "--source",
        default="0",
        help="Webcam index (e.g. 0), video file path, or IP camera URL.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="Path to a YAML pipeline configuration file.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(logging.DEBUG if args.verbose else logging.INFO)

    config = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig()
    app = DetectionApp(source=_parse_source(args.source), config=config)

    try:
        app.run()
    except KeyboardInterrupt:
        logging.getLogger(__name__).info("Interrupted by user.")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
