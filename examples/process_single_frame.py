"""Example: run the pipeline on a single image and save the annotated result.

This is handy for debugging detection parameters without a live camera::

    python examples/process_single_frame.py path/to/frame.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

from app.config import PipelineConfig
from app.detector import LaneDetector
from app.utils import configure_logging


def main(image_path: str) -> None:
    configure_logging()
    frame = cv2.imread(image_path)
    if frame is None:
        raise SystemExit(f"Could not read image: {image_path}")

    detector = LaneDetector(PipelineConfig())
    result = detector.process(frame)

    out = Path(image_path).with_name("annotated.png")
    cv2.imwrite(str(out), result.annotated)
    print(f"Saved annotated frame → {out}")
    print(f"Detected lanes: {[lane.side.value for lane in result.lanes.lanes]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python examples/process_single_frame.py <image>")
    main(sys.argv[1])
