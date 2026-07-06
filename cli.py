"""
Command-line runner for the lane detector (desktop OpenCV window).

Prefer the web dashboard (`python app.py`) for the full experience; this CLI is
handy for quick local testing or headless benchmarking.

Usage:
    python cli.py                       # bundled demo clip
    python cli.py --source 0            # webcam
    python cli.py --source path/to.mp4  # a video file
    python cli.py --source demo --benchmark   # measure detection rate, no window

Controls (windowed mode):
    q  quit        1 detection view     3 bird's-eye
    p  pause       2 threshold mask     4 warp region
"""

import argparse
import time

import cv2

from lane_detector import LaneDetector

VIEW_KEYS = {ord("1"): "final", ord("2"): "threshold", ord("3"): "birdseye", ord("4"): "roi"}


def resolve_source(source):
    if source in (None, "demo"):
        return "test3.mp4"
    if source.isdigit():
        return int(source)
    return source


def run(source, benchmark=False):
    cap = cv2.VideoCapture(resolve_source(source))
    if not cap.isOpened():
        print(f"Error: could not open source '{source}'")
        return

    detector = LaneDetector(frame_size=(640, 360))
    view_mode, paused = "final", False
    frames, locked = 0, 0
    t_start = time.time()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            output = detector.process(frame, view_mode)
            frames += 1
            if detector.status not in ("SEARCHING", "INITIALISING"):
                locked += 1

        if benchmark:
            if frames % 100 == 0:
                print(
                    f"  {frames} frames | lock {100*locked/max(frames,1):.0f}% "
                    f"| {detector.metrics()['fps']:.0f} fps"
                )
            continue

        cv2.imshow("LaneVision", output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
        elif key in VIEW_KEYS:
            view_mode = VIEW_KEYS[key]

    cap.release()
    cv2.destroyAllWindows()
    dur = time.time() - t_start
    print(
        f"\nProcessed {frames} frames in {dur:.1f}s "
        f"({frames/max(dur,1e-6):.1f} fps) | lane-locked {100*locked/max(frames,1):.0f}%"
    )


def main():
    ap = argparse.ArgumentParser(description="LaneVision CLI")
    ap.add_argument(
        "--source", default="demo", help="'demo', webcam index (e.g. 0), or a video path"
    )
    ap.add_argument(
        "--benchmark", action="store_true", help="run headless and report detection rate / fps"
    )
    args = ap.parse_args()
    run(args.source, args.benchmark)


if __name__ == "__main__":
    main()
