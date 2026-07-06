# Legacy

`lane_detection_irl.py` is the original v1 detector: a single-file Canny +
Hough-transform pipeline that fits straight lines. It is kept here for
reference. The current system (colour/gradient threshold → perspective warp →
sliding-window polynomial fit) lives in `lane_detector/` at the repo root and
handles curved lanes, which the straight-line approach could not.
