"""
LaneVision web server.

FastAPI backend that runs the lane-detection pipeline and streams annotated
frames to the browser as MJPEG, while exposing live telemetry as JSON.

Run:
    pip install -r requirements.txt
    python app.py
    # open http://localhost:8000

Endpoints:
    GET  /                -> dashboard UI
    GET  /api/stream      -> multipart MJPEG of processed frames
    GET  /api/metrics     -> live telemetry (curvature, offset, fps, status)
    POST /api/view        -> set view mode (final|threshold|birdseye|roi)
    POST /api/source      -> switch to 'demo' or 'webcam'
    POST /api/upload      -> upload a video file to process
"""

import os
import threading
import time

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from lane_detector import LaneDetector

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_VIDEO = os.path.join(BASE_DIR, "test3.mp4")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

FRAME_SIZE = (640, 360)


class StreamState:
    """Shared, thread-safe state driving the video stream."""

    def __init__(self):
        self.lock = threading.Lock()
        self.detector = LaneDetector(frame_size=FRAME_SIZE)
        self.view_mode = "final"
        self.source = DEMO_VIDEO
        self.cap = cv2.VideoCapture(self.source)
        self.is_file = True

    def set_view(self, mode):
        with self.lock:
            self.view_mode = mode

    def set_source(self, source, is_file):
        with self.lock:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(source)
            self.source = source
            self.is_file = is_file
            self.detector = LaneDetector(frame_size=FRAME_SIZE)

    def read(self):
        """Return the next frame, looping demo/uploaded files at EOF."""
        with self.lock:
            ret, frame = self.cap.read()
            if not ret:
                if self.is_file:  # loop the clip
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                if not ret:
                    return None
            return frame

    def process(self, frame):
        with self.lock:
            return self.detector.process(frame, self.view_mode), self.detector.metrics()


state = StreamState()
latest_metrics = {
    "status": "INITIALISING",
    "fps": 0.0,
    "curvature_m": None,
    "offset_m": None,
    "confidence": 0.0,
}

app = FastAPI(title="LaneVision")


def mjpeg_generator():
    global latest_metrics
    target_dt = 1 / 30
    while True:
        t0 = time.time()
        frame = state.read()
        if frame is None:
            time.sleep(0.05)
            continue
        output, metrics = state.process(frame)
        latest_metrics = metrics
        ok, buffer = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        # pace to ~30 fps so playback speed matches real time
        elapsed = time.time() - t0
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)


@app.get("/api/stream")
def stream():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/metrics")
def metrics():
    return JSONResponse(latest_metrics)


@app.post("/api/view")
def set_view(payload: dict):
    mode = payload.get("mode", "final")
    if mode in ("final", "threshold", "birdseye", "roi"):
        state.set_view(mode)
        return {"ok": True, "mode": mode}
    return JSONResponse({"ok": False, "error": "invalid mode"}, status_code=400)


@app.post("/api/source")
def set_source(payload: dict):
    src = payload.get("source", "demo")
    if src == "demo":
        state.set_source(DEMO_VIDEO, is_file=True)
    elif src == "webcam":
        state.set_source(0, is_file=False)
    else:
        return JSONResponse({"ok": False, "error": "unknown source"}, status_code=400)
    return {"ok": True, "source": src}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    dest = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest, "wb") as f:
        f.write(await file.read())
    state.set_source(dest, is_file=True)
    return {"ok": True, "filename": file.filename}


# Serve the built React (Vite) frontend from static/dist.
DIST_DIR = os.path.join(BASE_DIR, "static", "dist")


@app.get("/")
def index():
    index_html = os.path.join(DIST_DIR, "index.html")
    if os.path.exists(index_html):
        return FileResponse(index_html)
    return JSONResponse(
        {"error": "Frontend not built. Run: cd frontend && npm install && npm run build"},
        status_code=503,
    )


if os.path.isdir(os.path.join(DIST_DIR, "assets")):
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(DIST_DIR, "assets")),
        name="assets",
    )


if __name__ == "__main__":
    import uvicorn

    # Hosted platforms (Render, Railway, ...) inject the port via $PORT.
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
