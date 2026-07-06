import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEFAULT_METRICS,
  DEFAULT_SETTINGS,
  type DetectionSettings,
  type EngineStatus,
  type Metrics,
  type SourceKind,
} from "../types";
import { renderFrame, type OverlayState } from "../lib/overlay";
import { clamp, lerp } from "../lib/format";

interface EngineApi {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  status: EngineStatus;
  source: SourceKind;
  metrics: Metrics;
  settings: DetectionSettings;
  hasMedia: boolean;
  error: string | null;
  start: () => void;
  pause: () => void;
  reset: () => void;
  loadFile: (file: File) => void;
  startWebcam: () => Promise<void>;
  updateSettings: (patch: Partial<DetectionSettings>) => void;
}

/**
 * Drives the live preview: owns the animation loop, real FPS timing, the canvas
 * overlay, and the derived metrics that feed the dashboard.
 */
export function useDetectionEngine(): EngineApi {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rafRef = useRef<number>(0);
  const streamRef = useRef<MediaStream | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const overlay = useRef<OverlayState>({
    deviation: 0,
    confidence: 0,
    fps: 0,
    frameCount: 0,
  });
  const fpsWindow = useRef<number[]>([]);
  const lastTs = useRef<number>(0);
  const lastMetricsPush = useRef<number>(0);
  const frames = useRef<number>(0);

  const [status, setStatus] = useState<EngineStatus>("idle");
  const [source, setSource] = useState<SourceKind>("idle");
  const [metrics, setMetrics] = useState<Metrics>(DEFAULT_METRICS);
  const [settings, setSettings] = useState<DetectionSettings>(DEFAULT_SETTINGS);
  const [hasMedia, setHasMedia] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const statusRef = useRef(status);
  statusRef.current = status;

  const sizeCanvas = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    const w = video.videoWidth || 960;
    const h = video.videoHeight || 540;
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }, []);

  const loop = useCallback(
    (ts: number) => {
      rafRef.current = requestAnimationFrame(loop);
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      sizeCanvas();

      // Real FPS from frame delta.
      const dt = lastTs.current ? ts - lastTs.current : 16;
      lastTs.current = ts;
      const instFps = clamp(1000 / dt, 0, 240);
      const win = fpsWindow.current;
      win.push(instFps);
      if (win.length > 30) win.shift();

      // Evolve simulated lane state smoothly (slow sway + gentle noise).
      const t = ts / 1000;
      const target =
        Math.sin(t * 0.5) * 0.5 + Math.sin(t * 1.3) * 0.15 + (Math.random() - 0.5) * 0.05;
      const s = settingsRef.current;
      overlay.current.deviation = lerp(overlay.current.deviation, target, 0.05);
      const targetConf = clamp(0.7 + s.sensitivity * 0.25 - Math.abs(target) * 0.2, 0, 0.99);
      overlay.current.confidence = lerp(overlay.current.confidence, targetConf, 0.04);

      frames.current += 1;
      overlay.current.fps = instFps;
      overlay.current.frameCount = frames.current;
      renderFrame(ctx, video, s, overlay.current);

      // Throttle React state updates to ~12 Hz.
      if (ts - lastMetricsPush.current > 80) {
        lastMetricsPush.current = ts;
        const avg = win.reduce((a, b) => a + b, 0) / win.length;
        setMetrics({
          fps: instFps,
          avgFps: avg,
          confidence: overlay.current.confidence,
          frameCount: frames.current,
          latencyMs: 1000 / clamp(avg, 1, 240),
          deviationPx: Math.round(overlay.current.deviation * 80),
        });
      }
    },
    [sizeCanvas],
  );

  const play = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    lastTs.current = 0;
    rafRef.current = requestAnimationFrame(loop);
  }, [loop]);

  const start = useCallback(() => {
    if (!hasMedia) return;
    const video = videoRef.current;
    if (video && source === "upload") void video.play();
    setStatus("running");
    play();
  }, [hasMedia, source, play]);

  const pause = useCallback(() => {
    setStatus("paused");
    cancelAnimationFrame(rafRef.current);
    if (source === "upload") videoRef.current?.pause();
  }, [source]);

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((tr) => tr.stop());
    streamRef.current = null;
  }, []);

  const reset = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    stopStream();
    if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    objectUrlRef.current = null;
    const video = videoRef.current;
    if (video) {
      video.pause();
      video.srcObject = null;
      video.removeAttribute("src");
      video.load();
    }
    const canvas = canvasRef.current;
    canvas?.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
    fpsWindow.current = [];
    frames.current = 0;
    overlay.current = { deviation: 0, confidence: 0, fps: 0, frameCount: 0 };
    setMetrics(DEFAULT_METRICS);
    setStatus("idle");
    setSource("idle");
    setHasMedia(false);
    setError(null);
  }, [stopStream]);

  const loadFile = useCallback(
    (file: File) => {
      reset();
      const video = videoRef.current;
      if (!video) return;
      const url = URL.createObjectURL(file);
      objectUrlRef.current = url;
      video.src = url;
      video.loop = true;
      video.muted = true;
      video.onloadeddata = () => {
        setSource("upload");
        setHasMedia(true);
        setStatus("running");
        void video.play();
        play();
      };
      video.onerror = () => setError("Could not load that video file.");
    },
    [reset, play],
  );

  const startWebcam = useCallback(async () => {
    reset();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) return;
      video.srcObject = stream;
      video.muted = true;
      video.onloadeddata = () => {
        setSource("webcam");
        setHasMedia(true);
        setStatus("running");
        void video.play();
        play();
      };
    } catch {
      setError("Camera access was denied or is unavailable.");
    }
  }, [reset, play]);

  const updateSettings = useCallback((patch: Partial<DetectionSettings>) => {
    setSettings((prev) => ({ ...prev, ...patch }));
  }, []);

  // Cleanup on unmount.
  useEffect(() => {
    return () => {
      cancelAnimationFrame(rafRef.current);
      stopStream();
      if (objectUrlRef.current) URL.revokeObjectURL(objectUrlRef.current);
    };
  }, [stopStream]);

  return {
    videoRef,
    canvasRef,
    status,
    source,
    metrics,
    settings,
    hasMedia,
    error,
    start,
    pause,
    reset,
    loadFile,
    startWebcam,
    updateSettings,
  };
}
