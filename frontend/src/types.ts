export type SourceKind = "idle" | "upload" | "webcam";

export type EngineStatus = "idle" | "running" | "paused";

export interface Metrics {
  fps: number;
  avgFps: number;
  confidence: number; // 0..1
  frameCount: number;
  latencyMs: number;
  deviationPx: number; // signed; negative = left, positive = right
}

export interface DetectionSettings {
  /** Canny / edge sensitivity, 0..1. */
  sensitivity: number;
  /** Hough vote threshold as a normalized 0..1 value. */
  threshold: number;
  showLaneFill: boolean;
  showEdges: boolean;
  showHud: boolean;
}

export const DEFAULT_METRICS: Metrics = {
  fps: 0,
  avgFps: 0,
  confidence: 0,
  frameCount: 0,
  latencyMs: 0,
  deviationPx: 0,
};

export const DEFAULT_SETTINGS: DetectionSettings = {
  sensitivity: 0.6,
  threshold: 0.4,
  showLaneFill: true,
  showEdges: false,
  showHud: true,
};
