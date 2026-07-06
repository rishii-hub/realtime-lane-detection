// Shared types for the LaneVision dashboard.

export type LaneStatus =
  | "CENTERED"
  | "SEARCHING"
  | "INITIALISING"
  | `LANE DEPARTURE ${"LEFT" | "RIGHT"}`;

export interface Metrics {
  curvature_m: number | null;
  offset_m: number | null;
  status: LaneStatus;
  fps: number;
  confidence: number;
}

export type ViewMode = "final" | "threshold" | "birdseye" | "roi";
export type SourceKind = "demo" | "webcam";

export type SignalLevel = "ok" | "warn" | "alert" | "idle";

/** Map a detector status to a UI signal level. */
export function signalFor(status: LaneStatus): SignalLevel {
  if (status.startsWith("LANE DEPARTURE")) return "alert";
  if (status === "CENTERED") return "ok";
  if (status === "SEARCHING") return "warn";
  return "idle";
}

export const SIGNAL_COLOR: Record<SignalLevel, string> = {
  ok: "#4ade80",
  warn: "#f5b041",
  alert: "#ff5a52",
  idle: "#7a8a94",
};
