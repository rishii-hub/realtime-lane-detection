// Typed client for the FastAPI backend.
import type { Metrics, SourceKind, ViewMode } from "./types";

const json = { "Content-Type": "application/json" };

export const STREAM_URL = "/api/stream";

export async function fetchMetrics(signal?: AbortSignal): Promise<Metrics> {
  const res = await fetch("/api/metrics", { signal });
  if (!res.ok) throw new Error(`metrics ${res.status}`);
  return (await res.json()) as Metrics;
}

export async function setViewMode(mode: ViewMode): Promise<void> {
  await fetch("/api/view", {
    method: "POST",
    headers: json,
    body: JSON.stringify({ mode }),
  });
}

export async function setSource(source: SourceKind): Promise<void> {
  await fetch("/api/source", {
    method: "POST",
    headers: json,
    body: JSON.stringify({ source }),
  });
}

export async function uploadVideo(file: File): Promise<{ ok: boolean; filename?: string }> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch("/api/upload", { method: "POST", body: form });
  return (await res.json()) as { ok: boolean; filename?: string };
}
