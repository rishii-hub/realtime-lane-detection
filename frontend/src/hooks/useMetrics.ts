import { useEffect, useRef, useState } from "react";
import { fetchMetrics } from "../api";
import type { Metrics } from "../types";

const INITIAL: Metrics = {
  curvature_m: null,
  offset_m: null,
  status: "INITIALISING",
  fps: 0,
  confidence: 0,
};

/** Poll the telemetry endpoint on an interval; tolerant of transient errors. */
export function useMetrics(intervalMs = 250): Metrics {
  const [metrics, setMetrics] = useState<Metrics>(INITIAL);
  const timer = useRef<number>();

  useEffect(() => {
    let alive = true;
    const controller = new AbortController();

    const tick = async () => {
      try {
        const data = await fetchMetrics(controller.signal);
        if (alive) setMetrics(data);
      } catch {
        /* keep last good value on a dropped poll */
      }
    };

    tick();
    timer.current = window.setInterval(tick, intervalMs);
    return () => {
      alive = false;
      controller.abort();
      window.clearInterval(timer.current);
    };
  }, [intervalMs]);

  return metrics;
}
