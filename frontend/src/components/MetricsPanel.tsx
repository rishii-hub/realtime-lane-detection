import { Activity, Crosshair, Gauge, Layers, Timer } from "lucide-react";
import type { Metrics } from "../types";
import { fmt, fmtInt, pct } from "../lib/format";
import { MetricCard } from "./MetricCard";

export function MetricsPanel({ metrics }: { metrics: Metrics }) {
  const dev = metrics.deviationPx;
  const devLabel = dev === 0 ? "centered" : dev > 0 ? "right" : "left";

  return (
    <div className="grid grid-cols-2 gap-3">
      <MetricCard
        icon={Gauge}
        label="FPS"
        value={fmt(metrics.avgFps, 0)}
        unit="frames/s"
        accent="green"
        progress={metrics.avgFps / 120}
        delay={0.02}
      />
      <MetricCard
        icon={Activity}
        label="Confidence"
        value={pct(metrics.confidence)}
        accent="blue"
        progress={metrics.confidence}
        delay={0.06}
      />
      <MetricCard
        icon={Timer}
        label="Latency"
        value={fmt(metrics.latencyMs, 1)}
        unit="ms"
        accent="yellow"
        progress={1 - Math.min(1, metrics.latencyMs / 60)}
        delay={0.1}
      />
      <MetricCard
        icon={Layers}
        label="Frames"
        value={fmtInt(metrics.frameCount)}
        accent="neutral"
        delay={0.14}
      />
      <div className="col-span-2">
        <MetricCard
          icon={Crosshair}
          label="Lane deviation"
          value={`${dev > 0 ? "+" : ""}${dev}`}
          unit={`px · ${devLabel}`}
          accent={Math.abs(dev) > 45 ? "yellow" : "green"}
          progress={0.5 + dev / 160}
          delay={0.18}
        />
      </div>
    </div>
  );
}
