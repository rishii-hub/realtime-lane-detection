import type { Metrics } from "../types";
import { LanePositionGauge } from "./LanePositionGauge";
import { ReadoutCard } from "./ReadoutCard";
import { ConfidenceBar } from "./ConfidenceBar";
import { SourceControls } from "./SourceControls";
import styles from "./TelemetryPanel.module.css";

interface Props {
  metrics: Metrics;
  onSourceChanged: () => void;
}

export function TelemetryPanel({ metrics, onSourceChanged }: Props) {
  const curvature =
    metrics.curvature_m != null ? Math.round(metrics.curvature_m).toLocaleString() : "--";

  return (
    <aside className={styles.panel}>
      <div className="eyebrow">
        <span>Telemetry</span>
      </div>

      <LanePositionGauge offsetM={metrics.offset_m} status={metrics.status} />

      <div className={styles.grid}>
        <ReadoutCard label="Curve radius" value={curvature} unit="m" />
        <ReadoutCard label="Process rate" value={metrics.fps.toFixed(0)} unit="fps" />
        <ReadoutCard label="Track confidence" value="" wide>
          <ConfidenceBar value={metrics.confidence} />
        </ReadoutCard>
      </div>

      <SourceControls onSourceChanged={onSourceChanged} />
    </aside>
  );
}
