import { signalFor, SIGNAL_COLOR } from "../types";
import type { LaneStatus } from "../types";
import styles from "./LanePositionGauge.module.css";

interface Props {
  offsetM: number | null;
  status: LaneStatus;
}

const RANGE_M = 0.9; // clamp window either side of centre

export function LanePositionGauge({ offsetM, status }: Props) {
  const hasFix = offsetM !== null && offsetM !== undefined;
  const clamped = hasFix ? Math.max(-RANGE_M, Math.min(RANGE_M, offsetM)) : 0;
  const leftPct = 50 + (clamped / RANGE_M) * 42;
  const color = SIGNAL_COLOR[signalFor(status)];

  return (
    <div className={styles.card}>
      <div className={styles.head}>
        <span>Lane position</span>
        <span className={styles.value}>
          {hasFix ? `${offsetM >= 0 ? "+" : ""}${offsetM.toFixed(2)} m` : "-- m"}
        </span>
      </div>

      <div className={styles.track}>
        <span className={styles.centerLine} />
        <span
          className={styles.vehicle}
          style={{ left: `${leftPct}%`, background: color, color }}
        />
      </div>

      <div className={styles.scale}>
        <span>L</span>
        <span>CENTER</span>
        <span>R</span>
      </div>
    </div>
  );
}
