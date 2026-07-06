import type { LaneStatus } from "../types";
import { signalFor } from "../types";
import styles from "./StatusBadge.module.css";

export function StatusBadge({ status }: { status: LaneStatus }) {
  const level = signalFor(status);
  return (
    <div className={styles.badge}>
      <span className={`${styles.dot} ${styles[level]}`} />
      <span className={styles.label}>{status}</span>
    </div>
  );
}
