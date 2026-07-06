import type { LaneStatus } from "../types";
import { StatusBadge } from "./StatusBadge";
import styles from "./TopBar.module.css";

export function TopBar({ status }: { status: LaneStatus }) {
  return (
    <header className={styles.topbar}>
      <div className={styles.brand}>
        <span className={styles.mark} aria-hidden />
        <div>
          <h1 className={styles.name}>LaneVision</h1>
          <p className={styles.sub}>ADAS Perception Unit</p>
        </div>
      </div>
      <StatusBadge status={status} />
    </header>
  );
}
