import type { ViewMode } from "../types";
import styles from "./SegmentedControl.module.css";

const MODES: { value: ViewMode; label: string }[] = [
  { value: "final", label: "Detection" },
  { value: "threshold", label: "Threshold mask" },
  { value: "birdseye", label: "Bird's-eye" },
  { value: "roi", label: "Warp region" },
];

interface Props {
  value: ViewMode;
  onChange: (mode: ViewMode) => void;
}

export function ViewModeToggle({ value, onChange }: Props) {
  return (
    <div className={styles.row} role="group" aria-label="View mode">
      {MODES.map((m) => (
        <button
          key={m.value}
          className={`${styles.btn} ${value === m.value ? styles.active : ""}`}
          aria-pressed={value === m.value}
          onClick={() => onChange(m.value)}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}
