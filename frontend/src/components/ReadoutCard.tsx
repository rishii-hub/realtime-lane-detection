import type { ReactNode } from "react";
import styles from "./ReadoutCard.module.css";

interface Props {
  label: string;
  value: ReactNode;
  unit?: string;
  wide?: boolean;
  children?: ReactNode;
}

export function ReadoutCard({ label, value, unit, wide, children }: Props) {
  return (
    <div className={`${styles.card} ${wide ? styles.wide : ""}`}>
      <span className={styles.label}>{label}</span>
      {children}
      {value !== "" && value != null && (
        <span className={styles.value}>
          {value}
          {unit && <em>{unit}</em>}
        </span>
      )}
    </div>
  );
}
