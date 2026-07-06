import styles from "./ConfidenceBar.module.css";

export function ConfidenceBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  return (
    <>
      <div className={styles.bar}>
        <span style={{ width: `${pct}%` }} />
      </div>
      <span className={styles.value}>{pct}%</span>
    </>
  );
}
