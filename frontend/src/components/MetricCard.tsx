import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";

interface Props {
  icon: LucideIcon;
  label: string;
  value: string;
  unit?: string;
  accent?: "green" | "blue" | "yellow" | "neutral";
  /** 0..1 progress for the mini bar; omit to hide. */
  progress?: number;
  delay?: number;
}

const ACCENTS = {
  green: { text: "text-accent", bar: "bg-accent" },
  blue: { text: "text-info", bar: "bg-info" },
  yellow: { text: "text-warn", bar: "bg-warn" },
  neutral: { text: "text-ink", bar: "bg-ink-muted" },
};

export function MetricCard({
  icon: Icon,
  label,
  value,
  unit,
  accent = "neutral",
  progress,
  delay = 0,
}: Props) {
  const a = ACCENTS[accent];
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.35 }}
      className="card p-4"
    >
      <div className="flex items-center justify-between">
        <span className="label">{label}</span>
        <Icon size={15} className={a.text} />
      </div>
      <div className="mt-2 flex items-baseline gap-1">
        <span className="font-mono text-2xl font-semibold tabular-nums text-ink">
          {value}
        </span>
        {unit && <span className="text-xs text-ink-muted">{unit}</span>}
      </div>
      {progress !== undefined && (
        <div className="mt-3 h-1 w-full overflow-hidden rounded-full bg-base-700">
          <motion.div
            className={`h-full rounded-full ${a.bar}`}
            animate={{ width: `${Math.round(Math.max(0, Math.min(1, progress)) * 100)}%` }}
            transition={{ type: "spring", stiffness: 120, damping: 20 }}
          />
        </div>
      )}
    </motion.div>
  );
}
