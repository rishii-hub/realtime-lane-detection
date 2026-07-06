import { motion } from "framer-motion";
import { Cpu, Gauge } from "lucide-react";
import type { EngineStatus } from "../types";
import { fmt } from "../lib/format";

const STATUS_STYLES: Record<EngineStatus, { dot: string; label: string; text: string }> = {
  idle: { dot: "bg-ink-faint", label: "Idle", text: "text-ink-muted" },
  running: { dot: "bg-accent", label: "Live", text: "text-accent" },
  paused: { dot: "bg-warn", label: "Paused", text: "text-warn" },
};

export function TopBar({ status, fps }: { status: EngineStatus; fps: number }) {
  const s = STATUS_STYLES[status];
  return (
    <header className="flex items-center justify-between gap-4 border-b border-base-600/50 px-5 py-4 lg:px-8">
      <div>
        <h1 className="text-lg font-bold tracking-tight text-ink lg:text-xl">
          Real-Time Lane Detection
        </h1>
        <p className="text-xs text-ink-muted">
          Canny + Hough pipeline · temporal smoothing · live overlay
        </p>
      </div>

      <div className="flex items-center gap-2.5">
        <div className="chip">
          <Cpu size={13} className="text-info" />
          On-device
        </div>
        <div className="hidden chip sm:inline-flex">
          <Gauge size={13} className="text-accent" />
          {fmt(fps, 0)} FPS
        </div>
        <div className="chip">
          <motion.span
            key={status}
            className={`h-2 w-2 rounded-full ${s.dot} ${status === "running" ? "animate-pulse-dot" : ""}`}
            initial={{ scale: 0.6 }}
            animate={{ scale: 1 }}
          />
          <span className={s.text}>{s.label}</span>
        </div>
      </div>
    </header>
  );
}
