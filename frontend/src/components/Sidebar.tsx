import { motion } from "framer-motion";
import {
  Activity,
  Github,
  LayoutDashboard,
  Radio,
  Settings2,
  Video,
} from "lucide-react";
import type { SourceKind } from "../types";

const NAV = [
  { icon: LayoutDashboard, label: "Dashboard", active: true },
  { icon: Video, label: "Sources" },
  { icon: Activity, label: "Metrics" },
  { icon: Settings2, label: "Settings" },
];

export function Sidebar({ source }: { source: SourceKind }) {
  return (
    <aside className="hidden w-60 shrink-0 flex-col border-r border-base-600/50 bg-base-900/40 p-4 lg:flex">
      <div className="flex items-center gap-2.5 px-2 py-3">
        <img src="/logo.svg" alt="" className="h-9 w-9" />
        <div className="leading-tight">
          <p className="text-sm font-bold text-ink">LaneVision</p>
          <p className="text-[11px] text-ink-faint">Detection Suite</p>
        </div>
      </div>

      <nav className="mt-4 flex flex-col gap-1">
        {NAV.map(({ icon: Icon, label, active }) => (
          <button
            key={label}
            className={`flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors ${
              active
                ? "bg-accent/10 text-accent"
                : "text-ink-muted hover:bg-base-800/70 hover:text-ink"
            }`}
          >
            <Icon size={18} strokeWidth={2} />
            {label}
          </button>
        ))}
      </nav>

      <div className="mt-auto space-y-3">
        <div className="card-glass p-3">
          <div className="flex items-center gap-2">
            <Radio
              size={16}
              className={source === "idle" ? "text-ink-faint" : "text-accent"}
            />
            <span className="text-xs font-semibold text-ink">Pipeline</span>
          </div>
          <p className="mt-1.5 text-[11px] leading-relaxed text-ink-muted">
            {source === "idle"
              ? "Awaiting a video or webcam source."
              : `Live ${source} stream · classical CV overlay`}
          </p>
        </div>

        <a
          href="https://github.com/rishii-hub/realtime-lane-detection"
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-2 rounded-xl px-3 py-2 text-xs font-medium text-ink-muted transition-colors hover:text-ink"
        >
          <Github size={16} />
          View on GitHub
        </a>
      </div>

      <motion.div
        aria-hidden
        className="pointer-events-none mt-2 h-px w-full bg-gradient-to-r from-transparent via-accent/40 to-transparent"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
      />
    </aside>
  );
}
