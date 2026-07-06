import { Pause, Play, RotateCcw } from "lucide-react";
import type { DetectionSettings, EngineStatus } from "../types";
import { Slider } from "./Slider";
import { Toggle } from "./Toggle";

interface Props {
  status: EngineStatus;
  hasMedia: boolean;
  settings: DetectionSettings;
  onStart: () => void;
  onPause: () => void;
  onReset: () => void;
  onSettings: (patch: Partial<DetectionSettings>) => void;
}

export function ControlPanel({
  status,
  hasMedia,
  settings,
  onStart,
  onPause,
  onReset,
  onSettings,
}: Props) {
  const running = status === "running";
  return (
    <div className="card p-4">
      <h3 className="label mb-3">Controls</h3>

      {/* Transport */}
      <div className="grid grid-cols-3 gap-2">
        <button
          className="btn-primary"
          disabled={!hasMedia}
          onClick={running ? onPause : onStart}
        >
          {running ? <Pause size={16} /> : <Play size={16} />}
          {running ? "Pause" : "Start"}
        </button>
        <button className="btn-ghost col-span-2" disabled={!hasMedia} onClick={onReset}>
          <RotateCcw size={15} />
          Reset
        </button>
      </div>

      {/* Sliders */}
      <div className="mt-5 space-y-4">
        <Slider
          label="Threshold"
          value={settings.threshold}
          onChange={(v) => onSettings({ threshold: v })}
          hint="Hough vote threshold — higher rejects weak lines."
        />
        <Slider
          label="Sensitivity"
          value={settings.sensitivity}
          onChange={(v) => onSettings({ sensitivity: v })}
          hint="Edge sensitivity — higher detects fainter markings."
        />
      </div>

      {/* Overlay toggles */}
      <div className="mt-5 border-t border-base-600/50 pt-3">
        <h4 className="label mb-1.5">Overlays</h4>
        <Toggle
          label="Lane fill"
          checked={settings.showLaneFill}
          onChange={(v) => onSettings({ showLaneFill: v })}
        />
        <Toggle
          label="Edge map"
          checked={settings.showEdges}
          onChange={(v) => onSettings({ showEdges: v })}
        />
        <Toggle
          label="HUD metrics"
          checked={settings.showHud}
          onChange={(v) => onSettings({ showHud: v })}
        />
      </div>
    </div>
  );
}
