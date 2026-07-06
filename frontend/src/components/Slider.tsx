interface Props {
  label: string;
  value: number; // 0..1
  onChange: (v: number) => void;
  hint?: string;
}

export function Slider({ label, value, onChange, hint }: Props) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-ink">{label}</span>
        <span className="font-mono text-xs tabular-nums text-accent">
          {Math.round(value * 100)}%
        </span>
      </div>
      <input
        type="range"
        min={0}
        max={1}
        step={0.01}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      {hint && <p className="text-[11px] text-ink-faint">{hint}</p>}
    </div>
  );
}
