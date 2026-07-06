import { motion } from "framer-motion";

interface Props {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
}

export function Toggle({ label, checked, onChange }: Props) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      onClick={() => onChange(!checked)}
      className="flex w-full items-center justify-between rounded-xl px-1 py-1.5 text-sm text-ink transition-colors hover:text-white"
    >
      <span className="font-medium">{label}</span>
      <span
        className={`relative flex h-6 w-11 items-center rounded-full p-0.5 transition-colors ${
          checked ? "bg-accent" : "bg-base-600"
        }`}
      >
        <motion.span
          layout
          transition={{ type: "spring", stiffness: 500, damping: 34 }}
          className={`h-5 w-5 rounded-full bg-white shadow-sm ${checked ? "ml-auto" : ""}`}
        />
      </span>
    </button>
  );
}
