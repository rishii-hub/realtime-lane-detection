import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { Camera, Loader2, Sparkles, UploadCloud } from "lucide-react";

interface Props {
  onFile: (file: File) => void;
  onWebcam: () => void;
}

/** Path to the bundled demo clip (served from /public). */
const SAMPLE_CLIP = `${import.meta.env.BASE_URL}sample-drive.mp4`;

export function Dropzone({ onFile, onWebcam }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [loadingSample, setLoadingSample] = useState(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file?.type.startsWith("video/")) onFile(file);
  };

  const loadSample = async () => {
    try {
      setLoadingSample(true);
      const res = await fetch(SAMPLE_CLIP);
      const blob = await res.blob();
      onFile(new File([blob], "sample-drive.mp4", { type: "video/mp4" }));
    } finally {
      setLoadingSample(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      className="flex h-full w-full flex-col items-center justify-center gap-6 p-6"
    >
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`group flex w-full max-w-md cursor-pointer flex-col items-center gap-3 rounded-2xl border-2 border-dashed px-8 py-12 text-center transition-all ${
          dragging
            ? "border-accent bg-accent/5"
            : "border-base-600 hover:border-base-500 hover:bg-base-800/40"
        }`}
      >
        <motion.div
          animate={{ y: dragging ? -4 : 0 }}
          className="grid h-14 w-14 place-items-center rounded-2xl bg-base-800 text-accent ring-1 ring-base-600/70 group-hover:ring-accent/40"
        >
          <UploadCloud size={26} />
        </motion.div>
        <div>
          <p className="text-sm font-semibold text-ink">
            Drop a driving clip here
          </p>
          <p className="mt-1 text-xs text-ink-muted">
            or <span className="text-accent">browse</span> — MP4, WebM, MOV
          </p>
        </div>
        <input
          ref={inputRef}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
        />
      </div>

      <div className="flex items-center gap-3 text-xs text-ink-faint">
        <span className="h-px w-10 bg-base-600" />
        or
        <span className="h-px w-10 bg-base-600" />
      </div>

      <div className="flex flex-wrap items-center justify-center gap-3">
        <button className="btn-primary" onClick={loadSample} disabled={loadingSample}>
          {loadingSample ? (
            <Loader2 size={16} className="animate-spin" />
          ) : (
            <Sparkles size={16} />
          )}
          Try a sample clip
        </button>
        <button className="btn-ghost" onClick={onWebcam}>
          <Camera size={16} />
          Use live webcam
        </button>
      </div>
    </motion.div>
  );
}
