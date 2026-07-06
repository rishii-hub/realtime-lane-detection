import { AnimatePresence, motion } from "framer-motion";
import { AlertCircle, ScanLine } from "lucide-react";
import type { EngineStatus, SourceKind } from "../types";
import { Dropzone } from "./Dropzone";

interface Props {
  videoRef: React.RefObject<HTMLVideoElement>;
  canvasRef: React.RefObject<HTMLCanvasElement>;
  hasMedia: boolean;
  status: EngineStatus;
  source: SourceKind;
  error: string | null;
  onFile: (file: File) => void;
  onWebcam: () => void;
}

export function VideoStage({
  videoRef,
  canvasRef,
  hasMedia,
  status,
  source,
  error,
  onFile,
  onWebcam,
}: Props) {
  return (
    <div className="card relative aspect-video w-full overflow-hidden">
      {/* The video element is the raw source; the canvas is the annotated view. */}
      <video
        ref={videoRef}
        playsInline
        className="pointer-events-none absolute inset-0 h-full w-full object-cover opacity-0"
      />
      <canvas
        ref={canvasRef}
        className={`absolute inset-0 h-full w-full object-cover transition-opacity duration-300 ${
          hasMedia ? "opacity-100" : "opacity-0"
        }`}
      />

      {/* Corner framing + source tag */}
      {hasMedia && (
        <>
          <div className="pointer-events-none absolute left-3 top-3 flex items-center gap-1.5 rounded-lg bg-base-950/70 px-2.5 py-1 text-[11px] font-medium text-ink backdrop-blur">
            <ScanLine size={13} className="text-accent" />
            {source === "webcam" ? "Webcam" : "Uploaded clip"}
            {status === "paused" && <span className="text-warn">· paused</span>}
          </div>
          <div className="pointer-events-none absolute inset-3 rounded-xl ring-1 ring-inset ring-white/5" />
        </>
      )}

      <AnimatePresence>
        {!hasMedia && (
          <motion.div
            key="empty"
            exit={{ opacity: 0 }}
            className="absolute inset-0 grid place-items-center"
          >
            <Dropzone onFile={onFile} onWebcam={onWebcam} />
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ y: 12, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute bottom-3 left-1/2 flex -translate-x-1/2 items-center gap-2 rounded-xl border border-danger/40 bg-danger/10 px-3 py-2 text-xs font-medium text-danger backdrop-blur"
          >
            <AlertCircle size={14} />
            {error}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
