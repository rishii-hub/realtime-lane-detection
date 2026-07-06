import { motion } from "framer-motion";
import { useDetectionEngine } from "./hooks/useDetectionEngine";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { VideoStage } from "./components/VideoStage";
import { MetricsPanel } from "./components/MetricsPanel";
import { ControlPanel } from "./components/ControlPanel";

export default function App() {
  const engine = useDetectionEngine();

  return (
    <div className="flex h-full w-full overflow-hidden">
      <Sidebar source={engine.source} />

      <div className="flex min-w-0 flex-1 flex-col">
        <TopBar status={engine.status} fps={engine.metrics.avgFps} />

        <main className="flex-1 overflow-y-auto p-5 lg:p-8">
          <div className="mx-auto grid max-w-7xl grid-cols-1 gap-5 xl:grid-cols-[1fr_340px]">
            {/* Left: video stage + info */}
            <motion.section
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4 }}
              className="min-w-0 space-y-5"
            >
              <VideoStage
                videoRef={engine.videoRef}
                canvasRef={engine.canvasRef}
                hasMedia={engine.hasMedia}
                status={engine.status}
                source={engine.source}
                error={engine.error}
                onFile={engine.loadFile}
                onWebcam={engine.startWebcam}
              />

              <div className="grid grid-cols-1 gap-5 sm:grid-cols-3">
                <InfoTile
                  title="Pipeline"
                  body="Grayscale → CLAHE → Canny → ROI mask → Hough → temporal smoothing."
                />
                <InfoTile
                  title="On-device"
                  body="Runs entirely in your browser — no upload, no server, no tracking."
                />
                <InfoTile
                  title="Companion UI"
                  body="A visual front-end for the production Python detector in this repo."
                />
              </div>
            </motion.section>

            {/* Right: metrics + controls */}
            <aside className="space-y-5">
              <MetricsPanel metrics={engine.metrics} />
              <ControlPanel
                status={engine.status}
                hasMedia={engine.hasMedia}
                settings={engine.settings}
                onStart={engine.start}
                onPause={engine.pause}
                onReset={engine.reset}
                onSettings={engine.updateSettings}
              />
            </aside>
          </div>
        </main>
      </div>
    </div>
  );
}

function InfoTile({ title, body }: { title: string; body: string }) {
  return (
    <div className="card p-4">
      <p className="text-sm font-semibold text-ink">{title}</p>
      <p className="mt-1.5 text-xs leading-relaxed text-ink-muted">{body}</p>
    </div>
  );
}
