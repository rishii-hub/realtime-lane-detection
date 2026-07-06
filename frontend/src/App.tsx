import { useCallback, useState } from "react";
import { useMetrics } from "./hooks/useMetrics";
import { setViewMode } from "./api";
import type { ViewMode } from "./types";
import { TopBar } from "./components/TopBar";
import { VideoViewport } from "./components/VideoViewport";
import { ViewModeToggle } from "./components/ViewModeToggle";
import { TelemetryPanel } from "./components/TelemetryPanel";
import styles from "./App.module.css";

export default function App() {
  const metrics = useMetrics(250);
  const [view, setView] = useState<ViewMode>("final");
  const [streamKey, setStreamKey] = useState(0);

  const onViewChange = useCallback(async (mode: ViewMode) => {
    setView(mode);
    await setViewMode(mode);
  }, []);

  // Force the MJPEG connection to re-open after switching source
  const reloadStream = useCallback(() => setStreamKey((k) => k + 1), []);

  return (
    <div className="app">
      <div className="grid-overlay" aria-hidden />

      <TopBar status={metrics.status} />

      <main className="console">
        <section className={styles.viewportPanel}>
          <VideoViewport streamKey={streamKey} />
          <ViewModeToggle value={view} onChange={onViewChange} />
        </section>

        <TelemetryPanel metrics={metrics} onSourceChanged={reloadStream} />
      </main>

      <footer className={styles.footer}>
        <span>
          Colour + gradient threshold → perspective warp → sliding-window polyfit
        </span>
        <span className={styles.sep}>/</span>
        <span>OpenCV · FastAPI · React</span>
      </footer>
    </div>
  );
}
