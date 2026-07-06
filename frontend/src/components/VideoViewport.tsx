import { STREAM_URL } from "../api";
import styles from "./VideoViewport.module.css";

interface Props {
  /** bump this to force the MJPEG <img> to reconnect after a source change */
  streamKey: number;
}

export function VideoViewport({ streamKey }: Props) {
  return (
    <>
      <div className="eyebrow">
        <span>Forward camera</span>
        <span className={styles.live}>● LIVE</span>
      </div>
      <div className={styles.viewport}>
        <span className={`${styles.bracket} ${styles.tl}`} />
        <span className={`${styles.bracket} ${styles.tr}`} />
        <span className={`${styles.bracket} ${styles.bl}`} />
        <span className={`${styles.bracket} ${styles.br}`} />
        <img
          className={styles.feed}
          src={`${STREAM_URL}?t=${streamKey}`}
          alt="Processed lane-detection feed"
        />
        <div className={styles.scanline} aria-hidden />
      </div>
    </>
  );
}
