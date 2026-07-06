import { useRef, useState } from "react";
import type { ChangeEvent } from "react";
import { setSource, uploadVideo } from "../api";
import type { SourceKind } from "../types";
import styles from "./SourceControls.module.css";

interface Props {
  onSourceChanged: () => void;
}

type Selected = SourceKind | "upload";

const HINTS: Record<Selected, string> = {
  demo: "Looping bundled highway clip.",
  webcam: "Live webcam — grant camera access on the host machine.",
  upload: "",
};

export function SourceControls({ onSourceChanged }: Props) {
  const [selected, setSelected] = useState<Selected>("demo");
  const [hint, setHint] = useState(HINTS.demo);
  const fileInput = useRef<HTMLInputElement>(null);

  const pick = async (source: SourceKind) => {
    setSelected(source);
    setHint(HINTS[source]);
    await setSource(source);
    onSourceChanged();
  };

  const onUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelected("upload");
    setHint(`Uploading ${file.name}…`);
    try {
      const res = await uploadVideo(file);
      setHint(res.ok ? `Processing ${res.filename}.` : "Upload failed — try another file.");
      onSourceChanged();
    } catch {
      setHint("Upload failed — check the connection and retry.");
    }
  };

  return (
    <>
      <div className="eyebrow">
        <span>Input source</span>
      </div>
      <div className={styles.controls}>
        <button
          className={`${styles.btn} ${selected === "demo" ? styles.active : ""}`}
          onClick={() => pick("demo")}
        >
          Demo clip
        </button>
        <button
          className={`${styles.btn} ${selected === "webcam" ? styles.active : ""}`}
          onClick={() => pick("webcam")}
        >
          Webcam
        </button>
        <button
          className={`${styles.btn} ${selected === "upload" ? styles.active : ""}`}
          onClick={() => fileInput.current?.click()}
        >
          Upload video
        </button>
        <input
          ref={fileInput}
          type="file"
          accept="video/*"
          hidden
          onChange={onUpload}
        />
      </div>
      <p className={styles.hint}>{hint}</p>
    </>
  );
}
