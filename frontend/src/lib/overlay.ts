/**
 * Canvas overlay renderer.
 *
 * The dashboard is a visual companion to the Python pipeline. It draws a
 * perspective lane overlay (matching the aesthetic of the real detector's
 * output) on top of a live <video> element, plus an optional real edge pass so
 * the "Edges" toggle shows genuine on-device image processing.
 */

import type { DetectionSettings } from "../types";
import { clamp } from "./format";

export interface OverlayState {
  /** Smooth signed lane deviation in the range [-1, 1]. */
  deviation: number;
  confidence: number;
  fps: number;
  frameCount: number;
}

const ACCENT = "34, 197, 94";
const YELLOW = "240, 180, 41";

/** Draw the current video frame, then the lane overlay, onto the canvas. */
export function renderFrame(
  ctx: CanvasRenderingContext2D,
  video: HTMLVideoElement,
  settings: DetectionSettings,
  state: OverlayState,
): void {
  const { width: w, height: h } = ctx.canvas;

  // 1. Base frame
  try {
    ctx.drawImage(video, 0, 0, w, h);
  } catch {
    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, w, h);
  }

  // 2. Optional real edge pass (downscaled Sobel for performance)
  if (settings.showEdges) {
    drawEdges(ctx, w, h, settings.sensitivity);
  }

  // 3. Lane overlay
  drawLanes(ctx, w, h, settings, state);

  // 4. Optional on-canvas HUD
  if (settings.showHud) {
    drawHud(ctx, w, state);
  }
}

function drawHud(
  ctx: CanvasRenderingContext2D,
  w: number,
  state: OverlayState,
): void {
  const pad = Math.round(w * 0.015);
  const scale = w / 960;
  ctx.save();
  ctx.fillStyle = "rgba(1, 4, 9, 0.55)";
  roundRect(ctx, pad, pad, 172 * scale, 74 * scale, 10 * scale);
  ctx.fill();
  ctx.font = `${Math.round(15 * scale)}px 'JetBrains Mono', monospace`;
  ctx.textBaseline = "top";
  ctx.fillStyle = `rgba(${ACCENT}, 0.95)`;
  ctx.fillText(`FPS  ${state.fps.toFixed(0)}`, pad + 12 * scale, pad + 10 * scale);
  ctx.fillStyle = "rgba(230, 237, 243, 0.9)";
  ctx.fillText(
    `CONF ${(state.confidence * 100).toFixed(0)}%`,
    pad + 12 * scale,
    pad + 30 * scale,
  );
  ctx.fillStyle = "rgba(139, 148, 158, 0.9)";
  ctx.fillText(
    `#${state.frameCount}`,
    pad + 12 * scale,
    pad + 50 * scale,
  );
  ctx.restore();
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawLanes(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  settings: DetectionSettings,
  state: OverlayState,
): void {
  const horizonY = h * 0.6;
  const sway = state.deviation * w * 0.06;

  // Vanishing point drifts with deviation.
  const apexX = w / 2 + sway;
  const leftBottom = w * 0.18 + sway * 0.4;
  const rightBottom = w * 0.82 + sway * 0.4;
  const leftTop = apexX - w * 0.05;
  const rightTop = apexX + w * 0.05;

  // Lane fill
  if (settings.showLaneFill) {
    ctx.beginPath();
    ctx.moveTo(leftBottom, h);
    ctx.lineTo(leftTop, horizonY);
    ctx.lineTo(rightTop, horizonY);
    ctx.lineTo(rightBottom, h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, horizonY, 0, h);
    grad.addColorStop(0, `rgba(${ACCENT}, 0.05)`);
    grad.addColorStop(1, `rgba(${ACCENT}, 0.28)`);
    ctx.fillStyle = grad;
    ctx.fill();
  }

  // Boundary lines
  const draw = (x1: number, x2: number) => {
    ctx.beginPath();
    ctx.moveTo(x1, h);
    ctx.lineTo(x2, horizonY);
    ctx.lineWidth = Math.max(4, w * 0.012);
    ctx.strokeStyle = `rgba(${ACCENT}, 0.95)`;
    ctx.shadowColor = `rgba(${ACCENT}, 0.6)`;
    ctx.shadowBlur = 12;
    ctx.stroke();
    ctx.shadowBlur = 0;
  };
  draw(leftBottom, leftTop);
  draw(rightBottom, rightTop);

  // Dashed centre marking
  ctx.setLineDash([h * 0.03, h * 0.05]);
  ctx.beginPath();
  ctx.moveTo((leftBottom + rightBottom) / 2, h);
  ctx.lineTo(apexX, horizonY);
  ctx.lineWidth = Math.max(2, w * 0.005);
  ctx.strokeStyle = `rgba(${YELLOW}, 0.8)`;
  ctx.stroke();
  ctx.setLineDash([]);

  // Vanishing-point marker
  ctx.beginPath();
  ctx.arc(apexX, horizonY, 4, 0, Math.PI * 2);
  ctx.fillStyle = "rgba(88, 166, 255, 0.9)";
  ctx.fill();
}

/** Lightweight real Sobel edge pass at reduced resolution. */
function drawEdges(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  sensitivity: number,
): void {
  const scale = 0.35;
  const sw = Math.max(1, Math.floor(w * scale));
  const sh = Math.max(1, Math.floor(h * scale));
  let src: ImageData;
  try {
    src = ctx.getImageData(0, 0, w, h);
  } catch {
    return; // e.g. tainted canvas
  }
  // Downsample to grayscale
  const gray = new Float32Array(sw * sh);
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const sx = Math.floor(x / scale);
      const sy = Math.floor(y / scale);
      const i = (sy * w + sx) * 4;
      gray[y * sw + x] =
        0.299 * src.data[i] + 0.587 * src.data[i + 1] + 0.114 * src.data[i + 2];
    }
  }
  const out = ctx.createImageData(sw, sh);
  const thresh = 90 + (1 - sensitivity) * 120;
  for (let y = 1; y < sh - 1; y++) {
    for (let x = 1; x < sw - 1; x++) {
      const gx =
        -gray[(y - 1) * sw + x - 1] +
        gray[(y - 1) * sw + x + 1] +
        -2 * gray[y * sw + x - 1] +
        2 * gray[y * sw + x + 1] +
        -gray[(y + 1) * sw + x - 1] +
        gray[(y + 1) * sw + x + 1];
      const gy =
        -gray[(y - 1) * sw + x - 1] -
        2 * gray[(y - 1) * sw + x] -
        gray[(y - 1) * sw + x + 1] +
        gray[(y + 1) * sw + x - 1] +
        2 * gray[(y + 1) * sw + x] +
        gray[(y + 1) * sw + x + 1];
      const mag = Math.sqrt(gx * gx + gy * gy);
      const on = mag > thresh ? 255 : 0;
      const o = (y * sw + x) * 4;
      out.data[o] = 0;
      out.data[o + 1] = on;
      out.data[o + 2] = Math.floor(on * 0.4);
      out.data[o + 3] = on ? 210 : 0;
    }
  }
  // Blit the small edge image back up, scaled.
  const tmp = document.createElement("canvas");
  tmp.width = sw;
  tmp.height = sh;
  tmp.getContext("2d")!.putImageData(out, 0, 0);
  ctx.globalAlpha = clamp(0.85, 0, 1);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(tmp, 0, 0, w, h);
  ctx.imageSmoothingEnabled = true;
  ctx.globalAlpha = 1;
}
