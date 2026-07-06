# Lane Detection · Dashboard

A modern, dark-themed dashboard for the
[Real-Time Lane Detection](../README.md) project. Built with **React 18**,
**Vite**, **TypeScript**, **Tailwind CSS**, and **Framer Motion**.

> The dashboard is a **visual companion** to the Python pipeline. It runs a
> real, on-device canvas overlay (including a genuine Sobel edge pass) on top of
> an uploaded clip or your webcam — no backend, no upload, no tracking.

**Live demo:** [rishii-hub.github.io/realtime-lane-detection](https://rishii-hub.github.io/realtime-lane-detection/) —
deployed automatically from `main` via GitHub Pages. Click *"Try a sample clip"*
for an instant, zero-setup demo.

## Features

- 🎥 **Drag-and-drop** video upload and **live webcam** preview
- 🛣️ Real-time perspective **lane overlay** with lane fill and deviation
- 📊 **Metrics panel** — FPS, confidence, latency, frame count, lane deviation
- 🎛️ **Controls** — start / pause / reset, overlay toggles, threshold &
  sensitivity sliders
- 🌘 Polished dark UI inspired by Vercel, Linear, and GitHub
- ✨ Smooth motion via Framer Motion, fully responsive layout

## Getting started

```bash
cd frontend
npm install
npm run dev      # http://localhost:5173
```

### Build for production

```bash
npm run build
npm run preview
```

## Project layout

```
src/
├── App.tsx                  # Layout composition
├── types.ts                 # Shared types & defaults
├── hooks/
│   └── useDetectionEngine.ts # Animation loop, FPS timing, metrics
├── lib/
│   ├── overlay.ts           # Canvas overlay + Sobel edge pass
│   └── format.ts            # Formatting helpers
└── components/              # Sidebar, TopBar, VideoStage, MetricsPanel, ...
```

## Tech stack

| Concern     | Choice                     |
| ----------- | -------------------------- |
| Framework   | React 18                   |
| Build tool  | Vite 5                     |
| Language    | TypeScript (strict)        |
| Styling     | Tailwind CSS 3             |
| Animation   | Framer Motion 11           |
| Icons       | lucide-react               |
