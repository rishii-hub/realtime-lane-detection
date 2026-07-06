import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, proxy API + MJPEG stream to the FastAPI backend on :8000.
// In prod, `npm run build` emits to ../static/dist, which FastAPI serves.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../static/dist",
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
