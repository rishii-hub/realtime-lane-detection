import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
//
// `VITE_BASE` lets CI build for a sub-path deployment (GitHub Pages serves the
// app from /realtime-lane-detection/); local dev and preview default to "/".
export default defineConfig({
  base: process.env.VITE_BASE ?? "/",
  plugins: [react()],
  server: {
    port: 5173,
    open: true,
  },
});
