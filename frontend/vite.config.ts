import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the Python backend runs on :8765 (`vctts-server`); Vite proxies to it.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8765",
      "/ws": { target: "ws://127.0.0.1:8765", ws: true },
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
