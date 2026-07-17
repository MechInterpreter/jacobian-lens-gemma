import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Static single-page app; `base: "./"` keeps the production build servable
// from any subpath (GitHub Pages project sites included).
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { outDir: "dist", sourcemap: false },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["src/test/setup.ts"],
    css: false,
  },
});
