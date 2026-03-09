import { defineConfig } from "vite";

export default defineConfig({
  resolve: {
    dedupe: ["lit", "@lit/context"],
  },
  build: {
    target: "esnext",
  },
});
