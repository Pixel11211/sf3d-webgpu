import { defineConfig } from "vite";

// Vanilla TypeScript + three.js + onnxruntime-web.
// ONNX Runtime's WASM binaries are loaded at runtime from `ort.env.wasm.wasmPaths`
// (see src/core/onnx.ts); we do not bundle them here.
export default defineConfig({
  base: "./",
  build: {
    target: "esnext",
    chunkSizeWarningLimit: 6000,
    sourcemap: false,
  },
  server: { host: "127.0.0.1", port: 5173 },
  test: {
    environment: "node",
    include: ["tests/**/*.test.ts"],
  },
} as any);
