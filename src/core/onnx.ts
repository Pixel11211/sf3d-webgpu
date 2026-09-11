import * as ort from "onnxruntime-web";
import { ORT_WASM_PATHS } from "../config";

let configured = false;
export function configureOrt(): void {
  if (configured) return;
  ort.env.wasm.wasmPaths = ORT_WASM_PATHS;
  ort.env.wasm.numThreads = 1; // avoid worker/proxy complexity; WebGPU does the heavy lifting
  ort.env.wasm.proxy = false;
  configured = true;
}

export function hasWebGPU(): boolean {
  return typeof navigator !== "undefined" && !!(navigator as unknown as { gpu?: unknown }).gpu;
}

export async function createSession(model: ArrayBuffer | Uint8Array, preferWebGPU = true): Promise<ort.InferenceSession> {
  configureOrt();
  const bytes = model instanceof Uint8Array ? model : new Uint8Array(model);
  const executionProviders: (string | Record<string, unknown>)[] = preferWebGPU && hasWebGPU()
    ? ["webgpu", "wasm"]
    : ["wasm"];
  return ort.InferenceSession.create(bytes, {
    executionProviders: executionProviders as never,
    graphOptimizationLevel: "all",
    logSeverityLevel: 2,
  });
}

export function f32Tensor(data: Float32Array | ArrayBuffer, dims: number[]): ort.Tensor {
  const arr = data instanceof Float32Array ? data : new Float32Array(data);
  return new ort.Tensor("float32", arr, dims);
}

export function outF32(t: ort.Tensor): Float32Array {
  return t.data as Float32Array;
}

export type { ort };
