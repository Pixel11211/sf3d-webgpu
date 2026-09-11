import * as ort from "onnxruntime-web";
import { ORT_WASM_PATHS } from "../config";

export interface WebGPUDiagnostics {
  available: boolean;
  adapterInfo?: string;
  isFallbackAdapter?: boolean;
  shaderF16?: boolean;
  maxStorageBufferBindingSize?: number;
  maxBufferSize?: number;
  maxComputeWorkgroupStorageSize?: number;
  deviceProvided?: boolean;
  error?: string;
}

let configured = false;
export function configureOrt(): void {
  if (configured) return;
  ort.env.wasm.wasmPaths = ORT_WASM_PATHS;
  ort.env.wasm.numThreads = 1; // WebGPU does the heavy lifting; avoid WASM worker/proxy complexity
  ort.env.wasm.proxy = false;
  configured = true;
}

export function hasWebGPU(): boolean {
  return typeof navigator !== "undefined" && !!(navigator as unknown as { gpu?: unknown }).gpu;
}

/**
 * Create our OWN WebGPU device and hand it to ONNX Runtime BEFORE any session is created.
 *
 * Why: the SF3D backbone is FP16 and its attention/triplane tensors are large. ORT's default
 * device may (a) not enable the optional `shader-f16` feature and (b) cap
 * `maxStorageBufferBindingSize` at the 128 MB default. Either makes the FP16/large compute
 * pipelines fail with "Invalid ComputePipeline … due to a previous error", which silently
 * yields garbage and an empty mesh. Requesting `shader-f16` + the adapter's MAX limits fixes both.
 */
export async function initWebGPU(): Promise<WebGPUDiagnostics> {
  const diag: WebGPUDiagnostics = { available: hasWebGPU() };
  if (!diag.available) { diag.error = "WebGPU not available in this browser"; return diag; }
  try {
    configureOrt();
    const gpu = (navigator as unknown as { gpu: GPU }).gpu;
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) { diag.error = "requestAdapter returned null (no suitable GPU)"; return diag; }

    const info = (adapter as unknown as { info?: GPUAdapterInfo }).info;
    diag.adapterInfo = info
      ? [info.vendor, info.device, info.architecture, info.description].filter(Boolean).join(" ") || "unknown"
      : "unknown";
    diag.isFallbackAdapter = (adapter as unknown as { isFallbackAdapter?: boolean }).isFallbackAdapter;

    const hasF16 = adapter.features.has("shader-f16" as GPUFeatureName);
    diag.shaderF16 = hasF16;

    const L = adapter.limits as unknown as Record<string, number>;
    diag.maxStorageBufferBindingSize = L.maxStorageBufferBindingSize;
    diag.maxBufferSize = L.maxBufferSize;
    diag.maxComputeWorkgroupStorageSize = L.maxComputeWorkgroupStorageSize;

    const requiredFeatures: GPUFeatureName[] = hasF16 ? ["shader-f16" as GPUFeatureName] : [];
    const requiredLimits: Record<string, number> = {
      maxStorageBufferBindingSize: L.maxStorageBufferBindingSize,
      maxBufferSize: L.maxBufferSize,
      maxComputeWorkgroupStorageSize: L.maxComputeWorkgroupStorageSize,
    };

    const device = await adapter.requestDevice({ requiredFeatures, requiredLimits } as GPUDeviceDescriptor);
    // Hand our device+adapter to ORT (must be before the first WebGPU session is created).
    (ort.env as unknown as { webgpu: Record<string, unknown> }).webgpu.adapter = adapter;
    (ort.env as unknown as { webgpu: Record<string, unknown> }).webgpu.device = device;
    diag.deviceProvided = true;

    device.lost.then((e) => console.warn("[sf3d] WebGPU device lost:", e.reason, e.message));
    device.addEventListener?.("uncapturederror", (ev) => {
      console.warn("[sf3d] WebGPU uncaptured error:", (ev as GPUUncapturedErrorEvent).error?.message);
    });
  } catch (e) {
    diag.error = `${(e as Error).name}: ${(e as Error).message}`;
  }
  return diag;
}

/** Push/pop a WebGPU validation error scope to surface the ROOT pipeline error. */
export async function withWebGPUErrorScope<T>(fn: () => Promise<T>, label: string): Promise<T> {
  const device = (ort.env as unknown as { webgpu?: { device?: GPUDevice } }).webgpu?.device;
  if (!device?.pushErrorScope) return fn();
  device.pushErrorScope("validation");
  device.pushErrorScope("out-of-memory");
  try {
    return await fn();
  } finally {
    const oom = await device.popErrorScope();
    const validation = await device.popErrorScope();
    if (oom) console.error(`[sf3d] WebGPU OUT-OF-MEMORY during ${label}:`, oom.message);
    if (validation) console.error(`[sf3d] WebGPU VALIDATION root error during ${label}:`, validation.message);
  }
}

export async function createSession(model: ArrayBuffer | Uint8Array, preferWebGPU = true): Promise<ort.InferenceSession> {
  configureOrt();
  const bytes = model instanceof Uint8Array ? model : new Uint8Array(model);
  const executionProviders: (string | Record<string, unknown>)[] =
    preferWebGPU && hasWebGPU() ? ["webgpu", "wasm"] : ["wasm"];
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
