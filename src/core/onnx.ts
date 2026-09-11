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
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  // Prefer the discrete GPU. ORT reads this when it creates its own adapter/device.
  try { (ort.env as unknown as { webgpu: Record<string, unknown> }).webgpu.powerPreference = "high-performance"; } catch { /* older ORT */ }
  configured = true;
}

export function hasWebGPU(): boolean {
  return typeof navigator !== "undefined" && !!(navigator as unknown as { gpu?: unknown }).gpu;
}

/** graphOptimizationLevel can be overridden with ?opt=basic|extended|all|disabled (debugging). */
function optLevel(): "all" | "extended" | "basic" | "disabled" {
  if (typeof location !== "undefined") {
    const v = new URLSearchParams(location.search).get("opt");
    if (v === "basic" || v === "extended" || v === "all" || v === "disabled") return v;
  }
  return "all";
}
export const graphOptLevel = optLevel;

/**
 * Diagnostics only. IMPORTANT: we must NOT create a device or set env.webgpu.adapter here.
 * In WebGPU an adapter can create only ONE device; if we consume it, ORT fails to create its
 * own device and silently drops the WebGPU EP (CPU fallback -> garbage). ORT 1.29 already
 * requests `shader-f16` and the adapter's MAX buffer limits when it builds its device, so we
 * just inspect capabilities (requestAdapter does not consume the adapter — requestDevice does).
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
    diag.shaderF16 = adapter.features.has("shader-f16" as GPUFeatureName);
    const L = adapter.limits as unknown as Record<string, number>;
    diag.maxStorageBufferBindingSize = L.maxStorageBufferBindingSize;
    diag.maxBufferSize = L.maxBufferSize;
    diag.maxComputeWorkgroupStorageSize = L.maxComputeWorkgroupStorageSize;
    diag.deviceProvided = false; // ORT creates its own device (with shader-f16 + max limits)
    // do NOT keep a reference / do NOT requestDevice — leave the adapter unconsumed for ORT
  } catch (e) {
    diag.error = `${(e as Error).name}: ${(e as Error).message}`;
  }
  return diag;
}

/** Push/pop WebGPU error scopes around a run to surface the ROOT pipeline/OOM error. */
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
    graphOptimizationLevel: optLevel(),
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
