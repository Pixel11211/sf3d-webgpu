import { ASSETS, HF_BASE, type AssetKey } from "../config";
import type { ProgressFn } from "../types";

const CACHE_NAME = "sf3d-assets-v1";

export async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  const h = await crypto.subtle.digest("SHA-256", buf);
  return Array.from(new Uint8Array(h)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

/**
 * Stream an artifact from HF (resolve URL -> CDN), verify sha256, and best-effort cache it.
 * Progress is reported as a fraction of total bytes.
 */
export async function fetchAsset(key: AssetKey, onProgress?: ProgressFn): Promise<ArrayBuffer> {
  const spec = ASSETS[key];
  const url = `${HF_BASE}/${spec.path}`;

  if (typeof caches !== "undefined") {
    try {
      const cache = await caches.open(CACHE_NAME);
      const hit = await cache.match(url);
      if (hit) {
        const buf = await hit.arrayBuffer();
        if (buf.byteLength === spec.bytes) { onProgress?.(`asset:${key}`, 1, "cached"); return buf; }
      }
    } catch { /* cache unavailable */ }
  }

  const res = await fetch(url, { mode: "cors" });
  if (!res.ok || !res.body) throw new Error(`fetch ${spec.path}: HTTP ${res.status}`);
  const total = Number(res.headers.get("Content-Length") || spec.bytes);
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (value) { chunks.push(value); got += value.length; onProgress?.(`asset:${key}`, total ? got / total : 0, `${(got / 1e6).toFixed(1)}/${(total / 1e6).toFixed(1)} MB`); }
  }
  const buf = new ArrayBuffer(got);
  const view = new Uint8Array(buf);
  let off = 0;
  for (const c of chunks) { view.set(c, off); off += c.length; }

  const hex = await sha256Hex(buf);
  if (hex !== spec.sha256) throw new Error(`sha256 mismatch for ${spec.path}\n  expected ${spec.sha256}\n  got      ${hex}`);

  if (typeof caches !== "undefined") {
    try {
      const cache = await caches.open(CACHE_NAME);
      await cache.put(url, new Response(buf, { headers: { "Content-Type": "application/octet-stream", "Content-Length": String(got) } }));
    } catch { /* quota / unavailable — ignore, we still have the bytes */ }
  }
  return buf;
}

export async function fetchFloat32(key: AssetKey, onProgress?: ProgressFn): Promise<Float32Array> {
  return new Float32Array(await fetchAsset(key, onProgress));
}
export async function fetchInt32(key: AssetKey, onProgress?: ProgressFn): Promise<Int32Array> {
  return new Int32Array(await fetchAsset(key, onProgress));
}
export async function fetchJson<T = unknown>(key: AssetKey, onProgress?: ProgressFn): Promise<T> {
  const buf = await fetchAsset(key, onProgress);
  return JSON.parse(new TextDecoder().decode(buf)) as T;
}
