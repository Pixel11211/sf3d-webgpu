import "./ui/styles.css";
import { SF3DPipeline } from "./core/pipeline";
import { hasWebGPU } from "./core/onnx";
import { Viewer } from "./render/viewer";
import { buildThreeMesh, exportGLB } from "./core/mesh";
import type { MeshData } from "./types";

const app = document.getElementById("app")!;
app.innerHTML = `
  <header>
    <h1>SF3D · WebGPU</h1>
    <span class="sub">Stable Fast 3D image&#8209;to&#8209;3D in the browser · streams needle-tools/SF3D-webgpu</span>
    <span style="flex:1"></span>
    <span id="gpu" class="badge">checking WebGPU…</span>
  </header>
  <main>
    <section class="panel">
      <div id="drop" class="drop">
        <div><strong>Drop an image</strong> or <label style="color:var(--acc);cursor:pointer">browse<input id="file" type="file" accept="image/*" hidden></label></div>
        <div class="hint" style="margin-top:6px">Best: a foreground cutout with transparency (PNG). SF3D expects a segmented object.</div>
        <img id="thumb" class="thumb" hidden alt="preview">
      </div>
      <div class="row">
        <button id="load">1 · Load model (~1.7 GB)</button>
        <button id="gen" class="secondary" disabled>2 · Generate 3D</button>
      </div>
      <div>
        <div class="bar"><i id="bar"></i></div>
        <div id="status" class="status" style="margin-top:6px">Idle.</div>
      </div>
      <div class="row">
        <button id="dl" class="secondary" disabled>Download .glb</button>
        <span id="stats" class="stats"></span>
      </div>
      <div id="log" class="log"></div>
      <div class="hint">Runs entirely client-side. First load downloads &amp; sha256-verifies the ONNX artifacts, then caches them.</div>
    </section>
    <section class="stage"><canvas id="view"></canvas></section>
  </main>
  <footer class="foot">
    <strong>Powered by Stability AI</strong> · Stable Fast 3D model © Stability AI Ltd. (Stability AI
    Community License) · ONNX artifacts by needle-tools · this app's code is MIT
  </footer>`;

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const dropEl = $("drop"); const fileEl = $<HTMLInputElement>("file"); const thumb = $<HTMLImageElement>("thumb");
const loadBtn = $<HTMLButtonElement>("load"); const genBtn = $<HTMLButtonElement>("gen"); const dlBtn = $<HTMLButtonElement>("dl");
const bar = $<HTMLElement>("bar"); const statusEl = $("status"); const statsEl = $("stats"); const logEl = $("log"); const gpuEl = $("gpu");
const view = $<HTMLCanvasElement>("view");

const pipeline = new SF3DPipeline();
const viewer = new Viewer(view);
let picked: File | null = null;
let mesh: MeshData | null = null;

function log(msg: string) { const t = new Date().toLocaleTimeString(); logEl.textContent = `[${t}] ${msg}\n` + logEl.textContent; }
function progress(stage: string, frac: number, detail?: string) {
  bar.style.width = `${Math.round(Math.min(1, Math.max(0, frac)) * 100)}%`;
  statusEl.textContent = detail ? `${stage} — ${detail}` : stage;
}
function setGpu() {
  const ok = hasWebGPU();
  gpuEl.textContent = ok ? "WebGPU ✓" : "WebGPU ✗ (will use WASM/CPU — slow)";
  gpuEl.className = `badge ${ok ? "ok" : "err"}`;
}
setGpu();

function setFile(f: File) {
  picked = f;
  const url = URL.createObjectURL(f);
  thumb.src = url; thumb.hidden = false;
  dropEl.classList.add("has");
  genBtn.disabled = !pipeline.loaded;
  log(`image: ${f.name} (${(f.size / 1024).toFixed(0)} KB)`);
}
fileEl.addEventListener("change", () => { if (fileEl.files?.[0]) setFile(fileEl.files[0]); });
["dragover", "drop"].forEach((ev) => dropEl.addEventListener(ev, (e) => {
  e.preventDefault();
  if (ev === "drop") { const f = (e as DragEvent).dataTransfer?.files?.[0]; if (f) setFile(f); }
}));

loadBtn.addEventListener("click", async () => {
  loadBtn.disabled = true;
  try {
    log("loading model artifacts…");
    await pipeline.load(progress);
    log("model ready.");
    progress("ready", 1, "model loaded");
    genBtn.disabled = !picked;
  } catch (e) {
    log(`LOAD ERROR: ${(e as Error).message}`);
    statusEl.textContent = "Load failed (see log).";
    loadBtn.disabled = false;
  }
});

genBtn.addEventListener("click", async () => {
  if (!picked) return;
  genBtn.disabled = true; dlBtn.disabled = true;
  try {
    log("generating…");
    const t0 = performance.now();
    mesh = await pipeline.generate(picked, progress);
    const dt = ((performance.now() - t0) / 1000).toFixed(1);
    viewer.setMesh(buildThreeMesh(mesh));
    statsEl.textContent = `${mesh.positions.length / 3} verts · ${mesh.indices.length / 3} tris · ${dt}s`;
    log(`mesh: ${mesh.positions.length / 3} verts, ${mesh.indices.length / 3} tris in ${dt}s`);
    dlBtn.disabled = false; genBtn.disabled = false;
  } catch (e) {
    log(`GENERATE ERROR: ${(e as Error).message}`);
    statusEl.textContent = "Generation failed (see log).";
    genBtn.disabled = false;
  }
});

dlBtn.addEventListener("click", async () => {
  if (!mesh) return;
  try {
    const glb = await exportGLB(mesh);
    const blob = new Blob([glb], { type: "model/gltf-binary" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "sf3d_mesh.glb"; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
    log(`exported GLB (${(glb.byteLength / 1024).toFixed(0)} KB)`);
  } catch (e) { log(`EXPORT ERROR: ${(e as Error).message}`); }
});
