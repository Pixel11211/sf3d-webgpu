# SF3D · WebGPU

**Stable Fast 3D (SF3D) image‑to‑3D, running entirely in your browser** via WebGPU +
ONNX Runtime Web. Drop in a picture of an object, get a textured 3D mesh you can orbit
and export as `.glb` — no server, no Python, no GPU install.

It streams the browser‑ready ONNX artifacts published by
[`needle-tools/SF3D-webgpu`](https://huggingface.co/needle-tools/SF3D-webgpu) (a derivative of
[`stabilityai/stable-fast-3d`](https://huggingface.co/stabilityai/stable-fast-3d), arXiv 2408.00653)
and reproduces the full inference + mesh‑extraction pipeline in TypeScript.

> **Status:** complete, compiling, unit‑tested implementation of the whole pipeline.
> The geometry/colour math is proven by tests here; the actual 912 MB FP16 backbone inference
> and on‑screen 3D require a **WebGPU browser** (see *Verification* below).

---

## Highlights
- **100% client‑side.** WebGPU inference (WASM/CPU fallback), three.js viewer, GLB export.
- **Faithful pipeline** ported from the reference implementation — verified constants, exact
  marching‑tetrahedra tables, `align_corners` triplane sampling, the real color‑head weights.
- **Integrity‑checked streaming.** Every artifact is SHA‑256 verified against
  `assets-manifest.json` and cached (Cache API) so it downloads once.
- **Honest about the boundary** between what's proven in CI and what needs a GPU browser.

## How it works
```
image ──prepare──▶ rgb[1,512,512,3] + c2w + intrinsic_normed
      ──image_tokenizer_single.onnx──▶ image_tokens[1,1297,1024]
      ──backbone_fp16.onnx──────────▶ triplane[1,3,40,384,384]
      ──decoder_single.onnx(triplane, gridPos[1,535882,3])──▶ density, vertex_offset
      sdf = density − 10.0 ; deformed = tets + (1/160)·tanh(vertex_offset)
      ──marching tetrahedra (2,971,452 tets)──▶ mesh (verts, faces)
      ──query_triplane + features_mlp (120→64→64→64→3)──▶ per‑vertex albedo
      ──vertex normals + glTF orientation──▶ three.js / .glb
```
Full spec with every constant, tensor shape and source: **[`docs/model-contract.md`](docs/model-contract.md)**.
Module/data‑flow map: **[`docs/architecture.md`](docs/architecture.md)**.
ONNX I/O ground truth (auto‑extracted): **[`docs/onnx-io.json`](docs/onnx-io.json)**.

## Quick start
**Requirements:** a WebGPU‑capable browser (Chrome/Edge 113+, or recent Firefox/Safari Tech Preview),
Node 20+, and ~1.7 GB of bandwidth on first load (the models).

```bash
npm install
npm run dev        # vite dev server → open the printed URL
# or
npm run build && npm run preview
```
1. Click **Load model** (downloads + verifies + caches the artifacts; watch the progress bar).
2. Drop an image — **ideally a cutout PNG with transparency** (SF3D expects a segmented object).
3. Click **Generate 3D**, orbit the result, then **Download .glb**.

## Model artifacts
Streamed from HF at runtime (never committed here):

| file | size | role |
|---|---|---|
| `onnx/image_tokenizer_single.onnx` | 764 MB | DINOv2 image → 1297 tokens |
| `onnx/backbone_fp16.onnx` | 912 MB | tokens → triplane scene codes |
| `onnx/decoder_single.onnx` | 104 KB | triplane+positions → density, vertex_offset |
| `tets_vertices.bin` / `tets_indices.bin` | 6.4 / 47.5 MB | res‑160 tetrahedral grid |
| `features_mlp_weights.json` | 350 KB | color head (120→3 albedo) |

## Project structure
```
src/
  config.ts            constants + asset manifest (sha256) + verified model params
  types.ts             shared types
  core/
    preprocess.ts      prepare_image: bbox crop (0.85) → 512 → composite on gray
    assets.ts          HF streaming + sha256 verify + Cache API
    onnx.ts            ONNX Runtime Web session manager (WebGPU/WASM)
    camera.ts          c2w + normalized intrinsics (exact)
    triplane.ts        query_triplane bilinear sample (align_corners=True)
    marchingTets.ts    faithful port of MarchingTetrahedraHelper._forward
    colorHead.ts       features_mlp (SiLU MLP 120→64→64→64→3)
    geometry.ts        vertex normals, glTF orientation, winding
    pipeline.ts        end-to-end orchestration
    mesh.ts            three.js mesh + GLB export
  render/viewer.ts     three.js scene / OrbitControls / lighting
  main.ts, ui/         UI wiring + styles
tests/                 node:test unit tests for the pure math
tools/inspect_onnx_io.py  RAM-safe ONNX graph I/O inspector (how the contract was derived)
docs/                  model-contract.md, architecture.md, onnx-io.json
```

## Tests & verification
```bash
npm run typecheck   # tsc --noEmit  (clean)
npm run build       # vite build    (clean)
npm test            # node:test — marching tets, triplane, color head
```
**Proven in this repo (no GPU needed):** TypeScript build, and unit tests that
(i) reconstruct an *exact* isosurface from a linear SDF (validates the marching‑tets
interpolation + triangle tables), (ii) recover a sphere from a radial SDF, (iii) reproduce
`align_corners` bilinear sampling exactly, (iv) exercise the color head, (v) check camera math.

**Needs your WebGPU browser (can't run headless here):** the real FP16 backbone inference and
on‑screen 3D. The ONNX I/O it relies on was verified directly against the model graphs.

### Assumptions to confirm at runtime
These follow the reference code but the upstream `config.yaml` is gated, so verify visually:
- **`rgb` normalization** — fed as raw `[0,1]` HWC; DINOv2 mean/std is assumed baked into the
  ONNX graph. If output is garbage, toggle normalization in `core/preprocess.ts`.
- **Color‑head output activation** — `config.COLOR_OUTPUT_ACTIVATION` defaults to `none`
  (MaterialMLP default) + clamp; flip to `sigmoid` if albedo looks washed/dark.
- **Roughness/metallic** — not emitted by the graphs → fixed defaults in `config.ts`.
- **Orientation** — reference export transform applied (`rotateToGlTF`); adjust if upside‑down.

## Segmentation note
SF3D expects a **foreground cutout**. This app uses the image's alpha channel as the mask and
composites onto gray (matching `prepare_image`). For photos without transparency, plug a
background‑removal model into the `Segmenter` interface (`src/types.ts`) — e.g.
`@imgly/background-removal` — before `preprocessToRgbCond`.

## Self‑hosting ONNX Runtime WASM (optional)
By default the ORT WASM binaries load from a pinned jsDelivr CDN (`config.ORT_WASM_PATHS`).
For offline use, copy `node_modules/onnxruntime-web/dist/*.{wasm,mjs}` into `public/ort/`
and set `ORT_WASM_PATHS = "ort/"`.

## Performance
~1.7 GB of models; the backbone is the heavy step and wants a discrete GPU. Marching tetrahedra
runs over ~3M tets on the main thread (~1–3 s); move it to a Web Worker if you need a
non‑blocking UI. Vertex colours replace the reference CUDA UV texture baker (not portable to
the browser) — geometry and albedo are faithful, UV atlas baking is out of scope.

## License & attribution
- **This source code**: MIT — see [`LICENSE`](LICENSE).
- **Model weights / ONNX artifacts**: **Stability AI Community License**, © Stability AI Ltd.
  They are *streamed from Hugging Face at runtime and not redistributed here*. Review the license
  before use. Derivative artifacts by `needle-tools`. See [`NOTICE`](NOTICE).

## Disclaimer
Not affiliated with Stability AI or needle-tools. Provided as‑is for research/education.
