# SF3D WebGPU — Model Contract (ground truth)

Everything here was **verified** by inspecting the real artifacts, not guessed:
- ONNX I/O signatures: extracted with `tools/inspect_onnx_io.py` → `docs/onnx-io.json`.
- Tet grid: parsed from `tets_vertices.bin` / `tets_indices.bin`.
- Color head: parsed from `features_mlp_weights.json`.
- Pre/post-processing math: ported from `Stability-AI/stable-fast-3d`
  (`run.py`, `sf3d/system.py`, `sf3d/models/{isosurface,network,utils}.py`, `sf3d/utils.py`).

Source artifacts: **`needle-tools/SF3D-webgpu`** (Hugging Face), a derivative of
**`stabilityai/stable-fast-3d`** (arXiv 2408.00653), Stability AI Community License.

## ONNX graphs (producer: pytorch 2.10, see docs/onnx-io.json)

### image_tokenizer_single.onnx (opset 18, ~764 MB)
- inputs
  - `rgb` `float32[B,512,512,3]` — **HWC**, in `[0,1]`, background pre-composited (no mask input).
  - `c2w` `float32[B,4,4]` — camera-to-world (row-major).
  - `intrinsic_normed` `float32[B,3,3]` — normalized intrinsics (row-major).
- outputs
  - `image_tokens` `float32[B,1297,1024]` — 1 CLS + 1296 DINOv2 patch tokens.
- Note: DINOv2 mean/std normalization is **inside** the graph (module-level export); pass raw `[0,1]` rgb.

### backbone_fp16.onnx (opset 18, ~912 MB)
- inputs: `image_tokens` `float32[1,1297,1024]`
- outputs: `triplane` `float32[1,3,40,384,384]` — **scene codes** (already post-PixelShuffle upsample).

### decoder_single.onnx (~104 KB)
- inputs
  - `triplane` `float32[B,3,40,384,384]`
  - `positions` `float32[B,N,3]` — world coords in `[-1,1]`.
- outputs
  - `density` `float32[B,N,1]`
  - `vertex_offset` `float32[B,N,3]`
- Internally performs the triplane bilinear sample (align_corners=True) + geometry MLP.

## Fixed constants (stable-fast-3d defaults)
- `cond_image_size = 512`, `background_color = [0.5,0.5,0.5]`, `foreground_ratio = 0.85`
- `isosurface_resolution = 160`, `isosurface_threshold = 10.0`, `radius = 1.0`
- `default_fovy_deg = 40.0`, `default_distance = 1.6`
- triplane: `[3, 40, 384, 384]` → feature dim `120`
- tet grid: `535882` vertices `float32 xyz ∈ [0,1]`, `2971452` tets `int32 x4`

### Camera (exact)
```
c2w(1.6) = [[0,0,1,1.6],[1,0,0,0],[0,1,0,0],[0,0,0,1]]   (row-major)
f = 0.5*512 / tan(0.5*40deg) = 703.3521
intrinsic_normed = [[f/512,0,0.5],[0,f/512,0.5],[0,0,1]] = [[1.373734,0,0.5],[0,1.373734,0.5],[0,0,1]]
```

## Pipeline (end to end)
1. **Preprocess**: load RGBA → foreground bbox crop to `0.85` → resize `512` →
   `rgb = lerp(bg=0.5, rgb, alpha)`; `rgb_cond ∈ [0,1]`, HWC.
2. **Tokenizer**: `rgb,c2w,intrinsic_normed → image_tokens[1,1297,1024]`.
3. **Backbone**: `image_tokens → triplane[1,3,40,384,384]`.
4. **Geometry**:
   - `gridPos = tets_vertices*2-1` (`[-1,1]`) → `decoder(triplane, gridPos)` → `density, vertex_offset`.
   - `sdf = density - 10.0`.
   - `deformed = tets_vertices + (1/160)*tanh(vertex_offset)` (still `[0,1]` space).
   - **Marching tetrahedra**(`deformed`, `sdf`, `tets_indices`) → `verts∈[0,1]`, `faces`.
   - `v_pos = verts*2-1` (`[-1,1]`).
5. **Appearance** (browser path = per-vertex color; the CUDA UV baker is not portable):
   - `query_triplane(triplane, v_pos)` → `[M,120]` (bilinear, align_corners=True, planes xy,xz,yz).
   - `features_mlp` (JSON) → albedo `[M,3]`.
   - `roughness/metallic` not emitted by the ONNX graphs → fixed defaults (see config).
6. **Mesh**: vertex normals from faces; export GLB (three.js) with vertex colors + PBR factors.
7. Final mesh orientation in the reference is `rot(-90°,X)`, `rot(+90°,Y)`, `invert`; the viewer
   applies an equivalent transform so +Z is up/front as in the original export.

## query_triplane (align_corners=True)
For position `p=(x,y,z)∈[-1,1]`, sample plane `P∈{0,1,2}` at `(u,v)`:
`P0:(x,y)  P1:(x,z)  P2:(y,z)`; pixel `= (coord+1)/2*(size-1)`; bilinear; concat 3×40 → 120.

## features_mlp (color head, from features_mlp_weights.json)
`120 → 64 → 64 → 64 → 3`; SiLU after each hidden layer; final `w3[3,64]·h2 + b3`.
Output activation is **configurable** (`config.COLOR_OUTPUT_ACTIVATION`, default `none`→clamp[0,1]);
the upstream `config.yaml` is gated, and `MaterialMLP` defaults to identity. Flip to `sigmoid`
if albedo looks wrong at runtime.

## Marching tetrahedra tables (from isosurface.py)
- `base_tet_edges = [0,1, 0,2, 0,3, 1,2, 1,3, 2,3]`
- `num_triangles_table = [0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0]`
- `triangle_table`: 16×6 (see src/core/marchingTets.ts)
- occupancy bit `tetindex = o0*1 + o1*2 + o2*4 + o3*8`; edge vertex = zero-crossing lerp
  `v = p_min*(-s_max/(s_min-s_max)) + p_max*(s_min/(s_min-s_max))`.

## Verification status (this environment: no GPU, no browser, ~1 GiB RAM)
Proven here: TypeScript build, unit tests (marching tets on synthetic SDF, triplane sampling,
color head, camera math, preprocess math), ONNX I/O signature match, asset sha256 scheme.
NOT runnable here (needs your WebGPU browser): the actual 912 MB FP16 backbone inference and
on-screen 3D. See README "Running" + "Assumptions to confirm at runtime".
