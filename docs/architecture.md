# Architecture

## Data flow (with tensor shapes)
```
File/Blob
  └─ preprocess.preprocessToRgbCond
       ├─ foregroundBounds(alpha)            -> bbox
       ├─ foregroundCrop(bbox, 0.85)         -> square crop (source px)
       ├─ canvas transform crop+resize       -> 512x512 RGBA (transparent pad)
       └─ compositeBackground                -> rgb Float32[512*512*3] (HWC, [0,1]), mask
  └─ camera.defaultCondC2W(1.6)              -> Float32[16]  (c2w, row-major)
  └─ camera.intrinsicNormed(40,512,512)      -> Float32[9]
pipeline.generate
  ├─ tokenizer.run({rgb[1,512,512,3], c2w[1,4,4], intrinsic_normed[1,3,3]})
  │     └─ image_tokens Float32[1,1297,1024]
  ├─ backbone.run({image_tokens})            -> triplane Float32[1,3,40,384,384]
  ├─ gridPos = tets_vertices*2-1             -> Float32[1,535882,3]
  ├─ decoder.run({triplane, positions=gridPos})
  │     ├─ density Float32[535882]
  │     └─ vertex_offset Float32[535882*3]
  ├─ sdf = density - 10.0
  ├─ defGrid = tets_vertices + (1/160)*tanh(vertex_offset)   (still [0,1])
  ├─ marchingTets.marchingTetrahedra(defGrid, sdf, tets_indices)
  │     └─ vertices Float32[M*3] ([0,1]), indices Uint32[F*3]
  ├─ scalePositions(vertices, 0,1, -1,1)     -> world [-1,1]
  ├─ triplane.queryTriplane(triplane, vertices, M)  -> feats Float32[M*120]
  ├─ colorHead.colorHeadForward(weights, feats, M)  -> colors Float32[M*3]
  ├─ geometry.rotateToGlTF(vertices); geometry.flipWinding(indices)
  └─ geometry.computeVertexNormals(vertices, indices) -> normals
mesh.buildThreeMesh / mesh.exportGLB        -> THREE.Mesh / GLB ArrayBuffer
render.Viewer.setMesh                        -> three.js scene
```

## Module boundaries
- **Pure, dependency-free (unit-tested on Node):** `camera`, `math`, `triplane`,
  `marchingTets`, `colorHead`, `geometry`, and the pure parts of `preprocess`
  (`foregroundBounds`, `foregroundCrop`, `compositeBackground`).
- **Browser/IO:** `assets` (fetch+crypto+caches), `onnx` (ORT Web), the canvas parts of
  `preprocess`, `mesh`/`viewer` (three.js), `main`/`ui` (DOM).
- **Orchestration:** `pipeline` ties them together; it is the only module that knows the
  end-to-end order and the ONNX I/O names.

## Key decisions
1. **Vertex colors instead of UV baking.** The reference bakes albedo into a UV atlas with a
   CUDA/Metal kernel (`texture_baker`). That is not portable to the browser, so we sample the
   triplane at each mesh vertex and store per-vertex albedo. Geometry and color values are
   identical; only the texture representation differs.
2. **Color head from JSON.** `decoder_single.onnx` emits only `density`/`vertex_offset`; the
   albedo MLP ships separately as `features_mlp_weights.json`, so we run it in TS.
3. **Single tet-grid decode pass.** `decoder_single` does the triplane sampling for geometry
   internally; we only re-sample in TS for color (at the far smaller set of surface vertices).
4. **RAM-safe contract extraction.** `tools/inspect_onnx_io.py` streams the protobuf so the
   912 MB graph's I/O can be read without loading weights (how `docs/onnx-io.json` was made).
