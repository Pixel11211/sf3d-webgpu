import type * as ort from "onnxruntime-web";
import {
  DEFAULT_METALLIC, DEFAULT_ROUGHNESS, ISO_RESOLUTION, ISO_THRESHOLD,
  TET_VERTEX_COUNT, TRIPLANE_CHANNELS, TRIPLANE_PLANES, TRIPLANE_SIZE,
} from "../config";
import type { MeshData, ProgressFn, RGBImage } from "../types";
import { fetchAsset, fetchInt32, fetchJson } from "./assets";
import { defaultCondC2W, intrinsicNormed } from "./camera";
import { colorHeadForward, parseColorHead, type ColorHeadJson, type ColorHeadWeights } from "./colorHead";
import { computeVertexNormals, flipWinding, rotateToGlTF } from "./geometry";
import { marchingTetrahedra } from "./marchingTets";
import { scalePositions } from "./math";
import { createSession, f32Tensor, initWebGPU, outF32, withWebGPUErrorScope, type WebGPUDiagnostics } from "./onnx";
import { preprocessToRgbCond } from "./preprocess";
import { makeTriplane, queryTriplane } from "./triplane";

export class SF3DPipeline {
  private tokenizer?: ort.InferenceSession;
  private backbone?: ort.InferenceSession;
  private decoder?: ort.InferenceSession;
  private tetsVerts?: Float32Array;
  private tetsIdx?: Int32Array;
  private color?: ColorHeadWeights;
  private gpu?: WebGPUDiagnostics;
  private _loaded = false;

  get loaded(): boolean { return this._loaded; }
  get gpuDiagnostics(): WebGPUDiagnostics | undefined { return this.gpu; }

  /** Download + verify all artifacts and create the ONNX sessions. */
  async load(onProgress?: ProgressFn): Promise<void> {
    if (this._loaded) return;
    onProgress?.("webgpu", 0, "requesting device (shader-f16 + max limits)");
    this.gpu = await initWebGPU();
    if (this.gpu.error) console.warn("[sf3d] WebGPU init:", this.gpu.error);
    console.info("[sf3d] WebGPU diagnostics:", this.gpu);

    onProgress?.("grid+color-head", 0, "tet grid, color MLP");
    const [vertsBuf, idx, mlp] = await Promise.all([
      fetchAsset("tetsVertices", onProgress),
      fetchInt32("tetsIndices", onProgress),
      fetchJson<ColorHeadJson>("featuresMlp", onProgress),
    ]);
    this.tetsVerts = new Float32Array(vertsBuf);
    this.tetsIdx = idx;
    this.color = parseColorHead(mlp);

    onProgress?.("decoder", 0, "decoder_single.onnx");
    this.decoder = await createSession(await fetchAsset("decoder", onProgress));

    onProgress?.("tokenizer", 0, "image_tokenizer_single.onnx (~764 MB)");
    this.tokenizer = await createSession(await fetchAsset("imageTokenizer", onProgress));

    onProgress?.("backbone", 0, "backbone_fp16.onnx (~912 MB)");
    this.backbone = await createSession(await fetchAsset("backbone", onProgress));

    this._loaded = true;
    onProgress?.("ready", 1, "model ready");
  }

  /** image -> MeshData (positions/normals/colors/indices in glTF orientation). */
  async generate(src: Blob | RGBImage, onProgress?: ProgressFn): Promise<MeshData> {
    if (!this._loaded || !this.tokenizer || !this.backbone || !this.decoder || !this.tetsVerts || !this.tetsIdx || !this.color)
      throw new Error("pipeline not loaded");

    onProgress?.("preprocess", 0.02, "crop/resize/composite -> 512");
    const { rgb } = await preprocessToRgbCond(src as RGBImage);

    onProgress?.("tokenizer", 0.1, "DINOv2 image tokens");
    const tok = await withWebGPUErrorScope(() => this.tokenizer!.run({
      rgb: f32Tensor(rgb, [1, 512, 512, 3]),
      c2w: f32Tensor(defaultCondC2W(), [1, 4, 4]),
      intrinsic_normed: f32Tensor(intrinsicNormed(), [1, 3, 3]),
    }), "image_tokenizer");
    const imageTokens = tok["image_tokens"];

    onProgress?.("backbone", 0.35, "triplane [1,3,40,384,384]");
    const bb = await withWebGPUErrorScope(() => this.backbone!.run({ image_tokens: imageTokens }), "backbone_fp16");
    const triData = outF32(bb["triplane"]);
    const triplane = makeTriplane(triData, TRIPLANE_PLANES, TRIPLANE_CHANNELS, TRIPLANE_SIZE);

    onProgress?.("geometry", 0.6, "density + vertex_offset at grid");
    const nv = TET_VERTEX_COUNT;
    const gridPos = new Float32Array(this.tetsVerts);
    scalePositions(gridPos, 0, 1, -1, 1);
    const dec = await withWebGPUErrorScope(() => this.decoder!.run({
      triplane: f32Tensor(triData, [1, TRIPLANE_PLANES, TRIPLANE_CHANNELS, TRIPLANE_SIZE, TRIPLANE_SIZE]),
      positions: f32Tensor(gridPos, [1, nv, 3]),
    }), "decoder_single");
    const density = outF32(dec["density"]);
    const voff = outF32(dec["vertex_offset"]);

    // Diagnostics: distinguish "WebGPU produced garbage" from "threshold/segmentation".
    const stat = (a: Float32Array) => {
      let mn = Infinity, mx = -Infinity, sum = 0, fin = 0;
      for (let i = 0; i < a.length; i++) { const v = a[i]; if (Number.isFinite(v)) { fin++; if (v < mn) mn = v; if (v > mx) mx = v; sum += v; } }
      return { min: mn, max: mx, mean: a.length ? sum / a.length : 0, finite: fin, len: a.length };
    };
    let inside = 0;
    for (let i = 0; i < density.length; i++) if (density[i] - ISO_THRESHOLD > 0) inside++;
    console.info("[sf3d] triplane stats:", stat(triData));
    console.info("[sf3d] density stats:", stat(density), "| threshold:", ISO_THRESHOLD, "| inside(sdf>0):", `${inside}/${density.length}`);

    const sdf = new Float32Array(nv);
    for (let i = 0; i < nv; i++) sdf[i] = density[i] - ISO_THRESHOLD;
    const defGrid = new Float32Array(nv * 3);
    const k = 1 / ISO_RESOLUTION;
    for (let i = 0; i < nv * 3; i++) defGrid[i] = this.tetsVerts[i] + k * Math.tanh(voff[i]);

    onProgress?.("marching-tets", 0.75, `${(this.tetsIdx!.length / 4) | 0} tets`);
    const { vertices, indices } = marchingTetrahedra(defGrid, sdf, this.tetsIdx);
    const m = vertices.length / 3;
    if (m === 0) throw new Error(
      `empty mesh: no isosurface extracted (inside verts ${inside}/${nv}). Open DevTools console for the ` +
      `[sf3d] triplane/density stats and any WebGPU root error. Likely: WebGPU FP16 pipeline failure ` +
      `(see the GPU line in the log — shader-f16 should be "yes"), or the input is not a clean single-object cutout.`,
    );
    scalePositions(vertices, 0, 1, -1, 1); // -> world [-1,1]

    onProgress?.("color", 0.88, `${m} verts`);
    const feats = queryTriplane(triplane, vertices, m);           // sample in SF3D world space
    const colors = colorHeadForward(this.color, feats, m);        // albedo [m,3]

    rotateToGlTF(vertices);                                       // export orientation
    flipWinding(indices);
    const normals = computeVertexNormals(vertices, indices);

    onProgress?.("done", 1, `${m} verts, ${(indices.length / 3) | 0} tris`);
    return { positions: vertices, normals, colors, indices, roughness: DEFAULT_ROUGHNESS, metallic: DEFAULT_METALLIC };
  }
}
