// Central configuration + verified constants for the SF3D WebGPU pipeline.
// Values cross-checked against Stability-AI/stable-fast-3d source and the
// needle-tools/SF3D-webgpu ONNX I/O. See docs/model-contract.md.

export const HF_REPO = "needle-tools/SF3D-webgpu";
export const HF_BASE = `https://huggingface.co/${HF_REPO}/resolve/main`;

export interface AssetSpec {
  path: string;
  bytes: number;
  sha256: string;
}

// From assets-manifest.json (schemaVersion 1). Streamed from HF, sha256-verified.
export const ASSETS = {
  imageTokenizer: { path: "onnx/image_tokenizer_single.onnx", bytes: 763808910, sha256: "cc8c76276a9cf86ece8a854827ed570bdbf71e458df09bde5664af46af29c21e" },
  backbone:       { path: "onnx/backbone_fp16.onnx",          bytes: 911863351, sha256: "8bcde6d22589e8bbb753c4ca1a91f2c800f27a794b75405ef0dbee6b07b0da12" },
  decoder:        { path: "onnx/decoder_single.onnx",         bytes: 104430,     sha256: "ec4655df567128cc86b18b8a86b7b79d092d5223c8521a6d9993d08209e2785d" },
  tetsVertices:   { path: "tets_vertices.bin",                bytes: 6430584,    sha256: "16f4ee01a050d1757c19b13a7e1dbd4d0918d08208d961c105ff74cbfc345dac" },
  tetsIndices:    { path: "tets_indices.bin",                 bytes: 47543232,   sha256: "606cc8b47f8744a64ff6f1f3c088d9d9113ff80539bd62cb1651f8dc629d1f1e" },
  featuresMlp:    { path: "features_mlp_weights.json",        bytes: 349798,     sha256: "b493e908039ebb70bf99f451b3ff85dc70cb2a62b29cefd17453ebdf27b8b4d7" },
} as const satisfies Record<string, AssetSpec>;

export type AssetKey = keyof typeof ASSETS;

// ONNX Runtime Web WASM binaries, pinned to the installed version.
// To self-host: copy node_modules/onnxruntime-web/dist/*.{wasm,mjs} into public/ort/
// and set this to "ort/" (see README).
export const ORT_VERSION = "1.29.0";
export const ORT_WASM_PATHS = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;

// --- pipeline constants (stable-fast-3d defaults) ---
export const COND_IMAGE_SIZE = 512;
export const BACKGROUND_COLOR: readonly [number, number, number] = [0.5, 0.5, 0.5];
export const FOREGROUND_RATIO = 0.85;

export const ISO_RESOLUTION = 160;
export const ISO_THRESHOLD = 10.0;
export const RADIUS = 1.0;
export const DEFAULT_FOVY_DEG = 40.0;
export const DEFAULT_DISTANCE = 1.6;

export const TRIPLANE_PLANES = 3;
export const TRIPLANE_CHANNELS = 40;
export const TRIPLANE_SIZE = 384;
export const TRIPLANE_FEATURE_DIM = TRIPLANE_PLANES * TRIPLANE_CHANNELS; // 120

export const TET_VERTEX_COUNT = 535882;
export const TET_COUNT = 2971452;

// The ONNX graphs do not emit roughness/metallic; use fixed PBR material defaults.
export const DEFAULT_ROUGHNESS = 0.75;
export const DEFAULT_METALLIC = 0.0;

export type ColorActivation = "none" | "sigmoid";
export const COLOR_OUTPUT_ACTIVATION: ColorActivation = "none";

// Image tokenizer DINOv2 normalization (informational): baked into the ONNX graph,
// so the app feeds raw [0,1] rgb. Kept here for reference / toggling experiments.
export const DINOV2_MEAN: readonly [number, number, number] = [0.485, 0.456, 0.406];
export const DINOV2_STD: readonly [number, number, number] = [0.229, 0.224, 0.225];
