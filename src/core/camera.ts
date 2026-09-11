import { COND_IMAGE_SIZE, DEFAULT_DISTANCE, DEFAULT_FOVY_DEG } from "../config";

/**
 * default_cond_c2w(distance) — the fixed camera-to-world matrix SF3D conditions on.
 * Row-major 4x4, matching the `c2w` ONNX input [B,4,4].
 */
export function defaultCondC2W(distance: number = DEFAULT_DISTANCE): Float32Array {
  return new Float32Array([
    0, 0, 1, distance,
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 0, 1,
  ]);
}

/**
 * create_intrinsic_from_fov_deg(...)[1] — the normalized intrinsic passed to the
 * tokenizer as `intrinsic_normed` [B,3,3] (row-major).
 *   f = 0.5*H / tan(0.5*fov);  normalized: [0,0]/=W, [1,1]/=H, [0,2]/=W, [1,2]/=H
 */
export function intrinsicNormed(
  fovDeg: number = DEFAULT_FOVY_DEG,
  h: number = COND_IMAGE_SIZE,
  w: number = COND_IMAGE_SIZE,
): Float32Array {
  const fov = (fovDeg * Math.PI) / 180;
  const f = (0.5 * h) / Math.tan(0.5 * fov);
  return new Float32Array([
    f / w, 0, (w / 2) / w,
    0, f / h, (h / 2) / h,
    0, 0, 1,
  ]);
}
