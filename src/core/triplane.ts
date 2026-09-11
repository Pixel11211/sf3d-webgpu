import { clamp } from "./math";

export interface Triplane {
  data: Float32Array; // layout [planes, channels, size, size] row-major
  planes: number;
  channels: number;
  size: number;
}

export function makeTriplane(data: Float32Array, planes: number, channels: number, size: number): Triplane {
  return { data, planes, channels, size };
}

/**
 * query_triplane: bilinear sample (grid_sample, align_corners=True) of the 3 planes
 * at N positions in [-1,1]. Plane 0 uses (x,y), plane 1 (x,z), plane 2 (y,z).
 * Returns [N, planes*channels] (120 for SF3D).
 */
export function queryTriplane(tp: Triplane, positions: Float32Array, count: number, out?: Float32Array): Float32Array {
  const { data, planes, channels, size } = tp;
  const C = channels, W = size, H = size;
  const planeStride = C * H * W;
  const cw = H * W;
  const featDim = planes * C;
  const result = out && out.length >= count * featDim ? out : new Float32Array(count * featDim);
  const s = (size - 1) * 0.5; // (coord+1)*0.5*(size-1)

  for (let n = 0; n < count; n++) {
    const x = positions[n * 3], y = positions[n * 3 + 1], z = positions[n * 3 + 2];
    const outBase = n * featDim;
    for (let p = 0; p < planes; p++) {
      const u = p === 2 ? y : x;      // plane0:(x,.) plane1:(x,.) plane2:(y,.)
      const v = p === 0 ? y : z;      // plane0:(.,y) plane1:(.,z) plane2:(.,z)
      const fx = (u + 1) * s;
      const fy = (v + 1) * s;
      let x0 = Math.floor(fx), y0 = Math.floor(fy);
      const tx = fx - x0, ty = fy - y0;
      let x1 = x0 + 1, y1 = y0 + 1;
      x0 = clamp(x0, 0, W - 1); x1 = clamp(x1, 0, W - 1);
      y0 = clamp(y0, 0, H - 1); y1 = clamp(y1, 0, H - 1);
      const pOff = p * planeStride;
      const r00 = pOff + y0 * W + x0, r01 = pOff + y0 * W + x1;
      const r10 = pOff + y1 * W + x0, r11 = pOff + y1 * W + x1;
      const w00 = (1 - tx) * (1 - ty), w01 = tx * (1 - ty), w10 = (1 - tx) * ty, w11 = tx * ty;
      const o = outBase + p * C;
      for (let c = 0; c < C; c++) {
        const cOff = c * cw;
        result[o + c] = data[r00 + cOff] * w00 + data[r01 + cOff] * w01 + data[r10 + cOff] * w10 + data[r11 + cOff] * w11;
      }
    }
  }
  return result;
}
