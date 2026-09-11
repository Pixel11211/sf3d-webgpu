// features_mlp color head: 120 -> 64 -> 64 -> 64 -> 3 (SiLU hidden), from
// needle-tools features_mlp_weights.json. Pure TS, unit-testable.
import { clamp, sigmoid, silu } from "./math";
import { COLOR_OUTPUT_ACTIVATION, TRIPLANE_FEATURE_DIM } from "../config";

export interface ColorHeadWeights {
  w0: Float32Array; b0: Float32Array; // [64,120] [64]
  w1: Float32Array; b1: Float32Array; // [64,64]  [64]
  w2: Float32Array; b2: Float32Array; // [64,64]  [64]
  w3: Float32Array; b3: Float32Array; // [3,64]   [3]
}

const HID = 64;
const IN = TRIPLANE_FEATURE_DIM; // 120
const OUT = 3;

export interface ColorHeadJson {
  w0: number[][]; b0: number[];
  w1: number[][]; b1: number[];
  w2: number[][]; b2: number[];
  w3: number[][]; b3: number[];
}

function flat2d(a: number[][]): Float32Array {
  const rows = a.length, cols = a[0].length;
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) out[i * cols + j] = a[i][j];
  return out;
}

export function parseColorHead(json: ColorHeadJson): ColorHeadWeights {
  return {
    w0: flat2d(json.w0), b0: Float32Array.from(json.b0),
    w1: flat2d(json.w1), b1: Float32Array.from(json.b1),
    w2: flat2d(json.w2), b2: Float32Array.from(json.b2),
    w3: flat2d(json.w3), b3: Float32Array.from(json.b3),
  };
}

/** Run the color head on [count,120] features -> [count,3] albedo in [0,1]. */
export function colorHeadForward(w: ColorHeadWeights, feats: Float32Array, count: number, out?: Float32Array): Float32Array {
  const result = out && out.length >= count * OUT ? out : new Float32Array(count * OUT);
  const h0 = new Float32Array(HID), h1 = new Float32Array(HID), h2 = new Float32Array(HID);
  for (let n = 0; n < count; n++) {
    const xOff = n * IN;
    for (let i = 0; i < HID; i++) {
      let acc = w.b0[i]; const row = i * IN;
      for (let j = 0; j < IN; j++) acc += w.w0[row + j] * feats[xOff + j];
      h0[i] = silu(acc);
    }
    for (let i = 0; i < HID; i++) {
      let acc = w.b1[i]; const row = i * HID;
      for (let j = 0; j < HID; j++) acc += w.w1[row + j] * h0[j];
      h1[i] = silu(acc);
    }
    for (let i = 0; i < HID; i++) {
      let acc = w.b2[i]; const row = i * HID;
      for (let j = 0; j < HID; j++) acc += w.w2[row + j] * h1[j];
      h2[i] = silu(acc);
    }
    for (let i = 0; i < OUT; i++) {
      let acc = w.b3[i]; const row = i * HID;
      for (let j = 0; j < HID; j++) acc += w.w3[row + j] * h2[j];
      const v = COLOR_OUTPUT_ACTIVATION === "sigmoid" ? sigmoid(acc) : acc;
      result[n * OUT + i] = clamp(v, 0, 1);
    }
  }
  return result;
}
