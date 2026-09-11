export function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

/** scale_tensor(dat, (inLo,inHi), (outLo,outHi)) from stable-fast-3d. */
export function scaleValue(x: number, inLo: number, inHi: number, outLo: number, outHi: number): number {
  return ((x - inLo) / (inHi - inLo)) * (outHi - outLo) + outLo;
}

export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/** SiLU / swish: x * sigmoid(x). */
export function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

export function tanh(x: number): number {
  return Math.tanh(x);
}

/** In-place scale a flat [N*3] array from [inLo,inHi] to [outLo,outHi]. */
export function scalePositions(arr: Float32Array, inLo: number, inHi: number, outLo: number, outHi: number): Float32Array {
  const s = (outHi - outLo) / (inHi - inLo);
  for (let i = 0; i < arr.length; i++) arr[i] = (arr[i] - inLo) * s + outLo;
  return arr;
}
