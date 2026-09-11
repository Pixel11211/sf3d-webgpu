export interface MeshData {
  positions: Float32Array; // [V*3], world coords in [-1,1]
  normals: Float32Array;   // [V*3]
  colors: Float32Array;    // [V*3], albedo in [0,1]
  indices: Uint32Array;    // [F*3]
  roughness: number;
  metallic: number;
}

export type ProgressFn = (stage: string, frac: number, detail?: string) => void;

export interface RGBImage {
  width: number;
  height: number;
  data: Uint8ClampedArray; // RGBA, row-major
}

export interface Segmenter {
  /** Return an RGBA image whose alpha is the foreground mask. */
  segment(img: RGBImage): Promise<RGBImage>;
}
