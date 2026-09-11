// Faithful port of MarchingTetrahedraHelper._forward from
// Stability-AI/stable-fast-3d sf3d/models/isosurface.py.
// Pure stdlib TypeScript (no three/ORT) so it is unit-testable on Node.

// 16 cases x 6 edge-slots. Values index the 6 tet edges (see BASE_TET_EDGES); -1 = unused.
const TRIANGLE_TABLE: readonly (readonly number[])[] = [
  [-1, -1, -1, -1, -1, -1],
  [ 1,  0,  2, -1, -1, -1],
  [ 4,  0,  3, -1, -1, -1],
  [ 1,  4,  2,  1,  3,  4],
  [ 3,  1,  5, -1, -1, -1],
  [ 2,  3,  0,  2,  5,  3],
  [ 1,  4,  0,  1,  5,  4],
  [ 4,  2,  5, -1, -1, -1],
  [ 4,  5,  2, -1, -1, -1],
  [ 4,  1,  0,  4,  5,  1],
  [ 3,  2,  0,  3,  5,  2],
  [ 1,  3,  5, -1, -1, -1],
  [ 4,  1,  2,  4,  3,  1],
  [ 3,  0,  4, -1, -1, -1],
  [ 2,  0,  1, -1, -1, -1],
  [-1, -1, -1, -1, -1, -1],
];
const NUM_TRIANGLES_TABLE: readonly number[] = [0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0];
// 6 edges per tet as local vertex-index pairs, ordered to match TRIANGLE_TABLE slots.
const BASE_TET_EDGES: readonly number[] = [0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3];

export interface MarchingTetsResult {
  vertices: Float32Array; // [M*3] surface vertices (same coord space as `pos`)
  indices: Uint32Array;   // [F*3] triangle vertex indices
}

/**
 * Extract the sdf>0 isosurface from a tetrahedral grid.
 * @param pos  grid vertex positions, flat [Nv*3]
 * @param sdf  per-vertex level, [Nv]; inside = sdf > 0
 * @param tets tetrahedra vertex indices, [Nt*4] (int32), referencing pos
 */
export function marchingTetrahedra(
  pos: Float32Array,
  sdf: Float32Array,
  tets: Int32Array,
): MarchingTetsResult {
  const nv = sdf.length;
  const nt = (tets.length / 4) | 0;
  const occ = new Uint8Array(nv);
  for (let i = 0; i < nv; i++) occ[i] = sdf[i] > 0 ? 1 : 0;

  const verts: number[] = [];
  const faces: number[] = [];
  // key = min*nv + max (unique, < 2^53 for nv up to ~5.4e5) -> new vertex index
  const edgeMap = new Map<number, number>();
  const idxMap = new Int32Array(6);
  const local = new Int32Array(4);

  for (let t = 0; t < nt; t++) {
    const b = t * 4;
    local[0] = tets[b]; local[1] = tets[b + 1]; local[2] = tets[b + 2]; local[3] = tets[b + 3];
    const o0 = occ[local[0]], o1 = occ[local[1]], o2 = occ[local[2]], o3 = occ[local[3]];
    const sum = o0 + o1 + o2 + o3;
    if (sum === 0 || sum === 4) continue;
    const tetIndex = o0 | (o1 << 1) | (o2 << 2) | (o3 << 3);
    const nTri = NUM_TRIANGLES_TABLE[tetIndex];
    if (nTri === 0) continue;

    for (let e = 0; e < 6; e++) {
      const a = local[BASE_TET_EDGES[e * 2]];
      const c = local[BASE_TET_EDGES[e * 2 + 1]];
      if (occ[a] === occ[c]) { idxMap[e] = -1; continue; }
      const mn = a < c ? a : c;
      const mx = a < c ? c : a;
      const key = mn * nv + mx;
      let vi = edgeMap.get(key);
      if (vi === undefined) {
        const s0 = sdf[mn], s1 = sdf[mx];
        const denom = s0 - s1;
        const w0 = denom !== 0 ? -s1 / denom : 0.5;
        const w1 = denom !== 0 ? s0 / denom : 0.5;
        const p0 = mn * 3, p1 = mx * 3;
        vi = verts.length / 3;
        verts.push(
          pos[p0] * w0 + pos[p1] * w1,
          pos[p0 + 1] * w0 + pos[p1 + 1] * w1,
          pos[p0 + 2] * w0 + pos[p1 + 2] * w1,
        );
        edgeMap.set(key, vi);
      }
      idxMap[e] = vi;
    }

    const tri = TRIANGLE_TABLE[tetIndex];
    faces.push(idxMap[tri[0]], idxMap[tri[1]], idxMap[tri[2]]);
    if (nTri === 2) faces.push(idxMap[tri[3]], idxMap[tri[4]], idxMap[tri[5]]);
  }

  return { vertices: new Float32Array(verts), indices: new Uint32Array(faces) };
}
