// Pure mesh geometry helpers (no three.js) — unit-testable.

/** Accumulate face normals per vertex, then normalize. positions/indices in the same space. */
export function computeVertexNormals(positions: Float32Array, indices: Uint32Array): Float32Array {
  const n = new Float32Array(positions.length);
  for (let f = 0; f < indices.length; f += 3) {
    const ia = indices[f] * 3, ib = indices[f + 1] * 3, ic = indices[f + 2] * 3;
    const ax = positions[ia], ay = positions[ia + 1], az = positions[ia + 2];
    const ux = positions[ib] - ax, uy = positions[ib + 1] - ay, uz = positions[ib + 2] - az;
    const vx = positions[ic] - ax, vy = positions[ic + 1] - ay, vz = positions[ic + 2] - az;
    const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    n[ia] += nx; n[ia + 1] += ny; n[ia + 2] += nz;
    n[ib] += nx; n[ib + 1] += ny; n[ib + 2] += nz;
    n[ic] += nx; n[ic + 1] += ny; n[ic + 2] += nz;
  }
  for (let i = 0; i < n.length; i += 3) {
    const l = Math.hypot(n[i], n[i + 1], n[i + 2]) || 1;
    n[i] /= l; n[i + 1] /= l; n[i + 2] /= l;
  }
  return n;
}

/** rot(-90°,X) then rot(+90°,Y)  =>  (x,y,z) -> (-y, z, -x). In place. */
export function rotateToGlTF(arr: Float32Array): void {
  for (let i = 0; i < arr.length; i += 3) {
    const x = arr[i], y = arr[i + 1], z = arr[i + 2];
    arr[i] = -y; arr[i + 1] = z; arr[i + 2] = -x;
  }
}

/** trimesh.invert(): reverse triangle winding. In place. */
export function flipWinding(indices: Uint32Array): void {
  for (let f = 0; f < indices.length; f += 3) {
    const t = indices[f + 1]; indices[f + 1] = indices[f + 2]; indices[f + 2] = t;
  }
}
