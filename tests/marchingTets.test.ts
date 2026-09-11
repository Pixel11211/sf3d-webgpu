import test from "node:test";
import assert from "node:assert/strict";
import { marchingTetrahedra } from "../src/core/marchingTets";

// Build an (n+1)^3 grid over [0,1]^3, each cube -> 5 tets.
function buildGrid(n: number) {
  const N = n + 1;
  const pos = new Float32Array(N * N * N * 3);
  const idx = (i: number, j: number, k: number) => (i * N + j) * N + k;
  let p = 0;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) for (let k = 0; k < N; k++) {
    pos[p++] = i / n; pos[p++] = j / n; pos[p++] = k / n;
  }
  const tets: number[] = [];
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) {
    // cube corners: v000..v111 ; local index by (di,dj,dk)
    const v = (di: number, dj: number, dk: number) => idx(i + di, j + dj, k + dk);
    const v000 = v(0,0,0), v100 = v(1,0,0), v110 = v(1,1,0), v010 = v(0,1,0);
    const v001 = v(0,0,1), v101 = v(1,0,1), v111 = v(1,1,1), v011 = v(0,1,1);
    // standard 5-tet decomposition
    tets.push(v000, v100, v010, v001);
    tets.push(v100, v110, v010, v111);
    tets.push(v100, v001, v101, v111);
    tets.push(v010, v001, v111, v011);
    tets.push(v100, v010, v001, v111);
  }
  return { pos, tets: Int32Array.from(tets), nv: N * N * N };
}

test("marching tets: linear SDF plane -> vertices lie exactly on the plane", () => {
  const n = 5;
  const { pos, tets, nv } = buildGrid(n);
  const plane = 0.53;
  const sdf = new Float32Array(nv);
  for (let i = 0; i < nv; i++) sdf[i] = pos[i * 3] - plane; // linear in x
  const { vertices, indices } = marchingTetrahedra(pos, sdf, tets);
  const m = vertices.length / 3;
  assert.ok(m > 0, "expected some surface vertices");
  assert.equal(indices.length % 3, 0, "faces are triangles");
  assert.ok(indices.length / 3 > 0, "expected some faces");
  for (let i = 0; i < m; i++) {
    assert.ok(Math.abs(vertices[i * 3] - plane) < 1e-4, `vertex x=${vertices[i*3]} not on plane`);
  }
  for (let f = 0; f < indices.length; f++) {
    assert.ok(indices[f] >= 0 && indices[f] < m, "face index in range");
  }
});

test("marching tets: sphere SDF -> closed surface near the radius", () => {
  const n = 8;
  const { pos, tets, nv } = buildGrid(n);
  const c = 0.5, r = 0.3;
  const sdf = new Float32Array(nv);
  for (let i = 0; i < nv; i++) {
    const dx = pos[i*3]-c, dy = pos[i*3+1]-c, dz = pos[i*3+2]-c;
    sdf[i] = r - Math.sqrt(dx*dx+dy*dy+dz*dz); // >0 inside
  }
  const { vertices, indices } = marchingTetrahedra(pos, sdf, tets);
  const m = vertices.length / 3;
  assert.ok(m > 200, `expected a substantial sphere mesh, got ${m} verts`);
  let maxErr = 0;
  for (let i = 0; i < m; i++) {
    const dx = vertices[i*3]-c, dy = vertices[i*3+1]-c, dz = vertices[i*3+2]-c;
    maxErr = Math.max(maxErr, Math.abs(Math.sqrt(dx*dx+dy*dy+dz*dz) - r));
  }
  assert.ok(maxErr < 0.08, `sphere radius error too large: ${maxErr}`);
  // every vertex used by at least the index range
  assert.ok(indices.length / 3 > 200, "expected many faces");
});

test("marching tets: empty when all outside", () => {
  const n = 3;
  const { pos, tets, nv } = buildGrid(n);
  const sdf = new Float32Array(nv).fill(-1); // all outside
  const { vertices, indices } = marchingTetrahedra(pos, sdf, tets);
  assert.equal(vertices.length, 0);
  assert.equal(indices.length, 0);
});
