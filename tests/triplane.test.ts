import test from "node:test";
import assert from "node:assert/strict";
import { makeTriplane, queryTriplane } from "../src/core/triplane";

test("triplane: constant planes sample to their constants; feature dim = planes*channels", () => {
  const planes = 3, C = 1, size = 3;
  const data = new Float32Array(planes * C * size * size);
  const consts = [2, 7, 9];
  for (let p = 0; p < planes; p++) for (let i = 0; i < C*size*size; i++) data[p*(C*size*size)+i] = consts[p];
  const tp = makeTriplane(data, planes, C, size);
  const positions = new Float32Array([0.1, -0.4, 0.7]); // single point
  const out = queryTriplane(tp, positions, 1);
  assert.equal(out.length, planes * C);
  assert.ok(Math.abs(out[0] - consts[0]) < 1e-5);
  assert.ok(Math.abs(out[1] - consts[1]) < 1e-5);
  assert.ok(Math.abs(out[2] - consts[2]) < 1e-5);
});

test("triplane: align_corners linear ramp is reproduced exactly (plane0 = x)", () => {
  const planes = 3, C = 1, size = 3;
  const data = new Float32Array(planes * C * size * size);
  const planeStride = C * size * size;
  // plane0 data[r][c] = c (0..size-1); planes 1,2 constant 5
  for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) data[0*planeStride + r*size + c] = c;
  for (let p = 1; p < planes; p++) for (let i = 0; i < planeStride; i++) data[p*planeStride+i] = 5;
  const tp = makeTriplane(data, planes, C, size);
  // align_corners: fx=(x+1)*0.5*(size-1)=(x+1); value at column fx (linear) = fx
  for (const x of [-1, -0.5, 0, 0.5, 1]) {
    const out = queryTriplane(tp, new Float32Array([x, 0, 0]), 1);
    const expected = (x + 1) * 0.5 * (size - 1); // = x+1
    assert.ok(Math.abs(out[0] - expected) < 1e-4, `x=${x}: got ${out[0]} want ${expected}`);
    assert.ok(Math.abs(out[1] - 5) < 1e-4);
    assert.ok(Math.abs(out[2] - 5) < 1e-4);
  }
});
