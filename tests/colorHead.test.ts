import test from "node:test";
import assert from "node:assert/strict";
import { parseColorHead, colorHeadForward, type ColorHeadWeights } from "../src/core/colorHead";

function zeros(r: number, c: number) { return Array.from({length:r},()=>Array.from({length:c},()=>0)); }

test("color head: parse shapes from nested-array JSON", () => {
  const json = {
    w0: zeros(64,120), b0: Array(64).fill(0),
    w1: zeros(64,64),  b1: Array(64).fill(0),
    w2: zeros(64,64),  b2: Array(64).fill(0),
    w3: zeros(3,64),   b3: [0.2,0.5,0.8],
  };
  const w = parseColorHead(json as any);
  assert.equal(w.w0.length, 64*120);
  assert.equal(w.w3.length, 3*64);
  assert.equal(w.b3.length, 3);
  // all-zero weights => output == b3 (clamped)
  const feats = new Float32Array(2*120).fill(0.1);
  const out = colorHeadForward(w, feats, 2);
  assert.equal(out.length, 6);
  for (let n=0;n<2;n++){
    assert.ok(Math.abs(out[n*3+0]-0.2)<1e-6);
    assert.ok(Math.abs(out[n*3+1]-0.5)<1e-6);
    assert.ok(Math.abs(out[n*3+2]-0.8)<1e-6);
  }
});

test("color head: output always within [0,1]", () => {
  // random-ish weights
  const rnd = (r:number,c:number,scale:number)=>Array.from({length:r},()=>Array.from({length:c},()=>(Math.random()-0.5)*scale));
  const w = parseColorHead({
    w0: rnd(64,120,0.2), b0: rnd(64,1,0.1).map(a=>a[0]),
    w1: rnd(64,64,0.2),  b1: rnd(64,1,0.1).map(a=>a[0]),
    w2: rnd(64,64,0.2),  b2: rnd(64,1,0.1).map(a=>a[0]),
    w3: rnd(3,64,0.5),   b3: rnd(3,1,2).map(a=>a[0]),
  } as any);
  const feats = new Float32Array(5*120); for(let i=0;i<feats.length;i++) feats[i]=(Math.random()-0.5)*4;
  const out = colorHeadForward(w, feats, 5);
  for (let i=0;i<out.length;i++) assert.ok(out[i]>=0 && out[i]<=1, `albedo out of range: ${out[i]}`);
});
