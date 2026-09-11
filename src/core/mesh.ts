import * as THREE from "three";
import { GLTFExporter } from "three/examples/jsm/exporters/GLTFExporter.js";
import type { MeshData } from "../types";

export function buildThreeMesh(md: MeshData): THREE.Mesh {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(md.positions, 3));
  g.setAttribute("normal", new THREE.BufferAttribute(md.normals, 3));
  g.setAttribute("color", new THREE.BufferAttribute(md.colors, 3));
  g.setIndex(new THREE.BufferAttribute(md.indices, 1));
  g.computeBoundingSphere();
  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true, roughness: md.roughness, metalness: md.metallic, side: THREE.DoubleSide,
  });
  return new THREE.Mesh(g, mat);
}

export function exportGLB(md: MeshData): Promise<ArrayBuffer> {
  const mesh = buildThreeMesh(md);
  const exporter = new GLTFExporter();
  return new Promise((resolve, reject) => {
    exporter.parse(mesh, (res) => resolve(res as ArrayBuffer), (err) => reject(err), { binary: true });
  });
}
