import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export class Viewer {
  private renderer: THREE.WebGLRenderer;
  private scene = new THREE.Scene();
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private current?: THREE.Object3D;
  private raf = 0;

  constructor(private canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene.background = new THREE.Color(0x0b0d12);
    this.camera = new THREE.PerspectiveCamera(40, 1, 0.01, 100);
    this.camera.position.set(2.2, 1.6, 2.6);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.6); key.position.set(3, 5, 4); this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x8899ff, 0.5); fill.position.set(-4, -1, -3); this.scene.add(fill);
    const grid = new THREE.GridHelper(4, 16, 0x2a2f3a, 0x1a1e27); grid.position.y = -1.05; this.scene.add(grid);

    addEventListener("resize", () => this.resize());
    this.resize();
    this.loop();
  }

  setMesh(obj: THREE.Object3D): void {
    if (this.current) { this.scene.remove(this.current); this.disposeObject(this.current); }
    this.current = obj; this.scene.add(obj);
    const box = new THREE.Box3().setFromObject(obj);
    const c = box.getCenter(new THREE.Vector3());
    obj.position.sub(c); // recenter
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }

  private disposeObject(obj: THREE.Object3D): void {
    obj.traverse((o) => {
      const m = o as THREE.Mesh;
      if (m.geometry) m.geometry.dispose();
      const mat = m.material as THREE.Material | THREE.Material[] | undefined;
      if (Array.isArray(mat)) mat.forEach((x) => x.dispose()); else mat?.dispose();
    });
  }

  private resize(): void {
    const w = this.canvas.clientWidth || 640, h = this.canvas.clientHeight || 480;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }

  private loop = (): void => {
    this.raf = requestAnimationFrame(this.loop);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  dispose(): void { cancelAnimationFrame(this.raf); this.renderer.dispose(); }
}
