import { BACKGROUND_COLOR, COND_IMAGE_SIZE, FOREGROUND_RATIO } from "../config";
import type { RGBImage } from "../types";
import { clamp } from "./math";

export interface Bounds { x0: number; y0: number; x1: number; y1: number; }
export interface Crop { left: number; top: number; size: number; }

/** get_bbox_from_mask: first/last column & row with mask>thr. Pure/testable. */
export function foregroundBounds(mask: ArrayLike<number>, w: number, h: number, thr = 0.5): Bounds | null {
  let x0 = -1, x1 = -1, y0 = -1, y1 = -1;
  for (let y = 0; y < h; y++) {
    let rowHit = false;
    for (let x = 0; x < w; x++) {
      if (mask[y * w + x] > thr) {
        rowHit = true;
        if (x0 < 0 || x < x0) x0 = x;
        if (x > x1) x1 = x;
      }
    }
    if (rowHit) { if (y0 < 0) y0 = y; y1 = y; }
  }
  if (x0 < 0) return null;
  return { x0, y0, x1, y1 };
}

/** resize_foreground: square crop centered on the bbox so the foreground fills `ratio`. Pure/testable. */
export function foregroundCrop(b: Bounds, ratio = FOREGROUND_RATIO): Crop {
  const bw = b.x1 - b.x0, bh = b.y1 - b.y0;
  const xc = (b.x0 + b.x1) / 2, yc = (b.y0 + b.y1) / 2;
  const size = Math.max(bw, bh) / ratio;
  return { left: xc - size / 2, top: yc - size / 2, size };
}

/** rgb_cond = lerp(bg, rgb, alpha). Writes HWC float32 [size*size*3] in [0,1]. Pure/testable. */
export function compositeBackground(rgba: Uint8ClampedArray, size: number, out?: Float32Array): { rgb: Float32Array; mask: Float32Array } {
  const rgb = out ?? new Float32Array(size * size * 3);
  const mask = new Float32Array(size * size);
  const [br, bg, bb] = BACKGROUND_COLOR;
  for (let i = 0; i < size * size; i++) {
    const a = rgba[i * 4 + 3] / 255;
    mask[i] = a;
    rgb[i * 3 + 0] = br * (1 - a) + (rgba[i * 4 + 0] / 255) * a;
    rgb[i * 3 + 1] = bg * (1 - a) + (rgba[i * 4 + 1] / 255) * a;
    rgb[i * 3 + 2] = bb * (1 - a) + (rgba[i * 4 + 2] / 255) * a;
  }
  return { rgb, mask };
}

// ---- browser glue (canvas) ----
function makeCanvas(size: number): { canvas: any; ctx: any } {
  if (typeof OffscreenCanvas !== "undefined") {
    const c = new OffscreenCanvas(size, size);
    return { canvas: c, ctx: c.getContext("2d", { willReadFrequently: true })! };
  }
  const c = document.createElement("canvas");
  c.width = c.height = size;
  return { canvas: c, ctx: c.getContext("2d", { willReadFrequently: true })! };
}

export async function loadImageRGBA(src: Blob | string): Promise<RGBImage> {
  const bitmap = typeof src === "string"
    ? await createImageBitmap(await (await fetch(src)).blob())
    : await createImageBitmap(src);
  const { canvas, ctx } = makeCanvas(bitmap.width);
  canvas.height = bitmap.height;
  ctx.drawImage(bitmap, 0, 0);
  const d = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  bitmap.close?.();
  return { width: bitmap.width, height: bitmap.height, data: d.data };
}

/**
 * Full prepare_image + resize_foreground: crop the foreground to `ratio`, resize to 512,
 * composite onto the gray background. Returns the tokenizer `rgb` input (HWC [0,1]) + mask.
 */
export async function preprocessToRgbCond(
  src: Blob | RGBImage,
  ratio = FOREGROUND_RATIO,
  size = COND_IMAGE_SIZE,
): Promise<{ rgb: Float32Array; mask: Float32Array }> {
  const img: RGBImage = src instanceof Object && "data" in (src as any) ? (src as RGBImage) : await loadImageRGBA(src as Blob);
  const { width: w, height: h, data } = img;
  const mask = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) mask[i] = data[i * 4 + 3] / 255;
  let bounds = foregroundBounds(mask, w, h, 0.5);
  if (!bounds) bounds = { x0: 0, y0: 0, x1: w - 1, y1: h - 1 };
  const crop = foregroundCrop(bounds, ratio);

  // One-shot crop+resize via a canvas transform; areas outside the source stay transparent.
  const { canvas, ctx } = makeCanvas(size);
  ctx.clearRect(0, 0, size, size);
  const f = size / crop.size;
  // Need an image source for drawImage; if given RGBImage, blit it into a temp canvas first.
  let source: CanvasImageSource;
  if (src instanceof Object && "data" in (src as any)) {
    const tmp = makeCanvas(w); (tmp.canvas as any).height = h;
    tmp.ctx.putImageData(new ImageData(data as unknown as Uint8ClampedArray<ArrayBuffer>, w, h), 0, 0);
    source = tmp.canvas;
  } else {
    source = await createImageBitmap(src as Blob);
  }
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.setTransform(f, 0, 0, f, -crop.left * f, -crop.top * f);
  ctx.drawImage(source, 0, 0);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const out = ctx.getImageData(0, 0, size, size);
  void canvas;
  return compositeBackground(out.data, size);
}

export { clamp };
