// The seed-0 MNIST LeNet (src/models.py) run in plain TypeScript, so "Draw a digit" needs no ML runtime.
// conv5x5(1→32, pad 2) → ReLU → maxpool2 → conv5x5(32→64) → ReLU → maxpool2 → flatten(1600) → fc128 → ReLU → fc10.
// Weights come from scripts/export_mnist_net.py; scripts/check-explorer.mjs checks the logits against torch.

type Layer = { shape: number[]; data: string };
export type NetJson = { mean: number; std: number; layers: Record<string, Layer>; check: { img: string; logits: number[] }[] };
export type Net = { mean: number; std: number; w: Record<string, Float32Array> };

const f32 = (b64: string) => {
  const bin = atob(b64), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(u8.buffer);
};
export const loadNet = (j: NetJson): Net => ({ mean: j.mean, std: j.std, w: Object.fromEntries(Object.entries(j.layers).map(([k, l]) => [k, f32(l.data)])) });

function conv(x: Float32Array, cin: number, h: number, wd: number, W: Float32Array, B: Float32Array, cout: number, pad: number) {
  const oh = h + 2 * pad - 4, ow = wd + 2 * pad - 4, out = new Float32Array(cout * oh * ow);
  for (let o = 0; o < cout; o++) for (let y = 0; y < oh; y++) for (let xx = 0; xx < ow; xx++) {
    let s = B[o];
    for (let c = 0; c < cin; c++) for (let ky = 0; ky < 5; ky++) {
      const iy = y + ky - pad; if (iy < 0 || iy >= h) continue;
      for (let kx = 0; kx < 5; kx++) { const ix = xx + kx - pad; if (ix >= 0 && ix < wd) s += x[(c * h + iy) * wd + ix] * W[((o * cin + c) * 5 + ky) * 5 + kx]; }
    }
    out[(o * oh + y) * ow + xx] = s > 0 ? s : 0; // ReLU fused
  }
  return { out, h: oh, w: ow };
}
function pool(x: Float32Array, c: number, h: number, w: number) {
  const oh = h >> 1, ow = w >> 1, out = new Float32Array(c * oh * ow);
  for (let k = 0; k < c; k++) for (let y = 0; y < oh; y++) for (let xx = 0; xx < ow; xx++) {
    const i = (k * h + 2 * y) * w + 2 * xx;
    out[(k * oh + y) * ow + xx] = Math.max(x[i], x[i + 1], x[i + w], x[i + w + 1]);
  }
  return out;
}
function fc(x: Float32Array, W: Float32Array, B: Float32Array, relu: boolean) {
  const n = B.length, m = x.length, out = new Float32Array(n);
  for (let o = 0; o < n; o++) { let s = B[o]; for (let i = 0; i < m; i++) s += W[o * m + i] * x[i]; out[o] = relu && s < 0 ? 0 : s; }
  return out;
}

/** Logits for one 28x28 image of pixel values in [0, 255] (white digit on black, like MNIST). */
export function lenet(net: Net, px: ArrayLike<number>): number[] {
  const x = new Float32Array(784);
  for (let i = 0; i < 784; i++) x[i] = (px[i] / 255 - net.mean) / net.std;
  const w = net.w;
  const c1 = conv(x, 1, 28, 28, w["body.0.weight"], w["body.0.bias"], 32, 2), p1 = pool(c1.out, 32, c1.h, c1.w);
  const c2 = conv(p1, 32, 14, 14, w["body.3.weight"], w["body.3.bias"], 64, 0), p2 = pool(c2.out, 64, c2.h, c2.w);
  return Array.from(fc(fc(p2, w["body.7.weight"], w["body.7.bias"], true), w["head.weight"], w["head.bias"], false));
}
