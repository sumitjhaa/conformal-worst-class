// Split conformal prediction in the browser. Port of site/app.js, which mirrors src/scores.py + src/conformal.py:
// k = ceil((n+1)(1-alpha)), q = k-th smallest calibration score (inf if k > n), C(x) = {y : s(x,y) <= q}.
// PAC variants pick the smallest k with P(Beta(k, n+1-k) >= 1-alpha) >= 1-delta_g, delta_g = 1-(1-delta)^(1/G).

export type RawBlock = { n: number; logits: string; labels?: number[]; imgs?: string[] };
export type Block = { n: number; K: number; logits: Float32Array; labels: Int32Array | null; imgs: string[] | null; salt: number; p?: Float32Array; pT?: number; u?: Float32Array };
export type Method = "top1" | "naive" | "aps" | "raps" | "mondrian" | "clustered_pac";
export type Raps = { lam: number; k_reg: number };
export type Params = { method: Method; alpha: number; T: number; delta: number; raps?: Raps };
export type Cal = { q?: number; qc?: number[] } | null;

export function i16(b64: string, scale = 100): Float32Array {
  const bin = atob(b64), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  const a = new Int16Array(u8.buffer), out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] / scale;
  return out;
}
export function decodeBlock(b: RawBlock, salt: number, K: number): Block {
  return { n: b.n, K, logits: i16(b.logits), labels: b.labels ? Int32Array.from(b.labels) : null, imgs: b.imgs || null, salt };
}
export function rng(seed: number) { // mulberry32
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// ---------------------------------------------------------------- special functions
function lgamma(z: number): number { // Lanczos
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61503916999185, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
  z -= 1; let x = c[0]; for (let i = 1; i < 9; i++) x += c[i] / (z + i);
  const t = z + 7.5; return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}
function betacf(x: number, a: number, b: number) { // continued fraction (Numerical Recipes)
  const FP = 1e-300, qab = a + b, qap = a + 1, qam = a - 1; let c = 1, d = 1 - (qab * x) / qap;
  if (Math.abs(d) < FP) d = FP; d = 1 / d; let h = d;
  for (let m = 1; m <= 5000; m++) {
    const m2 = 2 * m; let aa = (m * (b - m) * x) / ((qam + m2) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FP) d = FP; c = 1 + aa / c; if (Math.abs(c) < FP) c = FP; d = 1 / d; h *= d * c;
    aa = (-(a + m) * (qab + m) * x) / ((a + m2) * (qap + m2));
    d = 1 + aa * d; if (Math.abs(d) < FP) d = FP; c = 1 + aa / c; if (Math.abs(c) < FP) c = FP; d = 1 / d;
    const del = d * c; h *= del; if (Math.abs(del - 1) < 1e-12) break;
  }
  return h;
}
function ibeta(x: number, a: number, b: number) {
  if (x <= 0) return 0; if (x >= 1) return 1;
  const lbt = lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x);
  return x < (a + 1) / (a + b + 2) ? Math.exp(lbt) * betacf(x, a, b) / a : 1 - Math.exp(lbt) * betacf(1 - x, b, a) / b;
}
export const betaSf = (x: number, a: number, b: number) => 1 - ibeta(x, a, b);
export const kIndex = (n: number, alpha: number) => Math.ceil(Math.round((n + 1) * (1 - alpha) * 1e9) / 1e9);
export function pacK(n: number, alpha: number, delta: number): number | null {
  let lo = Math.max(1, kIndex(n, alpha)), hi = n;
  if (n < 1 || lo > n || betaSf(1 - alpha, n, 1) < 1 - delta) return null;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (betaSf(1 - alpha, mid, n + 1 - mid) >= 1 - delta) hi = mid; else lo = mid + 1; }
  return lo;
}
export function betaPdf(x: number, a: number, b: number) {
  if (x <= 0 || x >= 1) return 0;
  return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) + lgamma(a + b) - lgamma(a) - lgamma(b));
}

/** Distribution of the worst of K independent per-class true coverages, each Beta(k, n+1-k) (the project's worst-class law). */
export function worstLaw(K: number, n: number, alpha: number, k: number | null = kIndex(n, alpha)) {
  if (k == null || k > n) return null;
  const a = k, b = n + 1 - k, G = 300;
  const t0 = Math.max(0, 1 - alpha - 8 * Math.sqrt(alpha / Math.max(n, 1)) - 0.05);
  const pts: [number, number][] = []; let prev = 1 - Math.pow(betaSf(t0, a, b), K), mean = t0, q10 = NaN, q90 = NaN;
  for (let i = 1; i <= G; i++) {
    const t = t0 + ((1 - t0) * i) / G, c = 1 - Math.pow(betaSf(t, a, b), K);
    pts.push([t, (c - prev) / ((1 - t0) / G)]); mean += ((1 - t0) / G) * (1 - c);
    if (Number.isNaN(q10) && c >= 0.1) q10 = t; if (Number.isNaN(q90) && c >= 0.9) q90 = t; prev = c;
  }
  return { pts, mean, q10, q90, pBelow: 1 - Math.pow(betaSf(1 - alpha, a, b), K) };
}

// ---------------------------------------------------------------- conformal maths
export function probs(b: Block, T: number): Float32Array {
  if (b.p && b.pT === T) return b.p;
  const K = b.K, z = b.logits, p = new Float32Array(b.n * K);
  for (let i = 0; i < b.n; i++) {
    let m = -Infinity;
    for (let k = 0; k < K; k++) m = Math.max(m, z[i * K + k] / T);
    let s = 0;
    for (let k = 0; k < K; k++) { const e = Math.exp(z[i * K + k] / T - m); p[i * K + k] = e; s += e; }
    for (let k = 0; k < K; k++) p[i * K + k] /= s;
  }
  b.p = p; b.pT = T;
  return p;
}
function uniforms(n: number, salt: number) { const r = rng(salt * 7919 + 13), u = new Float32Array(n); for (let i = 0; i < n; i++) u[i] = r(); return u; }

/** Nonconformity score of every (image, label): naive = 1 - p(y|x); APS = randomised cumulative probability mass up to y;
 *  RAPS = APS + lam * max(rank - k_reg, 0) with 1-indexed rank (src/scores.py raps_scores). */
export function scoreMatrix(b: Block, m: Method, T: number, raps: Raps = { lam: 0.1, k_reg: 1 }): Float32Array {
  const K = b.K, n = b.n, S = new Float32Array(n * K), p = probs(b, T);
  if (m === "naive" || m === "top1") { for (let j = 0; j < n * K; j++) S[j] = 1 - p[j]; return S; }
  b.u = b.u || uniforms(n, b.salt);
  const idx = new Array<number>(K);
  for (let i = 0; i < n; i++) {
    for (let k = 0; k < K; k++) idx[k] = k;
    idx.sort((a, c) => p[i * K + c] - p[i * K + a]);
    let cum = 0;
    for (let r = 0; r < K; r++) { const k = idx[r], pk = p[i * K + k]; S[i * K + k] = cum + b.u[i] * pk + (m === "raps" ? raps.lam * Math.max(r + 1 - raps.k_reg, 0) : 0); cum += pk; }
  }
  return S;
}
const scoreOf = (m: Method): Method => (m === "mondrian" || m === "clustered_pac" ? "aps" : m);
function kth(arr: number[], k: number | null) { if (k == null || k > arr.length) return Infinity; if (k <= 0) return -Infinity; return Float64Array.from(arr).sort()[k - 1]; }
const quantile = (arr: number[], alpha: number) => kth(arr, kIndex(arr.length, alpha));

function groupThresholds(scores: number[], groups: number[], G: number, alpha: number, delta: number | null) {
  const per: number[][] = Array.from({ length: G }, () => []);
  scores.forEach((s, i) => { if (groups[i] >= 0) per[groups[i]].push(s); });
  const present = per.filter((a) => a.length).length || 1, dg = delta == null ? null : 1 - Math.pow(1 - delta, 1 / present);
  return per.map((a) => (dg == null ? quantile(a, alpha) : kth(a, pacK(a.length, alpha, dg))));
}

/** Calibrate on D_cal. Mondrian: one threshold per class. Clustered-PAC: one Beta-corrected threshold per cluster. */
export function calibrate(cal: Block, P: Params, clusters?: number[]): Cal {
  if (P.method === "top1") return null;
  const K = cal.K, S = scoreMatrix(cal, scoreOf(P.method), P.T, P.raps), y = cal.labels!;
  const s = Array.from({ length: cal.n }, (_, i) => S[i * K + y[i]]), ys = Array.from(y);
  if (P.method === "mondrian") return { qc: groupThresholds(s, ys, K, P.alpha, null) };
  if (P.method === "clustered_pac" && clusters) {
    const G = Math.max(...clusters) + 1, qg = groupThresholds(s, ys.map((c) => clusters[c]), G, P.alpha, P.delta), q0 = quantile(s, P.alpha);
    return { qc: clusters.map((g) => (g >= 0 ? qg[g] : q0)) };
  }
  return { q: quantile(s, P.alpha) };
}

export function predictSets(b: Block, P: Params, cal: Cal): Uint8Array {
  const n = b.n, K = b.K, C = new Uint8Array(n * K);
  if (P.method === "top1" || !cal) {
    for (let i = 0; i < n; i++) { let best = 0; for (let k = 1; k < K; k++) if (b.logits[i * K + k] > b.logits[i * K + best]) best = k; C[i * K + best] = 1; }
    return C;
  }
  const S = scoreMatrix(b, scoreOf(P.method), P.T, P.raps);
  for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) C[i * K + k] = S[i * K + k] <= threshold(cal, k) ? 1 : 0;
  return C;
}

export function metrics(b: Block, C: Uint8Array) {
  const n = b.n, K = b.K, y = b.labels!, cHit = new Array(K).fill(0), cN = new Array(K).fill(0);
  let hits = 0, size = 0, empty = 0, single = 0;
  for (let i = 0; i < n; i++) {
    let s = 0; for (let k = 0; k < K; k++) s += C[i * K + k];
    size += s; if (s === 0) empty++; if (s === 1) single++;
    const h = C[i * K + y[i]]; hits += h; cHit[y[i]] += h; cN[y[i]]++;
  }
  const classCov = cHit.map((h, c) => (cN[c] ? h / cN[c] : NaN));
  return { cov: hits / n, mean: size / n, empty: empty / n, single: single / n, classCov, worst: Math.min(...classCov.filter(Number.isFinite)) };
}

/** The cut-off label k is compared against (per-class methods have one per class). */
export const threshold = (cal: Cal, k: number) => (!cal ? NaN : cal.qc ? cal.qc[k] : cal.q!);

/** Equal-width confidence bins (src/calibrate.py reliability_bins) and ECE = sum_b |B_b|/n * |acc_b - conf_b|. */
export function reliability(b: Block, T: number, bins = 15) {
  const K = b.K, p = probs(b, T), y = b.labels!, count = new Array(bins).fill(0), acc = new Array(bins).fill(0), conf = new Array(bins).fill(0);
  for (let i = 0; i < b.n; i++) {
    let best = 0; for (let k = 1; k < K; k++) if (p[i * K + k] > p[i * K + best]) best = k;
    const c = p[i * K + best], j = Math.min(bins - 1, Math.floor(c * bins));
    count[j]++; conf[j] += c; acc[j] += best === y[i] ? 1 : 0;
  }
  let ece = 0;
  for (let j = 0; j < bins; j++) if (count[j]) { acc[j] /= count[j]; conf[j] /= count[j]; ece += (count[j] / b.n) * Math.abs(acc[j] - conf[j]); }
  return { count, acc, conf, ece };
}

/** K x K counts: how often labels a and b are in the same set (diagonal = how often a is in a set at all). */
export function cooccurrence(C: Uint8Array, n: number, K: number) {
  const M = new Float64Array(K * K), inSet: number[] = [];
  for (let i = 0; i < n; i++) {
    inSet.length = 0; for (let k = 0; k < K; k++) if (C[i * K + k]) inSet.push(k);
    for (const a of inSet) for (const c of inSet) M[a * K + c]++;
  }
  return M;
}

export function setOf(C: Uint8Array, i: number, K: number) { const out: number[] = []; for (let k = 0; k < K; k++) if (C[i * K + k]) out.push(k); return out; }

/** Coverage of many random re-splits of the pooled calibration + test true-label scores (naive score). */
export function resplitCoverages(cal: Block, test: Block, T: number, nCal: number, alpha: number, reps: number, seed: number) {
  const pool: number[] = [];
  for (const b of [cal, test]) { const p = probs(b, T); for (let i = 0; i < b.n; i++) pool.push(1 - p[i * b.K + b.labels![i]]); }
  const r = rng(seed), out: number[] = [], N = pool.length, a = Float64Array.from(pool), k = kIndex(nCal, alpha);
  for (let rep = 0; rep < reps; rep++) {
    for (let i = N - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = a[i]; a[i] = a[j]; a[j] = t; }
    const q = k > nCal ? Infinity : Float64Array.from(a.subarray(0, nCal)).sort()[k - 1];
    let hit = 0; for (let i = nCal; i < N; i++) if (a[i] <= q) hit++;
    out.push(hit / (N - nCal));
  }
  return out;
}

export const pct = (v: number, d = 1) => (Number.isFinite(v) ? (v * 100).toFixed(d) + "%" : "—");
export const mean = (a: number[]) => { const v = a.filter(Number.isFinite); return v.length ? v.reduce((x, y) => x + y, 0) / v.length : NaN; };
