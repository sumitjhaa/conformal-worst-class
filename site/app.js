// Live conformal prediction in the browser. Mirrors src/scores.py + src/conformal.py:
// k = ceil((n+1)(1-alpha)), q_hat = k-th smallest calibration score (inf if k > n), C(x) = {y : s(x,y) <= q_hat}.
// PAC variants pick the smallest k with P(Beta(k, n+1-k) >= 1-alpha) >= 1-delta_g, delta_g = 1-(1-delta)^(1/G).
"use strict";
const $ = (id) => document.getElementById(id);
const DS_LABEL = { mnist: "MNIST", cifar10: "CIFAR-10", cifar10_r18: "CIFAR-10 · ResNet-18", cifar100: "CIFAR-100", imagenet: "ImageNet" };
const RUN_ORDER = ["mnist", "cifar10", "cifar10_r18", "cifar100", "imagenet"];
const SHORT = { cifar10: ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"] };
const METHODS = {
  top1: { name: "Top-1 (baseline)", hint: "Always one label. No coverage guarantee." },
  naive: { name: "Naive  1 − p(y|x)", hint: "Least-ambiguous sets; smallest on average." },
  margin: { name: "Margin", hint: "max other p − p(y). Another probability score." },
  aps: { name: "APS", hint: "Adds labels in probability order until the mass reaches q̂." },
  raps: { name: "RAPS", hint: "APS with a penalty λ on labels ranked below k_reg." },
  saps: { name: "SAPS", hint: "Keeps only the top probability; λ per extra rank." },
  learned: { name: "Learned score", hint: "Residual head trained on D_design for small sets (fixed before D_cal)." },
  dist_naive: { name: "Naive + distance", hint: "1 − p_T(y|x) + γ·d̃_M(x,y). First proposal." },
  dist_aps: { name: "APS + distance", hint: "s_APS(p_T) + γ·d̃_M(x,y). First proposal." },
  mondrian: { name: "Mondrian (per-class APS)", hint: "One q̂ per class: per-class coverage on average over calibration draws." },
  mondrian_pac: { name: "Mondrian-PAC", hint: "Beta-corrected q̂ per class: every class ≥ 1−α at once, w.p. ≥ 1−δ." },
  clustered: { name: "Clustered", hint: "One q̂ per cluster of similar classes (clusters fit on D_design)." },
  clustered_pac: { name: "Clustered-PAC (proposed)", hint: "Beta-corrected q̂ per cluster: high-probability guarantee, far cheaper." },
};
const PRETTY = { top1: "Top-1", naive: "Naive", aps: "APS", raps: "RAPS", temp_naive: "Naive (T)", temp_aps: "APS (T)", margin: "Margin",
  saps: "SAPS", conftr: "Learned", learned: "Learned", dist_naive: "Naive+dist", dist_aps: "APS+dist", mondrian: "Mondrian",
  mondrian_pac: "Mondrian-PAC", clustered: "Clustered", clustered_pac: "Clustered-PAC", ls_bbse: "Weighted (BBSE)", ls_oracle: "Weighted (oracle)" };
const GROUPED = new Set(["mondrian", "mondrian_pac", "clustered", "clustered_pac"]);
const APS_LIKE = new Set(["aps", "raps", "dist_aps", ...GROUPED]);

const D = {};            // dataset name -> decoded data
let R = null;            // results.json
let IN = null;           // ImageNet gallery
const st = { ds: "cifar10", method: "aps", alpha: 0.1, T: 1, gamma: 0, lam: 0.1, kreg: 1, delta: 0.1, ncal: 5000, calSeed: 1, filter: "all", sel: null, sev: 0, src: "synthetic" };

// ---------------------------------------------------------------- decoding
function i16(b64, scale = 100) {
  const bin = atob(b64), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  const a = new Int16Array(u8.buffer), out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] / scale;
  return out;
}
function decodeBlock(b, salt, K) {
  return { n: b.n, K, logits: i16(b.logits), learned: b.learned ? i16(b.learned) : null, dist: b.dist ? i16(b.dist) : null,
    labels: b.labels ? Int32Array.from(b.labels) : null, imgs: b.imgs || null, salt, cache: {} };
}
function rng(seed) { // mulberry32
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function uniforms(n, salt) { const r = rng(salt * 7919 + 13), u = new Float32Array(n); for (let i = 0; i < n; i++) u[i] = r(); return u; }

// ---------------------------------------------------------------- special functions
function lgamma(z) { // Lanczos
  const g = 7, c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61503916999185, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - lgamma(1 - z);
  z -= 1; let x = c[0]; for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const t = z + g + 0.5; return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}
function betacf(x, a, b) { // continued fraction for the incomplete beta (Numerical Recipes)
  const FP = 1e-300; let qab = a + b, qap = a + 1, qam = a - 1, c = 1, d = 1 - (qab * x) / qap;
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
function ibeta(x, a, b) { // regularised incomplete beta I_x(a, b)
  if (x <= 0) return 0; if (x >= 1) return 1;
  const lbt = lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x);
  return x < (a + 1) / (a + b + 2) ? Math.exp(lbt) * betacf(x, a, b) / a : 1 - Math.exp(lbt) * betacf(1 - x, b, a) / b;
}
const betaSf = (x, a, b) => 1 - ibeta(x, a, b);
const kIndex = (n, alpha) => Math.ceil(Math.round((n + 1) * (1 - alpha) * 1e9) / 1e9);
function pacK(n, alpha, delta) { // smallest k with P(Beta(k, n+1-k) >= 1-alpha) >= 1-delta, or null
  let lo = Math.max(1, kIndex(n, alpha)), hi = n;
  if (n < 1 || lo > n || betaSf(1 - alpha, n, 1) < 1 - delta) return null;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (betaSf(1 - alpha, mid, n + 1 - mid) >= 1 - delta) hi = mid; else lo = mid + 1; }
  return lo;
}
function bbSurvival(m, a, b) { // S[j] = P(X >= j), X ~ BetaBinomial(m, a, b), j = 0..m+1
  const lB = lgamma(a) + lgamma(b) - lgamma(a + b), lm = lgamma(m + 1), pmf = new Float64Array(m + 1);
  for (let j = 0; j <= m; j++) pmf[j] = Math.exp(lm - lgamma(j + 1) - lgamma(m - j + 1) + lgamma(j + a) + lgamma(m - j + b) - lgamma(m + a + b) - lB);
  const S = new Float64Array(m + 2); for (let j = m; j >= 0; j--) S[j] = S[j + 1] + pmf[j];
  return S;
}
function gammaSample(r, k) { // Marsaglia-Tsang, k > 0
  if (k < 1) return gammaSample(r, k + 1) * Math.pow(r(), 1 / k);
  const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) { let x, v; do { const u1 = r(), u2 = r(); x = Math.sqrt(-2 * Math.log(u1 || 1e-12)) * Math.cos(2 * Math.PI * u2); v = 1 + c * x; } while (v <= 0);
    v = v * v * v; const u = r(); if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v; }
}
function dirichlet(r, K, beta) { const g = Array.from({ length: K }, () => gammaSample(r, beta)), s = g.reduce((a, b) => a + b, 0); return g.map((v) => v / s); }

// ---------------------------------------------------------------- conformal maths
function probs(b, T, src = "logits") {
  const key = src + T, K = b.K;
  if (b.cache[key]) return b.cache[key];
  const z = b[src], p = new Float32Array(b.n * K), t = src === "learned" ? 1 : T;
  for (let i = 0; i < b.n; i++) {
    let m = -Infinity;
    for (let k = 0; k < K; k++) m = Math.max(m, z[i * K + k] / t);
    let s = 0;
    for (let k = 0; k < K; k++) { const e = Math.exp(z[i * K + k] / t - m); p[i * K + k] = e; s += e; }
    for (let k = 0; k < K; k++) p[i * K + k] /= s;
  }
  for (const k of Object.keys(b.cache)) if (k.startsWith(src)) delete b.cache[k]; // one temperature per source
  b.cache[key] = p;
  return p;
}
function scoreMatrix(b, m, P) {
  const K = b.K, n = b.n, S = new Float32Array(n * K);
  if (m === "learned" && b.learned) { const p = probs(b, 1, "learned"); for (let j = 0; j < n * K; j++) S[j] = 1 - p[j]; return S; }
  const p = probs(b, m === "top1" ? 1 : P.T);
  if (m === "naive" || m === "top1" || m === "dist_naive" || m === "learned") {
    for (let j = 0; j < n * K; j++) S[j] = 1 - p[j];
  } else if (m === "margin") {
    for (let i = 0; i < n; i++) { let a = -1, bb = -1; for (let k = 0; k < K; k++) { const v = p[i * K + k]; if (v > a) { bb = a; a = v; } else if (v > bb) bb = v; }
      for (let k = 0; k < K; k++) { const v = p[i * K + k]; S[i * K + k] = (v === a ? bb : a) - v; } }
  } else {
    b.u = b.u || uniforms(n, b.salt);
    const idx = new Array(K);
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < K; k++) idx[k] = k;
      idx.sort((a, c) => p[i * K + c] - p[i * K + a]);
      if (m === "saps") {
        const pm = p[i * K + idx[0]];
        for (let r = 0; r < K; r++) S[i * K + idx[r]] = r === 0 ? b.u[i] * pm : pm + (r - 1 + b.u[i]) * P.lam;
        continue;
      }
      let cum = 0;
      for (let r = 0; r < K; r++) {
        const k = idx[r], pk = p[i * K + k];
        S[i * K + k] = cum + b.u[i] * pk + (m === "raps" ? P.lam * Math.max(r + 1 - P.kreg, 0) : 0);
        cum += pk;
      }
    }
  }
  if (m.startsWith("dist_") && b.dist) for (let j = 0; j < n * K; j++) S[j] += P.gamma * b.dist[j];
  return S;
}
function sortedQuantileK(arr, k) { if (k == null || k > arr.length) return Infinity; if (k <= 0) return -Infinity; return Float64Array.from(arr).sort()[k - 1]; }
function quantile(arr, alpha) { return sortedQuantileK(arr, kIndex(arr.length, alpha)); }
function calIndex(d, P) {
  if (!d.calPerm || d.calPermSeed !== P.calSeed) {
    const r = rng(P.calSeed), a = Array.from({ length: d.cal.n }, (_, i) => i);
    for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    d.calPerm = a; d.calPermSeed = P.calSeed;
  }
  return d.calPerm.slice(0, Math.min(P.ncal, d.cal.n));
}
// Group thresholds: one per group of calibration points; pac => simultaneous high-probability version.
function groupThresholds(scores, groups, G, alpha, delta) {
  const per = Array.from({ length: G }, () => []);
  scores.forEach((s, i) => { if (groups[i] >= 0) per[groups[i]].push(s); });
  const present = per.filter((a) => a.length).length || 1, dg = delta == null ? null : 1 - Math.pow(1 - delta, 1 / present);
  return per.map((a) => dg == null ? quantile(a, alpha) : sortedQuantileK(a, pacK(a.length, alpha, dg)));
}
// Calibrate on D_cal: returns {q: number} or {qc: per-class thresholds}.
function calibrate(d, P) {
  if (P.method === "top1") return null;
  const K = d.K, S = scoreMatrix(d.cal, P.method, P), idx = calIndex(d, P), y = d.cal.labels;
  const s = idx.map((i) => S[i * K + y[i]]), ys = idx.map((i) => y[i]);
  if (P.method === "mondrian" || P.method === "mondrian_pac") return { qc: groupThresholds(s, ys, K, P.alpha, P.method === "mondrian_pac" ? P.delta : null) };
  if (P.method === "clustered" || P.method === "clustered_pac") {
    const cmap = d.clusters, G = Math.max(...cmap) + 1, qg = groupThresholds(s, ys.map((c) => cmap[c]), G, P.alpha, P.method === "clustered_pac" ? P.delta : null);
    const q0 = quantile(s, P.alpha);
    return { qc: cmap.map((g) => (g >= 0 ? qg[g] : q0)), G };
  }
  return { q: quantile(s, P.alpha) };
}
const thr = (cal, k) => (cal.qc ? cal.qc[k] : cal.q);
function predictSets(b, P, cal) {
  const n = b.n, K = b.K, C = new Uint8Array(n * K);
  if (P.method === "top1") {
    for (let i = 0; i < n; i++) { let best = 0; for (let k = 1; k < K; k++) if (b.logits[i * K + k] > b.logits[i * K + best]) best = k; C[i * K + best] = 1; }
    return { C, S: null };
  }
  const S = scoreMatrix(b, P.method, P);
  for (let i = 0; i < n; i++) for (let k = 0; k < K; k++) C[i * K + k] = S[i * K + k] <= thr(cal, k) ? 1 : 0;
  return { C, S };
}
function metrics(b, C, T) {
  const n = b.n, K = b.K, y = b.labels, sizes = new Int32Array(n), hit = new Uint8Array(n);
  const hist = new Array(K + 1).fill(0), cHit = new Array(K).fill(0), cN = new Array(K).fill(0);
  for (let i = 0; i < n; i++) {
    let s = 0; for (let k = 0; k < K; k++) s += C[i * K + k];
    sizes[i] = s; hist[s]++;
    if (y) { hit[i] = C[i * K + y[i]]; cHit[y[i]] += hit[i]; cN[y[i]]++; }
  }
  const mean = sizes.reduce((a, v) => a + v, 0) / n;
  const out = { n, sizes, hist: hist.map((v) => v / n), mean, empty: hist[0] / n, single: hist[1] / n };
  if (y) {
    out.cov = hit.reduce((a, v) => a + v, 0) / n;
    out.classCov = cHit.map((h, c) => (cN[c] ? h / cN[c] : NaN));
    out.classN = cN;
    const p = probs(b, T), conf = Array.from({ length: n }, (_, i) => { let m = 0; for (let k = 0; k < K; k++) m = Math.max(m, p[i * K + k]); return [m, i]; });
    conf.sort((a, c) => a[0] - c[0]);
    out.deciles = Array.from({ length: 10 }, (_, dd) => {
      const part = conf.slice(Math.floor((dd * n) / 10), Math.floor(((dd + 1) * n) / 10));
      return { cov: part.reduce((a, [, i]) => a + hit[i], 0) / part.length, conf: part[Math.floor(part.length / 2)][0] };
    });
  }
  out.hit = hit;
  return out;
}
const finite = (a) => a.filter((v) => Number.isFinite(v));
const minF = (a) => Math.min(...finite(a));

// ---------------------------------------------------------------- tiny SVG chart kit
const fmt = (v, d = 1) => (Number.isFinite(v) ? (v * 100).toFixed(d) + "%" : "—");
const f2 = (v, d = 2) => (Number.isFinite(v) ? (v >= 100 ? v.toFixed(0) : v.toFixed(d)) : "—");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function scale(d0, d1, r0, r1) { return (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0); }
function logScale(d0, d1, r0, r1) { const a = Math.log(d0), b = Math.log(d1); return (v) => r0 + ((Math.log(Math.max(v, 1e-9)) - a) / (b - a || 1)) * (r1 - r0); }
function frame(w, h, m, x, y, yTicks, yFmt, xTicks) {
  let s = `<g class="grid">`;
  for (const t of yTicks) s += `<line x1="${m.l}" x2="${w - m.r}" y1="${y(t)}" y2="${y(t)}"/>`;
  s += `</g>`;
  for (const t of yTicks) s += `<text x="${m.l - 6}" y="${y(t) + 4}" text-anchor="end">${yFmt(t)}</text>`;
  s += `<line class="axis" x1="${m.l}" x2="${w - m.r}" y1="${h - m.b}" y2="${h - m.b}"/>`;
  for (const [v, lab] of xTicks || []) s += `<text x="${x(v)}" y="${h - m.b + 15}" text-anchor="middle">${lab}</text>`;
  return s;
}
function svg(w, h, body, label) { return `<svg class="chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}">${body}</svg>`; }
function niceDomain(vals, target, pad = 0.02) {
  const lo = Math.max(0, Math.min(...finite(vals), target) - pad);
  return [Math.floor(lo * 50) / 50, 1];
}
function bars(el, vals, { labels, target, yDom, yFmt = (v) => fmt(v, 0), tip, under, ghost, label, h = 220, w = 400, every = 1 }) {
  const m = { l: 44, r: 8, t: 10, b: 26 }, n = vals.length;
  const y = scale(yDom[0], yDom[1], h - m.b, m.t), bw = (w - m.l - m.r) / n, gap = bw > 8 ? 2 : bw > 3 ? 0.6 : 0;
  const ticks = [0, 1, 2, 3, 4].map((i) => yDom[0] + ((yDom[1] - yDom[0]) * i) / 4);
  let s = frame(w, h, m, (i) => m.l + bw * (i + 0.5), y, ticks, yFmt, labels.map((l, i) => [i, l]).filter((_, i) => i % every === 0));
  vals.forEach((v, i) => {
    if (!Number.isFinite(v)) return;
    const x0 = m.l + bw * i + gap, x1 = x0 + bw - 2 * gap, yv = Math.max(m.t, y(Math.max(v, yDom[0])));
    if (ghost && Number.isFinite(ghost[i])) { const g = Math.max(m.t, y(Math.max(ghost[i], yDom[0]))); s += `<rect class="ghost" x="${x0}" y="${g}" width="${Math.max(0.5, (x1 - x0) / 2 - 0.5)}" height="${h - m.b - g}" rx="1"/>`; }
    const bx = ghost ? x0 + (x1 - x0) / 2 + 0.5 : x0, bwid = Math.max(0.6, ghost ? (x1 - x0) / 2 - 0.5 : x1 - x0), r = Math.min(3, bwid / 2);
    s += `<path class="bar${under && under(v, i) ? " under" : ""}" d="M${bx},${h - m.b}V${yv + r}q0,-${r} ${r},-${r}h${Math.max(0, bwid - 2 * r)}q${r},0 ${r},${r}V${h - m.b}Z"/>`;
    s += `<rect class="hit" x="${m.l + bw * i}" y="${m.t}" width="${bw}" height="${h - m.t - m.b}" data-tip="${esc(tip(v, i))}"/>`;
  });
  if (target != null) s += `<line class="target" x1="${m.l}" x2="${w - m.r}" y1="${y(target)}" y2="${y(target)}"/>`;
  el.innerHTML = svg(w, h, s, label);
}
// Generic multi-series line chart. series: [{pts:[[x,y]], cls:"s1", dash, hollow, label, band:[[x,lo,hi]], tips}]
function lines(el, series, { w = 420, h = 240, xlog = false, xDom, yDom, xFmt = (v) => v, yFmt = (v) => fmt(v, 0), target, xTicks, label, extra = "" }) {
  const m = { l: 46, r: 14, t: 12, b: 30 };
  const xs = series.flatMap((s) => s.pts.map((p) => p[0])), ys = series.flatMap((s) => [...s.pts.map((p) => p[1]), ...(s.band || []).flatMap((b) => [b[1], b[2]])]);
  const xd = xDom || [Math.min(...xs), Math.max(...xs)], yd = yDom || [Math.min(...finite(ys), target ?? 1) - 0.02, Math.min(1, Math.max(...finite(ys), target ?? 0) + 0.02)];
  const X = (xlog ? logScale : scale)(xd[0], xd[1], m.l, w - m.r), Y = scale(yd[0], yd[1], h - m.b, m.t);
  const yt = [0, 1, 2, 3, 4].map((i) => yd[0] + ((yd[1] - yd[0]) * i) / 4);
  const xt = xTicks || (xlog ? [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000].filter((v) => v >= xd[0] && v <= xd[1]) : [0, 1, 2, 3, 4].map((i) => xd[0] + ((xd[1] - xd[0]) * i) / 4));
  let s = frame(w, h, m, X, Y, yt, yFmt, xt.map((v) => [v, xFmt(v)]));
  if (target != null) s += `<line class="target" x1="${m.l}" x2="${w - m.r}" y1="${Y(target)}" y2="${Y(target)}"/>`;
  for (const se of series) {
    if (se.band && se.band.length) s += `<path class="area ${se.cls}" d="M${se.band.map((b) => `${X(b[0]).toFixed(1)},${Y(b[2]).toFixed(1)}`).join("L")}L${[...se.band].reverse().map((b) => `${X(b[0]).toFixed(1)},${Y(b[1]).toFixed(1)}`).join("L")}Z"/>`;
    if (se.err) se.err.forEach(([x, lo, hi]) => { s += `<line class="${se.cls}" style="stroke-width:1.2" x1="${X(x)}" x2="${X(x)}" y1="${Y(lo)}" y2="${Y(hi)}"/>`; });
    if (se.pts.length > 1 && !se.noLine) s += `<path class="ln ${se.cls}${se.dash ? " dash" : ""}" d="M${se.pts.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join("L")}"/>`;
    if (se.dots !== false) se.pts.forEach(([x, y], i) => {
      s += `<circle class="${se.cls}${se.hollow ? " hollow" : ""}" cx="${X(x)}" cy="${Y(y)}" r="${se.r || 3.5}"/>`;
      if (se.tips) s += `<circle class="hit" cx="${X(x)}" cy="${Y(y)}" r="10" data-tip="${esc(se.tips[i])}"/>`;
    });
  }
  s += extra ? extra(X, Y, m) : "";
  el.innerHTML = svg(w, h, s, label);
}

// ---------------------------------------------------------------- tooltip + downloads
const tip = $("tip");
document.addEventListener("pointermove", (e) => {
  const t = e.target.closest && e.target.closest("[data-tip]");
  if (!t) { tip.style.opacity = 0; return; }
  tip.textContent = t.getAttribute("data-tip"); tip.style.opacity = 1;
  const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8), y = e.clientY + 16 + tip.offsetHeight > innerHeight ? e.clientY - tip.offsetHeight - 10 : e.clientY + 16;
  tip.style.left = x + "px"; tip.style.top = y + "px";
});
document.addEventListener("click", (e) => {
  const b = e.target.closest && e.target.closest("button.dl"); if (!b) return;
  const src = $(b.dataset.dl) && $(b.dataset.dl).querySelector("svg"); if (!src) return;
  try {
    const clone = src.cloneNode(true), all = src.querySelectorAll("*"), call = clone.querySelectorAll("*");
    all.forEach((node, i) => { const cs = getComputedStyle(node), keep = ["fill", "stroke", "stroke-width", "stroke-dasharray", "opacity", "font-family", "font-size", "font-weight"];
      call[i].setAttribute("style", keep.map((k) => `${k}:${cs.getPropertyValue(k)}`).join(";")); });
    clone.querySelectorAll(".hit").forEach((n) => n.remove());
    const vb = src.viewBox.baseVal, bg = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    bg.setAttribute("width", vb.width); bg.setAttribute("height", vb.height); bg.setAttribute("fill", getComputedStyle(document.body).backgroundColor);
    clone.insertBefore(bg, clone.firstChild); clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    const url = URL.createObjectURL(new Blob([clone.outerHTML], { type: "image/svg+xml" }));
    const a = document.createElement("a"); a.href = url; a.download = `${b.dataset.dl}-${st.ds}.svg`; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  } catch (err) { console.warn("download failed", err); }
});

// ---------------------------------------------------------------- shared UI bits
function seg(el, options, value, onPick) {
  el.innerHTML = options.map(([v, l]) => `<button type="button" data-v="${v}" aria-pressed="${String(v) === String(value)}">${esc(l)}</button>`).join("");
  el.onclick = (e) => { const b = e.target.closest("button"); if (b) onPick(b.dataset.v); };
}
function names(ds) { return SHORT[ds] || D[ds].classes; }
function chips(C, i, d, trueY, max = 4) {
  const nm = names(d.ds), K = d.K, inSet = [];
  for (let k = 0; k < K; k++) if (C[i * K + k]) inSet.push(k);
  if (!inSet.length) return `<span class="chip empty">∅</span>`;
  inSet.sort((a, b) => (b === trueY) - (a === trueY));
  const shown = inSet.slice(0, max).map((k) => `<span class="chip${k === trueY ? " true" : ""}">${esc(nm[k])}</span>`).join("");
  return shown + (inSet.length > max ? `<span class="chip more">+${inSet.length - max}</span>` : "");
}
function card(img, C, i, d, y, extraTip) {
  const covered = y == null ? null : C[i * d.K + y] === 1;
  const cls = y == null ? "nolabel" : covered ? "" : "miss";
  const t = y == null ? extraTip : `true: ${d.classes[y]} · ${covered ? "covered" : "missed"}`;
  return `<button class="card ${cls}" data-i="${i}" type="button" data-tip="${esc(t)}"><img src="${img}" alt="${esc(t)}"><span class="chips">${chips(C, i, d, y)}</span></button>`;
}
function params(over = {}) { return { ...st, ...over }; }
const runsIn = (rows) => RUN_ORDER.filter((r) => rows && rows.some((x) => x.dataset === r));
const mean = (a) => { const v = finite(a); return v.length ? v.reduce((x, y) => x + y, 0) / v.length : NaN; };

// ---------------------------------------------------------------- hero
function renderHero() {
  const d = D[st.ds]; if (!d) return;
  const P = params({ method: "aps", T: d.tuned.T, ncal: d.cal.n, calSeed: 1 });
  const cal = calibrate(d, P), { C } = predictSets(d.test, P, cal), mt = metrics(d.test, C, P.T);
  $("heroStrip").innerHTML = d.gallery.slice(0, 18).map((g) => card(g.img, C, g.i, d, d.test.labels[g.i])).join("");
  $("heroStats").innerHTML = `APS(T) · coverage <b>${fmt(mt.cov)}</b> · mean size <b>${mt.mean.toFixed(2)}</b> · empty ${fmt(mt.empty)}`;
}
function renderKpis() {
  if (!R) return;
  const g = (ds, m, f) => mean(R.main.filter((r) => r.dataset === ds && r.method === m && Math.abs(r.alpha - 0.1) < 1e-9).map((r) => r[f]));
  const wc = (ds, m, f) => mean((R.worst_class || []).filter((r) => r.dataset === ds && r.method === m && Math.abs(r.alpha - 0.1) < 1e-9).map((r) => r[f]));
  const big = runsIn(R.main).includes("imagenet") ? "imagenet" : runsIn(R.main).slice(-1)[0];
  const tiles = [
    [`Worst class, ${DS_LABEL[big]}`, fmt(g(big, "temp_naive", "min_class_cov"), 0), `under Naive(T), at 90% marginal coverage`],
    [`Predicted vs measured`, `${fmt(wc("imagenet", "mondrian", "pred_mean"), 1)} · ${fmt(wc("imagenet", "mondrian", "measured_min"), 1)}`, `ImageNet Mondrian worst class: exact law vs experiment`],
    [`Mondrian-PAC, ${DS_LABEL[big]}`, f2(g(big, "mondrian_pac", "mean_size"), 0), `labels per set: a per-class guarantee is infeasible with ~20 images per class`],
    [`Clustered-PAC, ${DS_LABEL[big]}`, f2(g(big, "clustered_pac", "mean_size"), 1), `labels per set for a high-probability guarantee on clusters`],
  ];
  $("heroKpis").innerHTML = tiles.map(([k, v, s]) => `<div class="tile"><span class="k">${esc(k)}</span><span class="v">${v}</span><span class="s">${esc(s)}</span></div>`).join("");
}

// ---------------------------------------------------------------- set lab
function syncControls() {
  const d = D[st.ds];
  $("alpha").value = st.alpha; $("alphaV").textContent = st.alpha.toFixed(3);
  $("temp").value = st.T; $("tempV").textContent = st.T.toFixed(2); $("tFit").textContent = d.tuned.T.toFixed(3);
  $("gamma").value = st.gamma; $("gammaV").textContent = st.gamma.toFixed(2);
  const gk = st.method === "dist_naive" ? "dist_naive" : "dist_aps"; $("gFit").textContent = d.tuned.gamma[gk];
  $("lam").value = st.lam; $("kreg").value = st.kreg; $("lamV").textContent = st.method === "saps" ? st.lam.toFixed(2) : `${st.lam.toFixed(2)} · ${st.kreg}`;
  $("kreg").style.display = st.method === "saps" ? "none" : "";
  $("delta").value = st.delta; $("deltaV").textContent = st.delta.toFixed(2);
  $("ncal").max = d.cal.n; $("ncal").value = st.ncal; $("ncalV").textContent = st.ncal;
  const opts = Object.keys(METHODS).filter((m) => !(m.startsWith("dist_") && !d.cal.dist) && !(m === "learned" && !d.cal.learned));
  if ($("method").options.length !== opts.length) $("method").innerHTML = opts.map((k) => `<option value="${k}">${esc(METHODS[k].name)}</option>`).join("");
  $("method").value = st.method; $("methodHint").textContent = METHODS[st.method].hint;
  $("tCtl").classList.toggle("off", st.method === "top1" || st.method === "learned");
  $("gCtl").classList.toggle("off", !st.method.startsWith("dist_"));
  $("rCtl").classList.toggle("off", st.method !== "raps" && st.method !== "saps");
  $("dCtl").classList.toggle("off", !st.method.endsWith("_pac"));
  const n = Math.min(st.ncal, d.cal.n);
  $("band").textContent = GROUPED.has(st.method) ? `per group; ~${Math.round(n / d.K)} images per class` : `[${fmt(1 - st.alpha)}, ${fmt(Math.min(1, 1 - st.alpha + 1 / (n + 1)))}]`;
  seg($("labDs"), D_LIST(), st.ds, pickDs);
  seg($("heroDs"), D_LIST(), st.ds, pickDs);
}
function D_LIST() { return Object.keys(D).map((k) => [k, DS_LABEL[k]]); }
function pickDs(ds, keep = false) {
  st.ds = ds; const d = D[ds];
  if (!keep) Object.assign(st, { T: d.tuned.T, gamma: d.tuned.gamma.dist_aps, lam: st.method === "saps" ? d.tuned.saps_lam : d.tuned.raps.lam, kreg: d.tuned.raps.k_reg, ncal: d.cal.n, sel: null, sev: 0, src: "synthetic" });
  if ((st.method.startsWith("dist_") && !d.cal.dist)) st.method = "aps";
  renderAll();
}
let lab = null;
function renderLab() {
  const d = D[st.ds], P = params(), cal = calibrate(d, P), out = predictSets(d.test, P, cal), mt = metrics(d.test, out.C, P.T);
  lab = { cal, ...out, mt };
  const n = Math.min(st.ncal, d.cal.n), lo = 1 - st.alpha;
  const se = Math.sqrt(lo * (1 - lo) / mt.n) * 2;
  const state = st.method === "top1" ? ["warn", "no guarantee"] : mt.cov >= lo - se ? ["ok", "✓ meets target"] : ["bad", "✕ below target"];
  const qtxt = cal == null ? "—" : cal.qc ? `${cal.qc.filter((q) => !Number.isFinite(q)).length ? cal.qc.filter((q) => !Number.isFinite(q)).length + " classes ∞" : "per " + (cal.G ? "cluster" : "class")}` : Number.isFinite(cal.q) ? cal.q.toFixed(3) : "∞";
  const cc = mt.classCov, worst = minF(cc), wcI = cc.indexOf(worst);
  $("tiles").innerHTML = `
    <div class="tile"><span class="k">Coverage</span><span class="v">${fmt(mt.cov)}</span><span class="pill ${state[0]}">${state[1]}</span></div>
    <div class="tile"><span class="k">Mean set size</span><span class="v">${mt.mean.toFixed(2)}</span><span class="s">singletons ${fmt(mt.single, 0)}</span></div>
    <div class="tile"><span class="k">Empty sets</span><span class="v">${fmt(mt.empty)}</span><span class="s">q̂ = ${qtxt}</span></div>
    <div class="tile"><span class="k">Worst class</span><span class="v">${fmt(worst)}</span><span class="s">class "${esc(d.classes[wcI])}"${d.K > 10 ? ` · ${mt.classN[wcI]} imgs` : ""}</span></div>`;
  const many = d.K > 12;
  const order = many ? cc.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]) : cc.map((v, i) => [v, i]);
  bars($("classChart"), order.map((o) => o[0]), { labels: many ? order.map((_, j) => (j === 0 ? "worst" : j === order.length - 1 ? "best" : "")) : names(st.ds), target: lo,
    yDom: niceDomain(cc, lo), under: (v) => v < lo - 0.02, tip: (v, j) => `${d.classes[order[j][1]]}: ${fmt(v)} covered (${mt.classN[order[j][1]]} test images)`, label: "Coverage per class" });
  $("classNote").textContent = many ? `${d.K} classes, sorted worst → best` : "Marginal ≠ conditional";
  const maxS = Math.min(d.K, Math.max(10, mt.sizes.reduce((a, v) => Math.max(a, v), 0))), cap = Math.min(maxS, 30);
  const hist = mt.hist.slice(0, cap + 1); hist[cap] = mt.hist.slice(cap).reduce((a, v) => a + v, 0);
  bars($("sizeChart"), hist, { labels: hist.map((_, i) => (i === cap && cap < d.K ? `${i}+` : String(i))), every: cap > 12 ? 5 : 1,
    yDom: [0, Math.max(0.2, Math.ceil(Math.max(...hist) * 10) / 10)], tip: (v, i) => `size ${i}${i === cap && cap < d.K ? "+" : ""}: ${fmt(v)} of images`, label: "Set size histogram" });
  const dc = mt.deciles.map((x) => x.cov);
  bars($("decileChart"), dc, { labels: mt.deciles.map((x) => (x.conf >= 0.995 ? ">.99" : x.conf.toFixed(2))), target: lo, yDom: niceDomain(dc, lo, 0.05), h: 170, w: 760,
    under: (v) => v < lo - 0.02, tip: (v, i) => `decile ${i + 1} (median conf ${mt.deciles[i].conf.toFixed(3)}): ${fmt(v)}`, label: "Coverage by confidence decile" });
  renderGallery();
}
function renderGallery() {
  const d = D[st.ds], { C, mt } = lab, y = d.test.labels;
  const keep = { all: () => true, miss: (i) => !mt.hit[i], multi: (i) => mt.sizes[i] >= 2, empty: (i) => mt.sizes[i] === 0 }[st.filter];
  const items = d.gallery.filter((g) => keep(g.i));
  seg($("galFilter"), [["all", "All"], ["miss", "Missed"], ["multi", "Size ≥ 2"], ["empty", "Empty"]], st.filter, (v) => { st.filter = v; renderGallery(); });
  $("gallery").innerHTML = items.length ? items.map((g) => card(g.img, C, g.i, d, y[g.i])).join("") : `<p class="note">No images in this sample match. Try another filter or α.</p>`;
  renderDetail();
}
function renderDetail() {
  const el = $("detail"), d = D[st.ds], K = d.K;
  if (st.sel == null) { el.hidden = true; return; }
  const i = st.sel, g = d.gallery.find((x) => x.i === i); if (!g) { el.hidden = true; return; }
  el.hidden = false;
  const { C, S, cal } = lab, y = d.test.labels[i], p = probs(d.test, st.method === "top1" ? 1 : st.T);
  let rows = Array.from({ length: K }, (_, k) => k);
  if (K > 12) { rows.sort((a, b) => p[i * K + b] - p[i * K + a]); rows = rows.slice(0, 10); if (!rows.includes(y)) rows.push(y); }
  const w = 520, rowH = 22, h = rows.length * rowH + 30, m = { l: 110, r: 70 };
  const val = (k) => (S ? S[i * K + k] : 1 - p[i * K + k]);
  const th = (k) => (cal ? thr(cal, k) : null);
  const fin = [...rows.map(val), ...rows.map(th)].filter((v) => v != null && Number.isFinite(v));
  const x = scale(Math.min(0, ...fin), Math.max(1, ...fin), m.l, w - m.r);
  let s = `<text x="${m.l}" y="12">score s(x, y), smaller = more plausible</text>`;
  rows.forEach((k, r) => {
    const yy = 22 + r * rowH, inSet = C[i * K + k];
    s += `<text x="${m.l - 8}" y="${yy + 14}" text-anchor="end" class="${k === y ? "lbl-ink" : ""}">${esc(d.classes[k].slice(0, 14))}${k === y ? " ◂" : ""}</text>`;
    s += `<rect x="${m.l}" y="${yy + 4}" width="${Math.max(2, x(val(k)) - m.l)}" height="12" rx="2" class="${inSet ? "bar" : "ghost"}" style="${inSet ? "" : "opacity:.5"}"/>`;
    s += `<text x="${w - m.r + 6}" y="${yy + 14}">p=${p[i * K + k].toFixed(3)}</text>`;
    const t = th(k);
    if (t != null && Number.isFinite(t)) { const x1 = x(t); s += cal.qc ? `<line class="target" x1="${x1}" x2="${x1}" y1="${yy + 2}" y2="${yy + 18}"/>` : r === 0 ? `<line class="target" x1="${x1}" x2="${x1}" y1="18" y2="${h - 6}"/><text x="${x1 + 4}" y="${h}" style="fill:var(--target)">q̂</text>` : ""; }
    s += `<rect class="hit" x="0" y="${yy}" width="${w}" height="${rowH}" data-tip="${esc(`${d.classes[k]}: score ${val(k).toFixed(3)} ${inSet ? "≤" : ">"} q̂ → ${inSet ? "in set" : "out"}`)}"/>`;
  });
  el.innerHTML = `<div><img src="${g.img}" alt="selected test image"><p class="note" style="margin-top:6px">True label: <strong>${esc(d.classes[y])}</strong><br>Set size ${lab.mt.sizes[i]}</p></div>` +
    svg(w, h + 6, s, "Scores of each label against the threshold");
}

// ---------------------------------------------------------------- theorem: re-split law
function renderTheorem() {
  const d = D[st.ds], K = d.K, method = GROUPED.has(st.method) || st.method === "top1" ? "aps" : st.method, P = params({ method });
  const Sc = scoreMatrix(d.cal, method, P), St = scoreMatrix(d.test, method, P);
  const pool = new Float64Array(d.cal.n + d.test.n);
  for (let i = 0; i < d.cal.n; i++) pool[i] = Sc[i * K + d.cal.labels[i]];
  for (let i = 0; i < d.test.n; i++) pool[d.cal.n + i] = St[i * K + d.test.labels[i]];
  const N = pool.length, n = Math.min(st.ncal, d.cal.n), m = N - n, reps = 400, r = rng(99 + st.calSeed), a = Float64Array.from(pool), covs = [];
  for (let t = 0; t < reps; t++) {
    for (let i = 0; i < n; i++) { const j = i + Math.floor(r() * (N - i)); const tmp = a[i]; a[i] = a[j]; a[j] = tmp; }
    const q = quantile(a.subarray(0, n), st.alpha); let h = 0;
    for (let i = n; i < N; i++) h += a[i] <= q;
    covs.push(h / m);
  }
  const k = kIndex(n, st.alpha), A = k, B = n + 1 - k;
  const mu = covs.reduce((x, v) => x + v, 0) / reps, sd = Math.sqrt(covs.reduce((x, v) => x + (v - mu) ** 2, 0) / reps);
  const lo = Math.min(...covs), hi = Math.max(...covs), span = Math.max(hi - lo, 0.01);
  const x0 = Math.max(0, lo - span * 0.3), x1 = Math.min(1, hi + span * 0.3), nb = 30, bw = (x1 - x0) / nb;
  const hist = new Array(nb).fill(0); covs.forEach((v) => { hist[Math.min(nb - 1, Math.max(0, Math.floor((v - x0) / bw)))]++; });
  const dens = hist.map((c) => c / reps / bw);
  const lB = lgamma(A) + lgamma(B) - lgamma(A + B), lm = lgamma(m + 1), curve = [];
  if (k <= n) for (let j = Math.max(0, Math.floor(x0 * m)); j <= Math.min(m, Math.ceil(x1 * m)); j += Math.max(1, Math.floor(m / 600))) {
    const lp = lm - lgamma(j + 1) - lgamma(m - j + 1) + lgamma(j + A) + lgamma(m - j + B) - lgamma(m + A + B) - lB;
    curve.push([j / m, Math.exp(lp) * m]);
  }
  const ymax = Math.max(...dens, ...curve.map((c) => c[1]), 1) * 1.08;
  const w = 760, h = 240, mg = { l: 44, r: 10, t: 10, b: 28 }, X = scale(x0, x1, mg.l, w - mg.r), Y = scale(0, ymax, h - mg.b, mg.t);
  const xt = [0, 1, 2, 3, 4].map((i) => x0 + ((x1 - x0) * i) / 4);
  let s = `<rect class="band" x="${X(1 - st.alpha)}" y="${mg.t}" width="${Math.max(2, X(Math.min(1, 1 - st.alpha + 1 / (n + 1))) - X(1 - st.alpha))}" height="${h - mg.t - mg.b}"/>`;
  s += frame(w, h, mg, X, Y, [0, ymax / 2, ymax].map((v) => Math.round(v)), (v) => v, xt.map((v) => [v, fmt(v, 1)]));
  dens.forEach((v, i) => { const xa = X(x0 + i * bw) + 1, xb = X(x0 + (i + 1) * bw) - 1; s += `<rect class="bar" style="opacity:.55" x="${xa}" y="${Y(v)}" width="${Math.max(1, xb - xa)}" height="${h - mg.b - Y(v)}" rx="1"/><rect class="hit" x="${xa}" y="${mg.t}" width="${Math.max(1, xb - xa)}" height="${h - mg.t - mg.b}" data-tip="${fmt(x0 + i * bw)}–${fmt(x0 + (i + 1) * bw)}: ${hist[i]} of ${reps} splits"/>`; });
  if (curve.length) s += `<path class="curve" d="M${curve.map(([cx, cy]) => `${X(cx).toFixed(1)},${Y(cy).toFixed(1)}`).join("L")}"/>`;
  $("rsChart").innerHTML = svg(w, h, s, "Coverage across re-splits with the exact law");
  $("rsSummary").innerHTML = `${DS_LABEL[st.ds]} · n = ${n}, ${reps} splits · mean ${fmt(mu, 2)} (theory ${fmt(A / (A + B), 2)}) · sd ${fmt(sd, 2)} · below 1−α in ${fmt(covs.filter((v) => v < 1 - st.alpha).length / reps, 0)}`;
  $("nCalFull").textContent = d.cal.n;
  const nf = d.cal.n, kf = kIndex(nf, st.alpha), af = kf, bf = nf + 1 - kf;
  $("rsSd").textContent = fmt(Math.sqrt((af * bf) / ((af + bf) ** 2 * (af + bf + 1))), 2);
}

// ---------------------------------------------------------------- results (precomputed, all seeds)
const res = { ds: "cifar10", alpha: 0.1, ccDs: "cifar10", ccA: "temp_naive", ccB: "mondrian" };
function agg(rows, key, fields) {
  const g = {};
  for (const r of rows) (g[r[key]] = g[r[key]] || []).push(r);
  return Object.fromEntries(Object.entries(g).map(([k, rs]) => [k, Object.fromEntries(fields.map((f) => {
    const v = rs.map((r) => r[f]).filter((x) => x != null && Number.isFinite(x)), m = v.reduce((a, b) => a + b, 0) / (v.length || 1);
    return [f, { m, s: v.length > 1 ? Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1)) : 0, n: v.length }];
  }))]));
}
function renderResults() {
  if (!R || !R.main.length) { $("resTable").innerHTML = `<p class="note">Run scripts/run_all.py to fill this section.</p>`; return; }
  const dss = runsIn(R.main);
  if (!dss.includes(res.ds)) res.ds = dss[0];
  seg($("resDs"), dss.map((k) => [k, DS_LABEL[k]]), res.ds, (v) => { res.ds = v; renderResults(); });
  seg($("resAlpha"), R.config.alphas.map((a) => [a, "α " + a]), res.alpha, (v) => { res.alpha = +v; renderResults(); });
  const rows = R.main.filter((r) => r.dataset === res.ds && Math.abs(r.alpha - res.alpha) < 1e-9);
  const F = ["coverage", "mean_size", "p90_size", "singleton_rate", "empty_rate", "min_class_cov", "sscv", "fsc_min"];
  const A = agg(rows, "method", F), order = R.config.scores.filter((m) => A[m]);
  const valid = order.filter((m) => m !== "top1");
  const best = { mean_size: Math.min(...valid.map((m) => A[m].mean_size.m)), min_class_cov: Math.max(...valid.map((m) => A[m].min_class_cov.m)),
    sscv: Math.min(...valid.map((m) => A[m].sscv.m).filter(Number.isFinite)), fsc_min: Math.max(...valid.map((m) => A[m].fsc_min.m).filter(Number.isFinite)) };
  const nseed = A[order[0]].coverage.n;
  const pm = (o, pct, d = 1) => (!Number.isFinite(o.m) || !o.n ? "—" : (pct ? fmt(o.m, d) : f2(o.m)) + (o.n > 1 ? `<span style="color:var(--muted)"> ±${pct ? (o.s * 100).toFixed(1) : f2(o.s)}</span>` : ""));
  const cell = (m, f, pct) => `<td class="${best[f] != null && m !== "top1" && Math.abs(A[m][f].m - best[f]) < 1e-9 ? "best" : ""}">${pm(A[m][f], pct)}</td>`;
  $("resTable").innerHTML = `<table><thead><tr><th>Method</th><th>Coverage</th><th>Mean size</th><th>P90 size</th><th>Singleton</th><th>Empty</th><th>Worst class</th><th>SSCV ↓</th><th>Worst feature cluster</th></tr></thead><tbody>${
    order.map((m) => `<tr class="${m === st.method || (m === "temp_aps" && st.method === "aps") ? "sel" : ""}"><td style="font-family:var(--body)">${PRETTY[m]}</td>${cell(m, "coverage", 1)}${cell(m, "mean_size", 0)}${cell(m, "p90_size", 0)}${cell(m, "singleton_rate", 1)}${cell(m, "empty_rate", 1)}${cell(m, "min_class_cov", 1)}${cell(m, "sscv", 1)}${cell(m, "fsc_min", 1)}</tr>`).join("")
  }</tbody></table><p class="note" style="padding:8px 10px 0">Target coverage ${fmt(1 - res.alpha, 0)}. ${nseed} seed${nseed > 1 ? "s" : ""}${res.ds === "imagenet" ? " (random re-splits of the 50k images)" : ""}; ± is the std across seeds. Green marks the best guaranteed method per column. SSCV: worst gap to target over set-size strata. Worst feature cluster: coverage of the worst of 10 k-means clusters of test features.</p>`;
  // frontier
  const fo = order.filter((m) => m !== "top1"), xs = fo.map((m) => A[m].mean_size.m), ys = fo.map((m) => A[m].min_class_cov.m);
  const big = Math.max(...xs) / Math.max(Math.min(...xs), 0.1) > 4;
  const series = fo.map((m) => ({ pts: [[A[m].mean_size.m, A[m].min_class_cov.m]], cls: m === st.method || (m === "temp_aps" && st.method === "aps") ? "s1" : "sg",
    hollow: m.endsWith("_pac"), r: 5, err: [[A[m].mean_size.m, A[m].min_class_cov.m - A[m].min_class_cov.s, A[m].min_class_cov.m + A[m].min_class_cov.s]],
    tips: [`${PRETTY[m]}: size ${f2(A[m].mean_size.m, 3)}, worst class ${fmt(A[m].min_class_cov.m)}`] }));
  const xlo = Math.min(...xs), xhi = Math.max(...xs);
  lines($("frontier"), series, { w: 420, h: 300, xlog: big, xDom: big ? [xlo * 0.8, xhi * 1.25] : [xlo - (xhi - xlo) * 0.1 - 0.005, xhi + (xhi - xlo) * 0.1 + 0.005],
    yDom: [Math.max(0, Math.min(...ys, 1 - res.alpha) - 0.03), Math.min(1.005, Math.max(...ys, 1 - res.alpha) + 0.02)], target: 1 - res.alpha,
    xFmt: (v) => f2(v), label: "Mean set size against worst-class coverage",
    extra: (X, Y) => fo.filter((m) => ["temp_naive", "temp_aps", "mondrian", "mondrian_pac", "clustered_pac"].includes(m)).map((m) =>
      `<text x="${X(A[m].mean_size.m) + 7}" y="${Y(A[m].min_class_cov.m) - 6}" style="font-size:10px">${PRETTY[m]}</text>`).join("") });
  renderForest(); renderHeat(); renderFindings(A);
}
function renderForest() {
  const rows = (R.bootstrap || []).filter((r) => r.dataset === res.ds && r.metric === "min_class_cov");
  if (!rows.length) { $("forest").innerHTML = `<p class="note">Run scripts/research.py to fill this.</p>`; return; }
  const w = 420, rowH = 26, m = { l: 150, r: 14, t: 8, b: 26 }, h = rows.length * rowH + m.t + m.b;
  const lo = Math.min(...rows.map((r) => r.ci_lo), 0), hi = Math.max(...rows.map((r) => r.ci_hi), 0), pad = (hi - lo) * 0.08 || 0.01;
  const X = scale(lo - pad, hi + pad, m.l, w - m.r);
  let s = `<line class="axis" x1="${X(0)}" x2="${X(0)}" y1="${m.t}" y2="${h - m.b}" style="stroke:var(--ink-2)"/>`;
  [lo, 0, hi].forEach((v) => { s += `<text x="${X(v)}" y="${h - 8}" text-anchor="middle">${v > 0 ? "+" : ""}${(v * 100).toFixed(1)}</text>`; });
  rows.forEach((r, i) => {
    const y = m.t + i * rowH + rowH / 2, sig = r.ci_lo > 0 || r.ci_hi < 0;
    s += `<text x="${m.l - 8}" y="${y + 4}" text-anchor="end" class="${sig ? "lbl-ink" : ""}">${PRETTY[r.a]} − ${PRETTY[r.b]}</text>`;
    s += `<line class="${sig ? "s1" : "sg"}" style="stroke-width:2" x1="${X(r.ci_lo)}" x2="${X(r.ci_hi)}" y1="${y}" y2="${y}"/><circle class="${sig ? "s1" : "sg"}" cx="${X(r.estimate)}" cy="${y}" r="4"/>`;
    s += `<rect class="hit" x="0" y="${y - rowH / 2}" width="${w}" height="${rowH}" data-tip="${esc(`${PRETTY[r.a]} − ${PRETTY[r.b]}: ${(r.estimate * 100).toFixed(2)} pts [${(r.ci_lo * 100).toFixed(2)}, ${(r.ci_hi * 100).toFixed(2)}], ${r.n_seeds} seeds`)}"/>`;
  });
  $("forest").innerHTML = svg(w, h, s, "Bootstrap intervals for worst-class differences") + `<p class="note" style="margin:0">Points: difference in worst-class coverage (percentage points).</p>`;
}
function renderHeat() {
  const rows = (R.sweep || []).filter((r) => r.dataset === res.ds && r.method === "dist_aps");
  if (!rows.length) { $("heat").innerHTML = `<p class="note">${res.ds === "imagenet" ? "No features for ImageNet, so no distance term." : "Run scripts/sweep.py to fill this."}</p>`; return; }
  const gs = [...new Set(rows.map((r) => r.gamma))].sort((a, b) => a - b), as = R.config.alphas;
  const cellv = (a, g, f) => { const v = rows.filter((r) => Math.abs(r.alpha - a) < 1e-9 && r.gamma === g).map((r) => r[f]); return v.reduce((x, y) => x + y, 0) / v.length; };
  const tbl = (f, title, pct) => {
    const all = as.flatMap((a) => gs.map((g) => cellv(a, g, f))), lo = Math.min(...all), hi = Math.max(...all);
    return `<div><div class="eyebrow" style="margin:6px 0">${title}</div><div class="tbl"><table class="heat"><thead><tr><th style="text-transform:none">α \\ γ</th>${gs.map((g) => `<th style="text-align:center">${g}</th>`).join("")}</tr></thead><tbody>${
      as.map((a) => `<tr><td>${a}</td>${gs.map((g) => { const v = cellv(a, g, f), t = (v - lo) / (hi - lo || 1);
        return `<td style="background:color-mix(in srgb, var(--accent) ${Math.round(8 + t * 55)}%, transparent)" data-tip="α=${a}, γ=${g}: ${pct ? fmt(v, 2) : v.toFixed(3)}">${pct ? fmt(v, 1) : v.toFixed(2)}</td>`; }).join("")}</tr>`).join("")
    }</tbody></table></div></div>`;
  };
  $("heat").innerHTML = `<div class="two">${tbl("mean_size", "Mean set size", false)}${tbl("min_class_cov", "Worst-class coverage", true)}</div>`;
}
function renderFindings(A) {
  const g = (m, f) => (A[m] ? A[m][f].m : NaN);
  const acc = mean((R.tuned || []).filter((t) => t.dataset === res.ds).map((t) => t.accuracy));
  const items = [`Every conformal method lands within sampling error of ${fmt(1 - res.alpha, 0)} coverage; Top-1 gives ${fmt(g("top1", "coverage"))} with no control over it.`];
  items.push(acc > 1 - res.alpha
    ? `Accuracy (${fmt(acc)}) beats the target, so sets reach ${fmt(1 - res.alpha, 0)} by <strong>returning empty sets</strong> (${fmt(g("temp_naive", "empty_rate"))} for Naive(T)); section 13 proves why.`
    : `Accuracy (${fmt(acc)}) is below the target, so sets must grow beyond one label (mean ${f2(g("temp_aps", "mean_size"))} for APS(T)).`);
  items.push(`Naive(T) gives the smallest sets (${f2(g("temp_naive", "mean_size"))}) but its worst class reaches only ${fmt(g("temp_naive", "min_class_cov"))}; APS(T) reaches ${fmt(g("temp_aps", "min_class_cov"))} at ${f2(g("temp_aps", "mean_size"))}.`);
  items.push(`Mondrian guarantees each class only on average: its realised worst class is ${fmt(g("mondrian", "min_class_cov"))}. Mondrian-PAC lifts it to ${fmt(g("mondrian_pac", "min_class_cov"))} at size ${f2(g("mondrian_pac", "mean_size"))}; Clustered-PAC reaches ${fmt(g("clustered_pac", "min_class_cov"))} at ${f2(g("clustered_pac", "mean_size"))}.`);
  if (A.dist_naive) items.push(`The distance term is a diagnosed weak result: Naive(T) → +dist moves the worst class ${fmt(g("temp_naive", "min_class_cov"))} → ${fmt(g("dist_naive", "min_class_cov"))} at size ${f2(g("temp_naive", "mean_size"))} → ${f2(g("dist_naive", "mean_size"))} (section 13 explains why).`);
  if (A.conftr) items.push(`The learned score gives size ${f2(g("conftr", "mean_size"))} vs ${f2(g("temp_naive", "mean_size"))} for Naive(T): training for size cannot beat the size-optimal score by much, and it does not fix the worst class (${fmt(g("conftr", "min_class_cov"))}).`);
  $("findings").innerHTML = `<h3>Reading the table · ${DS_LABEL[res.ds]}, α = ${res.alpha}</h3><ul class="clean">${items.map((t) => `<li>${t}</li>`).join("")}</ul>`;
}
function renderClassAudit() {
  if (!R || !R.class_cov.length) return;
  const dss = runsIn(R.class_cov); if (!dss.includes(res.ccDs)) res.ccDs = dss[0];
  seg($("ccDs"), dss.map((k) => [k, DS_LABEL[k]]), res.ccDs, (v) => { res.ccDs = v; renderClassAudit(); });
  const ms = R.config.scores.filter((m) => R.class_cov.some((r) => r.dataset === res.ccDs && r.method === m));
  for (const [id, key] of [["ccA", "ccA"], ["ccB", "ccB"]]) {
    if (!ms.includes(res[key])) res[key] = key === "ccA" ? "temp_naive" : "mondrian";
    $(id).innerHTML = ms.map((m) => `<option value="${m}">${PRETTY[m]}</option>`).join(""); $(id).value = res[key];
    $(id).onchange = (e) => { res[key] = e.target.value; renderClassAudit(); };
  }
  const rows = R.class_cov.filter((r) => r.dataset === res.ccDs), K = Math.max(...rows.map((r) => r.cls)) + 1;
  const per = (m) => { const a = new Array(K).fill(NaN); rows.filter((r) => r.method === m).forEach((r) => { a[r.cls] = r.coverage; }); return a; };
  const a = per(res.ccA), b = per(res.ccB), t = 1 - R.config.main_alpha;
  if (K <= 12) {
    const cls = res.ccDs === "mnist" ? [...Array(10).keys()].map(String) : SHORT.cifar10;
    bars($("ccChart"), b, { labels: cls, ghost: a, target: t, yDom: niceDomain([...a, ...b], t, 0.03), under: (v) => v < t - 0.02,
      tip: (v, i) => `${cls[i]}: ${PRETTY[res.ccA]} ${fmt(a[i])} · ${PRETTY[res.ccB]} ${fmt(v)}`, label: "Per-class coverage, two methods", h: 220, w: 760 });
    $("ccTitle").textContent = "Per-class coverage";
    $("ccLegend").innerHTML = `<span><i style="height:10px;background:var(--muted);opacity:.35"></i>${PRETTY[res.ccA]} (left bar)</span><span><i style="height:10px"></i>${PRETTY[res.ccB]} (right bar)</span><span><i class="t"></i>target ${fmt(t, 0)}</span><span><i class="r"></i>more than 2 pts short</span>`;
  } else {
    const sa = [...finite(a)].sort((x, y) => x - y), sb = [...finite(b)].sort((x, y) => x - y);
    const toPts = (arr) => arr.map((v, i) => [(i + 0.5) / arr.length, v]);
    lines($("ccChart"), [{ pts: toPts(sa), cls: "sg", dots: false }, { pts: toPts(sb), cls: "s1", dots: false }], { w: 760, h: 240, xDom: [0, 1], yDom: [Math.max(0, Math.min(sa[0], sb[0]) - 0.03), 1.002], target: t,
      xFmt: (v) => fmt(v, 0), label: "Sorted per-class coverage",
      extra: (X, Y) => `<text x="${X(0.01)}" y="${Y(sa[0]) + 14}">${PRETTY[res.ccA]} worst ${fmt(sa[0])}</text><text x="${X(0.01)}" y="${Y(sb[0]) - 6}" class="lbl-ink">${PRETTY[res.ccB]} worst ${fmt(sb[0])}</text>` });
    const below = (arr) => arr.filter((v) => v < t - 0.05).length;
    $("ccTitle").textContent = `${K} classes, sorted from worst to best (mean over seeds)`;
    $("ccLegend").innerHTML = `<span><span class="sw sg"></span>${PRETTY[res.ccA]}: ${below(sa)} classes more than 5 pts short</span><span><span class="sw s1"></span>${PRETTY[res.ccB]}: ${below(sb)} classes more than 5 pts short</span><span><i class="t"></i>target ${fmt(t, 0)}</span>`;
  }
}

// ---------------------------------------------------------------- worst-class law explorer
const WK = [2, 5, 10, 20, 50, 100, 200, 500, 1000];
const logSlider = (v) => Math.round(5 * Math.pow(10, (v / 100) * 3));          // 5 .. 5000
const invLog = (n) => Math.round((Math.log10(n / 5) / 3) * 100);
const wst = { K: 10, n: 500, m: 1000, alpha: 0.1, delta: 0.1, nsDs: "cifar100" };
function worstLaw(K, n, m, alpha, k) { // returns {grid:[t], dens:[...], mean, q10, q90, pBelow}
  if (k == null || k > n) return null;
  const a = k, b = n + 1 - k;
  if (m) {
    const S = bbSurvival(m, a, b), pts = [];
    let mu = 0;
    for (let j = 0; j <= m; j++) { const pr = Math.pow(S[j], K) - Math.pow(S[j + 1], K); mu += pr * (j / m); pts.push([j / m, pr * m]); }
    const cdf = (t) => 1 - Math.pow(S[Math.ceil(t * m - 1e-9)] ?? 0, K);
    const quant = (p) => { for (let j = 0; j <= m; j++) if (1 - Math.pow(S[j + 1], K) >= p) return j / m; return 1; };
    return { pts, mean: mu, q10: quant(0.1), q90: quant(0.9), pBelow: cdf(1 - alpha) };
  }
  const G = 400, t0 = Math.max(0, 1 - alpha - 8 * Math.sqrt(alpha / Math.max(n, 1)) - 0.05), pts = [];
  let prev = 1 - Math.pow(betaSf(t0, a, b), K), mu = t0, q10 = null, q90 = null;
  for (let i = 1; i <= G; i++) {
    const t = t0 + ((1 - t0) * i) / G, c = 1 - Math.pow(betaSf(t, a, b), K);
    pts.push([t, (c - prev) / ((1 - t0) / G)]); mu += 1 - c > 0 ? ((1 - t0) / G) * (1 - c) : 0;
    if (q10 == null && c >= 0.1) q10 = t; if (q90 == null && c >= 0.9) q90 = t; prev = c;
  }
  return { pts, mean: mu, q10, q90, pBelow: 1 - Math.pow(betaSf(1 - alpha, a, b), K) };
}
function nNeeded(K, alpha, eps, delta) {
  for (let n = Math.max(1, Math.ceil(1 / alpha) - 1); n <= 200000; n = n < 200 ? n + 1 : Math.ceil(n * 1.02)) {
    const k = kIndex(n, alpha); if (k <= n && Math.pow(betaSf(1 - alpha - eps, k, n + 1 - k), K) >= 1 - delta) return n;
  }
  return null;
}
function renderWorst() {
  $("wK").value = WK.indexOf(wst.K); $("wKV").textContent = wst.K; $("wN").value = invLog(wst.n); $("wNV").textContent = wst.n;
  $("wM").value = invLog(wst.m); $("wMV").textContent = wst.m; $("wA").value = wst.alpha; $("wAV").textContent = wst.alpha.toFixed(2);
  $("wD").value = wst.delta; $("wDV").textContent = wst.delta.toFixed(2);
  const k = kIndex(wst.n, wst.alpha), dg = 1 - Math.pow(1 - wst.delta, 1 / wst.K), kp = pacK(wst.n, wst.alpha, dg);
  const obs = worstLaw(wst.K, wst.n, wst.m, wst.alpha, k), tru = worstLaw(wst.K, wst.n, 0, wst.alpha, k), pac = kp ? worstLaw(wst.K, wst.n, wst.m, wst.alpha, kp) : null;
  const need = nNeeded(wst.K, wst.alpha, 0.02, wst.delta), minPac = Math.ceil(Math.log(dg) / Math.log(1 - wst.alpha));
  $("wTiles").innerHTML = `
    <div class="tile"><span class="k">Mondrian worst class (mean)</span><span class="v">${obs ? fmt(obs.mean) : "100%"}</span><span class="s">10–90%: ${obs ? `${fmt(obs.q10)} – ${fmt(obs.q90)}` : "full sets"}</span></div>
    <div class="tile"><span class="k">P(some class truly &lt; 1−α)</span><span class="v">${tru ? fmt(tru.pBelow, 0) : "0%"}</span><span class="s">with one q̂ per class</span></div>
    <div class="tile"><span class="k">Mondrian-PAC</span><span class="v">${kp ? `k=${kp}` : "infeasible"}</span><span class="s">${kp ? `vs k=${k}; worst mean ${fmt(pac.mean)}` : `needs ≥ ${minPac} images/class; else full sets`}</span></div>
    <div class="tile"><span class="k">Images/class for ±2 pts</span><span class="v">${need ? need.toLocaleString() : "—"}</span><span class="s">all K within 2 pts w.p. ${fmt(1 - wst.delta, 0)}</span></div>`;
  const series = [];
  if (tru) series.push({ pts: tru.pts, cls: "s4", dots: false, dash: true });
  if (obs) series.push({ pts: obs.pts, cls: "s4", dots: false });
  if (pac) series.push({ pts: pac.pts, cls: "s1", dots: false });
  if (!series.length) { $("wChart").innerHTML = `<p class="note">k &gt; n: too few images per class, every set is the full label set.</p>`; }
  else {
    const all = series.flatMap((s) => s.pts), xs = all.filter((p) => p[1] > 1e-3 * Math.max(...all.map((q) => q[1]))).map((p) => p[0]);
    const xlo = Math.max(0, Math.min(...xs, 1 - wst.alpha) - 0.01), ymax = Math.max(...all.filter((p) => p[0] >= xlo).map((p) => p[1]));
    const clip = (s) => ({ ...s, pts: s.pts.filter((p) => p[0] >= xlo) });
    lines($("wChart"), series.map(clip), { w: 760, h: 240, xDom: [xlo, 1], yDom: [0, ymax * 1.08], yFmt: (v) => v.toFixed(0), xFmt: (v) => fmt(v, 0), label: "Worst-class coverage distribution",
      extra: (X, Y, m) => `<line class="target" x1="${X(1 - wst.alpha)}" x2="${X(1 - wst.alpha)}" y1="${m.t}" y2="${240 - m.b}"/>` +
        (obs ? `<line class="s4" style="stroke-width:1.5" x1="${X(obs.mean)}" x2="${X(obs.mean)}" y1="${m.t}" y2="${240 - m.b}"/><text x="${X(obs.mean) - 4}" y="${m.t + 10}" text-anchor="end" class="lbl-ink">mean ${fmt(obs.mean)}</text>` : "") });
  }
  // prediction vs measurement table
  const W = (R && R.worst_class || []).filter((r) => Math.abs(r.alpha - 0.1) < 1e-9);
  const tr = runsIn(W).map((ds) => {
    const a = W.filter((r) => r.dataset === ds && r.method === "mondrian"), p = W.filter((r) => r.dataset === ds && r.method === "mondrian_pac");
    const pit = mean(a.map((r) => r.pred_cdf_at_measured));
    return `<tr data-tip="${esc(`${DS_LABEL[ds]}: ${a.length} seeds · mean P(worst ≤ measured) under the law = ${Number.isFinite(pit) ? pit.toFixed(2) : "—"} (≈0.5 if calibration and test are exchangeable)`)}"><td style="font-family:var(--body)">${DS_LABEL[ds]}</td><td>${a[0].K}</td><td>${Math.round(mean(a.map((r) => r.n_cal_per_class)))}</td><td>${fmt(mean(a.map((r) => r.pred_mean)))}</td><td><strong style="${pit < 0.2 ? "color:var(--bad)" : ""}">${fmt(mean(a.map((r) => r.measured_min)))}</strong></td><td>${p.length && mean(p.map((r) => r.pred_mean)) < 0.999 ? `<strong>${fmt(mean(p.map((r) => r.measured_min)))}</strong>` : "full sets"}</td></tr>`;
  }).join("");
  $("wTable").innerHTML = tr ? `<table style="font-size:12px"><thead><tr><th>Run</th><th>K</th><th>n/cls</th><th>Pred.</th><th>Meas.</th><th>PAC</th></tr></thead><tbody>${tr}</tbody></table>` : `<p class="note">Run scripts/research.py.</p>`;
  seg($("wPreset"), runsIn(W).map((ds) => [ds, DS_LABEL[ds]]), "", (ds) => {
    const a = W.find((r) => r.dataset === ds && r.method === "mondrian"); Object.assign(wst, { K: WK.reduce((b, v) => (Math.abs(Math.log(v / a.K)) < Math.abs(Math.log(b / a.K)) ? v : b), 2), n: Math.round(a.n_cal_per_class), m: Math.round(a.n_test_per_class), alpha: 0.1 }); renderWorst(); });
  // re-split sweep
  const NS = (R && R.ncal_sweep) || [], nds = runsIn(NS); if (!nds.includes(wst.nsDs)) wst.nsDs = nds[0];
  seg($("nsDs"), nds.map((ds) => [ds, DS_LABEL[ds]]), wst.nsDs, (v) => { wst.nsDs = v; renderWorst(); });
  const rows = NS.filter((r) => r.dataset === wst.nsDs).sort((a, b) => a.n_c - b.n_c);
  if (rows.length) lines($("nsChart"), [
    { pts: rows.map((r) => [r.n_c, r.pred_mean]), band: rows.map((r) => [r.n_c, r.pred_q10, r.pred_q90]), cls: "s2", dots: false },
    { pts: rows.map((r) => [r.n_c, r.measured_mean]), err: rows.map((r) => [r.n_c, r.measured_q10, r.measured_q90]), cls: "s1", noLine: true, tips: rows.map((r) => `${r.n_c} per class: measured ${fmt(r.measured_mean)}, theory ${fmt(r.pred_mean)}`) },
  ], { w: 420, h: 240, xlog: true, xDom: [rows[0].n_c * 0.8, rows[rows.length - 1].n_c * 1.25], target: 0.9, label: "Worst class vs calibration size per class" });
}

// ---------------------------------------------------------------- PAC + clusters
const cst = { ds: "imagenet", Mi: 3, inM: "temp_aps" };
function renderClusters() {
  const CL = (R && R.clusters) || [], dss = runsIn(CL); if (!dss.length) return;
  if (!dss.includes(cst.ds)) cst.ds = dss[dss.length - 1];
  seg($("clDs"), dss.map((k) => [k, DS_LABEL[k]]), cst.ds, (v) => { cst.ds = v; cst.Mi = 2; renderClusters(); });
  const rows = CL.filter((r) => r.dataset === cst.ds), Ms = [...new Set(rows.map((r) => r.M))].sort((a, b) => a - b);
  cst.Mi = Math.min(cst.Mi, Ms.length - 1); $("clM").max = Ms.length - 1; $("clM").value = cst.Mi;
  const M = Ms[cst.Mi]; $("clMV").textContent = M + (M === 1 ? " (marginal)" : M === Ms[Ms.length - 1] ? " (Mondrian)" : "");
  const get = (meth) => Ms.map((m) => rows.find((r) => r.M === m && r.method === meth)).filter(Boolean);
  const a = get("clustered"), b = get("clustered_pac"), xs = [...a, ...b].map((r) => r.mean_size), ys = [...a, ...b].map((r) => r.min_class_cov);
  const big = Math.max(...xs) / Math.min(...xs) > 3;
  lines($("clChart"), [
    { pts: a.map((r) => [r.mean_size, r.min_class_cov]), cls: "s5", tips: a.map((r) => `Clustered, M=${r.M}: size ${f2(r.mean_size)}, worst ${fmt(r.min_class_cov)}`) },
    { pts: b.map((r) => [r.mean_size, r.min_class_cov]), cls: "s5", dash: true, hollow: true, tips: b.map((r) => `Clustered-PAC, M=${r.M}: size ${f2(r.mean_size)}, worst ${fmt(r.min_class_cov)}`) },
  ], { w: 420, h: 270, xlog: big, xDom: big ? [Math.min(...xs) * 0.8, Math.max(...xs) * 1.25] : [Math.min(...xs) * 0.98, Math.max(...xs) * 1.02], yDom: [Math.max(0, Math.min(...ys, 0.9) - 0.03), Math.min(1.005, Math.max(...ys, 0.9) + 0.02)],
    target: 0.9, xFmt: (v) => (big ? f2(v, 1) : v.toFixed(3)), label: "Cluster count trade-off",
    extra: (X, Y) => [a.find((r) => r.M === M), b.find((r) => r.M === M)].filter(Boolean).map((r) => `<circle class="s1" cx="${X(r.mean_size)}" cy="${Y(r.min_class_cov)}" r="7" style="fill:none;stroke-width:2.5"/>`).join("") });
  const ra = a.find((r) => r.M === M), rb = b.find((r) => r.M === M);
  const mpr = (R.main || []).filter((r) => r.dataset === cst.ds && r.method === "mondrian_pac" && Math.abs(r.alpha - 0.1) < 1e-9); // the real Mondrian-PAC run
  const mp = { min_class_cov: mean(mpr.map((r) => r.min_class_cov)), mean_size: mean(mpr.map((r) => r.mean_size)) };
  $("clNow").textContent = `${DS_LABEL[cst.ds]} · K = ${rows[0].K} · M = ${M}`;
  $("clTiles").innerHTML = [[`Clustered`, ra], [`Clustered-PAC`, rb]].map(([k, r]) => `<div class="tile"><span class="k">${k}</span><span class="v">${fmt(r.min_class_cov)}</span><span class="s">worst class · size ${f2(r.mean_size)}</span></div>`).join("");
  const bestPac = b.filter((r) => r.min_class_cov >= mp.min_class_cov - 0.01).sort((x, y) => x.mean_size - y.mean_size)[0];
  const K = rows[0].K, mond = (R.main || []).filter((r) => r.dataset === cst.ds && r.method === "mondrian" && Math.abs(r.alpha - 0.1) < 1e-9);
  const mo = { w: mean(mond.map((r) => r.min_class_cov)), s: mean(mond.map((r) => r.mean_size)) }, top = [...b].sort((x, y) => y.min_class_cov - x.min_class_cov).find((r) => r.mean_size < 0.1 * K) || b[0];
  $("clCall").innerHTML = !Number.isFinite(mp.mean_size) ? `<p>No Mondrian-PAC result for this run.</p>`
    : mp.mean_size >= 0.99 * K ? `<p>Mondrian-PAC is <strong>infeasible</strong> here: with this few calibration images per class it must include every label (size ${f2(mp.mean_size, 0)}). Clustered-PAC with M = ${top.M} keeps a high-probability guarantee per cluster and reaches a worst class of <strong>${fmt(top.min_class_cov)}</strong> at size <strong>${f2(top.mean_size, 1)}</strong>${Number.isFinite(mo.w) ? `, versus Mondrian's ${fmt(mo.w)} at ${f2(mo.s, 1)}` : ""}.</p>`
    : bestPac ? `<p>Mondrian-PAC reaches a worst class of <strong>${fmt(mp.min_class_cov)}</strong> at size <strong>${f2(mp.mean_size)}</strong>. Clustered-PAC with M = ${bestPac.M} gets within 1 point (${fmt(bestPac.min_class_cov)}) at size <strong>${f2(bestPac.mean_size)}</strong>, ${mp.mean_size > 1.1 * bestPac.mean_size ? `${(mp.mean_size / bestPac.mean_size).toFixed(1)}× smaller` : "about the same size"}.</p>`
    : `<p>Mondrian-PAC reaches a worst class of <strong>${fmt(mp.min_class_cov)}</strong> at size <strong>${f2(mp.mean_size)}</strong>; no cluster count matches it within a point.</p>`;
  renderImagenet();
}
function renderImagenet() {
  if (!IN) { $("inGallery").innerHTML = `<p class="note">ImageNet examples not exported.</p>`; return; }
  const lab_ = { temp_naive: "Naive(T)", temp_aps: "APS(T)", mondrian: "Mondrian", clustered: "Clustered", clustered_pac: "Clustered-PAC" };
  seg($("inMethod"), IN.methods.map((m) => [m, lab_[m] || m]), cst.inM, (v) => { cst.inM = v; renderImagenet(); });
  const cov = IN.items.filter((it) => it.sets[cst.inM].covered).length, sz = mean(IN.items.map((it) => it.sets[cst.inM].size));
  $("inGallery").innerHTML = IN.items.map((it) => { const s = it.sets[cst.inM];
    return `<div class="ic${s.covered ? "" : " miss"}" data-tip="${esc(`true: ${it.true} · ${s.covered ? "covered" : "missed"} · set size ${s.size}`)}"><img src="${it.img}" alt="${esc(it.true)}" loading="lazy"><div class="tl">${esc(it.true)}</div>
      ${it.top.slice(0, 5).map(([nm, p], j) => `<div class="row${s.in[j] ? " in" : ""}"><span>${esc(nm.slice(0, 18))}</span><span>${p.toFixed(2)}</span></div>`).join("")}
      <div class="sz">set size ${s.size}${s.size > 5 ? ` (${s.size - s.in.slice(0, 5).filter(Boolean).length} more)` : ""}</div></div>`; }).join("") +
    `<p class="note" style="grid-column:1/-1">${cov} of ${IN.items.length} shown covered · mean set size ${f2(sz, 1)} on these images.</p>`;
}

// ---------------------------------------------------------------- label shift (live)
const BETAS = [0.1, 0.3, 1, 3, 10, 30, 100];
const lsst = { bi: 2, draw: 1 };
let lsCache = { key: "" };
function renderLabelShift() {
  const d = D[st.ds], K = d.K, key = `${st.ds}|${st.alpha}|${st.calSeed}|${st.ncal}`;
  if (lsCache.key !== key) {
    const per = {};
    for (const m of ["naive", "aps", "mondrian"]) { const P = params({ method: m, T: d.tuned.T }), cal = calibrate(d, P); per[m] = metrics(d.test, predictSets(d.test, P, cal).C, P.T).classCov; }
    lsCache = { key, per };
  }
  const beta = BETAS[lsst.bi]; $("lsB").value = lsst.bi; $("lsBV").textContent = beta;
  const prior = dirichlet(rng(1000 + lsst.draw * 31 + lsst.bi), K, beta);
  const cov = (cc) => prior.reduce((a, p, c) => a + p * (Number.isFinite(cc[c]) ? cc[c] : 1), 0);
  const rows = [["Naive (T)", "naive", "s1"], ["APS (T)", "aps", "s2"], ["Mondrian", "mondrian", "s4"]].map(([l, m, c]) => ({ l, c, v: cov(lsCache.per[m]), worst: minF(lsCache.per[m]) }));
  const w = 420, rowH = 46, mg = { l: 90, r: 20, t: 10, b: 26 }, h = rows.length * rowH + mg.t + mg.b;
  const lo = Math.max(0, Math.min(...rows.map((r) => Math.min(r.v, r.worst)), 1 - st.alpha) - 0.04), X = scale(lo, 1, mg.l, w - mg.r);
  let s = [lo, (lo + 1) / 2, 1].map((t) => `<text x="${X(t)}" y="${h - 8}" text-anchor="middle">${fmt(t, 0)}</text>`).join("");
  rows.forEach((r, i) => { const y = mg.t + i * rowH;
    s += `<text x="${mg.l - 8}" y="${y + 20}" text-anchor="end" class="lbl-ink">${r.l}</text><rect class="${r.c}" x="${mg.l}" y="${y + 6}" width="${Math.max(1, X(r.v) - mg.l)}" height="20" rx="3"/>`;
    s += `<text x="${X(r.v) + 5}" y="${y + 20}" class="lbl-ink">${fmt(r.v)}</text><line class="sg" style="stroke-width:2" x1="${X(r.worst)}" x2="${X(r.worst)}" y1="${y + 2}" y2="${y + 32}"/>`;
    s += `<rect class="hit" x="0" y="${y}" width="${w}" height="${rowH}" data-tip="${esc(`${r.l}: ${fmt(r.v)} on this mix; worst possible under any label shift = its worst class, ${fmt(r.worst)}`)}"/>`; });
  s += `<line class="target" x1="${X(1 - st.alpha)}" x2="${X(1 - st.alpha)}" y1="${mg.t}" y2="${h - mg.b}"/>`;
  $("lsChart").innerHTML = svg(w, h, s, "Coverage under the drawn label shift");
  const tv = 0.5 * prior.reduce((a, p) => a + Math.abs(p - 1 / K), 0);
  $("lsNote").textContent = `${DS_LABEL[st.ds]}, α = ${st.alpha.toFixed(3)} · this draw moves ${fmt(tv, 0)} of the class mass (total variation). Grey tick: the worst any label shift can do = that method's worst class.`;
  const order = prior.map((p, c) => [p, c]).sort((a, b) => b[0] - a[0]);
  bars($("lsPrior"), order.map((o) => o[0]), { labels: K <= 12 ? order.map((o) => names(st.ds)[o[1]]) : order.map((_, j) => (j === 0 ? "most" : "")), yDom: [0, Math.max(...prior) * 1.1], yFmt: (v) => fmt(v, 0),
    tip: (v, j) => `${d.classes[order[j][1]]}: ${fmt(v)} of test images (uniform: ${fmt(1 / K)})`, label: "Shifted class proportions", target: 1 / K, h: 200 });
  const LS = (R && R.label_shift || []).filter((r) => r.dataset === st.ds);
  if (!LS.length) { $("lsTable").innerHTML = `<p class="note">No precomputed label-shift draws for ${DS_LABEL[st.ds]}.</p>`; return; }
  const bs = [...new Set(LS.map((r) => r.beta))].sort((a, b) => a - b), cols = ["naive", "temp_aps", "mondrian", "mondrian_pac", "ls_bbse"].filter((c) => LS[0][c] != null);
  $("lsTable").innerHTML = `<table><thead><tr><th>β</th><th>shift (TV)</th>${cols.map((c) => `<th>${PRETTY[c] || c}</th>`).join("")}</tr></thead><tbody>${bs.map((b) => { const g = LS.filter((r) => r.beta === b);
    return `<tr><td>${b}</td><td>${fmt(mean(g.map((r) => r.tv_shift)), 0)}</td>${cols.map((c) => { const v = g.map((r) => r[c]); return `<td>${fmt(mean(v))} · <span style="color:${Math.min(...v) < 0.88 ? "var(--bad)" : "var(--muted)"}">${fmt(Math.min(...v))}</span></td>`; }).join("")}</tr>`; }).join("")}</tbody></table>`;
}

// ---------------------------------------------------------------- covariate shift (live) + precomputed real shift, ACI, weighted
function shiftSources(d) {
  const src = [["synthetic", d.ds === "mnist" ? "rotation" : "pixel noise"]];
  const cs = [...new Set(Object.keys(d.real || {}).filter((k) => k.includes(" · ")).map((k) => k.split(" · ")[0]))];
  if (d.real && d.real["CIFAR-10.1"]) src.push(["cifar10.1", "CIFAR-10.1"]);
  cs.forEach((c) => src.push([c, c]));
  return src;
}
function shiftLevels(d) { // [[label, block|null(clean)]]
  if (st.src === "synthetic") return [["clean", null], ...Object.keys(d.shift).sort((a, b) => +a - +b).map((k) => [d.ds === "mnist" ? `${k}°` : `σ=${k}`, d.shift[k]])];
  if (st.src === "cifar10.1") return [["clean", null], ["CIFAR-10.1", d.real["CIFAR-10.1"]]];
  return [["clean", null], ...[1, 2, 3, 4, 5].map((s) => [`sev ${s}`, d.real[`${st.src} · ${s}`]]).filter((x) => x[1])];
}
function renderShift() {
  const d = D[st.ds], P = params(), cal = lab.cal;
  const srcs = shiftSources(d); if (!srcs.some((s) => s[0] === st.src)) st.src = "synthetic";
  seg($("shiftSrc"), srcs, st.src, (v) => { st.src = v; st.sev = 0; renderShift(); });
  const lv = shiftLevels(d);
  const pts = lv.map(([l, b], i) => [i, b ? metrics(b, predictSets(b, P, cal).C, P.T) : lab.mt, l, b]);
  $("sev").max = lv.length - 1; st.sev = Math.min(st.sev, lv.length - 1); $("sev").value = st.sev;
  $("sevV").textContent = lv[st.sev][0];
  lines($("shiftChart"), [{ pts: pts.map((p) => [p[0], p[1].cov]), cls: "s1", tips: pts.map((p) => `${p[2]}: coverage ${fmt(p[1].cov)}, mean size ${p[1].mean.toFixed(2)}`) },
    { pts: [[st.sev, pts[st.sev][1].cov]], cls: "s2", r: 6 }], { w: 420, h: 240, xDom: [0, Math.max(1, lv.length - 1)], xTicks: lv.map((_, i) => i), xFmt: (i) => lv[i][0],
    yDom: [Math.max(0, Math.min(...pts.map((p) => p[1].cov), 1 - st.alpha) - 0.05), 1], target: 1 - st.alpha, label: "Coverage against shift severity" });
  $("shiftNote").textContent = `${PRETTY[st.method] || st.method}, α = ${st.alpha.toFixed(3)}`;
  const cur = pts[st.sev][1];
  $("shiftNow").innerHTML = `coverage <b>${fmt(cur.cov)}</b> · mean size ${cur.mean.toFixed(2)}`;
  const b = lv[st.sev][1];
  $("shiftStrip").innerHTML = !b ? d.gallery.slice(0, 8).map((g) => card(g.img, lab.C, g.i, d, d.test.labels[g.i])).join("")
    : (() => { const C = predictSets(b, P, cal).C; return b.imgs.map((img, i) => card(img, C, i, d, b.labels[i])).join(""); })();
}
const rst = { ds: "cifar10", aci: "cifar10" };
function renderReal() {
  const RS = (R && R.real_shift) || [], dss = runsIn(RS);
  if (!dss.length) { $("realChart").innerHTML = `<p class="note">Run scripts/research.py after CIFAR-10-C is cached.</p>`; }
  else {
    if (!dss.includes(rst.ds)) rst.ds = dss[0];
    seg($("rsDs"), dss.map((k) => [k, DS_LABEL[k]]), rst.ds, (v) => { rst.ds = v; renderReal(); });
    const ms = [["temp_naive", "s1"], ["temp_aps", "s2"], ["raps", "s3"], ["mondrian", "s4"], ["clustered", "s5"], ["top1", "sg"]];
    const ser = ms.map(([m, c]) => { const pts = [0, 1, 2, 3, 4, 5].map((sv) => { const g = RS.filter((r) => r.dataset === rst.ds && r.method === m && (sv === 0 ? r.family === "real" : r.family === "c" && r.severity === sv)); return g.length ? [sv, mean(g.map((r) => r.coverage))] : null; }).filter(Boolean);
      return { pts, cls: c, dash: m === "top1", tips: pts.map(([sv, v]) => `${PRETTY[m]} · ${sv === 0 ? "CIFAR-10.1" : "severity " + sv}: ${fmt(v)}`) }; }).filter((s) => s.pts.length);
    lines($("realChart"), ser, { w: 420, h: 250, xDom: [0, 5], xTicks: [0, 1, 2, 3, 4, 5], xFmt: (v) => (v === 0 ? "10.1" : "sev " + v), target: 0.9, label: "Coverage on CIFAR-10-C" });
    $("realLegend").innerHTML = ms.map(([m, c]) => `<span><span class="sw ${c}"></span>${PRETTY[m]}</span>`).join("") + `<span><i class="t"></i>target</span><span class="note">x = 0: CIFAR-10.1; 1–5: CIFAR-10-C severity, mean of 15 corruptions</span>`;
  }
  const AS = (R && R.aci_series) || [], ads = runsIn(AS);
  if (!ads.length) { $("aciChart").innerHTML = `<p class="note">Run scripts/research.py.</p>`; }
  else {
    if (!ads.includes(rst.aci)) rst.aci = ads[0];
    seg($("aciDs"), ads.map((k) => [k, DS_LABEL[k]]), rst.aci, (v) => { rst.aci = v; renderReal(); });
    const rows = AS.filter((r) => r.dataset === rst.aci && r.rolling_cov != null);
    const ser = [[0, "sg"], [0.005, "s1"], [0.02, "s2"]].map(([lr, c]) => ({ pts: rows.filter((r) => r.lr === lr).map((r) => [r.t, r.rolling_cov]), cls: c, dots: false }));
    const segs = []; rows.filter((r) => r.lr === 0).forEach((r, i, a) => { if (i === 0 || r.segment !== a[i - 1].segment) segs.push([r.t, r.segment]); });
    const tmax = Math.max(...rows.map((r) => r.t));
    lines($("aciChart"), ser, { w: 420, h: 250, xDom: [0, tmax], yDom: [0, 1.02], target: 0.9, xFmt: (v) => Math.round(v), label: "Rolling coverage over a drifting stream",
      extra: (X, Y, m) => segs.map(([t, l], i) => `<line class="axis" x1="${X(t)}" x2="${X(t)}" y1="${m.t}" y2="${250 - m.b}"/><text x="${X(t) + 3}" y="${m.t + 10 + (i % 2) * 11}" style="font-size:9.5px">${esc(String(l))}</text>`).join("") });
    const A = (R.aci || []).filter((r) => r.dataset === rst.aci), last = segs.length ? segs[segs.length - 1][1] : null;
    const g = (lr, segm, f) => mean(A.filter((r) => r.lr === lr && String(r.segment) === String(segm)).map((r) => r[f]));
    if (last != null) $("aciNote").innerHTML = `In the hardest segment (${esc(String(last))}): fixed split conformal covers <strong>${fmt(g(0, last, "coverage"))}</strong>; ACI (η = 0.005) covers <strong>${fmt(g(0.005, last, "coverage"))}</strong>, at mean set size ${f2(g(0.005, last, "mean_size"), 1)} vs ${f2(g(0, last, "mean_size"), 1)}. Over the whole stream: ${fmt(g(0, "all", "coverage"))} vs ${fmt(g(0.005, "all", "coverage"))}.`;
  }
  const WT = (R && R.weighted) || [], wds = runsIn(WT);
  if (!wds.length) { $("wtTable").innerHTML = `<p class="note">Run scripts/research.py.</p>`; return; }
  $("wtTable").innerHTML = `<table><thead><tr><th>Run</th><th>Shifted set</th><th>Split conformal</th><th>Weighted</th><th>Size: split → weighted</th><th>Effective n<sub>cal</sub></th></tr></thead><tbody>${
    WT.sort((a, b) => RUN_ORDER.indexOf(a.dataset) - RUN_ORDER.indexOf(b.dataset)).map((r) => `<tr><td style="font-family:var(--body)">${DS_LABEL[r.dataset]}</td><td>${esc(r.set.replace("shift/", "synthetic ").replace("real/cifar10_1", "CIFAR-10.1").replace("c/", "C: ").replace("_", " "))}</td><td>${fmt(r.cov_standard)}</td><td class="${r.cov_weighted - r.cov_standard > 0.01 ? "best" : ""}">${fmt(r.cov_weighted)}</td><td>${f2(r.size_standard)} → ${f2(r.size_weighted)}</td><td>${Math.round(r.ess_cal).toLocaleString()} / ${Math.round(r.n_cal).toLocaleString()}</td></tr>`).join("")
  }</tbody></table>`;
  const gain = mean(WT.map((r) => r.cov_weighted - r.cov_standard));
  const inf = mean(WT.map((r) => r.frac_inf));
  $("wtNote").textContent = `Mean gain ${(gain * 100).toFixed(1)} points, but bought with size: on average ${(inf * 100).toFixed(0)}% of shifted images get every label. When the domain classifier separates shifted from clean images almost perfectly, a test point's own weight exceeds the α budget and its set is the full label set; the effective calibration size collapses. That is the support-mismatch limit of covariate-shift conformal.`;
}

// ---------------------------------------------------------------- OOD (live)
function renderOOD() {
  const d = D[st.ds], P = params(), cal = lab.cal;
  const sets = [["in-distribution test", lab.mt]], strips = [];
  for (const [name, b] of Object.entries(d.ood)) { const o = predictSets(b, P, cal); sets.push([name, metrics(b, o.C, P.T)]); strips.push([name, b, o.C]); }
  const w = 760, rowH = 34, mg = { l: 150, r: 10 }, h = sets.length * rowH + 22, X = scale(0, 1, mg.l, w - mg.r);
  let s = "";
  [0, 0.25, 0.5, 0.75, 1].forEach((t) => { s += `<text x="${X(t)}" y="${h - 2}" text-anchor="middle">${fmt(t, 0)}</text>`; });
  sets.forEach(([name, m], r) => {
    const y = 4 + r * rowH, parts = [["empty", m.empty, "ghost"], ["singleton", m.single, "bar"], ["2+", 1 - m.empty - m.single, ""]];
    s += `<text x="${mg.l - 10}" y="${y + 18}" text-anchor="end" class="lbl-ink">${esc(name.replace("_", "-"))}</text>`;
    let x = 0;
    parts.forEach(([lab_, v, cls]) => {
      if (v <= 0) return;
      const x0 = X(x) + (x > 0 ? 1 : 0), x1 = X(x + v) - 1;
      s += `<rect x="${x0}" y="${y + 4}" width="${Math.max(1, x1 - x0)}" height="20" rx="2" class="${cls}" style="${cls === "" ? "fill:var(--target)" : cls === "ghost" ? "opacity:.5" : ""}" data-tip="${esc(name)}: ${lab_} ${fmt(v)}"/>`;
      if (v > 0.08) s += `<text x="${(x0 + x1) / 2}" y="${y + 18}" text-anchor="middle" style="fill:${cls === "ghost" ? "var(--ink)" : "#fff"}">${fmt(v, 0)}</text>`;
      x += v;
    });
  });
  $("oodChart").innerHTML = svg(w, h, s, "Set size mix for in-distribution and OOD inputs");
  $("oodStrips").innerHTML = strips.map(([name, b, C]) => `<div class="panel"><div class="panel-h"><h3>${esc(name.replace("_", "-"))}</h3><p>${fmt(metrics(b, C, P.T).single, 0)} confident singletons</p></div>
    <div class="strip">${b.imgs.map((img, i) => card(img, C, i, d, null, "no valid label")).join("")}</div></div>`).join("");
}

// ---------------------------------------------------------------- diagnostics
function renderDiagnostics() {
  const ES = ((R && R.empty_sets) || []).filter((r) => r.bound_applies >= 0.999 && (r.method === "naive" || r.method === "temp_naive"));
  if (ES.length) {
    const cls = { mnist: "s1", cifar10: "s2", cifar10_r18: "s3", cifar100: "s6", imagenet: "s4" };
    const mx = Math.max(...ES.map((r) => Math.max(r.empty_rate, r.bound))) * 1.08;
    lines($("emptyChart"), runsIn(ES).map((ds) => ({ pts: ES.filter((r) => r.dataset === ds).map((r) => [r.bound, r.empty_rate]), cls: cls[ds], noLine: true,
      tips: ES.filter((r) => r.dataset === ds).map((r) => `${DS_LABEL[ds]}, α=${r.alpha}, ${PRETTY[r.method]}: empty ${fmt(r.empty_rate)} ≥ bound ${fmt(r.bound)}`) })),
    { w: 420, h: 250, xDom: [0, mx], yDom: [0, mx], xFmt: (v) => fmt(v, 0), label: "Empty-set rate vs bound",
      extra: (X, Y) => `<path class="ln sg dash" d="M${X(0)},${Y(0)}L${X(mx)},${Y(mx)}"/>` + runsIn(ES).map((ds, i) => `<circle class="${cls[ds]}" cx="${X(mx * 0.04)}" cy="${Y(mx * (0.95 - i * 0.07))}" r="4"/><text x="${X(mx * 0.04) + 8}" y="${Y(mx * (0.95 - i * 0.07)) + 4}">${DS_LABEL[ds]}</text>`).join("") });
  }
  const DD = (R && R.distance_diag) || [];
  if (DD.length) {
    $("ddTable").innerHTML = `<table><thead><tr><th>Run</th><th>AUROC softmax</th><th>AUROC distance</th><th>Spearman ρ</th><th>tuned γ (naive)</th></tr></thead><tbody>${
      DD.sort((a, b) => RUN_ORDER.indexOf(a.dataset) - RUN_ORDER.indexOf(b.dataset)).map((r) => `<tr><td style="font-family:var(--body)">${DS_LABEL[r.dataset]}</td><td><strong>${r.auroc_rank2_softmax.toFixed(3)}</strong></td><td>${r.auroc_rank2_distance.toFixed(3)}</td><td>${r.spearman_softmax_distance.toFixed(2)}</td><td>${r.gamma_naive.toFixed(2)}</td></tr>`).join("")}</tbody></table>`;
    $("ddNote").textContent = `AUROC for "is the 2nd-ranked label the true one?". The softmax already ranks that decision better than the feature distance in every run, and the two are strongly correlated (ρ), so adding γ·d̃ mostly re-weights information the score already has.`;
  }
}

// ---------------------------------------------------------------- data, limitations, conclusion
const INFO = { mnist: ["LeNet", 48000, 6000, 6000, 10000], cifar10: ["SmallResNet", 40000, 5000, 5000, 10000], cifar10_r18: ["ResNet-18 (pretrained)", 40000, 5000, 5000, 10000],
  cifar100: ["SmallResNet", 40000, 5000, 5000, 10000], imagenet: ["ResNet-152 (published)", 0, 10000, 20000, 20000] };
const dst = { ds: "cifar10" };
function renderData() {
  const runs = R ? runsIn(R.main) : Object.keys(D);
  if (!runs.includes(dst.ds)) dst.ds = runs[0];
  seg($("splitDs"), runs.map((k) => [k, DS_LABEL[k]]), dst.ds, (v) => { dst.ds = v; renderData(); });
  const [, tr, de, ca, te] = INFO[dst.ds];
  const parts = [["D_train", tr, "var(--accent)"], ["D_design", de, "#7a5bd1"], ["D_cal", ca, "var(--target)"], ["D_test", te, "#4a5262"]].filter((p) => p[1] > 0);
  $("splitbar").innerHTML = parts.map(([n, v, c]) => `<div style="flex:${v};background:${c}" data-tip="${n}: ${v.toLocaleString()} images">${n} · ${v.toLocaleString()}</div>`).join("");
  if (!R) return;
  $("dataTable").innerHTML = `<table><thead><tr><th>Run</th><th>Model</th><th>K</th><th>Seeds</th><th>Test accuracy</th><th>Fitted T</th><th>ECE raw → T</th><th>Cal. images / class</th></tr></thead><tbody>${runs.map((ds) => {
    const t = (R.tuned || []).filter((r) => r.dataset === ds); if (!t.length) return "";
    return `<tr><td style="font-family:var(--body)">${DS_LABEL[ds]}</td><td style="font-family:var(--body)">${INFO[ds][0]}</td><td>${t[0].K || 10}</td><td>${t.length}</td><td>${fmt(mean(t.map((r) => r.accuracy)))}</td><td>${mean(t.map((r) => r.T)).toFixed(2)}</td><td>${mean(t.map((r) => r.ece_raw)).toFixed(3)} → ${mean(t.map((r) => r.ece_T)).toFixed(3)}</td><td>${Math.round(mean(t.map((r) => r.n_cal)) / (t[0].K || 10))}</td></tr>`; }).join("")}</tbody></table>`;
}
function renderConclusion() {
  if (!R) return;
  const g = (ds, m, f) => mean(R.main.filter((r) => r.dataset === ds && r.method === m && Math.abs(r.alpha - 0.1) < 1e-9).map((r) => r[f]));
  const wc = (ds, f, m = "mondrian") => mean((R.worst_class || []).filter((r) => r.dataset === ds && r.method === m && Math.abs(r.alpha - 0.1) < 1e-9).map((r) => r[f]));
  const runs = runsIn(R.main), big = runs.includes("imagenet") ? "imagenet" : runs[runs.length - 1];
  const lim = [
    `<strong>Marginal, not conditional.</strong> Only Mondrian-style methods control per-class coverage, and the in-expectation version still leaves a predictable worst class (${runsIn(R.worst_class || []).map((ds) => `${DS_LABEL[ds]} ${fmt(wc(ds, "measured_min"), 0)}`).join(", ")}). No method controls coverage for one image.`,
    `<strong>High-probability guarantees cost size.</strong> With ~20 calibration images per class (${DS_LABEL[big]}), Mondrian-PAC is infeasible and returns ${f2(g(big, "mondrian_pac", "mean_size"), 0)} labels; pooling classes is the only affordable route, and its guarantee is per cluster, not per class.`,
    `<strong>Exchangeability is the whole contract.</strong> Under real corruption and rotation, coverage falls far below target. ACI restores long-run coverage only by growing sets; weighted conformal recovers coverage only by returning every label when shifted and clean images barely overlap.`,
    `<strong>Scale and models.</strong> Small backbones trained here plus one published ImageNet model; seeds: ${runs.map((ds) => `${DS_LABEL[ds]} ${(R.tuned || []).filter((r) => r.dataset === ds).length}`).join(", ")}. CIFAR-10 test images were collected separately from training images, so exchangeability with D_cal is approximate (pooled re-splits average exactly 1 − α).`,
    `<strong>Sweeps on D_test are sensitivity only.</strong> γ, the number of clusters and ACI's η are reported across a grid; headline numbers use values fixed on D_design or the stated default.`,
  ];
  $("limitList").innerHTML = lim.map((t) => `<li>${t}</li>`).join("");
  $("conclusion").innerHTML = `<h3>Conclusion</h3><p>At a fixed guaranteed coverage, the score decides <em>who</em> is covered, and the grouping decides <em>which classes</em> are protected. The worst class under class-wise calibration is not a surprise: it follows an exact law that matches exchangeable re-splits to within about a point in every setting (ImageNet: ${fmt(wc("imagenet", "pred_mean"))} predicted, ${fmt(wc("imagenet", "measured_min"))} measured). Where it misses, it is informative: on CIFAR-100's official test set the measured worst class sits ${(100 * (wc("cifar100", "pred_mean") - wc("cifar100", "measured_min"))).toFixed(1)} points below it, exposing a per-class shift between the training pool and the separately collected test set. That law says per-class guarantees need thousands of images per class at ImageNet scale. A high-probability version is affordable only by sharing thresholds: Clustered-PAC reaches a worst class of ${fmt(g(big, "clustered_pac", "min_class_cov"))} on ${DS_LABEL[big]} with ${f2(g(big, "clustered_pac", "mean_size"), 1)} labels per set, where Mondrian-PAC needs the full ${f2(g(big, "mondrian_pac", "mean_size"), 0)}. Report the worst class and the empty-set rate next to marginal coverage.</p>`;
}

// ---------------------------------------------------------------- tour
const TOUR = [
  { id: "top", t: "Sets instead of answers", x: "Each image gets a set of labels. At α = 0.1 the true label is inside 90% of the time, on average. Green frames are covered, red are misses.", go: () => { pickDsIf("cifar10"); st.method = "aps"; st.alpha = 0.1; } },
  { id: "lab", t: "Small sets, uneven protection", x: "Naive gives the smallest sets, but look at the worst class tile and the red bars: some classes are covered far less than 90%.", go: () => { pickDsIf("cifar10"); st.method = "naive"; } },
  { id: "theorem", t: "The guarantee is an average", x: "Coverage of one calibrated predictor is random. Over 400 re-splits it follows an exact Beta–binomial law; at small n, many deployments sit below target.", go: () => { st.ncal = 200; } },
  { id: "classes", t: "Who gets covered", x: "With 100 or 1000 classes, the sorted per-class curve shows a long tail of under-covered classes, even for per-class (Mondrian) calibration.", go: () => { res.ccDs = R && runsIn(R.class_cov).includes("cifar100") ? "cifar100" : res.ccDs; st.ncal = D[st.ds].cal.n; } },
  { id: "worst", t: "The worst class is predictable", x: "Per-class coverages are independent Beta variables, so the worst class has an exact law. The table compares the prediction with every run.", go: () => { Object.assign(wst, { K: 100, n: 50, m: 100 }); } },
  { id: "fix", t: "Fixing it costs size, unless you pool", x: "Mondrian-PAC guarantees every class with high probability, but at ImageNet scale it returns all labels. Clustered-PAC keeps a high-probability guarantee with far smaller sets.", go: () => { cst.ds = "imagenet"; } },
  { id: "labelshift", t: "Why the worst class matters", x: "Under label shift, a marginal method can fall to its worst class. Draw shifts and compare with Mondrian, which stays at target.", go: () => { lsst.bi = 0; lsst.draw++; } },
  { id: "shift", t: "When images change", x: "Real corruptions break exchangeability and coverage collapses. Adaptive conformal (ACI) recovers long-run coverage by growing sets.", go: () => { st.src = "gaussian noise"; st.sev = 0; } },
  { id: "diagnostics", t: "Two explanations", x: "Empty sets are forced when accuracy beats 1 − α (a one-line bound), and the distance score adds little because softmax already ranks the decision better.", go: () => {} },
];
let tourI = -1;
function pickDsIf(ds) { if (D[ds] && st.ds !== ds) { st.ds = ds; const d = D[ds]; Object.assign(st, { T: d.tuned.T, ncal: d.cal.n, sel: null, sev: 0, src: "synthetic" }); } }
function showTour(i) {
  document.querySelectorAll("section.tour-on").forEach((s) => s.classList.remove("tour-on"));
  if (i < 0 || i >= TOUR.length) { $("tour").hidden = true; tourI = -1; return; }
  tourI = i; const s = TOUR[i];
  try { s.go(); } catch (e) { console.warn(e); }
  renderAll(); renderEverythingElse();
  $("tour").hidden = false; $("tourStep").textContent = `Step ${i + 1} of ${TOUR.length}`; $("tourTitle").textContent = s.t; $("tourText").textContent = s.x;
  $("tourDots").textContent = TOUR.map((_, j) => (j === i ? "●" : "○")).join(" "); $("tourPrev").disabled = i === 0; $("tourNext").textContent = i === TOUR.length - 1 ? "Finish" : "Next";
  const el = $(s.id); if (el) { el.classList.add("tour-on"); el.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" }); }
}

// ---------------------------------------------------------------- shareable state
const KEYS = { ds: "ds", m: "method", a: "alpha", T: "T", g: "gamma", l: "lam", k: "kreg", d: "delta", n: "ncal", c: "calSeed" };
function stateString() { return Object.entries(KEYS).map(([k, f]) => `${k}=${encodeURIComponent(typeof st[f] === "number" ? +st[f].toFixed(4) : st[f])}`).join("&"); }
function readState() {
  let q = "";
  try { q = location.search.slice(1) || (location.hash.startsWith("#v=") ? decodeURIComponent(location.hash.slice(3)) : ""); } catch { q = ""; }
  if (!q) return false;
  const p = new URLSearchParams(q); let any = false;
  for (const [k, f] of Object.entries(KEYS)) if (p.has(k)) { const v = p.get(k); st[f] = f === "ds" || f === "method" ? v : +v; any = true; }
  return any;
}

// ---------------------------------------------------------------- wiring
function renderAll() {
  if (!D[st.ds]) return;
  syncControls(); renderHero(); renderLab(); renderTheorem(); renderShift(); renderOOD(); renderLabelShift();
}
function renderEverythingElse() { renderResults(); renderClassAudit(); renderWorst(); renderClusters(); renderReal(); renderDiagnostics(); renderData(); renderConclusion(); renderKpis(); }
let raf = 0;
function schedule(fn) { cancelAnimationFrame(raf); raf = requestAnimationFrame(fn); }
function bind() {
  const on = (id, f) => $(id).addEventListener("input", (e) => { f(e.target.value); schedule(() => { syncControls(); renderLab(); renderShift(); renderOOD(); renderTheorem(); renderLabelShift(); renderResults(); }); });
  on("method", (v) => { st.method = v; if (v.startsWith("dist_")) st.gamma = D[st.ds].tuned.gamma[v]; if (v === "saps") st.lam = D[st.ds].tuned.saps_lam; if (v === "raps") st.lam = D[st.ds].tuned.raps.lam; });
  on("alpha", (v) => { st.alpha = +v; $("heroAlpha").value = v; $("heroAlphaV").textContent = (+v).toFixed(3); schedule(renderHero); });
  on("temp", (v) => { st.T = +v; });
  on("gamma", (v) => { st.gamma = +v; });
  on("lam", (v) => { st.lam = +v; });
  on("kreg", (v) => { st.kreg = +v; });
  on("delta", (v) => { st.delta = +v; });
  on("ncal", (v) => { st.ncal = +v; });
  $("tReset").onclick = () => { st.T = D[st.ds].tuned.T; renderAll(); };
  $("redraw").onclick = () => { st.calSeed++; renderAll(); };
  $("heroAlpha").addEventListener("input", (e) => {
    st.alpha = +e.target.value; $("heroAlphaV").textContent = st.alpha.toFixed(3);
    schedule(() => { renderHero(); syncControls(); renderLab(); renderShift(); renderOOD(); renderTheorem(); renderLabelShift(); });
  });
  $("sev").addEventListener("input", (e) => { st.sev = +e.target.value; schedule(renderShift); });
  $("gallery").onclick = (e) => { const c = e.target.closest(".card"); if (!c) return; st.sel = +c.dataset.i; renderDetail(); };
  $("heroStrip").onclick = (e) => { const c = e.target.closest(".card"); if (!c) return; st.sel = +c.dataset.i; renderDetail(); $("lab").scrollIntoView({ behavior: "smooth" }); };
  const wOn = (id, f) => $(id).addEventListener("input", (e) => { f(+e.target.value); schedule(renderWorst); });
  wOn("wK", (v) => { wst.K = WK[v]; }); wOn("wN", (v) => { wst.n = logSlider(v); }); wOn("wM", (v) => { wst.m = logSlider(v); });
  wOn("wA", (v) => { wst.alpha = v; }); wOn("wD", (v) => { wst.delta = v; });
  $("clM").addEventListener("input", (e) => { cst.Mi = +e.target.value; schedule(renderClusters); });
  $("lsB").addEventListener("input", (e) => { lsst.bi = +e.target.value; schedule(renderLabelShift); });
  $("lsDraw").onclick = () => { lsst.draw++; renderLabelShift(); };
  $("tourBtn").onclick = () => showTour(0);
  $("tourNext").onclick = () => showTour(tourI + 1);
  $("tourPrev").onclick = () => showTour(tourI - 1);
  $("tourClose").onclick = () => showTour(-1);
  document.addEventListener("keydown", (e) => { if (tourI >= 0 && e.key === "Escape") showTour(-1); });
  $("shareBtn").onclick = async () => {
    let base = ""; try { base = location.href.split(/[?#]/)[0]; } catch { base = ""; }
    const url = `${base}?${stateString()}`;
    try { await navigator.clipboard.writeText(url); $("shareNote").textContent = "Link copied."; }
    catch { $("shareNote").textContent = `#v=${stateString()}`; }
    setTimeout(() => { $("shareNote").textContent = ""; }, 6000);
  };
  const links = [...document.querySelectorAll("nav.rail a")];
  const io = new IntersectionObserver((es) => es.forEach((en) => { if (en.isIntersecting) links.forEach((a) => a.classList.toggle("on", a.getAttribute("href") === "#" + en.target.id)); }), { rootMargin: "-40% 0px -55% 0px" });
  document.querySelectorAll("main section").forEach((s) => io.observe(s));
}

async function load() {
  bind();
  try { R = await (await fetch("data/results.json")).json(); } catch { R = null; }
  try { IN = await (await fetch("data/imagenet_gallery.json")).json(); } catch { IN = null; }
  const list = (R && R.datasets && R.datasets.length) ? R.datasets : ["mnist", "cifar10"];
  for (const ds of list) {
    try {
      const j = await (await fetch(`data/${ds}.json`)).json(), K = j.K || 10;
      D[ds] = { ...j, ds, K, cal: decodeBlock(j.cal, 202, K), test: decodeBlock(j.test, 303, K),
        shift: Object.fromEntries(Object.entries(j.shift || {}).map(([k, b]) => [k, decodeBlock(b, 400 + Math.round(+k * 100), K)])),
        ood: Object.fromEntries(Object.entries(j.ood || {}).map(([k, b], i) => [k, decodeBlock(b, 500 + i, K)])),
        real: Object.fromEntries(Object.entries(j.real || {}).map(([k, b], i) => [k, decodeBlock(b, 700 + i, K)])) };
    } catch (e) { console.warn("could not load", ds, e); }
  }
  if (!Object.keys(D).length) { $("heroStrip").innerHTML = `<p class="note">Data files are missing. Run <code>python scripts/export_site.py</code>, then serve the site folder.</p>`; return; }
  if (R) res.alpha = R.config.main_alpha;
  const fromUrl = readState();
  if (!D[st.ds]) st.ds = D.cifar10 ? "cifar10" : Object.keys(D)[0];
  if (!METHODS[st.method]) st.method = "aps";
  pickDs(st.ds, fromUrl);
  if (fromUrl) { $("heroAlpha").value = st.alpha; $("heroAlphaV").textContent = st.alpha.toFixed(3); renderAll(); }
  renderEverythingElse();
}
load();
