// Self-check for the explorer's in-browser maths: `node scripts/check-explorer.mjs` (Node >= 23 runs the .ts imports).
// 1. conformal.ts on the exported seed-0 logits must match the Python results in results.json (main table, seed 0).
//    The browser uses a 5,000-image test subset, so agreement is checked within sampling noise, not exactly.
// 2. lenet.ts must give the same logits as torch for the two check images in mnist_net.json.
import { readFileSync } from "node:fs";
import { calibrate, decodeBlock, metrics, predictSets } from "../src/lib/conformal.ts";
import { lenet, loadNet } from "../src/lib/lenet.ts";

const data = (f) => JSON.parse(readFileSync(new URL(`../public/data/${f}`, import.meta.url)));
const R = data("results.json");
let fail = 0;
const check = (ok, msg) => { console.log(`${ok ? "ok  " : "FAIL"} ${msg}`); if (!ok) fail++; };

// browser method -> python method it reproduces (the browser always applies the tuned temperature)
const PAIRS = [["naive", "temp_naive"], ["aps", "temp_aps"], ["mondrian", "mondrian"], ["clustered_pac", "clustered_pac"]];
for (const ds of ["mnist", "cifar10"]) {
  const j = data(`${ds}.json`), K = j.K, cal = decodeBlock(j.cal, 202, K), test = decodeBlock(j.test, 303, K);
  for (const [m, py] of PAIRS) {
    const row = R.main.find((r) => r.dataset === ds && r.seed === 0 && Math.abs(r.alpha - 0.1) < 1e-9 && r.method === py);
    if (!row) { check(false, `${ds}/${py}: no seed-0 row in results.json`); continue; }
    const P = { method: m, alpha: 0.1, T: j.tuned.T, delta: j.tuned.delta ?? 0.1 };
    const out = metrics(test, predictSets(test, P, calibrate(cal, P, j.clusters)));
    const dc = Math.abs(out.cov - row.coverage), ds_ = Math.abs(out.mean - row.mean_size) / row.mean_size;
    check(dc < 0.02 && ds_ < 0.1, `${ds}/${m} vs ${py}: coverage ${out.cov.toFixed(4)} vs ${row.coverage.toFixed(4)}, size ${out.mean.toFixed(3)} vs ${row.mean_size.toFixed(3)}`);
  }
}

const nj = data("mnist_net.json"), net = loadNet(nj);
for (const [i, c] of nj.check.entries()) {
  const px = Uint8Array.from(atob(c.img), (ch) => ch.charCodeAt(0)), z = lenet(net, px);
  const err = Math.max(...z.map((v, k) => Math.abs(v - c.logits[k])));
  check(err < 1e-3, `lenet check image ${i}: max |logit - torch| = ${err.toExponential(2)}`);
}

if (fail) { console.error(`${fail} check(s) failed`); process.exit(1); }
console.log("all explorer checks passed");
