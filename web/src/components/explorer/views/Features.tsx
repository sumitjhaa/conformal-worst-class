"use client";
import { useMemo } from "react";
import { i16, pct, predictSets, probs, type Block } from "@/lib/conformal";
import { extraBlock } from "@/lib/data";
import { Seg } from "../../ui";
import { AlphaPick, DsPick, Figure, LIVE_METHODS, Loading, Scatter, Term, useLab, useQuery } from "../kit";

const N = 1500; // test points drawn; enough to see the shape, light enough to render as SVG

/** Per picture: confidence (top probability) and Mahalanobis distance to the predicted class. */
function points(b: Block, dist: Float32Array, T: number, n: number) {
  const p = probs(b, T), K = b.K, out: { conf: number; dist: number; i: number }[] = [];
  for (let i = 0; i < Math.min(n, b.n); i++) {
    let best = 0; for (let k = 1; k < K; k++) if (p[i * K + k] > p[i * K + best]) best = k;
    out.push({ conf: p[i * K + best], dist: dist[i * K + best], i });
  }
  return out;
}
const q = (v: number[], f: number) => [...v].sort((a, b) => a - b)[Math.floor(f * (v.length - 1))];

export default function Features({ p }: { p: string }) {
  const L = useLab(p), { d, out } = L;
  const [color, setColor] = useQuery<string>(`${p}col`, "ood");
  const view = useMemo(() => {
    if (!d || !out || !d.testDist) return null;
    const test = points(d.test, i16(d.testDist), d.tuned.T, N), oname = Object.keys(d.ood || {})[0], raw = oname ? d.ood[oname] : null;
    let ood: { conf: number; dist: number; i: number }[] = [], oC: Uint8Array | null = null;
    if (raw?.dist) { const b = extraBlock(d, "ood", oname)!; ood = points(b, i16(raw.dist), d.tuned.T, b.n); oC = predictSets(b, out.P, out.cal); }
    const size = (C: Uint8Array, i: number) => { let s = 0; for (let k = 0; k < d.K; k++) s += C[i * d.K + k]; return s; };
    const ds_ = [...test, ...ood].map((x) => x.dist), ymin = q(ds_, 0.005), ymax = q(ds_, 0.995); // standardised: can be negative
    return { test, ood, oname, ymin, ymax, tSize: test.map((x) => size(out.C, x.i)), oSize: oC ? ood.map((x) => size(oC!, x.i)) : [] };
  }, [d, out]);

  if (!d || !out) return <div className="stack"><DsPick value={L.ds} onChange={L.setDs} /><Loading error={L.error} what={L.ds} /></div>;
  if (!view) return <p className="loading">No feature distances were exported for {L.ds}.</p>;
  const all = [...view.test.map((x, j) => ({ ...x, s: view.tSize[j], ood: false })), ...view.ood.map((x, j) => ({ ...x, s: view.oSize[j], ood: true }))];
  const pt = (x: { conf: number; dist: number }) => [x.conf, x.dist] as [number, number];
  const groups = color === "ood"
    ? [{ color: "var(--s1)", pts: view.test.map(pt) }, { color: "var(--bad)", pts: view.ood.map(pt), r: 3 }]
    : [{ color: "var(--violet)", pts: all.filter((x) => x.s === 0).map(pt) }, { color: "var(--good)", pts: all.filter((x) => x.s === 1).map(pt) }, { color: "var(--s2)", pts: all.filter((x) => x.s >= 2).map(pt) }];
  const auc = (() => { // AUROC of distance for "foreign vs familiar": share of (ood, test) pairs where the ood picture is farther
    if (!view.ood.length) return NaN;
    const t = view.test.map((x) => x.dist).sort((a, b) => a - b); let s = 0;
    for (const o of view.ood) { let lo = 0, hi = t.length; while (lo < hi) { const m = (lo + hi) >> 1; if (t[m] < o.dist) lo = m + 1; else hi = m; } s += lo; }
    return s / (view.ood.length * t.length);
  })();
  return (
    <div className="stack" style={{ gap: 28 }}>
      <div className="stack xcontrols">
        <div className="row" style={{ justifyContent: "space-between" }}><DsPick value={L.ds} onChange={L.setDs} /><AlphaPick value={L.alpha} onChange={L.setAlpha} /></div>
        <div className="row"><Seg label="Rule" value={L.method} onChange={L.setMethod} options={LIVE_METHODS} /><Seg label="Colour" value={color} onChange={setColor} options={[["ood", "Familiar vs foreign"], ["size", "Set size"]]} /></div>
      </div>
      <Figure title="Confidence against feature distance" name={`features-${L.ds}`} tour="scatter"
        table={{ cols: ["picture", "foreign", "confidence", "distance", "set_size"], rows: all.map((x) => [x.i, x.ood ? "yes" : "no", x.conf, x.dist, x.s]) }}
        note={<>x: the model’s top probability (T = {d.tuned.T.toFixed(2)}). y: <Term t="mahalanobis distance">Mahalanobis distance</Term> from the picture’s internal features to the predicted class, standardised (0 = typical, above 0 = farther than usual). {view.ood.length > 0 && <>Distance separates foreign {view.oname.replace(/_/g, "-")} pictures from familiar ones with AUROC {pct(auc)}.</>}</>}>
        <Scatter label="Each picture by confidence and feature distance" xDom={[0, 1]} yDom={[view.ymin, view.ymax]} xFmt={(v) => pct(v, 0)} yFmt={(v) => v.toFixed(1)} groups={groups} h={360} />
        <div className="legend">{color === "ood" ? <><span><i style={{ background: "var(--s1)" }} />familiar test pictures ({view.test.length})</span><span><i style={{ background: "var(--bad)" }} />foreign: {view.oname?.replace(/_/g, "-")} ({view.ood.length})</span></> : <><span><i style={{ background: "var(--violet)" }} />empty set</span><span><i style={{ background: "var(--good)" }} />one label</span><span><i style={{ background: "var(--s2)" }} />two or more</span></>}</div>
      </Figure>
    </div>
  );
}
