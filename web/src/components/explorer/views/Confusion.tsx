"use client";
import { useMemo } from "react";
import { cooccurrence, pct } from "@/lib/conformal";
import { Seg } from "../../ui";
import { AlphaPick, DataTable, DsPick, Figure, Heatmap, LIVE_METHODS, Loading, useLab } from "../kit";

const TOP = 20; // a 100 x 100 map is unreadable: show the classes most often confused

export default function Confusion({ p }: { p: string }) {
  const L = useLab(p), { d, out } = L;
  const map = useMemo(() => {
    if (!d || !out) return null;
    const K = d.K, M = cooccurrence(out.C, d.test.n, K);
    const off = Array.from({ length: K }, (_, a) => { let s = 0; for (let b = 0; b < K; b++) if (b !== a) s += M[a * K + b]; return s; });
    const idx = Array.from({ length: K }, (_, k) => k).sort((a, b) => off[b] - off[a]).slice(0, Math.min(K, TOP)).sort((a, b) => a - b);
    const pairs: [number, number, number][] = [];
    for (let a = 0; a < K; a++) for (let b = a + 1; b < K; b++) if (M[a * K + b]) pairs.push([a, b, M[a * K + b]]);
    pairs.sort((x, y) => y[2] - x[2]);
    return { M, K, idx, pairs: pairs.slice(0, 12) };
  }, [d, out]);

  return (
    <div className="stack" style={{ gap: 28 }}>
      <div className="stack xcontrols">
        <div className="row" style={{ justifyContent: "space-between" }}><DsPick value={L.ds} onChange={L.setDs} /><AlphaPick value={L.alpha} onChange={L.setAlpha} /></div>
        <Seg label="Rule" value={L.method} onChange={L.setMethod} options={LIVE_METHODS.filter(([m]) => m !== "top1")} />
      </div>
      {!map || !out || !d ? <Loading error={L.error} what={L.ds} /> : (() => { const share = (r: number, c: number) => map.M[map.idx[r] * map.K + map.idx[c]] / (map.M[map.idx[r] * map.K + map.idx[r]] || 1); return (
        <div className="xsplit">
          <Figure title={map.K > TOP ? `The ${TOP} most-confused classes` : "Every class"} name={`confusion-${L.ds}-${L.method}`} tour="map"
            table={{ cols: ["a", "b", "share_of_a_sets_with_b"], rows: map.idx.flatMap((a) => map.idx.filter((b) => b !== a).map((b) => [out.names[a], out.names[b], map.M[a * map.K + b] / (map.M[a * map.K + a] || 1)])) }}
            note="Row a, column b: of the sets that contain a, the share that also contain b. Darker = more often confused. Hover a square for the number.">
            <Heatmap label="Share of sets containing the row label that also contain the column label" names={map.idx.map((k) => out.names[k])} value={share} max={Math.max(...map.idx.flatMap((_, r) => map.idx.map((__, c) => (r === c ? 0 : share(r, c)))))} />
          </Figure>
          <div className="stack" data-tour="pairs">
            <h3>Most-shared pairs</h3>
            <DataTable t={{ cols: ["labels", "sets with both", "of all sets"], rows: map.pairs.map(([a, b, c]) => [`${out.names[a]} + ${out.names[b]}`, c, pct(c / d.test.n)]) }} />
          </div>
        </div>
      ); })()}
    </div>
  );
}

