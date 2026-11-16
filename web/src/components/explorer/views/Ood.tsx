"use client";
import { useMemo } from "react";
import { pct, predictSets, setOf, type Block } from "@/lib/conformal";
import { extraBlock, loadResults, rows, useLoad } from "@/lib/data";
import { HBars, ImgCard, Seg, Stat } from "../../ui";
import { AlphaPick, DsPick, dsName, Figure, LIVE_METHODS, Loading, mname, Term, useLab, useQuery } from "../kit";

function sizeStats(C: Uint8Array, n: number, K: number) {
  let empty = 0, single = 0, full = 0, sum = 0;
  for (let i = 0; i < n; i++) { let s = 0; for (let k = 0; k < K; k++) s += C[i * K + k]; sum += s; if (s === 0) empty++; else if (s === 1) single++; if (s === K) full++; }
  return { mean: sum / n, empty: empty / n, single: single / n, full: full / n };
}

export default function Ood({ p }: { p: string }) {
  const L = useLab(p), { d, out } = L;
  const { data: R } = useLoad(loadResults);
  const names = d ? Object.keys(d.ood || {}) : [];
  const [o, setO] = useQuery<string>(`${p}ood`, "");
  const oname = names.includes(o) ? o : names[0];

  const live = useMemo(() => {
    const on = d ? (Object.keys(d.ood || {}).includes(o) ? o : Object.keys(d.ood || {})[0]) : "";
    if (!d || !out || !on) return null;
    const b = extraBlock(d, "ood", on) as Block, C = predictSets(b, out.P, out.cal);
    return { b, C, ood: sizeStats(C, b.n, d.K), ind: sizeStats(out.C, d.test.n, d.K) };
  }, [d, out, o]);
  const oodRows = rows(R, "ood").filter((r) => r.dataset === L.ds), atA = oodRows.filter((r) => Math.abs(Number(r.alpha) - L.alpha) < 1e-9);
  const table = (atA.length ? atA : oodRows.filter((r) => Math.abs(Number(r.alpha) - 0.1) < 1e-9)).sort((a, b) => Number(b.auroc_not_singleton) - Number(a.auroc_not_singleton));

  return (
    <div className="stack" style={{ gap: 28 }}>
      <div className="stack xcontrols">
        <div className="row" style={{ justifyContent: "space-between" }}><DsPick value={L.ds} onChange={(v) => { L.setDs(v); setO(""); }} /><AlphaPick value={L.alpha} onChange={L.setAlpha} /></div>
        <Seg label="Rule" value={L.method} onChange={L.setMethod} options={LIVE_METHODS} />
        {names.length > 1 && <Seg label="Foreign pictures" value={oname} onChange={setO} options={names.map((n) => [n, n.replace(/_/g, "-")])} />}
      </div>
      {!live || !out || !d ? <Loading error={L.error} what={L.ds} /> : (
        <>
          <p>The {dsName(L.ds)} model is shown <strong>{oname.replace(/_/g, "-")}</strong> pictures, which belong to none of its classes. Any set is wrong; a useful set is large or empty, as a warning. These are <Term t="ood">OOD</Term> pictures.</p>
          <div className="grid4" data-tour="stats">
            <Stat k="Mean set size" value={live.ood.mean} format={(v) => v.toFixed(2)} s={`familiar pictures: ${live.ind.mean.toFixed(2)}`} color="var(--s1)" />
            <Stat k="Single label (confidently wrong)" value={live.ood.single} format={(v) => pct(v)} s={`familiar: ${pct(live.ind.single)}`} tone={live.ood.single > 0.4 ? "bad" : "good"} />
            <Stat k="Empty (“none of these”)" value={live.ood.empty} format={(v) => pct(v)} s={`familiar: ${pct(live.ind.empty)}`} color="var(--violet)" />
            <Stat k="Every label (“no idea”)" value={live.ood.full} format={(v) => pct(v)} s={`familiar: ${pct(live.ind.full)}`} color="var(--s2)" />
          </div>
          {live.b.imgs && (
            <div data-tour="gallery">
              <h3 style={{ marginBottom: 12 }}>Foreign pictures and their sets</h3>
              <div className="gallery">{live.b.imgs.map((img, i) => { const s = setOf(live.C, i, d.K); return <ImgCard key={i} img={img} trueName="(not a class)" covered labels={s.map((k) => out.names[k])} />; })}</div>
            </div>
          )}
        </>
      )}
      {table.length > 0 && (
        <Figure title={<>Does “set is not a single label” flag foreign pictures? (<Term t="auroc">AUROC</Term>)</>} name={`ood-${L.ds}`} tour="table"
          table={{ cols: ["method", "ood", "mean_size", "empty_rate", "singleton_rate", "full_rate", "auroc"], rows: table.map((r) => [String(r.method), String(r.ood), Number(r.mean_size), Number(r.empty_rate), Number(r.singleton_rate), Number(r.full_rate), Number(r.auroc_not_singleton)]) }}
          note="From the experiments at α = 0.1, all methods. 50% = no better than chance.">
          <HBars label="AUROC per method" max={1} target={0.5} fmt={(v) => pct(v, 0)} items={table.map((r) => ({ name: `${mname(String(r.method))}${names.length > 1 ? ` · ${r.ood}` : ""}`, value: Number(r.auroc_not_singleton), color: Number(r.auroc_not_singleton) > 0.7 ? "var(--good)" : "var(--s2)" }))} />
        </Figure>
      )}
    </div>
  );
}
