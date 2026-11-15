"use client";
import { useEffect, useMemo, useState } from "react";
import { pct } from "@/lib/conformal";
import { loadResults, rows, useLoad } from "@/lib/data";
import { Play, Pause } from "../../icons";
import { LineChart, Seg, Slider } from "../../ui";
import { dsName, Figure, Loading, Term, useQuery } from "../kit";

const LR_COLOR: Record<string, string> = { "0": "var(--bad)", "0.005": "var(--s1)", "0.02": "var(--good)" };
const lrName = (lr: number) => (lr === 0 ? "fixed (lr 0)" : `ACI lr ${lr}`);

export default function Aci({ p }: { p: string }) {
  const { data: R, error } = useLoad(loadResults);
  const [ds, setDs] = useQuery<string>(`${p}ds`, "cifar10");
  const [t, setT] = useState(Infinity), [play, setPlay] = useState(false);

  const S = useMemo(() => {
    const rs = rows(R, "aci_series").filter((r) => r.dataset === ds), lrs = [...new Set(rs.map((r) => Number(r.lr)))].sort((a, b) => a - b);
    const by = lrs.map((lr) => ({ lr, pts: rs.filter((r) => Number(r.lr) === lr).sort((a, b) => Number(a.t) - Number(b.t)) }));
    const first = by[0]?.pts ?? [], T = first.length ? Number(first[first.length - 1].t) : 1;
    const cuts = first.filter((r, i) => i > 0 && r.segment !== first[i - 1].segment).map((r) => ({ x: Number(r.t), label: String(r.segment) }));
    return { by, T, cuts, seg0: String(first[0]?.segment ?? "") };
  }, [R, ds]);
  const dsets = useMemo(() => [...new Set(rows(R, "aci_series").map((r) => String(r.dataset)))], [R]);

  useEffect(() => {
    if (!play) return;
    const id = setInterval(() => setT((v) => { const n = (Number.isFinite(v) ? v : 0) + S.T / 120; if (n >= S.T) { setPlay(false); return S.T; } return n; }), 50);
    return () => clearInterval(id);
  }, [play, S.T]);

  if (!R) return <Loading error={error} what="results" />;
  const now = Math.min(t, S.T), upto = (k: string) => S.by.map((s) => ({ lr: s.lr, pts: s.pts.filter((r) => Number(r.t) <= now && r[k] != null).map((r) => [Number(r.t), Number(r[k])] as [number, number]) }));
  const vl = S.cuts.map((c) => ({ x: c.x, color: "var(--muted)", label: c.label }));
  const summary = rows(R, "aci").filter((r) => r.dataset === ds);
  const chart = (k: string, title: string, yDom: [number, number], yFmt?: (v: number) => string, target?: number, tour?: string) => (
    <Figure title={title} name={`aci-${ds}-${k}`} tour={tour} table={{ cols: ["lr", "t", k], rows: S.by.flatMap((s) => s.pts.map((r) => [s.lr, Number(r.t), Number(r[k] ?? NaN)])) }}>
      <LineChart still label={title} xDom={[0, S.T]} yDom={yDom} yFmt={yFmt} xFmt={(v) => String(Math.round(v))} target={target} vlines={[...vl, { x: now, color: "var(--accent)" }]}
        series={upto(k).map((s) => ({ pts: s.pts, color: LR_COLOR[String(s.lr)] ?? "var(--violet)", dots: false, width: 2 }))} />
    </Figure>
  );
  return (
    <div className="stack" style={{ gap: 24 }}>
      <div className="stack xcontrols">
        <Seg label="Dataset" value={ds} onChange={(v) => { setDs(v); setT(Infinity); }} options={dsets.map((d) => [d, dsName(d)])} />
        <div className="row" data-tour="play">
          <button type="button" className="btn primary" onClick={() => { if (now >= S.T) setT(0); setPlay(!play); }}>{play ? <><Pause /> Pause</> : <><Play /> Play the stream</>}</button>
          <Slider label="Time step" value={now} display={`${Math.round(now)} / ${S.T}`} min={0} max={S.T} step={1} onChange={(v) => { setPlay(false); setT(v); }} />
        </div>
        <p className="small">A stream starting with “{S.seg0}” pictures, then segments of growing shift (grey lines). <Term t="aci">ACI</Term> updates α after every picture with learning rate lr.</p>
        <div className="legend">{S.by.map((s) => <span key={s.lr}><i style={{ background: LR_COLOR[String(s.lr)] ?? "var(--violet)" }} />{lrName(s.lr)}</span>)}</div>
      </div>
      {chart("rolling_cov", "Rolling coverage", [0.4, 1], undefined, 0.9, "cov")}
      <div className="grid2">
        {chart("alpha_t", "α_t used at each step", [0, 0.2], (v) => v.toFixed(2), undefined, "alpha")}
        {chart("size", "Set size", [0, Math.max(2, ...S.by.flatMap((s) => s.pts.map((r) => Number(r.size))))], (v) => v.toFixed(1))}
      </div>
      <details className="xdetails" open>
        <summary>Coverage per segment</summary>
        <div className="scroll-x"><table className="t"><thead><tr><th>Segment</th>{S.by.map((s) => <th key={s.lr} className="n">{lrName(s.lr)}</th>)}</tr></thead>
          <tbody>{[...new Set(summary.map((r) => String(r.segment)))].map((seg) => <tr key={seg}><th scope="row">{seg}</th>{S.by.map((s) => { const r = summary.find((x) => x.segment === seg && Number(x.lr) === s.lr); return <td key={s.lr} className="n">{r ? `${pct(Number(r.coverage))} · ${Number(r.mean_size).toFixed(2)}` : "—"}</td>; })}</tr>)}</tbody></table></div>
        <p className="small muted">Each cell: coverage · mean set size.</p>
      </details>
    </div>
  );
}
