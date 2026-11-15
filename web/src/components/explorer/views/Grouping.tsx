"use client";
import { useMemo } from "react";
import { pct } from "@/lib/conformal";
import { loadResults, rows, useLoad } from "@/lib/data";
import { LineChart, Seg } from "../../ui";
import { dsName, Figure, Loading, Term, useQuery } from "../kit";

const LINES: [string, string, string][] = [["clustered", "Clustered", "var(--s1)"], ["clustered_pac", "Clustered-PAC", "var(--good)"]];

export default function Grouping({ p }: { p: string }) {
  const { data: R, error } = useLoad(loadResults);
  const [ds, setDs] = useQuery<string>(`${p}ds`, "cifar100");
  const all = useMemo(() => rows(R, "clusters").filter((r) => r.dataset === ds).sort((a, b) => Number(a.M) - Number(b.M)), [R, ds]);
  const dsets = useMemo(() => [...new Set(rows(R, "clusters").map((r) => String(r.dataset)))], [R]);
  if (!R) return <Loading error={error} what="results" />;
  const K = Number(all[0]?.K ?? 10), Ms = [...new Set(all.map((r) => Number(r.M)))];
  const line = (m: string, k: string) => all.filter((r) => r.method === m).map((r) => [Number(r.M), Number(r[k])] as [number, number]);
  const chart = (k: string, title: string, yDom: [number, number], yFmt?: (v: number) => string, target?: number) => (
    <Figure title={title} name={`grouping-${ds}-${k}`} table={{ cols: ["method", "M", k], rows: all.map((r) => [String(r.method), Number(r.M), Number(r[k])]) }}>
      <LineChart label={title} xlog xDom={[1, Math.max(2, K)]} xTicks={Ms} yDom={yDom} yFmt={yFmt} target={target} series={LINES.map(([m, , c]) => ({ pts: line(m, k), color: c }))} />
    </Figure>
  );
  const lo = Math.floor(Math.min(...all.map((r) => Number(r.min_class_cov))) * 20) / 20;
  return (
    <div className="stack" style={{ gap: 28 }}>
      <div className="stack xcontrols">
        <Seg label="Dataset" value={ds} onChange={setDs} options={dsets.map((d) => [d, dsName(d)])} />
        <p className="small">M = number of class groups. M = 1 is one shared cut-off (the plain rule); M = {K} is one cut-off per class (<Term t="mondrian">Mondrian</Term>). <Term t="pac">PAC</Term> adds a safety margin so each group meets 90% with high probability. α = 0.1.</p>
        <div className="legend">{LINES.map(([, l, c]) => <span key={l}><i style={{ background: c }} />{l}</span>)}<span><i style={{ background: "var(--target)" }} />target 90%</span></div>
      </div>
      <div className="grid2" data-tour="chart">
        {chart("min_class_cov", "Worst class", [lo, 1], undefined, 0.9)}
        {chart("mean_size", "Average set size", [0, Math.ceil(Math.max(...all.map((r) => Number(r.mean_size))) * 1.1 * 10) / 10], (v) => v.toFixed(1))}
        {chart("q05_class_cov", "5th-percentile class", [lo, 1], undefined, 0.9)}
        {chart("coverage", "Overall coverage", [0.85, 1], undefined, 0.9)}
      </div>
      <p className="small muted">Read across: more groups lift the worst class but cost labels; the PAC margin lifts it further for a few more labels. Best trade-off on {dsName(ds)}: {(() => { const pac = all.filter((r) => r.method === "clustered_pac" && Number(r.min_class_cov) >= 0.88).sort((a, b) => Number(a.mean_size) - Number(b.mean_size))[0]; return pac ? `Clustered-PAC with M = ${pac.M} keeps every class at ${pct(Number(pac.min_class_cov))} or better with ${Number(pac.mean_size).toFixed(2)} labels per set.` : "no grouping keeps every class within 2 pt of target here."; })()}</p>
    </div>
  );
}
