"use client";
import { useMemo } from "react";
import { pct } from "@/lib/conformal";
import { loadResults, rows, useLoad } from "@/lib/data";
import { Seg } from "../../ui";
import { ALL_DS, downloadCsv, dsName, Loading, mname, Select, Term, useQuery } from "../kit";

const COLS: [string, string, (v: number) => string, boolean][] = [ // key, header, format, higher is better
  ["coverage", "Coverage", (v) => pct(v), true], ["mean_size", "Mean size", (v) => v.toFixed(2), false], ["singleton_rate", "Single label", (v) => pct(v), true],
  ["empty_rate", "Empty", (v) => pct(v), false], ["min_class_cov", "Worst class", (v) => pct(v), true], ["sscv", "SSCV", (v) => v.toFixed(3), false],
];
const sd = (v: number[]) => { const m = v.reduce((a, b) => a + b, 0) / v.length; return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, v.length - 1)); };

export default function Leaderboard({ p }: { p: string }) {
  const { data: R, error } = useLoad(loadResults);
  const [ds, setDs] = useQuery<string>(`${p}ds`, "cifar10"), [alpha, setAlpha] = useQuery<number>(`${p}a`, 0.1);
  const [sort, setSort] = useQuery<string>(`${p}sort`, "mean_size"), [dir, setDir] = useQuery<number>(`${p}dir`, 1);
  const [pair, setPair] = useQuery<string>(`${p}pair`, "");

  const table = useMemo(() => {
    const by = new Map<string, Record<string, number[]>>();
    for (const r of rows(R, "main")) {
      if (r.dataset !== ds || Math.abs(Number(r.alpha) - alpha) > 1e-9) continue;
      const m = String(r.method), e = by.get(m) ?? {}; by.set(m, e);
      for (const [k] of COLS) (e[k] ??= []).push(Number(r[k]));
    }
    const out = [...by].map(([m, e]) => ({ m, n: e.coverage.length, mean: Object.fromEntries(COLS.map(([k]) => [k, e[k].reduce((a, b) => a + b, 0) / e[k].length])), sd: Object.fromEntries(COLS.map(([k]) => [k, sd(e[k])])) }));
    return out.sort((a, b) => dir * (a.mean[sort] - b.mean[sort]));
  }, [R, ds, alpha, sort, dir]);

  const pairs = useMemo(() => [...new Set(rows(R, "bootstrap").filter((r) => r.dataset === ds).map((r) => `${r.a}|${r.b}`))], [R, ds]);
  const cur = pairs.includes(pair) ? pair : pairs[0] || "";
  const boot = rows(R, "bootstrap").filter((r) => r.dataset === ds && `${r.a}|${r.b}` === cur);
  const tuned = rows(R, "tuned").filter((r) => r.dataset === ds);

  if (!R) return <Loading error={error} what="results" />;
  const head = (k: string, label: string) => (
    <th key={k} scope="col" className="n" style={{ padding: 0 }} aria-sort={sort === k ? (dir > 0 ? "ascending" : "descending") : "none"}>
      <button type="button" className="xsort" onClick={() => (sort === k ? setDir(-dir) : (setSort(k), setDir(k === "mean_size" || k === "empty_rate" || k === "sscv" ? 1 : -1)))}>{label}{sort === k ? (dir > 0 ? " ↑" : " ↓") : ""}</button>
    </th>
  );
  const csv = { cols: ["method", "seeds", ...COLS.flatMap(([k]) => [k, `${k}_sd`])], rows: table.map((t) => [t.m, t.n, ...COLS.flatMap(([k]) => [t.mean[k], t.sd[k]])]) };
  return (
    <div className="stack" style={{ gap: 28 }}>
      <div className="stack xcontrols" data-tour="controls">
        <Seg label="Dataset" value={ds} onChange={setDs} options={ALL_DS.map((d) => [d, dsName(d)])} />
        <Seg label="α" value={String(alpha)} onChange={(v) => setAlpha(+v)} options={["0.01", "0.05", "0.1", "0.2"].map((a) => [a, `α = ${a}`])} />
      </div>
      <div data-tour="table" className="stack">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <h3>{table.length} methods · {dsName(ds)} · target {Math.round((1 - alpha) * 100)}%</h3>
          <button type="button" className="xbtn" onClick={() => downloadCsv(`leaderboard-${ds}-${alpha}`, csv)}>CSV</button>
        </div>
        <div className="scroll-x">
          <table className="t">
            <thead><tr><th scope="col">Method</th><th scope="col" className="n">Seeds</th>{COLS.map(([k, l]) => head(k, l))}</tr></thead>
            <tbody>
              {table.map((t) => (
                <tr key={t.m} className={t.mean.coverage < 1 - alpha - 0.005 ? "under" : ""}>
                  <th scope="row">{mname(t.m)}{t.mean.coverage < 1 - alpha - 0.005 && <span className="xflag">under target</span>}</th>
                  <td className="n">{t.n}</td>
                  {COLS.map(([k, , f]) => <td key={k} className="n">{f(t.mean[k])}<span className="xsd"> ±{k === "mean_size" || k === "sscv" ? t.sd[k].toFixed(2) : (t.sd[k] * 100).toFixed(1)}</span></td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="small muted">Means over seeds ± one standard deviation. <Term t="sscv">SSCV</Term>: size-stratified coverage violation, the largest coverage gap across set sizes (lower is better).</p>
      </div>
      <div className="stack" data-tour="pair">
        <h3>Is the difference real? <span className="small muted">· paired <Term t="bootstrap ci">bootstrap</Term> over seeds</span></h3>
        {pairs.length ? (
          <>
            <Select label="Compare" value={cur} onChange={setPair} options={pairs.map((x) => { const [a, b] = x.split("|"); return [x, `${mname(a)} − ${mname(b)}`]; })} />
            <div className="grid3">
              {boot.map((b) => {
                const real = Number(b.ci_lo) > 0 || Number(b.ci_hi) < 0, f = b.metric === "mean_size" ? (v: number) => v.toFixed(3) : (v: number) => `${(v * 100).toFixed(2)} pt`;
                return (
                  <div key={String(b.metric)} className="stat" style={{ "--tone": real ? "var(--good)" : "var(--muted)" } as React.CSSProperties}>
                    <span className="k">{b.metric === "mean_size" ? "Set size" : b.metric === "coverage" ? "Coverage" : "Worst class"} difference</span>
                    <span className="v">{f(Number(b.estimate))}</span>
                    <span className="s">95% CI [{f(Number(b.ci_lo))}, {f(Number(b.ci_hi))}] · {real ? "real difference" : "could be noise"} · {String(b.n_seeds)} seeds</span>
                  </div>
                );
              })}
            </div>
          </>
        ) : <p className="small muted">No bootstrap comparisons were run for this dataset.</p>}
      </div>
      {tuned.length > 0 && (
        <details className="xdetails">
          <summary>Tuned settings per seed ({tuned.length})</summary>
          <div className="scroll-x"><table className="t"><thead><tr>{["seed", "T", "ECE raw", "ECE after T", "RAPS", "clusters", "accuracy"].map((c) => <th key={c} className="n">{c}</th>)}</tr></thead>
            <tbody>{tuned.map((t) => <tr key={String(t.seed)}><td className="n">{String(t.seed)}</td><td className="n">{Number(t.T).toFixed(3)}</td><td className="n">{pct(Number(t.ece_raw), 2)}</td><td className="n">{pct(Number(t.ece_T), 2)}</td><td className="n">{String(t.raps)}</td><td className="n">{String(t.n_clusters)}</td><td className="n">{pct(Number(t.accuracy))}</td></tr>)}</tbody></table></div>
        </details>
      )}
    </div>
  );
}
