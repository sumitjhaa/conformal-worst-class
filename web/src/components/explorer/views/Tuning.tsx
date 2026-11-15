"use client";
import { useMemo } from "react";
import { pct } from "@/lib/conformal";
import { loadResults, rows, useLoad } from "@/lib/data";
import { LineChart, Seg } from "../../ui";
import { DataTable, dsName, Figure, Loading, mname, SERIES, Term, useQuery } from "../kit";

const GAMMAS = [0, 0.01, 0.1, 0.3, 1];
const avg = (v: number[]) => (v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN);

export default function Tuning({ p }: { p: string }) {
  const { data: R, error } = useLoad(loadResults);
  const [ds, setDs] = useQuery<string>(`${p}ds`, "cifar10"), [a, setA] = useQuery<number>(`${p}a`, 0.1);
  const dsets = useMemo(() => [...new Set(rows(R, "sweep").map((r) => String(r.dataset)))], [R]);

  const sweep = useMemo(() => {
    const rs = rows(R, "sweep").filter((r) => r.dataset === ds && Math.abs(Number(r.alpha) - a) < 1e-9);
    const line = (m: string, k: string) => GAMMAS.map((g, i) => [i, avg(rs.filter((r) => r.method === m && Number(r.gamma) === g).map((r) => Number(r[k])))] as [number, number]).filter((q) => Number.isFinite(q[1]));
    const base = (m: string, k: string) => avg(rs.filter((r) => r.method === m && String(r.gamma) === "None").map((r) => Number(r[k])));
    return { line, base, n: rs.length };
  }, [R, ds, a]);

  if (!R) return <Loading error={error} what="results" />;
  const chart = (k: string, title: string, yFmt?: (v: number) => string, target?: number) => {
    const ls = ["dist_naive", "dist_aps"].map((m, j) => ({ m, pts: sweep.line(m, k), color: SERIES[j] })), b = sweep.base("temp_aps", k);
    const all = [...ls.flatMap((l) => l.pts.map((q) => q[1])), b].filter(Number.isFinite), lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.2 || 0.05;
    return (
      <Figure title={title} name={`gamma-${ds}-${k}`} table={{ cols: ["method", "gamma", k], rows: ls.flatMap((l) => l.pts.map(([i, v]) => [l.m, GAMMAS[i], v])) }}>
        <LineChart label={title} xDom={[0, GAMMAS.length - 1]} xTicks={GAMMAS.map((_, i) => i)} xFmt={(i) => `γ ${GAMMAS[Math.round(i)]}`} yDom={[Math.max(0, lo - pad), hi + pad]} yFmt={yFmt} target={target}
          series={[...ls.map((l) => ({ pts: l.pts, color: l.color })), ...(Number.isFinite(b) ? [{ pts: [[0, b], [GAMMAS.length - 1, b]] as [number, number][], color: "var(--muted)", dash: true, dots: false }] : [])]} />
      </Figure>
    );
  };
  const diag = rows(R, "distance_diag").find((r) => r.dataset === ds);
  const empty = rows(R, "empty_sets").filter((r) => r.dataset === ds), em = [...new Set(empty.map((r) => String(r.method)))], alphas = [...new Set(empty.map((r) => Number(r.alpha)))].sort((x, y) => x - y);
  const tuned = [...new Set(rows(R, "tuned").map((r) => String(r.dataset)))].map((dd) => { const t = rows(R, "tuned").filter((r) => r.dataset === dd); return [dsName(dd), t.length, avg(t.map((r) => Number(r.T))), avg(t.map((r) => Number(r.ece_raw))), avg(t.map((r) => Number(r.ece_T))), avg(t.map((r) => Number(r.accuracy)))] as (string | number)[]; });
  return (
    <div className="stack" style={{ gap: 32 }}>
      <div className="stack xcontrols">
        <Seg label="Dataset" value={ds} onChange={setDs} options={dsets.map((d) => [d, dsName(d)])} />
        <Seg label="α" value={String(a)} onChange={(v) => setA(+v)} options={["0.01", "0.05", "0.1", "0.2"].map((x) => [x, `α = ${x}`])} />
      </div>
      <section className="stack" data-tour="gamma">
        <h2 className="xh2">The distance weight γ</h2>
        <p>The distance scores add γ × the <Term t="mahalanobis distance">Mahalanobis distance</Term> to the usual score: γ = 0 is the plain score. Grey dashed line: plain APS for reference. Means over seeds.</p>
        <div className="grid2">{chart("mean_size", "Set size", (v) => v.toFixed(2))}{chart("min_class_cov", "Worst class", undefined, 1 - a)}</div>
        <div className="legend"><span><i style={{ background: SERIES[0] }} />{mname("dist_naive")}</span><span><i style={{ background: SERIES[1] }} />{mname("dist_aps")}</span><span><i style={{ background: "var(--muted)" }} />APS</span></div>
      </section>
      {diag && (
        <section className="stack">
          <h2 className="xh2">Is distance a useful signal?</h2>
          <DataTable t={{ cols: ["check", "softmax", "distance"], rows: [["AUROC: true label vs others", pct(Number(diag.auroc_true_label_softmax)), pct(Number(diag.auroc_true_label_distance))], ["AUROC: 2nd label vs rest", pct(Number(diag.auroc_rank2_softmax)), pct(Number(diag.auroc_rank2_distance))], ["Spearman(softmax, distance)", Number(diag.spearman_softmax_distance).toFixed(3), ""], ["Tuned γ (simple / APS)", String(diag.gamma_naive), String(diag.gamma_aps)]] }} />
          <p className="small muted">Distance ranks labels worse than the softmax and mostly repeats it (strong negative correlation), which is why the tuned γ is small.</p>
        </section>
      )}
      <section className="stack" data-tour="empty">
        <h2 className="xh2">When do empty sets appear?</h2>
        <Figure title="Empty-set rate against α" name={`empty-${ds}`} table={{ cols: ["method", "alpha", "accuracy", "empty_rate", "bound", "bound_applies"], rows: empty.map((r) => [String(r.method), Number(r.alpha), Number(r.accuracy), Number(r.empty_rate), Number(r.bound), Number(r.bound_applies)]) }}
          note="Empty sets appear once the model is more accurate than 1 − α: the rule must miss some pictures on purpose to use up its allowed error. The bound is the guaranteed minimum where it applies.">
          <LineChart label="Empty-set rate against alpha" xDom={[alphas[0] ?? 0, alphas[alphas.length - 1] ?? 0.2]} xTicks={alphas} xFmt={(v) => v.toFixed(2)} yDom={[0, Math.max(0.05, ...empty.map((r) => Number(r.empty_rate))) * 1.15]} yFmt={(v) => pct(v, 0)}
            series={em.map((m, j) => ({ pts: empty.filter((r) => r.method === m).map((r) => [Number(r.alpha), Number(r.empty_rate)] as [number, number]).sort((x, y) => x[0] - y[0]), color: SERIES[j % SERIES.length] }))} />
          <div className="legend">{em.map((m, j) => <span key={m}><i style={{ background: SERIES[j % SERIES.length] }} />{mname(m)}</span>)}</div>
        </Figure>
      </section>
      <section className="stack">
        <h2 className="xh2">Tuned temperature, every dataset</h2>
        <DataTable t={{ cols: ["dataset", "seeds", "mean T", "ECE raw", "ECE after T", "accuracy"], rows: tuned }} />
        <p className="small muted">T and every other knob were tuned on the design split only, never on calibration or test pictures.</p>
      </section>
    </div>
  );
}
