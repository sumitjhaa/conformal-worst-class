"use client";
import { useMemo } from "react";
import { pct, worstLaw } from "@/lib/conformal";
import { loadResults, rows, useLoad } from "@/lib/data";
import { LineChart, Seg, Stat, StepSlider } from "../../ui";
import { dsName, Figure, Loading, mname, Select, Term, useQuery } from "../kit";

const KS = [2, 5, 10, 20, 50, 100, 200, 500, 1000], NS = [10, 20, 30, 50, 100, 200, 500, 1000, 2000];

export default function WorstClass({ p }: { p: string }) {
  const { data: R, error } = useLoad(loadResults);
  const [ki, setKi] = useQuery<number>(`${p}K`, 4), [ni, setNi] = useQuery<number>(`${p}n`, 4), [la, setLa] = useQuery<number>(`${p}la`, 0.1);
  const [ds, setDs] = useQuery<string>(`${p}ds`, "cifar100"), [a, setA] = useQuery<number>(`${p}a`, 0.1), [m, setM] = useQuery<string>(`${p}m`, "mondrian");
  const K = KS[ki] ?? 50, n = NS[ni] ?? 100;
  const law = useMemo(() => worstLaw(K, n, la), [K, n, la]);

  const wc = useMemo(() => rows(R, "worst_class").filter((r) => r.dataset === ds && Math.abs(Number(r.alpha) - a) < 1e-9 && r.method === m).sort((x, y) => Number(x.seed) - Number(y.seed)), [R, ds, a, m]);
  const methods = useMemo(() => [...new Set(rows(R, "worst_class").filter((r) => r.dataset === ds).map((r) => String(r.method)))], [R, ds]);
  const dsets = useMemo(() => [...new Set(rows(R, "worst_class").map((r) => String(r.dataset)))], [R]);
  const nc = useMemo(() => rows(R, "ncal_sweep").filter((r) => r.dataset === ds).sort((x, y) => Number(x.n_c) - Number(y.n_c)), [R, ds]);
  const inBand = wc.filter((r) => Number(r.measured_min) >= Number(r.pred_q10) && Number(r.measured_min) <= Number(r.pred_q90)).length;

  return (
    <div className="stack" style={{ gap: 36 }}>
      <section className="stack" data-tour="law">
        <h2 className="xh2">The law, for any setting</h2>
        <p>With <b>K</b> classes and <b>n</b> calibration pictures per class, each class’s true coverage follows a <Term t="beta distribution">Beta</Term> law; the worst class is the lowest of K such draws. No test pictures needed.</p>
        <div className="row" style={{ gap: 28 }}>
          <StepSlider label="Classes K" index={ki} onChange={setKi} display={String(K)} stops={KS.map(String)} width={380} />
          <StepSlider label="Calibration pictures per class n" index={ni} onChange={setNi} display={String(n)} stops={NS.map(String)} width={380} />
          <Seg label="α" value={String(la)} onChange={(v) => setLa(+v)} options={[["0.05", "α 0.05"], ["0.1", "α 0.1"], ["0.2", "α 0.2"]]} />
        </div>
        {law ? (
          <>
            <div className="grid4">
              <Stat k="Expected worst class" value={law.mean} format={(v) => pct(v)} tone={law.mean < 1 - la - 0.02 ? "bad" : "good"} s={`target ${pct(1 - la, 0)}`} />
              <Stat k="80% range" value={law.q10} format={(v) => `${pct(v)}`} s={`to ${pct(law.q90)}`} color="var(--s1)" />
              <Stat k="Chance some class is under target" value={law.pBelow} format={(v) => pct(v)} color="var(--bad)" />
              <Stat k="Total calibration pictures" value={K * n} format={(v) => Math.round(v).toLocaleString()} color="var(--violet)" />
            </div>
            <Figure title="Where the worst class will land" name={`worst-law-K${K}-n${n}`} table={{ cols: ["coverage", "density"], rows: law.pts.map(([x, y]) => [x, y]) }}>
              <LineChart label="Density of the worst-class coverage" xDom={[law.pts[0][0], 1]} yDom={[0, Math.max(...law.pts.map((q) => q[1])) * 1.1]} yFmt={(v) => v.toFixed(0)} xFmt={(v) => pct(v, 0)}
                series={[{ pts: law.pts, color: "var(--accent)", dots: false, area: true }]} vlines={[{ x: 1 - la, color: "var(--target)", label: "target" }, { x: law.mean, color: "var(--bad)", label: "expected", end: true }]} />
            </Figure>
          </>
        ) : <p className="small muted">With n = {n}, α = {la} cannot be met: the set would have to include every label.</p>}
      </section>

      {!R ? <Loading error={error} what="results" /> : (
        <>
          <section className="stack" data-tour="seeds">
            <h2 className="xh2">Checked against every run</h2>
            <div className="row">
              <Seg label="Dataset" value={ds} onChange={setDs} options={dsets.map((d) => [d, dsName(d)])} />
              <Seg label="α" value={String(a)} onChange={(v) => setA(+v)} options={[["0.05", "α 0.05"], ["0.1", "α 0.1"]]} />
              <Select label="Rule" value={m} onChange={setM} options={methods.map((x) => [x, mname(x)])} />
            </div>
            {wc.length > 0 && (
              <Figure title={`Measured worst class vs the law’s 10–90% band · ${inBand} of ${wc.length} runs inside`} name={`worst-seeds-${ds}-${m}-${a}`}
                table={{ cols: ["seed", "measured_min", "pred_mean", "pred_q10", "pred_q90", "mean_size"], rows: wc.map((r) => [Number(r.seed), Number(r.measured_min), Number(r.pred_mean), Number(r.pred_q10), Number(r.pred_q90), Number(r.mean_size)]) }}>
                <LineChart label="Measured minimum class coverage per seed" xDom={[-0.5, wc.length - 0.5]} xTicks={wc.map((_, i) => i)} xFmt={(v) => `s${v}`}
                  yDom={[Math.floor(Math.min(...wc.map((r) => Math.min(Number(r.measured_min), Number(r.pred_q10)))) * 40) / 40, 1]} target={1 - a}
                  series={[{ pts: wc.map((r, i) => [i, Number(r.pred_mean)]), color: "var(--s1)", dash: true, dots: false, band: wc.map((r, i) => [i, Number(r.pred_q10), Number(r.pred_q90)]) }, { pts: wc.map((r, i) => [i, Number(r.measured_min)]), color: "var(--bad)", line: false }]} />
                <div className="legend"><span><i style={{ background: "var(--bad)" }} />measured worst class</span><span><i style={{ background: "var(--s1)" }} />law: mean and 10–90% band</span></div>
              </Figure>
            )}
          </section>
          {nc.length > 0 && (
            <section className="stack" data-tour="ncal">
              <Figure title="More calibration pictures per class lift the worst class" name={`ncal-${ds}`}
                table={{ cols: ["n_c", "measured_mean", "measured_q10", "measured_q90", "pred_mean", "pred_q10", "pred_q90"], rows: nc.map((r) => [Number(r.n_c), Number(r.measured_mean), Number(r.measured_q10), Number(r.measured_q90), Number(r.pred_mean), Number(r.pred_q10), Number(r.pred_q90)]) }}>
                <LineChart label="Worst class against calibration pictures per class" xlog xDom={[Number(nc[0].n_c), Number(nc[nc.length - 1].n_c)]} xTicks={nc.map((r) => Number(r.n_c))} yDom={[Math.floor(Number(nc[0].measured_q10) * 10) / 10, 1]} target={0.9}
                  series={[{ pts: nc.map((r) => [Number(r.n_c), Number(r.measured_mean)]), color: "var(--bad)", band: nc.map((r) => [Number(r.n_c), Number(r.measured_q10), Number(r.measured_q90)]) },
                    { pts: nc.map((r) => [Number(r.n_c), Number(r.pred_mean)]), color: "var(--s1)", dash: true, dots: false }]} />
                <div className="legend"><span><i style={{ background: "var(--bad)" }} />measured (10–90% band)</span><span><i style={{ background: "var(--s1)" }} />law</span></div>
              </Figure>
            </section>
          )}
          <details className="xdetails">
            <summary>Calibration pictures needed per class to keep the worst class within ε of target</summary>
            <div className="scroll-x"><table className="t"><thead><tr><th className="n">K</th><th className="n">ε</th><th className="n">n needed per class</th></tr></thead>
              <tbody>{rows(R, "n_needed").map((r, i) => <tr key={i}><td className="n">{String(r.K)}</td><td className="n">{String(r.eps)}</td><td className="n">{Number(r.n_needed).toLocaleString()}</td></tr>)}</tbody></table></div>
          </details>
        </>
      )}
    </div>
  );
}
