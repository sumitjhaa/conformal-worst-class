"use client";
import { useMemo } from "react";
import { betaPdf, kIndex, pct, resplitCoverages } from "@/lib/conformal";
import { LineChart, Stat, StepSlider } from "../../ui";
import { AlphaPick, DsPick, Figure, Loading, Term, useLab, useQuery } from "../kit";

const NCAL = [50, 100, 250, 500, 1000, 2500, 5000], REPS = [100, 300, 1000];

export default function Simulator({ p }: { p: string }) {
  const L = useLab(p, { m: "naive" }), d = L.d;
  const [ni, setNi] = useQuery<number>(`${p}n`, 2), [ri, setRi] = useQuery<number>(`${p}r`, 1), [seed, setSeed] = useQuery<number>(`${p}seed`, 1);
  const pool = d ? d.cal.n + d.test.n : 0, nCal = Math.min(NCAL[ni] ?? 250, Math.max(50, pool - 500)), reps = REPS[ri] ?? 300, a = L.alpha;

  const sim = useMemo(() => {
    if (!d) return null;
    const cov = resplitCoverages(d.cal, d.test, d.tuned.T, nCal, a, reps, seed), lo = Math.min(...cov), hi = Math.max(...cov);
    const mean = cov.reduce((x, y) => x + y, 0) / cov.length, sd = Math.sqrt(cov.reduce((x, y) => x + (y - mean) ** 2, 0) / cov.length);
    const k = kIndex(nCal, a), B = 30, x0 = Math.min(lo, 1 - a - 4 * sd), x1 = Math.min(1, Math.max(hi, 1 - a + 4 * sd)), w = (x1 - x0) / B, h = new Array(B).fill(0);
    cov.forEach((c) => h[Math.min(B - 1, Math.floor((c - x0) / w))]++);
    const hist = h.map((c, i) => [x0 + (i + 0.5) * w, c / (cov.length * w)] as [number, number]);
    const law = k > nCal ? [] : Array.from({ length: 120 }, (_, i) => { const x = x0 + ((x1 - x0) * i) / 119; return [x, betaPdf(x, k, nCal + 1 - k)] as [number, number]; });
    return { cov, mean, sd, below: cov.filter((c) => c < 1 - a).length / cov.length, hist, law, x0, x1, k };
  }, [d, nCal, a, reps, seed]);

  return (
    <div className="stack" style={{ gap: 28 }}>
      <div className="stack xcontrols" data-tour="controls">
        <div className="row" style={{ justifyContent: "space-between" }}><DsPick value={L.ds} onChange={L.setDs} /><AlphaPick value={a} onChange={L.setAlpha} /></div>
        <div className="row" style={{ gap: 28 }}>
          <StepSlider label="Calibration pictures per run" index={ni} onChange={setNi} display={String(nCal)} stops={NCAL.map(String)} />
          <StepSlider label="Reshuffles" index={ri} onChange={setRi} display={String(reps)} stops={REPS.map(String)} />
          <button type="button" className="btn primary" onClick={() => setSeed(seed + 1)}>Run again (new shuffles)</button>
        </div>
        <p className="small muted">Each run pools the {pool.toLocaleString()} calibration + test pictures, shuffles them, calibrates the simple rule on the first {nCal} and measures coverage on the rest.</p>
      </div>
      {!sim ? <Loading error={L.error} what={L.ds} /> : (
        <>
          <div className="grid4">
            <Stat k="Average coverage" value={sim.mean} format={(v) => pct(v, 2)} tone={sim.mean >= 1 - a - 0.003 ? "good" : "bad"} s={`guarantee: ${pct(1 - a, 1)} to ${pct(1 - a + 1 / (nCal + 1), 2)}`} />
            <Stat k="Spread (1 sd)" value={sim.sd} format={(v) => `±${(v * 100).toFixed(2)} pt`} s={`shrinks like 1/√n`} color="var(--s1)" />
            <Stat k="Runs under target" value={sim.below} format={(v) => pct(v, 0)} s="the promise is on average, not every run" color="var(--bad)" />
            <Stat k="Worst run" value={Math.min(...sim.cov)} format={(v) => pct(v)} s={`of ${reps} reshuffles`} color="var(--violet)" />
          </div>
          <Figure title="Coverage of every reshuffle" name={`simulator-${L.ds}-n${nCal}`} tour="hist"
            table={{ cols: ["run", "coverage"], rows: sim.cov.map((c, i) => [i + 1, c]) }}
            note={<>Curve: the exact <Term t="beta distribution">Beta</Term>(k, n + 1 − k) law with k = {sim.k}, n = {nCal}. The histogram is a little wider because each run’s test set is finite too.</>}>
            <LineChart label="Histogram of coverage over reshuffles with the Beta law" xDom={[sim.x0, sim.x1]} yDom={[0, Math.max(...sim.hist.map((h) => h[1]), ...sim.law.map((l) => l[1])) * 1.1]} xFmt={(v) => pct(v, 1)} yFmt={(v) => v.toFixed(0)}
              series={[{ pts: sim.hist, color: "var(--s1)", dots: false, area: true, width: 1.5 }, { pts: sim.law, color: "var(--violet)", dots: false }]}
              vlines={[{ x: 1 - a, color: "var(--target)", label: "1 − α" }, { x: Math.min(sim.x1, 1 - a + 1 / (nCal + 1)), color: "var(--target)", label: "+ 1/(n+1)", end: true }]} />
            <div className="legend"><span><i style={{ background: "var(--s1)" }} />reshuffles</span><span><i style={{ background: "var(--violet)" }} />exact law</span><span><i style={{ background: "var(--target)" }} />guarantee bounds on the average</span></div>
          </Figure>
        </>
      )}
    </div>
  );
}
