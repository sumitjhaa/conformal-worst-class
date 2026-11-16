"use client";
import { useMemo } from "react";
import { calibrate, metrics, pct, predictSets, reliability, type Method } from "@/lib/conformal";
import { LineChart, Slider, Stat } from "../../ui";
import { DsPick, Figure, Loading, Term, useLab, useQuery, VBars } from "../kit";

const TS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4];

export default function Calibration({ p }: { p: string }) {
  const L = useLab(p, { m: "naive" }), d = L.d;
  const [lt, setLt] = useQuery<number>(`${p}lt`, NaN); // log2 T; NaN = start at the tuned T
  const T = d ? (Number.isFinite(lt) ? 2 ** lt : d.tuned.T) : 1;

  const rel = useMemo(() => (d ? { now: reliability(d.test, T), raw: reliability(d.test, 1).ece, tuned: reliability(d.test, d.tuned.T).ece } : null), [d, T]);
  const sets = useMemo(() => {
    if (!d) return null;
    const run = (m: Method, t: number) => { const P = { method: m, alpha: 0.1, T: t, delta: 0.1 }; return metrics(d.test, predictSets(d.test, P, calibrate(d.cal, P))); };
    return { naive: run("naive", T), aps: run("aps", T), curve: TS.map((t) => ({ t, ...run("naive", t) })) };
  }, [d, T]);

  if (!d || !rel || !sets) return <div className="stack"><DsPick value={L.ds} onChange={L.setDs} /><Loading error={L.error} what={L.ds} /></div>;
  const r = rel.now, bins = r.count.map((c, j) => ({ c, acc: r.acc[j], conf: r.conf[j], mid: (j + 0.5) / r.count.length }));
  return (
    <div className="stack" style={{ gap: 28 }}>
      <div className="stack xcontrols" data-tour="temp">
        <DsPick value={L.ds} onChange={(v) => { L.setDs(v); setLt(NaN); }} />
        <Slider label="Temperature T" value={Math.log2(T)} display={`T = ${T.toFixed(2)}${Math.abs(T - d.tuned.T) < 0.01 ? " (tuned)" : ""}`} min={-2} max={2} step={0.02} marks={["0.25", "0.5", "1", "2", "4"]} onChange={setLt} />
        <div className="row"><button type="button" className="xbtn" onClick={() => setLt(0)}>T = 1 (raw model)</button><button type="button" className="xbtn" onClick={() => setLt(Math.log2(d.tuned.T))}>Tuned T = {d.tuned.T.toFixed(2)}</button></div>
        <p className="small"><Term t="temperature scaling">Temperature</Term> divides the model’s raw scores by T before turning them into probabilities. It never changes which label is most likely, only how sure the model claims to be.</p>
      </div>
      <div className="grid4">
        <Stat k="ECE at this T" value={r.ece} format={(v) => pct(v, 2)} tone={r.ece < 0.02 ? "good" : "bad"} s={<>raw {pct(rel.raw, 2)} · tuned {pct(rel.tuned, 2)}</>} />
        <Stat k="Coverage, simple rule" value={sets.naive.cov} format={(v) => pct(v)} tone={sets.naive.cov >= 0.89 ? "good" : "bad"} s="target 90%" />
        <Stat k="Set size, simple rule" value={sets.naive.mean} format={(v) => v.toFixed(3)} color="var(--s1)" s="labels per picture" />
        <Stat k="Set size, APS" value={sets.aps.mean} format={(v) => v.toFixed(3)} color="var(--violet)" s={`coverage ${pct(sets.aps.cov)}`} />
      </div>
      <div className="grid2">
        <Figure title="Reliability diagram" name={`reliability-${L.ds}-T${T.toFixed(2)}`} tour="rel"
          table={{ cols: ["bin", "count", "confidence", "accuracy"], rows: bins.map((b, j) => [j, b.c, b.conf, b.acc]) }}
          note={<>Each dot is a confidence bin: x = how sure the model says it is, y = how often it is right. On the dashed line = calibrated. <Term t="ece">ECE</Term> is the average gap.</>}>
          <LineChart label="Accuracy against confidence" xDom={[0, 1]} yDom={[0, 1]} xFmt={(v) => pct(v, 0)}
            series={[{ pts: [[0, 0], [1, 1]], color: "var(--target)", dash: true, dots: false }, { pts: bins.filter((b) => b.c >= 5).map((b) => [b.conf, b.acc]), color: "var(--accent)" }]} />
        </Figure>
        <Figure title="How many pictures per confidence bin" name={`confidence-hist-${L.ds}`}>
          <VBars label="Pictures per confidence bin" bars={bins.map((b) => ({ x: pct(b.mid, 0), v: b.c / d.test.n, color: "var(--s1)" }))} yDom={[0, Math.max(...bins.map((b) => b.c / d.test.n))]} yFmt={(v) => pct(v, 0)} />
        </Figure>
      </div>
      <Figure title="Coverage holds at every temperature; only the set size pays" name={`temp-sets-${L.ds}`} tour="sets"
        table={{ cols: ["T", "coverage", "mean_size"], rows: sets.curve.map((c) => [c.t, c.cov, c.mean]) }}
        note="Simple rule, α = 0.1, recalibrated at each T. Conformal prediction never trusts the probabilities, only their order on the calibration pictures.">
        <div className="grid2">
          <LineChart label="Coverage against temperature" xlog xDom={[0.25, 4]} xTicks={TS} yDom={[0.8, 1]} target={0.9} series={[{ pts: sets.curve.map((c) => [c.t, c.cov]), color: "var(--good)" }]} vlines={[{ x: T, color: "var(--accent)", label: "now" }]} />
          <LineChart label="Set size against temperature" xlog xDom={[0.25, 4]} xTicks={TS} yDom={[0, Math.max(2, ...sets.curve.map((c) => c.mean)) * 1.1]} yFmt={(v) => v.toFixed(1)} series={[{ pts: sets.curve.map((c) => [c.t, c.mean]), color: "var(--s1)" }]} vlines={[{ x: T, color: "var(--accent)" }]} />
        </div>
      </Figure>
    </div>
  );
}
