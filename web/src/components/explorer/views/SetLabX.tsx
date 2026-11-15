"use client";
import Link from "next/link";
import { useMemo } from "react";
import { calibrate, metrics, pct, predictSets, setOf, type Method } from "@/lib/conformal";
import { ImgCard, LineChart, Seg, Slider, Stat } from "../../ui";
import { ALPHAS, AlphaPick, DsPick, Figure, LIVE_METHODS, Loading, params, Term, useLab, useQuery } from "../kit";

const SINGLE: Method[] = ["naive", "aps", "raps"]; // one shared cut-off, so it can be dragged by hand

export default function SetLabX({ p }: { p: string }) {
  const L = useLab(p), { d, out, alpha } = L;
  const [manual, setManual] = useQuery<number>(`${p}q`, -1);
  const canDrag = SINGLE.includes(L.method) && !!out?.cal?.q;

  // manual cut-off: same scores, user-chosen q
  const shown = useMemo(() => {
    if (!d || !out) return null;
    if (!canDrag || manual < 0) return { ...out, q: out.cal?.q ?? NaN };
    const cal = { q: manual }, C = predictSets(d.test, out.P, cal);
    return { ...out, cal, C, m: metrics(d.test, C), q: manual };
  }, [d, out, manual, canDrag]);

  const curve = useMemo(() => {
    if (!d) return null;
    return ALPHAS.map((a) => { const P = params(d, L.method, a), C = predictSets(d.test, P, calibrate(d.cal, P, d.clusters)); return { a, ...metrics(d.test, C) }; });
  }, [d, L.method]);

  const qMax = L.method === "naive" ? 1 : L.method === "raps" ? 2 : 1.05;
  return (
    <div className="stack" style={{ gap: 28 }}>
      <div className="stack xcontrols" data-tour="controls">
        <div className="row" style={{ justifyContent: "space-between" }}>
          <DsPick value={L.ds} onChange={(v) => { L.setDs(v); setManual(-1); }} />
          <AlphaPick value={alpha} onChange={(v) => { L.setAlpha(v); setManual(-1); }} />
        </div>
        <Seg label="Rule" value={L.method} onChange={(v) => { L.setMethod(v); setManual(-1); }} options={LIVE_METHODS} />
      </div>
      {!shown || !d ? <Loading error={L.error} what={`${L.ds} (CIFAR-100 is about 12 MB)`} /> : (
        <>
          <div className="grid4">
            <Stat k="Coverage" value={shown.m.cov} format={(v) => pct(v)} tone={shown.m.cov >= 1 - alpha - 0.01 ? "good" : "bad"} s={`target ${Math.round((1 - alpha) * 100)}%`} />
            <Stat k="Average set size" value={shown.m.mean} format={(v) => v.toFixed(2)} s="labels per picture" color="var(--s1)" />
            <Stat k="Single-label sets" value={shown.m.single} format={(v) => pct(v)} s="a normal, confident answer" color="var(--violet)" />
            <Stat k="Worst class" value={shown.m.worst} format={(v) => pct(v)} tone={shown.m.worst < 1 - alpha - 0.02 ? "bad" : "good"} s="lowest per-class coverage" />
          </div>
          <div data-tour="cutoff" className="stack">
            {canDrag ? (
              <>
                <Slider label={manual < 0 ? "Cut-off q̂ (calibrated)" : "Cut-off q̂ (your choice)"} value={shown.q} display={shown.q.toFixed(3)} min={0} max={qMax} step={0.001} onChange={setManual} />
                <p className="small">Labels whose <Term t="surprise (nonconformity) score">score</Term> is at or below the cut-off join the set. The calibrated value is <b className="mono">{out!.cal!.q!.toFixed(3)}</b>{manual >= 0 && <> · <button type="button" className="xbtn" onClick={() => setManual(-1)}>reset to calibrated</button></>}</p>
              </>
            ) : <p className="small muted">{L.method === "top1" ? "One guess has no cut-off: it always returns the top label." : "This rule has one cut-off per class or group, so it cannot be dragged as one number. Pick Simple, APS or RAPS to drag it."}</p>}
          </div>
          {curve && (
            <Figure title="Coverage and set size at every α" name={`set-lab-${L.ds}-${L.method}`} tour="curve"
              table={{ cols: ["alpha", "coverage", "mean_size", "worst_class"], rows: curve.map((c) => [c.a, c.cov, c.mean, c.worst]) }}>
              <div className="grid2">
                <LineChart label="Coverage against target" xDom={[0, 0.3]} yDom={[0.6, 1]} xTicks={[0.01, 0.1, 0.2, 0.3]} xFmt={(v) => v.toFixed(2)}
                  series={[{ pts: curve.map((c) => [c.a, 1 - c.a]), color: "var(--target)", dash: true, dots: false }, { pts: curve.map((c) => [c.a, c.cov]), color: "var(--good)" }, { pts: curve.map((c) => [c.a, c.worst]), color: "var(--bad)" }]}
                  vlines={[{ x: alpha, color: "var(--accent)", label: "now" }]} />
                <LineChart label="Average set size" xDom={[0, 0.3]} yDom={[0, Math.max(2, Math.ceil(Math.max(...curve.map((c) => c.mean))))]} xTicks={[0.01, 0.1, 0.2, 0.3]} xFmt={(v) => v.toFixed(2)} yFmt={(v) => v.toFixed(1)}
                  series={[{ pts: curve.map((c) => [c.a, c.mean]), color: "var(--s1)" }]} vlines={[{ x: alpha, color: "var(--accent)" }]} />
              </div>
              <div className="legend"><span><i style={{ background: "var(--good)" }} />coverage</span><span><i style={{ background: "var(--bad)" }} />worst class</span><span><i style={{ background: "var(--target)" }} />target 1 − α</span><span><i style={{ background: "var(--s1)" }} />set size</span></div>
            </Figure>
          )}
          <div data-tour="gallery">
            <h3 style={{ marginBottom: 12 }}>Test pictures <span className="small muted">· click one to inspect it</span></h3>
            <div className="gallery">
              {d.gallery.slice(0, 48).map((g) => {
                const y = d.test.labels![g.i], set = setOf(shown.C, g.i, d.K).sort((a, b) => Number(b === y) - Number(a === y));
                return (
                  <Link key={g.i} href={`/explorer/inspector/?ds=${L.ds}&m=${L.method}&a=${alpha}&i=${g.i}`} className="xcard-link" aria-label={`Inspect picture of ${shown.names[y]}`}>
                    <ImgCard img={g.img} trueName={shown.names[y]} covered={set.includes(y)} labels={set.map((k) => shown.names[k])} />
                  </Link>
                );
              })}
            </div>
          </div>
          <p className="small muted">Computed live from {d.cal.n.toLocaleString()} calibration and {d.test.n.toLocaleString()} test pictures.</p>
        </>
      )}
    </div>
  );
}
