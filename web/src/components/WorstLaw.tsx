"use client";
import { Shapes } from "lucide-react";
import { motion } from "motion/react";
import { useMemo, useState, type ReactNode } from "react";
import { pct, worstLaw } from "@/lib/conformal";
import { at, avg, DS_LABEL, loadResults, useLoad } from "@/lib/data";
import { Box, Chapter, LineChart, Reveal, Seg, StepSlider } from "./ui";

const KS = ["10", "100", "1000"];
const NS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000];

export default function WorstLaw() {
  const { data: R } = useLoad(loadResults);
  const [K, setK] = useState("100");
  const [ni, setNi] = useState(4);
  const n = NS[ni];
  const law = useMemo(() => worstLaw(+K, n, 0.1), [K, n]);
  const one = useMemo(() => worstLaw(1, n, 0.1), [n]);

  const table = (["mnist", "cifar10", "cifar10_r18", "cifar100", "imagenet"] as const).map((ds) => {
    const r = at(R, "worst_class", { dataset: ds, method: "mondrian", alpha: 0.1 });
    return { ds, K: r[0] ? Number(r[0].K) : NaN, n: avg(r, "n_cal_per_class"), pred: avg(r, "pred_mean"), meas: avg(r, "measured_min"), lo: avg(r, "pred_q10"), hi: avg(r, "pred_q90") };
  }).filter((r) => Number.isFinite(r.pred));

  const xlo = law ? Math.max(0, Math.min(law.q10 ?? 0.8, one?.q10 ?? 0.8) - 0.06) : 0.5;
  const clip = (pts: [number, number][]) => pts.filter((p) => p[0] >= xlo);
  const ymax = Math.max(...(law ? law.pts.map((p) => p[1]) : [1]), ...(one ? one.pts.map((p) => p[1]) : [1]));

  return (
    <Chapter id="predict" no={6} eyebrow="Our first result" title="We can predict the worst class before running anything"
      lead={<>Give every class its own cut-off (the “cut-off per class” rule). Each class’s coverage then wobbles randomly around 90%, like the histogram in chapter 4, and the classes wobble <strong>independently</strong>. The worst class is the unluckiest of K coin-flippers, and that has an exact formula.</>}>
      <Box kind="analogy" tag="Everyday analogy">
        <p>One student’s exam score varies a bit from day to day. In a class of 10, the lowest score is a little below average. In a school of 1,000
          students, the lowest score is <strong>far</strong> below average, even if nobody is a weak student. Pure chance makes the minimum drop as the
          group grows. Classes in a classifier behave exactly like that.</p>
      </Box>

      <Reveal>
        <div className="card stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <div className="row"><span className="small muted">Number of classes K:</span><Seg label="Classes" value={K} onChange={setK} options={KS.map((k) => [k, <span key={k} className="seg-ico"><Shapes aria-hidden />{Number(k).toLocaleString()} classes</span>]) as [string, ReactNode][]} /></div>
            <StepSlider label="Calibration pictures per class" index={ni} display={n.toLocaleString()} stops={NS.map((v) => (v >= 1000 ? `${v / 1000}k` : v))} onChange={setNi} />
          </div>
          {law && one ? (
            <>
              <LineChart label="Distribution of one class's coverage and of the worst class's coverage" xDom={[xlo, 1]} yDom={[0, ymax * 1.08]} yFmt={() => ""} xFmt={(v) => pct(v, 0)}
                shade={{ x0: xlo, x1: 0.9, color: "var(--bad)", label: "below the 90% promise" }}
                series={[{ pts: clip(one.pts), color: "var(--muted)", dots: false, dash: true, area: true, label: "one class" }, { pts: clip(law.pts), color: "var(--bad)", dots: false, area: true, label: "worst class", width: 3.5 }]}
                vlines={[{ x: 0.9, color: "var(--target)", label: "target 90%" }, { x: law.mean, color: "var(--bad)", label: `worst class ≈ ${pct(law.mean)} (${((law.mean - 0.9) * 100).toFixed(1)} pts)`, end: true }]} />
              <div className="legend"><span><i style={{ background: "var(--muted)" }} />one typical class (centred on 90%)</span><span><i style={{ background: "var(--bad)" }} />the worst of {K} classes</span></div>
              <div className="grid3">
                <div className="stat bad"><span className="k">Predicted worst class</span><span className="v">{pct(law.mean)}</span><span className="s">with {K} classes × {n.toLocaleString()} pictures each</span></div>
                <div className="stat" style={{ "--tone": "var(--violet)" } as React.CSSProperties}><span className="k">Chance some class is below 90%</span><span className="v">{pct(law.pBelow, 0)}</span><span className="s">true coverage, not just test noise</span></div>
                <div className="stat" style={{ "--tone": "var(--s1)" } as React.CSSProperties}><span className="k">Rule of thumb</span><span className="v" style={{ fontSize: "clamp(18px, 2vw, 22px)", lineHeight: 1.15 }}>more classes → more data</span><span className="s">every class within 2 points needs ≈1,300 / 2,400 / 3,400 pictures per class for 10 / 100 / 1,000 classes</span></div>
              </div>
            </>
          ) : <p className="muted">Too few pictures per class: the cut-off must include every label, so every set is the full list.</p>}
        </div>
      </Reveal>

      <Reveal>
        <div className="card stack">
          <h3>Did the formula work? Predicted vs measured on our real models</h3>
          {!R ? <p className="loading">loading…</p> : (() => {
            // main plot on the left (PW wide), zoom panel for the crowded 10-class runs on the right
            const lo0 = 0.38, hi0 = 0.95, PW = 600, W = 930, H = 500, m = { l: 52, r: 20, t: 18, b: 46 };
            const sx = (v: number) => Math.round((m.l + ((v - lo0) / (hi0 - lo0)) * (PW - m.l - m.r)) * 100) / 100;
            const sy = (v: number) => Math.round((H - m.b - ((v - lo0) / (hi0 - lo0)) * (H - m.t - m.b)) * 100) / 100;
            const LBL: Record<string, [number, number]> = { cifar100: [30, 6], imagenet: [30, 30] };
            const COL: Record<string, string> = { mnist: "var(--s1)", cifar10: "var(--s3)", cifar10_r18: "var(--s5)", cifar100: "var(--bad)", imagenet: "var(--s2)" };
            const rad = (K: number) => 9 + 4 * Math.log10(K);
            const zoomed = table.filter((r) => r.K === 10), zlo = 0.855, zhi = 0.882;
            const box = { x0: sx(0.855), x1: sx(0.896), y0: sy(0.9), y1: sy(0.842) }, pan = { x: PW + 40, y: m.t, w: W - PW - 44, h: H - m.t - m.b };
            const iy = (v: number) => Math.round((pan.y + pan.h - 34 - ((v - zlo) / (zhi - zlo)) * (pan.h - 80)) * 100) / 100;
            const SHORT: Record<string, string> = { mnist: "MNIST", cifar10: "CIFAR-10", cifar10_r18: "ResNet-18" };
            const zp = zoomed.length ? zoomed.reduce((a, r) => a + r.pred, 0) / zoomed.length : 0;
            const err = table.map((r) => Math.abs(r.meas - r.pred) * 100), hits = table.filter((r) => r.meas >= r.lo && r.meas <= r.hi).length;
            return (
              <div className="stack" style={{ gap: 18 }}>
                <div className="grid3">
                  <div className="stat good"><span className="k">Average miss</span><span className="v">{(err.reduce((a, b) => a + b, 0) / err.length).toFixed(1)} pts</span><span className="s">between predicted and measured worst class</span></div>
                  <div className="stat" style={{ "--tone": "var(--s1)" } as React.CSSProperties}><span className="k">Inside the forecast range</span><span className="v">{hits} of {table.length}</span><span className="s">measured value fell in the formula’s 10–90% range</span></div>
                </div>
                <svg className="chart" style={{ maxWidth: 980, margin: "0 auto", display: "block" }} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Predicted against measured worst-class coverage for each dataset; points on the diagonal mean the formula was exactly right">
                  {[0.4, 0.5, 0.6, 0.7, 0.8, 0.9].map((t) => (
                    <g key={t} className="grid"><line x1={m.l} x2={PW - m.r} y1={sy(t)} y2={sy(t)} /><line x1={sx(t)} x2={sx(t)} y1={m.t} y2={H - m.b} /></g>
                  ))}
                  {[0.4, 0.5, 0.6, 0.7, 0.8, 0.9].map((t) => <g key={`l${t}`}><text x={m.l - 8} y={sy(t) + 4} textAnchor="end">{pct(t, 0)}</text><text x={sx(t)} y={H - m.b + 18} textAnchor="middle">{pct(t, 0)}</text></g>)}
                  <text x={(m.l + PW - m.r) / 2} y={H - 6} textAnchor="middle" style={{ fontWeight: 600 }}>predicted by the formula →</text>
                  <text transform={`translate(14 ${(m.t + H - m.b) / 2}) rotate(-90)`} textAnchor="middle" style={{ fontWeight: 600 }}>measured on real models →</text>
                  <polygon points={`${sx(lo0)},${sy(lo0 + 0.03)} ${sx(hi0 - 0.03)},${sy(hi0)} ${sx(hi0)},${sy(hi0)} ${sx(hi0)},${sy(hi0 - 0.03)} ${sx(lo0 + 0.03)},${sy(lo0)} ${sx(lo0)},${sy(lo0)}`} fill="var(--good)" opacity={0.1} />
                  <motion.line x1={sx(lo0)} y1={sy(lo0)} x2={sx(hi0)} y2={sy(hi0)} stroke="var(--ink-2)" strokeWidth={1.5} strokeDasharray="6 5" initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }} viewport={{ once: true }} transition={{ duration: 1 }} />
                  <text transform={`translate(${sx(0.625)} ${sy(0.625) - 10}) rotate(${(Math.atan2(sy(0.9) - sy(0.4), sx(0.9) - sx(0.4)) * 180) / Math.PI})`} textAnchor="middle" style={{ fill: "var(--ink-2)" }}>on this line = formula exactly right</text>
                  {[...table].sort((a, b) => b.K - a.K).map((r, i) => {
                    const c = COL[r.ds] || "var(--s1)", rr = rad(r.K), [dx, dy] = LBL[r.ds] || [24, 0], x = sx(r.pred), y = sy(r.meas), e = (r.meas - r.pred) * 100;
                    return (
                      <motion.g key={r.ds} initial={{ y: -60, scale: 0.4 }} whileInView={{ y: 0, scale: 1 }} viewport={{ once: true }} transition={{ type: "spring", stiffness: 120, damping: 14, delay: 0.6 + i * 0.18 }}>
                        <title>{`${DS_LABEL[r.ds]}: predicted ${pct(r.pred)}, measured ${pct(r.meas)} (${e >= 0 ? "+" : "−"}${Math.abs(e).toFixed(1)} pts); forecast range ${pct(r.lo)}–${pct(r.hi)}`}</title>
                        <line x1={x} x2={x} y1={sy(r.lo)} y2={sy(r.hi)} stroke={c} strokeWidth={3} strokeLinecap="round" opacity={0.45} />
                        <circle cx={x} cy={y} r={rr} fill={c} fillOpacity={0.3} stroke={c} strokeWidth={2} />
                        <circle cx={x} cy={y} r={3} fill={c} />
                        {!zoomed.includes(r) && <>
                        <line x1={x + Math.sign(dx) * rr} y1={y + (dy * rr) / Math.max(rr, Math.hypot(dx, dy))} x2={x + dx * 0.8} y2={y + dy * 0.8} stroke={c} strokeWidth={1} opacity={0.6} />
                        <text x={x + dx} y={y + dy + 4} textAnchor={dx < 0 ? "end" : "start"} style={{ fill: c, fontWeight: 700 }}>{DS_LABEL[r.ds]}</text>
                        </>}
                        {r.ds === "cifar100" && <text x={x + dx} y={y + dy + 20} textAnchor="start" style={{ fill: "var(--bad)" }}>the one miss: {e.toFixed(1)} pts ↓</text>}
                      </motion.g>
                    );
                  })}
                  {zoomed.length > 0 && (
                    <g>
                      <rect x={box.x0} y={box.y0} width={box.x1 - box.x0} height={box.y1 - box.y0} rx={8} fill="none" stroke="var(--ink-2)" strokeWidth={1.2} strokeDasharray="4 3" />
                      <line x1={box.x1} y1={box.y0} x2={pan.x} y2={pan.y} stroke="var(--ink-2)" strokeWidth={0.8} strokeDasharray="3 4" opacity={0.6} />
                      <line x1={box.x1} y1={box.y1} x2={pan.x} y2={pan.y + pan.h} stroke="var(--ink-2)" strokeWidth={0.8} strokeDasharray="3 4" opacity={0.6} />
                      <rect x={pan.x} y={pan.y} width={pan.w} height={pan.h} rx={14} fill="var(--surface)" />
                      {[0.86, 0.87, 0.88].map((t) => <g key={t} className="grid"><line x1={pan.x + 46} x2={pan.x + pan.w - 12} y1={iy(t)} y2={iy(t)} /><text x={pan.x + 40} y={iy(t) + 4} textAnchor="end">{pct(t, 0)}</text></g>)}
                      <line x1={pan.x + 46} x2={pan.x + pan.w - 12} y1={iy(zp)} y2={iy(zp)} stroke="var(--ink-2)" strokeWidth={1.5} strokeDasharray="6 5" />
                      <text x={pan.x + pan.w - 12} y={iy(zp) - 8} textAnchor="end" style={{ fill: "var(--ink-2)" }}>formula: {pct(zp)}</text>
                      {zoomed.map((r, i) => {
                        const c = COL[r.ds] || "var(--s1)", rr = rad(r.K), y = iy(r.meas), x = pan.x + 70 + i * 34;
                        return (
                          <motion.g key={r.ds} initial={{ y: -40, scale: 0.4 }} whileInView={{ y: 0, scale: 1 }} viewport={{ once: true }} transition={{ type: "spring", stiffness: 120, damping: 14, delay: 1.4 + i * 0.15 }}>
                            <title>{`${DS_LABEL[r.ds]}: predicted ${pct(r.pred)}, measured ${pct(r.meas)}; forecast range ${pct(r.lo)}–${pct(r.hi)}`}</title>
                            <line x1={x} x2={x} y1={iy(Math.max(r.lo, zlo))} y2={iy(Math.min(r.hi, zhi))} stroke={c} strokeWidth={3} strokeLinecap="round" opacity={0.45} />
                            <circle cx={x} cy={y} r={rr} fill={c} fillOpacity={0.3} stroke={c} strokeWidth={2} />
                            <circle cx={x} cy={y} r={3} fill={c} />
                            <text x={x + rr + 6} y={y + 4} style={{ fill: c, fontWeight: 700, paintOrder: "stroke", stroke: "var(--surface)", strokeWidth: 4 }}>{SHORT[r.ds] || DS_LABEL[r.ds]} {pct(r.meas)}</text>
                          </motion.g>
                        );
                      })}
                      <text x={pan.x + 14} y={pan.y + pan.h - 12} style={{ fill: "var(--muted)" }}>measured worst class · 10 classes</text>
                    </g>
                  )}
                </svg>
                <div className="legend"><span><i style={{ background: "var(--ink-2)" }} />diagonal: prediction = measurement</span><span><i style={{ background: "var(--good-soft)" }} />within ±3 points</span><span><i style={{ background: "var(--muted)" }} />vertical whisker: formula’s 10–90% forecast</span><span>bigger bubble = more classes</span></div>
              </div>
            );
          })()}
        </div>
      </Reveal>

      <div className="stack" style={{ gap: 28 }}>
        <Box kind="keyidea" tag="Why this is useful">
          <p>The formula needs only <strong>how many pictures per class</strong> you have, not the pictures themselves. So before collecting any data you
            can say: “with 20 pictures per class and 1,000 classes, expect some class at about 49%.” On ImageNet we predicted 48.7% and measured 47.7%.</p>
        </Box>
        <Box kind="warnbox" tag="The one miss, and what it taught us">
          <p>On CIFAR-100 the prediction was 75% but we measured 70%. When we reshuffled the data randomly, the formula matched exactly. So the official
            CIFAR-100 test pictures are <strong>slightly different</strong> from the training pictures for some classes. The formula ended up working as a
            detector for that kind of hidden difference.</p>
        </Box>
      </div>
    </Chapter>
  );
}
