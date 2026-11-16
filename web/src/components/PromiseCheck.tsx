"use client";
import { motion, useInView } from "motion/react";
import { useMemo, useRef, useState } from "react";
import { betaPdf, kIndex, pct, resplitCoverages } from "@/lib/conformal";
import { loadDataset, useLoad } from "@/lib/data";
import Gif from "./Gif";
import { Box, Chapter, Reveal, Seg } from "./ui";

const ALPHA = 0.1, REPS = 300, BINS = 36;

export default function PromiseCheck() {
  const { data: d } = useLoad(() => loadDataset("cifar10"));
  const [n, setN] = useState("20");
  const [run, setRun] = useState(1); // start pre-run with 20 calibration pictures
  const cardRef = useRef<HTMLDivElement>(null), inView = useInView(cardRef, { once: true }); // card is always mounted; the chart waits for data
  const nCal = +n;

  const res = useMemo(() => {
    if (!d || run === 0) return null;
    const cov = resplitCoverages(d.cal, d.test, d.tuned.T, nCal, ALPHA, REPS, run * 101 + nCal);
    const lo = Math.max(0.6, Math.min(...cov) - 0.01), hi = Math.min(1, Math.max(...cov) + 0.01);
    const counts = new Array(BINS).fill(0);
    cov.forEach((c) => counts[Math.min(BINS - 1, Math.floor(((c - lo) / (hi - lo)) * BINS))]++);
    const below = cov.filter((c) => c < 1 - ALPHA).length / cov.length;
    const avg = cov.reduce((a, b) => a + b, 0) / cov.length;
    const sorted = [...cov].sort((a, b) => a - b), q = (p: number) => sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))];
    const mine = (run * 7919 + nCal * 31) % cov.length; // "your deployed model": one run, picked fresh each time
    return { counts, lo, hi, below, avg, cov, q10: q(0.1), q90: q(0.9), mine };
  }, [d, run, nCal]);

  const W = 640, H = 280, m = { l: 40, r: 14, t: 16, b: 34 };
  const k = kIndex(nCal, ALPHA);

  return (
    <Chapter id="promise" no={4} eyebrow="Checking the promise" title="Is the 90% promise real? Let’s test it 300 times"
      lead={<>A promise is only worth something if it holds up. I re-shuffle which pictures are used for calibration and which for testing, rebuild the sets, and measure coverage again. And again. 300 times.</>}>
      <Gif name="resplit" flip fig="Fig. 4a · 160 re-splits" title="One promise, many calibration draws" alt="Animation: a histogram of coverage over 160 random calibration re-splits builds up and matches the exact Beta-law curve centred near 90 percent.">
        <p>Every re-split draws a fresh calibration set of 200 pictures and measures coverage on the rest. The bars pile up into the exact shape the maths predicts (orange curve): 90% on average, with a known spread around it.</p>
      </Gif>
      <Reveal>
        <div ref={cardRef} className="card stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <div className="row">
              <span className="small muted">Calibration pictures:</span>
              <Seg label="Calibration size" value={n} onChange={(v) => { setN(v); if (run) setRun((r) => r + 1); }} options={[["20", "20"], ["100", "100"], ["1000", "1,000"], ["5000", "5,000"]]} />
            </div>
            <button className="btn primary" onClick={() => setRun((r) => r + 1)}>{run ? "Run 300 more" : "Run the experiment 300 times"}</button>
          </div>
          {!d && <p className="loading">loading…</p>}
          {d && !res && <p className="muted" style={{ padding: "60px 0", textAlign: "center" }}>Press the button. Each run is a full conformal prediction from scratch.</p>}
          {res && (
            <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Histogram of coverage over 300 re-splits">
              {(() => {
                const X = (v: number) => m.l + ((v - res.lo) / (res.hi - res.lo)) * (W - m.l - m.r);
                const maxC = Math.max(...res.counts), bw = (W - m.l - m.r) / BINS, plotH = H - m.t - m.b;
                // each run is one dot, stacked in its bin; dot size chosen so the tallest stack matches the curve's peak
                const r = Math.min(bw / 2 - 0.5, (0.92 * plotH) / (2 * maxC)), seen = new Array(BINS).fill(0);
                const dots = res.cov.map((c) => { const b = Math.min(BINS - 1, Math.floor(((c - res.lo) / (res.hi - res.lo)) * BINS)); return { b, j: seen[b]++, below: c < 1 - ALPHA }; });
                // theory: Beta(k, n+1-k) density for the coverage of one calibration draw (test-set noise adds a little spread)
                const a = k, b = nCal + 1 - k, pts: string[] = [];
                let peak = 0; const xs = Array.from({ length: 160 }, (_, i) => res.lo + ((res.hi - res.lo) * i) / 159);
                const dens = xs.map((x) => { const v = k > nCal ? 0 : betaPdf(x, a, b); peak = Math.max(peak, v); return v; });
                xs.forEach((x, i) => pts.push(`${X(x).toFixed(1)},${(H - m.b - (dens[i] / (peak || 1)) * (H - m.t - m.b) * 0.92).toFixed(1)}`));
                return (
                  <>
                    <line className="axis" x1={m.l} x2={W - m.r} y1={H - m.b} y2={H - m.b} />
                    {[0, 0.25, 0.5, 0.75, 1].map((t) => { const v = res.lo + t * (res.hi - res.lo); return <text key={t} x={X(v)} y={H - m.b + 17} textAnchor="middle">{pct(v, 0)}</text>; })}
                    {dots.map((d, i) => (
                      <motion.circle key={`${run}-${i}`} cx={m.l + (d.b + 0.5) * bw} r={r} fill={d.below ? "var(--s2)" : "var(--s1)"}
                        initial={{ cy: m.t, scale: 0 }} animate={inView ? { cy: H - m.b - r - d.j * 2 * r, scale: 1 } : undefined} transition={{ delay: i * 0.0008, type: "spring", stiffness: 520, damping: 34 }} />
                    ))}
                    {(() => {
                      const d = dots[res.mine], cx = m.l + (d.b + 0.5) * bw, cy = H - m.b - r - d.j * 2 * r, c = d.below ? "var(--s2)" : "var(--s1)";
                      return (
                        <motion.g key={`mine-${run}`} initial={{ scale: 0 }} animate={inView ? { scale: 1 } : undefined} transition={{ delay: 0.45, type: "spring", stiffness: 320, damping: 20 }} style={{ transformOrigin: `${cx}px ${cy}px` }}>
                          <circle cx={cx} cy={cy} r={r + 2} fill={c} stroke="var(--bad)" strokeWidth={2.5} />
                          <motion.circle cx={cx} cy={cy} r={r + 2} fill="none" stroke="var(--bad)" strokeWidth={2} animate={{ scale: [1, 2.6], opacity: [0.8, 0] }} transition={{ duration: 1.4, repeat: Infinity, ease: "easeOut" }} style={{ transformOrigin: `${cx}px ${cy}px` }} />
                        </motion.g>
                      );
                    })()}
                    {k <= nCal && <motion.polyline key={`th-${run}`} points={pts.join(" ")} fill="none" stroke="var(--ink)" strokeWidth={2} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ delay: 0.1, duration: 0.5 }} />}
                    <line x1={X(1 - ALPHA)} x2={X(1 - ALPHA)} y1={m.t} y2={H - m.b} stroke="var(--target)" strokeWidth={2} strokeDasharray="6 4" />
                    <text x={X(1 - ALPHA) + 5} y={m.t + 10} style={{ fill: "var(--target)" }}>90% target</text>
                  </>
                );
              })()}
            </svg>
          )}
          {res && (
            <div className="grid4">
              <div className={`stat ${res.avg >= 0.895 ? "good" : "bad"}`}><span className="k">Average coverage over 300 runs</span><span className="v">{pct(res.avg)}</span><span className="s">the promise: at least 90%</span></div>
              <div className="stat" style={{ "--tone": "var(--s2)" } as React.CSSProperties}><span className="k">Runs below 90%</span><span className="v">{pct(res.below, 0)}</span><span className="s">orange dots</span></div>
              <div className="stat" style={{ "--tone": "var(--violet)" } as React.CSSProperties}><span className="k">Middle 80% of runs</span><span className="v">{pct(res.q10, 0)}–{pct(res.q90, 0)}</span><span className="s">how far one run can wander; shrinks with more pictures</span></div>
              <div className="stat" style={{ "--tone": "var(--ink-2)" } as React.CSSProperties}><span className="k">The curve</span><span className="v" style={{ fontSize: "clamp(24px, 3vw, 30px)" }}>exact theory</span><span className="s">Beta({k}, {nCal + 1 - k}) law, drawn before the data</span></div>
            </div>
          )}
          <div className="legend"><span><i style={{ background: "var(--s1)", borderRadius: "50%" }} />one run at or above 90%</span><span><i style={{ background: "var(--s2)", borderRadius: "50%" }} />one run below 90%</span><span><i style={{ background: "var(--ink)", height: 2 }} />theory curve</span></div>
        </div>
      </Reveal>

      <div className="stack" style={{ gap: 28 }}>
        <Box kind="keyidea" tag="What you should see">
          <p>The dots pile up around 90%, right where the theory curve says. <strong>On average</strong> the promise holds. But individual runs land
            above or below it. With only 20 or 100 calibration pictures the spread is wide; with 5,000 it is tight.</p>
        </Box>
        <Box kind="warnbox" tag="The fine print">
          <p>“90% guaranteed” is a promise <strong>on average over many possible calibration sets</strong>. Your one deployed model is one dot of
            this histogram. That small detail matters a lot in the next chapter, where it gets much bigger.</p>
        </Box>
      </div>
    </Chapter>
  );
}
