"use client";
import { AnimatePresence, motion } from "motion/react";
import { useMemo, useState } from "react";
import { kIndex, probs } from "@/lib/conformal";
import { loadDataset, SHORT_NAMES, useLoad } from "@/lib/data";
import { Box, Chapter, Reveal, StepSlider } from "./ui";

const N = 40; // calibration dots shown; small so every dot is visible
const ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3];
const BIN = 0.05; // dot-histogram bin width on the score axis
const STEPS = [
  { t: "Collect fresh examples", d: "Take pictures the model never saw during training, where we know the right answer. This is the calibration set." },
  { t: "Score how surprised the model is", d: "For each picture, the surprise score is 1 − (probability the model gave to the right label). Sure and right → near 0. Fooled → near 1." },
  { t: "Line them up and draw a cut-off", d: "Sort the scores. Put a line so that 90% of the dots sit to its left. That line is the threshold q̂." },
  { t: "Answer a new picture", d: "For a new picture, compute the surprise score of every possible label. Every label left of the line goes into the set." },
];

export default function HowItWorks() {
  const { data: d } = useLoad(() => loadDataset("cifar10"));
  const [step, setStep] = useState(0);
  const [ai, setAi] = useState(1), alpha = ALPHAS[ai];
  const [pic, setPic] = useState(0);

  const demo = useMemo(() => {
    if (!d) return null;
    const p = probs(d.cal, d.tuned.T), K = d.K, y = d.cal.labels!;
    const scores = Array.from({ length: N }, (_, i) => 1 - p[i * K + y[i]]);
    const order = scores.map((s, i) => [s, i] as const).sort((a, b) => a[0] - b[0]);
    const pt = probs(d.test, d.tuned.T);
    // stack position of each dot in its score bin, two dots per row
    const fill: Record<number, number> = {}, slot = scores.map((s) => { const b = Math.min(Math.floor(s / BIN), 1 / BIN - 1); const j = (fill[b] = (fill[b] ?? -1) + 1); return { b, j }; });
    return { scores, sorted: order.map((o) => o[0]), pt, K, slot };
  }, [d]);

  const k = kIndex(N, alpha), q = demo ? (k > N ? 1 : demo.sorted[k - 1]) : 0.5;

  // step 4 cycles through test pictures; the first few are ones the model finds hard, so the cut-off matters
  const test = useMemo(() => {
    if (!d || !demo) return null;
    const hard = d.gallery.filter((g) => { const s = 1 - demo.pt[g.i * demo.K + d.test.labels![g.i]]; return s > 0.05; });
    const list = [...hard, ...d.gallery.filter((g) => !hard.includes(g))];
    const pick = list[pic % list.length];
    const labels = Array.from({ length: demo.K }, (_, j) => ({ j, s: 1 - demo.pt[pick.i * demo.K + j] })).sort((a, b) => a.s - b.s);
    return { img: pick.img, labels, truth: d.test.labels![pick.i] };
  }, [d, demo, pic]);

  const W = 480, H = 250, L = 22, R = 14, X = (s: number) => L + s * (W - L - R), base = H - 46, bw = (W - L - R) * BIN;
  const names = SHORT_NAMES.cifar10;
  const covered = demo ? demo.scores.filter((s) => s <= q).length : 0;

  return (
    <Chapter id="how" no={2} eyebrow="How it works" title="Four steps, no magic"
      lead={<>Below are <strong>real numbers</strong> from our CIFAR-10 model (it recognises 10 kinds of photo). Click through the steps.</>}>
      <div className="stack" style={{ gap: 16 }}>
        <div className="stepper" role="list">
          {STEPS.map((s, i) => (
            <button key={s.t} type="button" role="listitem" onClick={() => setStep(i)} aria-current={i === step ? "step" : undefined} className={`stepbtn ${i < step ? "done" : ""}`}
              style={{ opacity: i <= step ? 1 : 0.5, transition: "opacity .25s" }}>
              <span className="dot">{String(i + 1).padStart(2, "0")}</span>
              <span><span className="t">{s.t}</span><span className="d small">{s.d}</span></span>
            </button>
          ))}
        </div>
        <div className="row">
          <button className="btn" onClick={() => setStep((v) => Math.max(0, v - 1))} disabled={step === 0}>← Back</button>
          <button className="btn primary" onClick={() => setStep((v) => Math.min(3, v + 1))} disabled={step === 3}>Next step →</button>
        </div>
      </div>

      {!demo ? <p className="loading">loading the model’s outputs…</p> : (
        <div className="grid2 how-panels">
          <div className="card stack">
            <h3 className="panel-h"><span className="panel-no">A</span>Calibration pictures</h3>
                <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Calibration scores on a number line with the threshold">
                  {step >= 2 && <motion.rect y={10} height={base - 10} x={L} animate={{ width: Math.max(0, X(q) - L) }} fill="var(--s1)" opacity={0.08} rx={6} />}
                  <line className="axis" x1={L} x2={W - R} y1={base} y2={base} />
                  {[0, 0.25, 0.5, 0.75, 1].map((t) => <text key={t} x={X(t)} y={base + 18} textAnchor="middle">{t}</text>)}
                  <text x={W / 2} y={H - 4} textAnchor="middle">surprise score →</text>
                  {demo.scores.map((s, i) => {
                    const sorted = step >= 1, { b, j } = demo.slot[i];
                    const cx = sorted ? X(b * BIN) + bw / 2 + ((j % 2) - 0.5) * 9.6 : L + ((i + 0.5) / N) * (W - L - R);
                    const cy = sorted ? base - 7 - Math.floor(j / 2) * 9.4 : 40 + ((i * 7) % 6) * 24;
                    const left = step >= 2 && s <= q;
                    return <motion.circle key={i} r={sorted ? 4.6 : 6.2} initial={{ cx, cy: -10, opacity: 0 }} animate={{ cx, cy, opacity: 1, fill: step >= 2 ? (left ? "var(--s1)" : "var(--s2)") : "var(--muted)" }}
                      transition={{ delay: step === 0 ? i * 0.03 : 0, type: "spring", stiffness: 120, damping: 16 }} stroke="var(--surface)" strokeWidth={sorted ? 1.2 : 2} />;
                  })}
                  <AnimatePresence>
                    {step >= 2 && (
                      <motion.g key="q" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                        <motion.line animate={{ x1: X(q), x2: X(q) }} y1={10} y2={base} stroke="var(--target)" strokeWidth={3} strokeDasharray="7 5" />
                        <motion.text animate={{ x: X(q) + 8 }} y={24} style={{ fill: "var(--target)", fontWeight: 700, fontSize: 13 }}>q̂ = {q.toFixed(2)}</motion.text>
                        <motion.text animate={{ x: X(q) - 8 }} y={24} textAnchor="end" style={{ fill: "var(--s1)", fontWeight: 700, fontSize: 11 }}>{covered}/{N} dots</motion.text>
                      </motion.g>
                    )}
                  </AnimatePresence>
                </svg>
            <AnimatePresence mode="wait">
              <motion.div key={step} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                {step === 0 && <p>Each dot is one calibration picture ({N} shown here; the real run uses 5,000). The model has never trained on them.</p>}
                {step === 1 && <p>The dots now pile up by their surprise score, like a histogram. Most sit near 0: the model is usually sure and right. The few on the right are pictures that fooled it.</p>}
                {step >= 2 && <p><strong style={{ color: "var(--s1)" }}>{covered} of {N}</strong> dots are left of the line ({Math.round((covered / N) * 100)}%). The rule picks the {k}th smallest score, where {k} = ⌈({N}+1)×(1−α)⌉. Move α and watch the line.</p>}
              </motion.div>
            </AnimatePresence>
            {step >= 2 && <StepSlider label="α (allowed error)" index={ai} onChange={setAi} width={640} display={`${alpha.toFixed(2)} → aim for ${Math.round((1 - alpha) * 100)}% coverage`} stops={ALPHAS.map((v) => v.toFixed(2))} />}
          </div>

          <div className={`card stack ${step < 3 ? "panel-wait" : ""}`}>
            <h3 className="panel-h"><span className="panel-no">B</span>A new picture</h3>
            {step < 3 || !test ? (
              <p className="muted">Step 4 puts a new picture here and scores every label against the same cut-off q̂ from panel A.</p>
            ) : (
              <>
                <div className="row" style={{ alignItems: "center", flexWrap: "nowrap", gap: 14 }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={test.img} alt="new test picture" style={{ width: 72, height: 72, borderRadius: 10, imageRendering: "pixelated", flex: "none" }} />
                  <div className="stack" style={{ gap: 6 }}>
                    <div className="row" style={{ gap: 6 }}>
                      <span className="small muted">Set:</span>
                      {test.labels.filter((l) => l.s <= q).map((l) => <span key={l.j} className={`chip ${l.j === test.truth ? "true" : ""}`} style={{ fontSize: 13 }}>{names[l.j]}</span>)}
                      {test.labels.every((l) => l.s > q) && <span className="chip empty">empty set</span>}
                    </div>
                    <p className="small muted">True label: <strong>{names[test.truth]}</strong> (★ below).</p>
                  </div>
                  <button className="btn" style={{ marginLeft: "auto", flex: "none" }} onClick={() => setPic((v) => v + 1)}>Another picture →</button>
                </div>
                  <svg className="chart" viewBox={`0 0 ${W} ${test.labels.length * 22 + 6}`} role="img" aria-label="This picture's surprise score for every label, against the same cut-off">
                    
                    {test.labels.map((l, r) => {
                      const y = 3 + r * 22, inSet = l.s <= q;
                      return (
                        <motion.g key={`${pic}-${l.j}`} initial={{ opacity: 0.001, x: -12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: r * 0.04 }}>
                          <rect x={L} y={y} height={15} rx={4} width={Math.max(3, X(l.s) - L)} fill={inSet ? "var(--good-soft)" : "var(--surface-2)"} stroke={inSet ? "var(--good)" : "none"} strokeWidth={1.5} />
                          <text x={L + 6} y={y + 12} style={{ fill: inSet ? "var(--good)" : "var(--muted)", fontWeight: l.j === test.truth ? 800 : 500, paintOrder: "stroke", stroke: inSet ? "var(--surface)" : "var(--surface-2)", strokeWidth: 4 }}>
                            {names[l.j]}{l.j === test.truth ? " ★" : ""}{inSet ? " · in set" : ""}
                          </text>
                        </motion.g>
                      );
                    })}
                    <motion.line animate={{ x1: X(q), x2: X(q) }} y1={0} y2={test.labels.length * 22 + 2} stroke="var(--target)" strokeWidth={3} strokeDasharray="7 5" />
                  </svg>
                <p className="small">Each bar is this picture’s surprise score for one label, on the same axis as panel A. Green bars end left of q̂, so those labels go in the set. Lower α to push q̂ right and watch the set grow.</p>
              </>
            )}
          </div>
        </div>
      )}

      <Box kind="keyidea" tag="Why the 90% promise is true">
        <p>Imagine the new picture as the 41st dot. If it is “just another picture like the others” (statisticians say <strong>exchangeable</strong>),
          it is equally likely to land in any position in the sorted line. So the chance it lands left of the 37th dot is 37 out of 41, about 90%. That
          single counting argument is the whole guarantee, and it holds whatever the model is.</p>
      </Box>
      <Reveal>
        <p className="small muted">The set rule shown here is the simplest one (“Naive” in the paper). The project also tests smarter scores such as APS, which adds labels
          in order of probability, and many variants; the next chapter lets you try them.</p>
      </Reveal>
    </Chapter>
  );
}
