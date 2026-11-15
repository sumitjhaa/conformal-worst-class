"use client";
import { AnimatePresence, motion, useReducedMotion } from "motion/react";
import { useEffect, useState } from "react";
import { asset, SHORT_NAMES } from "@/lib/data";
import { ArrowRight, Compass, FileText } from "./icons";
import { Count } from "./ui";

// Three illustrative pictures with made-up but realistic probabilities over the 10 CIFAR-10 classes.
// Every demo further down uses the real model outputs; this one only shows the idea.
const NAMES = SHORT_NAMES.cifar10;
const CUT = 0.15; // labels whose probability clears this line join the set
const EXAMPLES = [
  { name: "a clear truck", truth: 9, p: [0.005, 0.04, 0.002, 0.003, 0.002, 0.003, 0.002, 0.003, 0.01, 0.93], note: "Easy picture: one bar clears the line, so the set is just “truck”, the same as a normal answer." },
  { name: "a blurry cat", truth: 3, p: [0.01, 0.01, 0.03, 0.38, 0.06, 0.46, 0.02, 0.02, 0.005, 0.005], note: "The single guess says “dog” and is wrong. The set says “dog or cat”, and the truth is inside." },
  { name: "a dark deer", truth: 4, p: [0.02, 0.01, 0.06, 0.08, 0.27, 0.18, 0.03, 0.34, 0.005, 0.005], note: "Very unsure, so three labels clear the line. The size of the set is how the model admits doubt." },
];

const STATS: [number, string, string][] = [[5, "", "settings, digits to ImageNet"], [13, "", "conformal methods audited"], [1000, "", "classes in the largest test"], [10, "×", "seeds per setting, at most"]];

export default function Hero() {
  const [i, setI] = useState(0);
  const [paused, setPaused] = useState(false);
  const still = useReducedMotion();
  useEffect(() => {
    if (paused || still) return;
    const t = setInterval(() => setI((v) => (v + 1) % EXAMPLES.length), 4800);
    return () => clearInterval(t);
  }, [paused, still]);
  const ex = EXAMPLES[i], guess = ex.p.indexOf(Math.max(...ex.p)), inSet = ex.p.map((v) => v >= CUT);

  return (
    <header className="hero-wrap">
      <div className="aurora" aria-hidden><i /><i /><i /></div>
      <div className="wrap">
      <div className="hero">
        <div className="stack" style={{ gap: 22 }}>
          <h1>
            Teaching a classifier to say{" "}
            <span className="scribble serif-i">
              “I’m not sure”
              <svg viewBox="0 0 200 12" preserveAspectRatio="none" aria-hidden>
                <path d="M2 8 C 40 2, 80 11, 120 6 S 180 3, 198 7" pathLength={1} fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
              </svg>
            </span>
            , with a promise you can check.
          </h1>
          <p className="lead">
            A normal classifier gives one answer, even when it is guessing. This project makes it give a <strong>short list of labels</strong> that
            contains the right one <strong>90% of the time</strong>, proves the promise holds, then asks the harder question: <strong>is it
            fair to every kind of picture?</strong>
          </p>
          <div className="row">
            <a className="btn primary" href="#idea">Start reading <ArrowRight /></a>
            <a className="btn" href={asset("papers/es678_conformal.pdf")} target="_blank" rel="noopener noreferrer"><FileText /> Paper</a>
            <a className="btn" href={asset("explorer/")}><Compass /> Explorer</a>
          </div>
          <p className="byline"><span>Sumit Kumar Jha</span><span>26210180</span><span>ES 678 · IIT Gandhinagar</span><span>Nov 2026</span></p>
        </div>

        <figure className="hero-fig" style={{ margin: 0 }} onMouseEnter={() => setPaused(true)} onMouseLeave={() => setPaused(false)} onFocus={() => setPaused(true)} onBlur={() => setPaused(false)}>
          <div className="row" style={{ justifyContent: "space-between", marginBottom: 18 }}>
            <div className="seg" role="group" aria-label="Example picture">
              {EXAMPLES.map((e, j) => (
                <button key={e.name} type="button" aria-pressed={i === j} onClick={() => setI(j)}>
                  {i === j && <motion.span layoutId="hero-pill" className="pill" transition={{ type: "spring", stiffness: 520, damping: 40 }} />}
                  {/* keyed on state: forces a fresh paint, browsers left the old tab's text colour stale */}
                  <span key={String(i === j)}>{e.name}</span>
                </button>
              ))}
            </div>
          </div>
          <div className="hbars-v" role="img" aria-label={`Model probabilities for ${ex.name}. Labels above the cut-off: ${NAMES.filter((_, k) => inSet[k]).join(", ")}.`}>
            {ex.p.map((v, k) => (
              <motion.div key={k} className="bar" initial={false} animate={{ height: `${Math.max(v, 0.008) * 100}%`, backgroundColor: inSet[k] ? "var(--accent)" : "var(--surface-2)" }}
                transition={{ type: "spring", stiffness: 140, damping: 20, delay: k * 0.02 }} />
            ))}
            <div className="cut" style={{ bottom: `${CUT * 100}%` }}><span>cut-off</span></div>
          </div>
          <div className="hbars-v-lbl" aria-hidden>
            {NAMES.map((n, k) => <span key={n} className={inSet[k] ? "in" : ""} style={k === ex.truth ? { textDecoration: "underline", textDecorationColor: "var(--accent)", textDecorationThickness: 2, textUnderlineOffset: 3 } : undefined}>{n}</span>)}
          </div>
          <div className="grid2" style={{ marginTop: 18, gap: 14 }}>
            <div>
              <div className="small muted">Usual answer</div>
              <AnimatePresence mode="wait" initial={false}>
                <motion.div key={i} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} className="set-read">
                  {NAMES[guess]} <span style={{ color: guess === ex.truth ? "var(--good)" : "var(--bad)", fontSize: 13, fontWeight: 500 }}>{guess === ex.truth ? "right" : "wrong"}</span>
                </motion.div>
              </AnimatePresence>
            </div>
            <div>
              <div className="small muted">Honest set</div>
              <AnimatePresence mode="wait" initial={false}>
                <motion.div key={i} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} className="set-read">
                  {"{ "}{NAMES.filter((_, k) => inSet[k]).join(", ")}{" }"}
                </motion.div>
              </AnimatePresence>
            </div>
          </div>
          <figcaption aria-live="polite">{ex.note} <span className="muted">Illustration; the true label is underlined.</span></figcaption>
        </figure>
      </div>
      <div className="statstrip reveal">
        {STATS.map(([v, suffix, k]) => (
          <div key={k}><div className="v"><Count value={v} format={(x) => Math.round(x).toLocaleString("en-US")} />{suffix && <small>{suffix}</small>}</div><div className="k">{k}</div></div>
        ))}
      </div>
      </div>
    </header>
  );
}
