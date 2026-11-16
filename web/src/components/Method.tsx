"use client";
import { motion } from "motion/react";
import { GLOSSARY } from "@/lib/glossary";
import { BrainCircuit, Crosshair, Images, Sigma, Split, Thermometer } from "lucide-react";
import { Lock } from "./icons";
import { Tech } from "./tech";
import { Chapter, Reveal } from "./ui";

const SPLIT = [
  { name: "Train", pct: 80, color: "var(--s1)", d: "the model learns from these" },
  { name: "Design", pct: 10, color: "var(--s5)", d: "tune settings (temperature, groups)" },
  { name: "Calibrate", pct: 10, color: "var(--s2)", d: "set the cut-off q̂, nothing else" },
];
// pictures, split, train, temperature tuning, cut-off, maths
const STEP_ICONS = [Images, Split, BrainCircuit, Thermometer, Crosshair, Sigma];
const TOOLS: [string, string[]][] = [["Train", ["Python", "PyTorch"]], ["Analyse", ["NumPy", "SciPy", "pandas"]], ["Write", ["Matplotlib", "LaTeX"]], ["Web", ["TypeScript", "Next.js", "React 19", "Motion"]]];
const STEPS = [
  ["Get data", "MNIST digits, CIFAR-10 and CIFAR-100 photos, and published ImageNet model outputs. Plus corrupted versions to test shift."],
  ["Split it four ways", "Training / design / calibration pictures from the training set, and the official test set kept sealed. A check in the code fails if any picture appears twice."],
  ["Train models", "Small ResNets and a LeNet trained on a laptop GPU (Apple M5), 3–10 times each with different random seeds; a second, pre-trained ResNet-18 as a cross-check."],
  ["Tune on the design set", "Temperature scaling (fixes over-confidence), score settings and class groups, all chosen without touching calibration or test pictures."],
  ["Calibrate, then test once", "Compute the cut-off from calibration pictures only, then measure coverage, set size and worst class on the sealed test set."],
  ["Check the maths", "Compare every measurement with the exact formulas, re-split hundreds of times, and put confidence intervals on every difference."],
];
export function Method() {
  return (
    <Chapter id="method" no={11} eyebrow="How I did it" title="The whole pipeline, step by step"
      lead={<>Every number on this page and in the paper is produced by scripts and written into the paper automatically, so nothing is typed by hand.</>}>
      <Reveal>
        <div className="card stack">
          <h3>One rule above all: each picture has one job</h3>
          {/* treemap: tile area = share of the training pictures; the sealed test set sits apart */}
          <div className="split-map">
            {SPLIT.map((s, i) => (
              <motion.div key={s.name} className={`tile t-${s.name.toLowerCase()}`} style={{ "--c": s.color } as React.CSSProperties}
                initial={{ opacity: 0, scale: 0.94 }} whileInView={{ opacity: 1, scale: 1 }} viewport={{ once: true }} transition={{ duration: 0.5, delay: i * 0.15, ease: [0.22, 1, 0.36, 1] }}>
                <span className="pct">{s.pct}%</span><strong>{s.name}</strong><span className="d">{s.d}</span>
              </motion.div>
            ))}
            <motion.div className="tile t-test" initial={{ opacity: 0, x: 12 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ delay: 0.6 }}>
              <span className="pct"><Lock /></span><strong>Test</strong><span className="d">official test set, sealed until the end</span>
            </motion.div>
          </div>
          <p className="small" style={{ maxWidth: "none" }}>If calibration pictures were also used for tuning, the “just another picture” argument from chapter 2 would break, and so would the guarantee.</p>
        </div>
      </Reveal>
      <ol className="pipeline">
        <motion.span className="pipe-line" aria-hidden initial={{ clipPath: "inset(0 0 100% 0)" }} whileInView={{ clipPath: "inset(0 0 0% 0)" }} viewport={{ once: true }} transition={{ duration: 1.2, ease: "easeInOut" }} />
        {STEPS.map(([t, d], i) => {
          const Icon = STEP_ICONS[i];
          return (
            <motion.li key={t} className="step" initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.12 }}>
              <span className="no" aria-hidden>{String(i + 1).padStart(2, "0")}</span>
              <span className="node"><Icon strokeWidth={1.6} aria-hidden /></span>
              <div><h3>{t}</h3><p>{d}</p></div>
            </motion.li>
          );
        })}
      </ol>
      <Reveal>
        <div className="card">
          <h3>Tools</h3>
          <div className="tool-groups">
            {TOOLS.map(([g, names]) => (
              <div key={g}><span className="kicker">{g}</span><ul className="tech">{names.map((n) => <Tech key={n} name={n} />)}</ul></div>
            ))}
          </div>
          <p className="small" style={{ marginTop: 14 }}>Code, logs and checksums of every data file and trained model are in the repository.</p>
        </div>
      </Reveal>
    </Chapter>
  );
}

export function Glossary() {
  return (
    <Chapter id="glossary" no={12} eyebrow="Words used" title="Glossary">
      <Reveal>
        <table className="gl-table">
          <thead><tr><th scope="col">Word</th><th scope="col">Meaning</th></tr></thead>
          <tbody>{GLOSSARY.map(([t, d]) => <tr key={t}><th scope="row">{t}</th><td>{d}</td></tr>)}</tbody>
        </table>
      </Reveal>
    </Chapter>
  );
}
