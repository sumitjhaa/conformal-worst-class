"use client";
import { motion } from "motion/react";
import { useState } from "react";
import { asset } from "@/lib/data";
import { ArrowRight, Check, Compass, Copy, FileText, Globe, Hash, Mail, School } from "./icons";
import { Tech, TECH } from "./tech";

const EMAIL = "sumitjha@iitgn.ac.in";
const PORTFOLIO = "https://sumitjha.vercel.app";

/** A tiny looping prediction set: chips join and leave as an imaginary cut-off moves. */
function SetLoop() {
  const labels = ["cat", "dog", "deer", "horse", "frog"];
  return (
    <svg viewBox="0 0 320 120" role="img" aria-label="Decorative: a prediction set growing and shrinking" style={{ width: "100%", maxWidth: 360, justifySelf: "end" }}>
      {labels.map((l, i) => (
        <motion.g key={l} animate={{ opacity: [0.25, 1, 1, 0.25], y: [6, 0, 0, 6] }} transition={{ duration: 4, repeat: Infinity, delay: i * 0.45, times: [0, 0.2, 0.7, 1] }}>
          <rect x={8 + i * 62} y={30} width={56} height={26} rx={7} fill={i === 0 ? "var(--accent)" : "var(--surface-2)"} />
          <text x={36 + i * 62} y={47} textAnchor="middle" style={{ font: "500 12px var(--mono)", fill: i === 0 ? "var(--on-accent)" : "var(--ink-2)" }}>{l}</text>
        </motion.g>
      ))}
      <text x={6} y={18} style={{ font: "500 11px var(--mono)", fill: "var(--muted)" }}>{"C(x) = { y : s(x, y) ≤ q̂ }"}</text>
      <line x1={8} x2={312} y1={84} y2={84} stroke="var(--line)" />
      <motion.line y1={74} y2={94} stroke="var(--target)" strokeWidth={2} animate={{ x1: [70, 300, 70], x2: [70, 300, 70] }} transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }} />
      <text x={8} y={110} style={{ font: "500 11px var(--mono)", fill: "var(--muted)" }}>moving q̂ → the set grows or shrinks</text>
    </svg>
  );
}

function CopyEmail() {
  const [done, setDone] = useState(false);
  const copy = async () => {
    try { await navigator.clipboard.writeText(EMAIL); setDone(true); setTimeout(() => setDone(false), 1600); } catch { location.href = `mailto:${EMAIL}`; }
  };
  return <button type="button" className="copy" onClick={copy} aria-live="polite">{done ? <><Check /> Copied</> : <><Copy /> Copy email</>}</button>;
}

export default function Footer() {
  const glow = (e: React.PointerEvent<HTMLElement>) => {
    const w = e.currentTarget.querySelector<HTMLElement>(".wordmark");
    w?.style.setProperty("--wx", `${e.clientX - w.getBoundingClientRect().left}px`);
  };
  return (
    <footer className="site-foot" onPointerMove={glow}>
      <div className="wrap">
        <div className="foot-cta">
          <div className="stack" style={{ gap: 14 }}>
            <h2>Who does a conformal guarantee <em>leave out?</em></h2>
            <p>The full story, with proofs, every table and every number, is in the papers. The explorer has every method and chart from the project.</p>
            <div className="row">
              <a className="btn primary" href={asset("papers/es678_conformal.pdf")} target="_blank" rel="noopener noreferrer"><FileText /> Course paper</a>
              <a className="btn" href={asset("papers/conformal_worst_class.pdf")} target="_blank" rel="noopener noreferrer">Extended paper <ArrowRight /></a>
            </div>
          </div>
          <SetLoop />
        </div>

        <div className="foot-grid">
          <div>
            <h4>About</h4>
            <p className="small">An interactive tour of split conformal prediction on MNIST, CIFAR-10, CIFAR-100 and ImageNet: the promise, the classes it leaves out, an exact law for the worst class, and a cheap fix.</p>
            <ul className="tech">{TECH.map(([t]) => <Tech key={t} name={t} />)}</ul>
          </div>
          <div>
            <h4>Read</h4>
            <ul>
              <li><a href={asset("papers/es678_conformal.pdf")} target="_blank" rel="noopener noreferrer"><FileText /> 4-page paper</a></li>
              <li><a href={asset("papers/conformal_worst_class.pdf")} target="_blank" rel="noopener noreferrer"><FileText /> Extended paper</a></li>
              <li><a href={asset("explorer/")}><Compass /> Explorers</a></li>
              <li><a href="#glossary"><Hash /> Glossary</a></li>
            </ul>
          </div>
          <div className="who">
            <h4>Author</h4>
            <div><div className="name">Sumit Kumar Jha</div><div className="roll">Roll no. 26210180</div></div>
            <a className="foot-link" href={`mailto:${EMAIL}`}><Mail /> {EMAIL}</a>
            <a className="foot-link" href={PORTFOLIO} target="_blank" rel="noreferrer"><Globe /> sumitjha.vercel.app</a>
            <span className="foot-link" style={{ cursor: "default" }}><School /> ES 678 · IIT Gandhinagar</span>
            <CopyEmail />
          </div>
        </div>

        <div className="wordmark" aria-hidden>conformal</div>

        <div className="foot-bottom">
          <span>© 2026 Sumit Kumar Jha · Live demos use exported outputs for 5,000 test pictures (3,000 for CIFAR-100). AI assistance (Claude) was used for code and writing.</span>
        </div>
      </div>
    </footer>
  );
}
