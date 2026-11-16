"use client";
import { animate, AnimatePresence, motion, MotionConfig, useInView, useMotionValue, useTransform } from "motion/react";
import Link from "next/link";
import { useEffect, useId, useRef, useSyncExternalStore, type ReactNode } from "react";
import { EXPLORERS } from "@/lib/explorers";

/** Respect the OS "reduce motion" setting for every motion animation on the page. */
export function MotionRoot({ children }: { children: ReactNode }) {
  return <MotionConfig reducedMotion="user">{children}</MotionConfig>;
}

const motionQuery = () => matchMedia("(prefers-reduced-motion: reduce)");
/** The OS "reduce motion" setting. The server can't know it, so the first (hydration) render always says false and the real value
 *  follows straight after; reading it during render directly made server and browser HTML differ. */
export function useStill() {
  return useSyncExternalStore((cb) => { const m = motionQuery(); m.addEventListener("change", cb); return () => m.removeEventListener("change", cb); }, () => motionQuery().matches, () => false);
}

/** Content is always rendered visible; supporting browsers animate it in with a CSS scroll timeline (see .reveal). */
export function Reveal({ children }: { children: ReactNode; delay?: number; y?: number }) {
  return <div className="reveal">{children}</div>;
}

export function Chapter({ id, no, eyebrow, title, lead, children }: { id: string; no: number; eyebrow: string; title: ReactNode; lead?: ReactNode; children: ReactNode }) {
  return (
    <section className="chapter" id={id} aria-labelledby={`${id}-h`}>
      <div className="wrap">
        <div className="chapter-head reveal">
          <div className="ch-meta"><span className="ch-no" aria-hidden>{String(no).padStart(2, "0")}</span><span className="kicker">{eyebrow}</span></div>
          <h2 id={`${id}-h`}>{title}</h2>
          {lead && <p className="lead">{lead}</p>}
          {EXPLORERS.some((e) => e.chapter === id) && (
            <p className="ch-explore">Open in explorer: {EXPLORERS.filter((e) => e.chapter === id).map((e) => <Link key={e.slug} href={`/explorer/${e.slug}/`}>{e.short}</Link>)}</p>
          )}
        </div>
        <div className="stack" style={{ gap: 28 }}>{children}</div>
      </div>
    </section>
  );
}

export function Box({ kind, tag, children }: { kind: "analogy" | "keyidea" | "warnbox"; tag: string; children: ReactNode }) {
  return <aside className={`note ${kind} reveal`}><span className="tag">{tag}</span><div>{children}</div></aside>;
}

export function Seg<T extends string>({ options, value, onChange, label }: { options: [T, ReactNode][]; value: T; onChange: (v: T) => void; label: string }) {
  const id = useId();
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map(([v, l]) => (
        <button key={v} type="button" aria-pressed={v === value} onClick={() => onChange(v)}>
          {v === value && <motion.span layoutId={id} className="pill" transition={{ type: "spring", stiffness: 520, damping: 40 }} />}
          {/* keyed on state: forces a fresh paint, browsers left the old tab's text colour stale */}
          <span key={String(v === value)}>{l}</span>
        </button>
      ))}
    </div>
  );
}

/** marks: labels drawn evenly from min to max under the track, so the scale is readable without dragging. */
export function Slider({ label, value, display, min, max, step, onChange, marks }: { label: string; value: number; display: string; min: number; max: number; step: number; onChange: (v: number) => void; marks?: ReactNode[] }) {
  return (
    <div className="slider">
      <label>{label} <b>{display}</b></label>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} aria-label={label} />
      {marks && marks.length > 1 && <div className="ticks" aria-hidden>{marks.map((t, i) => <span key={i} style={{ "--p": i / (marks.length - 1) } as React.CSSProperties}>{t}</span>)}</div>}
    </div>
  );
}

/** Slider over a few named stops, with clickable labels centred under each stop. */
export function StepSlider({ label, stops, index, display, onChange }: { label: string; stops: ReactNode[]; index: number; display: string; onChange: (i: number) => void }) {
  const last = stops.length - 1;
  return (
    <div className="stepslider">
      <Slider label={label} value={index} display={display} min={0} max={last} step={1} onChange={onChange} />
      <div className="ticks">{stops.map((t, i) => <button key={i} type="button" style={{ "--p": i / last } as React.CSSProperties} aria-pressed={i === index} onClick={() => onChange(i)}>{t}</button>)}</div>
    </div>
  );
}

/** A number that counts to its new value. format turns the animated value into text. */
export function Count({ value, format }: { value: number; format: (v: number) => string }) {
  const ref = useRef<HTMLSpanElement>(null), inView = useInView(ref, { once: true }), still = useStill();
  const mv = useMotionValue(0), text = useTransform(mv, (v) => (Number.isFinite(value) ? format(v) : "—"));
  useEffect(() => {
    if (!Number.isFinite(value)) return;
    if (still) { mv.set(value); return; }
    if (inView) { const c = animate(mv, value, { duration: 0.7, ease: [0.22, 1, 0.36, 1] }); return () => c.stop(); }
  }, [inView, value, mv, still]);
  return <motion.span ref={ref}>{text}</motion.span>;
}

export function Stat({ k, value, format, s, tone, color }: { k: string; value: number; format: (v: number) => string; s?: ReactNode; tone?: "good" | "bad"; color?: string }) {
  return (
    <div className={`stat ${tone || ""}`} style={color ? ({ "--tone": color } as React.CSSProperties) : undefined}>
      <span className="k">{k}</span>
      <span className="v"><Count value={value} format={format} /></span>
      {s && <span className="s">{s}</span>}
    </div>
  );
}

export function ImgCard({ img, labels, trueName, covered }: { img: string; labels: string[]; trueName: string; covered: boolean }) {
  return (
    <motion.div layout className={`imgcard ${covered ? "" : "miss"}`} title={`true label: ${trueName} · ${covered ? "inside the set" : "missed"}`}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={img} alt={trueName} />
      <div className="chips">
        <AnimatePresence initial={false} mode="popLayout">
          {labels.length === 0 && <motion.span key="empty" className="chip empty" initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.6, opacity: 0 }}>empty</motion.span>}
          {labels.slice(0, 4).map((l) => (
            <motion.span layout key={l} className={`chip ${l === trueName ? "true" : ""}`} initial={{ scale: 0.6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} exit={{ scale: 0.6, opacity: 0 }} transition={{ type: "spring", stiffness: 500, damping: 30 }}>{l}</motion.span>
          ))}
          {labels.length > 4 && <motion.span key="more" className="chip">+{labels.length - 4}</motion.span>}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}

// ---------------------------------------------------------------- tiny SVG charts
/** About n round tick values inside [a, b] (steps of 1, 2, 2.5 or 5 × 10^k), so axes read 70%, 80%… not 65%, 74%…. */
export function niceTicks(a: number, b: number, n = 5): number[] {
  if (!(b > a)) return [a];
  const raw = (b - a) / Math.max(1, n - 1), mag = 10 ** Math.floor(Math.log10(raw));
  // the round step whose tick count lands closest to n
  const count = (s: number) => Math.floor(b / s + 1e-9) - Math.ceil(a / s - 1e-9) + 1;
  const step = [0.5, 1, 2, 2.5, 5, 10].map((m) => m * mag).reduce((best, s) => (Math.abs(count(s) - n) < Math.abs(count(best) - n) ? s : best));
  const out: number[] = [];
  for (let v = Math.ceil(a / step - 1e-9) * step; v <= b + step * 1e-9; v += step) out.push(Math.round(v / step) * step);
  return out;
}

export const lin = (d0: number, d1: number, r0: number, r1: number) => (v: number) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
export const logs = (d0: number, d1: number, r0: number, r1: number) => { const a = Math.log(d0), b = Math.log(d1); return (v: number) => r0 + ((Math.log(Math.max(v, 1e-9)) - a) / (b - a || 1)) * (r1 - r0); };

export type Series = { pts: [number, number][]; color: string; dash?: boolean; dots?: boolean; label?: string; band?: [number, number, number][]; width?: number; area?: boolean; line?: boolean };
export function LineChart({ series, xDom, yDom, w = 560, h = 280, xTicks, xFmt = String, yFmt = (v) => `${Math.round(v * 100)}%`, target, xlog, vlines = [], shade, label, still }:
  { still?: boolean; series: Series[]; xDom: [number, number]; yDom: [number, number]; w?: number; h?: number; xTicks?: number[]; xFmt?: (v: number) => string; yFmt?: (v: number) => string; target?: number; xlog?: boolean; vlines?: { x: number; color: string; label?: string; end?: boolean }[]; shade?: { x0: number; x1: number; color: string; label?: string }; label: string }) {
  const m = { l: 48, r: 14, t: 14, b: 32 };
  // round to 0.01px: Node and the browser differ in the last float digits, which breaks hydration
  const sx = (xlog ? logs : lin)(xDom[0], xDom[1], m.l, w - m.r), sy = lin(yDom[0], yDom[1], h - m.b, m.t);
  const X = (v: number) => Math.round(sx(v) * 100) / 100, Y = (v: number) => Math.round(sy(v) * 100) / 100;
  const yt = niceTicks(yDom[0], yDom[1]);
  const xt = xTicks || (xlog ? [0, 1, 2, 3, 4].map((i) => Math.exp(Math.log(xDom[0]) + ((Math.log(xDom[1]) - Math.log(xDom[0])) * i) / 4)) : niceTicks(xDom[0], xDom[1]));
  const path = (pts: [number, number][]) => "M" + pts.map(([x, y]) => `${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join("L");
  return (
    <svg className="chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}>
      {shade && <g><rect x={X(shade.x0)} y={m.t} width={Math.max(0, X(shade.x1) - X(shade.x0))} height={h - m.b - m.t} fill={shade.color} opacity={0.09} />
        {shade.label && <text x={X(shade.x0) + 8} y={h - m.b - 10} style={{ fill: shade.color, fontWeight: 600 }}>{shade.label}</text>}</g>}
      <g className="grid">{yt.map((t) => <line key={t} x1={m.l} x2={w - m.r} y1={Y(t)} y2={Y(t)} />)}</g>
      {yt.map((t) => <text key={t} x={m.l - 6} y={Y(t) + 4} textAnchor="end">{yFmt(t)}</text>)}
      <line className="axis" x1={m.l} x2={w - m.r} y1={h - m.b} y2={h - m.b} />
      {xt.map((t) => <text key={t} x={X(t)} y={h - m.b + 17} textAnchor="middle">{xFmt(t)}</text>)}
      {target != null && <line x1={m.l} x2={w - m.r} y1={Y(target)} y2={Y(target)} stroke="var(--target)" strokeWidth={2} strokeDasharray="6 4" />}
      {vlines.map((v, i) => (
        <g key={i}><line x1={X(v.x)} x2={X(v.x)} y1={m.t} y2={h - m.b} stroke={v.color} strokeWidth={2} strokeDasharray="4 3" />
          {v.label && <text className="vlabel" x={X(v.x) + (v.end ? -5 : 5)} y={m.t + 12} textAnchor={v.end ? "end" : "start"} style={{ fill: v.color }}>{v.label}</text>}</g>
      ))}
      {series.map((s, i) => (
        <g key={i + (s.label || "")}>
          {s.area && s.pts.length > 1 && <motion.path key={`a${path(s.pts)}`} d={`${path(s.pts)}L${X(s.pts[s.pts.length - 1][0])},${Y(yDom[0])}L${X(s.pts[0][0])},${Y(yDom[0])}Z`} fill={s.color} initial={{ opacity: 0 }} animate={{ opacity: 0.16 }} transition={{ duration: 0.8 }} />}
          {s.band && <motion.path initial={{ opacity: 0 }} animate={{ opacity: 0.18 }} d={`M${s.band.map((b) => `${X(b[0])},${Y(b[2])}`).join("L")}L${[...s.band].reverse().map((b) => `${X(b[0])},${Y(b[1])}`).join("L")}Z`} fill={s.color} />}
          {s.pts.length > 1 && s.line !== false && <motion.path key={path(s.pts)} d={path(s.pts)} fill="none" stroke={s.color} strokeWidth={s.width || 2.5} strokeDasharray={s.dash ? "6 5" : undefined} strokeLinejoin="round"
            initial={still ? false : { pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: 0.9, ease: "easeInOut" }} />}
          {s.dots !== false && s.pts.map(([x, y], j) => <motion.circle key={j} cx={X(x)} cy={Y(y)} r={4.5} fill={s.color} stroke="var(--surface)" strokeWidth={2} initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: 0.2 + j * 0.05 }} />)}
        </g>
      ))}
    </svg>
  );
}

/** Horizontal bars with labels (HTML, so text stays readable at any width); each bar grows from 0. */
export function HBars({ items, max, fmt, target, label }: { items: { name: string; value: number; color: string }[]; max: number; fmt: (v: number) => string; target?: number; label: string; w?: number }) {
  const pc = (v: number) => `${Math.max(0, Math.min(1, v / max)) * 100}%`;
  return (
    <div role="list" aria-label={label} className="hbars">
      {items.map((it, i) => (
        <div role="listitem" key={`${i}-${it.name}`} className="hbar">
          <span className="hbar-name">{it.name}</span>
          <span className="hbar-track">
            <motion.span className="hbar-fill" style={{ background: it.color }} initial={{ width: 0 }} whileInView={{ width: pc(it.value) }} viewport={{ once: true }} transition={{ duration: 0.9, delay: i * 0.08, ease: [0.22, 1, 0.36, 1] }} />
            {target != null && <span className="hbar-target" style={{ left: pc(target) }} />}
          </span>
          <span className="hbar-val">{fmt(it.value)}</span>
        </div>
      ))}
    </div>
  );
}
