"use client";
import { motion } from "motion/react";
import { useState } from "react";
import { pct } from "@/lib/conformal";
import { at, avg, loadResults, SHORT_NAMES, useLoad } from "@/lib/data";
import { Box, Chapter, Reveal, Seg } from "./ui";
import Gif from "./Gif";

const RULES: [string, string][] = [["temp_naive", "Simple sets"], ["temp_aps", "Adaptive sets"], ["mondrian", "Cut-off per class"], ["clustered_pac", "My fix"]];

export default function Fairness() {
  const { data: R } = useLoad(loadResults);
  const [rule, setRule] = useState("temp_naive");
  const cc = at(R, "class_cov", { dataset: "cifar10", method: rule }).sort((a, b) => Number(a.cls) - Number(b.cls));
  const vals = cc.map((r) => Number(r.coverage));
  const overall = avg(at(R, "main", { dataset: "cifar10", method: rule, alpha: 0.1 }), "coverage");
  const names = SHORT_NAMES.cifar10;
  const W = 640, H = 270, m = { l: 44, r: 10, t: 14, b: 30 }, lo = 0.7, hi = 1;
  const Y = (v: number) => H - m.b - ((Math.max(lo, v) - lo) / (hi - lo)) * (H - m.t - m.b), bw = (W - m.l - m.r) / 10;

  const worstRows = (["mnist", "cifar10", "cifar100", "imagenet"] as const).map((ds) => ({ ds, K: { mnist: 10, cifar10: 10, cifar100: 100, imagenet: 1000 }[ds], rows: at(R, "main", { dataset: ds, method: "temp_naive", alpha: 0.1 }) }));

  return (
    <Chapter id="fair" no={5} eyebrow="The catch" title="90% on average… but not for everyone"
      lead={<>The promise is about <strong>all pictures mixed together</strong>. It says nothing about each kind of picture. Some classes can be covered far less than 90% while others are covered more, and the average still looks perfect.</>}>
      <Box kind="analogy" tag="Everyday analogy">
        <p>A school says “our students average 90% attendance”. That can be true even if one class section attends only 70% and another 100%. The
          average hides who is left out. My project measures the section with the worst attendance: the <strong>worst-covered class</strong>.</p>
      </Box>

      <Gif name="worst" fig="Fig. 5a · per-class coverage" title="Some class is always left behind" alt="Animation: per-class coverage bars for the ten CIFAR-10 classes change with every random calibration draw; the average stays near 90 percent while the worst class, often cat, falls well below.">
        <p>36 random calibration draws, one per frame. The average stays near 90%, but the red bar, the worst class that draw, sits well below the dashed target, and it is usually <strong>cat</strong>. That gap is what chapters 6 and 7 predict and fix.</p>
      </Gif>

      <Reveal>
        <div className="card stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h3>Coverage of each kind of photo (CIFAR-10, averaged over 10 trained models)</h3>
            <Seg label="Rule" value={rule} onChange={setRule} options={RULES} />
          </div>
          {!R ? <p className="loading">loading results…</p> : (
            <svg className="chart" style={{ maxWidth: 720, margin: "0 auto" }} viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Per-class coverage">
              {[0.7, 0.8, 0.9, 1].map((t) => <g key={t} className="grid"><line x1={m.l} x2={W - m.r} y1={Y(t)} y2={Y(t)} /><text x={m.l - 6} y={Y(t) + 4} textAnchor="end">{pct(t, 0)}</text></g>)}
              {vals.map((v, i) => {
                const bad = v < 0.88;
                return (
                  <g key={i}>
                    <motion.rect x={m.l + i * bw + 6} width={bw - 12} rx={6} fill={bad ? "var(--bad)" : "var(--s1)"} style={{ transition: "fill .3s" }} animate={{ y: Y(v), height: H - m.b - Y(v) }} initial={false} transition={{ type: "spring", stiffness: 120, damping: 18, delay: i * 0.03 }} />
                    <motion.text x={m.l + (i + 0.5) * bw} animate={{ y: Y(v) - 6 }} initial={false} textAnchor="middle" style={{ fill: bad ? "var(--bad)" : "var(--ink)", fontWeight: 600 }}>{Math.round(v * 100)}</motion.text>
                    <text x={m.l + (i + 0.5) * bw} y={H - m.b + 17} textAnchor="middle">{names[i]}</text>
                  </g>
                );
              })}
              <line x1={m.l} x2={W - m.r} y1={Y(0.9)} y2={Y(0.9)} stroke="var(--target)" strokeWidth={2} strokeDasharray="6 4" />
            </svg>
          )}
          <p className="small">Overall coverage with this rule: <strong>{pct(overall)}</strong>. Red bars: classes more than 2 points below the 90% target.
            {rule === "temp_naive" && <> With <strong>Simple sets</strong>, cats are covered only about {pct(Math.min(...vals), 0)} of the time, while the best class gets {pct(Math.max(...vals), 0)}.</>}
            {rule === "temp_aps" && <> Adaptive sets spread the coverage more evenly, at the cost of slightly bigger sets.</>}
            {rule === "mondrian" && <> One cut-off per class targets 90% for every class, but each class only gets about 500 calibration pictures, so some still land a bit low by chance.</>}
            {rule === "clustered_pac" && <> My fix adds a safety margin, so classes land at or above target.</>}
          </p>
        </div>
      </Reveal>

      <Reveal>
        <div className="card stack">
          <h3>It gets much worse with more classes</h3>
          <p className="small">Every dot is one class (Simple sets, averaged over models). All four rows average about 90%, but the more classes there are, the wider they spread, and the further the unluckiest class falls.</p>
          {R && <ClassSpread R={R} />}
          <p className="small muted">ImageNet: 1,000 classes, about 20 calibration pictures each. In a single run its worst class is covered only about {pct(avg(worstRows[3].rows, "min_class_cov"), 0)} of the time.</p>
        </div>
      </Reveal>

      <Box kind="keyidea" tag="The research question">
        <p>So the real question of this project is not “does the 90% promise hold?” (it does, on average) but: <strong>which class gets the worst
          deal, can I predict how bad it will be before running anything, and can I fix it cheaply?</strong></p>
      </Box>
    </Chapter>
  );
}

const SPREAD: [string, string][] = [["mnist", "Digits"], ["cifar10", "Photos"], ["cifar100", "CIFAR-100"], ["imagenet", "ImageNet"]];

/** One row per dataset, one dot per class: same ~90% average, wider spread as the class count grows. */
function ClassSpread({ R }: { R: NonNullable<Parameters<typeof at>[0]> }) {
  const W = 900, rowH = 74, m = { l: 128, r: 24, t: 26, b: 52 }, lo = 0.35, hi = 1;
  const rows = SPREAD.map(([ds, name]) => ({ ds, name, v: at(R, "class_cov", { dataset: ds, method: "temp_naive" }).map((r) => Number(r.coverage)) })).filter((r) => r.v.length);
  const H = m.t + rows.length * rowH + m.b;
  const X = (v: number) => Math.round((m.l + ((Math.max(lo, v) - lo) / (hi - lo)) * (W - m.l - m.r)) * 100) / 100;
  // deterministic vertical jitter so dense rows read as a cloud, not a line
  const jit = (i: number) => (((i * 2654435761) >>> 0) % 1000) / 1000 - 0.5;
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Per-class coverage for each dataset: the average stays near 90% while the spread and the lowest class get worse as the number of classes grows">
      {[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1].map((t) => <g key={t} className="grid"><line x1={X(t)} x2={X(t)} y1={m.t - 8} y2={H - m.b} /><text x={X(t)} y={H - m.b + 18} textAnchor="middle">{pct(t, 0)}</text></g>)}
      <line x1={X(0.9)} x2={X(0.9)} y1={m.t - 14} y2={H - m.b} stroke="var(--target)" strokeWidth={2} strokeDasharray="6 4" />
      <text x={X(0.9)} y={m.t - 16} textAnchor="middle" style={{ fill: "var(--target)", fontWeight: 600 }}>90% promise</text>
      {rows.map((r, ri) => {
        const cy = m.t + ri * rowH + rowH / 2, n = r.v.length, rad = n > 500 ? 2.6 : n > 50 ? 4 : 6.5, spread = n > 50 ? rowH - 22 : 10;
        const mean = r.v.reduce((a, b) => a + b, 0) / n, wi = r.v.indexOf(Math.min(...r.v)), w = r.v[wi];
        return (
          <motion.g key={r.ds} initial={{ scaleY: 0.1 }} whileInView={{ scaleY: 1 }} viewport={{ once: true }} transition={{ type: "spring", stiffness: 110, damping: 16, delay: ri * 0.15 }} style={{ transformOrigin: `0px ${cy}px` }}>
            <rect x={m.l - 6} y={cy - rowH / 2 + 4} width={W - m.l - m.r + 12} height={rowH - 8} rx={10} fill="var(--surface-2)" opacity={0.35} />
            <text x={m.l - 14} y={cy - 2} textAnchor="end" style={{ fill: "var(--ink)", fontWeight: 700, fontSize: 12.5 }}>{r.name}</text>
            <text x={m.l - 14} y={cy + 14} textAnchor="end">{n.toLocaleString()} classes</text>
            {r.v.map((v, i) => i !== wi && (
              <circle key={i} cx={X(v)} cy={cy + jit(i + ri * 7919) * spread} r={rad} fill={v < 0.85 ? "var(--bad)" : "var(--s1)"} fillOpacity={n > 500 ? 0.5 : 0.65} />
            ))}
            <line x1={X(mean)} x2={X(mean)} y1={cy - rowH / 2 + 8} y2={cy + rowH / 2 - 8} stroke="var(--ink)" strokeWidth={2.5} strokeLinecap="round" />
            <circle cx={X(w)} cy={cy} r={rad + 4} fill="var(--bad)" stroke="var(--surface)" strokeWidth={2} />
            <text x={X(w) - rad - 10} y={cy + 4} textAnchor="end" style={{ fill: "var(--bad)", fontWeight: 700, paintOrder: "stroke", stroke: "var(--bg)", strokeWidth: 4 }}>lowest {pct(w, 0)}</text>
          </motion.g>
        );
      })}
      <g transform={`translate(${m.l} ${H - 6})`}>
        <line x1={0} x2={0} y1={-9} y2={1} stroke="var(--ink)" strokeWidth={2.5} /><text x={8} y={0}>average</text>
        <circle cx={90} cy={-4} r={4} fill="var(--s1)" /><text x={98} y={0}>one class</text>
        <circle cx={186} cy={-4} r={4} fill="var(--bad)" /><text x={194} y={0}>class more than 5 points below</text>
      </g>
    </svg>
  );
}
