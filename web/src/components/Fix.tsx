"use client";
import { Boxes, LayoutGrid, ShieldCheck, SlidersHorizontal, Square, Tags, type LucideIcon } from "lucide-react";
import { motion } from "motion/react";
import { useMemo, useState } from "react";
import { pct, rng } from "@/lib/conformal";
import { at, avg, loadGallery, loadResults, useLoad } from "@/lib/data";
import { Box, Chapter, HBars, Reveal, Seg, StepSlider } from "./ui";

const COLORS = ["var(--s1)", "var(--s2)", "var(--s3)", "var(--s4)", "var(--s5)", "var(--good)"];
const NC = 30; // toy classes in the grouping animation

const GROUP_OPTS = [1, 2, 4, 6, 10, 30];
const PICS = 20; // calibration pictures per class, as on ImageNet

function Grouping() {
  const [gi, setGi] = useState(2);
  const M = GROUP_OPTS[gi];
  // each toy class has a hidden "difficulty"; grouping sorts classes of similar difficulty together, like the real clustering
  const diff = useMemo(() => { const r = rng(7); return Array.from({ length: NC }, () => r()).sort((a, b) => a - b); }, []);
  const per = Math.ceil(NC / M), nG = Math.ceil(NC / per);
  const W = 640, H = 150, gap = 8, tw = (W - 20 - gap * (nG - 1)) / NC, ty = 28, th = 46;
  const r2 = (v: number) => Math.round(v * 100) / 100;
  return (
    <div className="card stack">
      <div className="row" style={{ justifyContent: "space-between", alignItems: "flex-start" }}>
        <h3>Grouping similar classes</h3>
        <StepSlider label="Number of groups" index={gi} onChange={setGi} stops={GROUP_OPTS}
          display={M === 1 ? "1 · everyone shares" : M === NC ? `${NC} · one per class` : `${M} groups`} />
      </div>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${NC} classes in ${nG} groups; each cut-off is set from about ${per * PICS} pictures`}>
        <text x={10} y={14}>easy classes</text><text x={W - 10} y={14} textAnchor="end">hard classes</text>
        {diff.map((d, i) => {
          const g = Math.floor(i / per), x = r2(10 + i * tw + g * gap);
          return <motion.rect key={i} y={ty} width={r2(tw - 2)} height={th} rx={4} initial={false}
            animate={{ x, fill: COLORS[g % COLORS.length], opacity: 0.35 + 0.65 * d }} transition={{ type: "spring", stiffness: 170, damping: 22 }} />;
        })}
        {Array.from({ length: nG }, (_, g) => {
          const n = Math.min(per, NC - g * per), x0 = r2(10 + g * per * tw + g * gap), w = r2(n * tw - 2), yb = ty + th + 10, c = COLORS[g % COLORS.length];
          return (
            <motion.g key={`${M}-${g}`} initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.25 }}>
              <path d={`M${x0},${yb} v6 H${r2(x0 + w)} v-6`} fill="none" stroke={c} strokeWidth={2} />
              {w > 54 && <text x={r2(x0 + w / 2)} y={yb + 24} textAnchor="middle" style={{ fill: c, fontWeight: 600 }}>{n * PICS} pics</text>}
            </motion.g>
          );
        })}
      </svg>
      <p className="small">{M === 1 ? `One group: all classes share one cut-off set from ${NC * PICS} pictures. Lots of data, but hard classes get short-changed (chapter 5).`
        : M === NC ? `One group per class: fair on paper, but each cut-off is set from only ${PICS} pictures, so it is noisy (chapter 6).`
          : `${M} groups: classes that behave alike share a cut-off set from about ${per * PICS} pictures, so it is both fair-ish and well-measured.`}</p>
    </div>
  );
}

export default function Fix() {
  const { data: R } = useLoad(loadResults);
  const { data: G } = useLoad(loadGallery);
  const [gm, setGm] = useState("temp_aps");
  const methods: [string, string, string][] = [["temp_naive", "Simple sets", "var(--muted)"], ["temp_aps", "Adaptive sets", "var(--s2)"], ["mondrian", "Cut-off per class", "var(--s4)"], ["mondrian_pac", "Per class + margin", "var(--s5)"], ["clustered_pac", "Our fix (groups + margin)", "var(--good)"]];
  const im = methods.map(([m, name, color]) => { const r = at(R, "main", { dataset: "imagenet", method: m, alpha: 0.1 }); return { name, color, worst: avg(r, "min_class_cov"), size: avg(r, "mean_size") }; });
  const lab: Record<string, string> = { temp_naive: "Simple", temp_aps: "Adaptive", mondrian: "Per class", clustered: "Groups", clustered_pac: "Our fix" };
  const ico: Record<string, LucideIcon> = { temp_naive: Square, temp_aps: SlidersHorizontal, mondrian: LayoutGrid, clustered: Boxes, clustered_pac: ShieldCheck };
  const items = G ? G.items.slice(0, 12) : [];

  return (
    <Chapter id="fix" no={7} eyebrow="Our second result" title="A cheap fix: share cut-offs between similar classes, add a safety margin"
      lead={<>To protect every class, two things are needed: enough pictures behind each cut-off, and a margin for bad luck. Our method, <strong>Clustered-PAC</strong>, does both.</>}>
      <div className="stack" style={{ gap: 28 }}>
        <Box kind="analogy" tag="Idea 1 · share">
          <p>Instead of one cut-off per class (too little data each) or one for all (unfair), group classes that the model finds equally hard and give
            each group its own cut-off. Like seating students by level so each group gets the right difficulty of homework.</p>
        </Box>
        <Box kind="analogy" tag="Idea 2 · safety margin">
          <p>A cut-off set from a sample is a bit random. So we move it slightly further, just enough that <strong>every group</strong> reaches 90%
            with 90% confidence at the same time. Like leaving 10 minutes early so you are on time even on a bad-traffic day. (“PAC” = probably approximately correct.)</p>
        </Box>
      </div>
      <Reveal><Grouping /></Reveal>

      <Reveal>
        <div className="card stack">
          <h3 style={{ marginBottom: 18 }}>Result on ImageNet (1,000 classes, ≈20 calibration pictures each)</h3>
          {!R ? <p className="loading">loading…</p> : (
            <div className="grid2">
              <div><p className="small" style={{ marginBottom: 6 }}><strong>Worst-covered class</strong> (higher is better)</p>
                <HBars label="Worst class coverage on ImageNet" max={1} target={0.9} fmt={(v) => pct(v)} items={im.map((r) => ({ name: r.name, value: r.worst, color: r.color }))} /></div>
              <div><p className="small" style={{ marginBottom: 6 }}><strong>Average set size</strong> (smaller is better; log scale)</p>
                <HBars label="Average set size on ImageNet" max={3} fmt={(v) => (Math.pow(10, v)).toFixed(v >= 2 ? 0 : 1)} items={im.map((r) => ({ name: r.name, value: Math.log10(Math.max(1, r.size)), color: r.color }))} /></div>
            </div>
          )}
          <p className="small">The per-class safety margin is <strong>impossible</strong> here: with only 20 pictures per class it must list all 1,000 labels.
            Our fix lifts the worst class from {pct(im[0].worst, 0)} (simple) and {pct(im[2].worst, 0)} (per class) to <strong>{pct(im[4].worst, 0)}</strong>, with about {im[4].size.toFixed(0)} labels per picture out of 1,000.</p>
        </div>
      </Reveal>

      <Reveal>
        <div className="card stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h3>The same ImageNet pictures under each rule</h3>
            {G && <Seg label="Rule" value={gm} onChange={setGm} options={G.methods.map((m) => { const I = ico[m]; return [m, <span key={m} className="seg-ico">{I && <I aria-hidden />}{lab[m] || m}</span>]; })} />}
          </div>
          {!G ? <p className="loading">loading pictures…</p> : (
            <div className="gallery" style={{ gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))" }}>
              {items.map((it, idx) => {
                const s = it.sets[gm];
                return (
                  <motion.div layout key={idx} className={`imgcard ${s.covered ? "" : "miss"}`}>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={it.img} alt={it.true} style={{ imageRendering: "auto" }} loading="lazy" />
                    <div className="imgmeta">
                      <motion.span key={gm} initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className={`setbadge ${s.covered ? "ok" : "miss"}`}
                        title={`${s.size} label${s.size === 1 ? "" : "s"} in the set · ${s.covered ? "covered" : "missed"}`}>
                        <Tags aria-hidden />{s.size}
                      </motion.span>
                      <span className="small imgname">{it.true}</span>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          )}
        </div>
      </Reveal>
    </Chapter>
  );
}
