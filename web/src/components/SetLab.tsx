"use client";
import { Camera, Crosshair, Images, LayoutGrid, PenLine, ShieldCheck, SlidersHorizontal, Square, type LucideIcon } from "lucide-react";
import { useDeferredValue, useMemo, useState, type ReactNode } from "react";
import { calibrate, metrics, pct, predictSets, setOf, type Method } from "@/lib/conformal";
import { loadDataset, SHORT_NAMES, useLoad, type DsName } from "@/lib/data";
import Gif from "./Gif";
import { Box, Chapter, ImgCard, Reveal, Seg, Stat, StepSlider } from "./ui";

const ALPHAS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3];
const DATASETS: [DsName, string, LucideIcon][] = [["mnist", "Handwritten digits", PenLine], ["cifar10", "Photos (10 kinds)", Camera], ["cifar100", "Photos (100 kinds)", Images]];
const RULE_ICON: Record<string, LucideIcon> = { top1: Crosshair, naive: Square, aps: SlidersHorizontal, mondrian: LayoutGrid, clustered_pac: ShieldCheck };
const tab = (Icon: LucideIcon, label: string): ReactNode => <span className="seg-ico"><Icon aria-hidden />{label}</span>;

const METHODS: [Method, string, string][] = [
  ["top1", "One guess", "The normal AI: always exactly one label, no promise."],
  ["naive", "Simple sets", "Keep every label whose surprise score is under the cut-off (the rule from chapter 2). Smallest sets."],
  ["aps", "Adaptive sets", "Add labels from most to least likely until enough probability is covered (APS). Larger, but fairer across classes."],
  ["mondrian", "Cut-off per class", "A separate cut-off for each class, each set from that class’s own calibration pictures (Mondrian)."],
  ["clustered_pac", "Our fix", "Group similar classes and give each group a safety-margin cut-off (Clustered-PAC, chapter 6)."],
];

export default function SetLab() {
  const [ds, setDs] = useState<DsName>("cifar10");
  const [method, setMethod] = useState<Method>("aps");
  const [ai, setAi] = useState(3), alpha = ALPHAS[ai];
  const { data: d, error } = useLoad(() => loadDataset(ds), [ds]);
  // React renders the controls at once and recomputes the sets in the background; old results dim until the new ones land.
  const a = useDeferredValue(alpha), meth = useDeferredValue(method), stale = a !== alpha || meth !== method;

  const out = useMemo(() => {
    if (!d || d.ds !== ds) return null;
    const P = { method: meth, alpha: a, T: d.tuned.T, delta: d.tuned.delta ?? 0.1 };
    const cal = calibrate(d.cal, P, d.clusters), C = predictSets(d.test, P, cal), m = metrics(d.test, C);
    const names = SHORT_NAMES[ds] || d.classes;
    const worstC = m.classCov.indexOf(m.worst);
    return { m, C, names, worstName: names[worstC] };
  }, [d, ds, meth, a]);

  return (
    <Chapter id="try" no={3} eyebrow="Try it yourself" title="Make the AI answer with sets"
      lead={<>Everything here is recomputed live in your browser from the trained models’ real outputs on thousands of test pictures. Pick a dataset and a rule, then move α.</>}>
      <Reveal>
        <div className="card stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <Seg label="Dataset" value={ds} onChange={setDs} options={DATASETS.map(([v, l, I]) => [v, tab(I, l)] as [DsName, ReactNode])} />
            <StepSlider label="α (allowed error)" index={ai} onChange={setAi} display={`${alpha.toFixed(2)} → target ${Math.round((1 - alpha) * 100)}%`} stops={ALPHAS.map((v) => v.toFixed(2))} />
          </div>
          <Seg label="Rule" value={method} onChange={setMethod} options={METHODS.map(([v, l]) => [v, tab(RULE_ICON[v], l)] as [Method, ReactNode])} />
          <p className="small">{METHODS.find((m) => m[0] === method)![2]}</p>
        </div>
      </Reveal>

      {error && <p className="loading">Could not load data: {error}</p>}
      {!out ? <p className="loading">loading {ds}… (CIFAR-100 is about 12 MB)</p> : (
        <div className={`stack ${stale ? "stale" : ""}`} style={{ gap: 28 }} aria-busy={stale}>
          <div className="grid4">
            <Stat k="Coverage (truth inside the set)" value={out.m.cov} format={(v) => pct(v)} tone={out.m.cov >= 1 - a - 0.01 ? "good" : "bad"} s={`target ${Math.round((1 - a) * 100)}%`} />
            <Stat k="Average set size" value={out.m.mean} format={(v) => v.toFixed(2)} s="labels per picture" color="var(--s1)" />
            <Stat k="Empty sets" value={out.m.empty} format={(v) => pct(v)} s="model says “none of these”" color="var(--s2)" />
            <Stat k="Worst-treated class" value={out.m.worst} format={(v) => pct(v)} tone={out.m.worst < 1 - a - 0.02 ? "bad" : "good"} s={`class “${out.worstName}”`} />
          </div>
          <Reveal>
            <div>
              <div className="gallery">
                {d!.gallery.slice(0, 30).map((g) => {
                  const y = d!.test.labels![g.i], set = setOf(out.C, g.i, d!.K);
                  set.sort((a, b) => Number(b === y) - Number(a === y));
                  return <ImgCard key={g.i} img={g.img} trueName={out.names[y]} covered={set.includes(y)} labels={set.map((k) => out.names[k])} />;
                })}
              </div>
              <div className="legend" style={{ marginTop: 14 }}>
                <span><i style={{ background: "var(--ink)" }} />true label (inside the set)</span>
                <span><i style={{ border: "2px solid var(--good)" }} />green frame: truth covered</span>
                <span><i style={{ border: "2px solid var(--bad)" }} />red frame: truth missed</span>
              </div>
            </div>
          </Reveal>
        </div>
      )}

      <Gif name="sets" fig="Fig. 3a · APS on CIFAR-10" title="Watch the sets breathe" alt="Animation: eight hard CIFAR-10 test pictures; their prediction sets grow as alpha falls from 0.30 to 0.02 and shrink as it rises again, with live coverage and mean set size.">
        <p>Eight pictures the model finds hard. As α falls from 0.30 to 0.02 the target rises from 70% to 98%, and every set grows to keep the promise. Coverage and mean size in the header are measured on all 5,000 test pictures.</p>
      </Gif>

      <div className="stack" style={{ gap: 28 }}>
        <Box kind="keyidea" tag="Try this">
          <p>Pick <strong>One guess</strong> on 100 kinds of photo: coverage falls to about 71%, the model’s accuracy. Switch to <strong>Adaptive sets</strong>: coverage jumps to about 90%, and hard pictures get longer lists.</p>
        </Box>
        <Box kind="keyidea" tag="Try this">
          <p>Move α towards 0.01 (99% target). Sets grow: being almost never wrong costs a lot of labels. Move it to 0.3: sets shrink, even to empty.</p>
        </Box>
        <Box kind="warnbox" tag="Surprise">
          <p>On handwritten digits the model is 99% accurate, more than the 90% we asked for. So the rule <strong>returns empty sets</strong> about 9% of the time to “use up” its allowed error. Being too good also has odd effects.</p>
        </Box>
      </div>
    </Chapter>
  );
}
