"use client";
import { BrainCircuit, CloudLightning, Microscope, Ruler, Scale, ShieldCheck, Telescope, WandSparkles, type LucideIcon } from "lucide-react";
import { motion } from "motion/react";
import { pct } from "@/lib/conformal";
import { at, avg, loadResults, useLoad } from "@/lib/data";
import { Chapter, Reveal, Stat } from "./ui";

const WORKED: [LucideIcon, string, string][] = [
  [ShieldCheck, "The 90% promise holds", "Every conformal rule covered 89–92% on average, on all five settings, across 3–10 independently trained models each."],
  [Telescope, "The worst class is predictable", "My formula predicted the worst class within about 1 point on ImageNet (48.7% vs 47.7%) and on every random reshuffle, using only the count of pictures per class."],
  [WandSparkles, "The cheap fix works", "Clustered-PAC raised ImageNet’s worst class from 38% to 61% with about 17 labels per picture, where the per-class version needs all 1,000."],
  [Scale, "Honest checks", "Every comparison has a 95% confidence interval from resampling models and pictures, so I know which differences are real and which are noise."],
];
const DIDNT: [LucideIcon, string, string][] = [
  [Ruler, "Distance idea: small gain", "My first idea (add how far a picture is from each class in the model’s feature space) helped only 2 points on CIFAR-10 and not at all on CIFAR-100. I explain why in the paper."],
  [BrainCircuit, "Learned score: no gain", "Training an extra network to make sets smaller did not beat the simple rule."],
  [CloudLightning, "Shift breaks everything", "No method keeps its promise on corrupted pictures. The adaptive fix (ACI) restores coverage only by making sets much bigger."],
  [Microscope, "One miss of the formula", "On CIFAR-100 the formula was 5 points off. That turned out to reveal a real difference between CIFAR-100’s training and test pictures."],
];

export default function Results() {
  const { data: R } = useLoad(loadResults);
  const g = (ds: string, m: string, k: string) => avg(at(R, "main", { dataset: ds, method: m, alpha: 0.1 }), k);
  return (
    <Chapter id="results" no={9} eyebrow="So, are the results good?" title="What worked, what didn’t, and what it means"
      lead={<>Short answer: <strong>yes</strong>. The project confirms the theory, adds a new formula that predicts unfairness in advance, and gives a fix that works where the standard fix is impossible. It also reports what failed, honestly.</>}>
      <div className="grid4">
        <Stat k="Settings tested" value={5} format={(v) => Math.round(v).toString()} s="digits → 1,000-class ImageNet" />
        <Stat k="Methods compared" value={13} format={(v) => Math.round(v).toString()} s="plus 3 repairs for shift" />
        <Stat k="ImageNet worst class, before" value={g("imagenet", "temp_naive", "min_class_cov")} format={(v) => pct(v)} tone="bad" s="simple sets" />
        <Stat k="ImageNet worst class, my fix" value={g("imagenet", "clustered_pac", "min_class_cov")} format={(v) => pct(v)} tone="good" s={`${g("imagenet", "clustered_pac", "mean_size").toFixed(1)} labels / picture`} />
      </div>
      <div className="grid2">
        <Reveal>
          <div className="verdict good">
            <h3>What worked</h3>
            {WORKED.map(([Icon, t, d], i) => (
              <motion.div key={t} className="vitem" initial={{ opacity: 0, x: -16 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.1 }}>
                <span className="vicon"><Icon aria-hidden /></span><div><strong>{t}</strong><p className="small">{d}</p></div>
              </motion.div>
            ))}
          </div>
        </Reveal>
        <Reveal delay={0.1}>
          <div className="verdict meh">
            <h3>What didn’t (and why that’s still useful)</h3>
            {DIDNT.map(([Icon, t, d], i) => (
              <motion.div key={t} className="vitem" initial={{ opacity: 0, x: 16 }} whileInView={{ opacity: 1, x: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.1 }}>
                <span className="vicon"><Icon aria-hidden /></span><div><strong>{t}</strong><p className="small">{d}</p></div>
              </motion.div>
            ))}
          </div>
        </Reveal>
      </div>
      <Reveal>
        <div className="card">
          <h3>The takeaway in one sentence</h3>
          <p style={{ fontSize: 19, marginTop: 6 }}>An AI’s “90% guaranteed” can hide classes it serves badly; you can <strong>predict</strong> how badly from the amount of data
            alone, and <strong>fix</strong> it cheaply by sharing safety-margin cut-offs between similar classes. Always report the worst class next to the average.</p>
        </div>
      </Reveal>
    </Chapter>
  );
}
