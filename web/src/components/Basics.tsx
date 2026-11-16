"use client";
import { useMemo, useState } from "react";
import { probs } from "@/lib/conformal";
import { loadDataset, loadResults, useLoad } from "@/lib/data";
import { Box, Chapter, HBars, Reveal, Slider } from "./ui";

// sizes are the public dataset sizes; accuracy comes from my own trained models (results.json)
const DATA = [
  { ds: "mnist", name: "MNIST", what: "handwritten digits 0–9", px: "28×28, grey", K: 10, n: "70,000 (60k train + 10k test)", model: "LeNet, a small 1998-style network, trained by me" },
  { ds: "cifar10", name: "CIFAR-10", what: "photos: plane, car, bird, cat, deer, dog, frog, horse, ship, truck", px: "32×32, colour", K: 10, n: "60,000 (50k + 10k)", model: "a small ResNet trained by me, plus a pre-trained ResNet-18" },
  { ds: "cifar100", name: "CIFAR-100", what: "photos of 100 kinds of thing (apple, bee, castle…)", px: "32×32, colour", K: 100, n: "60,000 (50k + 10k)", model: "a small ResNet trained by me" },
  { ds: "imagenet", name: "ImageNet", what: "photos of 1,000 kinds of thing", px: "full-size photos", K: 1000, n: "50,000 validation photos", model: "a published ResNet-152; I only use its saved outputs" },
];

function SoftmaxDemo() {
  const { data: d } = useLoad(() => loadDataset("cifar10"));
  const [T, setT] = useState(1);
  const [pick, setPick] = useState(0);
  const view = useMemo(() => {
    if (!d) return null;
    const p1 = probs(d.test, 1), K = d.K, y = d.test.labels!;
    // pictures where the model hesitates make the point best: top probability under 80%
    const unsure = d.gallery.filter((g) => Math.max(...Array.from(p1.subarray(g.i * K, g.i * K + K))) < 0.8);
    const g = (unsure.length ? unsure : d.gallery)[pick % Math.max(1, (unsure.length ? unsure : d.gallery).length)];
    const pT = probs(d.test, T), row = Array.from(pT.subarray(g.i * K, g.i * K + K));
    const top = row.map((v, j) => ({ j, v })).sort((a, b) => b.v - a.v).slice(0, 5);
    return { img: g.img, top, truth: y[g.i], best: d.tuned.T };
  }, [d, T, pick]);

  if (!d || !view) return <p className="loading">loading a real CIFAR-10 picture…</p>;
  return (
    <div className="grid2">
      <div className="stack" style={{ gap: 12 }}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={view.img} alt={`test picture, true label ${d.classes[view.truth]}`} style={{ width: 128, imageRendering: "pixelated", borderRadius: 8 }} />
        <p className="small">True label: <strong>{d.classes[view.truth]}</strong>. This is a real test picture the model never saw while training.</p>
        <Slider label="Temperature T" value={T} display={T.toFixed(2)} min={0.5} max={3} step={0.05} onChange={setT} marks={["0.5", "1", "1.5", "2", "2.5", "3"]} />
        <p className="small">T = 1 is the raw model. My tuned value is T = {view.best.toFixed(2)}: bigger T flattens the bars, smaller T sharpens them.</p>
        <button className="btn" onClick={() => setPick((v) => v + 1)}>Another picture →</button>
      </div>
      <div className="card">
        <h3>The model’s top 5 guesses</h3>
        <HBars label="Top five class probabilities" max={1} fmt={(v) => `${(v * 100).toFixed(1)}%`}
          items={view.top.map((t) => ({ name: d.classes[t.j], value: t.v, color: t.j === view.truth ? "var(--good)" : "var(--s2)" }))} />
        <p className="small" style={{ marginTop: 10 }}>Green is the true label. The longest bar is the model’s single answer (its “top-1” guess).</p>
      </div>
    </div>
  );
}

export default function Basics() {
  const { data: R } = useLoad(loadResults);
  const acc = (ds: string) => {
    const rows = (R?.tuned || []).filter((r) => r.dataset === ds);
    return rows.length ? rows.reduce((s, r) => s + Number(r.accuracy), 0) / rows.length : NaN;
  };
  return (
    <Chapter id="basics" no={0} eyebrow="Start here" title="The basics, before anything else"
      lead={<>No machine-learning background needed. This chapter covers the five ideas the rest of the page builds on. If you already know them, skip to chapter 1.</>}>

      <Reveal>
        <div className="card stack">
          <h3>1 · Data: lots of pictures, each with the right answer</h3>
          <p>A dataset here is a big pile of pictures, each tagged with its correct label by a person. One <strong>row</strong> is one picture
            plus its label. I used four standard, public datasets, from easy to hard:</p>
          <div style={{ overflowX: "auto" }}>
            <table className="t">
              <thead><tr><th>Dataset</th><th>What is in it</th><th>Picture</th><th className="n">Classes</th><th>Size</th><th>Model</th><th className="n">Accuracy</th></tr></thead>
              <tbody>{DATA.map((x) => (
                <tr key={x.ds}><th scope="row">{x.name}</th><td>{x.what}</td><td>{x.px}</td><td className="n">{x.K}</td><td>{x.n}</td><td>{x.model}</td>
                  <td className="n">{Number.isFinite(acc(x.ds)) ? `${(acc(x.ds) * 100).toFixed(1)}%` : "…"}</td></tr>
              ))}</tbody>
            </table>
          </div>
          <p className="small">Accuracy = the share of test pictures where the model’s single best guess is right, averaged over my repeated runs.
            Classes are close to balanced (roughly the same number of pictures each) and no labels are missing.</p>
        </div>
      </Reveal>

      <Reveal>
        <div className="card stack">
          <h3>2 · What a classifier actually outputs</h3>
          <p>A trained network doesn’t just output a label. For every possible class it gives a raw number (a <em>logit</em>). A formula
            called <strong>softmax</strong> turns these into probabilities that add up to 100%. The class with the biggest probability is the
            model’s answer. Dividing the logits by a <strong>temperature</strong> T before softmax makes the model more or less sure of itself,
            without changing which class wins. Try it on a picture the model finds hard:</p>
          <SoftmaxDemo />
        </div>
      </Reveal>

      <Reveal>
        <div className="card stack">
          <h3>3 · Training vs testing, and why I hide data</h3>
          <p>A model learns from <strong>training</strong> pictures. If I graded it on those same pictures, it could score well just by
            memorising them. So I keep a separate <strong>test</strong> set locked away and only open it at the very end. The same logic is why
            this project splits data four ways (chapter 11): each picture has exactly one job, so no result is graded on data that helped make it.</p>
        </div>
      </Reveal>

      <Reveal>
        <div className="card stack">
          <h3>4 · What “right 90% of the time” means</h3>
          <p>It’s a long-run frequency, like a fair coin landing heads 50% of the time. Toss it 10 times and you might get 7 heads. Toss it
            10,000 times and you’ll be very close to 50%. So a promise of “90% coverage” is checked over many pictures, and even then a single
            batch can land a bit above or below. Chapter 4 measures exactly how much.</p>
        </div>
      </Reveal>

      <Reveal>
        <div className="card stack">
          <h3>5 · Percentiles (quantiles)</h3>
          <p>Sort 100 exam scores from low to high. The 90th percentile is the score that 90 of them fall below. Conformal prediction uses this
            one idea: it sorts the model’s “surprise” on 100s of fresh pictures and puts a cut-off at the 90th percentile. That’s the whole
            trick behind the guarantee, and chapter 2 shows it on real numbers.</p>
        </div>
      </Reveal>

      <Box kind="keyidea" tag="How the page is ordered">
        <p>Chapters 1–3 explain the method with demos. Chapters 4–8 are what I found: the promise holds on average, some classes are left
          behind, I can predict which and by how much, and a cheap fix. Chapter 9 sums up the results, chapter 10 has the maths for anyone who
          wants it, and chapter 11 shows exactly how the experiments were run.</p>
      </Box>
    </Chapter>
  );
}
