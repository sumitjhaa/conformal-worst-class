"use client";
import { useMemo, useRef, useState, type ReactNode } from "react";
import { betaSf, kIndex, pacK, worstLaw } from "@/lib/conformal";
import { asset, loadResults, useLoad } from "@/lib/data";
import { Box, Chapter, Reveal, StepSlider } from "./ui";

const NS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
const KS = [10, 100, 1000];
const ALPHAS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3], DELTAS = [0.01, 0.05, 0.1, 0.2, 0.3];
const pct = (v: number, d = 1) => (Number.isFinite(v) ? `${(v * 100).toFixed(d)}%` : "—");

/** Live calculator: everything here is computed in the browser from the formulas below, nothing is looked up. */
function Calculator() {
  const [ni, setNi] = useState(1), [ki, setKi] = useState(2), [ai, setAi] = useState(1), [di, setDi] = useState(2);
  const n = NS[ni], K = KS[ki], a = ALPHAS[ai], delta = DELTAS[di];
  const out = useMemo(() => {
    const k = kIndex(n, a), feasible = k <= n;
    const below = feasible ? 1 - betaSf(1 - a, k, n + 1 - k) : 0;
    const law = worstLaw(K, n, a);
    const dg = 1 - Math.pow(1 - delta, 1 / K), kp = pacK(n, a, dg);
    const nMin = Math.ceil(Math.log(dg) / Math.log(1 - a));
    return { k, feasible, below, sd: Math.sqrt(a * (1 - a) / n), law, dg, kp, nMin };
  }, [n, K, a, delta]);
  return (
    <div className="grid2">
      <div className="stack" style={{ gap: 14 }}>
        <StepSlider label="Calibration pictures per class (n)" stops={NS.map(String)} index={ni} display={String(n)} onChange={setNi} />
        <StepSlider label="Number of classes (K)" stops={KS.map(String)} index={ki} display={String(K)} onChange={setKi} />
        <StepSlider label="Allowed error α" stops={ALPHAS.map((v) => v.toFixed(2))} index={ai} display={a.toFixed(2)} onChange={setAi} />
        <StepSlider label="PAC failure budget δ" stops={DELTAS.map((v) => v.toFixed(2))} index={di} display={delta.toFixed(2)} onChange={setDi} />
      </div>
      <div className="card">
        <table className="t">
          <tbody>
            <tr><th scope="row">Cut-off index k = ⌈(n+1)(1−α)⌉</th><td className="n">{out.feasible ? out.k : `${out.k} > n → keep every label`}</td></tr>
            <tr><th scope="row">Wobble of one class’s coverage (s.d.)</th><td className="n">±{pct(out.sd)}</td></tr>
            <tr><th scope="row">Chance one class lands below {pct(1 - a, 0)}</th><td className="n">{pct(out.below)}</td></tr>
            <tr><th scope="row">Expected worst true coverage of {K} classes</th><td className="n">{out.law ? pct(out.law.mean) : "—"}</td></tr>
            <tr><th scope="row">Chance the worst class is below {pct(1 - a, 0)}</th><td className="n">{out.law ? pct(out.law.pBelow) : "—"}</td></tr>
            <tr><th scope="row">PAC level per class δ_g = 1−(1−δ)^(1/K)</th><td className="n">{out.dg.toExponential(2)}</td></tr>
            <tr><th scope="row">PAC cut-off index</th><td className="n">{out.kp ?? "none: keep every label"}</td></tr>
            <tr><th scope="row">Minimum n for a finite PAC cut-off</th><td className="n">{out.nMin}</td></tr>
          </tbody>
        </table>
        <p className="small" style={{ marginTop: 10 }}>Try n = 20, K = 1000 (ImageNet): the PAC rule needs {Math.ceil(Math.log(1 - Math.pow(0.9, 1 / 1000)) / Math.log(0.9))} pictures per class at α = δ = 0.1, so with 20 it has to return all 1,000 labels. These are <em>true</em> coverages; measuring each class on only ~20 test pictures adds more noise, which is why ImageNet’s measured worst class is lower still (about 48%).</p>
      </div>
    </div>
  );
}

function Topic({ no, title, ch, children, open }: { no: number; title: string; ch?: string; children: ReactNode; open?: boolean }) {
  return (
    <details className="acc" open={open}>
      <summary>
        <svg className="acc-chev" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden><path d="M6 3.5 10.5 8 6 12.5" /></svg>
        <span className="acc-no">{String(no).padStart(2, "0")}</span><span className="acc-t">{title}</span><span className="acc-ch">{ch ?? ""}</span>
      </summary>
      <div className="acc-body">{children}</div>
    </details>
  );
}

/** Opens or closes every section in the list at once; the sections stay plain <details>, so each still toggles on its own. */
function TopicList({ children }: { children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  const all = (open: boolean) => ref.current?.querySelectorAll("details").forEach((d) => { d.open = open; });
  return (
    <div className="stack" style={{ gap: 12 }}>
      <div className="acc-tools">
        <h3>Step by step</h3>
        <div className="row" style={{ gap: 6 }}><button type="button" className="xbtn" onClick={() => all(true)}>Open all</button><button type="button" className="xbtn" onClick={() => all(false)}>Close all</button></div>
      </div>
      <div className="acc-list" ref={ref}>{children}</div>
    </div>
  );
}

export default function DeepDive() {
  const { data: R } = useLoad(loadResults);
  const need = (R?.n_needed || []).filter((r) => Math.abs(Number(r.eps) - 0.02) < 1e-9);
  return (
    <Chapter id="deep" no={10} eyebrow="For the curious" title="The maths behind every chapter"
      lead={<>Everything above, now with formulas. Open any section; each one says which part of the project it supports. Notation: n calibration pictures, K classes, α allowed error (0.1 = 10%).</>}>

      <Reveal><div className="card stack"><h3>Play with the formulas</h3><Calculator /></div></Reveal>

      <TopicList>
      <Topic no={1} title="Split conformal: the recipe" open>
        <p>Pick a <strong>score</strong> s(x, y) that is small when label y fits picture x. For the simple rule, s = 1 − p̂(y | x).
          Compute it for the true label of each of the n calibration pictures, sort them, and take the k-th smallest, with
          <code> k = ⌈(n+1)(1−α)⌉</code>. That value is the cut-off q̂. For a new picture, the set is every label whose score is at most q̂:
          <code> C(x) = &#123;y : s(x, y) ≤ q̂&#125;</code>. If k &gt; n there aren’t enough pictures and the only safe set is all labels.</p>
      </Topic>

      <Topic no={2} title="Why the 90% promise holds">
        <p>Assume the new picture is <strong>exchangeable</strong> with the calibration pictures: it’s just one more draw from the same pile.
          Then its score is equally likely to land in any of the n+1 positions when all scores are sorted together. The true label is in the
          set exactly when the new score ranks at or below k, so</p>
        <div className="eq">P(Y ∈ C(X)) = k / (n+1)  ∈  [1−α, 1−α + 1/(n+1))</div>
        <p>Nothing about the model was assumed. A bad model still gets 90%, it just pays with bigger sets.</p>
      </Topic>

      <Topic no={3} title="Why one calibration set can still miss" ch="chapter 4">
        <p>The promise is an average over all possible calibration sets. For the one set you actually drew, coverage is a random number
          with an exact law: <code>Beta(k, n+1−k)</code>. Its spread is about <code>√(α(1−α)/n)</code>: ±3 points at n = 100, ±1 point at
          n = 1,000. At n = 100 close to half of all draws land below 90% (44% of my 200 re-splits), which is what chapter 4’s histogram shows.</p>
      </Topic>

      <Topic no={4} title="The scores I compared" ch="chapter 3">
        <ul>
          <li><strong>Naive (LAC):</strong> 1 − p̂(y|x). Gives the smallest sets on average when probabilities are calibrated.</li>
          <li><strong>APS:</strong> add labels from most to least likely until their probabilities sum past the cut-off. Sets grow on hard pictures.</li>
          <li><strong>RAPS / SAPS:</strong> APS with a penalty for long tails of unlikely labels, so sets stay short.</li>
          <li><strong>Margin:</strong> how far the label’s probability is behind the best other label.</li>
          <li><strong>Learned score:</strong> a small network trained (on the design split) to make sets small.</li>
          <li><strong>Distance score:</strong> naive score + γ × the Mahalanobis distance of the picture’s features from the class average. This was my first idea, and it barely helped: sets grow by adding the second-best label, and the softmax already judges that label better than the distance does (AUROC 0.92 vs 0.74 on CIFAR-10).</li>
          <li><strong>Temperature (T):</strong> divide logits by T before softmax, with T tuned on the design split, so probabilities aren’t over-confident.</li>
        </ul>
      </Topic>

      <Topic no={5} title="Per-class cut-offs and the worst-class law" ch="chapter 6">
        <p>Mondrian conformal gives each class c its own cut-off from that class’s own n_c calibration pictures. Those piles don’t overlap,
          so the classes’ coverages are <strong>independent</strong>, each <code>Beta(k_c, n_c+1−k_c)</code>. Measured on m_c test pictures,
          each is Beta–binomial. The chance that every class is at least t is then a product:</p>
        <div className="eq">P(min_c cov_c ≥ t) = ∏_c P(cov_c ≥ t)</div>
        <p>That is the worst-class law. It needs only the counts, no pictures. It matched my re-split experiments to within 1.4 points
          from 10 to 1,000 classes. On CIFAR-100’s official test set it was 5 points too hopeful, which turned out to be a real difference
          between that test set and the training pool.</p>
        {need.length > 0 && (
          <div>
            <p>Turned around, it says how many calibration pictures per class you need so that <em>every</em> class is within 2 points of
              target with 90% probability:</p>
            <table className="t">
              <thead><tr><th>Classes K</th><th className="n">Pictures per class</th></tr></thead>
              <tbody>{need.map((r) => <tr key={String(r.K)}><th scope="row">{Number(r.K).toLocaleString()}</th><td className="n">{Number(r.n_needed).toLocaleString()}</td></tr>)}</tbody>
            </table>
          </div>
        )}
      </Topic>

      <Topic no={6} title="The safety margin: PAC over groups" ch="chapter 7">
        <p>Instead of k = ⌈(n+1)(1−α)⌉, pick the smallest k for which <code>P(Beta(k, n+1−k) ≥ 1−α) ≥ 1−δ_g</code>. Then that group meets
          the target with probability at least 1−δ_g, not just on average. To protect G groups <em>at once</em> with probability 1−δ, use
          <code> δ_g = 1 − (1−δ)^(1/G)</code>. Because the groups are independent this is exact, slightly better than the usual union bound δ/G.</p>
        <p>The catch: even the largest possible k only works if <code>n ≥ log δ_g / log(1−α)</code>. With 1,000 classes that is 87 pictures
          per class. ImageNet gives about 20, so per-class PAC must return all 1,000 labels.</p>
      </Topic>

      <Topic no={7} title="Clustered-PAC: the fix" ch="chapter 7">
        <p>Group classes that behave alike, then apply the PAC rule per group. To group them, describe each class by its 50th–90th
          percentile scores on the design split and run k-means with M clusters. M = 1 is one shared cut-off: precise but unfair. M = K is
          one per class: fair but noisy, with a huge margin. In between, groups are big enough for a small margin and similar enough to be
          fair. On ImageNet this lifted the worst class from 48% (Mondrian) to 61% with 17 labels per set instead of 35.</p>
      </Topic>

      <Topic no={8} title="When the data changes" ch="chapter 8">
        <ul>
          <li><strong>Label shift</strong> (the class mix changes): coverage becomes <code>Σ_c π′_c · cov_c</code>, so in the worst case it
            falls to the worst class. Per-class methods are protected; weighting by estimated class ratios (BBSE) also repairs it.</li>
          <li><strong>Covariate shift</strong> (pictures look different, e.g. fog or noise): no guarantee survives. Weighted conformal
            re-weights calibration pictures by how similar they are to the new ones, but when the two barely overlap it returns all labels.</li>
          <li><strong>ACI</strong> (online repair): after each picture, nudge the error level with <code>α_(t+1) = α_t + η(α − err_t)</code>.
            Misses lower α, so sets grow until coverage recovers.</li>
        </ul>
      </Topic>

      <Topic no={9} title="Empty sets are forced when the model is very accurate">
        <p>If every set has at most one label, a miss is either an empty set or a wrong top-1 guess, so
          <code> P(empty) ≥ accuracy − coverage</code>. MNIST is 99% accurate, so to land at 90% coverage conformal must leave at least 9%
          of sets empty. I measured 9.4%.</p>
      </Topic>

      <Topic no={10} title="How I evaluated, and what I don’t claim">
        <p>Each picture has one job: train (80% of the training set), design (10%, for every tuned setting), calibration (10%, cut-offs
          only), and the official test set, opened once. Each experiment was repeated 3–10 times with new random splits (seeds). I report
          coverage, set size, the worst class, empty-set rate, and coverage by confidence level, with 95% bootstrap intervals on differences.</p>
        <p>Limits: the worst-class law assumes the test pictures of a class look like its calibration pictures. Clustered-PAC protects
          groups, not each single class. The shifts I tested are benchmark corruptions, not a real deployment.</p>
      </Topic>
      </TopicList>

      <Box kind="keyidea" tag="Where to read more">
        <p>The <a href={asset("papers/es678_conformal.pdf")}>4-page paper</a> has the full argument, with every number produced by the project’s scripts.</p>
      </Box>
    </Chapter>
  );
}
