"use client";
import { AudioWaveform, CloudFog, Grid2x2, Wind, type LucideIcon } from "lucide-react";
import type { ReactNode } from "react";
import { useMemo, useState } from "react";
import { calibrate, metrics, pct, predictSets, setOf } from "@/lib/conformal";
import { at, avg, loadDataset, loadResults, rows, SHORT_NAMES, useLoad } from "@/lib/data";
import { Box, Chapter, ImgCard, LineChart, Reveal, Seg, StepSlider } from "./ui";

const CORR_ICON: Record<string, LucideIcon> = { fog: CloudFog, "gaussian noise": AudioWaveform, "jpeg compression": Grid2x2, "motion blur": Wind };

export default function Shift() {
  const { data: d } = useLoad(() => loadDataset("cifar10"));
  const { data: R } = useLoad(loadResults);
  const corrs = useMemo(() => (d ? [...new Set(Object.keys(d.real).filter((k) => k.includes(" · ")).map((k) => k.split(" · ")[0]))] : []), [d]);
  const [corr, setCorr] = useState("gaussian noise");
  const [sev, setSev] = useState(0);

  const live = useMemo(() => {
    if (!d) return null;
    const P = { method: "aps" as const, alpha: 0.1, T: d.tuned.T, delta: 0.1 }, cal = calibrate(d.cal, P);
    const b = sev === 0 ? d.test : d.real[`${corr} · ${sev}`];
    if (!b) return null;
    const C = predictSets(b, P, cal), m = metrics(b, C);
    const cleanCov = sev === 0 ? m.cov : metrics(d.test, predictSets(d.test, P, cal)).cov;
    const imgs = sev === 0 ? d.gallery.slice(0, 12).map((g) => ({ img: g.img, i: g.i })) : (b.imgs || []).slice(0, 12).map((img, i) => ({ img, i }));
    return { m, C, b, imgs, cleanCov };
  }, [d, corr, sev]);

  const rs = rows(R, "real_shift").filter((r) => r.dataset === "cifar10");
  const curve = (m: string) => [1, 2, 3, 4, 5].map((s) => [s, avg(rs.filter((r) => r.family === "c" && r.method === m && Number(r.severity) === s), "coverage")] as [number, number]);
  const clean = (m: string) => avg(at(R, "main", { dataset: "cifar10", method: m, alpha: 0.1 }), "coverage");
  const aci = rows(R, "aci_series").filter((r) => r.dataset === "cifar10" && r.rolling_cov != null);
  const aciS = (lr: number) => aci.filter((r) => Number(r.lr) === lr).map((r) => [Number(r.t), Number(r.rolling_cov)] as [number, number]);
  const ls = at(R, "label_shift", { dataset: "cifar10", beta: 0.1 });
  const names = SHORT_NAMES.cifar10;

  return (
    <Chapter id="shift" no={8} eyebrow="When the world changes" title="The promise breaks if new pictures look different"
      lead={<>The whole guarantee rests on one assumption: new pictures are <strong>like the calibration pictures</strong>. What if the camera gets foggy, the image is noisy, or the mix of objects changes?</>}>
      <Reveal>
        <div className="card stack">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <Seg label="Corruption" value={corr} onChange={(v) => setCorr(v)} options={(corrs.length ? corrs : ["gaussian noise"]).map((c) => { const I = CORR_ICON[c]; return [c, <span key={c} className="seg-ico">{I && <I aria-hidden />}{c}</span>]; }) as [string, ReactNode][]} />
            <StepSlider label="How strong" index={sev} display={sev === 0 ? "clean" : `level ${sev} of 5`} stops={["clean", 1, 2, 3, 4, 5]} onChange={setSev} />
          </div>
          {!live ? <p className="loading">loading…</p> : (
            <>
              <div className="grid3">
                <div className={`stat ${live.m.cov < 0.88 ? "bad" : "good"}`}><span className="k">Coverage now</span><span className="v">{pct(live.m.cov)}</span><span className="s">promised: 90%{sev > 0 && (() => { const dp = Math.round((live.m.cov - live.cleanCov) * 100); return <span className={`delta ${dp < 0 ? "down" : "up"}`}>{dp < 0 ? "−" : "+"}{Math.abs(dp)} pts vs clean</span>; })()}</span>
                  <span className="gauge" role="img" aria-label={`coverage ${pct(live.m.cov)} against the 90% promise`}><i style={{ width: `${live.m.cov * 100}%` }} /><b style={{ left: "90%" }} /></span></div>
                <div className="stat"><span className="k">Average set size</span><span className="v">{live.m.mean.toFixed(2)}</span><span className="s">labels per picture</span></div>
                <div className="stat"><span className="k">Cut-off</span><span className="v" style={{ fontSize: "clamp(24px, 3vw, 30px)" }}>unchanged</span><span className="s">still set on clean pictures</span></div>
              </div>
              <div className="gallery">
                {live.imgs.map(({ img, i }) => {
                  const y = live.b.labels![i], set = setOf(live.C, i, 10).sort((a, b) => Number(b === y) - Number(a === y));
                  return <ImgCard key={`${sev}-${corr}-${i}`} img={img} trueName={names[y]} covered={set.includes(y)} labels={set.map((k) => names[k])} />;
                })}
              </div>
            </>
          )}
        </div>
      </Reveal>

      <div className="grid2">
        <Reveal>
          <div className="card stack">
            <h3>Average over 15 kinds of corruption</h3>
            {R && <LineChart label="Coverage against corruption strength" xDom={[0, 5]} yDom={[0.3, 1]} target={0.9} xTicks={[0, 1, 2, 3, 4, 5]} xFmt={(v) => (v === 0 ? "clean" : `lvl ${v}`)} vlines={[{ x: sev, color: "var(--accent)", label: "your slider", end: sev >= 4 }]}
              series={[{ pts: [[0, clean("temp_aps")], ...curve("temp_aps")], color: "var(--s2)", label: "Adaptive" }, { pts: [[0, clean("naive")], ...curve("naive")], color: "var(--muted)", label: "Simple" }]} />}
            <div className="legend"><span><i style={{ background: "var(--s2)" }} />Adaptive sets</span><span><i style={{ background: "var(--muted)" }} />Simple sets</span></div>
          </div>
        </Reveal>
        <Reveal delay={0.1}>
          <div className="card stack">
            <h3>Repair: let the cut-off adapt as it goes (ACI)</h3>
            {aci.length > 0 && <LineChart label="Rolling coverage over a stream that gets noisier" xDom={[0, Math.max(...aci.map((r) => Number(r.t)))]} yDom={[0.3, 1]} target={0.9} xFmt={(v) => (v === 0 ? "start" : "")}
              series={[{ pts: aciS(0), color: "var(--muted)", dots: false, label: "fixed" }, { pts: aciS(0.005), color: "var(--s3)", dots: false, label: "adaptive", width: 3 }]} />}
            <div className="legend"><span><i style={{ background: "var(--muted)" }} />fixed cut-off</span><span><i style={{ background: "var(--s3)" }} />adaptive cut-off (ACI)</span></div>
            <p className="small">A stream of pictures that gets noisier over time. ACI nudges the cut-off after every mistake, so coverage stays near 90%, by making sets bigger.</p>
          </div>
        </Reveal>
      </div>

      <div className="stack" style={{ gap: 28 }}>
        <Box kind="warnbox" tag="What we found">
          {R && <div className="drops">
            <div><span className="n good">{pct(clean("temp_aps"))}</span><span>clean pictures</span></div><span className="arr" aria-hidden>→</span>
            <div><span className="n bad">{pct(curve("temp_aps")[4][1])}</span><span>Adaptive, heavy corruption</span></div>
            <div><span className="n bad">{pct(curve("naive")[4][1])}</span><span>Simple, heavy corruption</span></div>
          </div>}
          <p>With heavy corruption, coverage of Adaptive sets drops from about 89% to about 69% (Simple sets: about 44%). The model gets worse, but the
            cut-off was set for clean pictures, so it doesn’t know. Even CIFAR-10.1, a new test set collected the same way, drops coverage to about 88%.</p>
        </Box>
        <Box kind="keyidea" tag="When the mix of classes changes">
          <p>If tomorrow’s pictures are mostly cats, coverage becomes the cats’ coverage. Across many random mixes, Simple sets fell as low as
            <strong> {pct(Math.min(...ls.map((r) => Number(r.naive))))}</strong>, while the per-class rule with a safety margin stayed at or above
            <strong> {pct(Math.min(...ls.map((r) => Number(r.mondrian_pac))))}</strong>. Protecting the worst class also protects against this.</p>
        </Box>
      </div>
    </Chapter>
  );
}
