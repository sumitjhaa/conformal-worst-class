"use client";
import { useMemo } from "react";
import { metrics, pct, predictSets, setOf, type Block } from "@/lib/conformal";
import { extraBlock, loadResults, rows, useLoad } from "@/lib/data";
import { ImgCard, LineChart, Seg } from "../../ui";
import { ALL_DS, AlphaPick, DsPick, dsName, Figure, LIVE_METHODS, Loading, mname, Select, SERIES, Term, useLab, useQuery } from "../kit";

const TABS: [string, string][] = [["live", "Live"], ["c", "CIFAR-10-C"], ["syn", "Synthetic"], ["w", "Weighted"], ["label", "Label shift"]];
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / (v.length || 1);

function Live({ p }: { p: string }) {
  const L = useLab(p), { d, out } = L;
  const [pick, setPick] = useQuery<string>(`${p}set`, "");
  const sets = useMemo(() => {
    if (!d) return [];
    const list: [string, Block][] = [["clean test", d.test], ...Object.entries(d.real)];
    for (const k of Object.keys(d.shift || {})) { const b = extraBlock(d, "shift", k); if (b) list.push([d.ds === "mnist" ? `rotated ${k}°` : `noise σ ${k}`, b]); }
    return list;
  }, [d]);
  const res = useMemo(() => (out ? sets.map(([name, b]) => { const C = predictSets(b, out.P, out.cal), m = metrics(b, C); let acc = 0; for (let i = 0; i < b.n; i++) { let best = 0; for (let k = 1; k < b.K; k++) if (b.logits[i * b.K + k] > b.logits[i * b.K + best]) best = k; acc += best === b.labels![i] ? 1 : 0; } return { name, b, C, m, acc: acc / b.n }; }) : []), [sets, out]);
  if (!d || !out) return <Loading error={L.error} what={L.ds} />;
  const cur = res.find((r) => r.name === pick) ?? res[1] ?? res[0];
  return (
    <div className="stack" style={{ gap: 24 }} data-tour="live">
      <div className="stack xcontrols">
        <div className="row" style={{ justifyContent: "space-between" }}><DsPick value={L.ds} onChange={(v) => { L.setDs(v); setPick(""); }} /><AlphaPick value={L.alpha} onChange={L.setAlpha} /></div>
        <Seg label="Rule" value={L.method} onChange={L.setMethod} options={LIVE_METHODS} />
      </div>
      <p className="small">The cut-off is calibrated on clean pictures, then applied unchanged to shifted ones: <Term t="exchangeable">exchangeability</Term> no longer holds, so the 90% promise can break.</p>
      <Figure title="Coverage on each kind of picture" name={`shift-live-${L.ds}-${L.method}`}
        table={{ cols: ["set", "n", "accuracy", "coverage", "mean_size"], rows: res.map((r) => [r.name, r.b.n, r.acc, r.m.cov, r.m.mean]) }}>
        <div className="scroll-x"><table className="t">
          <thead><tr><th>Pictures</th><th className="n">n</th><th className="n">Accuracy</th><th className="n">Coverage</th><th className="n">Mean size</th></tr></thead>
          <tbody>{res.map((r) => (
            <tr key={r.name} aria-current={r === cur ? "true" : undefined} onClick={() => setPick(r.name)} style={{ cursor: "pointer" }}>
              <th scope="row"><button type="button" className="xlinkbtn" onClick={() => setPick(r.name)}>{r.name}</button></th><td className="n">{r.b.n}</td><td className="n">{pct(r.acc)}</td>
              <td className="n" style={{ color: r.m.cov < 1 - L.alpha - 0.02 ? "var(--bad)" : undefined }}>{pct(r.m.cov)}</td><td className="n">{r.m.mean.toFixed(2)}</td>
            </tr>))}</tbody>
        </table></div>
      </Figure>
      {cur?.b.imgs && (
        <div>
          <h3 style={{ marginBottom: 12 }}>{cur.name}: sample pictures</h3>
          <div className="gallery">{cur.b.imgs.slice(0, 8).map((img, i) => { const y = cur.b.labels![i], s = setOf(cur.C, i, d.K); return <ImgCard key={i} img={img} trueName={out.names[y]} covered={s.includes(y)} labels={s.map((k) => out.names[k])} />; })}</div>
        </div>
      )}
    </div>
  );
}

export default function ShiftLab({ p }: { p: string }) {
  const [tab, setTab] = useQuery<string>(`${p}tab`, "live");
  const { data: R, error } = useLoad(loadResults);
  const [ds, setDs] = useQuery<string>(`${p}rds`, "cifar10"), [cor, setCor] = useQuery<string>(`${p}cor`, "gaussian_noise");

  const body = () => {
    if (tab === "live") return <Live p={p} />;
    if (!R) return <Loading error={error} what="results" />;
    if (tab === "c") {
      const rs = rows(R, "real_shift").filter((r) => r.dataset === "cifar10" && r.family === "c" && r.corruption === cor);
      const ms = [...new Set(rs.map((r) => String(r.method)))], cors = [...new Set(rows(R, "real_shift").filter((r) => r.family === "c").map((r) => String(r.corruption)))];
      const acc = [1, 2, 3, 4, 5].map((s) => [s, mean(rs.filter((r) => Number(r.severity) === s).map((r) => Number(r.accuracy)))] as [number, number]);
      return (
        <Figure title={`CIFAR-10-C · ${cor.replace(/_/g, " ")}`} name={`cifar10c-${cor}`} table={{ cols: ["method", "severity", "accuracy", "coverage", "mean_size"], rows: rs.map((r) => [String(r.method), Number(r.severity), Number(r.accuracy), Number(r.coverage), Number(r.mean_size)]) }}>
          <Select label="Corruption" value={cor} onChange={setCor} options={cors.map((c) => [c, c.replace(/_/g, " ")])} />
          <LineChart label="Coverage against corruption severity" xDom={[1, 5]} xTicks={[1, 2, 3, 4, 5]} yDom={[0.3, 1]} target={0.9}
            series={[...ms.map((m, j) => ({ pts: rs.filter((r) => r.method === m).map((r) => [Number(r.severity), Number(r.coverage)] as [number, number]).sort((a, b) => a[0] - b[0]), color: SERIES[j % SERIES.length], dots: false })), { pts: acc, color: "var(--ink-2)", dash: true, dots: false }]} />
          <div className="legend">{ms.map((m, j) => <span key={m}><i style={{ background: SERIES[j % SERIES.length] }} />{mname(m)}</span>)}<span><i style={{ background: "var(--ink-2)" }} />model accuracy</span></div>
        </Figure>
      );
    }
    if (tab === "syn") {
      const rs = rows(R, "shift").filter((r) => r.dataset === ds), ms = [...new Set(rs.map((r) => String(r.method)))], sev = [...new Set(rs.map((r) => Number(r.severity)))].sort((a, b) => a - b);
      return (
        <Figure title={`${dsName(ds)} · ${ds === "mnist" ? "rotation (degrees)" : "Gaussian noise σ"}`} name={`shift-${ds}`} table={{ cols: ["method", "severity", "coverage", "mean_size"], rows: rs.map((r) => [String(r.method), Number(r.severity), Number(r.coverage), Number(r.mean_size)]) }}>
          <Seg label="Dataset" value={ds} onChange={setDs} options={ALL_DS.map((d) => [d, dsName(d)])} />
          <LineChart label="Coverage against shift severity" xDom={[sev[0] ?? 0, sev[sev.length - 1] ?? 1]} xTicks={sev} yDom={[0.2, 1]} target={0.9}
            series={ms.map((m, j) => ({ pts: rs.filter((r) => r.method === m).map((r) => [Number(r.severity), Number(r.coverage)] as [number, number]).sort((a, b) => a[0] - b[0]), color: SERIES[j % SERIES.length], dots: false }))} />
          <div className="legend">{ms.map((m, j) => <span key={m}><i style={{ background: SERIES[j % SERIES.length] }} />{mname(m)}</span>)}</div>
        </Figure>
      );
    }
    if (tab === "w") {
      const rs = rows(R, "weighted").filter((r) => r.dataset === ds), dsets = [...new Set(rows(R, "weighted").map((r) => String(r.dataset)))];
      return (
        <Figure title="Standard vs importance-weighted calibration" name={`weighted-${ds}`} note="Weighted calibration reweights calibration pictures to look like the shifted ones; ESS is the effective number of calibration pictures left."
          table={{ cols: ["set", "cov_standard", "cov_weighted", "size_standard", "size_weighted", "ess_cal"], rows: rs.map((r) => [String(r.set), Number(r.cov_standard), Number(r.cov_weighted), Number(r.size_standard), Number(r.size_weighted), Number(r.ess_cal)]) }}>
          <Seg label="Dataset" value={ds} onChange={setDs} options={dsets.map((d) => [d, dsName(d)])} />
          <div className="hbars" role="list">{rs.map((r) => (
            <div key={String(r.set)} role="listitem" className="xpair">
              <span className="hbar-name">{String(r.set).replace(/^c\//, "").replace(/_/g, " ")}</span>
              <span className="xpair-bars">
                <span className="hbar-track"><span className="hbar-fill" style={{ width: pct(Number(r.cov_standard)), background: "var(--s2)" }} /><span className="hbar-target" style={{ left: "90%" }} /></span>
                <span className="hbar-track"><span className="hbar-fill" style={{ width: pct(Number(r.cov_weighted)), background: "var(--good)" }} /><span className="hbar-target" style={{ left: "90%" }} /></span>
              </span>
              <span className="hbar-val">{pct(Number(r.cov_standard))}<br />{pct(Number(r.cov_weighted))}</span>
            </div>))}</div>
          <div className="legend"><span><i style={{ background: "var(--s2)" }} />standard</span><span><i style={{ background: "var(--good)" }} />weighted</span></div>
        </Figure>
      );
    }
    const rs = rows(R, "label_shift").filter((r) => r.dataset === ds), dsets = [...new Set(rows(R, "label_shift").map((r) => String(r.dataset)))], betas = [...new Set(rs.map((r) => Number(r.beta)))].sort((a, b) => a - b);
    const keys: [string, string][] = [["naive", "Simple"], ["temp_aps", "APS"], ["mondrian", "Per class"], ["mondrian_pac", "Per class + PAC"], ["ls_oracle", "Reweighted (true mix)"], ["ls_bbse", "Reweighted (estimated)"], ["worst_case_bound", "Worst-case bound"]];
    const line = (k: string) => betas.map((b) => [b, mean(rs.filter((r) => Number(r.beta) === b).map((r) => Number(r[k])))] as [number, number]);
    return (
      <Figure title={<><Term t="label shift">Label shift</Term>: the class mix changes</>} name={`label-shift-${ds}`} note="β is the concentration of the new class mix: small β = a very lopsided mix, large β = close to the original. Means over seeds and draws."
        table={{ cols: ["beta", ...keys.map((k) => k[0])], rows: betas.map((b, i) => [b, ...keys.map((k) => line(k[0])[i][1])]) }}>
        <Seg label="Dataset" value={ds} onChange={setDs} options={dsets.map((d) => [d, dsName(d)])} />
        <LineChart label="Coverage against label-shift strength" xlog xDom={[betas[0] ?? 0.1, betas[betas.length - 1] ?? 10]} xTicks={betas} yDom={[0.7, 1]} target={0.9}
          series={keys.map(([k], j) => ({ pts: line(k), color: SERIES[j % SERIES.length], dash: k === "worst_case_bound" }))} />
        <div className="legend">{keys.map(([k, l], j) => <span key={k}><i style={{ background: SERIES[j % SERIES.length] }} />{l}</span>)}</div>
      </Figure>
    );
  };
  return (
    <div className="stack" style={{ gap: 24 }}>
      <div data-tour="tabs"><Seg label="Kind of shift" value={tab} onChange={setTab} options={TABS} /></div>
      {body()}
    </div>
  );
}
