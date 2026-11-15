"use client";
import { useMemo } from "react";
import { calibrate, pct, predictSets, probs, scoreMatrix, setOf, threshold, type Method } from "@/lib/conformal";
import { Seg } from "../../ui";
import { AlphaPick, DsPick, Figure, LIVE_METHODS, Loading, params, Term, useLab, useQuery } from "../kit";

const scoreKind = (m: Method): Method => (m === "mondrian" || m === "clustered_pac" ? "aps" : m);

export default function Inspector({ p }: { p: string }) {
  const L = useLab(p), { d, out } = L;
  const [i0, setI] = useQuery<number>(`${p}i`, -1);
  const i = d ? (d.gallery.some((g) => g.i === i0) ? i0 : d.gallery[0].i) : -1;

  const view = useMemo(() => {
    if (!d || !out || i < 0) return null;
    const K = d.K, y = d.test.labels![i], pr = probs(d.test, d.tuned.T), S = L.method === "top1" ? null : scoreMatrix(d.test, scoreKind(L.method), d.tuned.T, out.P.raps);
    const order = Array.from({ length: K }, (_, k) => k).sort((a, b) => pr[i * K + b] - pr[i * K + a]).slice(0, Math.min(K, 12));
    const labels = order.map((k) => ({ k, p: pr[i * K + k], s: S ? S[i * K + k] : NaN, q: threshold(out.cal, k), inSet: !!out.C[i * K + k] }));
    return { y, labels };
  }, [d, out, i, L.method]);

  // the same picture under every rule (calibrated once per dataset and α)
  const rules = useMemo(() => {
    if (!d) return null;
    return LIVE_METHODS.map(([m, name]) => { const P = params(d, m, L.alpha), C = predictSets(d.test, P, calibrate(d.cal, P, d.clusters)); return { m, name, C }; });
  }, [d, L.alpha]);

  if (!d || !out || !view) return <div className="stack"><DsPick value={L.ds} onChange={L.setDs} /><Loading error={L.error} what={L.ds} /></div>;
  const img = d.gallery.find((g) => g.i === i)!.img, sMax = Math.max(1, ...view.labels.map((l) => (Number.isFinite(l.s) ? l.s : 0)), ...view.labels.map((l) => (Number.isFinite(l.q) ? l.q : 0)));
  return (
    <div className="stack" style={{ gap: 28 }}>
      <div className="stack xcontrols">
        <div className="row" style={{ justifyContent: "space-between" }}><DsPick value={L.ds} onChange={(v) => { L.setDs(v); setI(-1); }} /><AlphaPick value={L.alpha} onChange={L.setAlpha} /></div>
        <Seg label="Rule" value={L.method} onChange={L.setMethod} options={LIVE_METHODS} />
      </div>
      <div className="xstrip" data-tour="picker" role="listbox" aria-label="Pick a test picture">
        {d.gallery.map((g) => (
          <button key={g.i} type="button" role="option" aria-selected={g.i === i} onClick={() => setI(g.i)} title={out.names[d.test.labels![g.i]]}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={g.img} alt={out.names[d.test.labels![g.i]]} />
          </button>
        ))}
      </div>
      <div className="xinspect">
        <div className="stack">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className="xbig" src={img} alt={`test picture, true label ${out.names[view.y]}`} />
          <p><strong>True label:</strong> {out.names[view.y]}</p>
          <p><strong>Set:</strong> {"{ "}{setOf(out.C, i, d.K).map((k) => out.names[k]).join(", ") || "empty"}{" }"} <span className={`setbadge ${out.C[i * d.K + view.y] ? "ok" : "miss"}`}>{out.C[i * d.K + view.y] ? "covered" : "missed"}</span></p>
          <p className="small muted">Picture #{i} of {d.test.n.toLocaleString()} test pictures.</p>
        </div>
        <Figure title="Each label’s score against its cut-off" name={`inspector-${L.ds}-${i}`} tour="scores"
          table={{ cols: ["label", "probability", "score", "cutoff", "in_set"], rows: view.labels.map((l) => [out.names[l.k], l.p, l.s, l.q, l.inSet ? "yes" : "no"]) }}
          note={<>Probability after temperature T = {d.tuned.T.toFixed(2)}. A label joins when its <Term t="surprise (nonconformity) score">score</Term> is at or below the cut-off (the tick). {L.method === "top1" && "One guess has no scores: it takes the most likely label."}</>}>
          <div className="xscores" role="list">
            {view.labels.map((l) => (
              <div key={l.k} role="listitem" className={`xscore ${l.inSet ? "in" : ""} ${l.k === view.y ? "truth" : ""}`}>
                <span className="xscore-name">{out.names[l.k]}{l.k === view.y && " ✓"}</span>
                <span className="xscore-p mono">{pct(l.p)}</span>
                <span className="xscore-track">
                  {Number.isFinite(l.s) && <i style={{ width: `${(Math.min(l.s, sMax) / sMax) * 100}%` }} />}
                  {Number.isFinite(l.q) && <b style={{ left: `${(Math.min(l.q, sMax) / sMax) * 100}%` }} title={`cut-off ${l.q.toFixed(3)}`} />}
                </span>
                <span className="mono small">{Number.isFinite(l.s) ? l.s.toFixed(3) : "—"}</span>
              </div>
            ))}
          </div>
        </Figure>
      </div>
      {rules && (
        <div data-tour="rules" className="stack">
          <h3>This picture under every rule</h3>
          <div className="scroll-x"><table className="t">
            <thead><tr><th>Rule</th><th>Set</th><th className="n">Size</th><th>Truth inside?</th></tr></thead>
            <tbody>{rules.map((r) => { const set = setOf(r.C, i, d.K); return (
              <tr key={r.m} aria-current={r.m === L.method ? "true" : undefined}><th scope="row">{r.name}</th><td>{set.map((k) => <span key={k} className={`chip ${k === view.y ? "true" : ""}`} style={{ marginRight: 4 }}>{out.names[k]}</span>)}{!set.length && <span className="chip empty">empty</span>}</td><td className="n">{set.length}</td><td>{set.includes(view.y) ? "yes" : "no"}</td></tr>
            ); })}</tbody>
          </table></div>
        </div>
      )}
    </div>
  );
}
