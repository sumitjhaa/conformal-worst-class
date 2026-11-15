"use client";
import { useMemo } from "react";
import { calibrate, metrics, pct, predictSets, setOf, type Method } from "@/lib/conformal";
import { Stat } from "../../ui";
import { AlphaPick, DataTable, DsPick, LIVE_METHODS, Loading, params, Select, useLab, useQuery } from "../kit";

export default function Diff({ p }: { p: string }) {
  const L = useLab(p, { m: "naive" }), d = L.d;
  const [mb, setMb] = useQuery<string>(`${p}mb`, "aps");
  const ma = L.method, name = (m: string) => LIVE_METHODS.find((x) => x[0] === m)?.[1] ?? m;

  const r = useMemo(() => {
    if (!d) return null;
    const run = (m: Method) => { const P = params(d, m, L.alpha); return predictSets(d.test, P, calibrate(d.cal, P, d.clusters)); };
    const A = run(ma), B = run(mb as Method), K = d.K, y = d.test.labels!;
    let differ = 0, added = 0, removed = 0, rescued = 0, lost = 0;
    const diffIdx = new Set<number>();
    for (let i = 0; i < d.test.n; i++) {
      let dif = false;
      for (let k = 0; k < K; k++) {
        const a = A[i * K + k], b = B[i * K + k];
        if (a === b) continue;
        dif = true;
        if (b) { added++; if (k === y[i]) rescued++; } else { removed++; if (k === y[i]) lost++; }
      }
      if (dif) { differ++; diffIdx.add(i); }
    }
    const mA = metrics(d.test, A), mB = metrics(d.test, B);
    const cls = mA.classCov.map((c, k) => ({ k, a: c, b: mB.classCov[k] })).sort((x, z) => Math.abs(z.b - z.a) - Math.abs(x.b - x.a));
    return { A, B, differ, added, removed, rescued, lost, diffIdx, mA, mB, cls };
  }, [d, ma, mb, L.alpha]);

  return (
    <div className="stack" style={{ gap: 28 }}>
      <div className="stack xcontrols" data-tour="controls">
        <div className="row" style={{ justifyContent: "space-between" }}><DsPick value={L.ds} onChange={L.setDs} /><AlphaPick value={L.alpha} onChange={L.setAlpha} /></div>
        <div className="row">
          <Select label="Rule A" value={ma} onChange={(v) => L.setMethod(v)} options={LIVE_METHODS} />
          <Select label="Rule B" value={mb} onChange={setMb} options={LIVE_METHODS} />
        </div>
      </div>
      {!r || !d || !L.out ? <Loading error={L.error} what={L.ds} /> : (
        <>
          <div className="grid4" data-tour="counts">
            <Stat k="Pictures whose sets differ" value={r.differ / d.test.n} format={(v) => pct(v)} s={`${r.differ.toLocaleString()} of ${d.test.n.toLocaleString()}`} color="var(--s1)" />
            <Stat k={`Labels ${name(mb)} adds`} value={r.added} format={(v) => Math.round(v).toLocaleString()} s={`${r.rescued.toLocaleString()} were the true label`} tone="good" />
            <Stat k={`Labels ${name(mb)} drops`} value={r.removed} format={(v) => Math.round(v).toLocaleString()} s={`${r.lost.toLocaleString()} were the true label`} tone={r.lost ? "bad" : "good"} />
            <Stat k="Coverage A → B" value={r.mB.cov - r.mA.cov} format={(v) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)} pt`} s={`size ${r.mA.mean.toFixed(2)} → ${r.mB.mean.toFixed(2)}`} color="var(--violet)" />
          </div>
          <p className="small">{r.added ? <>Of every 100 labels {name(mb)} adds, <b>{Math.round((100 * r.rescued) / r.added)}</b> are the truth; the rest only make sets longer.</> : `${name(mb)} never adds a label ${name(ma)} lacks.`}</p>
          <div data-tour="gallery">
            <h3 style={{ marginBottom: 12 }}>Sample pictures where they disagree</h3>
            <div className="xdiff">
              {d.gallery.filter((g) => r.diffIdx.has(g.i)).slice(0, 18).map((g) => {
                const y = d.test.labels![g.i], a = setOf(r.A, g.i, d.K), b = setOf(r.B, g.i, d.K), nm = L.out!.names;
                const chips = (s: number[], other: number[]) => s.map((k) => <span key={k} className={`chip ${k === y ? "true" : ""} ${other.includes(k) ? "" : "xnew"}`}>{nm[k]}</span>);
                return (
                  <div key={g.i} className="xdiff-row">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={g.img} alt={nm[y]} />
                    <div><span className="small muted">A</span> <span className="chips">{a.length ? chips(a, b) : <span className="chip empty">empty</span>}</span></div>
                    <div><span className="small muted">B</span> <span className="chips">{b.length ? chips(b, a) : <span className="chip empty">empty</span>}</span></div>
                  </div>
                );
              })}
            </div>
            <p className="small muted">Outlined chips appear in only one of the two sets; the filled chip is the true label. Only the {d.gallery.length} pictures with stored images can be shown.</p>
          </div>
          <div className="stack">
            <h3>Classes that change most</h3>
            <DataTable t={{ cols: ["class", "coverage A", "coverage B", "change (pt)"], rows: r.cls.slice(0, 15).map((c) => [L.out!.names[c.k], pct(c.a), pct(c.b), +((c.b - c.a) * 100).toFixed(2)]) }} />
          </div>
        </>
      )}
    </div>
  );
}
