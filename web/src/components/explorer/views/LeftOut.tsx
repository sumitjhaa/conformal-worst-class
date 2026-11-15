"use client";
import { useMemo } from "react";
import { pct } from "@/lib/conformal";
import { loadResults, rows, SHORT_NAMES, useLoad } from "@/lib/data";
import { LineChart, Seg, Stat } from "../../ui";
import { ALL_DS, dsName, Figure, Loading, mname, Select, SERIES, Term, useQuery, VBars } from "../kit";

const COMPARE = ["temp_naive", "temp_aps", "mondrian", "clustered_pac"];

export default function LeftOut({ p }: { p: string }) {
  const { data: R, error } = useLoad(loadResults);
  const [ds, setDs] = useQuery<string>(`${p}ds`, "cifar100"), [m, setM] = useQuery<string>(`${p}m`, "temp_aps");
  const methods = useMemo(() => [...new Set(rows(R, "class_cov").filter((r) => r.dataset === ds).map((r) => String(r.method)))], [R, ds]);

  const cls = useMemo(() => rows(R, "class_cov").filter((r) => r.dataset === ds && r.method === m).map((r) => ({ c: Number(r.cls), v: Number(r.coverage) })).sort((a, b) => a.v - b.v), [R, ds, m]);
  const dec = useMemo(() => COMPARE.concat(COMPARE.includes(m) ? [] : [m]).map((mm, j) => ({ mm, color: SERIES[j], pts: rows(R, "deciles").filter((r) => r.dataset === ds && r.method === mm).map((r) => [Number(r.decile) + 1, Number(r.coverage)] as [number, number]).sort((a, b) => a[0] - b[0]) })).filter((s) => s.pts.length), [R, ds, m]);
  const sizes = useMemo(() => {
    const r = rows(R, "sizes").filter((x) => x.dataset === ds && x.method === m), tot = r.reduce((a, x) => a + Number(x.count), 0) || 1;
    return r.map((x) => ({ s: Number(x.size), f: Number(x.count) / tot })).sort((a, b) => a.s - b.s);
  }, [R, ds, m]);

  if (!R) return <Loading error={error} what="results" />;
  const nm = (c: number) => (ds === "mnist" ? `digit ${c}` : (SHORT_NAMES[ds.replace("_r18", "")] || [])[c] ?? `class ${c}`), worst = cls[0], below = cls.filter((c) => c.v < 0.9 - 0.02).length;
  const shownSizes = sizes.filter((s) => s.s <= 20), more = sizes.filter((s) => s.s > 20).reduce((a, s) => a + s.f, 0);
  return (
    <div className="stack" style={{ gap: 28 }}>
      <div className="stack xcontrols">
        <Seg label="Dataset" value={ds} onChange={setDs} options={ALL_DS.map((d) => [d, dsName(d)])} />
        <Select label="Rule" value={m} onChange={setM} options={methods.map((x) => [x, mname(x)])} />
        <p className="small muted">All numbers at α = 0.1 (target 90%), pooled over seeds, from the experiments.</p>
      </div>
      {cls.length > 0 && (
        <div className="grid3">
          <Stat k="Worst class" value={worst.v} format={(v) => pct(v)} tone={worst.v < 0.88 ? "bad" : "good"} s={nm(worst.c)} />
          <Stat k="Best class" value={cls[cls.length - 1].v} format={(v) => pct(v)} s={nm(cls[cls.length - 1].c)} color="var(--s1)" />
          <Stat k="Classes more than 2 pt under target" value={below} format={(v) => String(Math.round(v))} s={`of ${cls.length}`} tone={below ? "bad" : "good"} />
        </div>
      )}
      <Figure title={<><Term t="class-conditional coverage">Coverage of each class</Term>, lowest first</>} name={`classes-${ds}-${m}`} tour="classes"
        table={{ cols: ["class", "name", "coverage"], rows: cls.map((c) => [c.c, nm(c.c), c.v]) }}>
        <LineChart label="Per-class coverage sorted" xDom={[0, Math.max(1, cls.length - 1)]} yDom={[Math.min(0.7, Math.floor((worst?.v ?? 0.8) * 20) / 20), 1]} target={0.9} xFmt={(v) => String(Math.round(v) + 1)}
          series={[{ pts: cls.map((c, i) => [i, c.v]), color: "var(--accent)", dots: cls.length <= 100, width: 2 }]} />
      </Figure>
      <div className="grid2">
        <Figure title="Coverage from easiest to hardest pictures" name={`deciles-${ds}`} tour="deciles"
          table={{ cols: ["method", "decile", "coverage"], rows: dec.flatMap((s) => s.pts.map(([x, y]) => [s.mm, x, y])) }}
          note="Pictures split into 10 equal groups by how unsure the model is (1 = most sure).">
          <LineChart label="Coverage by difficulty decile" xDom={[1, 10]} yDom={[0.5, 1]} target={0.9} xTicks={[1, 4, 7, 10]} series={dec.map((s) => ({ pts: s.pts, color: s.color, dots: false }))} />
          <div className="legend">{dec.map((s) => <span key={s.mm}><i style={{ background: s.color }} />{mname(s.mm)}</span>)}</div>
        </Figure>
        <Figure title="How many labels per set" name={`sizes-${ds}-${m}`} tour="sizes" table={{ cols: ["size", "share"], rows: sizes.map((s) => [s.s, s.f]) }}
          note={more > 0 ? `${pct(more)} of sets have more than 20 labels (not drawn).` : undefined}>
          <VBars label="Set-size histogram" bars={shownSizes.map((s) => ({ x: String(s.s), v: s.f, color: s.s === 0 ? "var(--bad)" : s.s === 1 ? "var(--good)" : "var(--s1)" }))} yDom={[0, Math.max(0.1, ...shownSizes.map((s) => s.f))]} yFmt={(v) => pct(v, 0)} />
        </Figure>
      </div>
    </div>
  );
}
