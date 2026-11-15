"use client";
import { useEffect, useId, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from "react";
import { calibrate, metrics, predictSets, type Method, type Params } from "@/lib/conformal";
import { DS_LABEL, loadDataset, SHORT_NAMES, useLoad, type Dataset, type DsName } from "@/lib/data";
import { TERMS } from "@/lib/glossary";
import { lin, Seg, StepSlider } from "../ui";

// ---------------------------------------------------------------- URL state
// Each explorer keeps its controls in the query string, so a view is a shareable link. `p` prefixes the keys:
// compare mode renders the same explorer twice, "" on the left and "b." on the right.
const listeners = new Set<() => void>();
const subscribe = (f: () => void) => { listeners.add(f); addEventListener("popstate", f); return () => { listeners.delete(f); removeEventListener("popstate", f); }; };
const search = () => location.search;

export function useQuery<T extends string | number>(key: string, def: T): [T, (v: T) => void] {
  const qs = useSyncExternalStore(subscribe, search, () => "");
  const raw = new URLSearchParams(qs).get(key);
  const value = (raw == null ? def : typeof def === "number" ? (Number.isFinite(+raw) ? +raw : def) : raw) as T;
  const set = (v: T) => {
    const u = new URLSearchParams(location.search);
    if (Object.is(v, def)) u.delete(key); else u.set(key, String(v)); // Object.is: a NaN default must match itself
    const q = u.toString();
    history.replaceState(null, "", location.pathname + (q ? `?${q}` : "") + location.hash);
    listeners.forEach((f) => f());
  };
  return [value, set];
}

// ---------------------------------------------------------------- glossary on hover / focus
export function Term({ t, children }: { t: string; children?: ReactNode }) {
  const id = useId(), def = TERMS[t.toLowerCase()];
  if (!def) return <>{children ?? t}</>;
  return <span className="term" tabIndex={0} aria-describedby={id}>{children ?? t}<span role="tooltip" id={id}>{def}</span></span>;
}

// ---------------------------------------------------------------- figure: chart + data table + CSV / PNG export
export type Table = { cols: string[]; rows: (string | number)[][] };
const cell = (v: string | number) => (typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(4)) : v);

function download(name: string, blob: Blob) {
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}
export function downloadCsv(name: string, t: Table) {
  const esc = (v: string | number) => { const s = cell(v); return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s; };
  download(`${name}.csv`, new Blob([[t.cols, ...t.rows].map((r) => r.map(esc).join(",")).join("\n")], { type: "text/csv" }));
}
const PAINT = ["fill", "stroke", "stroke-width", "stroke-dasharray", "opacity", "fill-opacity", "stroke-opacity", "font-family", "font-size", "font-weight"];
/** SVG → PNG. Colours come from CSS variables, which a standalone SVG cannot see, so computed paint is inlined first. */
export function downloadPng(name: string, svg: SVGSVGElement) {
  const clone = svg.cloneNode(true) as SVGSVGElement, src = [svg, ...svg.querySelectorAll("*")], dst = [clone, ...clone.querySelectorAll("*")];
  src.forEach((el, i) => { const cs = getComputedStyle(el), d = dst[i] as SVGElement; PAINT.forEach((p) => d.style.setProperty(p, cs.getPropertyValue(p))); });
  const vb = svg.viewBox.baseVal, w = vb.width || svg.clientWidth, h = vb.height || svg.clientHeight, s = 2;
  clone.setAttribute("width", String(w)); clone.setAttribute("height", String(h));
  const img = new Image();
  img.onload = () => {
    const c = document.createElement("canvas"); c.width = w * s; c.height = h * s;
    const g = c.getContext("2d")!; g.fillStyle = getComputedStyle(document.body).backgroundColor; g.fillRect(0, 0, c.width, c.height);
    g.drawImage(img, 0, 0, c.width, c.height); c.toBlob((b) => b && download(`${name}.png`, b));
  };
  img.src = "data:image/svg+xml;charset=utf-8," + encodeURIComponent(new XMLSerializer().serializeToString(clone));
}

export function DataTable({ t, max = 400 }: { t: Table; max?: number }) {
  return (
    <div className="scroll-x" style={{ maxHeight: 360 }}>
      <table className="t">
        <thead><tr>{t.cols.map((c) => <th key={c} className="n" scope="col">{c}</th>)}</tr></thead>
        <tbody>{t.rows.slice(0, max).map((r, i) => <tr key={i}>{r.map((v, j) => <td key={j} className={typeof v === "number" ? "n" : ""}>{cell(v)}</td>)}</tr>)}</tbody>
      </table>
      {t.rows.length > max && <p className="small muted">First {max} of {t.rows.length} rows shown; the CSV has all of them.</p>}
    </div>
  );
}

export function Figure({ title, name, table, children, tour, note }: { title: ReactNode; name: string; table?: Table; children: ReactNode; tour?: string; note?: ReactNode }) {
  const [show, setShow] = useState(false), [hasSvg, setHasSvg] = useState(false), ref = useRef<HTMLDivElement>(null);
  useEffect(() => setHasSvg(!!ref.current?.querySelector("svg.chart")), [children]);
  const png = () => { const svg = ref.current?.querySelector("svg.chart"); if (svg) downloadPng(name, svg as SVGSVGElement); };
  return (
    <figure className="xfig" data-tour={tour}>
      <figcaption className="xfig-h">
        <h3>{title}</h3>
        <span className="xfig-tools">
          {table && <button type="button" className="xbtn" aria-pressed={show} onClick={() => setShow(!show)}>{show ? "Chart" : "Table"}</button>}
          {table && <button type="button" className="xbtn" onClick={() => downloadCsv(name, table)}>CSV</button>}
          {hasSvg && <button type="button" className="xbtn" onClick={png}>PNG</button>}
        </span>
      </figcaption>
      <div ref={ref} hidden={show}>{children}</div>
      {show && table && <DataTable t={table} />}
      {note && <p className="small muted" style={{ marginTop: 8 }}>{note}</p>}
    </figure>
  );
}

// ---------------------------------------------------------------- extra charts (the line chart lives in ui.tsx)
const M = { l: 48, r: 14, t: 14, b: 32 };
const r2 = (v: number) => Math.round(v * 100) / 100;

/** Vertical bars over categories. */
export function VBars({ bars, yDom, label, h = 240, w = 560, yFmt = (v: number) => String(v), target }: { bars: { x: string; v: number; color?: string }[]; yDom: [number, number]; label: string; h?: number; w?: number; yFmt?: (v: number) => string; target?: number }) {
  const n = bars.length || 1, bw = (w - M.l - M.r) / n, Y = (v: number) => r2(lin(yDom[0], yDom[1], h - M.b, M.t)(v));
  const every = Math.ceil(n / 14), yt = [0, 1, 2, 3, 4].map((i) => yDom[0] + ((yDom[1] - yDom[0]) * i) / 4);
  return (
    <svg className="chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}>
      <g className="grid">{yt.map((t) => <line key={t} x1={M.l} x2={w - M.r} y1={Y(t)} y2={Y(t)} />)}</g>
      {yt.map((t) => <text key={t} x={M.l - 6} y={Y(t) + 4} textAnchor="end">{yFmt(t)}</text>)}
      {bars.map((b, i) => (
        <g key={b.x}>
          <rect x={r2(M.l + i * bw + bw * 0.12)} width={r2(bw * 0.76)} y={Math.min(Y(b.v), Y(yDom[0]))} height={Math.abs(Y(yDom[0]) - Y(b.v))} rx={3} fill={b.color || "var(--accent)"}><title>{`${b.x}: ${yFmt(b.v)}`}</title></rect>
          {i % every === 0 && <text x={r2(M.l + (i + 0.5) * bw)} y={h - M.b + 16} textAnchor="middle">{b.x}</text>}
        </g>
      ))}
      <line className="axis" x1={M.l} x2={w - M.r} y1={h - M.b} y2={h - M.b} />
      {target != null && <line x1={M.l} x2={w - M.r} y1={Y(target)} y2={Y(target)} stroke="var(--target)" strokeWidth={2} strokeDasharray="6 4" />}
    </svg>
  );
}

/** Dots only, for many points (no per-point animation). */
export function Scatter({ groups, xDom, yDom, label, xFmt = String, yFmt = String, w = 560, h = 320, target }: { groups: { color: string; pts: [number, number][]; r?: number; label?: string }[]; xDom: [number, number]; yDom: [number, number]; label: string; xFmt?: (v: number) => string; yFmt?: (v: number) => string; w?: number; h?: number; target?: number }) {
  const X = (v: number) => r2(lin(xDom[0], xDom[1], M.l, w - M.r)(v)), Y = (v: number) => r2(lin(yDom[0], yDom[1], h - M.b, M.t)(v));
  const t5 = (d: [number, number]) => [0, 1, 2, 3, 4].map((i) => d[0] + ((d[1] - d[0]) * i) / 4);
  return (
    <svg className="chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}>
      <g className="grid">{t5(yDom).map((t) => <line key={t} x1={M.l} x2={w - M.r} y1={Y(t)} y2={Y(t)} />)}</g>
      {t5(yDom).map((t) => <text key={t} x={M.l - 6} y={Y(t) + 4} textAnchor="end">{yFmt(t)}</text>)}
      {t5(xDom).map((t) => <text key={t} x={X(t)} y={h - M.b + 17} textAnchor="middle">{xFmt(t)}</text>)}
      <line className="axis" x1={M.l} x2={w - M.r} y1={h - M.b} y2={h - M.b} />
      {target != null && <line x1={M.l} x2={w - M.r} y1={Y(target)} y2={Y(target)} stroke="var(--target)" strokeWidth={2} strokeDasharray="6 4" />}
      {groups.map((g, gi) => <g key={gi} fill={g.color} opacity={0.75}>{g.pts.map(([x, y], i) => <circle key={i} cx={X(x)} cy={Y(Math.min(Math.max(y, yDom[0]), yDom[1]))} r={g.r ?? 2.6} />)}</g>)}
    </svg>
  );
}

/** Square heatmap; value / max maps to accent opacity (the tooltip shows the raw value). */
export function Heatmap({ names, value, label, max = 1 }: { names: string[]; value: (r: number, c: number) => number; label: string; max?: number }) {
  const n = names.length, lab = 78, cs = Math.max(14, Math.min(40, Math.floor(480 / n))), w = lab + n * cs, h = lab + n * cs;
  return (
    <svg className="chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label} style={{ maxWidth: w * 1.4 }}>
      {names.map((nm, i) => <text key={`r${i}`} x={lab - 6} y={lab + i * cs + cs / 2 + 4} textAnchor="end">{nm.slice(0, 11)}</text>)}
      {names.map((nm, i) => <text key={`c${i}`} transform={`translate(${lab + i * cs + cs / 2 + 4},${lab - 6}) rotate(-60)`}>{nm.slice(0, 11)}</text>)}
      {names.map((_, r) => names.map((__, c) => {
        const v = value(r, c);
        return <rect key={`${r}-${c}`} x={lab + c * cs} y={lab + r * cs} width={cs - 1.5} height={cs - 1.5} rx={2} fill={r === c ? "var(--surface-2)" : "var(--accent)"} fillOpacity={r === c ? 1 : 0.06 + 0.94 * Math.min(1, v / (max || 1))}><title>{`${names[r]} → ${names[c]}: ${(v * 100).toFixed(1)}%`}</title></rect>;
      }))}
    </svg>
  );
}

// ---------------------------------------------------------------- shared controls + names
export const LAB_DS: [DsName, string][] = [["mnist", "Digits (MNIST)"], ["cifar10", "CIFAR-10"], ["cifar100", "CIFAR-100"]];
export const ALL_DS = ["mnist", "cifar10", "cifar10_r18", "cifar100", "imagenet"];
export const ALPHAS = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3];
export const LIVE_METHODS: [Method, string][] = [["top1", "One guess"], ["naive", "Simple"], ["aps", "APS"], ["raps", "RAPS"], ["mondrian", "Per class"], ["clustered_pac", "Clustered-PAC"]];
/** Display names for the method keys in results.json. */
export const METHOD_NAME: Record<string, string> = {
  top1: "One guess", naive: "Simple (raw)", temp_naive: "Simple", aps: "APS (raw)", temp_aps: "APS", raps: "RAPS", saps: "SAPS", margin: "Margin",
  dist_naive: "Simple + distance", dist_aps: "APS + distance", conftr: "Learned score", mondrian: "Per class", mondrian_pac: "Per class + PAC",
  clustered: "Clustered", clustered_pac: "Clustered-PAC",
};
export const mname = (m: string) => METHOD_NAME[m] || m;
export const dsName = (ds: string) => DS_LABEL[ds] || ds;
export const SERIES = ["var(--s1)", "var(--good)", "var(--violet)", "var(--s2)", "var(--bad)", "var(--s3)", "var(--s4)", "var(--target)"];

export function DsPick({ value, onChange, options = LAB_DS }: { value: string; onChange: (v: DsName) => void; options?: [string, string][] }) {
  return <Seg label="Dataset" value={value as DsName} onChange={onChange} options={options as [DsName, ReactNode][]} />;
}
export function Select({ label, value, onChange, options }: { label: string; value: string; onChange: (v: string) => void; options: [string, string][] }) {
  return <label className="xsel"><span>{label}</span><select value={value} onChange={(e) => onChange(e.target.value)}>{options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}</select></label>;
}
export function AlphaPick({ value, onChange, alphas = ALPHAS }: { value: number; onChange: (v: number) => void; alphas?: number[] }) {
  const i = Math.max(0, alphas.findIndex((a) => Math.abs(a - value) < 1e-9));
  return <StepSlider label="α (allowed error)" index={i} onChange={(j) => onChange(alphas[j])} display={`${value.toFixed(2)} → target ${Math.round((1 - value) * 100)}%`} stops={alphas.map((v) => v.toFixed(2))} width={420} />;
}

/** Dataset + method + α from the URL, the dataset loaded, and the calibrated sets on its test block. */
export function useLab(p: string, defs: { ds?: DsName; m?: Method; a?: number } = {}) {
  const [ds, setDs] = useQuery<string>(`${p}ds`, defs.ds ?? "cifar10");
  const [method, setMethod] = useQuery<string>(`${p}m`, defs.m ?? "aps");
  const [alpha, setAlpha] = useQuery<number>(`${p}a`, defs.a ?? 0.1);
  const { data: d, error } = useLoad(() => loadDataset(ds as DsName), [ds]);
  const out = useMemo(() => {
    if (!d || d.ds !== ds) return null;
    const P = params(d, method as Method, alpha), cal = calibrate(d.cal, P, d.clusters), C = predictSets(d.test, P, cal);
    return { P, cal, C, m: metrics(d.test, C), names: names(d) };
  }, [d, ds, method, alpha]);
  return { ds: ds as DsName, setDs, method: method as Method, setMethod, alpha, setAlpha, d: d && d.ds === ds ? d : null, error, out };
}
export const params = (d: Dataset, method: Method, alpha: number): Params => ({ method, alpha, T: d.tuned.T, delta: d.tuned.delta ?? 0.1, raps: (d.tuned as { raps?: Params["raps"] }).raps });
export const names = (d: Dataset) => SHORT_NAMES[d.ds] || d.classes;

export function Loading({ error, what = "data" }: { error?: string | null; what?: string }) {
  return error ? <p className="loading" role="alert">Could not load {what}: {error}</p> : <p className="loading">loading {what}…</p>;
}
