"use client";
import { useEffect, useState } from "react";
import { decodeBlock, type Block, type RawBlock } from "./conformal";
import type { NetJson } from "./lenet";

const BASE = process.env.NEXT_PUBLIC_BASE_PATH || "";
export const asset = (p: string) => `${BASE}/${p}`;

export type DsName = "mnist" | "cifar10" | "cifar100";
export const DS_LABEL: Record<string, string> = { mnist: "MNIST (digits)", cifar10: "CIFAR-10 (photos)", cifar10_r18: "CIFAR-10 · ResNet-18", cifar100: "CIFAR-100 (100 classes)", imagenet: "ImageNet (1000 classes)" };
export const SHORT_NAMES: Record<string, string[]> = { cifar10: ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"] };

export type Dataset = {
  ds: DsName; K: number; classes: string[]; accuracy: number; clusters: number[];
  tuned: { T: number; delta: number; n_clusters: number };
  cal: Block; test: Block; gallery: { i: number; img: string }[];
  real: Record<string, Block>; train_n: number; design_n: number;
  ood: Record<string, RawBlock & { dist?: string }>; shift: Record<string, RawBlock>; testDist?: string;
};
type Row = Record<string, number | string>;
export type Results = Record<string, Row[]> & { config?: unknown };
export type Gallery = { K: number; methods: string[]; items: { img: string; true: string; top: [string, number][]; sets: Record<string, { covered: boolean; size: number; in: boolean[] }> }[] };

const cache: Record<string, Promise<unknown>> = {};
function once<T>(key: string, f: () => Promise<T>): Promise<T> { return (cache[key] ??= f()) as Promise<T>; }
const getJson = (p: string) => fetch(asset(p)).then((r) => { if (!r.ok) throw new Error(`${p}: ${r.status}`); return r.json(); });

export const loadResults = () => once("results", () => getJson("data/results.json") as Promise<Results>);
export const loadGallery = () => once("gallery", () => getJson("data/imagenet_gallery.json") as Promise<Gallery>);
export const loadMnistNet = () => once("mnist_net", () => getJson("data/mnist_net.json") as Promise<NetJson>);
export const loadDataset = (ds: DsName) => once(ds, async () => {
  const j = await getJson(`data/${ds}.json`), K = j.K || 10;
  return { ...j, ds, K, cal: decodeBlock(j.cal, 202, K), test: decodeBlock(j.test, 303, K),
    testDist: j.test.dist, real: Object.fromEntries(Object.entries((j.real || {}) as Record<string, RawBlock>).map(([k, b], i) => [k, decodeBlock(b, 700 + i, K)])) } as Dataset;
});

/** Load something once and re-render when it arrives; `error` is set if the file is missing. */
export function useLoad<T>(f: (() => Promise<T>) | null, deps: unknown[] = []) {
  const [state, set] = useState<{ data: T | null; error: string | null }>({ data: null, error: null });
  useEffect(() => {
    let live = true;
    if (!f) return;
    f().then((data) => live && set({ data, error: null })).catch((e) => live && set({ data: null, error: String(e) }));
    return () => { live = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return state;
}

const extras = new Map<string, Block>();
/** Decode an OOD or shifted block on first use (they stay base64 until an explorer asks for them). */
export function extraBlock(d: Dataset, kind: "ood" | "shift", name: string): Block | null {
  const raw = d[kind]?.[name], key = `${d.ds}/${kind}/${name}`;
  if (!raw) return null;
  if (!extras.has(key)) extras.set(key, decodeBlock(raw, 900 + extras.size, d.K));
  return extras.get(key)!;
}

export const rows = (R: Results | null, key: string) => ((R && R[key]) || []) as Row[];
export const num = (r: Row | undefined, k: string) => (r ? Number(r[k]) : NaN);
export function avg(list: Row[], k: string) { const v = list.map((r) => Number(r[k])).filter(Number.isFinite); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN; }
export const at = (R: Results | null, key: string, where: Record<string, string | number>) =>
  rows(R, key).filter((r) => Object.entries(where).every(([k, v]) => (typeof v === "number" ? Math.abs(Number(r[k]) - v) < 1e-9 : r[k] === v)));
