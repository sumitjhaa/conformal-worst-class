"use client";
import { useEffect, useMemo, useRef, useState } from "react";
import { calibrate, pct, predictSets, probs, setOf, type Block, type Method } from "@/lib/conformal";
import { loadDataset, loadMnistNet, useLoad } from "@/lib/data";
import { lenet, loadNet } from "@/lib/lenet";
import { HBars, Seg } from "../../ui";
import { AlphaPick, Loading, params, useQuery } from "../kit";

const SIDE = 280, METHODS: [Method, string][] = [["naive", "Simple"], ["aps", "APS"], ["mondrian", "Per class"], ["clustered_pac", "Clustered-PAC"]];

/** Canvas → 28x28 the way MNIST was made: crop to the ink, fit the longer side to 20 px, centre by centre of mass. */
function toMnist(src: HTMLCanvasElement): Float32Array | null {
  const g = src.getContext("2d")!, { data } = g.getImageData(0, 0, SIDE, SIDE);
  let x0 = SIDE, y0 = SIDE, x1 = -1, y1 = -1;
  for (let y = 0; y < SIDE; y++) for (let x = 0; x < SIDE; x++) if (data[(y * SIDE + x) * 4] > 30) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  if (x1 < 0) return null;
  const w = x1 - x0 + 1, h = y1 - y0 + 1, s = 20 / Math.max(w, h), tw = Math.max(1, Math.round(w * s)), th = Math.max(1, Math.round(h * s));
  const small = document.createElement("canvas"); small.width = tw; small.height = th;
  const sg = small.getContext("2d")!; sg.imageSmoothingQuality = "high"; sg.drawImage(src, x0, y0, w, h, 0, 0, tw, th);
  const px = sg.getImageData(0, 0, tw, th).data;
  let m = 0, cx = 0, cy = 0;
  for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) { const v = px[(y * tw + x) * 4]; m += v; cx += v * x; cy += v * y; }
  const ox = Math.round(14 - cx / m), oy = Math.round(14 - cy / m), out = new Float32Array(784);
  for (let y = 0; y < th; y++) for (let x = 0; x < tw; x++) { const X = x + ox, Y = y + oy; if (X >= 0 && X < 28 && Y >= 0 && Y < 28) out[Y * 28 + X] = px[(y * tw + x) * 4]; }
  return out;
}

export default function Draw({ p }: { p: string }) {
  const net = useLoad(loadMnistNet), ds = useLoad(() => loadDataset("mnist"));
  const model = useMemo(() => (net.data ? loadNet(net.data) : null), [net.data]);
  const [method, setMethod] = useQuery<string>(`${p}m`, "aps"), [alpha, setAlpha] = useQuery<number>(`${p}a`, 0.1);
  const cv = useRef<HTMLCanvasElement>(null), last = useRef<[number, number] | null>(null), timer = useRef(0);
  const [z, setZ] = useState<number[] | null>(null), [thumb, setThumb] = useState<string | null>(null);

  const cal = useMemo(() => (ds.data ? { P: params(ds.data, method as Method, alpha), d: ds.data } : null), [ds.data, method, alpha]);
  const calib = useMemo(() => (cal ? calibrate(cal.d.cal, cal.P, cal.d.clusters) : null), [cal]);

  const blank = () => { const g = cv.current?.getContext("2d"); if (g) { g.fillStyle = "#000"; g.fillRect(0, 0, SIDE, SIDE); } };
  const clear = () => { blank(); setZ(null); setThumb(null); };
  useEffect(blank, [net.data, ds.data]); // the canvas mounts once both have loaded

  const predict = () => {
    if (!model || !cv.current) return;
    const x = toMnist(cv.current);
    if (!x) { setZ(null); setThumb(null); return; }
    setZ(lenet(model, x));
    const t = document.createElement("canvas"); t.width = t.height = 28; const tg = t.getContext("2d")!, im = tg.createImageData(28, 28);
    x.forEach((v, i) => { im.data.set([v, v, v, 255], i * 4); }); tg.putImageData(im, 0, 0); setThumb(t.toDataURL());
  };
  const schedule = () => { clearTimeout(timer.current); timer.current = window.setTimeout(predict, 60); };

  const pos = (e: React.PointerEvent): [number, number] => { const r = cv.current!.getBoundingClientRect(); return [((e.clientX - r.left) / r.width) * SIDE, ((e.clientY - r.top) / r.height) * SIDE]; };
  const stroke = (a: [number, number], b: [number, number]) => { const g = cv.current!.getContext("2d")!; g.strokeStyle = "#fff"; g.lineWidth = 20; g.lineCap = g.lineJoin = "round"; g.beginPath(); g.moveTo(...a); g.lineTo(...b); g.stroke(); };
  const loadDigit = () => {
    const gal = ds.data?.gallery; if (!gal) return;
    const img = new Image();
    img.onload = () => { const g = cv.current!.getContext("2d")!; g.imageSmoothingEnabled = false; g.drawImage(img, 0, 0, SIDE, SIDE); predict(); };
    img.src = gal[Math.floor(Math.random() * gal.length)].img;
  };

  const answer = useMemo(() => {
    if (!z || !cal || !calib) return null;
    const b: Block = { n: 1, K: 10, logits: Float32Array.from(z), labels: null, imgs: null, salt: Math.round(Math.abs(z[0]) * 1e3) };
    return { set: setOf(predictSets(b, cal.P, calib), 0, 10), p: Array.from(probs(b, cal.d.tuned.T)) };
  }, [z, cal, calib]);

  if (!model || !ds.data) return <Loading error={net.error || ds.error} what="the MNIST network (1.4 MB)" />;
  return (
    <div className="stack" style={{ gap: 24 }}>
      <div className="row" style={{ justifyContent: "space-between" }}><Seg label="Rule" value={method} onChange={setMethod} options={METHODS} /><AlphaPick value={alpha} onChange={setAlpha} /></div>
      <div className="xdraw">
        <div className="stack" data-tour="canvas">
          <canvas ref={cv} width={SIDE} height={SIDE} className="xcanvas" aria-label="Drawing area: draw a digit with mouse, finger or pen" role="img"
            onPointerDown={(e) => { cv.current!.setPointerCapture(e.pointerId); last.current = pos(e); stroke(last.current, last.current); schedule(); }}
            onPointerMove={(e) => { if (!last.current) return; const q = pos(e); stroke(last.current, q); last.current = q; schedule(); }}
            onPointerUp={() => { last.current = null; predict(); }} onPointerCancel={() => { last.current = null; }} />
          <div className="row"><button type="button" className="btn" onClick={clear}>Clear</button><button type="button" className="btn" onClick={() => { clear(); loadDigit(); }}>Load a test digit</button></div>
        </div>
        <div className="stack" data-tour="answer" aria-live="polite">
          {!answer ? <p className="muted">Draw a digit 0–9 in the black square.</p> : (
            <>
              <div className="row" style={{ alignItems: "center", gap: 16 }}>
                {/* eslint-disable-next-line @next/next/no-img-element */}
                {thumb && <img src={thumb} alt="what the network sees (28 by 28)" className="xthumb" />}
                <div>
                  <span className="small muted">Honest answer ({Math.round((1 - alpha) * 100)}% promise)</span>
                  <div className="set-read">{answer.set.length ? `{ ${answer.set.join(", ")} }` : "{ } none of these"}</div>
                  <span className="small">{answer.set.length === 1 ? "Confident: one label." : answer.set.length === 0 ? "The rule gave an empty set: it does not recognise this as a digit it knows." : `Unsure between ${answer.set.length} digits.`}</span>
                </div>
              </div>
              <HBars label="Network probabilities" max={1} fmt={(v) => pct(v, 1)} items={answer.p.map((v, k) => ({ name: `${k}${answer.set.includes(k) ? " · in set" : ""}`, value: v, color: answer.set.includes(k) ? "var(--accent)" : "var(--surface-2)" }))} />
            </>
          )}
          <p className="small muted">The trained LeNet (seed 0) runs in your browser; the cut-off comes from its {ds.data.cal.n.toLocaleString()} real calibration digits. Your drawing is never sent anywhere.</p>
        </div>
      </div>
    </div>
  );
}
