"use client";
import { useReducedMotion } from "motion/react";
import { useState, type ReactNode } from "react";
import { asset } from "@/lib/data";
import { Expand, Pause, Play } from "./icons";

/** An animated GIF rendered from the project's data (scripts/make_gifs.py). Click or press the button to pause;
 *  a still poster is shown instead when the reader prefers reduced motion. Tilts slightly toward the pointer. */
export default function Gif({ name, fig, title, alt, flip, children }: { name: string; fig: string; title: string; alt: string; flip?: boolean; children: ReactNode }) {
  const still = useReducedMotion();
  const [paused, setPaused] = useState<boolean | null>(null);
  const playing = paused === null ? !still : !paused;
  const src = asset(`gifs/${name}.${playing ? "gif" : "png"}`);
  const tilt = (e: React.PointerEvent<HTMLDivElement>) => {
    if (still) return;
    const r = e.currentTarget.getBoundingClientRect(), x = (e.clientX - r.left) / r.width - 0.5, y = (e.clientY - r.top) / r.height - 0.5;
    e.currentTarget.style.setProperty("--ry", `${x * 6}deg`); e.currentTarget.style.setProperty("--rx", `${-y * 6}deg`);
  };
  const reset = (e: React.PointerEvent<HTMLDivElement>) => { e.currentTarget.style.setProperty("--ry", "0deg"); e.currentTarget.style.setProperty("--rx", "0deg"); };

  return (
    <figure className={`gif reveal ${flip ? "flip" : ""}`}>
      <div className="gif-media" onPointerMove={tilt} onPointerLeave={reset} onClick={() => setPaused(playing)}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt={alt} width={760} height={340} loading="lazy" decoding="async" />
        <div className="ctl" onClick={(e) => e.stopPropagation()}>
          <button type="button" onClick={() => setPaused(playing)} aria-pressed={!playing}>{playing ? <><Pause /> Pause</> : <><Play /> Play</>}</button>
          <a href={asset(`gifs/${name}.gif`)} target="_blank" rel="noreferrer" aria-label={`Open ${title} full size`}><Expand /></a>
        </div>
      </div>
      <figcaption>
        <span className="fig-no">{fig}</span>
        <h3>{title}</h3>
        {children}
        <span className="src">rendered from the exported model outputs</span>
      </figcaption>
    </figure>
  );
}
