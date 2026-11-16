"use client";
import { motion, useScroll, useSpring } from "motion/react";
import { useEffect, useState } from "react";
import { ACCENTS, FLAVORS, setTheme, useTheme } from "@/lib/theme";
import { Check, Menu, Palette } from "./icons";

export const CHAPTERS = [["basics", "Basics"], ["idea", "Problem"], ["how", "How it works"], ["try", "Try it"], ["promise", "Check"], ["fair", "Catch"], ["predict", "Predict"], ["fix", "Fix"], ["shift", "Shift"], ["results", "Results"], ["deep", "Maths"], ["method", "Method"], ["glossary", "Glossary"]] as const;
const IDS = CHAPTERS.map(([id]) => id);

/** The chapter whose top has most recently passed the upper third of the viewport. */
function useActiveChapter() {
  const [active, setActive] = useState<string | null>(null);
  useEffect(() => {
    let raf = 0;
    const update = () => {
      raf = 0;
      // look chapters up on every update: a chapter that re-mounts (e.g. hot reload) would otherwise leave a detached node whose top reads 0
      const els = IDS.map((id) => document.getElementById(id)).filter((e): e is HTMLElement => !!e);
      const line = innerHeight / 3, passed = els.filter((e) => e.getBoundingClientRect().top < line);
      setActive(passed.length ? passed[passed.length - 1].id : null);
    };
    const onScroll = () => { if (!raf) raf = requestAnimationFrame(update); };
    update();
    addEventListener("scroll", onScroll, { passive: true });
    return () => { removeEventListener("scroll", onScroll); cancelAnimationFrame(raf); };
  }, []);
  return active;
}

/** j / k: next / previous chapter. t: theme panel. Ignored while typing in a field. */
function useShortcuts(active: string | null) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (e.metaKey || e.ctrlKey || e.altKey || t.closest("input:not([type=range]), textarea, select, [contenteditable]")) return;
      const i = active ? IDS.indexOf(active as (typeof IDS)[number]) : -1;
      if (e.key === "j" || e.key === "k") {
        const next = IDS[Math.min(IDS.length - 1, Math.max(0, i + (e.key === "j" ? 1 : -1)))];
        document.getElementById(next)?.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
      } else if (e.key === "t") {
        document.getElementById("theme-panel")?.togglePopover();
      }
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, [active]);
}

export function ThemePanel() {
  const { flavor, accent } = useTheme();
  return (
    <>
      <div className="sec">
        <h4><span>Theme</span><span className="mono">{flavor}</span></h4>
        <div className="flavors">
          {FLAVORS.map((f) => (
            <button key={f.id} type="button" className="flavor" aria-pressed={flavor === f.id} onClick={() => setTheme({ flavor: f.id })}
              style={{ background: f.sw[0], color: f.sw[1] }}>
              <span className="name">{f.name}{flavor === f.id && <span style={{ width: 14, color: f.sw[2] }}><Check /></span>}</span>
              <span className="sub">{f.sub}</span>
              <span className="sw">{f.sw.slice(1).map((c) => <i key={c} style={{ background: c }} />)}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="sec">
        <h4><span>Accent</span><span className="mono">{accent}</span></h4>
        <div className="accents">
          {ACCENTS.map((a) => (
            <button key={a} type="button" className="accent" aria-pressed={accent === a} aria-label={a} title={a} onClick={() => setTheme({ accent: a })} style={{ background: `var(--ctp-${a})` }}>
              {accent === a && <Check />}
            </button>
          ))}
        </div>
      </div>
      <div className="keys"><span><kbd>j</kbd> <kbd>k</kbd> next / previous chapter</span><span><kbd>t</kbd> this panel</span><span>Saved on this device</span></div>
    </>
  );
}

export default function Sidebar() {
  const active = useActiveChapter();
  useShortcuts(active);
  const { scrollYProgress } = useScroll();
  const smooth = useSpring(scrollYProgress, { stiffness: 200, damping: 30, restDelta: 0.001 });

  return (
    <>
      <nav className="rail" aria-label="Chapters">
        <a className="logo" href="#top">conformal/</a>
        <ol>
          {CHAPTERS.map(([id, label]) => (
            <li key={id}>
              <a className="item" href={`#${id}`} aria-current={active === id ? "true" : undefined}>{label}</a>
            </li>
          ))}
        </ol>
        <div className="tools">
          <button type="button" className="item" popoverTarget="theme-panel"><Palette /> Theme <kbd>t</kbd></button>
        </div>
      </nav>

      <button type="button" className="fab" popoverTarget="sheet"><Menu /> Chapters &amp; theme</button>

      <div id="theme-panel" className="pop" popover="auto"><ThemePanel /></div>
      <div id="sheet" className="pop" popover="auto">
        <div className="sec">
          <h4><span>Chapters</span></h4>
          <nav className="sheet-nav" aria-label="Chapters">
            {CHAPTERS.map(([id, label]) => (
              <a key={id} href={`#${id}`} aria-current={active === id ? "true" : undefined} onClick={() => document.getElementById("sheet")?.hidePopover()}>{label}</a>
            ))}
          </nav>
        </div>
        <ThemePanel />
      </div>

      <motion.div className="progress" style={{ scaleX: smooth }} aria-hidden />
    </>
  );
}
