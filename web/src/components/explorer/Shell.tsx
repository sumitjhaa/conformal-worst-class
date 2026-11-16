"use client";
import Link from "next/link";
import { useEffect, useState, type ComponentType, type ReactNode } from "react";
import { EXPLORERS, GROUPS, type ExplorerInfo } from "@/lib/explorers";
import { asset } from "@/lib/data";
import { Check, Copy, Menu, Palette } from "../icons";
import { ThemePanel } from "../Sidebar";
import { useQuery } from "./kit";
import { Guard } from "../Guard";

/** Left rail (desktop) / bottom sheet (mobile) listing every explorer, plus the theme panel. */
function Nav({ current }: { current?: string }) {
  const list = (onPick?: () => void) => GROUPS.map((g) => (
    <div key={g} className="xnav-group">
      <span className="xnav-h">{g}</span>
      {EXPLORERS.filter((e) => e.group === g).map((e) => (
        <Link key={e.slug} className="item" href={`/explorer/${e.slug}/`} aria-current={current === e.slug ? "page" : undefined} onClick={onPick}>{e.short}</Link>
      ))}
    </div>
  ));
  return (
    <>
      <nav className="rail xrail" aria-label="Explorers">
        <a className="logo" href={asset("")}>conformal/</a>
        <Link className="item" href="/explorer/" aria-current={!current ? "page" : undefined}>All explorers</Link>
        {list()}
        <div className="tools"><button type="button" className="item" popoverTarget="theme-panel"><Palette /> Theme</button></div>
      </nav>
      <button type="button" className="fab" popoverTarget="sheet"><Menu /> Explorers &amp; theme</button>
      <div id="theme-panel" className="pop" popover="auto"><ThemePanel /></div>
      <div id="sheet" className="pop" popover="auto">
        <div className="sec xsheet">
          <a href={asset("")}>← Main page</a> <Link href="/explorer/">All explorers</Link>
          {list(() => document.getElementById("sheet")?.hidePopover())}
        </div>
        <ThemePanel />
      </div>
    </>
  );
}

function CopyLink() {
  const [done, setDone] = useState(false);
  const copy = async () => { try { await navigator.clipboard.writeText(location.href); setDone(true); setTimeout(() => setDone(false), 1600); } catch {} };
  return <button type="button" className="xbtn" onClick={copy} aria-live="polite">{done ? <><Check /> Copied</> : <><Copy /> Copy link</>}</button>;
}

/** Step-by-step walkthrough: highlights each [data-tour] element in turn. */
function TourBox({ info, onClose }: { info: ExplorerInfo; onClose: () => void }) {
  const [i, setI] = useState(0), step = info.tour[i];
  useEffect(() => {
    const el = document.querySelector(`[data-tour="${step.at}"]`);
    if (!el) return;
    el.classList.add("tour-hi");
    el.scrollIntoView({ block: "center", behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    return () => el.classList.remove("tour-hi");
  }, [step.at]);
  useEffect(() => { const k = (e: KeyboardEvent) => e.key === "Escape" && onClose(); addEventListener("keydown", k); return () => removeEventListener("keydown", k); }, [onClose]);
  return (
    <div className="tourbox" role="dialog" aria-label="Guided tour" aria-live="polite">
      <span className="small muted">Step {i + 1} of {info.tour.length}</span>
      <p>{step.text}</p>
      <div className="row" style={{ justifyContent: "flex-end", gap: 8 }}>
        <button type="button" className="xbtn" onClick={onClose}>Close</button>
        <button type="button" className="xbtn" disabled={i === 0} onClick={() => setI(i - 1)}>Back</button>
        {i < info.tour.length - 1 ? <button type="button" className="btn primary" onClick={() => setI(i + 1)}>Next</button> : <button type="button" className="btn primary" onClick={onClose}>Done</button>}
      </div>
    </div>
  );
}

/** What this explorer teaches, how to read it, and the project's finding: the same three notes on every explorer. */
function Lesson({ info }: { info: ExplorerInfo }) {
  return (
    <section className="xlesson" aria-label="About this explorer">
      <div><span className="xlesson-k">What you’ll learn</span><p>{info.learn}</p></div>
      <div><span className="xlesson-k">How to read it</span><p>{info.read}</p></div>
      <div className="found"><span className="xlesson-k">What I found</span><p>{info.found}</p></div>
    </section>
  );
}

/** Previous / next explorer, so the list reads as one path from the basics to the advanced material. */
function PrevNext({ info }: { info: ExplorerInfo }) {
  const i = EXPLORERS.indexOf(info), prev = EXPLORERS[i - 1], next = EXPLORERS[i + 1];
  return (
    <nav className="xpath" aria-label="Learning path">
      {prev ? <Link href={`/explorer/${prev.slug}/`} className="xpath-l"><span className="small muted">← Previous</span><strong>{prev.short}</strong></Link> : <a href={asset("#basics")} className="xpath-l"><span className="small muted">← Before this</span><strong>The basics chapter</strong></a>}
      {next ? <Link href={`/explorer/${next.slug}/`} className="xpath-r"><span className="small muted">Next →</span><strong>{next.short}</strong></Link> : <a href={asset("#deep")} className="xpath-r"><span className="small muted">Finish with →</span><strong>The maths chapter</strong></a>}
    </nav>
  );
}

/** Frame for one explorer: header, tools (compare / tour / link) and the view, once or side by side. */
export function Shell({ info, view: View }: { info: ExplorerInfo; view: ComponentType<{ p: string }> }) {
  const [cmp, setCmp] = useQuery<number>("cmp", 0), [tour, setTour] = useState(false);
  const no = EXPLORERS.indexOf(info) + 1;
  return (
    <>
      <Nav current={info.slug} />
      <main className="xmain" id="top">
        <div className="wrap">
          <header className="xhead">
            <div className="ch-meta"><span className="ch-no">{String(no).padStart(2, "0")}</span><span className="kicker">{info.group}</span></div>
            <h1>{info.title}</h1>
            <p className="lead">{info.blurb}</p>
            <div className="row xtools">
              <button type="button" className="xbtn" aria-pressed={!!cmp} onClick={() => setCmp(cmp ? 0 : 1)}>Compare side by side</button>
              <button type="button" className="xbtn" onClick={() => setTour(true)}>Guided tour</button>
              <CopyLink />
              {info.chapter && <a className="xbtn" href={asset(`#${info.chapter}`)}>Read the chapter</a>}
            </div>
          </header>
          <Lesson info={info} />
          {cmp ? (
            <div className="xcompare">
              <section aria-label="Left configuration"><span className="xpin">A</span><Guard what="the left view"><View p="" /></Guard></section>
              <section aria-label="Right configuration"><span className="xpin">B</span><Guard what="the right view"><View p="b." /></Guard></section>
            </div>
          ) : <Guard what="this explorer"><View p="" /></Guard>}
          <PrevNext info={info} />
        </div>
      </main>
      {tour && <TourBox info={info} onClose={() => setTour(false)} />}
    </>
  );
}

export function IndexShell({ children }: { children: ReactNode }) {
  return <><Nav /><main className="xmain" id="top"><div className="wrap">{children}</div></main></>;
}

