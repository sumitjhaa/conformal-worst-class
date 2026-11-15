"use client";
import Link from "next/link";
import { useEffect, useState, type ComponentType, type ReactNode } from "react";
import { EXPLORERS, GROUPS, type ExplorerInfo } from "@/lib/explorers";
import { asset } from "@/lib/data";
import { Check, Copy, Menu, Palette } from "../icons";
import { ThemePanel } from "../Sidebar";
import { useQuery } from "./kit";

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
          {cmp ? (
            <div className="xcompare">
              <section aria-label="Left configuration"><span className="xpin">A</span><View p="" /></section>
              <section aria-label="Right configuration"><span className="xpin">B</span><View p="b." /></section>
            </div>
          ) : <View p="" />}
        </div>
      </main>
      {tour && <TourBox info={info} onClose={() => setTour(false)} />}
    </>
  );
}

export function IndexShell({ children }: { children: ReactNode }) {
  return <><Nav /><main className="xmain" id="top"><div className="wrap">{children}</div></main></>;
}

