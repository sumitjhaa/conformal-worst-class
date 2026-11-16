import type { Metadata } from "next";
import Link from "next/link";
import { IndexShell } from "@/components/explorer/Shell";
import { EXPLORERS, GROUPS } from "@/lib/explorers";

// server component: asset() from lib/data is client-only, so build the two main-page links here
const home = (hash: string) => `${process.env.NEXT_PUBLIC_BASE_PATH || ""}/${hash}`;

export const metadata: Metadata = { title: "Conformal explorer", description: "Seventeen interactive explorers for the conformal prediction project: live prediction sets, every method's results, the worst-class law, shift, and a digit you can draw." };

export default function ExplorerIndex() {
  return (
    <IndexShell>
      <header className="xhead">
        <div className="ch-meta"><span className="ch-no">{EXPLORERS.length}</span><span className="kicker">interactive explorers</span></div>
        <h1>Explore the results yourself</h1>
        <p className="lead">Every number in the project, and most of the maths, runs live here: real model outputs, recomputed in your browser. Each explorer keeps its settings in the link, so any view can be shared.</p>
      </header>
      <section className="xstart" aria-label="How to use the explorers">
        <a href={home("#basics")} className="xstart-card"><span className="xlesson-k">New to this?</span><strong>Start with the basics chapter</strong><span className="small muted">Datasets, probabilities, train vs test, percentiles: five minutes.</span></a>
        <div className="xstart-card"><span className="xlesson-k">How the path works</span><strong>Top to bottom, easy to hard</strong>
          <span className="small muted xlegend"><i className="lv-beginner" />beginner <i className="lv-intermediate" />intermediate <i className="lv-advanced" />advanced: the card colour shows the level, and every explorer ends with a link to the next.</span></div>
        <a href={home("#deep")} className="xstart-card"><span className="xlesson-k">Want the proofs?</span><strong>Read the maths chapter</strong><span className="small muted">Formulas for every result, with a live calculator.</span></a>
      </section>
      {GROUPS.map((g) => (
        <section key={g} className="xgroup" aria-labelledby={`g-${g}`}>
          <h2 id={`g-${g}`}>{g}</h2>
          <ol className="xcards">
            {EXPLORERS.filter((e) => e.group === g).map((e) => (
              <li key={e.slug}>
                <Link href={`/explorer/${e.slug}/`} className={`xcard lv-${e.level.toLowerCase()}`} title={`${e.level} level`}>
                  <span className="xcard-top"><span className="xcard-no">{String(EXPLORERS.indexOf(e) + 1).padStart(2, "0")}</span></span>
                  <strong>{e.short}</strong>
                  <span className="xcard-d">{e.blurb}</span>
                  <span className="xcard-f"><span className="xlesson-k">Finding</span> {e.found}</span>
                </Link>
              </li>
            ))}
          </ol>
        </section>
      ))}
    </IndexShell>
  );
}
