import type { Metadata } from "next";
import Link from "next/link";
import { IndexShell } from "@/components/explorer/Shell";
import { EXPLORERS, GROUPS } from "@/lib/explorers";

export const metadata: Metadata = { title: "Conformal explorer", description: "Seventeen interactive explorers for the conformal prediction project: live prediction sets, every method's results, the worst-class law, shift, and a digit you can draw." };

export default function ExplorerIndex() {
  return (
    <IndexShell>
      <header className="xhead">
        <div className="ch-meta"><span className="ch-no">{EXPLORERS.length}</span><span className="kicker">interactive explorers</span></div>
        <h1>Explore the results yourself</h1>
        <p className="lead">Every number in the project, and most of the maths, runs live here: real model outputs, recomputed in your browser. Each explorer keeps its settings in the link, so any view can be shared.</p>
      </header>
      {GROUPS.map((g) => (
        <section key={g} className="xgroup" aria-labelledby={`g-${g}`}>
          <h2 id={`g-${g}`}>{g}</h2>
          <table className="gl-table xindex">
            <thead><tr><th scope="col">#</th><th scope="col">Explorer</th><th scope="col">What you can do</th></tr></thead>
            <tbody>
              {EXPLORERS.filter((e) => e.group === g).map((e) => (
                <tr key={e.slug}>
                  <td className="mono">{String(EXPLORERS.indexOf(e) + 1).padStart(2, "0")}</td>
                  <th scope="row"><Link href={`/explorer/${e.slug}/`}>{e.short}</Link></th>
                  <td>{e.blurb}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      ))}
    </IndexShell>
  );
}
