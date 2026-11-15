"use client";
import { pct } from "@/lib/conformal";
import { loadGallery, useLoad } from "@/lib/data";
import { Seg } from "../../ui";
import { Loading, mname, useQuery } from "../kit";

export default function ImageNet({ p }: { p: string }) {
  const { data: g, error } = useLoad(loadGallery);
  const [m, setM] = useQuery<string>(`${p}m`, "temp_aps");
  if (!g) return <Loading error={error} what="ImageNet gallery" />;
  const all = m === "all", method = g.methods.includes(m) ? m : g.methods[0];
  const covered = all ? NaN : g.items.filter((it) => it.sets[method]?.covered).length;
  return (
    <div className="stack" style={{ gap: 24 }}>
      <div className="stack xcontrols" data-tour="controls">
        <Seg label="Rule" value={all ? "all" : method} onChange={setM} options={[...g.methods.map((x) => [x, mname(x)] as [string, string]), ["all", "Compare all"]]} />
        <p className="small muted">Published ResNet-152 outputs on ImageNet ({g.K} classes), α = 0.1. Bars are the model’s top probabilities{all ? "" : `; highlighted labels are in the ${mname(method)} set`}. {!all && `${covered} of ${g.items.length} pictures have the truth inside the set.`}</p>
      </div>
      <div className="xinet">
        {g.items.map((it, i) => (
          <article key={i} className={`xinet-card ${!all && !it.sets[method]?.covered ? "miss" : ""}`}>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={it.img} alt={it.true} loading="lazy" />
            <p className="small"><strong>{it.true}</strong></p>
            {all ? (
              <table className="t xinet-t"><tbody>{g.methods.map((mm) => { const s = it.sets[mm]; return <tr key={mm}><th scope="row">{mname(mm)}</th><td className="n">{s.size}</td><td>{s.covered ? "✓" : "✗"}</td></tr>; })}</tbody></table>
            ) : (
              <ul className="xinet-top">
                {it.top.map(([name, pr], j) => (
                  <li key={name} className={it.sets[method]?.in[j] ? "in" : ""}>
                    <span className="xinet-bar" style={{ width: `${Math.max(2, pr * 100)}%` }} />
                    <span className="xinet-name">{name === it.true ? `${name} ✓` : name}</span><span className="mono">{pct(pr, 1)}</span>
                  </li>
                ))}
                {(it.sets[method]?.size ?? 0) > it.top.length && <li className="small muted">+{it.sets[method].size - it.top.length} more labels in the set</li>}
              </ul>
            )}
          </article>
        ))}
      </div>
    </div>
  );
}
