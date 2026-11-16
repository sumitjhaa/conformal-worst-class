"use client";
import { useEffect, useState } from "react";
import { LOAD_FAILED_EVENT, retryLoads } from "@/lib/data";

/** One notice for the whole site when any data file fails to load, so sections that only show “loading…” are never stuck silently. */
export default function LoadToast() {
  const [what, setWhat] = useState<string | null>(null);
  useEffect(() => {
    const on = (e: Event) => setWhat((e as CustomEvent<string>).detail || "some data");
    const ok = () => setWhat(null);
    addEventListener(LOAD_FAILED_EVENT, on); addEventListener("conformal:retry", ok);
    return () => { removeEventListener(LOAD_FAILED_EVENT, on); removeEventListener("conformal:retry", ok); };
  }, []);
  if (!what) return null;
  return (
    <div className="xtoast" role="alert">
      <p>Couldn’t load {what}. Check your connection.</p>
      <div className="row" style={{ gap: 8 }}>
        <button type="button" className="btn primary" onClick={retryLoads}>Try again</button>
        <button type="button" className="xbtn" onClick={() => setWhat(null)}>Dismiss</button>
      </div>
    </div>
  );
}
