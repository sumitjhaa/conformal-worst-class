"use client";
import { Component, type ReactNode } from "react";
import { retryLoads } from "@/lib/data";

// A script chunk that failed to download stays failed in the module cache, so only a reload can fetch it again.
const isChunkError = (e: unknown) => /ChunkLoadError|Loading chunk|dynamically imported module|Failed to fetch|Importing a module script failed/i.test(String((e as Error)?.name) + " " + String((e as Error)?.message ?? e));

export function LoadFailed({ what, error }: { what: string; error?: unknown }) {
  const chunk = isChunkError(error);
  return (
    <div className="xerror" role="alert">
      <p>{chunk ? `Part of the page (${what}) did not download.` : `Something went wrong while showing ${what}.`} <span className="small muted">The rest of the page still works.</span></p>
      <div className="row" style={{ gap: 8 }}>
        {chunk ? <button type="button" className="btn primary" onClick={() => location.reload()}>Reload page</button>
          : <button type="button" className="btn primary" onClick={() => { retryLoads(); dispatchEvent(new Event("conformal:guard-reset")); }}>Try again</button>}
      </div>
    </div>
  );
}

/** Error boundary: a crash in one chapter or explorer shows a small notice instead of blanking the whole page. */
export class Guard extends Component<{ what: string; children: ReactNode }, { error: unknown }> {
  state = { error: null as unknown };
  static getDerivedStateFromError(error: unknown) { return { error }; }
  private reset = () => this.setState({ error: null });
  componentDidMount() { addEventListener("conformal:guard-reset", this.reset); }
  componentWillUnmount() { removeEventListener("conformal:guard-reset", this.reset); }
  componentDidCatch(error: unknown) { console.error(`[${this.props.what}]`, error); }
  render() { return this.state.error ? <LoadFailed what={this.props.what} error={this.state.error} /> : this.props.children; }
}
