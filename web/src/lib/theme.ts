"use client";
import { useSyncExternalStore } from "react";

// Theme (Catppuccin flavours plus other palettes mapped onto the same --ctp-* slots) + accent, stored on <html data-flavor data-accent> and in localStorage under "ctp".
// layout.tsx applies the saved choice before first paint; this module reads and changes it.
export const FLAVORS = [
  { id: "latte", name: "Latte", sub: "light", sw: ["#eff1f5", "#4c4f69", "#8839ef", "#40a02b", "#fe640b", "#1e66f5"] },
  { id: "frappe", name: "Frappé", sub: "dim", sw: ["#303446", "#c6d0f5", "#ca9ee6", "#a6d189", "#ef9f76", "#8caaee"] },
  { id: "macchiato", name: "Macchiato", sub: "dark", sw: ["#24273a", "#cad3f5", "#c6a0f6", "#a6da95", "#f5a97f", "#8aadf4"] },
  { id: "mocha", name: "Mocha", sub: "darkest", sw: ["#1e1e2e", "#cdd6f4", "#cba6f7", "#a6e3a1", "#fab387", "#89b4fa"] },
  { id: "dawn", name: "Rosé Pine Dawn", sub: "light", sw: ["#faf4ed", "#575279", "#907aa9", "#518a5a", "#d7704a", "#286983"] },
  { id: "rosepine", name: "Rosé Pine", sub: "dark", sw: ["#191724", "#e0def4", "#c4a7e7", "#a6d0a0", "#f0a274", "#9ccfd8"] },
  { id: "nord", name: "Nord", sub: "dark", sw: ["#2e3440", "#eceff4", "#b48ead", "#a3be8c", "#d08770", "#88c0d0"] },
  { id: "gruvbox", name: "Gruvbox", sub: "dark", sw: ["#282828", "#ebdbb2", "#d3869b", "#b8bb26", "#fe8019", "#83a598"] },
  { id: "tokyo", name: "Tokyo Night", sub: "dark", sw: ["#1a1b26", "#c0caf5", "#bb9af7", "#9ece6a", "#ff9e64", "#7aa2f7"] },
] as const;
export const ACCENTS = ["rosewater", "flamingo", "pink", "mauve", "red", "maroon", "peach", "yellow", "green", "teal", "sky", "sapphire", "blue", "lavender"] as const;
export type Flavor = (typeof FLAVORS)[number]["id"];
export type Accent = (typeof ACCENTS)[number];

const listeners = new Set<() => void>();
const subscribe = (f: () => void) => { listeners.add(f); return () => { listeners.delete(f); }; };
const snap = () => `${document.documentElement.dataset.flavor || "latte"}:${document.documentElement.dataset.accent || "mauve"}`;

/** Current theme; re-renders when it changes. Server render assumes the defaults. */
export function useTheme() {
  const [flavor, accent] = useSyncExternalStore(subscribe, snap, () => "latte:mauve").split(":") as [Flavor, Accent];
  return { flavor, accent };
}

export function setTheme(next: { flavor?: Flavor; accent?: Accent }) {
  const el = document.documentElement;
  const apply = () => {
    if (next.flavor) el.dataset.flavor = next.flavor;
    if (next.accent) el.dataset.accent = next.accent;
    try { localStorage.setItem("ctp", JSON.stringify({ flavor: el.dataset.flavor, accent: el.dataset.accent })); } catch {}
    listeners.forEach((f) => f());
  };
  // cross-fade the page where the View Transitions API exists
  const d = document as Document & { startViewTransition?: (f: () => void) => void };
  if (next.flavor && d.startViewTransition && !matchMedia("(prefers-reduced-motion: reduce)").matches) d.startViewTransition(apply); else apply();
}

/** Inline script for <head>: saved choice, else Latte or Mocha from the OS setting. */
export const THEME_INIT = `try{var s=JSON.parse(localStorage.getItem("ctp")||"{}"),e=document.documentElement;e.dataset.flavor=s.flavor||(matchMedia("(prefers-color-scheme: dark)").matches?"mocha":"latte");e.dataset.accent=s.accent||"mauve"}catch(x){document.documentElement.dataset.flavor="latte"}`;
