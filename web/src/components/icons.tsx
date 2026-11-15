// Small stroke icons (24-unit grid, currentColor) shared by the sidebar, GIF figures and footer.
const P = { fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round", strokeLinejoin: "round" } as const;
const I = ({ d, children }: { d?: string; children?: React.ReactNode }) => <svg viewBox="0 0 24 24" aria-hidden {...P}>{d ? <path d={d} /> : children}</svg>;

export const Palette = () => <I><circle cx="13.5" cy="6.5" r="1.2" /><circle cx="17.5" cy="10.5" r="1.2" /><circle cx="8.5" cy="7.5" r="1.2" /><circle cx="6.5" cy="12.5" r="1.2" /><path d="M12 2a10 10 0 1 0 0 20c1.1 0 2-.9 2-2 0-.5-.2-1-.5-1.3-.3-.4-.5-.8-.5-1.3 0-1.1.9-2 2-2h2.4A5.6 5.6 0 0 0 22 9.8C22 5.5 17.5 2 12 2z" /></I>;
export const ArrowUp = () => <I d="M12 19V5M5 12l7-7 7 7" />;
export const ArrowRight = () => <I d="M5 12h14M13 5l7 7-7 7" />;
export const Menu = () => <I d="M4 7h16M4 12h16M4 17h10" />;
export const Mail = () => <I><rect x="3" y="5" width="18" height="14" rx="2" /><path d="m3 7 9 6 9-6" /></I>;
export const Globe = () => <I><circle cx="12" cy="12" r="9" /><path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18" /></I>;
export const FileText = () => <I><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" /><path d="M14 3v5h5M9 13h6M9 17h6" /></I>;
export const Compass = () => <I><circle cx="12" cy="12" r="9" /><path d="m15.5 8.5-2 5-5 2 2-5z" /></I>;
export const Copy = () => <I><rect x="9" y="9" width="11" height="11" rx="2" /><path d="M5 15V6a2 2 0 0 1 2-2h9" /></I>;
export const Check = () => <I d="M5 12.5 10 17 19 7" />;
export const Play = () => <svg viewBox="0 0 24 24" aria-hidden fill="currentColor"><path d="M8 5v14l11-7z" /></svg>;
export const Pause = () => <svg viewBox="0 0 24 24" aria-hidden fill="currentColor"><path d="M7 5h4v14H7zM13 5h4v14h-4z" /></svg>;
export const Expand = () => <I d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />;
export const School = () => <I><path d="m2 9 10-5 10 5-10 5z" /><path d="M6 11v5c3 2.5 9 2.5 12 0v-5" /></I>;
export const Lock = () => <I><rect x="5" y="11" width="14" height="10" rx="2" /><path d="M8 11V7a4 4 0 0 1 8 0v4" /></I>;
export const Hash = () => <I d="M5 9h14M5 15h14M10 3 8 21M16 3l-2 18" />;
