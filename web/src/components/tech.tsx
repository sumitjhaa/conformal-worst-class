import { siFramer, siLatex, siNextdotjs, siNumpy, siPandas, siPython, siPytorch, siReact, siScipy, siTypescript } from "simple-icons";

// simple-icons has no Matplotlib mark: a small polar bar chart in its spirit
const MATPLOTLIB = "M12 1a11 11 0 1 0 0 22 11 11 0 0 0 0-22zm0 2a9 9 0 1 1 0 18 9 9 0 0 1 0-18zm0 9 6.5-4.5A8 8 0 0 0 12 4zm0 0-7.5 1.5a7.5 7.5 0 0 0 3 5.5zm0 0 2.5 5.5A6 6 0 0 0 18 12z";
// [name, svg path, colour on light themes, colour on dark themes]: brand colours, lightened where the brand colour vanishes on dark
export const TECH: [string, string, string, string][] = [
  ["Python", siPython.path, "#3776ab", "#5a9fd4"], ["PyTorch", siPytorch.path, "#ee4c2c", "#f2694d"], ["NumPy", siNumpy.path, "#013243", "#4dabcf"],
  ["SciPy", siScipy.path, "#0054a6", "#8caae6"], ["pandas", siPandas.path, "#150458", "#e70488"], ["Matplotlib", MATPLOTLIB, "#11557c", "#5fa8d3"],
  ["TypeScript", siTypescript.path, "#3178c6", "#4d9be6"], ["Next.js", siNextdotjs.path, "currentColor", "currentColor"], ["React 19", siReact.path, "#087ea4", "#61dafb"],
  ["Motion", siFramer.path, "#0055ff", "#6b9bff"], ["LaTeX", siLatex.path, "#008080", "#3cbcbc"],
];
const BY_NAME = Object.fromEntries(TECH.map((t) => [t[0], t]));

/** Brand logo + name; `name` must be one of TECH. */
export function Tech({ name }: { name: string }) {
  const [, d, light, dark] = BY_NAME[name];
  return <li><svg viewBox="0 0 24 24" aria-hidden style={{ color: `light-dark(${light}, ${dark})` }}><path d={d} /></svg>{name}</li>;
}
