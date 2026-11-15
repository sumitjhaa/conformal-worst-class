# Honest AI answers — the project explained (Next.js)

A plain-language, animated tour of the project, written for someone with class-12 maths. It reruns conformal prediction
live in the browser from the trained models' real outputs.

```sh
cd web
yarn            # install (Node 20+, Yarn 1)
yarn dev        # http://localhost:3000  (copies ../site/data, ../site and ../paper/*.pdf into public/ first)
yarn build      # static site in out/ ; `yarn start` serves it on :3000
```

Data comes from `python scripts/export_site.py` (writes `site/data/*.json`); `scripts/sync-data.mjs` copies it into
`public/` before every dev/build, so nothing generated is committed. For GitHub Pages under `/<repo>/`, build with
`NEXT_PUBLIC_BASE_PATH=/<repo> yarn build` and publish `out/`.

| Chapter | What it shows | Component |
|---|---|---|
| 1 Problem | why one answer is not enough; weather-forecast analogy | `Idea.tsx` |
| 2 How it works | the 4 steps on real CIFAR-10 scores, animated | `HowItWorks.tsx` |
| 3 Try it | live sets on digits / CIFAR-10 / CIFAR-100, 5 rules, α slider | `SetLab.tsx` |
| 4 Check | 300 live re-splits vs the exact Beta law | `PromiseCheck.tsx` |
| 5 Catch | per-class coverage; worst class by dataset | `Fairness.tsx` |
| 6 Predict | the worst-class law, predicted vs measured | `WorstLaw.tsx` |
| 7 Fix | grouping animation, Clustered-PAC on ImageNet, gallery | `Fix.tsx` |
| 8 Shift | corruption slider, CIFAR-10-C curve, ACI, label shift | `Shift.tsx` |
| 9 Results | what worked / what didn't | `Results.tsx` |
| 10–11 | pipeline, glossary | `Method.tsx` |

`src/lib/conformal.ts` is a TypeScript port of the conformal maths in `site/app.js` (which mirrors `src/conformal.py`).
The original single-file explorer is still served at `/explorer/` for every method and chart.
