# Who Does a Conformal Guarantee Leave Out? — ES 678 Course Project

**Predicting and repairing the worst class in conformal classification.**

Split conformal prediction turns any classifier into one that returns label *sets* containing the truth with
probability `≥ 1 − α`. That promise averages over classes and over calibration draws. This project measures who it
leaves out, predicts the worst class with an exact law, and repairs it with a cheap high-probability correction.

| Contribution | Where |
|---|---|
| **Exact worst-class law** under class-wise calibration (independent Beta–binomials), validated on 5 settings | `src/theory.py`, paper §4, site §8 |
| **Calibration-size rule**: images per class needed so every class is within ε, w.p. 1−δ | `theory.n_per_class_needed` |
| **Clustered-PAC**: simultaneous high-probability guarantee over clusters of classes (exact δ_g, no union bound) | `src/conformal.py`, paper §4–5, site §9 |
| Label-shift corollary (worst case = worst class), empty-set lemma | `src/theory.py`, paper §4 |
| Audit of 13 methods × 5 settings × 3–10 seeds, label/covariate shift, ACI, weighted CP, OOD, bootstrap CIs | `scripts/research.py` |
| Interactive site that reruns conformal prediction in the browser | [live site](https://sumitjhaa.github.io/conformal-worst-class/), `web/` |

## Settings (runs)
| Run | Model | Classes | Seeds |
|---|---|---|---|
| `mnist` | LeNet (trained here) | 10 | 10 |
| `cifar10` | SmallResNet (trained here) | 10 | 10 |
| `cifar10_r18` | ImageNet-pretrained ResNet-18, fine-tuned | 10 | 5 |
| `cifar100` | SmallResNet (trained here) | 100 | 3 |
| `imagenet` | published ResNet-152 outputs on the 50k val images | 1000 | 10 re-splits |

Shift / OOD sets: rotation & pixel noise, **CIFAR-10.1** (natural), **CIFAR-10-C** (15 corruptions × 5 severities),
FashionMNIST/KMNIST, CIFAR-100, CIFAR-10. SVHN's host was unreachable, so it is skipped automatically.

## Methods
Top-1 baseline · Naive (LAC) · Margin · APS · RAPS · SAPS · learned score (conformal training) · temperature scaling ·
distance-augmented score (first proposal; diagnosed weak result) · Mondrian · **Mondrian-PAC** · Clustered ·
**Clustered-PAC** (proposed) · ACI · covariate-shift weighted CP · label-shift weighted CP with BBSE weights.

## Reproduce
```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/                      # 22 checks: quantiles, scores, PAC, weighted/label-shift CP, ACI, theory, learned head
python scripts/train_all.py        # trains every (run, seed), caches logits/features, adds CIFAR-10.1 / CIFAR-10-C
python scripts/run_all.py          # main tables + depth analyses + figures (CPU, minutes)
python scripts/sweep.py            # α × γ grid and temperature ablation
python scripts/research.py         # worst-class law, PAC/clusters, label shift, ACI, weighted CP, real shift, bootstrap
python scripts/make_figures.py     # rebuild every figure/table from the CSVs
python scripts/paper_numbers.py    # results/numbers.tex: every number quoted in the papers, as LaTeX macros
python scripts/export_site.py      # site/data/*.json
python scripts/make_gifs.py        # site/gifs/*.gif: animated figures for the web tour, from the exported outputs
cd web && yarn && yarn dev         # interactive site at http://localhost:3000
```
Downloads: torchvision datasets automatically; CIFAR-10.1 v6 `.npy` files into `data/cifar10_1/`
(github.com/modestyachts/CIFAR-10.1); CIFAR-10-C into `data/cifar10_c/` (Zenodo record 2535967); ImageNet ResNet-152
outputs (`imagenet-resnet152.npz`) into `data/imagenet/` from the archive used by the conformal-prediction tutorial
notebooks (github.com/aangelopoulos/conformal-prediction). Missing optional sets are skipped.

Training happens once per (run, seed); logits and features are cached in `results/cache/`, so every analysis reruns
on CPU. Delete a cache file to retrain.

## Key rule (validity)
Everything tuned — temperature `T`, RAPS/SAPS `λ`, `γ`, the learned score, the class clusters — uses **`D_design`** only.
The conformal thresholds use **`D_cal`** only. **`D_test`** is opened once. Mixing these breaks the guarantee.

## Outputs
- `results/main.csv`, `class_cov.csv`, `deciles.csv`, `sizes.csv`, `ood.csv`, `shift.csv`, `resplit.csv`, `tuned.csv`
- `results/research/*.csv` — worst_class, ncal_sweep, clusters, label_shift, aci(_series), weighted, real_shift,
  empty_sets, distance_diag, bootstrap, n_needed
- `results/figures/*.pdf`, `results/table_*.tex`, `results/numbers.tex`

## Papers
- `paper/es678_conformal.tex` — the 4-page course paper (supplied NeurIPS 2026 style, ≤ 4 pages incl. references).
- `paper/conformal_worst_class.tex` — extended version with proofs, all tables and figures.
Both read every number from `results/numbers.tex`; compile with `tectonic -X compile <file>.tex` from `paper/`.

## Website
**Live:** https://sumitjhaa.github.io/conformal-worst-class/

`web/` is a Next.js + React app exported as a static site: a plain-language, animated tour (problem, method on real scores,
worst-class law, the fix, shift, results, glossary) plus 17 explorers that rerun conformal prediction in the browser from
the exported logits (live set lab with every method, per-class audit, worst-class law, cluster trade-off, label and
covariate shift, ImageNet gallery, draw-a-digit). Its data comes from `site/data/` and `site/gifs/` (written by
`scripts/export_site.py` and `scripts/make_gifs.py`) and the paper PDFs, copied in by `web/scripts/sync-data.mjs`.

- Local: `cd web && yarn && yarn dev`; see `web/README.md`. `node web/scripts/check-explorer.mjs` checks the browser maths
  against the Python results.
- Deploy: `.github/workflows/pages.yml` builds and publishes it on every push to `main`. One-time setup on GitHub:
  Settings → Pages → Source: **GitHub Actions**.

## AI-use disclosure
AI assistance (Claude) helped organise the code, build parts of the interactive site and explorers, and edit the report's
wording and structure. The experiments, analysis and results are my own. External datasets and methods are cited in `paper/refs.bib`.
