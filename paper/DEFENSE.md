# Defence prep — likely questions and short answers

Rubric: "can explain the theorem, a limitation, and a small change." Numbers below come from `results/numbers.tex`
(regenerate with `python scripts/paper_numbers.py`); ImageNet figures are over 10 random splits at α = 0.1.

## 1. The theorem
**Q: Why does split conformal give `P(Y ∈ C(X)) ≥ 1 − α`?**
Scores of the n calibration points and the test point are exchangeable, so the test score's rank among all n+1 is
uniform (ties broken at random, which APS's `u` does). `Y ∈ C(X)` iff the test score ≤ the k-th smallest calibration
score, i.e. its rank ≤ k, which has probability `k/(n+1)`. With `k = ⌈(n+1)(1−α)⌉` that is between `1−α` and `1−α+1/(n+1)`.

**Q: Then why is one deployed predictor sometimes below target?**
The guarantee averages over calibration sets. Given D_cal, coverage is `F(S_(k))`; `F(S_i)` are i.i.d. uniform, so
coverage is the k-th uniform order statistic, `Beta(k, n+1−k)`, with s.d. ≈ `sqrt(α(1−α)/n)`. At n = 100 about half of
re-splits fall below 1 − α. The site's §5 shows the live histogram against this law.

**Q: What is your worst-class law?**
With one threshold per class, each class's calibration set is disjoint from the others, so per-class coverages are
independent `Beta(k_c, n_c+1−k_c)`; measured on m_c test images they are Beta–binomial. The minimum of independent
variables has survival `Π_c P(cov_c ≥ t)`. It needs only the counts, not the data. On exchangeable re-splits it matches
experiment within about a point from 10 to 1,000 classes (ImageNet: 48.7% predicted, 47.7% measured).

**Q: But on CIFAR-100 it missed by 5 points — doesn't that break it?**
No: the law assumes calibration and test images are exchangeable *within each class*. When I pool D_cal and D_test and
re-split (exchangeable by construction) it matches CIFAR-100 exactly (74.5% vs 74.3% at 50 images/class). On the official
split the measured worst class sits in the law's lower tail in 2 of 3 seeds (Fisher p ≈ 0.01): some classes are harder in
CIFAR's separately collected test set than in the training pool (Recht et al., 2018). So the law doubles as a detector of
per-class shift — and it is sharper than a χ² test on per-class accuracy, which is weak on CIFAR-100 but fires on MNIST,
where the worst class is unaffected.

**Q: Why is the PAC level `δ_g = 1 − (1−δ)^(1/G)` exact rather than a union bound?**
Because the group coverages are independent (disjoint calibration sets), `P(all groups ok) = Π_g P(group ok)
≥ (1−δ_g)^G = 1−δ`. A union bound would use `δ/G`, slightly stricter, and is only needed under dependence.

**Q: When is Mondrian-PAC infeasible?**
The largest index k = n gives `P(cov ≥ 1−α) = 1 − (1−α)^n`. That reaches `1 − δ_g` only if `n ≥ log δ_g / log(1−α)`.
For ImageNet (G = 1000, δ = 0.1, α = 0.1) that is 87 images per class; we have about 20, so the only valid choice is
`q̂ = ∞`, i.e. every label in every set (size 1000).

## 2. Results to remember (α = 0.1)
- ImageNet worst class: Naive(T) 37.9%, APS(T) 52.9%, Mondrian 47.7% (size 35), **Clustered-PAC 60.7% (size 17.2)**,
  Mondrian-PAC: full sets.
- Every in-expectation method is marginally valid (89.2–92.3% at α = 0.1); PAC variants over-cover by design.
- MNIST: accuracy 99.1% > 90%, so ~9–10% of sets are **empty** — forced by `P(empty) ≥ acc − coverage`.
- Label shift: a marginal method can fall to its worst class; class-wise methods cannot.
- Real shift: coverage collapses on CIFAR-10-C; ACI restores it by growing sets; weighted CP lifts shifted CIFAR-10 from 70.1% to
  89.0%, but only by giving 31% of images the full label set (support mismatch: size 1.93 → 4.34).

## 3. Limitations (pick one and explain)
- **Clustered-PAC protects clusters, not classes.** Its guarantee is for the mixture of classes in a cluster; a class unlike
  its cluster-mates can still be under-covered. The clusters come from ~10 design images per class on ImageNet.
- **Exchangeability.** Every guarantee here breaks under covariate shift (CIFAR-10-C). ACI gives long-run, not per-input, coverage.
- **ImageNet seeds** re-split one model's outputs, so they capture calibration/test randomness, not training randomness.
- **The distance term** is a weak result: +2.1 points on CIFAR-10's worst class (CI [1.0, 3.0]), nothing on CIFAR-100; the
  softmax already decides the second label better (AUROC 0.916 vs 0.740), so γ·d adds little.

## 4. "A small change" — be ready to say what would happen
- **Raise α to 0.2 on MNIST** → even more empty sets (bound: acc − coverage ≈ 19%).
- **Halve calibration images per class** → worst-class law shifts left; the site's §8 explorer shows it live.
- **More clusters M** → thresholds get noisier and the PAC correction bigger; fewer → heterogeneous groups. §9 slider.
- **Use δ = 0.5 instead of 0.1** → smaller PAC sets, weaker guarantee (half of deployments may miss somewhere).
- **Apply the same pipeline to a new model** → only `D_design`/`D_cal` logits are needed; no retraining for conformal.

## 5. Process questions
- **Leakage?** Four disjoint splits with an assertion; everything tuned uses D_design; D_cal only sets q̂; D_test used once.
- **Why bootstrap over seeds and test images?** Seeds capture training/split randomness, test images capture evaluation
  noise; intervals that exclude 0 are differences neither can explain.
- **AI use?** Claude helped with code, debugging, plots, the site and editing; every number is produced by the scripts and
  written into the paper as a macro by `scripts/paper_numbers.py`.
