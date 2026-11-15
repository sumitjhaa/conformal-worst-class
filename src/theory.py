"""Closed-form predictions that the experiments are checked against.

1. Marginal law (Vovk 2012; Angelopoulos & Bates 2021): given D_cal, coverage ~ Beta(k, n + 1 - k), k = ceil((n+1)(1-a)).
2. Worst-class law (this project): class-conditional (Mondrian) calibration uses disjoint per-class calibration sets,
   so per-class coverages are independent Beta(k_c, n_c + 1 - k_c) variables. Measured on m_c test points each,
   observed coverage is Beta-binomial. The worst class is their minimum, whose survival function is a product:
       P(min_c cov_c >= t) = prod_c P(cov_c >= t).
3. Calibration size needed per class so that every class is within eps of target with probability >= 1 - delta.
4. Empty-set bound: if every set has at most one label, P(empty) >= acc - (1 - alpha) - slack.
5. Label shift: coverage under any re-weighting of classes is sum_c pi'_c cov_c, so its worst case is min_c cov_c.
"""
from __future__ import annotations

import functools
import math

import numpy as np
from scipy.stats import beta, betabinom


def k_index(n: int, alpha: float) -> int:
    return math.ceil(round((n + 1) * (1 - alpha), 9))


def marginal_law(n: int, alpha: float) -> tuple[int, int] | None:
    """Beta(a, b) parameters of coverage given D_cal, or None when k > n (q = +inf, coverage 1)."""
    k = k_index(n, alpha)
    return None if k > n else (k, n + 1 - k)


@functools.lru_cache(maxsize=4096)
def _class_survival_cached(grid: int, n_cal: int, n_test: int | None, alpha: float, k: int | None) -> np.ndarray:
    """Per-class survival on the standard grid; classes share (n_c, m_c, k), so this is computed once per distinct triple."""
    return _class_survival(np.linspace(0, 1, grid), n_cal, n_test, alpha, k)


def _class_survival(t: np.ndarray, n_cal: int, n_test: int | None, alpha: float, k: int | None = None) -> np.ndarray:
    """P(observed coverage of one class >= t) on a grid t. n_test=None means the true (infinite-test) coverage."""
    k = k_index(n_cal, alpha) if k is None else k
    if k > n_cal:                       # q = +inf: this class is always covered
        return np.ones_like(t)
    a, b = k, n_cal + 1 - k
    if n_test is None:
        return beta.sf(t, a, b)
    # observed = X / m with X ~ BetaBinomial(m, a, b); P(X/m >= t) = P(X >= ceil(t m)) = sf(ceil(t m) - 1)
    return betabinom.sf(np.ceil(t * n_test - 1e-9) - 1, n_test, a, b)


def worst_class_law(n_cal: list[int], n_test: list[int] | None, alpha: float, ks: list[int | None] | None = None,
                    grid: int = 4001) -> dict:
    """Distribution of min over classes of (observed or true) coverage under class-conditional calibration.

    ks optionally overrides the order-statistic index per class (e.g. the PAC-corrected index).
    Returns the grid, survival P(min >= t), mean, and the 10/50/90% quantiles.
    """
    t = np.linspace(0, 1, grid)
    surv = np.ones_like(t)
    for c, n in enumerate(n_cal):
        m = None if n_test is None else n_test[c]
        surv *= _class_survival_cached(grid, int(n), None if m is None else int(m), alpha, None if ks is None else int(ks[c]))
    cdf = 1 - surv                                   # P(min < t), nondecreasing
    mean = float(np.trapezoid(surv, t))              # E[min] = int_0^1 P(min >= t) dt for min in [0, 1]
    q = {p: float(t[min(np.searchsorted(cdf, p), len(t) - 1)]) for p in (0.1, 0.5, 0.9)}  # all-infinite thresholds: min = 1
    return {"t": t, "survival": surv, "mean": mean, "q10": q[0.1], "q50": q[0.5], "q90": q[0.9]}


def simulate_worst_class(n_cal: int, n_test: int, K: int, alpha: float, reps: int = 20000, seed: int = 0) -> np.ndarray:
    """Monte Carlo of the same law (used only to test worst_class_law)."""
    rng = np.random.default_rng(seed)
    k = k_index(n_cal, alpha)
    if k > n_cal:
        return np.ones(reps)
    cov = rng.beta(k, n_cal + 1 - k, size=(reps, K))
    return (rng.binomial(n_test, cov) / n_test).min(1)


def n_per_class_needed(K: int, alpha: float, eps: float, delta: float, n_max: int = 100000) -> int | None:
    """Smallest per-class calibration size n so that, with probability >= 1 - delta, every one of K classes has true
    coverage >= 1 - alpha - eps under class-conditional calibration: beta.sf(1 - alpha - eps, k, n + 1 - k)^K >= 1 - delta."""
    n = max(1, math.ceil(1 / alpha) - 1)
    while n <= n_max:
        k = k_index(n, alpha)
        if k <= n and beta.sf(1 - alpha - eps, k, n + 1 - k) ** K >= 1 - delta:
            return n
        n = n + 1 if n < 200 else int(n * 1.02) + 1  # shortcut: 2% steps above 200, so the answer can overshoot by <2%
    return None


def empty_rate_lower_bound(accuracy: float, coverage: float) -> float:
    """If every prediction set has size <= 1 and a non-empty set is the top-1 label, then
    miss = P(empty) + P(non-empty and top-1 wrong) <= P(empty) + (1 - acc), so P(empty) >= acc - coverage."""
    return max(0.0, accuracy - coverage)


def label_shift_coverage(class_cov: np.ndarray, target_prior: np.ndarray) -> float:
    """Coverage under label shift = sum_c pi'_c cov_c (per-class coverage is unchanged by re-weighting classes)."""
    return float(np.dot(target_prior, class_cov))
