"""Conformal prediction: quantile threshold and set construction.

Theorem (marginal coverage): under exchangeability of (X_i, Y_i),
    1 - alpha  <=  P(Y in C(X))  <=  1 - alpha + 1/(n+1).
"""
from __future__ import annotations

import math

import torch


def conformal_quantile(cal_scores: torch.Tensor, alpha: float) -> float:
    """Split-conformal threshold.

    k = ceil((n+1) * (1 - alpha));  q_hat = k-th smallest of the n calibration scores.
    Return +inf if k > n (too few points: the only valid set is the full label space).
    """
    n = len(cal_scores)
    # round() guards float error, e.g. (9+1)*0.9 = 9.000000000000002 must give k=9, not 10.
    k = math.ceil(round((n + 1) * (1 - alpha), 9))
    if k > n:
        return math.inf
    if k <= 0:
        return -math.inf
    return cal_scores.sort().values[k - 1].item()  # k is 1-indexed


def build_sets(score_matrix: torch.Tensor, q_hat: float) -> torch.Tensor:
    """Return a boolean membership matrix C (n, K): C[i, y] = (S[i, y] <= q_hat)."""
    return score_matrix <= q_hat


def mondrian_quantiles(cal_scores: torch.Tensor, cal_labels: torch.Tensor, alpha: float, num_classes: int) -> torch.Tensor:
    """Class-conditional (Mondrian) thresholds q_hat_c, one per class.

    cal_scores is the TRUE-label score vector S[i, y_i]. For class c, the conformal quantile is taken
    over calibration points with label c only. Gives P(Y in C(X) | Y = c) >= 1 - alpha per class.
    """
    return torch.tensor([conformal_quantile(cal_scores[cal_labels == c], alpha) for c in range(num_classes)])


def build_sets_mondrian(score_matrix: torch.Tensor, q_hats: torch.Tensor) -> torch.Tensor:
    """C[i, y] = (S[i, y] <= q_hats[y]). The test label is unknown, so each candidate y uses its own q_hat_y."""
    return score_matrix <= q_hats[None, :].to(score_matrix.dtype)


def true_label_scores(score_matrix: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """S[i, y_i]: the calibration score of each point's own label."""
    return score_matrix[torch.arange(len(labels)), labels]


def coverage(membership: torch.Tensor, labels: torch.Tensor) -> float:
    """Marginal empirical coverage: mean over i of C[i, labels[i]]."""
    return membership[torch.arange(len(labels)), labels].float().mean().item()


def class_conditional_coverage(membership: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Return per-class empirical coverage, shape (K,). NaN for a class absent from `labels`."""
    hit = membership[torch.arange(len(labels)), labels].float()
    counts = torch.bincount(labels, minlength=num_classes).float()
    return torch.bincount(labels, weights=hit, minlength=num_classes) / counts


# ---------------------------------------------------------------------------------------------------------
# Training-conditional (PAC) and group-conditional calibration
# ---------------------------------------------------------------------------------------------------------
def pac_k(n: int, alpha: float, delta: float) -> int | None:
    """Smallest order-statistic index k with P(coverage >= 1 - alpha) >= 1 - delta, else None (use q = +inf).

    Given n calibration scores, the coverage of q = S_(k) is Beta(k, n + 1 - k) (continuous scores), so we need
    Beta(k, n + 1 - k).sf(1 - alpha) >= 1 - delta. This is a stricter k than ceil((n + 1)(1 - alpha)).
    """
    from scipy.stats import beta
    if n == 0:
        return None
    ks = range(max(1, math.ceil((n + 1) * (1 - alpha))), n + 1)
    for k in ks:  # sf is increasing in k, so the first hit is the smallest
        if beta.sf(1 - alpha, k, n + 1 - k) >= 1 - delta:
            return k
    return None


def kth_quantile(cal_scores: torch.Tensor, k: int | None) -> float:
    """S_(k) (1-indexed), or +inf when k is None or exceeds n."""
    if k is None or k > len(cal_scores):
        return math.inf
    return cal_scores.sort().values[k - 1].item()


def group_quantiles(cal_scores: torch.Tensor, cal_groups: torch.Tensor, alpha: float, n_groups: int,
                    delta: float | None = None) -> torch.Tensor:
    """One threshold per group (Mondrian over groups). Group -1 (if present) gets the pooled marginal threshold.

    delta=None: the standard in-expectation guarantee per group.
    delta given: every group simultaneously has true coverage >= 1 - alpha with probability >= 1 - delta.
        Group calibration sets are disjoint and independent given the labels, so a per-group level of
        delta_g = 1 - (1 - delta)^(1/G) is exact (no union bound needed).
    """
    g_ = sum(1 for g in range(n_groups) if (cal_groups == g).any())
    dg = None if delta is None else 1 - (1 - delta) ** (1 / max(g_, 1))
    q = []
    for g in range(n_groups):
        s = cal_scores[cal_groups == g]
        q.append(conformal_quantile(s, alpha) if dg is None else kth_quantile(s, pac_k(len(s), alpha, dg)))
    return torch.tensor(q, dtype=torch.float64)


def mondrian_quantiles_pac(cal_scores: torch.Tensor, cal_labels: torch.Tensor, alpha: float, num_classes: int,
                           delta: float) -> torch.Tensor:
    """Class-conditional thresholds with a simultaneous high-probability guarantee over all classes."""
    return group_quantiles(cal_scores, cal_labels, alpha, num_classes, delta)


def kmeans(x: torch.Tensor, k: int, weights: torch.Tensor | None = None, iters: int = 100, seed: int = 0) -> torch.Tensor:
    """Weighted k-means with k-means++ seeding. Returns the cluster index of each row of x."""
    g = torch.Generator().manual_seed(seed)
    x = x.double()
    w = torch.ones(len(x), dtype=torch.float64) if weights is None else weights.double()
    k = min(k, len(x))
    cent = x[torch.multinomial(w, 1, generator=g)]
    for _ in range(1, k):
        d2 = torch.cdist(x, cent).min(1).values ** 2
        p = w * d2
        cent = torch.cat([cent, x[torch.multinomial(p / p.sum() if p.sum() > 0 else w, 1, generator=g)]])
    for _ in range(iters):
        assign = torch.cdist(x, cent).argmin(1)
        new = torch.stack([(x[assign == j] * w[assign == j, None]).sum(0) / w[assign == j].sum()
                           if (assign == j).any() else cent[j] for j in range(k)])
        if torch.allclose(new, cent):
            break
        cent = new
    return torch.cdist(x, cent).argmin(1)


def cluster_classes(scores: torch.Tensor, labels: torch.Tensor, num_classes: int, n_clusters: int,
                    levels=(0.5, 0.6, 0.7, 0.8, 0.9), min_count: int = 5, seed: int = 0) -> torch.Tensor:
    """Clustered conformal (Ding et al., NeurIPS 2023): group classes whose true-label score distributions look alike.

    Each class is embedded by quantiles of its scores at `levels` and clustered by k-means weighted by sqrt(count).
    Classes with fewer than `min_count` points go to the null cluster (-1), which uses the marginal threshold.
    Fit on data disjoint from D_cal (here D_design), so the cluster map is fixed before calibration.
    """
    lv = torch.tensor(levels, dtype=torch.float64)
    emb, ok, cnt = [], [], []
    for c in range(num_classes):
        s = scores[labels == c].double()
        if len(s) >= min_count:
            emb.append(torch.quantile(s, lv))
            ok.append(c)
            cnt.append(len(s))
    out = torch.full((num_classes,), -1, dtype=torch.long)
    if ok:
        out[torch.tensor(ok)] = kmeans(torch.stack(emb), n_clusters, torch.tensor(cnt).double().sqrt(), seed=seed)
    return out


def build_sets_grouped(score_matrix: torch.Tensor, class_group: torch.Tensor, q_group: torch.Tensor, q_null: float) -> torch.Tensor:
    """C[i, y] = S[i, y] <= q[group(y)], with the null group (-1) using q_null."""
    q = torch.where(class_group >= 0, q_group[class_group.clamp(min=0)], torch.tensor(q_null, dtype=torch.float64))
    return score_matrix <= q[None, :].to(score_matrix.dtype)


# ---------------------------------------------------------------------------------------------------------
# Weighted conformal: covariate shift (Tibshirani et al., 2019) and label shift (Podkopaev & Ramdas, 2021)
# ---------------------------------------------------------------------------------------------------------
def weighted_thresholds(cal_scores: torch.Tensor, cal_w: torch.Tensor, test_w: torch.Tensor, alpha: float) -> torch.Tensor:
    """Per-test-point threshold: the (1 - alpha) quantile of sum_i p_i delta_{S_i} + p_test delta_{+inf},
    with p_i = w_i / (sum_j w_j + w_test). Vectorised over test points via one sort and a cumulative sum."""
    order = cal_scores.argsort()
    s, w = cal_scores[order].double(), cal_w[order].double()
    cw = w.cumsum(0)
    need = (1 - alpha) * (cw[-1] + test_w.double())       # mass that must lie at or below q
    idx = torch.searchsorted(cw, need)                    # first index with cumulative weight >= need
    q = torch.full_like(need, math.inf)
    ok = idx < len(s)
    q[ok] = s[idx[ok]]
    return q


def bbse_weights(cal_logits: torch.Tensor, cal_labels: torch.Tensor, target_logits: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Black-box shift estimation (Lipton et al., 2018): w_y = q(y) / p(y) from the confusion matrix on D_cal and
    the predicted-label histogram on the unlabelled target. Clipped at 0 and renormalised so E_p[w] = 1."""
    pred = cal_logits.argmax(1)
    C = torch.zeros(num_classes, num_classes, dtype=torch.float64)
    C.index_put_((pred, cal_labels), torch.ones(len(pred), dtype=torch.float64), accumulate=True)
    C /= len(pred)
    mu = torch.bincount(target_logits.argmax(1), minlength=num_classes).double() / len(target_logits)
    w = torch.linalg.lstsq(C, mu[:, None]).solution[:, 0].clamp(min=0)
    p = torch.bincount(cal_labels, minlength=num_classes).double() / len(cal_labels)
    return w / (w * p).sum()


def label_shift_sets(cal_true_scores: torch.Tensor, cal_labels: torch.Tensor, score_matrix: torch.Tensor,
                     w_class: torch.Tensor, alpha: float) -> torch.Tensor:
    """Label-shift weighted conformal: candidate y is tested with test weight w_y, calibration point i has w_{y_i}."""
    q = weighted_thresholds(cal_true_scores, w_class[cal_labels], w_class, alpha)  # one threshold per candidate y
    return score_matrix <= q[None, :].to(score_matrix.dtype)


def aci(cal_true_scores: torch.Tensor, stream_scores: torch.Tensor, stream_labels: torch.Tensor, alpha: float,
        lr: float) -> dict:
    """Adaptive conformal inference (Gibbs & Candes, 2021): alpha_{t+1} = alpha_t + lr * (alpha - err_t).

    lr = 0 is ordinary split conformal. alpha_t <= 0 gives the full set; alpha_t >= 1 the empty set.
    Returns per-step coverage indicator, set size and alpha_t.
    """
    s = cal_true_scores.sort().values
    n = len(s)
    a, hit, size, alphas = alpha, [], [], []
    for t in range(len(stream_labels)):
        if a <= 0:
            q = math.inf
        elif a >= 1:
            q = -math.inf
        else:
            k = math.ceil(round((n + 1) * (1 - a), 9))
            q = math.inf if k > n else s[max(k, 1) - 1].item()
        row = stream_scores[t] <= q
        h = bool(row[stream_labels[t]])
        hit.append(h)
        size.append(int(row.sum()))
        alphas.append(a)
        a = a + lr * (alpha - (0.0 if h else 1.0))
    return {"hit": torch.tensor(hit, dtype=torch.float64), "size": torch.tensor(size, dtype=torch.float64),
            "alpha_t": torch.tensor(alphas)}
