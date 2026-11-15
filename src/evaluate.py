"""Evaluation: coverage, efficiency, class-conditional coverage, OOD behaviour."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.stats import rankdata

from src.conformal import class_conditional_coverage, coverage


@dataclass
class Metrics:
    coverage: float
    mean_size: float
    median_size: float
    p90_size: float
    singleton_rate: float
    empty_rate: float
    worst_class_gap: float
    min_class_cov: float  # the safety-relevant number: the most under-covered class
    sscv: float = float("nan")     # size-stratified coverage violation (Angelopoulos et al., 2021)
    fsc_min: float = float("nan")  # worst coverage over feature-space clusters of the test inputs


def set_sizes(membership: torch.Tensor) -> torch.Tensor:
    """Return per-example set size = row sums of the boolean membership matrix."""
    return membership.sum(1)


def worst_class_gap(class_coverage: torch.Tensor, alpha: float) -> float:
    """max_c |cov_c - (1 - alpha)| across classes."""
    return (class_coverage - (1 - alpha)).abs().max().item()


SSCV_BINS = [(0, 1), (2, 3), (4, 10), (11, 100), (101, 1000)]


def sscv(membership: torch.Tensor, labels: torch.Tensor, alpha: float, min_count: int = 25) -> float:
    """max over set-size strata of |coverage - (1 - alpha)|, strata as in Angelopoulos et al. (2021), sizes 0-1 pooled."""
    hit = membership[torch.arange(len(labels)), labels].float()
    sizes = set_sizes(membership)
    gaps = [abs(hit[m].mean().item() - (1 - alpha)) for lo, hi in SSCV_BINS
            if (m := (sizes >= lo) & (sizes <= hi)).sum() >= min_count]
    return max(gaps) if gaps else float("nan")


def group_min_coverage(membership: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor, min_count: int = 50) -> float:
    """Worst coverage over groups of test points (e.g. feature-space clusters), ignoring groups smaller than min_count."""
    hit = membership[torch.arange(len(labels)), labels].float()
    covs = [hit[groups == g].mean().item() for g in groups.unique() if (groups == g).sum() >= min_count]
    return min(covs) if covs else float("nan")


def evaluate_scores(membership: torch.Tensor, labels: torch.Tensor, alpha: float, num_classes: int,
                    feature_groups: torch.Tensor | None = None) -> Metrics:
    """Compute the full Metrics bundle for one method/alpha."""
    sizes = set_sizes(membership).float()
    cc = class_conditional_coverage(membership, labels, num_classes)
    cc = cc[~cc.isnan()]  # classes absent from this test set (possible under label shift or small sets)
    return Metrics(
        coverage=coverage(membership, labels),
        mean_size=sizes.mean().item(),
        median_size=sizes.median().item(),
        p90_size=sizes.quantile(0.9).item(),
        singleton_rate=(sizes == 1).float().mean().item(),
        empty_rate=(sizes == 0).float().mean().item(),
        worst_class_gap=worst_class_gap(cc, alpha),
        min_class_cov=cc.min().item(),
        sscv=sscv(membership, labels, alpha),
        fsc_min=float("nan") if feature_groups is None else group_min_coverage(membership, labels, feature_groups),
    )


def subgroup_coverage(membership: torch.Tensor, labels: torch.Tensor, confidence: torch.Tensor, n_bins: int = 10) -> torch.Tensor:
    """Coverage as a function of model-confidence decile (bin 0 = least confident)."""
    hit = membership[torch.arange(len(labels)), labels].float()
    edges = confidence.quantile(torch.linspace(0, 1, n_bins + 1, dtype=confidence.dtype)[1:-1])
    b = torch.bucketize(confidence, edges)
    return torch.bincount(b, weights=hit, minlength=n_bins) / torch.bincount(b, minlength=n_bins).clamp(min=1)


def size_stratified_coverage(membership: torch.Tensor, labels: torch.Tensor) -> dict[int, tuple[float, int]]:
    """{set size: (coverage, count)}. Good adaptive sets cover ~1-alpha at every size."""
    hit = membership[torch.arange(len(labels)), labels].float()
    sizes = set_sizes(membership)
    return {int(s): (hit[sizes == s].mean().item(), int((sizes == s).sum())) for s in sizes.unique()}


def auroc(neg: torch.Tensor, pos: torch.Tensor) -> float:
    """P(pos > neg) + 0.5 P(tie): the Mann-Whitney form of AUROC."""
    ranks = rankdata(torch.cat([neg, pos]).numpy())  # average ranks over ties
    r_pos = ranks[len(neg):].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ood_report(membership: torch.Tensor, alpha: float, in_dist_membership: torch.Tensor | None = None) -> dict[str, float]:
    """Set-size statistics on an OOD set. OOD inputs have no label in {0..K-1}, so coverage is undefined:
    exchangeability is violated and we report behaviour, not a guarantee.

    If the in-distribution test membership is given, also report AUROC of "set size differs from 1"
    used as an OOD detector (empty or multi-label sets flag the input).
    """
    sizes = set_sizes(membership).float()
    out = {
        "alpha": alpha,
        "mean_size": sizes.mean().item(),
        "empty_rate": (sizes == 0).float().mean().item(),
        "singleton_rate": (sizes == 1).float().mean().item(),
        "full_rate": (sizes == membership.shape[1]).float().mean().item(),
    }
    if in_dist_membership is not None:
        flag = lambda m: (set_sizes(m) != 1).float()  # noqa: E731
        out["auroc_not_singleton"] = auroc(flag(in_dist_membership), flag(membership))
    return out
