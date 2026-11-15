"""Conformity scores. Convention: smaller = more conforming.

All scores return a matrix S of shape (n, K) where S[i, y] = s(x_i, y).
Any fixed function of (x, y) is a valid score, as long as everything it depends on
(T, mu, Sigma, normalisation stats, gamma, lambda, k_reg) is fixed before D_cal is touched.
"""
from __future__ import annotations

import torch

from src.mahalanobis import mahalanobis_per_class


def naive_scores(probs: torch.Tensor) -> torch.Tensor:
    """s(x, y) = 1 - p(y|x).  Shape (n, K)."""
    return 1 - probs


def _sorted_cumulative(probs: torch.Tensor, randomised: bool, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """APS score in sorted order (n, K) and the sort index used to scatter it back."""
    p_sorted, idx = probs.sort(dim=1, descending=True)
    if randomised:
        g = torch.Generator().manual_seed(seed)
        u = torch.rand(len(probs), 1, generator=g, dtype=probs.dtype)  # one u per example, shared over classes
    else:
        u = 1.0
    # mass strictly above y in the ranking, plus u * p(y)
    return p_sorted.cumsum(1) - p_sorted + u * p_sorted, idx


def aps_scores(probs: torch.Tensor, randomised: bool = True, seed: int = 0) -> torch.Tensor:
    """Adaptive Prediction Sets (Romano et al., 2020).

    Sort descending p_(1) >= ... >= p_(K).
    randomised:      s(x,y) = sum_{j<k_y} p_(j) + u * p_(k_y),  u ~ U(0,1)
    non-randomised:  s(x,y) = sum_{j<k_y} p_(j) + p_(k_y)
    where k_y is the rank of y.
    """
    s_sorted, idx = _sorted_cumulative(probs, randomised, seed)
    return torch.empty_like(probs).scatter_(1, idx, s_sorted)


def raps_scores(probs: torch.Tensor, lam: float, k_reg: int, randomised: bool = True, seed: int = 0) -> torch.Tensor:
    """Regularised APS (Angelopoulos et al., 2021): s = s_APS + lam * max(k_y - k_reg, 0)."""
    s_sorted, idx = _sorted_cumulative(probs, randomised, seed)
    rank = torch.arange(1, probs.shape[1] + 1, dtype=probs.dtype)  # 1-indexed k_y in sorted order
    s_sorted = s_sorted + lam * (rank - k_reg).clamp(min=0)
    return torch.empty_like(probs).scatter_(1, idx, s_sorted)


def mahalanobis_scores(features: torch.Tensor, mu: torch.Tensor, sigma_inv: torch.Tensor) -> torch.Tensor:
    """d_M(x, y) = sqrt((phi(x) - mu_y)^T Sigma^-1 (phi(x) - mu_y)).  Shape (n, K)."""
    return mahalanobis_per_class(features, mu, sigma_inv)


def distance_augmented_scores(
    base_scores: torch.Tensor,
    d_m_normalised: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Proposed score: s_gamma(x, y) = base_scores(x, y) + gamma * d_m_normalised(x, y).

    gamma = 0 recovers the base score (clean ablation).
    """
    return base_scores + gamma * d_m_normalised


def normalise_distance(d_m: torch.Tensor, ref_mean: float, ref_std: float) -> torch.Tensor:
    """Standardise d_M using design-split statistics: (d_M - mean) / std."""
    return (d_m - ref_mean) / ref_std


def margin_scores(probs: torch.Tensor) -> torch.Tensor:
    """Margin score: s(x, y) = max_{y' != y} p(y'|x) - p(y|x). Negative only for the top label."""
    top2 = probs.topk(2, dim=1).values
    best_other = torch.where(probs == top2[:, :1], top2[:, 1:2], top2[:, :1])
    return best_other - probs


def saps_scores(probs: torch.Tensor, lam: float, seed: int = 0) -> torch.Tensor:
    """Sorted Adaptive Prediction Sets (Huang et al., 2024). Keeps only the top probability:
    rank 1: s = u * p_max;  rank r >= 2: s = p_max + (r - 2 + u) * lam,  u ~ U(0, 1) per example."""
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(len(probs), 1, generator=g, dtype=probs.dtype)
    pmax = probs.max(1, keepdim=True).values
    rank = probs.argsort(1, descending=True).argsort(1) + 1          # 1 = most probable
    return torch.where(rank == 1, u * pmax, pmax + (rank - 2 + u) * lam)
