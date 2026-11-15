"""Mahalanobis distance in feature space (Lee et al., 2018).

Fit per-class Gaussians with one shared covariance Sigma on D_train penultimate features.
"""
from __future__ import annotations

import torch


def fit_class_gaussians(features: torch.Tensor, labels: torch.Tensor, num_classes: int, reg: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (mu, sigma_inv).

    mu:        (K, D) class means
    sigma_inv: (D, D) inverse of the shared (tied) covariance, regularised by `reg` on the diagonal.
    """
    f = features.double()  # float64: Sigma is often near-singular (ReLU features have dead units)
    mu = torch.stack([f[labels == c].mean(0) for c in range(num_classes)])
    centred = f - mu[labels]
    sigma = centred.T @ centred / len(f)
    # reg is relative to the average variance, so it means the same thing for MNIST and CIFAR features
    sigma += reg * sigma.trace() / len(sigma) * torch.eye(len(sigma), dtype=f.dtype)
    return mu, torch.linalg.inv(sigma)


def mahalanobis_per_class(features: torch.Tensor, mu: torch.Tensor, sigma_inv: torch.Tensor) -> torch.Tensor:
    """Return (n, K) matrix of d_M(x_i, y) for every class y."""
    diff = features.double()[:, None, :] - mu[None]                    # (n, K, D)
    sq = torch.einsum("nkd,de,nke->nk", diff, sigma_inv, diff)
    return sq.clamp(min=0).sqrt().float()
