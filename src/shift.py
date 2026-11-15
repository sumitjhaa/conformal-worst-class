"""Likelihood-ratio weights for covariate shift via a probabilistic domain classifier (Tibshirani et al., 2019).

w(x) = dP_target / dP_source (x) is estimated as  p(target | x) / p(source | x) * n_source / n_target,
with an L2-regularised logistic regression on the classifier's penultimate features.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def fit_logistic(x: torch.Tensor, y: torch.Tensor, l2: float = 1e-3, iters: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    """Binary logistic regression by LBFGS. Returns (weights, bias) for standardised inputs."""
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1, max_iter=iters, line_search_fn="strong_wolfe")
    x, y = x.double(), y.double()

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ w + b, y) + l2 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def domain_weights(source: torch.Tensor, target_fit: torch.Tensor, *apply_to: torch.Tensor, clip: float = 100.0) -> list[torch.Tensor]:
    """Fit source-vs-target on features, return w(x) for each tensor in apply_to (clipped at `clip` for stability)."""
    mu, sd = source.double().mean(0), source.double().std(0).clamp(min=1e-6)
    z = lambda t: (t.double() - mu) / sd  # noqa: E731
    x = torch.cat([z(source), z(target_fit)])
    y = torch.cat([torch.zeros(len(source)), torch.ones(len(target_fit))])
    w, b = fit_logistic(x, y)
    ratio = len(source) / len(target_fit)
    return [((z(t) @ w + b).exp() * ratio).clamp(max=clip) for t in apply_to]


def effective_sample_size(w: torch.Tensor) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2: how many calibration points the weights really use."""
    return (w.sum() ** 2 / (w ** 2).sum()).item()
