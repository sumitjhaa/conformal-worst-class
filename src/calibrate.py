"""Temperature scaling (Guo et al., 2017).

Fit T > 0 on D_design by minimising NLL. Preserves argmax; rescales score magnitudes.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Return the scalar temperature T via LBFGS on logits (design split only).

    p_T(y|x) = softmax(z(x) / T)
    """
    log_t = torch.zeros(1, requires_grad=True)  # optimise log T so T stays positive
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return log_t.exp().item()


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return softmax(logits / T)."""
    return torch.softmax(logits / temperature, dim=1)


def reliability_bins(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per equal-width confidence bin: (count, accuracy, mean confidence). Shared by ECE and the plot."""
    conf, pred = probs.max(1)
    correct = (pred == labels).float()
    b = (conf * n_bins).long().clamp(max=n_bins - 1)
    count = torch.bincount(b, minlength=n_bins).float()
    safe = count.clamp(min=1)
    acc = torch.bincount(b, weights=correct, minlength=n_bins) / safe
    mean_conf = torch.bincount(b, weights=conf, minlength=n_bins) / safe
    return count, acc, mean_conf


def expected_calibration_error(probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """ECE: sum_b (|B_b|/n) * |acc(B_b) - conf(B_b)|."""
    count, acc, conf = reliability_bins(probs, labels, n_bins)
    return (count / count.sum() * (acc - conf).abs()).sum().item()
