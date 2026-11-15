"""Learned conformity score: a small residual network on the classifier's log-probabilities, trained for small sets.

Conformal training after Stutz et al. (ICLR 2022): inside each batch, half the points act as a calibration set,
a differentiable quantile gives q, and smooth set membership sigmoid((q - s) / tau) gives a set size to minimise.
A cross-entropy term keeps the transformed probabilities a sensible classifier.

Validity: the network is fit on D_design only (half A to train, half B to early-stop), so it is a fixed
function before D_cal is touched, and split conformal on its scores keeps the finite-sample guarantee.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.conformal import build_sets, conformal_quantile, true_label_scores


class ResidualHead(nn.Module):
    """h -> h + MLP(h) on temperature-scaled log-probabilities; the last layer starts at zero (identity map)."""

    def __init__(self, k: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(k, hidden), nn.ReLU(), nn.Linear(hidden, k))
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, logits: torch.Tensor, T: float) -> torch.Tensor:
        h = F.log_softmax(logits / T, 1)
        return h + self.net(h)


def _size_on(head, T, logits, labels, alpha) -> float:
    """Exact (non-smooth) mean set size of the naive score on the head's probabilities, by a 50/50 split."""
    with torch.no_grad():
        S = 1 - torch.softmax(head(logits, T), 1)
    half = len(labels) // 2
    q = conformal_quantile(true_label_scores(S[:half], labels[:half]), alpha)
    return build_sets(S[half:], q).sum(1).float().mean().item()


def train_conformal_head(logits: torch.Tensor, labels: torch.Tensor, T: float, alpha: float, steps: int = 400,
                         batch: int = 1000, tau: float = 0.1, ce_weight: float = 1.0, seed: int = 0) -> ResidualHead:
    """Fit on D_design: half A trains, half B picks the step with the smallest exact mean set size."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    perm = torch.randperm(len(labels), generator=g)
    a, b = perm[: len(perm) // 2], perm[len(perm) // 2:]
    head = ResidualHead(logits.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    best, best_size = copy.deepcopy(head.state_dict()), _size_on(head, T, logits[b], labels[b], alpha)
    for step in range(steps):
        idx = a[torch.randint(len(a), (min(batch, len(a)),), generator=g)]
        z, y = logits[idx], labels[idx]
        out = head(z, T)
        S = 1 - torch.softmax(out, 1)
        h = len(y) // 2
        q = torch.quantile(S[:h][torch.arange(h), y[:h]], 1 - alpha)          # differentiable in S
        soft = torch.sigmoid((q - S[h:]) / tau)                                # smooth membership
        loss = (soft.sum(1) - 1).clamp(min=0).mean() + ce_weight * F.cross_entropy(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 20 == 19:
            size = _size_on(head, T, logits[b], labels[b], alpha)
            if size < best_size:
                best, best_size = copy.deepcopy(head.state_dict()), size
    head.load_state_dict(best)
    head.eval()
    return head
