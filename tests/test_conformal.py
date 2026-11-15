"""Checks for the conformal core. Run: pytest tests/"""
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibrate import apply_temperature, expected_calibration_error, fit_temperature  # noqa: E402
from src.conformal import (  # noqa: E402
    build_sets, build_sets_mondrian, class_conditional_coverage, conformal_quantile, coverage,
    mondrian_quantiles, true_label_scores,
)
from src.evaluate import auroc  # noqa: E402
from src.mahalanobis import fit_class_gaussians, mahalanobis_per_class  # noqa: E402
from src.scores import aps_scores, distance_augmented_scores, naive_scores, raps_scores  # noqa: E402


def test_quantile_index():
    s = torch.arange(1.0, 10.0)                       # n=9, alpha=0.1 -> k=ceil(9.0)=9 -> largest
    assert conformal_quantile(s, 0.1) == 9.0
    assert conformal_quantile(s[:8], 0.1) == math.inf  # n=8 -> k=9 > n
    assert conformal_quantile(s, 0.5) == 5.0           # k=ceil(5.0)=5


def test_aps_hand_example():
    p = torch.tensor([[0.2, 0.5, 0.3]])
    assert torch.allclose(aps_scores(p, randomised=False), torch.tensor([[1.0, 0.5, 0.8]]))
    # RAPS with k_reg=1: ranks 1,2,3 get penalty 0, lam, 2*lam
    assert torch.allclose(raps_scores(p, lam=0.1, k_reg=1, randomised=False), torch.tensor([[1.2, 0.5, 0.9]]))


def test_gamma_zero_is_baseline():
    base = torch.rand(5, 3)
    assert torch.equal(distance_augmented_scores(base, torch.randn(5, 3), 0.0), base)


def _synthetic(n, k=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    probs = torch.distributions.Dirichlet(torch.ones(k)).sample((n,))
    labels = torch.multinomial(probs, 1, generator=g).squeeze(1)  # labels truly drawn from probs
    return probs, labels


def test_marginal_and_mondrian_validity():
    """Average coverage over many calibration draws must sit in [1-alpha, 1-alpha+1/(n+1)]."""
    torch.manual_seed(0)
    alpha, n, m, reps = 0.1, 500, 2000, 100
    probs, labels = _synthetic(reps * (n + m))
    for score_fn in (naive_scores, lambda p: aps_scores(p, seed=1), lambda p: raps_scores(p, 0.05, 2, seed=1)):
        S = score_fn(probs)
        covs, mond = [], []
        for r in range(reps):
            cal = slice(r * (n + m), r * (n + m) + n)
            test = slice(r * (n + m) + n, (r + 1) * (n + m))
            q = conformal_quantile(true_label_scores(S[cal], labels[cal]), alpha)
            covs.append(coverage(build_sets(S[test], q), labels[test]))
            qs = mondrian_quantiles(true_label_scores(S[cal], labels[cal]), labels[cal], alpha, 5)
            mond.append(class_conditional_coverage(build_sets_mondrian(S[test], qs), labels[test], 5))
        mean = sum(covs) / reps
        mc_err = 3 * (alpha * (1 - alpha) / (reps * m)) ** 0.5 + 0.005  # sampling noise + cal-set variance
        assert 1 - alpha - mc_err <= mean <= 1 - alpha + 1 / (n + 1) + mc_err, mean
        # Mondrian: each class has ~n/5 cal points, so the per-class slack is ~1/(n/5+1)
        assert (torch.stack(mond).mean(0) >= 1 - alpha - 0.02).all()


def test_temperature_recovers_known_t():
    torch.manual_seed(0)
    z = torch.randn(20000, 4) * 3
    labels = torch.multinomial(torch.softmax(z / 2.0, 1), 1).squeeze(1)  # true T = 2
    t = fit_temperature(z, labels)
    assert abs(t - 2.0) < 0.1, t
    assert expected_calibration_error(apply_temperature(z, t), labels) < 0.02


def test_mahalanobis_identity_cov():
    feats = torch.cat([torch.randn(5000, 2), torch.randn(5000, 2) + 4])
    labels = torch.cat([torch.zeros(5000), torch.ones(5000)]).long()
    mu, inv = fit_class_gaussians(feats, labels, 2)
    d = mahalanobis_per_class(torch.tensor([[0.0, 0.0]]), mu, inv)
    assert abs(d[0, 0].item()) < 0.1 and abs(d[0, 1].item() - 32 ** 0.5) < 0.2


def test_auroc():
    assert auroc(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0])) == 1.0
    assert auroc(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0])) == 0.5
