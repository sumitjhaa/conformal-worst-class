"""Checks for the research extensions: PAC / group calibration, weighted and label-shift conformal, ACI,
new scores, clustering, the worst-class law and the learned head. Run: pytest tests/"""
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import theory  # noqa: E402
from src.conformal import (  # noqa: E402
    aci, bbse_weights, build_sets, class_conditional_coverage, conformal_quantile, coverage, group_quantiles, kmeans,
    label_shift_sets, pac_k, true_label_scores, weighted_thresholds,
)
from src.evaluate import sscv  # noqa: E402
from src.learned import ResidualHead  # noqa: E402
from src.scores import margin_scores, saps_scores  # noqa: E402


def test_pac_k_is_stricter_and_valid():
    from scipy.stats import beta
    n, a, d = 500, 0.1, 0.1
    k = pac_k(n, a, d)
    assert k > math.ceil((n + 1) * (1 - a))
    assert beta.sf(1 - a, k, n + 1 - k) >= 1 - d > beta.sf(1 - a, k - 1, n + 2 - k)
    assert pac_k(5, 0.1, 0.1) is None  # too few points: only the full set is valid


def test_pac_holds_over_calibration_draws():
    """Uniform scores: true coverage of S_(k) is S_(k) itself. P(coverage >= 1-a) should be >= 1-d."""
    g = torch.Generator().manual_seed(0)
    n, a, d = 200, 0.1, 0.1
    k = pac_k(n, a, d)
    covs = torch.rand(4000, n, generator=g).sort(1).values[:, k - 1]
    assert (covs >= 1 - a).float().mean() >= 1 - d - 0.02


def test_group_pac_is_simultaneous():
    """G groups calibrated independently: all groups covered at once with probability >= 1 - delta."""
    g = torch.Generator().manual_seed(1)
    G, n, a, d, reps = 10, 300, 0.1, 0.1, 1500
    ok = 0
    for _ in range(reps):
        s = torch.rand(G * n, generator=g)
        q = group_quantiles(s, torch.arange(G).repeat_interleave(n), a, G, d)
        ok += bool((q.clamp(max=1) >= 1 - a).all())   # uniform scores: true coverage of q is q
    assert ok / reps >= 1 - d - 0.03


def test_weighted_with_unit_weights_is_split_conformal():
    s = torch.rand(999, generator=torch.Generator().manual_seed(2))
    q = weighted_thresholds(s, torch.ones(999), torch.ones(5), 0.1)
    assert torch.allclose(q, torch.full((5,), conformal_quantile(s, 0.1), dtype=torch.float64))


def test_label_shift_unit_weights_match_standard():
    g = torch.Generator().manual_seed(3)
    S_cal, y_cal = torch.rand(500, 4, generator=g), torch.randint(0, 4, (500,), generator=g)
    S = torch.rand(50, 4, generator=g)
    s_true = true_label_scores(S_cal, y_cal)
    assert torch.equal(label_shift_sets(s_true, y_cal, S, torch.ones(4, dtype=torch.float64), 0.1),
                       build_sets(S, conformal_quantile(s_true, 0.1)))


def test_bbse_recovers_weights():
    """Noisy classifier with a known confusion matrix; target prior differs from source."""
    rng = np.random.default_rng(0)
    K, n = 3, 60000
    conf = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])  # P(pred | y)

    def sample(prior, m):
        y = rng.choice(K, size=m, p=prior)
        pred = np.array([rng.choice(K, p=conf[c]) for c in y])
        logits = torch.full((m, K), -5.0)
        logits[torch.arange(m), torch.as_tensor(pred)] = 5.0
        return logits, torch.as_tensor(y)

    p, q = np.array([1 / 3] * 3), np.array([0.6, 0.3, 0.1])
    lc, yc = sample(p, n)
    lt, _ = sample(q, n)
    w = bbse_weights(lc, yc, lt, K)
    assert torch.allclose(w, torch.as_tensor(q / p), atol=0.06)


def test_aci_zero_lr_is_split_conformal_and_adapts():
    g = torch.Generator().manual_seed(4)
    cal = torch.rand(1000, generator=g)
    S = torch.rand(3000, 3, generator=g) * torch.linspace(1, 1.6, 3000)[:, None]   # scores drift upward
    y = torch.randint(0, 3, (3000,), generator=g)
    fixed = aci(cal, S, y, 0.1, 0.0)
    q = conformal_quantile(cal, 0.1)
    assert torch.equal(fixed["hit"].bool(), (S[torch.arange(3000), y] <= q))
    adapt = aci(cal, S, y, 0.1, 0.02)
    assert abs(adapt["hit"].mean().item() - 0.9) < 0.02 < abs(fixed["hit"].mean().item() - 0.9)


def test_margin_and_saps_hand_example():
    p = torch.tensor([[0.5, 0.3, 0.2]])
    assert torch.allclose(margin_scores(p), torch.tensor([[-0.2, 0.2, 0.3]]))
    s = saps_scores(p, lam=0.1, seed=0)
    u = s[0, 0] / 0.5
    assert torch.allclose(s, torch.tensor([[0.5 * u, 0.5 + u * 0.1, 0.5 + (1 + u) * 0.1]]))


def test_kmeans_separates_obvious_clusters():
    x = torch.cat([torch.randn(30, 2) * 0.1, torch.randn(30, 2) * 0.1 + 5])
    a = kmeans(x, 2)
    assert len(a[:30].unique()) == 1 and len(a[30:].unique()) == 1 and a[0] != a[30]


def test_worst_class_law_matches_simulation():
    law = theory.worst_class_law([300] * 20, [400] * 20, 0.1)
    sim = theory.simulate_worst_class(300, 400, 20, 0.1, reps=20000)
    assert abs(law["mean"] - sim.mean()) < 0.002
    assert abs(law["q10"] - np.quantile(sim, 0.1)) < 0.004


def test_n_needed_grows_with_classes():
    ns = [theory.n_per_class_needed(K, 0.1, 0.02, 0.1) for K in (2, 20, 200)]
    assert ns[0] < ns[1] < ns[2]


def test_empty_bound_holds_on_singletons():
    """Sets of size <= 1 built from a top-1 label: empty rate >= accuracy - coverage."""
    g = torch.Generator().manual_seed(5)
    logits = torch.randn(5000, 5, generator=g) * 3
    y = torch.where(torch.rand(5000, generator=g) < 0.9, logits.argmax(1), torch.randint(0, 5, (5000,), generator=g))
    p = torch.softmax(logits, 1)
    C = build_sets(1 - p, conformal_quantile(true_label_scores(1 - p, y), 0.2))
    assert C.sum(1).max() <= 1
    acc = (logits.argmax(1) == y).float().mean().item()
    assert (C.sum(1) == 0).float().mean().item() >= theory.empty_rate_lower_bound(acc, coverage(C, y)) - 1e-9


def test_sscv_zero_for_perfectly_stratified():
    C = torch.zeros(100, 5, dtype=torch.bool)
    C[:, 0] = True
    y = torch.zeros(100, dtype=torch.long)
    y[:10] = 1                                   # 90% covered, all sets size 1
    assert abs(sscv(C, y, 0.1)) < 1e-6  # float32 mean


def test_label_shift_coverage_is_prior_weighted():
    cc = np.array([0.95, 0.8])
    assert abs(theory.label_shift_coverage(cc, np.array([0.25, 0.75])) - (0.25 * 0.95 + 0.75 * 0.8)) < 1e-12
    C = torch.tensor([[1, 0], [0, 0], [1, 1]], dtype=torch.bool)
    assert torch.allclose(class_conditional_coverage(C, torch.tensor([0, 1, 1]), 2), torch.tensor([1.0, 0.5]))


def test_learned_head_starts_as_temperature_scaling():
    z = torch.randn(7, 6)
    h = ResidualHead(6)
    assert torch.allclose(torch.softmax(h(z, 1.7), 1), torch.softmax(z / 1.7, 1), atol=1e-6)
