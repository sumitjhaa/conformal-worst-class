#!/usr/bin/env python3
"""Research analyses on the cached logits (CPU only, no training). Writes results/research/*.csv.

    A  worst_class      measured worst-class coverage vs the exact worst-class law (theory.py)
    B  ncal_sweep       worst class vs calibration images per class, re-split from D_cal + D_test, vs theory
    C  clusters         number of class clusters M: 1 (marginal) ... K (Mondrian), with and without the PAC correction
    D  label_shift      Dirichlet label shift: standard, Mondrian, label-shift weighted CP (oracle and BBSE weights)
    E  aci              online stream drifting from clean to corrupted: fixed split conformal vs ACI
    F  weighted         covariate-shift weighted CP with a domain classifier on features
    G  real_shift       CIFAR-10.1 and CIFAR-10-C coverage for CIFAR-10 models
    H  empty_sets       empty-set rate vs the lower bound acc - coverage (size <= 1 regime)
    I  distance_diag    why the Mahalanobis term adds little: AUROC of softmax vs distance for the true label
    J  bootstrap        hierarchical bootstrap CIs (seeds, then test images) for the key paired comparisons
    K  class_shift      are classes equally hard in D_cal and D_test? (chi-square over per-class accuracy gaps)

Usage: python scripts/research.py [--only A C] [--runs cifar10 cifar100]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import theory  # noqa: E402
from src.conformal import (  # noqa: E402
    aci, bbse_weights, build_sets, build_sets_mondrian, class_conditional_coverage, conformal_quantile, coverage,
    label_shift_sets, mondrian_quantiles, pac_k, true_label_scores, weighted_thresholds,
)
from src.evaluate import auroc, evaluate_scores  # noqa: E402
from src.pipeline import Scorer, build_cache, membership, methods_for, seeds_for  # noqa: E402
from src.shift import domain_weights, effective_sample_size  # noqa: E402

OUT = ROOT / "results" / "research"
A0 = 0.1
MAIN = ["naive", "temp_aps", "raps", "saps", "mondrian", "mondrian_pac", "clustered", "clustered_pac", "top1"]


def scorers(cfg: dict, runs: list[str], max_seeds: int | None = None):
    """Yield (run, seed, tuned Scorer) for every cached (run, seed)."""
    for ds in runs:
        seeds = seeds_for(cfg, ds)[:max_seeds] if max_seeds else seeds_for(cfg, ds)
        for seed in seeds:
            if ds != "imagenet" and not (ROOT / "results" / "cache" / f"{ds}_s{seed}.pt").exists():
                continue
            sc = Scorer(build_cache(ds, seed, cfg), cfg)
            sc.tune(cfg, A0)
            print(f"  [{ds} s{seed}] K={sc.K}", flush=True)
            yield ds, seed, sc


def counts(labels: torch.Tensor, K: int) -> list[int]:
    return torch.bincount(labels, minlength=K).tolist()


# ------------------------------------------------------------------------------------------------ A
def worst_class(sc: Scorer, ds: str, seed: int) -> list[dict]:
    rows = []
    y_cal, y_te = sc.cache["cal"]["labels"], sc.cache["test"]["labels"]
    n_cal, n_te = counts(y_cal, sc.K), counts(y_te, sc.K)
    for alpha in (0.05, 0.1):
        dg = 1 - (1 - sc.delta) ** (1 / sc.K)
        laws = {"mondrian": theory.worst_class_law(n_cal, n_te, alpha),
                "mondrian_pac": theory.worst_class_law(n_cal, n_te, alpha, ks=[pac_k(n, alpha, dg) or n + 1 for n in n_cal]),
                "mondrian_true": theory.worst_class_law(n_cal, None, alpha)}
        for m in ("mondrian", "mondrian_pac", "temp_aps", "naive", "clustered", "clustered_pac"):
            C = membership(sc, m, "test", alpha)
            cc = class_conditional_coverage(C, y_te, sc.K)
            law = laws.get(m)
            rows.append({"dataset": ds, "seed": seed, "alpha": alpha, "method": m, "K": sc.K,
                         "n_cal_per_class": float(np.mean(n_cal)), "n_test_per_class": float(np.mean(n_te)),
                         "measured_min": cc.min().item(), "measured_q05": cc.quantile(0.05).item(),
                         "pred_mean": law["mean"] if law else np.nan, "pred_q10": law["q10"] if law else np.nan,
                         "pred_q90": law["q90"] if law else np.nan,
                         "pred_true_mean": laws["mondrian_true"]["mean"] if m == "mondrian" else np.nan,
                         # P(worst class <= measured) under the law: ~Uniform(0,1) if calibration and test are exchangeable
                         "pred_cdf_at_measured": float(1 - np.interp(cc.min().item() + 1e-6, law["t"], law["survival"])) if law else np.nan,
                         "mean_size": C.sum(1).float().mean().item()})
    return rows


# ------------------------------------------------------------------------------------------------ B
def ncal_sweep(sc: Scorer, ds: str, seed: int, reps: int = 30) -> list[dict]:
    S = torch.cat([sc.scores("temp_aps", "cal"), sc.scores("temp_aps", "test")])
    y = torch.cat([sc.cache["cal"]["labels"], sc.cache["test"]["labels"]])
    s_true = true_label_scores(S, y)
    by_c = [torch.nonzero(y == c).flatten() for c in range(sc.K)]
    pool = min(len(i) for i in by_c)
    grid = [n for n in (10, 20, 30, 50, 100, 200, 500, 1000) if n <= pool - 10]
    g = torch.Generator().manual_seed(seed)
    rows = []
    for n_c in grid:
        meas = []
        for _ in range(reps):
            cal, te = [], []
            for idx in by_c:
                p = idx[torch.randperm(len(idx), generator=g)]
                cal.append(p[:n_c])
                te.append(p[n_c:])
            cal, te = torch.cat(cal), torch.cat(te)
            q = mondrian_quantiles(s_true[cal], y[cal], A0, sc.K)
            meas.append(class_conditional_coverage(build_sets_mondrian(S[te], q), y[te], sc.K).min().item())
        law = theory.worst_class_law([n_c] * sc.K, [len(i) - n_c for i in by_c], A0)
        rows.append({"dataset": ds, "seed": seed, "K": sc.K, "n_c": n_c, "measured_mean": float(np.mean(meas)),
                     "measured_q10": float(np.quantile(meas, 0.1)), "measured_q90": float(np.quantile(meas, 0.9)),
                     "pred_mean": law["mean"], "pred_q10": law["q10"], "pred_q90": law["q90"]})
    return rows


# ------------------------------------------------------------------------------------------------ C
def clusters(sc: Scorer, ds: str, seed: int) -> list[dict]:
    y = sc.cache["test"]["labels"]
    Ms = sorted({m for m in (1, 2, 3, 5, 10, 20, 32, 50, 100, 200, 500, sc.K) if m <= sc.K})
    rows = []
    for M in Ms:
        for m in ("clustered", "clustered_pac"):
            met = evaluate_scores(membership(sc, m, "test", A0, n_clusters=M), y, A0, sc.K)
            rows.append({"dataset": ds, "seed": seed, "K": sc.K, "M": M, "method": m, "coverage": met.coverage,
                         "mean_size": met.mean_size, "min_class_cov": met.min_class_cov,
                         "q05_class_cov": class_conditional_coverage(membership(sc, m, "test", A0, n_clusters=M), y, sc.K).quantile(0.05).item()})
    return rows


# ------------------------------------------------------------------------------------------------ D
def label_shift(sc: Scorer, ds: str, seed: int, draws: int = 30, n_target: int = 2000) -> list[dict]:
    K = sc.K
    y_cal, y_te = sc.cache["cal"]["labels"], sc.cache["test"]["labels"]
    S_cal, S_te = sc.scores("temp_aps", "cal"), sc.scores("temp_aps", "test")
    s_true = true_label_scores(S_cal, y_cal)
    p_src = torch.bincount(y_cal, minlength=K).double() / len(y_cal)
    cc = {m: class_conditional_coverage(membership(sc, m, "test", A0), y_te, K).double()
          for m in ("naive", "temp_aps", "mondrian", "mondrian_pac")}
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    by_c = [torch.nonzero(y_te == c).flatten() for c in range(K)]
    rows = []
    for beta_ in (0.1, 0.3, 1.0, 3.0, 10.0):
        for d in range(draws):
            prior = torch.as_tensor(rng.dirichlet(np.full(K, beta_)))
            # an unlabelled target sample drawn from the shifted prior (for BBSE)
            cnt = torch.multinomial(prior, n_target, replacement=True, generator=g).bincount(minlength=K)
            tgt = torch.cat([by_c[c][torch.randint(len(by_c[c]), (int(cnt[c]),), generator=g)] for c in range(K) if cnt[c] > 0])
            row = {"dataset": ds, "seed": seed, "K": K, "beta": beta_, "draw": d,
                   "tv_shift": 0.5 * (prior - p_src).abs().sum().item()}
            for m, v in cc.items():
                row[m] = theory.label_shift_coverage(v.numpy(), prior.numpy())
            for name, w in (("ls_oracle", prior / p_src.clamp(min=1e-12)),
                            ("ls_bbse", bbse_weights(sc.cache["cal"]["logits"], y_cal, sc.cache["test"]["logits"][tgt], K))):
                C = label_shift_sets(s_true, y_cal, S_te, w, A0)
                row[name] = theory.label_shift_coverage(class_conditional_coverage(C, y_te, K).double().nan_to_num(1.0).numpy(), prior.numpy())
                row[name + "_size"] = float((C.sum(1).double() * prior[y_te] / torch.bincount(y_te, minlength=K).double()[y_te]).sum())
            row["temp_aps_size"] = float((S_te <= conformal_quantile(s_true, A0)).sum(1).double().mul(prior[y_te] / torch.bincount(y_te, minlength=K).double()[y_te]).sum())
            row["worst_case_bound"] = cc["temp_aps"].min().item()       # any label shift: coverage >= min_c cov_c
            row["worst_case_bound_naive"] = cc["naive"].min().item()
            rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------ E
def stream_sets(sc: Scorer, ds: str) -> list[tuple[str, str]]:
    """(segment label, cached set name) in drift order: clean test first, then rising severity."""
    if any(k.startswith("c/gaussian_noise/") for k in sc.cache):
        return [("clean", "test")] + [(f"noise sev {s}", f"c/gaussian_noise/{s}") for s in range(1, 6)]
    return [("clean", "test")] + [(k.split("/")[1], k) for k in sorted((k for k in sc.cache if k.startswith("shift/")),
                                                                       key=lambda k: float(k.split("/")[1]))]


def aci_runs(sc: Scorer, ds: str, seed: int, per_seg: int = 1000, keep_series: bool = False) -> tuple[list[dict], list[dict]]:
    g = torch.Generator().manual_seed(seed)
    segs, Ss, ys = [], [], []
    for label, name in stream_sets(sc, ds):
        n = min(per_seg * (2 if label == "clean" else 1), len(sc.cache[name]["labels"]))
        idx = torch.randperm(len(sc.cache[name]["labels"]), generator=g)[:n]
        Ss.append(sc.scores("temp_aps", name)[idx])
        ys.append(sc.cache[name]["labels"][idx])
        segs += [label] * n
    S, y = torch.cat(Ss), torch.cat(ys)
    s_true = true_label_scores(sc.scores("temp_aps", "cal"), sc.cache["cal"]["labels"])
    summ, series = [], []
    for lr in (0.0, 0.005, 0.02):
        r = aci(s_true, S, y, A0, lr)
        df = pd.DataFrame({"seg": segs, "hit": r["hit"].numpy(), "size": r["size"].numpy(), "alpha_t": r["alpha_t"].numpy()})
        for seg, d in df.groupby("seg", sort=False):
            summ.append({"dataset": ds, "seed": seed, "lr": lr, "segment": seg, "coverage": d.hit.mean(), "mean_size": d["size"].mean()})
        summ.append({"dataset": ds, "seed": seed, "lr": lr, "segment": "all", "coverage": df.hit.mean(), "mean_size": df["size"].mean()})
        if keep_series:
            roll = df.hit.rolling(250, min_periods=50).mean()
            for t in range(0, len(df), 25):
                series.append({"dataset": ds, "lr": lr, "t": t, "segment": df.seg[t], "rolling_cov": roll[t],
                               "size": df["size"][max(0, t - 125):t + 125].mean(), "alpha_t": df.alpha_t[t]})
    return summ, series


# ------------------------------------------------------------------------------------------------ F
def weighted(sc: Scorer, ds: str, seed: int) -> list[dict]:
    if not sc.has_features:
        return []
    names = [k for k in sc.cache if k.startswith("shift/") or k.startswith("real/")]
    names += [f"c/{c}/{s}" for c in ("gaussian_noise", "fog", "contrast", "jpeg_compression") for s in (3, 5) if f"c/{c}/{s}" in sc.cache]
    S_cal = sc.scores("temp_aps", "cal")
    y_cal = sc.cache["cal"]["labels"]
    s_true = true_label_scores(S_cal, y_cal)
    q0 = conformal_quantile(s_true, A0)
    g = torch.Generator().manual_seed(seed)
    rows = []
    for name in names:
        n = len(sc.cache[name]["labels"])
        p = torch.randperm(n, generator=g)
        a, b = p[: n // 2], p[n // 2:]
        f = sc.cache[name]["features"].float()
        w_cal, w_b = domain_weights(sc.cache["design"]["features"].float(), f[a], sc.cache["cal"]["features"].float(), f[b])
        S_b, y_b = sc.scores("temp_aps", name)[b], sc.cache[name]["labels"][b]
        q = weighted_thresholds(s_true, w_cal, w_b, A0)
        Cw = S_b <= q[:, None].to(S_b.dtype)
        C0 = S_b <= q0
        rows.append({"dataset": ds, "seed": seed, "set": name, "cov_standard": coverage(C0, y_b), "size_standard": C0.sum(1).float().mean().item(),
                     "cov_weighted": coverage(Cw, y_b), "size_weighted": Cw.sum(1).float().mean().item(),
                     "ess_cal": effective_sample_size(w_cal), "n_cal": len(w_cal), "frac_inf": (q == float("inf")).float().mean().item()})
    return rows


# ------------------------------------------------------------------------------------------------ G
def real_shift(sc: Scorer, ds: str, seed: int) -> list[dict]:
    rows = []
    for name in [k for k in sc.cache if k.startswith("real/") or k.startswith("c/")]:
        y = sc.cache[name]["labels"]
        acc = (sc.cache[name]["logits"].argmax(1) == y).float().mean().item()
        for m in MAIN:
            C = membership(sc, m, name, A0)
            parts = name.split("/")
            rows.append({"dataset": ds, "seed": seed, "set": name, "family": parts[0],
                         "corruption": parts[1] if parts[0] == "c" else "cifar10.1", "severity": int(parts[2]) if parts[0] == "c" else 0,
                         "method": m, "accuracy": acc, "coverage": coverage(C, y), "mean_size": C.sum(1).float().mean().item()})
    return rows


# ------------------------------------------------------------------------------------------------ H
def empty_sets(sc: Scorer, ds: str, seed: int, alphas) -> list[dict]:
    y = sc.cache["test"]["labels"]
    acc = (sc.cache["test"]["logits"].argmax(1) == y).float().mean().item()
    rows = []
    for alpha in alphas:
        for m in ("naive", "temp_naive", "margin", "conftr", "temp_aps", "mondrian"):
            C = membership(sc, m, "test", alpha)
            sizes = C.sum(1)
            cov = coverage(C, y)
            rows.append({"dataset": ds, "seed": seed, "alpha": alpha, "method": m, "accuracy": acc, "coverage": cov,
                         "empty_rate": (sizes == 0).float().mean().item(), "frac_size_ge2": (sizes >= 2).float().mean().item(),
                         "bound": theory.empty_rate_lower_bound(acc, cov), "bound_applies": bool((sizes <= 1).all())})
    return rows


# ------------------------------------------------------------------------------------------------ I
def distance_diag(sc: Scorer, ds: str, seed: int) -> list[dict]:
    if not sc.has_features:
        return []
    pr = sc.parts("test")
    y = sc.cache["test"]["labels"]
    pT, d = pr["pT"], pr["d"]
    true = torch.zeros_like(pT, dtype=torch.bool).scatter_(1, y[:, None], True)
    sub = torch.randperm(pT.numel(), generator=torch.Generator().manual_seed(0))[:200000]
    flat_t, flat_p, flat_d = true.flatten()[sub], pT.flatten()[sub], d.flatten()[sub]
    # rank-2 decision: is the second most probable label the true one? (the label a set decides to add or not)
    r2 = pT.argsort(1, descending=True)[:, 1]
    is_true = (r2 == y)
    p2, d2 = pT[torch.arange(len(y)), r2], d[torch.arange(len(y)), r2]
    rho = np.corrcoef(pd.Series(flat_p.numpy()).rank(), pd.Series(flat_d.numpy()).rank())[0, 1]
    return [{"dataset": ds, "seed": seed,
             "auroc_true_label_softmax": auroc(flat_p[~flat_t], flat_p[flat_t]),
             "auroc_true_label_distance": auroc(-flat_d[~flat_t], -flat_d[flat_t]),
             "auroc_rank2_softmax": auroc(p2[~is_true], p2[is_true]),
             "auroc_rank2_distance": auroc(-d2[~is_true], -d2[is_true]),
             "spearman_softmax_distance": float(rho), "gamma_naive": sc.gamma["dist_naive"], "gamma_aps": sc.gamma["dist_aps"]}]


# ------------------------------------------------------------------------------------------------ K
def class_shift(sc: Scorer, ds: str, seed: int) -> list[dict]:
    """Is each class equally hard in D_cal and D_test? Per-class two-proportion z-tests on top-1 correctness,
    summed into a chi-square with K degrees of freedom (exchangeable splits give chi2 ~ chi2_K)."""
    from scipy.stats import chi2
    K = sc.K
    stats = {}
    for name in ("cal", "test"):
        y = sc.cache[name]["labels"]
        ok = (sc.cache[name]["logits"].argmax(1) == y).double()
        stats[name] = (torch.bincount(y, weights=ok, minlength=K), torch.bincount(y, minlength=K).double())
    (kc, nc), (kt, nt) = stats["cal"], stats["test"]
    keep = (nc > 0) & (nt > 0)
    pool = (kc + kt) / (nc + nt)
    var = pool * (1 - pool) * (1 / nc + 1 / nt)
    good = keep & (var > 0)
    z2 = ((kt / nt - kc / nc) ** 2 / var)[good]
    stat, dof = float(z2.sum()), int(good.sum())
    gap = (kt / nt - kc / nc)[good]
    return [{"dataset": ds, "seed": seed, "K": K, "chi2": stat, "dof": dof, "p_value": float(chi2.sf(stat, dof)),
             "excess_ratio": stat / dof, "gap_sd": float(gap.std()), "sampling_sd": float(var[good].sqrt().mean())}]


# ------------------------------------------------------------------------------------------------ J
COMPARISONS = [("temp_aps", "naive"), ("mondrian", "temp_aps"), ("mondrian_pac", "mondrian"), ("clustered_pac", "mondrian_pac"),
               ("clustered", "mondrian"), ("dist_naive", "temp_naive"), ("conftr", "temp_naive"), ("saps", "raps")]


def per_point(sc: Scorer, methods) -> dict:
    y = sc.cache["test"]["labels"]
    out = {}
    for m in methods:
        C = membership(sc, m, "test", A0)
        out[m] = (C[torch.arange(len(y)), y].double(), C.sum(1).double())
    return out | {"_y": y}


def bootstrap(store: dict, B: int = 1000) -> list[dict]:
    rows = []
    rng = np.random.default_rng(0)
    for ds, seeds in store.items():
        seed_ids = list(seeds)
        for a, b in COMPARISONS:
            if a not in seeds[seed_ids[0]] or b not in seeds[seed_ids[0]]:
                continue
            per_seed = {}
            for s in seed_ids:
                st = seeds[s]
                y = st["_y"]
                K = int(y.max()) + 1
                onehot = torch.nn.functional.one_hot(y, K).double()
                per_seed[s] = (st[a], st[b], onehot)
            diffs = {"mean_size": [], "min_class_cov": [], "coverage": []}
            obs = {k: [] for k in diffs}
            for s in seed_ids:  # point estimate: mean over seeds of the per-seed difference
                (ha, sa), (hb, sb), oh = per_seed[s]
                obs["mean_size"].append((sa.mean() - sb.mean()).item())
                obs["coverage"].append((ha.mean() - hb.mean()).item())
                obs["min_class_cov"].append(((ha @ oh) / oh.sum(0)).min().item() - ((hb @ oh) / oh.sum(0)).min().item())
            for _ in range(B):
                pick = rng.choice(seed_ids, size=len(seed_ids), replace=True)
                acc = {k: [] for k in diffs}
                for s in pick:
                    (ha, sa), (hb, sb), oh = per_seed[s]
                    w = torch.as_tensor(rng.poisson(1.0, size=len(ha)), dtype=torch.float64)
                    acc["mean_size"].append(((sa - sb) @ w / w.sum()).item())
                    acc["coverage"].append(((ha - hb) @ w / w.sum()).item())
                    n_c = (w @ oh).clamp(min=1)
                    acc["min_class_cov"].append((((ha * w) @ oh) / n_c).min().item() - (((hb * w) @ oh) / n_c).min().item())
                for k in diffs:
                    diffs[k].append(np.mean(acc[k]))
            for k, v in diffs.items():
                v = np.array(v)
                rows.append({"dataset": ds, "a": a, "b": b, "metric": k, "estimate": float(np.mean(obs[k])),
                             "ci_lo": float(np.quantile(v, 0.025)), "ci_hi": float(np.quantile(v, 0.975)),
                             "p_le_0": float((v <= 0).mean()), "n_seeds": len(seed_ids)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--only", nargs="+", help="subset of A..J")
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--part", help="write to results/research/parts/PART/ (merge with scripts/merge_parts.py)")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    runs = args.runs or cfg["datasets"]
    todo = set(args.only or list("ABCDEFGHIJK"))
    out = OUT / "parts" / args.part if args.part else OUT
    out.mkdir(parents=True, exist_ok=True)
    T = {k: [] for k in ("worst_class", "ncal_sweep", "clusters", "label_shift", "aci", "aci_series", "weighted",
                         "real_shift", "empty_sets", "distance_diag", "class_shift")}
    boot_store: dict = {}
    for ds, seed, sc in scorers(cfg, runs):
        first = seed == seeds_for(cfg, ds)[0]
        if "A" in todo:
            T["worst_class"] += worst_class(sc, ds, seed)
        if "B" in todo and seed < 3:
            T["ncal_sweep"] += ncal_sweep(sc, ds, seed)
        if "C" in todo and seed < 5:
            T["clusters"] += clusters(sc, ds, seed)
        if "D" in todo and seed < 5 and sc.K <= 100:  # BBSE needs a well-conditioned K x K confusion matrix
            T["label_shift"] += label_shift(sc, ds, seed)
        if "E" in todo and seed < 5 and ds != "imagenet":
            s, ser = aci_runs(sc, ds, seed, keep_series=first)
            T["aci"] += s
            T["aci_series"] += ser
        if "F" in todo and seed < 5:
            T["weighted"] += weighted(sc, ds, seed)
        if "G" in todo:
            T["real_shift"] += real_shift(sc, ds, seed)
        if "H" in todo:
            T["empty_sets"] += empty_sets(sc, ds, seed, cfg["alphas"])
        if "I" in todo:
            T["distance_diag"] += distance_diag(sc, ds, seed)
        if "K" in todo:
            T["class_shift"] += class_shift(sc, ds, seed)
        if "J" in todo:
            boot_store.setdefault(ds, {})[seed] = per_point(sc, [m for m in methods_for(cfg, ds) if m != "top1"])
    for k, v in T.items():
        if v:
            pd.DataFrame(v).to_csv(out / f"{k}.csv", index=False)
            print(f"wrote {k}.csv ({len(v)} rows)")
    if boot_store:
        pd.DataFrame(bootstrap(boot_store)).to_csv(out / "bootstrap.csv", index=False)
        print("wrote bootstrap.csv")


if __name__ == "__main__":
    main()
