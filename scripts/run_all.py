#!/usr/bin/env python3
"""End-to-end entry point: regenerates every figure and table into results/.

Usage:
    python scripts/run_all.py --config configs/default.yaml
    python scripts/run_all.py --config configs/default.yaml --dry-run
    python scripts/run_all.py --datasets mnist --quick      # 1 seed, 2 epochs: smoke test

Stages: train+cache (skipped if cached) -> tune on D_design -> calibrate on D_cal -> evaluate on D_test
        -> depth analyses (per-class, deciles, OOD, shift, re-split) -> tables + figures.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import plots  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts"))
import make_figures  # noqa: E402
from src.calibrate import expected_calibration_error  # noqa: E402
from src.conformal import class_conditional_coverage, conformal_quantile, coverage, true_label_scores  # noqa: E402
from src.evaluate import evaluate_scores, ood_report, set_sizes, subgroup_coverage  # noqa: E402
from src.pipeline import Scorer, build_cache, membership, methods_for, seeds_for  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    p.add_argument("--datasets", nargs="+", help="override cfg datasets")
    p.add_argument("--quick", action="store_true", help="1 seed, 2 epochs, separate cache: smoke test only")
    p.add_argument("--part", help="write tables to results/parts/PART/ only (merge with scripts/merge_parts.py)")
    return p.parse_args()


def resplit_coverages(scorer: Scorer, method: str, alpha: float, n_cal: int, reps: int, seed: int) -> list[float]:
    """Pool D_cal and D_test (exactly exchangeable after shuffling), re-split `reps` times, return coverages."""
    S = torch.cat([scorer.scores(method, "cal"), scorer.scores(method, "test")])
    y = torch.cat([scorer.cache["cal"]["labels"], scorer.cache["test"]["labels"]])
    s_true = true_label_scores(S, y)
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(reps):
        perm = torch.randperm(len(y), generator=g)
        cal, ev = perm[:n_cal], perm[n_cal:]
        # Y in C(X) iff s(X, Y) <= q, so coverage only needs the true-label scores (no K-wide sets)
        out.append((s_true[ev] <= conformal_quantile(s_true[cal], alpha)).double().mean().item())
    return out


def run_one(ds: str, seed: int, cfg: dict, quick: bool, tables: dict) -> None:
    cache = build_cache(ds, seed, cfg, quick)
    scorer = Scorer(cache, cfg)
    a0 = cfg["main_alpha"]
    tuned = scorer.tune(cfg, a0)
    test_y = cache["test"]["labels"]
    K = scorer.K
    fg = scorer.feature_groups("test")
    tuned |= {
        "K": K,
        "accuracy": (cache["test"]["logits"].argmax(1) == test_y).float().mean().item(),
        "ece_raw": expected_calibration_error(scorer.parts("test")["p"], test_y),
        "ece_T": expected_calibration_error(scorer.parts("test")["pT"], test_y),
        "n_cal": len(cache["cal"]["labels"]),
    }
    tables["tuned"].append({"dataset": ds, "seed": seed, **tuned})
    print(f"[{ds} s{seed}] acc={tuned['accuracy']:.4f} T={tuned['T']:.3f} "
          f"ECE {tuned['ece_raw']:.4f}->{tuned['ece_T']:.4f} raps={tuned['raps']} gamma={tuned['gamma']}")

    methods = methods_for(cfg, ds)
    for alpha in cfg["alphas"]:
        for m in methods:
            C = membership(scorer, m, "test", alpha)
            tables["main"].append({"dataset": ds, "seed": seed, "alpha": alpha, "method": m,
                                   **asdict(evaluate_scores(C, test_y, alpha, K, fg))})

    # ---- depth analyses at the main alpha
    conf = scorer.parts("test")["pT"].max(1).values
    for m in methods:
        C = membership(scorer, m, "test", a0)
        for c, v in enumerate(class_conditional_coverage(C, test_y, K).tolist()):
            tables["class_cov"].append({"dataset": ds, "seed": seed, "method": m, "cls": c, "coverage": v})
        for b, v in enumerate(subgroup_coverage(C, test_y, conf).tolist()):
            tables["deciles"].append({"dataset": ds, "seed": seed, "method": m, "decile": b, "coverage": v})
        for s, cnt in zip(*set_sizes(C).unique(return_counts=True)):
            tables["sizes"].append({"dataset": ds, "seed": seed, "method": m, "size": int(s), "count": int(cnt)})
        for name in [k for k in cache if k.startswith("ood/")]:
            tables["ood"].append({"dataset": ds, "seed": seed, "method": m, "ood": name[4:],
                                  **ood_report(membership(scorer, m, name, a0), a0, C)})
        for name in [k for k in cache if k.startswith("shift/")]:
            met = evaluate_scores(membership(scorer, m, name, a0), cache[name]["labels"], a0, K)
            tables["shift"].append({"dataset": ds, "seed": seed, "method": m, "severity": float(name[6:]),
                                    "coverage": met.coverage, "mean_size": met.mean_size})
        tables["shift"].append({"dataset": ds, "seed": seed, "method": m, "severity": 0.0,
                                "coverage": coverage(C, test_y), "mean_size": set_sizes(C).float().mean().item()})

    for n_cal in cfg["resplit"]["n_cal"] + [tuned["n_cal"]]:
        for v in resplit_coverages(scorer, "temp_aps", a0, n_cal, 20 if quick else cfg["resplit"]["reps"], seed):
            tables["resplit"].append({"dataset": ds, "seed": seed, "n_cal": n_cal, "coverage": v})

    if seed == seeds_for(cfg, ds)[0]:  # reliability diagram from one seed is enough
        plots.plot_reliability(scorer.parts("test")["p"], scorer.parts("test")["pT"], test_y,
                               ROOT / "results" / "figures" / f"reliability_{ds}.pdf")


def write_outputs(tables: dict, cfg: dict) -> None:
    out = ROOT / "results"
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    dfs = {k: pd.DataFrame(v) for k, v in tables.items() if v}
    for k, df in dfs.items():
        df.to_csv(out / f"{k}.csv", index=False)
    (out / "tuned.json").write_text(json.dumps(tables["tuned"], indent=2))

    main = dfs["main"]
    agg = main.groupby(["dataset", "alpha", "method"]).agg(["mean", "std"]).drop(columns="seed")
    agg.to_csv(out / "main_summary.csv")
    make_figures.main()
    print(f"\nwrote tables to {out} and figures to {figs}")
    print(agg.xs(cfg["main_alpha"], level="alpha")[[("coverage", "mean"), ("mean_size", "mean"),
                                                     ("min_class_cov", "mean")]].round(4).to_string())


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    if args.datasets:
        cfg["datasets"] = args.datasets
    if args.quick:
        cfg["seeds"] = cfg["seeds"][:1]

    if args.dry_run:
        print("Plan:")
        for k in ("datasets", "seeds", "alphas", "scores", "gamma_grid", "main_alpha"):
            print(f"  {k:10s}: {cfg[k]}")
        print("OK (dry run)")
        return

    tables = {k: [] for k in ("tuned", "main", "class_cov", "deciles", "sizes", "ood", "shift", "resplit")}
    for ds in cfg["datasets"]:
        for seed in (cfg["seeds"] if args.quick else seeds_for(cfg, ds)):
            if not args.quick and not (ROOT / "results" / "cache" / f"{ds}_s{seed}.pt").exists() and ds != "imagenet":
                print(f"[skip] {ds} s{seed}: not trained yet (run scripts/train_all.py)")
                continue
            run_one(ds, seed, cfg, args.quick, tables)
    if args.part:
        out = ROOT / "results" / "parts" / args.part
        out.mkdir(parents=True, exist_ok=True)
        for k, v in tables.items():
            if v:
                pd.DataFrame(v).to_csv(out / f"{k}.csv", index=False)
        print(f"wrote part {args.part}")
        return
    write_outputs(tables, cfg)


if __name__ == "__main__":
    main()
