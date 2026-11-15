#!/usr/bin/env python3
"""Sweeps: alpha x gamma grid and seed aggregation, on cached logits/features (no training).

This is REPORTED sensitivity on D_test, not selection: gamma is chosen on D_design in run_all.py.

Usage:
    python scripts/sweep.py --config configs/default.yaml      # run after run_all.py has built the cache
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import plots  # noqa: E402
from src.evaluate import evaluate_scores  # noqa: E402
from src.pipeline import Scorer, build_cache, membership, seeds_for  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--datasets", nargs="+", help="override cfg datasets")
    p.add_argument("--quick", action="store_true", help="use the --quick cache from run_all.py")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    datasets = args.datasets or cfg["datasets"]
    rows = []
    for ds in datasets:
        for seed in (cfg["seeds"][:1] if args.quick else seeds_for(cfg, ds)):
            if ds == "imagenet" or not (ROOT / "results" / "cache" / f"{ds}_s{seed}{'_quick' if args.quick else ''}.pt").exists():
                continue  # no features for ImageNet; untrained runs are skipped
            scorer = Scorer(build_cache(ds, seed, cfg, args.quick), cfg)
            K = scorer.K
            y = scorer.cache["test"]["labels"]
            for alpha in cfg["alphas"]:
                for method in ("dist_naive", "dist_aps"):
                    for g in cfg["gamma_grid"]:
                        m = evaluate_scores(membership(scorer, method, "test", alpha, gamma=g), y, alpha, K)
                        rows.append({"dataset": ds, "seed": seed, "alpha": alpha, "method": method, "gamma": g, **asdict(m)})
                # temperature ablation: APS on raw probs (T=1) vs APS on temperature-scaled probs
                for method in ("aps", "temp_aps"):
                    m = evaluate_scores(membership(scorer, method, "test", alpha), y, alpha, K)
                    rows.append({"dataset": ds, "seed": seed, "alpha": alpha, "method": method, "gamma": None, **asdict(m)})

    df = pd.DataFrame(rows)
    out = ROOT / "results"
    df.to_csv(out / "sweep.csv", index=False)
    summary = df.groupby(["dataset", "method", "alpha", "gamma"], dropna=False)[
        ["coverage", "mean_size", "min_class_cov", "worst_class_gap", "empty_rate"]].agg(["mean", "std"])
    summary.to_csv(out / "sweep_summary.csv")
    d = df[df.method == "dist_aps"]
    plots.plot_sweep_heatmap(d, "mean_size", out / "figures" / "sweep_size.pdf")
    plots.plot_sweep_heatmap(d, "min_class_cov", out / "figures" / "sweep_min_cov.pdf")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
