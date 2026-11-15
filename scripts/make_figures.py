#!/usr/bin/env python3
"""Rebuild every figure and LaTeX table from the CSVs in results/ (no model or cache needed).

    python scripts/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import plots, theory  # noqa: E402

RES, FIG, RQ = ROOT / "results", ROOT / "results" / "figures", ROOT / "results" / "research"
PAPER_METHODS = ["temp_naive", "temp_aps", "raps", "saps", "conftr", "dist_naive", "mondrian", "mondrian_pac",
                 "clustered", "clustered_pac", "top1"]


def read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    a0 = cfg["main_alpha"]
    FIG.mkdir(parents=True, exist_ok=True)
    main_ = read(RES / "main.csv")
    if main_ is not None:
        m0 = main_[main_.alpha == a0]
        plots.write_latex_table(main_, a0, PAPER_METHODS, RES / "table_main.tex", runs=["cifar10", "cifar100", "imagenet"])
        plots.write_latex_table(main_, a0, cfg["scores"], RES / "table_full_a.tex", runs=["mnist", "cifar10", "cifar10_r18"])
        plots.write_latex_table(main_, a0, cfg["scores"], RES / "table_full_b.tex", runs=["cifar100", "imagenet"], size_std=True)
        plots.plot_coverage_vs_size(m0, FIG / "frontier.pdf", [m for m in PAPER_METHODS if m != "top1"])
    for name, fn in (("class_cov", lambda d: plots.plot_class_coverage(d, a0, FIG / "class_coverage.pdf")),
                     ("sizes", lambda d: plots.plot_size_histogram(d, FIG / "set_sizes.pdf")),
                     ("deciles", lambda d: plots.plot_decile_coverage(d, a0, FIG / "decile_coverage.pdf")),
                     ("shift", lambda d: plots.plot_shift_curve(d, a0, FIG / "shift.pdf")),
                     ("resplit", lambda d: plots.plot_coverage_histogram(d, a0, FIG / "resplit.pdf"))):
        d = read(RES / f"{name}.csv")
        if d is not None:
            fn(d)
    sw = read(RES / "sweep.csv")
    if sw is not None:
        d = sw[sw.method == "dist_aps"]
        plots.plot_sweep_heatmap(d, "mean_size", FIG / "sweep_size.pdf")
        plots.plot_sweep_heatmap(d, "min_class_cov", FIG / "sweep_min_cov.pdf")

    for name, fn in (("worst_class", lambda d: plots.plot_worst_class_theory(d, FIG / "worst_class_theory.pdf")),
                     ("ncal_sweep", lambda d: plots.plot_ncal_sweep(d, a0, FIG / "ncal_sweep.pdf")),
                     ("clusters", lambda d: plots.plot_clusters(d, a0, FIG / "clusters.pdf")),
                     ("label_shift", lambda d: plots.plot_label_shift(d, a0, FIG / "label_shift.pdf")),
                     ("aci_series", lambda d: plots.plot_aci(d, a0, FIG / "aci.pdf")),
                     ("real_shift", lambda d: plots.plot_real_shift(d, a0, FIG / "real_shift.pdf")),
                     ("empty_sets", lambda d: plots.plot_empty_bound(d, FIG / "empty_bound.pdf")),
                     ("bootstrap", lambda d: (plots.plot_bootstrap(d, FIG / "bootstrap_worst.pdf"),
                                              plots.plot_bootstrap(d, FIG / "bootstrap_size.pdf", "mean_size")))):
        d = read(RQ / f"{name}.csv")
        if d is not None and len(d):
            fn(d)

    rows = [{"K": K, "eps": e, "n_needed": theory.n_per_class_needed(K, a0, e, 0.1)}
            for e in (0.01, 0.02, 0.05) for K in (2, 5, 10, 20, 50, 100, 200, 500, 1000)]
    nn = pd.DataFrame(rows)
    RQ.mkdir(parents=True, exist_ok=True)
    nn.to_csv(RQ / "n_needed.csv", index=False)
    plots.plot_n_needed(nn, FIG / "n_needed.pdf")
    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
