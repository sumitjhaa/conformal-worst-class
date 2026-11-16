"""Figures and LaTeX tables for the papers.

Colour encodes the method *family* (fixed order from the validated reference palette, never cycled); within a family the
variant is the marker, and the high-probability (PAC) variants are hollow / dashed, so identity never rests on colour
alone. Every figure is a PDF sized for the 5.5in text width of the NeurIPS-style template.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import beta  # noqa: E402

from src.calibrate import reliability_bins  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GREY = "#8a8984"
COLORS = {
    "naive": BLUE, "temp_naive": BLUE, "margin": BLUE,
    "aps": ORANGE, "temp_aps": ORANGE,
    "raps": AQUA, "saps": AQUA,
    "conftr": YELLOW,
    "dist_naive": VIOLET, "dist_aps": VIOLET,
    "mondrian": RED, "mondrian_pac": RED,
    "clustered": MAGENTA, "clustered_pac": MAGENTA,
    "top1": GREY,
}
MARKERS = {"naive": "o", "temp_naive": "s", "margin": "d", "aps": "o", "temp_aps": "s", "raps": "o", "saps": "s",
           "conftr": "P", "dist_naive": "o", "dist_aps": "s", "mondrian": "D", "mondrian_pac": "D",
           "clustered": "^", "clustered_pac": "^", "top1": "X"}
HOLLOW = {"mondrian_pac", "clustered_pac"}
LABELS = {"naive": "Naive", "aps": "APS", "raps": "RAPS", "temp_naive": "Naive(T)", "temp_aps": "APS(T)",
          "dist_naive": "Naive(T)+dist", "dist_aps": "APS(T)+dist", "margin": "Margin", "saps": "SAPS",
          "conftr": "Learned", "mondrian": "Mondrian", "mondrian_pac": "Mondrian-PAC", "clustered": "Clustered",
          "clustered_pac": "Clustered-PAC", "top1": "Top-1"}
RUN_LABEL = {"mnist": "MNIST", "cifar10": "CIFAR-10", "cifar100": "CIFAR-100", "cifar10_r18": "CIFAR-10 (ResNet-18)",
             "imagenet": "ImageNet"}
RUN_ORDER = ["mnist", "cifar10", "cifar10_r18", "cifar100", "imagenet"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "legend.fontsize": 6,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4, "pdf.fonttype": 42,
    "legend.frameon": False,
})


def _runs(df: pd.DataFrame) -> list[str]:
    present = set(df["dataset"])
    return [r for r in RUN_ORDER if r in present]


def _fig(ncols: int, height: float = 2.0, width: float | None = None):
    fig, axes = plt.subplots(1, ncols, figsize=(width or min(2.3 * ncols, 7.2), height), squeeze=False)
    return fig, axes[0]


def _save(fig, out_path) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _mk(m: str, **kw) -> dict:
    """Marker style for a method: family colour, variant marker, hollow for PAC variants."""
    c = COLORS[m]
    return dict(marker=MARKERS[m], color=c, mfc="white" if m in HOLLOW else c, mec=c, **kw)


def _target(ax, alpha, axis="h") -> None:
    (ax.axhline if axis == "h" else ax.axvline)(1 - alpha, color=INK, ls="--", lw=0.7, zorder=0)


# ----------------------------------------------------------------------------------------- core results
def plot_coverage_vs_size(df: pd.DataFrame, out_path: str, methods: list[str] | None = None) -> None:
    """Efficiency frontier at one alpha: mean set size (x, log for many classes) vs worst-class coverage (y)."""
    alpha = df["alpha"].iloc[0]
    runs = _runs(df)
    fig, axes = _fig(len(runs), 2.1)
    for ax, ds in zip(axes, runs):
        d = df[(df.dataset == ds) & (df.method != "top1")]
        for m, g in d.groupby("method"):
            if methods and m not in methods:
                continue
            ax.errorbar(g.mean_size.mean(), g.min_class_cov.mean(), xerr=g.mean_size.std(), yerr=g.min_class_cov.std(),
                        ms=4.5, mew=0.9, elinewidth=0.7, ls="none", label=LABELS[m], **_mk(m))
        _target(ax, alpha)
        if d.mean_size.max() > 5:
            ax.set_xscale("log")
        ax.set(title=RUN_LABEL[ds], xlabel="mean set size")
    axes[0].set_ylabel("worst-class coverage")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 8), bbox_to_anchor=(0.5, -0.12))
    _save(fig, out_path)


def plot_size_histogram(sizes: pd.DataFrame, out_path: str) -> None:
    """Set-size distribution (share of test points) for four representative methods, pooled over seeds."""
    runs = _runs(sizes)
    fig, axes = _fig(len(runs), 1.9)
    methods = [m for m in ("temp_naive", "temp_aps", "mondrian", "clustered") if m in set(sizes.method)]
    for ax, ds in zip(axes, runs):
        d = sizes[sizes.dataset == ds]
        for m in methods:
            s = d[d.method == m].groupby("size")["count"].sum()
            s = s / s.sum()
            ax.step(s.index, s.values, where="mid", color=COLORS[m], ls="--" if m in HOLLOW else "-", lw=1.1, label=LABELS[m])
        if d["size"].max() > 20:
            ax.set_xscale("symlog", linthresh=10)
        ax.set(title=RUN_LABEL[ds], xlabel="set size")
    axes[0].set_ylabel("share of test images")
    axes[-1].legend(loc="upper right")
    _save(fig, out_path)


def plot_class_coverage(class_cov: pd.DataFrame, alpha: float, out_path: str) -> None:
    """Per-class coverage, sorted from worst to best (mean over seeds): the tail a marginal guarantee averages away."""
    runs = _runs(class_cov)
    methods = [m for m in ("temp_naive", "temp_aps", "mondrian", "clustered") if m in set(class_cov.method)]
    fig, axes = _fig(len(runs), 1.9)
    for ax, ds in zip(axes, runs):
        d = class_cov[class_cov.dataset == ds].groupby(["method", "cls"]).coverage.mean()
        for m in methods:
            v = np.sort(d[m].values)
            ax.plot(np.linspace(0, 1, len(v)), v, color=COLORS[m], lw=1.2, label=LABELS[m],
                    marker=MARKERS[m] if len(v) <= 10 else None, ms=3)
        _target(ax, alpha)
        ax.set(title=RUN_LABEL[ds], xlabel="classes, worst to best (fraction)")
    axes[0].set_ylabel("class coverage")
    axes[-1].legend(loc="lower right")
    _save(fig, out_path)


def plot_decile_coverage(deciles: pd.DataFrame, alpha: float, out_path: str) -> None:
    """Coverage by model-confidence decile (0 = least confident)."""
    runs = _runs(deciles)
    fig, axes = _fig(len(runs), 1.9)
    for ax, ds in zip(axes, runs):
        d = deciles[deciles.dataset == ds].groupby(["method", "decile"]).coverage.mean()
        for m in ("temp_naive", "temp_aps", "raps", "mondrian"):
            if m in d.index.get_level_values(0):
                ax.plot(d[m].index, d[m].values, ms=3, lw=1.1, label=LABELS[m], **_mk(m))
        _target(ax, alpha)
        ax.set(title=RUN_LABEL[ds], xlabel="confidence decile")
    axes[0].set_ylabel("coverage")
    axes[-1].legend(loc="lower right")
    _save(fig, out_path)


def plot_shift_curve(shift: pd.DataFrame, alpha: float, out_path: str) -> None:
    """Coverage vs synthetic shift severity, calibrated on clean data."""
    runs = _runs(shift)
    fig, axes = _fig(len(runs), 1.9)
    xlabel = {"mnist": "rotation (degrees)"}
    for ax, ds in zip(axes, runs):
        d = shift[shift.dataset == ds].groupby(["method", "severity"]).coverage.mean()
        for m in ("temp_naive", "temp_aps", "raps", "mondrian", "top1"):
            if m in d.index.get_level_values(0):
                ax.plot(d[m].index, d[m].values, ms=3, lw=1.1, label=LABELS[m], **_mk(m))
        _target(ax, alpha)
        ax.set(title=RUN_LABEL[ds], xlabel=xlabel.get(ds, "Gaussian noise std"))
    axes[0].set_ylabel("coverage")
    axes[-1].legend(loc="lower left")
    _save(fig, out_path)


def plot_coverage_histogram(resplit: pd.DataFrame, alpha: float, out_path: str) -> None:
    """Coverage over random calibration splits vs the exact law Beta(k, n+1-k), k = ceil((n+1)(1-alpha))."""
    runs = _runs(resplit)
    fig, axes = _fig(len(runs), 1.9)
    for ax, ds in zip(axes, runs):
        d = resplit[resplit.dataset == ds]
        ns = sorted(d.n_cal.unique())
        for n, color in ((ns[0], BLUE), (ns[-1], ORANGE)):
            v = d[d.n_cal == n].coverage
            ax.hist(v, bins=30, density=True, color=color, alpha=0.45, label=f"re-splits, n={n}")
            k = int(np.ceil((n + 1) * (1 - alpha)))
            xs = np.linspace(v.min() - 0.01, min(v.max() + 0.01, 1), 300)
            ax.plot(xs, beta.pdf(xs, k, n + 1 - k), color=color, lw=1.1, label=f"Beta law, n={n}")
        _target(ax, alpha, "v")
        ax.set(title=RUN_LABEL[ds], xlabel="coverage")
    axes[0].set_ylabel("density")
    axes[-1].legend(loc="upper left")
    _save(fig, out_path)


def plot_reliability(probs_raw, probs_t, labels, out_path: str, n_bins: int = 15) -> None:
    """Reliability diagram before/after temperature scaling (same bins as the ECE)."""
    fig, axes = _fig(1, 2.0, 2.4)
    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=0.8)
    for p, name, color in ((probs_raw, "raw", BLUE), (probs_t, "temperature-scaled", ORANGE)):
        count, acc, conf = reliability_bins(p, labels, n_bins)
        keep = count > 0
        ax.plot(conf[keep], acc[keep], marker="o", ms=3, color=color, label=name)
    ax.set(xlabel="confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left")
    _save(fig, out_path)


def plot_sweep_heatmap(sweep: pd.DataFrame, value: str, out_path: str) -> None:
    """alpha x gamma heatmap of `value` (mean over seeds), one panel per run; sequential single hue."""
    runs = _runs(sweep)
    fig, axes = _fig(len(runs), 2.0)
    for ax, ds in zip(axes, runs):
        t = sweep[sweep.dataset == ds].groupby(["alpha", "gamma"])[value].mean().unstack()
        im = ax.imshow(t.values, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(t.columns)), [f"{g:g}" for g in t.columns])
        ax.set_yticks(range(len(t.index)), [f"{a:g}" for a in t.index])
        ax.grid(False)
        for (i, j), v in np.ndenumerate(t.values):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                    color="white" if v > np.nanmean(t.values) else INK)
        ax.set(title=f"{RUN_LABEL[ds]}", xlabel=r"$\gamma$", ylabel=r"$\alpha$")
        fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, out_path)


# ----------------------------------------------------------------------------------------- research figures
def plot_worst_class_theory(wc: pd.DataFrame, out_path: str) -> None:
    """Measured worst-class coverage against the exact worst-class law, every run x seed x alpha (Mondrian, Mondrian-PAC)."""
    fig, axes = _fig(1, 2.4, 2.6)
    ax = axes[0]
    d = wc[wc.method.isin(["mondrian", "mondrian_pac"])].dropna(subset=["pred_mean"])
    d = d[d.pred_mean < 0.999]  # trivial full-set cases (no finite PAC threshold) are reported separately
    lo = min(d.pred_q10.min(), d.measured_min.min()) - 0.02
    ax.plot([lo, 1], [lo, 1], color=MUTED, lw=0.7, ls="--")
    for i, (ds, g) in enumerate(d.groupby("dataset")):
        for m, gm in g.groupby("method"):
            ax.errorbar(gm.pred_mean, gm.measured_min, xerr=np.clip([gm.pred_mean - gm.pred_q10, gm.pred_q90 - gm.pred_mean], 0, None),
                        ls="none", elinewidth=0.4, alpha=0.8, ms=3.5,
                        marker="osD^v"[RUN_ORDER.index(ds)], color=[BLUE, ORANGE, AQUA, VIOLET, RED][RUN_ORDER.index(ds)],
                        mfc="white" if m == "mondrian_pac" else [BLUE, ORANGE, AQUA, VIOLET, RED][RUN_ORDER.index(ds)],
                        label=f"{RUN_LABEL[ds]}" if m == "mondrian" else None)
    ax.set(xlabel="predicted worst-class coverage (exact law)", ylabel="measured worst-class coverage", xlim=(lo, 1), ylim=(lo, 1))
    ax.legend(loc="upper left", title="filled: Mondrian, hollow: PAC", title_fontsize=5.5)
    _save(fig, out_path)


def plot_ncal_sweep(ns: pd.DataFrame, alpha: float, out_path: str) -> None:
    """Worst-class coverage vs calibration images per class: re-split measurements (dots) vs exact law (line, 10-90% band)."""
    runs = _runs(ns)
    fig, axes = _fig(len(runs), 1.9)
    for ax, ds in zip(axes, runs):
        g = ns[ns.dataset == ds].groupby("n_c").mean(numeric_only=True)
        ax.fill_between(g.index, g.pred_q10, g.pred_q90, color=ORANGE, alpha=0.18, lw=0, label="theory 10-90%")
        ax.plot(g.index, g.pred_mean, color=ORANGE, lw=1.2, label="theory mean")
        ax.errorbar(g.index, g.measured_mean, yerr=np.clip([g.measured_mean - g.measured_q10, g.measured_q90 - g.measured_mean], 0, None),
                    fmt="o", ms=3, color=BLUE, elinewidth=0.7, label="measured")
        _target(ax, alpha)
        ax.set_xscale("log")
        ax.set(title=f"{RUN_LABEL[ds]} (K={int(g.K.iloc[0])})", xlabel="calibration images per class")
    axes[0].set_ylabel("worst-class coverage (Mondrian)")
    axes[0].legend(loc="lower right")
    _save(fig, out_path)


def plot_n_needed(table: pd.DataFrame, out_path: str) -> None:
    """Calibration images per class needed so every class is within eps of target w.p. 1-delta, vs number of classes."""
    fig, axes = _fig(1, 2.0, 2.6)
    ax = axes[0]
    for (eps, g), c in zip(table.groupby("eps"), (BLUE, ORANGE, AQUA)):
        ax.plot(g.K, g.n_needed, marker="o", ms=3, color=c, label=fr"$\varepsilon$={eps:g}")
    ax.set(xscale="log", yscale="log", xlabel="number of classes K", ylabel="images per class needed")
    ax.legend(title=r"$\alpha$=0.1, $\delta$=0.1", title_fontsize=5.5)
    _save(fig, out_path)


def plot_clusters(cl: pd.DataFrame, alpha: float, out_path: str, runs: list[str] | None = None) -> None:
    """Number of class clusters M from 1 (marginal) to K (one cluster per class): mean size vs worst-class coverage."""
    runs = [r for r in _runs(cl) if runs is None or r in runs]
    fig, axes = _fig(len(runs), 2.0)
    for ax, ds in zip(axes, runs):
        d = cl[cl.dataset == ds].groupby(["method", "M"]).mean(numeric_only=True)
        for m in ("clustered", "clustered_pac"):
            g = d.loc[m]
            # two colours as well as solid/dashed, so the pair is readable in greyscale and at small size
            c = BLUE if m in HOLLOW else COLORS[m]
            ax.plot(g.mean_size, g.min_class_cov, lw=1.0, ls="--" if m in HOLLOW else "-", ms=3.5, label=LABELS[m] if ax is axes[0] else None,
                    **{**_mk(m), "color": c, "mec": c, "mfc": "white" if m in HOLLOW else c})
            for M in (1, g.index.max()):
                ax.annotate(f"M={M}", (g.mean_size[M], g.min_class_cov[M]), fontsize=5, color=MUTED,
                            xytext=(3, -7 if m == "clustered" else 4), textcoords="offset points")
        _target(ax, alpha)
        ax.margins(y=0.12)  # room for the M= labels under the lowest point
        if d.mean_size.max() > 5:
            ax.set_xscale("log")
        ax.set(title=RUN_LABEL[ds], xlabel="mean set size")
    axes[0].set_ylabel("worst-class coverage")
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08), frameon=False)  # outside the panels: never covers data
    _save(fig, out_path)


def plot_label_shift(ls: pd.DataFrame, alpha: float, out_path: str) -> None:
    """Coverage under Dirichlet label shift vs total-variation size of the shift."""
    runs = _runs(ls)
    fig, axes = _fig(len(runs), 1.9)
    for ax, ds in zip(axes, runs):
        d = ls[ls.dataset == ds]
        for col, m, lab in (("naive", "naive", "Naive"), ("temp_aps", "temp_aps", "APS(T)"), ("mondrian", "mondrian", "Mondrian"),
                            ("ls_bbse", "clustered", "weighted (BBSE)")):
            if col not in d:
                continue
            ax.scatter(d.tv_shift, d[col], s=3, alpha=0.35, color=COLORS[m], marker=MARKERS[m], lw=0, label=lab)
        _target(ax, alpha)
        ax.set(title=RUN_LABEL[ds], xlabel="label shift (total variation)")
    axes[0].set_ylabel("coverage on shifted test set")
    axes[-1].legend(loc="lower left", markerscale=2.5)
    _save(fig, out_path)


def plot_aci(series: pd.DataFrame, alpha: float, out_path: str) -> None:
    """Rolling coverage over a stream drifting from clean to heavily corrupted, fixed vs adaptive conformal."""
    runs = _runs(series)
    fig, axes = _fig(len(runs), 1.9)
    for ax, ds in zip(axes, runs):
        d = series[series.dataset == ds]
        for lr, c, lab in ((0.0, GREY, "split conformal"), (0.005, BLUE, r"ACI $\eta$=0.005"), (0.02, ORANGE, r"ACI $\eta$=0.02")):
            g = d[d.lr == lr]
            ax.plot(g.t, g.rolling_cov, color=c, lw=1.0, label=lab)
        for seg, g in d[d.lr == 0].groupby("segment", sort=False):
            ax.axvline(g.t.min(), color=GRID, lw=0.6)
        _target(ax, alpha)
        ax.set(title=RUN_LABEL[ds], xlabel="time step (stream drifts →)", ylim=(0, 1.02))
    axes[0].set_ylabel("rolling coverage (250)")
    axes[-1].legend(loc="lower left")
    _save(fig, out_path)


def plot_real_shift(rs: pd.DataFrame, alpha: float, out_path: str) -> None:
    """CIFAR-10-C: coverage vs corruption severity (mean over 15 corruptions); CIFAR-10.1 at severity 0 offset."""
    runs = _runs(rs)
    fig, axes = _fig(len(runs), 1.9)
    for ax, ds in zip(axes, runs):
        d = rs[(rs.dataset == ds) & (rs.family == "c")].groupby(["method", "severity"]).coverage.mean()
        n1 = rs[(rs.dataset == ds) & (rs.family == "real")].groupby("method").coverage.mean()
        for m in ("temp_naive", "naive", "temp_aps", "raps", "mondrian", "clustered", "top1"):
            if m in d.index.get_level_values(0):
                ax.plot(d[m].index, d[m].values, ms=3, lw=1.0, label=LABELS[m], **_mk(m))
                if m in n1:
                    ax.plot([0], [n1[m]], ms=3.5, ls="none", **_mk(m))
        _target(ax, alpha)
        ax.set_xticks(range(6), ["10.1"] + [str(s) for s in range(1, 6)])
        ax.set(title=RUN_LABEL[ds], xlabel="CIFAR-10.1 | CIFAR-10-C severity")
    axes[0].set_ylabel("coverage")
    axes[-1].legend(loc="lower left")
    _save(fig, out_path)


def plot_empty_bound(es: pd.DataFrame, out_path: str) -> None:
    """Empty-set rate vs the lower bound acc - coverage, where every set has at most one label."""
    fig, axes = _fig(1, 2.2, 2.4)
    ax = axes[0]
    d = es[es.bound_applies & es.method.isin(["naive", "temp_naive"])]
    for i, ds in enumerate(_runs(d)):
        g = d[d.dataset == ds]
        ax.scatter(g.bound, g.empty_rate, s=10, color=[BLUE, ORANGE, AQUA, VIOLET, RED][RUN_ORDER.index(ds)],
                   marker="osD^v"[RUN_ORDER.index(ds)], label=RUN_LABEL[ds], lw=0)
    m = max(d.empty_rate.max(), d.bound.max()) * 1.05
    ax.plot([0, m], [0, m], color=MUTED, lw=0.7, ls="--")
    ax.set(xlabel="bound: accuracy - coverage", ylabel="measured empty-set rate")
    ax.legend(loc="upper left")
    _save(fig, out_path)


def plot_bootstrap(bs: pd.DataFrame, out_path: str, metric: str = "min_class_cov") -> None:
    """Forest plot: paired differences with 95% hierarchical-bootstrap intervals."""
    d = bs[bs.metric == metric].copy()
    d["label"] = d.apply(lambda r: f"{RUN_LABEL[r.dataset]}: {LABELS[r.a]} − {LABELS[r.b]}", axis=1)
    d = d.iloc[::-1]
    fig, axes = _fig(1, 0.16 * len(d) + 0.5, 4.0)
    ax = axes[0]
    y = np.arange(len(d))
    ax.errorbar(d.estimate, y, xerr=np.clip([d.estimate - d.ci_lo, d.ci_hi - d.estimate], 0, None), fmt="o", ms=2.5, color=BLUE, elinewidth=0.8)
    ax.axvline(0, color=INK, lw=0.6)
    ax.set_yticks(y, d.label)
    ax.tick_params(axis="y", labelsize=5)
    ax.set(xlabel={"min_class_cov": "difference in worst-class coverage", "mean_size": "difference in mean set size"}[metric])
    _save(fig, out_path)


# ----------------------------------------------------------------------------------------- tables
def write_latex_table(main: pd.DataFrame, alpha: float, methods: list[str], out_path, runs: list[str] | None = None,
                      size_std: bool = False) -> None:
    """booktabs table: coverage / mean size / worst-class coverage per run at `alpha`, mean over seeds."""
    d = main[main.alpha == alpha]
    runs = runs or _runs(d)
    lines = [r"\begin{tabular}{l" + "ccc" * len(runs) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{RUN_LABEL[ds]}}}" for ds in runs) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(runs))),
             "Method" + " & Cov. & Size & Worst" * len(runs) + r" \\", r"\midrule"]
    for m in methods:
        cells = []
        for ds in runs:
            r = d[(d.dataset == ds) & (d.method == m)]
            if r.empty:
                cells += ["--"] * 3
                continue
            size = r.mean_size.mean()
            sz = f"{size:.2f}" if size < 100 else f"{size:.0f}"
            if size_std and len(r) > 1:
                sz += rf"\,{{\scriptsize$\pm${r.mean_size.std():.2f}}}"
            cells += [f"{r.coverage.mean():.3f}", sz, f"{r.min_class_cov.mean():.3f}"]
        lines.append(LABELS[m] + " & " + " & ".join(cells) + r" \\")
        if m in ("conftr", "dist_aps"):
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}"]
    Path(out_path).write_text("\n".join(lines) + "\n")
