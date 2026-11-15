#!/usr/bin/env python3
"""Write results/numbers.tex: one LaTeX macro per number quoted in the papers, computed from the result CSVs.

The papers never contain a hand-typed result; re-running this after new experiments updates every sentence.
Missing inputs render as a bold "??" so a gap is visible in the PDF.
    python scripts/paper_numbers.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import theory  # noqa: E402

RES, RQ = ROOT / "results", ROOT / "results" / "research"
TAG = {"mnist": "MNIST", "cifar10": "CTen", "cifar10_r18": "CTenR", "cifar100": "CHun", "imagenet": "INet"}
M_TAG = {"naive": "Naive", "temp_naive": "NaiveT", "temp_aps": "APST", "aps": "APS", "raps": "RAPS", "saps": "SAPS", "margin": "Margin",
         "conftr": "Learned", "dist_naive": "Dist", "dist_aps": "DistAPS", "mondrian": "Mond", "mondrian_pac": "MondPAC",
         "clustered": "Clus", "clustered_pac": "ClusPAC", "top1": "Top"}
out: dict[str, str] = {}


def read(p: Path):
    return pd.read_csv(p) if p.exists() else None


def pct(v, d=1):
    return "\\textbf{??}" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.{d}f}"


def num(v, d=2):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "\\textbf{??}"
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.{d}f}"


def put(name: str, value: str) -> None:
    assert name.isalpha(), name
    out[name] = value


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    main_ = read(RES / "main.csv")
    tuned = read(RES / "tuned.csv")
    if tuned is None and (RES / "tuned.json").exists():
        tuned = pd.read_json(RES / "tuned.json")
    a0 = cfg["main_alpha"]

    # ---- per run: accuracy, seeds, T, ECE
    for ds, tag in TAG.items():
        t = tuned[tuned.dataset == ds] if tuned is not None else pd.DataFrame()
        put(f"Acc{tag}", pct(t.accuracy.mean() if len(t) else None))
        put(f"Seeds{tag}", str(len(t)) if len(t) else "\\textbf{??}")
        put(f"T{tag}", num(t["T"].mean() if len(t) else None))
        put(f"EceRaw{tag}", num(t.ece_raw.mean() if len(t) else None, 3))
        put(f"EceT{tag}", num(t.ece_T.mean() if len(t) else None, 3))
        put(f"Ncls{tag}", num(t.n_cal.mean() / t.K.mean() if len(t) and "K" in t else None, 0))

    # ---- main metrics at the main alpha
    if main_ is not None:
        m0 = main_[main_.alpha == a0]
        for ds, tag in TAG.items():
            for m, mt in M_TAG.items():
                g = m0[(m0.dataset == ds) & (m0.method == m)]
                put(f"Cov{tag}{mt}", pct(g.coverage.mean() if len(g) else None))
                put(f"Size{tag}{mt}", num(g.mean_size.mean() if len(g) else None))
                put(f"Worst{tag}{mt}", pct(g.min_class_cov.mean() if len(g) else None))
                put(f"Empty{tag}{mt}", pct(g.empty_rate.mean() if len(g) else None))
                put(f"Fsc{tag}{mt}", pct(g.fsc_min.mean() if len(g) and "fsc_min" in g else None))
                put(f"Sscv{tag}{mt}", pct(g.sscv.mean() if len(g) and "sscv" in g else None))
        covs = m0[~m0.method.isin(["top1", "mondrian_pac", "clustered_pac"])].groupby(["dataset", "method"]).coverage.mean()  # PAC over-covers by design
        put("CovLo", pct(covs.min()))
        put("CovHi", pct(covs.max()))

    # ---- worst-class law
    wc = read(RQ / "worst_class.csv")
    for ds, tag in TAG.items():
        for m, mt in (("mondrian", "Mond"), ("mondrian_pac", "MondPAC")):
            g = None if wc is None else wc[(wc.dataset == ds) & (wc.method == m) & (wc.alpha == a0)]
            ok = g is not None and len(g)
            put(f"Pred{tag}{mt}", pct(g.pred_mean.mean() if ok else None))
            put(f"PredLo{tag}{mt}", pct(g.pred_q10.mean() if ok else None))
            put(f"PredHi{tag}{mt}", pct(g.pred_q90.mean() if ok else None))
            put(f"Meas{tag}{mt}", pct(g.measured_min.mean() if ok else None))
    if wc is not None:
        g = wc[wc.method.isin(["mondrian", "mondrian_pac"]) & (wc.pred_mean < 0.999)]
        err = (g.measured_min - g.pred_mean).abs()
        put("WcMaxErr", pct(err.groupby([g.dataset, g.method, g.alpha]).mean().max()))
        inside = ((g.measured_min >= g.pred_q10 - 1e-9) & (g.measured_min <= g.pred_q90 + 1e-9)).mean()
        put("WcInside", pct(inside, 0))
        put("WcN", str(len(g)))
    # exchangeability diagnostics: probability-integral-transform of the measured worst class under the law
    from scipy.stats import chi2 as _chi2, kstest
    for ds, tag in TAG.items():
        g = None if wc is None or "pred_cdf_at_measured" not in wc else wc[(wc.dataset == ds) & (wc.method == "mondrian") & (wc.alpha == a0)]
        ok = g is not None and len(g)
        put(f"Pit{tag}", num(g.pred_cdf_at_measured.mean() if ok else None))
        put(f"Fisher{tag}", num(float(_chi2.sf(-2 * np.log(g.pred_cdf_at_measured.clip(1e-12)).sum(), 2 * len(g))) if ok else None, 3))
        put(f"Ks{tag}", num(float(kstest(g.pred_cdf_at_measured, "uniform").pvalue) if ok and len(g) >= 5 else None, 2))
        put(f"MissPts{tag}", num(100 * (g.pred_mean - g.measured_min).mean() if ok else None, 1))
    cs = read(RQ / "class_shift.csv")
    for ds, tag in TAG.items():
        g = None if cs is None else cs[cs.dataset == ds]
        ok = g is not None and len(g)
        put(f"ChiRatio{tag}", num(g.excess_ratio.mean() if ok else None))
        put(f"ChiP{tag}", num(float(_chi2.sf(g.chi2.sum(), g.dof.sum())) if ok else None, 3))
    ns = read(RQ / "ncal_sweep.csv")
    if ns is not None:
        put("NsMaxErr", pct((ns.measured_mean - ns.pred_mean).abs().groupby([ns.dataset, ns.n_c]).mean().max()))

    # ---- calibration size needed
    for K, kt in ((10, "Ten"), (100, "Hun"), (1000, "Thou")):
        put(f"Need{kt}", num(theory.n_per_class_needed(K, a0, 0.02, 0.1), 0))
    put("PacMinINet", str(math.ceil(math.log(1 - (1 - 0.1) ** (1 / 1000)) / math.log(1 - a0))))  # n >= log(delta_g)/log(1-alpha)

    # ---- clusters
    cl = read(RQ / "clusters.csv")
    if cl is not None:
        for ds, tag in TAG.items():
            g = cl[cl.dataset == ds].groupby(["method", "M"]).mean(numeric_only=True)
            if "clustered_pac" not in g.index.get_level_values(0):
                continue
            pac = g.loc["clustered_pac"]
            # reference = the actual Mondrian-PAC run (the M = K cluster row differs where rare classes use the marginal threshold)
            mp = main_[(main_.alpha == a0) & (main_.dataset == ds) & (main_.method == "mondrian_pac")] if main_ is not None else pd.DataFrame()
            if not len(mp):
                continue
            ref = pd.Series({"min_class_cov": mp.min_class_cov.mean(), "mean_size": mp.mean_size.mean()})
            ok = pac[pac.min_class_cov >= ref.min_class_cov - 0.01]
            if not len(ok):
                continue
            best = ok.sort_values("mean_size").iloc[0]
            put(f"ClBestM{tag}", str(int(best.name)))
            put(f"ClBestSize{tag}", num(best.mean_size))
            put(f"ClBestWorst{tag}", pct(best.min_class_cov))
            put(f"ClRatio{tag}", num(ref.mean_size / best.mean_size, 1))

    # ---- label shift (strongest shift)
    ls = read(RQ / "label_shift.csv")
    if ls is not None:
        for ds, tag in TAG.items():
            g = ls[(ls.dataset == ds) & (ls.beta == ls.beta.min())]
            for col, ct in (("naive", "Naive"), ("temp_aps", "APST"), ("mondrian", "Mond"), ("mondrian_pac", "MondPAC"), ("ls_bbse", "Bbse"), ("ls_oracle", "Oracle")):
                ok = len(g) and col in g
                put(f"Ls{tag}{ct}Mean", pct(g[col].mean() if ok else None))
                put(f"Ls{tag}{ct}Min", pct(g[col].min() if ok else None))
            put(f"Ls{tag}Tv", pct(g.tv_shift.mean() if len(g) else None, 0))

    # ---- ACI
    ac = read(RQ / "aci.csv")
    if ac is not None:
        for ds, tag in TAG.items():
            g = ac[ac.dataset == ds]
            if not len(g):
                continue
            segs = [s for s in g.segment.unique() if s != "all"]
            last = segs[-1]
            for lr, lt in ((0.0, "Fix"), (0.005, "Aci")):
                h = g[(g.lr == lr) & (g.segment == last)]
                put(f"Aci{tag}{lt}Cov", pct(h.coverage.mean()))
                put(f"Aci{tag}{lt}Size", num(h.mean_size.mean(), 1))
                put(f"Aci{tag}{lt}All", pct(g[(g.lr == lr) & (g.segment == "all")].coverage.mean()))

    # ---- weighted covariate-shift CP
    wt = read(RQ / "weighted.csv")
    if wt is not None:
        gain = wt.cov_weighted - wt.cov_standard
        put("WtGainMean", pct(gain.mean()))
        put("WtGainMax", pct(gain.groupby([wt.dataset, wt.set]).mean().max()))
        put("WtEssMin", pct((wt.ess_cal / wt.n_cal).min(), 1))
        put("WtEssMinN", f"{wt.ess_cal.min():.1f}")
        for ds, tag in TAG.items():
            g = wt[wt.dataset == ds]
            for col, nm in (("cov_standard", "CovStd"), ("cov_weighted", "CovW"), ("frac_inf", "Inf")):
                put(f"Wt{nm}{tag}", pct(g[col].mean() if len(g) else None))
            for col, nm in (("size_standard", "SizeStd"), ("size_weighted", "SizeW")):
                put(f"Wt{nm}{tag}", num(g[col].mean() if len(g) else None))

    # ---- real shift
    rs = read(RQ / "real_shift.csv")
    if rs is not None:
        for ds, tag in (("cifar10", "CTen"), ("cifar10_r18", "CTenR")):
            for m, mt in (("naive", "Naive"), ("temp_aps", "APST"), ("top1", "Top")):
                g = rs[(rs.dataset == ds) & (rs.method == m)]
                put(f"Ten{tag}{mt}", pct(g[g.family == "real"].coverage.mean() if len(g) else None))
                for s in (1, 5):
                    put(f"Cc{tag}{mt}S{'One' if s == 1 else 'Five'}", pct(g[(g.family == "c") & (g.severity == s)].coverage.mean() if len(g) else None))
            g = rs[(rs.dataset == ds) & (rs.family == "real")]
            put(f"TenAcc{tag}", pct(g.accuracy.mean() if len(g) else None))

    # ---- empty sets, distance diagnosis
    es = read(RQ / "empty_sets.csv")
    if es is not None:
        g = es[(es.dataset == "mnist") & (es.alpha == a0) & (es.method == "naive")]
        put("EmptyMnist", pct(g.empty_rate.mean()))
        put("EmptyBoundMnist", pct(g.bound.mean()))
        h = es[es.bound_applies & es.method.isin(["naive", "temp_naive"])]
        put("EmptyHolds", pct(((h.empty_rate + 1e-9) >= h.bound).mean(), 0))
        put("EmptyN", str(len(h)))
    dd = read(RQ / "distance_diag.csv")
    if dd is not None:
        for ds, tag in TAG.items():
            g = dd[dd.dataset == ds]
            put(f"DdSoft{tag}", num(g.auroc_rank2_softmax.mean() if len(g) else None, 3))
            put(f"DdDist{tag}", num(g.auroc_rank2_distance.mean() if len(g) else None, 3))
            put(f"DdRho{tag}", num(g.spearman_softmax_distance.mean() if len(g) else None))

    # ---- re-split
    rp = read(RES / "resplit.csv")
    if rp is not None:
        g = rp[rp.n_cal == 100]
        put("BelowHundred", pct((g.coverage < 1 - a0).mean(), 0))
        put("SdHundred", pct(g.coverage.std(), 1))

    # ---- bootstrap headline intervals (worst-class differences)
    bs = read(RQ / "bootstrap.csv")
    if bs is not None:
        for ds, tag in TAG.items():
            for (a, b), ct in (((("mondrian_pac", "mondrian")), "PacVsMond"), (("clustered_pac", "mondrian_pac"), "ClusVsPac"),
                               (("temp_aps", "naive"), "ApsVsNaive"), (("dist_naive", "temp_naive"), "DistVsNaive")):
                for metric, mt in (("min_class_cov", "W"), ("mean_size", "S")):
                    g = bs[(bs.dataset == ds) & (bs.a == a) & (bs.b == b) & (bs.metric == metric)]
                    f = (lambda v: pct(v)) if metric == "min_class_cov" else (lambda v: num(v))
                    put(f"Bs{tag}{ct}{mt}", f(g.estimate.iloc[0]) if len(g) else "\\textbf{??}")
                    put(f"Bs{tag}{ct}{mt}Lo", f(g.ci_lo.iloc[0]) if len(g) else "\\textbf{??}")
                    put(f"Bs{tag}{ct}{mt}Hi", f(g.ci_hi.iloc[0]) if len(g) else "\\textbf{??}")

    # every macro a paper may use exists, even if its experiment has not run yet (renders as a bold ??)
    q = "\\textbf{??}"
    names = ["WtEssMinN"] + [f"Wt{n}{t}" for t in TAG.values() for n in ("CovStd", "CovW", "Inf", "SizeStd", "SizeW")] + ["WcMaxErr", "WcInside", "WcN", "NsMaxErr", "WtGainMean", "WtGainMax", "WtEssMin", "EmptyMnist", "EmptyBoundMnist",
             "EmptyHolds", "EmptyN", "BelowHundred", "SdHundred", "CovLo", "CovHi"]
    for tag in TAG.values():
        names += [f"ClBest{x}{tag}" for x in ("M", "Size", "Worst")] + [f"ClRatio{tag}", f"Ls{tag}Tv", f"TenAcc{tag}"]
        names += [f"Ls{tag}{c}{s}" for c in ("Naive", "APST", "Mond", "MondPAC", "Bbse", "Oracle") for s in ("Mean", "Min")]
        names += [f"Aci{tag}{l}{f}" for l in ("Fix", "Aci") for f in ("Cov", "Size", "All")]
        names += [f"Dd{x}{tag}" for x in ("Soft", "Dist", "Rho")] + [f"{x}{tag}" for x in ("Pit", "Fisher", "Ks", "MissPts", "ChiRatio", "ChiP")]
        names += [f"{x}{tag}{m}" for x in ("Pred", "PredLo", "PredHi", "Meas") for m in ("Mond", "MondPAC")]
        names += [f"Ten{tag}{m}" for m in ("Naive", "APST", "Top")] + [f"Cc{tag}{m}S{s}" for m in ("Naive", "APST", "Top") for s in ("One", "Five")]
        names += [f"Bs{tag}{c}{m}{e}" for c in ("PacVsMond", "ClusVsPac", "ApsVsNaive", "DistVsNaive") for m in ("W", "S") for e in ("", "Lo", "Hi")]
    for n in names:
        out.setdefault(n, q)

    lines = ["% Auto-generated by scripts/paper_numbers.py. Do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(out.items())]
    (RES / "numbers.tex").write_text("\n".join(lines) + "\n")
    missing = [k for k, v in out.items() if "??" in v]
    print(f"wrote {len(out)} macros to results/numbers.tex ({len(missing)} missing)")


if __name__ == "__main__":
    main()
