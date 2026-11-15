"""Glue shared by scripts/run_all.py and scripts/sweep.py.

Stage 1 (expensive, once per dataset x seed): train the classifier and cache logits/features for every
split, OOD set and shift level. Everything after that is post-hoc tensor maths on the cache.

Stage 2 (cheap): fit T, Mahalanobis, tune RAPS/gamma on D_design only, then build score matrices.
"""
from __future__ import annotations

import functools
from pathlib import Path

import torch

from src import data, models
from src.calibrate import apply_temperature, fit_temperature
from src.conformal import (
    build_sets, build_sets_grouped, build_sets_mondrian, cluster_classes, conformal_quantile, group_quantiles, kmeans,
    mondrian_quantiles, mondrian_quantiles_pac, true_label_scores,
)
from src.learned import train_conformal_head
from src.evaluate import evaluate_scores
from src.mahalanobis import fit_class_gaussians, mahalanobis_per_class
from src.scores import (
    aps_scores, distance_augmented_scores, margin_scores, naive_scores, normalise_distance, raps_scores, saps_scores,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT_SEEDS = {"design": 101, "cal": 202, "test": 303}  # distinct APS-randomisation streams per split


# --------------------------------------------------------------------------------------------- stage 1
def _cache_path(ds: str, seed: int, quick: bool = False) -> Path:
    return ROOT / "results" / "cache" / f"{ds}_s{seed}{'_quick' if quick else ''}.pt"


@functools.lru_cache(maxsize=1)
def _imagenet_outputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Load the 50k x 1000 softmax once per process (kept in memory, not duplicated into per-seed caches)."""
    import numpy as np
    z = np.load(ROOT / "data" / "imagenet" / "imagenet-resnet152.npz")
    logits = torch.as_tensor(z["smx"], dtype=torch.float32).clamp_min(1e-12).log()  # softmax(log p) = p
    return logits, torch.as_tensor(z["labels"].astype("int64"))


def _imagenet_cache(seed: int, cfg: dict) -> dict:
    """ImageNet val (50k, 1000 classes) from published ResNet-152 softmax outputs (Angelopoulos et al.).

    No images or features ship with it, so only logit-based methods run. Each seed re-draws a disjoint
    design / cal / test split of the 50k images (fractions in cfg["imagenet"]).
    """
    logits, labels = _imagenet_outputs()
    perm = torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed))
    n_de = round(cfg["imagenet"]["design"] * len(labels))
    n_ca = round(cfg["imagenet"]["calib"] * len(labels))
    idx = {"design": perm[:n_de], "cal": perm[n_de:n_de + n_ca], "test": perm[n_de + n_ca:]}
    cache = {k: {"logits": logits[v], "labels": labels[v]} for k, v in idx.items()}
    # T and the learned score are fit on design; "train" exists only so code that reads it has a split to read
    cache["train"] = cache["design"]
    cache["meta"] = {"split_idx": idx, "train_info": {"source": "imagenet-resnet152.npz"}}
    return cache


def load_net(ds: str, seed: int, cfg: dict, quick: bool = False):
    """Rebuild the trained network for (run, seed) from its checkpoint, with the D_train normalisation stats."""
    x_tr, *_ = data.load_in_distribution(ds, str(ROOT / "data"))
    sp = data.four_way_split(len(x_tr), cfg["split"], seed)
    mean, std = data.channel_stats(x_tr[sp.train])
    net = models.build_model(ds)
    net.load_state_dict(torch.load(ROOT / "checkpoints" / _cache_path(ds, seed, quick).with_suffix(".ckpt").name, map_location="cpu"))
    return net.to(models.get_device()), mean, std


def build_cache(ds: str, seed: int, cfg: dict, quick: bool = False) -> dict:
    """Train once and return {name: {"logits", "features", "labels"?}} for every evaluation set."""
    path = _cache_path(ds, seed, quick)
    if path.exists():
        return torch.load(path)
    if ds == "imagenet":  # cheap to rebuild from the shared outputs, so never written to disk
        return _imagenet_cache(seed, cfg)

    models.seed_everything(seed)
    x_tr, y_tr, x_te, y_te = data.load_in_distribution(ds, str(ROOT / "data"))
    sp = data.four_way_split(len(x_tr), cfg["split"], seed)
    mean, std = data.channel_stats(x_tr[sp.train])  # D_train only

    mcfg = dict(cfg["model"][ds])
    if quick:
        mcfg["epochs"] = 2
    net = models.build_model(ds)
    print(f"[train] {ds} seed={seed} on {models.get_device()} ({len(sp.train)} train / {len(sp.design)} design)", flush=True)
    ckpt = ROOT / "checkpoints" / path.with_suffix(".ckpt").name
    if ckpt.exists():  # weights already trained (e.g. a run that died while caching): reuse, don't retrain
        net.load_state_dict(torch.load(ckpt, map_location="cpu"))
        net.to(models.get_device())
        info = {"loaded_checkpoint": str(ckpt.name)}
    else:
        info = models.train_model(net, x_tr, y_tr, sp.train, sp.design, mean, std, data.build_transforms(ds), mcfg, ckpt)

    def run(x, y=None):
        logits, feats = models.predict_logits(net, x, mean, std)
        return {"logits": logits, "features": feats} | ({"labels": y} if y is not None else {})

    cache = {
        "train": run(x_tr[sp.train], y_tr[sp.train]),
        "design": run(x_tr[sp.design], y_tr[sp.design]),
        "cal": run(x_tr[sp.calib], y_tr[sp.calib]),
        "test": run(x_te, y_te),
        "meta": {"split_idx": sp.__dict__, "mean": mean, "std": std, "train_info": info},
    }
    for name in cfg["ood"].get(ds, []):
        try:
            cache[f"ood/{name}"] = run(data.load_ood(name, ds, str(ROOT / "data")))
        except (OSError, RuntimeError) as e:  # URLError is an OSError; OOD is optional, training is not
            print(f"[warn] skipping OOD set {name}: {e}")
    for sev in cfg["shift"].get(ds, []):
        cache[f"shift/{sev}"] = run(data.shift_images(x_te, ds, sev, seed), y_te)

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return cache


def extend_cache(ds: str, seed: int, cfg: dict) -> bool:
    """Add real-shift sets (CIFAR-10.1, CIFAR-10-C) to an existing CIFAR-10 cache. Returns True if it changed.

    Features are stored as float16 for the CIFAR-10-C sets to keep the cache small.
    """
    if data.base(ds) != "cifar10":
        return False
    path = _cache_path(ds, seed)
    cache = torch.load(path)
    root = str(ROOT / "data")
    todo = []
    if "real/cifar10_1" not in cache and (ROOT / "data" / "cifar10_1").exists():
        todo.append(("real/cifar10_1", lambda: data.load_cifar10_1(root), False))
    if (ROOT / "data" / "cifar10_c" / "CIFAR-10-C").exists():
        rc = cfg["real_shift"]["cifar10_c"]
        for c in rc["corruptions"]:
            for s in rc["severities"]:
                if f"c/{c}/{s}" not in cache:
                    todo.append((f"c/{c}/{s}", lambda c=c, s=s: data.load_cifar10_c(c, s, rc["n"], root), True))
    if not todo:
        return False
    net, mean, std = load_net(ds, seed, cfg)
    for name, load, half in todo:
        x, y = load()
        logits, feats = models.predict_logits(net, x, mean, std)
        cache[name] = {"logits": logits, "features": feats.half() if half else feats, "labels": y}
    torch.save(cache, path)
    print(f"[extend] {ds} s{seed}: +{len(todo)} sets", flush=True)
    return True


# --------------------------------------------------------------------------------------------- stage 2
GROUP_METHODS = ("mondrian", "mondrian_pac", "clustered", "clustered_pac")
FEATURE_METHODS = ("dist_naive", "dist_aps")


def methods_for(cfg: dict, ds: str) -> list[str]:
    """Methods that apply to a run (no feature-based methods where features are unavailable, e.g. ImageNet)."""
    skip = set(cfg.get("skip_methods", {}).get(ds, []))
    return [m for m in cfg["scores"] if m not in skip]


class Scorer:
    """Holds everything fitted on D_train / D_design and turns any cached set into score matrices."""

    def __init__(self, cache: dict, cfg: dict) -> None:
        self.cache = cache
        self.cfg = cfg
        de = cache["design"]
        self.K = de["logits"].shape[1]
        self.T = fit_temperature(de["logits"], de["labels"])
        self.has_features = "features" in cache["train"]
        if self.has_features:
            tr = cache["train"]
            self.mu, self.sigma_inv = fit_class_gaussians(tr["features"].float(), tr["labels"], self.K)
            d_design = mahalanobis_per_class(de["features"].float(), self.mu, self.sigma_inv)
            self.d_mean, self.d_std = d_design.mean().item(), d_design.std().item()  # D_design stats only
            # feature-space strata for FSC: k-means centroids on (a subsample of) D_train features
            f = tr["features"].float()
            sub = torch.randperm(len(f), generator=torch.Generator().manual_seed(0))[:10000]
            assign = kmeans(f[sub], cfg.get("fsc_clusters", 10), seed=0)
            self.centroids = torch.stack([f[sub][assign == j].mean(0) for j in assign.unique()])
        self.raps = {"lam": 0.0, "k_reg": 1}
        self.saps_lam = 0.1
        self.gamma = {"dist_naive": 0.0, "dist_aps": 0.0}
        self.head = None
        self.n_clusters = cfg.get("n_clusters") or max(2, round(self.K ** 0.5))
        self.delta = cfg.get("pac_delta", 0.1)
        self._memo: dict = {}
        self._clusters: dict = {}

    def feature_groups(self, name: str) -> torch.Tensor | None:
        if not self.has_features or "features" not in self.cache[name]:
            return None
        return torch.cdist(self.cache[name]["features"].float(), self.centroids).argmin(1)

    def parts(self, name: str) -> dict:
        """Raw probs, temperature-scaled probs and normalised distances for one cached set."""
        if name not in self._memo:
            c = self.cache[name]
            out = {"p": torch.softmax(c["logits"], 1), "pT": apply_temperature(c["logits"], self.T)}
            if self.has_features and "features" in c:
                d = mahalanobis_per_class(c["features"].float(), self.mu, self.sigma_inv)
                out["d"] = normalise_distance(d, self.d_mean, self.d_std)
            self._memo[name] = out
        return self._memo[name]

    def scores(self, method: str, name: str, gamma: float | None = None, raps: dict | None = None,
               saps_lam: float | None = None) -> torch.Tensor:
        """(n, K) score matrix for `method` on cached set `name`. Group methods reuse the temp_aps score."""
        pr = self.parts(name)
        seed = SPLIT_SEEDS.get(name, SPLIT_SEEDS["test"])  # OOD/shift sets are test-like
        raps = raps or self.raps
        if method in ("naive", "top1"):
            return naive_scores(pr["p"])
        if method == "aps":
            return aps_scores(pr["p"], seed=seed)
        if method == "raps":
            return raps_scores(pr["p"], raps["lam"], raps["k_reg"], seed=seed)
        if method == "temp_naive":
            return naive_scores(pr["pT"])
        if method == "temp_aps" or method in GROUP_METHODS:
            return aps_scores(pr["pT"], seed=seed)
        if method == "margin":
            return margin_scores(pr["pT"])
        if method == "saps":
            return saps_scores(pr["pT"], self.saps_lam if saps_lam is None else saps_lam, seed=seed)
        if method == "conftr":
            with torch.no_grad():
                return naive_scores(torch.softmax(self.head(self.cache[name]["logits"], self.T), 1))
        if method == "dist_naive":
            g = self.gamma["dist_naive"] if gamma is None else gamma
            return distance_augmented_scores(naive_scores(pr["pT"]), pr["d"], g)
        if method == "dist_aps":
            g = self.gamma["dist_aps"] if gamma is None else gamma
            return distance_augmented_scores(aps_scores(pr["pT"], seed=seed), pr["d"], g)
        raise ValueError(method)

    def clusters(self, n_clusters: int | None = None) -> torch.Tensor:
        """Class -> cluster map fit on D_design temp_aps scores (Ding et al., 2023). Cached per cluster count."""
        m = n_clusters or self.n_clusters
        if m not in self._clusters:
            S = self.scores("temp_aps", "design")
            y = self.cache["design"]["labels"]
            self._clusters[m] = cluster_classes(true_label_scores(S, y), y, self.K, m)
        return self._clusters[m]

    # ---- tuning on D_design only: split it in halves A/B, calibrate on one, evaluate on the other, swap.
    def _design_eval(self, method: str, alpha: float, **kw) -> tuple[float, float]:
        S = self.scores(method, "design", **kw)
        y = self.cache["design"]["labels"]
        half = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))
        a, b = half[: len(y) // 2], half[len(y) // 2:]
        sizes, mins = [], []
        for cal, ev in ((a, b), (b, a)):
            q = conformal_quantile(true_label_scores(S[cal], y[cal]), alpha)
            m = evaluate_scores(build_sets(S[ev], q), y[ev], alpha, self.K)
            sizes.append(m.mean_size)
            mins.append(m.min_class_cov)
        return sum(sizes) / 2, sum(mins) / 2

    def tune(self, cfg: dict, alpha: float) -> dict:
        """RAPS / SAPS: smallest mean size. gamma: best min-class coverage with mean size <= 1.10x the gamma=0 size.
        Learned score: trained on D_design halves. Class clusters: fit on D_design scores."""
        grid = [{"lam": l, "k_reg": k} for l in cfg["raps"]["lambda_grid"] for k in cfg["raps"]["k_reg_grid"]]
        self.raps = min(grid, key=lambda r: self._design_eval("raps", alpha, raps=r)[0])
        self.saps_lam = min(cfg["saps"]["lambda_grid"], key=lambda l: self._design_eval("saps", alpha, saps_lam=l)[0])
        if self.has_features:
            for method in self.gamma:
                base_size = self._design_eval(method, alpha, gamma=0.0)[0]
                ok = []
                for g in cfg["gamma_grid"]:
                    size, min_cov = self._design_eval(method, alpha, gamma=g)
                    if size <= 1.10 * base_size:
                        ok.append((min_cov, -g, g))  # tie-break toward the smaller gamma
                self.gamma[method] = max(ok)[2]
        de = self.cache["design"]
        self.head = train_conformal_head(de["logits"], de["labels"], self.T, alpha)
        self.clusters()
        return {"T": self.T, "raps": self.raps, "saps_lam": self.saps_lam, "gamma": dict(self.gamma),
                "n_clusters": self.n_clusters}


def membership(scorer: Scorer, method: str, name: str, alpha: float, cal: str = "cal", n_clusters: int | None = None,
               delta: float | None = None, **kw) -> torch.Tensor:
    """Calibrate `method` on D_cal and return the boolean (n, K) sets for cached set `name`."""
    if method == "top1":  # baseline: a single label, no guarantee
        logits = scorer.cache[name]["logits"]
        return torch.zeros_like(logits, dtype=torch.bool).scatter_(1, logits.argmax(1, keepdim=True), True)
    cal_y = scorer.cache[cal]["labels"]
    cal_s = true_label_scores(scorer.scores(method, cal, **kw), cal_y)
    S = scorer.scores(method, name, **kw)
    if method == "mondrian":
        return build_sets_mondrian(S, mondrian_quantiles(cal_s, cal_y, alpha, scorer.K))
    if method == "mondrian_pac":
        return build_sets_mondrian(S, mondrian_quantiles_pac(cal_s, cal_y, alpha, scorer.K, delta or scorer.delta))
    if method in ("clustered", "clustered_pac"):
        cmap = scorer.clusters(n_clusters)
        m = int(cmap.max()) + 1
        d = (delta or scorer.delta) if method == "clustered_pac" else None
        qg = group_quantiles(cal_s, cmap[cal_y], alpha, m, d)
        return build_sets_grouped(S, cmap, qg, conformal_quantile(cal_s, alpha))  # null cluster: marginal q (Ding et al.)
    return build_sets(S, conformal_quantile(cal_s, alpha))


def seeds_for(cfg: dict, ds: str) -> list[int]:
    """Seeds for a run: cfg["run_seeds"][ds] if given, else the global list."""
    return cfg.get("run_seeds", {}).get(ds, cfg["seeds"])
