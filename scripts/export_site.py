#!/usr/bin/env python3
"""Export cached logits + precomputed results into site/data/ for the interactive frontend.

The browser recomputes conformal sets live from these logits, so every slider is real conformal maths,
not a lookup table. Run after scripts/run_all.py, scripts/sweep.py and scripts/research.py:
    python scripts/export_site.py
"""
from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data  # noqa: E402
from src.pipeline import Scorer, build_cache, membership, seeds_for  # noqa: E402

OUT = ROOT / "site" / "data"
CIFAR10 = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
LAB = {"mnist": 5000, "cifar10": 5000, "cifar100": 3000}          # live-lab runs and their test subset sizes
N_SHIFT, N_OOD, N_GALLERY, N_OOD_IMG, N_REAL = 1500, 500, 96, 12, 500
REAL_C = ["gaussian_noise", "motion_blur", "fog", "jpeg_compression"]


def b64_i16(x: torch.Tensor, scale: float = 100) -> str:
    """Quantise to int16 at 1/scale resolution and base64 it (row-major). Decoded in site/app.js."""
    a = (x.float() * scale).round().clamp(-32767, 32767).to(torch.int16).numpy()
    return base64.b64encode(a.astype("<i2").tobytes()).decode()


def png(img: torch.Tensor) -> str:
    a = img.permute(1, 2, 0).numpy()
    im = Image.fromarray(a[..., 0] if a.shape[2] == 1 else a)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def jpeg_file(path: Path, side: int = 112) -> str:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=72)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def block(scorer: Scorer, name: str, idx: torch.Tensor | None = None) -> dict:
    c = scorer.cache[name]
    sel = (lambda t: t) if idx is None else (lambda t: t[idx])
    logits = sel(c["logits"])
    out = {"n": len(logits), "logits": b64_i16(logits)}
    with torch.no_grad():
        out["learned"] = b64_i16(scorer.head(logits, scorer.T))  # residual head's log-probabilities
    if "d" in scorer.parts(name):
        out["dist"] = b64_i16(sel(scorer.parts(name)["d"]))
    if "labels" in c:
        out["labels"] = sel(c["labels"]).tolist()
    return out


def class_names(ds: str) -> list[str]:
    if ds == "mnist":
        return [str(i) for i in range(10)]
    if data.base(ds) == "cifar10":
        return CIFAR10
    if ds == "cifar100":
        import torchvision
        return torchvision.datasets.CIFAR100(str(ROOT / "data"), train=False, download=False).classes
    return json.loads((ROOT / "data" / "imagenet" / "human_readable_labels.json").read_text())


def export_lab(ds: str, cfg: dict) -> dict:
    seed = seeds_for(cfg, ds)[0]
    scorer = Scorer(build_cache(ds, seed, cfg), cfg)
    tuned = scorer.tune(cfg, cfg["main_alpha"])
    g = torch.Generator().manual_seed(7)
    test_idx = torch.randperm(len(scorer.cache["test"]["labels"]), generator=g)[:LAB[ds]].sort().values
    _, _, x_te, _ = data.load_in_distribution(ds, str(ROOT / "data"))
    gallery = [{"i": i, "img": png(x_te[test_idx[i]])} for i in range(N_GALLERY)]

    shift = {}
    for sev in cfg["shift"].get(ds, []):
        sidx = test_idx[:N_SHIFT]
        shift[str(sev)] = block(scorer, f"shift/{sev}", sidx)
        shift[str(sev)]["imgs"] = [png(x) for x in data.shift_images(x_te[sidx[:8]], ds, sev, seed)]

    ood = {}
    for name in [k[4:] for k in scorer.cache if k.startswith("ood/")]:
        x_ood = data.load_ood(name, ds, str(ROOT / "data"))
        oidx = torch.randperm(len(x_ood), generator=g)[:N_OOD]
        ood[name] = block(scorer, f"ood/{name}", oidx)
        ood[name]["imgs"] = [png(x_ood[j]) for j in oidx[:N_OOD_IMG]]

    real = {}
    if "real/cifar10_1" in scorer.cache:
        x1, _ = data.load_cifar10_1(str(ROOT / "data"))
        ridx = torch.randperm(len(x1), generator=g)[:N_REAL].sort().values
        real["CIFAR-10.1"] = block(scorer, "real/cifar10_1", ridx) | {"imgs": [png(x1[j]) for j in ridx[:8]]}
    for c in REAL_C:
        for s in range(1, 6):
            key = f"c/{c}/{s}"
            if key in scorer.cache:
                xc, _ = data.load_cifar10_c(c, s, cfg["real_shift"]["cifar10_c"]["n"], str(ROOT / "data"))
                real[f"{c.replace('_', ' ')} · {s}"] = block(scorer, key, torch.arange(N_REAL)) | {"imgs": [png(x) for x in xc[:8]]}

    acc = (scorer.cache["test"]["logits"].argmax(1) == scorer.cache["test"]["labels"]).float().mean().item()
    return {
        "dataset": ds, "seed": seed, "K": scorer.K, "classes": class_names(ds), "accuracy": acc,
        "tuned": {"T": tuned["T"], "raps": tuned["raps"], "saps_lam": tuned["saps_lam"], "gamma": tuned["gamma"],
                  "n_clusters": tuned["n_clusters"], "delta": scorer.delta},
        "clusters": scorer.clusters().tolist(),
        "cal": block(scorer, "cal"), "test": block(scorer, "test", test_idx), "gallery": gallery,
        "shift": shift, "ood": ood, "real": real,
        "train_n": len(scorer.cache["train"]["labels"]), "design_n": len(scorer.cache["design"]["labels"]),
    }


def export_imagenet_gallery(cfg: dict, n: int = 36) -> dict:
    """Example ImageNet test images with the sets several methods give them at the main alpha (seed 0 split)."""
    import numpy as np_
    seed = seeds_for(cfg, "imagenet")[0]
    sc = Scorer(build_cache("imagenet", seed, cfg), cfg)
    sc.tune(cfg, cfg["main_alpha"])
    names = class_names("imagenet")
    ex = np_.load(ROOT / "data" / "imagenet" / "imagenet-resnet152.npz")["example_indexes"].astype(int)
    test_pos = {int(v): i for i, v in enumerate(sc.cache["meta"]["split_idx"]["test"].tolist())}
    rows = [(int(e), test_pos[int(e)]) for e in ex if int(e) in test_pos and (ROOT / "data" / "imagenet" / "examples" / f"{int(e)}.jpeg").exists()][:n]
    methods = ["temp_naive", "temp_aps", "mondrian", "clustered", "clustered_pac"]
    sets = {m: membership(sc, m, "test", cfg["main_alpha"]) for m in methods}
    pT = sc.parts("test")["pT"]
    y = sc.cache["test"]["labels"]
    items = []
    for e, i in rows:
        top = pT[i].argsort(descending=True)[:8]
        items.append({"img": jpeg_file(ROOT / "data" / "imagenet" / "examples" / f"{e}.jpeg"), "true": names[int(y[i])],
                      "top": [[names[int(k)], round(float(pT[i, k]), 3)] for k in top],
                      "sets": {m: {"size": int(sets[m][i].sum()), "covered": bool(sets[m][i, y[i]]),
                                   "in": [bool(sets[m][i, k]) for k in top]} for m in methods}})
    return {"items": items, "methods": methods, "K": sc.K}


def records(df: pd.DataFrame | None) -> list[dict]:
    if df is None or not len(df):
        return []
    return json.loads(df.round(5).to_json(orient="records"))


def read(p: Path) -> pd.DataFrame | None:
    return pd.read_csv(p) if p.exists() else None


def summary(cfg: dict, done: list[str]) -> dict:
    res, rq = ROOT / "results", ROOT / "results" / "research"
    S: dict = {"datasets": done}
    S["main"] = records(read(res / "main.csv"))
    cc = read(res / "class_cov.csv")
    S["class_cov"] = records(None if cc is None else cc.groupby(["dataset", "method", "cls"], as_index=False).coverage.mean())
    dec = read(res / "deciles.csv")
    S["deciles"] = records(None if dec is None else dec.groupby(["dataset", "method", "decile"], as_index=False).coverage.mean())
    sz = read(res / "sizes.csv")
    S["sizes"] = records(None if sz is None else sz.groupby(["dataset", "method", "size"], as_index=False)["count"].sum())
    for k in ("ood", "shift"):
        d = read(res / f"{k}.csv")
        keys = ["dataset", "method", "ood"] if k == "ood" else ["dataset", "method", "severity"]
        S[k] = records(None if d is None else d.groupby(keys, as_index=False).mean(numeric_only=True).drop(columns="seed"))
    S["sweep"] = records(read(res / "sweep.csv"))
    S["tuned"] = json.loads((res / "tuned.json").read_text()) if (res / "tuned.json").exists() else []
    # research
    wc = read(rq / "worst_class.csv")
    S["worst_class"] = records(wc)
    ns = read(rq / "ncal_sweep.csv")
    S["ncal_sweep"] = records(None if ns is None else ns.groupby(["dataset", "K", "n_c"], as_index=False).mean(numeric_only=True).drop(columns="seed"))
    cl = read(rq / "clusters.csv")
    S["clusters"] = records(None if cl is None else cl.groupby(["dataset", "K", "M", "method"], as_index=False).mean(numeric_only=True).drop(columns="seed"))
    ls = read(rq / "label_shift.csv")
    S["label_shift"] = records(None if ls is None else ls.sample(frac=1, random_state=0).groupby("dataset").head(400))
    S["aci_series"] = records(read(rq / "aci_series.csv"))
    ac = read(rq / "aci.csv")
    S["aci"] = records(None if ac is None else ac.groupby(["dataset", "lr", "segment"], as_index=False, sort=False).mean(numeric_only=True).drop(columns="seed"))
    wt = read(rq / "weighted.csv")
    S["weighted"] = records(None if wt is None else wt.groupby(["dataset", "set"], as_index=False).mean(numeric_only=True).drop(columns="seed"))
    rs = read(rq / "real_shift.csv")
    S["real_shift"] = records(None if rs is None else rs.groupby(["dataset", "family", "corruption", "severity", "method"], as_index=False).mean(numeric_only=True).drop(columns="seed"))
    es = read(rq / "empty_sets.csv")
    S["empty_sets"] = records(None if es is None else es.groupby(["dataset", "alpha", "method"], as_index=False).mean(numeric_only=True).drop(columns="seed"))
    dd = read(rq / "distance_diag.csv")
    S["distance_diag"] = records(None if dd is None else dd.groupby("dataset", as_index=False).mean(numeric_only=True).drop(columns="seed"))
    S["bootstrap"] = records(read(rq / "bootstrap.csv"))
    S["n_needed"] = records(read(rq / "n_needed.csv"))
    S["config"] = {k: cfg[k] for k in ("alphas", "scores", "gamma_grid", "main_alpha", "split", "seeds", "run_seeds", "shift", "ood", "pac_delta")}
    S["config"]["runs"] = cfg["datasets"]
    return S


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    done = []
    for ds in LAB:
        if not (ROOT / "results" / "cache" / f"{ds}_s{seeds_for(cfg, ds)[0]}.pt").exists():
            print(f"skip {ds}: no cache (run scripts/train_all.py first)")
            continue
        (OUT / f"{ds}.json").write_text(json.dumps(export_lab(ds, cfg), separators=(",", ":")))
        done.append(ds)
        print(f"wrote {ds}.json ({(OUT / f'{ds}.json').stat().st_size / 1e6:.2f} MB)", flush=True)
    if (ROOT / "data" / "imagenet" / "imagenet-resnet152.npz").exists():
        (OUT / "imagenet_gallery.json").write_text(json.dumps(export_imagenet_gallery(cfg), separators=(",", ":")))
        print(f"wrote imagenet_gallery.json ({(OUT / 'imagenet_gallery.json').stat().st_size / 1e6:.2f} MB)")
    (OUT / "results.json").write_text(json.dumps(summary(cfg, done), separators=(",", ":")))
    print(f"wrote results.json ({(OUT / 'results.json').stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
