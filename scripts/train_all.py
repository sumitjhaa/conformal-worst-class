#!/usr/bin/env python3
"""Stage 1 only: train every (run, seed) and cache logits/features, then add the real-shift sets.

GPU-bound and resumable: finished caches and checkpoints are skipped. Run before run_all.py / research.py.
    python scripts/train_all.py                      # everything in configs/default.yaml
    python scripts/train_all.py --runs cifar100      # one run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import build_cache, extend_cache, seeds_for  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--runs", nargs="+")
    p.add_argument("--extend-only", action="store_true", help="only add CIFAR-10.1 / CIFAR-10-C to existing caches")
    p.add_argument("--seeds", nargs="+", type=int, help="subset of seeds (lets two processes share the GPU without overlap)")
    a = p.parse_args()
    cfg = yaml.safe_load((ROOT / a.config).read_text())
    runs = a.runs or cfg["datasets"]
    if not a.extend_only:
        for ds in runs:
            for seed in [s for s in seeds_for(cfg, ds) if not a.seeds or s in a.seeds]:
                build_cache(ds, seed, cfg)
                print(f"[cached] {ds} s{seed}", flush=True)
    for ds in runs:
        for seed in [s for s in seeds_for(cfg, ds) if not a.seeds or s in a.seeds]:
            extend_cache(ds, seed, cfg)
