#!/usr/bin/env python3
"""Concatenate per-run result parts (run_all.py / research.py --part) into results/ and rebuild figures.

Lets finished runs be analysed while others still train:
    python scripts/run_all.py --datasets mnist --part mnist
    python scripts/research.py --runs mnist --part mnist
    python scripts/merge_parts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import make_figures  # noqa: E402


def merge(parts_dir: Path, out: Path) -> list[str]:
    names = sorted({f.name for d in parts_dir.glob("*") if d.is_dir() for f in d.glob("*.csv")})
    for name in names:
        dfs = [pd.read_csv(f) for f in sorted(parts_dir.glob(f"*/{name}"))]
        pd.concat(dfs, ignore_index=True).to_csv(out / name, index=False)
    return names


if __name__ == "__main__":
    res = ROOT / "results"
    done = merge(res / "parts", res) if (res / "parts").exists() else []
    if "tuned.csv" in done:
        t = pd.read_csv(res / "tuned.csv")
        (res / "tuned.json").write_text(json.dumps(json.loads(t.to_json(orient="records")), indent=2))
    if "main.csv" in done:
        m = pd.read_csv(res / "main.csv")
        m.groupby(["dataset", "alpha", "method"]).agg(["mean", "std"]).drop(columns="seed").to_csv(res / "main_summary.csv")
    rdone = merge(res / "research" / "parts", res / "research") if (res / "research" / "parts").exists() else []
    print("merged", done, rdone)
    make_figures.main()
