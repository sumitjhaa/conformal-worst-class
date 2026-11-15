#!/usr/bin/env python3
"""Export the seed-0 MNIST LeNet as JSON so the explorer's "Draw a digit" page can run it in the browser.

Weights go out as base64 float32 with their shapes, plus the D_train normalisation stats, and two test images with
the logits torch gives them: the browser forward pass is checked against those (web/scripts/check-explorer.mjs).
    .venv/bin/python scripts/export_mnist_net.py
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import data  # noqa: E402
from src.pipeline import load_net  # noqa: E402


def b64_f32(t: torch.Tensor) -> str:
    return base64.b64encode(t.detach().cpu().float().numpy().astype("<f4").tobytes()).decode()


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())
    net, mean, std = load_net("mnist", 0, cfg)
    net = net.cpu().eval()
    layers = {k: {"shape": list(v.shape), "data": b64_f32(v)} for k, v in net.state_dict().items()}

    _, _, x_te, _ = data.load_in_distribution("mnist", str(ROOT / "data"))
    xs = x_te[:2]  # uint8 (2, 1, 28, 28)
    with torch.no_grad():
        logits = net(data.normalise(xs, mean, std))
    check = [{"img": base64.b64encode(x.numpy().astype(np.uint8).tobytes()).decode(), "logits": z.tolist()} for x, z in zip(xs, logits)]

    out = {"arch": "lenet", "mean": float(mean[0]), "std": float(std[0]), "layers": layers, "check": check}
    path = ROOT / "site" / "data" / "mnist_net.json"
    path.write_text(json.dumps(out))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
