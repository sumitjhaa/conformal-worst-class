"""Classifier architectures and training.

Use MPS on Apple Silicon when available.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data import normalise


class LeNet(nn.Module):
    """Small CNN for MNIST (28x28 grayscale). Expose penultimate features for Mahalanobis."""

    def __init__(self, num_classes: int = 10, feature_dim: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),   # 32x14x14
            nn.Conv2d(32, 64, 5), nn.ReLU(), nn.MaxPool2d(2),              # 64x5x5
            nn.Flatten(), nn.Linear(64 * 5 * 5, feature_dim), nn.ReLU(),
        )
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits z(x)."""
        return self.head(self.features(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return penultimate features phi(x) used by the Mahalanobis score."""
        return self.body(x)


class _Block(nn.Module):
    """ResNet basic block: two 3x3 convs + identity (or 1x1 projection) shortcut."""

    def __init__(self, cin: int, cout: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.short = nn.Sequential() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)) + self.short(x))


class SmallResNet(nn.Module):
    """Compact ResNet for CIFAR-10 (32x32 RGB): 3 stages x 2 blocks, ~2.8M params, ~90% test accuracy."""

    def __init__(self, num_classes: int = 10, feature_dim: int = 256) -> None:
        super().__init__()
        widths = [64, 128, feature_dim]
        layers = [nn.Conv2d(3, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU()]
        cin = 64
        for i, w in enumerate(widths):
            layers += [_Block(cin, w, 1 if i == 0 else 2), _Block(w, w, 1)]
            cin = w
        self.body = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class PretrainedResNet18(nn.Module):
    """Second architecture: ImageNet-pretrained torchvision ResNet-18, fine-tuned on 32x32 inputs upsampled to 64x64.

    Different family, pre-training and calibration from SmallResNet, to test that conclusions are not model-specific.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        import torchvision
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.body = nn.Sequential(*list(net.children())[:-1], nn.Flatten())  # up to global average pool
        self.head = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False))


def build_model(run: str) -> nn.Module:
    return {"mnist": lambda: LeNet(), "cifar10": lambda: SmallResNet(10), "cifar100": lambda: SmallResNet(100),
            "cifar10_r18": lambda: PretrainedResNet18(10)}[run]()


def get_device() -> torch.device:
    """Prefer MPS, then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(model, x, y, train_idx, design_idx, mean, std, augment, cfg: dict, ckpt_path: Path) -> dict:
    """Train with cross-entropy on D_train; keep the epoch with the lowest D_design NLL (early stopping).

    cfg keys: epochs, batch_size, lr, optimizer ("adam" | "sgd").
    """
    dev = get_device()
    model.to(dev)
    steps = (len(train_idx) + cfg["batch_size"] - 1) // cfg["batch_size"]
    if cfg["optimizer"] == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=5e-4, nesterov=True)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=cfg["epochs"] * steps)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        sched = None

    hist, best, best_nll = [], None, float("inf")
    for epoch in range(cfg["epochs"]):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        total = 0.0
        for b in perm.split(cfg["batch_size"]):
            xb = augment(normalise(x[b].to(dev), mean, std))
            loss = F.cross_entropy(model(xb), y[b].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched:
                sched.step()
            total += loss.item() * len(b)
        logits, _ = predict_logits(model, x[design_idx], mean, std)
        nll = F.cross_entropy(logits, y[design_idx]).item()
        acc = (logits.argmax(1) == y[design_idx]).float().mean().item()
        hist.append({"epoch": epoch, "train_loss": total / len(train_idx), "design_nll": nll, "design_acc": acc})
        print(f"  epoch {epoch:2d}  train_loss {total / len(train_idx):.4f}  design_nll {nll:.4f}  design_acc {acc:.4f}", flush=True)
        if nll < best_nll:
            best_nll, best = nll, copy.deepcopy(model.state_dict())

    model.load_state_dict(best)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best, ckpt_path)
    return {"history": hist, "best_design_nll": best_nll}


@torch.no_grad()
def predict_logits(model, x_uint8: torch.Tensor, mean, std, batch_size: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (logits, features) on CPU for all images, in model.eval() mode."""
    model.eval()
    dev = next(model.parameters()).device
    logits, feats = [], []
    for xb in x_uint8.split(batch_size):
        f = model.features(normalise(xb.to(dev), mean, std))
        feats.append(f.cpu())
        logits.append(model.head(f).cpu())
    return torch.cat(logits), torch.cat(feats)
