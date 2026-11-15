"""Data loading and the leakage-safe four-way split.

Splits (disjoint):
    D_train  -> fit the classifier, normalisation stats, Mahalanobis Gaussians
    D_design -> early stopping, temperature T, RAPS lambda/k_reg, gamma, method selection
    D_cal    -> compute the conformal quantile q_hat  (NEVER used for tuning)
    D_test   -> final reported metrics only

Images are held in memory as uint8 tensors (N, C, H, W); all datasets here fit easily, and on-device
batching/augmentation is much faster on MPS than a PIL DataLoader.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

_ID = {"mnist": torchvision.datasets.MNIST, "cifar10": torchvision.datasets.CIFAR10, "cifar100": torchvision.datasets.CIFAR100}
# A "run" names a (dataset, architecture) pair; runs that reuse a dataset with another model map back here.
_BASE = {"cifar10_r18": "cifar10"}


def base(run: str) -> str:
    """Dataset behind a run name, e.g. cifar10_r18 -> cifar10."""
    return _BASE.get(run, run)


@dataclass
class Splits:
    """Index tensors into the official train set (test = the official test set, untouched)."""
    train: torch.Tensor
    design: torch.Tensor
    calib: torch.Tensor


def _to_tensors(ds) -> tuple[torch.Tensor, torch.Tensor]:
    """torchvision dataset -> (uint8 images (N, C, H, W), int64 labels)."""
    x = torch.as_tensor(ds.data)
    if x.ndim == 3:            # MNIST family: (N, H, W)
        x = x.unsqueeze(1)
    elif x.shape[-1] == 3:     # CIFAR family: (N, H, W, C)
        x = x.permute(0, 3, 1, 2)
    y = torch.as_tensor(ds.targets if hasattr(ds, "targets") else ds.labels, dtype=torch.long)
    return x.contiguous(), y


def load_in_distribution(dataset: str, root: str = "data") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (x_train, y_train, x_test, y_test) via torchvision. dataset in {mnist, cifar10, cifar100} or a run name."""
    cls = _ID[base(dataset)]
    return (*_to_tensors(cls(root, train=True, download=True)), *_to_tensors(cls(root, train=False, download=True)))


def load_ood(dataset: str, parent: str, root: str = "data") -> torch.Tensor:
    """Return OOD test images (labels are meaningless for the parent's label space, so dropped).

    FashionMNIST/KMNIST for mnist; SVHN/CIFAR-100 for cifar10.
    """
    d = torchvision.datasets
    loaders = {
        "fashion_mnist": lambda: d.FashionMNIST(root, train=False, download=True),
        "kmnist": lambda: d.KMNIST(root, train=False, download=True),
        "svhn": lambda: d.SVHN(root, split="test", download=True),
        "cifar100": lambda: d.CIFAR100(root, train=False, download=True),
        "cifar10": lambda: d.CIFAR10(root, train=False, download=True),
    }
    return _to_tensors(loaders[dataset]())[0]  # SVHN is already (N, C, H, W); _to_tensors leaves it alone


def four_way_split(n_train: int, fractions: dict[str, float], seed: int) -> Splits:
    """Disjoint train/design/calib index sets over the official train set, with a fixed seed."""
    perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(seed))
    n_tr = round(fractions["train"] * n_train)
    n_de = round(fractions["design"] * n_train)
    s = Splits(train=perm[:n_tr], design=perm[n_tr:n_tr + n_de], calib=perm[n_tr + n_de:])
    allidx = torch.cat([s.train, s.design, s.calib])
    assert len(allidx) == n_train and len(allidx.unique()) == n_train, "splits must be disjoint and complete"
    return s


def channel_stats(x_train: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean/std of D_train images in [0, 1]. Fit on D_train only (no leakage)."""
    x = x_train.float() / 255
    return x.mean(dim=(0, 2, 3)), x.std(dim=(0, 2, 3))


def normalise(x_uint8: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """uint8 (B, C, H, W) -> standardised float on the same device."""
    x = x_uint8.float() / 255
    return (x - mean.to(x.device)[None, :, None, None]) / std.to(x.device)[None, :, None, None]


def augment_cifar(x: torch.Tensor) -> torch.Tensor:
    """Train-only augmentation on a float batch: random horizontal flip + random 32x32 crop of a 4-px padded image."""
    b, c, h, w = x.shape
    flip = torch.rand(b, 1, 1, 1, device=x.device) < 0.5
    x = torch.where(flip, x.flip(3), x)
    p = F.pad(x, (4, 4, 4, 4))
    di = torch.randint(0, 9, (b,), device=x.device)
    dj = torch.randint(0, 9, (b,), device=x.device)
    rows = (di[:, None] + torch.arange(h, device=x.device))[:, None, :, None]
    cols = (dj[:, None] + torch.arange(w, device=x.device))[:, None, None, :]
    return p[torch.arange(b, device=x.device)[:, None, None, None], torch.arange(c, device=x.device)[None, :, None, None], rows, cols]


def build_transforms(dataset: str):
    """Return the train-time augmentation (identity for MNIST). Eval uses `normalise` only."""
    return augment_cifar if base(dataset).startswith("cifar") else (lambda x: x)


def shift_images(x_uint8: torch.Tensor, dataset: str, severity: float, seed: int = 0) -> torch.Tensor:
    """Labelled covariate shift: same images and labels, corrupted inputs. Coverage stays measurable.

    mnist:   rotate by `severity` degrees.
    cifar10: additive Gaussian noise with std `severity` (pixels in [0, 1]).
    """
    if base(dataset) == "mnist":
        return TF.rotate(x_uint8, severity)
    g = torch.Generator().manual_seed(seed)
    x = x_uint8.float() / 255 + severity * torch.randn(x_uint8.shape, generator=g)
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)


# ------------------------------------------------------------------ natural / benchmark shift for CIFAR-10
CIFAR10C = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
            "snow", "frost", "fog", "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression"]


def load_cifar10_1(root: str = "data") -> tuple[torch.Tensor, torch.Tensor]:
    """CIFAR-10.1 v6 (Recht et al., 2018): 2,000 new test images collected like the original. Natural shift."""
    import numpy as np
    d = f"{root}/cifar10_1"
    x = torch.as_tensor(np.load(f"{d}/cifar10.1_v6_data.npy")).permute(0, 3, 1, 2).contiguous()
    return x, torch.as_tensor(np.load(f"{d}/cifar10.1_v6_labels.npy"), dtype=torch.long)


def load_cifar10_c(corruption: str, severity: int, n: int, root: str = "data", seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """CIFAR-10-C (Hendrycks & Dietterich, 2019): the 10k test images under a corruption at severity 1..5.

    A fixed random subset of `n` images (same indices for every corruption/severity) keeps the cache small.
    """
    import numpy as np
    d = f"{root}/cifar10_c/CIFAR-10-C"
    lo = (severity - 1) * 10000
    idx = torch.randperm(10000, generator=torch.Generator().manual_seed(seed))[:n].sort().values.numpy()
    x = np.load(f"{d}/{corruption}.npy", mmap_mode="r")[lo:lo + 10000][idx]
    y = np.load(f"{d}/labels.npy")[lo:lo + 10000][idx]
    return torch.as_tensor(np.ascontiguousarray(x)).permute(0, 3, 1, 2).contiguous(), torch.as_tensor(y, dtype=torch.long)
