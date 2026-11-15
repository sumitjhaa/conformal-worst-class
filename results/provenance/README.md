# Provenance — what was run, on what, from where

Everything needed to audit this project is in the repository; only re-downloadable or regenerable bulk is left out.

## In the repo (proof of work)
| What | Where |
|---|---|
| Training logs, one line per epoch (train loss, design NLL, design accuracy) for every (run, seed) | `results/train_all.log`, `results/train_s1.log`, `results/train_s2.log`, `results/cifar_cache.log` |
| Analysis logs | `results/run_all.log`, `results/sweep.log`, `results/export.log`, `results/analysis_*.log` |
| Every metric for every (run, seed, α, method) | `results/main.csv` (+ `class_cov`, `deciles`, `sizes`, `ood`, `shift`, `resplit`, `tuned`) |
| Research analyses | `results/research/*.csv` (per-run parts in `results/research/parts/`) |
| Figures and tables | `results/figures/*.pdf`, `results/table_*.tex`, `results/numbers.tex` |
| Hardware and exact package versions | `results/provenance/environment.txt` |
| SHA-256 of every downloaded data file | `results/provenance/data_sha256.txt` |
| SHA-256 of every trained checkpoint | `results/provenance/checkpoints_sha256.txt` |
| Screenshots of the site and papers | `docs/screenshots/` |

## Not in the repo (too large for GitHub; regenerable)
| What | Size | How to get it back |
|---|---|---|
| `data/` | ~4 GB | downloaded by the scripts / links below; verify with `data_sha256.txt` |
| `results/cache/` | ~4.3 GB | `python scripts/train_all.py` (reuses checkpoints if present) |
| `checkpoints/` | ~0.36 GB (29 files, each < 100 MB) | `python scripts/train_all.py`; or publish as a GitHub Release / Git LFS and verify with `checkpoints_sha256.txt` |

## Data sources
- MNIST, FashionMNIST, KMNIST, CIFAR-10, CIFAR-100: `torchvision.datasets` (automatic download).
- CIFAR-10.1 v6: `cifar10.1_v6_data.npy`, `cifar10.1_v6_labels.npy` from github.com/modestyachts/CIFAR-10.1 (`datasets/`).
- CIFAR-10-C: `CIFAR-10-C.tar` from Zenodo record 2535967 (Hendrycks & Dietterich, 2019).
- ImageNet ResNet-152 validation softmax: `imagenet-resnet152.npz` (+ example JPEGs, class names) from the data archive used by the
  notebooks of github.com/aangelopoulos/conformal-prediction.
- ImageNet-pretrained ResNet-18 weights: `torchvision.models.ResNet18_Weights.IMAGENET1K_V1` (automatic download).
- SVHN was configured but its host refused connections during the runs; it is skipped automatically.

## Determinism
Seeds and split fractions are in `configs/default.yaml`. Splits are seeded `torch.randperm`; APS randomisation uses fixed
per-split seeds; ImageNet splits are seeded re-draws of the 50k outputs. MPS kernels are not bit-deterministic, so retraining
reproduces metrics to sampling precision, not bit-for-bit; the cached logits make every analysis bit-reproducible.
