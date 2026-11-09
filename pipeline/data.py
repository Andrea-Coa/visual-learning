"""Dataset loading.

Whole datasets are kept as uint8 tensors (N, C, H, W) on the training device and normalized
per batch. They are small enough (< 250 MB each) to fit in a T4's memory, and this avoids the
per-sample overhead of a DataLoader, which dominates with the paper's small batch sizes.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets

from .config import DatasetSpec


def load_split(name: str, root: Path, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (images uint8 [N, C, H, W], labels int64 [N]) for one split."""
    root = Path(root)
    if name == "mnist":
        ds = datasets.MNIST(root, train=train, download=True)
        x, y = ds.data.unsqueeze(1), ds.targets
    elif name == "cifar10":
        ds = datasets.CIFAR10(root, train=train, download=True)
        x, y = torch.from_numpy(ds.data).permute(0, 3, 1, 2), torch.tensor(ds.targets)
    elif name == "svhn":
        # torchvision already maps label 10 (digit zero) to 0.
        ds = datasets.SVHN(root, split="train" if train else "test", download=True)
        x, y = torch.from_numpy(ds.data), torch.from_numpy(ds.labels)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return x.contiguous(), y.long()


def channel_stats(x: torch.Tensor, chunk: int = 10_000) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean and std of uint8 images scaled to [0, 1], computed in chunks."""
    total = torch.zeros(x.shape[1], dtype=torch.float64)
    total_sq = torch.zeros_like(total)
    for start in range(0, len(x), chunk):
        batch = x[start:start + chunk].double() / 255.0
        total += batch.sum(dim=(0, 2, 3))
        total_sq += (batch ** 2).sum(dim=(0, 2, 3))
    count = len(x) * x.shape[2] * x.shape[3]
    mean = total / count
    std = (total_sq / count - mean ** 2).sqrt()
    return mean.float(), std.float()


@dataclass
class ImageData:
    spec: DatasetSpec
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    mean: torch.Tensor  # [C]
    std: torch.Tensor   # [C]

    def to(self, device: torch.device) -> "ImageData":
        for field in ("x_train", "y_train", "x_test", "y_test", "mean", "std"):
            setattr(self, field, getattr(self, field).to(device))
        return self

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """uint8 [N, C, H, W] -> standardized float32."""
        return (x.float() / 255.0 - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)


def load_dataset(spec: DatasetSpec, root: Path) -> ImageData:
    x_train, y_train = load_split(spec.name, root, train=True)
    x_test, y_test = load_split(spec.name, root, train=False)
    mean, std = channel_stats(x_train)
    return ImageData(spec, x_train, y_train, x_test, y_test, mean, std)


def stratified_subset(labels: np.ndarray, size: int, n_classes: int, seed: int) -> np.ndarray:
    """Sorted indices of `size // n_classes` random observations per class.

    Deterministic for a given seed, so every machine/Colab account picks the same subset.
    """
    rng = np.random.RandomState(seed)
    per_class = size // n_classes
    picked = [rng.choice(np.flatnonzero(labels == c), per_class, replace=False) for c in range(n_classes)]
    return np.sort(np.concatenate(picked))


def save_subset(path: Path, data: ImageData, indices: np.ndarray) -> None:
    """Store the subset (indices, labels, raw images as [N, H, W, C] uint8) for later export."""
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = torch.as_tensor(indices)
    images = data.x_test.cpu()[idx].permute(0, 2, 3, 1).numpy()
    labels = data.y_test.cpu()[idx].numpy()
    np.savez_compressed(path, indices=indices, labels=labels, images=images,
                        classes=np.array(data.spec.classes))


def batch_indices(n: int, batch_size: int, generator: torch.Generator, device: torch.device):
    """Yield shuffled index tensors covering range(n) once (one epoch)."""
    perm = torch.randperm(n, generator=generator).to(device)
    for start in range(0, n, batch_size):
        yield perm[start:start + batch_size]
