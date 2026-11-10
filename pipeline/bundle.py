"""Trail bundling of the t-SNE projections (paper Sec. 5.4, "trail bundling").

A trail joins the 2D positions of one observation over the steps of a view:
    T1: over the snapshot epochs (one set of 1000 trails per layer)
    T2: over the layers Input -> hidden -> Output (one set of 1000 trails per epoch)
Drawing them straight gives heavy clutter, so they are bundled with a simplified kernel density estimation
edge bundling (KDEEB, Hurter et al. 2012), the CPU counterpart of the paper's GPU bundling (CUBu):
    1. every segment between two consecutive steps is subdivided into equally spaced samples;
    2. repeat: splat all samples on a grid (weighted by segment length, so density follows trail length, not
       sample count) and blur it with a Gaussian kernel; move every sample a small step up the density
       gradient; smooth every trail; shrink the kernel.
The first and last sample of each trail (first and last step) never move. Samples keep their position along the
trail, so sample i always belongs to step i / SAMPLES_PER_SEGMENT (the UI uses this for the progress lightness).

Bundling is class-aware: each trail is attracted only by the density of its own (true) class. Global bundling
merged trails of different classes into shared "highways", which, with hue encoding class, falsely suggested that
those classes follow the same path through the network.

Two strengths are stored, "medium" and "strong" (more iterations). Unbundled trails are drawn by the UI
directly from the projected points, and PCA trails are never bundled (see PLAN.md, key decisions).

Usage:
    python -m pipeline.bundle                       # all configurations
    python -m pipeline.bundle --dataset svhn --model mlp --force

Output: raw/<dataset>/<model>/trails_tsne.npz with
    t1   uint16 [len(t1_layers), len(strengths), N, (n_epochs - 1) * SAMPLES_PER_SEGMENT + 1, 2]
    t2   uint16 [len(epochs),    len(strengths), N, (n_layers - 1) * SAMPLES_PER_SEGMENT + 1, 2]
    strengths, epochs, t1_layers, t2_layers, samples_per_segment, scale
Coordinates are quantized to 0..scale (9999) in the unit square of the projection files.
"""
import argparse
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

from . import config

GRID = 256                   # density grid resolution over the unit square
SUBDIVISIONS = 8             # samples per segment while bundling
SAMPLES_PER_SEGMENT = 4      # samples per segment kept in the output (every 2nd)
STRENGTHS = {"medium": 6, "strong": 24}  # strength name -> number of iterations
KERNEL_START = 0.06          # Gaussian kernel width (fraction of the unit square) in the first iteration
KERNEL_DECAY = 0.85          # kernel shrinks by this factor every iteration ...
KERNEL_MIN = 0.012           # ... down to this width
STEP = 0.4                   # move per iteration, as a fraction of the current kernel width
SMOOTHING = 0.5              # Laplacian smoothing weight (one pass per iteration)
SCALE = 9999                 # quantization of the output coordinates


def subdivide(nodes: np.ndarray) -> np.ndarray:
    """[N, K, 2] step positions -> [N, (K - 1) * SUBDIVISIONS + 1, 2] equally spaced samples per segment."""
    t = np.linspace(0, 1, SUBDIVISIONS, endpoint=False)[None, None, :, None]
    start, end = nodes[:, :-1, None], nodes[:, 1:, None]
    samples = (start + (end - start) * t).reshape(len(nodes), -1, 2)
    return np.concatenate([samples, nodes[:, -1:]], axis=1)


def sample_weights(nodes: np.ndarray) -> np.ndarray:
    """Weight of every sample = length of its segment / samples per segment (the last sample inherits)."""
    seg = np.linalg.norm(np.diff(nodes, axis=1), axis=2) / SUBDIVISIONS  # [N, K - 1]
    weights = np.repeat(seg, SUBDIVISIONS, axis=1)
    return np.concatenate([weights, seg[:, -1:]], axis=1)


def density_gradient(points: np.ndarray, weights: np.ndarray, kernel: float) -> np.ndarray:
    """Gradient of the blurred sample density, evaluated at every sample. Returns [..., 2] (x, y)."""
    cells = np.clip(np.rint(points * (GRID - 1)).astype(int), 0, GRID - 1)
    grid = np.zeros((GRID, GRID))
    np.add.at(grid, (cells[..., 1].ravel(), cells[..., 0].ravel()), weights.ravel())
    density = gaussian_filter(grid, sigma=kernel * (GRID - 1))
    grad_y, grad_x = np.gradient(density)
    coords = [points[..., 1].ravel() * (GRID - 1), points[..., 0].ravel() * (GRID - 1)]
    grad = np.stack([map_coordinates(grad_x, coords, order=1), map_coordinates(grad_y, coords, order=1)], axis=-1)
    return grad.reshape(points.shape)


def bundle(nodes: np.ndarray, groups: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Bundle the trails through `nodes` [N, K, 2]; returns {strength: [N, (K-1)*SAMPLES_PER_SEGMENT+1, 2]}.

    With `groups` (e.g. class labels [N]), each trail is attracted only by the density of its own group.
    """
    points = subdivide(nodes.astype(np.float64))
    weights = sample_weights(nodes)
    masks = [groups == g for g in np.unique(groups)] if groups is not None else [slice(None)]
    kernel, keep = KERNEL_START, SUBDIVISIONS // SAMPLES_PER_SEGMENT
    result = {}
    for iteration in range(1, max(STRENGTHS.values()) + 1):
        grad = np.empty_like(points)
        for mask in masks:
            grad[mask] = density_gradient(points[mask], weights[mask], kernel)
        norm = np.linalg.norm(grad, axis=-1, keepdims=True)
        step = np.divide(grad, norm, out=np.zeros_like(grad), where=norm > 1e-12) * STEP * kernel
        points[:, 1:-1] += step[:, 1:-1]
        points[:, 1:-1] = ((1 - SMOOTHING) * points[:, 1:-1]
                           + SMOOTHING * (points[:, :-2] + points[:, 2:]) / 2)
        np.clip(points, 0, 1, out=points)
        kernel = max(kernel * KERNEL_DECAY, KERNEL_MIN)
        for name, iterations in STRENGTHS.items():
            if iteration == iterations:
                result[name] = points[:, ::keep].copy()
    return result


def quantize(points: np.ndarray) -> np.ndarray:
    return np.rint(points * SCALE).astype(np.uint16)


def bundle_config(cfg_dir: Path) -> Path:
    proj = np.load(cfg_dir / "proj_tsne.npz")
    labels, strengths = proj["labels"], list(STRENGTHS)
    # T1: proj["t1"] is [layer, epoch, obs, 2] -> trails over epochs, per layer.
    t1 = np.stack([np.stack([quantize(b[s]) for s in strengths])
                   for b in (bundle(layer.transpose(1, 0, 2), labels) for layer in proj["t1"])])
    # T2: proj["t2"] is [epoch, layer, obs, 2] -> trails over layers, per epoch.
    t2 = np.stack([np.stack([quantize(b[s]) for s in strengths])
                   for b in (bundle(epoch.transpose(1, 0, 2), labels) for epoch in proj["t2"])])
    out = cfg_dir / "trails_tsne.npz"
    np.savez_compressed(out, t1=t1, t2=t2, strengths=np.array(strengths), epochs=proj["epochs"],
                        t1_layers=proj["t1_layers"], t2_layers=proj["t2_layers"],
                        samples_per_segment=SAMPLES_PER_SEGMENT, scale=SCALE)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", nargs="+", choices=list(config.DATASETS), default=list(config.DATASETS))
    p.add_argument("--model", nargs="+", choices=list(config.TRAINING), default=list(config.TRAINING))
    p.add_argument("--raw", type=Path, default=config.DEFAULT_RAW_DIR, help="training outputs folder")
    p.add_argument("--force", action="store_true", help="recompute existing trails")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.dataset:
        for model in args.model:
            cfg_dir = args.raw / dataset / model
            if (cfg_dir / "trails_tsne.npz").exists() and not args.force:
                print(f"[{dataset}/{model}] trails exist, skipping (use --force to recompute).")
                continue
            if not (cfg_dir / "proj_tsne.npz").exists():
                print(f"[{dataset}/{model}] no proj_tsne.npz, run `python -m pipeline.project` first.")
                continue
            start = time.perf_counter()
            out = bundle_config(cfg_dir)
            print(f"[{dataset}/{model}] {out.name} ({out.stat().st_size / 1e6:.1f} MB, "
                  f"{time.perf_counter() - start:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
