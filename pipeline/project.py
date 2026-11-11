"""2D projections of the recorded activations (t-SNE, PCA, UMAP and variants) and their neighborhood hit.

T1 (inter-epoch): for every layer (hidden layers + Output), the frames of all snapshot epochs.
    t-SNE: one joint projection of all epochs stacked, so movement between epochs reflects real changes
    (paper Sec. 5.4); normalized to the unit square per layer, over all epochs.
    PCA (of log1p(activations), see pca2): one PCA per epoch, each aligned to the previous epoch with
    orthogonal Procrustes and normalized per frame. (A joint PCA collapses the early epochs to a dot:
    activations grow by orders of magnitude during training.)
T2 (inter-layer): for every snapshot epoch, one projection per layer (Input -> hidden layers -> Output).
    t-SNE: every layer is initialized with the t-SNE of the Input layer (the network input, identical at every
    epoch), which keeps clusters roughly in place across layers and across epochs (paper Sec. 5.4).
    PCA: each layer aligned to the previous one (starting at the Input PCA) with orthogonal Procrustes.
    Normalized to the unit square per frame.

Alternatives compared in PLAN S.4 (the T2 alignment above is only "soft": a layer that reorganizes neighborhoods
can rotate or flip its map):
    tsne_proc  t-SNE + Procrustes. T1 = t-SNE. T2 = the t-SNE frames, each layer rotated/reflected onto the previous
               one (orthogonal Procrustes). Rigid, so the neighborhood hit is unchanged. Derived from proj_tsne.npz.
    umap       UMAP. T1: one joint UMAP per layer over all epochs (more neighbors, see UMAP_JOINT_NEIGHBORS).
               T2: every layer initialized with the UMAP of the Input layer (the same scheme as t-SNE).
    aumap      Aligned UMAP. T1: the joint UMAP of `umap` (reused if it exists). T2: for every epoch, one AlignedUMAP
               over the sequence Input -> ... -> Output (same observations in every frame), which penalizes moving a
               point away from its position in the neighboring layers. The whole sequence is then rotated onto the
               Input UMAP so that epochs share an orientation.
    grid       Aligned UMAP over the layer x epoch grid: ONE set of coordinates for both views (T1 frames = T2
               frames). Every AlignedUMAP frame is one layer with all its epochs stacked (as the joint T1 map, so
               epochs are exactly aligned) and consecutive layers are aligned with the movement penalty (as T2).
               The Input UMAP is rotated onto the first hidden layer. One normalization for the whole grid.

Normalization keeps the aspect ratio (the shorter side is centered).

Usage:
    python -m pipeline.project                          # all configurations, t-SNE and PCA
    python -m pipeline.project --dataset mnist --model mlp --method pca
    python -m pipeline.project --method umap aumap tsne_proc   # the alternatives of PLAN S.4

Output: raw/<dataset>/<model>/proj_<method>.npz with
    t1         float32 [len(t1_layers), len(epochs), N, 2]
    t2         float32 [len(epochs), len(t2_layers), N, 2]
    nh_t1      float32 [len(t1_layers), len(epochs)]   neighborhood hit (k = 6) of each T1 frame
    nh_t2      float32 [len(epochs), len(t2_layers)]
    epochs, t1_layers, t2_layers, labels
    seconds    wall time of the projection (methods added in S.4 only)
"""
import argparse
import json
import time
from pathlib import Path

import warnings

import numpy as np
from openTSNE import TSNE
from openTSNE.initialization import rescale
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from . import config

METHODS = ("tsne", "pca", "umap", "aumap", "tsne_proc", "grid")
DEFAULT_METHODS = ("tsne", "pca")
PCA_DIMS = 50     # activations are reduced to 50 dims before t-SNE (standard practice, faster)
PERPLEXITY = 30   # openTSNE default, as the paper used default parameters
NH_K = 6          # neighborhood size of the neighborhood hit, as in the paper
UMAP_NEIGHBORS = 15       # umap-learn defaults
UMAP_JOINT_NEIGHBORS = 50 # joint T1 UMAP: every observation appears once per epoch, and a layer that stops changing
                          # (e.g. MNIST CNN conv1 after epoch 1) makes those copies near-duplicates; with 15 neighbors
                          # the graph then links mostly copies of the same image (NH 0.41 vs t-SNE 0.90). 50 -> 0.91.
UMAP_MIN_DIST = 0.1      # compatibility_layout=True pins umap-learn's classic (0.5.x) layout, so results stay
                         # reproducible when a future release changes the default optimizer
ALIGNMENT_REG = 0.1       # AlignedUMAP alignment_regularisation, chosen with `pipeline.compare --sweep` (PLAN S.4):
                          # the smallest value without any rotated/mirrored layer pair; 1.0 already costs NH/trust
# A fixed random_state makes UMAP single-threaded (reproducible); umap-learn warns about it on every fit.
warnings.filterwarnings("ignore", message="n_jobs value .* overridden to 1 by setting random_state")


# --------------------------------------------------------------------------- data access

def load_input(raw: Path, dataset: str, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Labels and network-input vectors (normalized pixels, as the network sees them) of the subset."""
    with np.load(raw / dataset / "subset.npz") as subset:
        labels = subset["labels"]
        images = subset["images"].astype(np.float32) / 255.0  # [N, H, W, C]
    norm = cfg["normalization"]
    images = (images - np.array(norm["mean"], np.float32)) / np.array(norm["std"], np.float32)
    return labels, images.reshape(len(images), -1)


def snapshot_path(cfg_dir: Path, epoch: int) -> Path:
    return cfg_dir / "snapshots" / f"epoch_{epoch:03d}.npz"


def load_snapshot(cfg_dir: Path, epoch: int) -> dict[str, np.ndarray]:
    """All hidden layers of one snapshot plus 'output' (softmax probabilities), as float32."""
    with np.load(snapshot_path(cfg_dir, epoch)) as snap:
        acts = {str(name): snap[str(name)].astype(np.float32) for name in snap["layers"]}
        acts["output"] = snap["probs"].astype(np.float32)
    return acts


def load_layer(cfg_dir: Path, epoch: int, layer: str) -> np.ndarray:
    """One layer of one snapshot ('output' = softmax probabilities), as float32."""
    with np.load(snapshot_path(cfg_dir, epoch)) as snap:
        return snap["probs" if layer == "output" else layer].astype(np.float32)


# --------------------------------------------------------------------------- projection helpers

def reduce_dims(x: np.ndarray) -> np.ndarray:
    if x.shape[1] <= PCA_DIMS:
        return x
    return PCA(PCA_DIMS, random_state=config.SEED).fit_transform(x)


def tsne(x: np.ndarray, init: np.ndarray | None = None) -> np.ndarray:
    """t-SNE of x; `init` (a previous embedding of the same points) fixes the starting layout."""
    initialization = "pca" if init is None else rescale(init)
    embedding = TSNE(perplexity=PERPLEXITY, initialization=initialization,
                     n_jobs=-1, random_state=config.SEED).fit(reduce_dims(x))
    return np.asarray(embedding, dtype=np.float32)


def umap2(x: np.ndarray, init: np.ndarray | None = None, n_neighbors: int = UMAP_NEIGHBORS) -> np.ndarray:
    """UMAP of x; `init` (a previous embedding of the same points) fixes the starting layout."""
    import umap  # slow import (numba): only when UMAP is used
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=UMAP_MIN_DIST, random_state=config.SEED,
                        init="spectral" if init is None else init, compatibility_layout=True)
    return reducer.fit_transform(reduce_dims(x)).astype(np.float32)


def aligned_umap(frames: list[np.ndarray], ref: np.ndarray, regularisation: float = ALIGNMENT_REG) -> list[np.ndarray]:
    """AlignedUMAP of a sequence of frames of the same observations (frames may differ in dimension).

    The sequence is rotated as a whole so that its first frame matches `ref` (a 2D embedding of frame 0).
    """
    import umap
    n = len(frames[0])
    identity = {i: i for i in range(n)}
    mapper = umap.AlignedUMAP(n_neighbors=UMAP_NEIGHBORS, min_dist=UMAP_MIN_DIST, random_state=config.SEED,
                              alignment_regularisation=regularisation, compatibility_layout=True)
    mapper.fit([reduce_dims(x) for x in frames], relations=[identity] * (len(frames) - 1))
    embeddings = [np.asarray(e, np.float32) for e in mapper.embeddings_]
    center = embeddings[0].mean(0)
    rotation, _ = orthogonal_procrustes(embeddings[0] - center, ref - ref.mean(0))
    return [((e - center) @ rotation).astype(np.float32) for e in embeddings]


def pca2(x: np.ndarray, log: bool = False) -> np.ndarray:
    """PCA to 2D. `log` applies log1p first: ReLU activations are heavy-tailed, and without it a few long
    arms dominate the variance and crowd most points into the center (tested: higher NH, less crowding)."""
    return PCA(2, random_state=config.SEED).fit_transform(np.log1p(x) if log else x).astype(np.float32)


def procrustes_align(y: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Rotate/reflect y (centered) to best match ref; scale is normalized later anyway."""
    y = y - y.mean(0)
    rotation, _ = orthogonal_procrustes(y, ref - ref.mean(0))
    return y @ rotation


def normalize(y: np.ndarray) -> np.ndarray:
    """Map [..., 2] coordinates into the unit square, keeping the aspect ratio and centering the shorter side."""
    flat = y.reshape(-1, 2)
    lo, hi = flat.min(0), flat.max(0)
    span = (hi - lo).max()
    return ((y - lo) / span + (1 - (hi - lo) / span) / 2).astype(np.float32)


def neighborhood_hit(y: np.ndarray, labels: np.ndarray, k: int = NH_K) -> float:
    """Mean fraction of each point's k nearest 2D neighbors that share its class (Paulovich et al.)."""
    idx = NearestNeighbors(n_neighbors=k + 1).fit(y).kneighbors(y, return_distance=False)[:, 1:]
    return float((labels[idx] == labels[:, None]).mean())


# --------------------------------------------------------------------------- per configuration

def project_config(raw: Path, dataset: str, model: str, method: str,
                   labels: np.ndarray, input_ref: np.ndarray) -> Path:
    t0 = time.perf_counter()
    cfg_dir = raw / dataset / model
    cfg = json.loads((cfg_dir / "config.json").read_text())
    epochs = cfg["snapshot_epochs"]
    hidden = list(cfg["layers"])
    t1_layers, t2_layers = hidden + ["output"], ["input"] + hidden + ["output"]
    n = len(labels)
    tag = f"[{dataset}/{model}/{method}]"

    # T2 first: one pass over the snapshot files.
    t2 = np.empty((len(epochs), len(t2_layers), n, 2), np.float32)
    x_input = load_input(raw, dataset, cfg)[1] if method == "aumap" else None
    for ei, epoch in enumerate(epochs):
        start = time.perf_counter()
        acts = load_snapshot(cfg_dir, epoch)
        if method == "aumap":
            frames = aligned_umap([x_input] + [acts[layer] for layer in t2_layers[1:]], input_ref)
            t2[ei] = [normalize(y) for y in frames]
            print(f"{tag} T2 epoch {epoch:2d}: {time.perf_counter() - start:.0f}s", flush=True)
            continue
        previous = input_ref
        for li, layer in enumerate(t2_layers):
            if layer == "input":
                y = input_ref
            elif method == "tsne":
                y = tsne(acts[layer], init=input_ref)
            elif method == "umap":
                y = umap2(acts[layer], init=input_ref)
            else:
                y = previous = procrustes_align(pca2(acts[layer], log=True), previous)
            t2[ei, li] = normalize(y)
        print(f"{tag} T2 epoch {epoch:2d}: {time.perf_counter() - start:.0f}s", flush=True)

    # T1: per layer, the frames of all epochs.
    t1 = np.empty((len(t1_layers), len(epochs), n, 2), np.float32)
    joint_umap = cfg_dir / "proj_umap.npz"
    reuse_umap = method == "aumap" and joint_umap.exists()  # T1 of Aligned UMAP = the joint UMAP
    if reuse_umap:
        with np.load(joint_umap) as previous_run:
            t1[:] = previous_run["t1"]
        print(f"{tag} T1 reused from {joint_umap.name}", flush=True)
    for li, layer in enumerate([] if reuse_umap else t1_layers):
        start = time.perf_counter()
        frames = [load_layer(cfg_dir, e, layer) for e in epochs]
        if method == "tsne":
            t1[li] = normalize(tsne(np.concatenate(frames)).reshape(len(epochs), n, 2))
        elif method in ("umap", "aumap"):
            joint = umap2(np.concatenate(frames), n_neighbors=UMAP_JOINT_NEIGHBORS)
            t1[li] = normalize(joint.reshape(len(epochs), n, 2))
        else:
            previous = None
            for ei, x in enumerate(frames):
                y = pca2(x, log=True)
                previous = y if previous is None else procrustes_align(y, previous)
                t1[li, ei] = normalize(previous)
        print(f"{tag} T1 {layer}: {time.perf_counter() - start:.0f}s", flush=True)

    nh_t1 = np.array([[neighborhood_hit(frame, labels) for frame in layer] for layer in t1], np.float32)
    nh_t2 = np.array([[neighborhood_hit(frame, labels) for frame in epoch] for epoch in t2], np.float32)

    out = cfg_dir / f"proj_{method}.npz"
    extra = {} if method in DEFAULT_METHODS else {"seconds": np.float32(time.perf_counter() - t0)}
    np.savez_compressed(out, t1=t1, t2=t2, nh_t1=nh_t1, nh_t2=nh_t2, epochs=np.array(epochs),
                        t1_layers=np.array(t1_layers), t2_layers=np.array(t2_layers), labels=labels, **extra)
    print(f"{tag} NH input {nh_t2[0, 0]:.3f} | last hidden @epoch0 {nh_t2[0, -2]:.3f} -> "
          f"@epoch{epochs[-1]} {nh_t2[-1, -2]:.3f} | output @epoch{epochs[-1]} {nh_t2[-1, -1]:.3f}", flush=True)
    return out


def project_grid(raw: Path, dataset: str, model: str, labels: np.ndarray, input_ref: np.ndarray) -> Path:
    """`grid`: one Aligned UMAP over the layers, each frame = the layer's activations of all epochs stacked."""
    t0 = time.perf_counter()
    cfg_dir = raw / dataset / model
    cfg = json.loads((cfg_dir / "config.json").read_text())
    epochs = cfg["snapshot_epochs"]
    hidden = list(cfg["layers"])
    t1_layers, t2_layers = hidden + ["output"], ["input"] + hidden + ["output"]
    n, tag = len(labels), f"[{dataset}/{model}/grid]"
    import umap
    stacked = [reduce_dims(np.concatenate([load_layer(cfg_dir, e, layer) for e in epochs])) for layer in t1_layers]
    identity = {i: i for i in range(len(epochs) * n)}  # observation i at epoch e is row e * n + i in every frame
    mapper = umap.AlignedUMAP(n_neighbors=UMAP_JOINT_NEIGHBORS, min_dist=UMAP_MIN_DIST, random_state=config.SEED,
                              alignment_regularisation=ALIGNMENT_REG, compatibility_layout=True)
    mapper.fit(stacked, relations=[identity] * (len(stacked) - 1))
    layers = np.stack([np.asarray(e, np.float32).reshape(len(epochs), n, 2) for e in mapper.embeddings_])
    # Input (identical at every epoch): rigid + scale match onto the first hidden layer, averaged over epochs.
    target = layers[0].mean(0)
    src, dst = input_ref - input_ref.mean(0), target - target.mean(0)
    rotation, _ = orthogonal_procrustes(src, dst)
    x_in = src @ rotation * (np.linalg.norm(dst) / np.linalg.norm(src)) + target.mean(0)
    grid = normalize(np.concatenate([np.broadcast_to(x_in, (1, len(epochs), n, 2)), layers]))  # [T2 layer, epoch, ...]
    t1, t2 = grid[1:], grid.transpose(1, 0, 2, 3)
    nh_t1 = np.array([[neighborhood_hit(frame, labels) for frame in layer] for layer in t1], np.float32)
    nh_t2 = np.array([[neighborhood_hit(frame, labels) for frame in epoch] for epoch in t2], np.float32)
    out = cfg_dir / "proj_grid.npz"
    np.savez_compressed(out, t1=t1, t2=np.ascontiguousarray(t2), nh_t1=nh_t1, nh_t2=nh_t2, epochs=np.array(epochs),
                        t1_layers=np.array(t1_layers), t2_layers=np.array(t2_layers), labels=labels,
                        unified=True, seconds=np.float32(time.perf_counter() - t0))
    print(f"{tag} {time.perf_counter() - t0:.0f}s | NH last hidden @epoch{epochs[-1]} {nh_t2[-1, -2]:.3f}", flush=True)
    return out


def procrustes_chain(cfg_dir: Path) -> Path:
    """t-SNE + Procrustes (`tsne_proc`): the t-SNE frames, with every T2 layer rotated/reflected onto the previous one."""
    t0 = time.perf_counter()
    with np.load(cfg_dir / "proj_tsne.npz") as base:
        proj = {key: base[key] for key in base.files}
    t2 = proj["t2"].copy()
    for ei in range(len(t2)):
        previous = t2[ei, 0]
        for li in range(1, t2.shape[1]):
            previous = procrustes_align(t2[ei, li], previous)
            t2[ei, li] = normalize(previous)
    proj["t2"] = t2  # rigid transforms: nh_t1 / nh_t2 are unchanged
    out = cfg_dir / "proj_tsne_proc.npz"
    np.savez_compressed(out, **proj, seconds=np.float32(time.perf_counter() - t0))
    print(f"[{cfg_dir.parent.name}/{cfg_dir.name}/tsne_proc] written from proj_tsne.npz", flush=True)
    return out


# --------------------------------------------------------------------------- entry point

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", nargs="+", choices=list(config.DATASETS), default=list(config.DATASETS))
    p.add_argument("--model", nargs="+", choices=list(config.TRAINING), default=list(config.TRAINING))
    p.add_argument("--method", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    p.add_argument("--raw", type=Path, default=config.DEFAULT_RAW_DIR, help="training outputs folder")
    p.add_argument("--force", action="store_true", help="recompute existing projections")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for dataset in args.dataset:
        for method in args.method:
            todo = [m for m in args.model
                    if args.force or not (args.raw / dataset / m / f"proj_{method}.npz").exists()]
            if not todo:
                print(f"[{dataset}/{method}] projections exist, skipping (use --force to recompute).")
                continue
            if method == "tsne_proc":
                for model in todo:
                    procrustes_chain(args.raw / dataset / model)
                continue
            # The Input projection depends only on the dataset: compute it once for both models.
            cfg = json.loads((args.raw / dataset / todo[0] / "config.json").read_text())
            labels, x_input = load_input(args.raw, dataset, cfg)
            input_ref = {"tsne": tsne, "umap": umap2, "aumap": umap2, "grid": umap2}.get(method, pca2)(x_input)
            for model in todo:
                if method == "grid":
                    project_grid(args.raw, dataset, model, labels, input_ref)
                else:
                    project_config(args.raw, dataset, model, method, labels, input_ref)


if __name__ == "__main__":
    main()
