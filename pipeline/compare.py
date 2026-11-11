"""Compare the 2D projection methods of pipeline.project (PLAN S.4) with the same metrics.

Per configuration and method (every raw/<dataset>/<model>/proj_<method>.npz found):
    nh_t1        mean neighborhood hit (k = 6) of all T1 frames (every layer, every epoch)
    nh_t2        mean neighborhood hit of the T2 frames of the last epoch (hidden layers + Output)
    trust        trustworthiness (k = 10) of the T2 frames of the last epoch against the activations:
                 are the 2D neighbors also neighbors in the layer? (label-free check of faithfulness)
    move_t2      mean displacement of a point between consecutive T2 layers, divided by the displacement
                 after shuffling the points (1 = as unstable as random; lower = steadier), all epochs
    flips_t2     share of consecutive T2 layer pairs whose best rigid match needs a rotation > 90° or a
                 reflection, i.e. the map visibly turned or mirrored between two layers
    move_t1      the same as move_t2 between consecutive epochs of T1
    seconds      projection wall time (stored by the S.4 methods only)

The S.4 decision weighs steadiness (move, flips) against faithfulness (nh, trust): an aligned method that
pins points in place could hide a real reorganization, which would show up as lower nh / trust.

Usage:
    python -m pipeline.compare                                   # table of every method found in raw/
    python -m pipeline.compare --sweep 0.001 0.01 0.1 --dataset svhn --model cnn
        # Aligned UMAP alignment_regularisation sweep on the last epoch (T2 only), to choose ALIGNMENT_REG

Output: printed Markdown tables; the full numbers go to raw/projection_comparison.json.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import trustworthiness

from . import config
from .project import (METHODS, aligned_umap, load_input, load_snapshot, neighborhood_hit, normalize,
                      reduce_dims, umap2)

TRUST_K = 10
METRICS = ("nh_t1", "nh_t2", "trust", "move_t2", "flips_t2", "move_t1", "seconds")


def frame_change(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> tuple[float, bool]:
    """(displacement ratio vs shuffled points, turned-or-mirrored) between two frames of the same points."""
    moved = np.linalg.norm(a - b, axis=1).mean()
    shuffled = np.linalg.norm(a - b[rng.permutation(len(b))], axis=1).mean()
    rotation, _ = orthogonal_procrustes(b - b.mean(0), a - a.mean(0))
    angle = abs(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
    return float(moved / shuffled), bool(np.linalg.det(rotation) < 0 or angle > 90)


def sequence_change(frames: np.ndarray, rng: np.random.Generator) -> tuple[list[float], list[bool]]:
    changes = [frame_change(a, b, rng) for a, b in zip(frames, frames[1:])]
    return [c[0] for c in changes], [c[1] for c in changes]


def trust_t2(frames: np.ndarray, layers: list[str], acts: dict[str, np.ndarray]) -> float:
    """Mean trustworthiness of the T2 frames (hidden layers + Output) of one epoch."""
    return float(np.mean([trustworthiness(reduce_dims(acts[layer]), frame, n_neighbors=TRUST_K)
                          for frame, layer in zip(frames[1:], layers[1:])]))


def evaluate(proj: np.lib.npyio.NpzFile, last_acts: dict[str, np.ndarray]) -> dict:
    rng = np.random.default_rng(config.SEED)
    t1, t2, layers = proj["t1"], proj["t2"], [str(name) for name in proj["t2_layers"]]
    move_t2, flips_t2 = zip(*(sequence_change(epoch_frames, rng) for epoch_frames in t2))
    move_t1 = [sequence_change(layer_frames, rng)[0] for layer_frames in t1]
    return {
        "nh_t1": float(proj["nh_t1"].mean()),
        "nh_t2": float(proj["nh_t2"][-1, 1:].mean()),
        "trust": trust_t2(t2[-1], layers, last_acts),
        "move_t2": float(np.mean(move_t2)),
        "flips_t2": float(np.mean(flips_t2)),
        "move_t1": float(np.mean(move_t1)),
        "seconds": float(proj["seconds"]) if "seconds" in proj.files else None,
    }


def fmt(value: float | None, metric: str) -> str:
    if value is None:
        return "—"
    if metric == "seconds":
        return f"{value:.0f}"
    if metric == "flips_t2":
        return f"{value:.0%}"
    return f"{value:.3f}"


def table(rows: list[dict], keys: tuple[str, ...]) -> str:
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(row[k] if isinstance(row[k], str) else fmt(row[k], k) for k in keys) + " |"
              for row in rows]
    return "\n".join(lines)


def compare(raw: Path, datasets: list[str], models: list[str]) -> list[dict]:
    rows = []
    for dataset in datasets:
        for model in models:
            cfg_dir = raw / dataset / model
            found = [m for m in METHODS if (cfg_dir / f"proj_{m}.npz").exists()]
            if not found:
                continue
            last_epoch = json.loads((cfg_dir / "config.json").read_text())["snapshot_epochs"][-1]
            last_acts = load_snapshot(cfg_dir, last_epoch)
            for method in found:
                with np.load(cfg_dir / f"proj_{method}.npz") as proj:
                    rows.append({"config": f"{dataset}/{model}", "method": method, **evaluate(proj, last_acts)})
                print(f"[{dataset}/{model}] {method} evaluated", flush=True)
    return rows


def sweep(raw: Path, datasets: list[str], models: list[str], values: list[float]) -> list[dict]:
    """Aligned UMAP on the last epoch's T2 sequence for several alignment_regularisation values."""
    rows = []
    for dataset in datasets:
        for model in models:
            cfg_dir = raw / dataset / model
            if not (cfg_dir / "config.json").exists():
                continue
            cfg = json.loads((cfg_dir / "config.json").read_text())
            labels, x_input = load_input(raw, dataset, cfg)
            acts = load_snapshot(cfg_dir, cfg["snapshot_epochs"][-1])
            layers = ["input"] + list(cfg["layers"]) + ["output"]
            ref = umap2(x_input)
            for value in values:
                start = time.perf_counter()
                frames = np.array([normalize(y) for y in
                                   aligned_umap([x_input] + [acts[layer] for layer in layers[1:]], ref, value)])
                move, flips = sequence_change(frames, np.random.default_rng(config.SEED))
                rows.append({"config": f"{dataset}/{model}", "alignment": f"{value:g}",
                             "nh_t2": float(np.mean([neighborhood_hit(f, labels) for f in frames[1:]])),
                             "trust": trust_t2(frames, layers, acts), "move_t2": float(np.mean(move)),
                             "flips_t2": float(np.mean(flips)), "seconds": time.perf_counter() - start})
                print(f"[{dataset}/{model}] alignment {value:g}: {rows[-1]['seconds']:.0f}s", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", nargs="+", choices=list(config.DATASETS), default=list(config.DATASETS))
    p.add_argument("--model", nargs="+", choices=list(config.TRAINING), default=list(config.TRAINING))
    p.add_argument("--raw", type=Path, default=config.DEFAULT_RAW_DIR, help="pipeline outputs folder")
    p.add_argument("--sweep", nargs="+", type=float, metavar="REG",
                   help="instead of the table: Aligned UMAP alignment_regularisation values to try")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.sweep:
        rows = sweep(args.raw, args.dataset, args.model, args.sweep)
        print(table(rows, ("config", "alignment", "nh_t2", "trust", "move_t2", "flips_t2", "seconds")))
        return
    rows = compare(args.raw, args.dataset, args.model)
    print(table(rows, ("config", "method") + METRICS))
    out = args.raw / "projection_comparison.json"
    out.write_text(json.dumps(rows, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
