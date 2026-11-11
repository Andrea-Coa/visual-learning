"""Export the data files of the static site (app/data/) from the pipeline outputs in raw/.

Usage:
    python -m pipeline.export                  # all configurations found in raw/
    python -m pipeline.export --out app/data

Layout (compact JSON; coordinates are integers 0..SCALE in the unit square, x right, y down):
    index.json                                datasets, models, projections, sprite sheets
    <dataset>/sprites.png                     the 1000 subset images, row-major grid, in observation order
    <dataset>/<model>/meta.json               classes, architecture, snapshot epochs, learning curves, metrics
    <dataset>/<model>/nodes.json              labels, test indices, sample order, per-epoch predictions/probabilities
    <dataset>/<model>/coords_<method>.json    all T1 frames (per layer, per epoch) and T2 frames (per epoch, per layer)
    <dataset>/<model>/trails/<method>/t1_<layer>.json   bundled trails over epochs, per strength
    <dataset>/<model>/trails/<method>/t2_e<epoch>.json  bundled trails over layers, per strength
The full schema is documented in README.md ("Site data files").
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from . import config

SCALE = 9999          # coordinate quantization (same as pipeline.bundle)
PROB_SCALE = 1000     # probabilities are stored as integers in per mille
SPRITE_COLUMNS = 40   # 1000 images -> 40 x 25 grid
METHODS = ("tsne", "pca", "umap", "aumap", "tsne_proc", "grid")  # exported when raw/ has proj_<method>.npz
DISPLAY_NAMES = {"mnist": "MNIST", "svhn": "SVHN", "cifar10": "CIFAR-10", "mlp": "MLP", "cnn": "CNN"}

# Layer descriptions for the architecture panel. Must match pipeline/models.py.
LAYER_TYPES = {
    "mlp": {name: ("dense · ReLU", dropout)
            for name, dropout in zip(("h1", "h2", "h3", "h4"), (0.2, 0.3, 0.4, 0.5))},
    "cnn": {"conv1": ("2× conv 3×3 ×32 · ReLU · max-pool", 0.25),
            "conv2": ("2× conv 3×3 ×64 · ReLU · max-pool", 0.25),
            "fc": ("dense · ReLU", 0.5)},
}


def write_json(path: Path, obj) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    return len(text)


def quantize(coords: np.ndarray) -> list:
    """[..., N, 2] floats in the unit square -> nested lists whose innermost item is a flat [x0, y0, x1, y1, ...].

    y is flipped so that 'up' in the projection is 'up' on screen (SVG y grows downwards).
    """
    flipped = coords.copy()
    flipped[..., 1] = 1 - flipped[..., 1]
    q = np.rint(flipped * SCALE).astype(np.int32)
    return q.reshape(*q.shape[:-2], -1).tolist()


def nested_sample_order(labels: np.ndarray, seed: int) -> list[int]:
    """Observation ids ordered so that every prefix of length 10k holds k observations per class.

    The UI shows the first n ids, so growing the sample only adds points (VIS_PLAN.md D10).
    """
    rng = np.random.RandomState(seed)
    per_class = [rng.permutation(np.flatnonzero(labels == c)) for c in np.unique(labels)]
    rounds = min(len(ids) for ids in per_class)
    blocks = [rng.permutation([ids[r] for ids in per_class]) for r in range(rounds)]
    return np.concatenate(blocks).tolist()


# --------------------------------------------------------------------------- per dataset

def export_sprites(raw: Path, out: Path, dataset: str) -> dict:
    with np.load(raw / dataset / "subset.npz") as subset:
        images = subset["images"]  # [N, H, W, C] uint8
    n, size, _, channels = images.shape
    rows = -(-n // SPRITE_COLUMNS)
    sheet = np.zeros((rows * size, SPRITE_COLUMNS * size, channels), np.uint8)
    for i, image in enumerate(images):
        r, c = divmod(i, SPRITE_COLUMNS)
        sheet[r * size:(r + 1) * size, c * size:(c + 1) * size] = image
    path = out / dataset / "sprites.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet.squeeze(-1) if channels == 1 else sheet).save(path, optimize=True)
    return {"file": f"{dataset}/sprites.png", "cell": size, "columns": SPRITE_COLUMNS, "count": n}


# --------------------------------------------------------------------------- per configuration

def architecture(model: str, cfg: dict, spec: config.DatasetSpec) -> list[dict]:
    layers = [{"id": "input", "label": "Input", "units": spec.channels * spec.size ** 2,
               "type": f"pixels {spec.size}×{spec.size}×{spec.channels}", "dropout": None}]
    for name, units in cfg["layers"].items():
        kind, dropout = LAYER_TYPES[model][name]
        layers.append({"id": name, "label": name, "units": units, "type": kind, "dropout": dropout})
    layers.append({"id": "output", "label": "Output", "units": spec.n_classes, "type": "softmax", "dropout": None})
    return layers


def export_meta(cfg_dir: Path, dataset: str, model: str, proj: np.lib.npyio.NpzFile) -> dict:
    cfg = json.loads((cfg_dir / "config.json").read_text())
    history = json.loads((cfg_dir / "history.json").read_text())
    metrics = json.loads((cfg_dir / "metrics.json").read_text())
    spec = config.DATASETS[dataset]
    best = metrics["best_test_epoch"]
    return {
        "dataset": dataset, "model": model,
        "classes": list(spec.classes),
        "epochs": [int(e) for e in proj["epochs"]],
        "layers": architecture(model, cfg, spec),
        "t1Layers": [str(name) for name in proj["t1_layers"]],
        "t2Layers": [str(name) for name in proj["t2_layers"]],
        "training": {k: cfg["training"][k] for k in ("epochs", "batch_size", "lr", "momentum")},
        "history": {
            "epoch": [r["epoch"] for r in history],
            "trainAcc": [round(r["train_acc"], 5) for r in history],
            "testAcc": [round(r["test_acc"], 5) for r in history],
            "trainLoss": [round(r["train_loss"], 5) for r in history],
            "testLoss": [round(r["test_loss"], 5) for r in history],
        },
        "metrics": {
            "testAcc": round(metrics["test"]["accuracy"], 5),
            "trainAcc": round(metrics["train"]["accuracy"], 5),
            "paperTestAcc": metrics["paper_test_accuracy"],
            "bestEpoch": best["epoch"], "bestTestAcc": round(best["test_acc"], 5),
        },
    }


def export_nodes(raw: Path, cfg_dir: Path, dataset: str, epochs: list[int]) -> dict:
    with np.load(raw / dataset / "subset.npz") as subset:
        labels, test_index = subset["labels"], subset["indices"]
    preds, probs = [], []
    for epoch in epochs:
        with np.load(cfg_dir / "snapshots" / f"epoch_{epoch:03d}.npz") as snap:
            preds.append(snap["preds"].astype(int).tolist())
            p = np.rint(snap["probs"].astype(np.float32) * PROB_SCALE).astype(int)
            probs.append(p.ravel().tolist())  # flat [obs * classes]
    return {
        "n": len(labels), "labels": labels.tolist(), "testIndex": test_index.tolist(),
        "sampleOrder": nested_sample_order(labels, config.SEED),  # same for every model
        "probScale": PROB_SCALE, "preds": preds, "probs": probs,
    }


def export_coords(proj: np.lib.npyio.NpzFile) -> dict:
    coords = {
        "scale": SCALE,
        # t1[layer][epoch] and t2[epoch][layer] are flat [x0, y0, x1, y1, ...] arrays
        "t1": dict(zip([str(n) for n in proj["t1_layers"]], quantize(proj["t1"]))),
        "t2": quantize(proj["t2"]),
    }
    if "unified" in proj.files and bool(proj["unified"]):
        coords["unified"] = True  # T1 and T2 share coordinates (grid): the UI keeps points on a view switch
    return coords


def export_trails(cfg_dir: Path, out_dir: Path, method: str) -> int:
    trails = np.load(cfg_dir / f"trails_{method}.npz")
    strengths = [str(s) for s in trails["strengths"]]
    common = {"scale": int(trails["scale"]), "samplesPerSegment": int(trails["samples_per_segment"]),
              "strengths": strengths}

    def payload(views: np.ndarray) -> dict:  # views: [strength, N, P, 2] uint16
        flipped = views.astype(np.int32)
        flipped[..., 1] = int(trails["scale"]) - flipped[..., 1]
        return {**common, "trails": {s: flipped[i].reshape(len(flipped[i]), -1).tolist()
                                     for i, s in enumerate(strengths)}}

    size = 0
    for layer, views in zip(trails["t1_layers"], trails["t1"]):
        size += write_json(out_dir / "trails" / method / f"t1_{layer}.json", payload(views))
    for epoch, views in zip(trails["epochs"], trails["t2"]):
        size += write_json(out_dir / "trails" / method / f"t2_e{int(epoch)}.json", payload(views))
    return size


def export_config(raw: Path, out: Path, dataset: str, model: str) -> dict:
    cfg_dir, out_dir = raw / dataset / model, out / dataset / model
    if out_dir.exists():
        shutil.rmtree(out_dir)  # generated output only
    methods = [m for m in METHODS if (cfg_dir / f"proj_{m}.npz").exists()]
    proj = np.load(cfg_dir / f"proj_{methods[0]}.npz")
    meta = export_meta(cfg_dir, dataset, model, proj)
    size = write_json(out_dir / "meta.json", meta)
    size += write_json(out_dir / "nodes.json", export_nodes(raw, cfg_dir, dataset, meta["epochs"]))
    for method in methods:
        size += write_json(out_dir / f"coords_{method}.json", export_coords(np.load(cfg_dir / f"proj_{method}.npz")))
    bundled = [m for m in methods if (cfg_dir / f"trails_{m}.npz").exists()]
    for method in bundled:
        size += export_trails(cfg_dir, out_dir, method)
    print(f"[{dataset}/{model}] {size / 1e6:.1f} MB ({', '.join(methods)}; trails: {', '.join(bundled) or 'none'})")
    return {"id": model, "name": DISPLAY_NAMES[model], "methods": methods, "trails": bundled}


# --------------------------------------------------------------------------- entry point

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=config.DEFAULT_RAW_DIR, help="pipeline outputs folder")
    p.add_argument("--out", type=Path, default=config.PROJECT_ROOT / "app" / "data", help="site data folder")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    datasets = []
    for dataset, spec in config.DATASETS.items():
        models = [m for m in config.TRAINING if any((args.raw / dataset / m).glob("proj_*.npz"))]
        if not models:
            print(f"[{dataset}] no projections in {args.raw / dataset}, skipping.")
            continue
        sprite = export_sprites(args.raw, args.out, dataset)
        datasets.append({
            "id": dataset, "name": DISPLAY_NAMES[dataset], "classes": list(spec.classes), "sprite": sprite,
            "models": [export_config(args.raw, args.out, dataset, m) for m in models],
        })
    write_json(args.out / "index.json", {"scale": SCALE, "datasets": datasets})
    total = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    print(f"index.json written; app/data total {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
