"""Train the MLP / CNN models and record hidden-layer activations of a fixed test subset.

Examples:
    python -m pipeline.train                                   # all 3 datasets x 2 models
    python -m pipeline.train --dataset svhn --model cnn        # a single configuration
    python -m pipeline.train --dataset mnist --model mlp --epochs 1 --max-batches 50 --out smoke

Outputs, per configuration, in <out>/<dataset>/<model>/ (see README.md for the full layout):
    config.json            hyperparameters, layer sizes, snapshot epochs, environment
    history.json           per-epoch train/test loss and accuracy (rewritten every epoch)
    snapshots/epoch_XXX.npz  hidden activations + predictions of the subset at each snapshot epoch
    metrics.json           final metrics (written last: its presence means the run is complete)
    model_final.pt         final weights
    checkpoint.pt          resume state; deleted once the run completes
<out>/<dataset>/subset.npz holds the subset (test indices, labels, images), shared by both models.

Re-running a command resumes interrupted runs from checkpoint.pt and skips completed ones.
"""
import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from . import config
from .data import ImageData, batch_indices, load_dataset, save_subset, stratified_subset
from .metrics import classification_report, confusion_matrix
from .models import build_model


# --------------------------------------------------------------------------- I/O helpers

def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def save_atomic(obj, path: Path) -> None:
    """torch.save through a temp file, so a disconnect never leaves a corrupt checkpoint."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# --------------------------------------------------------------------------- evaluation

@torch.no_grad()
def predict(model, data: ImageData, x: torch.Tensor, with_hidden: bool = False):
    """Eval-mode forward pass of uint8 images in batches.

    Returns logits [N, classes], plus {layer: float16 ndarray [N, features]} if `with_hidden`.
    """
    model.eval()
    logits, hidden = [], {}
    for start in range(0, len(x), config.EVAL_BATCH_SIZE):
        xb = data.normalize(x[start:start + config.EVAL_BATCH_SIZE])
        if with_hidden:
            out, acts = model(xb, return_hidden=True)
            for name, a in acts.items():
                hidden.setdefault(name, []).append(a.half().cpu())
        else:
            out = model(xb)
        logits.append(out.float())
    logits = torch.cat(logits)
    if with_hidden:
        return logits, {name: torch.cat(parts).numpy() for name, parts in hidden.items()}
    return logits


def evaluate(model, data: ImageData, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float, torch.Tensor]:
    logits = predict(model, data, x)
    preds = logits.argmax(1)
    return F.cross_entropy(logits, y).item(), (preds == y).float().mean().item(), preds


def save_snapshot(path: Path, epoch: int, model, data: ImageData, subset_idx: torch.Tensor) -> float:
    """Store hidden activations, softmax probabilities and predictions of the subset. Returns subset accuracy."""
    logits, hidden = predict(model, data, data.x_test[subset_idx], with_hidden=True)
    probs = logits.softmax(1)
    preds = probs.argmax(1)
    np.savez_compressed(
        path, epoch=epoch, layers=np.array(list(hidden)),
        probs=probs.half().cpu().numpy(), preds=preds.to(torch.int8).cpu().numpy(), **hidden,
    )
    return (preds == data.y_test[subset_idx]).float().mean().item()


# --------------------------------------------------------------------------- training

def train_epoch(model, data: ImageData, optimizer, scheduler, batch_size: int,
                generator: torch.Generator, max_batches: int | None) -> tuple[float, float]:
    """One pass over the training set. Returns running (dropout-on) loss and accuracy."""
    model.train()
    device = data.x_train.device
    loss_sum = torch.zeros((), device=device)
    correct = torch.zeros((), dtype=torch.long, device=device)
    seen = 0
    for i, idx in enumerate(batch_indices(len(data.x_train), batch_size, generator, device)):
        if max_batches is not None and i >= max_batches:
            break
        xb, yb = data.normalize(data.x_train[idx]), data.y_train[idx]
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        # Accumulate on the device: calling .item() every step would force a GPU sync.
        loss_sum += loss.detach() * len(idx)
        correct += (logits.argmax(1) == yb).sum()
        seen += len(idx)
    mean_loss = loss_sum.item() / seen
    if not np.isfinite(mean_loss):
        raise FloatingPointError("Training loss diverged (NaN/inf).")
    return mean_loss, correct.item() / seen


def epoch_record(epoch: int, model, data: ImageData, optimizer, running=(None, None), train_time=0.0) -> dict:
    """Eval-mode metrics on the full train and test sets (dropout off, so they are comparable)."""
    train_loss, train_acc, _ = evaluate(model, data, data.x_train, data.y_train)
    test_loss, test_acc, _ = evaluate(model, data, data.x_test, data.y_test)
    return {
        "epoch": epoch,
        "train_loss": train_loss, "train_acc": train_acc,
        "test_loss": test_loss, "test_acc": test_acc,
        "running_loss": running[0], "running_acc": running[1],
        "lr": optimizer.param_groups[0]["lr"],
        "train_time_s": train_time,
    }


def run_config(dataset: str, model_name: str, data: ImageData, subset_idx: torch.Tensor,
               args: argparse.Namespace, device: torch.device) -> None:
    spec, hp = config.DATASETS[dataset], config.TRAINING[model_name]
    epochs = args.epochs or hp.epochs
    snapshot_epochs = config.snapshot_schedule(epochs)
    out_dir = args.out / dataset / model_name
    snap_dir = out_dir / "snapshots"
    ckpt_path = out_dir / "checkpoint.pt"
    tag = f"[{dataset}/{model_name}]"

    if args.restart and out_dir.exists():
        shutil.rmtree(out_dir)
    if (out_dir / "metrics.json").exists():
        print(f"{tag} already complete, skipping (use --restart to retrain).")
        return
    snap_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    model = build_model(model_name, spec).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=hp.lr, momentum=hp.momentum)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda it: 1.0 / (1.0 + hp.decay * it))
    generator = torch.Generator().manual_seed(args.seed)  # batch shuffling

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        generator.set_state(ckpt["generator"])
        torch.set_rng_state(ckpt["cpu_rng"])
        if device.type == "cuda" and ckpt["cuda_rng"] is not None:
            torch.cuda.set_rng_state(ckpt["cuda_rng"])
        history, start_epoch = ckpt["history"], ckpt["epoch"] + 1
        print(f"{tag} resuming from epoch {ckpt['epoch']}")
    else:
        _, dims = predict(model, data, data.x_test[:1], with_hidden=True)
        write_json(out_dir / "config.json", {
            "dataset": dataset, "model": model_name, "classes": list(spec.classes),
            "training": {**hp.to_dict(), "epochs": epochs, "max_batches": args.max_batches},
            "seed": args.seed, "subset_size": len(subset_idx), "snapshot_epochs": snapshot_epochs,
            "layers": {name: a.shape[1] for name, a in dims.items()},
            "n_params": sum(p.numel() for p in model.parameters()),
            "normalization": {"mean": data.mean.tolist(), "std": data.std.tolist()},
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "torch_version": torch.__version__,
        })
        record = epoch_record(0, model, data, optimizer)
        record["subset_acc"] = save_snapshot(snap_dir / "epoch_000.npz", 0, model, data, subset_idx)
        history, start_epoch = [record], 1
        print(f"{tag} untrained: test acc {record['test_acc']:.4f}")

    run_start = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.perf_counter()
        running = train_epoch(model, data, optimizer, scheduler, hp.batch_size, generator, args.max_batches)
        record = epoch_record(epoch, model, data, optimizer, running, train_time=time.perf_counter() - t0)
        if epoch in snapshot_epochs:
            record["subset_acc"] = save_snapshot(snap_dir / f"epoch_{epoch:03d}.npz", epoch, model, data, subset_idx)
        record["epoch_time_s"] = time.perf_counter() - t0
        history.append(record)

        write_json(out_dir / "history.json", history)
        save_atomic({
            "epoch": epoch, "history": history,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "generator": generator.get_state(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        }, ckpt_path)

        per_epoch = (time.perf_counter() - run_start) / (epoch - start_epoch + 1)
        print(f"{tag} epoch {epoch:3d}/{epochs} | train {record['train_loss']:.4f} / {record['train_acc']:.4f}"
              f" | test {record['test_loss']:.4f} / {record['test_acc']:.4f}"
              f" | {record['epoch_time_s']:.1f}s | ETA {per_epoch * (epochs - epoch) / 60:.1f} min", flush=True)

    write_final_metrics(out_dir, dataset, model_name, model, data, history)
    torch.save(model.state_dict(), out_dir / "model_final.pt")
    ckpt_path.unlink(missing_ok=True)
    print(f"{tag} done -> {out_dir}")


def write_final_metrics(out_dir: Path, dataset: str, model_name: str, model, data: ImageData,
                        history: list[dict]) -> None:
    spec = data.spec
    test_loss, _, test_preds = evaluate(model, data, data.x_test, data.y_test)
    cm = confusion_matrix(data.y_test.cpu().numpy(), test_preds.cpu().numpy(), spec.n_classes)
    final = history[-1]
    best = max(history, key=lambda r: r["test_acc"])
    write_json(out_dir / "metrics.json", {
        "dataset": dataset, "model": model_name, "epochs": final["epoch"],
        "test": {"loss": test_loss, **classification_report(cm, spec.classes)},
        "train": {"loss": final["train_loss"], "accuracy": final["train_acc"]},
        "generalization_gap": final["train_acc"] - final["test_acc"],
        "best_test_epoch": {"epoch": best["epoch"], "test_acc": best["test_acc"]},
        "paper_test_accuracy": config.PAPER_ACCURACY[dataset][model_name],
        "subset_accuracy": {r["epoch"]: r["subset_acc"] for r in history if "subset_acc" in r},
        "training_time_s": sum(r.get("epoch_time_s", 0.0) for r in history),
    })


# --------------------------------------------------------------------------- entry point

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", nargs="+", choices=list(config.DATASETS), default=list(config.DATASETS))
    p.add_argument("--model", nargs="+", choices=list(config.TRAINING), default=list(config.TRAINING))
    p.add_argument("--epochs", type=int, default=None, help="override the epochs in config.TRAINING")
    p.add_argument("--data-dir", type=Path, default=config.DEFAULT_DATA_DIR, help="torchvision download folder")
    p.add_argument("--out", type=Path, default=config.DEFAULT_RAW_DIR, help="output folder")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-batches", type=int, default=None, help="limit batches per epoch (smoke tests)")
    p.add_argument("--restart", action="store_true", help="delete previous outputs and train from scratch")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Device: {torch.cuda.get_device_name(device)}")
    else:
        print("Device: CPU (training will be slow; use Colab for the full runs)")

    for dataset in args.dataset:
        spec = config.DATASETS[dataset]
        pending = [m for m in args.model
                   if args.restart or not (args.out / dataset / m / "metrics.json").exists()]
        if not pending:
            print(f"[{dataset}] all requested models complete, skipping.")
            continue

        print(f"[{dataset}] loading data...")
        data = load_dataset(spec, args.data_dir)
        subset = stratified_subset(data.y_test.numpy(), config.SUBSET_SIZE, spec.n_classes, args.seed)
        subset_path = args.out / dataset / "subset.npz"
        if not subset_path.exists():
            save_subset(subset_path, data, subset)
        data.to(device)
        subset_idx = torch.as_tensor(subset, device=device)

        for model_name in pending:
            run_config(dataset, model_name, data, subset_idx, args, device)

        del data
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
