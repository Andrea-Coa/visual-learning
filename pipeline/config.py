"""Central configuration: datasets, training hyperparameters, snapshot schedule and paths.

Hyperparameters follow Rauber et al. (2017), Sec. 4.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"  # torchvision downloads
DEFAULT_RAW_DIR = PROJECT_ROOT / "raw"    # training outputs (activations, metrics, weights)

SEED = 42
SUBSET_SIZE = 1000  # test observations whose activations are recorded (stratified)
EVAL_BATCH_SIZE = 1000


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    channels: int
    size: int  # images are size x size
    classes: tuple[str, ...]

    @property
    def n_classes(self) -> int:
        return len(self.classes)


DIGITS = tuple(str(i) for i in range(10))

DATASETS: dict[str, DatasetSpec] = {
    "mnist": DatasetSpec("mnist", channels=1, size=28, classes=DIGITS),
    "svhn": DatasetSpec("svhn", channels=3, size=32, classes=DIGITS),
    "cifar10": DatasetSpec(
        "cifar10", channels=3, size=32,
        classes=("airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"),
    ),
}


@dataclass(frozen=True)
class TrainSpec:
    batch_size: int
    decay: float  # Keras-style learning-rate decay: lr_t = lr / (1 + decay * iteration)
    lr: float = 0.01
    momentum: float = 0.9
    epochs: int = 50

    def to_dict(self) -> dict:
        return asdict(self)


# Test accuracies reported in the paper (Table 1), kept for comparison in metrics.json.
PAPER_ACCURACY = {
    "mnist": {"mlp": 0.9852, "cnn": 0.9962},
    "svhn": {"mlp": 0.7738, "cnn": 0.9376},
    "cifar10": {"mlp": 0.5291, "cnn": 0.7919},
}

TRAINING: dict[str, TrainSpec] = {
    "mlp": TrainSpec(batch_size=16, decay=1e-9),
    "cnn": TrainSpec(batch_size=32, decay=1e-6),
}

# Per-(dataset, model) deviations from TRAINING. With the paper's lr = 0.01 the MLP diverges on the
# 3072-dimensional SVHN / CIFAR-10 inputs: accuracy peaks in epoch 1, collapses to a single predicted
# class within a few epochs and the loss ends in NaN. lr = 0.001 trains stably (see PLAN.md, 2.3).
TRAINING_OVERRIDES: dict[tuple[str, str], dict] = {
    ("svhn", "mlp"): {"lr": 0.001},
    ("cifar10", "mlp"): {"lr": 0.001},
}


def training_spec(dataset: str, model: str) -> TrainSpec:
    """Hyperparameters for one configuration: model defaults plus dataset-specific overrides."""
    return replace(TRAINING[model], **TRAINING_OVERRIDES.get((dataset, model), {}))

# Epochs at which subset activations are stored. Denser at the start, where
# representations change the most. Epoch 0 = untrained network.
SNAPSHOT_EPOCHS = (0, 1, 2, 3, 5, 8, 12, 18, 25, 35, 50)


def snapshot_schedule(epochs: int) -> list[int]:
    """Snapshot epochs that fit in a run of `epochs` epochs; always includes 0 and the last one."""
    return sorted({e for e in SNAPSHOT_EPOCHS if e <= epochs} | {0, epochs})
