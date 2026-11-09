# Visual Learning: how a neural network learns, epoch by epoch and layer by layer

DS5343 Data Visualization (UTEC) homework. We train neural networks on **MNIST, SVHN and CIFAR-10**,
record the hidden-layer activations of 1000 test images during training, project them to 2D with t-SNE
and explore their evolution with D3:

- **T1**: evolution across **epochs** (how representations change while the network trains).
- **T2**: evolution across **layers** (how representations change as an image flows through the network).

Based on Rauber et al., *Visualizing the Hidden Activity of Artificial Neural Networks* (IEEE TVCG 2017), in `paper/`.
The assignment is in `task-description.md`; the plan and its progress are in `PLAN.md`.

## Project layout

```
pipeline/                 Python pipeline (runs on Colab or locally)
  config.py               datasets, hyperparameters, snapshot epochs, paths
  data.py                 dataset loading, normalization, fixed stratified test subset
  models.py               MLP and CNN from the paper, exposing hidden activations
  metrics.py              confusion matrix, precision / recall / F1
  train.py                training + activation snapshots + metrics (entry point)
notebooks/colab_train.ipynb   runs pipeline/train.py on a Colab T4
app/                      Flask + D3 app (later phases)
raw/                      training outputs (git-ignored, large)
data/                     downloaded datasets (git-ignored)
```

## Models and training

As in the paper (Sec. 4), with plain SGD (lr 0.01, momentum 0.9, Keras-style lr decay):

| Model | Architecture | Recorded hidden layers | Batch |
|---|---|---|---|
| MLP | 4 × Dense(1000, ReLU), dropout 0.2/0.3/0.4/0.5 → softmax | `h1`, `h2`, `h3`, `h4` (1000 each) | 16 |
| CNN | [conv32 conv32 pool] → [conv64 conv64 pool] → Dense(512) → softmax; dropout 0.25/0.25/0.5 | `conv1` (6272 / 8192), `conv2` (3136 / 4096), `fc` (512) | 32 |

- 50 epochs per configuration (paper: 100). Activations are snapshotted at epochs
  `0, 1, 2, 3, 5, 8, 12, 18, 25, 35, 50` (epoch 0 = untrained), more often early because that's when representations change most.
- The subset is 1000 test images, 100 per class, chosen with a fixed seed: the same images in every
  snapshot, model and Colab account.

## Step by step: training

### Option A: Google Colab (recommended, T4 GPU)

The code comes from GitHub (`https://github.com/Andrea-Coa/visual-learning`, public, so no token needed). Only the
**outputs** go to Google Drive, so they survive disconnects.

1. **Push your changes.** Colab trains whatever is on `master` on GitHub, so push any local changes to `pipeline/` first.
2. **Open the notebook in Colab.** Go to `File > Open notebook > GitHub`, paste the repo URL and pick
   `notebooks/colab_train.ipynb`. Or open
   `https://colab.research.google.com/github/Andrea-Coa/visual-learning/blob/master/notebooks/colab_train.ipynb`.
3. **Select the GPU.** `Runtime > Change runtime type > T4 GPU`.
4. **Configure.** In the configuration cell, choose which `DATASETS` / `MODELS` this account trains
   (default: all six configurations).
5. **Run all cells** (`Runtime > Run all`, and accept the Drive permission prompt). The notebook:
   - clones the repo to `/content/visual-learning` (or `git pull`s it if it's already there) and `cd`s into it,
   - mounts Drive; outputs go to `MyDrive/visual-learning/raw/`, datasets to the local `/content/data`,
   - runs an optional ~1 minute smoke test,
   - trains: `python -m pipeline.train --dataset ... --model ... --data-dir /content/data --out <Drive>/raw`,
   - prints a summary of final accuracies next to the paper's.
6. **If Colab disconnects**, reconnect and run all cells again. The code is cloned again and the outputs on Drive
   are kept, so finished configurations are skipped and interrupted ones resume from the last completed epoch.
7. **Download the results.** Download `MyDrive/visual-learning/raw/` from Drive and put it as `raw/` in the local
   project root. `raw/` is git-ignored: the outputs are too large for the repo.

Changes made to the notebook inside Colab are not saved to the repo. Edit `notebooks/colab_train.ipynb` locally
and push, or use `File > Save a copy in GitHub`.

#### Splitting the work across several Colab accounts

Every account opens the same notebook from GitHub and writes to its own Drive. Every configuration writes only to
its own folder `raw/<dataset>/<model>/`, and the subset is identical everywhere (fixed seed). So each account can
train a different part (set `DATASETS` in the configuration cell). Merge them by copying every account's
`raw/<dataset>/` folders into one local `raw/`.

| Account | `DATASETS` | `MODELS` | Estimated time |
|---|---|---|---|
| 1 | `mnist` | `mlp cnn` | ~15–20 min |
| 2 | `svhn` | `mlp cnn` | ~20–30 min |
| 3 | `cifar10` | `mlp cnn` | ~12–18 min |

(With two accounts: `mnist cifar10` on one, `svhn` on the other.)

### Option B: locally

```bash
git clone https://github.com/Andrea-Coa/visual-learning.git
cd visual-learning
python -m venv .venv
.venv/Scripts/activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python -m pipeline.train        # all configurations, or e.g. --dataset mnist --model mlp
```

The PyPI build of PyTorch on Windows is CPU-only, and a full run on CPU takes many hours. Use local runs for
smoke tests only:

```bash
python -m pipeline.train --dataset mnist --model mlp cnn --epochs 1 --max-batches 50 --out smoke
```

### Command-line options (`python -m pipeline.train --help`)

| Option | Default | Meaning |
|---|---|---|
| `--dataset` | all | one or more of `mnist svhn cifar10` |
| `--model` | all | one or more of `mlp cnn` |
| `--epochs` | 50 | override number of epochs |
| `--data-dir` | `data/` | where torchvision downloads datasets |
| `--out` | `raw/` | output folder |
| `--max-batches` | none | limit batches per epoch (smoke tests) |
| `--restart` | off | delete previous outputs of the selected configs and retrain |
| `--seed` | 42 | seed for weights, shuffling and subset selection |

## Training time estimate

These are **estimates**, not measurements: step counts × typical T4 step times for these small models, where Python/kernel-launch
overhead dominates because the batches are small (16 for MLP, 32 for CNN). Each epoch also evaluates the full train and test sets.
After the first epoch, every log line prints the real epoch time and ETA, so check it early.

| Config | Steps / epoch | Est. time / epoch | Est. 50 epochs |
|---|---|---|---|
| MNIST MLP | 3750 | 7–11 s | 6–9 min |
| MNIST CNN | 1875 | 8–12 s | 7–10 min |
| SVHN MLP | 4579 | 9–14 s | 8–12 min |
| SVHN CNN | 2290 | 10–15 s | 8–13 min |
| CIFAR-10 MLP | 3125 | 6–9 s | 5–8 min |
| CIFAR-10 CNN | 1563 | 7–10 s | 6–9 min |
| **Total** | | | **~45 min – 1 h** (+ ~5 min downloads) |

Plan for up to **2 h** to be safe. That usually fits in one free Colab session, but free-tier GPU quotas vary, so
splitting across accounts (table above) is a reasonable fallback. If the real epoch times are much higher, lower
`--epochs` (e.g. 35): the snapshot schedule adapts automatically.

Disk usage on Drive (measured on MNIST, extrapolated): about 4 MB per MLP snapshot and 9–12 MB per CNN
snapshot, so **~60–70 MB per MLP and ~100–140 MB per CNN configuration**. That's **~0.6 GB** in total, plus a
temporary `checkpoint.pt` (15–50 MB) per configuration while it trains.

## Training outputs

```
raw/<dataset>/
  subset.npz                    indices (into the test set), labels, images [1000, H, W, C] uint8, classes
  <model>/
    config.json                 hyperparameters, layer sizes, snapshot epochs, normalization, device, versions
    history.json                one record per epoch (0..E): train/test loss & accuracy (eval mode, full sets),
                                running (dropout-on) loss/accuracy, lr, timings, subset accuracy at snapshots
    snapshots/epoch_XXX.npz     epoch, layers, <layer> [1000, features] float16, probs [1000, 10], preds [1000]
    metrics.json                final test metrics (accuracy, loss, per-class precision/recall/F1, confusion matrix),
                                final train accuracy, generalization gap, best epoch, paper accuracy, training time
    model_final.pt              final weights (state_dict)
```

`metrics.json` is written last, so its presence means the configuration finished. Load a snapshot with:

```python
import numpy as np
s = np.load("raw/mnist/mlp/snapshots/epoch_050.npz")
s["layers"]            # ['h1' 'h2' 'h3' 'h4']
s["h4"].shape          # (1000, 1000)
```
