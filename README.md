<p align="center">
  <img src="app/img/jamona.svg" alt="JaMoNA logo: a pink piglet holding a magnifying glass" width="160">
</p>

<h1 align="center">JaMoNA</h1>
<p align="center"><b>Jimena and Andrea's Model Network Analysis</b><br>
how a neural network learns, epoch by epoch and layer by layer</p>

DS5343 Data Visualization (UTEC) homework. We train neural networks on **MNIST, SVHN and CIFAR-10**,
record the hidden-layer activations of 1000 test images during training, project them to 2D with t-SNE
and explore their evolution with D3:

- **T1**: evolution across **epochs** (how representations change while the network trains).
- **T2**: evolution across **layers** (how representations change as an image flows through the network).

Based on Rauber et al., *Visualizing the Hidden Activity of Artificial Neural Networks* (IEEE TVCG 2017), in `paper/`.
The assignment is in `task-description.md`; the plan and its progress are in `PLAN.md`; the UI design spec is in
`VIS_PLAN.md`; the in-class demo script is in `PRESENTATION.md`.

**Live site:** <https://andrea-coa.github.io/jamona/> (GitHub Pages, see "Run the site").
**Quick start (local):** `python -m http.server 8000 -d app`, or `pip install flask && python app.py`
(<http://127.0.0.1:5000>). Everything below "Run the site" explains the site; the sections before it rebuild the data.

## Project layout

```
pipeline/                 Python pipeline (runs on Colab or locally)
  config.py               datasets, hyperparameters, snapshot epochs, paths
  data.py                 dataset loading, normalization, fixed stratified test subset
  models.py               MLP and CNN from the paper, exposing hidden activations
  metrics.py              confusion matrix, precision / recall / F1
  train.py                training + activation snapshots + metrics (entry point)
  project.py              t-SNE / PCA projections for T1 and T2 + neighborhood hit (entry point)
  bundle.py               class-aware bundling of the t-SNE trails (entry point)
  export.py               writes the site data files to app/data/ (entry point)
notebooks/colab_train.ipynb   runs pipeline/train.py on a Colab T4
app/                      static D3 site (HTML/CSS/JS + app/data/ JSON), deployed on GitHub Pages
app.py                    Flask wrapper that serves app/ (the assignment asks for a Flask app)
package.py                builds the submission ZIP (dist/jamona.zip)
raw/                      training outputs (git-ignored, large)
data/                     downloaded datasets (git-ignored)
```

## Models and training

As in the paper (Sec. 4), with plain SGD (lr 0.01, momentum 0.9, Keras-style lr decay):

| Model | Architecture | Recorded hidden layers | Batch |
|---|---|---|---|
| MLP | 4 × Dense(1000, ReLU), dropout 0.2/0.3/0.4/0.5 → softmax | `h1`, `h2`, `h3`, `h4` (1000 each) | 16 |
| CNN | [conv32 conv32 pool] → [conv64 conv64 pool] → Dense(512) → softmax; dropout 0.25/0.25/0.5 | `conv1` (6272 / 8192), `conv2` (3136 / 4096), `fc` (512) | 32 |

- **Deviation from the paper:** the SVHN and CIFAR-10 MLPs use **lr 0.001**. With lr 0.01 they diverged on Colab:
  SVHN MLP test accuracy peaked at 44% after epoch 1, then collapsed to predicting one class (~20%), and the loss
  became NaN at epoch 14. In a local 4-epoch check, lr 0.001 kept improving (SVHN 64% → 68% in two epochs).
  Scaling pixels to [0, 1] instead of standardizing them did not help (it collapsed in epoch 1). Set in
  `TRAINING_OVERRIDES` in `pipeline/config.py`.
- 50 epochs per configuration (paper: 100). Activations are snapshotted at epochs
  `0, 1, 2, 3, 5, 8, 12, 18, 25, 35, 50` (epoch 0 = untrained), more often early because that's when representations change most.
- The subset is 1000 test images, 100 per class, chosen with a fixed seed: the same images in every
  snapshot, model and Colab account.

## Step by step: training

### Option A: Google Colab (recommended, T4 GPU)

The code comes from GitHub (`https://github.com/Andrea-Coa/jamona`, public, so no token needed). Only the
**outputs** go to Google Drive, so they survive disconnects.

1. **Push your changes.** Colab trains whatever is on `main` on GitHub, so push any local changes to `pipeline/` first.
2. **Open the notebook in Colab.** Go to `File > Open notebook > GitHub`, paste the repo URL and pick
   `notebooks/colab_train.ipynb`. Or open
   `https://colab.research.google.com/github/Andrea-Coa/jamona/blob/main/notebooks/colab_train.ipynb`.
3. **Select the GPU.** `Runtime > Change runtime type > T4 GPU`.
4. **Configure.** In the configuration cell, choose which `DATASETS` / `MODELS` this account trains
   (default: all six configurations).
5. **Run all cells** (`Runtime > Run all`, and accept the Drive permission prompt). The notebook:
   - clones the repo to `/content/jamona` (or `git pull`s it if it's already there) and `cd`s into it,
   - mounts Drive; outputs go to `MyDrive/jamona/raw/`, datasets to the local `/content/data`,
   - runs an optional ~1 minute smoke test,
   - trains: `python -m pipeline.train --dataset ... --model ... --data-dir /content/data --out <Drive>/raw`,
   - prints a summary of final accuracies next to the paper's.
6. **If Colab disconnects**, reconnect and run all cells again. The code is cloned again and the outputs on Drive
   are kept, so finished configurations are skipped and interrupted ones resume from the last completed epoch.
7. **Download the results.** Download `MyDrive/jamona/raw/` from Drive and put it as `raw/` in the local
   project root. `raw/` is git-ignored: the outputs are too large for the repo.

8. **If a configuration diverges** (the log prints `FAILED: Training loss diverged`), the run continues with the
   other configurations. Its folder on Drive keeps the diverged checkpoint, so it must be retrained from scratch.
   Fix the cause (e.g. its learning rate in `pipeline/config.py`), push, and then either delete
   `MyDrive/jamona/raw/<dataset>/<model>/` in Drive and run all cells again, or run this in a new cell:
   `!python -m pipeline.train --dataset <dataset> --model <model> --restart --data-dir {DATA} --out {OUT}`.
   `--restart` deletes the previous outputs of every selected configuration, so name only the failed one.

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
git clone https://github.com/Andrea-Coa/jamona.git
cd jamona
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

## Step by step: projections (local)

After `raw/` contains the training outputs:

```bash
python -m pipeline.project                    # t-SNE and PCA for all configurations
python -m pipeline.project --method pca       # only PCA (seconds); --dataset / --model select configurations
```

Results go to `raw/<dataset>/<model>/proj_<method>.npz`. Existing files are skipped unless you pass `--force`.
On an 8-core laptop, PCA takes ~10 s per configuration and t-SNE ~15 min (~1.5 h for all six).

- **T1 (epochs):** for every layer (hidden layers + output), t-SNE of all snapshot epochs together, so the
  points of different epochs share one map. PCA: one PCA per epoch, each aligned (Procrustes) to the previous one.
- **T2 (layers):** for every snapshot epoch, Input → hidden layers → Output. Every t-SNE starts from the t-SNE of
  the Input layer, which keeps clusters in roughly the same place across layers and epochs. PCA: each layer aligned
  to the previous one.
- PCA uses `log1p` of the activations (ReLU outputs are heavy-tailed); t-SNE uses PCA-50 of the raw activations.
- The neighborhood hit (k = 6, share of each point's 6 nearest neighbors in the same class) is stored for every frame.

## Step by step: trail bundling (local)

After the projections exist:

```bash
python -m pipeline.bundle                     # all configurations (~20 s each); --force recomputes
```

Results go to `raw/<dataset>/<model>/trails_tsne.npz`. Every observation gets a trail through its t-SNE positions
(T1: over epochs, one set per layer; T2: over layers, one set per epoch). The trails are bundled with kernel density
edge bundling (KDEEB), the CPU counterpart of the paper's GPU bundling:
- **Class-aware:** a trail is only pulled towards trails of its own class, so bundles never mix class colors.
- **Two strengths are stored:** medium and strong. Unbundled trails are drawn directly from the projected points.
- **Fixed endpoints:** the first and last points of a trail never move.
- **Format:** coordinates are stored as integers 0–9999 in the unit square. Sample `i` of a trail belongs to step `i / 4`.

## Step by step: export the site data (local)

```bash
python -m pipeline.export                     # raw/ -> app/data/ (~1 min, ~66 MB)
```

The export replaces the generated files of every configuration that has projections in `raw/`. `app/data/` is
committed: it's what the site (and GitHub Pages) serves.

## Run the site

The site in `app/` is static: HTML, CSS, D3 v7 (copied into `app/js/vendor/`, so it also works offline) and the
JSON in `app/data/`. There is no build step and no server code.

**Locally:** browsers don't let a page loaded from `file://` read its data files, so serve the folder:

```bash
python -m http.server 8000 -d app          # then open http://localhost:8000
```

**With Flask** (same site; Flask only serves the folder):

```bash
pip install flask
python app.py                              # then open http://127.0.0.1:5000
```

**GitHub Pages:** `.github/workflows/pages.yml` publishes `app/` on every push to `main` that changes `app/`,
or on a manual run from the Actions tab.
1. One-time setup on GitHub: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
2. Push. The site appears at `https://andrea-coa.github.io/jamona/`.

The site lives under `/jamona/`, so all paths in the code are relative. Keep it that way.

**Code layout (`app/js/`):**

| File | Role |
|---|---|
| `main.js` | entry point: creates the data store, the UI state and the panels; loads the selected configuration |
| `data.js` | fetches and caches the data files, decodes them into typed arrays (`Index`, `Config`, `Coords`, `Trails`) |
| `state.js` | shared UI state (`VIS_PLAN.md` §3) with change notifications and the reset rules |
| `controls.js` | controls panel (`VIS_PLAN.md` §5.1): dropdowns and sample spinner, two-way bound to the state |
| `ui.js` | shared floating tooltip and ⓘ info icons |
| `encoding.js` | shared encodings: class colors, color-by modes, trail progress colors, animation timing |
| `scatter.js` | embeddings view (`VIS_PLAN.md` §5.4): points, legend, zoom/pan, brush, tooltip, pin, badges, path view, trajectory |
| `paths.js` | trails of the path view: grouped SVG paths, draw-in, morph between bundling strengths, hit testing |
| `metrics.js` | neighborhood hit, sample accuracy, confusion matrix, per-class recall / precision / F1 |
| `steppers.js` | layer / epoch rows inside the plot (draft-v2), Play, speed, All, keyboard shortcuts (`VIS_PLAN.md` §5.5) |
| `architecture.js` | architecture panel with the NH separation bars (`VIS_PLAN.md` §5.2) |
| `curve.js` | learning curve with the current-epoch marker (`VIS_PLAN.md` §5.3) |
| `model-metrics.js` | confusion matrix and per-class bars at the current epoch (`VIS_PLAN.md` §5.7) |
| `images.js` | image view and active-filter chips (`VIS_PLAN.md` §5.6) |
| `datapoint.js` | data point panel: softmax bars and "when was it learned?" (`VIS_PLAN.md` §5.8) |

`data.js`, `state.js` and `metrics.js` use neither D3 nor the DOM, so they can be tested with Node. In the browser console,
`app.store.set({ dataset: "svhn" })` changes the state by hand.

## Site data files (`app/data/`)

All JSON is compact. **Coordinates** are integers `0..9999` in the unit square, with x to the right and y down
(already flipped for SVG). A **flat point array** is `[x0, y0, x1, y1, …]`, indexed by observation id
`0..999` (the position in the subset).

| File | Content |
|---|---|
| `index.json` | `scale`; `datasets[]`: `id`, `name`, `classes`, `sprite` {`file`, `cell` px, `columns`, `count`}, `models[]` {`id`, `name`, `methods` (`tsne`, `pca`), `trails` (`["tsne"]`)} |
| `<ds>/sprites.png` | the 1000 subset images in a grid of `columns` cells of `cell`×`cell` px; image `i` is at column `i % columns`, row `floor(i / columns)` |
| `<ds>/<model>/meta.json` | `classes`, `epochs` (snapshot epochs), `layers[]` {`id`, `label`, `units`, `type`, `dropout`} from Input to Output, `t1Layers`, `t2Layers`, `training` (hyperparameters), `history` {`epoch`, `trainAcc`, `testAcc`, `trainLoss`, `testLoss`} for every epoch 0–50, `metrics` {`testAcc`, `trainAcc`, `paperTestAcc`, `bestEpoch`, `bestTestAcc`} |
| `<ds>/<model>/nodes.json` | `n`, `labels[obs]`, `testIndex[obs]` (index in the original test set), `sampleOrder` (the first 100·k ids hold 10·k per class; identical for both models), `preds[epoch][obs]`, `probs[epoch]` = flat `[obs × 10]` probabilities in per mille (`probScale` = 1000) |
| `<ds>/<model>/coords_<method>.json` | `t1[layer][epoch]` and `t2[epoch][layer]`, each a flat point array. `epoch` indexes `meta.epochs`; `layer` names follow `t1Layers` (keys) / `t2Layers` (order) |
| `<ds>/<model>/trails/t1_<layer>.json`, `t2_e<epoch>.json` | t-SNE only: `strengths` (`medium`, `strong`), `samplesPerSegment` (4), `trails[strength][obs]` = flat point array of the bundled trail; point `i` belongs to step `i / 4` (an epoch index for T1, a layer index for T2). Unbundled trails: join the points of `coords` directly |

## Using the site

| Task | How |
|---|---|
| **T1 (epochs)** | *View by: epoch (T1)*. Pick a layer in the Layer row, press ▶ (or Space) on the Epoch row: points move from the untrained network (epoch 0) to epoch 50. `All` shows every epoch at once as bundled trails. |
| **T2 (layers)** | *View by: layer (T2)*, or click a box in the Architecture panel. ▶ plays Input → hidden layers → Output; the separation bars show the NH of every layer at the current epoch. |
| Filter | Legend or per-class bar (click = isolate, Shift + click = add, double-click = all), confusion-matrix cell, brush (`Brush` or Shift + drag). Filters combine; chips above the images clear them. |
| Details | Hover a point / trail / thumbnail (tooltip with the image); click to pin it: Data point panel, trajectory over all steps. |
| Keyboard | Space play / pause · ← → previous / next step · A path view · Esc unpin, then clear the brush |

## Design rationale

- **Audience and message.** Classmates and the instructor on a projector, who know what a neural network is but not
  t-SNE. Message: *a network learns by pulling same-class observations together, over epochs and layer after layer;
  where classes stay mixed, it makes its mistakes.* Every panel answers one question of `VIS_PLAN.md` §1.
- **Two projections for two tasks.** T1 uses **one joint t-SNE per layer over all snapshot epochs**: every epoch
  lives in the same map, so a point that moves really changed its representation (no projection noise between
  frames; paper Sec. 5.4). T2 projects every layer of an epoch separately, but **every t-SNE starts from the Input
  t-SNE**, which keeps clusters roughly in place across layers and across epochs. PCA (of `log1p` activations,
  Procrustes-aligned) is offered for contrast: it keeps global variance and shows how much t-SNE "cleans up".
- **Quality you can read.** The neighborhood hit (NH, k = 6), the paper's metric, is computed in the browser on the
  points shown, so it follows the sample size. It appears as a badge with its change from the previous step, as the
  separation bars of the architecture panel and in the tooltips of the Layer / Epoch steps. Sample accuracy sits next to it.
- **Encodings.** Hue = true class (`schemeCategory10` = matplotlib tab10, the paper's colors); misclassified points
  are triangles in the **predicted** class color (paper convention), so errors are visible without another panel.
  UI chrome is gray / near-black, so a saturated color always means a class; selection is a black ring. Trails use
  lightness for progress (dark start, pure class color at the end, as in paper Fig. 1). Sequential data use
  single-hue scales: Blues (confusion matrix), viridis ("learned at"). No explanatory text on plots: ⓘ icons.
- **Evolution: animation + bundling.** Stepping animates points between snapshot frames (750 ms, glyph switch at
  mid-transition); Play adds a dwell so each frame can be read, and every linked view (badges, matrix, bars,
  softmax, curve marker, separation bars) tweens in sync. The path view draws all steps at once as trails bundled
  with our own KDE edge bundling (the CPU counterpart of the paper's GPU bundling), **class-aware** so that a bundle
  never mixes hues; trails draw in by segment and morph between bundling strengths.
- **Layout.** The panels follow the team's mockup (`drafts/draft-v2.png`) and nothing scrolls: every chart sizes
  itself to its panel and the image view is paged, so the whole state is visible at once on a projector.
- **Visualization mantra.** *Overview*: path view (all steps at once), learning curve, separation bars.
  *Zoom and filter*: zoom/pan, legend, confusion cells, brush, color-by modes, sample size, bundling strength.
  *Details on demand*: tooltip with the image, pinned Data point panel (softmax, p(true) over epochs, trajectory),
  image view. All views read one shared state, so every interaction is linked.
- **Fair comparisons.** The same 1000 test images (100 per class) are used for every model, epoch and layer; the
  sample spinner shows nested prefixes, balanced at every size. Datasets × models are all selectable.
- **Engineering.** Everything is precomputed (static JSON, 66 MB) so the site runs on GitHub Pages and offline;
  D3 is vendored. Trails are drawn as ~200 SVG paths (one per class × segment) instead of 10 000.

## Findings

Numbers are t-SNE NH on all 1000 points and test accuracy (full test set, `meta.json`); paper values in brackets.

| Config | Input NH | last hidden NH, epoch 0 → 50 | NH by layer, epoch 50 | test acc (paper) | best epoch |
|---|---|---|---|---|---|
| MNIST MLP | 0.85 (0.89) | h4 0.76 (0.84) → 0.96 (0.98) | 0.94 · 0.97 · 0.97 · 0.96 · out 0.98 | 98.4 % (98.5) | 48 |
| MNIST CNN | 0.85 | fc 0.83 → 0.99 | 0.89 · 0.97 · 0.99 · out 1.00 | 99.7 % (99.6) | 36 |
| SVHN MLP | 0.16 | h4 0.15 → 0.79 | 0.45 · 0.64 · 0.77 · 0.79 · out 0.79 | 85.3 % (77.4) | 47 |
| SVHN CNN | 0.16 | fc 0.14 → 0.82 (0.85) | 0.27 · 0.65 · 0.82 · out 0.86 | 90.9 % (93.8) | 14 (92.9 %) |
| CIFAR-10 MLP | 0.19 | h4 0.20 → 0.43 | 0.24 · 0.32 · 0.41 · 0.43 · out 0.44 | 57.1 % (52.9) | 49 |
| CIFAR-10 CNN | 0.19 | fc 0.20 → 0.56 (0.53) | 0.28 · 0.40 · 0.56 · out 0.65 | 74.5 % (79.2) | 21 (76.8 %) |

- **MNIST: easy, and learned in one epoch.** Raw pixels already separate digits (NH 0.85) and the *untrained* MLP
  keeps most of that structure (h4 0.76), as in the paper. In T1, the epoch-0 map is a central blob and epoch 1 is
  almost the final picture: test accuracy goes 9 % → 96 %, and 93 % of the sample is classified correctly from
  epoch 1 to the end. The few errors sit between 3 ↔ 5 and 5 ↔ 8.
- **SVHN: layers do the work.** Raw pixels barely separate classes (NH 0.16): t-SNE of the input is an elongated
  strip ordered by brightness (light vs dark backgrounds). Untrained layers stay at 0.15; after training, separation
  grows **layer by layer** (MLP 0.45 → 0.64 → 0.77 → 0.79; CNN conv1 0.27 → conv2 0.65 → fc 0.82), the clearest T2
  story. T1 is gradual for the MLP (h4 NH 0.48 at epoch 1 → 0.78), which makes it the best Play demo.
- **CIFAR-10: a confusion zone.** Even the best layer separates classes only partly (MLP 0.43, CNN fc 0.56); the
  center of the map stays mixed. Cat ↔ dog is the most confused pair for both models (39 and 35 of 1000), then
  airplane ↔ ship and bird ↔ deer. Color by *learned at* shows 43 % of the MLP sample (27 % for the CNN) is never
  learned.
- **Overfitting is visible.** The CNNs peak early (SVHN epoch 14, CIFAR-10 epoch 21) and then lose test accuracy while
  train accuracy keeps rising (CIFAR-10: 90 % train vs 74.5 % test). In T1, CIFAR-10 CNN `fc` NH peaks at epoch 25
  (0.63) and drops to 0.57 at epoch 50; at epoch 25, 8 of 10 per-class recall bars sit above their "final" marks.
- **Across models.** The CNN beats the MLP in the last hidden layer on every dataset, and its first block changes
  least (conv1 is only 0.04–0.11 NH above the raw input): low-level features are generic, class separation happens deeper. At epoch 0
  the Output layer has the lowest NH of all (0.11–0.40): an untrained softmax mixes everything.
- **Deviation from the paper:** SVHN / CIFAR-10 MLPs use lr 0.001 (lr 0.01 diverged), which explains their higher
  accuracy than the paper's MLPs; 50 epochs instead of 100.

## Submission ZIP

```bash
python package.py                          # -> dist/jamona.zip (site + data files + Flask app + code + docs)
```

The ZIP contains `app.py` (Flask), `app/` (site and its data files: `nodes.json` = observations, trails = edges),
`pipeline/`, the notebook and the docs. It leaves out `raw/`, datasets, the paper PDF and virtual environments.
