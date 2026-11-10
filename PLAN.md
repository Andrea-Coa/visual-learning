# Visual Learning — Implementation Plan

High-level plan for the DS5343 "Visual Learning Models" assignment (see `task-description.md`),
based on Rauber et al., *Visualizing the Hidden Activity of Artificial Neural Networks* (`paper/`).

## How to use this file

- Status markers: `[ ]` not started · `[~]` in progress · `[x]` done · `[-]` dropped (say why).
- When you finish (or start) an item, update its marker and, if useful, add a short note under it
  (`> note: ...`) with anything the next person must know (file paths, gotchas, deviations).
- Append a line to the **Progress log** at the bottom for every working session.
- If you change a decision in **Key decisions**, edit it there and log why.

---

## Key decisions

| Topic | Decision | Rationale |
|---|---|---|
| Datasets | **MNIST, SVHN, CIFAR-10** (all three) | Grade bonus for >1 dataset; shows easy → hard progression, as in the paper. |
| Models | **MLP** (4 ReLU hidden layers × 1000, dropout 0.2→0.5) and **CNN** (paper architecture; recorded hidden layers = flattened output of both conv blocks (`conv1`, `conv2`) + the 512-unit FC layer (`fc`)) | Same architectures as the paper (Sec. 4), so results are comparable. MLP has equal-width layers → clean inter-layer comparison. `conv2` is the paper's "4096 (or 3136)" layer. |
| Framework | PyTorch + torchvision, trained on **Colab T4** | Free GPU; torchvision ships all three datasets. |
| Optimizer | SGD, momentum 0.9, lr 0.01 (batch 16 MLP / 32 CNN), as in paper; **lr 0.001 for SVHN/CIFAR-10 MLP** | Slower than Adam → more visible evolution between epochs. lr 0.01 made the 3072-input MLPs diverge (2026-11-09). |
| Observation subset | **1000 test images, stratified (100/class), fixed seed**, same subset for every epoch/layer | Task suggests 1000; fixed subset is required to link points across epochs/layers. |
| Epochs & snapshots | **50 epochs**; snapshots at 0,1,2,3,5,8,12,18,25,35,50 (0 = untrained) — `pipeline/config.py` | Paper used 100; 50 keeps Colab time ~1 h. Largest changes happen in the first epochs (paper Sec. 5.4). |
| Projection | PCA → 50 dims, then **t-SNE** (`openTSNE`) | Paper uses (Barnes-Hut) t-SNE; openTSNE supports custom initialization, needed below. |
| T1 (epochs) projection | **One joint t-SNE per layer** (every hidden layer + Output) over the union of all epoch snapshots (paper Sec. 5.4, inter-epoch) | Single projection ⇒ no spurious frame-to-frame motion. Every layer is needed by the UI (`VIS_PLAN.md` D1). |
| T2 (layers) projection | **For every snapshot epoch**, layers Input → hidden → Output; t-SNE of the Input (network input pixels, same at every epoch) is the **initialization** of every other t-SNE (paper Sec. 5.4, inter-layer) | Shared init keeps clusters roughly in place across layers *and* across epochs. |
| PCA alternative | PCA(2) of **log1p(activations)**, one PCA per frame (T1 and T2), each frame Procrustes-aligned to the previous one | "Dim. red." dropdown (`VIS_PLAN.md` D4). Joint T1 PCA collapsed early epochs to a dot (activation scale grows in training); raw ReLU activations crowd most points into the center (log1p: higher NH, less crowding in 4/4 tested cases). |
| Evolution view | **Trail bundling** precomputed in Python (KDE edge bundling, own implementation), **t-SNE only, strengths medium + strong**; "none" drawn from node coordinates; + **animated point transitions** in D3 | Paper uses bundling (CUBu); the assignment additionally *requires* animation. Storing only 2 bundled levels for t-SNE keeps the data ~50 MB. |
| Quality metric | **Neighborhood hit** (k = 6): computed **in the browser** on the visible sample; also in Python on all 1000 points (report / sanity check) | Same as paper. Browser computation follows the Sample-size control. |
| Frontend | **D3 v7 only** (no other vis libs), served by **Flask** | Assignment rule. |

Note: the assignment's T1/T2 are *not* the paper's T1/T2. Here **T1 = inter-epoch**, **T2 = inter-layer**
(both are the paper's Sec. 5.4). The paper's neuron projections (its T2, Sec. 6) are out of scope (optional stretch).

---

## Proposed layout

```
visual-learning/
├── PLAN.md, CLAUDE.md, README.md, requirements.txt
├── pipeline/                  # Python: training + processing (runs on Colab or locally)
│   ├── config.py              # datasets, models, epochs to snapshot, seeds, paths
│   ├── data.py                # torchvision loaders + fixed stratified subset
│   ├── models.py              # MLP, CNN with forward hooks returning hidden activations
│   ├── train.py               # training loop; saves activations/predictions per snapshot
│   ├── project.py             # PCA + t-SNE (T1 joint, T2 init-chained), neighborhood hit
│   ├── bundle.py              # KDE edge bundling of trails
│   └── export.py              # writes app/data/** JSON + image sprite sheets
├── notebooks/colab_train.ipynb  # thin wrapper: clone/upload repo, run pipeline on T4
├── raw/                       # (git-ignored) .npz activations from Colab
└── app/                       # Flask deliverable
    ├── app.py
    ├── templates/index.html
    ├── static/css/style.css
    ├── static/js/             # main.js, scatter.js, trails.js, controls.js, ...
    └── data/<dataset>/<model>/  # nodes + edges files (deliverable)
```

---

## Phase 0 — Setup

- [x] 0.1 Create folder skeleton above; add `.gitignore` (`.venv/`, `raw/`, `__pycache__/`).
  > note: `/data/` and `/raw/` are root-anchored in `.gitignore` so `app/data/` (deliverable) is NOT ignored.
- [x] 0.2 Fill `requirements.txt` (local): `torch`, `torchvision`, `numpy`, `scikit-learn`, `openTSNE`, `scipy`, `pillow`, `flask`.
  > note: installed in `.venv` (Python 3.13, torch 2.15 CPU, openTSNE 1.0.4) without issues.
- [x] 0.3 Colab notebook skeleton that installs deps, mounts Drive, and runs `pipeline/train.py`.
  > note: `notebooks/colab_train.ipynb`; no installs needed (Colab ships torch/torchvision/scipy). Not yet run on Colab.
  > The notebook clones/pulls the GitHub repo (code) and writes outputs to Drive `MyDrive/visual-learning/raw/`. Push code changes before training.

## Phase 1 — Data & models

- [x] 1.1 `data.py`: load MNIST / SVHN / CIFAR-10 (train + test), normalize, flatten for MLP.
  > note: whole dataset kept as uint8 on the GPU, normalized per batch (per-channel mean/std of train set); no DataLoader.
- [x] 1.2 Fixed stratified subset of 1000 test images per dataset (seeded); save indices.
  > note: `raw/<ds>/subset.npz` also stores the images (for sprites in 5.2) and class names.
- [x] 1.3 `models.py`: MLP and CNN as in paper Sec. 4; expose hidden-layer activations by name
      (MLP: `h1..h4`; CNN: `conv1`, `conv2` (flattened block outputs), `fc`).
- [x] 1.4 Weight init: Glorot-uniform, biases 0 (paper Sec. 4).

## Phase 2 — Training & activation extraction (Colab T4)

- [x] 2.1 `train.py`: SGD loop with per-epoch train/test loss & accuracy logging.
  > note: history uses eval-mode metrics on the full train/test sets (+ running dropout-on metrics).
  > Resumable: per-epoch `checkpoint.pt`; finished configs (with `metrics.json`) are skipped. Verified locally (CPU smoke test + simulated interruption).
- [x] 2.2 At each snapshot epoch (incl. epoch 0), run the subset through the model in eval mode and save
      all hidden activations (float16), predicted labels, and softmax confidence.
  > note: one file per snapshot: `raw/<ds>/<model>/snapshots/epoch_XXX.npz`. Final metrics (confusion matrix,
  > per-class P/R/F1, train/test gap, paper accuracy) in `metrics.json`; weights in `model_final.pt`. Layout in README.
- [x] 2.3 Train all 6 configs (3 datasets × 2 models) on Colab; compare final test accuracy with paper Table 1.
  > note: final test acc (paper): MNIST MLP 98.42 (98.52) · MNIST CNN 99.65 (99.62) · SVHN MLP 85.33 (77.38) ·
  > SVHN CNN 90.93 (93.76) · CIFAR MLP 57.07 (52.91) · CIFAR CNN 74.54 (79.19). 5–10 min each on T4.
  > MNIST is ~96–98% already at epoch 1 (little movement after); SVHN/CIFAR MLP improve gradually.
  > SVHN CNN peaks at epoch 14 (92.9%) and CIFAR CNN at epoch 21 (76.8%), then overfit (test loss rises).
  > note: steps + time estimate in README ("Step by step: training"). Can be split across Colab accounts by dataset.
  > note: done: MNIST MLP 98.42% (paper 98.52), MNIST CNN 99.65% (paper 99.62), ~6–8 min each on T4.
  > note: SVHN MLP diverged with lr 0.01 (NaN at epoch 14). SVHN/CIFAR-10 MLP now use lr 0.001 (`TRAINING_OVERRIDES`
  > in config.py), checked locally on CPU for a few epochs. Before retraining, delete Drive `raw/svhn/mlp/` (diverged checkpoint).
  > A diverged config no longer stops the run: it is reported and the others continue.
- [x] 2.4 Download `raw/` from Drive into the local project root (merge folders if several accounts were used).
  > note: `raw/` is 392 MB locally, all 6 configs complete (11 snapshots each, no NaNs, no leftover checkpoints).

## Phase 3 — Projections & metrics

- [x] 3.1 `project.py`: PCA(50) + openTSNE helper with fixed seed.
  > note: `python -m pipeline.project [--dataset ..] [--model ..] [--method tsne pca] [--force]` →
  > `raw/<ds>/<model>/proj_<method>.npz` (arrays `t1` [layer, epoch, obs, 2], `t2` [epoch, layer, obs, 2], `nh_t1`, `nh_t2`, ...).
  > Skips existing outputs. Time (local CPU, 8 cores): PCA ~10 s/config; t-SNE ~15 min/config (T1 joint of 11 k points ~2.5 min/layer).
- [x] 3.2 **T1 (epochs):** for each config and **every layer** (hidden + Output), stack all snapshots →
      one joint t-SNE → split back into per-epoch coordinates. PCA: per-epoch frames, Procrustes-chained.
  > note: all 6 configs × {t-SNE, PCA} done (12 `proj_*.npz`). CIFAR-10 CNN `fc` NH peaks at epoch 25 (0.63) and drops
  > to 0.57 at 50: the overfitting is visible in T1.
- [x] 3.3 **T2 (layers):** for each config and **every snapshot epoch**, project Input → hidden → Output;
      every t-SNE is initialized from the Input t-SNE (PCA: each layer Procrustes-aligned to the previous one).
  > note: t-SNE NH of the last hidden layer, untrained → epoch 50 (paper, trained): MNIST MLP 0.76 → 0.96 (0.98),
  > MNIST CNN 0.83 → 0.99, SVHN MLP 0.15 → 0.79 (0.67), SVHN CNN 0.14 → 0.82 (0.85), CIFAR-10 MLP 0.20 → 0.43,
  > CIFAR-10 CNN 0.20 → 0.56 (0.53). SVHN raw-pixel t-SNE is an elongated strip (brightness-dominated).
- [x] 3.4 Normalize all coordinates to a common [0,1] box per view (T1 t-SNE: per layer over all epochs; else per frame).
- [x] 3.5 Neighborhood hit (k=6) for every projection (all 1000 points); the Input frame is the raw-pixel baseline (paper Fig. 2).
  > note: MNIST MLP t-SNE: input 0.85, h4 untrained 0.76 → trained 0.96 (paper: 0.89 / 0.84 / 0.98).
- [x] 3.6 Quick matplotlib sanity check of a few projections (not a deliverable).
  > note: plotting script lives outside the repo (scratch). matplotlib installed in `.venv` only for this (not in requirements).
  > T1 t-SNE on MNIST: epoch 0 is a central blob (as in the paper), epochs ≥1 are nearly identical tight clusters.

## Phase 4 — Trail bundling

- [ ] 4.1 `bundle.py`: build one polyline per observation through its sequence of positions
      (epochs for T1, layers for T2); resample to ~4 points per segment.
- [ ] 4.2 KDE edge bundling: density map of trail points → move points up the density gradient →
      smooth → repeat (a few iterations); keep endpoints fixed.
- [ ] 4.3 Two bundling strengths (medium, strong), **t-SNE only**; "none" = straight polylines from node coordinates in the UI.
- [ ] 4.4 Quantize coordinates to integers 0–9999 to keep JSON small (~50 MB total for all views).

## Phase 5 — Export (deliverable data files)

- [ ] 5.1 Define the JSON schema (document it in README). One file per view, loaded on demand by the UI:
  - `meta.json` — dataset, model, classes & colors, architecture (names, widths, types, dropout), epochs,
    learning curves (`history.json`), final metrics, nested sample order (`VIS_PLAN.md` D10).
  - `nodes.json` — per observation: id, true label, per-epoch `pred` and softmax `probs`.
  - `<method>/t1_<layer>.json` — coords [epoch][obs]; `<method>/t2_e<epoch>.json` — coords [layer][obs].
  - `tsne/t1_<layer>_trails.json`, `tsne/t2_e<epoch>_trails.json` — bundled polylines per strength.
- [ ] 5.2 Image sprite sheet (`sprites.png`, 1000 thumbnails) per dataset for hover details.
- [ ] 5.3 Write everything to `app/data/<dataset>/<model>/`.

## Phase 6 — Flask app

- [ ] 6.1 `app.py`: index route + static serving of `app/data/**`; a `/api/configs` route listing available configs.
- [ ] 6.2 `index.html` layout: header with dataset/model selectors; two tabs/panels **T1 Epochs** and **T2 Layers**.

## Phase 7 — D3 visualizations

> note: **Full UI design spec: `VIS_PLAN.md`** (based on `drafts/draft-v2.png`); its §11 lists the data the design
> needs. Feasibility review done 2026-11-09: all feasible, resolutions written into §11 and Phases 3–5 above.
> Where 7.x below and `VIS_PLAN.md` differ, `VIS_PLAN.md` wins. No explanatory text on plots (ⓘ icons instead);
> style references in `style-references/`.
> note: UI layout mockups in `drafts/` (latest: `draft-v2.png`, reviewed 2026-11-09). Panels: controls,
> architecture (click layer → layer view), learning curve, embeddings view (+ layer/epoch steppers, play), image view
> (brush-linked), model metrics (per-epoch confusion matrix, per-class bars), data-point panel (softmax probs).

Shared scatter component: points colored by true class (categorical palette, 10 classes),
misclassified points drawn as **triangles colored by predicted class** (paper convention).

- [ ] 7.1 **T1 — Inter-epoch view**
  - Main scatter of the selected epoch; **animated transitions** (`d3.transition`) when the epoch changes.
  - Epoch slider + Play/Pause (animation through all snapshots).
  - Trail layer: bundled trails, hue = class, lightness = epoch (dark = early, bright = late), toggle on/off.
  - Linked **learning curve** (train/test accuracy & loss vs epoch) with the current epoch marked; clicking the curve jumps to that epoch.
  - Small multiples of a few epochs (overview).
- [ ] 7.2 **T2 — Inter-layer view**
  - Same structure, with a layer slider instead of epochs ("forward propagation" animation).
  - Small multiples `A_p[1] … A_p[L]` beside the bundled trail image (as in Figure 1).
  - Before/after-training toggle (epoch 0 vs final).
- [ ] 7.3 **Interaction** (mandatory; follows the mantra)
  - *Overview:* small multiples + bundled trails.
  - *Zoom & filter:* `d3.zoom` pan/zoom; click legend to filter/isolate classes; brush to select a group;
    "show only misclassified" toggle; bundling-strength selector.
  - *Details on demand:* `mouseover` tooltip with image thumbnail, true/predicted label, confidence;
    `click` a point to pin it and highlight its full trajectory across epochs/layers in all views.
- [ ] 7.4 NH and accuracy shown next to each projection.
- [ ] 7.5 Visual polish: consistent color palette, legends, axis-free scatter, readable at projector resolution.

## Phase 8 — Documentation & delivery

- [ ] 8.1 `README.md`: how to run (Colab + local Flask), data schema, **design rationale** (why t-SNE,
      joint vs chained projection, bundling + animation, color/glyph encodings, interaction mapping to mantra).
- [ ] 8.2 Findings write-up per dataset (compare with paper: untrained MLP already separates MNIST; later layers
      separate better; SVHN light/dark-background split; CIFAR-10 confusion zone; early epochs at center of T1).
- [ ] 8.3 Test the app end-to-end on a clean venv (`pip install -r requirements.txt && python app/app.py`).
- [ ] 8.4 Build the submission ZIP (Flask app + data files; exclude `.venv`, `raw/`).
- [ ] 8.5 Prepare the in-class presentation (live demo script + 3–5 key insights).

## Stretch (only if time allows)

- [-] S.1 T1 layer selector (joint epoch projection for every hidden layer, not only the last).
  > note: moved into the main scope (3.2), required by `VIS_PLAN.md`.
- [ ] S.2 Neuron projections / discriminative neuron map (paper Sec. 6).
- [ ] S.3 Train-subset vs test-subset comparison (paper Figs. 7–8).
  > note: only test activations were recorded; train activations can be extracted for the final epoch only (`model_final.pt`).

---

## Progress log

| Date | Who | What |
|---|---|---|
| 2026-11-09 | Claude | Read task + paper; wrote this plan. |
| 2026-11-09 | Claude | Phase 0 done; Phase 1 + training code (2.1, 2.2) implemented and smoke-tested on CPU. README with Colab steps and time estimates. Next: run 2.3 on Colab. |
| 2026-11-09 | Claude | Colab: MNIST MLP/CNN trained OK; SVHN MLP diverged. Fixed with lr 0.001 for SVHN/CIFAR-10 MLP (tested on CPU); divergence no longer aborts the whole run. |
| 2026-11-09 | Claude | All 6 configs trained (user, Colab); `raw/` verified. Reviewed `VIS_PLAN.md` against the training outputs: compatible, with open points on output size, NH definition and demo order. |
| 2026-11-09 | Claude | Reviewed UI drafts v1 → v2 (`drafts/`); feedback on missing pieces given in chat. |
| 2026-11-09 | Claude | Wrote `VIS_PLAN.md` (visualization design spec from draft-v2 + review); awaiting feasibility review by code agent. |
| 2026-11-09 | Claude | Feasibility review written into `VIS_PLAN.md` §11 (+ §6, §10, ⓘ-icon rule) and Phases 3–5. Phase 3 done: `pipeline/project.py`, t-SNE + PCA for all 6 configs, NH, sanity plots. PCA switched to per-frame log1p + Procrustes chain after visual checks. Next: Phase 4 (bundling). |
