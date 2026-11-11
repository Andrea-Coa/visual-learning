# JaMoNA — Implementation Plan

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
| Evolution view | **Trail bundling** precomputed in Python (KDE edge bundling, own implementation, **class-aware**), **t-SNE only, strengths medium + strong**; "none" drawn from node coordinates; + **animated point transitions** in D3 | Paper uses bundling (CUBu); the assignment additionally *requires* animation. Storing only 2 bundled levels for t-SNE keeps the data ~50 MB. |
| Quality metric | **Neighborhood hit** (k = 6): computed **in the browser** on the visible sample; also in Python on all 1000 points (report / sanity check) | Same as paper. Browser computation follows the Sample-size control. |
| Frontend | **Static site**: plain HTML/CSS/JS (ES modules, no build step) + **D3 v7 only** (vendored, pinned), reading precomputed JSON; deployed on **GitHub Pages**; `app.py` = small Flask wrapper serving `app/` | D3-only is an assignment rule. Flask as a backend dropped (user decision, 2026-11-09): all data is precomputed, so a server adds nothing, and a static site can be shared as a link. The wrapper (8.5) was added on 2026-11-10 because the assignment asks for "a Flask application" in the ZIP. |

Note: the assignment's T1/T2 are *not* the paper's T1/T2. Here **T1 = inter-epoch**, **T2 = inter-layer**
(both are the paper's Sec. 5.4). The paper's neuron projections (its T2, Sec. 6) are out of scope (optional stretch).

---

## Proposed layout

```
jamona/
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
├── .github/workflows/pages.yml  # deploys app/ to GitHub Pages
└── app/                       # static site = the deliverable (committed, data included)
    ├── index.html
    ├── css/style.css
    ├── js/                    # main.js (state + wiring), one module per panel, vendor/d3.v7.min.js
    └── data/                  # index.json + <dataset>/<model>/** (nodes + edges files), <dataset>/sprites.png
```

---

## Phase 0 — Setup

- [x] 0.1 Create folder skeleton above; add `.gitignore` (`.venv/`, `raw/`, `__pycache__/`).
  > note: `/data/` and `/raw/` are root-anchored in `.gitignore` so `app/data/` (deliverable) is NOT ignored.
- [x] 0.2 Fill `requirements.txt` (local): `torch`, `torchvision`, `numpy`, `scikit-learn`, `openTSNE`, `scipy`, `pillow`, `flask`.
  > note: `flask` is now optional (only for the 8.5 wrapper); the site itself is static.
  > note: installed in `.venv` (Python 3.13, torch 2.15 CPU, openTSNE 1.0.4) without issues.
- [x] 0.3 Colab notebook skeleton that installs deps, mounts Drive, and runs `pipeline/train.py`.
  > note: `notebooks/colab_train.ipynb`; no installs needed (Colab ships torch/torchvision/scipy). Not yet run on Colab.
  > The notebook clones/pulls the GitHub repo (code) and writes outputs to Drive `MyDrive/jamona/raw/`. Push code changes before training.

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

- [x] 4.1 `bundle.py`: build one polyline per observation through its sequence of positions
      (epochs for T1, layers for T2); resample to ~4 points per segment.
  > note: `python -m pipeline.bundle [--dataset ..] [--model ..] [--force]` → `raw/<ds>/<model>/trails_tsne.npz`:
  > `t1` uint16 [layer, strength, obs, 41, 2], `t2` uint16 [epoch, strength, obs, 17|21, 2]. Sample i belongs to
  > step i / 4 (use it for the dark→bright progress color). ~20 s per config.
- [x] 4.2 KDE edge bundling: density map of trail points → move points up the density gradient →
      smooth → repeat (a few iterations); keep endpoints fixed.
  > note: **class-aware** (each trail attracted only by its own class's density): global bundling merged different
  > classes into shared highways, misleading with hue = class. Samples weighted by segment length. Endpoints verified exact.
- [x] 4.3 Two bundling strengths (medium, strong), **t-SNE only**; "none" = straight polylines from node coordinates in the UI.
  > note: medium = 6 iterations, strong = 24 (14 looked almost identical to medium).
- [x] 4.4 Quantize coordinates to integers 0–9999 to keep JSON small (~50 MB total for all views).
  > note: done in `bundle.py` (uint16). All trails ≈ 9.4 M integers ≈ 47 MB as JSON text.

## Phase 5 — Export (deliverable data files)

- [x] 5.1 Define the JSON schema (document it in README):
  > note: schema in README "Site data files". y is flipped at export (SVG y-down); coordinates int 0–9999;
  > probabilities per mille. `sampleOrder` is the same for MLP and CNN of a dataset.
  - `app/data/index.json` — available datasets × models × projections, sprite info (replaces a server route).
  - `meta.json` — dataset, model, classes, architecture (names, widths, types, dropout), epochs,
    learning curves (`history.json`), final metrics.
  - `nodes.json` — per observation: true label, test-set index, nested sample order (`VIS_PLAN.md` D10),
    per-epoch `pred` and softmax `probs`.
  - `coords_<method>.json` — all T1 and T2 frames of one configuration (needed together: filmstrip, separation bars).
  - `trails/t1_<layer>.json`, `trails/t2_e<epoch>.json` — bundled t-SNE polylines per strength, loaded on demand.
- [x] 5.2 Image sprite sheet (`sprites.png`, 1000 thumbnails) per dataset for hover details.
  > note: 40 columns, native size (MNIST 28 px grayscale, others 32 px RGB), ≤ 2.2 MB.
- [x] 5.3 Write everything to `app/data/<dataset>/<model>/`.
  > note: `python -m pipeline.export` (~1 min). 118 JSON + 3 PNG, **66 MB total**, largest file 2.2 MB (sprites) /
  > 1.2 MB (coords). Verified: all JSON parse, coords and trail endpoints match `raw/`, preds match snapshots,
  > probability sums 997–1003 ‰, sample-order prefixes balanced.
  > note: re-verified 2026-11-10 from `app/data/` alone: 154 of 744 k trail-endpoint coordinates differ from
  > `coords_tsne.json` by 1 unit (of 9999, sub-pixel) because of float rounding at quantization. Harmless; don't "fix".

## Phase 6 — Static site skeleton & deployment

- [x] 6.1 `app/index.html` + `css/style.css`: three-column layout of `VIS_PLAN.md` §2 with empty panels.
  > note: panel style after `style-references/` (VISTA-like: white panels, thin borders, small uppercase titles with a
  > square glyph). CSS tokens on `:root`; `.info` class for ⓘ icons; `.loading`/`.error` states. Model metrics and
  > Data point are `<details>` (collapsible without JS). Checked with headless Edge at 1920×1080 and 1150 px (stacked).
- [x] 6.2 Vendor D3 v7 (`app/js/vendor/d3.v7.9.0.min.js`, pinned 7.9.0) so the in-class demo also works offline.
  > note: loaded as a classic script → global `d3` in the modules.
- [x] 6.3 `js/data.js`: loads `index.json` and the per-view files with `fetch` (relative URLs, cached per view).
  > note: decodes into typed arrays once: `Index` (+ `sprite()` CSS helper), `Config` (`sample(n)`, `pred`, `probsOf`,
  > `epochIndex`, `lastHiddenLayer`), `Coords` (`frame(mode, …)`, `sequence(mode, …)`), `Trails` (`trail`, `stepOf`).
  > All 6 configs load + decode in < 1 s. Tested in Node against a local server (scratch test, not in repo).
- [x] 6.4 `js/state.js`: the shared state + change notifications of `VIS_PLAN.md` §3 (single source of truth).
  > note: `store.set(patch)`, `store.subscribe(fn, keys)`, `store.toggleInSet(key, v, {exclusive})`. `epoch` holds the
  > epoch *value*; `layer`/`epoch` null = not set yet (main.js fills last hidden layer / last epoch after loading).
  > Reset rules: dataset → selections + classFilter + layer; model → selections + layer; projection → selections.
- [x] 6.5 Local run: `python -m http.server -d app` (needed because `fetch` does not work from `file://`).
  > note: documented in README "Run the site". `main.js` = entry point (loads the selected config, generated title,
  > loading/error states); `window.app` exposes `{data, store, current}` for the console.
- [~] 6.6 GitHub Pages: `.github/workflows/pages.yml` uploads `app/` (repo setting: Pages → Source: GitHub Actions).
  > note: Pages serves the site under `/jamona/`, so every URL in the code must be relative.
  > note: workflow written (triggers on pushes to `main` touching `app/`, or manually). Not run yet: needs the one-time
  > repo setting + a push by the user.

## Phase 7 — D3 visualizations

> note: **Full UI design spec: `VIS_PLAN.md`** (based on `drafts/draft-v2.png`); its §11 lists the data the design
> needs. Feasibility review done 2026-11-09: all feasible, resolutions written into §11 and Phases 3–5 above.
> Where 7.x below and `VIS_PLAN.md` differ, `VIS_PLAN.md` wins. No explanatory text on plots (ⓘ icons instead);
> style references in `style-references/`.
> note: UI layout mockups in `drafts/` (latest: `draft-v2.png`, reviewed 2026-11-09). Panels: controls,
> architecture (click layer → layer view), learning curve, embeddings view (+ layer/epoch steppers, play), image view
> (brush-linked), model metrics (per-epoch confusion matrix, per-class bars), data-point panel (softmax probs).

One item per panel, in build order; § = section of `VIS_PLAN.md`, which holds the details. [M] items first,
[S] items after the [M] items of all panels work.

- [x] 7.1 Controls panel (§5.1): dataset, dim. red., ANN, view by, sample size, color by, bundling.
  > note: `js/controls.js` (`initControls`), shared tooltip + ⓘ icon helper in `js/ui.js` (reuse them in later panels).
  > Controls only write state; `colorBy` / `bundling` have no visual effect until 7.4 / 7.8. Bundling is disabled unless
  > `showAll` and the projection has trails. Changing dataset keeps model/projection valid (falls back to the first
  > available). Tested in headless Chrome (interaction test in scratch, not in repo).
- [x] 7.2 Embeddings view, scatter mode (§5.4, §4.1–4.2): circles/triangles, title, legend filter, zoom/pan,
      brush, tooltip with thumbnail, pin; NH + sample-acc badges computed in the browser.
  > note: `js/scatter.js` (`initScatter` → `setConfiguration`, called by main.js after loading), NH / accuracy in
  > `js/metrics.js` (browser NH = Python NH: MNIST MLP T2 h4 e50 0.96, Input 0.85). Already animates §6 for step change
  > (750 ms move, glyph switch at mid), dataset/model/projection/mode swap (fade), sample size (grow/shrink), filters
  > (opacity). Brush = own `d3.drag` rectangle (d3.brush's Shift = axis lock clashed with Shift+drag). In T1 the Input
  > layer shows the T2 Input frame (no T1 frames). Esc clears pin, then brush. Loading message is now an overlay.
  > note (testing): headless Chrome with `--virtual-time-budget` never fires `requestAnimationFrame` inside an iframe,
  > so D3 transitions hang there. Test pages must load the app in the same window (`<base href>`) with an rAF→setTimeout
  > shim before D3.
- [x] 7.3 Steppers + animation (§5.5, §6): layer/epoch rows, Play/speed on the active row, animated transitions;
      then the canvas filmstrip [S].
  > note: `js/steppers.js`. Active row = canvas filmstrip (64 px → 48 px → plain buttons as width shrinks; hover = NH
  > tooltip, class filter dims thumbnails) + Play / All / speed; epoch filmstrip has a compact proportional time axis
  > (leader lines to the true epoch). Play: first step immediately, then every (750 + 450) / speed ms; stops after the
  > last step; ▶ at the last step restarts; any manual step change or dataset/model/projection/mode change pauses.
  > Scatter motion and badge tweens are divided by `speed` while playing. Keyboard: Space, ← / →, A (Esc in scatter).
  > Switching to T1 while on Input moves to the first hidden layer. Shared helpers: `frameOf` (data.js),
  > `cachedNeighborhoodHit` (metrics.js). Legend now starts outside the square plot when there is side room.
  > Not done: the ▶/All "slide" on mode switch (rows are rebuilt instantly); comet tail is 7.8.
  > note (2026-11-10, later): reorganized to follow draft-v2: the steppers are now two compact rows overlaid at the
  > bottom-right of the plot; the canvas filmstrip and the time axis were removed (NH per step is in the tooltip).
  > note (testing): ResizeObserver also doesn't fire under `--virtual-time-budget` after load; test widths with
  > `--window-size` instead.
- [x] 7.4 Path view (§4.3, §5.4): trails at none/medium/strong, draw-in animation, pinned unbundled trajectory.
  > note: `js/paths.js` (`PathLayer`) + path code in `scatter.js`. Trails grouped into one SVG path per
  > (class, segment, filter state) ≈ 200 elements; "none" = straight lines resampled to 4 samples / segment, so all
  > strengths morph (600 ms). Draw-in = segment-by-segment fade, staggered by class (not stroke-dash). Hover = nearest
  > trail sample (quadtree); brush uses trail ends; pinned trajectory (unbundled, dot + label per step) also in
  > scatter mode (`trajectory` state, Data point checkbox). Trail opacity uses `stroke-opacity` / `fill-opacity`
  > attributes, not CSS opacity (no compositing layer per path).
- [x] 7.5 Learning curve (§5.3) and architecture panel (§5.2, click → T2; separation bars [S]).
  > note: `js/curve.js` ([Acc | Loss] in the panel title = `curve` state), `js/architecture.js` (bars = T2 NH of
  > every layer at the current epoch; in T2 path view a gray dark→light stripe is the trail-lightness legend).
- [x] 7.6 Model metrics (§5.7): per-epoch confusion matrix with cell filter, per-class bars.
  > note: `js/model-metrics.js`; hover cell = `cellHover` state (temporary highlight in the scatter), click = `cellFilter`;
  > hovered / pinned point's cell outlined; ghost = thin mark at the final-epoch value; `metric` state for the dropdown.
- [x] 7.7 Image view (§5.6) and data-point panel (§5.8), both linked to the scatter.
  > note: `js/images.js` (images = sample ∩ all filters, header names the top-priority source; order fixed until the
  > source / sort changes, badges update per epoch), `js/datapoint.js`. `Config.learnedAt` added in data.js.
- [x] 7.8 [S] items of §12 (color-by modes, ghost bars, learned-at chart, keyboard, filter chips, …).
  > note: color-by class / correctness / learned at (legend switches; never learned = hollow triangle), comet tail of
  > the pinned point during Play, filter chips in the image view. Shared encodings in `js/encoding.js`.
  > Not done (minor): the ▶/All "slide" on mode switch and the filmstrip cross-fade on dataset change.
- [x] 7.9 Polish against `style-references/`; projector check (1920×1080); ⓘ icons instead of text on plots;
      empty/loading states (§9).
  > note: screenshots at 1920×1080 and 1280×720; controls never shrink (left column: architecture scrolls, curve
  > shrinks to 160 px). 4 browser suites in scratch (scatter 29, steppers 33, paths 27, panels 39 checks) all pass.
  > note (testing): headless Chrome screenshots under `--virtual-time-budget` miss SVG changes made after the last
  > composited frame (the DOM is right). Clone the SVGs (`el.replaceWith(el.cloneNode(true))`) before the shot.

## Phase 8 — Documentation & delivery

- [x] 8.1 `README.md`: link to the GitHub Pages site, how to run (Colab training, local pipeline, local server),
      data schema, **design rationale** (why t-SNE, joint vs chained projection, bundling + animation,
      color/glyph encodings, interaction mapping to mantra).
  > note: sections "Using the site", "Design rationale", "Findings", "Submission ZIP"; code table lists every module.
- [x] 8.2 Findings write-up per dataset (compare with paper: untrained MLP already separates MNIST; later layers
      separate better; SVHN light/dark-background split; CIFAR-10 confusion zone; early epochs at center of T1).
  > note: in README "Findings" (table + 6 bullets); numbers = browser NH on all 1000 points (same `metrics.js`).
- [~] 8.3 Test end-to-end: deployed Pages site + local `python -m http.server -d app`, in Chrome and Firefox.
  > note: done: local http.server + Flask wrapper (also from the extracted ZIP), Chrome headless, all 6 configs ×
  > 2 methods. Pending (user): Pages deploy (6.6) and Firefox / Safari by hand (not installed / not scriptable here).
- [x] 8.4 Build the submission ZIP (`app/` = site + data files; exclude `.venv`, `raw/`).
  > note: `python package.py` → `dist/jamona.zip` (28 MB, git-ignored). Rebuild after any change.
  > note: 2026-11-10: `drafts/` added to the ZIP (user request) → 159 files, 28.8 MB. Rebuilt from commit 7e395bd and
  > tested from a clean extraction: `python app.py` serves every file type (200, correct MIME), full UI renders.
- [x] 8.5 *(optional)* `app.py`: ~10-line Flask wrapper serving `app/` as static files, only if the course insists
      on a Flask app in the ZIP.
  > note: done (the assignment text asks for a Flask application). Tested: MIME types OK (`text/javascript` for modules).
  > `.venv` on the macOS machine only has Flask (Python 3.9); the pipeline needs the full `requirements.txt`.
- [x] 8.6 Prepare the in-class presentation (live demo script + 3–5 key insights; keep a local copy for offline use).
  > note: `PRESENTATION.md` (6-step demo, takeaways, fallback).

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
| 2026-11-09 | Claude | Phase 4 done: `pipeline/bundle.py` (class-aware KDEEB, medium/strong, uint16), trails for all 6 configs, visually checked. Next: Phase 5 (export). |
| 2026-11-09 | Claude | Plan revised: Flask dropped → static site in `app/` deployed on GitHub Pages (optional Flask wrapper, 8.5). Phases 6–8 rewritten; Phase 7 now one item per `VIS_PLAN.md` panel. `app/` skeleton restructured. |
| 2026-11-09 | Claude | Phase 5 done: `pipeline/export.py` → `app/data/` (index, meta, nodes, coords per method, trails per view, sprites; 66 MB), verified, schema in README. Next: Phase 6 (site skeleton + Pages). |
| 2026-11-09 | Claude | Phase 6 done except the first Pages deploy (6.6, needs user): layout, vendored D3 7.9.0, `data.js`, `state.js`, `main.js`, Pages workflow; Node tests + headless-Edge screenshots. Default branch is `main` (README links fixed). Next: Phase 7.1 (controls). |
| 2026-11-10 | Claude | Full check on macOS (no `raw/`, no `.venv`): pipeline compiles, `app/data/` consistent (only ±1 rounding on trail endpoints), all 6 configs × 2 methods + trails load in Chrome. 7.1 controls panel done (`controls.js`, `ui.js`). Next: 7.2 scatter. |
| 2026-11-10 | Claude | 7.2 scatter done (`scatter.js`, `metrics.js`): glyphs, legend filter, zoom/pan, brush, tooltip, pin, NH/acc badges, §6 animations of the scatter; 29-check browser test passes. Next: 7.3 steppers + Play. |
| 2026-11-10 | Claude | 7.3 done (`steppers.js`): filmstrip rows, Play/speed/All, keyboard, time axis; legend placed outside the plot when room. Steppers test (33 checks) + scatter regression (29) pass; screenshots at 1920 / 1300 px. Next: 7.4 path view. |
| 2026-11-10 | Claude | Layout reorganized to match `drafts/draft-v2.png` and never scroll (user request): steppers as compact rows inside the plot (filmstrip dropped), legend bottom-left, plot fills the panel, paged image view, panels sized to their height, `<details>` → collapsible sections. 4 suites + no-scroll check at 1280×720 … 2560×1440 pass. |
| 2026-11-10 | Claude | 7.4–7.9 done (path view, curve, architecture, metrics, images, data point, color-by, comet, chips; layout fixes). 8.1/8.2/8.4/8.5/8.6 done (README rationale + findings, `package.py`, `app.py`, `PRESENTATION.md`). Left for the user: 6.6 Pages setting + push, 8.3 Firefox check. |
| 2026-11-10 | Claude | Submission ZIP rebuilt with `drafts/` (`package.py` INCLUDE); 159 files, 28.8 MB; tested from a clean extraction with the Flask wrapper + headless Edge. |
| 2026-11-10 | Claude | UI review fixes: pinned trajectory in selection color + per-step glyphs; narrower legend (wrapping key); draggable steppers; collapsible image view; all non-zero confusion counts; side columns 16/64/20 %. Fixed two overflows this caused at 1280×720 (curve toggle, metrics headers now measured). Checked in headless Edge at 1280×720, 1366×768, 1920×1080, 2560×1440: no scroll, no clipped panel. |
| 2026-11-10 | Claude | Left column widened 16 → 19 % (min 220 → 250 px), center 64 → 61 %: controls were too tight. No overflow at 1280×720 / 1920×1080. ZIP not rebuilt (user: change not for the submission). |
| 2026-11-10 | Claude | Left column widened again 19 → 22 % (min 280 px), center 58 %. No overflow at 1280×720 / 1920×1080. ZIP not rebuilt. |
| 2026-11-11 | Claude | Project renamed to **JaMoNA** (Jimena and Andrea's Model Network Analysis); slug `jamona` in URLs, Colab/Drive paths and the ZIP. New logo `app/img/jamona.svg` (piglet + magnifying glass) in the README, site header and favicon. GitHub repo rename is up to the owner. |
