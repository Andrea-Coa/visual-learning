# In-class presentation: demo script

About 8 minutes: 1 min context, 6 min live demo, 1 min takeaways. Numbers come from README "Findings" (all 1000 points; with the default sample of 500 they differ slightly).

## Before class

- Open the live site (<https://andrea-coa.github.io/jamona/>) **and** a local copy as a backup:
  `python -m http.server 8000 -d app` (or `python app.py`). D3 is vendored, so the local copy works offline.
- Browser at 1920×1080 (or full screen), zoom 100 %. Load each dataset once so its files are cached.
- Start state: MNIST · MLP · Aligned UMAP grid (the default) · View by epoch · sample 500.

## Context (1 min)

- Task: show **how a neural network learns**: T1 = across **epochs**, T2 = across **layers** (Rauber et al. 2017).
- Pipeline: MLP and CNN trained on MNIST, SVHN, CIFAR-10 (Colab T4, 50 epochs); hidden activations of the same
  1000 test images at 11 snapshot epochs; t-SNE as in the paper (T1: one joint map per layer; T2: per epoch,
  initialized from the input map); default **Aligned UMAP grid**: t-SNE layer maps turn or mirror between layers
  (42 % of pairs), and the grid is one map aligned across epochs *and* layers, so T1 and T2 share positions
  (switching view keeps every point in place) at a small cost in faithfulness; class-aware trail bundling; static D3 site.
- How to read it: color = true class; triangle = misclassified, colored by the predicted class; NH = how well classes
  separate (share of 6 nearest neighbors with the same class).

## Live demo (6 min)

1. **T1, gradual learning** — SVHN · MLP · layer h4 · View by epoch. Click epoch `untrained`, press ▶ (Space).
   Points move from one central blob into digit clusters; NH badge ≈ 0.1 → ≈ 0.8; the confusion matrix concentrates
   on the diagonal; the learning-curve marker moves in sync.
2. **Overview** — press `All` (A): bundled trails show the whole training at once (dark = early, light = late).
   Switch bundling none → medium → strong (trails morph). Hover a trail: its true, unbundled trajectory.
3. **Mistakes** — back to scatter (A). Brush a mixed region (Shift + drag): the image view shows ambiguous house
   numbers. Click one image: Data point panel → softmax bars, "learned at epoch …" chart; its trajectory over epochs.
4. **T2, layer by layer** — MNIST · MLP, click `h1` in the architecture panel (switches to View by layer). ▶ plays
   Input → Output. Separation bars: ≈ 0.85 (pixels) → 0.94 → 0.97 → … → 0.98. Click epoch `0` (untrained): the untrained
   network already keeps the pixel structure (h4 0.76), but the Output layer mixes everything (0.27).
5. **Hard dataset** — CIFAR-10 · MLP, Color by `learned at`: 43 % of the sample is never learned (hollow triangles).
   Confusion matrix: click the cat → dog cell (cat ↔ dog: 39 of the 1000 images, the most confused pair):
   those points and images light up.
6. **CNN vs MLP, overfitting** — CIFAR-10 · CNN: last layer NH 0.56 vs 0.43 (MLP). Learning curve: test accuracy
   peaks at epoch 21 while train keeps rising; in T1 `fc` NH peaks at epoch 25 (0.63) and drops to 0.57; the ghost
   ("final") marks of the per-class bars sit below the bars of epoch 25.

## Takeaways (1 min)

1. Learning = pulling same-class observations together; NH grows over epochs (T1) and over layers (T2).
2. MNIST is learned in one epoch; SVHN and CIFAR-10 need many epochs and deep layers.
3. Untrained networks keep the input structure; the separation is built in the last layers (CNN conv1 barely
   changes).
4. Where classes stay mixed (cat ↔ dog, airplane ↔ ship), the model makes its mistakes.
5. The final epoch is not always the best one: the CNNs overfit, and it is visible in the projections.

## If something goes wrong

- Site does not load (network): use the local copy (`python -m http.server 8000 -d app`).
- Something looks stuck: reload the page (all state resets to the start state).
- Too slow on the projector laptop: lower the sample to 300.
