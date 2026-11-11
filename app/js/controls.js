// Controls panel (VIS_PLAN.md §5.1). Every control writes to the shared state and is re-rendered from it, so a
// change made elsewhere (e.g. clicking a layer in the architecture panel switches "View by") shows up here too.
// Options without exported data are disabled with a "not available" tooltip (§9).
import { infoIcon } from "./ui.js";

const PROJECTIONS = [
  { id: "grid", name: "Aligned UMAP grid" }, { id: "aumap", name: "Aligned UMAP" }, { id: "umap", name: "UMAP" },
  { id: "tsne", name: "t-SNE" },
  { id: "tsne_proc", name: "t-SNE + Procrustes" }, { id: "pca", name: "PCA" }];
const MODES = [{ id: "epoch", name: "epoch (T1)" }, { id: "layer", name: "layer (T2)" }];
const COLOR_BY = [{ id: "class", name: "class" }, { id: "correctness", name: "correctness" },
  { id: "learnedAt", name: "learned at" }];
const BUNDLING = [{ id: "none", name: "none" }, { id: "medium", name: "medium" }, { id: "strong", name: "strong" }];
export const SAMPLE = Object.freeze({ min: 100, max: 1000, step: 100 });

const INFO = {
  projection: "t-SNE and UMAP keep neighbors together; PCA keeps global variance. Layers (T2): t-SNE and UMAP start "
    + "from the Input map; t-SNE + Procrustes also turns each layer onto the previous one; Aligned UMAP penalizes "
    + "points that move between layers. Epochs (T1): one joint map of all epochs. Aligned UMAP grid: one map for "
    + "both views (aligned across epochs and layers), so switching T1 / T2 keeps every point in place.",
  mode: "T1 (epoch): one layer across the training epochs. T2 (layer): one epoch from the input to the output layer.",
  sampleSize: "Balanced sample of test images: the same points at every epoch and layer. "
    + "Growing it only adds points.",
  colorBy: "class: true class (misclassified points are triangles in the predicted class color). "
    + "correctness: right vs. wrong. learned at: first epoch from which a point stays correctly classified.",
  bundling: "Pulls the trails of each class together in the path view (bundled projections only).",
};

/** Builds the controls inside `container` and keeps them in sync with `store`. */
export function initControls(container, { store, index }) {
  const form = d3.select(container).append("form").attr("class", "controls")
    .on("submit", (event) => event.preventDefault());

  const modelIds = [...new Set(index.datasets.flatMap((d) => d.models.map((m) => m.id)))];
  const modelName = (id) => index.datasets.flatMap((d) => d.models).find((m) => m.id === id).name;

  // Each select: its state key and the options for the current state ({id, name, disabled}).
  const selects = [
    { key: "dataset", label: "Dataset", options: () => index.datasets.map(({ id, name }) => ({ id, name })) },
    {
      key: "projection", label: "Dim. red.", info: INFO.projection,
      options: (s) => PROJECTIONS.map((p) => ({ ...p, disabled: !index.model(s.dataset, s.model)?.methods.includes(p.id) })),
    },
    {
      key: "model", label: "ANN",
      options: (s) => modelIds.map((id) => ({ id, name: modelName(id), disabled: !index.model(s.dataset, id) })),
    },
    { key: "mode", label: "View by", info: INFO.mode, options: () => MODES },
    { key: "sampleSize", label: "Sample", info: INFO.sampleSize, spinner: true },
    { key: "colorBy", label: "Color by", info: INFO.colorBy, options: () => COLOR_BY, separatorBefore: true },
    {
      key: "bundling", label: "Bundling", info: INFO.bundling, options: () => BUNDLING,
      disabled: (s) => !s.showAll || !index.model(s.dataset, s.model)?.trails.includes(s.projection),
      disabledReason: "Path view with Aligned UMAP (grid), Aligned UMAP or t-SNE only",
    },
  ];

  /** A state patch for `key = value` that keeps the dataset / model / projection combination available. */
  function patchFor(key, value) {
    const s = { ...store.state, [key]: value };
    const patch = { [key]: value };
    const dataset = index.dataset(s.dataset);
    if (!index.model(s.dataset, s.model)) patch.model = s.model = dataset.models[0].id;
    const { methods } = index.model(s.dataset, s.model);
    if (!methods.includes(s.projection)) patch.projection = methods[0];
    return patch;
  }

  const rows = {};
  for (const spec of selects) {
    if (spec.separatorBefore) form.append("hr").attr("class", "controls-sep");
    const id = `control-${spec.key}`;
    form.append("label").attr("for", id).text(spec.label);
    const cell = form.append("div").attr("class", "control-cell");
    rows[spec.key] = spec.spinner ? buildSpinner(cell, id, store) : cell.append("select")
      .attr("id", id)
      .attr("class", "control-input")
      .on("change", (event) => store.set(patchFor(spec.key, event.target.value)));
    const infoCell = form.append("span").attr("class", "control-info");
    if (spec.info) infoCell.node().append(infoIcon(spec.info));
    if (spec.spinner) form.append("div").attr("class", "control-note").attr("id", `${id}-note`);
  }

  function render(state) {
    for (const spec of selects) {
      if (spec.spinner) {
        renderSpinner(rows[spec.key], state.sampleSize);
        const nClasses = index.dataset(state.dataset).classes.length;
        form.select(`#control-${spec.key}-note`).text(`balanced · ${state.sampleSize / nClasses} per class`);
        continue;
      }
      const disabled = spec.disabled?.(state) ?? false;
      rows[spec.key]
        .property("disabled", disabled)
        .attr("title", disabled ? spec.disabledReason : null)
        .selectAll("option")
        .data(spec.options(state), (d) => d.id)
        .join("option")
        .attr("value", (d) => d.id)
        .text((d) => d.name)
        .property("disabled", (d) => Boolean(d.disabled))
        .attr("title", (d) => (d.disabled ? "not available" : null));
      rows[spec.key].property("value", state[spec.key]);
    }
  }

  store.subscribe(render, selects.map((s) => s.key).concat("showAll"));
  render(store.state);
}

// --------------------------------------------------------------------------- sample-size spinner

function clampSample(value) {
  const rounded = Math.round(value / SAMPLE.step) * SAMPLE.step;
  return Math.min(SAMPLE.max, Math.max(SAMPLE.min, rounded));
}

/** [−] [number] [+] spinner for the sample size; typing a value snaps it to the step on change. */
function buildSpinner(cell, id, store) {
  const spinner = cell.append("div").attr("class", "spinner");
  const step = (sign) => store.set({ sampleSize: clampSample(store.get("sampleSize") + sign * SAMPLE.step) });
  spinner.append("button").attr("type", "button").attr("class", "spinner-dec")
    .attr("aria-label", "Fewer points").text("−").on("click", () => step(-1));
  spinner.append("input")
    .attr("id", id)
    .attr("type", "number")
    .attr("inputmode", "numeric")
    .attr("min", SAMPLE.min).attr("max", SAMPLE.max).attr("step", SAMPLE.step)
    .on("change", (event) => {
      const value = Number(event.target.value);
      const next = Number.isFinite(value) && event.target.value !== "" ? clampSample(value) : store.get("sampleSize");
      event.target.value = next; // also restores the field when the value did not change the state
      store.set({ sampleSize: next });
    });
  spinner.append("button").attr("type", "button").attr("class", "spinner-inc")
    .attr("aria-label", "More points").text("+").on("click", () => step(1));
  return spinner;
}

function renderSpinner(spinner, value) {
  spinner.select("input").property("value", value);
  spinner.select(".spinner-dec").property("disabled", value <= SAMPLE.min);
  spinner.select(".spinner-inc").property("disabled", value >= SAMPLE.max);
}
