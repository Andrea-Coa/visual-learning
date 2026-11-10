// Entry point: creates the shared data store and UI state, loads the configuration selected in the state and
// keeps it in sync. The panels (Phase 7) subscribe to the same store; until they exist, this module renders the
// generated title and the loading / error states (VIS_PLAN.md §5.4, §9).
import { initControls } from "./controls.js";
import { DataStore } from "./data.js";
import { Store } from "./state.js";

const PROJECTION_NAMES = { tsne: "t-SNE", pca: "PCA" };

const data = new DataStore("data/");
const store = new Store();

/** Everything the views need for the current dataset / model / projection. */
export const current = { index: null, config: null, coords: null };

function plotMessage(text, className = "loading") {
  const plot = document.getElementById("plot");
  plot.replaceChildren();
  const div = document.createElement("div");
  div.className = className;
  div.textContent = text;
  plot.append(div);
}

/** Generated plot title, e.g. "MNIST · MLP · layer h4 · epoch 8 · t-SNE" (§5.4). */
function title(state) {
  const { index, config } = current;
  const loaded = config && config.dataset === state.dataset && config.model === state.model;
  if (!loaded || state.layer === null || state.epoch === null) return ""; // configuration still loading
  const dataset = index.dataset(state.dataset).name;
  const model = index.model(state.dataset, state.model).name;
  const epoch = state.epoch === 0 ? "epoch 0 (untrained)" : `epoch ${state.epoch}`;
  const step = state.mode === "epoch" ? `layer ${state.layer}` : epoch;
  const fixed = state.mode === "epoch" ? epoch : `layer ${state.layer}`;
  return [dataset, model, step, fixed, PROJECTION_NAMES[state.projection]].join(" · ");
}

function renderTitle(state) {
  document.getElementById("embeddings-title").textContent = title(state);
}

let loadToken = 0;

/** Loads the selected configuration and fills in the state keys that depend on it (layer, epoch). */
async function loadConfiguration(state) {
  const token = ++loadToken;
  const { index } = current;
  plotMessage(`Loading ${index.dataset(state.dataset).name} · ${index.model(state.dataset, state.model).name}…`);
  try {
    const [config, coords] = await Promise.all([
      data.config(state.dataset, state.model),
      data.coords(state.dataset, state.model, state.projection),
    ]);
    if (token !== loadToken) return; // a newer selection superseded this one
    Object.assign(current, { config, coords });

    const patch = {};
    if (!config.t2Layers.includes(state.layer)) patch.layer = config.lastHiddenLayer;
    if (!config.epochs.includes(state.epoch)) patch.epoch = config.epochs[config.epochs.length - 1];
    if (Object.keys(patch).length) store.set(patch);
    plotMessage("", "plot-ready");
    renderTitle(store.state);
  } catch (err) {
    if (token === loadToken) plotMessage(`Could not load the data (${err.message}).`, "error");
    console.error(err);
  }
}

async function main() {
  current.index = await data.index();
  initControls(document.getElementById("controls"), { store, index: current.index });
  store.subscribe(loadConfiguration, ["dataset", "model", "projection"]);
  store.subscribe(renderTitle, ["mode", "layer", "epoch"]);
  await loadConfiguration(store.state);
}

// Handy in the browser console while developing: app.store.set({ dataset: "svhn" }).
globalThis.app = { data, store, current };

main().catch((err) => {
  plotMessage(`Could not start (${err.message}).`, "error");
  console.error(err);
});
