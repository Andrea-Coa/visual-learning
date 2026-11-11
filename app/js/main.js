// Entry point: creates the shared data store and UI state, builds the panels, loads the configuration selected in
// the state and hands it to the panels. Loading / error states are an overlay on the plot (VIS_PLAN.md §9).
import { initArchitecture } from "./architecture.js";
import { initControls } from "./controls.js";
import { initCurve } from "./curve.js";
import { DataStore } from "./data.js";
import { initDataPoint } from "./datapoint.js";
import { initImages } from "./images.js";
import { initModelMetrics } from "./model-metrics.js";
import { initScatter } from "./scatter.js";
import { Store } from "./state.js";
import { initSteppers } from "./steppers.js";

const PROJECTION_NAMES = { tsne: "t-SNE", pca: "PCA" };

const data = new DataStore("data/");
const store = new Store();

/** Everything the views need for the current dataset / model / projection. */
export const current = { index: null, config: null, coords: null };

const plot = document.getElementById("plot");
const message = document.createElement("div");
plot.append(message);

/** Shows `text` over the plot (class "loading" or "error"); no text hides the overlay. */
function plotMessage(text, className = "loading") {
  message.className = `plot-message ${className}`;
  message.textContent = text;
  message.hidden = !text;
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
const panels = [];   // views that receive every loaded configuration via setConfiguration()

/** Loads the selected configuration, fills in the state keys that depend on it (layer, epoch), updates the views. */
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
    if (!config.t2Layers.includes(store.state.layer)) patch.layer = config.lastHiddenLayer;
    if (!config.epochs.includes(store.state.epoch)) patch.epoch = config.epochs[config.epochs.length - 1];
    if (Object.keys(patch).length) store.set(patch);
    plotMessage("");
    renderTitle(store.state);
    const { dataset, model, projection } = state;
    const view = { index, config, coords, dataset, model, projection, spriteUrl: data.spriteUrl(dataset) };
    for (const panel of panels) panel.setConfiguration(view);
  } catch (err) {
    if (token === loadToken) plotMessage(`Could not load the data (${err.message}).`, "error");
    console.error(err);
  }
}

/**
 * Collapsible panels (draft-v2 ▾): the title (and the arrow, where the title also holds tools) toggles the panel;
 * the other panel of the column takes the space.
 */
function initCollapsible() {
  for (const button of document.querySelectorAll(".collapsible .collapse-btn")) {
    button.addEventListener("click", () => {
      const panel = button.closest(".collapsible");
      const collapsed = panel.classList.toggle("collapsed");
      for (const b of panel.querySelectorAll(".collapse-btn")) b.setAttribute("aria-expanded", String(!collapsed));
      const arrow = panel.querySelector(".collapse-arrow");
      const name = panel.querySelector(".collapse-label")?.textContent.trim().toLowerCase() ?? "panel";
      arrow?.setAttribute("aria-label", `${collapsed ? "Expand" : "Collapse"} ${name}`);
    });
  }
}

async function main() {
  initCollapsible();
  current.index = await data.index();
  initControls(document.getElementById("controls"), { store, index: current.index });
  panels.push(
    initScatter(plot, {
      store, data, tools: document.getElementById("embeddings-tools"), overlay: document.getElementById("steppers"),
    }),
    initSteppers(document.getElementById("steppers"), { store }),
    initArchitecture(document.getElementById("architecture"), {
      store, subtitle: document.getElementById("architecture-model"), tools: document.getElementById("architecture-tools"),
    }),
    initCurve(document.getElementById("learning-curve"), { store, tools: document.getElementById("curve-tools") }),
    initModelMetrics(document.getElementById("metrics"), { store }),
    initImages(document.getElementById("images"), {
      store, subtitle: document.getElementById("images-source"), tools: document.getElementById("images-tools"),
    }),
    initDataPoint(document.getElementById("datapoint"), { store }),
  );
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
