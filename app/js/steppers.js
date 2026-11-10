// Layer / epoch steppers and Play (VIS_PLAN.md §5.5, §6), laid out as in drafts/draft-v2.png: two compact rows at
// the bottom-right of the embeddings view, "Layer [ … ]" and "▶ Epoch [ … ] [All]". The row of the active dimension
// (epochs in T1, layers in T2) gets Play, All and the speed toggle; the other row only fixes the other parameter.
// The player advances the active dimension one step at a time and pauses on any manual step change.
import { frameOf } from "./data.js";
import { DWELL_MS, MOVE_MS, ms } from "./encoding.js";
import { cachedNeighborhoodHit } from "./metrics.js";
import { tooltip } from "./ui.js";

const SPEEDS = [0.5, 1, 2];
const INPUT_STATIC = "Raw pixels do not change during training";

const dimOf = (mode) => (mode === "epoch" ? "epoch" : "layer");

/** Builds the two stepper rows inside `container`. Returns { setConfiguration(view) } (called by main.js). */
export function initSteppers(container, { store }) {
  let view = null;             // { config, coords, dataset, model, projection }
  let built = null;            // structure key of the rows currently in the DOM
  let timer = null;
  let advancing = false;       // true while the player itself changes the step
  const root = d3.select(container);

  function isReady(state) {
    if (!view || state.layer === null || state.epoch === null) return false;
    return view.dataset === state.dataset && view.model === state.model && view.projection === state.projection
      && view.config.t2Layers.includes(state.layer) && view.config.epochs.includes(state.epoch);
  }

  /** Steps of a dimension: { value, label, title, disabled }. */
  function stepsOf(dim, state) {
    const { config } = view;
    if (dim === "epoch") {
      return config.epochs.map((e) => ({ value: e, label: String(e), title: e === 0 ? "epoch 0 (untrained)" : `epoch ${e}` }));
    }
    return config.layers.map((layer) => {
      const disabled = state.mode === "epoch" && !config.t1Layers.includes(layer.id);
      return { value: layer.id, label: layer.label, disabled, title: disabled ? INPUT_STATIC : `layer ${layer.label}` };
    });
  }

  const enabledValues = (state) => stepsOf(dimOf(state.mode), state).filter((d) => !d.disabled).map((d) => d.value);

  // ------------------------------------------------------------------ rows
  function render(state) {
    if (!isReady(state)) return;
    const structure = `${view.dataset}/${view.model}/${state.mode}`;
    if (structure !== built) build(state, structure);
    update(state);
  }

  /** One grid row per dimension: [▶] [label] [steps] [All] [speed]; cells are empty in the inactive row. */
  function build(state, structure) {
    built = structure;
    root.selectAll("*").remove();
    const active = dimOf(state.mode);
    for (const dim of ["layer", "epoch"]) {
      const isActive = dim === active;
      const row = root.append("div").attr("class", "stepper-row").attr("data-dim", dim).classed("active", isActive);
      if (isActive) {
        row.append("button").attr("type", "button").attr("class", "play-btn")
          .on("click", () => store.set({ playing: !store.get("playing") }));
      } else {
        row.append("span");
      }
      row.append("span").attr("class", "stepper-label").text(dim === "epoch" ? "Epoch" : "Layer");
      row.append("div").attr("class", "seg compact").attr("role", "group")
        .attr("aria-label", dim === "epoch" ? "Epoch" : "Layer")
        .selectAll("button").data(stepsOf(dim, state)).join("button")
        .attr("type", "button")
        .attr("class", "step")
        .property("disabled", (d) => Boolean(d.disabled))
        .text((d) => d.label)
        .on("click", (event, d) => store.set({ [dim]: d.value })) // a manual step change pauses Play
        .on("mouseenter", (event, d) => tooltip.show(stepTooltip(dim, d), event))
        .on("mousemove", (event) => tooltip.move(event))
        .on("mouseleave", () => tooltip.hide());
      if (isActive) {
        row.append("button").attr("type", "button").attr("class", "tool-btn all-btn").text("All")
          .attr("title", "Show every step at once (path view) · A")
          .on("click", () => store.set({ showAll: !store.get("showAll") }));
        row.append("div").attr("class", "seg speed").attr("role", "group").attr("aria-label", "Play speed")
          .selectAll("button").data(SPEEDS).join("button")
          .attr("type", "button")
          .text((d) => `${d}×`)
          .on("click", (event, d) => store.set({ speed: d }));
      } else {
        row.append("span");
        row.append("span");
      }
    }
  }

  function update(state) {
    root.selectAll(".step").each(function (d) {
      const current = d.value === state[this.closest(".stepper-row").dataset.dim];
      d3.select(this).classed("current", current).attr("aria-pressed", String(current));
    });
    root.select(".play-btn")
      .classed("playing", state.playing)
      .attr("aria-label", state.playing ? "Pause" : "Play")
      .attr("title", `${state.playing ? "Pause" : "Play"} · Space`);
    root.select(".all-btn").classed("active", state.showAll).attr("aria-pressed", String(state.showAll));
    root.selectAll(".speed button").classed("active", (d) => d === state.speed)
      .attr("aria-pressed", (d) => String(d === state.speed));
  }

  /** Step tooltip: its name and the NH of its projection on the shown points. */
  function stepTooltip(dim, d) {
    if (d.disabled) return d.title;
    const state = store.state;
    const { config, coords } = view;
    const frame = dim === "epoch"
      ? frameOf(config, coords, { mode: "epoch", layer: state.layer, epoch: d.value })
      : frameOf(config, coords, { mode: "layer", layer: d.value, epoch: state.epoch });
    const nh = cachedNeighborhoodHit(frame, config.sample(state.sampleSize), config.labels);
    return `${d.title} · NH ${nh.toFixed(2)}`;
  }

  // ------------------------------------------------------------------ player
  function advance(patch) {
    advancing = true;
    try {
      store.set(patch);
    } finally {
      advancing = false;
    }
  }

  function schedule() {
    clearTimeout(timer);
    timer = setTimeout(tick, (ms(MOVE_MS) + DWELL_MS) / store.get("speed"));
  }

  /** One Play step: move to the next step, or stop after the last one has been shown. */
  function tick() {
    timer = null;
    const state = store.state;
    if (!state.playing || !isReady(state)) return;
    const dim = dimOf(state.mode);
    const values = enabledValues(state);
    const i = values.indexOf(state[dim]);
    if (i < 0 || i >= values.length - 1) {
      store.set({ playing: false });
      return;
    }
    advance({ [dim]: values[i + 1] });
    schedule();
  }

  function onPlaying(state) {
    clearTimeout(timer);
    timer = null;
    if (!state.playing) return;
    if (!isReady(state)) {
      store.set({ playing: false });
      return;
    }
    const dim = dimOf(state.mode);
    const values = enabledValues(state);
    if (values.indexOf(state[dim]) >= values.length - 1) { // ▶ at the last step restarts from the first
      advance({ [dim]: values[0] });
      schedule();
    } else {
      tick();
    }
  }

  /** Previous / next step of the active dimension (keyboard ← / →). */
  function stepBy(delta) {
    const state = store.state;
    if (!isReady(state)) return;
    const dim = dimOf(state.mode);
    const values = enabledValues(state);
    const i = values.indexOf(state[dim]);
    const next = values[Math.max(0, Math.min(values.length - 1, (i < 0 ? 0 : i) + delta))];
    if (next !== state[dim]) store.set({ [dim]: next });
  }

  store.subscribe(onPlaying, ["playing"]);
  store.subscribe((state) => { if (state.playing && !advancing) store.set({ playing: false }); }, ["layer", "epoch"]);
  store.subscribe((state) => {
    if (state.playing) store.set({ playing: false });
    // T1 has no Input frames: switching to "epoch" while on Input moves to the first hidden layer.
    if (state.mode === "epoch" && isReady(state) && !view.config.t1Layers.includes(state.layer)) {
      store.set({ layer: view.config.t1Layers[0] });
    }
  }, ["dataset", "model", "projection", "mode"]);
  store.subscribe(render, ["mode", "layer", "epoch", "playing", "showAll", "speed"]);

  // ------------------------------------------------------------------ keyboard (§5.5): Space, ← / →, A
  const typing = (event) => event.target.closest?.("input, select, textarea") || event.metaKey || event.ctrlKey
    || event.altKey;
  document.addEventListener("keydown", (event) => {
    if (typing(event)) return;
    if (event.key === " ") {
      event.preventDefault();
      if (!event.repeat) store.set({ playing: !store.get("playing") });
    } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      stepBy(event.key === "ArrowLeft" ? -1 : 1);
    } else if (event.key === "a" || event.key === "A") {
      store.set({ showAll: !store.get("showAll") });
    }
  });
  // Space must not also "click" the focused button (e.g. the last step clicked).
  document.addEventListener("keyup", (event) => { if (event.key === " " && !typing(event)) event.preventDefault(); });

  return {
    setConfiguration(next) {
      view = next;
      built = null;
      render(store.state);
    },
  };
}
