// Layer / epoch steppers and Play (VIS_PLAN.md §5.5, §6). The row of the active dimension (epochs in T1, layers
// in T2) is a filmstrip of mini scatterplots (canvas) with Play, All and speed; the other row only fixes the other
// parameter. The player advances the active dimension one step at a time and pauses on any manual step change.
import { frameOf } from "./data.js";
import { cachedNeighborhoodHit } from "./metrics.js";
import { CLASS_COLORS, MOVE_MS, ms } from "./scatter.js";
import { tooltip } from "./ui.js";

const DWELL_MS = 450;               // pause on each frame during Play, at 1×
const SPEEDS = [0.5, 1, 2];
const THUMB_SIZES = [64, 48];       // filmstrip thumbnails; plain buttons if even the smallest doesn't fit
const GAP = 6;                      // px between steps (keep in sync with .filmstrip gap)
const LEAD_WIDTH = 96;              // px of the row's first column (keep in sync with CSS)
const TAIL_WIDTH = 170;             // px reserved for All + speed
const AXIS_HEIGHT = 14;
const INPUT_STATIC = "Raw pixels do not change during training";

const dimOf = (mode) => (mode === "epoch" ? "epoch" : "layer");

/** Builds the two stepper rows inside `container`. Returns { setConfiguration(view) } (called by main.js). */
export function initSteppers(container, { store }) {
  let view = null;             // { config, coords, dataset, model, projection }
  let thumb = THUMB_SIZES[0];  // current thumbnail size, 0 = plain buttons
  let built = null;            // structure key of the rows currently in the DOM
  let drawnKey = null;         // content key of the thumbnails currently drawn
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
      return config.epochs.map((e) => ({
        value: e, label: e === 0 ? "untrained" : String(e), short: String(e),
        title: e === 0 ? "epoch 0 (untrained)" : `epoch ${e}`,
      }));
    }
    return config.layers.map((layer) => {
      const disabled = state.mode === "epoch" && !config.t1Layers.includes(layer.id);
      return {
        value: layer.id, label: layer.label, short: layer.label, disabled,
        title: disabled ? INPUT_STATIC : `${layer.label} · ${layer.units} units · ${layer.type}`,
      };
    });
  }

  const enabledValues = (state) => stepsOf(dimOf(state.mode), state).filter((d) => !d.disabled).map((d) => d.value);

  // ------------------------------------------------------------------ rows
  function render(state) {
    if (!isReady(state)) return;
    const structure = `${view.dataset}/${view.model}/${state.mode}/${thumb}`;
    if (structure !== built) build(state, structure);
    update(state);
  }

  function build(state, structure) {
    built = structure;
    drawnKey = null;
    root.selectAll(".stepper-row").remove();
    const active = dimOf(state.mode);
    for (const dim of ["layer", "epoch"]) {
      const isActive = dim === active;
      const row = root.append("div").attr("class", "stepper-row").attr("data-dim", dim)
        .classed("active", isActive)
        .classed("has-film", isActive && thumb > 0)
        .style("--thumb", `${thumb}px`);
      const lead = row.append("div").attr("class", "stepper-lead");
      if (isActive) {
        lead.append("button").attr("type", "button").attr("class", "play-btn")
          .on("click", () => store.set({ playing: !store.get("playing") }));
      } else {
        lead.append("span").attr("class", "play-spacer");
      }
      lead.append("span").attr("class", "stepper-label").text(dim === "epoch" ? "Epoch" : "Layer");

      const cell = row.append("div").attr("class", "stepper-steps");
      const steps = stepsOf(dim, state);
      if (isActive && thumb > 0) buildFilmstrip(cell, dim, steps);
      else buildCompact(cell, dim, steps);
      if (isActive && thumb > 0 && dim === "epoch") buildAxis(cell, steps);

      const tail = row.append("div").attr("class", "stepper-tail");
      if (isActive) {
        tail.append("button").attr("type", "button").attr("class", "tool-btn all-btn").text("All")
          .attr("title", "Show every step at once (path view) · A")
          .on("click", () => store.set({ showAll: !store.get("showAll") }));
        tail.append("div").attr("class", "seg speed").attr("role", "group").attr("aria-label", "Play speed")
          .selectAll("button").data(SPEEDS).join("button")
          .attr("type", "button")
          .text((d) => `${d}×`)
          .on("click", (event, d) => store.set({ speed: d }));
      }
    }
  }

  function selectStep(dim, value) {
    store.set({ [dim]: value }); // a manual step change pauses Play (see the subscription below)
  }

  function buildCompact(cell, dim, steps) {
    cell.append("div").attr("class", "seg compact").attr("role", "group")
      .selectAll("button").data(steps).join("button")
      .attr("type", "button")
      .attr("class", "step")
      .attr("title", (d) => d.title)
      .property("disabled", (d) => Boolean(d.disabled))
      .text((d) => d.short)
      .on("click", (event, d) => selectStep(dim, d.value));
  }

  function buildFilmstrip(cell, dim, steps) {
    cell.append("div").attr("class", "filmstrip")
      .selectAll("button").data(steps).join((enter) => {
        const button = enter.append("button").attr("type", "button").attr("class", "step step-thumb")
          .attr("aria-label", (d) => d.title)
          .property("disabled", (d) => Boolean(d.disabled))
          .on("click", (event, d) => selectStep(dim, d.value))
          .on("mouseenter", (event, d) => tooltip.show(thumbTooltip(dim, d), event))
          .on("mousemove", (event) => tooltip.move(event))
          .on("mouseleave", () => tooltip.hide());
        button.append("canvas").style("width", `${thumb}px`).style("height", `${thumb}px`);
        button.append("span").attr("class", "step-label").text((d) => d.label);
        return button;
      });
  }

  /** Compact time axis under the epoch filmstrip: each thumbnail is linked to its true position on 0..max epoch. */
  function buildAxis(cell, steps) {
    const width = steps.length * (thumb + GAP) - GAP;
    const center = (i) => i * (thumb + GAP) + thumb / 2;
    const t = d3.scaleLinear().domain(d3.extent(steps, (d) => d.value)).range([center(0), center(steps.length - 1)]);
    const svg = cell.append("svg").attr("class", "stepper-axis").attr("width", width).attr("height", AXIS_HEIGHT)
      .attr("aria-hidden", true);
    svg.append("line").attr("class", "axis-base")
      .attr("x1", t.range()[0]).attr("x2", t.range()[1]).attr("y1", AXIS_HEIGHT - 3).attr("y2", AXIS_HEIGHT - 3);
    const tick = svg.selectAll("g").data(steps).join("g").attr("class", "axis-tick");
    tick.append("line").attr("x1", (d, i) => center(i)).attr("y1", 0)
      .attr("x2", (d) => t(d.value)).attr("y2", AXIS_HEIGHT - 3);
    tick.append("circle").attr("cx", (d) => t(d.value)).attr("cy", AXIS_HEIGHT - 3).attr("r", 2);
  }

  function update(state) {
    const active = dimOf(state.mode);
    root.selectAll(".step").classed("current", function (d) {
      return d.value === state[this.closest(".stepper-row").dataset.dim];
    }).attr("aria-pressed", function (d) {
      return String(d.value === state[this.closest(".stepper-row").dataset.dim]);
    });
    root.selectAll(".axis-tick").classed("current", (d) => d.value === state[active]);
    root.select(".play-btn")
      .classed("playing", state.playing)
      .attr("aria-label", state.playing ? "Pause" : "Play")
      .attr("title", `${state.playing ? "Pause" : "Play"} · Space`);
    root.select(".all-btn").classed("active", state.showAll).attr("aria-pressed", String(state.showAll));
    root.selectAll(".speed button").classed("active", (d) => d === state.speed)
      .attr("aria-pressed", (d) => String(d === state.speed));
    drawThumbnails(state);
  }

  // ------------------------------------------------------------------ filmstrip thumbnails (canvas)
  function thumbFrame(dim, value, state) {
    const { config, coords } = view;
    return dim === "epoch"
      ? frameOf(config, coords, { mode: "epoch", layer: state.layer, epoch: value })
      : frameOf(config, coords, { mode: "layer", layer: value, epoch: state.epoch });
  }

  function drawThumbnails(state) {
    if (thumb === 0) return;
    const dim = dimOf(state.mode);
    const filter = state.classFilter ? [...state.classFilter].sort().join(",") : "";
    const key = [built, view.projection, dim === "epoch" ? state.layer : state.epoch, state.sampleSize, filter].join("|");
    if (key === drawnKey) return;
    drawnKey = key;

    const { labels } = view.config;
    const ids = view.config.sample(state.sampleSize);
    const dpr = window.devicePixelRatio || 1;
    const pad = 3;
    const span = thumb - 2 * pad;
    root.selectAll(".step-thumb canvas").each(function (d) {
      this.width = thumb * dpr;
      this.height = thumb * dpr;
      const ctx = this.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, thumb, thumb);
      const frame = thumbFrame(dim, d.value, state);
      for (const id of ids) {
        const cls = labels[id];
        ctx.globalAlpha = state.classFilter && !state.classFilter.has(cls) ? 0.08 : 0.8;
        ctx.fillStyle = CLASS_COLORS[cls];
        ctx.fillRect(pad + frame[2 * id] * span - 0.75, pad + frame[2 * id + 1] * span - 0.75, 1.5, 1.5);
      }
    });
  }

  function thumbTooltip(dim, d) {
    const state = store.state;
    if (d.disabled) return d.title;
    const ids = view.config.sample(state.sampleSize);
    const nh = cachedNeighborhoodHit(thumbFrame(dim, d.value, state), ids, view.config.labels);
    return `${d.title} · NH ${nh.toFixed(2)}`;
  }

  // ------------------------------------------------------------------ thumbnail size from the available width
  function layout() {
    if (!view) return;
    const state = store.state;
    const n = isReady(state) ? stepsOf(dimOf(state.mode), state).length : view.config.epochs.length;
    const available = container.clientWidth - LEAD_WIDTH - TAIL_WIDTH - 2 * GAP;
    const next = THUMB_SIZES.find((size) => n * (size + GAP) - GAP <= available) ?? 0;
    if (next !== thumb) {
      thumb = next;
      render(state);
    }
  }
  new ResizeObserver(layout).observe(container);

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
    if (next !== state[dim]) selectStep(dim, next);
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
  store.subscribe(render, ["mode", "layer", "epoch", "playing", "showAll", "speed", "sampleSize", "classFilter"]);

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
      layout();
      render(store.state);
    },
  };
}
