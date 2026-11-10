// Embeddings view, scatter mode (VIS_PLAN.md §5.4; encodings §4.1–4.2; animations §6).
// Every observation is an SVG group keyed by its id, so selections survive epoch/layer changes and the points can
// move between frames. Positions live in "base" pixels; zoom/pan transforms the whole layer and scales every glyph
// back by 1/k, so the point size stays constant while zooming.
import { frameOf } from "./data.js";
import { cachedNeighborhoodHit, sampleAccuracy } from "./metrics.js";
import { infoIcon, tooltip } from "./ui.js";

export const CLASS_COLORS = d3.schemeCategory10;  // = matplotlib tab10, the colors of the paper's figures

const BASE_OPACITY = 0.75;
const DIM_OPACITY = 0.08;
const EMPHASIS = 1.8;      // radius factor of the hovered / pinned point
const PAD = 16;            // px between the frame and the outermost points
const MAX_ZOOM = 8;
const HOVER_DELAY = 80;    // ms before the tooltip appears
const NH_K = 6;
const LEGEND_MARGIN = 8;   // px between the legend and the plot frame
const DURATION = { move: 750, swapOut: 200, swapIn: 200, sample: 300, fade: 250, reset: 400 };

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
export const ms = (duration) => (reducedMotion.matches ? 0 : duration);
export const MOVE_MS = DURATION.move;

const NH_INFO = "NH (neighborhood hit): share of each point's 6 nearest neighbors in this projection that have "
  + "its class. Sample acc: accuracy of the shown points at this epoch. ▲ ▼: change from the previous step.";

/** The frame shown for a state, and the frame of the previous step of the active dimension (null at the first). */
function framesFor(config, coords, state) {
  const frame = frameOf(config, coords, state);
  const e = config.epochIndex(state.epoch);
  const l = config.t2Layers.indexOf(state.layer);
  if (state.mode === "epoch") {
    const sequence = coords.t1[state.layer]; // undefined for Input, whose frame doesn't change with training
    return { frame, previous: sequence && e > 0 ? sequence[e - 1] : null };
  }
  return { frame, previous: l > 0 ? coords.t2[e][l - 1] : null };
}

/** Whether observation `d` passes the active filters (they combine with AND; VIS_PLAN.md §3). */
function passes(d, state, { ignoreClass = false } = {}) {
  if (!ignoreClass && state.classFilter && !state.classFilter.has(d.label)) return false;
  if (state.cellFilter && (d.label !== state.cellFilter.truth || d.pred !== state.cellFilter.pred)) return false;
  if (state.brushed && !state.brushed.has(d.id)) return false;
  return true;
}

const filtersActive = (state) => Boolean(state.classFilter || state.cellFilter || state.brushed);

/**
 * Builds the scatterplot inside `container` (#plot) and its toolbar inside `tools`.
 * Returns { setConfiguration(view) }, called by main.js once a configuration's files are loaded.
 */
export function initScatter(container, { store, tools }) {
  let view = null;              // { index, config, coords, dataset, model, projection, spriteUrl }
  let shown = null;             // what is drawn: { key, layer, epoch, n }
  let size = 0;
  let radius = 3;
  let transform = d3.zoomIdentity;
  let brushMode = false;
  let legendHover = null;       // class highlighted while hovering its legend entry
  let swapping = false;
  let hoverTimer = null;
  let lastPointer = null;
  let ids = [];                 // observation ids currently drawn (the sample)
  const nodes = new Map();      // id -> datum { id, label, x, y, pred, drawnPred }, reused across steps
  const x = d3.scaleLinear().domain([0, 1]);
  const y = d3.scaleLinear().domain([0, 1]);

  // ------------------------------------------------------------------ DOM
  const root = d3.select(container).append("div").attr("class", "scatter");
  const svg = root.append("svg").attr("class", "scatter-svg").attr("role", "img")
    .attr("aria-label", "Scatterplot of the projected observations");
  const frameRect = svg.append("rect").attr("class", "scatter-frame");
  const zoomLayer = svg.append("g").attr("class", "zoom-layer");
  const pointsG = zoomLayer.append("g").attr("class", "points");
  const brushRect = svg.append("rect").attr("class", "brush-rect").style("display", "none");

  const badges = root.append("div").attr("class", "badges");
  const badgeRows = {};
  for (const [key, label] of [["nh", "NH"], ["acc", "Sample acc"]]) {
    const row = badges.append("div").attr("class", "badge");
    row.append("span").attr("class", "badge-label").text(label);
    badgeRows[key] = {
      value: row.append("span").attr("class", "metric badge-value"),
      delta: row.append("span").attr("class", "badge-delta"),
    };
  }
  badges.node().append(infoIcon(NH_INFO));

  const legend = root.append("div").attr("class", "legend");
  const legendHeader = legend.append("div").attr("class", "legend-header");
  legendHeader.append("span").text("Legend");
  legendHeader.append("button").attr("type", "button").attr("class", "legend-toggle")
    .attr("aria-label", "Collapse legend")
    .on("click", () => {
      const collapsed = !legend.classed("collapsed");
      legend.classed("collapsed", collapsed);
      legendHeader.select(".legend-toggle").attr("aria-label", collapsed ? "Expand legend" : "Collapse legend");
    });
  const legendItems = legend.append("div").attr("class", "legend-items");
  const legendKey = legend.append("div").attr("class", "legend-key");
  legendKey.append("svg").attr("width", 12).attr("height", 12).attr("viewBox", "-6 -6 12 12")
    .append("path").attr("d", d3.symbol(d3.symbolTriangle, 40)());
  legendKey.append("span").text("misclassified (predicted class color)");

  const empty = root.append("div").attr("class", "scatter-empty").attr("hidden", true);
  empty.append("span").text("No observations match");
  empty.append("button").attr("type", "button").attr("class", "tool-btn").text("Clear filters")
    .on("click", () => store.set({ classFilter: null, cellFilter: null, brushed: null }));

  // Toolbar in the panel title: [Pan | Brush] and reset zoom.
  const toolbar = d3.select(tools);
  const modeButtons = toolbar.append("div").attr("class", "seg").attr("role", "group")
    .attr("aria-label", "Drag mode")
    .selectAll("button")
    .data([{ id: "pan", name: "Pan" }, { id: "brush", name: "Brush" }])
    .join("button")
    .attr("type", "button")
    .attr("title", (d) => (d.id === "brush" ? "Select points (or Shift + drag)" : "Drag to pan, scroll to zoom"))
    .text((d) => d.name)
    .on("click", (event, d) => setBrushMode(d.id === "brush"));
  toolbar.append("button").attr("type", "button").attr("class", "tool-btn tool-icon")
    .attr("title", "Reset zoom").attr("aria-label", "Reset zoom").text("⟲")
    .on("click", () => svg.transition().duration(ms(DURATION.reset)).call(zoom.transform, d3.zoomIdentity));

  function setBrushMode(on) {
    brushMode = on;
    modeButtons.classed("active", (d) => (d.id === "brush") === on)
      .attr("aria-pressed", (d) => String((d.id === "brush") === on));
    svg.classed("brush-mode", on);
  }
  setBrushMode(false);

  // ------------------------------------------------------------------ glyphs
  const symbolCache = new Map();
  function symbolPath(wrong, factor) {
    const key = `${wrong}|${radius}|${factor}`;
    if (!symbolCache.has(key)) {
      const area = Math.PI * (radius * factor) ** 2; // triangles get the same area as circles
      symbolCache.set(key, d3.symbol(wrong ? d3.symbolTriangle : d3.symbolCircle, area)());
    }
    return symbolCache.get(key);
  }

  const isEmphasized = (d, state) => d.id === state.hovered || d.id === state.pinned;

  /** Shape and color of a point from what is currently drawn (`drawnPred` switches at mid-transition). */
  function paintGlyph(path, d, state) {
    const emphasized = isEmphasized(d, state);
    path.attr("d", symbolPath(d.drawnPred !== d.label, emphasized ? EMPHASIS : 1))
      .attr("fill", CLASS_COLORS[d.drawnPred])
      .classed("emph", emphasized);
  }

  const position = (d) => `translate(${d.x},${d.y})`;
  const glyphScale = () => `scale(${1 / transform.k})`;

  // ------------------------------------------------------------------ zoom, pan, brush
  const zoom = d3.zoom()
    .scaleExtent([1, MAX_ZOOM])
    .filter((event) => (!event.ctrlKey || event.type === "wheel") && !event.button
      && (event.type === "wheel" || (!brushMode && !event.shiftKey)))
    .on("zoom", (event) => {
      transform = event.transform;
      zoomLayer.attr("transform", transform);
      pointsG.selectAll(".glyph").attr("transform", glyphScale());
      tooltip.hide();
    });
  svg.call(zoom).on("dblclick.zoom", null);

  let brushOrigin = null;
  const brushBox = (event) => {
    const [x0, y0] = brushOrigin;
    const cx = Math.max(0, Math.min(size, event.x));
    const cy = Math.max(0, Math.min(size, event.y));
    return [Math.min(x0, cx), Math.min(y0, cy), Math.max(x0, cx), Math.max(y0, cy)];
  };
  svg.call(d3.drag()
    .container(svg.node())
    .filter((event) => (brushMode || event.shiftKey) && !event.button)
    .on("start", (event) => {
      brushOrigin = [event.x, event.y];
      tooltip.hide();
    })
    .on("drag", (event) => {
      const [x0, y0, x1, y1] = brushBox(event);
      brushRect.style("display", null).attr("x", x0).attr("y", y0).attr("width", x1 - x0).attr("height", y1 - y0);
    })
    .on("end", (event) => {
      brushRect.style("display", "none");
      const [x0, y0, x1, y1] = brushBox(event);
      if (x1 - x0 < 3 && y1 - y0 < 3) return; // a click, not a brush
      const [a0, b0] = transform.invert([x0, y0]);
      const [a1, b1] = transform.invert([x1, y1]);
      const inside = ids.filter((id) => {
        const d = nodes.get(id);
        return d.x >= a0 && d.x <= a1 && d.y >= b0 && d.y <= b1;
      });
      store.set({ brushed: inside.length ? new Set(inside) : null }); // an empty brush clears itself (§9)
    }));

  svg.on("click", (event) => {
    const point = event.target.closest(".pt");
    if (point) {
      const { id } = d3.select(point).datum();
      store.set({ pinned: store.get("pinned") === id ? null : id });
    } else if (store.get("brushed")) {
      store.set({ brushed: null });
    }
  });
  svg.on("mousemove.pointer", (event) => { lastPointer = event; });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || event.target.closest?.("input, select, textarea")) return;
    if (store.get("pinned") !== null) store.set({ pinned: null });
    else if (store.get("brushed")) store.set({ brushed: null });
  });

  // ------------------------------------------------------------------ hover + tooltip
  function onEnter(event, d) {
    lastPointer = event;
    store.set({ hovered: d.id });
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(() => tooltip.show(tooltipContent(d), lastPointer), HOVER_DELAY);
  }
  function onMove(event) {
    lastPointer = event;
    tooltip.move(event);
  }
  function onLeave(event, d) {
    clearTimeout(hoverTimer);
    tooltip.hide();
    if (store.get("hovered") === d.id) store.set({ hovered: null });
  }

  function tooltipContent(d) {
    const { index, config, dataset, spriteUrl } = view;
    const state = store.state;
    const pred = config.pred(state.epoch, d.id);
    const conf = config.probsOf(state.epoch, d.id)[pred];
    const box = d3.create("div").attr("class", "tip-point");
    const thumb = box.append("div").attr("class", "thumb tip-thumb");
    Object.assign(thumb.node().style, index.sprite(dataset, d.id, 64, spriteUrl));
    const text = box.append("div").attr("class", "tip-text");
    text.append("div").attr("class", "tip-id").text(`#${d.id}`);
    const row = (label, cls, value) => {
      const line = text.append("div").attr("class", "tip-row");
      line.append("span").attr("class", "chip").style("background", CLASS_COLORS[cls]);
      line.append("span").text(`${label}: ${config.classes[cls]}${value ?? ""}`);
    };
    row("true", d.label);
    row(pred === d.label ? "predicted" : "▲ predicted", pred, ` (conf ${conf.toFixed(2)})`);
    return box.node();
  }

  // ------------------------------------------------------------------ rendering
  const keyOf = (state) => `${state.dataset}/${state.model}/${state.projection}/${state.mode}`;

  function isReady(state) {
    if (!view || state.layer === null || state.epoch === null) return false;
    const { config } = view;
    return view.dataset === state.dataset && view.model === state.model && view.projection === state.projection
      && config.t2Layers.includes(state.layer) && config.epochs.includes(state.epoch);
  }

  function render(state, changed = new Set()) {
    if (swapping || !isReady(state) || size === 0) return;
    if (!shown || shown.key !== keyOf(state)) {
      swap();
      return;
    }
    const stepChanged = shown.layer !== state.layer || shown.epoch !== state.epoch;
    const sampleChanged = shown.n !== state.sampleSize;
    if (stepChanged || sampleChanged) {
      draw(state, { move: stepChanged, sample: sampleChanged });
    } else {
      restyle(state, { fade: ["classFilter", "cellFilter", "brushed"].some((k) => changed.has(k)) });
    }
  }

  /** Dataset / model / projection / mode change: positions are not comparable, so fade out and back in (§6). */
  function swap() {
    swapping = true;
    tooltip.hide();
    zoomLayer.transition("swap").duration(shown ? ms(DURATION.swapOut) : 0).style("opacity", 0)
      .on("end", () => {
        swapping = false;
        pointsG.selectAll(".pt").remove();
        nodes.clear();
        shown = null;
        const state = store.state;
        if (!isReady(state)) return;
        svg.call(zoom.transform, d3.zoomIdentity);
        renderLegendItems();
        if (!legendDragged) placeLegend(); // class names change the legend width (e.g. CIFAR-10)
        draw(state, {});
        zoomLayer.transition("swap").duration(ms(DURATION.swapIn)).style("opacity", 1);
      });
  }

  /** Step or sample change: joins the sample, moves the points (animated on a step change), updates the badges. */
  function draw(state, { move = false, sample = false }) {
    const { config, coords } = view;
    ids = Array.from(config.sample(state.sampleSize));
    const { frame, previous } = framesFor(config, coords, state);
    const preds = config.preds[config.epochIndex(state.epoch)];
    const nextRadius = state.sampleSize <= 300 ? 4 : 3;
    const radiusChanged = nextRadius !== radius;
    radius = nextRadius;

    for (const id of ids) {
      let d = nodes.get(id);
      if (!d) nodes.set(id, (d = { id, label: config.labels[id] }));
      d.x = x(frame[2 * id]);
      d.y = y(frame[2 * id + 1]);
      d.pred = preds[id];
    }

    const points = pointsG.selectAll(".pt:not(.exiting)")
      .data(ids.map((id) => nodes.get(id)), (d) => d.id)
      .join(
        (enter) => {
          const g = enter.append("g").attr("class", "pt").attr("transform", position)
            .each((d) => { d.drawnPred = d.pred; })
            .on("mouseenter", onEnter).on("mousemove", onMove).on("mouseleave", onLeave);
          const glyph = g.append("g").attr("class", "glyph").attr("transform", sample ? "scale(0)" : glyphScale());
          glyph.append("path");
          if (sample) glyph.transition("grow").duration(ms(DURATION.sample)).attr("transform", glyphScale());
          return g;
        },
        (update) => update,
        (exit) => {
          exit.classed("exiting", true);
          exit.select(".glyph").transition("grow").duration(ms(DURATION.sample)).attr("transform", "scale(0)");
          exit.transition("exit").duration(ms(DURATION.sample)).remove();
        },
      );

    const moveMs = ms(DURATION.move) / (state.playing ? state.speed : 1); // Play speed scales the motion (§6)
    if (move) {
      points.transition("move").duration(moveMs).ease(d3.easeCubicInOut).attr("transform", position);
      // Glyph changes (circle <-> triangle, predicted color) happen at mid-transition (§6).
      points.transition("glyph").delay(moveMs / 2).duration(0).on("start", function (d) {
        d.drawnPred = d.pred;
        paintGlyph(d3.select(this).select("path"), d, store.state);
      });
    } else {
      points.interrupt("move").interrupt("glyph").attr("transform", position).each((d) => { d.drawnPred = d.pred; });
    }
    if (radiusChanged) symbolCache.clear();

    shown = { key: keyOf(state), layer: state.layer, epoch: state.epoch, n: state.sampleSize };
    renderBadges(state, frame, previous, move ? moveMs : 0);
    restyle(state, { fade: true });
  }

  /** Opacity (filters), emphasis (hover / pin), pin ring + label, legend counts, empty-state message. */
  function restyle(state, { fade = false } = {}) {
    const points = pointsG.selectAll(".pt:not(.exiting)");
    const anyFilter = filtersActive(state);
    const opacity = (d) => {
      if (legendHover !== null) return d.label === legendHover ? 1 : DIM_OPACITY;
      if (!anyFilter) return BASE_OPACITY;
      return passes(d, state) ? 1 : DIM_OPACITY;
    };
    (fade ? points.transition("fade").duration(ms(DURATION.fade)) : points.interrupt("fade"))
      .style("opacity", opacity);

    points.select("path").each(function (d) { paintGlyph(d3.select(this), d, state); });
    points.select(".glyph").selectAll(".pin-ring")
      .data((d) => (d.id === state.pinned ? [d] : []))
      .join("circle")
      .attr("class", "pin-ring")
      .attr("r", radius * EMPHASIS + 3);
    points.select(".glyph").selectAll(".pin-label")
      .data((d) => (d.id === state.pinned ? [d] : []))
      .join("text")
      .attr("class", "pin-label")
      .attr("x", radius * EMPHASIS + 6)
      .attr("dy", "0.35em")
      .text((d) => `#${d.id}`);
    points.filter((d) => isEmphasized(d, state)).raise();

    const visible = ids.filter((id) => passes(nodes.get(id), state)).length;
    empty.attr("hidden", anyFilter && visible === 0 ? null : true);
    renderLegendCounts(state);
  }

  // ------------------------------------------------------------------ badges
  const nhOf = (frame) => cachedNeighborhoodHit(frame, ids, view.config.labels, NH_K);

  /** `tweenMs` > 0 tweens the numbers in sync with the point motion. */
  function renderBadges(state, frame, previous, tweenMs) {
    const { config } = view;
    const e = config.epochIndex(state.epoch);
    const acc = sampleAccuracy(config.preds[e], ids, config.labels);
    const accPrevious = state.mode === "epoch" && e > 0 ? sampleAccuracy(config.preds[e - 1], ids, config.labels) : null;
    setBadge(badgeRows.nh, nhOf(frame), previous ? nhOf(previous) : null, (v) => v.toFixed(2), tweenMs);
    setBadge(badgeRows.acc, acc, accPrevious, (v) => `${(v * 100).toFixed(1)} %`, tweenMs);
  }

  function setBadge(row, value, previous, format, tweenMs) {
    const node = row.value.node();
    const from = node.__value ?? value;
    node.__value = value;
    row.value.transition("tween").duration(tweenMs).ease(d3.easeCubicInOut)
      .tween("text", () => {
        const interpolate = d3.interpolateNumber(from, value);
        return (t) => { node.textContent = format(interpolate(t)); };
      });
    if (previous === null) {
      row.delta.text("");
      return;
    }
    const delta = value - previous;
    const magnitude = format(Math.abs(delta)).replace(" %", "");
    const arrow = Number(magnitude) === 0 ? "=" : delta > 0 ? "▲" : "▼"; // "=" when it rounds to zero
    row.delta.text(`${arrow} ${magnitude}`);
  }

  // ------------------------------------------------------------------ legend (filter by class)
  function renderLegendItems() {
    legendItems.selectAll(".legend-item")
      .data(view.config.classes.map((name, cls) => ({ name, cls })), (d) => `${view.dataset}:${d.cls}`)
      .join((enter) => {
        const item = enter.append("button").attr("type", "button").attr("class", "legend-item")
          .on("click", (event, d) => store.toggleInSet("classFilter", d.cls, { exclusive: !event.shiftKey }))
          .on("dblclick", () => store.set({ classFilter: null }))
          .on("mouseenter", (event, d) => { legendHover = d.cls; restyle(store.state); })
          .on("mouseleave", () => { legendHover = null; restyle(store.state); });
        item.append("span").attr("class", "chip");
        item.append("span").attr("class", "legend-name");
        item.append("span").attr("class", "legend-count");
        return item;
      })
      .attr("title", (d) => `${d.name}: click to isolate, Shift + click to add, double-click to show all`)
      .call((item) => item.select(".chip").style("background", (d) => CLASS_COLORS[d.cls]))
      .call((item) => item.select(".legend-name").text((d) => d.name));
  }

  function renderLegendCounts(state) {
    const counts = new Array(view.config.nClasses).fill(0);
    for (const id of ids) {
      const d = nodes.get(id);
      if (passes(d, state, { ignoreClass: true })) counts[d.label]++;
    }
    legendItems.selectAll(".legend-item")
      .classed("off", (d) => Boolean(state.classFilter && !state.classFilter.has(d.cls)))
      .attr("aria-pressed", (d) => String(Boolean(state.classFilter?.has(d.cls))))
      .select(".legend-count").text((d) => counts[d.cls]);
  }

  // The legend starts in the free space left of the square plot when there is room (so it hides no points), else
  // inside its top-left corner; it can be dragged by its header anywhere in the plot area.
  let legendDragged = false;
  legendHeader.call(d3.drag()
    .container(root.node())
    .filter((event) => !event.button && !event.target.closest("button"))
    .on("drag", (event) => {
      legendDragged = true;
      moveLegend(event.dx, event.dy);
    }));

  const sideSpace = () => Math.max(0, Math.floor((container.clientWidth - size) / 2));

  function placeLegend() {
    const width = legend.node().offsetWidth;
    const outside = sideSpace() >= width + 2 * LEGEND_MARGIN;
    legend.style("left", `${outside ? -(width + LEGEND_MARGIN) : LEGEND_MARGIN}px`).style("top", `${outside ? 0 : LEGEND_MARGIN}px`);
  }

  function moveLegend(dx = 0, dy = 0) {
    const node = legend.node();
    const free = sideSpace();
    const left = Math.max(-free, Math.min(size + free - node.offsetWidth, node.offsetLeft + dx));
    const top = Math.max(0, Math.min(size - node.offsetHeight, node.offsetTop + dy));
    legend.style("left", `${left}px`).style("top", `${top}px`);
  }

  // ------------------------------------------------------------------ layout (square plot, §2)
  function layout() {
    const { width, height } = container.getBoundingClientRect();
    const next = Math.max(0, Math.floor(Math.min(width, height)));
    if (next === size) return;
    size = next;
    root.style("width", `${size}px`).style("height", `${size}px`);
    svg.attr("width", size).attr("height", size);
    frameRect.attr("x", 0.5).attr("y", 0.5).attr("width", Math.max(0, size - 1)).attr("height", Math.max(0, size - 1));
    x.range([PAD, size - PAD]);
    y.range([PAD, size - PAD]);
    zoom.extent([[0, 0], [size, size]]).translateExtent([[0, 0], [size, size]]);
    svg.call(zoom.transform, d3.zoomIdentity);
    if (legendDragged) moveLegend();
    else placeLegend();
    if (shown && isReady(store.state)) {
      const state = store.state;
      shown = { ...shown, layer: null };
      draw(state, {});
    } else {
      render(store.state);
    }
  }
  new ResizeObserver(layout).observe(container);

  store.subscribe(render, ["mode", "layer", "epoch", "sampleSize", "classFilter", "cellFilter", "brushed",
    "hovered", "pinned"]);

  return {
    setConfiguration(next) {
      view = next;
      render(store.state);
    },
  };
}
