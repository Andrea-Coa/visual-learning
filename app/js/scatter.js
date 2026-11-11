// Embeddings view (VIS_PLAN.md §5.4; encodings §4; animations §6): scatter mode and path view.
// Every observation is an SVG group keyed by its id, so selections survive epoch/layer changes and the points can
// move between frames. Positions live in "base" pixels; zoom/pan transforms the whole layer and scales every glyph
// back by 1/k, so marks keep a constant size while zooming. The path view (`All`) draws the trails of paths.js on top.
// Layout as in drafts/draft-v2.png: the plot fills the panel, legend inside at the bottom-left, steppers (an overlay
// owned by steppers.js) at the bottom-right; the data square (same scale on both axes) stays clear of the steppers.
import { frameOf } from "./data.js";
import { CLASS_COLORS, INK, CORRECT_GRAY, MOVE_MS, learnedColor, ms, pointStyle } from "./encoding.js";
import { cachedNeighborhoodHit, sampleAccuracy } from "./metrics.js";
import { PathLayer, straightTrail } from "./paths.js";
import { infoIcon, tooltip } from "./ui.js";

const BASE_OPACITY = 0.75;
const DIM_OPACITY = 0.08;
const PATH_POINT_OPACITY = 0.15;  // points behind the trails in the path view (§6)
const EMPHASIS = 1.8;      // radius factor of the hovered / pinned point
const PAD = 16;            // px between the frame and the outermost points
const MAX_ZOOM = 8;
const HOVER_DELAY = 80;    // ms before the tooltip appears
const TRAIL_HIT_PX = 8;    // hover distance to a trail in the path view
const NH_K = 6;
const LEGEND_MARGIN = 8;   // px between the legend and the plot frame
const LEGEND_MIN_PLOT = 480; // plot height (px) below which the legend starts collapsed
const OVERLAY_MARGIN = 10; // px around the steppers overlay (keep in sync with #steppers in CSS)
const COMET = [0.5, 0.32, 0.16];  // opacity of the pinned point's last positions during Play (§6)
const DURATION = { swapOut: 200, swapIn: 200, sample: 300, fade: 250, reset: 400 };

const NH_INFO = "NH (neighborhood hit): share of each point's 6 nearest neighbors in this projection that have "
  + "its class. Sample acc: accuracy of the shown points at this epoch. +/−: change from the previous step "
  + "(previous epoch or layer).";

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
export function passes(d, state, { ignoreClass = false } = {}) {
  if (!ignoreClass && state.classFilter && !state.classFilter.has(d.label)) return false;
  if (state.cellFilter && (d.label !== state.cellFilter.truth || d.pred !== state.cellFilter.pred)) return false;
  if (state.brushed && !state.brushed.has(d.id)) return false;
  return true;
}

export const filtersActive = (state) => Boolean(state.classFilter || state.cellFilter || state.brushed);

/**
 * Builds the embeddings view inside `container` (#plot) and its toolbar inside `tools`; `data` loads the trails;
 * `overlay` (the steppers) is kept clear of the data. Returns { setConfiguration(view) }, called by main.js.
 */
export function initScatter(container, { store, data, tools, overlay }) {
  let view = null;              // { index, config, coords, dataset, model, projection, spriteUrl }
  let shown = null;             // what is drawn: { key, layer, epoch, n }
  let width = 0;                // plot size in px (the whole panel body)
  let height = 0;
  let bandHeight = 0;           // steppers overlay (px), kept clear of the data
  let bandWidth = 0;
  let radius = 3;
  let transform = d3.zoomIdentity;
  let brushMode = false;
  let legendHover = null;       // class highlighted while hovering its legend entry
  let swapping = false;
  let hoverTimer = null;
  let lastPointer = null;
  let ids = [];                 // observation ids currently drawn (the sample)
  let comet = [];               // last positions of the pinned point during Play
  let pathToken = 0;
  let drawnPaths = null;        // { geometry, strength } of the trails on screen
  const nodes = new Map();      // id -> datum { id, label, learnedAt, x, y, pred, drawnPred }
  const geometries = new Map(); // geometry key -> { key, labels, frames, nodePaths, trails: Map strength -> Map }
  const x = d3.scaleLinear().domain([0, 1]);
  const y = d3.scaleLinear().domain([0, 1]);

  // ------------------------------------------------------------------ DOM
  const root = d3.select(container).append("div").attr("class", "scatter");
  const svg = root.append("svg").attr("class", "scatter-svg").attr("role", "img")
    .attr("aria-label", "Scatterplot of the projected observations");
  const frameRect = svg.append("rect").attr("class", "scatter-frame");
  const zoomLayer = svg.append("g").attr("class", "zoom-layer");
  const pointsG = zoomLayer.append("g").attr("class", "points");
  const glyphScale = () => `scale(${1 / transform.k})`;
  const paths = new PathLayer(zoomLayer, glyphScale);
  const cometG = zoomLayer.append("g").attr("class", "comet");
  const trajectoryG = zoomLayer.append("g").attr("class", "trajectory");
  const lassoPath = svg.append("path").attr("class", "lasso").style("display", "none");

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
  legendHeader.append("span").attr("class", "legend-title").text("Legend");
  legendHeader.append("button").attr("type", "button").attr("class", "legend-toggle")
    .attr("aria-label", "Collapse legend")
    .on("click", () => {
      legendToggled = true;
      setLegendCollapsed(!legend.classed("collapsed"));
    });
  let legendToggled = false;   // once the user opens / closes the legend, the layout no longer decides
  function setLegendCollapsed(collapsed) {
    legend.classed("collapsed", collapsed);
    legendHeader.select(".legend-toggle").attr("aria-label", collapsed ? "Expand legend" : "Collapse legend");
  }
  const legendItems = legend.append("div").attr("class", "legend-items");
  const legendKey = legend.append("div").attr("class", "legend-key");

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
    .attr("title", (d) => (d.id === "brush" ? "Draw a freeform shape around points to select them (or Shift + drag)"
      : "Drag to pan, scroll to zoom"))
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
    const style = pointStyle(state.colorBy, { label: d.label, pred: d.drawnPred, learnedAt: d.learnedAt },
      view.config.epochs.length);
    path.attr("d", symbolPath(style.wrong, emphasized ? EMPHASIS : 1))
      .attr("fill", style.fill)
      .classed("hollow", style.hollow)
      .classed("emph", emphasized);
  }

  const position = (d) => `translate(${d.x},${d.y})`;

  // ------------------------------------------------------------------ zoom, pan, brush
  const zoom = d3.zoom()
    .scaleExtent([1, MAX_ZOOM])
    .filter((event) => (!event.ctrlKey || event.type === "wheel") && !event.button
      && (event.type === "wheel" || (!brushMode && !event.shiftKey)))
    .on("zoom", (event) => {
      transform = event.transform;
      zoomLayer.attr("transform", transform);
      zoomLayer.selectAll(".glyph").attr("transform", glyphScale());
      tooltip.hide();
    });
  svg.call(zoom).on("dblclick.zoom", null);

  // Freeform brush (lasso): the pointer path is closed into a polygon; the points inside it are brushed.
  let lasso = [];               // polygon vertices in screen px (svg coordinates)
  const LASSO_STEP = 3;         // px between recorded vertices
  const lassoD = () => `M${lasso.map((p) => p.join(",")).join("L")}Z`;
  const clampToPlot = (event) => [Math.max(0, Math.min(width, event.x)), Math.max(0, Math.min(height, event.y))];
  svg.call(d3.drag()
    .container(svg.node())
    .filter((event) => (brushMode || event.shiftKey) && !event.button)
    .on("start", (event) => {
      lasso = [clampToPlot(event)];
      tooltip.hide();
    })
    .on("drag", (event) => {
      const p = clampToPlot(event);
      const last = lasso[lasso.length - 1];
      if (Math.hypot(p[0] - last[0], p[1] - last[1]) < LASSO_STEP) return;
      lasso.push(p);
      lassoPath.style("display", null).attr("d", lassoD());
    })
    .on("end", () => {
      lassoPath.style("display", "none");
      const [x0, x1] = d3.extent(lasso, (p) => p[0]);
      const [y0, y1] = d3.extent(lasso, (p) => p[1]);
      if (lasso.length < 3 || (x1 - x0 < 4 && y1 - y0 < 4)) return; // a click, not a brush
      const polygon = lasso.map((p) => transform.invert(p));       // into base px, like the point positions
      const inside = ids.filter((id) => {
        // In the path view the visible marks are the trail ends, so those are brushed.
        const point = paths.active ? paths.end(id) : [nodes.get(id).x, nodes.get(id).y];
        return d3.polygonContains(polygon, point);
      });
      lasso = [];
      store.set({ brushed: inside.length ? new Set(inside) : null }); // an empty brush clears itself (§9)
    }));

  svg.on("click", (event) => {
    const point = event.target.closest(".pt");
    const id = point ? d3.select(point).datum().id : paths.active ? store.get("hovered") : null;
    if (id !== null) store.set({ pinned: store.get("pinned") === id ? null : id });
    else if (store.get("brushed")) store.set({ brushed: null });
  });

  // Hover in the path view: the trail closest to the cursor (points are not interactive there).
  svg.on("mousemove.pointer", (event) => {
    lastPointer = event;
    if (!paths.active) return;
    const [px, py] = transform.invert(d3.pointer(event, svg.node()));
    const id = paths.nearest(px, py, TRAIL_HIT_PX / transform.k);
    if (id !== store.get("hovered")) {
      if (id === null) leaveHover();
      else enterHover(id);
    }
    tooltip.move(event);
  });
  svg.on("mouseleave.pointer", () => { if (paths.active) leaveHover(); });

  document.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || event.target.closest?.("input, select, textarea")) return;
    if (store.get("pinned") !== null) store.set({ pinned: null });
    else if (store.get("brushed")) store.set({ brushed: null });
  });

  // ------------------------------------------------------------------ hover + tooltip
  function enterHover(id) {
    store.set({ hovered: id });
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(() => tooltip.show(tooltipContent(id), lastPointer), HOVER_DELAY);
  }
  function leaveHover() {
    clearTimeout(hoverTimer);
    tooltip.hide();
    if (store.get("hovered") !== null) store.set({ hovered: null });
  }

  function onEnter(event, d) {
    lastPointer = event;
    enterHover(d.id);
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

  function tooltipContent(id) {
    const { index, config, dataset, spriteUrl } = view;
    const state = store.state;
    const label = config.labels[id];
    const pred = config.pred(state.epoch, id);
    const conf = config.probsOf(state.epoch, id)[pred];
    const box = d3.create("div").attr("class", "tip-point");
    const thumb = box.append("div").attr("class", "thumb tip-thumb");
    Object.assign(thumb.node().style, index.sprite(dataset, id, 64, spriteUrl));
    const text = box.append("div").attr("class", "tip-text");
    text.append("div").attr("class", "tip-id").text(`#${id}`);
    const row = (name, cls, value) => {
      const line = text.append("div").attr("class", "tip-row");
      line.append("span").attr("class", "chip").style("background", CLASS_COLORS[cls]);
      line.append("span").text(`${name}: ${config.classes[cls]}${value ?? ""}`);
    };
    row("true", label);
    row(pred === label ? "predicted" : "▲ predicted", pred, ` (conf ${conf.toFixed(2)})`);
    return box.node();
  }

  // ------------------------------------------------------------------ rendering
  const keyOf = (state) => `${state.dataset}/${state.model}/${state.projection}/${state.mode}`;
  const PATH_KEYS = ["showAll", "bundling", "layer", "epoch", "sampleSize", "classFilter", "cellFilter", "brushed"];
  const TRAJECTORY_KEYS = ["pinned", "hovered", "trajectory", "showAll", "layer", "epoch", "colorBy"];

  function isReady(state) {
    if (!view || state.layer === null || state.epoch === null) return false;
    const { config } = view;
    return view.dataset === state.dataset && view.model === state.model && view.projection === state.projection
      && config.t2Layers.includes(state.layer) && config.epochs.includes(state.epoch);
  }

  function render(state, changed = new Set()) {
    if (swapping || !isReady(state) || width === 0) return;
    if (!shown || shown.key !== keyOf(state)) {
      swap();
      return;
    }
    const stepChanged = shown.layer !== state.layer || shown.epoch !== state.epoch;
    const sampleChanged = shown.n !== state.sampleSize;
    if (stepChanged || sampleChanged) {
      draw(state, { move: stepChanged, sample: sampleChanged });
    } else {
      const fadeKeys = ["classFilter", "cellFilter", "brushed", "showAll", "cellHover"];
      restyle(state, { fade: fadeKeys.some((k) => changed.has(k)) });
    }
    if (changed.has("colorBy")) renderLegend(state);
    if (PATH_KEYS.some((k) => changed.has(k))) updatePaths(state);
    if (TRAJECTORY_KEYS.some((k) => changed.has(k))) renderTrajectory(state);
    if (changed.has("pinned") || (changed.has("playing") && !state.playing)) clearComet();
  }

  /** Dataset / model / projection / mode change: positions are not comparable, so fade out and back in (§6). */
  function swap() {
    swapping = true;
    tooltip.hide();
    zoomLayer.transition("swap").duration(shown ? ms(DURATION.swapOut) : 0).style("opacity", 0)
      .on("end", () => {
        swapping = false;
        pointsG.selectAll(".pt").remove();
        paths.clear();
        drawnPaths = null;
        nodes.clear();
        geometries.clear();
        clearComet();
        shown = null;
        const state = store.state;
        if (!isReady(state)) return;
        svg.call(zoom.transform, d3.zoomIdentity);
        renderLegend(state);
        if (!legendDragged) placeLegend(); // class names change the legend width (e.g. CIFAR-10)
        draw(state, {});
        updatePaths(state);
        renderTrajectory(state);
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
    if (nextRadius !== radius) symbolCache.clear();
    radius = nextRadius;

    const pinned = state.pinned !== null ? nodes.get(state.pinned) : null;
    if (move && state.playing && pinned) comet = [[pinned.x, pinned.y], ...comet].slice(0, COMET.length);

    for (const id of ids) {
      let d = nodes.get(id);
      if (!d) nodes.set(id, (d = { id, label: config.labels[id], learnedAt: config.learnedAt[id] }));
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

    const moveMs = ms(MOVE_MS) / (state.playing ? state.speed : 1); // Play speed scales the motion (§6)
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

    shown = { key: keyOf(state), layer: state.layer, epoch: state.epoch, n: state.sampleSize };
    renderBadges(state, frame, previous, move ? moveMs : 0);
    restyle(state, { fade: true });
    renderComet(moveMs);
  }

  /** Opacity (filters, highlights, path view), emphasis (hover / pin), pin ring + label, legend, empty state. */
  function restyle(state, { fade = false } = {}) {
    const points = pointsG.selectAll(".pt:not(.exiting)");
    const anyFilter = filtersActive(state);
    const cell = state.cellHover;
    const highlight = legendHover !== null ? (d) => d.label === legendHover
      : cell ? (d) => d.label === cell.truth && d.pred === cell.pred : null;
    const opacity = (d) => {
      let value;
      if (highlight) value = highlight(d) ? 1 : DIM_OPACITY;
      else if (!anyFilter) value = BASE_OPACITY;
      else value = passes(d, state) ? 1 : DIM_OPACITY;
      return state.showAll ? Math.min(value, PATH_POINT_OPACITY) : value;
    };
    pointsG.classed("inert", state.showAll);
    (fade ? points.transition("fade").duration(ms(DURATION.fade)) : points.interrupt("fade"))
      .style("opacity", opacity);

    points.select("path").each(function (d) { paintGlyph(d3.select(this), d, state); });
    const pinnedHere = (d) => (d.id === state.pinned && !state.showAll ? [d] : []);
    points.select(".glyph").selectAll(".pin-ring").data(pinnedHere).join("circle")
      .attr("class", "pin-ring")
      .attr("r", radius * EMPHASIS + 3);
    points.select(".glyph").selectAll(".pin-label").data(pinnedHere).join("text")
      .attr("class", "pin-label")
      .attr("x", radius * EMPHASIS + 6)
      .attr("dy", "0.35em")
      .text((d) => `#${d.id}`);
    points.filter((d) => isEmphasized(d, state)).raise();

    const visible = ids.filter((id) => passes(nodes.get(id), state)).length;
    empty.attr("hidden", anyFilter && visible === 0 ? null : true);
    renderLegendCounts(state);
  }

  // ------------------------------------------------------------------ path view (trails)
  /** Frames along the active dimension, their labels and the index of the current step. */
  function stepFrames(state) {
    const { config, coords } = view;
    if (state.mode === "epoch") {
      return {
        frames: config.epochs.map((epoch) => frameOf(config, coords, { mode: "epoch", layer: state.layer, epoch })),
        labels: config.epochs.map((e) => `e${e}`),
        current: config.epochIndex(state.epoch),
      };
    }
    return {
      frames: config.t2Layers.map((layer) => frameOf(config, coords, { mode: "layer", layer, epoch: state.epoch })),
      labels: config.layers.map((l) => l.label),
      current: config.t2Layers.indexOf(state.layer),
    };
  }

  const geometryKey = (state) => `${keyOf(state)}|${state.mode === "epoch" ? state.layer : state.epoch}|${width}x${height}`;

  /** Unbundled step positions (base px) of every observation along the active dimension (cached). */
  function geometryFor(state) {
    const key = geometryKey(state);
    if (!geometries.has(key)) {
      const { frames, labels } = stepFrames(state);
      const nodePaths = new Map();
      for (let id = 0; id < view.config.n; id++) {
        const p = new Float32Array(frames.length * 2);
        frames.forEach((f, s) => { p[2 * s] = x(f[2 * id]); p[2 * s + 1] = y(f[2 * id + 1]); });
        nodePaths.set(id, p);
      }
      geometries.set(key, { key, labels, nodePaths, trails: new Map() });
    }
    return geometries.get(key);
  }

  const bundledAvailable = (state) => view.index.model(view.dataset, view.model).trails.includes(view.projection)
    && (state.mode === "layer" || view.config.t1Layers.includes(state.layer));

  /** Trails of a geometry at a strength: "none" = straight segments; else the bundled t-SNE file (loaded once). */
  async function trailsFor(state, geometry, strength) {
    if (geometry.trails.has(strength)) return geometry.trails.get(strength);
    let trails;
    if (strength === "none") {
      trails = new Map([...geometry.nodePaths].map(([id, p]) => [id, straightTrail(p, 4)]));
    } else {
      const file = await data.trails(view.dataset, view.model, state.mode, state.mode === "epoch" ? state.layer : state.epoch);
      trails = new Map();
      for (let id = 0; id < view.config.n; id++) {
        const t = file.trail(strength, id);
        const p = new Float32Array(t.length);
        for (let i = 0; i < t.length; i += 2) { p[i] = x(t[i]); p[i + 1] = y(t[i + 1]); }
        trails.set(id, p);
      }
    }
    geometry.trails.set(strength, trails);
    return trails;
  }

  async function updatePaths(state) {
    const token = ++pathToken;
    if (!state.showAll || !isReady(state)) {
      paths.hide();
      drawnPaths = null;
      return;
    }
    const geometry = geometryFor(state);
    const strength = state.bundling !== "none" && bundledAvailable(state) ? state.bundling : "none";
    let trails;
    try {
      trails = await trailsFor(state, geometry, strength);
    } catch (err) {
      console.error(err);
      return;
    }
    const now = store.state;
    if (token !== pathToken || !now.showAll || geometryKey(now) !== geometry.key) return;
    let animate = null;
    if (!drawnPaths || drawnPaths.geometry !== geometry.key) animate = "in";
    else if (drawnPaths.strength !== strength) animate = "morph";
    paths.render({
      trails, ids, labels: view.config.labels, k: 4, steps: geometry.labels.length,
      pass: (id) => passes(nodes.get(id), now), anyFilter: filtersActive(now),
    }, animate);
    drawnPaths = { geometry: geometry.key, strength };
    trajectoryG.raise();
  }

  /**
   * The unbundled trajectory (§4.3): pinned point (if "Show trajectory" is on) and, in the path view, the hovered
   * one. The line is the selection color (near-black with a white halo, §4.1: selection is never a hue); every step
   * carries the point's own glyph at that step (circle = correct, triangle = wrong, filled by the "Color by" mode),
   * so in T1 the glyph changes at the epoch where the observation is learned. The current step is enlarged.
   */
  function renderTrajectory(state) {
    if (!isReady(state)) return;
    const { config } = view;
    const shownIds = [];
    if (state.pinned !== null && state.trajectory) shownIds.push(state.pinned);
    if (state.showAll && state.hovered !== null && state.hovered !== state.pinned) shownIds.push(state.hovered);
    const geometry = shownIds.length ? geometryFor(state) : null;
    const { current } = shownIds.length ? stepFrames(state) : { current: -1 };
    // Prediction at a step: T1 steps are epochs; in T2 every layer shares the current epoch's prediction.
    const epochNow = config.epochIndex(state.epoch);
    const predAt = (id, s) => config.preds[state.mode === "epoch" ? s : epochNow][id];
    const lineOf = (id) => {
      const p = geometry.nodePaths.get(id);
      return d3.line()(d3.range(geometry.labels.length).map((s) => [p[2 * s], p[2 * s + 1]]));
    };

    const traj = trajectoryG.selectAll(".traj").data(shownIds, (id) => id).join((enter) => {
      const g = enter.append("g").attr("class", "traj");
      g.append("path").attr("class", "traj-halo");
      g.append("path").attr("class", "traj-line");
      return g;
    });
    traj.select(".traj-halo").attr("d", lineOf);
    traj.select(".traj-line").attr("d", lineOf);
    traj.selectAll(".traj-step")
      .data((id) => geometry.labels.map((label, s) => ({ id, s, label, p: geometry.nodePaths.get(id) })))
      .join((enter) => {
        const g = enter.append("g").attr("class", "traj-step");
        const glyph = g.append("g").attr("class", "glyph");
        glyph.append("path");
        glyph.append("text").attr("x", 7).attr("y", -7);
        return g;
      })
      .attr("transform", (d) => `translate(${d.p[2 * d.s]},${d.p[2 * d.s + 1]})`)
      .classed("current", (d) => d.s === current)
      .call((g) => g.select(".glyph").attr("transform", glyphScale()))
      .call((g) => g.select("path").each(function (d) {
        const style = pointStyle(state.colorBy,
          { label: config.labels[d.id], pred: predAt(d.id, d.s), learnedAt: config.learnedAt[d.id] }, config.epochs.length);
        d3.select(this).attr("d", symbolPath(style.wrong, d.s === current ? EMPHASIS : 1.2))
          .attr("fill", style.hollow ? "none" : style.fill);
      }))
      .call((g) => g.select("text").text((d) => d.label));
    trajectoryG.raise();
  }

  /** The pinned point's last positions during Play, in the selection color (like its trajectory). */
  function renderComet(moveMs) {
    cometG.selectAll(".comet-dot").data(comet).join((enter) => {
      const g = enter.append("g").attr("class", "comet-dot");
      g.append("g").attr("class", "glyph").append("circle").attr("r", radius);
      return g;
    })
      .attr("transform", (p) => `translate(${p[0]},${p[1]})`)
      .call((g) => g.select(".glyph").attr("transform", glyphScale()))
      .style("fill", INK)
      .transition("comet").duration(moveMs)
      .style("opacity", (p, i) => COMET[i]);
  }

  function clearComet() {
    comet = [];
    cometG.selectAll("*").remove();
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
    // Signed change from the previous step (+0.06 / −0.04 / ±0.00). No ▲ ▼: triangles mean "misclassified".
    const delta = value - previous;
    const magnitude = format(Math.abs(delta)).replace(" %", "");
    const sign = Number(magnitude) === 0 ? "±" : delta > 0 ? "+" : "−"; // "±" when it rounds to zero
    row.delta.text(`${sign}${magnitude}`)
      .attr("title", "change from the previous step");
  }

  // ------------------------------------------------------------------ legend (depends on "Color by")
  /** Legend entries: { key, name, swatch: { shape, color }, cls (class filter), count(d) }. */
  function legendEntries(state) {
    const { config } = view;
    if (state.colorBy === "correctness") {
      return [
        { key: "ok", name: "correct", swatch: { shape: "circle", color: CORRECT_GRAY }, count: (d) => d.pred === d.label },
        { key: "wrong", name: "misclassified", swatch: { shape: "triangle", color: INK }, count: (d) => d.pred !== d.label },
      ];
    }
    if (state.colorBy === "learnedAt") {
      const n = config.epochs.length;
      return config.epochs.map((epoch, i) => ({
        key: `l${i}`, name: `epoch ${epoch}`, swatch: { shape: "square", color: learnedColor(i, n) },
        count: (d) => d.learnedAt === i,
      })).concat({ key: "never", name: "never learned", swatch: { shape: "hollow", color: INK }, count: (d) => d.learnedAt < 0 });
    }
    return config.classes.map((name, cls) => ({
      key: `c${cls}`, name, cls, swatch: { shape: "circle", color: CLASS_COLORS[cls] }, count: (d) => d.label === cls,
    }));
  }

  const SWATCH = { circle: d3.symbolCircle, triangle: d3.symbolTriangle, hollow: d3.symbolTriangle, square: d3.symbolSquare };

  function renderLegend(state) {
    const entries = legendEntries(state);
    const byClass = state.colorBy === "class";
    legend.classed("by-class", byClass);
    legend.select(".legend-title").text({ class: "Legend", correctness: "Correctness", learnedAt: "Learned at" }[state.colorBy]);
    legendItems.selectAll(".legend-item").data(entries, (d) => `${view.dataset}:${state.colorBy}:${d.key}`)
      .join((enter) => {
        const item = enter.append("button").attr("type", "button").attr("class", "legend-item")
          .on("click", (event, d) => {
            if (d.cls !== undefined) store.toggleInSet("classFilter", d.cls, { exclusive: !event.shiftKey });
          })
          .on("dblclick", (event, d) => { if (d.cls !== undefined) store.set({ classFilter: null }); })
          .on("mouseenter", (event, d) => { if (d.cls !== undefined) { legendHover = d.cls; restyle(store.state); } })
          .on("mouseleave", () => { if (legendHover !== null) { legendHover = null; restyle(store.state); } });
        item.append("svg").attr("class", "swatch").attr("width", 10).attr("height", 10).attr("viewBox", "-5 -5 10 10")
          .append("path");
        item.append("span").attr("class", "legend-name");
        item.append("span").attr("class", "legend-count");
        return item;
      })
      .property("disabled", (d) => d.cls === undefined)
      .attr("title", (d) => (d.cls !== undefined
        ? `${d.name}: click to isolate, Shift + click to add, double-click to show all` : null))
      .call((item) => item.select(".swatch path")
        .attr("d", (d) => d3.symbol(SWATCH[d.swatch.shape], d.swatch.shape === "square" ? 64 : 50)())
        .attr("fill", (d) => (d.swatch.shape === "hollow" ? "none" : d.swatch.color))
        .attr("stroke", (d) => (d.swatch.shape === "hollow" ? d.swatch.color : "none")))
      .call((item) => item.select(".legend-name").text((d) => d.name));

    legendKey.selectAll("*").remove();
    if (byClass) {
      legendKey.append("svg").attr("width", 12).attr("height", 12).attr("viewBox", "-6 -6 12 12")
        .append("path").attr("d", d3.symbol(d3.symbolTriangle, 40)());
      legendKey.append("span").text("misclassified (predicted class color)");
    }
    legendKey.style("display", byClass ? null : "none");
    if (shown) renderLegendCounts(state);
  }

  function renderLegendCounts(state) {
    const visible = ids.map((id) => nodes.get(id)).filter((d) => passes(d, state, { ignoreClass: true }));
    legendItems.selectAll(".legend-item")
      .classed("off", (d) => d.cls !== undefined && Boolean(state.classFilter && !state.classFilter.has(d.cls)))
      .attr("aria-pressed", (d) => (d.cls !== undefined ? String(Boolean(state.classFilter?.has(d.cls))) : null))
      .select(".legend-count").text((d) => visible.filter(d.count).length);
  }

  // The legend starts at the bottom-left of the plot (draft-v2); it can be collapsed and dragged by its header.
  let legendDragged = false;
  legendHeader.call(d3.drag()
    .container(root.node())
    .filter((event) => !event.button && !event.target.closest("button"))
    .on("drag", (event) => {
      legendDragged = true;
      moveLegend(event.dx, event.dy);
    }));

  /** Default position, anchored at the bottom so that collapsing keeps the legend in its corner; on short plots
   * (e.g. a 720 px projector) it starts collapsed so it hides no points. */
  function placeLegend() {
    if (!legendToggled) setLegendCollapsed(height < LEGEND_MIN_PLOT);
    // Bottom-left (draft-v2), unless the steppers (in their default bottom-right corner) reach that far left: then
    // top-left. Once the user has moved the steppers, their position says nothing about the free corners.
    const avoidSteppers = overlay && !overlay.classList.contains("moved");
    const room = avoidSteppers ? overlay.getBoundingClientRect().left - container.getBoundingClientRect().left : width;
    const fitsBottom = room >= legend.node().offsetWidth + 2 * LEGEND_MARGIN;
    legend.style("left", `${LEGEND_MARGIN}px`)
      .style("top", fitsBottom ? "auto" : `${LEGEND_MARGIN}px`)
      .style("bottom", fitsBottom ? `${LEGEND_MARGIN}px` : "auto");
  }

  function moveLegend(dx = 0, dy = 0) {
    const node = legend.node();
    const left = Math.max(0, Math.min(width - node.offsetWidth, node.offsetLeft + dx));
    const top = Math.max(0, Math.min(height - node.offsetHeight, node.offsetTop + dy));
    legend.style("left", `${left}px`).style("top", `${top}px`).style("bottom", "auto");
  }

  // ------------------------------------------------------------------ layout (§2: same scale on both axes)
  function layout() {
    const box = container.getBoundingClientRect();
    const w = Math.max(0, Math.floor(box.width));
    const h = Math.max(0, Math.floor(box.height));
    const band = overlay ? overlay.offsetHeight + OVERLAY_MARGIN : 0;
    const overlayWidth = overlay ? overlay.offsetWidth : 0;   // the legend placement depends on it
    if (w === width && h === height && band === bandHeight && overlayWidth === bandWidth) return;
    width = w;
    height = h;
    bandHeight = band;
    bandWidth = overlayWidth;
    svg.attr("width", width).attr("height", height);
    frameRect.attr("x", 0.5).attr("y", 0.5).attr("width", Math.max(0, width - 1)).attr("height", Math.max(0, height - 1));
    // The data square: as large as possible above the steppers band, centered horizontally.
    const side = Math.max(0, Math.min(width, height - band) - 2 * PAD);
    const x0 = (width - side) / 2;
    const y0 = PAD + Math.max(0, (height - band - 2 * PAD - side) / 2);
    x.range([x0, x0 + side]);
    y.range([y0, y0 + side]);
    zoom.extent([[0, 0], [width, height]]).translateExtent([[0, 0], [width, height]]);
    svg.call(zoom.transform, d3.zoomIdentity);
    if (legendDragged) moveLegend();
    else placeLegend();
    geometries.clear(); // trails are cached in base px
    if (shown && isReady(store.state)) {
      const state = store.state;
      shown = { ...shown, layer: null };
      clearComet();
      draw(state, {});
      drawnPaths = null;
      paths.clear();
      updatePaths(state);
      renderTrajectory(state);
    } else {
      render(store.state);
    }
  }
  const resizeObserver = new ResizeObserver(layout);
  resizeObserver.observe(container);
  if (overlay) resizeObserver.observe(overlay);

  store.subscribe(render, ["mode", "layer", "epoch", "sampleSize", "classFilter", "cellFilter", "brushed",
    "hovered", "pinned", "colorBy", "showAll", "bundling", "cellHover", "trajectory", "playing"]);

  return {
    setConfiguration(next) {
      view = next;
      render(store.state);
    },
  };
}
