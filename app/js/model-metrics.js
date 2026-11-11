// Model metrics panel (VIS_PLAN.md §5.7): confusion matrix and per-class bars at the current epoch, computed on the
// shown sample. Both tween between epochs, so during Play the matrix visibly concentrates on the diagonal.
import { CLASS_COLORS, INK, MOVE_MS, ms, shortName } from "./encoding.js";
import { confusionMatrix, perClassScores } from "./metrics.js";
import { infoIcon, tooltip } from "./ui.js";

const LAYER_NOTE = "Predictions come from the output layer, so this panel does not depend on the selected layer.";
const MATRIX_INFO = "Rows: true class · columns: predicted class, at the current epoch, on the shown sample. "
  + "Color: share of the row (row-normalized). "
  + "Click a cell to filter its points; hide the diagonal to make the errors stand out.";
const METRICS = [{ id: "recall", name: "Recall (class accuracy)" }, { id: "precision", name: "Precision" },
  { id: "f1", name: "F1" }];
const M = { left: 40, top: 30 };                         // matrix header sizes
const BARS = { top: 8, bottom: 26, left: 28, right: 4, min: 90, max: 130 };  // height = 30 % of the panel
const MAX_SHARE = 0.55;     // the panel is sized to its content, using at most this share of the column height
const HEAD_MARGINS = 26;    // px of vertical margin around the two panel-internal headers (CSS .mm-head)
const MIN_TEXT_CELL = 12;   // px: below this cell size only the cells of ≥ 5 % show their count
const SMALL_CELL = 16;      // px: below this cell size the counts use a smaller font
const pct = d3.format(".0%");

/** Builds the panel inside `container`. Returns { setConfiguration(view) }. */
export function initModelMetrics(container, { store }) {
  let view = null;
  let width = 0;
  let hideDiagonal = false;
  const root = d3.select(container).append("div").attr("class", "mm");

  // Confusion matrix
  const head = root.append("div").attr("class", "mm-head");
  head.append("span").attr("class", "mm-title").text("Confusion matrix");
  const layerNote = head.append("span").attr("class", "mm-note");
  layerNote.node().append(infoIcon(LAYER_NOTE));
  head.node().append(infoIcon(MATRIX_INFO));
  const diag = head.append("label").attr("class", "mm-check");
  diag.append("input").attr("type", "checkbox").on("change", (event) => {
    hideDiagonal = event.target.checked;
    render(store.state);
  });
  diag.append("span").text("hide diagonal");
  const matrixSvg = root.append("svg").attr("class", "mm-matrix");
  const corner = matrixSvg.append("text").attr("class", "mm-corner").attr("x", 2).attr("y", 10);
  corner.append("tspan").text("true \\");
  corner.append("tspan").attr("x", 2).attr("dy", 11).text("pred.");
  const rowHeads = matrixSvg.append("g");
  const colHeads = matrixSvg.append("g");
  const cellsG = matrixSvg.append("g");
  const focusRect = matrixSvg.append("rect").attr("class", "mm-focus").style("display", "none");

  // Per-class bars
  const barHead = root.append("div").attr("class", "mm-head");
  barHead.append("span").attr("class", "mm-title").text("Per class");
  const metricSelect = barHead.append("select").attr("class", "control-input mm-select")
    .attr("aria-label", "Per-class metric")
    .on("change", (event) => store.set({ metric: event.target.value }));
  metricSelect.selectAll("option").data(METRICS).join("option").attr("value", (d) => d.id).text((d) => d.name);
  const barKey = barHead.append("span").attr("class", "mm-key");
  barKey.append("span").attr("class", "ghost-swatch");
  barKey.append("span").text("final");
  const barsSvg = root.append("svg").attr("class", "mm-bars");
  const barsAxis = barsSvg.append("g").attr("class", "axis");
  const barsG = barsSvg.append("g");
  const barLabels = barsSvg.append("g").attr("class", "mm-bar-labels");

  const isReady = (state) => view && view.dataset === state.dataset && view.model === state.model
    && state.epoch !== null;

  const panel = container.closest(".panel");
  const column = panel.parentElement;

  /**
   * Body height the matrix + bars may use. The panel is sized to its content (CSS: flex none) so the data-point
   * panel below gets the rest of the column; the content may take at most MAX_SHARE of the column height, or all
   * the room the other panels leave when they are collapsed. Derived from the column, not from the panel itself
   * (whose height follows from this result).
   */
  function availableBodyHeight() {
    const others = [...column.children].filter((p) => p !== panel);
    const gap = parseFloat(getComputedStyle(column).rowGap) || 0;
    const maxPanel = others.every((p) => p.classList.contains("collapsed"))
      ? column.clientHeight - d3.sum(others, (p) => p.offsetHeight + gap)
      : Math.round(column.clientHeight * MAX_SHARE);
    const chrome = panel.querySelector(".panel-title").offsetHeight + 2 + 16;   // borders + body padding
    return maxPanel - chrome;
  }

  /** Confusion counts and scores at an epoch index, on the shown sample. */
  function statsAt(state, e) {
    const { config } = view;
    const counts = confusionMatrix(config.preds[e], config.labels, config.sample(state.sampleSize), config.nClasses);
    return { counts, scores: perClassScores(counts, config.nClasses) };
  }

  function render(state, changed = new Set()) {
    if (!isReady(state)) return;
    width = container.clientWidth - 20;
    const bodyHeight = availableBodyHeight();
    if (width <= 0 || bodyHeight <= 0) return;
    // No scrolling (draft-v2): the matrix and the bars share the panel height. The headers are measured: in a
    // narrow panel they wrap onto two lines.
    const heads = head.node().offsetHeight + barHead.node().offsetHeight + HEAD_MARGINS;
    const barsHeight = Math.max(BARS.min, Math.min(BARS.max, Math.round(bodyHeight * 0.3)));
    const { config } = view;
    const n = config.nClasses;
    const e = config.epochIndex(state.epoch);
    const { counts, scores } = statsAt(state, e);
    const final = statsAt(state, config.epochs.length - 1).scores;
    const duration = changed.has("epoch") ? ms(MOVE_MS) / (state.playing ? state.speed : 1) : 0;
    const names = config.classes;

    layerNote.style("display", state.mode === "layer" ? null : "none");
    metricSelect.property("value", state.metric);

    // ---------------------------------------------------------------- matrix
    const cell = Math.max(8, Math.floor(Math.min((width - M.left) / n, (bodyHeight - heads - barsHeight - M.top) / n)));
    matrixSvg.attr("width", M.left + n * cell).attr("height", M.top + n * cell);
    const rowTotal = d3.range(n).map((t) => d3.sum(d3.range(n), (p) => counts[t * n + p]));
    const cells = [];
    for (let t = 0; t < n; t++) {
      for (let p = 0; p < n; p++) {
        const count = counts[t * n + p];
        cells.push({ t, p, count, frac: rowTotal[t] ? count / rowTotal[t] : 0 });
      }
    }
    const max = hideDiagonal ? d3.max(cells.filter((c) => c.t !== c.p), (c) => c.frac) || 1 : 1;
    const color = d3.scaleSequential(d3.interpolateBlues).domain([0, max]);
    const fill = (c) => (hideDiagonal && c.t === c.p ? "#f3f4f5" : color(Math.min(c.frac, max)));
    const dark = (c) => !(hideDiagonal && c.t === c.p) && Math.min(c.frac, max) / max > 0.55;
    const inFilter = (cls) => !state.classFilter || state.classFilter.has(cls);
    const selected = (c) => state.cellFilter && state.cellFilter.truth === c.t && state.cellFilter.pred === c.p;

    const header = (g) => g.selectAll(".mm-h").data(names.map((name, cls) => ({ name, cls })))
      .join((enter) => {
        const h = enter.append("g").attr("class", "mm-h");
        h.append("circle").attr("r", 3.5);
        h.append("text");
        h.append("title");
        return h;
      })
      .classed("off", (d) => !inFilter(d.cls))
      .call((h) => h.select("circle").attr("fill", (d) => CLASS_COLORS[d.cls]))
      .call((h) => h.select("title").text((d) => d.name));
    header(rowHeads).attr("transform", (d) => `translate(0,${M.top + d.cls * cell + cell / 2})`)
      .call((h) => h.select("circle").attr("cx", 5))
      .call((h) => h.select("text").attr("x", 12).attr("dy", "0.35em").text((d) => shortName(d.name)));
    header(colHeads).attr("transform", (d) => `translate(${M.left + d.cls * cell + cell / 2},0)`)
      .call((h) => h.select("circle").attr("cy", 7))
      .call((h) => h.select("text").attr("y", 24).attr("text-anchor", "middle")
        .text((d) => (cell >= 24 ? shortName(d.name) : "")));   // narrow cells: chips only (name in the tooltip)

    const cellSel = cellsG.selectAll(".mm-cell").data(cells, (c) => `${c.t},${c.p}`)
      .join((enter) => {
        const g = enter.append("g").attr("class", "mm-cell")
          .on("mouseenter", (event, c) => {
            const live = cellAt(c.t, c.p);
            tooltip.show(`true ${names[c.t]}, predicted ${names[c.p]}: ${live.count} (${pct(live.frac)})`, event);
            store.set({ cellHover: { truth: c.t, pred: c.p } });
          })
          .on("mousemove", (event) => tooltip.move(event))
          .on("mouseleave", () => {
            tooltip.hide();
            store.set({ cellHover: null });
          })
          .on("click", (event, c) => {
            const same = store.get("cellFilter")?.truth === c.t && store.get("cellFilter")?.pred === c.p;
            store.set({ cellFilter: same ? null : { truth: c.t, pred: c.p } });
          });
        g.append("rect");
        g.append("text").attr("text-anchor", "middle").attr("dy", "0.35em");
        return g;
      })
      .attr("transform", (c) => `translate(${M.left + c.p * cell},${M.top + c.t * cell})`)
      .classed("selected", selected)
      .classed("dim", (c) => !(inFilter(c.t) || inFilter(c.p)));
    cellSel.select("rect").attr("width", cell - 1).attr("height", cell - 1)
      .transition("fill").duration(duration).attr("fill", fill);
    // Every non-zero count is printed (single errors matter: with 50 images per class, 1 error = 2 %); zeros stay
    // empty. Only when the cells are too small for text are the counts limited to the cells of ≥ 5 %.
    const showCount = (c) => c.count > 0 && (cell >= MIN_TEXT_CELL || c.frac >= 0.05);
    cellSel.select("text").attr("x", (cell - 1) / 2).attr("y", (cell - 1) / 2)
      .style("font-size", cell < SMALL_CELL ? "8px" : null)
      .attr("fill", (c) => (dark(c) ? "#fff" : INK))
      .text((c) => (showCount(c) ? c.count : ""));

    function cellAt(t, p) {
      const live = statsAt(store.state, config.epochIndex(store.state.epoch)).counts;
      const total = d3.sum(d3.range(n), (k) => live[t * n + k]);
      return { count: live[t * n + p], frac: total ? live[t * n + p] / total : 0 };
    }

    // Outline the cell of the hovered (or pinned) observation (§7).
    const focus = state.hovered ?? state.pinned;
    if (focus !== null) {
      const t = config.labels[focus];
      const p = config.preds[e][focus];
      focusRect.style("display", null).attr("x", M.left + p * cell - 1).attr("y", M.top + t * cell - 1)
        .attr("width", cell + 1).attr("height", cell + 1);
    } else {
      focusRect.style("display", "none");
    }

    // ---------------------------------------------------------------- per-class bars
    const values = scores[state.metric];
    const ghost = final[state.metric];
    const bw = width;
    barsSvg.attr("width", bw).attr("height", barsHeight);
    const x = d3.scaleBand().domain(d3.range(n)).range([BARS.left, bw - BARS.right]).padding(0.18);
    const y = d3.scaleLinear().domain([0, 1]).range([barsHeight - BARS.bottom, BARS.top]);
    barsAxis.attr("transform", `translate(${BARS.left},0)`)
      .call(d3.axisLeft(y).ticks(2).tickFormat(d3.format(".1f")).tickSizeOuter(0));
    const bars = barsG.selectAll(".mm-bar").data(d3.range(n)).join((enter) => {
      const g = enter.append("g").attr("class", "mm-bar")
        .on("click", (event, c) => store.toggleInSet("classFilter", c, { exclusive: !event.shiftKey }))
        .on("mouseenter", (event, c) => {
          const s = store.state;
          const name = METRICS.find((m) => m.id === s.metric).name.split(" ")[0].toLowerCase();
          const now = statsAt(s, config.epochIndex(s.epoch)).scores[s.metric][c];
          const fin = statsAt(s, config.epochs.length - 1).scores[s.metric][c];
          const f = (v) => (Number.isNaN(v) ? "n/a" : v.toFixed(2));
          tooltip.show(`${names[c]}: ${name} ${f(now)} (final ${f(fin)})`, event);
        })
        .on("mousemove", (event) => tooltip.move(event))
        .on("mouseleave", () => tooltip.hide());
      g.append("rect").attr("class", "mm-hit");
      g.append("rect").attr("class", "mm-bar-fill");
      g.append("rect").attr("class", "mm-ghost");
      return g;
    }).classed("off", (c) => !inFilter(c));
    const v = (c) => (Number.isNaN(values[c]) ? 0 : values[c]);
    bars.select(".mm-hit").attr("x", (c) => x(c)).attr("width", x.bandwidth())
      .attr("y", BARS.top).attr("height", y(0) - BARS.top);
    bars.select(".mm-bar-fill").attr("x", (c) => x(c)).attr("width", x.bandwidth()).attr("fill", (c) => CLASS_COLORS[c])
      .transition("bar").duration(duration).ease(d3.easeCubicInOut)
      .attr("y", (c) => y(v(c))).attr("height", (c) => y(0) - y(v(c)));
    // Ghost bar: thin outline of the final-epoch value ("final", not "target": an earlier epoch can be higher).
    bars.select(".mm-ghost").attr("x", (c) => x(c) - 0.5).attr("width", x.bandwidth() + 1)
      .attr("y", (c) => y(Number.isNaN(ghost[c]) ? 0 : ghost[c])).attr("height", 1.5);
    barLabels.selectAll("text").data(d3.range(n)).join("text")
      .attr("x", (c) => x(c) + x.bandwidth() / 2).attr("y", barsHeight - BARS.bottom + 12)
      .attr("text-anchor", "middle").classed("off", (c) => !inFilter(c))
      .text((c) => shortName(names[c]));
  }

  // The column (window size) and its panels (collapsing the data point frees room); rendering itself changes the
  // panel height, which re-renders once with the same result.
  const resize = new ResizeObserver(() => render(store.state));
  for (const el of [container, column, ...column.children]) resize.observe(el);
  store.subscribe(render, ["epoch", "sampleSize", "classFilter", "cellFilter", "hovered", "pinned", "mode", "metric"]);

  return {
    setConfiguration(next) {
      view = next;
      render(store.state);
    },
  };
}
