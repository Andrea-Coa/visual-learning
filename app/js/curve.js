// Learning curve (VIS_PLAN.md §5.3): train (dashed) and test (solid) accuracy or loss for every epoch, snapshot
// epochs as ticks, a current-epoch marker that moves with the state, hover crosshair and click-to-select.
import { INK, MOVE_MS, ms } from "./encoding.js";
import { tooltip } from "./ui.js";

const MARGIN = { top: 10, right: 34, bottom: 22, left: 34 };
const TRAIN_GRAY = "#8a8f95";
const SERIES = [
  { key: "test", name: "test", color: INK, dash: null },
  { key: "train", name: "train", color: TRAIN_GRAY, dash: "4 3" },
];

/** Builds the chart in `container` and the [Acc | Loss] toggle in `tools`. Returns { setConfiguration(view) }. */
export function initCurve(container, { store, tools }) {
  let view = null;
  let width = 0;
  let height = 0;
  const x = d3.scaleLinear();
  const y = d3.scaleLinear();

  const toggle = d3.select(tools).append("div").attr("class", "seg").attr("role", "group")
    .attr("aria-label", "Curve metric")
    .selectAll("button").data([{ id: "acc", name: "Acc" }, { id: "loss", name: "Loss" }]).join("button")
    .attr("type", "button").text((d) => d.name)
    .on("click", (event, d) => store.set({ curve: d.id }));

  const svg = d3.select(container).append("svg").attr("class", "curve-svg");
  const plot = svg.append("g");
  const xAxisG = plot.append("g").attr("class", "axis");
  const yAxisG = plot.append("g").attr("class", "axis");
  const snapshotsG = plot.append("g").attr("class", "snapshot-ticks");
  const linesG = plot.append("g");
  const labelsG = plot.append("g").attr("class", "end-labels");
  const marker = plot.append("g").attr("class", "curve-marker");
  marker.append("line").attr("class", "marker-rule");
  const hover = plot.append("g").attr("class", "curve-hover").style("display", "none");
  hover.append("line");
  const overlay = plot.append("rect").attr("class", "curve-overlay");

  const series = (key, state) => {
    const h = view.config.history;
    const field = `${key}${state.curve === "acc" ? "Acc" : "Loss"}`;
    return h.epoch.map((epoch, i) => ({ epoch, value: h[field][i] }));
  };
  const format = (state) => (state.curve === "acc" ? (v) => `${(v * 100).toFixed(1)} %` : (v) => v.toFixed(3));

  function isReady(state) {
    return view && view.dataset === state.dataset && view.model === state.model && state.epoch !== null;
  }

  function resize() {
    const style = getComputedStyle(container);   // content box: the SVG must not overflow the padding
    const boxWidth = container.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
    const boxHeight = container.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
    width = Math.max(0, boxWidth - MARGIN.left - MARGIN.right);
    height = Math.max(0, boxHeight - MARGIN.top - MARGIN.bottom);
    svg.attr("width", Math.max(0, boxWidth)).attr("height", Math.max(0, boxHeight));
    plot.attr("transform", `translate(${MARGIN.left},${MARGIN.top})`);
    if (isReady(store.state)) draw(store.state);
  }

  /** Full redraw: axes, lines, labels (configuration or metric change, resize). */
  function draw(state) {
    if (!isReady(state) || width <= 0 || height <= 0) return;
    toggle.classed("active", (d) => d.id === state.curve).attr("aria-pressed", (d) => String(d.id === state.curve));
    const all = SERIES.map((s) => ({ ...s, points: series(s.key, state) }));
    const values = all.flatMap((s) => s.points.map((p) => p.value));
    const maxEpoch = d3.max(view.config.history.epoch);
    x.domain([0, maxEpoch]).range([0, width]);
    if (state.curve === "acc") y.domain([Math.floor(d3.min(values) * 10) / 10, 1]);
    else y.domain([0, d3.max(values) * 1.05]);
    y.range([height, 0]).nice();

    xAxisG.attr("transform", `translate(0,${height})`).call(d3.axisBottom(x).ticks(5).tickSizeOuter(0));
    yAxisG.call(d3.axisLeft(y).ticks(4).tickFormat(state.curve === "acc" ? d3.format(".0%") : d3.format(".2f"))
      .tickSizeOuter(0));
    snapshotsG.attr("transform", `translate(0,${height})`).selectAll("line").data(view.config.epochs).join("line")
      .attr("x1", (e) => x(e)).attr("x2", (e) => x(e)).attr("y1", -4).attr("y2", 0);

    const line = d3.line().x((p) => x(p.epoch)).y((p) => y(p.value));
    linesG.selectAll("path").data(all, (s) => s.key).join("path")
      .attr("class", "curve-line")
      .attr("stroke", (s) => s.color)
      .attr("stroke-dasharray", (s) => s.dash)
      .attr("d", (s) => line(s.points));
    // Inline end labels instead of a legend box.
    const ends = all.map((s) => ({ ...s, y: y(s.points.at(-1).value) }));
    if (Math.abs(ends[0].y - ends[1].y) < 11) {
      const [hi, lo] = ends[0].y < ends[1].y ? [ends[0], ends[1]] : [ends[1], ends[0]];
      const mid = (hi.y + lo.y) / 2;
      hi.y = mid - 6;
      lo.y = mid + 6;
    }
    labelsG.selectAll("text").data(ends, (s) => s.key).join("text")
      .attr("x", width + 4).attr("y", (s) => s.y).attr("dy", "0.35em")
      .attr("fill", (s) => s.color).text((s) => s.name);
    overlay.attr("width", width).attr("height", height);
    marker.select(".marker-rule").attr("y1", 0).attr("y2", height);
    hover.select("line").attr("y1", 0).attr("y2", height);
    moveMarker(state, false);
  }

  /** Current-epoch marker: rule + a dot on each line + value labels; moves with a transition (§5.3). */
  function moveMarker(state, animate = true) {
    if (!isReady(state) || width <= 0) return;
    const i = view.config.history.epoch.indexOf(state.epoch);
    const points = SERIES.map((s) => ({ ...s, value: series(s.key, state)[i].value }));
    const duration = animate ? ms(MOVE_MS) / (state.playing ? state.speed : 1) : 0;
    const px = x(state.epoch);
    const left = px > width - 70;   // labels on the left of the rule near the right edge
    const fmt = format(state);
    marker.transition("move").duration(duration).ease(d3.easeCubicInOut).attr("transform", `translate(${px},0)`);
    const dots = marker.selectAll(".marker-dot").data(points, (p) => p.key).join((enter) => {
      const g = enter.append("g").attr("class", "marker-dot");
      g.append("circle").attr("r", 3);
      g.append("text").attr("dy", "0.35em");
      return g;
    });
    dots.select("circle").attr("fill", (p) => p.color);
    const ys = points.map((p) => y(p.value));
    const sep = Math.abs(ys[0] - ys[1]) < 11 ? (ys[0] <= ys[1] ? [-6, 6] : [6, -6]) : [0, 0];
    dots.transition("move").duration(duration).ease(d3.easeCubicInOut)
      .attr("transform", (p) => `translate(0,${y(p.value)})`);
    dots.select("text")
      .attr("x", left ? -6 : 6).attr("text-anchor", left ? "end" : "start")
      .attr("y", (p, k) => sep[k])
      .attr("fill", (p) => p.color)
      .text((p) => fmt(p.value));
  }

  // Hover: crosshair + values at any epoch; click: nearest snapshot epoch (§5.3).
  overlay
    .on("mousemove", (event) => {
      const state = store.state;
      if (!isReady(state)) return;
      const epoch = Math.round(x.invert(d3.pointer(event)[0]));
      const i = view.config.history.epoch.indexOf(Math.max(0, Math.min(epoch, x.domain()[1])));
      if (i < 0) return;
      hover.style("display", null).attr("transform", `translate(${x(view.config.history.epoch[i])},0)`);
      const fmt = format(state);
      const text = SERIES.map((s) => `${s.name} ${fmt(series(s.key, state)[i].value)}`).join(" · ");
      tooltip.show(`epoch ${view.config.history.epoch[i]} · ${text}`, event);
    })
    .on("mouseleave", () => {
      hover.style("display", "none");
      tooltip.hide();
    })
    .on("click", (event) => {
      const state = store.state;
      if (!isReady(state)) return;
      const epoch = x.invert(d3.pointer(event)[0]);
      const nearest = d3.least(view.config.epochs, (e) => Math.abs(e - epoch));
      store.set({ epoch: nearest });
    });

  new ResizeObserver(resize).observe(container);
  store.subscribe((state) => draw(state), ["curve"]);
  store.subscribe((state) => moveMarker(state), ["epoch"]);

  return {
    setConfiguration(next) {
      view = next;
      resize();
    },
  };
}
