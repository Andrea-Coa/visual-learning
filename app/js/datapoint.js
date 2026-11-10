// Data point panel (VIS_PLAN.md §5.8): details of the pinned observation (a light-gray preview of the hovered one
// while nothing is pinned): thumbnail and labels, softmax bars at the current epoch, and "when was it learned?",
// the probability of the true class over the snapshot epochs.
import { CLASS_COLORS, MOVE_MS, ms, shortName } from "./encoding.js";
import { tooltip } from "./ui.js";

const SOFTMAX = { height: 112, top: 8, bottom: 22, left: 28, right: 4 };
const LEARNED = { height: 112, top: 18, bottom: 22, left: 28, right: 10 };

/** Builds the panel inside `container`. Returns { setConfiguration(view) }. */
export function initDataPoint(container, { store }) {
  let view = null;
  const root = d3.select(container).append("div").attr("class", "dp");
  const hint = root.append("p").attr("class", "dp-hint").text("Click a point to inspect it");
  const body = root.append("div").attr("class", "dp-body");

  const header = body.append("div").attr("class", "dp-header");
  const thumb = header.append("div").attr("class", "thumb dp-thumb");
  const info = header.append("div").attr("class", "dp-info");
  const idLine = info.append("div").attr("class", "dp-id");
  const trueLine = info.append("div").attr("class", "dp-row");
  const predLine = info.append("div").attr("class", "dp-row");
  const actions = info.append("div").attr("class", "dp-actions");
  const trajectory = actions.append("label").attr("class", "mm-check");
  trajectory.append("input").attr("type", "checkbox")
    .on("change", (event) => store.set({ trajectory: event.target.checked }));
  trajectory.append("span").text("show trajectory");
  actions.append("button").attr("type", "button").attr("class", "tool-btn dp-unpin").text("unpin")
    .on("click", () => store.set({ pinned: null }));

  body.append("div").attr("class", "dp-title").text("Softmax at this epoch");
  const softSvg = body.append("svg").attr("class", "dp-softmax");
  const softAxis = softSvg.append("g").attr("class", "axis");
  const softBars = softSvg.append("g");
  const softLabels = softSvg.append("g").attr("class", "mm-bar-labels");

  body.append("div").attr("class", "dp-title").text("When was it learned?");
  const learnSvg = body.append("svg").attr("class", "dp-learned");
  const learnAxisX = learnSvg.append("g").attr("class", "axis");
  const learnAxisY = learnSvg.append("g").attr("class", "axis");
  const half = learnSvg.append("line").attr("class", "dp-half");
  const learnLine = learnSvg.append("path").attr("class", "dp-line");
  const learnDots = learnSvg.append("g");
  const learnRule = learnSvg.append("line").attr("class", "marker-rule");
  const learnNote = learnSvg.append("text").attr("class", "dp-note").attr("x", LEARNED.left).attr("y", 10);
  const learnHit = learnSvg.append("rect").attr("class", "curve-overlay");

  const isReady = (state) => view && view.dataset === state.dataset && view.model === state.model
    && state.epoch !== null;

  function render(state, changed = new Set()) {
    if (!isReady(state)) return;
    const id = state.pinned ?? state.hovered;
    hint.style("display", id === null ? null : "none");
    body.style("display", id === null ? "none" : null).classed("preview", state.pinned === null);
    if (id === null) return;
    const { config, index, dataset, spriteUrl } = view;
    const n = config.nClasses;
    const label = config.labels[id];
    const pred = config.pred(state.epoch, id);
    const probs = config.probsOf(state.epoch, id);
    const width = Math.max(0, container.clientWidth - 20);
    const duration = changed.has("epoch") ? ms(MOVE_MS) / (state.playing ? state.speed : 1) : 0;

    Object.assign(thumb.node().style, index.sprite(dataset, id, 96, spriteUrl));
    idLine.text(`#${id} · test image ${config.testIndex[id]}`);
    const chipRow = (sel, text, cls) => {
      sel.selectAll("*").remove();
      sel.append("span").attr("class", "chip").style("background", CLASS_COLORS[cls]);
      sel.append("span").text(text);
    };
    chipRow(trueLine, `true: ${config.classes[label]}`, label);
    chipRow(predLine, `${pred === label ? "" : "▲ "}predicted: ${config.classes[pred]} (${probs[pred].toFixed(2)})`, pred);
    trajectory.select("input").property("checked", state.trajectory);
    actions.select(".dp-unpin").style("display", state.pinned === null ? "none" : null);

    // Softmax bars: class colors, the true class outlined (§5.8).
    softSvg.attr("width", width).attr("height", SOFTMAX.height);
    const x = d3.scaleBand().domain(d3.range(n)).range([SOFTMAX.left, width - SOFTMAX.right]).padding(0.18);
    const y = d3.scaleLinear().domain([0, 1]).range([SOFTMAX.height - SOFTMAX.bottom, SOFTMAX.top]);
    softAxis.attr("transform", `translate(${SOFTMAX.left},0)`)
      .call(d3.axisLeft(y).ticks(2).tickFormat(d3.format(".1f")).tickSizeOuter(0));
    softBars.selectAll("rect").data(d3.range(n)).join("rect")
      .attr("x", (c) => x(c)).attr("width", x.bandwidth())
      .attr("fill", (c) => CLASS_COLORS[c])
      .classed("true-class", (c) => c === label)
      .transition("bar").duration(duration).ease(d3.easeCubicInOut)
      .attr("y", (c) => y(probs[c])).attr("height", (c) => y(0) - y(probs[c]));
    softLabels.selectAll("text").data(d3.range(n)).join("text")
      .attr("x", (c) => x(c) + x.bandwidth() / 2).attr("y", SOFTMAX.height - SOFTMAX.bottom + 12)
      .attr("text-anchor", "middle").classed("strong", (c) => c === label)
      .text((c) => shortName(config.classes[c]));

    // p(true class) over the snapshot epochs, with the 0.5 reference and the current-epoch rule.
    learnSvg.attr("width", width).attr("height", LEARNED.height);
    const series = config.epochs.map((epoch) => ({ epoch, p: config.probsOf(epoch, id)[label] }));
    const lx = d3.scaleLinear().domain(d3.extent(config.epochs)).range([LEARNED.left, width - LEARNED.right]);
    const ly = d3.scaleLinear().domain([0, 1]).range([LEARNED.height - LEARNED.bottom, LEARNED.top]);
    learnAxisX.attr("transform", `translate(0,${ly(0)})`).call(d3.axisBottom(lx).ticks(5).tickSizeOuter(0));
    learnAxisY.attr("transform", `translate(${LEARNED.left},0)`)
      .call(d3.axisLeft(ly).ticks(2).tickFormat(d3.format(".1f")).tickSizeOuter(0));
    half.attr("x1", lx.range()[0]).attr("x2", lx.range()[1]).attr("y1", ly(0.5)).attr("y2", ly(0.5));
    learnLine.attr("stroke", CLASS_COLORS[label])
      .attr("d", d3.line().x((d) => lx(d.epoch)).y((d) => ly(d.p))(series));
    learnDots.selectAll("circle").data(series).join("circle").attr("r", 2.5)
      .attr("cx", (d) => lx(d.epoch)).attr("cy", (d) => ly(d.p))
      .attr("fill", (d) => (config.preds[config.epochIndex(d.epoch)][id] === label ? CLASS_COLORS[label] : "#fff"))
      .attr("stroke", CLASS_COLORS[label]);
    learnRule.attr("y1", LEARNED.top).attr("y2", ly(0))
      .transition("move").duration(duration).ease(d3.easeCubicInOut)
      .attr("x1", lx(state.epoch)).attr("x2", lx(state.epoch));
    const learned = config.learnedAt[id];
    learnNote.text(learned < 0 ? "never learned" : `learned at epoch ${config.epochs[learned]}`);
    learnHit.attr("x", lx.range()[0]).attr("y", LEARNED.top).attr("width", lx.range()[1] - lx.range()[0])
      .attr("height", ly(0) - LEARNED.top)
      .on("click", (event) => {
        const epoch = lx.invert(d3.pointer(event)[0]);
        store.set({ epoch: d3.least(config.epochs, (e) => Math.abs(e - epoch)) });
      })
      .on("mousemove", (event) => {
        const epoch = d3.least(config.epochs, (e) => Math.abs(e - lx.invert(d3.pointer(event)[0])));
        tooltip.show(`epoch ${epoch}: p(${config.classes[label]}) = ${config.probsOf(epoch, id)[label].toFixed(2)}`, event);
      })
      .on("mouseleave", () => tooltip.hide());
  }

  new ResizeObserver(() => render(store.state)).observe(container);
  store.subscribe(render, ["pinned", "hovered", "epoch", "trajectory"]);

  return {
    setConfiguration(next) {
      view = next;
      render(store.state);
    },
  };
}
