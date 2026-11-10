// Architecture panel (VIS_PLAN.md §5.2): one box per layer, Input at the top; click = T2 view of that layer.
// Separation bars next to the boxes = neighborhood hit of each layer's T2 projection at the current epoch, so the
// bars growing downward tell the inter-layer story at a glance. They tween when the epoch changes.
import { frameOf } from "./data.js";
import { MOVE_MS, ms, progressGray } from "./encoding.js";
import { cachedNeighborhoodHit } from "./metrics.js";
import { infoIcon, tooltip } from "./ui.js";

const INPUT_STATIC = "Raw pixels do not change during training";
const NH_INFO = "Separation: neighborhood hit (NH) of each layer's T2 projection at the current epoch, "
  + "on the shown points. Longer bar = classes better separated.";

/** Builds the panel inside `container`; `subtitle` shows the model name. Returns { setConfiguration(view) }. */
export function initArchitecture(container, { store, subtitle }) {
  let view = null;
  let built = null;
  const root = d3.select(container).append("div").attr("class", "arch");
  const head = root.append("div").attr("class", "arch-head");
  head.append("span").text("layer");
  const nhHead = head.append("span").attr("class", "arch-head-nh").text("NH ");
  nhHead.node().append(infoIcon(NH_INFO));
  const list = root.append("div").attr("class", "arch-list");

  const isReady = (state) => view && view.dataset === state.dataset && view.model === state.model
    && view.projection === state.projection && state.epoch !== null && state.layer !== null;

  function build() {
    const { config } = view;
    built = `${view.dataset}/${view.model}`;
    d3.select(subtitle).text(view.index.model(view.dataset, view.model).name);
    list.selectAll("*").remove();
    const rows = list.selectAll(".arch-row").data(config.layers, (d) => d.id).join("div").attr("class", "arch-row");
    rows.filter((d, i) => i > 0).insert("div", ":first-child").attr("class", "arch-arrow").attr("aria-hidden", true);
    const line = rows.append("div").attr("class", "arch-line");
    const box = line.append("button").attr("type", "button").attr("class", "arch-box")
      .on("click", (event, d) => store.set({ mode: "layer", layer: d.id }))
      .on("mouseenter", (event, d) => tooltip.show(details(d), event))
      .on("mousemove", (event) => tooltip.move(event))
      .on("mouseleave", () => tooltip.hide());
    box.append("span").attr("class", "arch-stripe");
    const text = box.append("span").attr("class", "arch-text");
    text.append("span").attr("class", "arch-name").text((d) => d.label);
    text.append("span").attr("class", "arch-units").text((d) => d3.format(",")(d.units));
    text.append("span").attr("class", "arch-type").text((d) => d.type);
    const bar = line.append("div").attr("class", "arch-bar");
    bar.append("div").attr("class", "arch-bar-track").append("div").attr("class", "arch-bar-fill");
    bar.append("span").attr("class", "arch-bar-value metric");
  }

  function details(layer) {
    const state = store.state;
    const parts = [`${layer.label}: ${d3.format(",")(layer.units)} ${layer.id === "input" ? "values" : "units"}`,
      layer.type];
    if (layer.dropout) parts.push(`dropout ${layer.dropout}`);
    if (state.mode === "epoch" && layer.id === "input") parts.push(INPUT_STATIC);
    else parts.push("click: compare layers (T2)");
    return parts.join(" · ");
  }

  function render(state, changed = new Set()) {
    if (!isReady(state)) return;
    if (built !== `${view.dataset}/${view.model}`) build();
    const { config, coords } = view;
    const ids = config.sample(state.sampleSize);
    const nLayers = config.layers.length;
    const rows = list.selectAll(".arch-row");
    rows.classed("current", (d) => d.id === state.layer)
      .classed("static", (d) => state.mode === "epoch" && !config.t1Layers.includes(d.id))
      .classed("ramp", state.mode === "layer" && state.showAll);
    // T2 path view: the stripe is the legend of the trail lightness (dark = first layer, light = last).
    rows.select(".arch-stripe").style("background", (d, i) => progressGray(nLayers > 1 ? i / (nLayers - 1) : 1));
    rows.select(".arch-box").attr("aria-pressed", (d) => String(d.id === state.layer));

    const nh = (d) => cachedNeighborhoodHit(frameOf(config, coords, { mode: "layer", layer: d.id, epoch: state.epoch }),
      ids, config.labels);
    const duration = changed.has("epoch") ? ms(MOVE_MS) / (state.playing ? state.speed : 1) : 0;
    rows.select(".arch-bar-fill").transition("nh").duration(duration).ease(d3.easeCubicInOut)
      .style("width", (d) => `${nh(d) * 100}%`);
    rows.select(".arch-bar-value").each(function (d) {
      const node = this;
      const to = nh(d);
      const from = node.__value ?? to;
      node.__value = to;
      d3.select(node).transition("nh").duration(duration).tween("text", () => {
        const i = d3.interpolateNumber(from, to);
        return (t) => { node.textContent = i(t).toFixed(2); };
      });
    });
  }

  store.subscribe(render, ["mode", "layer", "epoch", "sampleSize", "showAll", "playing"]);

  return {
    setConfiguration(next) {
      view = next;
      built = null;
      render(store.state);
    },
  };
}
