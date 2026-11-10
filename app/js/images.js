// Image view (VIS_PLAN.md §5.6) with the active-filter chips (§3): thumbnails of the points that pass the filters
// (brushed / confusion cell / classes; a random sample when nothing is filtered), linked both ways to the scatter.
import { CLASS_COLORS, shortName } from "./encoding.js";
import { filtersActive, passes } from "./scatter.js";

const THUMB = 48;
const GAP = 4;
const INITIAL_ROWS = 2;
const MORE = 100;   // thumbnails added per "show more" click
const SORTS = [{ id: "wrong", name: "misclassified first" }, { id: "confidence", name: "least confident first" }];

/** Builds the view inside `container`; `subtitle` names the image source. Returns { setConfiguration(view) }. */
export function initImages(container, { store, subtitle }) {
  let view = null;
  let sort = "wrong";
  let seed = 1;
  let limit = 0;
  let orderKey = null;
  let order = [];

  const root = d3.select(container).append("div").attr("class", "images");
  const bar = root.append("div").attr("class", "img-bar");
  const chips = bar.append("div").attr("class", "chips");
  const sortSelect = bar.append("select").attr("class", "control-input img-sort").attr("aria-label", "Sort images")
    .on("change", (event) => { sort = event.target.value; render(store.state); });
  sortSelect.selectAll("option").data(SORTS).join("option").attr("value", (d) => d.id).text((d) => d.name);
  const reshuffle = bar.append("button").attr("type", "button").attr("class", "tool-btn tool-icon")
    .attr("title", "New random sample").attr("aria-label", "New random sample").text("⟳")
    .on("click", () => { seed++; render(store.state); });
  const grid = root.append("div").attr("class", "img-grid");
  const more = root.append("button").attr("type", "button").attr("class", "tool-btn img-more").text("show more")
    .on("click", () => { limit += MORE; render(store.state, new Set(), { keepOrder: true }); });

  const isReady = (state) => view && view.dataset === state.dataset && view.model === state.model
    && state.epoch !== null;

  /** Header text for the current source, in the priority brushed -> cell -> class -> random (§5.6). */
  function sourceText(state, count) {
    const { classes } = view.config;
    const images = `${count} image${count === 1 ? "" : "s"}`;
    if (state.brushed) return `Brushed: ${images}`;
    if (state.cellFilter) return `True ${classes[state.cellFilter.truth]} → predicted ${classes[state.cellFilter.pred]}: ${images}`;
    if (state.classFilter) {
      const names = [...state.classFilter].sort((a, b) => a - b).map((c) => classes[c]);
      return `${names.length === 1 ? "Class" : "Classes"} ${names.map((n) => `'${n}'`).join(", ")}: ${images}`;
    }
    return "Random sample";
  }

  function renderChips(state) {
    const { classes } = view.config;
    const list = [];
    if (state.brushed) list.push({ key: "brushed", text: `brushed (${state.brushed.size})`, clear: { brushed: null } });
    if (state.cellFilter) {
      list.push({ key: "cell", text: `${shortName(classes[state.cellFilter.truth])} → ${shortName(classes[state.cellFilter.pred])}`,
        clear: { cellFilter: null } });
    }
    if (state.classFilter) {
      const names = [...state.classFilter].sort((a, b) => a - b).map((c) => classes[c]).join(", ");
      list.push({ key: "class", text: `class: ${names}`, clear: { classFilter: null } });
    }
    const chip = chips.selectAll(".chip-filter").data(list, (d) => d.key).join((enter) => {
      const c = enter.append("span").attr("class", "chip-filter");
      c.append("span").attr("class", "chip-text");
      c.append("button").attr("type", "button").attr("class", "chip-clear").text("✕");
      return c;
    });
    chip.select(".chip-text").text((d) => d.text);
    chip.select(".chip-clear").attr("aria-label", (d) => `Clear filter ${d.text}`)
      .on("click", (event, d) => store.set(d.clear));
  }

  function render(state, changed = new Set(), { keepOrder = false } = {}) {
    if (!isReady(state)) return;
    const { config, index, dataset, spriteUrl } = view;
    const e = config.epochIndex(state.epoch);
    const preds = config.preds[e];
    const datum = (id) => ({ id, label: config.labels[id], pred: preds[id] });
    renderChips(state);
    reshuffle.style("display", filtersActive(state) ? "none" : null);

    // The set and its order only change with the source or the sort, not with the epoch (badges update in place).
    const key = [dataset, view.model, state.sampleSize, state.brushed ? [...state.brushed].join() : "",
      JSON.stringify(state.cellFilter), state.classFilter ? [...state.classFilter].join() : "", sort, seed,
      state.cellFilter ? state.epoch : ""].join("|");
    if (key !== orderKey && !keepOrder) {
      orderKey = key;
      let ids = Array.from(config.sample(state.sampleSize)).filter((id) => passes(datum(id), state));
      if (!filtersActive(state)) ids = d3.shuffler(d3.randomLcg(seed / 1000))(ids);
      const conf = (id) => config.probs[e][id * config.nClasses + preds[id]];
      if (sort === "wrong") ids = d3.sort(ids, (id) => (preds[id] === config.labels[id] ? 1 : 0));
      else ids = d3.sort(ids, conf);
      order = ids;
      const columns = Math.max(1, Math.floor((grid.node().clientWidth + GAP) / (THUMB + GAP)));
      limit = columns * INITIAL_ROWS;
    }
    d3.select(subtitle).text(sourceText(state, order.length));

    const shownIds = order.slice(0, limit);
    const thumbs = grid.selectAll(".img-thumb").data(shownIds, (id) => id).join((enter) => {
      const t = enter.append("button").attr("type", "button").attr("class", "img-thumb thumb")
        .on("mouseenter", (event, id) => store.set({ hovered: id }))
        .on("mouseleave", (event, id) => { if (store.get("hovered") === id) store.set({ hovered: null }); })
        .on("click", (event, id) => store.set({ pinned: store.get("pinned") === id ? null : id }));
      t.append("span").attr("class", "img-truth");
      t.append("span").attr("class", "img-badge");
      return t;
    }).order();
    thumbs.each(function (id) { Object.assign(this.style, index.sprite(dataset, id, THUMB, spriteUrl)); })
      .attr("aria-label", (id) => `#${id}, true ${config.classes[config.labels[id]]}`)
      .classed("hovered", (id) => id === state.hovered)
      .classed("pinned", (id) => id === state.pinned);
    thumbs.select(".img-truth").style("background", (id) => CLASS_COLORS[config.labels[id]]);
    thumbs.select(".img-badge")
      .style("display", (id) => (preds[id] === config.labels[id] ? "none" : null))
      .style("--badge", (id) => CLASS_COLORS[preds[id]])
      .attr("title", (id) => `predicted ${config.classes[preds[id]]}`)
      .text((id) => shortName(config.classes[preds[id]]).slice(0, 2));
    more.style("display", order.length > limit ? null : "none")
      .text(`show more (${Math.min(MORE, order.length - limit)})`);
  }

  store.subscribe(render, ["epoch", "sampleSize", "brushed", "cellFilter", "classFilter", "hovered", "pinned"]);

  return {
    setConfiguration(next) {
      view = next;
      orderKey = null;
      render(store.state);
    },
  };
}
