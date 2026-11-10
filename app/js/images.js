// Image view (VIS_PLAN.md §5.6, drafts/draft-v2.png): thumbnails of the points that pass the filters (brushed /
// confusion cell / classes; a random sample when nothing is filtered), linked both ways to the scatter. No
// scrolling: the thumbnails are sized to fill the panel height and the set is paged. The active-filter chips, sort
// and pager live in the panel title.
import { CLASS_COLORS, shortName } from "./encoding.js";
import { filtersActive, passes } from "./scatter.js";

const GAP = 6;
const TARGET = 84;          // preferred thumbnail size (px); the actual size fills the panel height exactly
const MAX_THUMB = 104;
const SORTS = [{ id: "wrong", name: "misclassified first" }, { id: "confidence", name: "least confident first" }];

/**
 * Builds the view inside `container`; `subtitle` names the image source and `tools` (panel title) holds the chips,
 * sort, reshuffle and pager. Returns { setConfiguration(view) }.
 */
export function initImages(container, { store, subtitle, tools }) {
  let view = null;
  let sort = "wrong";
  let seed = 1;
  let page = 0;
  let orderKey = null;
  let order = [];

  const bar = d3.select(tools);
  const chips = bar.append("div").attr("class", "chips");
  const sortSelect = bar.append("select").attr("class", "control-input img-sort").attr("aria-label", "Sort images")
    .on("change", (event) => { sort = event.target.value; render(store.state); });
  sortSelect.selectAll("option").data(SORTS).join("option").attr("value", (d) => d.id).text((d) => d.name);
  const reshuffle = bar.append("button").attr("type", "button").attr("class", "tool-btn tool-icon")
    .attr("title", "New random sample").attr("aria-label", "New random sample").text("⟳")
    .on("click", () => { seed++; render(store.state); });
  const pager = bar.append("div").attr("class", "seg pager").attr("role", "group").attr("aria-label", "Pages");
  const prev = pager.append("button").attr("type", "button").attr("aria-label", "Previous page").text("‹")
    .on("click", () => { page--; render(store.state); });
  const pageText = pager.append("span").attr("class", "pager-text");
  const next = pager.append("button").attr("type", "button").attr("aria-label", "Next page").text("›")
    .on("click", () => { page++; render(store.state); });

  const grid = d3.select(container).append("div").attr("class", "img-grid");

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
      const names = [...state.classFilter].sort((a, b) => a - b).map((c) => shortName(classes[c])).join(", ");
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

  /** Thumbnail size and page size that fill the panel without scrolling. */
  function geometry() {
    const w = container.clientWidth - 20;    // content box (panel-body padding 8 px × 10 px)
    const h = container.clientHeight - 16;
    const rows = Math.max(1, Math.round((h + GAP) / (TARGET + GAP)));
    const size = Math.max(24, Math.min(MAX_THUMB, Math.floor((h + GAP) / rows) - GAP));
    const columns = Math.max(1, Math.floor((w + GAP) / (size + GAP)));
    return { size, perPage: rows * columns };
  }

  function render(state) {
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
    if (key !== orderKey) {
      orderKey = key;
      page = 0;
      let ids = Array.from(config.sample(state.sampleSize)).filter((id) => passes(datum(id), state));
      if (!filtersActive(state)) ids = d3.shuffler(d3.randomLcg(seed / 1000))(ids);
      const conf = (id) => config.probs[e][id * config.nClasses + preds[id]];
      ids = sort === "wrong" ? d3.sort(ids, (id) => (preds[id] === config.labels[id] ? 1 : 0)) : d3.sort(ids, conf);
      order = ids;
    }
    d3.select(subtitle).text(sourceText(state, order.length));

    const { size, perPage } = geometry();
    const pages = Math.max(1, Math.ceil(order.length / perPage));
    page = Math.max(0, Math.min(page, pages - 1));
    const first = page * perPage;
    const shownIds = order.slice(first, first + perPage);
    prev.property("disabled", page === 0);
    next.property("disabled", page >= pages - 1);
    pageText.text(order.length ? `${first + 1}–${first + shownIds.length} / ${order.length}` : "0");
    pager.style("display", pages > 1 ? null : "none");

    const thumbs = grid.style("--thumb", `${size}px`).selectAll(".img-thumb").data(shownIds, (id) => id)
      .join((enter) => {
        const t = enter.append("button").attr("type", "button").attr("class", "img-thumb thumb")
          .on("mouseenter", (event, id) => store.set({ hovered: id }))
          .on("mouseleave", (event, id) => { if (store.get("hovered") === id) store.set({ hovered: null }); })
          .on("click", (event, id) => store.set({ pinned: store.get("pinned") === id ? null : id }));
        const tag = t.append("span").attr("class", "img-tag");
        tag.append("span").attr("class", "chip");
        tag.append("span").attr("class", "img-tag-text");
        t.append("span").attr("class", "img-badge");
        return t;
      })
      .order();
    thumbs.each(function (id) { Object.assign(this.style, index.sprite(dataset, id, size, spriteUrl)); })
      .attr("aria-label", (id) => `#${id}, true ${config.classes[config.labels[id]]}`)
      .classed("wrong", (id) => preds[id] !== config.labels[id])
      .classed("hovered", (id) => id === state.hovered)
      .classed("pinned", (id) => id === state.pinned);
    // Draft-v2: true label (chip + name) in the bottom-right corner; the prediction, if wrong, in the top-right one.
    thumbs.select(".img-tag .chip").style("background", (id) => CLASS_COLORS[config.labels[id]]);
    thumbs.select(".img-tag-text").text((id) => shortName(config.classes[config.labels[id]]));
    thumbs.select(".img-badge")
      .style("display", (id) => (preds[id] === config.labels[id] ? "none" : null))
      .style("--badge", (id) => CLASS_COLORS[preds[id]])
      .attr("title", (id) => `predicted ${config.classes[preds[id]]}`)
      .text((id) => `▲ ${shortName(config.classes[preds[id]])}`);
  }

  new ResizeObserver(() => render(store.state)).observe(container);
  store.subscribe(render, ["epoch", "sampleSize", "brushed", "cellFilter", "classFilter", "hovered", "pinned"]);

  return {
    setConfiguration(nextView) {
      view = nextView;
      orderKey = null;
      render(store.state);
    },
  };
}
