// Shared UI state (VIS_PLAN.md §3): the single source of truth. Every view reads it and re-renders when the keys
// it depends on change, so all interactions are linked automatically. No D3 or DOM here (testable in Node).

export const DEFAULTS = Object.freeze({
  dataset: "mnist",        // mnist | svhn | cifar10
  model: "mlp",            // mlp | cnn
  projection: "grid",      // grid | aumap | umap | tsne | tsne_proc | pca (grid: one map for T1 and T2, PLAN S.4/S.5)
  mode: "epoch",           // "epoch" (T1, inter-epoch) | "layer" (T2, inter-layer)
  layer: null,             // layer id; null = last hidden layer of the loaded configuration
  epoch: null,             // snapshot epoch value (not index); null = last snapshot epoch
  showAll: false,          // path view (trails over every step of the active dimension)
  playing: false,
  speed: 1,                // 0.5 | 1 | 2
  sampleSize: 500,         // 100..1000, step 100
  colorBy: "class",        // class | correctness | learnedAt
  bundling: "medium",      // none | medium | strong
  classFilter: null,       // Set of class indices, or null = all classes
  cellFilter: null,        // { truth, pred } confusion-matrix cell, or null
  brushed: null,           // Set of observation ids, or null
  hovered: null,           // observation id, or null
  pinned: null,            // observation id, or null
  cellHover: null,         // { truth, pred } confusion-matrix cell under the mouse, or null (temporary highlight)
  trajectory: true,        // draw the pinned point's unbundled trajectory (Data point panel toggle)
  metric: "recall",        // per-class bars: recall | precision | f1
  curve: "acc",            // learning curve: acc | loss
});

// Selections hold observation ids, so they survive epoch/layer changes and Play, but not a change of what the
// ids are compared across (VIS_PLAN.md §3, rules).
const SELECTION_RESET = { brushed: null, cellFilter: null, cellHover: null, pinned: null, hovered: null };
const RESETS = {
  dataset: { ...SELECTION_RESET, classFilter: null, layer: null }, // other classes, other layers
  model: { ...SELECTION_RESET, layer: null },                      // other layers
  projection: SELECTION_RESET,
};

export class Store {
  constructor(initial = {}) {
    this.state = { ...DEFAULTS, ...initial };
    this.listeners = new Set();
  }

  get(key) {
    return this.state[key];
  }

  /**
   * Applies a partial update, plus the resets triggered by the changed keys (an explicit value in the same
   * patch wins over a reset). Listeners are called once with the set of keys whose value actually changed.
   */
  set(patch) {
    const next = { ...this.state };
    for (const [key, value] of Object.entries(patch)) {
      if (!(key in DEFAULTS)) throw new Error(`unknown state key: ${key}`);
      if (next[key] !== value && RESETS[key]) Object.assign(next, RESETS[key]);
    }
    Object.assign(next, patch);

    const changed = new Set(Object.keys(next).filter((key) => next[key] !== this.state[key]));
    if (changed.size === 0) return changed;
    this.state = next;
    for (const { fn, keys } of [...this.listeners]) {
      if (!keys || keys.some((key) => changed.has(key))) fn(this.state, changed);
    }
    return changed;
  }

  /** Calls fn(state, changedKeys) after every change of any of `keys` (all keys if omitted). Returns unsubscribe. */
  subscribe(fn, keys = null) {
    const entry = { fn, keys };
    this.listeners.add(entry);
    return () => this.listeners.delete(entry);
  }

  /** Toggles a value in a Set-valued filter key (e.g. classFilter); an empty set becomes null (= no filter). */
  toggleInSet(key, value, { exclusive = false } = {}) {
    const current = this.state[key];
    let next;
    if (exclusive) {
      next = current?.size === 1 && current.has(value) ? null : new Set([value]);
    } else {
      next = new Set(current ?? []);
      next.has(value) ? next.delete(value) : next.add(value);
      if (next.size === 0) next = null;
    }
    this.set({ [key]: next });
  }
}
