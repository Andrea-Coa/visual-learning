// Data access for the site: fetches the files written by pipeline/export.py (schema: README "Site data files"),
// decodes them once into typed arrays and caches them. No D3 or DOM here, so it also runs in Node.

/** Fetches and caches JSON files relative to a base URL (relative URLs keep GitHub Pages sub-paths working). */
export class DataStore {
  constructor(base = "data/") {
    const root = typeof document !== "undefined" ? document.baseURI : undefined;
    this.base = new URL(base, root);
    this.cache = new Map();
  }

  url(path) {
    return new URL(path, this.base).href;
  }

  async fetchJson(path) {
    const res = await fetch(this.url(path));
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}: ${path}`);
    return res.json();
  }

  /** Caches the (decoded) result of `factory` under `key`; a failure is evicted so it can be retried. */
  memo(key, factory) {
    if (!this.cache.has(key)) {
      const promise = factory().catch((err) => {
        this.cache.delete(key);
        throw err;
      });
      this.cache.set(key, promise);
    }
    return this.cache.get(key);
  }

  index() {
    return this.memo("index", async () => new Index(await this.fetchJson("index.json")));
  }

  /** meta.json + nodes.json of one configuration. */
  config(dataset, model) {
    return this.memo(`config:${dataset}/${model}`, async () => {
      const [meta, nodes] = await Promise.all([
        this.fetchJson(`${dataset}/${model}/meta.json`),
        this.fetchJson(`${dataset}/${model}/nodes.json`),
      ]);
      return new Config(meta, nodes);
    });
  }

  coords(dataset, model, method) {
    const path = `${dataset}/${model}/coords_${method}.json`;
    return this.memo(path, async () => new Coords(await this.fetchJson(path)));
  }

  /** Bundled t-SNE trails of one view: mode "epoch" (T1, key = layer id) or "layer" (T2, key = epoch value). */
  trails(dataset, model, mode, key) {
    const path = `${dataset}/${model}/trails/${mode === "epoch" ? `t1_${key}` : `t2_e${key}`}.json`;
    return this.memo(path, async () => new Trails(await this.fetchJson(path)));
  }

  spriteUrl(dataset) {
    return this.url(`${dataset}/sprites.png`);
  }
}

/** index.json: the available datasets, models and projections. */
export class Index {
  constructor(raw) {
    this.scale = raw.scale;
    this.datasets = raw.datasets;
  }

  dataset(id) {
    return this.datasets.find((d) => d.id === id);
  }

  model(datasetId, modelId) {
    return this.dataset(datasetId)?.models.find((m) => m.id === modelId);
  }

  /** CSS background values that show image `id` of a dataset's sprite sheet at `size` px. */
  sprite(datasetId, id, size, spriteUrl) {
    const { cell, columns, count } = this.dataset(datasetId).sprite;
    const k = size / cell;
    const rows = Math.ceil(count / columns);
    return {
      backgroundImage: `url("${spriteUrl}")`,
      backgroundSize: `${columns * cell * k}px ${rows * cell * k}px`,
      backgroundPosition: `${-(id % columns) * cell * k}px ${-Math.floor(id / columns) * cell * k}px`,
    };
  }
}

/** One configuration (dataset × model): architecture, curves and per-epoch predictions. */
export class Config {
  constructor(meta, nodes) {
    this.meta = meta;
    this.dataset = meta.dataset;
    this.model = meta.model;
    this.classes = meta.classes;
    this.epochs = meta.epochs;             // snapshot epochs, e.g. [0, 1, 2, 3, 5, ...]
    this.layers = meta.layers;             // Input -> hidden -> Output, with units/type/dropout
    this.t1Layers = meta.t1Layers;         // layers that have T1 (epoch) frames
    this.t2Layers = meta.t2Layers;         // layers that have T2 (layer) frames, in forward order
    this.history = meta.history;
    this.metrics = meta.metrics;

    this.n = nodes.n;
    this.labels = Uint8Array.from(nodes.labels);
    this.testIndex = nodes.testIndex;
    this.sampleOrder = Uint16Array.from(nodes.sampleOrder);
    this.preds = nodes.preds.map((p) => Uint8Array.from(p));
    this.probScale = nodes.probScale;
    this.probs = nodes.probs.map((p) => Uint16Array.from(p));
    this.nClasses = this.classes.length;
  }

  epochIndex(epoch) {
    const i = this.epochs.indexOf(epoch);
    if (i < 0) throw new Error(`epoch ${epoch} is not a snapshot epoch`);
    return i;
  }

  get lastHiddenLayer() {
    return this.t1Layers[this.t1Layers.length - 2];
  }

  /** The first n observation ids of the nested stratified order (n / 10 per class). */
  sample(n) {
    return this.sampleOrder.subarray(0, Math.min(n, this.n));
  }

  pred(epoch, id) {
    return this.preds[this.epochIndex(epoch)][id];
  }

  /** Softmax probabilities of observation `id` at `epoch`, as numbers in [0, 1]. */
  probsOf(epoch, id) {
    const p = this.probs[this.epochIndex(epoch)];
    return Array.from(p.subarray(id * this.nClasses, (id + 1) * this.nClasses), (v) => v / this.probScale);
  }
}

/** All 2D frames of one configuration and method; every frame is a Float32Array [x0, y0, x1, y1, ...] in [0, 1]. */
export class Coords {
  constructor(raw) {
    const toFrame = (flat) => Float32Array.from(flat, (v) => v / raw.scale);
    this.t1 = Object.fromEntries(Object.entries(raw.t1).map(([layer, frames]) => [layer, frames.map(toFrame)]));
    this.t2 = raw.t2.map((frames) => frames.map(toFrame));
  }

  /**
   * The frame shown for (mode, layer, epoch index, layer index):
   *   mode "epoch" (T1): joint projection of `layer` over epochs -> frame of epoch `epochIdx`
   *   mode "layer" (T2): projection of epoch `epochIdx` -> frame of layer `layerIdx` (index in t2Layers)
   */
  frame(mode, { layer, epochIdx, layerIdx }) {
    if (mode === "epoch") {
      const frames = this.t1[layer];
      if (!frames) throw new Error(`layer ${layer} has no T1 frames`);
      return frames[epochIdx];
    }
    return this.t2[epochIdx][layerIdx];
  }

  /** All frames along the animated dimension (T1: every epoch of a layer; T2: every layer of an epoch). */
  sequence(mode, { layer, epochIdx }) {
    return mode === "epoch" ? this.t1[layer] : this.t2[epochIdx];
  }
}

/**
 * The frame shown for (mode, layer, epoch value). In T1 the Input layer has no frames of its own (raw pixels do
 * not change during training), so its T2 frame is used.
 */
export function frameOf(config, coords, { mode, layer, epoch }) {
  const e = config.epochIndex(epoch);
  if (mode === "epoch" && coords.t1[layer]) return coords.t1[layer][e];
  return coords.t2[e][config.t2Layers.indexOf(layer)];
}

/** Bundled trails of one view: trail(strength, id) is a Float32Array [x0, y0, ...] in [0, 1]. */
export class Trails {
  constructor(raw) {
    this.strengths = raw.strengths;
    this.samplesPerSegment = raw.samplesPerSegment;
    this.trails = Object.fromEntries(
      Object.entries(raw.trails).map(([s, list]) => [s, list.map((flat) => Float32Array.from(flat, (v) => v / raw.scale))]),
    );
  }

  trail(strength, id) {
    return this.trails[strength][id];
  }

  /** Step position (0 = first epoch/layer) of point i of a trail; used for the dark -> bright progress color. */
  stepOf(i) {
    return i / this.samplesPerSegment;
  }
}
