// Quality metrics computed in the browser on the points currently shown (VIS_PLAN.md §5.4, D6).
// No D3 or DOM here, so it also runs in Node.

/**
 * Neighborhood hit: the average share of each point's k nearest neighbors (2D, Euclidean) that have its class.
 * `frame` is a flat [x0, y0, x1, y1, ...] array indexed by observation id; only the observations in `ids` count.
 */
export function neighborhoodHit(frame, ids, labels, k = 6) {
  const n = ids.length;
  if (n <= k) return NaN;
  const xs = new Float64Array(n);
  const ys = new Float64Array(n);
  const cls = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    xs[i] = frame[2 * ids[i]];
    ys[i] = frame[2 * ids[i] + 1];
    cls[i] = labels[ids[i]];
  }

  const bestDist = new Float64Array(k);  // the k smallest distances so far, ascending
  const bestIdx = new Int32Array(k);
  let hits = 0;
  for (let i = 0; i < n; i++) {
    bestDist.fill(Infinity);
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dx = xs[i] - xs[j];
      const dy = ys[i] - ys[j];
      const d = dx * dx + dy * dy;
      if (d >= bestDist[k - 1]) continue;
      let p = k - 1;                       // insertion into the sorted top-k
      while (p > 0 && bestDist[p - 1] > d) {
        bestDist[p] = bestDist[p - 1];
        bestIdx[p] = bestIdx[p - 1];
        p--;
      }
      bestDist[p] = d;
      bestIdx[p] = j;
    }
    for (let p = 0; p < k; p++) if (cls[bestIdx[p]] === cls[i]) hits++;
  }
  return hits / (n * k);
}

/** Share of the observations in `ids` whose prediction equals their label. */
export function sampleAccuracy(preds, ids, labels) {
  if (ids.length === 0) return NaN;
  let correct = 0;
  for (const id of ids) if (preds[id] === labels[id]) correct++;
  return correct / ids.length;
}
