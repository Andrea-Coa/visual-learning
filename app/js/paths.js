// Path view (VIS_PLAN.md §4.3, §5.4, §6): one trail per observation through its positions over the active dimension
// (epochs in T1, layers in T2). Trails are drawn as one SVG path per (class, segment, filter state): up to
// 1000 trails × 10 segments stay ~200 elements, each with its own progress color (hue = class, dark -> light).
// All bundling strengths have the same number of samples per trail, so trails can morph between strengths.
import { CLASS_COLORS, ms, progressColor } from "./encoding.js";

const OPACITY = { base: 0.25, in: 0.6, out: 0.04 };   // §4.3: normal, filtered-in, filtered-out
const DRAW_IN_MS = 1100;      // the last segment starts appearing after this (plus the class stagger)
const CLASS_STAGGER_MS = 60;  // §6: draw-in staggered by class
const SEGMENT_FADE_MS = 250;
const MORPH_MS = 600;         // bundling strength change
const LEAVE_MS = 400;
const END_RADIUS = 2;

/** Linear resampling of the step positions (S points) to `k` samples per segment: the unbundled trail. */
export function straightTrail(points, k) {
  const steps = points.length / 2;
  const out = new Float32Array(((steps - 1) * k + 1) * 2);
  for (let s = 0; s < steps - 1; s++) {
    for (let i = 0; i < k; i++) {
      const t = i / k;
      out[2 * (s * k + i)] = points[2 * s] + (points[2 * s + 2] - points[2 * s]) * t;
      out[2 * (s * k + i) + 1] = points[2 * s + 1] + (points[2 * s + 3] - points[2 * s + 1]) * t;
    }
  }
  out[out.length - 2] = points[points.length - 2];
  out[out.length - 1] = points[points.length - 1];
  return out;
}

/** Path data of one group: segment `seg` of every trail in the group, interpolated between two geometries. */
function groupPath(group, from, to, k, t) {
  const start = group.seg * k;
  let d = "";
  for (const id of group.ids) {
    const a = from.get(id);
    const b = to.get(id);
    for (let i = start; i <= start + k; i++) {
      const px = a[2 * i] + (b[2 * i] - a[2 * i]) * t;
      const py = a[2 * i + 1] + (b[2 * i + 1] - a[2 * i + 1]) * t;
      d += `${i === start ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`;
    }
  }
  return d;
}

export class PathLayer {
  /** `parent`: the zoomed layer; `glyphScale()`: the 1/k transform that keeps marks a constant size. */
  constructor(parent, glyphScale) {
    this.trailsG = parent.append("g").attr("class", "trails");
    this.endsG = parent.append("g").attr("class", "trail-ends");
    this.glyphScale = glyphScale;
    this.trails = null;   // Map id -> Float32Array of base px, the geometry currently drawn
    this.k = 4;
    this.steps = 0;
    this.quadtree = null;
  }

  get active() {
    return this.trails !== null;
  }

  /**
   * Draws (or redraws) the trails. `trails`: Map id -> base px samples; `ids`: drawn ids; `labels`; `pass(id)`:
   * filter test; `animate`: "in" (draw-in), "morph" (from the current geometry) or null.
   */
  render({ trails, ids, labels, pass, anyFilter, k, steps }, animate = null) {
    const from = this.trails;
    const morph = animate === "morph" && from && ids.every((id) => from.has(id));
    this.trails = trails;
    this.k = k;
    this.steps = steps;

    const groups = new Map();
    for (const id of ids) {
      const ok = pass(id);
      for (let seg = 0; seg < steps - 1; seg++) {
        const key = `${labels[id]}|${seg}|${ok ? 1 : 0}`;
        if (!groups.has(key)) groups.set(key, { key, cls: labels[id], seg, pass: ok, ids: [] });
        groups.get(key).ids.push(id);
      }
    }
    const opacity = (d) => (d.pass ? (anyFilter ? OPACITY.in : OPACITY.base) : OPACITY.out);
    const paths = this.trailsG.selectAll("path:not(.leaving)").data([...groups.values()], (d) => d.key)
      .join("path")
      .attr("stroke", (d) => progressColor(d.cls, steps > 1 ? (d.seg + 0.5) / (steps - 1) : 1));

    // Paint attributes (stroke-opacity / fill-opacity), not CSS opacity: no compositing layer per path.
    if (morph) {
      paths.interrupt("trail").attr("stroke-opacity", opacity)
        .transition("morph").duration(ms(MORPH_MS)).ease(d3.easeCubicInOut)
        .attrTween("d", (d) => (t) => groupPath(d, from, trails, k, t));
    } else {
      paths.interrupt("morph").attr("d", (d) => groupPath(d, trails, trails, k, 0));
      if (animate === "in") {
        const perSegment = steps > 1 ? DRAW_IN_MS / (steps - 1) : 0;
        paths.attr("stroke-opacity", 0).transition("trail")
          .delay((d) => ms(d.cls * CLASS_STAGGER_MS + d.seg * perSegment))
          .duration(ms(SEGMENT_FADE_MS))
          .attr("stroke-opacity", opacity);
      } else {
        paths.interrupt("trail").attr("stroke-opacity", opacity);
      }
    }

    // End positions as small dots on top, so the clusters stay readable (§4.3); they fade in last.
    const last = ((steps - 1) * k) * 2;
    const ends = this.endsG.selectAll(".trail-end:not(.leaving)").data(ids, (id) => id)
      .join((enter) => {
        const g = enter.append("g").attr("class", "trail-end");
        g.append("g").attr("class", "glyph").append("circle").attr("r", END_RADIUS);
        return g;
      })
      .attr("transform", (id) => `translate(${trails.get(id)[last]},${trails.get(id)[last + 1]})`);
    ends.select(".glyph").attr("transform", this.glyphScale());
    const dots = ends.select("circle").attr("fill", (id) => CLASS_COLORS[labels[id]]);
    const endOpacity = (id) => (pass(id) ? 0.9 : 0.08);
    if (animate === "in") {
      dots.attr("fill-opacity", 0).transition("trail")
        .delay(ms(DRAW_IN_MS + 9 * CLASS_STAGGER_MS)).duration(ms(SEGMENT_FADE_MS)).attr("fill-opacity", endOpacity);
    } else {
      dots.interrupt("trail").attr("fill-opacity", endOpacity);
    }

    // Hit testing: nearest trail sample (any trail point = that observation), filtered-in trails only.
    const points = [];
    for (const id of ids) {
      if (anyFilter && !pass(id)) continue;
      const c = trails.get(id);
      for (let i = 0; i < c.length; i += 2) points.push([c[i], c[i + 1], id]);
    }
    this.quadtree = d3.quadtree(points);
  }

  /** The observation whose trail passes closest to the base-px point (x, y), within `radius`, or null. */
  nearest(x, y, radius) {
    return this.quadtree?.find(x, y, radius)?.[2] ?? null;
  }

  /** End position (base px) of a trail. */
  end(id) {
    const c = this.trails?.get(id);
    return c ? [c[c.length - 2], c[c.length - 1]] : null;
  }

  hide() {
    if (!this.active) return;
    this.trails = null;
    this.quadtree = null;
    this.trailsG.selectAll(":scope > *").classed("leaving", true).interrupt("trail").interrupt("morph")
      .transition("leave").duration(ms(LEAVE_MS)).attr("stroke-opacity", 0).remove();
    const ends = this.endsG.selectAll(":scope > *").classed("leaving", true);
    ends.select("circle").interrupt("trail").transition("leave").duration(ms(LEAVE_MS)).attr("fill-opacity", 0);
    ends.transition("leave").duration(ms(LEAVE_MS)).remove();
  }

  clear() {
    this.trails = null;
    this.quadtree = null;
    this.trailsG.selectAll("*").remove();
    this.endsG.selectAll("*").remove();
  }
}
