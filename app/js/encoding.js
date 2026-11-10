// Visual encodings shared by every view (VIS_PLAN.md §4): class colors, color-by modes, trail progress colors and
// animation timing. One meaning per color: hue = class; UI chrome is neutral gray / near-black.

export const CLASS_COLORS = d3.schemeCategory10;  // = matplotlib tab10, the colors of the paper's figures
export const INK = "#222222";                      // selection, rules, "wrong" in correctness mode
export const CORRECT_GRAY = "#c8c8c8";

const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
/** Animation duration, or 0 when the user prefers reduced motion (Play still steps, §6). */
export const ms = (duration) => (reducedMotion.matches ? 0 : duration);
export const MOVE_MS = 750;      // step change (§6)
export const DWELL_MS = 450;     // pause on each frame during Play, at 1×

/** Trail color at progress t in [0, 1]: class color mixed 60 % with near-black at the start, pure at the end (§4.3). */
export function progressColor(cls, t) {
  const pure = CLASS_COLORS[cls];
  return d3.interpolateRgb(d3.interpolateRgb(pure, INK)(0.6), pure)(t);
}

/** Neutral version of the progress ramp (architecture panel legend for the trail lightness). */
export const progressGray = (t) => d3.interpolateRgb("#4a4a4a", "#d6d6d6")(t);

/** Color of the "learned at" snapshot index (viridis); index < 0 = never learned. */
export function learnedColor(index, nSnapshots) {
  return index < 0 ? null : d3.interpolateViridis(nSnapshots > 1 ? index / (nSnapshots - 1) : 1);
}

/**
 * Fill, shape and outline of a point (§4.1–4.2). `pred` is the prediction currently drawn.
 * Returns { fill, wrong, hollow }.
 */
export function pointStyle(colorBy, { label, pred, learnedAt }, nSnapshots) {
  const wrong = pred !== label;
  if (colorBy === "correctness") return { fill: wrong ? INK : CORRECT_GRAY, wrong, hollow: false };
  if (colorBy === "learnedAt") {
    if (learnedAt < 0) return { fill: "none", wrong: true, hollow: true };
    return { fill: learnedColor(learnedAt, nSnapshots), wrong, hollow: false };
  }
  return { fill: CLASS_COLORS[pred], wrong, hollow: false };
}

/** Short class name for tight headers (CIFAR-10 names abbreviated to 4 letters; digits unchanged). */
export const shortName = (name) => (name.length > 4 ? name.slice(0, 4) : name);
