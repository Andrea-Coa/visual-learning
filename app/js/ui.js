// Small UI helpers shared by the panels: the floating tooltip (#tooltip in index.html) and ⓘ info icons.
// Explanations live behind an ⓘ icon, never as text on a plot (CLAUDE.md, VIS_PLAN.md §9).

const OFFSET = 14;   // px between the cursor (or anchor) and the tooltip
const MARGIN = 8;    // px kept free at the window edges

const tooltipEl = () => document.getElementById("tooltip");

/** The shared tooltip. `content` is a string or a DOM node; `at` is a mouse event or a DOMRect to anchor to. */
export const tooltip = {
  show(content, at) {
    const el = tooltipEl();
    el.replaceChildren(typeof content === "string" ? document.createTextNode(content) : content);
    el.hidden = false;
    this.move(at);
  },

  /** Places the tooltip below-right of the anchor, flipped to the other side when it would leave the window. */
  move(at) {
    const el = tooltipEl();
    if (el.hidden || !at) return;
    const isRect = "width" in at && "left" in at;
    const x = isRect ? at.right : at.clientX;
    const y = isRect ? at.bottom : at.clientY;
    const { width, height } = el.getBoundingClientRect();
    let left = x + OFFSET;
    let top = y + OFFSET;
    if (left + width > window.innerWidth - MARGIN) left = (isRect ? at.left : x) - OFFSET - width;
    if (top + height > window.innerHeight - MARGIN) top = (isRect ? at.top : y) - OFFSET - height;
    el.style.left = `${Math.max(MARGIN, left)}px`;
    el.style.top = `${Math.max(MARGIN, top)}px`;
  },

  hide() {
    tooltipEl().hidden = true;
  },
};

/** An ⓘ icon that shows `text` in the shared tooltip on hover and on keyboard focus. */
export function infoIcon(text) {
  const icon = document.createElement("span");
  icon.className = "info";
  icon.textContent = "i";
  icon.tabIndex = 0;
  icon.setAttribute("role", "img");
  icon.setAttribute("aria-label", text);
  icon.addEventListener("mouseenter", (event) => tooltip.show(text, event));
  icon.addEventListener("mousemove", (event) => tooltip.move(event));
  icon.addEventListener("focus", () => tooltip.show(text, icon.getBoundingClientRect()));
  for (const type of ["mouseleave", "blur"]) icon.addEventListener(type, () => tooltip.hide());
  return icon;
}
