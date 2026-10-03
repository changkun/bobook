// A small string builder for SVG. Figures render to markup strings so the same
// code runs at build time and in the browser. Colors are passed as CSS values
// (usually theme tokens from theme.ts) and emitted in a style attribute, so a
// var(--fig-...) reference resolves against the page theme in both the static
// fallback and the live figure.

export type Attrs = Record<string, string | number | boolean | null | undefined>;

const STYLE_PROPS = new Set(["fill", "stroke"]);

export function esc(s: string | number): string {
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// Round coordinates so markup stays compact and stable across platforms.
export function r(n: number): number {
  return Math.round(n * 10) / 10;
}

// A caller's own `style` is merged with the color properties, after them, so
// passing both fill and style never produces two style attributes (the
// second of which a browser would drop).
function attrString(a: Attrs): string {
  let out = "";
  let style = "";
  let own = "";
  for (const [k, v] of Object.entries(a)) {
    if (v == null || v === false) continue;
    if (STYLE_PROPS.has(k)) style += `${k}:${v};`;
    else if (k === "style") own = String(v);
    else if (v === true) out += ` ${k}`;
    else out += ` ${k}="${typeof v === "number" ? r(v) : esc(v)}"`;
  }
  const all = style + own;
  return all ? `${out} style="${esc(all)}"` : out;
}

const SELF_CLOSING = new Set(["rect", "circle", "line", "path", "polyline", "polygon", "ellipse", "use", "stop"]);

// el("rect", { x: 1, fill: C.c1 }) or el("g", {}, child1, child2)
export function el(tag: string, a: Attrs = {}, ...children: Array<string | false | null | undefined>): string {
  const inner = children.filter(Boolean).join("");
  return inner || !SELF_CLOSING.has(tag) ? `<${tag}${attrString(a)}>${inner}</${tag}>` : `<${tag}${attrString(a)}/>`;
}

// Text node. `class` picks a typographic role from the stylesheet
// (fig-t-strong, fig-t-muted, fig-t-num, fig-t-halo). A one-letter symbol
// written base_sub (x_n, f_i, σ_n) is set with a lowered subscript, and
// base^sup (σ^2) with a raised superscript, so readouts show notation instead
// of raw underscores and carets.
const SCRIPT = /(?<![A-Za-z0-9])([A-Za-zα-ωΑ-Ω])([_^])(\{[^}]*\}|[A-Za-z0-9α-ω*+−-]+)/g;

// Estimated width in pixels of a label in the figures' 12 px interface font:
// `latin` per Latin character (an average) and a full em per Chinese
// character or full-width punctuation mark. For Latin-only text this equals
// length * latin, so English layouts do not move.
export function labelWidth(s: string, latin = 6.3, em = 12): number {
  let wide = 0, narrow = 0;
  for (const ch of s) if (/[\u2e80-\u9fff\u3000-\u303f\uff00-\uffef“”‘’]/.test(ch)) wide++; else narrow++;
  return narrow * latin + wide * em;
}

export function text(x: number, y: number, s: string | number, a: Attrs = {}): string {
  const str = String(s);
  if (!/[_^]/.test(str)) return el("text", { x, y, ...a }, esc(str));
  const size = Number(a["font-size"] ?? 12);
  const fs = Math.round(size * 0.75);
  let out = "", i = 0, shifted = 0;
  for (const m of str.matchAll(SCRIPT)) {
    const pre = str.slice(i, m.index) + m[1];
    out += shifted ? `<tspan dy="${-shifted}">${esc(pre)}</tspan>` : esc(pre);
    const d = m[2] === "_" ? Math.round(size * 2.8) / 10 : -Math.round(size * 3.6) / 10;
    const body = m[3].replace(/^\{|\}$/g, "");
    out += `<tspan dy="${d}" font-size="${fs}">${esc(body)}</tspan>`;
    shifted = d;
    i = (m.index ?? 0) + m[0].length;
  }
  const rest = str.slice(i);
  if (rest) out += shifted ? `<tspan dy="${-shifted}">${esc(rest)}</tspan>` : esc(rest);
  else if (shifted) out += `<tspan dy="${-shifted}"></tspan>`;
  return el("text", { x, y, ...a }, out);
}

export function g(a: Attrs, ...children: Array<string | false | null | undefined>): string {
  return el("g", a, ...children);
}

// The root element. viewBox only, no width attribute: the stylesheet makes the
// SVG fill its column, so there is never a fixed width wider than the column.
export function svg(w: number, h: number, label: string, ...children: Array<string | false | null | undefined>): string {
  return `<svg class="fig-svg" viewBox="0 0 ${r(w)} ${r(h)}" role="img" aria-label="${esc(label)}" xmlns="http://www.w3.org/2000/svg">${children.filter(Boolean).join("")}</svg>`;
}

export function linePath(pts: Array<[number, number]>): string {
  return pts.map(([x, y], i) => `${i ? "L" : "M"}${r(x)},${r(y)}`).join("");
}

// A closed band between an upper and a lower curve sharing x positions:
// credible intervals, shaded regions under a density.
export function bandPath(upper: Array<[number, number]>, lower: Array<[number, number]>): string {
  if (!upper.length) return "";
  return linePath(upper) + lower.slice().reverse().map(([x, y]) => `L${r(x)},${r(y)}`).join("") + "Z";
}

export function hatch(id: string, color: string, gap = 5, width = 1.2): string {
  return el("pattern", { id, width: gap, height: gap, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" },
    el("line", { x1: 0, y1: 0, x2: 0, y2: gap, stroke: color, "stroke-width": width }));
}

// An arrowhead marker for annotation lines; reference it with
// marker-end="url(#<id>)".
export function arrowMarker(id: string, color: string, size = 6): string {
  return el("marker", { id, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: size, markerHeight: size, orient: "auto-start-reverse" },
    el("path", { d: "M0,0L10,5L0,10z", fill: color }));
}
