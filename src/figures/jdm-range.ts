// Range normalization makes a person's scale session-relative. One fixed
// utility u(x) over a design parameter; two sessions show the person
// different parts of the space: session A the whole range, session B either
// the region near the best design or a poor region. A person whose values are
// range-normalized experiences v_S(x) = (1 - λ) u(x) + λ (u(x) - min_S) /
// (max_S - min_S), where min_S and max_S are the worst and best utility shown
// in session S, and answers comparisons with probit noise on that scale.
// Ratings (100 v) of the same design differ between sessions; the order of a
// pair does not, but the probability of the answer does, so a model fitted to
// one session's comparisons learns a utility on that session's scale. Outside
// the region a session explored, the normalized scale is extrapolated
// (dotted). An idealized model in the spirit of range-adaptation studies, not
// fitted to data.

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid } from "./lib/scale.ts";
import { Phi } from "./lib/stats.ts";
import { clip, curve, frame, frameAxes, hitArea } from "./lib/plot.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "design parameter x",
    y: "experienced value",
    a: "session A (explores everything)",
    bNear: "session B (explores near the best)",
    bLow: "session B (explores a poor region)",
    absolute: "absolute utility u(x)",
    range: "explored",
    readTitle: "The same two designs in the two sessions",
    colA: "session A",
    colB: "session B",
    rate1: "rating of x₁ (0 to 100)",
    rate2: "rating of x₂ (0 to 100)",
    prob: "P(x₂ preferred to x₁)",
    describe: "With adaptation {l}, design x₁ = {x1} is rated {ra1} in session A and {rb1} in session B; x₂ = {x2} is rated {ra2} and {rb2}. The probability of preferring x₂ is {pa} in session A and {pb} in session B.",
  },
  zh: {
    x: "设计参数 x",
    y: "体验到的价值",
    a: "会话 A（探索全部范围）",
    bNear: "会话 B（在最优设计附近探索）",
    bLow: "会话 B（在较差的区域探索）",
    absolute: "绝对效用 u(x)",
    range: "已探索",
    readTitle: "同样两个设计在两次会话中的情况",
    colA: "会话 A",
    colB: "会话 B",
    rate1: "x₁ 的评分（0 至 100）",
    rate2: "x₂ 的评分（0 至 100）",
    prob: "P(x₂ 优于 x₁)",
    describe: "适应程度为 {l} 时，设计 x₁ = {x1} 在会话 A 中的评分为 {ra1}，在会话 B 中为 {rb1}；x₂ = {x2} 的评分分别为 {ra2} 和 {rb2}。偏好 x₂ 的概率在会话 A 中为 {pa}，在会话 B 中为 {pb}。",
  },
};

const params = {
  lambda: { kind: "range", label: { en: "Range adaptation λ", zh: "范围适应 λ" }, min: 0, max: 1, default: 1, step: 0.05 },
  regionB: { kind: "choice", label: { en: "Session B explores", zh: "会话 B 的探索范围" }, options: [{ value: "near", label: { en: "Near the best", zh: "最优设计附近" } }, { value: "low", label: { en: "A poor region", zh: "较差的区域" } }], default: "near" },
  x1: { kind: "range", label: { en: "Design x₁", zh: "设计 x₁" }, min: 0, max: 1, default: 0.58, step: 0.01 },
  x2: { kind: "range", label: { en: "Design x₂", zh: "设计 x₂" }, min: 0, max: 1, default: 0.72, step: 0.01 },
} as const;

type P = { lambda: number; regionB: "near" | "low"; x1: number; x2: number };

const SIGMA = 0.1; // probit noise per option on the experienced scale
const XS = grid(0, 1, 201);
const u = (x: number) => 0.1 + 0.8 * Math.exp(-((x - 0.68) ** 2) / (2 * 0.17 * 0.17));
const RANGES: Record<string, [number, number]> = { A: [0, 1], near: [0.5, 0.86], low: [0.06, 0.42] };

function scaleFor(r: [number, number]) {
  let lo = Infinity, hi = -Infinity;
  for (const x of XS) if (x >= r[0] - 1e-9 && x <= r[1] + 1e-9) { lo = Math.min(lo, u(x)); hi = Math.max(hi, u(x)); }
  return { lo, hi };
}

function experienced(p: P, r: [number, number]) {
  const { lo, hi } = scaleFor(r);
  return (x: number) => (1 - p.lambda) * u(x) + p.lambda * (u(x) - lo) / (hi - lo);
}

function compute(p: P) {
  const rB = RANGES[p.regionB];
  const vA = experienced(p, RANGES.A), vB = experienced(p, rB);
  const pr = (v: (x: number) => number) => Phi((v(p.x2) - v(p.x1)) / (Math.SQRT2 * SIGMA));
  return { rB, vA, vB, pa: pr(vA), pb: pr(vB) };
}

const rating = (v: number) => String(Math.round(100 * v));

function describe(st: State<P>): string {
  const p = st.p;
  const c = compute(p);
  return tpl(labels[st.lang ?? "en"].describe, {
    l: fixed(p.lambda, 2), x1: fixed(p.x1, 2), x2: fixed(p.x2, 2),
    ra1: rating(c.vA(p.x1)), rb1: rating(c.vB(p.x1)), ra2: rating(c.vA(p.x2)), rb2: rating(c.vB(p.x2)),
    pa: fixed(c.pa, 2), pb: fixed(c.pb, 2),
  });
}

function render(st: State<P>): string {
  const L = labels[st.lang ?? "en"];
  const p = st.p, w = st.w, narrow = w < 480;
  const c = compute(p);
  const parts: string[] = [];
  // legend rows
  const lgY = 14;
  const lgItems: Array<[string, string, string]> = [
    [L.absolute, C.truth, "5 4"], [L.a, C.c1, ""], [p.regionB === "near" ? L.bNear : L.bLow, C.c2, ""],
  ];
  let lx = narrow ? 8 : 52, ly = lgY;
  for (const [lab, col, dash] of lgItems) {
    const tw = labelWidth(lab, 6.1) + 30;
    if (lx + tw > w - 8 && lx > 52) { lx = narrow ? 8 : 52; ly += 18; }
    parts.push(el("line", { x1: lx, x2: lx + 18, y1: ly - 1, y2: ly - 1, stroke: col, "stroke-width": 2.2, "stroke-dasharray": dash || undefined }), text(lx + 24, ly + 3, lab, { "font-size": TYPE.small, class: "fig-t-muted" }));
    lx += tw;
  }
  const top = ly + 28;
  const f = frame({ w, top, height: narrow ? 190 : 230, yDomain: [-0.35, 1.25], yTitle: L.y });
  const cid = `${st.uid}-clip`;
  parts.push(el("defs", {}, clip(f, cid)));
  parts.push(frameAxes(f, { yTitle: L.y, xTitle: L.x, yCount: 4 }));
  // explored ranges as shaded strips
  const strip = (r: [number, number], color: string, y: number, lab: string) => g({},
    el("rect", { x: f.x(r[0]), y, width: f.x(r[1]) - f.x(r[0]), height: 6, rx: 3, fill: color, opacity: 0.75 }),
    text(f.x(r[1]) - 2, y - 3, lab, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-faint" }),
  );
  parts.push(el("rect", { x: f.x(c.rB[0]), y: f.top, width: f.x(c.rB[1]) - f.x(c.rB[0]), height: f.bottom - f.top, fill: C.acqFill }));
  const inB = XS.filter((x) => x >= c.rB[0] - 1e-9 && x <= c.rB[1] + 1e-9);
  const outLo = XS.filter((x) => x <= c.rB[0] + 1e-9), outHi = XS.filter((x) => x >= c.rB[1] - 1e-9);
  parts.push(g({ "clip-path": `url(#${cid})` },
    curve(f, XS, XS.map(u), { stroke: C.truth, "stroke-dasharray": "5 4", "stroke-width": 1.6 }),
    curve(f, XS, XS.map(c.vA), { stroke: C.c1, "stroke-width": 2.2 }),
    outLo.length > 1 ? curve(f, outLo, outLo.map(c.vB), { stroke: C.c2, "stroke-width": 1.6, "stroke-dasharray": "1.5 3.5" }) : "",
    outHi.length > 1 ? curve(f, outHi, outHi.map(c.vB), { stroke: C.c2, "stroke-width": 1.6, "stroke-dasharray": "1.5 3.5" }) : "",
    curve(f, inB, inB.map(c.vB), { stroke: C.c2, "stroke-width": 2.4 }),
  ));
  // the test pair
  for (const [name, x] of [["x₁", p.x1], ["x₂", p.x2]] as const) {
    parts.push(el("line", { x1: f.x(x), x2: f.x(x), y1: f.top, y2: f.bottom, stroke: C.ink3, "stroke-dasharray": "3 3" }));
    parts.push(el("circle", { cx: f.x(x), cy: f.y(c.vA(x)), r: 4, fill: C.c1, stroke: C.paper, "stroke-width": 1.4 }));
    parts.push(el("circle", { cx: f.x(x), cy: f.y(c.vB(x)), r: 4, fill: C.c2, stroke: C.paper, "stroke-width": 1.4 }));
    parts.push(text(f.x(x), f.top - 4, name, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-strong" }));
  }
  parts.push(strip(c.rB, C.c2, f.bottom - 12, L.range));
  parts.push(hitArea(f));
  // readout table
  let y = f.bottom + 52;
  parts.push(text(narrow ? 8 : f.left, y, L.readTitle, { "font-size": TYPE.label, class: "fig-t-strong" }));
  y += 20;
  const c0 = narrow ? 8 : f.left, cA = narrow ? w - 120 : f.left + 260, cB = narrow ? w - 50 : f.left + 360;
  parts.push(text(cA, y, L.colA, { "font-size": TYPE.small, "text-anchor": "middle", style: `fill:${C.c1}`, class: "fig-t-strong" }), text(cB, y, L.colB, { "font-size": TYPE.small, "text-anchor": "middle", style: `fill:${C.c2}`, class: "fig-t-strong" }));
  const rowsT: Array<[string, string, string]> = [
    [L.rate1, rating(c.vA(p.x1)), rating(c.vB(p.x1))],
    [L.rate2, rating(c.vA(p.x2)), rating(c.vB(p.x2))],
    [L.prob, fixed(c.pa, 2), fixed(c.pb, 2)],
  ];
  for (const [lab, a, b] of rowsT) {
    y += 19;
    parts.push(el("line", { x1: c0, x2: narrow ? w - 8 : cB + 50, y1: y - 14, y2: y - 14, stroke: C.grid }));
    parts.push(text(c0, y, lab, { "font-size": TYPE.body }), text(cA, y, a, { "font-size": TYPE.body, "text-anchor": "middle", class: "fig-t-num" }), text(cB, y, b, { "font-size": TYPE.body, "text-anchor": "middle", class: "fig-t-num" }));
  }
  return svg(w, y + 12, describe(st), ...parts);
}

export default defineFigure({
  name: "jdm-range",
  title: { en: "Range normalization makes the scale session-relative", zh: "范围归一化使量表依赖于会话" },
  labels,
  params,
  hint: { en: "Click the plot to move the nearer of the two designs, or use the sliders.", zh: "点击图中位置，移动两个设计中较近的一个，或使用滑块。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click" || e.target !== "plot" || !e.data) return null;
    const x = Math.round(Math.min(1, Math.max(0, e.data.x)) * 100) / 100;
    return Math.abs(x - p.x1) <= Math.abs(x - p.x2) ? { ...p, x1: x } : { ...p, x2: x };
  },
});
