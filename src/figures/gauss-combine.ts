// Two ways to combine two Gaussians, on one axis. "Sum of variables" draws
// the density of X + Y for independent X ~ N(m1, s1^2) and Y ~ N(m2, s2^2):
// the means add and the variances add, so the result is wider than either
// input. "Product of densities" multiplies the two density curves pointwise
// and renormalizes: the precisions add and the mean is a precision-weighted
// average, so the result is narrower than either input. The raw product is
// drawn too; its area is the constant Z = N(m1; m2, s1^2 + s2^2). Drag near
// either input's peak to move its mean.

import { defineFigure, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { clamp, grid } from "./lib/scale.ts";
import { normalPdf } from "./lib/stats.ts";
import { clip, frame, frameAxes, hitArea, legend } from "./lib/plot.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "x",
    y: "density",
    inX: "density of X",
    inY: "density of Y",
    sum: "density of X + Y",
    inP: "p(x) = N(x; μ_1, σ_1²)",
    inQ: "q(x) = N(x; μ_2, σ_2²)",
    prod: "normalized product",
    raw: "raw product p(x) q(x)",
    sumMean: "mean μ_1 + μ_2 = {m}",
    sumVar: "variance σ_1² + σ_2² = {v},  sd {s}",
    sumNote: "wider than either input (largest sd {mx}): variances add",
    prodPrec: "precision 1/σ² = 1/σ_1² + 1/σ_2² = {pr},  sd {s}",
    prodMean: "mean = {w1} μ_1 + {w2} μ_2 = {m}  (weights ∝ precision)",
    prodZ: "area of the raw product Z = N(μ_1; μ_2, σ_1² + σ_2²) = {z}",
    describeSum: "Sum of independent Gaussians: X has mean {m1} and sd {s1}, Y has mean {m2} and sd {s2}; X + Y has mean {m} and sd {s}.",
    describeProd: "Product of two Gaussian densities with means {m1} and {m2} and sds {s1} and {s2}: after normalizing, a Gaussian with mean {m} and sd {s}, narrower than both. The raw product has area {z}.",
  },
  zh: {
    x: "x",
    y: "密度",
    inX: "X 的密度",
    inY: "Y 的密度",
    sum: "X + Y 的密度",
    inP: "p(x) = N(x; μ_1, σ_1²)",
    inQ: "q(x) = N(x; μ_2, σ_2²)",
    prod: "归一化的乘积",
    raw: "原始乘积 p(x) q(x)",
    sumMean: "均值 μ_1 + μ_2 = {m}",
    sumVar: "方差 σ_1² + σ_2² = {v}，标准差 {s}",
    sumNote: "比两个输入都宽（输入中较大的标准差为 {mx}）：方差相加",
    prodPrec: "精度 1/σ² = 1/σ_1² + 1/σ_2² = {pr}，标准差 {s}",
    prodMean: "均值 = {w1} μ_1 + {w2} μ_2 = {m}（权重 ∝ 精度）",
    prodZ: "原始乘积的面积 Z = N(μ_1; μ_2, σ_1² + σ_2²) = {z}",
    describeSum: "独立高斯变量之和：X 的均值为 {m1}、标准差为 {s1}，Y 的均值为 {m2}、标准差为 {s2}；X + Y 的均值为 {m}、标准差为 {s}。",
    describeProd: "两个高斯密度的乘积，均值分别为 {m1} 和 {m2}，标准差分别为 {s1} 和 {s2}：归一化后是均值为 {m}、标准差为 {s} 的高斯分布，比两者都窄。原始乘积的面积为 {z}。",
  },
};

const params = {
  op: { kind: "choice", label: { en: "Combine", zh: "组合方式" }, options: [{ value: "sum", label: { en: "Sum of variables", zh: "变量之和" } }, { value: "product", label: { en: "Product of densities", zh: "密度之积" } }], default: "product", control: "buttons" },
  m1: { kind: "range", label: { en: "Mean μ₁", zh: "均值 μ₁" }, min: -3, max: 3, default: -1, step: 0.1 },
  s1: { kind: "range", label: { en: "Sd σ₁", zh: "标准差 σ₁" }, min: 0.3, max: 2.5, default: 1.2, step: 0.05 },
  m2: { kind: "range", label: { en: "Mean μ₂", zh: "均值 μ₂" }, min: -3, max: 3, default: 1.5, step: 0.1 },
  s2: { kind: "range", label: { en: "Sd σ₂", zh: "标准差 σ₂" }, min: 0.3, max: 2.5, default: 0.6, step: 0.05 },
  grab: { kind: "range", label: { en: "Dragged input", zh: "拖动的输入" }, min: 0, max: 2, default: 0, step: 1, control: false },
} as const;

type P = { op: "sum" | "product"; m1: number; s1: number; m2: number; s2: number; grab: number };

const XS = grid(-6, 6, 241);

function result(p: P) {
  if (p.op === "sum") {
    const v = p.s1 * p.s1 + p.s2 * p.s2;
    return { m: p.m1 + p.m2, s: Math.sqrt(v), v, z: NaN, w1: NaN, w2: NaN, prec: NaN };
  }
  const p1 = 1 / (p.s1 * p.s1), p2 = 1 / (p.s2 * p.s2);
  const prec = p1 + p2, v = 1 / prec;
  const m = v * (p1 * p.m1 + p2 * p.m2);
  const z = normalPdf(p.m1, p.m2, Math.sqrt(p.s1 * p.s1 + p.s2 * p.s2));
  return { m, s: Math.sqrt(v), v, z, w1: p1 / prec, w2: p2 / prec, prec };
}

function describe(st: State<P>): string {
  const p = st.p, r = result(p);
  const base = { m1: fixed(p.m1, 1), m2: fixed(p.m2, 1), s1: fixed(p.s1, 2), s2: fixed(p.s2, 2), m: fixed(r.m, 2), s: fixed(r.s, 2), z: fixed(r.z, 3) };
  const L = labels[st.lang ?? "en"];
  return tpl(p.op === "sum" ? L.describeSum : L.describeProd, base);
}

function render(st: State<P>): string {
  const p = st.p, L = labels[st.lang ?? "en"], w = st.w;
  const narrow = w < 480;
  const r = result(p);
  const d1 = XS.map((x) => normalPdf(x, p.m1, p.s1));
  const d2 = XS.map((x) => normalPdf(x, p.m2, p.s2));
  const dr = XS.map((x) => normalPdf(x, r.m, r.s));
  const raw = XS.map((_, i) => d1[i] * d2[i]);
  const isSum = p.op === "sum";
  const lg = legend(narrow ? 8 : 52, 14, w - 16, [
    { kind: "band", color: C.c5, label: isSum ? L.inX : L.inP },
    { kind: "band", color: C.c6, label: isSum ? L.inY : L.inQ },
    { kind: "line", color: C.model, label: isSum ? L.sum : L.prod },
    ...(isSum ? [] : [{ kind: "dash" as const, color: C.ink2, label: L.raw }]),
  ]);
  const top = 14 + lg.height + 8;
  const peak = Math.max(...d1, ...d2, ...dr);
  const f = frame({ w, top, height: narrow ? 170 : 210, xDomain: [-6, 6], yDomain: [0, peak * 1.08], yTitle: L.y });
  const cid = `${st.uid}-clip`;
  const area = (ys: number[], color: string) => el("path", {
    d: linePath(XS.map((x, i) => [f.x(x), f.y(ys[i])])) + `L${f.x(6)},${f.y(0)}L${f.x(-6)},${f.y(0)}Z`,
    fill: color, "fill-opacity": 0.16, stroke: color, "stroke-width": 1.6,
  });
  const mark = (m: number, color: string) => el("line", { x1: f.x(m), x2: f.x(m), y1: f.bottom, y2: f.bottom + 6, stroke: color, "stroke-width": 2.5 });
  const parts: string[] = [
    el("defs", {}, clip(f, cid)),
    lg.svg,
    frameAxes(f, { yTitle: L.y, xTitle: L.x, yCount: 3 }),
    g({ "clip-path": `url(#${cid})` },
      area(d1, C.c5),
      area(d2, C.c6),
      isSum ? "" : el("path", { d: linePath(XS.map((x, i) => [f.x(x), f.y(raw[i])])), fill: "none", stroke: C.ink2, "stroke-width": 1.4, "stroke-dasharray": "4 3" }),
      el("path", { d: linePath(XS.map((x, i) => [f.x(x), f.y(dr[i])])), fill: "none", stroke: C.model, "stroke-width": 2.6 }),
    ),
    mark(p.m1, C.c5), mark(p.m2, C.c6), mark(r.m, C.model),
  ];
  // readout under the axis
  let y = f.bottom + 52;
  const x0 = narrow ? 8 : f.left;
  const line = (s: string) => { parts.push(text(x0, y, s, { "font-size": narrow ? TYPE.small : TYPE.body, class: "fig-t-muted fig-t-num" })); y += 18; };
  if (isSum) {
    line(tpl(L.sumMean, { m: fixed(r.m, 2) }));
    line(tpl(L.sumVar, { v: fixed(r.v, 2), s: fixed(r.s, 2) }));
    line(tpl(L.sumNote, { mx: fixed(Math.max(p.s1, p.s2), 2) }));
  } else {
    line(tpl(L.prodPrec, { pr: fixed(r.prec, 2), s: fixed(r.s, 2) }));
    line(tpl(L.prodMean, { w1: fixed(r.w1, 2), w2: fixed(r.w2, 2), m: fixed(r.m, 2) }));
    line(tpl(L.prodZ, { z: fixed(r.z, 3) }));
  }
  parts.push(hitArea(f));
  return svg(w, y - 4, describe(st), ...parts);
}

export default defineFigure({
  name: "gauss-combine",
  title: { en: "Adding two Gaussian variables versus multiplying two Gaussian densities", zh: "两个高斯变量相加与两个高斯密度相乘" },
  labels,
  params,
  hint: { en: "Use the sliders, or drag near either input's peak to move its mean.", zh: "使用滑块，或在任一输入的峰值附近拖动，以移动它的均值。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    const q = p as P;
    if (e.target !== "plot" || !e.data) return null;
    const x = Math.round(clamp(e.data.x, -3, 3) * 10) / 10;
    if (e.phase === "down" || e.phase === "click") {
      const which = Math.abs(e.data.x - q.m1) <= Math.abs(e.data.x - q.m2) ? 1 : 2;
      return which === 1 ? { ...q, grab: 1, m1: x } : { ...q, grab: 2, m2: x };
    }
    if (e.phase === "move") {
      if (q.grab === 1) return { ...q, m1: x };
      if (q.grab === 2) return { ...q, m2: x };
      return null;
    }
    return { ...q, grab: 0 };
  },
});
