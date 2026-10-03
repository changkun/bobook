// Improvement at a single input. The model's belief about f(x) is a Gaussian;
// what matters for improvement-based acquisition functions is the difference
// D = f(x) - f*_n, Gaussian with mean delta and standard deviation s. The top
// panel draws the density of D: the shaded area to the right of zero is the
// probability of improvement, and the second curve, the improvement max(D, 0)
// times the density, has area equal to the expected improvement. The lower
// panels hold delta fixed and vary s, so the reader sees that EI always grows
// with uncertainty while PI shrinks with it once the mean is above the
// incumbent.

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { Phi, phi } from "./lib/stats.ts";
import { fixed, tpl } from "./lib/format.ts";
import { legend, type Frame } from "./lib/plot.ts";

const labels = {
  en: {
    d: "improvement over the best value so far, D = f(x) − f*ₙ",
    density: "density of D",
    weighted: "max(D, 0) × density (area = EI)",
    pi: "P(D > 0) (area = PI)",
    readout: "δ = {d}, s = {s}:  PI = {pi},  EI = {ei}",
    piTitle: "PI as s grows (δ fixed)",
    eiTitle: "EI as s grows (δ fixed)",
    s: "standard deviation s",
    describe: "The difference D between f(x) and the best value so far is Gaussian with mean {d} and standard deviation {s}. The probability of improvement is {pi} and the expected improvement is {ei}. At this mean, increasing s {piDir} the probability of improvement and increases the expected improvement.",
    dec: "decreases",
    inc: "increases",
    same: "does not change",
  },
  zh: {
    d: "相对当前最优值的改进 D = f(x) − f*ₙ",
    density: "D 的密度",
    weighted: "max(D, 0) × 密度（面积 = EI）",
    pi: "D 大于 0 的概率（面积 = PI）",
    readout: "δ = {d}，s = {s}：PI = {pi}，EI = {ei}",
    piTitle: "PI 随 s 的变化（δ 固定）",
    eiTitle: "EI 随 s 的变化（δ 固定）",
    s: "标准差 s",
    describe: "f(x) 与当前最优值之差 D 服从均值为 {d}、标准差为 {s} 的高斯分布。改进概率为 {pi}，期望改进为 {ei}。在这个均值下，增大 s {piDir}改进概率，并增大期望改进。",
    dec: "会减小",
    inc: "会增大",
    same: "不改变",
  },
};

const params = {
  delta: { kind: "range", label: { en: "Mean δ", zh: "均值 δ" }, min: -1.5, max: 1.5, default: -0.3, step: 0.05 },
  s: { kind: "range", label: { en: "Std. dev. s", zh: "标准差 s" }, min: 0.05, max: 1.5, default: 0.5, step: 0.05 },
} as const;

type P = { delta: number; s: number };

const DX = grid(-3, 3, 241);
const SX = grid(0.02, 1.5, 120);

const piOf = (d: number, s: number) => Phi(d / s);
const eiOf = (d: number, s: number) => d * Phi(d / s) + s * phi(d / s);

function describe(st: State<P>): string {
  const { delta: d, s } = st.p;
  const L = labels[st.lang ?? "en"];
  return tpl(L.describe, {
    d: fixed(d, 2), s: fixed(s, 2), pi: fixed(piOf(d, s), 3), ei: fixed(eiOf(d, s), 3),
    piDir: d > 0 ? L.dec : d < 0 ? L.inc : L.same,
  });
}

function panel(left: number, right: number, top: number, h: number, xd: [number, number], yd: [number, number], w: number): Frame {
  return { x: linear(xd, [left, right]), y: linear(yd, [top + h, top]), left, right, top, bottom: top + h, w };
}

function render(st: State<P>): string {
  const { delta: d, s } = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const parts: string[] = [];

  const lg = legend(narrow ? 8 : 40, 14, st.w - 16, [
    { kind: "line", color: C.model, label: L.density },
    { kind: "band", color: C.acqFill, label: L.pi },
    { kind: "dash", color: C.acq, label: L.weighted },
  ]);
  parts.push(lg.svg);

  // Top: the density of D.
  const top = 14 + lg.height + 22;
  const left = narrow ? 34 : 48;
  const H1 = narrow ? 150 : 170;
  const dens = DX.map((x) => phi((x - d) / s) / s);
  const wgt = DX.map((x, i) => Math.max(x, 0) * dens[i]);
  const ymax = Math.max(1.0, Math.max(...dens), Math.max(...wgt)) * 1.08;
  const f = panel(left, st.w - 10, top, H1, [-3, 3], [0, ymax], st.w);
  const area = (ys: number[], from: number) => {
    const xs = DX.filter((x) => x >= from);
    const i0 = DX.length - xs.length;
    return `M${f.x(xs[0])},${f.y(0)}` + xs.map((x, k) => `L${f.x(x)},${f.y(ys[i0 + k])}`).join("") + `L${f.x(xs[xs.length - 1])},${f.y(0)}Z`;
  };
  const line = (ys: number[]) => DX.map((x, i) => `${i ? "L" : "M"}${f.x(x)},${f.y(Math.min(ys[i], ymax))}`).join("");
  parts.push(
    text(f.left, top - 8, tpl(L.readout, { d: fixed(d, 2), s: fixed(s, 2), pi: fixed(piOf(d, s), 3), ei: fixed(eiOf(d, s), 3) }), { "font-size": TYPE.small, class: "fig-t-num fig-t-strong" }),
    axis({ scale: f.y, orient: "left", at: f.left, span: [f.left, f.right], count: 3 }),
    axis({ scale: f.x, orient: "bottom", at: f.bottom, span: [f.top, f.bottom], title: narrow ? "D = f(x) − f*ₙ" : L.d, ticks: [-3, -2, -1, 0, 1, 2, 3], grid: false }),
    el("path", { d: area(dens, 0), fill: C.acqFill, stroke: "none" }),
    el("path", { d: line(dens), fill: "none", stroke: C.model, "stroke-width": 2 }),
    el("path", { d: line(wgt), fill: "none", stroke: C.acq, "stroke-width": 2, "stroke-dasharray": "5 3" }),
    el("line", { x1: f.x(0), x2: f.x(0), y1: f.top, y2: f.bottom, stroke: C.ink, "stroke-width": 1.4 }),
    text(f.x(0) + 4, f.top + 10, "f*ₙ", { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
    el("line", { x1: f.x(d), x2: f.x(d), y1: f.y(phi(0) / s > ymax ? ymax : phi(0) / s), y2: f.bottom, stroke: C.model, "stroke-width": 1, "stroke-dasharray": "2 3" }),
    text(f.x(d), f.bottom - 4, "δ", { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted fig-t-halo" }),
  );

  // Bottom: PI and EI against s at this delta.
  const bTop = f.bottom + 70;
  const H2 = narrow ? 92 : 104;
  const piCurve = SX.map((v) => piOf(d, v));
  const eiCurve = SX.map((v) => eiOf(d, v));
  const eiMax = Math.max(...eiCurve) * 1.1 || 1;
  const colW = narrow ? st.w - left - 10 : (st.w - left - 10 - 56) / 2;
  const fp = panel(left, left + colW, bTop, H2, [0, 1.5], [0, 1], st.w);
  const eTop = narrow ? bTop + H2 + 62 : bTop;
  const eLeft = narrow ? left : left + colW + 56;
  const fe = panel(eLeft, eLeft + colW, eTop, H2, [0, 1.5], [0, eiMax], st.w);
  const path = (fr: Frame, ys: number[]) => SX.map((v, i) => `${i ? "L" : "M"}${fr.x(v)},${fr.y(ys[i])}`).join("");
  for (const [fr, ys, title, val] of [[fp, piCurve, L.piTitle, piOf(d, s)], [fe, eiCurve, L.eiTitle, eiOf(d, s)]] as Array<[Frame, number[], string, number]>) {
    parts.push(
      text(fr.left, fr.top - 10, title, { "font-size": TYPE.small, class: "fig-t-muted" }),
      axis({ scale: fr.y, orient: "left", at: fr.left, span: [fr.left, fr.right], count: 3 }),
      axis({ scale: fr.x, orient: "bottom", at: fr.bottom, span: [fr.top, fr.bottom], title: L.s, ticks: [0, 0.5, 1, 1.5], grid: false }),
      el("path", { d: path(fr, ys), fill: "none", stroke: C.acq, "stroke-width": 2 }),
      el("line", { x1: fr.x(s), x2: fr.x(s), y1: fr.top, y2: fr.bottom, stroke: C.ink3, "stroke-width": 1, "stroke-dasharray": "3 3" }),
      el("circle", { cx: fr.x(s), cy: fr.y(val), r: 4.2, fill: C.acq, stroke: C.paper, "stroke-width": 1.5 }),
    );
  }
  // The EI floor max(delta, 0): what EI would be with no uncertainty.
  if (d > 0) parts.push(el("line", { x1: fe.left, x2: fe.right, y1: fe.y(d), y2: fe.y(d), stroke: C.ink3, "stroke-width": 1, "stroke-dasharray": "5 4" }),
    text(fe.right - 2, fe.y(d) - 4, "δ", { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo" }));

  const H = fe.bottom + 44;
  return svg(st.w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "acq-improvement",
  title: { en: "Probability and expected improvement at one input", zh: "单个输入处的改进概率与期望改进" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
