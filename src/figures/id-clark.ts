// The expected maximum of two jointly Gaussian values. The reader sets the
// two means, the two standard deviations, and the correlation. The plot shows
// the densities of A and B, and the exact density of max(A, B),
//   p(m) = p_A(m) P(B <= m | A = m) + p_B(m) P(A <= m | B = m),
// with its mean marked. The readout compares that mean, integrated
// numerically from the density, with Clark's closed form
//   mu_A Phi(d / s) + mu_B Phi(-d / s) + s phi(d / s),
// and shows the bonus over the better of the two means.

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid } from "./lib/scale.ts";
import { expectedMax2 } from "./lib/acq.ts";
import { Phi, normalPdf } from "./lib/stats.ts";
import { band, clip, curve, frame, frameAxes, legend } from "./lib/plot.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "value",
    y: "density",
    a: "density of A",
    b: "density of B",
    max: "density of max(A, B)",
    mean: "E[max]",
    better: "better mean",
    readout: "E[max(A, B)] = {e} by Clark's formula, {n} by integrating the density · better mean {m} · bonus {d} · s = {s}",
    r1: "E[max(A, B)] = {e} (formula), {n} (integral)",
    r2: "better mean {m} · bonus {d} · s = {s}",
    describe: "A has mean {ma} and standard deviation {sa}; B has mean {mb} and standard deviation {sb}; their correlation is {r}. The expected maximum is {e}, which is {d} above the better mean; the difference A − B has standard deviation {s}.",
  },
  zh: {
    x: "取值",
    y: "密度",
    a: "A 的密度",
    b: "B 的密度",
    max: "max(A, B) 的密度",
    mean: "E[max]",
    better: "较优均值",
    readout: "E[max(A, B)] = {e}（Clark 公式），{n}（对密度数值积分）· 较优均值 {m} · 额外收益 {d} · s = {s}",
    r1: "E[max(A, B)] = {e}（公式），{n}（积分）",
    r2: "较优均值 {m} · 额外收益 {d} · s = {s}",
    describe: "A 的均值为 {ma}，标准差为 {sa}；B 的均值为 {mb}，标准差为 {sb}；两者的相关系数为 {r}。期望最大值为 {e}，比较优均值高 {d}；差 A − B 的标准差为 {s}。",
  },
};

const params = {
  muA: { kind: "range", label: { en: "Mean of A", zh: "A 的均值" }, min: -2, max: 2, default: 0.4, step: 0.05 },
  muB: { kind: "range", label: { en: "Mean of B", zh: "B 的均值" }, min: -2, max: 2, default: 0, step: 0.05 },
  sdA: { kind: "range", label: { en: "Sd of A", zh: "A 的标准差" }, min: 0.1, max: 2, default: 1, step: 0.05 },
  sdB: { kind: "range", label: { en: "Sd of B", zh: "B 的标准差" }, min: 0.1, max: 2, default: 0.5, step: 0.05 },
  rho: { kind: "range", label: { en: "Correlation ρ", zh: "相关系数 ρ" }, min: -0.95, max: 0.95, default: 0, step: 0.05 },
} as const;

type P = { muA: number; muB: number; sdA: number; sdB: number; rho: number };

const XD: [number, number] = [-5, 5];
const XS = grid(XD[0], XD[1], 401);

function compute(p: P) {
  const c = p.rho * p.sdA * p.sdB;
  const clark = expectedMax2(p.muA, p.sdA ** 2, p.muB, p.sdB ** 2, c);
  const s = Math.sqrt(Math.max(0, p.sdA ** 2 + p.sdB ** 2 - 2 * c));
  const q = Math.sqrt(1 - p.rho * p.rho);
  const pa = XS.map((x) => normalPdf(x, p.muA, p.sdA));
  const pb = XS.map((x) => normalPdf(x, p.muB, p.sdB));
  // P(B <= m | A = m): B given A = m is Gaussian with mean
  // mu_B + rho (sd_B / sd_A)(m - mu_A) and standard deviation sd_B sqrt(1 - rho^2).
  const pm = XS.map((m, i) =>
    pa[i] * Phi((m - p.muB - p.rho * (p.sdB / p.sdA) * (m - p.muA)) / (p.sdB * q))
    + pb[i] * Phi((m - p.muA - p.rho * (p.sdA / p.sdB) * (m - p.muB)) / (p.sdA * q)));
  // Integrate m p(m) over a wider range than is plotted, so the check is fair.
  const W = grid(-14, 14, 5601), h = W[1] - W[0];
  let num = 0;
  for (const m of W) {
    const d = normalPdf(m, p.muA, p.sdA) * Phi((m - p.muB - p.rho * (p.sdB / p.sdA) * (m - p.muA)) / (p.sdB * q))
      + normalPdf(m, p.muB, p.sdB) * Phi((m - p.muA - p.rho * (p.sdA / p.sdB) * (m - p.muB)) / (p.sdA * q));
    num += m * d * h;
  }
  return { clark, s, pa, pb, pm, num, better: Math.max(p.muA, p.muB) };
}

function describe(st: State<P>): string {
  const p = st.p, c = compute(p);
  return tpl(labels[st.lang ?? "en"].describe, {
    ma: fixed(p.muA, 2), sa: fixed(p.sdA, 2), mb: fixed(p.muB, 2), sb: fixed(p.sdB, 2), r: fixed(p.rho, 2),
    e: fixed(c.clark, 3), d: fixed(c.clark - c.better, 3), s: fixed(c.s, 2),
  });
}

function render(st: State<P>): string {
  const p = st.p, L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const c = compute(p);
  const lg = legend(narrow ? 8 : 40, 14, st.w - 16, [
    { kind: "line", color: C.c7, label: L.a },
    { kind: "line", color: C.c6, label: L.b },
    { kind: "band", color: C.band, label: L.max },
  ]);
  const top = 14 + lg.height + 16;
  const H0 = narrow ? 190 : 230;
  const ymax = Math.max(...c.pa, ...c.pb, ...c.pm) * 1.08;
  const f = frame({ w: st.w, top, height: H0, yDomain: [0, ymax], xDomain: XD, yTitle: L.y });
  const cid = `${st.uid}-clip`;
  const xe = f.x(c.clark), xb = f.x(c.better);
  const lines = narrow
    ? [tpl(L.r1, { e: fixed(c.clark, 3), n: fixed(c.num, 3) }), tpl(L.r2, { m: fixed(c.better, 2), d: fixed(c.clark - c.better, 3), s: fixed(c.s, 2) })]
    : [tpl(L.readout, { e: fixed(c.clark, 3), n: fixed(c.num, 3), m: fixed(c.better, 2), d: fixed(c.clark - c.better, 3), s: fixed(c.s, 2) })];
  const readTop = f.bottom + 56;
  return svg(st.w, readTop + lines.length * 16 - 4, describe(st),
    el("defs", {}, clip(f, cid)),
    lg.svg,
    frameAxes(f, { yTitle: L.y, xTitle: L.x, yCount: 3 }),
    g({ "clip-path": `url(#${cid})` },
      band(f, XS, XS.map(() => 0), c.pm),
      curve(f, XS, c.pm, { "stroke-width": 2.2 }),
      curve(f, XS, c.pa, { stroke: C.c7, "stroke-width": 1.5 }),
      curve(f, XS, c.pb, { stroke: C.c6, "stroke-width": 1.5 }),
    ),
    el("line", { x1: xb, x2: xb, y1: f.top, y2: f.bottom, stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "4 3" }),
    el("line", { x1: xe, x2: xe, y1: f.top, y2: f.bottom, stroke: C.model, "stroke-width": 2 }),
    text(xb - 4, f.top - 5, L.better, { "text-anchor": "end", "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
    text(xe + 4, f.top - 5, L.mean, { "text-anchor": "start", "font-size": TYPE.small, class: "fig-t-halo" }),
    ...lines.map((s, i) => text(narrow ? 8 : f.left, readTop + i * 16, s, { "font-size": TYPE.small, class: "fig-t-num" })),
  );
}

export default defineFigure({
  name: "id-clark",
  title: { en: "The density and the expected value of the maximum of two Gaussians", zh: "两个高斯变量最大值的密度与期望" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "Equal means", zh: "均值相等" }, run: (p) => ({ ...p, muA: 0, muB: 0 }) },
    { label: { en: "One certain option", zh: "一个确定的选项" }, run: (p) => ({ ...p, sdB: 0.1, rho: 0 }) },
  ],
});
