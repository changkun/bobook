// One lengthscale per input, in two dimensions. A function of two inputs is
// drawn from a Gaussian process prior with the RBF kernel
//   k(x, x') = exp(-(x1 - x1')^2 / (2 l1^2) - (x2 - x2')^2 / (2 l2^2)),
// shown as a map over the unit square, with two cuts through its center: one
// along each input. When one lengthscale grows, the map turns into stripes
// and the cut along that input goes flat: the function has stopped depending
// on it, which is what "irrelevant" means for automatic relevance
// determination.
//
// The RBF kernel with one lengthscale per input is a product of two
// one-dimensional kernels, so on a grid its covariance matrix is a Kronecker
// product K1 ⊗ K2 and a draw is L1 Z L2^T with Z a matrix of standard
// normals. That shortcut is exact for this kernel only; a Matérn kernel of
// the scaled distance does not factor this way.

import { defineFigure, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { cholesky } from "./lib/linalg.ts";
import { memo, normals, rng } from "./lib/random.ts";
import { fixed, sig, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x1: "input x₁",
    x2: "input x₂",
    map: "f(x₁, x₂): blue above zero, red below",
    cut1: "cut along x₁ (at x₂ = 0.5)",
    cut2: "cut along x₂ (at x₁ = 0.5)",
    readout: "relevance 1/ℓ: {a} for x₁, {b} for x₂ · the cut along x₁ spans {s1}, the cut along x₂ spans {s2}",
    readoutShort1: "relevance 1/ℓ: x₁ {a}, x₂ {b}",
    readoutShort2: "span of the cuts: x₁ {s1}, x₂ {s2}",
    describe: "A function of two inputs drawn from a Gaussian process prior with lengthscale {l1} along the first input and {l2} along the second. Through the center, it varies over a range of {s1} along the first input and {s2} along the second.",
  },
  zh: {
    x1: "输入 x₁",
    x2: "输入 x₂",
    map: "f(x₁, x₂)：蓝色高于零，红色低于零",
    cut1: "沿 x₁ 的切片（x₂ = 0.5 处）",
    cut2: "沿 x₂ 的切片（x₁ = 0.5 处）",
    readout: "相关性 1/ℓ：x₁ 为 {a}，x₂ 为 {b} · 沿 x₁ 的切片跨度 {s1}，沿 x₂ 的切片跨度 {s2}",
    readoutShort1: "相关性 1/ℓ：x₁ {a}，x₂ {b}",
    readoutShort2: "切片跨度：x₁ {s1}，x₂ {s2}",
    describe: "从高斯过程先验中抽取的一个双输入函数，沿第一个输入的长度尺度为 {l1}，沿第二个输入的长度尺度为 {l2}。穿过中心时，它沿第一个输入的变化范围为 {s1}，沿第二个输入的变化范围为 {s2}。",
  },
};

const params = {
  l1: { kind: "range", label: { en: "Lengthscale ℓ₁", zh: "长度尺度 ℓ₁" }, min: 0.05, max: 5, default: 0.15, scale: "log" },
  l2: { kind: "range", label: { en: "Lengthscale ℓ₂", zh: "长度尺度 ℓ₂" }, min: 0.05, max: 5, default: 1.5, scale: "log" },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 3, step: 1, control: false },
} as const;

type P = { l1: number; l2: number; seed: number };

const N = 41;
const XS = grid(0, 1, N);
const MID = (N - 1) / 2;

const factor = memo((ell: number) => cholesky(XS.map((a) => XS.map((b) => Math.exp(-((a - b) ** 2) / (2 * ell * ell)))), 1e-8), 64);

// F[i][j] = f(x1 = XS[i], x2 = XS[j]) = (L1 Z L2^T)[i][j].
const draw = memo((l1: number, l2: number, seed: number) => {
  const L1 = factor(l1), L2 = factor(l2);
  const r = rng(seed);
  const Z = Array.from({ length: N }, () => normals(r, N));
  const A = L1.map((row, i) => XS.map((_, q) => { let s = 0; for (let k = 0; k <= i; k++) s += row[k] * Z[k][q]; return s; }));
  return A.map((row) => XS.map((_, j) => { let s = 0; for (let q = 0; q <= j; q++) s += row[q] * L2[j][q]; return s; }));
}, 16);

const span = (v: number[]) => Math.max(...v) - Math.min(...v);

function cuts(F: number[][]) {
  return { c1: F.map((row) => row[MID]), c2: F[MID].slice() };
}

function describe(st: State<P>): string {
  const { c1, c2 } = cuts(draw(st.p.l1, st.p.l2, st.p.seed));
  return tpl(labels[st.lang ?? "en"].describe, { l1: sig(st.p.l1, 2), l2: sig(st.p.l2, 2), s1: fixed(span(c1), 2), s2: fixed(span(c2), 2) });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 560;
  const F = draw(p.l1, p.l2, p.seed);
  const { c1, c2 } = cuts(F);
  const parts: string[] = [];
  const mx0 = 46, my0 = 28;
  const cell = Math.max(3, Math.floor((narrow ? Math.min(st.w - mx0 - 12, 250) : 250) / N));
  const size = cell * N;
  const cells: string[] = [];
  const op = (v: number) => Math.min(0.95, Math.abs(v) / 2.4).toFixed(3);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const v = F[i][j];
    cells.push(el("rect", { x: mx0 + i * cell, y: my0 + size - (j + 1) * cell, width: cell, height: cell, fill: v >= 0 ? C.c1 : C.c8, opacity: op(v) }));
  }
  parts.push(g({ "shape-rendering": "crispEdges" }, ...cells));
  const ax = linear([0, 1], [mx0, mx0 + size]), ay = linear([0, 1], [my0 + size, my0]);
  parts.push(
    el("rect", { x: mx0, y: my0, width: size, height: size, fill: "none", stroke: C.rule }),
    // the two cuts
    el("line", { x1: mx0, x2: mx0 + size, y1: ay(0.5), y2: ay(0.5), stroke: C.c7, "stroke-width": 1.6, "stroke-dasharray": "5 3" }),
    el("line", { x1: ax(0.5), x2: ax(0.5), y1: my0, y2: my0 + size, stroke: C.c6, "stroke-width": 1.6, "stroke-dasharray": "5 3" }),
    axis({ scale: ax, orient: "bottom", at: my0 + size, title: L.x1, count: 2, grid: false }),
    axis({ scale: ay, orient: "left", at: mx0, title: L.x2, count: 2, grid: false }),
    text(mx0, my0 - 9, L.map, { "font-size": TYPE.small, class: "fig-t-muted" }),
  );
  // cut panels: to the right when wide, below when narrow
  const cx0 = narrow ? mx0 : mx0 + size + 62;
  const cw = (narrow ? st.w - 12 : st.w - 10) - cx0;
  const ch = narrow ? 84 : (size - 44) / 2;
  const top1 = narrow ? my0 + size + 62 : my0;
  const top2 = top1 + ch + 44;
  const lim = Math.max(2.6, Math.ceil(Math.max(...c1.map(Math.abs), ...c2.map(Math.abs)) * 2) / 2);
  const panel = (top: number, vals: number[], color: string, title: string, xt: string) => {
    const sx = linear([0, 1], [cx0, cx0 + cw]), sy = linear([-lim, lim], [top + ch, top]);
    return g({},
      text(cx0, top - 7, title, { "font-size": TYPE.small, class: "fig-t-muted" }),
      axis({ scale: sy, orient: "left", at: cx0, span: [cx0, cx0 + cw], count: 2 }),
      axis({ scale: sx, orient: "bottom", at: top + ch, count: 2, grid: false, title: narrow ? undefined : xt }),
      el("line", { x1: cx0, x2: cx0 + cw, y1: sy(0), y2: sy(0), stroke: C.rule }),
      el("path", { d: linePath(XS.map((x, i) => [sx(x), sy(vals[i])])), fill: "none", stroke: color, "stroke-width": 2 }),
    );
  };
  parts.push(panel(top1, c1, C.c7, L.cut1, ""), panel(top2, c2, C.c6, L.cut2, ""));
  const bottom = Math.max(my0 + size + 40, top2 + ch + 24);
  const vars = { a: sig(1 / p.l1, 2), b: sig(1 / p.l2, 2), s1: fixed(span(c1), 2), s2: fixed(span(c2), 2) };
  const lines = narrow ? [tpl(L.readoutShort1, vars), tpl(L.readoutShort2, vars)] : [tpl(L.readout, vars)];
  lines.forEach((s, i) => parts.push(text(narrow ? 8 : mx0, bottom + 14 + i * 16, s, { "font-size": TYPE.small, class: "fig-t-num" })));
  return svg(st.w, bottom + 14 + lines.length * 16 - 6, describe(st), g({}, ...parts));
}

export default defineFigure({
  name: "kern-ard",
  title: { en: "A function of two inputs drawn with one lengthscale per input", zh: "每个输入各有一个长度尺度时抽取的双输入函数" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "Draw again", zh: "重新抽取" }, primary: true, run: (p) => ({ ...p, seed: (p.seed % 9999) + 1 }) },
    { label: { en: "Swap the lengthscales", zh: "交换长度尺度" }, run: (p) => ({ ...p, l1: p.l2, l2: p.l1 }) },
  ],
});
