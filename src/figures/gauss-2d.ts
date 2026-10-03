// A two-dimensional Gaussian with mean zero, standard deviations s1 and s2,
// and correlation rho, in three views. "Shape" draws the density as the
// ellipses at Mahalanobis distance 1, 2, and 3, with the eigenvector axes and
// the two marginal densities in the margins. "Samples" draws standard normal
// points z and their images L z under the Cholesky factor L of the covariance,
// with a few of them joined by arrows. "Condition" slices the density at an
// observed value x1 = a, which the reader drags: the margin on the right
// compares the marginal density of x2 with its conditional density given the
// observation, and a dashed line traces the conditional mean for every a.

import { defineFigure, type State } from "./types.ts";
import { arrowMarker, el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { clamp, grid, linear, type Scale } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { eig2 } from "./lib/linalg.ts";
import { normals, rng } from "./lib/random.ts";
import { normalPdf } from "./lib/stats.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x1: "x_1",
    x2: "x_2",
    cov: "Covariance matrix",
    eig: "eigenvalues λ_1 = {l1}, λ_2 = {l2}",
    det: "det Σ = σ_1²σ_2²(1 − ρ²) = {d}",
    rings: "Ellipses at Mahalanobis distance",
    rings2: "1, 2, 3 hold 39%, 86%, 99%",
    axes: "Dashed: eigenvector axes",
    chol: "Cholesky factor, L Lᵀ = Σ",
    map: "x = L z,  z ~ N(0, I)",
    grey: "grey: {n} draws of z",
    blue: "blue: the same draws as L z",
    corr: "sample correlation {r} (ρ = {rho})",
    obs: "Observed x_1 = {a}",
    cond: "x_2 | x_1 ~ N(m, s²)",
    m: "m = ρ (σ_2/σ_1) x_1 = {m}",
    s: "s = σ_2 √(1 − ρ²) = {s}",
    before: "before observing: sd σ_2 = {s2}",
    removed: "variance removed: ρ² = {pct}",
    legCond: "conditional of x_2",
    legMarg: "marginal of x_2",
    legLine: "conditional mean E[x_2 | x_1]",
    describeShape: "A two-dimensional Gaussian with standard deviations {s1} and {s2} and correlation {rho}. Its density contours are ellipses; the eigenvalues of the covariance are {l1} and {l2}.",
    describeSamples: "{n} standard normal draws z and their images L z under the Cholesky factor. The mapped points have sample correlation {r}, against a target of {rho}.",
    describeCond: "Conditioning on x1 = {a}: x2 given x1 is Gaussian with mean {m} and standard deviation {s}, against a marginal standard deviation of {s2}.",
  },
  zh: {
    x1: "x_1",
    x2: "x_2",
    cov: "协方差矩阵",
    eig: "特征值 λ_1 = {l1}，λ_2 = {l2}",
    det: "det Σ = σ_1²σ_2²(1 − ρ²) = {d}",
    rings: "Mahalanobis 距离 1、2、3 处的椭圆",
    rings2: "分别包含 39%、86%、99% 的概率",
    axes: "虚线：特征向量轴",
    chol: "Cholesky 因子，L Lᵀ = Σ",
    map: "x = L z,  z ~ N(0, I)",
    grey: "灰点：z 的 {n} 个样本",
    blue: "蓝点：同一批样本映射为 L z",
    corr: "样本相关系数 {r}（ρ = {rho}）",
    obs: "观测值 x_1 = {a}",
    cond: "x_2 | x_1 ~ N(m, s²)",
    m: "m = ρ (σ_2/σ_1) x_1 = {m}",
    s: "s = σ_2 √(1 − ρ²) = {s}",
    before: "观测之前：标准差 σ_2 = {s2}",
    removed: "消除的方差比例：ρ² = {pct}",
    legCond: "x_2 的条件分布",
    legMarg: "x_2 的边际分布",
    legLine: "条件均值 E[x_2 | x_1]",
    describeShape: "二维高斯分布，标准差为 {s1} 和 {s2}，相关系数为 {rho}。其密度等高线是椭圆；协方差矩阵的特征值为 {l1} 和 {l2}。",
    describeSamples: "{n} 个标准正态样本 z 及其在 Cholesky 因子映射下的像 L z。映射后的点的样本相关系数为 {r}，目标值为 {rho}。",
    describeCond: "以 x1 = {a} 为条件：给定 x1 时，x2 服从均值为 {m}、标准差为 {s} 的高斯分布，而其边际标准差为 {s2}。",
  },
};

const VIEWS = [
  { value: "shape", label: { en: "Shape", zh: "形状" } },
  { value: "samples", label: { en: "Samples", zh: "样本" } },
  { value: "condition", label: { en: "Condition", zh: "条件化" } },
] as const;

const params = {
  view: { kind: "choice", label: { en: "View", zh: "视图" }, options: VIEWS, default: "shape", control: "buttons" },
  s1: { kind: "range", label: { en: "sd of x₁, σ₁", zh: "x₁ 的标准差 σ₁" }, min: 0.4, max: 1.8, default: 1, step: 0.05 },
  s2: { kind: "range", label: { en: "sd of x₂, σ₂", zh: "x₂ 的标准差 σ₂" }, min: 0.4, max: 1.8, default: 1, step: 0.05 },
  rho: { kind: "range", label: { en: "Correlation ρ", zh: "相关系数 ρ" }, min: -0.95, max: 0.95, default: 0.7, step: 0.05 },
  a: { kind: "range", label: { en: "Observed x₁", zh: "观测值 x₁" }, min: -3, max: 3, default: 1.5, step: 0.05, control: false },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 5, step: 1, control: false },
} as const;

type P = { view: "shape" | "samples" | "condition"; s1: number; s2: number; rho: number; a: number; seed: number };

const D = 3.6; // plotted range [-D, D] on both axes
const MASS = [1, 2, 3];

function model(p: P) {
  const { s1, s2, rho } = p;
  const c = rho * s1 * s2;
  const L = [[s1, 0], [rho * s2, s2 * Math.sqrt(1 - rho * rho)]];
  const e = eig2(s1 * s1, c, s2 * s2);
  const m = rho * (s2 / s1) * p.a;
  const s = s2 * Math.sqrt(1 - rho * rho);
  return { c, L, e, m, s, det: s1 * s1 * s2 * s2 * (1 - rho * rho) };
}

function draws(p: P, n: number) {
  const z = normals(rng(p.seed * 7 + 1), 2 * n);
  const { L } = model(p);
  const zs: Array<[number, number]> = [], xs: Array<[number, number]> = [];
  for (let i = 0; i < n; i++) {
    const z1 = z[2 * i], z2 = z[2 * i + 1];
    zs.push([z1, z2]);
    xs.push([L[0][0] * z1, L[1][0] * z1 + L[1][1] * z2]);
  }
  return { zs, xs };
}

function sampleCorr(pts: Array<[number, number]>): number {
  const n = pts.length;
  const mx = pts.reduce((s, q) => s + q[0], 0) / n, my = pts.reduce((s, q) => s + q[1], 0) / n;
  let sxx = 0, syy = 0, sxy = 0;
  for (const [x, y] of pts) { sxx += (x - mx) ** 2; syy += (y - my) ** 2; sxy += (x - mx) * (y - my); }
  return sxy / Math.sqrt(sxx * syy);
}

function nDraws(w: number) { return w < 480 ? 160 : 250; }

function describe(st: State<P>): string {
  const p = st.p, L = labels[st.lang ?? "en"], md = model(p);
  if (p.view === "samples") {
    const n = nDraws(st.w);
    return tpl(L.describeSamples, { n, r: fixed(sampleCorr(draws(p, n).xs), 2), rho: fixed(p.rho, 2) });
  }
  if (p.view === "condition") return tpl(L.describeCond, { a: fixed(p.a, 2), m: fixed(md.m, 2), s: fixed(md.s, 2), s2: fixed(p.s2, 2) });
  return tpl(L.describeShape, { s1: fixed(p.s1, 2), s2: fixed(p.s2, 2), rho: fixed(p.rho, 2), l1: fixed(md.e.values[0], 2), l2: fixed(md.e.values[1], 2) });
}

// A 2 x 2 matrix with bracket glyphs, entries right-aligned in two columns.
function matrix(x: number, y: number, name: string, m: number[][], colW = 52): { svg: string; h: number } {
  const rowH = 17, h = rowH * 2 + 4;
  const bx = x + 26;
  const br = (xx: number, dir: 1 | -1) => el("path", { d: `M${xx + 5 * dir},${y} L${xx},${y} L${xx},${y + h} L${xx + 5 * dir},${y + h}`, fill: "none", stroke: C.ink2, "stroke-width": 1.2 });
  const cells: string[] = [];
  m.forEach((row, i) => row.forEach((v, j) => cells.push(text(bx + 6 + colW * (j + 1) - 8, y + rowH * (i + 1) - 2, fixed(v, 2), { "text-anchor": "end", "font-size": TYPE.body, class: "fig-t-num" }))));
  return {
    svg: g({}, text(x, y + h / 2 + 4, `${name} =`, { "font-size": TYPE.body, class: "fig-t-math" }), br(bx, 1), ...cells, br(bx + 6 + colW * 2 + 2, -1)),
    h: h + 8,
  };
}

function ellipse(px: Scale, py: Scale, L: number[][], r: number): string {
  const pts: Array<[number, number]> = [];
  for (let i = 0; i <= 72; i++) {
    const t = (2 * Math.PI * i) / 72, z1 = r * Math.cos(t), z2 = r * Math.sin(t);
    pts.push([px(L[0][0] * z1), py(L[1][0] * z1 + L[1][1] * z2)]);
  }
  return linePath(pts) + "Z";
}

function render(st: State<P>): string {
  const p = st.p, L = labels[st.lang ?? "en"], w = st.w;
  const narrow = w < 480;
  const md = model(p);
  const left = narrow ? 40 : 46;
  const stripW = narrow ? 46 : 58, stripH = narrow ? 36 : 44, gap = 6;
  const S = narrow ? Math.max(180, w - left - gap - stripW - 8) : Math.min(330, w - left - gap - stripW - 230);
  const top = 8 + stripH + gap;
  const px = linear([-D, D], [left, left + S]);
  const py = linear([-D, D], [top + S, top]);
  const parts: string[] = [];
  const uid = st.uid;
  parts.push(el("defs", {},
    el("clipPath", { id: `${uid}-c` }, el("rect", { x: left, y: top, width: S, height: S })),
    arrowMarker(`${uid}-ar`, C.acq, 5),
  ));
  parts.push(el("rect", { x: left, y: top, width: S, height: S, fill: C.panel, stroke: "none" }));
  parts.push(axis({ scale: px, orient: "bottom", at: top + S, span: [top, top + S], title: L.x1, count: 4 }));
  parts.push(axis({ scale: py, orient: "left", at: left, span: [left, left + S], title: L.x2, count: 4 }));

  const inner: string[] = [];
  // density ellipses
  const rings = p.view === "samples" ? [2] : p.view === "condition" ? [1, 2] : MASS;
  for (const r of rings.slice().reverse()) {
    inner.push(el("path", {
      d: ellipse(px, py, md.L, r),
      fill: p.view === "samples" ? "none" : C.band, "fill-opacity": p.view === "samples" ? undefined : 0.55,
      stroke: C.model, "stroke-width": p.view === "samples" ? 1.2 : 1.4, "stroke-dasharray": p.view === "samples" ? "4 3" : undefined,
    }));
  }
  if (p.view === "shape") {
    // eigenvector axes out to the r = 2 ellipse
    md.e.vectors.forEach((v, i) => {
      const len = 2 * Math.sqrt(Math.max(0, md.e.values[i]));
      inner.push(el("line", { x1: px(-v[0] * len), y1: py(-v[1] * len), x2: px(v[0] * len), y2: py(v[1] * len), stroke: C.ink2, "stroke-width": 1.2, "stroke-dasharray": "5 4" }));
      const ex = v[0] * (len + 0.35), ey = v[1] * (len + 0.35);
      inner.push(text(px(ex), py(ey) + 4, `u_${i + 1}`, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));
    });
  }
  if (p.view === "samples") {
    const n = nDraws(w);
    const { zs, xs } = draws(p, n);
    for (const [a, b] of zs) inner.push(el("circle", { cx: px(a), cy: py(b), r: 1.7, fill: C.ink3, opacity: 0.55 }));
    for (const [a, b] of xs) inner.push(el("circle", { cx: px(a), cy: py(b), r: 2.1, fill: C.model, opacity: 0.8 }));
    // a few pairs joined by arrows, one per direction around the cloud, so
    // the reader sees where z goes on every side
    const picks: number[] = [];
    for (let sector = 0; sector < 6; sector++) {
      const lo = (sector / 6) * 2 * Math.PI - Math.PI;
      const hi = lo + Math.PI / 3;
      for (let i = 0; i < zs.length; i++) {
        const ang = Math.atan2(zs[i][1], zs[i][0]);
        const rz = Math.hypot(zs[i][0], zs[i][1]);
        if (ang >= lo && ang < hi && rz > 1.2 && rz < 2.2) { picks.push(i); break; }
      }
    }
    for (const i of picks) {
      inner.push(
        el("line", { x1: px(zs[i][0]), y1: py(zs[i][1]), x2: px(xs[i][0]), y2: py(xs[i][1]), stroke: C.acq, "stroke-width": 1.4, "marker-end": `url(#${uid}-ar)` }),
        el("circle", { cx: px(zs[i][0]), cy: py(zs[i][1]), r: 3.4, fill: C.paper, stroke: C.acq, "stroke-width": 1.4 }),
        el("circle", { cx: px(xs[i][0]), cy: py(xs[i][1]), r: 3.4, fill: C.acq, stroke: C.paper, "stroke-width": 1 }),
      );
    }
  }
  if (p.view === "condition") {
    // the line of conditional means, x2 = rho (s2/s1) x1
    const k = p.rho * (p.s2 / p.s1);
    inner.push(el("line", { x1: px(-D), y1: py(-D * k), x2: px(D), y2: py(D * k), stroke: C.c5, "stroke-width": 1.5, "stroke-dasharray": "6 4" }));
    // the slice
    const xa = px(p.a);
    inner.push(el("line", { x1: xa, x2: xa, y1: top, y2: top + S, stroke: C.acq, "stroke-width": 1.6 }));
    const lo = md.m - 1.96 * md.s, hi = md.m + 1.96 * md.s;
    inner.push(
      el("line", { x1: xa, x2: xa, y1: py(lo), y2: py(hi), stroke: C.acq, "stroke-width": 5, "stroke-linecap": "round", opacity: 0.55 }),
      el("circle", { cx: xa, cy: py(md.m), r: 4.5, fill: C.acq, stroke: C.paper, "stroke-width": 1.5 }),
    );
  }
  parts.push(g({ "clip-path": `url(#${uid}-c)` }, ...inner));
  if (p.view === "condition") {
    // drag handles on the slice, outside the clip so they stay visible
    const xa = px(p.a);
    parts.push(el("path", { d: `M${xa - 6},${top - 1} L${xa + 6},${top - 1} L${xa},${top + 7}Z`, fill: C.acq }));
  }

  // marginal strips: x1 on top, x2 on the right
  const xsG = grid(-D, D, 121);
  const topBase = top - gap + 2;
  const m1 = xsG.map((v) => normalPdf(v, 0, p.s1));
  const scaleTop = (stripH - 6) / Math.max(...m1);
  parts.push(el("line", { x1: left, x2: left + S, y1: topBase, y2: topBase, stroke: C.rule }));
  parts.push(el("path", { d: linePath(xsG.map((v, i) => [px(v), topBase - m1[i] * scaleTop])) + `L${px(D)},${topBase}L${px(-D)},${topBase}Z`, fill: C.band, stroke: C.model, "stroke-width": 1.4 }));
  if (p.view === "condition") parts.push(el("line", { x1: px(p.a), x2: px(p.a), y1: topBase - stripH + 4, y2: topBase, stroke: C.acq, "stroke-width": 1.6 }));
  const rightBase = left + S + gap - 2;
  const m2 = xsG.map((v) => normalPdf(v, 0, p.s2));
  const cd = xsG.map((v) => normalPdf(v, md.m, md.s));
  const peakR = p.view === "condition" ? Math.max(...m2, ...cd) : Math.max(...m2);
  const scaleR = (stripW - 6) / peakR;
  parts.push(el("line", { x1: rightBase, x2: rightBase, y1: top, y2: top + S, stroke: C.rule }));
  parts.push(el("path", { d: linePath(xsG.map((v, i) => [rightBase + m2[i] * scaleR, py(v)])) + `L${rightBase},${py(D)}L${rightBase},${py(-D)}Z`, fill: C.band, stroke: C.model, "stroke-width": 1.4 }));
  if (p.view === "condition") {
    parts.push(el("path", { d: linePath(xsG.map((v, i) => [rightBase + cd[i] * scaleR, py(v)])) + `L${rightBase},${py(D)}L${rightBase},${py(-D)}Z`, fill: C.acqFill, stroke: C.acq, "stroke-width": 1.6 }));
  }

  // readout panel: to the right on wide layouts, below on narrow ones
  const panelX = narrow ? left - 30 : rightBase + stripW + 16;
  let y = narrow ? top + S + 44 : top + 4;
  const lineH = 18;
  const line = (s: string, cls = "fig-t-muted", size: number = TYPE.body) => { parts.push(text(panelX, y + 12, s, { "font-size": size, class: cls })); y += lineH; };
  const swatch = (kind: "fill" | "dash", color: string, fill: string, s: string) => {
    parts.push(kind === "fill"
      ? el("rect", { x: panelX, y: y + 3, width: 14, height: 10, rx: 2, fill, stroke: color, "stroke-width": 1.2 })
      : el("line", { x1: panelX, x2: panelX + 14, y1: y + 8, y2: y + 8, stroke: color, "stroke-width": 1.5, "stroke-dasharray": "5 3" }));
    parts.push(text(panelX + 20, y + 12, s, { "font-size": TYPE.small, class: "fig-t-muted" }));
    y += lineH - 2;
  };
  if (p.view === "shape") {
    line(L.cov, "fig-t-strong");
    const mx = matrix(panelX, y, "Σ", [[p.s1 * p.s1, md.c], [md.c, p.s2 * p.s2]]);
    parts.push(mx.svg); y += mx.h;
    line(tpl(L.eig, { l1: fixed(md.e.values[0], 2), l2: fixed(md.e.values[1], 2) }));
    line(tpl(L.det, { d: fixed(md.det, 2) }));
    y += 4;
    line(L.rings, "fig-t-muted", TYPE.small);
    line(L.rings2, "fig-t-muted", TYPE.small);
    line(L.axes, "fig-t-muted", TYPE.small);
  } else if (p.view === "samples") {
    const n = nDraws(w);
    line(L.chol, "fig-t-strong");
    const mx = matrix(panelX, y, "L", md.L);
    parts.push(mx.svg); y += mx.h;
    line(L.map);
    y += 4;
    line(tpl(L.grey, { n }), "fig-t-muted", TYPE.small);
    line(L.blue, "fig-t-muted", TYPE.small);
    line(tpl(L.corr, { r: fixed(sampleCorr(draws(p, n).xs), 2), rho: fixed(p.rho, 2) }), "fig-t-muted", TYPE.small);
  } else {
    line(tpl(L.obs, { a: fixed(p.a, 2) }), "fig-t-strong");
    line(L.cond);
    line(tpl(L.m, { m: fixed(md.m, 2) }));
    line(tpl(L.s, { s: fixed(md.s, 2) }));
    line(tpl(L.before, { s2: fixed(p.s2, 2) }), "fig-t-muted", TYPE.small);
    line(tpl(L.removed, { pct: `${Math.round(p.rho * p.rho * 100)}%` }), "fig-t-muted", TYPE.small);
    y += 4;
    swatch("fill", C.acq, C.acqFill, L.legCond);
    swatch("fill", C.model, C.band, L.legMarg);
    swatch("dash", C.c5, "none", L.legLine);
  }
  const plotBottom = top + S + 40;
  const H = Math.max(plotBottom, y + 6);
  // pointer target last, so it receives drags anywhere over the square
  if (p.view === "condition") {
    const map = [px.range[0], px.range[1], px.domain[0], px.domain[1], py.range[0], py.range[1], py.domain[0], py.domain[1]].join(",");
    parts.push(el("rect", { x: left, y: top - 8, width: S, height: S + 8, fill: "transparent", "data-fig-hit": "plot", "data-fig-map": map }));
  }
  return svg(w, H, describe(st), ...parts);
}

const step = (p: P, d: number): P => ({ ...p, a: Math.round(clamp(p.a + d, -3, 3) * 20) / 20 });

export default defineFigure({
  name: "gauss-2d",
  title: { en: "A two-dimensional Gaussian: its shape, its samples, and a slice through it", zh: "二维高斯分布：形状、样本与切片" },
  labels,
  params,
  hint: { en: "In the Condition view, click or drag across the plot to move the observed value x₁.", zh: "在“条件化”视图中，点击或在图上拖动，以移动观测值 x₁。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if ((p as P).view !== "condition" || e.target !== "plot" || !e.data) return null;
    if (e.phase === "up") return null;
    return { ...p, a: Math.round(clamp(e.data.x, -3, 3) * 20) / 20 };
  },
  actions: [
    { label: { en: "◀ Move x₁", zh: "◀ 移动 x₁" }, run: (p) => step(p as P, -0.25), enabled: (p) => p.view === "condition" },
    { label: { en: "Move x₁ ▶", zh: "移动 x₁ ▶" }, run: (p) => step(p as P, 0.25), enabled: (p) => p.view === "condition" },
    { label: { en: "New samples", zh: "新样本" }, run: (p) => ({ ...p, seed: (p.seed % 999) + 1 }), enabled: (p) => p.view === "samples" },
  ],
});
