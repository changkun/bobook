// Two latent utilities, one kind of answer. The prior over (g(A), g(B)) is a
// correlated Gaussian, the correlation coming from the kernel (high when A and
// B are close). Comparisons of A with B multiply in probit factors that depend
// only on the difference g(A) - g(B). The exact posterior is therefore skewed
// along the difference and exactly Gaussian along the sum: the skew lives in
// the directions that comparisons touch. The plot shows the exact posterior
// (shading and its 50% and 90% highest-density contours), the Laplace and EP
// ellipses at the same levels, and, along the top, the marginal of g(A), which
// is much less skewed than the difference.

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { normalPdf } from "./lib/stats.ts";
import { memo } from "./lib/random.ts";
import { ep, exactGrid, laplace, type Gauss1, type Problem } from "./lib/approx-probit.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    gA: "g(A)",
    gB: "g(B)",
    tie: "g(A) = g(B)",
    exact: "exact posterior (50%, 90%)",
    prior: "prior (90%)",
    laplace: "Laplace",
    ep: "EP",
    marg: "marginal of g(A)",
    skew: "skewness: difference {sd} · g(A) alone {sa}",
    describe: "Prior correlation {r}, noise σ = {s}, A won {w} of {n}. The posterior is skewed along the difference (skewness {sd}) but the marginal of g(A) alone has skewness {sa}.",
  },
  zh: {
    gA: "g(A)",
    gB: "g(B)",
    tie: "g(A) = g(B)",
    exact: "精确后验（50%，90%）",
    prior: "先验（90%）",
    laplace: "Laplace 近似",
    ep: "期望传播",
    marg: "g(A) 的边际分布",
    skew: "偏度：效用差 {sd} · 单看 g(A) {sa}",
    describe: "先验相关系数 {r}，噪声 σ = {s}，{n} 次中 A 赢 {w} 次。后验沿效用差的方向偏斜（偏度 {sd}），而单看 g(A) 的边际分布，偏度为 {sa}。",
  },
};

const params = {
  rho: { kind: "range", label: { en: "Prior correlation ρ", zh: "先验相关系数 ρ" }, min: 0, max: 0.95, default: 0.5, step: 0.05 },
  sigma: { kind: "range", label: { en: "Noise σ", zh: "噪声 σ" }, min: 0.01, max: 2, default: 0.05, scale: "log" },
  wins: { kind: "range", label: { en: "A won", zh: "A 赢的次数" }, min: 1, max: 6, default: 1, step: 1 },
  laplace: { kind: "toggle", label: { en: "Laplace", zh: "Laplace 近似" }, default: true },
  ep: { kind: "toggle", label: { en: "EP", zh: "期望传播" }, default: false },
} as const;

type P = { rho: number; sigma: number; wins: number; laplace: boolean; ep: boolean };

const LIM = 3;
const N = 96;
const GX = grid(-LIM, LIM, N);

// Highest-density levels: the density value above which a given mass lies.
function hpdLevels(vals: number[][], cellArea: number, masses: number[]): number[] {
  const flat = vals.flat().sort((a, b) => b - a);
  const out: number[] = [];
  let acc = 0, k = 0;
  for (const m of masses) {
    while (k < flat.length && acc < m) { acc += flat[k] * cellArea; k++; }
    out.push(flat[Math.max(0, k - 1)]);
  }
  return out;
}

// Marching squares: line segments of the level set {v = c} on a grid whose
// values are vals[i][j] at (xs[i], ys[j]).
function contour(vals: number[][], xs: number[], ys: number[], c: number): Array<[number, number, number, number]> {
  const segs: Array<[number, number, number, number]> = [];
  const lerp = (a: number, b: number, va: number, vb: number) => a + ((c - va) / (vb - va)) * (b - a);
  for (let i = 0; i < xs.length - 1; i++) for (let j = 0; j < ys.length - 1; j++) {
    const v00 = vals[i][j], v10 = vals[i + 1][j], v11 = vals[i + 1][j + 1], v01 = vals[i][j + 1];
    const pts: Array<[number, number]> = [];
    if ((v00 > c) !== (v10 > c)) pts.push([lerp(xs[i], xs[i + 1], v00, v10), ys[j]]);
    if ((v10 > c) !== (v11 > c)) pts.push([xs[i + 1], lerp(ys[j], ys[j + 1], v10, v11)]);
    if ((v11 > c) !== (v01 > c)) pts.push([lerp(xs[i + 1], xs[i], v11, v01), ys[j + 1]]);
    if ((v01 > c) !== (v00 > c)) pts.push([xs[i], lerp(ys[j + 1], ys[j], v01, v00)]);
    if (pts.length >= 2) segs.push([pts[0][0], pts[0][1], pts[1][0], pts[1][1]]);
    if (pts.length === 4) segs.push([pts[2][0], pts[2][1], pts[3][0], pts[3][1]]);
  }
  return segs;
}

const solve = memo((rho: number, sigma: number, wins: number) => {
  const vd = 2 - 2 * rho, vu = 2 + 2 * rho;
  const pr: Problem = { v0: vd, wins, losses: 0, s: Math.SQRT2 * sigma };
  const ex = exactGrid(pr, -8, 8, 4001);
  const la = laplace(pr);
  const e = ep(pr);
  const dens1 = (d: number) => {
    const h = ex.xs[1] - ex.xs[0];
    const k = Math.floor((d - ex.xs[0]) / h);
    if (k < 0 || k >= ex.xs.length - 1) return 0;
    const t = (d - ex.xs[k]) / h;
    return ex.dens[k] * (1 - t) + ex.dens[k + 1] * t;
  };
  // joint density of (gA, gB): p_d(gA - gB) N(gA + gB; 0, vu) times the
  // Jacobian 2 of (gA, gB) -> (d, u)
  const vals = GX.map((a) => GX.map((b) => 2 * dens1(a - b) * normalPdf(a + b, 0, Math.sqrt(vu))));
  const cell = (GX[1] - GX[0]) ** 2;
  const levels = hpdLevels(vals, cell, [0.5, 0.9]);
  // skewness of d and of the marginal of gA = (u + d) / 2
  let m3 = 0;
  for (let k = 0; k < ex.xs.length; k++) m3 += ex.dens[k] * (ex.xs[k] - ex.mean) ** 3 * (ex.xs[1] - ex.xs[0]);
  const skD = m3 / ex.sd ** 3;
  const sdA = Math.sqrt((ex.sd ** 2 + vu) / 4);
  const skA = (m3 / 8) / sdA ** 3;
  // marginal of gA on a 1-D grid: convolve the d density with the u Gaussian
  const MX = grid(-LIM, LIM, 181);
  const margA = MX.map((a) => {
    let sum = 0;
    const h = 0.02;
    for (let d = -8; d <= 8; d += h) sum += dens1(d) * normalPdf(2 * a - d, 0, Math.sqrt(vu)) * 2 * h;
    return sum;
  });
  return { vd, vu, ex, la, ep: e, vals, levels, skD, skA, MX, margA };
}, 24);

function ellipse(gs: Gauss1, sdU: number, r: number, sx: (v: number) => number, sy: (v: number) => number): string {
  let d = "";
  for (let k = 0; k <= 72; k++) {
    const t = (2 * Math.PI * k) / 72;
    const dd = gs.mean + r * gs.sd * Math.cos(t), uu = r * sdU * Math.sin(t);
    const a = (uu + dd) / 2, b = (uu - dd) / 2;
    d += `${k ? "L" : "M"}${sx(a).toFixed(1)},${sy(b).toFixed(1)}`;
  }
  return d + "Z";
}

const R50 = Math.sqrt(-2 * Math.log(0.5)), R90 = Math.sqrt(-2 * Math.log(0.1));

function describe(st: State<P>): string {
  const p = st.p;
  const s = solve(p.rho, p.sigma, p.wins);
  return tpl(labels[st.lang ?? "en"].describe, { r: fixed(p.rho, 2), s: fixed(p.sigma, 2), w: p.wins, n: p.wins, sd: fixed(s.skD, 2), sa: fixed(s.skA, 2) });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const s = solve(p.rho, p.sigma, p.wins);
  const parts: string[] = [];
  // legend
  const items: Array<[string, string, string]> = [[L.exact, C.ink, "line"], [L.prior, C.ink3, "dash"]];
  if (p.laplace) items.push([L.laplace, C.model, "line"]);
  if (p.ep) items.push([L.ep, C.c6, "line"]);
  let lx = 8, ly = 14;
  for (const [name, color, kind] of items) {
    const tw = labelWidth(name, 6.4) + 32;
    if (lx + tw > st.w - 8) { lx = 8; ly += 17; }
    parts.push(el("line", { x1: lx, x2: lx + 18, y1: ly - 4, y2: ly - 4, stroke: color, "stroke-width": 2, "stroke-dasharray": kind === "dash" ? "4 3" : undefined }), text(lx + 23, ly, name, { "font-size": TYPE.small, class: "fig-t-muted" }));
    lx += tw;
  }
  const size = Math.min(narrow ? st.w - 58 : 330, st.w - 58);
  const x0 = narrow ? 46 : Math.max(52, (st.w - size) / 2 - 40);
  const margH = 54;
  const my0 = ly + 22;
  const y0 = my0 + margH + 14;
  const sx = linear([-LIM, LIM], [x0, x0 + size]);
  const sy = linear([-LIM, LIM], [y0 + size, y0]);

  // top strip: marginal of g(A)
  const mmax = Math.max(...s.margA) * 1.15;
  const my = (v: number) => my0 + margH - (v / mmax) * margH;
  const sdU = Math.sqrt(s.vu);
  const lineOf = (ys: number[], color: string, w = 1.8, dash?: string) => el("path", { d: s.MX.map((a, k) => `${k ? "L" : "M"}${sx(a).toFixed(1)},${my(ys[k]).toFixed(1)}`).join(""), fill: "none", stroke: color, "stroke-width": w, "stroke-dasharray": dash });
  parts.push(el("line", { x1: x0, x2: x0 + size, y1: my0 + margH, y2: my0 + margH, stroke: C.rule }));
  parts.push(el("path", { d: `M${sx(-LIM)},${my0 + margH}` + s.MX.map((a, k) => `L${sx(a).toFixed(1)},${my(s.margA[k]).toFixed(1)}`).join("") + `L${sx(LIM)},${my0 + margH}Z`, fill: C.ink, opacity: 0.08 }));
  const gm = (gs: Gauss1) => s.MX.map((a) => normalPdf(a, gs.mean / 2, Math.sqrt((gs.sd ** 2 + s.vu) / 4)));
  if (p.laplace) parts.push(lineOf(gm(s.la), C.model, 1.8));
  if (p.ep) parts.push(lineOf(gm(s.ep), C.c6, 1.8));
  parts.push(lineOf(s.margA, C.ink, 1.8), text(x0 + 2, my0 + 10, L.marg, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));

  // main panel
  const cw = size / N;
  const vmax = Math.max(...s.vals.flat());
  const cells: string[] = [];
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const v = s.vals[i][j] / vmax;
    if (v < 0.01) continue;
    cells.push(el("rect", { x: sx(GX[i]) - cw / 2, y: sy(GX[j]) - cw / 2, width: cw + 0.3, height: cw + 0.3, fill: C.ink, opacity: (0.22 * v ** 0.8).toFixed(3) }));
  }
  parts.push(
    el("rect", { x: x0, y: y0, width: size, height: size, fill: "none", stroke: C.rule }),
    axis({ scale: sx, orient: "bottom", at: y0 + size, span: [y0, y0 + size], title: L.gA, count: 4 }),
    axis({ scale: sy, orient: "left", at: x0, span: [x0, x0 + size], title: L.gB, count: 4 }),
    g({}, ...cells),
    el("line", { x1: sx(-LIM), y1: sy(-LIM), x2: sx(LIM), y2: sy(LIM), stroke: C.ink3, "stroke-width": 1, "stroke-dasharray": "2 3" }),
    text(sx(LIM) - 4, sy(LIM) + 14, L.tie, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-faint fig-t-halo" }),
  );
  // prior 90% ellipse: d ~ N(0, vd), u ~ N(0, vu)
  parts.push(el("path", { d: ellipse({ mean: 0, sd: Math.sqrt(s.vd) }, sdU, R90, sx, sy), fill: "none", stroke: C.ink3, "stroke-width": 1.3, "stroke-dasharray": "4 3" }));
  // exact contours
  for (const lev of s.levels) {
    const segs = contour(s.vals, GX, GX, lev);
    parts.push(el("path", { d: segs.map(([a, b, c, d]) => `M${sx(a).toFixed(1)},${sy(b).toFixed(1)}L${sx(c).toFixed(1)},${sy(d).toFixed(1)}`).join(""), fill: "none", stroke: C.ink, "stroke-width": 1.7, "stroke-linecap": "round" }));
  }
  const gaussians: Array<[Gauss1, string]> = [];
  if (p.laplace) gaussians.push([s.la, C.model]);
  if (p.ep) gaussians.push([s.ep, C.c6]);
  for (const [gs, color] of gaussians) {
    for (const r of [R50, R90]) parts.push(el("path", { d: ellipse(gs, sdU, r, sx, sy), fill: "none", stroke: color, "stroke-width": 1.8 }));
    parts.push(el("circle", { cx: sx(gs.mean / 2), cy: sy(-gs.mean / 2), r: 3.2, fill: color, stroke: C.paper, "stroke-width": 1.2 }));
  }
  // exact mean
  parts.push(el("circle", { cx: sx(s.ex.mean / 2), cy: sy(-s.ex.mean / 2), r: 3.6, fill: C.ink, stroke: C.paper, "stroke-width": 1.4 }));
  const ry = y0 + size + 56;
  parts.push(text(x0, ry, tpl(L.skew, { sd: fixed(s.skD, 2), sa: fixed(s.skA, 2) }), { "font-size": TYPE.small, class: "fig-t-num fig-t-strong" }));
  return svg(st.w, ry + 8, describe(st), g({}, ...parts));
}

export default defineFigure({
  name: "approx-posterior-2d",
  title: { en: "The posterior of two utilities after comparisons: skewed along the difference, Gaussian along the sum", zh: "比较之后两个效用的后验：沿差的方向偏斜，沿和的方向为高斯分布" },
  labels,
  params,
  hint: { en: "Change the prior correlation and the noise; the skew stays across the diagonal.", zh: "改变先验相关系数与噪声；偏斜始终在横跨对角线的方向上。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
