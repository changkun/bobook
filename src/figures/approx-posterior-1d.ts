// The posterior of one utility difference Δ = g(A) - g(B) after comparisons
// of A with B, and the Gaussian approximations to it. The prior is N(0, 1);
// each comparison multiplies in a probit factor. The exact posterior is
// computed on a fine grid; the approximations are the Laplace approximation
// (a Gaussian at the mode with the curvature there), expectation propagation
// (moment matching, site by site), Gaussian variational inference (the
// Gaussian closest to the posterior in KL(q || p)), and elliptical slice
// sampling. A table compares the mean, the standard deviation, and the
// probability that B is in fact better.

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid } from "./lib/scale.ts";
import { normalPdf } from "./lib/stats.ts";
import { memo } from "./lib/random.ts";
import { clip, curve, frame, frameAxes } from "./lib/plot.ts";
import { fixed, tpl } from "./lib/format.ts";
import { ep, ess, exactGrid, laplace, logLik, pNegGauss, vi, type Gauss1, type Problem } from "./lib/approx-probit.ts";

const labels = {
  en: {
    x: "utility difference Δ = g(A) − g(B)",
    y: "density",
    exact: "exact posterior",
    prior: "prior",
    lik: "likelihood (scaled)",
    laplace: "Laplace",
    ep: "EP",
    vi: "variational",
    mcmc: "samples",
    mode: "mode",
    mean: "mean",
    method: "method",
    meanH: "mean",
    sdH: "sd",
    pH: "P(Δ < 0)",
    obs: "A won {w} of {n} · noise σ = {s}",
    describe: "After A won {w} of {n} comparisons with noise σ = {s}, the exact posterior of the difference has mean {m}, mode {mo}, and sd {sd}; it gives B a probability of {p} of being better.{extra}",
    extraLa: " The Laplace approximation has mean {m} and sd {sd}, and gives B a probability of {p}.",
  },
  zh: {
    x: "效用差 Δ = g(A) − g(B)",
    y: "密度",
    exact: "精确后验",
    prior: "先验",
    lik: "似然（已缩放）",
    laplace: "Laplace 近似",
    ep: "期望传播",
    vi: "变分推断",
    mcmc: "样本",
    mode: "众数",
    mean: "均值",
    method: "方法",
    meanH: "均值",
    sdH: "标准差",
    pH: "P(Δ < 0)",
    obs: "{n} 次中 A 赢 {w} 次 · 噪声 σ = {s}",
    describe: "噪声 σ = {s}，{n} 次比较中 A 赢了 {w} 次，此时效用差的精确后验均值为 {m}，众数为 {mo}，标准差为 {sd}；它给出 B 实际上更好的概率为 {p}。{extra}",
    extraLa: "Laplace 近似的均值为 {m}，标准差为 {sd}，给出 B 更好的概率为 {p}。",
  },
};

const METHODS = [
  { value: "exact", label: { en: "Exact only", zh: "仅精确后验" } },
  { value: "laplace", label: { en: "Laplace", zh: "Laplace 近似" } },
  { value: "ep", label: { en: "EP", zh: "期望传播" } },
  { value: "vi", label: { en: "Variational", zh: "变分推断" } },
  { value: "mcmc", label: { en: "Sampling", zh: "采样" } },
  { value: "all", label: { en: "All", zh: "全部" } },
] as const;

const params = {
  sigma: { kind: "range", label: { en: "Noise σ", zh: "噪声 σ" }, min: 0.01, max: 2, default: 0.3, scale: "log" },
  wins: { kind: "range", label: { en: "A won", zh: "A 赢的次数" }, min: 0, max: 8, default: 1, step: 1 },
  losses: { kind: "range", label: { en: "B won", zh: "B 赢的次数" }, min: 0, max: 8, default: 0, step: 1 },
  method: { kind: "choice", label: { en: "Show", zh: "显示" }, options: METHODS, default: "laplace", control: "buttons" },
  factors: { kind: "toggle", label: { en: "Show prior and likelihood", zh: "显示先验与似然" }, default: false },
  samples: { kind: "range", label: { en: "Samples", zh: "样本数" }, min: 20, max: 5000, default: 400, scale: "log", control: false },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 4, step: 1, control: false },
} as const;

type P = { sigma: number; wins: number; losses: number; method: string; factors: boolean; samples: number; seed: number };

const DOM: [number, number] = [-3, 3];
const XS = grid(DOM[0], DOM[1], 361);

const problem = (p: P): Problem => ({ v0: 1, wins: p.wins, losses: p.losses, s: Math.SQRT2 * p.sigma });

const solve = memo((sigma: number, wins: number, losses: number) => {
  const pr: Problem = { v0: 1, wins, losses, s: Math.SQRT2 * sigma };
  const ex = exactGrid(pr, -6, 6, 3001);
  const la = laplace(pr);
  const e = ep(pr);
  const v = vi(pr, la);
  return { pr, ex, la, ep: e, vi: v };
}, 24);

const sampler = memo((sigma: number, wins: number, losses: number, n: number, seed: number) => ess({ v0: 1, wins, losses, s: Math.SQRT2 * sigma }, n, seed), 8);

// The exact density at the plotting grid, by linear interpolation.
function interp(xs: number[], ys: number[], x: number): number {
  const h = xs[1] - xs[0];
  const i = Math.floor((x - xs[0]) / h);
  if (i < 0 || i >= xs.length - 1) return 0;
  const t = (x - xs[i]) / h;
  return ys[i] * (1 - t) + ys[i + 1] * t;
}

function describe(st: State<P>): string {
  const p = st.p;
  const s = solve(p.sigma, p.wins, p.losses);
  const L = labels[st.lang ?? "en"];
  const extra = p.method === "laplace" || p.method === "all" ? tpl(L.extraLa, { m: fixed(s.la.mean, 2), sd: fixed(s.la.sd, 2), p: fixed(pNegGauss(s.la), 2) }) : "";
  return tpl(L.describe, { w: p.wins, n: p.wins + p.losses, s: fixed(p.sigma, 2), m: fixed(s.ex.mean, 2), mo: fixed(s.ex.mode, 2), sd: fixed(s.ex.sd, 2), p: fixed(s.ex.pNeg, 3), extra });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const s = solve(p.sigma, p.wins, p.losses);
  const ex = s.ex;
  const show = (m: string) => p.method === m || (p.method === "all" && m !== "mcmc");
  const gaussians: Array<[string, Gauss1, string]> = [];
  if (show("laplace")) gaussians.push([L.laplace, s.la, C.model]);
  if (show("ep")) gaussians.push([L.ep, s.ep, C.c6]);
  if (show("vi")) gaussians.push([L.vi, s.vi, C.c5]);
  const exD = XS.map((x) => interp(ex.xs, ex.dens, x));
  const curves = gaussians.map(([, gs]) => XS.map((x) => normalPdf(x, gs.mean, gs.sd)));
  let samples: number[] = [];
  let hist: Array<[number, number, number]> = [];
  if (p.method === "mcmc") {
    samples = sampler(p.sigma, p.wins, p.losses, Math.round(p.samples), p.seed);
    const bins = 48, bw = (DOM[1] - DOM[0]) / bins;
    const cnt = new Array<number>(bins).fill(0);
    for (const v of samples) { const b = Math.floor((v - DOM[0]) / bw); if (b >= 0 && b < bins) cnt[b]++; }
    hist = cnt.map((c, b) => [DOM[0] + b * bw, DOM[0] + (b + 1) * bw, c / (samples.length * bw)]);
  }
  const peak = Math.max(...exD, ...curves.flat(), ...hist.map((h) => h[2]), 0.5);
  const parts: string[] = [];

  // Legend and observation summary.
  const items: Array<[string, string, string]> = [[L.exact, C.ink, "line"]];
  if (p.factors) items.push([L.prior, C.ink3, "dot"], [L.lik, C.c4, "dash"]);
  for (const [name, , color] of gaussians) items.push([name, color, "line"]);
  if (p.method === "mcmc") items.push([L.mcmc, C.c7, "box"]);
  let lx = narrow ? 8 : 52, ly = 14;
  for (const [name, color, kind] of items) {
    const tw = labelWidth(name, 6.4) + 32;
    if (lx + tw > st.w - 8) { lx = narrow ? 8 : 52; ly += 17; }
    parts.push(kind === "box"
      ? el("rect", { x: lx, y: ly - 9, width: 16, height: 10, rx: 2, fill: color, opacity: 0.45 })
      : el("line", { x1: lx, x2: lx + 18, y1: ly - 4, y2: ly - 4, stroke: color, "stroke-width": kind === "line" ? 2.2 : 1.6, "stroke-dasharray": kind === "dash" ? "5 3" : kind === "dot" ? "1.5 3" : undefined }));
    parts.push(text(lx + 23, ly, name, { "font-size": TYPE.small, class: "fig-t-muted" }));
    lx += tw;
  }
  const top = ly + 30;
  const f = frame({ w: st.w, top, height: narrow ? 170 : 210, xDomain: DOM, yDomain: [0, peak * 1.12], yTitle: L.y });
  const cid = `${st.uid}-clip`;
  parts.push(el("defs", {}, clip(f, cid)), frameAxes(f, { xTitle: L.x, yTitle: L.y, yCount: 3 }));
  // d = 0: A and B equally good
  parts.push(el("line", { x1: f.x(0), x2: f.x(0), y1: f.top, y2: f.bottom, stroke: C.ink3, "stroke-width": 1 }));
  const layers: string[] = [];
  if (hist.length) for (const [a, b, h] of hist) layers.push(el("rect", { x: f.x(a) + 0.5, y: f.y(h), width: Math.max(0.5, f.x(b) - f.x(a) - 1), height: f.bottom - f.y(h), fill: C.c7, opacity: 0.42 }));
  // exact posterior, filled
  layers.push(el("path", { d: `M${f.x(XS[0])},${f.bottom}` + XS.map((x, i) => `L${f.x(x)},${f.y(exD[i])}`).join("") + `L${f.x(XS[XS.length - 1])},${f.bottom}Z`, fill: C.ink, opacity: 0.08 }));
  if (p.factors) {
    const pr = problem(p);
    const lik = XS.map((x) => Math.exp(logLik(pr, x)));
    const lmax = Math.max(...lik) || 1;
    layers.push(
      curve(f, XS, XS.map((x) => normalPdf(x, 0, 1)), { stroke: C.ink3, "stroke-width": 1.6, "stroke-dasharray": "1.5 3" }),
      curve(f, XS, lik.map((v) => (v / lmax) * peak), { stroke: C.c4, "stroke-width": 1.6, "stroke-dasharray": "5 3" }),
    );
  }
  gaussians.forEach(([, , color], i) => layers.push(curve(f, XS, curves[i], { stroke: color, "stroke-width": 2.2 })));
  layers.push(curve(f, XS, exD, { stroke: C.ink, "stroke-width": 2 }));
  parts.push(g({ "clip-path": `url(#${cid})` }, ...layers));
  // mode and mean of the exact posterior
  const xm = f.x(ex.mode), xa = f.x(ex.mean);
  const close = Math.abs(xm - xa) < 34;
  parts.push(
    el("line", { x1: xm, x2: xm, y1: f.y(interp(ex.xs, ex.dens, ex.mode)), y2: f.bottom, stroke: C.ink, "stroke-width": 1.2, "stroke-dasharray": "3 3" }),
    el("line", { x1: xa, x2: xa, y1: f.y(interp(ex.xs, ex.dens, ex.mean)), y2: f.bottom, stroke: C.ink, "stroke-width": 1.2 }),
    text(xm + (close ? -3 : 0), f.top - 6, L.mode, { "text-anchor": close ? "end" : "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
    text(xa + (close ? 3 : 0), f.top - 6, L.mean, { "text-anchor": close ? "start" : "middle", "font-size": TYPE.small, class: "fig-t-strong fig-t-halo" }),
  );

  // The table.
  const rows: Array<[string, string, number, number, number]> = [[L.exact, C.ink, ex.mean, ex.sd, ex.pNeg]];
  for (const [name, gs, color] of gaussians) rows.push([name, color, gs.mean, gs.sd, pNegGauss(gs)]);
  if (samples.length) {
    const m = samples.reduce((a, b) => a + b, 0) / samples.length;
    const sd = Math.sqrt(samples.reduce((a, b) => a + (b - m) ** 2, 0) / samples.length);
    rows.push([`${L.mcmc} (${samples.length})`, C.c7, m, sd, samples.filter((v) => v < 0).length / samples.length]);
  }
  const ty = f.bottom + 74;
  parts.push(text(narrow ? 8 : f.left, f.bottom + 50, tpl(L.obs, { w: p.wins, n: p.wins + p.losses, s: fixed(p.sigma, 2) }), { "font-size": TYPE.small, class: "fig-t-muted" }));
  const c0 = narrow ? 8 : f.left, c1 = narrow ? st.w * 0.5 : f.left + 190, c2 = c1 + (narrow ? 52 : 80), c3 = c2 + (narrow ? 64 : 90);
  parts.push(
    text(c0, ty, L.method, { "font-size": TYPE.small, class: "fig-t-faint" }),
    text(c1, ty, L.meanH, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-faint" }),
    text(c2, ty, L.sdH, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-faint" }),
    text(c3, ty, L.pH, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-faint" }),
    el("line", { x1: c0, x2: c3, y1: ty + 5, y2: ty + 5, stroke: C.rule }),
  );
  rows.forEach(([name, color, m, sd, pn], i) => {
    const y = ty + 21 + i * 17;
    parts.push(
      el("rect", { x: c0, y: y - 9, width: 10, height: 10, rx: 2, fill: color, opacity: color === C.c7 ? 0.5 : 1 }),
      text(c0 + 15, y, name, { "font-size": TYPE.small, class: i === 0 ? "fig-t-strong" : "fig-t-muted" }),
      text(c1, y, fixed(m, 2), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-num" }),
      text(c2, y, fixed(sd, 2), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-num" }),
      text(c3, y, pn < 0.0005 ? "< 0.001" : fixed(pn, 3), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-num" }),
    );
  });
  const H = ty + 21 + rows.length * 17 + 4;
  return svg(st.w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "approx-posterior-1d",
  title: { en: "The posterior of a utility difference after comparisons, and its Gaussian approximations", zh: "比较之后效用差的后验及其高斯近似" },
  labels,
  params,
  hint: { en: "Move the noise toward 0.01 and watch the mode and the mean separate.", zh: "把噪声调向 0.01，观察众数与均值如何分开。" },
  actions: [
    { label: { en: "More samples", zh: "更多样本" }, run: (p) => ({ ...p, samples: Math.min(5000, Math.round(p.samples * 5)) }), enabled: (p) => p.method === "mcmc" && p.samples < 5000 },
    { label: { en: "Fewer samples", zh: "更少样本" }, run: (p) => ({ ...p, samples: Math.max(20, Math.round(p.samples / 5)) }), enabled: (p) => p.method === "mcmc" && p.samples > 20 },
    { label: { en: "New chain", zh: "新链" }, run: (p) => ({ ...p, seed: (p.seed % 999) + 1 }), enabled: (p) => p.method === "mcmc" },
  ],
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
