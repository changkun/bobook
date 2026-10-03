// What EUBO looks like over all possible pairs. Left (or top): the latent
// utility posterior after a few duels on the running objective. Right (or
// below): EUBO(x1, x2) for every pair on a grid, as a heatmap; the matrix is
// symmetric, its diagonal equals the posterior mean (a pair of identical
// options is worth exactly one option), and its maximum, marked, is the next
// query. Click a cell to read how that pair's value splits into the two means
// and the uncertainty of their difference.

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { fitPreference, kernel, pointIndex, predictPreference, type Duel, type X } from "./lib/gp.ts";
import { expectedMax2 } from "./lib/acq.ts";
import { parseDuels, validDuels, parseNumbers, validNumbers } from "./lib/params.ts";
import { memo } from "./lib/random.ts";
import { running } from "./lib/objectives.ts";
import { band, curve, frame, frameAxes } from "./lib/plot.ts";
import { axis } from "./lib/axis.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "x",
    u: "latent utility",
    x1: "first option x₁",
    x2: "second option x₂",
    heat: "EUBO(x₁, x₂)",
    next: "next pair",
    sel: "selected pair",
    readout: "x₁ = {a}, x₂ = {b}: μ₁ = {m1}, μ₂ = {m2}, sd of difference s = {s}, EUBO = {v}",
    describe: "After {n} duels, EUBO is largest for the pair x₁ = {a}, x₂ = {b}, with value {v}. The best single option by posterior mean is at x = {xb}.",
  },
  zh: {
    x: "x",
    u: "潜在效用",
    x1: "第一个选项 x₁",
    x2: "第二个选项 x₂",
    heat: "EUBO(x₁, x₂)",
    next: "下一对",
    sel: "选中的对",
    readout: "x₁ = {a}，x₂ = {b}：μ₁ = {m1}，μ₂ = {m2}，差值的标准差 s = {s}，EUBO = {v}",
    describe: "经过 {n} 次对决，EUBO 在 x₁ = {a}、x₂ = {b} 这一对上最大，值为 {v}。按后验均值，最好的单个选项位于 x = {xb}。",
  },
};

const params = {
  duels: { kind: "data", label: { en: "Answers", zh: "回答" }, default: "0.25>0.05,0.25>0.5,0.7>0.5,0.95>0.85,0.25>0.95", validate: validDuels },
  select: { kind: "data", label: { en: "Selected pair", zh: "选中的对" }, default: "", validate: validNumbers },
  lengthscale: { kind: "range", label: { en: "Lengthscale ℓ", zh: "长度尺度 ℓ" }, min: 0.05, max: 0.4, default: 0.12, scale: "log" },
} as const;

type P = { duels: string; select: string; lengthscale: number };

const G = 31;
const GX = grid(0, 1, G);
const XS = grid(0, 1, 121);

const compute = memo((duelStr: string, ls: number) => {
  const ds = parseDuels(duelStr);
  const xs: X[] = [];
  const duels: Duel[] = ds.map(([w, l]) => ({ winner: pointIndex(xs, w), loser: pointIndex(xs, l) }));
  const k = kernel("rbf", ls, 1);
  const fitted = fitPreference(k, xs, duels, 0.1);
  const post = predictPreference(fitted, XS);
  const gp = predictPreference(fitted, GX, true);
  const M: number[][] = GX.map((_, i) => GX.map((__, j) => expectedMax2(gp.mean[i], gp.cov![i][i], gp.mean[j], gp.cov![j][j], gp.cov![i][j])));
  let bi = 0, bj = 1, bv = -Infinity, lo = Infinity;
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
    lo = Math.min(lo, M[i][j]);
    if (i !== j && M[i][j] > bv) { bv = M[i][j]; bi = i; bj = j; }
  }
  return { xs: xs as number[], post, gp, M, best: [bi, bj] as [number, number], vmax: bv, vmin: lo };
}, 16);

function describe(st: State<P>): string {
  const c = compute(st.p.duels, st.p.lengthscale);
  const xb = XS[c.post.mean.indexOf(Math.max(...c.post.mean))];
  return tpl(labels[st.lang ?? "en"].describe, { n: parseDuels(st.p.duels).length, a: fixed(GX[c.best[0]], 2), b: fixed(GX[c.best[1]], 2), v: fixed(c.vmax, 2), xb: fixed(xb, 2) });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 560;
  const c = compute(p.duels, p.lengthscale);
  const ds = parseDuels(p.duels);
  const parts: string[] = [];
  // Layout: side by side when wide, stacked when narrow.
  const leftW = narrow ? st.w : Math.floor(st.w * 0.5);
  const f = frame({ w: leftW, top: 26, height: narrow ? 150 : 210, yDomain: [-2.4, 2.4], yTitle: L.u });
  const lo = c.post.mean.map((m, i) => m - 1.96 * Math.sqrt(c.post.var[i]));
  const hi = c.post.mean.map((m, i) => m + 1.96 * Math.sqrt(c.post.var[i]));
  parts.push(frameAxes(f, { yTitle: L.u, xTitle: L.x, yCount: 3 }), band(f, XS, lo, hi), curve(f, XS, c.post.mean),
    curve(f, XS, XS.map((x) => (running.f(x) - 0.2) * 2), { stroke: C.truth, "stroke-dasharray": "5 4", "stroke-width": 1.3, opacity: 0.8 }));
  // duels: an arrow from loser to winner along the top
  ds.forEach(([w, l], i) => {
    const y = f.top - 6 - (i % 3) * 6;
    parts.push(el("line", { x1: f.x(l), x2: f.x(w), y1: y, y2: y, stroke: C.ink3, "stroke-width": 1 }), el("circle", { cx: f.x(w), cy: y, r: 2.6, fill: C.ink }));
  });
  for (const x of c.xs) {
    const i = Math.round(x * (XS.length - 1));
    parts.push(el("circle", { cx: f.x(x), cy: f.y(c.post.mean[i]), r: 3.4, fill: C.ink, stroke: C.paper, "stroke-width": 1.4 }));
  }
  const [bi, bj] = c.best;
  for (const x of [GX[bi], GX[bj]]) parts.push(el("line", { x1: f.x(x), x2: f.x(x), y1: f.top, y2: f.bottom, stroke: C.acq, "stroke-width": 1.4, "stroke-dasharray": "4 3" }));
  // Heatmap.
  const hx0 = narrow ? 52 : leftW + 56;
  const hy0 = narrow ? f.bottom + 56 : 26;
  const size = narrow ? Math.min(st.w - hx0 - 14, 260) : Math.min(st.w - hx0 - 14, 210 + 40);
  const cell = size / G;
  const sx = linear([0, 1], [hx0 + cell / 2, hx0 + size - cell / 2]);
  const sy = linear([0, 1], [hy0 + size - cell / 2, hy0 + cell / 2]);
  const t = (v: number) => (v - c.vmin) / (c.vmax - c.vmin || 1);
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
    parts.push(el("rect", { x: hx0 + i * cell, y: hy0 + size - (j + 1) * cell, width: cell + 0.4, height: cell + 0.4, fill: C.model, opacity: (0.04 + 0.92 * t(c.M[i][j]) ** 1.5).toFixed(3), "data-fig-hit": `cell-${i}-${j}` }));
  }
  parts.push(el("line", { x1: hx0, y1: hy0 + size, x2: hx0 + size, y2: hy0, stroke: C.ink3, "stroke-width": 0.8, "stroke-dasharray": "2 3" }));
  parts.push(
    axis({ scale: linear([0, 1], [hx0, hx0 + size]), orient: "bottom", at: hy0 + size, title: L.x1, count: 2, grid: false }),
    axis({ scale: linear([0, 1], [hy0 + size, hy0]), orient: "left", at: hx0, title: L.x2, count: 2, grid: false }),
    text(hx0 + size / 2, hy0 - 8, L.heat, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }),
  );
  // the maximum (both symmetric copies)
  for (const [a, b] of [[bi, bj], [bj, bi]]) parts.push(el("rect", { x: hx0 + a * cell - 1, y: hy0 + size - (b + 1) * cell - 1, width: cell + 2, height: cell + 2, fill: "none", stroke: C.acq, "stroke-width": 2 }));
  // the selected cell
  const sel = parseNumbers(p.select);
  let readout = "";
  if (sel.length === 2) {
    const [a, b] = sel;
    parts.push(el("rect", { x: hx0 + a * cell - 1, y: hy0 + size - (b + 1) * cell - 1, width: cell + 2, height: cell + 2, fill: "none", stroke: C.ink, "stroke-width": 1.6 }));
    const g2 = c.gp;
    const s = Math.sqrt(Math.max(0, g2.cov![a][a] + g2.cov![b][b] - 2 * g2.cov![a][b]));
    readout = tpl(L.readout, { a: fixed(GX[a], 2), b: fixed(GX[b], 2), m1: fixed(g2.mean[a], 2), m2: fixed(g2.mean[b], 2), s: fixed(s, 2), v: fixed(c.M[a][b], 2) });
    for (const x of [GX[a], GX[b]]) parts.push(el("line", { x1: f.x(x), x2: f.x(x), y1: f.top, y2: f.bottom, stroke: C.ink, "stroke-width": 1.2 }));
  }
  void sx; void sy;
  const H = Math.max(f.bottom + 40, hy0 + size + 40) + (readout ? 24 : 6);
  if (readout) parts.push(text(narrow ? 8 : 40, H - 10, readout, { "font-size": TYPE.small, class: "fig-t-num" }));
  return svg(st.w, H, describe(st), g({}, ...parts));
}

export default defineFigure({
  name: "eubo-map",
  title: { en: "EUBO over all pairs", zh: "所有对上的 EUBO" },
  labels,
  params,
  hint: { en: "Click a cell of the map to see how that pair's EUBO is made up.", zh: "点击图中的一个格子，查看这一对的 EUBO 由哪些部分构成。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click" || !e.target?.startsWith("cell-")) return null;
    const [, i, j] = e.target.split("-");
    return { ...p, select: `${i},${j}` };
  },
  actions: [
    {
      label: { en: "Ask the next pair", zh: "询问下一对" }, primary: true,
      run: (p) => {
        // Answer the EUBO pair the way the running objective would (noise free) and continue.
        const c = compute(p.duels, p.lengthscale);
        const a = GX[c.best[0]], b = GX[c.best[1]];
        const w = running.f(a) >= running.f(b) ? [a, b] : [b, a];
        return { ...p, duels: `${p.duels}${p.duels ? "," : ""}${w[0].toFixed(4)}>${w[1].toFixed(4)}`, select: "" };
      },
    },
  ],
});
