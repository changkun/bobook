// Where the acquisition function has anything to say, as the dimension grows.
// The objective is Hartmann-6 (a two-dimensional slice through its maximum
// for d = 2, and Hartmann-6 hidden among irrelevant inputs for d > 6). After a
// Latin hypercube of 10 + 2d evaluations, a Gaussian process with the
// dimension-scaled lengthscale 0.2 sqrt(d) is fitted, and expected
// improvement is computed at two candidate sets of 1,000 points each: points
// uniform in the cube, and Gaussian perturbations (sd 0.05 per coordinate) of
// the five best evaluations. The left panel compares how far each candidate
// is from the nearest evaluation, in lengthscales; the right panel compares
// the EI values on a log scale. In two dimensions the two sets do equally
// well; in twenty, the uniform candidates sit where EI is many orders of
// magnitude below its maximum, which is why practical maximizers screen
// candidates near the data and then climb with gradients.

import { defineFigure, type State } from "./types.ts";
import { el, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { fit, kernel, predict } from "./lib/gp.ts";
import { ei } from "./lib/acq.ts";
import { memo, rng } from "./lib/random.ts";
import { hartmann6, embeddedHartmann6 } from "./lib/objectives.ts";
import { candidates, dist, latinHypercube, uniformPoints } from "./lib/nd.ts";
import { sig, tpl } from "./lib/format.ts";

const labels = {
  en: {
    uniform: "uniform in the cube",
    local: "near the 5 best evaluations",
    distTitle: "distance to the nearest evaluation (lengthscales)",
    eiTitle: "expected improvement (log scale)",
    best: "best {v}",
    readout: "d = {d}, {n} evaluations: the best uniform candidate has EI {u}; the best local one {l} ({ratio})",
    readout1: "d = {d}, {n} evaluations. Best EI:",
    readout2: "uniform {u}, local {l} ({ratio})",
    ratio: "{r}× larger",
    ratioSame: "about the same",
    describe: "In {d} dimensions after {n} evaluations, the largest expected improvement among 1,000 uniform candidates is {u}, and among 1,000 candidates near the best evaluations it is {l}. {f} of the uniform candidates have expected improvement below one millionth.",
  },
  zh: {
    uniform: "在立方体中均匀分布",
    local: "最好的 5 次评估附近",
    distTitle: "到最近评估点的距离（以长度尺度计）",
    eiTitle: "期望改进（对数刻度）",
    best: "最大值 {v}",
    readout: "d = {d}，{n} 次评估：最好的均匀候选点 EI 为 {u}；最好的局部候选点为 {l}（{ratio}）",
    readout1: "d = {d}，{n} 次评估。最大 EI：",
    readout2: "均匀 {u}，局部 {l}（{ratio}）",
    ratio: "是前者的 {r} 倍",
    ratioSame: "大致相同",
    describe: "在 {d} 维中评估 {n} 次之后，1,000 个均匀候选点中最大的期望改进为 {u}，最好评估点附近的 1,000 个候选点中最大的为 {l}。均匀候选点中有 {f} 的期望改进低于百万分之一。",
  },
};

const DIMS = [
  { value: 2, label: { en: "d = 2", zh: "d = 2" } },
  { value: 6, label: { en: "6", zh: "6" } },
  { value: 10, label: { en: "10", zh: "10" } },
  { value: 20, label: { en: "20", zh: "20" } },
] as const;

const params = {
  d: { kind: "choice", label: { en: "Dimension", zh: "维度" }, options: DIMS, default: 20, control: "buttons" },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 11, step: 1, control: false },
} as const;

type P = { d: number; seed: number };

const NC = 1000;
const LOG_LO = -12;

function objective(x: number[], d: number): number {
  if (d === 2) { const z = hartmann6.argmax.slice(); z[0] = x[0]; z[1] = x[1]; return hartmann6.f(z); }
  return embeddedHartmann6(x);
}

const compute = memo((d: number, seed: number) => {
  const r = rng(seed * 101 + d);
  const n = 10 + 2 * d;
  const X = latinHypercube(r, n, d);
  const y = X.map((x) => objective(x, d));
  const ym = y.reduce((a, b) => a + b, 0) / n;
  const ys = Math.sqrt(y.reduce((a, b) => a + (b - ym) ** 2, 0) / (n - 1)) || 1;
  const z = y.map((v) => (v - ym) / ys);
  const ls = 0.2 * Math.sqrt(d);
  const gp = fit(kernel("rbf", ls, 1), X, z, 1e-4, 0);
  const best = Math.max(...z);
  const top = z.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 5).map((q) => X[q[1]]);
  const sets = [uniformPoints(r, NC, d), candidates(r, d, 0, top, NC, 0.05)];
  return {
    n,
    sets: sets.map((S) => {
      const post = predict(gp, S);
      const e = post.mean.map((m, i) => ei(m, Math.sqrt(post.var[i]), best, 0));
      const near = S.map((s) => Math.min(...X.map((x) => dist(s, x))) / ls);
      return { e, near, max: Math.max(...e), tiny: e.filter((v) => v < 1e-6).length / NC };
    }),
  };
}, 16);

function describe(st: State<P>): string {
  const c = compute(st.p.d, st.p.seed);
  return tpl(labels[st.lang ?? "en"].describe, {
    d: st.p.d, n: c.n, u: sig(c.sets[0].max, 2), l: sig(c.sets[1].max, 2), f: `${Math.round(100 * c.sets[0].tiny)}%`,
  });
}

// 10 to a negative integer power, with Unicode superscripts: 10⁻⁶.
const SUP: Record<string, string> = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
const sup = (v: number) => String(v).split("").map((ch) => SUP[ch] ?? ch).join("");

function hist(vals: number[], lo: number, hi: number, bins: number): number[] {
  const h = new Array<number>(bins).fill(0);
  for (const v of vals) h[Math.max(0, Math.min(bins - 1, Math.floor(((v - lo) / (hi - lo)) * bins)))]++;
  return h;
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const c = compute(p.d, p.seed);
  const parts: string[] = [];
  const ratio = c.sets[1].max / Math.max(c.sets[0].max, 1e-300);
  const rtxt = ratio > 1.5 ? tpl(L.ratio, { r: sig(ratio, 2) }) : L.ratioSame;
  const readout = tpl(L.readout, { d: p.d, n: c.n, u: sig(c.sets[0].max, 2), l: sig(c.sets[1].max, 2), ratio: rtxt });
  // The readout takes two shorter lines at phone width.
  if (narrow) {
    parts.push(text(8, 16, tpl(L.readout1, { d: p.d, n: c.n }), { "font-size": TYPE.small, class: "fig-t-num fig-t-strong" }),
      text(8, 31, tpl(L.readout2, { u: sig(c.sets[0].max, 2), l: sig(c.sets[1].max, 2), ratio: rtxt }), { "font-size": TYPE.small, class: "fig-t-num fig-t-strong" }));
  } else parts.push(text(16, 16, readout, { "font-size": TYPE.small, class: "fig-t-num fig-t-strong" }));

  const rowH = narrow ? 46 : 54;
  const rowGap = 26;
  const colGap = 44;
  const left = narrow ? 14 : 16;
  const top0 = narrow ? 64 : 52;
  const colW = narrow ? st.w - left - 12 : (st.w - left - 12 - colGap) / 2;
  const colors = [C.ink3, C.acq];
  const names = [L.uniform, L.local];

  const panelsTop = [top0, narrow ? top0 + 2 * (rowH + rowGap) + 52 : top0];
  const panelsLeft = [left, narrow ? left : left + colW + colGap];

  // Left: distance to the nearest evaluation; right: log10 EI.
  const specs = [
    { title: L.distTitle, lo: 0, hi: 4, bins: 32, vals: c.sets.map((s) => s.near), ticks: [0, 1, 2, 3, 4], fmt: (v: number) => String(v) },
    { title: L.eiTitle, lo: LOG_LO, hi: 0, bins: 36, vals: c.sets.map((s) => s.e.map((v) => Math.log10(Math.max(v, 1e-300)))), ticks: [-12, -9, -6, -3, 0], fmt: (v: number) => (v === 0 ? "1" : `10${sup(v)}`) },
  ];
  specs.forEach((sp, q) => {
    const x0 = panelsLeft[q], y0 = panelsTop[q];
    const sx = linear([sp.lo, sp.hi], [x0, x0 + colW]);
    parts.push(text(x0, y0 - 8, sp.title, { "font-size": TYPE.small, class: "fig-t-muted" }));
    sp.vals.forEach((vals, k) => {
      const h = hist(vals, sp.lo, sp.hi, sp.bins);
      const hmax = Math.max(...h, 1);
      const yb = y0 + (k + 1) * rowH + k * rowGap;
      const bw = colW / sp.bins;
      h.forEach((cnt, b) => {
        if (!cnt) return;
        const hh = ((rowH - 14) * cnt) / hmax;
        parts.push(el("rect", { x: x0 + b * bw + 0.5, y: yb - hh, width: bw - 1, height: hh, fill: colors[k], opacity: 0.75 }));
      });
      parts.push(
        el("line", { x1: x0, x2: x0 + colW, y1: yb, y2: yb, stroke: C.rule }),
        text(x0 + 2, yb - rowH + 10, names[k], { "font-size": TYPE.small, class: k ? "fig-t-strong" : "fig-t-muted" }),
      );
      if (q === 1) {
        const mx = Math.log10(Math.max(c.sets[k].max, 1e-300));
        const px = sx(Math.max(LOG_LO, Math.min(0, mx)));
        parts.push(
          el("line", { x1: px, x2: px, y1: yb - rowH + 14, y2: yb, stroke: colors[k], "stroke-width": 2 }),
          text(x0 + colW, yb - rowH + 10, tpl(L.best, { v: sig(c.sets[k].max, 2) }), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-num fig-t-halo" }),
        );
      }
      if (k === 1) parts.push(axis({ scale: sx, orient: "bottom", at: yb, ticks: sp.ticks, format: sp.fmt, grid: false }));
    });
  });
  const H = panelsTop[1] + 2 * rowH + rowGap + 30;
  return svg(st.w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "acq-highdim",
  title: { en: "Where expected improvement is not flat", zh: "期望改进在哪里不平坦" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "New draw", zh: "重新抽取" }, run: (p) => ({ ...p, seed: (p.seed % 999) + 1 }) },
  ],
});
