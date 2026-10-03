// The preference model on a two-dimensional design space. A simulated person
// whose utility is the Branin function (rescaled to the unit square, larger is
// better) answers comparisons with probit noise. The pairs follow one of four
// designs that spend the same number of comparisons differently: random pairs
// among a pool of designs, a chain, a star around one design, or isolated
// pairs that share no design. The Gaussian process preference model is fitted
// with the Laplace approximation (lib/gp.ts). Three maps: the hidden utility,
// the posterior mean, and the posterior standard deviation of the utility
// relative to its average over the square (the overall level is not something
// comparisons measure). The comparison graph is drawn over the maps.

import { defineFigure, tr, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, CATEGORICAL, TYPE } from "./lib/theme.ts";
import { grid } from "./lib/scale.ts";
import { fitPreference, kernel, predictPreference, type Duel } from "./lib/gp.ts";
import { memo, rng } from "./lib/random.ts";
import { Phi } from "./lib/stats.ts";
import { branin01 } from "./lib/objectives.ts";
import { fixed, pct, tpl } from "./lib/format.ts";

const labels = {
  en: {
    truth: "hidden utility (Branin), by rank",
    mean: "posterior mean, by rank",
    sd: "posterior sd, relative to the average",
    status: "{n} comparisons · {m} designs · {k} component{k:/s}",
    agree: "pairs of compared designs ordered correctly: {o} · rank agreement over the square: {r}",
    pickq: "utility of the recommended design: {b} · best shown {bs} · best possible {mx}",
    describe: "{d}: {n} comparisons among {m} designs in {k} connected component{k:/s}. The posterior mean orders {o} of the pairs of compared designs correctly and ranks the grid with Spearman correlation {r} against the hidden utility.",
  },
  zh: {
    truth: "隐藏效用（Branin），按秩着色",
    mean: "后验均值，按秩着色",
    sd: "后验标准差，相对于平均值",
    status: "{n} 次比较 · {m} 个设计 · {k} 个连通分量",
    agree: "已比较设计中排序正确的对：{o} · 整个正方形上的秩一致性：{r}",
    pickq: "推荐设计的效用：{b} · 已展示设计的最优值 {bs} · 可能的最优值 {mx}",
    describe: "{d}：{m} 个设计之间有 {n} 次比较，构成 {k} 个连通分量。在已比较设计的所有对中，后验均值对 {o} 排序正确；它在网格上的排序与隐藏效用的 Spearman 相关系数为 {r}。",
  },
};

const DESIGNS = [
  { value: "random", label: { en: "Random pairs", zh: "随机对" } },
  { value: "chain", label: { en: "Chain", zh: "链" } },
  { value: "star", label: { en: "Star", zh: "星形" } },
  { value: "isolated", label: { en: "Isolated pairs", zh: "孤立对" } },
] as const;

const params = {
  design: { kind: "choice", label: { en: "Design", zh: "布局" }, options: DESIGNS, default: "random" },
  comparisons: { kind: "range", label: { en: "Comparisons", zh: "比较次数" }, min: 4, max: 40, default: 20, step: 1 },
  lengthscale: { kind: "range", label: { en: "Lengthscale ℓ", zh: "长度尺度 ℓ" }, min: 0.08, max: 0.8, default: 0.25, scale: "log" },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 34, step: 1, control: false },
} as const;

type P = { design: string; comparisons: number; lengthscale: number; seed: number };

const G = 28;
const GX = grid(0, 1, G);
const CELLS: number[][] = [];
for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) CELLS.push([GX[i], GX[j]]);
const SIGMA = 0.1; // the person's noise and the model's; illustrative
const U = CELLS.map(([a, b]) => branin01(a, b));
const UMAX = Math.max(...U);
// The readout wraps by character count; a Chinese character counts as two.
const vlen = (s: string) => labelWidth(s, 1, 2);

function ranks(v: number[]): number[] {
  const idx = v.map((x, i) => [x, i] as [number, number]).sort((a, b) => a[0] - b[0]);
  const r = new Array<number>(v.length);
  idx.forEach(([, i], k) => { r[i] = k; });
  return r;
}

function spearman(a: number[], b: number[]): number {
  const ra = ranks(a), rb = ranks(b);
  const n = a.length, m = (n - 1) / 2;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < n; i++) { num += (ra[i] - m) * (rb[i] - m); da += (ra[i] - m) ** 2; db += (rb[i] - m) ** 2; }
  return num / Math.sqrt(da * db);
}

const compute = memo((design: string, m: number, ls: number, seed: number) => {
  const r = rng(seed * 7919 + 17);
  const pt = () => [r(), r()];
  let xs: number[][] = [];
  const pairs: Array<[number, number]> = [];
  if (design === "isolated") {
    xs = Array.from({ length: 2 * m }, pt);
    for (let i = 0; i < m; i++) pairs.push([2 * i, 2 * i + 1]);
  } else if (design === "chain") {
    xs = Array.from({ length: m + 1 }, pt);
    for (let i = 0; i < m; i++) pairs.push([i, i + 1]);
  } else if (design === "star") {
    xs = Array.from({ length: m + 1 }, pt);
    for (let i = 1; i <= m; i++) pairs.push([0, i]);
  } else {
    const n = Math.max(4, Math.ceil(0.75 * m) + 2);
    xs = Array.from({ length: n }, pt);
    let guard = 0;
    while (pairs.length < m && guard++ < 5000) {
      const a = Math.floor(r() * n), b = Math.floor(r() * n);
      if (a !== b && !pairs.some(([p, q]) => (p === a && q === b) || (p === b && q === a))) pairs.push([a, b]);
    }
  }
  // the simulated person answers each pair
  const duels: Duel[] = pairs.map(([a, b]) => {
    const pa = Phi((branin01(xs[a][0], xs[a][1]) - branin01(xs[b][0], xs[b][1])) / (Math.SQRT2 * SIGMA));
    return r() < pa ? { winner: a, loser: b } : { winner: b, loser: a };
  });
  const k = kernel("rbf", ls, 1);
  const fit = fitPreference(k, xs, duels, SIGMA);
  const post = predictPreference(fit, CELLS);
  // sd of g(x) minus the grid average: Var g(x) - 2 Cov(g(x), avg) + Var avg
  const N = CELLS.length, n = xs.length;
  const kx = CELLS.map((c) => xs.map((x) => k(c, x))); // N x n
  const kbar = xs.map((_, j) => kx.reduce((s, row) => s + row[j], 0) / N); // mean over cells of k(c, x_j)
  const Mkbar = fit.M.map((row) => row.reduce((s, v, j) => s + v * kbar[j], 0));
  const rowSum = CELLS.map((c) => CELLS.reduce((s, d) => s + k(c, d), 0) / N);
  const priorAvg = rowSum.reduce((s, v) => s + v, 0) / N;
  const varAvg = priorAvg - kbar.reduce((s, v, j) => s + v * Mkbar[j], 0);
  const sdRel = CELLS.map((_, i) => {
    const covAvg = rowSum[i] - kx[i].reduce((s, v, j) => s + v * Mkbar[j], 0);
    return Math.sqrt(Math.max(1e-12, post.var[i] - 2 * covAvg + varAvg));
  });
  // components
  const parent = xs.map((_, i) => i);
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  for (const d of duels) parent[find(d.winner)] = find(d.loser);
  const roots = new Map<number, number>();
  const comp = xs.map((_, i) => { const rt = find(i); if (!roots.has(rt)) roots.set(rt, roots.size); return roots.get(rt)!; });
  // the recommendation: the compared design with the highest posterior mean
  const at = predictPreference(fit, xs).mean;
  let bi = 0;
  for (let i = 1; i < n; i++) if (at[i] > at[bi]) bi = i;
  const shown = xs.map(([a, b]) => branin01(a, b));
  // how many pairs of compared designs the posterior mean orders correctly
  let ok = 0, tot = 0;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { tot++; if ((at[i] - at[j]) * (shown[i] - shown[j]) > 0) ok++; }
  return { xs, duels, post, sdRel, comp, k: roots.size, rho: spearman(post.mean, U), order: ok / tot, best: shown[bi], bestShown: Math.max(...shown), bestAt: xs[bi] };
}, 16);

function describe(st: State<P>): string {
  const p = st.p;
  const c = compute(p.design, p.comparisons, p.lengthscale, p.seed);
  return tpl(labels[st.lang ?? "en"].describe, { d: tr(DESIGNS.find((d) => d.value === p.design)!.label, st.lang ?? "en"), n: c.duels.length, m: c.xs.length, k: c.k, o: pct(c.order), r: fixed(c.rho, 2) });
}

// A heatmap of one value per grid cell. `byRank` colors by rank instead of
// value: comparisons identify the order of utilities, not their scale.
function heat(x0: number, y0: number, size: number, vals: number[], color: string, byRank: boolean): string {
  const cell = size / G;
  const out: string[] = [];
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const r = byRank ? ranks(vals) : [];
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
    const k = i * G + j;
    const v = byRank ? r[k] / (vals.length - 1) : (vals[k] - lo) / (hi - lo || 1);
    // an opaque mix of the series color with the paper, so overlapping cell
    // edges leave no seams in either theme
    const pct = Math.round(4 + 84 * v * v);
    out.push(el("rect", { x: x0 + i * cell, y: y0 + size - (j + 1) * cell, width: cell + 0.5, height: cell + 0.5, fill: `color-mix(in srgb, ${color} ${pct}%, ${C.paper})` }));
  }
  return g({}, g({}, ...out), el("rect", { x: x0, y: y0, width: size + 0.5, height: size + 0.5, fill: "none", stroke: C.rule }));
}

function overlay(x0: number, y0: number, size: number, c: ReturnType<typeof compute>): string {
  const X = (v: number) => x0 + v * size, Y = (v: number) => y0 + size - v * size;
  const colorOf = (k: number) => (k < CATEGORICAL.length ? CATEGORICAL[k] : C.ink3);
  const out: string[] = [];
  for (const d of c.duels) {
    const a = c.xs[d.winner], b = c.xs[d.loser];
    out.push(el("line", { x1: X(a[0]), y1: Y(a[1]), x2: X(b[0]), y2: Y(b[1]), stroke: C.ink, "stroke-width": 1.3, opacity: 0.75 }));
  }
  c.xs.forEach((x, i) => out.push(el("circle", { cx: X(x[0]), cy: Y(x[1]), r: 3.4, fill: colorOf(c.comp[i]), stroke: C.paper, "stroke-width": 1.2 })));
  return g({}, ...out);
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 560;
  const c = compute(p.design, p.comparisons, p.lengthscale, p.seed);
  const parts: string[] = [];
  const gap = 18;
  const size = narrow ? Math.min(st.w - 24, 260) : Math.floor((st.w - 2 * gap - 16) / 3);
  const pos = (k: number): [number, number] => (narrow ? [(st.w - size) / 2, 22 + k * (size + 30)] : [8 + k * (size + gap), 22]);
  const titles = [L.truth, L.mean, L.sd];
  for (let k = 0; k < 3; k++) {
    const [x0, y0] = pos(k);
    parts.push(text(x0, y0 - 7, titles[k], { "font-size": TYPE.small, class: "fig-t-muted" }));
    if (k === 0) parts.push(heat(x0, y0, size, U, C.truth, true));
    else if (k === 1) parts.push(heat(x0, y0, size, c.post.mean, C.model, true), overlay(x0, y0, size, c));
    else parts.push(heat(x0, y0, size, c.sdRel, C.c7, false), overlay(x0, y0, size, c));
    if (k === 1) {
      // the posterior's best guess
      const [bx, by] = c.bestAt;
      parts.push(el("path", { d: star(x0 + bx * size, y0 + size - by * size, 7), fill: C.c4, stroke: C.paper, "stroke-width": 1.2 }));
    }
  }
  const [, yLast] = pos(2);
  const sy = (narrow ? yLast + size : 22 + size) + 22;
  parts.push(
    text(8, sy, tpl(L.status, { n: c.duels.length, m: c.xs.length, k: c.k }), { "font-size": TYPE.small, class: "fig-t-num fig-t-strong" }),
  );
  // the readout, wrapped at " · " to the available width
  const segs = [...tpl(L.agree, { o: pct(c.order), r: fixed(c.rho, 2) }).split(" · "), ...tpl(L.pickq, { b: fixed(c.best, 2), bs: fixed(c.bestShown, 2), mx: fixed(UMAX, 2) }).split(" · ")];
  const maxChars = Math.floor((st.w - 16) / 5.9);
  const lines: string[] = [];
  for (const sg of segs) {
    const last = lines[lines.length - 1];
    if (last !== undefined && vlen(last) + 3 + vlen(sg) <= maxChars) lines[lines.length - 1] = `${last} · ${sg}`;
    else lines.push(sg);
  }
  lines.forEach((ln, i) => parts.push(text(8, sy + 17 + 17 * i, ln, { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" })));
  return svg(st.w, sy + 9 + 17 * lines.length, describe(st), g({}, ...parts));
}

function star(cx: number, cy: number, r: number): string {
  let d = "";
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    const rr = i % 2 ? r * 0.45 : r;
    d += `${i ? "L" : "M"}${(cx + rr * Math.cos(a)).toFixed(1)},${(cy + rr * Math.sin(a)).toFixed(1)}`;
  }
  return d + "Z";
}

export default defineFigure({
  name: "pref-duels-2d",
  title: { en: "The preference model on a two-dimensional design space, with the comparison graph", zh: "二维设计空间上的偏好模型及其比较图" },
  labels,
  params,
  hint: { en: "Change the design and the lengthscale; New draw places new designs and asks again.", zh: "改变布局和长度尺度；“重新抽取”会放置新的设计并重新提问。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "New draw", zh: "重新抽取" }, primary: true, run: (p) => ({ ...p, seed: (p.seed % 999) + 1 }) },
  ],
});
