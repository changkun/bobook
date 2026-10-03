// Three ways to spend the same evaluation budget on a function we cannot see:
// a grid, uniform random points, and Bayesian optimization with expected
// improvement. The left panel shows where each method has evaluated (1-D: the
// running objective with three rows of ticks; 2-D: a Branin landscape with
// three kinds of markers). The right panel shows the best value each method
// has found against the number of evaluations, with the true maximum as a
// line. The timeline is the budget; scrub it to watch the race.

import { defineFigure, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { fit, kernel, predict } from "./lib/gp.ts";
import { argmax, ei } from "./lib/acq.ts";
import { memo, rng } from "./lib/random.ts";
import { branin01, running } from "./lib/objectives.ts";
import { curve, frame, frameAxes, legend } from "./lib/plot.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "input x",
    f: "f(x)",
    best: "best value found",
    evals: "evaluations",
    max: "true maximum",
    grid: "grid",
    random: "random",
    bo: "Bayesian optimization",
    x1: "x₁",
    x2: "x₂",
    key: "{n} evaluation{n:/s}",
    describe: "After {n} evaluations in {d} dimension{d:/s}, the best value found is {g} by grid search, {r} by random search, and {b} by Bayesian optimization; the true maximum is {m}.",
  },
  zh: {
    x: "输入 x",
    f: "f(x)",
    best: "已找到的最优值",
    evals: "评估次数",
    max: "真实最大值",
    grid: "网格",
    random: "随机",
    bo: "贝叶斯优化",
    x1: "x₁",
    x2: "x₂",
    key: "{n} 次评估",
    describe: "在 {d} 维空间中进行 {n} 次评估后，网格搜索找到的最优值为 {g}，随机搜索为 {r}，贝叶斯优化为 {b}；真实最大值为 {m}。",
  },
};

const params = {
  dims: { kind: "choice", label: { en: "Dimensions", zh: "维度" }, options: [{ value: 1, label: { en: "1-D", zh: "一维" } }, { value: 2, label: { en: "2-D", zh: "二维" } }], default: 1 },
  budget: { kind: "range", label: { en: "Budget", zh: "预算" }, min: 9, max: 36, default: 16, step: 1, control: false },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 12, step: 1, control: false },
} as const;

type P = { dims: number; budget: number; seed: number };
type Pt = number[];

const F = (d: number) => (d === 1 ? (x: Pt) => running.f(x[0]) : (x: Pt) => branin01(x[0], x[1]));

function trueMax(d: number): number {
  if (d === 1) return running.max;
  let m = -Infinity;
  for (const u of grid(0, 1, 201)) for (const v of grid(0, 1, 201)) m = Math.max(m, branin01(u, v));
  return m;
}
const TRUE_MAX = [0, trueMax(1), trueMax(2)];

// Grid: for a budget n, the evaluated points are the order in which a person
// would fill a grid: a full grid of the largest size that fits, row by row.
function gridPoints(d: number, n: number): Pt[] {
  if (d === 1) return grid(0, 1, n).map((x) => [x]);
  const k = Math.max(2, Math.floor(Math.sqrt(n)));
  const pts: Pt[] = [];
  for (const u of grid(0, 1, k)) for (const v of grid(0, 1, k)) pts.push([u, v]);
  return pts;
}

const runs = memo((d: number, n: number, seed: number) => {
  const f = F(d);
  // grid: a grid sized for the full budget, evaluated in order
  const gp = gridPoints(d, n);
  // random
  const r = rng(seed);
  const rp: Pt[] = Array.from({ length: n }, () => Array.from({ length: d }, () => r()));
  // BO starts from the same first two points as random search, then follows EI
  const bo: Pt[] = [rp[0].slice(), rp[1].slice()];
  const cand: Pt[] = d === 1 ? grid(0, 1, 201).map((x) => [x]) : (() => { const c: Pt[] = []; for (const u of grid(0, 1, 41)) for (const v of grid(0, 1, 41)) c.push([u, v]); return c; })();
  const k = kernel("matern52", d === 1 ? 0.12 : 0.22, 0.45);
  while (bo.length < n) {
    const ys = bo.map(f);
    const m0 = ys.reduce((a, b) => a + b, 0) / ys.length;
    const model = fit(k, d === 1 ? bo.map((p) => p[0]) : bo, ys, 1e-5, m0);
    const post = predict(model, d === 1 ? cand.map((p) => p[0]) : cand);
    const best = Math.max(...ys);
    const a = post.mean.map((m, i) => ei(m, Math.sqrt(post.var[i]), best, 0.005));
    bo.push(cand[argmax(a)]);
  }
  const bestSoFar = (pts: Pt[]) => { let b = -Infinity; return pts.map((p) => (b = Math.max(b, f(p)))); };
  return { gp, rp, bo, curves: { grid: bestSoFar(gp), random: bestSoFar(rp), bo: bestSoFar(bo) } };
}, 16);

function describe(st: State<P>): string {
  const p = st.p;
  const n = Math.max(1, Math.round(st.t));
  const R = runs(p.dims, p.budget, p.seed);
  const at = (c: number[]) => fixed(c[Math.min(n, c.length) - 1], 2);
  return tpl(labels[st.lang ?? "en"].describe, { n, d: p.dims, g: at(R.curves.grid), r: at(R.curves.random), b: at(R.curves.bo), m: fixed(TRUE_MAX[p.dims], 2) });
}

const COLORS = { grid: C.c4, random: C.c7, bo: C.acq };

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 560;
  const n = Math.max(1, Math.min(p.budget, Math.round(st.t)));
  const R = runs(p.dims, p.budget, p.seed);
  const parts: string[] = [];
  const lg = legend(narrow ? 8 : 40, 14, st.w - 16, [
    { kind: "dot", color: COLORS.grid, label: L.grid },
    { kind: "dot", color: COLORS.random, label: L.random },
    { kind: "dot", color: COLORS.bo, label: L.bo },
    ...(p.dims === 1 ? [{ kind: "dash" as const, color: C.truth, label: "f" }] : []),
  ]);
  parts.push(lg.svg);
  const top = 14 + lg.height + 10;
  const leftW = narrow ? st.w : Math.floor(st.w * 0.56);
  const H0 = narrow ? 210 : 230;
  if (p.dims === 1) {
    const f = frame({ w: leftW, top: top + 40, height: H0 - 40, yDomain: [-0.7, 1.1], yTitle: L.f });
    parts.push(frameAxes(f, { yTitle: L.f, xTitle: L.x, yCount: 3 }));
    const xs = grid(0, 1, 201);
    parts.push(curve(f, xs, xs.map(running.f), { stroke: C.truth, "stroke-dasharray": "5 4", "stroke-width": 1.6 }));
    const rows: Array<[keyof typeof COLORS, Pt[]]> = [["grid", R.gp], ["random", R.rp], ["bo", R.bo]];
    rows.forEach(([name, pts], ri) => {
      const y = top + 8 + ri * 12;
      parts.push(el("line", { x1: f.left, x2: f.right, y1: y, y2: y, stroke: C.grid }));
      pts.slice(0, n).forEach((q, i) => {
        parts.push(el("line", { x1: f.x(q[0]), x2: f.x(q[0]), y1: y - 4, y2: y + 4, stroke: COLORS[name], "stroke-width": i === n - 1 ? 2.6 : 1.6 }));
      });
    });
    // the best point each method has found, on the curve
    for (const [name, pts] of rows) {
      const seen = pts.slice(0, n);
      const b = seen.reduce((a, q) => (running.f(q[0]) > running.f(a[0]) ? q : a), seen[0]);
      parts.push(el("circle", { cx: f.x(b[0]), cy: f.y(running.f(b[0])), r: 4.5, fill: COLORS[name], stroke: C.paper, "stroke-width": 1.5 }));
    }
  } else {
    // 2-D: heatmap of the landscape with markers
    const size = Math.min(leftW - 60, H0);
    const x0 = narrow ? (st.w - size) / 2 + 10 : 52, y0 = top;
    const cells = 40;
    const cs = size / cells;
    let lo = Infinity, hi = -Infinity;
    const vals: number[][] = [];
    for (let i = 0; i < cells; i++) { vals.push([]); for (let j = 0; j < cells; j++) { const v = branin01((i + 0.5) / cells, (j + 0.5) / cells); vals[i].push(v); lo = Math.min(lo, v); hi = Math.max(hi, v); } }
    for (let i = 0; i < cells; i++) for (let j = 0; j < cells; j++) {
      const t = (vals[i][j] - lo) / (hi - lo);
      parts.push(el("rect", { x: x0 + i * cs, y: y0 + size - (j + 1) * cs, width: cs + 0.3, height: cs + 0.3, fill: C.truth, opacity: (0.05 + 0.55 * t ** 3).toFixed(3) }));
    }
    const sx = linear([0, 1], [x0, x0 + size]), sy = linear([0, 1], [y0 + size, y0]);
    parts.push(axis({ scale: sx, orient: "bottom", at: y0 + size, title: L.x1, count: 2, grid: false }), axis({ scale: sy, orient: "left", at: x0, title: L.x2, count: 2, grid: false }));
    const mark = (q: Pt, color: string, shape: "sq" | "di" | "ci", last: boolean) => {
      const cx = sx(q[0]), cy = sy(q[1]), r0 = last ? 5.5 : 4;
      if (shape === "sq") return el("rect", { x: cx - r0 * 0.8, y: cy - r0 * 0.8, width: r0 * 1.6, height: r0 * 1.6, fill: color, stroke: C.paper, "stroke-width": 1 });
      if (shape === "di") return el("path", { d: `M${cx},${cy - r0}L${cx + r0},${cy}L${cx},${cy + r0}L${cx - r0},${cy}Z`, fill: color, stroke: C.paper, "stroke-width": 1 });
      return el("circle", { cx, cy, r: r0 * 0.85, fill: color, stroke: C.paper, "stroke-width": 1 });
    };
    R.gp.slice(0, n).forEach((q, i) => parts.push(mark(q, COLORS.grid, "sq", i === n - 1)));
    R.rp.slice(0, n).forEach((q, i) => parts.push(mark(q, COLORS.random, "di", i === n - 1)));
    R.bo.slice(0, n).forEach((q, i) => parts.push(mark(q, COLORS.bo, "ci", i === n - 1)));
  }
  // best-so-far panel
  const bx0 = narrow ? 52 : leftW + 56, bx1 = st.w - 12;
  const by0 = narrow ? top + H0 + 40 : top, bh = narrow ? 150 : H0 - 10;
  const bxs = linear([1, p.budget], [bx0, bx1]);
  const lo = Math.min(...Object.values(R.curves).map((c) => c[0])) - 0.05;
  const bys = linear([Math.min(lo, TRUE_MAX[p.dims] - 1), TRUE_MAX[p.dims] + 0.1], [by0 + bh, by0]);
  parts.push(
    axis({ scale: bxs, orient: "bottom", at: by0 + bh, span: [by0, by0 + bh], title: L.evals, count: 4 }),
    axis({ scale: bys, orient: "left", at: bx0, span: [bx0, bx1], title: narrow ? "" : L.best, count: 3 }),
    el("line", { x1: bx0, x2: bx1, y1: bys(TRUE_MAX[p.dims]), y2: bys(TRUE_MAX[p.dims]), stroke: C.truth, "stroke-dasharray": "5 4" }),
    text(bx1 - 2, bys(TRUE_MAX[p.dims]) - 5, L.max, { "text-anchor": "end", "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
  );
  for (const name of ["grid", "random", "bo"] as const) {
    const c = R.curves[name].slice(0, n);
    parts.push(el("path", { d: linePath(c.map((v, i) => [bxs(i + 1), bys(v)])), fill: "none", stroke: COLORS[name], "stroke-width": name === "bo" ? 2.4 : 1.8 }));
  }
  if (narrow) parts.push(text(bx0 + 4, by0 + 10, L.best, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));
  parts.push(el("line", { x1: bxs(n), x2: bxs(n), y1: by0, y2: by0 + bh, stroke: C.ink3, "stroke-dasharray": "2 3" }));
  const H = Math.max(top + H0 + 46, by0 + bh + 40);
  return svg(st.w, H, describe(st), g({}, ...parts));
}

export default defineFigure({
  name: "intro-search-race",
  title: { en: "Grid search, random search, and Bayesian optimization on the same budget", zh: "相同预算下的网格搜索、随机搜索与贝叶斯优化" },
  labels,
  params,
  timeline: {
    rate: 3,
    discrete: true,
    duration: (p) => p.budget,
    keyframes: (p, lang) => [2, Math.round(p.budget / 2), p.budget].map((t) => ({ t, label: tpl(labels[lang].key, { n: t }) })),
    poster: (p) => p.budget,
  },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [{ label: { en: "New random run", zh: "新一轮随机运行" }, run: (p) => ({ ...p, seed: (p.seed % 9999) + 1 }) }],
});
