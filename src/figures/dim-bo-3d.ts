// Bayesian optimization in three dimensions, on the Hartmann-3 function.
// Left: the unit cube with every evaluated point, a drop line to the floor
// for depth, the newest point ringed, and the true maximizer as a star; drag
// the cube to rotate it. Right: three slices of the model through the best
// point found so far, one per input, each showing the posterior mean and
// band against the true function along that line. Below: the best value
// found against the number of evaluations. The timeline steps the loop.

import { defineFigure, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { fit, kernel, predict } from "./lib/gp.ts";
import { argmax, ei } from "./lib/acq.ts";
import { memo, rng } from "./lib/random.ts";
import { hartmann3 } from "./lib/objectives.ts";
import { candidates, cubeEdges, latinHypercube, project, type Pt } from "./lib/nd.ts";
import { band } from "./lib/plot.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    slice: "slice along x_{i}",
    best: "best value found",
    evals: "evaluations",
    max: "true max {m}",
    key0: "{n} initial points (Latin hypercube)",
    key: "evaluation {i}",
    readout: "{n} evaluations · best {b} at ({p})",
    describe: "Bayesian optimization on the three-dimensional Hartmann function after {n} evaluations. The best value found is {b} of a maximum of {m}; the best point is ({p}), and the true maximizer is ({q}).",
  },
  zh: {
    slice: "沿 x_{i} 的切片",
    best: "找到的最优值",
    evals: "评估次数",
    max: "真实最大值 {m}",
    key0: "{n} 个初始点（拉丁超立方）",
    key: "第 {i} 次评估",
    readout: "{n} 次评估 · 最优值 {b}，位于 ({p})",
    describe: "3 维 Hartmann 函数上经过 {n} 次评估的贝叶斯优化。找到的最优值为 {b}，而最大值为 {m}；最佳点为 ({p})，真正的最大值点为 ({q})。",
  },
};

const params = {
  yaw: { kind: "range", label: { en: "Rotate", zh: "旋转" }, min: -3.14, max: 3.14, default: 0.6, step: 0.01 },
  pitch: { kind: "range", label: { en: "Tilt", zh: "俯仰" }, min: -0.2, max: 1.2, default: 0.42, step: 0.01, control: false },
  lengthscale: { kind: "range", label: { en: "Lengthscale ℓ", zh: "长度尺度 ℓ" }, min: 0.08, max: 0.6, default: 0.22, scale: "log" },
  slices: { kind: "toggle", label: { en: "Show true function in slices", zh: "在切片中显示真实函数" }, default: true },
  init: { kind: "range", label: { en: "Initial points", zh: "初始点数" }, min: 2, max: 10, default: 5, step: 1, control: false },
  steps: { kind: "range", label: { en: "Evaluations", zh: "评估次数" }, min: 10, max: 40, default: 25, step: 1, control: false },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 4, step: 1, control: false },
} as const;

type P = { yaw: number; pitch: number; lengthscale: number; slices: boolean; init: number; steps: number; seed: number };

const f = hartmann3.f;
const SX = grid(0, 1, 61);

const run = memo((ls: number, init: number, steps: number, seed: number) => {
  const r = rng(seed);
  const X: Pt[] = latinHypercube(r, init, 3);
  const k = kernel("matern52", ls, 1);
  while (X.length < steps) {
    const ys = X.map(f);
    const m = ys.reduce((a, b) => a + b, 0) / ys.length;
    const sd = Math.sqrt(ys.reduce((a, b) => a + (b - m) ** 2, 0) / ys.length) || 1;
    const model = fit(k, X, ys.map((y) => (y - m) / sd), 1e-4);
    const best = Math.max(...ys);
    const order = X.map((_, i) => i).sort((a, b) => ys[b] - ys[a]);
    const cand = candidates(r, 3, 900, order.slice(0, 4).map((i) => X[i]), 300, 0.08);
    const post = predict(model, cand);
    const a = post.mean.map((mu, i) => ei(mu, Math.sqrt(post.var[i]), (best - m) / sd, 0.01));
    X.push(cand[argmax(a)]);
  }
  return X;
}, 12);

function state(p: P, t: number) {
  const X = run(p.lengthscale, p.init, p.steps, p.seed);
  const n = Math.min(X.length, p.init + Math.round(t));
  const pts = X.slice(0, n);
  const ys = pts.map(f);
  const bi = argmax(ys);
  return { X: pts, ys, best: pts[bi], bestY: ys[bi], n, all: X };
}

const fmtPt = (p: Pt) => p.map((v) => fixed(v, 2)).join(", ");

function describe(st: State<P>): string {
  const s = state(st.p, st.t);
  return tpl(labels[st.lang ?? "en"].describe, { n: s.n, b: fixed(s.bestY, 2), m: fixed(hartmann3.max, 2), p: fmtPt(s.best), q: fmtPt(hartmann3.argmax) });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 560;
  const s = state(p, st.t);
  const parts: string[] = [];
  // the cube
  const cubeW = narrow ? st.w : Math.min(330, st.w * 0.48);
  const cs = narrow ? Math.min(st.w * 0.5, 170) : cubeW * 0.5;
  const cx = cubeW / 2, cy = cs * 1.02 + 6;
  const cubeH = cy + cs * 0.95;
  parts.push(el("rect", { x: 0, y: 0, width: cubeW, height: cubeH, fill: "transparent", "data-fig-hit": "cube" }));
  parts.push(cubeEdges(p.yaw, p.pitch, cx, cy, cs, ["x₁", "x₂", "x₃"]));
  const yMin = 0, yMax = hartmann3.max;
  const items = s.X.map((q, i) => ({ q, y: s.ys[i], i, pr: project(q, p.yaw, p.pitch, cx, cy, cs), fl: project([q[0], q[1], 0], p.yaw, p.pitch, cx, cy, cs) }));
  items.sort((a, b) => a.pr.z - b.pr.z);
  for (const it of items) {
    const t = (it.y - yMin) / (yMax - yMin);
    const isNew = it.i === s.n - 1 && s.n > p.init;
    const isBest = it.q === s.best;
    parts.push(
      el("line", { x1: it.pr.x, y1: it.pr.y, x2: it.fl.x, y2: it.fl.y, stroke: C.ink3, "stroke-width": 0.7, opacity: 0.6 }),
      el("circle", { cx: it.fl.x, cy: it.fl.y, r: 1.6, fill: C.ink3, opacity: 0.6 }),
      el("circle", { cx: it.pr.x, cy: it.pr.y, r: 2.5 + 4.5 * Math.max(0, t), fill: isBest ? C.acq : C.model, opacity: (0.35 + 0.6 * Math.max(0, t)).toFixed(2), stroke: C.paper, "stroke-width": 1 }),
      isNew ? el("circle", { cx: it.pr.x, cy: it.pr.y, r: 10, fill: "none", stroke: C.acq, "stroke-width": 1.8 }) : "",
    );
  }
  const star = project(hartmann3.argmax, p.yaw, p.pitch, cx, cy, cs);
  parts.push(el("path", { d: starPath(star.x, star.y, 7), fill: C.truth, stroke: C.paper, "stroke-width": 1 }));
  parts.push(text(8, cubeH + 4, tpl(L.readout, { n: s.n, b: fixed(s.bestY, 2), p: fmtPt(s.best) }), { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
  // slices
  const k = kernel("matern52", p.lengthscale, 1);
  const ym = s.ys.reduce((a, b) => a + b, 0) / s.ys.length;
  const ysd = Math.sqrt(s.ys.reduce((a, b) => a + (b - ym) ** 2, 0) / s.ys.length) || 1;
  const model = fit(k, s.X, s.ys.map((y) => (y - ym) / ysd), 1e-4);
  const sx0 = narrow ? 40 : cubeW + 40, sx1 = st.w - 10;
  const sy0 = narrow ? cubeH + 24 : 14;
  const sh = narrow ? 70 : 62, gap = 22;
  for (let d = 0; d < 3; d++) {
    const top = sy0 + d * (sh + gap);
    const xs = linear([0, 1], [sx0, sx1]);
    const ys = linear([-0.4, 4.2], [top + sh, top]);
    const line = SX.map((v) => { const q = s.best.slice(); q[d] = v; return q; });
    const post = predict(model, line);
    const mu = post.mean.map((m) => m * ysd + ym);
    const sdv = post.var.map((v) => Math.sqrt(v) * ysd);
    const fr = { x: xs, y: ys, left: sx0, right: sx1, top, bottom: top + sh, w: st.w };
    parts.push(
      axis({ scale: ys, orient: "left", at: sx0, span: [sx0, sx1], count: 2 }),
      el("line", { x1: sx0, x2: sx1, y1: top + sh, y2: top + sh, stroke: C.rule }),
      band(fr, SX, mu.map((m, i) => m - 1.96 * sdv[i]), mu.map((m, i) => m + 1.96 * sdv[i])),
      p.slices ? el("path", { d: linePath(SX.map((v, i) => [xs(v), ys(f(line[i]))])), fill: "none", stroke: C.truth, "stroke-dasharray": "5 4", "stroke-width": 1.4 }) : "",
      el("path", { d: linePath(SX.map((v, i) => [xs(v), ys(mu[i])])), fill: "none", stroke: C.model, "stroke-width": 1.8 }),
      el("line", { x1: xs(s.best[d]), x2: xs(s.best[d]), y1: top, y2: top + sh, stroke: C.acq, "stroke-dasharray": "3 3" }),
      el("circle", { cx: xs(s.best[d]), cy: ys(s.bestY), r: 3.5, fill: C.acq }),
      text(sx0 + 6, top + 11, tpl(L.slice, { i: d + 1 }), { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
    );
    if (d === 2) parts.push(axis({ scale: xs, orient: "bottom", at: top + sh, count: 4, grid: false }));
  }
  // best so far
  const by0 = (narrow ? sy0 + 3 * (sh + gap) + 18 : Math.max(cubeH + 30, sy0 + 3 * (sh + gap) + 14));
  const bh = 70;
  const bx = linear([1, p.steps], [40, st.w - 10]);
  const byS = linear([0, 4], [by0 + bh, by0]);
  let b = -Infinity;
  const curve = s.ys.map((y) => (b = Math.max(b, y)));
  parts.push(
    axis({ scale: bx, orient: "bottom", at: by0 + bh, span: [by0, by0 + bh], title: L.evals, count: 5 }),
    axis({ scale: byS, orient: "left", at: 40, span: [40, st.w - 10], count: 2 }),
    el("line", { x1: 40, x2: st.w - 10, y1: byS(hartmann3.max), y2: byS(hartmann3.max), stroke: C.truth, "stroke-dasharray": "5 4" }),
    text(st.w - 12, byS(hartmann3.max) - 4, tpl(L.max, { m: fixed(hartmann3.max, 2) }), { "text-anchor": "end", "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
    el("path", { d: linePath(curve.map((v, i) => [bx(i + 1), byS(v)])), fill: "none", stroke: C.acq, "stroke-width": 2 }),
    text(46, by0 + 10, L.best, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
    el("rect", { x: bx(1), y: by0, width: Math.max(0, bx(p.init) - bx(1)), height: bh, fill: C.panel, opacity: 0.6 }),
  );
  return svg(st.w, by0 + bh + 40, describe(st), g({}, ...parts));
}

function starPath(cx: number, cy: number, r: number): string {
  let d = "";
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    const rr = i % 2 ? r * 0.45 : r;
    d += `${i ? "L" : "M"}${(cx + rr * Math.cos(a)).toFixed(1)},${(cy + rr * Math.sin(a)).toFixed(1)}`;
  }
  return d + "Z";
}

let drag: { x: number; y: number; yaw: number; pitch: number } | undefined;

export default defineFigure({
  name: "dim-bo-3d",
  title: { en: "Bayesian optimization in three dimensions", zh: "三维中的贝叶斯优化" },
  labels,
  params,
  hint: { en: "Drag the cube to rotate it. The star is the true maximizer; dot size and color show the value found.", zh: "拖动立方体可以旋转它。星号是真正的最大值点；圆点的大小和颜色表示找到的值。" },
  timeline: {
    rate: 2,
    discrete: true,
    duration: (p) => p.steps - p.init,
    keyframes: (p, lang) => [{ t: 0, label: tpl(labels[lang].key0, { n: p.init }) }, ...Array.from({ length: p.steps - p.init }, (_, i) => ({ t: i + 1, label: tpl(labels[lang].key, { i: p.init + i + 1 }) }))],
    poster: (p) => p.steps - p.init,
  },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase === "down" && e.target === "cube") { drag = { x: e.x, y: e.y, yaw: p.yaw, pitch: p.pitch }; return null; }
    if ((e.phase === "move" || e.phase === "up") && drag) {
      const yaw = Math.max(-3.14, Math.min(3.14, drag.yaw + (e.x - drag.x) * 0.012));
      const pitch = Math.max(-0.2, Math.min(1.2, drag.pitch + (e.y - drag.y) * 0.01));
      if (e.phase === "up") drag = undefined;
      return { ...p, yaw, pitch };
    }
    return null;
  },
  actions: [{ label: { en: "New run", zh: "新一轮运行" }, run: (p) => ({ ...p, seed: (p.seed % 9999) + 1 }) }],
});
