// The exploration dial. The loop runs on the book's running objective with
// the rule "posterior mean plus a weight times the posterior standard
// deviation", for a weight the reader chooses. The top panel shows one run
// after its whole budget: the hidden objective, the final posterior, and where
// the evaluations landed. The lower left panel traces the best value found
// after each evaluation. The lower right panel repeats the run from 32 random
// initial designs for every weight on a grid and plots the average gap between
// the true maximum and the best value found, so the reader sees the whole
// trade-off at once: too little exploration gets stuck on the wide bump, too
// much spreads the budget evenly and never refines the tall one.
//
// The GP settings mirror bo-loop (RBF kernel, signal variance 0.45, constant
// mean equal to the data average, lengthscale 0.08, two random initial points
// drawn the same way), so both figures show the same posterior for the same
// data.

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { fit, kernel, predict } from "./lib/gp.ts";
import { argmax } from "./lib/acq.ts";
import { memo, rng } from "./lib/random.ts";
import { running } from "./lib/objectives.ts";
import { band, clip, curve, frame, frameAxes, hitArea, legend, type Frame } from "./lib/plot.ts";
import { axis } from "./lib/axis.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "input x",
    f: "f(x)",
    truth: "hidden objective",
    mean: "final posterior mean",
    band: "95% band",
    init: "initial design",
    chosen: "chosen by the rule",
    best: "best found",
    runTitle: "one run, weight √β = {w}: best {b}, true max {m}",
    traceTitle: "best value so far, this run",
    traceX: "evaluation",
    max: "true max",
    sweepTitle: "average gap after {n} evaluations, {s} starts",
    sweepX: "exploration weight √β",
    found: "{k} of {s} starts within 0.05 of the max",
    describe: "With exploration weight {w}, this run's best value after {n} evaluations is {b}; the true maximum is {m}. Over {s} random starts the average gap to the maximum is {gap}, and {k} of {s} runs came within 0.05 of the maximum, which lies at x = {xm}.",
  },
  zh: {
    x: "输入 x",
    f: "f(x)",
    truth: "隐藏的目标函数",
    mean: "最终的后验均值",
    band: "95% 区间",
    init: "初始设计",
    chosen: "由规则选出",
    best: "找到的最优值",
    runTitle: "一次运行，权重 √β = {w}：最优值 {b}，真实最大值 {m}",
    traceTitle: "迄今最优值，本次运行",
    traceX: "评估次数",
    max: "真实最大值",
    sweepTitle: "{n} 次评估后的平均差距，{s} 个起点",
    sweepX: "探索权重 √β",
    found: "{s} 个起点中有 {k} 个与最大值相差不到 0.05",
    describe: "探索权重为 {w} 时，本次运行在 {n} 次评估后的最优值为 {b}；真实最大值为 {m}。在 {s} 个随机起点上，与最大值的平均差距为 {gap}，{s} 次运行中有 {k} 次与最大值相差不到 0.05；最大值位于 x = {xm}。",
  },
};

const params = {
  weight: {
    kind: "range", label: { en: "Exploration weight √β", zh: "探索权重 √β" }, min: 0, max: 8, default: 0, step: 0.25,
    marks: [{ value: 0, label: { en: "greedy", zh: "贪心" } }, { value: 8, label: { en: "explore", zh: "探索" } }],
  },
  seed: { kind: "range", label: { en: "Start", zh: "起点" }, min: 1, max: 32, default: 12, step: 1, control: false },
} as const;

type P = { weight: number; seed: number };

const XS = grid(0, 1, 201);
const SF2 = 0.45;
const LS = 0.08;
const INIT = 2;
const STEPS = 10;
const BUDGET = INIT + STEPS;
const SEEDS = 32;
const WEIGHTS = grid(0, 8, 33); // the slider's own steps
const FOUND = 0.05; // a run counts as a success if its gap is below this
const KERNEL = kernel("rbf", LS, SF2);

interface Run { obs: Array<[number, number]>; trace: number[]; mean: number[]; sd: number[]; best: number }

function posterior(obs: Array<[number, number]>) {
  const ym = obs.reduce((a, o) => a + o[1], 0) / obs.length;
  const gp = fit(KERNEL, obs.map((o) => o[0]), obs.map((o) => o[1]), 1e-5, ym);
  const post = predict(gp, XS);
  return { mean: post.mean, sd: post.var.map(Math.sqrt) };
}

// One run of the loop: a pure function of the weight and the seed.
const runOnce = memo((w: number, seed: number): Run => {
  const r = rng(seed);
  const obs: Array<[number, number]> = [];
  for (let i = 0; i < INIT; i++) { const x = 0.05 + 0.9 * r(); obs.push([x, running.f(x)]); }
  for (let s = 0; s < STEPS; s++) {
    const { mean, sd } = posterior(obs);
    const x = XS[argmax(mean.map((m, i) => m + w * sd[i]))];
    obs.push([x, running.f(x)]);
  }
  const trace: number[] = [];
  let b = -Infinity;
  for (const o of obs) { b = Math.max(b, o[1]); trace.push(b); }
  const { mean, sd } = posterior(obs);
  return { obs, trace, mean, sd, best: b };
}, 64);

// The sweep over all weights and starts; it depends on nothing the reader
// changes, so it is computed once.
let sweepCache: { gap: number[]; found: number[] } | undefined;
function sweep() {
  if (sweepCache) return sweepCache;
  const gap: number[] = [], found: number[] = [];
  for (const w of WEIGHTS) {
    let s = 0, k = 0;
    for (let seed = 1; seed <= SEEDS; seed++) {
      const r = runOnceRaw(w, seed);
      s += running.max - r;
      if (running.max - r < FOUND) k++;
    }
    gap.push(s / SEEDS);
    found.push(k);
  }
  sweepCache = { gap, found };
  return sweepCache;
}

// The sweep needs only the final best value, so it skips the memo (which
// would evict the runs the reader is looking at).
function runOnceRaw(w: number, seed: number): number {
  const r = rng(seed);
  const obs: Array<[number, number]> = [];
  for (let i = 0; i < INIT; i++) { const x = 0.05 + 0.9 * r(); obs.push([x, running.f(x)]); }
  for (let s = 0; s < STEPS; s++) {
    const { mean, sd } = posterior(obs);
    const x = XS[argmax(mean.map((m, i) => m + w * sd[i]))];
    obs.push([x, running.f(x)]);
  }
  return Math.max(...obs.map((o) => o[1]));
}

const wIndex = (w: number) => Math.max(0, Math.min(WEIGHTS.length - 1, Math.round(w / 0.25)));

function describe(st: State<P>): string {
  const p = st.p;
  const run = runOnce(p.weight, p.seed);
  const sw = sweep();
  const i = wIndex(p.weight);
  return tpl(labels[st.lang ?? "en"].describe, {
    w: fixed(p.weight, 2), n: BUDGET, b: fixed(run.best, 2), m: fixed(running.max, 2), s: SEEDS,
    gap: fixed(sw.gap[i], 3), k: sw.found[i], xm: fixed(running.argmax, 2),
  });
}

// A frame for a sub-panel placed anywhere in the SVG.
function panel(left: number, right: number, top: number, height: number, xd: [number, number], yd: [number, number], w: number): Frame {
  return { x: linear(xd, [left, right]), y: linear(yd, [top + height, top]), left, right, top, bottom: top + height, w };
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const run = runOnce(p.weight, p.seed);
  const sw = sweep();
  const wi = wIndex(p.weight);

  const lg = legend(narrow ? 8 : 40, 14, st.w - 16, [
    { kind: "dash", color: C.truth, label: L.truth },
    { kind: "line", color: C.model, label: L.mean },
    { kind: "dot", color: C.ink, label: L.chosen },
    { kind: "band", color: C.band, label: L.band },
  ]);
  const parts: string[] = [lg.svg];

  // Top panel: one run after its budget.
  const top = 14 + lg.height + 22;
  const f = frame({ w: st.w, top, height: narrow ? 160 : 190, yDomain: running.range, yTitle: L.f });
  const cid = `${st.uid}-c`;
  const lo = run.mean.map((m, i) => m - 1.96 * run.sd[i]);
  const hi = run.mean.map((m, i) => m + 1.96 * run.sd[i]);
  const bestIdx = run.obs.findIndex((o) => o[1] === run.best);
  parts.push(
    el("defs", {}, clip(f, cid)),
    text(f.left, top - 8, tpl(L.runTitle, { w: fixed(p.weight, 2), b: fixed(run.best, 2), m: fixed(running.max, 2) }), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }),
    frameAxes(f, { yTitle: L.f, xTitle: L.x, yCount: 4 }),
    g({ "clip-path": `url(#${cid})` },
      band(f, XS, lo, hi),
      curve(f, XS, XS.map(running.f), { stroke: C.truth, "stroke-dasharray": "5 4", "stroke-width": 1.6 }),
      curve(f, XS, run.mean),
    ),
    // Evaluations: the initial design hollow, the rule's choices filled.
    g({}, ...run.obs.map(([x, y], i) => el("circle", {
      cx: f.x(x), cy: f.y(y), r: 4.2,
      fill: i < INIT ? C.paper : C.ink, stroke: C.ink, "stroke-width": i < INIT ? 1.6 : 1,
    }))),
    el("circle", { cx: f.x(run.obs[bestIdx][0]), cy: f.y(run.best), r: 8.5, fill: "none", stroke: C.acq, "stroke-width": 2 }),
  );

  // Lower panels: side by side when wide, stacked when narrow.
  const pH = narrow ? 104 : 116;
  const tTop = f.bottom + (narrow ? 66 : 70);
  const tLeft = narrow ? 40 : 52;
  const tRight = narrow ? st.w - 10 : Math.floor(st.w / 2) - 14;
  const sTop = narrow ? tTop + pH + 66 : tTop;
  const sLeft = narrow ? 40 : Math.floor(st.w / 2) + 44;
  const sRight = st.w - 10;

  // Best value so far.
  const ft = panel(tLeft, tRight, tTop, pH, [1, BUDGET], [-0.5, 1], st.w);
  const steps = run.trace.map((b, i) => `${i === 0 ? "M" : "L"}${ft.x(i + 1)},${ft.y(i === 0 ? b : run.trace[i - 1])}L${ft.x(i + 1)},${ft.y(b)}`).join("");
  parts.push(
    text(tLeft, tTop - 10, L.traceTitle, { "font-size": TYPE.small, class: "fig-t-muted" }),
    axis({ scale: ft.y, orient: "left", at: ft.left, span: [ft.left, ft.right], count: 3 }),
    axis({ scale: ft.x, orient: "bottom", at: ft.bottom, span: [ft.top, ft.bottom], title: L.traceX, ticks: narrow ? [1, 4, 8, 12] : [1, 2, 4, 6, 8, 10, 12], grid: false }),
    el("rect", { x: ft.x(1), y: ft.top, width: ft.x(INIT + 0.5) - ft.x(1), height: pH, fill: C.grid, opacity: 0.6 }),
    text(ft.x(1) + 3, ft.bottom - 6, L.init, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
    el("line", { x1: ft.left, x2: ft.right, y1: ft.y(running.max), y2: ft.y(running.max), stroke: C.truth, "stroke-width": 1.4, "stroke-dasharray": "5 4" }),
    text(ft.right - 2, ft.y(running.max) - 5, L.max, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo" }),
    el("path", { d: steps, fill: "none", stroke: C.ink, "stroke-width": 1.8 }),
    g({}, ...run.trace.map((b, i) => el("circle", { cx: ft.x(i + 1), cy: ft.y(b), r: 2.6, fill: C.ink }))),
  );

  // The sweep over weights.
  const gmax = Math.max(...sw.gap) * 1.15;
  const fs = panel(sLeft, sRight, sTop, pH, [0, 8], [0, gmax], st.w);
  const sweepPath = WEIGHTS.map((w, i) => `${i ? "L" : "M"}${fs.x(w)},${fs.y(sw.gap[i])}`).join("");
  const cx = fs.x(WEIGHTS[wi]);
  const cy = fs.y(sw.gap[wi]);
  parts.push(
    text(sLeft, sTop - 10, tpl(L.sweepTitle, { n: BUDGET, s: SEEDS }), { "font-size": TYPE.small, class: "fig-t-muted" }),
    axis({ scale: fs.y, orient: "left", at: fs.left, span: [fs.left, fs.right], count: 3 }),
    axis({ scale: fs.x, orient: "bottom", at: fs.bottom, span: [fs.top, fs.bottom], title: L.sweepX, ticks: [0, 2, 4, 6, 8], grid: false }),
    hitArea(fs, "sweep"),
    el("path", { d: sweepPath, fill: "none", stroke: C.model, "stroke-width": 2, "stroke-linejoin": "round" }),
    el("line", { x1: cx, x2: cx, y1: fs.top, y2: fs.bottom, stroke: C.acq, "stroke-width": 1.5, "stroke-dasharray": "4 3" }),
    el("circle", { cx, cy, r: 4.5, fill: C.acq, stroke: C.paper, "stroke-width": 1.5 }),
    text(fs.right - 2, fs.top + 10, tpl(L.found, { k: sw.found[wi], s: SEEDS }), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo fig-t-num" }),
  );

  const H = sTop + pH + 44;
  return svg(st.w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "loop-tradeoff",
  title: { en: "How much should the loop explore?", zh: "循环应当探索多少？" },
  labels,
  params,
  hint: { en: "Move the exploration weight, or click the lower right plot to pick one.", zh: "移动探索权重，或点击右下方的图来选择一个权重。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click" || e.target !== "sweep" || !e.data) return null;
    return { ...p, weight: WEIGHTS[wIndex(e.data.x)] };
  },
  actions: [
    { label: { en: "Another start", zh: "换一个起点" }, run: (p) => ({ ...p, seed: (p.seed % SEEDS) + 1 }) },
  ],
});
