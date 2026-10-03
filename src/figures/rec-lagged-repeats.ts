// Repeated pairs at different lags, as a diagnostic (a model, not data). A
// session shows some pairs twice, separated by a lag of L answers, and counts
// how often the person gives the same answer both times. The expected rate is
// computed from a simple model of one answer:
//
//   each pair has a utility difference delta ~ N(0, 1) across pairs
//   an answer picks A with probability Phi(delta / (sqrt(2) sigma))
//   drift: by the second showing, delta has moved by N(0, tau^2 L / 10)
//   induced change: the option chosen the first time gains c exp(-L / 10)
//   incomplete answers: with probability rho the answer is a coin flip
//
// so the probability of the same answer twice is an integral over delta,
// done here on a grid. Below the curve, the figure draws what an experiment
// with `people` participants and `repeats` repeated pairs per lag and person
// would measure: binomial counts with 95% Wilson intervals, from a seed.

import { defineFigure, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { log as logScale, linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { rng } from "./lib/random.ts";
import { Phi, phi } from "./lib/stats.ts";
import { legend } from "./lib/plot.ts";
import { tpl } from "./lib/format.ts";

const labels = {
  en: {
    y: "P(same answer twice)",
    x: "lag between the two showings (answers)",
    xShort: "lag (answers)",
    model: "expected under these settings",
    ref: "noise and incomplete answers only",
    data: "a simulated experiment, 95% interval",
    chance: "chance",
    readout: "{n} people × {k} repeats = {m} repeated pairs per lag",
    describe: "Simulation of a model, not data. The probability of giving the same answer to a repeated pair is {a}% at a lag of 2 answers and {b}% at a lag of 40; with noise and incomplete answers only it would be {r}% at every lag. A simulated experiment with {m} repeated pairs per lag measures it to within about ±{w} percentage points.",
  },
  zh: {
    y: "两次回答相同的概率",
    x: "两次呈现之间的间隔（回答数）",
    xShort: "间隔（回答数）",
    model: "当前设置下的期望值",
    ref: "仅有噪声与不完备回答",
    data: "一次模拟实验，95% 区间",
    chance: "随机水平",
    readout: "{n} 人 × {k} 次重复 = 每个间隔 {m} 个重复配对",
    describe: "这是模型的模拟，不是数据。在间隔为 2 个回答时，对一个重复配对给出相同回答的概率为 {a}%，间隔为 40 时为 {b}%；若只有噪声与不完备回答，每个间隔上都会是 {r}%。一次每个间隔有 {m} 个重复配对的模拟实验，测得的值误差约在 ±{w} 个百分点以内。",
  },
};

const params = {
  noise: { kind: "range", label: { en: "Noise σ", zh: "噪声 σ" }, min: 0.1, max: 1.5, default: 0.5, step: 0.05 },
  drift: { kind: "range", label: { en: "Drift", zh: "漂移" }, min: 0, max: 1.5, default: 1, step: 0.05 },
  induced: { kind: "range", label: { en: "Induced", zh: "引起的改变" }, min: 0, max: 1.5, default: 0, step: 0.05 },
  incomplete: { kind: "range", label: { en: "Incomplete", zh: "不完备" }, min: 0, max: 0.6, default: 0, step: 0.05 },
  people: { kind: "range", label: { en: "People", zh: "人数" }, min: 1, max: 60, default: 30, step: 1 },
  repeats: { kind: "range", label: { en: "Repeats", zh: "重复次数" }, min: 1, max: 6, default: 4, step: 1 },
  seed: { kind: "range", label: { en: "Seed", zh: "种子" }, min: 1, max: 9999, default: 5, step: 1, control: false },
} as const;

type P = { noise: number; drift: number; induced: number; incomplete: number; people: number; repeats: number; seed: number };

const LAGS = [2, 5, 10, 20, 40];
const TAU = 10; // time constant of induced change, and the unit of drift, in answers

// The grid over delta ~ N(0, 1) for the expectation.
const DG = Array.from({ length: 161 }, (_, i) => -5 + (10 * i) / 160);
const DW = (() => { const w = DG.map((d) => phi(d)); const s = w.reduce((a, b) => a + b, 0); return w.map((v) => v / s); })();

function agreement(p: { noise: number; drift: number; induced: number; incomplete: number }, lag: number): number {
  const s0 = Math.SQRT2 * p.noise;
  const sL = Math.sqrt(s0 * s0 + (p.drift * p.drift * lag) / TAU);
  const b = p.induced * Math.exp(-lag / TAU);
  const rho = p.incomplete;
  let total = 0;
  for (let i = 0; i < DG.length; i++) {
    const d = DG[i];
    const pA = (1 - rho) * Phi(d / s0) + rho / 2; // first answer: A
    const againA = (1 - rho) * Phi((d + b) / sL) + rho / 2; // second answer: A again
    const againB = (1 - rho) * Phi((-d + b) / sL) + rho / 2; // second answer: B again
    total += DW[i] * (pA * againA + (1 - pA) * againB);
  }
  return total;
}

function wilson(x: number, m: number): [number, number] {
  const z = 1.96, ph = x / m;
  const den = 1 + (z * z) / m;
  const c = (ph + (z * z) / (2 * m)) / den;
  const h = (z * Math.sqrt((ph * (1 - ph)) / m + (z * z) / (4 * m * m))) / den;
  return [Math.max(0, c - h), Math.min(1, c + h)];
}

function sample(p: P) {
  const m = p.people * p.repeats;
  return LAGS.map((lag, j) => {
    const r = rng(p.seed * 131 + j * 7 + 1);
    const a = agreement(p, lag);
    let x = 0;
    for (let i = 0; i < m; i++) if (r() < a) x++;
    return { lag, est: x / m, ci: wilson(x, m) };
  });
}

function describe(st: State<P>): string {
  const p = st.p;
  const ref = { ...p, drift: 0, induced: 0 };
  const s = sample(p);
  const w = Math.max(...s.map((q) => (q.ci[1] - q.ci[0]) / 2));
  return tpl(labels[st.lang ?? "en"].describe, {
    a: Math.round(agreement(p, 2) * 100), b: Math.round(agreement(p, 40) * 100), r: Math.round(agreement(ref, 10) * 100),
    m: p.people * p.repeats, w: Math.round(w * 100),
  });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const lg = legend(narrow ? 8 : 40, 14, st.w - 16, [
    { kind: "line", color: C.c5, label: L.model },
    { kind: "dash", color: C.ink3, label: L.ref },
    { kind: "dot", color: C.ink, label: L.data },
  ]);
  const top = 14 + lg.height + 10;
  const left = narrow ? 44 : 56, right = st.w - 14;
  const h = narrow ? 190 : 230;
  const x = logScale([1.6, 50], [left, right]);
  const y = linear([0.4, 1], [top + h, top]);
  const parts: string[] = [lg.svg];
  parts.push(axis({ scale: y, orient: "left", at: left, span: [left, right], title: L.y, count: 6, format: (v) => v.toFixed(1) }));
  parts.push(axis({ scale: x, orient: "bottom", at: top + h, span: [top, top + h], title: narrow ? L.xShort : L.x, ticks: LAGS, format: (v) => String(v) }));
  parts.push(el("line", { x1: left, x2: right, y1: y(0.5), y2: y(0.5), stroke: C.rule, "stroke-dasharray": "3 3" }));
  parts.push(text(left + 6, y(0.5) - 5, L.chance, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));
  const xs = Array.from({ length: 80 }, (_, i) => 1.6 * Math.pow(50 / 1.6, i / 79));
  const ref = { ...p, drift: 0, induced: 0 };
  parts.push(el("path", { d: linePath(xs.map((l) => [x(l), y(agreement(ref, l))])), fill: "none", stroke: C.ink3, "stroke-width": 1.6, "stroke-dasharray": "5 4" }));
  parts.push(el("path", { d: linePath(xs.map((l) => [x(l), y(agreement(p, l))])), fill: "none", stroke: C.c5, "stroke-width": 2.2 }));
  for (const s of sample(p)) {
    const cx = x(s.lag);
    parts.push(
      el("line", { x1: cx, x2: cx, y1: y(Math.max(0.4, s.ci[0])), y2: y(s.ci[1]), stroke: C.ink, "stroke-width": 1.5 }),
      el("line", { x1: cx - 4, x2: cx + 4, y1: y(Math.max(0.4, s.ci[0])), y2: y(Math.max(0.4, s.ci[0])), stroke: C.ink, "stroke-width": 1.5 }),
      el("line", { x1: cx - 4, x2: cx + 4, y1: y(s.ci[1]), y2: y(s.ci[1]), stroke: C.ink, "stroke-width": 1.5 }),
      el("circle", { cx, cy: y(Math.max(0.4, s.est)), r: 4.2, fill: C.ink, stroke: C.paper, "stroke-width": 1.6 }),
    );
  }
  const H = top + h + 64;
  parts.push(text(left, top + h + 54, tpl(L.readout, { n: p.people, k: p.repeats, m: p.people * p.repeats }), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  return svg(st.w, H, describe(st), g({}, ...parts));
}

export default defineFigure({
  name: "rec-lagged-repeats",
  title: { en: "Repeated pairs at different lags: what separates noise from drift (a simulation)", zh: "不同间隔下的重复配对：如何区分噪声与漂移（模拟）" },
  labels,
  params,
  hint: { en: "A model, not data. Move the sliders and watch the shape of the curve; change the number of people and repeats to see what an experiment of that size could detect.", zh: "这是模型，不是数据。移动滑块，观察曲线的形状；改变人数和重复次数，看看这种规模的实验能检测到什么。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [{ label: { en: "Rerun the experiment", zh: "重新运行实验" }, run: (p) => ({ ...p, seed: (Number(p.seed) % 9999) + 1 }) }],
});
