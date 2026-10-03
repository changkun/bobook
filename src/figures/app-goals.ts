// One model, four ways to choose the next evaluation, three ways to keep
// score. On the running objective (grid of 101 inputs, noise sd 0.05):
//   - uncertainty sampling evaluates where the posterior variance is largest
//     (active learning, experimental design);
//   - Bayesian optimization maximizes an upper confidence bound (√β = 2);
//   - a (1+1) evolution strategy mutates its current best point and keeps
//     the better of the two, adapting its step size by the one-fifth rule;
//   - random search draws inputs uniformly.
// The top panel shows where each rule evaluated in one run, up to the
// timeline position. The bottom panel shows a score averaged over 20 runs:
// the error of a Gaussian process fitted to the rule's evaluations, the
// simple regret, or the cumulative regret. Each rule wins on the score that
// matches its goal.

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { kernel } from "./lib/gp.ts";
import { memo, normal, rng } from "./lib/random.ts";
import { running } from "./lib/objectives.ts";
import type { Frame, Swatch } from "./lib/plot.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "input x",
    f: "f(x)",
    n: "evaluations",
    truth: "hidden objective",
    al: "uncertainty sampling",
    bo: "Bayesian optimization",
    es: "evolution strategy",
    rs: "random search",
    alShort: "uncert.",
    boShort: "BO",
    esShort: "evol.",
    rsShort: "random",
    rmse: "model error (RMSE)",
    simple: "simple regret",
    cumulative: "cumulative regret",
    avg: "average over {r} runs",
    where: "where each rule evaluated, run 1",
    key0: "No evaluations yet",
    key: "{n} evaluation{n:/s}",
    describe: "After {n} evaluations, averaged over {r} runs, the {m} is {a} for uncertainty sampling, {b} for Bayesian optimization, {e} for the evolution strategy, and {d} for random search.",
  },
  zh: {
    x: "输入 x",
    f: "f(x)",
    n: "评估次数",
    truth: "隐藏的目标函数",
    al: "不确定性采样",
    bo: "贝叶斯优化",
    es: "进化策略",
    rs: "随机搜索",
    alShort: "不确定性",
    boShort: "BO",
    esShort: "进化",
    rsShort: "随机",
    rmse: "模型误差（RMSE）",
    simple: "简单遗憾",
    cumulative: "累积遗憾",
    avg: "{r} 次运行的平均",
    where: "各规则的评估位置（第 1 次运行）",
    key0: "尚无评估",
    key: "{n} 次评估",
    describe: "经过 {n} 次评估，在 {r} 次运行上平均，{m}：不确定性采样为 {a}，贝叶斯优化为 {b}，进化策略为 {e}，随机搜索为 {d}。",
  },
};

const SCORES = [
  { value: "rmse", label: { en: "Model error", zh: "模型误差" } },
  { value: "simple", label: { en: "Simple regret", zh: "简单遗憾" } },
  { value: "cumulative", label: { en: "Cumulative regret", zh: "累积遗憾" } },
] as const;

const params = {
  score: { kind: "choice", label: { en: "Score", zh: "评分指标" }, options: SCORES, default: "rmse" },
  budget: { kind: "range", label: { en: "Budget", zh: "预算" }, min: 10, max: 40, default: 30, step: 1, control: false },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 1, step: 1, control: false },
} as const;

type P = { score: "rmse" | "simple" | "cumulative"; budget: number; seed: number };

const M = 101;
const XS = grid(0, 1, M);
const FX = XS.map(running.f);
const FMAX = Math.max(...FX);
const NOISE = 0.05, S2 = NOISE * NOISE;
const K = kernel("rbf", 0.08, 0.45);
const K0 = (() => {
  const a = new Float64Array(M * M);
  for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) a[i * M + j] = K(XS[i], XS[j]);
  return a;
})();
const MEAN0 = 0.2;
const RULES = ["al", "bo", "es", "rs"] as const;
type Rule = typeof RULES[number];
const COLOR: Record<Rule, string> = { al: C.c7, bo: C.acq, es: C.c6, rs: C.ink3 };
const RUNS = 20;

interface Trace { idx: number[]; rmse: number[]; simple: number[]; cumulative: number[] }

function runRule(rule: Rule, n: number, seed: number): Trace {
  const S = Float64Array.from(K0);
  const mu = new Float64Array(M).fill(MEAN0);
  const z = normal(rng(seed * 7919 + 17));
  const r = rng(seed * 104729 + 5 + RULES.indexOf(rule) * 13);
  const start = Math.floor(rng(seed * 31 + 1)() * M); // the same first point for every rule
  const idx: number[] = [], rmse: number[] = [], simple: number[] = [], cumulative: number[] = [];
  let bestF = -Infinity, cum = 0;
  // evolution strategy state
  let ex = start, ey = -Infinity, step = 0.15;
  const zs = normal(rng(seed * 2003 + 9));
  for (let t = 0; t < n; t++) {
    let i = start;
    if (t > 0) {
      if (rule === "al") { let bv = -1; for (let j = 0; j < M; j++) if (S[j * M + j] > bv) { bv = S[j * M + j]; i = j; } }
      else if (rule === "bo") { let bv = -Infinity; for (let j = 0; j < M; j++) { const v = mu[j] + 2 * Math.sqrt(Math.max(S[j * M + j], 0)); if (v > bv) { bv = v; i = j; } } }
      else if (rule === "rs") i = Math.floor(r() * M);
      else i = Math.max(0, Math.min(M - 1, Math.round(ex + (step * zs()) * (M - 1))));
    }
    const y = FX[i] + NOISE * z();
    if (rule === "es") {
      if (t === 0) { ey = y; } else if (y >= ey) { ex = i; ey = y; step = Math.min(0.5, step * 1.5); } else step = Math.max(0.01, step * 1.5 ** -0.25);
    }
    // the same Gaussian process is fitted to every rule's data, for scoring
    const col = new Float64Array(M);
    for (let j = 0; j < M; j++) col[j] = S[j * M + i];
    const d = col[i] + S2;
    const rr = (y - mu[i]) / d;
    for (let j = 0; j < M; j++) mu[j] += col[j] * rr;
    for (let a = 0; a < M; a++) { const ca = col[a] / d; if (ca !== 0) for (let b = 0; b < M; b++) S[a * M + b] -= ca * col[b]; }
    idx.push(i);
    bestF = Math.max(bestF, FX[i]);
    cum += FMAX - FX[i];
    let se = 0;
    for (let j = 0; j < M; j++) se += (mu[j] - FX[j]) ** 2;
    rmse.push(Math.sqrt(se / M));
    simple.push(FMAX - bestF);
    cumulative.push(cum);
  }
  return { idx, rmse, simple, cumulative };
}

interface All { first: Record<Rule, Trace>; mean: Record<Rule, Record<"rmse" | "simple" | "cumulative", number[]>> }

const compute = memo((n: number, seed: number): All => {
  const first = {} as Record<Rule, Trace>;
  const mean = {} as All["mean"];
  for (const rule of RULES) {
    const acc = { rmse: new Array<number>(n).fill(0), simple: new Array<number>(n).fill(0), cumulative: new Array<number>(n).fill(0) };
    for (let k = 0; k < RUNS; k++) {
      const tr = runRule(rule, n, seed * 100 + k);
      if (k === 0) first[rule] = tr;
      for (let t = 0; t < n; t++) { acc.rmse[t] += tr.rmse[t] / RUNS; acc.simple[t] += tr.simple[t] / RUNS; acc.cumulative[t] += tr.cumulative[t] / RUNS; }
    }
    mean[rule] = acc;
  }
  return { first, mean };
}, 6);

function legendRow(x: number, y: number, w: number, items: Swatch[]): { svg: string; height: number } {
  const parts: string[] = [];
  let cx = x, cy = y;
  for (const it of items) {
    const tw = labelWidth(it.label, 6.4) + 40;
    if (cx + tw > x + w && cx > x) { cx = x; cy += 18; }
    parts.push(
      el("line", { x1: cx, x2: cx + 18, y1: cy - 1, y2: cy - 1, stroke: it.color, "stroke-width": 2, "stroke-dasharray": it.kind === "dash" ? "4 3" : undefined }),
      text(cx + 24, cy + 3, it.label, { "font-size": TYPE.small, class: "fig-t-muted" }),
    );
    cx += tw;
  }
  return { svg: g({}, ...parts), height: cy - y + 18 };
}

function describe(st: State<P>): string {
  const p = st.p;
  const all = compute(p.budget, p.seed);
  const n = Math.max(1, Math.min(p.budget, Math.round(st.t)));
  const v = (r: Rule) => fixed(all.mean[r][p.score][n - 1], p.score === "cumulative" ? 1 : 3);
  const L = labels[st.lang ?? "en"];
  return tpl(L.describe, { n, r: RUNS, m: L[p.score], a: v("al"), b: v("bo"), e: v("es"), d: v("rs") });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const all = compute(p.budget, p.seed);
  const n = Math.max(0, Math.min(p.budget, Math.round(st.t)));
  const parts: string[] = [];
  const lg = legendRow(narrow ? 8 : 40, 14, st.w - 16, [
    { kind: "dash", color: C.truth, label: L.truth },
    ...RULES.map((r) => ({ kind: "line" as const, color: COLOR[r], label: L[r] })),
  ]);
  parts.push(lg.svg);
  const fl = narrow ? 52 : 136;
  const right = st.w - 14;
  // top: the objective and four rows of evaluation ticks
  const top = 14 + lg.height + 4;
  const fH = narrow ? 80 : 100;
  const fx = linear([0, 1], [fl, right]);
  const fy = linear(running.range, [top + fH, top]);
  parts.push(el("line", { x1: fl, x2: right, y1: top + fH, y2: top + fH, stroke: C.rule }));
  parts.push(el("path", { d: XS.map((x, i) => `${i ? "L" : "M"}${fx(x).toFixed(1)},${fy(FX[i]).toFixed(1)}`).join(""), fill: "none", stroke: C.truth, "stroke-width": 1.8, "stroke-dasharray": "5 4" }));
  parts.push(text(right, top + 10, L.where, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo" }));
  const rowH = 15;
  RULES.forEach((r, k) => {
    const y = top + fH + 6 + k * rowH;
    parts.push(text(fl - 6, y + 9, narrow ? L[`${r}Short`] : L[r], { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted" }));
    const seen = new Map<number, number>();
    for (const i of all.first[r].idx.slice(0, n)) seen.set(i, (seen.get(i) ?? 0) + 1);
    for (const [i, c] of seen) parts.push(el("line", { x1: fx(XS[i]), x2: fx(XS[i]), y1: y + 1, y2: y + Math.min(12, 5 + 2.5 * Math.log2(c + 1)), stroke: COLOR[r], "stroke-width": 1.6 }));
    const last = all.first[r].idx[n - 1];
    if (n > 0 && last !== undefined) parts.push(el("circle", { cx: fx(XS[last]), cy: y + 3, r: 2.6, fill: COLOR[r] }));
  });
  // bottom: the chosen score against evaluations
  const bTop = top + fH + 6 + RULES.length * rowH + (narrow ? 24 : 28);
  const bH = narrow ? 150 : 180;
  const series = RULES.map((r) => all.mean[r][p.score]);
  const ymax = Math.max(...series.flat()) * 1.08;
  const bx = linear([1, p.budget], [fl, right]);
  const by = linear([0, ymax], [bTop + bH, bTop]);
  const bf: Frame = { x: bx, y: by, left: fl, right, top: bTop, bottom: bTop + bH, w: st.w };
  parts.push(
    axis({ scale: by, orient: "left", at: bf.left, span: [bf.left, bf.right], count: 4, title: narrow ? undefined : L[p.score] }),
    axis({ scale: bx, orient: "bottom", at: bf.bottom, span: [bf.top, bf.bottom], title: L.n, count: narrow ? 4 : 6 }),
  );
  if (narrow) parts.push(text(bf.left + 4, bf.top + 10, L[p.score], { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));
  parts.push(text(bf.right - 4, bf.top + 10, tpl(L.avg, { r: RUNS }), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo" }));
  RULES.forEach((r, k) => {
    const v = series[k];
    parts.push(el("path", { d: v.map((y, i) => `${i ? "L" : "M"}${bx(i + 1).toFixed(1)},${by(y).toFixed(1)}`).join(""), fill: "none", stroke: COLOR[r], "stroke-width": 1.2, opacity: 0.35 }));
    if (n > 0) {
      parts.push(el("path", { d: v.slice(0, n).map((y, i) => `${i ? "L" : "M"}${bx(i + 1).toFixed(1)},${by(y).toFixed(1)}`).join(""), fill: "none", stroke: COLOR[r], "stroke-width": 2.2, "stroke-linejoin": "round" }));
      parts.push(el("circle", { cx: bx(n), cy: by(v[n - 1]), r: 3.2, fill: COLOR[r], stroke: C.paper, "stroke-width": 1.2 }));
    }
  });
  const H = bf.bottom + 42;
  return svg(st.w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "app-goals",
  title: { en: "One model, four ways to choose evaluations, three ways to keep score", zh: "一个模型，四种选择评估的方式，三种评分方式" },
  labels,
  params,
  timeline: {
    rate: 4,
    discrete: true,
    duration: (p) => p.budget,
    keyframes: (p, lang) => Array.from({ length: p.budget + 1 }, (_, i) => ({ t: i, label: i === 0 ? labels[lang].key0 : tpl(labels[lang].key, { n: i }) })),
    poster: (p) => p.budget,
  },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "New runs", zh: "重新运行" }, run: (p) => ({ ...p, seed: (p.seed % 9999) + 1 }) },
  ],
});
