// Bayesian optimization in 6 to 50 dimensions, replayed from recorded runs
// (tools/figure-data/dim-bo-race.ts). The objective is the six-dimensional
// Hartmann function hidden among irrelevant inputs. Three panels:
//   1. regret (true maximum minus best value found) against evaluations for
//      random search and BO with a fixed-scale or a dimension-scaled
//      lengthscale prior, median over six seeds with the range shaded;
//   2. every point one run evaluated, in parallel coordinates: one vertical
//      axis per input, the six that matter first and shaded, the best points
//      highlighted, the true maximizer in the objective's color;
//   3. the lengthscales the model learned for each input at the end of that
//      run: short means "this input matters", long means "ignored".
// The timeline replays the evaluations.

import { defineFigure, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear, log } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { legend } from "./lib/plot.ts";
import { parallel } from "./lib/nd.ts";
import { hartmann6 } from "./lib/objectives.ts";
import { fixed, sig, tpl } from "./lib/format.ts";
import data from "./data/dim-bo-race.json" with { type: "json" };

interface Run { regret: number[][]; points: number[][]; values: number[]; lengthscales?: number[] }
const RUNS = data.runs as unknown as Record<string, Record<string, Run>>;
const META = data.meta;

const labels = {
  en: {
    regret: "regret (log scale)",
    evals: "evaluations",
    random: "random search",
    fixed: "BO, fixed-scale prior",
    scaled: "BO, dimension-scaled prior",
    pc: "points evaluated by {m}, run 1 · inputs 1 to 6 matter",
    pcShort: "evaluated points, run 1 · inputs 1 to 6 matter",
    lsShort: "learned lengthscales (log) · prior mode {mode}",
    ls: "learned lengthscale per input (log scale) · prior mode {mode}",
    noModel: "random search learns no model, so it has no lengthscales",
    relevant: "inputs that matter",
    describe: "In {d} dimensions with {s} acquisition search, after {n} evaluations the median regret is {r} for random search, {f} for BO with a fixed-scale prior, and {c} for BO with a dimension-scaled prior.",
    searchLocal: "global plus local",
    searchGlobal: "global-only",
    kfInit: "{n} initial points",
    kfEvals: "{n} evaluations",
  },
  zh: {
    regret: "遗憾（对数刻度）",
    evals: "评估次数",
    random: "随机搜索",
    fixed: "贝叶斯优化（固定先验）",
    scaled: "贝叶斯优化（缩放先验）",
    pc: "评估过的点：{m}，第 1 次运行 · 输入 1 至 6 是重要的",
    pcShort: "评估过的点，第 1 次运行 · 输入 1 至 6 是重要的",
    lsShort: "学到的长度尺度（对数）· 先验众数 {mode}",
    ls: "每个输入学到的长度尺度（对数刻度）· 先验众数 {mode}",
    noModel: "随机搜索不学习模型，因此没有长度尺度",
    relevant: "重要的输入",
    describe: "在 {d} 维中、使用{s}采集函数搜索时，经过 {n} 次评估，随机搜索的中位数遗憾为 {r}，采用固定尺度先验的贝叶斯优化为 {f}，采用维度缩放先验的贝叶斯优化为 {c}。",
    searchLocal: "全局加局部的",
    searchGlobal: "仅全局的",
    kfInit: "{n} 个初始点",
    kfEvals: "{n} 次评估",
  },
};

const params = {
  dims: { kind: "choice", label: { en: "Dimensions", zh: "维度" }, options: META.dims.map((d: number) => ({ value: d, label: { en: String(d), zh: `${d} 维` } })), default: 20 },
  search: { kind: "choice", label: { en: "Acquisition search", zh: "采集函数搜索" }, options: [{ value: "local", label: { en: "Global + local", zh: "全局 + 局部" } }, { value: "global", label: { en: "Global only", zh: "仅全局" } }], default: "local" },
  show: { kind: "choice", label: { en: "Show run of", zh: "显示哪种运行" }, options: [{ value: "fixed", label: { en: "Fixed prior", zh: "固定先验" } }, { value: "scaled", label: { en: "Scaled prior", zh: "缩放先验" } }, { value: "random", label: { en: "Random", zh: "随机搜索" } }], default: "scaled" },
} as const;

type P = { dims: number; search: string; show: string };

const COLORS: Record<string, string> = { random: C.c7, fixed: C.c4, scaled: C.acq };
const key = (m: string, search: string) => (m === "random" ? "random" : `${m}-${search}`);

function median(xs: number[]): number {
  const s = xs.slice().sort((a, b) => a - b);
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
}

function describe(st: State<P>): string {
  const p = st.p;
  const n = META.init + Math.round(st.t);
  const at = (m: string) => fixed(median(RUNS[p.dims][key(m, p.search)].regret.map((c) => c[n - 1])), 2);
  const L = labels[st.lang ?? "en"];
  return tpl(L.describe, { d: p.dims, s: p.search === "local" ? L.searchLocal : L.searchGlobal, n, r: at("random"), f: at("fixed"), c: at("scaled") });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 560;
  const n = META.init + Math.round(st.t);
  const parts: string[] = [];
  const left = narrow ? 40 : 52, right = st.w - 10;
  // 1. regret curves
  const lg = legend(narrow ? 8 : left, 14, st.w - 16, (["random", "fixed", "scaled"] as const).map((m) => ({ kind: "line" as const, color: COLORS[m], label: L[m] })));
  parts.push(lg.svg);
  const top1 = 14 + lg.height + 8, h1 = narrow ? 150 : 170;
  const xs = linear([1, META.budget], [left, right]);
  const ys = log([0.03, 3.5], [top1 + h1, top1]);
  parts.push(
    axis({ scale: xs, orient: "bottom", at: top1 + h1, span: [top1, top1 + h1], title: L.evals, count: 5 }),
    axis({ scale: ys, orient: "left", at: left, span: [left, right], format: (v) => String(v) }),
    text(right - 4, top1 + 11, L.regret, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo" }),
    el("rect", { x: xs(1), y: top1, width: xs(META.init) - xs(1), height: h1, fill: C.panel, opacity: 0.6 }),
  );
  for (const m of ["random", "fixed", "scaled"]) {
    const R = RUNS[p.dims][key(m, p.search)].regret;
    const idx = Array.from({ length: n }, (_, i) => i);
    const clampY = (v: number) => ys(Math.max(0.03, Math.min(3.5, v)));
    const lo = idx.map((i) => Math.min(...R.map((c) => c[i])));
    const hi = idx.map((i) => Math.max(...R.map((c) => c[i])));
    const med = idx.map((i) => median(R.map((c) => c[i])));
    parts.push(
      el("path", { d: linePath(idx.map((i) => [xs(i + 1), clampY(hi[i])])) + idx.slice().reverse().map((i) => `L${xs(i + 1).toFixed(1)},${clampY(lo[i]).toFixed(1)}`).join("") + "Z", fill: COLORS[m], opacity: 0.12, stroke: "none" }),
      el("path", { d: linePath(idx.map((i) => [xs(i + 1), clampY(med[i])])), fill: "none", stroke: COLORS[m], "stroke-width": m === p.show ? 2.6 : 1.6 }),
    );
  }
  // 2. parallel coordinates of one run
  const run = RUNS[p.dims][key(p.show, p.search)];
  const D = p.dims;
  const top2 = top1 + h1 + 66, h2 = narrow ? 110 : 120;
  const axes = Array.from({ length: D }, (_, i) => i);
  const step = D > 1 ? (right - left) / (D - 1) : 0;
  parts.push(el("rect", { x: left - Math.min(8, step / 2), y: top2 - 4, width: step * 5 + Math.min(16, step), height: h2 + 8, fill: C.truth, opacity: 0.08, rx: 4 }));
  const shown = run.points.slice(0, n);
  const vals = run.values.slice(0, n);
  const ranked = vals.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).map((x) => x[1]);
  const top = new Set(ranked.slice(0, Math.max(3, Math.round(n * 0.08))));
  const pts = [...shown.map((q, i) => ({ q, i, hot: top.has(i) })).filter((x) => !x.hot), ...shown.map((q, i) => ({ q, i, hot: top.has(i) })).filter((x) => x.hot)];
  const star = [...hartmann6.argmax, ...new Array(D - 6).fill(NaN)];
  parts.push(
    parallel(pts.map((x) => x.q), axes, left, right, top2, top2 + h2,
      (i) => (pts[i].hot ? { stroke: COLORS[p.show], width: 1.8, opacity: 0.95 } : { stroke: C.ink3, width: 0.8, opacity: 0.35 }),
      (d) => (D <= 20 || d % 5 === 4 || d === 0 ? String(d + 1) : "")),
    el("path", { d: linePath(axes.slice(0, 6).map((d) => [left + d * step, top2 + h2 - h2 * star[d]])), fill: "none", stroke: C.truth, "stroke-width": 2.4, "stroke-dasharray": "5 3" }),
    text(left, top2 - 12, tpl(narrow ? L.pcShort : L.pc, { m: L[p.show as "random" | "fixed" | "scaled"] }), { "font-size": TYPE.small, class: "fig-t-muted" }),
  );
  // 3. learned lengthscales
  const top3 = top2 + h2 + 44, h3 = narrow ? 70 : 80;
  const mode = (META.priorMode as Record<string, { fixed: number; scaled: number }>)[String(D)][p.show === "scaled" ? "scaled" : "fixed"];
  parts.push(text(left, top3 - 10, run.lengthscales ? tpl(narrow ? L.lsShort : L.ls, { mode: sig(mode, 2) }) : L.noModel, { "font-size": TYPE.small, class: "fig-t-muted" }));
  if (run.lengthscales) {
    const ly = log([0.02, 100], [top3 + h3, top3]);
    const bw = Math.max(1.5, Math.min(14, step * 0.7 || 14));
    parts.push(axis({ scale: ly, orient: "left", at: left, span: [left, right], format: (v) => String(v) }));
    run.lengthscales.forEach((l, d) => {
      const x = left + d * step;
      parts.push(el("rect", { x: x - bw / 2, y: ly(l), width: bw, height: top3 + h3 - ly(l), fill: d < 6 ? C.truth : C.ink3, opacity: d < 6 ? 0.9 : 0.55 }));
    });
    parts.push(el("line", { x1: left, x2: right, y1: ly(mode), y2: ly(mode), stroke: COLORS[p.show], "stroke-dasharray": "4 3", "stroke-width": 1.4 }));
  }
  return svg(st.w, top3 + h3 + 14, describe(st), g({}, ...parts));
}

export default defineFigure({
  name: "dim-bo-race",
  title: { en: "Bayesian optimization in 6 to 50 dimensions", zh: "6 至 50 维中的贝叶斯优化" },
  labels,
  params,
  timeline: {
    rate: 12,
    discrete: true,
    duration: () => META.budget - META.init,
    keyframes: (_p, lang) => [{ t: 0, label: tpl(labels[lang].kfInit, { n: META.init }) }, { t: 30, label: tpl(labels[lang].kfEvals, { n: META.init + 30 }) }, { t: META.budget - META.init, label: tpl(labels[lang].kfEvals, { n: META.budget }) }],
    poster: () => META.budget - META.init,
  },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
