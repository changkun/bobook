// Recorded hyperparameter searches on a gradient-boosted tree classifier with
// seven hyperparameters: ten runs of random search and ten of Bayesian
// optimization (Gaussian process, expected improvement), 60 evaluations each.
// Nothing here is simulated: every point is a configuration that was trained
// and scored, and the figure replays the records. The top panel shows the
// best error found so far by each run (validation or held-out test error,
// against evaluations or compute seconds); the bottom panel draws the
// evaluated configurations in parallel coordinates, with each
// hyperparameter's main-effect share estimated from the random-search runs.
//
// Data: src/figures/data/cs-classifier-gbm.json, written by
//
//   .cache/venv-cases/bin/python tools/figure-data/cs-classifier-gbm.py
//
// (environment, protocol, and optimizer settings in the script's header).
// The data set is Spambase (Hopkins, Reeber, Forman, and Suermondt 1999, UCI
// Machine Learning Repository, doi:10.24432/C53G6X, CC BY 4.0); the model is
// scikit-learn's HistGradientBoostingClassifier.

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { parallel } from "./lib/nd.ts";
import { tpl } from "./lib/format.ts";
import gbm from "./data/cs-classifier-gbm.json" with { type: "json" };

const D = 7;
const IX = { cv: D, sd: D + 1, test: D + 2, secs: D + 3 };
type Ev = number[];
interface RunRec { method: string; seed: number; evals: Ev[] }
const RUNS = gbm.runs as RunRec[];
const BUDGET: number = gbm.budget;
const INIT: number = gbm.init;
const DEFAULT = gbm.default;
const DIMS = gbm.dims as Array<{ name: string; lo: number; hi: number; scale: string; int: boolean }>;

const NAMES: Record<string, [string, string]> = {
  learning_rate: ["learning rate", "rate"],
  max_iter: ["trees", "trees"],
  max_leaf_nodes: ["leaves", "leaves"],
  min_samples_leaf: ["min leaf", "min leaf"],
  l2_regularization: ["L2 penalty", "L2"],
  max_features: ["features", "feat."],
  max_bins: ["bins", "bins"],
};
const NAMES_ZH: Record<string, [string, string]> = {
  learning_rate: ["学习率", "学习率"],
  max_iter: ["树的数量", "树数"],
  max_leaf_nodes: ["叶数", "叶数"],
  min_samples_leaf: ["最小叶样本数", "最小叶"],
  l2_regularization: ["L2 惩罚", "L2"],
  max_features: ["特征比例", "特征"],
  max_bins: ["分箱数", "分箱"],
};

const labels = {
  en: {
    bo: "Bayesian optimization",
    boShort: "BO",
    random: "random search",
    median: "median of 10 runs",
    def: "scikit-learn default",
    evals: "evaluations",
    secs: "compute seconds (one CPU thread)",
    secsShort: "compute seconds",
    cvAxis: "best CV error (%)",
    testAxis: "test error of the pick (%)",
    errAxisPc: "CV error",
    showing: "{n} configuration{n:/s} from {which}; bold: CV error ≤ {thr}% (best 10% of all 1,200)",
    allRuns: "all ten {m} runs",
    oneRun: "{m} run {k}",
    share: "main effect",
    describe: "After {t} evaluations, the median best cross-validation error is {b}% for Bayesian optimization and {r}% for random search; the library default scores {d}%. The learning rate and the number of trees explain most of the variation in the error.",
    mBo: "BO",
    mRandom: "random-search",
    paren: "{s} ({v})",
    kfStart: "past BO's random start",
    kfHalf: "halfway",
    kfEnd: "budget spent",
  },
  zh: {
    bo: "贝叶斯优化",
    boShort: "BO",
    random: "随机搜索",
    median: "10 次运行的中位数",
    def: "scikit-learn 默认值",
    evals: "评估次数",
    secs: "计算秒数（单个 CPU 线程）",
    secsShort: "计算秒数",
    cvAxis: "当前最优 CV 误差（%）",
    testAxis: "所选配置的测试误差（%）",
    errAxisPc: "CV 误差",
    showing: "{which}中的 {n} 个配置；粗线：CV 误差 ≤ {thr}%（全部 1,200 次评估中最好的 10%）",
    allRuns: "{m}的全部十次运行",
    oneRun: "{m}的第 {k} 次运行",
    share: "主效应",
    describe: "{t} 次评估之后，当前最优交叉验证误差的中位数为：贝叶斯优化 {b}%，随机搜索 {r}%；库的默认设置为 {d}%。学习率与树的数量解释了误差变化的大部分。",
    mBo: "贝叶斯优化",
    mRandom: "随机搜索",
    paren: "{s}（{v}）",
    kfStart: "BO 随机起步之后",
    kfHalf: "过半",
    kfEnd: "预算用完",
  },
};

const params = {
  score: { kind: "choice", label: { en: "Score", zh: "评分" }, options: [{ value: "cv", label: { en: "Validation (CV)", zh: "验证（CV）" } }, { value: "test", label: { en: "Held-out test", zh: "留出测试" } }], default: "cv" },
  xaxis: { kind: "choice", label: { en: "Spend", zh: "花费" }, options: [{ value: "evals", label: { en: "Evaluations", zh: "评估次数" } }, { value: "secs", label: { en: "Compute", zh: "计算时间" } }], default: "evals" },
  show: { kind: "choice", label: { en: "Configurations of", zh: "显示其配置" }, options: [{ value: "bo", label: { en: "BO", zh: "贝叶斯优化" } }, { value: "random", label: { en: "Random", zh: "随机搜索" } }], default: "bo" },
  run: { kind: "choice", label: { en: "Run", zh: "运行" }, options: [{ value: 0, label: { en: "all", zh: "全部" } }, ...Array.from({ length: 10 }, (_, i) => ({ value: i + 1, label: { en: String(i + 1), zh: String(i + 1) } }))], default: 0, control: "select" },
} as const;

type P = { score: "cv" | "test"; xaxis: "evals" | "secs"; show: "bo" | "random"; run: number };

const COLORS: Record<string, string> = { bo: C.acq, random: C.c7 };

// Best-so-far by the measure the optimizer used (CV error), and the held-out
// test error of that same pick, which the optimizer never saw.
function curves(r: RunRec) {
  const cv: number[] = [], test: number[] = [], secs: number[] = [];
  let b = 0, s = 0;
  r.evals.forEach((e, i) => {
    if (e[IX.cv] < r.evals[b][IX.cv]) b = i;
    s += e[IX.secs];
    cv.push(r.evals[b][IX.cv]);
    test.push(r.evals[b][IX.test]);
    secs.push(s);
  });
  return { cv, test, secs };
}
const CURVES = RUNS.map((r) => ({ method: r.method, seed: r.seed, ...curves(r) }));

const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); const m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
function medianCurve(method: string, key: "cv" | "test"): number[] {
  const cs = CURVES.filter((c) => c.method === method);
  return Array.from({ length: BUDGET }, (_, i) => median(cs.map((c) => c[key][i])));
}
const MED = { bo: { cv: medianCurve("bo", "cv"), test: medianCurve("bo", "test") }, random: { cv: medianCurve("random", "cv"), test: medianCurve("random", "test") } };

// Main-effect share of each hyperparameter: random search samples the unit
// cube uniformly, so the share of the variance of log(CV error) explained by
// quintile bins of one coordinate estimates that coordinate's first-order
// effect (the idea behind functional ANOVA, Hutter et al. 2014, without the
// surrogate model).
const SHARE: number[] = (() => {
  const ev = RUNS.filter((r) => r.method === "random").flatMap((r) => r.evals);
  const y = ev.map((e) => Math.log(e[IX.cv]));
  const my = y.reduce((a, b) => a + b, 0) / y.length;
  const tot = y.reduce((a, v) => a + (v - my) ** 2, 0);
  return Array.from({ length: D }, (_, k) => {
    let between = 0;
    for (let b = 0; b < 5; b++) {
      const ys = y.filter((_, i) => Math.min(4, Math.floor(ev[i][k] * 5)) === b);
      if (ys.length) between += ys.length * (ys.reduce((a, v) => a + v, 0) / ys.length - my) ** 2;
    }
    return between / tot;
  });
})();

const ALL_CV = RUNS.flatMap((r) => r.evals.map((e) => e[IX.cv])).sort((a, b) => a - b);
const THR = ALL_CV[Math.floor(0.1 * ALL_CV.length) - 1];
const EMIN = ALL_CV[0], EMAX = ALL_CV[ALL_CV.length - 1];

const SUP: Record<string, string> = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
function fmtVal(v: number): string {
  if (v > 0 && v < 0.01) return "10" + String(Math.round(Math.log10(v))).split("").map((c) => SUP[c] ?? c).join("");
  return String(Number(v.toPrecision(3)));
}

const pct = (v: number) => (100 * v).toFixed(2);

function describe(st: State<P>): string {
  const t = Math.max(1, Math.min(BUDGET, Math.round(st.t)));
  return tpl(labels[st.lang ?? "en"].describe, { t, b: pct(MED.bo.cv[t - 1]), r: pct(MED.random.cv[t - 1]), d: pct(DEFAULT.cv) });
}

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 480;
  const t = Math.max(1, Math.min(BUDGET, Math.round(st.t)));
  const parts: string[] = [];

  // Legend.
  const items: Array<[string, string, number]> = [["bo", narrow ? L.boShort : L.bo, 2.4], ["random", L.random, 2.4]];
  let lx = narrow ? 8 : 52;
  for (const [k, name, w] of items) {
    parts.push(el("line", { x1: lx, x2: lx + 18, y1: 10, y2: 10, stroke: COLORS[k], "stroke-width": w }), text(lx + 24, 14, name, { "font-size": TYPE.small, class: "fig-t-muted" }));
    lx += labelWidth(name) + 40;
  }
  parts.push(el("line", { x1: lx, x2: lx + 18, y1: 10, y2: 10, stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "4 3" }), text(lx + 24, 14, L.def, { "font-size": TYPE.small, class: "fig-t-muted" }));

  // Panel A: best so far.
  const ax0 = narrow ? 44 : 60, ax1 = st.w - 12;
  const ay0 = narrow ? 44 : 34, ah = narrow ? 140 : 170;
  const key = p.score;
  const [ylo, yhi] = key === "cv" ? [0.042, 0.054] : [0.028, 0.054];
  const xmax = p.xaxis === "evals" ? BUDGET : Math.max(...CURVES.map((c) => c.secs[BUDGET - 1]));
  const xs = linear(p.xaxis === "evals" ? [1, BUDGET] : [0, xmax], [ax0, ax1]);
  const ys = linear([ylo, yhi], [ay0 + ah, ay0]);
  const yTicks = key === "cv" ? [0.044, 0.048, 0.052] : [0.03, 0.04, 0.05];
  parts.push(
    axis({ scale: xs, orient: "bottom", at: ay0 + ah, span: [ay0, ay0 + ah], title: p.xaxis === "evals" ? L.evals : narrow ? L.secsShort : L.secs, count: narrow ? 4 : 6 }),
    axis({ scale: ys, orient: "left", at: ax0, span: [ax0, ax1], title: narrow ? "" : key === "cv" ? L.cvAxis : L.testAxis, ticks: yTicks, format: (v) => (100 * v).toFixed(1) }),
  );
  if (narrow) parts.push(text(ax0 + 4, ay0 - 6, key === "cv" ? L.cvAxis : L.testAxis, { "font-size": TYPE.small, class: "fig-t-muted" }));
  const dv = key === "cv" ? DEFAULT.cv : DEFAULT.test;
  parts.push(el("line", { x1: ax0, x2: ax1, y1: ys(dv), y2: ys(dv), stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "4 3" }));
  const clampY = (v: number) => ys(Math.min(yhi, Math.max(ylo, v)));
  const runLine = (c: (typeof CURVES)[number], n: number, w: number, op: number) => {
    const pts = c[key].slice(0, n).map((v, i): [number, number] => [xs(p.xaxis === "evals" ? i + 1 : c.secs[i]), clampY(v)]);
    return el("path", { d: linePath(pts), fill: "none", stroke: COLORS[c.method], "stroke-width": w, opacity: op, "stroke-linejoin": "round" });
  };
  for (const c of CURVES) {
    const sel = p.run > 0 && c.seed === p.run;
    parts.push(runLine(c, t, sel ? 2.2 : 1, sel ? 1 : p.run > 0 ? 0.25 : 0.4));
  }
  if (p.xaxis === "evals" && p.run === 0) {
    for (const m of ["random", "bo"] as const) {
      const pts = MED[m][key].slice(0, t).map((v, i): [number, number] => [xs(i + 1), clampY(v)]);
      parts.push(el("path", { d: linePath(pts), fill: "none", stroke: COLORS[m], "stroke-width": 3, "stroke-linejoin": "round" }));
    }
  }
  if (p.xaxis === "evals") parts.push(el("line", { x1: xs(t), x2: xs(t), y1: ay0, y2: ay0 + ah, stroke: C.ink3, "stroke-dasharray": "2 3" }));
  parts.push(text(ax1 - 2, ys(dv) - 5, tpl(L.paren, { s: L.def, v: `${pct(dv)}%` }), { "text-anchor": "end", "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));

  // Panel B: parallel coordinates of the evaluated configurations.
  const py0 = ay0 + ah + (narrow ? 104 : 96), ph = narrow ? 170 : 190;
  const px0 = narrow ? 22 : 56, px1 = st.w - (narrow ? 26 : 56);
  const runs = RUNS.filter((r) => r.method === p.show && (p.run === 0 || r.seed === p.run));
  const shown: Ev[] = runs.flatMap((r) => r.evals.slice(0, t));
  const lastOnes = new Set(runs.map((r) => r.evals[t - 1]));
  const errU = (e: number) => Math.log(e / EMIN) / Math.log(EMAX / EMIN);
  // order: ordinary first, then the best 10%, then the latest evaluation of each run
  const pts = shown.map((e) => [...e.slice(0, D), errU(e[IX.cv])]);
  const order = shown.map((_, i) => i).sort((a, b) => {
    const ra = lastOnes.has(shown[a]) ? 2 : shown[a][IX.cv] <= THR ? 1 : 0, rb = lastOnes.has(shown[b]) ? 2 : shown[b][IX.cv] <= THR ? 1 : 0;
    return ra - rb || shown[b][IX.cv] - shown[a][IX.cv];
  });
  const ordered = order.map((i) => pts[i]);
  const styleOf = (k: number) => {
    const e = shown[order[k]];
    if (lastOnes.has(e) && p.run > 0) return { stroke: C.ink, width: 2.8, opacity: 0.95 };
    if (e[IX.cv] <= THR) return { stroke: COLORS[p.show], width: p.run > 0 ? 1.8 : 1.1, opacity: p.run > 0 ? 0.9 : 0.5 };
    return { stroke: COLORS[p.show], width: 0.8, opacity: p.run > 0 ? 0.3 : 0.1 };
  };
  const axesIdx = Array.from({ length: D + 1 }, (_, i) => i);
  parts.push(parallel(ordered, axesIdx, px0, px1, py0, py0 + ph, styleOf, () => ""));
  // Axis names, end values, and main-effect shares.
  const step = (px1 - px0) / D;
  for (let k = 0; k <= D; k++) {
    const x = px0 + k * step;
    const name = k < D ? (lang === "zh" ? NAMES_ZH : NAMES)[DIMS[k].name][narrow ? 1 : 0] : L.errAxisPc;
    const stagger = narrow && k % 2 === 1 ? 13 : 0;
    parts.push(text(x, py0 - 30 - (narrow ? 13 - stagger : 0), name, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-strong" }));
    const lo = k < D ? fmtVal(DIMS[k].lo) : `${(100 * EMIN).toFixed(1)}%`;
    const hi = k < D ? fmtVal(DIMS[k].hi) : `${(100 * EMAX).toFixed(0)}%`;
    parts.push(
      text(x, py0 - 6, hi, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-num fig-t-halo" }),
      text(x, py0 + ph + 15, lo, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-num fig-t-halo" }),
    );
    if (k < D) {
      // main-effect share bar under the axis
      const bw = Math.max(1.5, SHARE[k] * (narrow ? 60 : 110));
      const by = py0 + ph + 26;
      parts.push(el("rect", { x: x - bw / 2, y: by, width: bw, height: 6, fill: C.c1, rx: 1 }),
        text(x, by + 19, `${Math.round(100 * SHARE[k])}%`, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
    }
  }
  const by = py0 + ph + 26;
  parts.push(text(px1 + (narrow ? 20 : 50), by + 19, narrow ? "" : L.share, { "text-anchor": "end", "font-size": TYPE.small, class: "fig-t-faint" }));
  // Caption line for the bottom panel.
  const which = p.run === 0 ? tpl(L.allRuns, { m: p.show === "bo" ? L.mBo : L.mRandom }) : tpl(L.oneRun, { m: p.show === "bo" ? L.mBo : L.mRandom, k: p.run });
  const capY = by + 40;
  const cap = tpl(L.showing, { n: shown.length, which, thr: (100 * THR).toFixed(2) });
  if (narrow) {
    const zh = lang === "zh";
    const cut = cap.indexOf(zh ? "；" : "; ");
    parts.push(text(8, capY, cap.slice(0, cut + 1), { "font-size": TYPE.small, class: "fig-t-muted" }), text(8, capY + 15, cap.slice(cut + (zh ? 1 : 2)), { "font-size": TYPE.small, class: "fig-t-muted" }));
  } else parts.push(text(px0 - 30, capY, cap, { "font-size": TYPE.small, class: "fig-t-muted" }));
  const H = capY + (narrow ? 24 : 10);
  return svg(st.w, H, describe(st), g({}, ...parts));
}

export default defineFigure({
  name: "cs-classifier-many",
  title: { en: "Recorded searches over seven hyperparameters of a gradient-boosted tree classifier", zh: "梯度提升树分类器七个超参数上的搜索记录" },
  labels,
  params,
  timeline: {
    rate: 6,
    discrete: true,
    duration: () => BUDGET,
    unit: { symbol: { en: "evaluations", zh: "次评估" }, value: (t) => String(Math.round(t)) },
    keyframes: (_p, lang) => {
      const L = labels[lang ?? "en"];
      return [{ t: INIT, label: L.kfStart }, { t: 30, label: L.kfHalf }, { t: BUDGET, label: L.kfEnd }];
    },
    poster: () => BUDGET,
  },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
