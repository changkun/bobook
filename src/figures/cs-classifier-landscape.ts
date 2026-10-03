// Find the best setting of a real classifier yourself, then watch grid search,
// random search, and Bayesian optimization spend the same budget on the same
// measured landscape, then see the whole landscape.
//
// The landscape is measured, not simulated: the 5-fold cross-validation error
// of an RBF-kernel support vector machine on scikit-learn's handwritten digits
// (the test set of the UCI "Optical Recognition of Handwritten Digits" data,
// Alpaydin and Kaynak 1998, doi:10.24432/C50P49, CC BY 4.0) at every one of
// 41 x 37 settings of (log2 C, log2 gamma), each measured with five different
// random splits. Every evaluation in the figure returns one of those five
// measurements. The data file src/figures/data/cs-classifier-svm.json is
// written by
//
//   .cache/venv-cases/bin/python tools/figure-data/cs-classifier-svm.py
//
// (see the script's header for the environment), and the strategies live in
// lib/cs-classifier-race.ts, shared with tools/figure-data/cs-classifier-race.ts.
//
// Phases: while the reader has budget left, clicks evaluate cells; after the
// budget is spent (or "Skip to the race"), the timeline replays the three
// strategies, evaluation by evaluation, beside the reader's own sequence.

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { memo } from "./lib/random.ts";
import { fmtPoints, parsePoints, validPoints } from "./lib/params.ts";
import { hitArea, type Frame } from "./lib/plot.ts";
import { tpl } from "./lib/format.ts";
import {
  BEST, BEST_CELL, LOG2C, LOG2G, MEAN, NC, NG, NIMG, N_INIT, WORST,
  boRun, cellOf, gridRun, ijOf, model, modelError, nextCell, randomRun, readerRun, type Run,
} from "./lib/cs-classifier-race.ts";

const labels = {
  en: {
    you: "you",
    grid: "grid",
    random: "random",
    bo: "Bayesian optimization",
    boShort: "BO",
    cAxis: "C (penalty)",
    gAxis: "γ (kernel width)",
    evals: "evaluations",
    errAxis: "error of the pick (%)",
    best: "best of all 1,517 settings",
    hidden: "Unexplored settings are hidden.",
    left: "{n} of {b} evaluations left",
    last: "Last: log₂ C = {c}, log₂ γ = {g}: {e}% measured ({k} of 1,797 wrong)",
    yourBest: "Your best measured: {e}%",
    none: "Click a cell to evaluate that setting.",
    spent: "Budget spent.|Press play to race the three strategies.",
    racePrompt: "Your picks are scored here.|When your budget is spent, press play|to race grid, random search, and BO.",
    modelNote: "BO's model after {n} evaluations; ring: its next query",
    modelWait: "The model needs {n} evaluations.",
    errLegend: "measured error",
    describeGame: "Your search: {n} of {b} evaluations used. Your best measured validation error is {e}.",
    describeGameEmpty: "A hidden landscape of 1,517 settings of an SVM's C and γ. No setting has been evaluated yet; you have {b} evaluations.",
    describeRace: "After {t} evaluations, the five-split error of the setting each strategy would pick is {g} for grid search, {r} for random search{sep}{o} for Bayesian optimization{y}. The best of all 1,517 settings has {m}.",
    describeYou: ", and {v} for yours",
    sepMid: ", ",
    sepLast: ", and ",
    paren: "{s} ({v})",
    unitWait: "the race starts after your search",
    unitN: "{n} evaluations",
    kfStart: "past BO's random start",
    kfHalf: "halfway",
    kfEnd: "budget spent",
  },
  zh: {
    you: "你",
    grid: "网格",
    random: "随机",
    bo: "贝叶斯优化",
    boShort: "BO",
    cAxis: "C（惩罚）",
    gAxis: "γ（核宽度）",
    evals: "评估次数",
    errAxis: "所选设置的误差（%）",
    best: "全部 1,517 个设置中的最优",
    hidden: "未经评估的设置处于隐藏状态。",
    left: "还剩 {n} 次评估（共 {b} 次）",
    last: "上一次：log₂ C = {c}，log₂ γ = {g}：实测 {e}%（1,797 幅中错 {k} 幅）",
    yourBest: "你的最优实测值：{e}%",
    none: "点击一格即可评估该设置。",
    spent: "预算已用完。|按播放，让三种策略比一比。",
    racePrompt: "你选出的设置在这里计分。|预算用完后按播放，|让网格、随机搜索与 BO 比一比。",
    modelNote: "{n} 次评估后 BO 的模型；圆环：它的下一次查询",
    modelWait: "模型需要 {n} 次评估。",
    errLegend: "实测误差",
    describeGame: "你的搜索：已用 {n} 次评估（共 {b} 次）。你的最优实测验证误差为 {e}。",
    describeGameEmpty: "一个隐藏的地形：SVM 的 C 与 γ 的 1,517 个设置。尚未评估任何设置；你有 {b} 次评估。",
    describeRace: "{t} 次评估之后，各策略所选设置的五次划分平均误差为：网格搜索 {g}，随机搜索 {r}{sep}贝叶斯优化 {o}{y}。全部 1,517 个设置中的最优为 {m}。",
    describeYou: "，你的为 {v}",
    sepMid: "，",
    sepLast: "，",
    paren: "{s}（{v}）",
    unitWait: "你的搜索结束后开始比赛",
    unitN: "{n} 次评估",
    kfStart: "BO 随机起步之后",
    kfHalf: "过半",
    kfEnd: "预算用完",
  },
};

const BUDGETS = [12, 15, 20, 30] as const;

const params = {
  budget: { kind: "choice", label: { en: "Budget", zh: "预算" }, options: BUDGETS.map((b) => ({ value: b as number, label: { en: String(b), zh: String(b) } })), default: 15 },
  c: { kind: "range", label: { en: "log₂ C", zh: "log₂ C" }, min: -5, max: 15, step: 0.5, default: 5 },
  g: { kind: "range", label: { en: "log₂ γ", zh: "log₂ γ" }, min: -15, max: 3, step: 0.5, default: -6 },
  reveal: { kind: "toggle", label: { en: "Reveal the landscape", zh: "揭晓地形" }, default: false },
  model: { kind: "toggle", label: { en: "Show BO's model", zh: "显示 BO 的模型" }, default: false },
  picks: { kind: "data", label: { en: "Your evaluations", zh: "你的评估" }, default: "", validate: validPoints },
  skip: { kind: "toggle", label: { en: "Skip to the race", zh: "跳到比赛" }, default: false, control: false },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 1, step: 1, control: false },
} as const;

type P = { budget: number; c: number; g: number; reveal: boolean; model: boolean; picks: string; skip: boolean; seed: number };

const COLORS = { you: C.ink, grid: C.c4, random: C.c7, bo: C.acq };
const SUP: Record<string, string> = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", ".": "·" };
const pow2 = (v: number) => "2" + String(v).split("").map((ch) => SUP[ch] ?? ch).join("");
const num = (v: number) => String(v).replace("-", "−");
const pct = (e: number, d = 2) => (100 * e).toFixed(d);

const pickCells = (s: string) => parsePoints(s).map(([i, j]) => cellOf(i, j));
const inGame = (p: P) => !p.skip && parsePoints(p.picks).length < p.budget;

const runs = memo((budget: number, seed: number) => ({
  grid: gridRun(budget, seed),
  random: randomRun(budget, seed),
  bo: boRun(budget, seed),
}), 12);

const boModel = memo((budget: number, seed: number, n: number) => {
  const r = runs(budget, seed).bo;
  const m = model(r.cells.slice(0, n), r.obs.slice(0, n));
  const nx = nextCell(m);
  return { err: modelError(m, nx.mean), next: nx.cell };
}, 40);

const reader = memo((picks: string): Run => readerRun(pickCells(picks)), 8);

// Goodness of an error on a log scale: 1 at the best setting, 0 at the worst.
const LO = Math.log(BEST), HI = Math.log(WORST);
const shade = (e: number) => { const t = Math.min(1, Math.max(0, (HI - Math.log(e)) / (HI - LO))); return 0.05 + 0.9 * t ** 3; };

function heat(x0: number, y0: number, cs: number, vals: (c: number) => number | undefined): string {
  // One rect per horizontal run of cells with the same shade, to keep markup small.
  const parts: string[] = [];
  for (let j = 0; j < NG; j++) {
    let i = 0;
    while (i < NC) {
      const v = vals(cellOf(i, j));
      if (v === undefined) { i++; continue; }
      const q = Math.round(shade(v) * 40) / 40;
      let k = i + 1;
      while (k < NC) { const w = vals(cellOf(k, j)); if (w === undefined || Math.round(shade(w) * 40) / 40 !== q) break; k++; }
      parts.push(el("rect", { x: x0 + i * cs, y: y0 + (NG - 1 - j) * cs, width: (k - i) * cs, height: cs, fill: C.truth, opacity: q.toFixed(3) }));
      i = k;
    }
  }
  return g({ "shape-rendering": "crispEdges" }, ...parts);
}

function marker(kind: "you" | "grid" | "random" | "bo", cx: number, cy: number, r0: number, last: boolean): string {
  const r = last ? r0 * 1.35 : r0;
  const a = { fill: COLORS[kind], stroke: C.paper, "stroke-width": 1 };
  if (kind === "grid") return el("rect", { x: cx - r * 0.8, y: cy - r * 0.8, width: r * 1.6, height: r * 1.6, ...a });
  if (kind === "random") return el("path", { d: `M${cx},${cy - r}L${cx + r},${cy}L${cx},${cy + r}L${cx - r},${cy}Z`, ...a });
  return el("circle", { cx, cy, r: kind === "you" ? r * 0.8 : r * 0.85, ...a });
}

function describe(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const you = reader(p.picks);
  if (inGame(p)) {
    if (!you.cells.length) return tpl(L.describeGameEmpty, { b: p.budget });
    return tpl(L.describeGame, { n: you.cells.length, b: p.budget, e: `${pct(Math.min(...you.obs))}%` });
  }
  const t = Math.max(1, Math.min(p.budget, Math.round(st.t)));
  const R = runs(p.budget, p.seed);
  const at = (r: Run) => `${pct(r.rec[Math.min(t, r.rec.length) - 1])}%`;
  return tpl(L.describeRace, {
    t, g: at(R.grid), r: at(R.random), o: at(R.bo), m: `${pct(BEST)}%`, sep: you.cells.length ? L.sepMid : L.sepLast,
    y: you.cells.length ? tpl(L.describeYou, { v: at(you) }) : "",
  });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 560;
  const game = inGame(p);
  const t = game ? 0 : Math.max(0, Math.min(p.budget, Math.round(st.t)));
  const you = reader(p.picks);
  const R = runs(p.budget, p.seed);
  const parts: string[] = [];

  // Legend: the three strategies and you, plus the shade key.
  const items: Array<[keyof typeof COLORS, string]> = [["you", L.you], ["grid", L.grid], ["random", L.random], ["bo", narrow ? L.boShort : L.bo]];
  let lx = narrow ? 8 : 46;
  const ly = 14;
  for (const [k, name] of items) {
    parts.push(marker(k, lx + 5, ly - 4, 4, false), text(lx + 13, ly, name, { "font-size": TYPE.small, class: "fig-t-muted" }));
    lx += labelWidth(name) + 28;
  }
  // shade key
  const kw = narrow ? 70 : 96, kx = st.w - kw - (narrow ? 8 : 12), ky = narrow ? ly + 12 : ly - 9;
  for (let s = 0; s < 24; s++) {
    const e = Math.exp(HI + (LO - HI) * (s / 23));
    parts.push(el("rect", { x: kx + (s * kw) / 24, y: ky, width: kw / 24 + 0.3, height: 8, fill: C.truth, opacity: shade(e).toFixed(3) }));
  }
  parts.push(el("rect", { x: kx, y: ky, width: kw, height: 8, fill: "none", stroke: C.rule, "stroke-width": 0.6 }));
  parts.push(text(kx, ky + 19, `${pct(WORST, 0)}%`, { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }),
    text(kx + kw, ky + 19, `${pct(BEST, 1)}%`, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-num" }));
  if (!narrow) parts.push(text(kx - 6, ky + 8, L.errLegend, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted" }));

  // Heatmap geometry.
  const top = narrow ? 50 : 40;
  const x0 = narrow ? 44 : 52;
  const leftW = narrow ? st.w - x0 - 10 : Math.floor(st.w * 0.52) - x0;
  const cs = Math.max(4, Math.floor(Math.min(leftW / NC, (narrow ? 240 : 290) / NG) * 10) / 10);
  const hw = cs * NC, hh = cs * NG;
  const y0 = top;
  const sx = linear([LOG2C[0] - 0.25, LOG2C[NC - 1] + 0.25], [x0, x0 + hw]);
  const sy = linear([LOG2G[0] - 0.25, LOG2G[NG - 1] + 0.25], [y0 + hh, y0]);
  const cx = (c: number) => x0 + (ijOf(c)[0] + 0.5) * cs;
  const cy = (c: number) => y0 + (NG - 1 - ijOf(c)[1] + 0.5) * cs;

  parts.push(el("rect", { x: x0, y: y0, width: hw, height: hh, fill: C.panel, stroke: C.rule, "stroke-width": 0.8 }));
  // What is visible: everything when revealed; otherwise the cells someone has evaluated.
  const showModel = p.model && !game && t >= N_INIT;
  if (showModel) {
    const m = boModel(p.budget, p.seed, t);
    parts.push(heat(x0, y0, cs, (c) => m.err[c]));
  } else if (p.reveal) {
    parts.push(heat(x0, y0, cs, (c) => MEAN[c]));
  } else {
    const seen = new Map<number, number>();
    you.cells.forEach((c, k) => seen.set(c, you.obs[k]));
    if (!game) for (const r of [R.grid, R.random, R.bo]) r.cells.slice(0, t).forEach((c, k) => { if (!seen.has(c)) seen.set(c, r.obs[k]); });
    parts.push(heat(x0, y0, cs, (c) => seen.get(c)));
  }
  const tickC = [-5, 0, 5, 10, 15], tickG = [-15, -10, -5, 0];
  parts.push(
    axis({ scale: sx, orient: "bottom", at: y0 + hh, title: L.cAxis, ticks: tickC, format: pow2, grid: false }),
    axis({ scale: sy, orient: "left", at: x0, title: L.gAxis, ticks: tickG, format: pow2, grid: false }),
  );
  const f: Frame = { x: sx, y: sy, left: x0, right: x0 + hw, top: y0, bottom: y0 + hh, w: st.w };
  parts.push(hitArea(f, "heat"));

  // Markers.
  const mr = Math.max(2.6, cs * 0.5);
  if (!game) {
    const order: Array<["grid" | "random" | "bo", Run]> = [["grid", R.grid], ["random", R.random], ["bo", R.bo]];
    for (const [k, r] of order) r.cells.slice(0, t).forEach((c, i) => parts.push(marker(k, cx(c), cy(c), mr, i === t - 1)));
  }
  you.cells.forEach((c, i) => parts.push(marker("you", cx(c), cy(c), mr, game && i === you.cells.length - 1)));
  if (showModel) {
    const m = boModel(p.budget, p.seed, t);
    parts.push(el("circle", { cx: cx(m.next), cy: cy(m.next), r: mr * 2.2, fill: "none", stroke: C.acq, "stroke-width": 2 }));
  }
  if (p.reveal || showModel) {
    // the best measured setting
    const bx = cx(BEST_CELL), by = cy(BEST_CELL), r = mr * 1.6;
    let d = "";
    for (let i = 0; i < 10; i++) { const a = -Math.PI / 2 + (i * Math.PI) / 5, rr = i % 2 ? r * 0.45 : r; d += `${i ? "L" : "M"}${(bx + rr * Math.cos(a)).toFixed(1)},${(by + rr * Math.sin(a)).toFixed(1)}`; }
    parts.push(el("path", { d: d + "Z", fill: C.c4, stroke: C.ink, "stroke-width": 0.8 }));
  }
  if (game) {
    // the cursor for keyboard selection
    const ci = Math.round((p.c - LOG2C[0]) * 2), gj = Math.round((p.g - LOG2G[0]) * 2);
    parts.push(el("rect", { x: x0 + ci * cs - 1, y: y0 + (NG - 1 - gj) * cs - 1, width: cs + 2, height: cs + 2, fill: "none", stroke: C.ink2, "stroke-width": 1.4 }));
  }

  // Readout under the heatmap.
  let ry = y0 + hh + 46;
  const say = (s: string, cls = "fig-t-muted") => { parts.push(text(narrow ? 8 : x0 - 40, ry, s, { "font-size": TYPE.small, class: cls })); ry += 16; };
  if (game) {
    say(tpl(L.left, { n: p.budget - you.cells.length, b: p.budget }), "fig-t-strong");
    if (you.cells.length) {
      const c = you.cells[you.cells.length - 1], [i, j] = ijOf(c), e = you.obs[you.obs.length - 1];
      say(tpl(L.last, { c: num(LOG2C[i]), g: num(LOG2G[j]), e: pct(e), k: Math.round(e * NIMG) }));
      say(tpl(L.yourBest, { e: pct(Math.min(...you.obs)) }));
    } else say(L.none);
  } else if (showModel) {
    say(tpl(L.modelNote, { n: t }));
  } else if (p.model) {
    say(tpl(L.modelWait, { n: N_INIT }));
  } else if (!p.reveal) {
    say(L.hidden);
  }

  // Race panel: the five-split error of each strategy's current pick.
  const bx0 = narrow ? 52 : Math.floor(st.w * 0.52) + 54, bx1 = st.w - 12;
  const by0 = narrow ? ry + 6 : top, bh = narrow ? 150 : hh;
  const YLO = 0.006, YHI = 0.026;
  const bxs = linear([1, p.budget], [bx0, bx1]);
  const bys = linear([YLO, YHI], [by0 + bh, by0]);
  parts.push(
    axis({ scale: bxs, orient: "bottom", at: by0 + bh, span: [by0, by0 + bh], title: L.evals, count: 4 }),
    axis({ scale: bys, orient: "left", at: bx0, span: [bx0, bx1], title: narrow ? "" : L.errAxis, ticks: [0.01, 0.015, 0.02, 0.025], format: (v) => (100 * v).toFixed(1) }),
    el("line", { x1: bx0, x2: bx1, y1: bys(BEST), y2: bys(BEST), stroke: C.truth, "stroke-dasharray": "5 4", "stroke-width": 1.4 }),
    text(bx1 - 2, bys(BEST) + 13, tpl(L.paren, { s: L.best, v: `${pct(BEST)}%` }), { "text-anchor": "end", "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
  );
  if (narrow) parts.push(text(bx0 + 4, by0 - 6, L.errAxis, { "font-size": TYPE.small, class: "fig-t-muted" }));
  const line = (r: Run, n: number, color: string, wdt: number) => {
    const pts = r.rec.slice(0, n).map((v, i): [number, number] => [bxs(i + 1), bys(Math.min(v, YHI))]);
    if (!pts.length) return "";
    return g({},
      el("path", { d: linePath(pts), fill: "none", stroke: color, "stroke-width": wdt, "stroke-linejoin": "round" }),
      // values above the axis are pinned to its top edge and marked open
      ...r.rec.slice(0, n).map((v, i) => (v > YHI ? el("circle", { cx: bxs(i + 1), cy: bys(YHI), r: 2.2, fill: C.paper, stroke: color, "stroke-width": 1.2 }) : "")),
    );
  };
  if (game) {
    if (you.cells.length) parts.push(line(you, you.cells.length, COLORS.you, 2));
    (you.cells.length >= p.budget ? L.spent : L.racePrompt).split("|").forEach((s, i) => parts.push(text((bx0 + bx1) / 2 + 8, by0 + bh / 2 - 24 + 15 * i, s, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-faint fig-t-halo" })));
  } else {
    if (t >= 1) {
      // BO first and widest, so a thinner line that coincides with it stays visible on top.
      parts.push(line(R.bo, t, COLORS.bo, 3.2), line(R.grid, t, COLORS.grid, 1.6), line(R.random, t, COLORS.random, 1.6));
      if (you.cells.length) parts.push(line(you, Math.min(t, you.cells.length), COLORS.you, 1.6));
    }
    parts.push(el("line", { x1: bxs(Math.max(1, t)), x2: bxs(Math.max(1, t)), y1: by0, y2: by0 + bh, stroke: C.ink3, "stroke-dasharray": "2 3" }));
  }
  const H = Math.max(ry + 4, by0 + bh + 44);
  return svg(st.w, H, describe(st), ...parts);
}

function evaluate(p: P): P {
  if (!inGame(p)) return p;
  const pts = parsePoints(p.picks);
  pts.push([Math.round((p.c - LOG2C[0]) * 2), Math.round((p.g - LOG2G[0]) * 2)]);
  return { ...p, picks: fmtPoints(pts) };
}

export default defineFigure({
  name: "cs-classifier-landscape",
  title: { en: "Tuning a support vector machine on measured data: you, grid search, random search, and Bayesian optimization", zh: "用实测数据为支持向量机调参：你、网格搜索、随机搜索与贝叶斯优化" },
  labels,
  params,
  hint: { en: "Click a cell to evaluate that setting (or set log₂ C and log₂ γ and press Evaluate). Each evaluation returns one measured cross-validation error.", zh: "点击一格即可评估该设置（也可以设定 log₂ C 与 log₂ γ 后按“评估”）。每次评估返回一个实测的交叉验证误差。" },
  timeline: {
    rate: 3,
    discrete: true,
    duration: (p) => p.budget,
    // While the reader is still searching there is no race to scrub: no
    // keyframes, and the readout says so instead of "budget spent".
    unit: { symbol: { en: "", zh: "" }, value: (t, p, lang) => (inGame(p as P) ? labels[lang ?? "en"].unitWait : tpl(labels[lang ?? "en"].unitN, { n: Math.round(t) })) },
    keyframes: (p, lang) => {
      const L = labels[lang ?? "en"];
      return inGame(p as P) ? [] : [{ t: N_INIT, label: L.kfStart }, { t: Math.round(p.budget / 2), label: L.kfHalf }, { t: p.budget, label: L.kfEnd }];
    },
    poster: (p) => p.budget,
  },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click" || e.target !== "heat" || !e.data || !inGame(p as P)) return null;
    const c = Math.min(LOG2C[NC - 1], Math.max(LOG2C[0], Math.round(e.data.x * 2) / 2));
    const gg = Math.min(LOG2G[NG - 1], Math.max(LOG2G[0], Math.round(e.data.y * 2) / 2));
    return evaluate({ ...(p as P), c, g: gg });
  },
  update(p, key) {
    if (key === "budget") {
      const pts = parsePoints(p.picks);
      if (pts.length > p.budget) return { ...p, picks: fmtPoints(pts.slice(0, p.budget)) };
    }
    return p;
  },
  actions: [
    { label: { en: "Evaluate", zh: "评估" }, primary: true, run: (p) => evaluate(p as P), enabled: (p) => inGame(p as P) },
    { label: { en: "Undo", zh: "撤销" }, run: (p) => ({ ...p, picks: fmtPoints(parsePoints(p.picks).slice(0, -1)) }), enabled: (p) => inGame(p as P) && p.picks !== "" },
    { label: { en: "Skip to the race", zh: "跳到比赛" }, run: (p) => ({ ...p, skip: true }), enabled: (p) => inGame(p as P) },
    { label: { en: "Search again", zh: "重新搜索" }, run: (p) => ({ ...p, picks: "", skip: false }), enabled: (p) => !inGame(p as P) },
    { label: { en: "New random run", zh: "新的随机运行" }, run: (p) => ({ ...p, seed: (p.seed % 9999) + 1 }), enabled: (p) => !inGame(p as P) },
  ],
});
