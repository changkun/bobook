// Bayesian optimization replayed on the measured direct arylation data of
// Shields et al. (Nature 2021): every one of the 1,728 combinations of 12
// ligands, 4 bases, 4 solvents, 3 concentrations, and 3 temperatures was run
// once by high-throughput experimentation, so an "experiment" here looks up a
// real measured yield. The reader chooses how the optimizer encodes the
// reagents, how many experiments it runs per batch, and where it starts; the
// figure compares the run with random selection, with the 50 chemists and
// engineers who played the paper's reaction optimization game on the same
// data, and with the mean of the paper's own 50 optimizer runs.
//
// Data: src/figures/data/cs-chem-arylation.json, written by
//   .cache/venv-cases/bin/python tools/figure-data/cs-chem-arylation.py
// from github.com/b-shields/edbo and github.com/b-shields/EvML (MIT License).
// The optimizer (lib/cs-chem.ts) is a simplified re-implementation, not the
// paper's code; tools/figure-data/cs-chem-race.ts compares the two.

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { memo } from "./lib/random.ts";
import { tpl } from "./lib/format.ts";
import { YIELD, bestSoFar, boRun, describeReaction, randomPicks, type Encoding, type Init } from "./lib/cs-chem.ts";
import { cellCenter, curveFrame, humanMean, mapGeom, mapFooter, mapLabels, references, stepLine, yieldMap } from "./lib/cs-chem-draw.ts";

const BUDGET = 50;

const labels = {
  en: {
    bo: "this Bayesian optimization run",
    boShort: "BO run",
    random: "random selection",
    chemists: "the 50 chemists (mean)",
    chemistsShort: "chemists (mean)",
    each: "each chemist",
    paper: "paper's optimizer (mean of 50 runs)",
    paperShort: "paper's BO",
    best: "best yield so far (%)",
    exps: "experiments",
    batch: "Batch {k} ({n} experiment{n:/s}):",
    more: "… and {n} more",
    relevance: "What the model weighs (1 / lengthscale), batch {k}:",
    waiting: "The model is fitted after the first batch.",
    choice: "ligand|base|solvent|conc.|temp.",
    yieldKey: "measured yield",
    describe: "After {t} of {b} experiments in batches of {k}, with {e} encoding and a {s} first batch, Bayesian optimization's best measured yield is {o}% and random selection's is {r}%; the 50 chemists averaged {h}% ({n} of them were still playing).",
    encDesc: "descriptor",
    encOnehot: "one-hot",
    startRandom: "random",
    startChemist: "chemist's",
    na: "n/a",
    kfFirst: "first batch done",
    kfHalf: "halfway",
    kfEnd: "budget spent",
  },
  zh: {
    bo: "本次贝叶斯优化运行",
    boShort: "本次运行",
    random: "随机选择",
    chemists: "50 位化学家（均值）",
    chemistsShort: "化学家（均值）",
    each: "每位化学家",
    paper: "论文的优化器（50 次运行的均值）",
    paperShort: "论文的优化器",
    best: "目前的最高产率（%）",
    exps: "实验次数",
    batch: "第 {k} 批（{n} 次实验）：",
    more: "……另有 {n} 个",
    relevance: "第 {k} 批时模型看重什么（1 / 长度尺度）：",
    waiting: "模型在第一批之后拟合。",
    choice: "配体|碱|溶剂|浓度|温度",
    yieldKey: "实测产率",
    describe: "以每批 {k} 个、{e}、{s}的第一批进行 {t} 次实验（共 {b} 次）后，贝叶斯优化的实测最高产率为 {o}%，随机选择为 {r}%；50 位化学家的均值为 {h}%（其中 {n} 位仍在继续）。",
    encDesc: "描述符编码",
    encOnehot: "独热编码",
    startRandom: "随机选取",
    startChemist: "取自化学家",
    na: "无",
    kfFirst: "第一批完成",
    kfHalf: "进行到一半",
    kfEnd: "预算用完",
  },
};


const params = {
  enc: { kind: "choice", label: { en: "Encoding", zh: "编码" }, options: [{ value: "desc", label: { en: "Descriptors", zh: "描述符" } }, { value: "onehot", label: { en: "One-hot", zh: "独热" } }], default: "desc" },
  batch: { kind: "choice", label: { en: "Batch size", zh: "批量大小" }, options: [{ value: 1, label: { en: "1", zh: "1" } }, { value: 5, label: { en: "5", zh: "5" } }, { value: 10, label: { en: "10", zh: "10" } }], default: 5 },
  start: { kind: "choice", label: { en: "First batch", zh: "第一批" }, options: [{ value: "random", label: { en: "Random", zh: "随机" } }, { value: "chemist", label: { en: "A chemist's", zh: "一位化学家的" } }], default: "random" },
  reveal: { kind: "toggle", label: { en: "Reveal all yields", zh: "显示全部产率" }, default: false },
  players: { kind: "toggle", label: { en: "Show each chemist", zh: "显示每位化学家" }, default: true },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 12, step: 1, control: false },
} as const;

type P = { enc: Encoding; batch: number; start: Init; reveal: boolean; players: boolean; seed: number };

const run = memo((enc: Encoding, batch: number, start: Init, seed: number) => {
  const r = boRun(enc, batch, start, BUDGET, seed);
  return { picks: r.picks, fits: r.fits, best: bestSoFar(r.picks), rand: bestSoFar(randomPicks(BUDGET, seed)) };
}, 16);

function describe(st: State<P>): string {
  const p = st.p;
  const t = Math.max(1, Math.min(BUDGET, Math.round(st.t)));
  const R = run(p.enc, p.batch, p.start, p.seed);
  const h = humanMean(t);
  const L = labels[st.lang ?? "en"];
  return tpl(L.describe, {
    t, b: BUDGET, k: p.batch, e: p.enc === "desc" ? L.encDesc : L.encOnehot, s: p.start === "random" ? L.startRandom : L.startChemist,
    o: R.best[t - 1].toFixed(1), r: R.rand[t - 1].toFixed(1), n: h.count, h: Number.isFinite(h.mean) ? h.mean.toFixed(1) : L.na,
  });
}

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 560;
  const t = Math.max(0, Math.min(BUDGET, Math.round(st.t)));
  const R = run(p.enc, p.batch, p.start, p.seed);
  const out: string[] = [];

  // Legend.
  const items: Array<[string, string, Record<string, string | number>]> = [
    [narrow ? L.boShort : L.bo, C.acq, { "stroke-width": 3 }],
    [L.random, C.c7, { "stroke-width": 2 }],
    [narrow ? L.chemistsShort : L.chemists, C.ink, { "stroke-width": 2.2 }],
    [L.each, C.ink3, { "stroke-width": 1, opacity: 0.6 }],
    [narrow ? L.paperShort : L.paper, C.acq, { "stroke-width": 2, "stroke-dasharray": "5 4" }],
  ];
  let lx = narrow ? 8 : 12, ly = 12;
  for (const [name, color, a] of items) {
    const tw = labelWidth(name, 6.2) + 34;
    if (lx + tw > st.w - 8) { lx = narrow ? 8 : 12; ly += 17; }
    out.push(el("line", { x1: lx, x2: lx + 18, y1: ly - 4, y2: ly - 4, stroke: color, ...a }), text(lx + 24, ly, name, { "font-size": TYPE.small, class: "fig-t-muted" }));
    lx += tw;
  }
  const top = ly + 14;

  // The map of all 1,728 reactions.
  const mapW = narrow ? st.w - 8 : Math.floor(st.w * 0.58);
  const m = mapGeom(narrow ? 4 : 6, top + 6, mapW, narrow);
  const tried = new Map<number, number>();
  R.picks.slice(0, t).forEach((i) => tried.set(i, YIELD[i]));
  out.push(yieldMap(m, (i) => (p.reveal ? YIELD[i] : tried.get(i))), mapLabels(m, narrow, lang));
  // The latest batch, ringed; every BO experiment so far, dotted.
  const bIdx = t === 0 ? -1 : Math.floor((t - 1) / p.batch);
  const bStart = bIdx * p.batch, bEnd = Math.min(t, bStart + p.batch);
  R.picks.slice(0, t).forEach((i, k) => {
    const [cx, cy] = cellCenter(m, i);
    const latest = k >= bStart;
    out.push(el("circle", { cx, cy, r: latest ? 4.2 : 1.6, fill: latest ? "none" : C.acq, stroke: latest ? C.acq : "none", "stroke-width": 1.8 }));
  });

  // Under the map: the latest batch and what the model weighs.
  const foot = mapFooter(m, narrow, L.yieldKey, lang);
  out.push(foot.svg);
  let ty = foot.y + 10;
  const tx = narrow ? 8 : m.x0 - m.labelW + 4;
  if (bIdx >= 0) {
    out.push(text(tx, ty, tpl(L.batch, { k: bIdx + 1, n: bEnd - bStart }), { "font-size": TYPE.small, class: "fig-t-strong" }));
    ty += 15;
    const shown = R.picks.slice(bStart, bEnd);
    shown.slice(0, 5).forEach((i) => {
      out.push(text(tx + 8, ty, `${describeReaction(i, lang)}${lang === "zh" ? "：" : ": "}${YIELD[i].toFixed(1)}%`, { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
      ty += 14;
    });
    if (shown.length > 5) { out.push(text(tx + 8, ty, tpl(L.more, { n: shown.length - 5 }), { "font-size": TYPE.small, class: "fig-t-faint" })); ty += 14; }
  }
  ty += 8;
  const fitIdx = bIdx - 1; // the model that chose the latest batch
  if (fitIdx >= 0 && R.fits[fitIdx]) {
    const f = R.fits[fitIdx];
    out.push(text(tx, ty, tpl(L.relevance, { k: bIdx + 1 }), { "font-size": TYPE.small, class: "fig-t-strong" }));
    ty += 8;
    const bw = narrow ? (st.w - 24) / 5 : Math.min(64, (mapW - 20) / 5);
    L.choice.split("|").forEach((name, gi) => {
      const x = tx + gi * bw;
      const v = 1 / f.ls[gi];
      const h = Math.max(1.5, (v / 2) * 28);
      out.push(el("rect", { x: x + 4, y: ty + 30 - h, width: bw - 14, height: h, fill: C.model, rx: 1 }),
        text(x + 4 + (bw - 14) / 2, ty + 44, name, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }));
    });
    ty += 52;
  } else {
    out.push(text(tx, ty, L.waiting, { "font-size": TYPE.small, class: "fig-t-faint" }));
    ty += 14;
  }

  // Best-so-far panel.
  const cx0 = narrow ? 44 : mapW + 52, cx1 = st.w - 12;
  const cy0 = narrow ? ty + 26 : top + 44, ch = narrow ? 170 : m.h - 14;
  const fr = curveFrame({ x0: cx0, x1: cx1, y0: cy0, h: ch, budget: BUDGET, narrow, title: L.best, xTitle: L.exps });
  out.push(fr.svg, references(fr.xs, fr.ys, BUDGET, p.players));
  if (t >= 1) {
    out.push(stepLine(fr.xs, fr.ys, R.rand, t, { stroke: C.c7, "stroke-width": 2 }));
    out.push(stepLine(fr.xs, fr.ys, R.best, t, { stroke: C.acq, "stroke-width": 3 }));
    out.push(el("line", { x1: fr.xs(t), x2: fr.xs(t), y1: cy0, y2: cy0 + ch, stroke: C.ink3, "stroke-dasharray": "2 3" }));
  }
  const H = Math.max(ty + 4, cy0 + ch + 44);
  return svg(st.w, H, describe(st), g({}, ...out));
}

export default defineFigure({
  name: "cs-chem-replay",
  title: { en: "Bayesian optimization replayed on 1,728 measured reactions, against random selection and 50 chemists", zh: "在 1,728 个实测反应上回放贝叶斯优化，并与随机选择和 50 位化学家比较" },
  labels,
  params,
  timeline: {
    rate: 4,
    discrete: true,
    duration: () => BUDGET,
    unit: { symbol: { en: "experiments", zh: "次实验" }, value: (t) => String(Math.round(t)) },
    keyframes: (p, lang) => [{ t: p.batch, label: labels[lang].kfFirst }, { t: 25, label: labels[lang].kfHalf }, { t: BUDGET, label: labels[lang].kfEnd }],
    poster: () => 30,
  },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [{ label: { en: "New run", zh: "新一轮运行" }, run: (p) => ({ ...p, seed: (p.seed % 9999) + 1 }) }],
});
