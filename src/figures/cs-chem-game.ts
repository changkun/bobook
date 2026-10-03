// Play the reaction optimization game of Shields et al. (Nature 2021): choose
// up to five reactions per batch from the 1,728 combinations of ligand, base,
// solvent, concentration, and temperature, "run" them, and see their yields.
// The yields are the measured high-throughput results the 50 chemists and
// engineers of the original game received; nothing is simulated. Your best
// yield so far is drawn against theirs and against the paper's optimizer.
//
// Data: src/figures/data/cs-chem-arylation.json, written by
//   .cache/venv-cases/bin/python tools/figure-data/cs-chem-arylation.py
// from github.com/b-shields/edbo and github.com/b-shields/EvML (MIT License).

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { tpl } from "./lib/format.ts";
import { fmtNumbers, parseNumbers, validNumbers } from "./lib/params.ts";
import { BASES, CONC, LIGANDS, SOLVENTS, TEMP, YIELD, bestSoFar, describeReaction, parts, reaction, solventName } from "./lib/cs-chem.ts";
import { HUMAN_CURVES, cellAt, cellCenter, curveFrame, hitMap, mapFooter, mapGeom, mapLabels, references, rowCol, stepLine, yieldMap } from "./lib/cs-chem-draw.ts";

const BUDGET = 50, BATCH = 5;

const labels = {
  en: {
    you: "you",
    chemists: "the 50 chemists (mean)",
    chemistsShort: "chemists (mean)",
    each: "each chemist",
    paper: "paper's optimizer (mean of 50 runs)",
    paperShort: "paper's BO",
    best: "best yield so far (%)",
    exps: "experiments",
    left: "{n} of {b} experiments left",
    pending: "Next batch ({n} of 5):",
    pendingNone: "Next batch: click up to five cells,|or set the conditions above and press Add.",
    last: "Batch {k} results:",
    spent: "Budget spent.",
    rank: "Your best after {n} experiments: {y}%. {k} of the 50 chemists had a higher best by then.",
    rankNone: "Run a batch to compare with the 50 chemists.",
    yieldKey: "measured yield",
    describeEmpty: "The reaction optimization game: 1,728 reactions, a budget of 50 experiments in batches of five, no experiments run yet.",
    describe: "You have run {n} of 50 experiments; your best measured yield is {y}%. {p} reaction{p:/s} queued for the next batch.",
  },
  zh: {
    you: "你",
    chemists: "50 位化学家（均值）",
    chemistsShort: "化学家（均值）",
    each: "每位化学家",
    paper: "论文的优化器（50 次运行的均值）",
    paperShort: "论文的优化器",
    best: "目前的最高产率（%）",
    exps: "实验次数",
    left: "还剩 {n} 次实验（共 {b} 次）",
    pending: "下一批（已选 {n} 个，最多 5 个）：",
    pendingNone: "下一批：点击最多五个单元格，|或在上方设好条件后按“添加”。",
    last: "第 {k} 批的结果：",
    spent: "预算已用完。",
    rank: "{n} 次实验后你的最高产率：{y}%。50 位化学家中有 {k} 位此时的最高产率高于你。",
    rankNone: "运行一批之后即可与 50 位化学家比较。",
    yieldKey: "实测产率",
    describeEmpty: "反应优化游戏：1,728 个反应，预算为 50 次实验，每批五个，尚未运行任何实验。",
    describe: "你已运行 50 次实验中的 {n} 次，实测最高产率为 {y}%。下一批已排入 {p} 个反应。",
  },
};

const opt = (xs: Array<string | number>, unit = "", zh: Array<string | number> = xs) => xs.map((v, i) => ({ value: i, label: { en: `${v}${unit}`, zh: `${zh[i]}${unit}` } }));

const params = {
  ligand: { kind: "choice", label: { en: "Ligand", zh: "配体" }, options: opt(LIGANDS), default: 0, control: "select" },
  base: { kind: "choice", label: { en: "Base", zh: "碱" }, options: opt(BASES), default: 0, control: "select" },
  solvent: { kind: "choice", label: { en: "Solvent", zh: "溶剂" }, options: opt(SOLVENTS, "", SOLVENTS.map((_, i) => solventName(i, "zh"))), default: 0, control: "select" },
  conc: { kind: "choice", label: { en: "Concentration", zh: "浓度" }, options: opt(CONC, " M"), default: 1 },
  temp: { kind: "choice", label: { en: "Temperature", zh: "温度" }, options: opt(TEMP, " °C"), default: 1 },
  reveal: { kind: "toggle", label: { en: "Reveal all yields", zh: "显示全部产率" }, default: false },
  players: { kind: "toggle", label: { en: "Show each chemist", zh: "显示每位化学家" }, default: true },
  done: { kind: "data", label: { en: "Experiments run", zh: "已运行的实验" }, default: "", validate: validNumbers },
  pending: { kind: "data", label: { en: "Next batch", zh: "下一批" }, default: "", validate: validNumbers },
} as const;

type P = { ligand: number; base: number; solvent: number; conc: number; temp: number; reveal: boolean; players: boolean; done: string; pending: string };

const list = (s: string) => parseNumbers(s).map((v) => Math.round(v)).filter((v) => v >= 0 && v < YIELD.length);
const selected = (p: P) => reaction(p.ligand, p.base, p.solvent, p.conc, p.temp);
const room = (p: P) => Math.min(BATCH, BUDGET - list(p.done).length);

function toggle(p: P, i: number): P {
  const done = list(p.done), pend = list(p.pending);
  if (done.includes(i)) return p;
  const k = pend.indexOf(i);
  if (k >= 0) pend.splice(k, 1);
  else if (pend.length < room(p)) pend.push(i);
  else return p;
  const [l, b, s, c, t] = parts(i);
  return { ...p, pending: fmtNumbers(pend), ligand: l, base: b, solvent: s, conc: c, temp: t };
}

function runBatch(p: P): P {
  const done = list(p.done), pend = list(p.pending);
  return { ...p, done: fmtNumbers([...done, ...pend].slice(0, BUDGET)), pending: "" };
}

function describe(st: State<P>): string {
  const L = labels[st.lang ?? "en"];
  const done = list(st.p.done), pend = list(st.p.pending);
  if (!done.length) return L.describeEmpty;
  return tpl(L.describe, { n: done.length, y: Math.max(...done.map((i) => YIELD[i])).toFixed(1), p: pend.length });
}

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 560;
  const done = list(p.done), pend = list(p.pending);
  const out: string[] = [];

  // Legend.
  const items: Array<[string, string, Record<string, string | number>]> = [
    [L.you, C.c5, { "stroke-width": 3 }],
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

  // Map.
  const mapW = narrow ? st.w - 8 : Math.floor(st.w * 0.58);
  const m = mapGeom(narrow ? 4 : 6, top + 6, mapW, narrow);
  const seen = new Map<number, number>();
  done.forEach((i) => seen.set(i, YIELD[i]));
  out.push(yieldMap(m, (i) => (p.reveal ? YIELD[i] : seen.get(i))), mapLabels(m, narrow, lang), hitMap(m, "map"));
  done.forEach((i) => { const [cx, cy] = cellCenter(m, i); out.push(el("circle", { cx, cy, r: 1.5, fill: C.c5 })); });
  pend.forEach((i) => { const [cx, cy] = cellCenter(m, i); out.push(el("circle", { cx, cy, r: 4.4, fill: "none", stroke: C.c5, "stroke-width": 2 })); });
  {
    const [r, c] = rowCol(selected(p));
    out.push(el("rect", { x: m.x0 + c * m.cw - 1, y: m.y0 + r * m.ch - 1, width: m.cw + 2, height: m.ch + 2, fill: "none", stroke: C.ink2, "stroke-width": 1.3 }));
  }

  // Under the map: status, the queued batch, the last results.
  const foot = mapFooter(m, narrow, L.yieldKey, lang);
  out.push(foot.svg);
  let ty = foot.y + 10;
  const tx = narrow ? 8 : m.x0 - m.labelW + 4;
  const line = (s: string, cls = "fig-t-muted", dx = 0) => { out.push(text(tx + dx, ty, s, { "font-size": TYPE.small, class: cls })); ty += 14; };
  line(done.length >= BUDGET ? L.spent : tpl(L.left, { n: BUDGET - done.length, b: BUDGET }), "fig-t-strong");
  if (done.length < BUDGET) {
    if (pend.length) line(tpl(L.pending, { n: pend.length }));
    else for (const s of L.pendingNone.split("|")) line(s);
    pend.forEach((i) => line(describeReaction(i, lang), "fig-t-muted", 8));
  }
  if (done.length) {
    ty += 4;
    const k = Math.ceil(done.length / BATCH);
    const lastStart = (k - 1) * BATCH;
    line(tpl(L.last, { k }), "fig-t-strong");
    done.slice(lastStart).forEach((i) => line(`${describeReaction(i, lang)}${lang === "zh" ? "：" : ": "}${YIELD[i].toFixed(1)}%`, "fig-t-muted fig-t-num", 8));
  }

  // Curves.
  const cx0 = narrow ? 44 : mapW + 52, cx1 = st.w - 12;
  const cy0 = narrow ? ty + 30 : top + 44, ch = narrow ? 170 : m.h - 14;
  const fr = curveFrame({ x0: cx0, x1: cx1, y0: cy0, h: ch, budget: BUDGET, narrow, title: L.best, xTitle: L.exps });
  out.push(fr.svg, references(fr.xs, fr.ys, BUDGET, p.players));
  const mine = bestSoFar(done);
  if (mine.length) out.push(stepLine(fr.xs, fr.ys, mine, mine.length, { stroke: C.c5, "stroke-width": 3 }));
  // Rank among the chemists at the same number of experiments.
  const n = done.length;
  let rank: string;
  if (n) {
    // a chemist who stopped earlier keeps their final best
    const others = HUMAN_CURVES.map((c) => c[Math.min(n, c.length) - 1]);
    rank = tpl(L.rank, { n, y: mine[n - 1].toFixed(1), k: others.filter((v) => v > mine[n - 1]).length });
  } else rank = L.rankNone;
  const ry = cy0 + ch + 54;
  // wrap the rank sentence into two lines on narrow layouts
  const sep = lang === "zh" ? "。" : ". ";
  const cut = rank.indexOf(sep);
  if (cut > 0 && cut + sep.length < rank.length) {
    out.push(text(narrow ? 8 : cx0 - 40, ry, rank.slice(0, cut + 1), { "font-size": TYPE.small, class: "fig-t-strong" }),
      text(narrow ? 8 : cx0 - 40, ry + 15, rank.slice(cut + sep.length), { "font-size": TYPE.small, class: "fig-t-muted" }));
  } else out.push(text(narrow ? 8 : cx0 - 40, ry, rank, { "font-size": TYPE.small, class: "fig-t-muted" }));
  const H = Math.max(ty + 4, ry + 22);
  return svg(st.w, H, describe(st), g({}, ...out));
}

export default defineFigure({
  name: "cs-chem-game",
  title: { en: "The reaction optimization game: find the best of 1,728 measured reactions in batches of five", zh: "反应优化游戏：每批五个，在 1,728 个实测反应中找出最好的一个" },
  labels,
  params,
  hint: { en: "Click cells to queue up to five reactions (or choose the five conditions and press Add), then press Run batch. Each experiment returns the measured yield.", zh: "点击单元格，把最多五个反应排入下一批（或选好五个条件后按“添加”），然后按“运行这一批”。每次实验返回实测产率。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click" || e.target !== "map" || !e.data) return null;
    const col = Math.floor(e.data.x), row = Math.floor(e.data.y);
    if (col < 0 || col >= 36 || row < 0 || row >= 48) return null;
    if (list(p.done).length >= BUDGET) return null;
    return toggle(p as P, cellAt(row, col));
  },
  actions: [
    { label: { en: "Add", zh: "添加" }, run: (p) => toggle(p as P, selected(p as P)), enabled: (p) => list(p.done).length < BUDGET && list(p.pending).length < room(p as P) && !list(p.done).includes(selected(p as P)) && !list(p.pending).includes(selected(p as P)) },
    { label: { en: "Run batch", zh: "运行这一批" }, primary: true, run: (p) => runBatch(p as P), enabled: (p) => list(p.pending).length > 0 },
    { label: { en: "Remove last", zh: "移除最后一个" }, run: (p) => ({ ...p, pending: fmtNumbers(list(p.pending).slice(0, -1)) }), enabled: (p) => list(p.pending).length > 0 },
    { label: { en: "Start over", zh: "重新开始" }, run: (p) => ({ ...p, done: "", pending: "" }), enabled: (p) => p.done !== "" || p.pending !== "" },
  ],
});
