// How much of a set of comparisons can one utility explain? Five options A to
// E have utilities 1, 0.5, 0, -0.5, -1, and a rock-paper-scissors component of
// strength kappa is added among A, B, and C (A over B, B over C, C over A).
// Every pair is asked n times under a Bradley-Terry (logistic) model, and each
// edge carries the empirical log-odds of its outcomes. The Hodge decomposition
// splits this edge flow into a gradient part (differences of one score per
// option, fitted by least squares: HodgeRank) and a cyclic part (on the
// complete graph of five options, all of it is local curl around triangles).
// Sampling noise alone produces some cyclic share, so the figure also shows
// the share a purely transitive person (kappa = 0) produces with the same n,
// averaged over 200 simulated sessions.

import { defineFigure, type State } from "./types.ts";
import { el, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { memo, rng } from "./lib/random.ts";
import { sigmoid } from "./lib/stats.ts";
import { fixed, pct, tpl } from "./lib/format.ts";

const labels = {
  en: {
    graph: "answers on every pair",
    split: "share of the comparison flow",
    gradient: "explained by one utility",
    cyclic: "cyclic",
    floor: "chance level for a transitive person",
    scores: "utility per option",
    truthScore: "transitive part of the truth",
    fitted: "HodgeRank score",
    cycleNote: "majority cycle: A beats B, B beats C, C beats A",
    key1: "arrow from the option chosen more often;",
    key2: "width: |log-odds|; magenta: the cyclic residual",
    noCycle: "majorities are transitive",
    describe: "Cycle strength {k}, {n}. The gradient part explains {g} of the comparison flow and the cyclic part {c}{floor}. {cyc}",
    floorText: "; a transitive person with the same number of answers shows about {f} cyclic share by chance",
    exact: "exact win probabilities",
    perPair: "{n} answer{n:/s} per pair",
    cycYes: "The majorities form a cycle A, B, C.",
    cycNo: "The majorities are transitive.",
    floorLine: "{floor}: {f} cyclic",
  },
  zh: {
    graph: "每一对上的回答",
    split: "比较流的构成",
    gradient: "由一个效用解释",
    cyclic: "循环",
    floor: "传递性的人的偶然水平",
    scores: "各选项的效用",
    truthScore: "真值的传递部分",
    fitted: "HodgeRank 得分",
    cycleNote: "多数偏好构成循环：A 胜 B，B 胜 C，C 胜 A",
    key1: "箭头从被选中次数较多的选项指出；",
    key2: "宽度：|对数几率|；品红色：循环残差",
    noCycle: "多数偏好是传递的",
    describe: "循环强度 {k}，{n}。梯度部分解释了比较流的 {g}，循环部分占 {c}{floor}。{cyc}",
    floorText: "；一个传递性的人在回答数相同时，仅凭偶然就会显示约 {f} 的循环比例",
    exact: "精确的获胜概率",
    perPair: "每对 {n} 个回答",
    cycYes: "多数偏好构成循环 A、B、C。",
    cycNo: "多数偏好是传递的。",
    floorLine: "{floor}：循环 {f}",
  },
};

const NS = [
  { value: 1, label: { en: "1", zh: "1" } },
  { value: 5, label: { en: "5", zh: "5" } },
  { value: 20, label: { en: "20", zh: "20" } },
  { value: 100, label: { en: "100", zh: "100" } },
  { value: 0, label: { en: "exact", zh: "精确" } },
] as const;

const params = {
  kappa: { kind: "range", label: { en: "Cycle strength κ among A, B, C", zh: "A、B、C 的循环强度 κ" }, min: 0, max: 2, default: 0.6, step: 0.05 },
  n: { kind: "choice", label: { en: "Answers per pair", zh: "每对的回答数" }, options: NS, default: 0, control: "buttons" },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 4, step: 1, control: false },
} as const;

type P = { kappa: number; n: number; seed: number };

const K = 5;
const NAMES = ["A", "B", "C", "D", "E"];
const U = [1, 0.5, 0, -0.5, -1];
const PAIRS: Array<[number, number]> = [];
for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) PAIRS.push([i, j]);
// The rock-paper-scissors flow: +1 on A>B, B>C, and C>A (that is, -1 on A>C).
const CYC = (i: number, j: number) => (i === 0 && j === 1) || (i === 1 && j === 2) ? 1 : i === 0 && j === 2 ? -1 : 0;
const truthFlow = (kappa: number) => PAIRS.map(([i, j]) => U[i] - U[j] + kappa * CYC(i, j));

// Observed flow: exact log-odds, or empirical log-odds of n answers with a
// continuity correction of one half.
function observe(kappa: number, n: number, r: () => number): number[] {
  const y = truthFlow(kappa);
  if (!n) return y;
  return y.map((v) => {
    const p = sigmoid(v);
    let w = 0;
    for (let t = 0; t < n; t++) if (r() < p) w++;
    return Math.log((w + 0.5) / (n - w + 0.5));
  });
}

// On the complete graph the least-squares scores are s_i = (1/K) sum_j Y_ij.
function decompose(y: number[]) {
  const M = Array.from({ length: K }, () => new Array<number>(K).fill(0));
  PAIRS.forEach(([i, j], e) => { M[i][j] = y[e]; M[j][i] = -y[e]; });
  const s = M.map((row) => row.reduce((a, b) => a + b, 0) / K);
  const grad = PAIRS.map(([i, j]) => s[i] - s[j]);
  const curl = y.map((v, e) => v - grad[e]);
  const tot = y.reduce((a, v) => a + v * v, 0) || 1;
  const gs = grad.reduce((a, v) => a + v * v, 0) / tot;
  return { s, grad, curl, gradShare: gs, curlShare: 1 - gs };
}

const noiseFloor = memo((n: number): number => {
  if (!n) return 0;
  const R = 200;
  let acc = 0;
  for (let k = 0; k < R; k++) acc += decompose(observe(0, n, rng(9973 * k + 7))).curlShare;
  return acc / R;
}, 8);

const compute = memo((kappa: number, n: number, seed: number) => {
  const y = observe(kappa, n, rng(seed * 7919 + n));
  const d = decompose(y);
  const majCycle = y[0] > 0 && y[4] > 0 && y[1] < 0; // A>B, B>C, C>A by majority
  return { y, ...d, floor: noiseFloor(n), majCycle };
}, 32);

function describe(st: State<P>): string {
  const p = st.p;
  const c = compute(p.kappa, p.n, p.seed);
  const L = labels[st.lang ?? "en"];
  return tpl(L.describe, {
    k: fixed(p.kappa, 2), n: p.n ? tpl(L.perPair, { n: p.n }) : L.exact, g: pct(c.gradShare), c: pct(c.curlShare),
    floor: p.n ? tpl(L.floorText, { f: pct(c.floor) }) : "",
    cyc: c.majCycle ? L.cycYes : L.cycNo,
  });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const c = compute(p.kappa, p.n, p.seed);
  const parts: string[] = [];

  // The complete graph on a pentagon. Arrows point from the majority winner to
  // the loser; width shows |log-odds|; color shows the cyclic residual.
  const gw = narrow ? st.w : 300;
  const cx = gw / 2, cy = 34 + 112;
  const R = narrow ? 92 : 100;
  parts.push(text(narrow ? 8 : 12, 16, L.graph, { "font-size": TYPE.label, class: "fig-t-strong" }));
  const pos = Array.from({ length: K }, (_, i) => {
    const a = -Math.PI / 2 + (2 * Math.PI * i) / K;
    return [cx + R * Math.cos(a), cy + R * Math.sin(a)] as [number, number];
  });
  PAIRS.forEach(([i, j], e) => {
    const v = c.y[e];
    const [w, l] = v >= 0 ? [i, j] : [j, i];
    const [x1, y1] = pos[w], [x2, y2] = pos[l];
    const dx = x2 - x1, dy = y2 - y1, d = Math.hypot(dx, dy);
    const ux = dx / d, uy = dy / d;
    const res = Math.min(1, Math.abs(c.curl[e]) / 1.2);
    const width = 1 + Math.min(3.5, Math.abs(v) * 1.4);
    const sx = x1 + ux * 15, sy0 = y1 + uy * 15;
    const tx = x2 - ux * 15, ty = y2 - uy * 15;
    const hx = tx - ux * 9, hy = ty - uy * 9; // base of the arrowhead
    const head = `M${tx},${ty}L${hx - uy * 4.5},${hy + ux * 4.5}L${hx + uy * 4.5},${hy - ux * 4.5}Z`;
    for (const [col, op] of [[C.ink3, 0.8], [C.c5, res]] as const) {
      if (op < 0.04) continue;
      parts.push(el("line", { x1: sx, y1: sy0, x2: hx, y2: hy, stroke: col, "stroke-width": width, opacity: op }));
      parts.push(el("path", { d: head, fill: col, opacity: op }));
    }
  });
  pos.forEach(([x, y], i) => {
    parts.push(el("circle", { cx: x, cy: y, r: 13, fill: C.panel, stroke: C.ink2, "stroke-width": 1.4 }));
    parts.push(text(x, y + 4.5, NAMES[i], { "text-anchor": "middle", "font-size": TYPE.label, class: "fig-t-strong" }));
  });
  const ny = cy + R + 34;
  parts.push(text(narrow ? 8 : 12, ny, L.key1, { "font-size": TYPE.small, class: "fig-t-muted" }));
  parts.push(text(narrow ? 8 : 12, ny + 15, L.key2, { "font-size": TYPE.small, class: "fig-t-muted" }));
  parts.push(text(narrow ? 8 : 12, ny + 34, c.majCycle ? L.cycleNote : L.noCycle, { "font-size": TYPE.small, class: c.majCycle ? "fig-t-strong" : "fig-t-muted" }));
  const graphBottom = ny + 40;

  // The split of the flow, with the chance level.
  const px0 = narrow ? 8 : gw + 16;
  const px1 = st.w - 12;
  let py = narrow ? graphBottom + 26 : 16;
  parts.push(text(px0, py, L.split, { "font-size": TYPE.label, class: "fig-t-strong" }));
  py += 14;
  const bx = linear([0, 1], [px0, px1]);
  parts.push(el("rect", { x: bx(0), y: py, width: bx(c.gradShare) - bx(0), height: 18, fill: C.model }));
  parts.push(el("rect", { x: bx(c.gradShare), y: py, width: bx(1) - bx(c.gradShare), height: 18, fill: C.c5 }));
  if (p.n) {
    const fx = bx(1 - c.floor);
    parts.push(el("line", { x1: fx, x2: fx, y1: py - 4, y2: py + 22, stroke: C.ink, "stroke-width": 1.6, "stroke-dasharray": "3 2" }));
  }
  py += 34;
  parts.push(el("rect", { x: px0, y: py - 9, width: 10, height: 10, fill: C.model }));
  parts.push(text(px0 + 15, py, `${L.gradient} ${pct(c.gradShare)}`, { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
  py += 16;
  parts.push(el("rect", { x: px0, y: py - 9, width: 10, height: 10, fill: C.c5 }));
  parts.push(text(px0 + 15, py, `${L.cyclic} ${pct(c.curlShare)}`, { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
  if (p.n) {
    py += 16;
    parts.push(el("line", { x1: px0 + 5, x2: px0 + 5, y1: py - 10, y2: py + 1, stroke: C.ink, "stroke-width": 1.6, "stroke-dasharray": "3 2" }));
    parts.push(text(px0 + 15, py, tpl(L.floorLine, { floor: L.floor, f: pct(c.floor) }), { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
  }

  // Scores: HodgeRank against the transitive part of the truth.
  py += 32;
  parts.push(text(px0, py, L.scores, { "font-size": TYPE.label, class: "fig-t-strong" }));
  const sTop = py + 12, sH = 112;
  const sy = linear([-1.6, 1.6], [sTop + sH, sTop]);
  const colW = (px1 - px0 - 24) / K;
  parts.push(el("line", { x1: px0 + 18, x2: px1, y1: sy(0), y2: sy(0), stroke: C.rule }));
  for (const t of [-1, 0, 1]) parts.push(text(px0 + 12, sy(t) + 4, String(t).replace("-", "−"), { "text-anchor": "end", "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
  for (let i = 0; i < K; i++) {
    const x = px0 + 24 + colW * (i + 0.5);
    parts.push(el("circle", { cx: x, cy: sy(U[i]), r: 6, fill: "none", stroke: C.truth, "stroke-width": 2 }));
    parts.push(el("circle", { cx: x, cy: sy(Math.max(-1.6, Math.min(1.6, c.s[i]))), r: 4.2, fill: C.model, stroke: C.paper, "stroke-width": 1.4 }));
    parts.push(text(x, sTop + sH + 16, NAMES[i], { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }));
  }
  py = sTop + sH + 36;
  parts.push(el("circle", { cx: px0 + 5, cy: py - 4, r: 5, fill: "none", stroke: C.truth, "stroke-width": 2 }));
  parts.push(text(px0 + 15, py, L.truthScore, { "font-size": TYPE.small, class: "fig-t-muted" }));
  py += 16;
  parts.push(el("circle", { cx: px0 + 5, cy: py - 4, r: 4, fill: C.model }));
  parts.push(text(px0 + 15, py, L.fitted, { "font-size": TYPE.small, class: "fig-t-muted" }));
  const H = Math.max(py + 12, narrow ? 0 : graphBottom + 4);
  return svg(st.w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "nat-hodge",
  title: { en: "How much of a set of comparisons one utility can explain", zh: "一个效用能解释多少比较" },
  labels,
  params,
  hint: { en: "Raise the cycle strength and watch the cyclic share; compare it with the chance level for the same number of answers.", zh: "调高循环强度，观察循环比例；把它与相同回答数下的偶然水平比较。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "Draw new answers", zh: "重新抽取回答" }, primary: true, run: (p) => ({ ...p, seed: (Number(p.seed) % 9999) + 1 }), enabled: (p) => p.n !== 0 },
    { label: { en: "No cycle", zh: "无循环" }, run: (p) => ({ ...p, kappa: 0 }), enabled: (p) => p.kappa !== 0 },
  ],
});
