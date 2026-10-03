// You are the oracle, in more dimensions. Two generated designs with 3, 6,
// or 10 parameters are shown; choose the one you prefer. A Gaussian process
// preference model learns a utility over the whole design space, EUBO picks
// the next pair from a few hundred candidates, and the panels show the
// model's best guess so far, how the learned utility changes along each
// parameter through that guess, and, from recorded simulations, how many
// comparisons a model needs as the number of parameters grows. In the
// simulated mode a hidden utility answers, so the guess can be scored.

import { defineFigure, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { predictPreference } from "./lib/gp.ts";
import { memo } from "./lib/random.ts";
import { DESIGN_PARAMS, designCard } from "./lib/design-card.ts";
import { hidden, normRegret, simulatedAnswer, step } from "./lib/oracle-nd.ts";
import { legend } from "./lib/plot.ts";
import { fixed, tpl } from "./lib/format.ts";
import data from "./data/dim-oracle-nd.json" with { type: "json" };

const Q = data.quartiles as unknown as Record<string, Record<string, number[][]>>;

const labels = {
  en: {
    pick: "Which design do you prefer?",
    simPick: "The simulated person is asked:",
    guess: "best guess",
    favorite: "hidden favorite",
    effects: "learned utility along each parameter, through the best guess",
    curves: "recorded: how close the best guess gets (median of 24 simulated people)",
    regret: "remaining gap",
    comps: "comparisons",
    you: "this session",
    status: "{n} comparison{n:/s} · {d} parameters",
    scored: "gap to favorite {g} (1 = a random design, 0 = the favorite)",
    describe: "Preferential optimization over designs with {d} parameters after {n} comparisons.{s}",
    describeSim: " The best guess closes {p}% of the gap between a random design and the hidden favorite.",
    nParams: "{d} parameters",
    randomPairs: "random pairs",
  },
  zh: {
    pick: "你更喜欢哪个设计？",
    simPick: "模拟用户被问到：",
    guess: "最佳猜测",
    favorite: "隐藏的最爱",
    effects: "经过最佳猜测、沿每个参数学到的效用",
    curves: "记录：最佳猜测的接近程度（24 个模拟用户的中位数）",
    regret: "剩余差距",
    comps: "比较次数",
    you: "本次会话",
    status: "{n} 次比较 · {d} 个参数",
    scored: "与最爱的差距 {g}（1 = 随机设计，0 = 最爱）",
    describe: "在有 {d} 个参数的设计上做偏好优化，已进行 {n} 次比较。{s}",
    describeSim: "最佳猜测已缩小随机设计与隐藏最爱之间 {p}% 的差距。",
    nParams: "{d} 个参数",
    randomPairs: "随机对",
  },
};

// The names of DESIGN_PARAMS (lib/design-card.ts) in the Chinese edition.
const PARAM_NAMES_ZH = ["背景色相", "圆角程度", "形状大小", "饱和度", "条纹", "旋转", "亮度", "强调色色相", "形状位置", "形状数量"];

const params = {
  dims: { kind: "choice", label: { en: "Parameters", zh: "参数个数" }, options: [{ value: 3, label: { en: "3", zh: "3" } }, { value: 6, label: { en: "6", zh: "6" } }, { value: 10, label: { en: "10", zh: "10" } }], default: 3 },
  mode: { kind: "choice", label: { en: "Who answers", zh: "由谁回答" }, options: [{ value: "you", label: { en: "You", zh: "你" } }, { value: "sim", label: { en: "Simulated person", zh: "模拟用户" } }], default: "you" },
  acq: { kind: "choice", label: { en: "Next pair", zh: "下一对" }, options: [{ value: "eubo", label: { en: "EUBO", zh: "EUBO" } }, { value: "random", label: { en: "Random", zh: "随机" } }], default: "eubo" },
  answers: { kind: "data", label: { en: "Answers", zh: "回答" }, default: "" },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 3, step: 1, control: false },
} as const;

type P = { dims: number; mode: string; acq: string; answers: string; seed: number };

// Answers are stored as "w1 w2 ...|l1 l2 ..." pairs separated by ";".
const r2 = (v: number) => Math.round(v * 100) / 100;
function parsePairs(s: string): Array<[number[], number[]]> {
  return s ? s.split(";").map((p) => p.split("|").map((v) => v.split(" ").map(Number)) as [number[], number[]]) : [];
}
function fmtPairs(ps: Array<[number[], number[]]>): string {
  return ps.map(([a, b]) => `${a.map(r2).join(" ")}|${b.map(r2).join(" ")}`).join(";");
}

const stepMemo = memo((D: number, answers: string, seed: number, acq: string) => step(D, parsePairs(answers), seed, acq as "eubo" | "random"), 16);

function answer(p: P, choice: 0 | 1): P {
  const st = stepMemo(p.dims, p.answers, p.seed, p.acq);
  const ps = parsePairs(p.answers);
  ps.push(choice === 0 ? [st.pair[0], st.pair[1]] : [st.pair[1], st.pair[0]]);
  return { ...p, answers: fmtPairs(ps.slice(-40)) };
}

function simulate(p: P, k: number): P {
  let q = p;
  for (let i = 0; i < k; i++) {
    const st = stepMemo(q.dims, q.answers, q.seed, q.acq);
    const ps = parsePairs(q.answers);
    if (ps.length >= 40) break;
    ps.push(simulatedAnswer(q.dims, q.seed, 0.1, st.pair, ps.length));
    q = { ...q, answers: fmtPairs(ps) };
  }
  return q;
}

function describe(st: State<P>): string {
  const p = st.p;
  const n = parsePairs(p.answers).length;
  const s = stepMemo(p.dims, p.answers, p.seed, p.acq);
  const L = labels[st.lang ?? "en"];
  const sim = p.mode === "sim" ? tpl(L.describeSim, { p: Math.round(100 * (1 - Math.min(1, normRegret(p.dims, p.seed, s.best)))) }) : "";
  return tpl(L.describe, { d: p.dims, n, s: sim });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 560;
  const D = p.dims;
  const s = stepMemo(D, p.answers, p.seed, p.acq);
  const pairs = parsePairs(p.answers);
  const parts: string[] = [];
  // the question
  parts.push(text(st.w / 2, 18, p.mode === "you" ? L.pick : L.simPick, { "text-anchor": "middle", "font-size": TYPE.label, class: "fig-t-strong" }));
  const cw = narrow ? (st.w - 36) / 2 : Math.min(190, (st.w - 80) / 2), ch = cw * 0.72;
  const gap = narrow ? 12 : 30;
  const ax = st.w / 2 - cw - gap / 2, bx = st.w / 2 + gap / 2, cy = 30;
  parts.push(designCard(s.pair[0], ax, cy, cw, ch, st.uid, "A"), designCard(s.pair[1], bx, cy, cw, ch, st.uid, "B"));
  parts.push(text(ax + 10, cy + 20, "A", { "font-size": TYPE.title, class: "fig-t-strong", style: "fill:rgba(255,255,255,.95)" }), text(bx + 10, cy + 20, "B", { "font-size": TYPE.title, class: "fig-t-strong", style: "fill:rgba(255,255,255,.95)" }));
  // best guess (and favorite)
  const y1 = cy + ch + 26;
  const sw = narrow ? 76 : 92, sh = sw * 0.72;
  parts.push(text(8, y1 + sh + 30, tpl(L.status, { n: pairs.length, d: D }), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  parts.push(designCard(s.best, 8, y1, sw, sh, st.uid), text(8 + sw / 2, y1 + sh + 14, L.guess, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }));
  const h = hidden(D, p.seed);
  if (p.mode === "sim") {
    parts.push(designCard(h.c, 16 + sw, y1, sw, sh, st.uid), text(16 + sw + sw / 2, y1 + sh + 14, L.favorite, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }));
    parts.push(text(8, y1 + sh + 46, tpl(L.scored, { g: fixed(normRegret(D, p.seed, s.best), 2) }), { "font-size": TYPE.small, class: "fig-t-num" }));
  }
  // effect plots: utility along each parameter through the best guess
  const ex0 = narrow ? 8 : 24 + (p.mode === "sim" ? 2 * sw + 16 : Math.max(sw + 8, 170)), ey0 = narrow ? y1 + sh + (p.mode === "sim" ? 66 : 50) : y1;
  const cols = narrow ? (D > 6 ? 4 : 3) : D > 6 ? 5 : 3;
  const ew = ((narrow ? st.w - 16 : st.w - ex0 - 8) - (cols - 1) * 8) / cols, eh = 40;
  parts.push(text(ex0, ey0 - 6, L.effects, { "font-size": TYPE.small, class: "fig-t-muted" }));
  const line = grid(0, 1, 21);
  const curves = DESIGN_PARAMS.slice(0, D).map((_, d) => predictPreference(s.fitted, line.map((v) => { const q = s.best.slice(); q[d] = v; return q; })).mean);
  const all = curves.flat();
  const lo = Math.min(...all), hi = Math.max(...all) + 1e-9;
  const names: readonly string[] = st.lang === "zh" ? PARAM_NAMES_ZH : DESIGN_PARAMS;
  names.slice(0, D).forEach((name, d) => {
    const cx = ex0 + (d % cols) * (ew + 8), cyy = ey0 + 4 + Math.floor(d / cols) * (eh + 22);
    const xs = linear([0, 1], [cx, cx + ew]), ys = linear([lo, hi], [cyy + eh, cyy + 4]);
    parts.push(
      el("rect", { x: cx, y: cyy, width: ew, height: eh, fill: C.panel, rx: 3 }),
      el("path", { d: linePath(line.map((v, i) => [xs(v), ys(curves[d][i])])), fill: "none", stroke: C.model, "stroke-width": 1.6 }),
      el("line", { x1: xs(s.best[d]), x2: xs(s.best[d]), y1: cyy, y2: cyy + eh, stroke: C.acq, "stroke-width": 1.2 }),
      p.mode === "sim" ? el("path", { d: `M${xs(h.c[d])},${cyy + eh}l-4,6h8z`, fill: C.truth }) : "",
      text(cx + 2, cyy + eh + 13, name, { "font-size": TYPE.small, class: "fig-t-muted" }),
    );
  });
  const rows = Math.ceil(D / cols);
  // recorded curves
  const top3 = Math.max(ey0 + 4 + rows * (eh + 22) + 36, y1 + sh + (p.mode === "sim" ? 72 : 56));
  const lg = legend(narrow ? 8 : 40, top3, st.w - 16, [
    ...[3, 6, 10].map((d, i) => ({ kind: "line" as const, color: [C.c1, C.c6, C.c5][i], label: tpl(L.nParams, { d }) })),
    { kind: "dash" as const, color: C.ink3, label: L.randomPairs },
    ...(p.mode === "sim" ? [{ kind: "dot" as const, color: C.acq, label: L.you }] : []),
  ]);
  const t3 = top3 + lg.height + 14, h3 = narrow ? 120 : 130;
  const xs3 = linear([0, 40], [40, st.w - 10]), ys3 = linear([0, 1.05], [t3 + h3, t3]);
  parts.push(lg.svg, text(40, top3 - 10, L.curves, { "font-size": TYPE.small, class: "fig-t-muted" }),
    axis({ scale: xs3, orient: "bottom", at: t3 + h3, span: [t3, t3 + h3], title: L.comps, count: 4 }),
    axis({ scale: ys3, orient: "left", at: 40, span: [40, st.w - 10], count: 2 }),
    text(46, t3 + 10, L.regret, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));
  [3, 6, 10].forEach((d, i) => {
    const col = [C.c1, C.c6, C.c5][i];
    const on = d === D;
    for (const acq of ["eubo", "random"]) {
      const qv = Q[d][acq];
      const isSel = acq === p.acq;
      if (on && isSel) parts.push(el("path", { d: linePath(qv.map((q, k) => [xs3(k), ys3(q[2])])) + qv.slice().reverse().map((q, k) => `L${xs3(40 - k).toFixed(1)},${ys3(q[0]).toFixed(1)}`).join("") + "Z", fill: col, opacity: 0.12 }));
      parts.push(el("path", { d: linePath(qv.map((q, k) => [xs3(k), ys3(q[1])])), fill: "none", stroke: col, "stroke-width": on ? 2.4 : 1.2, opacity: on ? 1 : 0.45, "stroke-dasharray": acq === "random" ? "5 4" : undefined }));
    }
  });
  if (p.mode === "sim" && pairs.length) {
    // this session's trajectory
    const traj: number[] = [];
    for (let k = 0; k <= pairs.length; k++) traj.push(normRegret(D, p.seed, stepMemo(D, fmtPairs(pairs.slice(0, k)), p.seed, p.acq).best));
    parts.push(el("path", { d: linePath(traj.map((v, k) => [xs3(k), ys3(Math.min(1.05, v))])), fill: "none", stroke: C.acq, "stroke-width": 1.6 }),
      el("circle", { cx: xs3(pairs.length), cy: ys3(Math.min(1.05, traj[traj.length - 1])), r: 4, fill: C.acq }));
  }
  return svg(st.w, t3 + h3 + 40, describe(st), g({}, ...parts));
}

export default defineFigure({
  name: "dim-oracle-nd",
  title: { en: "You are the oracle, with more design parameters", zh: "你就是预言机：更多的设计参数" },
  labels,
  params,
  hint: { en: "Click the design you prefer. The small multiples show what the model has learned about each parameter.", zh: "点击你更喜欢的设计。各个小图显示模型对每个参数学到了什么。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click" || p.mode !== "you") return null;
    if (e.target === "A") return answer(p as P, 0);
    if (e.target === "B") return answer(p as P, 1);
    return null;
  },
  update(p, key) {
    if (key === "dims" || key === "mode") return { ...p, answers: "" };
    return p;
  },
  actions: [
    { label: { en: "Prefer A", zh: "更喜欢 A" }, primary: true, run: (p) => answer(p as P, 0), enabled: (p) => p.mode === "you" },
    { label: { en: "Prefer B", zh: "更喜欢 B" }, primary: true, run: (p) => answer(p as P, 1), enabled: (p) => p.mode === "you" },
    { label: { en: "Simulate ten", zh: "模拟十次" }, primary: true, run: (p) => simulate(p as P, 10), enabled: (p) => p.mode === "sim" },
    { label: { en: "Undo", zh: "撤销" }, run: (p) => ({ ...p, answers: fmtPairs(parsePairs(p.answers).slice(0, -1)) }), enabled: (p) => p.answers !== "" },
    { label: { en: "New person", zh: "换一个模拟用户" }, run: (p) => ({ ...p, seed: (p.seed % 9999) + 1, answers: "" }), enabled: (p) => p.mode === "sim" },
  ],
});
