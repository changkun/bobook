// Learning a utility from comparisons on the running objective. The reader
// chooses pairs of inputs by clicking twice on the plot (or with the buttons);
// either a simulated person whose utility is the running objective answers,
// with probit noise, or the reader answers. The model is the Gaussian process
// preference model of Chu and Ghahramani, fitted with the Laplace
// approximation (lib/gp.ts fitPreference): the curve is the posterior mean of
// the latent utility with a 95% band. Above the plot, the comparison graph:
// one arc per answered pair, colored by connected component, with the winner
// marked. The objective is drawn shifted and scaled to fit the posterior mean,
// because comparisons identify a utility only up to shift and scale.

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, CATEGORICAL, TYPE } from "./lib/theme.ts";
import { grid } from "./lib/scale.ts";
import { fitPreference, kernel, pointIndex, predictPreference, type Duel, type X } from "./lib/gp.ts";
import { fmtDuels, parseDuels, parseNumbers, validDuels, validNumbers } from "./lib/params.ts";
import { memo, rng } from "./lib/random.ts";
import { Phi } from "./lib/stats.ts";
import { running } from "./lib/objectives.ts";
import { band, clip, curve, frame, frameAxes, hitArea } from "./lib/plot.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "input x",
    u: "utility g(x)",
    mean: "posterior mean",
    band: "95% band",
    bandRel: "95% band for g(x) minus its average",
    truth: "objective, shifted and scaled to fit",
    graph: "comparison graph: one arc per answered pair, colored by connected component",
    graphShort: "comparison graph (one arc per answer)",
    status: "{n} comparison{n:/s} · {m} input{m:/s} · {k} component{k:/s} · Newton: {it} step{it:/s}",
    predicted: "before the last answer the model gave the winner P = {p}",
    pickFirst: "Click a second input to compare with x₁ = {a}.",
    pickPair: "Which is better, x₁ = {a} or x₂ = {b}?",
    empty: "Click two inputs to compare them, or press Random pair.",
    describe: "{n} comparison{n:/s} among {m} input{m:/s} in {k} connected component{k:/s}. The posterior mean is highest at x = {best}; the objective's maximum is at x = {xs}.",
  },
  zh: {
    x: "输入 x",
    u: "效用 g(x)",
    mean: "后验均值",
    band: "95% 区间带",
    bandRel: "g(x) 减去其平均值后的 95% 区间带",
    truth: "目标函数（经平移和缩放以便对照）",
    graph: "比较图：每个已回答的对画一条弧，按连通分量着色",
    graphShort: "比较图（每个回答一条弧）",
    status: "{n} 次比较 · {m} 个输入 · {k} 个连通分量 · Newton 法：{it} 步",
    predicted: "最后一次回答之前，模型给获胜选项的概率为 P = {p}",
    pickFirst: "再点击一个输入，与 x₁ = {a} 比较。",
    pickPair: "x₁ = {a} 和 x₂ = {b}，哪个更好？",
    empty: "点击两个输入进行比较，或按“随机一对”。",
    describe: "{m} 个输入之间有 {n} 次比较，构成 {k} 个连通分量。后验均值在 x = {best} 处最高；目标函数的最大值在 x = {xs} 处。",
  },
};

const params = {
  who: { kind: "choice", label: { en: "Who answers", zh: "由谁回答" }, options: [{ value: "sim", label: { en: "The objective", zh: "目标函数" } }, { value: "you", label: { en: "You", zh: "你" } }], default: "sim" },
  lengthscale: { kind: "range", label: { en: "Lengthscale ℓ", zh: "长度尺度 ℓ" }, min: 0.03, max: 0.5, default: 0.1, scale: "log" },
  sigma: { kind: "range", label: { en: "Model noise σ", zh: "模型噪声 σ" }, min: 0.01, max: 1, default: 0.1, scale: "log" },
  truth: { kind: "toggle", label: { en: "Show the objective", zh: "显示目标函数" }, default: true },
  relative: { kind: "toggle", label: { en: "Band relative to the average", zh: "相对平均值的区间带" }, default: false },
  duels: { kind: "data", label: { en: "Answers", zh: "回答" }, default: "", validate: validDuels },
  pick: { kind: "data", label: { en: "Pending pair", zh: "待回答的一对" }, default: "", validate: validNumbers },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 5, step: 1, control: false },
} as const;

type P = { who: "sim" | "you"; lengthscale: number; sigma: number; truth: boolean; relative: boolean; duels: string; pick: string; seed: number };

const XS = grid(0, 1, 201);
const PERSON_NOISE = 0.05; // the simulated person's noise; illustrative
const MAX = 40;
const snap = (x: number) => Math.round(Math.min(1, Math.max(0, x)) * 20) / 20; // 21 candidate inputs

const model = memo((duelStr: string, ls: number, sigma: number) => {
  const ds = parseDuels(duelStr);
  const xs: X[] = [];
  const duels: Duel[] = ds.map(([w, l]) => ({ winner: pointIndex(xs, w), loser: pointIndex(xs, l) }));
  const k = kernel("rbf", ls, 1);
  const fit = fitPreference(k, xs, duels, sigma);
  const post = predictPreference(fit, XS, true);
  // sd of g(x) minus its average over the grid: what comparisons can pin down
  const C2 = post.cov!;
  const N = XS.length;
  const rowMean = C2.map((row) => row.reduce((s2, v) => s2 + v, 0) / N);
  const allMean = rowMean.reduce((s2, v) => s2 + v, 0) / N;
  const sdRel = XS.map((_, i) => Math.sqrt(Math.max(1e-12, C2[i][i] - 2 * rowMean[i] + allMean)));
  // connected components of the comparison graph
  const parent = xs.map((_, i) => i);
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  for (const d of duels) parent[find(d.winner)] = find(d.loser);
  const roots = new Map<number, number>();
  const comp = xs.map((_, i) => { const r = find(i); if (!roots.has(r)) roots.set(r, roots.size); return roots.get(r)!; });
  return { xs: xs as number[], duels, fit, post, sdRel, comp, k: roots.size };
}, 24);

// The probability the model assigned to the observed winner of the last duel,
// before that answer was given: @eq-cmp-predictive on the joint posterior.
function predictedLast(p: P): number | null {
  const ds = parseDuels(p.duels);
  if (!ds.length) return null;
  const [w, l] = ds[ds.length - 1];
  const m = model(fmtDuels(ds.slice(0, -1)), p.lengthscale, p.sigma);
  const pp = predictPreference(m.fit, [w, l], true);
  const c = pp.cov!;
  const v = 2 * p.sigma * p.sigma + c[0][0] + c[1][1] - 2 * c[0][1];
  return Phi((pp.mean[0] - pp.mean[1]) / Math.sqrt(Math.max(v, 1e-12)));
}

function answer(p: P, a: number, b: number, aWins: boolean): P {
  const ds = parseDuels(p.duels);
  ds.push(aWins ? [a, b] : [b, a]);
  return { ...p, duels: fmtDuels(ds.slice(-MAX)), pick: "" };
}

// The simulated person: utility = the running objective, probit noise.
function simAnswer(p: P, a: number, b: number): P {
  const n = parseDuels(p.duels).length;
  const r = rng(p.seed * 7919 + n * 104729 + Math.round(a * 1000) * 31 + Math.round(b * 1000));
  r();
  const pa = Phi((running.f(a) - running.f(b)) / (Math.SQRT2 * PERSON_NOISE));
  return answer(p, a, b, r() < pa);
}

function propose(p: P, a: number, b: number): P {
  if (a === b) return p;
  return p.who === "sim" ? simAnswer(p, a, b) : { ...p, pick: `${a},${b}` };
}

function randomPair(p: P): [number, number] {
  const n = parseDuels(p.duels).length;
  const r = rng(p.seed * 31 + n * 977 + 3);
  r();
  const a = snap(r());
  let b = snap(r());
  if (b === a) b = snap((a + 0.37) % 1);
  return [a, b];
}

function bestPair(p: P): [number, number] {
  const m = model(p.duels, p.lengthscale, p.sigma);
  let bi = 0;
  for (let i = 1; i < XS.length; i++) if (m.post.mean[i] > m.post.mean[bi]) bi = i;
  const best = parseDuels(p.duels).length ? snap(XS[bi]) : 0.5;
  const [a] = randomPair(p);
  return [best, a === best ? snap((a + 0.37) % 1) : a];
}

function describe(st: State<P>): string {
  const p = st.p;
  const m = model(p.duels, p.lengthscale, p.sigma);
  let bi = 0;
  for (let i = 1; i < XS.length; i++) if (m.post.mean[i] > m.post.mean[bi]) bi = i;
  return tpl(labels[st.lang ?? "en"].describe, { n: m.duels.length, m: m.xs.length, k: m.k, best: fixed(XS[bi], 2), xs: fixed(running.argmax, 2) });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const m = model(p.duels, p.lengthscale, p.sigma);
  const ds = parseDuels(p.duels);
  const pick = parseNumbers(p.pick);
  const parts: string[] = [];
  // legend
  const items: Array<[string, string, string]> = [[L.mean, C.model, "line"], [p.relative ? L.bandRel : L.band, C.band, "band"]];
  if (p.truth) items.push([L.truth, C.truth, "dash"]);
  let lx = narrow ? 8 : 52, ly = 14;
  for (const [name, color, kind] of items) {
    const tw = labelWidth(name) + 32;
    if (lx + tw > st.w - 8) { lx = narrow ? 8 : 52; ly += 17; }
    parts.push(kind === "band"
      ? el("rect", { x: lx, y: ly - 9, width: 18, height: 10, rx: 2, fill: color })
      : el("line", { x1: lx, x2: lx + 18, y1: ly - 4, y2: ly - 4, stroke: color, "stroke-width": 2, "stroke-dasharray": kind === "dash" ? "5 4" : undefined }));
    parts.push(text(lx + 23, ly, name, { "font-size": TYPE.small, class: "fig-t-muted" }));
    lx += tw;
  }
  // comparison graph strip
  const gTop = ly + 20, gH = 58;
  const f = frame({ w: st.w, top: gTop + gH + 14, height: narrow ? 170 : 210, yDomain: [-2.6, 2.6], yTitle: L.u });
  const base = gTop + gH;
  parts.push(
    text(f.left, gTop, narrow ? L.graphShort : L.graph, { "font-size": TYPE.small, class: "fig-t-faint" }),
    el("line", { x1: f.left, x2: f.right, y1: base, y2: base, stroke: C.rule }),
  );
  const colorOf = (c: number) => (c < CATEGORICAL.length ? CATEGORICAL[c] : C.ink3);
  for (const d of m.duels) {
    const xa = f.x(m.xs[d.winner]), xb = f.x(m.xs[d.loser]);
    const h = Math.min(gH - 14, Math.abs(xb - xa) * 0.4 + 8);
    parts.push(el("path", { d: `M${xa},${base} C${xa},${base - h} ${xb},${base - h} ${xb},${base}`, fill: "none", stroke: colorOf(m.comp[d.winner]), "stroke-width": 1.8, opacity: 0.9 }));
  }
  m.xs.forEach((x, i) => parts.push(el("circle", { cx: f.x(x), cy: base, r: 3.6, fill: colorOf(m.comp[i]), stroke: C.paper, "stroke-width": 1.4 })));
  // main plot
  const sd = p.relative ? m.sdRel : m.post.var.map(Math.sqrt);
  const avg = p.relative ? m.post.mean.reduce((a, b) => a + b, 0) / XS.length : 0;
  const lo = m.post.mean.map((v, i) => v - 1.96 * sd[i]);
  const hi = m.post.mean.map((v, i) => v + 1.96 * sd[i]);
  const cid = `${st.uid}-clip`;
  const layers: string[] = [band(f, XS, lo, hi), curve(f, XS, m.post.mean)];
  if (p.relative) layers.push(el("line", { x1: f.left, x2: f.right, y1: f.y(avg), y2: f.y(avg), stroke: C.ink3, "stroke-dasharray": "2 3" }));
  if (p.truth) {
    // least-squares affine fit of the objective to the posterior mean
    const u = XS.map((x) => running.f(x));
    const mu = u.reduce((a, b) => a + b, 0) / u.length, mm = m.post.mean.reduce((a, b) => a + b, 0) / XS.length;
    let cov = 0, vu = 0;
    for (let i = 0; i < XS.length; i++) { cov += (u[i] - mu) * (m.post.mean[i] - mm); vu += (u[i] - mu) ** 2; }
    let a = cov / vu, b = mm - a * mu;
    if (!ds.length || a <= 0.05) { a = 1.6; b = -1.6 * mu; } // nothing to fit yet: standardize for display
    layers.unshift(curve(f, XS, u.map((v) => a * v + b), { stroke: C.truth, "stroke-width": 1.5, "stroke-dasharray": "5 4", opacity: 0.85 }));
  }
  parts.push(el("defs", {}, clip(f, cid)), frameAxes(f, { xTitle: L.x, yTitle: L.u, yCount: 4 }), hitArea(f), g({ "clip-path": `url(#${cid})` }, ...layers));
  for (const x of m.xs) {
    const i = Math.round(x * (XS.length - 1));
    parts.push(el("circle", { cx: f.x(x), cy: f.y(m.post.mean[i]), r: 3.4, fill: C.ink, stroke: C.paper, "stroke-width": 1.4 }));
  }
  // the pending pair
  pick.forEach((x, j) => {
    parts.push(
      el("line", { x1: f.x(x), x2: f.x(x), y1: f.top, y2: f.bottom, stroke: C.acq, "stroke-width": 1.6, "stroke-dasharray": "4 3" }),
      text(f.x(x), f.top + 12, j === 0 ? "x₁" : "x₂", { "text-anchor": "middle", "font-size": TYPE.label, class: "fig-t-strong fig-t-halo", "data-fig-hit": pick.length === 2 ? `ans-${j}` : undefined }),
    );
  });
  // status lines
  const sy = f.bottom + 46;
  parts.push(text(narrow ? 8 : f.left, sy, tpl(L.status, { n: m.duels.length, m: m.xs.length, k: m.k, it: m.fit.iters }), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  let msg = "";
  if (pick.length === 1) msg = tpl(L.pickFirst, { a: fixed(pick[0], 2) });
  else if (pick.length === 2) msg = tpl(L.pickPair, { a: fixed(pick[0], 2), b: fixed(pick[1], 2) });
  else if (!ds.length) msg = L.empty;
  else { const pr = predictedLast(p); if (pr !== null) msg = tpl(L.predicted, { p: fixed(pr, 2) }); }
  parts.push(text(narrow ? 8 : f.left, sy + 17, msg, { "font-size": TYPE.small, class: pick.length ? "fig-t-strong" : "fig-t-muted fig-t-num" }));
  return svg(st.w, sy + 26, describe(st), ...parts);
}

export default defineFigure({
  name: "pref-duels-1d",
  title: { en: "Learning a utility from comparisons on the running objective", zh: "在示例目标函数上从比较中学习效用" },
  labels,
  params,
  hint: { en: "Click two inputs to compare them, or use Random pair. In “You” mode, answer with the buttons or by clicking x₁ or x₂.", zh: "点击两个输入进行比较，或使用“随机一对”。在“你”模式下，用按钮回答，或点击 x₁ 或 x₂。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    const q = p as P;
    if (e.phase !== "click") return null;
    const pick = parseNumbers(q.pick);
    if (pick.length === 2 && (e.target === "ans-0" || e.target === "ans-1")) return answer(q, pick[0], pick[1], e.target === "ans-0");
    if (e.target !== "plot" || !e.data) return null;
    const x = snap(e.data.x);
    if (pick.length === 1) return pick[0] === x ? null : propose({ ...q, pick: "" }, pick[0], x);
    return { ...q, pick: String(x) };
  },
  actions: [
    { label: { en: "Random pair", zh: "随机一对" }, primary: true, run: (p) => { const [a, b] = randomPair(p as P); return propose({ ...(p as P), pick: "" }, a, b); }, enabled: (p) => parseNumbers(p.pick).length < 2 },
    { label: { en: "Best against random", zh: "最优对随机" }, run: (p) => { const [a, b] = bestPair(p as P); return propose({ ...(p as P), pick: "" }, a, b); }, enabled: (p) => parseNumbers(p.pick).length < 2 },
    { label: { en: "Ten random pairs", zh: "随机十对" }, run: (p) => { let q = { ...(p as P), pick: "" }; for (let i = 0; i < 10; i++) { const [a, b] = randomPair(q); q = propose(q, a, b); } return q; }, enabled: (p) => p.who === "sim" },
    { label: { en: "Prefer x₁", zh: "更喜欢 x₁" }, primary: true, run: (p) => { const k = parseNumbers(p.pick); return answer(p as P, k[0], k[1], true); }, enabled: (p) => parseNumbers(p.pick).length === 2 },
    { label: { en: "Prefer x₂", zh: "更喜欢 x₂" }, primary: true, run: (p) => { const k = parseNumbers(p.pick); return answer(p as P, k[0], k[1], false); }, enabled: (p) => parseNumbers(p.pick).length === 2 },
    { label: { en: "Undo", zh: "撤销" }, run: (p) => (p.pick ? { ...p, pick: "" } : { ...p, duels: fmtDuels(parseDuels(p.duels).slice(0, -1)) }), enabled: (p) => p.duels !== "" || p.pick !== "" },
  ],
  update(p, key) {
    if (key === "who") return { ...p, pick: "" };
    return p;
  },
});
