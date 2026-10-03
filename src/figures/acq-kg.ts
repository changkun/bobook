// The knowledge gradient as a lookahead. The reader picks a candidate input
// x. One more evaluation there would return a value the model can only
// predict as a Gaussian; for seven equally likely outcomes (the quantiles of
// that Gaussian) the top panel draws the posterior mean the model would have
// afterwards, and marks where each of those means is largest. The knowledge
// gradient is the average height of those maxima, taken over all outcomes,
// minus the height of the current mean's maximum. The strip below shows it for
// every candidate, computed exactly by the envelope method of Frazier, Powell
// and Dayanik (2009).

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { fit, kernel, predict } from "./lib/gp.ts";
import { argmax } from "./lib/acq.ts";
import { PhiInv } from "./lib/stats.ts";
import { memo } from "./lib/random.ts";
import { running } from "./lib/objectives.ts";
import { knowledgeGradient, fantasyMean } from "./lib/acq-lookahead.ts";
import { fmtNumbers, parseNumbers, validNumbers } from "./lib/params.ts";
import { band, clip, curve, dots, frame, frameAxes, hitArea, legend, type Frame } from "./lib/plot.ts";
import { axis } from "./lib/axis.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "input x",
    f: "f(x)",
    truth: "hidden objective",
    mean: "posterior mean now",
    fant: "mean after one more evaluation at x (7 outcomes)",
    best: "maximum of each",
    now: "best mean now μ*",
    readout: "x = {x}:  best mean now {b},  expected best mean after {e},  KG = {kg}",
    readoutShort: "x = {x}: μ* now {b}, after {e}, KG = {kg}",
    kg: "knowledge gradient",
    describe: "With {n} evaluations, the highest posterior mean is {b}. Evaluating at x = {x} would raise the highest mean to {e} on average, so the knowledge gradient there is {kg}. The knowledge gradient is largest at x = {xa}.",
  },
  zh: {
    x: "输入 x",
    f: "f(x)",
    truth: "隐藏的目标函数",
    mean: "当前后验均值",
    fant: "在 x 处再评估一次后的均值（7 种结果）",
    best: "各自的最大值",
    now: "当前最高均值 μ*",
    readout: "x = {x}：当前最高均值 {b}，评估后最高均值的期望 {e}，KG = {kg}",
    readoutShort: "x = {x}：当前 μ* {b}，评估后 {e}，KG = {kg}",
    kg: "知识梯度",
    describe: "已有 {n} 次评估，后验均值的最高值为 {b}。在 x = {x} 处评估，平均会把最高均值提升到 {e}，因此该处的知识梯度为 {kg}。知识梯度在 x = {xa} 处最大。",
  },
};

const params = {
  x: { kind: "range", label: { en: "Candidate x", zh: "候选点 x" }, min: 0, max: 1, default: 0.62, step: 0.005 },
  noise: { kind: "range", label: { en: "Noise sd σ_n", zh: "噪声标准差 σ_n" }, min: 0, max: 0.3, default: 0, step: 0.01 },
  truth: { kind: "toggle", label: { en: "Show hidden objective", zh: "显示隐藏的目标函数" }, default: false },
  points: { kind: "data", label: { en: "Evaluated inputs", zh: "已评估的输入" }, default: "0.1,0.25,0.45,0.92", validate: validNumbers },
} as const;

type P = { x: number; noise: number; truth: boolean; points: string };

const XS = grid(0, 1, 201);
const K = kernel("rbf", 0.08, 0.45);
const ZQ = Array.from({ length: 7 }, (_, i) => PhiInv((i + 0.5) / 7));

const compute = memo((pts: string, noise: number) => {
  const xs = parseNumbers(pts);
  const ys = xs.map(running.f);
  const ym = ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : 0;
  const nv = Math.max(noise * noise, 1e-6);
  const gp = fit(K, xs, ys, nv, ym);
  const post = predict(gp, XS, true);
  const kg = knowledgeGradient(post, nv);
  return { xs, ys, post, sd: post.var.map(Math.sqrt), kg, nv, best: Math.max(...post.mean) };
}, 32);

const idx = (x: number) => Math.max(0, Math.min(XS.length - 1, Math.round(x * (XS.length - 1))));

function describe(st: State<P>): string {
  const c = compute(st.p.points, st.p.noise);
  const i = idx(st.p.x);
  return tpl(labels[st.lang ?? "en"].describe, {
    n: c.xs.length, b: fixed(c.best, 3), x: fixed(XS[i], 2), e: fixed(c.best + c.kg[i], 3), kg: fixed(c.kg[i], 3), xa: fixed(XS[argmax(c.kg)], 2),
  });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const c = compute(p.points, p.noise);
  const i = idx(p.x);
  const fants = ZQ.map((z) => fantasyMean(c.post, i, c.nv, z));
  const lg = legend(narrow ? 8 : 40, 14, st.w - 16, [
    { kind: "line", color: C.model, label: L.mean },
    { kind: "line", color: C.c7, label: L.fant },
    { kind: "dot", color: C.c7, label: L.best },
    ...(p.truth ? [{ kind: "dash" as const, color: C.truth, label: L.truth }] : []),
  ]);
  const top = 14 + lg.height + 24;
  const f = frame({ w: st.w, top, height: narrow ? 180 : 220, yDomain: running.range, yTitle: L.f });
  const cid = `${st.uid}-c`;
  const lo = c.post.mean.map((m, k) => m - 1.96 * c.sd[k]);
  const hi = c.post.mean.map((m, k) => m + 1.96 * c.sd[k]);
  const expected = c.best + c.kg[i];
  const parts: string[] = [
    lg.svg,
    el("defs", {}, clip(f, cid)),
    text(narrow ? 8 : f.left, top - 8, tpl(narrow ? L.readoutShort : L.readout, { x: fixed(XS[i], 2), b: fixed(c.best, 2), e: fixed(expected, 2), kg: fixed(c.kg[i], 3) }), { "font-size": TYPE.small, class: "fig-t-num fig-t-strong" }),
    frameAxes(f, { yTitle: L.f, xTicks: false }),
    hitArea(f),
    g({ "clip-path": `url(#${cid})` },
      band(f, XS, lo, hi),
      p.truth ? curve(f, XS, XS.map(running.f), { stroke: C.truth, "stroke-dasharray": "5 4", "stroke-width": 1.5 }) : "",
      ...fants.map((m) => curve(f, XS, m, { stroke: C.c7, "stroke-width": 1.2, opacity: 0.8 })),
      curve(f, XS, c.post.mean),
    ),
    // The current best mean, and each fantasy's maximum.
    el("line", { x1: f.left, x2: f.right, y1: f.y(c.best), y2: f.y(c.best), stroke: C.model, "stroke-width": 1.2, "stroke-dasharray": "5 4" }),
    text(f.left + 4, f.y(c.best) - 5, L.now, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
    ...fants.map((m) => { const k = argmax(m); return el("circle", { cx: f.x(XS[k]), cy: f.y(m[k]), r: 3.6, fill: C.c7, stroke: C.paper, "stroke-width": 1.2 }); }),
    el("line", { x1: f.x(XS[i]), x2: f.x(XS[i]), y1: f.top, y2: f.bottom, stroke: C.acq, "stroke-width": 1.6, "stroke-dasharray": "4 3" }),
    dots(f, c.xs.map((x, k) => [x, c.ys[k]] as [number, number])),
  ];

  // KG strip.
  const kTop = f.bottom + 16;
  const kH = narrow ? 64 : 76;
  const kmax = Math.max(...c.kg, 1e-6) * 1.12;
  const fk: Frame = { ...f, y: linear([0, kmax], [kTop + kH, kTop]), top: kTop, bottom: kTop + kH };
  const ka = argmax(c.kg);
  parts.push(
    frameAxes(fk, { xTitle: L.x, yCount: 2 }),
    hitArea(fk, "kg"),
    el("path", { d: `M${fk.x(0)},${fk.y(0)}` + XS.map((x, k) => `L${fk.x(x)},${fk.y(c.kg[k])}`).join("") + `L${fk.x(1)},${fk.y(0)}Z`, fill: C.acqFill, stroke: C.acq, "stroke-width": 1.6 }),
    el("line", { x1: fk.x(XS[i]), x2: fk.x(XS[i]), y1: fk.top, y2: fk.bottom, stroke: C.acq, "stroke-width": 1.6, "stroke-dasharray": "4 3" }),
    el("circle", { cx: fk.x(XS[i]), cy: fk.y(c.kg[i]), r: 4.2, fill: C.acq, stroke: C.paper, "stroke-width": 1.5 }),
    el("path", { d: `M${fk.x(XS[ka])},${fk.y(c.kg[ka]) - 10}l-4,-6h8z`, fill: C.ink }),
    text(fk.left + 6, fk.top + 12, L.kg, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
  );
  return svg(st.w, fk.bottom + 40, describe(st), ...parts);
}

export default defineFigure({
  name: "acq-kg",
  title: { en: "The knowledge gradient as a one-step lookahead", zh: "作为一步前瞻的知识梯度" },
  labels,
  params,
  hint: { en: "Click either plot, or move the slider, to choose the candidate x.", zh: "点击任一图或拖动滑块，选择候选点 x。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click" && e.phase !== "move" && e.phase !== "down") return null;
    if ((e.target !== "plot" && e.target !== "kg") || !e.data) return null;
    return { ...p, x: Math.round(Math.min(1, Math.max(0, e.data.x)) * 200) / 200 };
  },
  actions: [
    { label: { en: "Go to the KG maximum", zh: "跳到知识梯度的最大处" }, run: (p) => { const c = compute(p.points, p.noise); return { ...p, x: XS[argmax(c.kg)] }; } },
    {
      label: { en: "Evaluate at x", zh: "在 x 处评估" }, primary: true,
      run: (p) => { const xs = parseNumbers(p.points); if (xs.length >= 20) xs.shift(); xs.push(p.x); return { ...p, points: fmtNumbers(xs) }; },
    },
  ],
});
