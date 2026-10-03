// The Bayesian optimization loop on the book's running objective. Each
// timeline step is one iteration: fit the Gaussian process to everything
// observed so far, compute the acquisition function, evaluate the objective
// where it is largest. The top panel shows the hidden objective (dashed), the
// posterior, and the observations, with the newest one ringed; the bottom
// panel shows the acquisition function and the query it picks.

import { defineFigure, tr, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid } from "./lib/scale.ts";
import { fit, kernel, predict, samplePosterior } from "./lib/gp.ts";
import { argmax, ei, pi, ucb } from "./lib/acq.ts";
import { memo, normal, rng } from "./lib/random.ts";
import { running } from "./lib/objectives.ts";
import { band, clip, curve, dots, frame, frameAxes, legend, vline } from "./lib/plot.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "input x",
    f: "f(x)",
    a: "acquisition",
    truth: "hidden objective",
    mean: "posterior mean",
    band: "95% band",
    obs: "observations",
    next: "next query",
    best: "best so far {b}, true max {m}",
    key0: "Initial design: {n} random evaluations",
    key: "Step {i}: evaluate x = {x}",
    describe: "Bayesian optimization with {acq} after {n} evaluations. The best value found is {b}; the true maximum is {m} at x = {xm}. The next query is at x = {x}.",
  },
  zh: {
    x: "输入 x",
    f: "f(x)",
    a: "采集函数",
    truth: "隐藏的目标函数",
    mean: "后验均值",
    band: "95% 区间",
    obs: "观测",
    next: "下一次查询",
    best: "目前最优 {b}，真实最大值 {m}",
    key0: "初始设计：{n} 次随机评估",
    key: "第 {i} 步：评估 x = {x}",
    describe: "使用{acq}的贝叶斯优化，已进行 {n} 次评估。找到的最优值为 {b}；真实最大值为 {m}，位于 x = {xm}。下一次查询位于 x = {x}。",
  },
};

const ACQ = [
  { value: "ei", label: { en: "EI", zh: "期望改进" } },
  { value: "ucb", label: { en: "UCB", zh: "上置信界" } },
  { value: "pi", label: { en: "PI", zh: "改进概率" } },
  { value: "ts", label: { en: "Thompson", zh: "Thompson 采样" } },
] as const;

const params = {
  acq: { kind: "choice", label: { en: "Acquisition", zh: "采集函数" }, options: ACQ, default: "ei" },
  lengthscale: { kind: "range", label: { en: "Lengthscale ℓ", zh: "长度尺度 ℓ" }, min: 0.03, max: 0.3, default: 0.08, scale: "log" },
  beta: { kind: "range", label: { en: "UCB weight √β", zh: "上置信界权重 √β" }, min: 0, max: 5, default: 2, step: 0.1 },
  init: { kind: "range", label: { en: "Initial points", zh: "初始点数" }, min: 1, max: 6, default: 2, step: 1, control: false },
  steps: { kind: "range", label: { en: "Iterations", zh: "迭代次数" }, min: 3, max: 25, default: 12, step: 1, control: false },
  noise: { kind: "range", label: { en: "Noise sd", zh: "噪声标准差" }, min: 0, max: 0.2, default: 0, step: 0.01, control: false },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 3, step: 1, control: false },
  truth: { kind: "toggle", label: { en: "Show hidden objective", zh: "显示隐藏的目标函数" }, default: true },
} as const;

type P = { acq: string; lengthscale: number; beta: number; init: number; steps: number; noise: number; seed: number; truth: boolean };

const XS = grid(0, 1, 201);
const SF2 = 0.45;

interface Frame { obs: Array<[number, number]>; mean: number[]; sd: number[]; acq: number[]; next: number; best: number }

// The whole run is a pure function of the parameters; frames are memoized.
const run = memo((acqName: string, ls: number, beta: number, init: number, steps: number, noise: number, seed: number): Frame[] => {
  const r = rng(seed);
  const z = normal(rng(seed * 7919 + 1));
  const evalF = (x: number) => running.f(x) + noise * z();
  const obs: Array<[number, number]> = [];
  for (let i = 0; i < init; i++) { const x = 0.05 + 0.9 * r(); obs.push([x, evalF(x)]); }
  const k = kernel("rbf", ls, SF2);
  const frames: Frame[] = [];
  for (let s = 0; s <= steps; s++) {
    const ym = obs.reduce((a, o) => a + o[1], 0) / obs.length;
    const gp = fit(k, obs.map((o) => o[0]), obs.map((o) => o[1]), Math.max(noise * noise, 1e-5), ym);
    const post = predict(gp, XS, acqName === "ts");
    const sd = post.var.map(Math.sqrt);
    const best = Math.max(...obs.map((o) => o[1]));
    let acq: number[];
    if (acqName === "ei") acq = post.mean.map((m, i) => ei(m, sd[i], best, 0.01));
    else if (acqName === "pi") acq = post.mean.map((m, i) => pi(m, sd[i], best, 0.01));
    else if (acqName === "ucb") acq = post.mean.map((m, i) => ucb(m, sd[i], beta));
    else acq = samplePosterior(post, rng(seed * 31 + s), 1)[0];
    const next = XS[argmax(acq)];
    frames.push({ obs: obs.slice(), mean: post.mean, sd, acq, next, best });
    obs.push([next, evalF(next)]);
  }
  return frames;
}, 16);

const frames = (p: P) => run(p.acq, p.lengthscale, p.beta, p.init, p.steps, p.noise, p.seed);

function describe(st: State<P>): string {
  const fr = frames(st.p)[Math.round(st.t)];
  const lang = st.lang ?? "en";
  return tpl(labels[lang].describe, {
    acq: tr(ACQ.find((a) => a.value === st.p.acq)!.label, lang), n: fr.obs.length, b: fixed(fr.best, 2), m: fixed(running.max, 2), xm: fixed(running.argmax, 2), x: fixed(fr.next, 2),
  });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const fr = frames(p)[Math.min(Math.round(st.t), p.steps)];
  const items = [
    ...(p.truth ? [{ kind: "dash" as const, color: C.truth, label: L.truth }] : []),
    { kind: "line" as const, color: C.model, label: L.mean },
    { kind: "band" as const, color: C.band, label: L.band },
    { kind: "line" as const, color: C.acq, label: L.next },
  ];
  const lg = legend(narrow ? 8 : 40, 14, st.w - 16, items);
  const top = 14 + lg.height + 8;
  const f = frame({ w: st.w, top, height: narrow ? 170 : 220, yDomain: running.range, yTitle: L.f });
  const aTop = f.bottom + 16;
  const aH = narrow ? 60 : 74;
  const amax = Math.max(1e-9, ...fr.acq), amin = p.acq === "ucb" || p.acq === "ts" ? Math.min(...fr.acq) : 0;
  const fa = frame({ w: st.w, top: aTop, height: aH, yDomain: [amin, amax + (amax - amin) * 0.1 + 1e-12], yTitle: L.f });
  const H = aTop + aH + 40;
  const lo = fr.mean.map((m, i) => m - 1.96 * fr.sd[i]);
  const hi = fr.mean.map((m, i) => m + 1.96 * fr.sd[i]);
  const cid = `${st.uid}-c`;
  const newest = fr.obs.length > p.init ? fr.obs[fr.obs.length - 1] : undefined;
  const acqPath = `M${fa.x(0)},${fa.y(amin)}` + XS.map((x, i) => `L${fa.x(x)},${fa.y(fr.acq[i])}`).join("") + `L${fa.x(1)},${fa.y(amin)}Z`;
  return svg(st.w, H, describe(st),
    el("defs", {}, clip(f, cid)),
    lg.svg,
    frameAxes(f, { yTitle: L.f, xTicks: false }),
    g({ "clip-path": `url(#${cid})` },
      band(f, XS, lo, hi),
      p.truth ? curve(f, XS, XS.map(running.f), { stroke: C.truth, "stroke-dasharray": "5 4", "stroke-width": 1.6 }) : "",
      curve(f, XS, fr.mean),
    ),
    vline(f, fr.next, C.acq),
    dots(f, fr.obs),
    newest ? el("circle", { cx: f.x(newest[0]), cy: f.y(newest[1]), r: 8.5, fill: "none", stroke: C.acq, "stroke-width": 2 }) : "",
    text(f.right - 4, f.top + 12, tpl(L.best, { b: fixed(fr.best, 2), m: fixed(running.max, 2) }), { "text-anchor": "end", "font-size": TYPE.small, class: "fig-t-muted fig-t-halo fig-t-num" }),
    // acquisition panel
    frameAxes({ ...fa }, { xTitle: L.x, yCount: 2, yTitle: narrow ? "" : "" }),
    el("path", { d: acqPath, fill: C.acqFill, stroke: C.acq, "stroke-width": 1.6 }),
    el("line", { x1: fa.x(fr.next), x2: fa.x(fr.next), y1: fa.top, y2: fa.bottom, stroke: C.acq, "stroke-width": 1.5, "stroke-dasharray": "4 3" }),
    el("circle", { cx: fa.x(fr.next), cy: fa.y(fr.acq[argmax(fr.acq)]), r: 4, fill: C.acq }),
    text(fa.left + 6, fa.top + 12, `${L.a}${st.lang === "zh" ? "：" : ": "}${tr(ACQ.find((a) => a.value === p.acq)!.label, st.lang ?? "en")}`, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
  );
}

export default defineFigure({
  name: "bo-loop",
  title: { en: "The Bayesian optimization loop", zh: "贝叶斯优化循环" },
  labels,
  params,
  timeline: {
    rate: 1.2,
    discrete: true,
    duration: (p) => p.steps,
    keyframes: (p, lang) => frames(p as P).map((fr, i) => ({ t: i, label: i === 0 ? tpl(labels[lang].key0, { n: (p as P).init }) : tpl(labels[lang].key, { i, x: fixed(frames(p as P)[i - 1].next, 2) }) })),
    poster: (p) => Math.min(5, p.steps),
  },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "New run", zh: "新一轮运行" }, run: (p) => ({ ...p, seed: (p.seed % 9999) + 1 }) },
  ],
});
