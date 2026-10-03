// What to report when evaluations are noisy. GP-UCB (√β = 2, with the noise
// level known to the model) runs on the running objective with Gaussian
// observation noise, on a grid of 101 inputs, using rank-one posterior
// updates. After the budget is spent, two rules pick the recommendation: the
// input with the best observed value, and the evaluated input with the best
// posterior mean. The top panel shows one run, with each rule's claimed value
// and the true value at the input it picks. The bottom panel repeats the
// experiment over 24 seeds at each of six noise levels and plots the average
// simple regret of each rule, and how much the best observed value overstates
// the truth at its input: the winner's curse.

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { kernel } from "./lib/gp.ts";
import { memo, normal, rng } from "./lib/random.ts";
import { running } from "./lib/objectives.ts";
import { band, clip, curve, frame, frameAxes, legend, type Frame } from "./lib/plot.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "input x",
    f: "f(x)",
    truth: "hidden objective",
    mean: "posterior mean",
    band: "95% band",
    obs: "noisy observations",
    bestY: "best observed y",
    bestMu: "best posterior mean",
    claim: "claimed",
    noise: "noise sd σ_n",
    avg: "average over {n} runs",
    regY: "regret, best observed y",
    regMu: "regret, best posterior mean",
    over: "overstatement of best y",
    readY: "best observed y claims {c}, true {t}",
    readMu: "best posterior mean claims {c}, true {t}",
    max: "true maximum {m}",
    describe: "With noise sd {s} and {n} evaluations, the best observed value claims {cy} at x = {xy} where the truth is {ty}; the best posterior mean claims {cm} at x = {xm} where the truth is {tm}. The true maximum is {m}.",
  },
  zh: {
    x: "输入 x",
    f: "f(x)",
    truth: "隐藏的目标函数",
    mean: "后验均值",
    band: "95% 区间",
    obs: "带噪声的观测",
    bestY: "最优观测 y",
    bestMu: "最优后验均值",
    claim: "声称值",
    noise: "噪声标准差 σ_n",
    avg: "{n} 次运行的平均",
    regY: "最优观测的遗憾",
    regMu: "最优均值的遗憾",
    over: "最优 y 的高估",
    readY: "最优观测 y 声称 {c}，真实值 {t}",
    readMu: "最优后验均值声称 {c}，真实值 {t}",
    max: "真实最大值 {m}",
    describe: "噪声标准差为 {s}、评估 {n} 次时，最优观测值在 x = {xy} 处声称 {cy}，而该处的真实值为 {ty}；最优后验均值在 x = {xm} 处声称 {cm}，而该处的真实值为 {tm}。真实最大值为 {m}。",
  },
};

const params = {
  noise: { kind: "range", label: { en: "Noise sd σ_n", zh: "噪声标准差 σ_n" }, min: 0, max: 0.5, default: 0.25, step: 0.05 },
  evals: { kind: "range", label: { en: "Evaluations", zh: "评估次数" }, min: 10, max: 40, default: 25, step: 1 },
  truth: { kind: "toggle", label: { en: "Show hidden objective", zh: "显示隐藏的目标函数" }, default: true },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 1, step: 1, control: false },
} as const;

type P = { noise: number; evals: number; truth: boolean; seed: number };

const M = 101;
const XS = grid(0, 1, M);
const FX = XS.map(running.f);
const FMAX = Math.max(...FX);
const SF2 = 0.45;
const K = kernel("rbf", 0.08, SF2);
const K0 = (() => {
  const a = new Float64Array(M * M);
  for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) a[i * M + j] = K(XS[i], XS[j]);
  return a;
})();
const MEAN0 = 0.2; // a constant prior mean near the objective's average

interface Run { obsIdx: number[]; ys: number[]; mu: Float64Array; sd: number[]; iy: number; im: number }

function bo(noise: number, n: number, seed: number): Run {
  const s2 = Math.max(noise * noise, 1e-6);
  const S = Float64Array.from(K0);
  const mu = new Float64Array(M).fill(MEAN0);
  const z = normal(rng(seed * 7919 + 11));
  const r = rng(seed * 104729 + 3);
  const obsIdx: number[] = [], ys: number[] = [];
  for (let t = 0; t < n; t++) {
    let bi = 0;
    if (t < 3) bi = Math.floor(r() * M);
    else {
      let bv = -Infinity;
      for (let i = 0; i < M; i++) { const v = mu[i] + 2 * Math.sqrt(Math.max(S[i * M + i], 0)); if (v > bv) { bv = v; bi = i; } }
    }
    const y = FX[bi] + noise * z();
    obsIdx.push(bi); ys.push(y);
    // rank-one update of mean and covariance
    const col = new Float64Array(M);
    for (let j = 0; j < M; j++) col[j] = S[j * M + bi];
    const d = col[bi] + s2;
    const rr = (y - mu[bi]) / d;
    for (let j = 0; j < M; j++) mu[j] += col[j] * rr;
    for (let a = 0; a < M; a++) { const ca = col[a] / d; if (ca !== 0) for (let b = 0; b < M; b++) S[a * M + b] -= ca * col[b]; }
  }
  let iy = 0;
  for (let k = 1; k < ys.length; k++) if (ys[k] > ys[iy]) iy = k;
  let im = 0;
  for (let k = 1; k < obsIdx.length; k++) if (mu[obsIdx[k]] > mu[obsIdx[im]]) im = k;
  const sd = Array.from({ length: M }, (_, i) => Math.sqrt(Math.max(S[i * M + i], 0)));
  return { obsIdx, ys, mu, sd, iy, im };
}

const one = memo((noise: number, n: number, seed: number) => bo(noise, n, seed), 16);

const LEVELS = [0, 0.1, 0.2, 0.3, 0.4, 0.5];
const RUNS = 24;
const sweep = memo((n: number) => LEVELS.map((nz) => {
  let ry = 0, rm = 0, ov = 0;
  for (let s = 1; s <= RUNS; s++) {
    const b = bo(nz, n, 1000 + s);
    const xy = b.obsIdx[b.iy], xm = b.obsIdx[b.im];
    ry += FMAX - FX[xy];
    rm += FMAX - FX[xm];
    ov += b.ys[b.iy] - FX[xy];
  }
  return { ry: ry / RUNS, rm: rm / RUNS, ov: ov / RUNS };
}), 8);

function describe(st: State<P>): string {
  const p = st.p;
  const b = one(p.noise, p.evals, p.seed);
  const xy = b.obsIdx[b.iy], xm = b.obsIdx[b.im];
  return tpl(labels[st.lang ?? "en"].describe, {
    s: fixed(p.noise, 2), n: p.evals, cy: fixed(b.ys[b.iy], 2), xy: fixed(XS[xy], 2), ty: fixed(FX[xy], 2),
    cm: fixed(b.mu[xm], 2), xm: fixed(XS[xm], 2), tm: fixed(FX[xm], 2), m: fixed(FMAX, 2),
  });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const b = one(p.noise, p.evals, p.seed);
  const COL_Y = C.c8, COL_M = C.c1;
  const lg = legend(narrow ? 8 : 40, 14, st.w - 16, [
    ...(p.truth ? [{ kind: "dash" as const, color: C.truth, label: L.truth }] : []),
    { kind: "line" as const, color: C.model, label: L.mean },
    { kind: "band" as const, color: C.band, label: L.band },
    { kind: "dot" as const, color: C.ink, label: L.obs },
  ]);
  const top = 14 + lg.height + 6;
  const f = frame({ w: st.w, top, height: narrow ? 180 : 220, yDomain: [-1.4, 2.0], yTitle: L.f });
  const cid = `${st.uid}-c`;
  const mu = Array.from(b.mu);
  const lo = mu.map((m, i) => m - 1.96 * b.sd[i]), hi = mu.map((m, i) => m + 1.96 * b.sd[i]);
  const parts: string[] = [el("defs", {}, clip(f, cid)), lg.svg, frameAxes(f, { yTitle: L.f, xTicks: false })];
  parts.push(g({ "clip-path": `url(#${cid})` },
    band(f, XS, lo, hi),
    p.truth ? curve(f, XS, FX, { stroke: C.truth, "stroke-dasharray": "5 4", "stroke-width": 1.6 }) : "",
    curve(f, XS, mu),
  ));
  b.obsIdx.forEach((i, k) => parts.push(el("circle", { cx: f.x(XS[i]), cy: f.y(Math.max(-1.4, Math.min(2, b.ys[k]))), r: 3.4, fill: C.ink, stroke: C.paper, "stroke-width": 1.4, opacity: 0.85 })));
  // the two recommendations: claimed value (marker) and truth (tick), joined
  const mark = (i: number, claim: number, color: string, shape: "ring" | "square") => {
    const x = f.x(XS[i]);
    const yc = f.y(claim), yt = f.y(FX[i]);
    return g({},
      el("line", { x1: x, x2: x, y1: yc, y2: yt, stroke: color, "stroke-width": 2 }),
      shape === "ring"
        ? el("circle", { cx: x, cy: yc, r: 7, fill: "none", stroke: color, "stroke-width": 2.2 })
        : el("rect", { x: x - 6, y: yc - 6, width: 12, height: 12, fill: "none", stroke: color, "stroke-width": 2.2 }),
      el("line", { x1: x - 7, x2: x + 7, y1: yt, y2: yt, stroke: color, "stroke-width": 2.4 }),
    );
  };
  const xy = b.obsIdx[b.iy], xm = b.obsIdx[b.im];
  parts.push(mark(xm, b.mu[xm], COL_M, "square"), mark(xy, b.ys[b.iy], COL_Y, "ring"));
  // readout under the plot
  const ry = f.bottom + 16;
  parts.push(
    el("circle", { cx: f.left + 6, cy: ry - 4, r: 5, fill: "none", stroke: COL_Y, "stroke-width": 2 }),
    text(f.left + 16, ry, tpl(L.readY, { c: fixed(b.ys[b.iy], 2), t: fixed(FX[xy], 2) }), { "font-size": TYPE.small, class: "fig-t-num" }),
    el("rect", { x: f.left + 1.5, y: ry + 16 - 8.5, width: 9, height: 9, fill: "none", stroke: COL_M, "stroke-width": 2 }),
    text(f.left + 16, ry + 16, tpl(L.readMu, { c: fixed(b.mu[xm], 2), t: fixed(FX[xm], 2) }), { "font-size": TYPE.small, class: "fig-t-num" }),
    text(f.left + 16, ry + 32, tpl(L.max, { m: fixed(FMAX, 2) }), { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }),
  );
  // bottom panel: averages over runs against noise
  const sw = sweep(p.evals);
  const bTop = ry + 62;
  const bH = narrow ? 120 : 140;
  const ymax = Math.max(0.2, ...sw.map((s) => Math.max(s.ry, s.rm, s.ov))) * 1.1;
  const fl = narrow ? 40 : 52;
  const bx = linear([0, 0.5], [fl, st.w - 14]);
  const by = linear([0, ymax], [bTop + bH, bTop]);
  const bf: Frame = { x: bx, y: by, left: fl, right: st.w - 14, top: bTop, bottom: bTop + bH, w: st.w };
  const lg2 = legend(narrow ? 8 : 40, bTop - 14, st.w - 16, [
    { kind: "line", color: COL_Y, label: L.regY },
    { kind: "line", color: COL_M, label: L.regMu },
    { kind: "dash", color: COL_Y, label: L.over },
  ]);
  const shift = lg2.height - 18;
  const bf2: Frame = { ...bf, top: bTop + shift, bottom: bTop + shift + bH, y: linear([0, ymax], [bTop + shift + bH, bTop + shift]) };
  parts.push(lg2.svg,
    axis({ scale: bf2.y, orient: "left", at: bf2.left, span: [bf2.left, bf2.right], count: 3, title: narrow ? undefined : "" }),
    axis({ scale: bx, orient: "bottom", at: bf2.bottom, span: [bf2.top, bf2.bottom], title: L.noise, count: 5 }),
  );
  const line = (vals: number[], color: string, dash?: string) => el("path", {
    d: LEVELS.map((nz, i) => `${i ? "L" : "M"}${bx(nz).toFixed(1)},${bf2.y(vals[i]).toFixed(1)}`).join(""),
    fill: "none", stroke: color, "stroke-width": 2, "stroke-dasharray": dash,
  });
  parts.push(
    line(sw.map((s) => s.ov), COL_Y, "5 4"),
    line(sw.map((s) => s.ry), COL_Y),
    line(sw.map((s) => s.rm), COL_M),
    ...LEVELS.flatMap((nz, i) => [
      el("circle", { cx: bx(nz), cy: bf2.y(sw[i].ry), r: 2.6, fill: COL_Y }),
      el("circle", { cx: bx(nz), cy: bf2.y(sw[i].rm), r: 2.6, fill: COL_M }),
    ]),
    el("line", { x1: bx(p.noise), x2: bx(p.noise), y1: bf2.top, y2: bf2.bottom, stroke: C.ink3, "stroke-width": 1.2, "stroke-dasharray": "2 3" }),
    text(bf2.left + 6, bf2.top + 12, tpl(L.avg, { n: RUNS }), { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
  );
  const H = bf2.bottom + 42;
  return svg(st.w, H, describe(st), ...parts);
}

// Exposed for tests and for the chapter's numbers.
export const sweepFor = (n: number) => sweep(n);

export default defineFigure({
  name: "practice-incumbent",
  title: { en: "Noise and the incumbent: what a noisy Bayesian optimization run should report", zh: "噪声与当前最优值：带噪声的贝叶斯优化运行应当报告什么" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "New run", zh: "再运行一次" }, primary: true, run: (p) => ({ ...p, seed: (p.seed % 9999) + 1 }) },
  ],
});
