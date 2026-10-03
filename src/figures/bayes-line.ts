// Bayesian linear regression for a straight line f(x) = w1 + w2 x, seen in
// two spaces at once. On the left, data space: the observations the reader
// places, the posterior mean line, a 95% band for the latent f(x), a dashed
// 95% band for a new noisy observation y, and a few lines drawn from the
// posterior. On the right, weight space: the prior and posterior over
// (intercept, slope) as ellipses, with each drawn line appearing as one dot
// of the same color. The readout gives the posterior mean line and the log
// evidence of the data, which peaks at an intermediate prior width.

import { defineFigure, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { clamp, grid, linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { cholesky } from "./lib/linalg.ts";
import { rng } from "./lib/random.ts";
import { blrPosterior, quad, sampleWeights } from "./lib/bayes-linear.ts";
import { fmtPoints, parsePoints, validPoints } from "./lib/params.ts";
import { band, clip, curve, dots, frame, frameAxes, hitArea, legend } from "./lib/plot.ts";
import { fixed, sig, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "input x",
    y: "y",
    mean: "posterior mean",
    bandF: "95% band for f(x)",
    bandY: "95% band for a new y",
    samples: "lines drawn from the posterior",
    obs: "observations",
    w1: "intercept w_1",
    w2: "slope w_2",
    wTitle: "weight space",
    priorE: "prior",
    postE: "posterior",
    fit: "mean line: y = {a} + {b} x",
    fitNone: "no data: the posterior is the prior",
    ev: "log evidence log p(y) = {e}",
    sd: "sd of slope {s}, of intercept {i}",
    describe: "Bayesian linear regression on {n} observation{n:/s} with prior sd {sp} and noise sd {sn}. The posterior mean line is y = {a} + {b} x; the log evidence is {e}.",
    empty: "No observations: the lines drawn are samples from the prior, fanning out away from x = 0.",
  },
  zh: {
    x: "输入 x",
    y: "y",
    mean: "后验均值",
    bandF: "f(x) 的 95% 区间带",
    bandY: "新 y 的 95% 区间带",
    samples: "从后验中抽取的直线",
    obs: "观测",
    w1: "截距 w_1",
    w2: "斜率 w_2",
    wTitle: "权重空间",
    priorE: "先验",
    postE: "后验",
    fit: "均值直线：y = {a} + {b} x",
    fitNone: "没有数据：后验即先验",
    ev: "对数证据 log p(y) = {e}",
    sd: "斜率的标准差 {s}，截距的标准差 {i}",
    describe: "对 {n} 个观测做贝叶斯线性回归，先验标准差为 {sp}，噪声标准差为 {sn}。后验均值直线为 y = {a} + {b} x；对数证据为 {e}。",
    empty: "没有观测：图中的直线是从先验中抽取的样本，离 x = 0 越远散得越开。",
  },
};

const params = {
  prior: { kind: "range", label: { en: "Prior sd of weights σₚ", zh: "权重的先验标准差 σₚ" }, min: 0.1, max: 4, default: 1, scale: "log" },
  noise: { kind: "range", label: { en: "Noise sd σₙ", zh: "噪声标准差 σₙ" }, min: 0.05, max: 1.5, default: 0.3, scale: "log" },
  samples: { kind: "toggle", label: { en: "Draw lines from the posterior", zh: "从后验中抽取直线" }, default: true },
  points: { kind: "data", label: { en: "Observations", zh: "观测" }, default: "-0.55:-0.2,0.35:0.65", validate: validPoints },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 3, step: 1, control: false },
} as const;

type P = { prior: number; noise: number; samples: boolean; points: string; seed: number };

const XS = grid(-1, 1, 81);
const YD: [number, number] = [-3, 3];
const WD = 2.5; // weight space plotted over [-WD, WD]^2
const NS = 8;

function compute(p: P) {
  const pts = parsePoints(p.points);
  const post = blrPosterior(pts.map(([x]) => [1, x]), pts.map(([, y]) => y), p.prior * p.prior, p.noise * p.noise, 2);
  const mean = XS.map((x) => post.mean[0] + post.mean[1] * x);
  const varF = XS.map((x) => quad(post.cov, [1, x]));
  const ws = p.samples ? sampleWeights(post, rng(p.seed * 17 + pts.length), NS) : [];
  return { pts, post, mean, varF, ws };
}

function describe(st: State<P>): string {
  const p = st.p;
  const { pts, post } = compute(p);
  const L = labels[st.lang ?? "en"];
  if (!pts.length) return L.empty;
  return tpl(L.describe, { n: pts.length, sp: sig(p.prior, 2), sn: sig(p.noise, 2), a: fixed(post.mean[0], 2), b: fixed(post.mean[1], 2), e: fixed(post.logEvidence, 2) });
}

function ellipse(cx: (v: number) => number, cy: (v: number) => number, mu: number[], cov: number[][], r: number): string {
  const L = cholesky(cov);
  const pts: Array<[number, number]> = [];
  for (let i = 0; i <= 72; i++) {
    const t = (2 * Math.PI * i) / 72, z1 = r * Math.cos(t), z2 = r * Math.sin(t);
    pts.push([cx(mu[0] + L[0][0] * z1), cy(mu[1] + L[1][0] * z1 + L[1][1] * z2)]);
  }
  return linePath(pts) + "Z";
}

function render(st: State<P>): string {
  const p = st.p, L = labels[st.lang ?? "en"], w = st.w;
  const narrow = w < 480;
  const { pts, post, mean, varF, ws } = compute(p);
  const sampleColor = C.c7;
  // data space, left (or top)
  const dataW = narrow ? w : Math.round(w * 0.57);
  const lg = legend(narrow ? 8 : 52, 14, dataW - 16, [
    { kind: "line", color: C.model, label: L.mean },
    { kind: "band", color: C.band, label: L.bandF },
    { kind: "dash", color: C.ink3, label: L.bandY },
    ...(p.samples ? [{ kind: "line" as const, color: sampleColor, label: L.samples }] : []),
  ]);
  const top = 14 + lg.height + 6;
  const f = frame({ w: dataW, top, height: narrow ? 200 : 250, xDomain: [-1, 1], yDomain: YD, yTitle: L.y });
  const sdF = varF.map(Math.sqrt);
  const sdY = varF.map((v) => Math.sqrt(v + p.noise * p.noise));
  const cid = `${st.uid}-clip`;
  const parts: string[] = [
    el("defs", {}, clip(f, cid)),
    lg.svg,
    frameAxes(f, { yTitle: L.y, xTitle: L.x, yCount: 4 }),
    hitArea(f),
    g({ "clip-path": `url(#${cid})` },
      band(f, XS, mean.map((m, i) => m - 1.96 * sdF[i]), mean.map((m, i) => m + 1.96 * sdF[i])),
      curve(f, XS, mean.map((m, i) => m - 1.96 * sdY[i]), { stroke: C.ink3, "stroke-width": 1.2, "stroke-dasharray": "5 4" }),
      curve(f, XS, mean.map((m, i) => m + 1.96 * sdY[i]), { stroke: C.ink3, "stroke-width": 1.2, "stroke-dasharray": "5 4" }),
      ...ws.map((wv) => curve(f, XS, XS.map((x) => wv[0] + wv[1] * x), { stroke: sampleColor, "stroke-width": 1.1, opacity: 0.7 })),
      curve(f, XS, mean),
    ),
    dots(f, pts, { hit: (i) => `pt-${i}` }),
  ];
  // weight space, right (or below)
  const S = narrow ? Math.min(230, w - 70) : Math.min(w - dataW - 62, 240);
  const wx0 = narrow ? Math.round((w - S) / 2) + 14 : dataW + 48;
  const wy0 = narrow ? f.bottom + 70 : top + 14;
  const sx = linear([-WD, WD], [wx0, wx0 + S]);
  const sy = linear([-WD, WD], [wy0 + S, wy0]);
  const wid = `${st.uid}-wclip`;
  const priorCov = [[p.prior * p.prior, 0], [0, p.prior * p.prior]];
  parts.push(
    el("defs", {}, el("clipPath", { id: wid }, el("rect", { x: wx0, y: wy0, width: S, height: S }))),
    el("rect", { x: wx0, y: wy0, width: S, height: S, fill: C.panel }),
    axis({ scale: sx, orient: "bottom", at: wy0 + S, span: [wy0, wy0 + S], title: L.w1, count: 3 }),
    axis({ scale: sy, orient: "left", at: wx0, span: [wx0, wx0 + S], title: L.w2, count: 3 }),
    text(wx0 + S / 2, wy0 - 6, L.wTitle, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-strong" }),
    g({ "clip-path": `url(#${wid})` },
      ...[2, 1].map((r) => el("path", { d: ellipse(sx, sy, [0, 0], priorCov, r), fill: "none", stroke: C.ink3, "stroke-width": 1.2, "stroke-dasharray": "5 4" })),
      ...[2, 1].map((r) => el("path", { d: ellipse(sx, sy, post.mean, post.cov, r), fill: C.band, "fill-opacity": 0.8, stroke: C.model, "stroke-width": 1.3 })),
      ...ws.map((wv) => el("circle", { cx: sx(wv[0]), cy: sy(wv[1]), r: 3, fill: sampleColor, stroke: C.paper, "stroke-width": 1 })),
      el("circle", { cx: sx(post.mean[0]), cy: sy(post.mean[1]), r: 4, fill: C.model, stroke: C.paper, "stroke-width": 1.4 }),
    ),
  );
  // readout under the weight panel (wide) or under everything (narrow)
  let y = wy0 + S + 48;
  const x0 = narrow ? 8 : wx0 - 40;
  const line = (s: string, cls = "fig-t-muted") => { parts.push(text(x0, y, s, { "font-size": TYPE.small, class: `${cls} fig-t-num` })); y += 16; };
  line(pts.length ? tpl(L.fit, { a: fixed(post.mean[0], 2), b: fixed(post.mean[1], 2) }) : L.fitNone, "fig-t-strong");
  line(tpl(L.sd, { s: fixed(Math.sqrt(post.cov[1][1]), 2), i: fixed(Math.sqrt(post.cov[0][0]), 2) }));
  if (pts.length) line(tpl(L.ev, { e: fixed(post.logEvidence, 2) }));
  const swY = y + 2;
  parts.push(
    el("line", { x1: x0, x2: x0 + 16, y1: swY, y2: swY, stroke: C.ink3, "stroke-width": 1.2, "stroke-dasharray": "5 4" }),
    text(x0 + 22, swY + 4, L.priorE, { "font-size": TYPE.small, class: "fig-t-muted" }),
    el("rect", { x: x0 + 70, y: swY - 5, width: 16, height: 10, rx: 2, fill: C.band, stroke: C.model }),
    text(x0 + 92, swY + 4, L.postE, { "font-size": TYPE.small, class: "fig-t-muted" }),
  );
  y += 16;
  const H = Math.max(f.bottom + 44, y);
  return svg(w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "bayes-line",
  title: { en: "Bayesian linear regression in data space and in weight space", zh: "数据空间与权重空间中的贝叶斯线性回归" },
  labels,
  params,
  hint: { en: "Click the left plot to add an observation; click a point to remove it.", zh: "点击左图添加一个观测；点击一个点将其删除。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click") return null;
    const pts = parsePoints(p.points);
    if (e.target?.startsWith("pt-")) {
      pts.splice(Number(e.target.slice(3)), 1);
      return { ...p, points: fmtPoints(pts) };
    }
    if (e.target !== "plot" || !e.data) return null;
    if (pts.length >= 20) pts.shift();
    pts.push([clamp(e.data.x, -1, 1), clamp(e.data.y, YD[0], YD[1])]);
    return { ...p, points: fmtPoints(pts) };
  },
  actions: [
    { label: { en: "New samples", zh: "新样本" }, run: (p) => ({ ...p, seed: (p.seed % 999) + 1 }), enabled: (p) => p.samples },
    { label: { en: "Clear", zh: "清空" }, run: (p) => ({ ...p, points: "" }), enabled: (p) => p.points !== "" },
  ],
});
