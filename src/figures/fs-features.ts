// From weights to functions. A Bayesian linear model with M Gaussian-bump
// features, f(x) = sum_j w_j phi_j(x) with w_j ~ N(0, sigma_w^2), is already a
// prior over functions. The main panel draws three functions from it with the
// prior's 95% band; the strip below plots the covariance k_M(x0, x) that the
// features induce, against the RBF kernel it converges to when the bumps are
// packed densely (Rasmussen and Williams 2006, eqs. 4.10 to 4.13). Choosing
// "∞" drops the features and samples from the kernel directly.
//
// Bumps have width s = ℓ / sqrt(2), so the limit kernel has lengthscale ℓ.
// Centers are evenly spaced on [-3s, 1 + 3s], spacing Δ, and the weight
// variance is Δ / (s sqrt(pi)), which makes the limit amplitude exactly 1.

import { defineFigure, type State } from "./types.ts";
import { bandPath, el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid } from "./lib/scale.ts";
import { kernel, samplePrior } from "./lib/gp.ts";
import { memo, normals, rng } from "./lib/random.ts";
import { band, clip, curve, frame, frameAxes, hitArea, legend } from "./lib/plot.ts";
import { sig, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "input x",
    y: "f(x)",
    cov: "k(x₀, x)",
    samples: "functions drawn from the prior",
    band: "95% prior band",
    feats: "weighted bumps of the first function",
    model: "covariance the features induce",
    limit: "RBF kernel (the limit)",
    readout: "{m} features: prior sd {sd} on [0, 1]; largest gap to the RBF kernel {gap}",
    readoutShort1: "{m} features: prior sd {sd}",
    readoutShort2: "largest gap to the RBF kernel: {gap}",
    readoutInf: "no features: samples drawn from the RBF kernel itself; prior sd 1 everywhere",
    readoutInfShort: "no features: the RBF kernel itself",
    describe: "Three functions drawn from a linear model with {m} Gaussian-bump features of width {s}. The prior standard deviation {sdLong}, and the covariance with f at x₀ = {x0} differs from the RBF kernel's by {gapLong}.",
    describeInf: "Three functions drawn from a Gaussian process with an RBF kernel of lengthscale {l}, the limit of infinitely many Gaussian-bump features. The prior standard deviation is 1 at every input.",
    sdFlat: "{v} everywhere",
    sdRange: "from {lo} to {hi}",
    sdFlatLong: "is {v} at every input",
    sdRangeLong: "varies from {lo} to {hi} across the inputs",
    gapTiny: "less than 0.001",
    gapAtMost: "at most {g}",
  },
  zh: {
    x: "输入 x",
    y: "f(x)",
    cov: "k(x₀, x)",
    samples: "从先验中抽取的函数",
    band: "95% 先验区间",
    feats: "第一个函数的加权鼓包",
    model: "特征诱导的协方差",
    limit: "RBF 核（极限）",
    readout: "{m} 个特征：[0, 1] 上的先验标准差{sd}；与 RBF 核的最大差距 {gap}",
    readoutShort1: "{m} 个特征：先验标准差{sd}",
    readoutShort2: "与 RBF 核的最大差距：{gap}",
    readoutInf: "无特征：样本直接从 RBF 核中抽取；先验标准差处处为 1",
    readoutInfShort: "无特征：RBF 核本身",
    describe: "从带有 {m} 个高斯鼓包特征（宽度 {s}）的线性模型中抽取的三个函数。先验标准差{sdLong}，在 x₀ = {x0} 处与 f 的协方差和径向基函数核给出的协方差相差{gapLong}。",
    describeInf: "从长度尺度为 {l} 的径向基函数核高斯过程中抽取的三个函数，这是无穷多个高斯鼓包特征的极限。先验标准差在每个输入处都为 1。",
    sdFlat: "处处为 {v}",
    sdRange: "从 {lo} 到 {hi}",
    sdFlatLong: "在每个输入处都为 {v}",
    sdRangeLong: "在各输入之间从 {lo} 变化到 {hi}",
    gapTiny: "不到 0.001",
    gapAtMost: "至多 {g}",
  },
};

const FEATURES = [
  { value: "4", label: { en: "4", zh: "4" } },
  { value: "6", label: { en: "6", zh: "6" } },
  { value: "8", label: { en: "8", zh: "8" } },
  { value: "12", label: { en: "12", zh: "12" } },
  { value: "16", label: { en: "16", zh: "16" } },
  { value: "inf", label: { en: "∞", zh: "∞" } },
] as const;

const params = {
  features: { kind: "choice", label: { en: "Number of features M", zh: "特征数 M" }, options: FEATURES, default: "6", control: "buttons" },
  lengthscale: { kind: "range", label: { en: "Lengthscale ℓ", zh: "长度尺度 ℓ" }, min: 0.05, max: 0.3, default: 0.1, scale: "log" },
  showFeatures: { kind: "toggle", label: { en: "Show the bumps", zh: "显示鼓包" }, default: true },
  x0: { kind: "range", label: { en: "Reference input x₀", zh: "参考输入 x₀" }, min: 0, max: 1, default: 0.5, step: 0.01 },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 14, step: 1, control: false },
} as const;

type P = { features: string; lengthscale: number; showFeatures: boolean; x0: number; seed: number };

const N = 161;
const XS = grid(0, 1, N);
const YDOM: [number, number] = [-3.6, 3.6];
const COLORS = [C.c7, C.c5, C.c6];

// The feature model for M bumps and limit lengthscale ell.
function features(M: number, ell: number) {
  const s = ell / Math.SQRT2;
  const a = 3 * s;
  const centers = grid(-a, 1 + a, M);
  const delta = (1 + 2 * a) / (M - 1);
  const sw2 = delta / (s * Math.sqrt(Math.PI));
  const phi = (x: number, c: number) => Math.exp(-((x - c) ** 2) / (2 * s * s));
  return { s, centers, sw2, phi };
}

const compute = memo((fs: string, ell: number, x0: number, seed: number) => {
  const rbf = (a: number, b: number) => Math.exp(-((a - b) ** 2) / (2 * ell * ell));
  const limit = XS.map((x) => rbf(x0, x));
  if (fs === "inf") {
    const S = samplePrior(kernel("rbf", ell, 1), XS, rng(seed), 3);
    return { inf: true as const, S, sd: XS.map(() => 1), covM: limit, limit, parts: [] as number[][], gap: 0, s: ell / Math.SQRT2, centers: [] as number[] };
  }
  const M = Number(fs);
  const { s, centers, sw2, phi } = features(M, ell);
  const r = rng(seed);
  const W = [0, 1, 2].map(() => normals(r, M).map((z) => Math.sqrt(sw2) * z));
  const Phi = XS.map((x) => centers.map((c) => phi(x, c)));
  const S = W.map((w) => Phi.map((row) => row.reduce((acc, v, j) => acc + v * w[j], 0)));
  const sd = Phi.map((row) => Math.sqrt(sw2 * row.reduce((acc, v) => acc + v * v, 0)));
  const p0 = centers.map((c) => phi(x0, c));
  const covM = Phi.map((row) => sw2 * row.reduce((acc, v, j) => acc + v * p0[j], 0));
  let gap = 0;
  for (let i = 0; i < N; i++) gap = Math.max(gap, Math.abs(covM[i] - limit[i]));
  // The first function's weighted bumps, only those that reach into [0, 1].
  const parts = centers.flatMap((c, j) => (c > -2.5 * s && c < 1 + 2.5 * s ? [XS.map((x) => W[0][j] * phi(x, c))] : []));
  return { inf: false as const, S, sd, covM, limit, parts, gap, s, centers };
}, 24);

function fmtGap(v: number): string {
  return v < 0.001 ? "< 0.001" : sig(v, 2);
}

// The range of the prior standard deviation, or a single value when flat.
function sdRange(sd: number[], L: typeof labels.en): { short: string; long: string } {
  const lo = Math.min(...sd), hi = Math.max(...sd);
  if (sig(lo, 2) === sig(hi, 2)) return { short: tpl(L.sdFlat, { v: sig(hi, 2) }), long: tpl(L.sdFlatLong, { v: sig(hi, 2) }) };
  return { short: tpl(L.sdRange, { lo: sig(lo, 2), hi: sig(hi, 2) }), long: tpl(L.sdRangeLong, { lo: sig(lo, 2), hi: sig(hi, 2) }) };
}

function describe(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const c = compute(p.features, p.lengthscale, p.x0, p.seed);
  if (c.inf) return tpl(L.describeInf, { l: sig(p.lengthscale, 2) });
  return tpl(L.describe, {
    m: p.features, s: sig(c.s, 2), sdLong: sdRange(c.sd, L).long,
    x0: sig(p.x0, 2), gapLong: c.gap < 0.001 ? L.gapTiny : tpl(L.gapAtMost, { g: sig(c.gap, 2) }),
  });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const c = compute(p.features, p.lengthscale, p.x0, p.seed);
  const showParts = p.showFeatures && !c.inf;
  const lg = legend(narrow ? 8 : 40, 14, st.w - 16, [
    { kind: "line", color: COLORS[0], label: L.samples },
    { kind: "band", color: C.band, label: L.band },
    ...(showParts ? [{ kind: "dash" as const, color: COLORS[0], label: L.feats }] : []),
  ]);
  const top = 14 + lg.height + 6;
  const mainH = narrow ? 190 : 230;
  const f = frame({ w: st.w, top, height: mainH, yDomain: YDOM, yTitle: L.y });
  // The covariance strip: its own legend row, then the plot.
  const lg2 = legend(narrow ? 8 : 40, f.bottom + 22, st.w - 16, c.inf
    ? [{ kind: "line", color: C.model, label: L.limit }]
    : [{ kind: "line", color: C.model, label: L.model }, { kind: "dash", color: C.ink, label: L.limit }]);
  const stripTop = f.bottom + 22 + lg2.height + 2;
  const stripH = narrow ? 70 : 84;
  const peak = Math.max(1, ...c.covM);
  const yTop = peak <= 1.05 ? 1.1 : Math.ceil(peak * 1.1 * 2) / 2;
  const fs = frame({ w: st.w, top: stripTop, height: stripH, yDomain: [0, yTop], yTitle: L.cov });
  const readTop = fs.bottom + 56;
  const lines = c.inf
    ? [narrow ? L.readoutInfShort : L.readoutInf]
    : narrow
      ? [tpl(L.readoutShort1, { m: p.features, sd: sdRange(c.sd, L).short }), tpl(L.readoutShort2, { gap: fmtGap(c.gap) })]
      : [tpl(L.readout, { m: p.features, sd: sdRange(c.sd, L).short, gap: fmtGap(c.gap) })];
  const H = readTop + lines.length * 16 + 4;
  const cid = `${st.uid}-clip`;
  const cid2 = `${st.uid}-clip2`;
  const lo = c.sd.map((v) => -1.96 * v);
  const hi = c.sd.map((v) => 1.96 * v);
  const x0px = f.x(p.x0);
  return svg(st.w, H, describe(st),
    el("defs", {}, clip(f, cid), clip(fs, cid2)),
    lg.svg,
    frameAxes(f, { yTitle: L.y, xTicks: false }),
    hitArea(f, "plot"),
    g({ "clip-path": `url(#${cid})` },
      band(f, XS, lo, hi),
      showParts ? c.parts.map((part) => el("path", {
        d: bandPath(XS.map((x, i) => [f.x(x), f.y(part[i])]), XS.map((x) => [f.x(x), f.y(0)])),
        fill: COLORS[0], opacity: 0.2,
      }) + curve(f, XS, part, { stroke: COLORS[0], "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0.8 })).join("") : "",
      c.S.map((s, i) => curve(f, XS, s, { stroke: COLORS[i], "stroke-width": i === 0 ? 2 : 1.5, opacity: i === 0 ? 1 : 0.85 })).join(""),
    ),
    // bump centers, as ticks under the plot
    c.centers.filter((x) => x >= 0 && x <= 1).map((x) => el("path", { d: `M${f.x(x)},${f.bottom - 1}l-4,7h8z`, fill: C.ink3 })).join(""),
    el("line", { x1: x0px, x2: x0px, y1: f.top, y2: fs.bottom, stroke: C.ink3, "stroke-width": 1, "stroke-dasharray": "2 3" }),
    // the strip
    lg2.svg,
    frameAxes(fs, { xTitle: L.x, yTitle: narrow ? "" : L.cov, yCount: 2 }),
    hitArea(fs, "strip"),
    g({ "clip-path": `url(#${cid2})` },
      c.inf ? "" : curve(fs, XS, c.limit, { stroke: C.ink, "stroke-width": 1.3, "stroke-dasharray": "5 4" }),
      curve(fs, XS, c.covM, { stroke: C.model, "stroke-width": 2 }),
    ),
    el("circle", { cx: x0px, cy: fs.y(c.covM[Math.round(p.x0 * (N - 1))]), r: 3.6, fill: C.model, stroke: C.paper, "stroke-width": 1.5 }),
    text(x0px, f.top - 4, "x_0", { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
    ...lines.map((s, i) => text(narrow ? 8 : f.left, readTop + i * 16, s, { "font-size": TYPE.small, class: "fig-t-num" })),
  );
}

function setX0(p: P, x: number | undefined): P | null {
  if (x === undefined) return null;
  return { ...p, x0: Math.round(Math.min(1, Math.max(0, x)) * 100) / 100 };
}

export default defineFigure({
  name: "fs-features",
  title: { en: "Functions built from Gaussian-bump features, and the kernel they converge to", zh: "由高斯鼓包特征构造的函数，以及它们收敛到的核函数" },
  labels,
  params,
  hint: { en: "Click or drag on either panel to move x₀.", zh: "在任一面板上点击或拖动，可以移动 x₀。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.target !== "plot" && e.target !== "strip") return null;
    return setX0(p, e.data?.x);
  },
  actions: [
    { label: { en: "Draw new weights", zh: "重新抽取权重" }, primary: true, run: (p) => ({ ...p, seed: (p.seed % 9999) + 1 }) },
  ],
});
