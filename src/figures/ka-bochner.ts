// Bochner's theorem and random Fourier features. A stationary kernel with
// unit amplitude is k(r) = E[cos(ω r)] for a frequency ω drawn from its
// spectral density p(ω): a Gaussian with standard deviation 1/ℓ for the RBF
// kernel, a Student-t with 2ν degrees of freedom and scale 1/ℓ for the Matérn
// kernel (Rasmussen and Williams 2006, eq. 4.15, rewritten as a density over
// angular frequency). Drawing M frequencies gives the random features of
// Rahimi and Recht (2007), z(x) = (cos ω_j x, sin ω_j x)_j / √M; their inner
// products average M cosines and approximate the kernel, and a weighted sum of
// them with standard normal weights is an approximate draw from the process.
// Frequencies are generated in a fixed order from the seed, so raising M adds
// features without replacing the earlier ones, and switching kernels reuses
// the same underlying normal draws.

import { defineFigure, tr, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear, log } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { kernel, samplePrior, type KernelName } from "./lib/gp.ts";
import { memo, normals, rng } from "./lib/random.ts";
import { clip, curve, frame, frameAxes, legend } from "./lib/plot.ts";
import { fixed, sig, tpl } from "./lib/format.ts";

const labels = {
  en: {
    spec: "Spectral density p(ω), relative to p(0)",
    specX: "frequency × lengthscale, ωℓ",
    kern: "Kernel k(r) and its estimate from M features",
    kernX: "distance r",
    exact: "exact k(r)",
    approx: "(1/M) Σ cos(ω_j r)",
    rbfRef: "RBF, for comparison",
    drawn: "drawn frequencies",
    sample: "A draw from M random features, and an exact draw",
    feat: "sum of M = {m} features",
    exactDraw: "exact draw (Cholesky)",
    x: "input x",
    gap: "Root-mean-square gap between estimate and kernel on [0, 1]: {g}; 1/√(2M) = {s}",
    gap1: "Root-mean-square gap on [0, 1]: {g};",
    gap2: "for comparison, 1/√(2M) = {s}",
    describe: "{k} kernel with lengthscale {l}, approximated by M = {m} random Fourier features. The root-mean-square gap between the estimated and the exact kernel on [0, 1] is {g}, against 1/√(2M) = {s}.",
  },
  zh: {
    spec: "谱密度 p(ω) 与 p(0) 之比",
    specX: "频率 × 长度尺度，ωℓ",
    kern: "核函数 k(r) 及其由 M 个特征得到的估计",
    kernX: "距离 r",
    exact: "精确的 k(r)",
    approx: "(1/M) Σ cos(ω_j r)",
    rbfRef: "RBF，供对照",
    drawn: "抽取的频率",
    sample: "由 M 个随机特征构造的样本与精确样本",
    feat: "M = {m} 个特征之和",
    exactDraw: "精确样本（Cholesky 方法）",
    x: "输入 x",
    gap: "估计与核函数在 [0, 1] 上的均方根差距：{g}；1/√(2M) = {s}",
    gap1: "[0, 1] 上的均方根差距：{g}；",
    gap2: "对照值 1/√(2M) = {s}",
    describe: "长度尺度为 {l} 的 {k} 核，用 M = {m} 个随机 Fourier 特征近似。估计的核函数与精确核函数在 [0, 1] 上的均方根差距为 {g}，对照值 1/√(2M) = {s}。",
  },
};

const KERNELS = [
  { value: "rbf", label: { en: "RBF", zh: "RBF" } },
  { value: "matern52", label: { en: "Matérn 5/2", zh: "Matérn 5/2" } },
  { value: "matern32", label: { en: "Matérn 3/2", zh: "Matérn 3/2" } },
  { value: "matern12", label: { en: "Matérn 1/2", zh: "Matérn 1/2" } },
] as const;
const COLOR: Record<string, string> = { rbf: C.c7, matern52: C.c6, matern32: C.c8, matern12: C.c5 };
const NU: Record<string, number> = { matern52: 2.5, matern32: 1.5, matern12: 0.5 };
const MMAX = 2000;

const params = {
  kernel: { kind: "choice", label: { en: "Kernel", zh: "核函数" }, options: KERNELS, default: "matern32" },
  lengthscale: { kind: "range", label: { en: "Lengthscale ℓ", zh: "长度尺度 ℓ" }, min: 0.03, max: 0.5, default: 0.1, scale: "log" },
  features: { kind: "range", label: { en: "Features M", zh: "特征数 M" }, min: 1, max: MMAX, default: 20, scale: "log" },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 5, step: 1, control: false },
} as const;

type P = { kernel: string; lengthscale: number; features: number; seed: number };

// Spectral density of a unit-amplitude kernel as a probability density over
// angular frequency, as a function of u = ωℓ and divided by its value at 0.
export function relDensity(kname: string, u: number): number {
  if (kname === "rbf") return Math.exp(-0.5 * u * u);
  const nu = NU[kname];
  return Math.pow(1 + (u * u) / (2 * nu), -(nu + 0.5));
}

// The underlying normal draws for a seed: z0 for the numerator of every
// frequency, five more per frequency for the chi-square denominator of a
// Student-t with up to 5 degrees of freedom, and the feature weights.
const base = memo((seed: number) => {
  const r = rng(seed);
  const z0 = normals(r, MMAX);
  const chi = Array.from({ length: MMAX }, () => normals(r, 5));
  const a = normals(rng(seed + 7919), MMAX), b = normals(rng(seed + 104729), MMAX);
  return { z0, chi, a, b };
}, 4);

// u_j = ω_j ℓ, a draw from the scaled spectral density.
export const freqs = memo((kname: string, seed: number): number[] => {
  const { z0, chi } = base(seed);
  if (kname === "rbf") return z0.slice();
  const dof = Math.round(2 * NU[kname]);
  return z0.map((z, j) => {
    let s = 0;
    for (let i = 0; i < dof; i++) s += chi[j][i] * chi[j][i];
    return z / Math.sqrt(s / dof);
  });
}, 8);

const RS = Array.from({ length: 121 }, (_, i) => i / 120);
const XS = Array.from({ length: 301 }, (_, i) => i / 300);
const XE = Array.from({ length: 151 }, (_, i) => i / 150);

const exactDraw = memo((kname: string, ls: number, seed: number) => samplePrior(kernel(kname as KernelName, ls, 1), XE, rng(seed + 31337), 1)[0], 8);

export function compute(p: P) {
  const M = Math.max(1, Math.min(MMAX, Math.round(p.features)));
  const u = freqs(p.kernel, p.seed).slice(0, M);
  const w = u.map((v) => v / p.lengthscale);
  const k = kernel(p.kernel as KernelName, p.lengthscale, 1);
  const kExact = RS.map((r) => k(0, r));
  const kHat = RS.map((r) => w.reduce((s, om) => s + Math.cos(om * r), 0) / M);
  let ss = 0;
  for (let i = 0; i < RS.length; i++) ss += (kHat[i] - kExact[i]) ** 2;
  const gap = Math.sqrt(ss / RS.length);
  const { a, b } = base(p.seed);
  const f = XS.map((x) => {
    let s = 0;
    for (let j = 0; j < M; j++) s += a[j] * Math.cos(w[j] * x) + b[j] * Math.sin(w[j] * x);
    return s / Math.sqrt(M);
  });
  return { M, u, kExact, kHat, gap, f, fe: exactDraw(p.kernel, p.lengthscale, p.seed) };
}

function describe(st: State<P>): string {
  const p = st.p;
  const c = compute(p);
  const lang = st.lang ?? "en";
  return tpl(labels[lang].describe, {
    k: tr(KERNELS.find((k) => k.value === p.kernel)!.label, lang), l: sig(p.lengthscale, 2), m: c.M,
    g: fixed(c.gap, 3), s: fixed(1 / Math.sqrt(2 * c.M), 3),
  });
}

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 480;
  const c = compute(p);
  const parts: string[] = [];
  const col = COLOR[p.kernel];

  // ---- headers
  const aw = narrow ? st.w : Math.floor(st.w * 0.5);
  const bx0 = narrow ? 0 : aw + 4;
  const bw = narrow ? st.w : st.w - aw - 4;
  const aItems = [
    { kind: "line" as const, color: col, label: tr(KERNELS.find((k) => k.value === p.kernel)!.label, lang) },
    ...(p.kernel !== "rbf" ? [{ kind: "dash" as const, color: C.ink3, label: L.rbfRef }] : []),
  ];
  const aLeg = legend(4, 30, aw - 8, aItems);
  const bItems = [{ kind: "dash" as const, color: C.ink, label: L.exact }, { kind: "line" as const, color: col, label: L.approx }];
  const bLeg = legend(bx0 + 4, 30, bw - 8, bItems);
  const head = narrow ? 30 + aLeg.height : 30 + Math.max(aLeg.height, bLeg.height);

  // ---- panel A: relative spectral density on a log scale, with a rug of drawn frequencies
  const topA = head, hA = narrow ? 130 : 150;
  const al = narrow ? 44 : 50, ar = aw - 10;
  const xu = linear([0, 10], [al, ar]);
  const yd = log([1e-4, 1.5], [topA + hA, topA]);
  parts.push(text(4, 14, L.spec, { "font-size": TYPE.small, class: "fig-t-strong" }), aLeg.svg);
  parts.push(
    axis({ scale: yd, orient: "left", at: al, span: [al, ar], ticks: [1e-4, 1e-3, 1e-2, 1e-1, 1], format: (v) => (v === 1 ? "1" : `10${{ "-4": "⁻⁴", "-3": "⁻³", "-2": "⁻²", "-1": "⁻¹" }[String(Math.round(Math.log10(v)))] ?? ""}`) }),
    axis({ scale: xu, orient: "bottom", at: topA + hA, span: [topA, topA + hA], ticks: [0, 2, 4, 6, 8, 10], title: L.specX, format: (v) => String(v) }),
  );
  const us = Array.from({ length: 201 }, (_, i) => i / 20);
  const aid = `${st.uid}-aclip`;
  const densPath = (kn: string) => linePath(us.map((u) => [xu(u), yd(Math.max(relDensity(kn, u), 1e-6))]));
  parts.push(
    el("defs", {}, el("clipPath", { id: aid }, el("rect", { x: al, y: topA - 2, width: ar - al, height: hA + 4 }))),
    g({ "clip-path": `url(#${aid})` },
      p.kernel !== "rbf" ? el("path", { d: densPath("rbf"), fill: "none", stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "4 3" }) : "",
      el("path", { d: densPath(p.kernel), fill: "none", stroke: col, "stroke-width": 2.2 }),
    ),
  );
  // rug: |u_j| for the first 200 features; those beyond the axis pile up at its end
  const rug = c.u.slice(0, 200).map((u) => Math.min(Math.abs(u), 10));
  parts.push(g({}, ...rug.map((u) => el("line", { x1: xu(u), x2: xu(u), y1: topA + hA - 7, y2: topA + hA, stroke: col, "stroke-width": 1, opacity: 0.7 }))));

  // ---- panel B: kernel and estimate
  const topB = narrow ? topA + hA + 52 + 30 + bLeg.height : topA;
  const hB = narrow ? 120 : 150;
  const fb = frame({ w: bw, top: topB, height: hB, yDomain: [-0.6, 1.1] });
  const bTitleY = narrow ? topB - 16 - bLeg.height : 14;
  const shift = (s: string) => g({ transform: `translate(${bx0},0)` }, s);
  parts.push(text(bx0 + 4, bTitleY, L.kern, { "font-size": TYPE.small, class: "fig-t-strong" }));
  parts.push(narrow ? legend(4, bTitleY + 16, bw - 8, bItems).svg : bLeg.svg);
  const bid = `${st.uid}-bclip`;
  parts.push(shift(
    el("defs", {}, clip(fb, bid)) + frameAxes(fb, { xTitle: L.kernX, yCount: 3 }) +
    g({ "clip-path": `url(#${bid})` },
      el("line", { x1: fb.left, x2: fb.right, y1: fb.y(0), y2: fb.y(0), stroke: C.rule }),
      curve(fb, RS, c.kHat, { stroke: col, "stroke-width": 2 }),
      curve(fb, RS, c.kExact, { stroke: C.ink, "stroke-width": 1.6, "stroke-dasharray": "5 3" }),
    ),
  ));

  // ---- panel C: draws
  const topC = (narrow ? topB + hB : topA + Math.max(hA, hB)) + 78;
  const hC = narrow ? 130 : 150;
  let smax = 2.5;
  for (const v of c.f) smax = Math.max(smax, Math.abs(v) * 1.1);
  for (const v of c.fe) smax = Math.max(smax, Math.abs(v) * 1.1);
  smax = Math.ceil(smax * 2) / 2;
  const fc = frame({ w: st.w, top: topC, height: hC, yDomain: [-smax, smax], yTitle: "f(x)" });
  const cLeg = legend(4, topC - 6, st.w - 8, [
    { kind: "line", color: col, label: tpl(L.feat, { m: c.M }) },
    { kind: "line", color: C.ink3, label: L.exactDraw },
  ]);
  parts.push(text(4, topC - 22 - (cLeg.height - 18), L.sample, { "font-size": TYPE.small, class: "fig-t-strong" }));
  const cid = `${st.uid}-cclip`;
  // shift the legend up when it wraps, so it never overlaps the plot
  parts.push(g({ transform: `translate(0,${-(cLeg.height - 18)})` }, cLeg.svg));
  parts.push(
    el("defs", {}, clip(fc, cid)),
    frameAxes(fc, { xTitle: L.x, yTitle: "f(x)", yCount: 4 }),
    g({ "clip-path": `url(#${cid})` },
      curve(fc, XE, c.fe, { stroke: C.ink3, "stroke-width": 1.4 }),
      curve(fc, XS, c.f, { stroke: col, "stroke-width": 2 }),
    ),
  );

  let ry = topC + hC + 50;
  const vals = { g: fixed(c.gap, 3), s: fixed(1 / Math.sqrt(2 * c.M), 3) };
  if (narrow) {
    parts.push(text(4, ry, tpl(L.gap1, vals), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
    ry += 16;
    parts.push(text(4, ry, tpl(L.gap2, vals), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  } else {
    parts.push(text(4, ry, tpl(L.gap, vals), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  }
  return svg(st.w, ry + 10, describe(st), ...parts);
}

export default defineFigure({
  name: "ka-bochner",
  title: { en: "A kernel's spectral density, and random Fourier features drawn from it", zh: "核函数的谱密度，以及从中抽取的随机 Fourier 特征" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "Draw new frequencies", zh: "重新抽取频率" }, primary: true, run: (p) => ({ ...p, seed: (p.seed % 9999) + 1 }) },
  ],
  snapshots: {
    "M = 200": (p) => ({ ...p, features: 200 }),
    "M = 2000": (p) => ({ ...p, features: 2000 }),
    "RBF": (p) => ({ ...p, kernel: "rbf" }),
    "Matérn 1/2": (p) => ({ ...p, kernel: "matern12" }),
  },
});
