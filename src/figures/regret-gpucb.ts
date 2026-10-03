// GP-UCB against its own regret bound. A function f is drawn from a Gaussian
// process on a finite set D of 160 inputs (a grid on [0, 1], or a Latin
// hypercube in [0, 1]^3 or [0, 1]^6), so the setting is exactly the
// one of the finite-domain theorem of Srinivas et al. (2010): f ~ GP(0, k)
// with k(x, x) = 1, Gaussian observation noise of known variance, and the
// algorithm uses the same kernel. Two GP-UCB runs face the same noise draws:
// one with the theorem's beta_t = 2 log(|D| t^2 pi^2 / (6 delta)), one with a
// constant multiplier of the reader's choosing. The bottom panel plots both
// cumulative regrets, the expected regret of uniformly random queries, and the
// bound sqrt(C1 t beta_t gamma_t), on a log scale.
//
// gamma_t, the maximum information gain, is NP-hard to compute. The figure
// uses the greedy maximizer (repeatedly observe the input with the largest
// posterior variance), which reaches at least (1 - 1/e) of the maximum, so
// the plotted bound can only understate the theorem's.

import { defineFigure, tr, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear, log as logScale } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { kernel, samplePrior, type KernelName, type X } from "./lib/gp.ts";
import { latinHypercube } from "./lib/nd.ts";
import type { Frame, Swatch } from "./lib/plot.ts";
import { memo, normal, rng } from "./lib/random.ts";
import { fixed, sig, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "input x (grid of 160 points)",
    xnd: "first coordinate x₁ of 160 points in {d} dimensions",
    f: "f(x)",
    t: "round t",
    regret: "cumulative regret (log scale)",
    regretShort: "regret (log)",
    truth: "f, drawn from the GP",
    theory: "GP-UCB, theorem's β_t",
    practical: "GP-UCB, √β = {b}",
    random: "random queries (expected)",
    bound: "bound √(C_1 t β_t γ_t)",
    rugT: "β_t",
    rugP: "√β={b}",
    consts: "C_1 = {c} · √β_T = {b} · γ_T ≥ {g}",
    ratio: "at T = {T}: bound {B} · random queries {rr}",
    runs: "regret {r} with the theorem's β_t · {q} with √β = {b}",
    runsShort: "regret {r} (β_t) · {q} (√β = {b})",
    describe: "A function drawn from a Gaussian process in {d} dimension{d:/s} with a {k} kernel, lengthscale {l}, noise sd {s}. After {T} rounds, GP-UCB with the theorem's beta has cumulative regret {rt}, GP-UCB with a constant multiplier of {b} has {rp}, and random queries would expect {rr}. The bound evaluates to {B}, with C1 = {c} and the greedy information gain {g}.",
  },
  zh: {
    x: "输入 x（160 个点的网格）",
    xnd: "{d} 维中 160 个点的第一个坐标 x₁",
    f: "f(x)",
    t: "轮次 t",
    regret: "累积遗憾（对数刻度）",
    regretShort: "遗憾（对数）",
    truth: "f，抽取自高斯过程",
    theory: "GP-UCB，定理中的 β_t",
    practical: "GP-UCB，√β = {b}",
    random: "随机查询（期望）",
    bound: "遗憾界 √(C_1 t β_t γ_t)",
    rugT: "β_t",
    rugP: "√β={b}",
    consts: "C_1 = {c} · √β_T = {b} · γ_T ≥ {g}",
    ratio: "T = {T} 时：遗憾界 {B} · 随机查询 {rr}",
    runs: "遗憾：用定理的 β_t 为 {r} · 用 √β = {b} 为 {q}",
    runsShort: "遗憾 {r}（β_t）· {q}（√β = {b}）",
    describe: "从 {d} 维高斯过程中抽取的函数，核函数为 {k}，长度尺度 {l}，噪声标准差 {s}。{T} 轮之后，采用定理中 β 的 GP-UCB 累积遗憾为 {rt}，采用常数乘子 {b} 的 GP-UCB 为 {rp}，随机查询的期望累积遗憾为 {rr}。遗憾界的值为 {B}，其中 C1 = {c}，贪心法估计的信息增益为 {g}。",
  },
};

const KERNELS = [
  { value: "rbf", label: { en: "RBF", zh: "RBF 核" } },
  { value: "matern52", label: { en: "Matérn 5/2", zh: "Matérn 5/2 核" } },
  { value: "matern12", label: { en: "Matérn 1/2", zh: "Matérn 1/2 核" } },
] as const;

const DIMS = [
  { value: 1, label: { en: "1-D", zh: "1 维" } },
  { value: 3, label: { en: "3-D", zh: "3 维" } },
  { value: 6, label: { en: "6-D", zh: "6 维" } },
] as const;

const params = {
  dim: { kind: "choice", label: { en: "Dimension", zh: "维度" }, options: DIMS, default: 1, control: "buttons" },
  kernel: { kind: "choice", label: { en: "Kernel", zh: "核函数" }, options: KERNELS, default: "rbf" },
  lengthscale: { kind: "range", label: { en: "Lengthscale ℓ", zh: "长度尺度 ℓ" }, min: 0.03, max: 0.5, default: 0.1, scale: "log" },
  noise: { kind: "range", label: { en: "Noise sd σ_n", zh: "噪声标准差 σ_n" }, min: 0.02, max: 1, default: 0.1, scale: "log" },
  beta: { kind: "range", label: { en: "Practical √β", zh: "实用 √β" }, min: 0, max: 4, default: 2, step: 0.1 },
  horizon: { kind: "range", label: { en: "Rounds T", zh: "轮数 T" }, min: 20, max: 300, default: 200, step: 10 },
  delta: { kind: "range", label: { en: "δ", zh: "δ" }, min: 0.01, max: 0.5, default: 0.1, step: 0.01, control: false },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 13, step: 1, control: false },
} as const;

type P = { dim: number; kernel: string; lengthscale: number; noise: number; beta: number; horizon: number; delta: number; seed: number };

const M = 160;
const XS = grid(0, 1, M);

interface Run { queries: number[]; cum: number[] }
interface Result {
  x1: number[]; f: number[]; fmax: number; fmean: number; argmax: number;
  theory: Run; practical: Run; gamma: number[]; betaT: number[]; c1: number; bound: number[]; random: number[];
}

// Sequential GP updates on the grid: the posterior mean and the full
// covariance over D, updated by a rank-one correction per observation.
function gpucb(K0: Float64Array, f: number[], sigma2: number, T: number, beta: (t: number) => number, z: number[]): Run {
  const S = Float64Array.from(K0);
  const mu = new Float64Array(M);
  const fmax = Math.max(...f);
  const queries: number[] = [], cum: number[] = [];
  let c = 0;
  for (let t = 1; t <= T; t++) {
    const b = Math.sqrt(beta(t));
    let bi = 0, bv = -Infinity;
    for (let i = 0; i < M; i++) {
      const v = mu[i] + b * Math.sqrt(Math.max(S[i * M + i], 0));
      if (v > bv) { bv = v; bi = i; }
    }
    const y = f[bi] + Math.sqrt(sigma2) * z[t - 1];
    update(S, mu, bi, y, sigma2);
    c += fmax - f[bi];
    queries.push(bi);
    cum.push(c);
  }
  return { queries, cum };
}

function update(S: Float64Array, mu: Float64Array | null, i: number, y: number, sigma2: number) {
  const s = new Float64Array(M);
  for (let j = 0; j < M; j++) s[j] = S[j * M + i];
  const d = s[i] + sigma2;
  if (mu) { const r = (y - mu[i]) / d; for (let j = 0; j < M; j++) mu[j] += s[j] * r; }
  for (let a = 0; a < M; a++) {
    const sa = s[a] / d;
    if (sa === 0) continue;
    for (let b = 0; b < M; b++) S[a * M + b] -= sa * s[b];
  }
}

// Greedy information gain: observe the input of largest posterior variance;
// the gain of each observation is (1/2) log(1 + var / sigma^2).
function greedyGamma(K0: Float64Array, sigma2: number, T: number): number[] {
  const S = Float64Array.from(K0);
  const out: number[] = [];
  let gsum = 0;
  for (let t = 1; t <= T; t++) {
    let bi = 0, bv = -Infinity;
    for (let i = 0; i < M; i++) if (S[i * M + i] > bv) { bv = S[i * M + i]; bi = i; }
    gsum += 0.5 * Math.log(1 + Math.max(bv, 0) / sigma2);
    out.push(gsum);
    update(S, null, bi, 0, sigma2);
  }
  return out;
}

// The finite domain D: a grid in one dimension, a fixed Latin hypercube in
// more (the same points for every function, so only f changes with the seed).
const domain = memo((dim: number): X[] => (dim === 1 ? XS.slice() : latinHypercube(rng(4000 + dim), M, dim)), 4);

const compute = memo((dim: number, kname: string, ls: number, noise: number, bp: number, T: number, delta: number, seed: number): Result => {
  const k = kernel(kname as KernelName, ls, 1);
  const D = domain(dim);
  const K0 = new Float64Array(M * M);
  for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) K0[i * M + j] = k(D[i], D[j]);
  const f = samplePrior(k, D, rng(seed * 101 + 1 + 7919 * (dim - 1)), 1)[0];
  const x1 = D.map((x) => (typeof x === "number" ? x : x[0]));
  const fmax = Math.max(...f);
  const fmean = f.reduce((a, b) => a + b, 0) / M;
  const zr = normal(rng(seed * 977 + 13));
  const z = Array.from({ length: T }, () => zr());
  const sigma2 = noise * noise;
  const betaTheory = (t: number) => 2 * Math.log((M * t * t * Math.PI ** 2) / (6 * delta));
  const theory = gpucb(K0, f, sigma2, T, betaTheory, z);
  const practical = gpucb(K0, f, sigma2, T, () => bp * bp, z);
  const gamma = greedyGamma(K0, sigma2, T);
  const c1 = 8 / Math.log(1 + 1 / sigma2);
  const betaT = Array.from({ length: T }, (_, i) => betaTheory(i + 1));
  const bound = gamma.map((gm, i) => Math.sqrt(c1 * (i + 1) * betaT[i] * gm));
  const random = Array.from({ length: T }, (_, i) => (i + 1) * (fmax - fmean));
  return { x1, f, fmax, fmean, argmax: f.indexOf(fmax), theory, practical, gamma, betaT, c1, bound, random };
}, 8);

const run = (p: P) => compute(p.dim, p.kernel, p.lengthscale, p.noise, p.beta, p.horizon, p.delta, p.seed);

function describe(st: State<P>): string {
  const p = st.p;
  const r = run(p);
  const T = p.horizon;
  const lang = st.lang ?? "en";
  return tpl(labels[lang].describe, {
    d: p.dim, k: tr(KERNELS.find((x) => x.value === p.kernel)!.label, lang), l: sig(p.lengthscale, 2), s: sig(p.noise, 2), T,
    rt: fixed(r.theory.cum[T - 1], 1), rp: fixed(r.practical.cum[T - 1], 1), b: fixed(p.beta, 1), rr: fixed(r.random[T - 1], 0),
    B: fixed(r.bound[T - 1], 0), c: fixed(r.c1, 2), g: fixed(r.gamma[T - 1], 1),
  });
}

// Characters drawn at full width (CJK and full-width punctuation): about 11 px
// at this font size, against 6.4 px for Latin. Zero for an English label.
const wide = (s: string) => (s.match(/[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]/g) ?? []).length;

// A legend row like lib/plot.ts legend, with more room per entry so that a
// label never runs into the next swatch.
function legend(x: number, y: number, w: number, items: Swatch[]): { svg: string; height: number } {
  const parts: string[] = [];
  let cx = x, cy = y;
  const rowH = 18;
  for (const it of items) {
    const tw = it.label.replace(/[_^]/g, "").length * 6.4 + wide(it.label) * 4.8 + 40;
    if (cx + tw > x + w && cx > x) { cx = x; cy += rowH; }
    parts.push(
      el("line", { x1: cx, x2: cx + 18, y1: cy - 1, y2: cy - 1, stroke: it.color, "stroke-width": 2, "stroke-dasharray": it.kind === "dash" ? "4 3" : undefined }),
      text(cx + 24, cy + 3, it.label, { "font-size": TYPE.small, class: "fig-t-muted" }),
    );
    cx += tw;
  }
  return { svg: g({}, ...parts), height: cy - y + rowH };
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const r = run(p);
  const T = p.horizon;
  const bl = fixed(p.beta, 1);
  const parts: string[] = [];
  const lg = legend(narrow ? 8 : 40, 14, st.w - 16, [
    { kind: "dash", color: C.truth, label: L.truth },
    { kind: "line", color: C.c7, label: L.theory },
    { kind: "line", color: C.acq, label: tpl(L.practical, { b: bl }) },
    { kind: "dash", color: C.ink3, label: L.random },
    { kind: "dash", color: C.c5, label: L.bound },
  ]);
  parts.push(lg.svg);

  // Top panel: the function and where each run queried.
  const fl = narrow ? 40 : 52;
  const top = 14 + lg.height + 6;
  const fH = narrow ? 84 : 104;
  const fmin = Math.min(...r.f), fmax = r.fmax;
  const pad = 0.12 * (fmax - fmin || 1);
  const fx = linear([0, 1], [fl, st.w - 14]);
  const fy = linear([fmin - pad, fmax + pad], [top + fH, top]);
  const ff: Frame = { x: fx, y: fy, left: fl, right: st.w - 14, top, bottom: top + fH, w: st.w };
  parts.push(axis({ scale: fy, orient: "left", at: ff.left, span: [ff.left, ff.right], title: narrow ? undefined : L.f, count: 3 }));
  parts.push(el("line", { x1: ff.left, x2: ff.right, y1: ff.bottom, y2: ff.bottom, stroke: C.rule }));
  if (p.dim === 1) {
    parts.push(el("path", { d: XS.map((x, i) => `${i ? "L" : "M"}${fx(x).toFixed(1)},${fy(r.f[i]).toFixed(1)}`).join(""), fill: "none", stroke: C.truth, "stroke-width": 1.8, "stroke-dasharray": "5 4" }));
  } else {
    // In more dimensions, every point of D plotted against its first coordinate.
    parts.push(...r.f.map((v, i) => el("circle", { cx: fx(r.x1[i]), cy: fy(v), r: 2.2, fill: C.truth, opacity: 0.7 })));
    parts.push(text(ff.right, ff.bottom - 5, tpl(L.xnd, { d: p.dim }), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo" }));
  }
  parts.push(el("circle", { cx: fx(r.x1[r.argmax]), cy: fy(fmax), r: 4, fill: C.truth, stroke: C.paper, "stroke-width": 1.5 }));
  // rugs below the panel: one row per run
  const rugY = ff.bottom + 6;
  const rug = (qs: number[], y: number, color: string) => {
    const seen = new Map<number, number>();
    for (const q of qs) seen.set(q, (seen.get(q) ?? 0) + 1);
    return [...seen.entries()].map(([q, c]) => el("line", { x1: fx(r.x1[q]), x2: fx(r.x1[q]), y1: y, y2: y + Math.min(12, 4 + 2 * Math.log2(c + 1)), stroke: color, "stroke-width": 1.4 })).join("");
  };
  parts.push(rug(r.theory.queries, rugY, C.c7), rug(r.practical.queries, rugY + 16, C.acq));
  parts.push(text(ff.left - 4, rugY + 9, L.rugT, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted" }));
  parts.push(text(ff.left - 4, rugY + 25, narrow ? bl : tpl(L.rugP, { b: fixed(p.beta, 1).replace(/\.0$/, "") }), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted" }));
  if (narrow) parts.push(text(ff.left + 4, ff.top + 10, L.f, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));

  // Bottom panel: cumulative regret on a log scale.
  const bTop = rugY + 16 + 14 + (narrow ? 22 : 26);
  const bH = narrow ? 180 : 220;
  const all = [...r.bound, ...r.random, ...r.theory.cum, ...r.practical.cum];
  const ymaxRaw = Math.max(...all) * 1.3;
  const ymax = 10 ** Math.ceil(Math.log10(ymaxRaw));
  const ymin = 0.1;
  const bx = linear([0, T], [fl, st.w - 14]);
  const by = logScale([ymin, ymax], [bTop + bH, bTop]);
  const bf: Frame = { x: bx, y: by, left: fl, right: st.w - 14, top: bTop, bottom: bTop + bH, w: st.w };
  parts.push(
    axis({ scale: by, orient: "left", at: bf.left, span: [bf.left, bf.right], title: narrow ? undefined : L.regretShort, format: (v) => (v >= 1 ? String(Math.round(v)) : String(v)) }),
    axis({ scale: bx, orient: "bottom", at: bf.bottom, span: [bf.top, bf.bottom], title: L.t, count: narrow ? 4 : 6 }),
  );
  if (narrow) parts.push(text(bf.left + 4, bf.top + 10, L.regret, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));
  const cid = `${st.uid}-clip`;
  parts.push(el("defs", {}, el("clipPath", { id: cid }, el("rect", { x: bf.left, y: bf.top - 2, width: bf.right - bf.left, height: bf.bottom - bf.top + 4 }))));
  const line = (vs: number[], a: Record<string, string | number>) => el("path", {
    d: vs.map((v, i) => `${i ? "L" : "M"}${bx(i + 1).toFixed(1)},${by(Math.max(ymin, v)).toFixed(1)}`).join(""),
    fill: "none", "stroke-linejoin": "round", ...a,
  });
  parts.push(g({ "clip-path": `url(#${cid})` },
    line(r.random, { stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "5 4" }),
    line(r.bound, { stroke: C.c5, "stroke-width": 2, "stroke-dasharray": "6 4" }),
    line(r.theory.cum, { stroke: C.c7, "stroke-width": 2 }),
    line(r.practical.cum, { stroke: C.acq, "stroke-width": 2 }),
  ));
  // Readout.
  const ry = bf.bottom + 46;
  parts.push(text(bf.left, ry, tpl(L.consts, { c: fixed(r.c1, 2), b: fixed(Math.sqrt(r.betaT[T - 1]), 1), g: fixed(r.gamma[T - 1], 1) }), { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
  parts.push(text(bf.left, ry + 16, tpl(L.ratio, { T, B: fixed(r.bound[T - 1], 0), rr: fixed(r.random[T - 1], 0) }), { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
  parts.push(text(bf.left, ry + 32, tpl(narrow ? L.runsShort : L.runs, { r: fixed(r.theory.cum[T - 1], 1), q: fixed(r.practical.cum[T - 1], 1), b: bl }), { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
  const H = ry + 42;
  return svg(st.w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "regret-gpucb",
  title: { en: "GP-UCB on a function drawn from its own prior, against its regret bound", zh: "在从自身先验中抽取的函数上运行 GP-UCB，并与其遗憾界比较" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "New function", zh: "换一个函数" }, primary: true, run: (p) => ({ ...p, seed: (p.seed % 9999) + 1 }) },
  ],
});
