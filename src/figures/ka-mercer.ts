// Mercer's theorem computed. The kernel's integral operator on [0, 1] with the
// uniform measure is approximated by the kernel matrix on a grid of N
// midpoints divided by N (the Nyström method of Rasmussen and Williams 2006,
// sec. 4.3.2): its eigenvalues approximate the operator's eigenvalues λ_i and
// its eigenvectors, scaled by √N, the eigenfunctions φ_i. The figure plots
// the eigenvalues of four kernels on log axes, the first four eigenfunctions
// of the chosen kernel, and a draw from the Gaussian process built from its
// first m terms, f_m(x) = Σ_{i ≤ m} √λ_i z_i φ_i(x) (the Karhunen-Loève
// expansion), against the draw with all N terms, which is an exact draw of the
// process on the grid.

import { defineFigure, tr, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear, log } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { kernel, type KernelName } from "./lib/gp.ts";
import { symEig } from "./lib/linalg-eigen.ts";
import { memo, normals, rng } from "./lib/random.ts";
import { clip, curve, frame, frameAxes, legend } from "./lib/plot.ts";
import { fixed, sig, tpl } from "./lib/format.ts";

const labels = {
  en: {
    spectrum: "Eigenvalues λ_i (log axes)",
    index: "index i",
    eigfun: "First four eigenfunctions",
    x: "input x",
    sample: "A draw built from the first m terms",
    full: "all {n} terms (exact on the grid)",
    trunc: "first {m} term{m:/s}",
    floor: "below 10⁻¹⁴: rounding error",
    lam: "λ_1 = {a}, λ_2 = {b}, λ_3 = {c}; the λ_i sum to {s}",
    frac: "The first {m} term{m:/s} hold{m:s/} {f} of the prior variance; the draw's squared RKHS norm, Σ z_i², is {r}",
    frac1: "The first {m} term{m:/s} hold{m:s/} {f} of the prior variance;",
    frac2: "the draw's squared RKHS norm, Σ z_i², is {r}",
    describe: "{k} kernel with lengthscale {l} on [0, 1]: the largest eigenvalues are {a}, {b}, and {c}. The first {m} term{m:/s} of the Karhunen-Loève expansion hold{m:s/} {f} of the prior variance, and the truncated draw has squared RKHS norm {r}.",
  },
  zh: {
    spectrum: "特征值 λ_i（双对数坐标）",
    index: "下标 i",
    eigfun: "前四个特征函数",
    x: "输入 x",
    sample: "由前 m 项构造的样本",
    full: "全部 {n} 项（网格上的精确样本）",
    trunc: "前 {m} 项",
    floor: "低于 10⁻¹⁴：舍入误差",
    lam: "λ_1 = {a}，λ_2 = {b}，λ_3 = {c}；λ_i 之和为 {s}",
    frac: "前 {m} 项占先验方差的 {f}；样本的 RKHS 范数平方 Σ z_i² 为 {r}",
    frac1: "前 {m} 项占先验方差的 {f}；",
    frac2: "样本的 RKHS 范数平方 Σ z_i² 为 {r}",
    describe: "[0, 1] 上长度尺度为 {l} 的 {k} 核：最大的特征值为 {a}、{b}、{c}。Karhunen-Loève 展开的前 {m} 项占先验方差的 {f}，截断样本的 RKHS 范数平方为 {r}。",
  },
};

const KERNELS = [
  { value: "rbf", label: { en: "RBF", zh: "RBF" } },
  { value: "matern52", label: { en: "Matérn 5/2", zh: "Matérn 5/2" } },
  { value: "matern32", label: { en: "Matérn 3/2", zh: "Matérn 3/2" } },
  { value: "matern12", label: { en: "Matérn 1/2", zh: "Matérn 1/2" } },
] as const;
const COLOR: Record<string, string> = { rbf: C.c7, matern52: C.c6, matern32: C.c8, matern12: C.c5 };
const FUN_COLORS = [C.c1, C.c2, C.c3, C.c4];

const N = 100; // grid midpoints (i + 1/2)/N
const XS = Array.from({ length: N }, (_, i) => (i + 0.5) / N);
const SHOW = 40; // eigenvalues plotted
const FLOOR = 1e-14;

const params = {
  kernel: { kind: "choice", label: { en: "Kernel", zh: "核函数" }, options: KERNELS, default: "rbf" },
  lengthscale: { kind: "range", label: { en: "Lengthscale ℓ", zh: "长度尺度 ℓ" }, min: 0.03, max: 0.5, default: 0.1, scale: "log" },
  terms: { kind: "range", label: { en: "Terms m", zh: "项数 m" }, min: 1, max: N, default: 10, step: 1 },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 11, step: 1, control: false },
} as const;

type P = { kernel: string; lengthscale: number; terms: number; seed: number };

// Eigenvalues (operator scale, descending, clamped at zero) and eigenfunctions
// (grid values, unit L2 norm on [0, 1], sign fixed so the value at the left
// end is positive).
export const eigen = memo((kname: string, ls: number) => {
  const k = kernel(kname as KernelName, ls, 1);
  const K = XS.map((a) => XS.map((b) => k(a, b)));
  const { values, vectors } = symEig(K);
  const lam = values.map((v) => Math.max(v / N, 0));
  const phi = vectors.map((v) => {
    let first = 0;
    while (first < N - 1 && Math.abs(v[first]) < 1e-3) first++;
    const s = v[first] < 0 ? -1 : 1;
    return v.map((x) => s * x * Math.sqrt(N));
  });
  return { lam, phi };
}, 16);

const draws = memo((seed: number) => normals(rng(seed), N), 4);

export function compute(p: P) {
  const { lam, phi } = eigen(p.kernel, p.lengthscale);
  const m = Math.max(1, Math.min(N, Math.round(p.terms)));
  const z = draws(p.seed);
  const full = new Array<number>(N).fill(0), trunc = new Array<number>(N).fill(0);
  for (let i = 0; i < N; i++) {
    const a = Math.sqrt(lam[i]) * z[i];
    for (let j = 0; j < N; j++) {
      full[j] += a * phi[i][j];
      if (i < m) trunc[j] += a * phi[i][j];
    }
  }
  const total = lam.reduce((s, v) => s + v, 0);
  const head = lam.slice(0, m).reduce((s, v) => s + v, 0);
  const norm2 = z.slice(0, m).reduce((s, v) => s + v * v, 0);
  return { lam, phi, m, full, trunc, total, frac: head / total, norm2 };
}

const SUP: Record<string, string> = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
const pow10 = (v: number) => (v === 1 ? "1" : `10${String(Math.round(Math.log10(v))).split("").map((ch) => SUP[ch] ?? ch).join("")}`);

const pctStr = (f: number) => (f > 0.9995 && f < 1 ? ">99.9%" : f >= 0.99 ? `${(f * 100).toFixed(1)}%` : `${(f * 100).toFixed(0)}%`);

function describe(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const c = compute(p);
  return tpl(L.describe, {
    k: tr(KERNELS.find((k) => k.value === p.kernel)!.label, lang), l: sig(p.lengthscale, 2),
    a: fixed(c.lam[0], 3), b: fixed(c.lam[1], 3), c: fixed(c.lam[2], 3), m: c.m, f: pctStr(c.frac), r: fixed(c.norm2, 1),
  });
}

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 480;
  const c = compute(p);
  const parts: string[] = [];
  const pad = 8;
  // ---- headers: each top panel has a title and a legend row; the plots
  // start below the taller of the two headers, so they line up when side by side
  const specW = narrow ? st.w : Math.floor(st.w * 0.48);
  const fx0 = narrow ? 0 : specW + 4;
  const fw = narrow ? st.w : st.w - specW - 4;
  const kLegend = legend(4, 30, specW - 8, KERNELS.map((k) => ({ kind: "line" as const, color: COLOR[k.value], label: tr(k.label, lang) + (k.value === p.kernel ? " ●" : "") })));
  const fLegend = legend(fx0 + 4, 30, fw - 8, [0, 1, 2, 3].map((i) => ({ kind: "line" as const, color: FUN_COLORS[i], label: `φ_${i + 1}` })));
  const head = narrow ? 30 + kLegend.height : 30 + Math.max(kLegend.height, fLegend.height);

  // ---- spectrum panel (log-log), all four kernels, the chosen one emphasized
  const top1 = head;
  const h1 = narrow ? 150 : 170;
  const sl = narrow ? 44 : 50, sr = specW - 10;
  const xi = log([1, SHOW], [sl, sr]);
  const yl = log([FLOOR, 1], [top1 + h1, top1]);
  parts.push(text(4, 14, L.spectrum, { "font-size": TYPE.small, class: "fig-t-strong" }), kLegend.svg);
  parts.push(
    axis({ scale: yl, orient: "left", at: sl, span: [sl, sr], ticks: [1e-14, 1e-10, 1e-6, 1e-2], format: pow10 }),
    axis({ scale: xi, orient: "bottom", at: top1 + h1, span: [top1, top1 + h1], ticks: [1, 2, 5, 10, 20, 40], title: L.index, format: (v) => String(v) }),
  );
  const order = KERNELS.map((k) => k.value as string).filter((k) => k !== p.kernel).concat([p.kernel]);
  for (const kn of order) {
    const { lam } = eigen(kn, p.lengthscale);
    const on = kn === p.kernel;
    const pts: Array<[number, number]> = [];
    for (let i = 0; i < SHOW; i++) if (lam[i] >= FLOOR) pts.push([xi(i + 1), yl(lam[i])]);
    parts.push(el("path", { d: linePath(pts), fill: "none", stroke: COLOR[kn], "stroke-width": on ? 2.4 : 1.2, opacity: on ? 1 : 0.55 }));
    if (on) for (const [x, y] of pts) parts.push(el("circle", { cx: x, cy: y, r: 2.4, fill: COLOR[kn] }));
  }
  parts.push(text(sl + 4, top1 + h1 - 5, L.floor, { "font-size": TYPE.small, class: "fig-t-faint fig-t-halo" }));

  // ---- eigenfunction panel
  const top2 = narrow ? top1 + h1 + 52 + 30 + fLegend.height : top1;
  const h2 = narrow ? 130 : 170;
  let ymax = 0;
  for (let i = 0; i < 4; i++) for (const v of c.phi[i]) ymax = Math.max(ymax, Math.abs(v));
  ymax = Math.max(1.6, Math.ceil(ymax * 2.2) / 2);
  const ff = frame({ w: fw, top: top2, height: h2, yDomain: [-ymax, ymax] });
  const shift = (s: string) => g({ transform: `translate(${fx0},0)` }, s);
  const fTitleY = narrow ? top2 - 16 - fLegend.height : 14;
  parts.push(text(fx0 + 4, fTitleY, L.eigfun, { "font-size": TYPE.small, class: "fig-t-strong" }));
  parts.push(narrow ? legend(4, fTitleY + 16, fw - 8, [0, 1, 2, 3].map((i) => ({ kind: "line" as const, color: FUN_COLORS[i], label: `φ_${i + 1}` }))).svg : fLegend.svg);
  const funs = [0, 1, 2, 3].map((i) => curve(ff, XS, c.phi[i], { stroke: FUN_COLORS[i], "stroke-width": i === 0 ? 2.2 : 1.6 })).join("");
  const fid = `${st.uid}-fclip`;
  parts.push(shift(el("defs", {}, clip(ff, fid)) + frameAxes(ff, { xTitle: L.x, yCount: 3 }) + g({ "clip-path": `url(#${fid})` }, funs)));

  // ---- sample panel
  const top3 = (narrow ? top2 + h2 : top1 + Math.max(h1, h2)) + 78;
  const h3 = narrow ? 140 : 150;
  let smax = 0;
  for (const v of c.full) smax = Math.max(smax, Math.abs(v));
  for (const v of c.trunc) smax = Math.max(smax, Math.abs(v));
  smax = Math.max(2.5, Math.ceil(smax * 1.15 * 2) / 2);
  const fs = frame({ w: st.w, top: top3, height: h3, yDomain: [-smax, smax], yTitle: "f(x)" });
  const sid = `${st.uid}-sclip`;
  parts.push(text(4, top3 - 22, L.sample, { "font-size": TYPE.small, class: "fig-t-strong" }));
  // legend line
  const lg1 = tpl(L.trunc, { m: c.m }), lg2 = tpl(L.full, { n: N });
  parts.push(
    el("line", { x1: 4, x2: 20, y1: top3 - 10, y2: top3 - 10, stroke: COLOR[p.kernel], "stroke-width": 2.2 }),
    text(24, top3 - 6, lg1, { "font-size": TYPE.small, class: "fig-t-muted" }),
    el("line", { x1: 24 + lg1.length * 6.3 + 14, x2: 24 + lg1.length * 6.3 + 30, y1: top3 - 10, y2: top3 - 10, stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "4 3" }),
    text(24 + lg1.length * 6.3 + 34, top3 - 6, lg2, { "font-size": TYPE.small, class: "fig-t-muted" }),
  );
  parts.push(
    el("defs", {}, clip(fs, sid)),
    frameAxes(fs, { xTitle: L.x, yTitle: "f(x)", yCount: 4 }),
    g({ "clip-path": `url(#${sid})` },
      curve(fs, XS, c.full, { stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "4 3" }),
      curve(fs, XS, c.trunc, { stroke: COLOR[p.kernel], "stroke-width": 2.2 }),
    ),
  );

  // ---- readouts
  let ry = top3 + h3 + 50;
  parts.push(text(pad / 2, ry, tpl(L.lam, { a: fixed(c.lam[0], 3), b: fixed(c.lam[1], 3), c: fixed(c.lam[2], 3), s: fixed(c.total, 2) }), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  const fr = { m: c.m, f: pctStr(c.frac), r: fixed(c.norm2, 1) };
  if (narrow) {
    ry += 16;
    parts.push(text(pad / 2, ry, tpl(L.frac1, fr), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
    ry += 16;
    parts.push(text(pad / 2, ry, tpl(L.frac2, fr), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  } else {
    ry += 16;
    parts.push(text(pad / 2, ry, tpl(L.frac, fr), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  }
  return svg(st.w, ry + 10, describe(st), ...parts);
}

export default defineFigure({
  name: "ka-mercer",
  title: { en: "Eigenvalues, eigenfunctions, and the Karhunen-Loève expansion of a kernel on [0, 1]", zh: "[0, 1] 上核函数的特征值、特征函数与 Karhunen-Loève 展开" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "Draw again", zh: "重新抽取" }, primary: true, run: (p) => ({ ...p, seed: (p.seed % 9999) + 1 }) },
  ],
  snapshots: {
    "Matérn 1/2": (p) => ({ ...p, kernel: "matern12" }),
    "one term": (p) => ({ ...p, terms: 1 }),
    "all terms": (p) => ({ ...p, terms: 100 }),
  },
});
