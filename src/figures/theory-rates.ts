// The kernelized regret bounds for preference feedback, side by side. The
// rates are stated in terms of gamma_T, the maximum information gain of the
// kernel, so how they compare depends on how fast gamma_T grows. The top
// panel computes gamma_T for a one-dimensional domain (greedy estimate on a
// grid of 300 inputs, regularization 0.25) and draws the shapes of the bounds
// with every constant, confidence multiplier, and link factor set to 1:
// gamma_T sqrt(T) (MaxMinLCB, PF-TS), sqrt(gamma_T T) (MR-LPF), and, for the
// squared exponential kernel, T^(3/4) (POP-BO), against T itself, the growth
// of a learner that never improves. The bottom panel gives the asymptotic
// exponents of T as the dimension grows, from the published orders of gamma_T
// for Matérn kernels (logarithmic factors ignored). Only the bottom panel is a
// statement about the rates; the top panel's levels are illustrative.

import { defineFigure, tr, type State } from "./types.ts";
import { el, g, labelWidth, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear, log } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { kernel, type KernelName } from "./lib/gp.ts";
import { memo } from "./lib/random.ts";
import { fixed, sig, tpl } from "./lib/format.ts";

const labels = {
  en: {
    top: "Bound shapes in one dimension, constants set to 1 (illustrative)",
    bottom: "Exponent a in T^a as the dimension d grows (asymptotic, logs ignored)",
    T: "rounds T",
    d: "dimension d",
    lin: "T: no learning",
    sqrtT: "√T",
    gsq: "γ_T √T: MaxMinLCB, PF-TS",
    sqg: "√(γ_T T): MR-LPF",
    pop: "T^{3/4}: POP-BO",
    gap: "gap ×√γ_T = {g}",
    gam: "greedy γ_T at T = {T}: {g}",
    above: "above 1: grows faster than T",
    bottomShort: "Exponent a in T^a vs dimension d (asymptotic)",
    topShort: "Bound shapes in 1-D, constants set to 1 (illustrative)",
    lbNote: "For Matérn kernels the exponent of √(γ_T T) equals that of the scalar lower bound.",
    rbfNote: "Squared exponential: γ_T grows like a power of log T, so both γ_T √T and √(γ_T T) have exponent 1/2; they differ by polylog factors. POP-BO: 3/4.",
    popYes: "POP-BO's Matérn result needs a smoothness above {t} at d = 1, more in higher dimensions; this kernel (smoothness {nu}) meets it only up to d = {dmax}.",
    popNo: "POP-BO's Matérn result needs a smoothness above {t} even at d = 1, so it does not cover this kernel (smoothness {nu}) in any dimension.",
    describe: "With the {k} kernel and lengthscale {l}, the greedy estimate of the maximum information gain after {T} rounds is {g}. With constants set to 1, γ_T √T is {a} and √(γ_T T) is {b}, against T = {T}; the two differ by the factor √γ_T = {r}.",
  },
  zh: {
    top: "一维上的界的形状，常数均取 1（示意）",
    bottom: "T^a 的指数 a 随维度 d 的变化（渐近，忽略对数因子）",
    T: "轮数 T",
    d: "维度 d",
    lin: "T：不学习",
    sqrtT: "√T",
    gsq: "γ_T √T：MaxMinLCB、PF-TS",
    sqg: "√(γ_T T)：MR-LPF",
    pop: "T^{3/4}：POP-BO",
    gap: "差距 ×√γ_T = {g}",
    gam: "T = {T} 时的贪心 γ_T：{g}",
    above: "高于 1：增长快于 T",
    bottomShort: "T^a 的指数 a 与维度 d（渐近）",
    topShort: "一维上的界的形状，常数取 1（示意）",
    lbNote: "对 Matérn 核，√(γ_T T) 的指数等于标量下界的指数。",
    rbfNote: "平方指数核：γ_T 按 log T 的幂增长，所以 γ_T √T 与 √(γ_T T) 的指数都是 1/2，二者只差多对数因子。POP-BO：3/4。",
    popYes: "POP-BO 的 Matérn 结果在 d = 1 时要求光滑度高于 {t}，维度越高要求越高；此核（光滑度 {nu}）只在 d 不超过 {dmax} 时满足。",
    popNo: "POP-BO 的 Matérn 结果即使在 d = 1 时也要求光滑度高于 {t}，因此在任何维度上都不涵盖此核（光滑度 {nu}）。",
    describe: "核函数：{k}；长度尺度：{l}。{T} 轮后，最大信息增益的贪心估计为 {g}。常数均取 1 时，γ_T √T 为 {a}，√(γ_T T) 为 {b}，而 T = {T}；两者相差因子 √γ_T = {r}。",
  },
};

// Wrap Chinese text, which has no spaces between words, to lines of at most
// `max` Latin-letter widths: a Chinese character counts 1.8, a run of Latin
// letters, digits, and symbols is kept whole, a line never starts with
// closing punctuation, and never ends with an opening bracket.
function wrapZh(s: string, max: number): string[] {
  const toks = s.match(/[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]|[^\s\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]+|\s+/g) ?? [];
  const width = (t: string) => labelWidth(t, 1, 1.8);
  const lines: string[] = [];
  let line = "", w = 0;
  for (const t of toks) {
    const tw = width(t);
    if (line && w + tw > max && !/^[，。；：、）]$/.test(t) && !/^\s+$/.test(t)) {
      // an opening bracket moves to the new line with what it opens
      const carry = /[（“]$/.test(line) ? line.slice(-1) : "";
      lines.push(line.slice(0, line.length - carry.length).trimEnd());
      line = carry; w = carry ? width(carry) : 0;
    }
    if (!line && /^\s+$/.test(t)) continue;
    line += t; w += tw;
  }
  if (line.trim()) lines.push(line.trimEnd());
  return lines;
}

const KERNELS = [
  { value: "rbf", label: { en: "Sq. exponential", zh: "平方指数" } },
  { value: "matern52", label: { en: "Matérn 5/2", zh: "Matérn 5/2" } },
  { value: "matern32", label: { en: "Matérn 3/2", zh: "Matérn 3/2" } },
  { value: "matern12", label: { en: "Matérn 1/2", zh: "Matérn 1/2" } },
] as const;
const NU: Record<string, number> = { matern52: 2.5, matern32: 1.5, matern12: 0.5 };

const params = {
  kernel: { kind: "choice", label: { en: "Kernel", zh: "核函数" }, options: KERNELS, default: "matern52" },
  horizon: { kind: "choice", label: { en: "Horizon T", zh: "时域 T" }, options: [{ value: 100, label: { en: "100", zh: "100" } }, { value: 300, label: { en: "300", zh: "300" } }, { value: 1000, label: { en: "1000", zh: "1000" } }], default: 1000 },
  lengthscale: { kind: "range", label: { en: "Lengthscale ℓ", zh: "长度尺度 ℓ" }, min: 0.05, max: 0.5, default: 0.1, scale: "log" },
} as const;

type P = { kernel: string; horizon: number; lengthscale: number };

const N = 300;
const LAMBDA = 0.25; // regularization: the largest variance a binary answer can have
const TMAX = 1000;

// Greedy information gain on a grid: repeatedly observe the input of largest
// posterior variance. It reaches at least (1 - 1/e) of the maximum.
const gammas = memo((kname: string, ls: number): number[] => {
  const k = kernel(kname as KernelName, ls, 1);
  const xs = Array.from({ length: N }, (_, i) => i / (N - 1));
  const S = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) S[i * N + j] = k(xs[i], xs[j]);
  const out: number[] = [];
  const col = new Float64Array(N);
  let acc = 0;
  for (let t = 0; t < TMAX; t++) {
    let b = 0, bv = -1;
    for (let i = 0; i < N; i++) { const v = S[i * N + i]; if (v > bv) { bv = v; b = i; } }
    acc += 0.5 * Math.log(1 + Math.max(bv, 0) / LAMBDA);
    out.push(acc);
    const den = bv + LAMBDA;
    for (let i = 0; i < N; i++) col[i] = S[i * N + b];
    for (let i = 0; i < N; i++) {
      const ci = col[i] / den;
      if (ci === 0) continue;
      const row = i * N;
      for (let j = 0; j < N; j++) S[row + j] -= ci * col[j];
    }
  }
  return out;
}, 16);

// POP-BO's Matérn condition: nu > (d/4)(3 + d + sqrt(d^2 + 14 d + 17)).
const popThreshold = (d: number) => (d / 4) * (3 + d + Math.sqrt(d * d + 14 * d + 17));

function describe(st: State<P>): string {
  const p = st.p;
  const gm = gammas(p.kernel, p.lengthscale);
  const T = p.horizon;
  const gT = gm[T - 1];
  const lang = st.lang ?? "en";
  return tpl(labels[lang].describe, {
    k: tr(KERNELS.find((k) => k.value === p.kernel)!.label, lang), l: sig(p.lengthscale, 2), T, g: fixed(gT, 1),
    a: fixed(gT * Math.sqrt(T), 0), b: fixed(Math.sqrt(gT * T), 0), r: fixed(Math.sqrt(gT), 1),
  });
}

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 480;
  const T = p.horizon;
  const gm = gammas(p.kernel, p.lengthscale);
  const isRbf = p.kernel === "rbf";
  const parts: string[] = [];

  // ---- top panel: bound shapes over T, log-log
  const left = narrow ? 44 : 56;
  const labW = narrow ? 0 : 168;
  const right = st.w - 12 - labW;
  let y0 = 16;
  parts.push(text(4, y0, narrow ? L.topShort : L.top, { "font-size": TYPE.small, class: "fig-t-strong" }));
  const ts = Array.from({ length: T }, (_, i) => i + 1);
  const curves = [
    { key: "lin", color: C.ink3, dash: "5 4", ys: ts.map((t) => t), label: L.lin },
    ...(isRbf ? [{ key: "pop", color: C.c7, dash: "", ys: ts.map((t) => t ** 0.75), label: L.pop }] : []),
    { key: "gsq", color: C.model, dash: "", ys: ts.map((t) => gm[t - 1] * Math.sqrt(t)), label: L.gsq },
    { key: "sqg", color: C.acq, dash: "", ys: ts.map((t) => Math.sqrt(gm[t - 1] * t)), label: L.sqg },
  ];
  const ymax = Math.max(...curves.map((c) => c.ys[T - 1])) * 1.3;
  const top = y0 + (narrow ? 14 : 12);
  const h = narrow ? 170 : 200;
  const xs = log([1, T], [left, right]);
  const ysc = log([0.3, Math.max(10, ymax)], [top + h, top]);
  const yTicks = [1, 10, 100, 1000, 10000, 100000].filter((v) => v <= ysc.domain[1]);
  const xTicks = [1, 10, 100, 1000].filter((v) => v <= T);
  parts.push(
    axis({ scale: ysc, orient: "left", at: left, span: [left, right], ticks: yTicks, format: (v) => (v >= 1000 ? `${v / 1000}k` : String(v)) }),
    axis({ scale: xs, orient: "bottom", at: top + h, span: [top, top + h], ticks: xTicks, title: L.T, format: (v) => String(v) }),
  );
  const cid = `${st.uid}-clip`;
  parts.push(el("defs", {}, el("clipPath", { id: cid }, el("rect", { x: left, y: top - 2, width: right - left, height: h + 4 }))));
  const curveG: string[] = [];
  for (const c of curves) {
    const pts: Array<[number, number]> = ts.filter((t) => t === 1 || t === T || t % Math.max(1, Math.floor(T / 200)) === 0).map((t) => [xs(t), ysc(c.ys[t - 1])]);
    curveG.push(el("path", { d: linePath(pts), fill: "none", stroke: c.color, "stroke-width": c.key === "lin" ? 1.4 : 2.2, "stroke-dasharray": c.dash || undefined }));
  }
  parts.push(g({ "clip-path": `url(#${cid})` }, ...curveG));
  // gap bracket at T between the two gamma curves
  const ya = ysc(gm[T - 1] * Math.sqrt(T)), yb = ysc(Math.sqrt(gm[T - 1] * T));
  const bx = right - 4;
  parts.push(el("path", { d: `M${bx - 5},${ya}H${bx}V${yb}H${bx - 5}`, fill: "none", stroke: C.ink2, "stroke-width": 1.2 }));
  // labels: at the right end on wide screens, as a legend below on narrow ones
  if (!narrow) {
    const ends = curves.map((c) => ({ c, y: ysc(c.ys[T - 1]) })).sort((a, b) => a.y - b.y);
    let last = -Infinity;
    for (const e of ends) {
      const yy = Math.max(e.y + 4, last + 14);
      last = yy;
      parts.push(text(right + 8, yy, e.c.label, { "font-size": TYPE.small, class: "fig-t-strong", fill: e.c.color }));
    }
    parts.push(text(right + 8, Math.max(last + 22, (ya + yb) / 2 + 4), tpl(L.gap, { g: fixed(Math.sqrt(gm[T - 1]), 1) }), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  }
  let yy = top + h + (narrow ? 52 : 40);
  if (narrow) {
    let lx = 4;
    for (const c of curves) {
      const wv = labelWidth(c.label, 6) + 30;
      if (lx + wv > st.w - 4 && lx > 4) { lx = 4; yy += 17; }
      parts.push(el("line", { x1: lx, x2: lx + 16, y1: yy - 4, y2: yy - 4, stroke: c.color, "stroke-width": 2, "stroke-dasharray": c.dash || undefined }), text(lx + 21, yy, c.label, { "font-size": TYPE.small, class: "fig-t-muted" }));
      lx += wv;
    }
    yy += 17;
    parts.push(text(4, yy, tpl(L.gap, { g: fixed(Math.sqrt(gm[T - 1]), 1) }), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  }
  yy += 17;
  parts.push(text(4, yy, tpl(L.gam, { T, g: fixed(gm[T - 1], 1) }), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));

  // ---- bottom panel: exponents over dimension
  y0 = yy + 30;
  parts.push(text(4, y0, narrow ? L.bottomShort : L.bottom, { "font-size": TYPE.small, class: "fig-t-strong" }));
  const top2 = y0 + 14;
  const h2 = narrow ? 140 : 150;
  const right2 = st.w - 12 - labW;
  const xd = linear([1, 10], [left, right2]);
  const ye = linear([0.4, 1.5], [top2 + h2, top2]);
  parts.push(
    el("rect", { x: left, y: ye(1.5), width: right2 - left, height: ye(1) - ye(1.5), fill: C.panel }),
    axis({ scale: ye, orient: "left", at: left, span: [left, right2], ticks: [0.5, 0.75, 1, 1.25, 1.5], format: (v) => String(v) }),
    axis({ scale: xd, orient: "bottom", at: top2 + h2, span: [top2, top2 + h2], ticks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], title: L.d, format: (v) => String(v) }),
    el("line", { x1: left, x2: right2, y1: ye(1), y2: ye(1), stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "5 4" }),
    text(left + 6, ye(1.5) + 13, L.above, { "font-size": TYPE.small, class: "fig-t-muted" }),
  );
  const ds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  const series: Array<{ color: string; ys: number[]; label: string; dash?: string }> = [];
  if (isRbf) {
    series.push({ color: C.model, ys: ds.map(() => 0.5), label: L.gsq });
    series.push({ color: C.acq, ys: ds.map(() => 0.5), label: L.sqg, dash: "6 4" });
    series.push({ color: C.c7, ys: ds.map(() => 0.75), label: L.pop });
  } else {
    const nu = NU[p.kernel];
    series.push({ color: C.model, ys: ds.map((d) => 0.5 + d / (2 * nu + d)), label: L.gsq });
    series.push({ color: C.acq, ys: ds.map((d) => (nu + d) / (2 * nu + d)), label: L.sqg });
  }
  for (const s of series) {
    parts.push(el("path", { d: linePath(ds.map((d, i) => [xd(d), ye(s.ys[i])])), fill: "none", stroke: s.color, "stroke-width": 2.2, "stroke-dasharray": s.dash }));
    for (let i = 0; i < ds.length; i++) parts.push(el("circle", { cx: xd(ds[i]), cy: ye(s.ys[i]), r: 2.6, fill: s.color }));
  }
  // marker at d = 1, the top panel's dimension
  parts.push(el("line", { x1: xd(1), x2: xd(1), y1: top2, y2: top2 + h2, stroke: C.ink3, "stroke-dasharray": "2 3" }));
  if (!narrow) {
    const ends = series.map((s) => ({ s, y: ye(s.ys[ds.length - 1]) })).sort((a, b) => a.y - b.y);
    let last = -Infinity;
    for (const e of ends) {
      const y1 = Math.max(e.y + 4, last + 14);
      last = y1;
      parts.push(text(right2 + 8, y1, e.s.label, { "font-size": TYPE.small, class: "fig-t-strong", fill: e.s.color }));
    }
  }
  let yb2 = top2 + h2 + (narrow ? 52 : 40);
  if (narrow) {
    let lx = 4;
    for (const s of series) {
      const wv = labelWidth(s.label, 6) + 30;
      if (lx + wv > st.w - 4 && lx > 4) { lx = 4; yb2 += 17; }
      parts.push(el("line", { x1: lx, x2: lx + 16, y1: yb2 - 4, y2: yb2 - 4, stroke: s.color, "stroke-width": 2, "stroke-dasharray": s.dash }), text(lx + 21, yb2, s.label, { "font-size": TYPE.small, class: "fig-t-muted" }));
      lx += wv;
    }
    yb2 += 6;
  }
  // POP-BO note
  let note: string;
  if (isRbf) note = L.rbfNote;
  else {
    const nu = NU[p.kernel];
    const dmax = ds.filter((d) => nu > popThreshold(d)).length;
    note = `${L.lbNote}${lang === "zh" ? "" : " "}${tpl(dmax ? L.popYes : L.popNo, { t: fixed(popThreshold(1), 2), nu: fixed(nu, 1), dmax })}`;
  }
  const maxChars = Math.floor((st.w - 8) / 6.1);
  let lines: string[] = [];
  if (lang === "zh") lines = wrapZh(note, maxChars);
  else {
    const words = note.split(" ");
    let line = "";
    for (const w of words) { if (line && (line + " " + w).length > maxChars) { lines.push(line); line = w; } else line = line ? `${line} ${w}` : w; }
    if (line) lines.push(line);
  }
  for (const ln of lines) { yb2 += 16; parts.push(text(4, yb2, ln, { "font-size": TYPE.small, class: "fig-t-muted" })); }

  return svg(st.w, yb2 + 10, describe(st), ...parts);
}

export default defineFigure({
  name: "theory-rates",
  title: { en: "Kernelized regret bounds for preference feedback, compared", zh: "偏好反馈下核化遗憾界的比较" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
