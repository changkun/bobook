// Why fixed lengthscale priors break in high dimension. The reader picks the
// input dimension D. The top panel shows the exact distribution of the
// distance between two independent uniform points in the unit cube [0, 1]^D,
// with the lengthscale that two default priors start from: a fixed-scale
// default (BoTorch PairwiseGP, optuna-dashboard, or the upper bound used by
// local PBO) and BoTorch's dimension-scaled log-normal prior. The bottom panel
// shows, on a log scale, the kernel value k(d) each default assigns at
// distance d, with the middle 90% of pair distances shaded. A fixed
// lengthscale leaves typical pairs many lengthscales apart, so their kernel
// values collapse toward zero; the dimension-scaled lengthscale grows with
// the typical distance and keeps typical kernel values near exp(-2).
//
// The distance law is computed, not simulated: the squared distance is a sum
// of D independent copies of V = (x - y)^2, whose density on [0, 1] is
// v^(-1/2) - 1. V is discretized on a fine grid by splitting each bit of
// probability linearly between its two neighbouring grid points (which keeps
// the mean exact), and the D-fold convolution is taken with one FFT. The mean
// D/6 and variance 7D/180 of the squared distance are reproduced to four
// decimals for every D shown.

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { legend } from "./lib/plot.ts";
import { memo } from "./lib/random.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    dist: "distance r between two random points in the d-dimensional unit cube",
    distShort: "distance r between two random points",
    kern: "kernel value k(r), log scale",
    x: "distance r",
    mid: "middle 90% of pairs",
    median: "median {m}",
    eps: "float64 rounding unit 2⁻⁵³",
    epsShort: "2⁻⁵³",
    scaled: "dimension-scaled prior",
    typical: "{name}, ℓ = {l}: a typical pair is {r} lengthscales apart, k = {k}",
    typicalA: "{name}, ℓ = {l}",
    typicalB: "typical pair {r} lengthscales apart, k = {k}",
    describe: "In d = {D} dimensions the median distance between two random points of the unit cube is {m}, and 90% of pairs lie between {lo} and {hi}. With the {name} (lengthscale {lf}) the kernel value at the median distance is {kf}; with the dimension-scaled prior (lengthscale {ls}) it is {ks}.",
    pairwisegp: "PairwiseGP default",
    optuna: "optuna-dashboard default",
    local: "local PBO upper bound",
  },
  zh: {
    dist: "d 维单位立方体中两个随机点之间的距离 r",
    distShort: "两个随机点之间的距离 r",
    kern: "核函数值 k(r)，对数刻度",
    x: "距离 r",
    mid: "中间 90% 的点对",
    median: "中位数 {m}",
    eps: "双精度舍入单位 2⁻⁵³",
    epsShort: "2⁻⁵³",
    scaled: "维度缩放先验",
    typical: "{name}，ℓ = {l}：典型点对相距 {r} 个长度尺度，k = {k}",
    typicalA: "{name}，ℓ = {l}",
    typicalB: "典型点对相距 {r} 个长度尺度，k = {k}",
    describe: "在 d = {D} 维中，单位立方体内两个随机点之间距离的中位数为 {m}，90% 的点对距离在 {lo} 至 {hi} 之间。{name}（长度尺度 {lf}）在中位距离处的核函数值为 {kf}；维度缩放先验（长度尺度 {ls}）为 {ks}。",
    pairwisegp: "PairwiseGP 默认值",
    optuna: "optuna-dashboard 默认值",
    local: "局部 PBO 上界",
  },
};

const FIXED = [
  { value: "pairwisegp", label: { en: "PairwiseGP, ℓ 0.52", zh: "PairwiseGP，ℓ 0.52" } },
  { value: "optuna", label: { en: "optuna-dashboard, ℓ 0.4", zh: "optuna-dashboard，ℓ 0.4" } },
  { value: "local", label: { en: "local PBO, ℓ ≤ 0.5", zh: "局部 PBO，ℓ ≤ 0.5" } },
] as const;

// The name of the fixed-scale default, in the edition's language.
const nameOf = (lang: "en" | "zh", f: string) => labels[lang][f as "pairwisegp" | "optuna" | "local"];

const params = {
  dim: {
    kind: "range", label: { en: "Dimension d", zh: "维度 d" }, min: 1, max: 200, default: 50, step: 1,
    marks: [2, 10, 20, 50, 100, 200].map((v) => ({ value: v, label: { en: String(v), zh: String(v) } })),
  },
  fixed: { kind: "choice", label: { en: "Fixed prior", zh: "固定尺度先验" }, options: FIXED, default: "pairwisegp", control: "select" },
} as const;

type P = { dim: number; fixed: string };

// --- the exact distance law ------------------------------------------------

const L = 1 << 15;

function fft(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      let t = re[i]; re[i] = re[j]; re[j] = t;
      t = im[i]; im[i] = im[j]; im[j] = t;
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = ((2 * Math.PI) / len) * (inverse ? 1 : -1);
    const wr = Math.cos(ang), wi = Math.sin(ang);
    const half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k, b = a + half;
        const br = re[b] * cr - im[b] * ci, bi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - br; im[b] = im[a] - bi;
        re[a] += br; im[a] += bi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
  if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
}

interface Law { cum: Float64Array; h: number } // cumulative mass of the squared distance on the grid j*h

// The squared distance S = sum of D copies of V. V has CDF 2 sqrt(v) - v on [0, 1].
const law = memo((D: number): Law => {
  const M = Math.min(4096, Math.floor((L - 1) / D));
  const h = 1 / M;
  const F = (v: number) => 2 * Math.sqrt(v) - v; // P(V <= v)
  const G = (v: number) => (2 / 3) * v ** 1.5 - (v * v) / 2; // integral of v f(v)
  const re = new Float64Array(L), im = new Float64Array(L);
  for (let j = 0; j < M; j++) {
    const a = j * h, b = (j + 1) * h;
    const m0 = F(b) - F(a), m1 = G(b) - G(a);
    re[j] += (b * m0 - m1) / h; // share of the cell's mass assigned to its left end
    re[j + 1] += (m1 - a * m0) / h; // and to its right end
  }
  fft(re, im, false);
  for (let i = 0; i < L; i++) {
    const r = Math.hypot(re[i], im[i]) ** D, th = Math.atan2(im[i], re[i]) * D;
    re[i] = r * Math.cos(th); im[i] = r * Math.sin(th);
  }
  fft(re, im, true);
  const n = D * M + 1;
  const cum = new Float64Array(n);
  let c = 0;
  for (let j = 0; j < n; j++) { c += Math.max(0, re[j]); cum[j] = c; }
  for (let j = 0; j < n; j++) cum[j] /= c;
  return { cum, h };
}, 24);

// P(S <= s), with each grid mass spread uniformly over its cell [(j - 1/2)h, (j + 1/2)h].
function cdfS(lw: Law, s: number): number {
  const u = s / lw.h + 0.5;
  if (u <= 0) return 0;
  const j = Math.floor(u);
  if (j >= lw.cum.length) return 1;
  const prev = j > 0 ? lw.cum[j - 1] : 0;
  return prev + (lw.cum[j] - prev) * (u - j);
}

// The distance d with P(dist <= d) = p.
function quantileD(lw: Law, p: number): number {
  let lo = 0, hi = lw.cum.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (lw.cum[m] < p) lo = m; else hi = m; }
  const prev = hi > 0 ? lw.cum[hi - 1] : 0;
  const frac = (p - prev) / Math.max(1e-300, lw.cum[hi] - prev);
  return Math.sqrt(Math.max(0, (hi - 0.5 + Math.min(1, Math.max(0, frac))) * lw.h));
}

// --- the defaults ----------------------------------------------------------

// Mode of BoTorch's LogNormal(sqrt(2) + log(D)/2, sqrt(3)) prior: exp(mu - sigma^2).
const scaledMode = (D: number) => Math.exp(Math.SQRT2 + 0.5 * Math.log(D) - 3);
const fixedLs = (f: string) => (f === "optuna" ? 4 / 10 : f === "local" ? 0.5 : 1.4 / 2.7);

// log10 of the kernel value at distance d. RBF in the GPyTorch convention
// exp(-d^2 / (2 l^2)); Matérn 3/2 is (1 + sqrt(3) r) exp(-sqrt(3) r), r = d / l.
function log10k(kind: "rbf" | "matern32", l: number, d: number): number {
  const r = d / l;
  if (kind === "rbf") return (-0.5 * r * r) / Math.LN10;
  return (Math.log(1 + Math.sqrt(3) * r) - Math.sqrt(3) * r) / Math.LN10;
}

const SUP: Record<string, string> = { "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹", "-": "⁻" };
const sup = (n: number) => String(n).split("").map((ch) => SUP[ch] ?? ch).join("");

// A kernel value from its log10: "0.14", or "2.0×10⁻⁷" when small.
function fmtK(lg: number): string {
  if (lg > -2) return String(Number((10 ** lg).toPrecision(2)));
  const e = Math.floor(lg);
  let m = 10 ** (lg - e);
  let ee = e;
  if (Number(m.toFixed(1)) >= 10) { m /= 10; ee += 1; }
  return `${m.toFixed(1)}×10${sup(ee)}`;
}

const XMAX = 6.5;
// The shaded middle 90% of distances: a translucent muted ink, legible in both themes.
const BANDC = "color-mix(in srgb, var(--fig-ink-3) 22%, transparent)";
const KMIN = -30;

function compute(p: P) {
  const D = Math.round(p.dim);
  const lw = law(D);
  const med = quantileD(lw, 0.5), lo = quantileD(lw, 0.05), hi = quantileD(lw, 0.95);
  const lf = fixedLs(p.fixed), ls = scaledMode(D);
  const kindF = p.fixed === "optuna" ? "matern32" as const : "rbf" as const;
  return { D, lw, med, lo, hi, lf, ls, kindF, kf: log10k(kindF, lf, med), ks: log10k("rbf", ls, med) };
}

function describe(st: State<P>): string {
  const c = compute(st.p);
  const lang = st.lang ?? "en";
  return tpl(labels[lang].describe, {
    D: c.D, m: fixed(c.med, 2), lo: fixed(c.lo, 2), hi: fixed(c.hi, 2), name: nameOf(lang, st.p.fixed),
    lf: fixed(c.lf, 2), ls: fixed(c.ls, 2), kf: fmtK(c.kf), ks: fmtK(c.ks),
  });
}

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const Lb = labels[lang];
  const narrow = st.w < 480;
  const c = compute(p);
  const left = narrow ? 44 : 56, right = st.w - (narrow ? 10 : 14);
  const x = linear([0, XMAX], [left, right]);
  const parts: string[] = [];

  const lg = legend(narrow ? 8 : left, 14, st.w - 16, [
    { kind: "line", color: C.c5, label: nameOf(lang, p.fixed) },
    { kind: "line", color: C.c1, label: Lb.scaled },
    { kind: "band", color: BANDC, label: Lb.mid },
  ]);
  parts.push(lg.svg);

  // Panel (a): density of the distance.
  const aTop = 14 + lg.height + 22, aH = narrow ? 92 : 118;
  const ay = linear([0, 2.15], [aTop + aH, aTop]);
  const bTop = aTop + aH + 44, bH = narrow ? 150 : 190;
  const by = linear([KMIN, 0], [bTop + bH, bTop]);

  // the middle 90% band through both panels
  parts.push(el("rect", { x: x(c.lo), y: aTop, width: Math.max(1, x(c.hi) - x(c.lo)), height: bTop + bH - aTop, fill: BANDC }));

  const NB = narrow ? 150 : 240;
  const edges = grid(0, XMAX, NB + 1);
  const dens: Array<[number, number]> = [];
  for (let i = 0; i < NB; i++) {
    const a = edges[i], b = edges[i + 1];
    const mass = cdfS(c.lw, b * b) - cdfS(c.lw, a * a);
    dens.push([(a + b) / 2, Math.min(2.15, mass / (b - a))]);
  }
  const areaD = `M${x(0)},${ay(0)}` + dens.map(([d, v]) => `L${x(d).toFixed(1)},${ay(v).toFixed(1)}`).join("") + `L${x(XMAX)},${ay(0)}Z`;
  parts.push(
    el("path", { d: areaD, fill: C.ink3, opacity: 0.35, stroke: C.ink2, "stroke-width": 1.2 }),
    el("line", { x1: left, x2: right, y1: aTop + aH, y2: aTop + aH, stroke: C.rule }),
    text(left, aTop - 8, narrow ? Lb.distShort : Lb.dist, { "font-size": TYPE.small, class: "fig-t-muted" }),
  );
  // median label in the gap below panel (a)
  const mx = Math.min(right - 34, Math.max(left + 34, x(c.med)));
  parts.push(text(mx, aTop + aH + 15, tpl(Lb.median, { m: fixed(c.med, 2) }), { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-num fig-t-halo" }));

  // lengthscale markers, spanning both panels
  const lines = [
    { l: c.lf, color: C.c5 },
    { l: c.ls, color: C.c1 },
  ].sort((u, v) => u.l - v.l);
  const close = Math.abs(x(lines[1].l) - x(lines[0].l)) < 64;
  lines.forEach((ln, i) => {
    const px = x(Math.min(ln.l, XMAX));
    parts.push(el("line", { x1: px, x2: px, y1: aTop, y2: bTop + bH, stroke: ln.color, "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
    const anchor = close && i === 0 ? "end" : "start";
    const dx = close && i === 0 ? -4 : 4;
    const yl = aTop + 12;
    parts.push(text(px + dx, yl, `ℓ = ${fixed(ln.l, 2)}`, { "font-size": TYPE.small, "text-anchor": anchor, class: "fig-t-num fig-t-halo", fill: ln.color }));
  });

  // Panel (b): log10 kernel value against distance.
  const decTicks = [0, -10, -20, -30];
  parts.push(
    axis({ scale: by, orient: "left", at: left, span: [left, right], ticks: decTicks, format: (v) => (v === 0 ? "1" : `10${sup(v)}`) }),
    axis({ scale: x, orient: "bottom", at: bTop + bH, span: [bTop, bTop + bH], title: Lb.x, count: narrow ? 4 : 7 }),
    text(narrow ? 8 : left, bTop - 8, Lb.kern, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
  );
  const eps = -53 * Math.log10(2);
  parts.push(
    el("line", { x1: left, x2: right, y1: by(eps), y2: by(eps), stroke: C.ink3, "stroke-width": 1, "stroke-dasharray": "2 3" }),
    text(right - 4, by(eps) - 5, narrow ? Lb.epsShort : Lb.eps, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo" }),
  );
  const cid = `${st.uid}-kclip`;
  parts.push(el("defs", {}, el("clipPath", { id: cid }, el("rect", { x: left, y: bTop - 2, width: right - left, height: bH + 4 }))));
  const XS = grid(0, XMAX, narrow ? 160 : 260);
  const path = (kind: "rbf" | "matern32", l: number) =>
    XS.map((d, i) => `${i ? "L" : "M"}${x(d).toFixed(1)},${by(Math.max(KMIN - 2, log10k(kind, l, d))).toFixed(1)}`).join("");
  parts.push(g({ "clip-path": `url(#${cid})` },
    el("path", { d: path(c.kindF, c.lf), fill: "none", stroke: C.c5, "stroke-width": 2.2 }),
    el("path", { d: path("rbf", c.ls), fill: "none", stroke: C.c1, "stroke-width": 2.2 }),
  ));
  for (const [lgk, color] of [[c.kf, C.c5], [c.ks, C.c1]] as const) {
    if (lgk >= KMIN) parts.push(el("circle", { cx: x(c.med), cy: by(lgk), r: 4.5, fill: color, stroke: C.paper, "stroke-width": 1.6 }));
  }

  // Readout: the typical pair under each default.
  const rows = [
    { name: nameOf(lang, p.fixed), l: c.lf, k: c.kf, color: C.c5 },
    { name: Lb.scaled, l: c.ls, k: c.ks, color: C.c1 },
  ];
  let ry = bTop + bH + 50;
  for (const r of rows) {
    const vars = { name: r.name, l: fixed(r.l, 2), r: fixed(c.med / r.l, 1), k: fmtK(r.k) };
    parts.push(el("circle", { cx: (narrow ? 8 : left) + 5, cy: ry - 4, r: 4.5, fill: r.color }));
    const tx = (narrow ? 8 : left) + 16;
    if (narrow) {
      parts.push(text(tx, ry, tpl(Lb.typicalA, vars), { "font-size": TYPE.small, class: "fig-t-strong" }));
      parts.push(text(tx, ry + 15, tpl(Lb.typicalB, vars), { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
      ry += 36;
    } else {
      parts.push(text(tx, ry, tpl(Lb.typical, vars), { "font-size": TYPE.body, class: "fig-t-num" }));
      ry += 20;
    }
  }
  return svg(st.w, ry - (narrow ? 14 : 6), describe(st), g({}, ...parts));
}

export default defineFigure({
  name: "hd-lengthscale",
  title: { en: "Distances in the unit cube against default lengthscales", zh: "单位立方体中的距离与默认长度尺度" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
