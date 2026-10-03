// How far the average of n coin flips can stray from the coin's mean, against
// the three classic bounds on that probability. A coin pays 1 with probability
// theta and 0 otherwise; the figure plots, against n on a log axis,
//
//   exact      P(average of n flips >= theta + a), summed from the binomial
//              distribution (no simulation, so tiny probabilities are exact)
//   Markov     theta / (theta + a), Markov's inequality applied to the average
//              itself, a nonnegative variable with mean theta; it ignores n
//   Chebyshev  theta (1 - theta) / (n a^2), Markov applied to the squared
//              deviation, with the coin's own variance
//   Hoeffding  exp(-2 n a^2), the Chernoff method for rewards in [0, 1]
//
// all one-sided (the average exceeds theta + a). With "union over T" set above
// 1, each curve is for T such averages that must all stay below theta + a: the
// bounds are multiplied by T (the union bound, capped at 1), and the exact
// curve is the probability that at least one of T independent averages
// exceeds theta + a, 1 - (1 - p)^T. The readout gives each curve's value at
// the marked n and the n from which each falls below 0.05.

import { defineFigure, type Lang, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { log as logScale } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { legend } from "./lib/plot.ts";
import { memo } from "./lib/random.ts";
import { sig, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "number of flips n (log scale)",
    y: "probability (log scale)",
    exact: "exact",
    exactT: "exact, any of T",
    markov: "Markov",
    chebyshev: "Chebyshev",
    hoeffding: "Hoeffding",
    level: "0.05",
    at: "n = {n}: exact {e} · Markov {m} · Chebyshev {c} · Hoeffding {h}",
    atT: "n = {n}, T = {T}: exact {e} · Markov {m} · Chebyshev {c} · Hoeffding {h}",
    atShort1: "n = {n}{tt}: exact {e} · Markov {m}",
    atShort2: "Chebyshev {c} · Hoeffding {h}",
    need: "below 0.05 from n = {e} (exact), {c} (Chebyshev), {h} (Hoeffding); Markov never",
    needMarkov: "below 0.05 from n = {e} (exact), {c} (Chebyshev), {h} (Hoeffding), {m} (Markov)",
    needShort1: "below 0.05 from n = {e} (exact),",
    needShort2: "{c} (Chebyshev), {h} (Hoeffding)",
    over: "more than 1000",
    describe: "For a coin with mean {p}, the probability that the average of {n} flips is at least {q} is {e} (exact). Markov's inequality bounds it by {m}, Chebyshev's by {c}, and Hoeffding's by {h}. The probability falls below 0.05 from {ne} flips; the bounds from {nc} flips (Chebyshev) and {nh} flips (Hoeffding){nm}.",
    describeT: "For a coin with mean {p}, the probability that at least one of {T} independent averages of {n} flips is at least {q} is {e} (exact). The union bound with Markov's inequality gives {m}, with Chebyshev's {c}, and with Hoeffding's {h}. The probability falls below 0.05 from {ne} flips; the bounds from {nc} flips (Chebyshev) and {nh} flips (Hoeffding){nm}.",
    markovNever: ", while Markov's never does",
    markovFrom: ", and Markov's from {nm} flips",
  },
  zh: {
    x: "抛掷次数 n（对数刻度）",
    y: "概率（对数刻度）",
    exact: "精确值",
    exactT: "精确值，T 个中任一个",
    markov: "Markov",
    chebyshev: "Chebyshev",
    hoeffding: "Hoeffding",
    level: "0.05",
    at: "n = {n}：精确值 {e} · Markov {m} · Chebyshev {c} · Hoeffding {h}",
    atT: "n = {n}，T = {T}：精确值 {e} · Markov {m} · Chebyshev {c} · Hoeffding {h}",
    atShort1: "n = {n}{tt}：精确值 {e} · Markov {m}",
    atShort2: "Chebyshev {c} · Hoeffding {h}",
    need: "从 n = {e}（精确值）、{c}（Chebyshev）、{h}（Hoeffding）起低于 0.05；Markov 界始终不会",
    needMarkov: "从 n = {e}（精确值）、{c}（Chebyshev）、{h}（Hoeffding）、{m}（Markov）起低于 0.05",
    needShort1: "从 n = {e}（精确值）、",
    needShort2: "{c}（Chebyshev）、{h}（Hoeffding）起低于 0.05",
    over: "1000 以上",
    describe: "均值为 {p} 的硬币抛 {n} 次，平均值不小于 {q} 的概率为 {e}（精确值）。Markov 不等式给出的上界为 {m}，Chebyshev 不等式为 {c}，Hoeffding 不等式为 {h}。该概率从 {ne} 次起低于 0.05；两个界分别从 {nc} 次（Chebyshev）和 {nh} 次（Hoeffding）起低于 0.05{nm}。",
    describeT: "均值为 {p} 的硬币，{T} 个相互独立的 {n} 次抛掷平均值中至少有一个不小于 {q} 的概率为 {e}（精确值）。联合界与 Markov 不等式结合给出 {m}，与 Chebyshev 不等式结合给出 {c}，与 Hoeffding 不等式结合给出 {h}。该概率从 {ne} 次起低于 0.05；两个界分别从 {nc} 次（Chebyshev）和 {nh} 次（Hoeffding）起低于 0.05{nm}。",
    markovNever: "，Markov 界则始终不会",
    markovFrom: "，Markov 界从 {nm} 次起",
  },
};

const UNIONS = [
  { value: 1, label: { en: "1", zh: "1" } },
  { value: 10, label: { en: "10", zh: "10" } },
  { value: 100, label: { en: "100", zh: "100" } },
  { value: 1000, label: { en: "1000", zh: "1000" } },
] as const;

const params = {
  mean: { kind: "range", label: { en: "Coin's mean θ", zh: "硬币的均值 θ" }, min: 0.05, max: 0.95, default: 0.5, step: 0.05 },
  dev: { kind: "range", label: { en: "Deviation a", zh: "偏差 a" }, min: 0.02, max: 0.3, default: 0.1, step: 0.01 },
  n: { kind: "range", label: { en: "Marked n", zh: "标记的 n" }, min: 1, max: 1000, default: 100, scale: "log" },
  union: { kind: "choice", label: { en: "Union over T averages", zh: "对 T 个平均值取联合界" }, options: UNIONS, default: 1, control: "buttons" },
} as const;

type P = { mean: number; dev: number; n: number; union: number };

const NMAX = 1000;
const LEVEL = 0.05;
const FLOOR = 1e-6;

// log(k!) for k = 0..NMAX.
const LF = (() => {
  const out = new Float64Array(NMAX + 1);
  for (let k = 1; k <= NMAX; k++) out[k] = out[k - 1] + Math.log(k);
  return out;
})();

// P(average of n flips >= q) for a coin with mean p, from the binomial pmf.
// The threshold k >= n q is taken with a small tolerance so that n q landing
// on an integer in exact arithmetic (n = 100, q = 0.6) counts that integer.
function upperTail(n: number, p: number, q: number): number {
  const kmin = Math.max(0, Math.ceil(n * q - 1e-9));
  if (kmin > n) return 0;
  const lp = Math.log(p), lq = Math.log1p(-p);
  let s = 0;
  for (let k = kmin; k <= n; k++) s += Math.exp(LF[n] - LF[k] - LF[n - k] + k * lp + (n - k) * lq);
  return Math.min(1, s);
}

// The exact single-average tail for n = 1..NMAX (index n - 1).
const tails = memo((p: number, a: number): Float64Array => {
  const out = new Float64Array(NMAX);
  for (let n = 1; n <= NMAX; n++) out[n - 1] = upperTail(n, p, p + a);
  return out;
}, 16);

// At least one of T independent averages exceeds the line.
const anyOf = (v: number, T: number) => (T === 1 ? v : v >= 1 ? 1 : -Math.expm1(T * Math.log1p(-v)));

interface Curves { exact: number; markov: number; chebyshev: number; hoeffding: number }

function at(p: P, n: number): Curves {
  const th = p.mean, a = p.dev, T = p.union;
  return {
    exact: anyOf(tails(th, a)[n - 1], T),
    markov: Math.min(1, (T * th) / (th + a)),
    chebyshev: Math.min(1, (T * th * (1 - th)) / (n * a * a)),
    hoeffding: Math.min(1, T * Math.exp(-2 * n * a * a)),
  };
}

// The n from which each curve stays below LEVEL. For the bounds, which fall
// with n, it is a formula and may exceed the plotted range; for the exact
// probability, which zigzags with n because the flips are whole numbers, it is
// the n after the last value above LEVEL among n <= 1000.
function needed(p: P) {
  const th = p.mean, a = p.dev, T = p.union;
  const cheb = Math.max(1, Math.ceil((T * th * (1 - th)) / (LEVEL * a * a) - 1e-9));
  const hoef = Math.max(1, Math.ceil(Math.log(T / LEVEL) / (2 * a * a) - 1e-9));
  const markovOk = (T * th) / (th + a) <= LEVEL;
  const tl = tails(th, a);
  let last = 0;
  for (let n = 1; n <= NMAX; n++) if (anyOf(tl[n - 1], T) > LEVEL) last = n;
  const exact = last >= NMAX ? Infinity : last + 1;
  return { cheb, hoef, markovOk, exact };
}

const nOf = (p: P) => Math.min(NMAX, Math.max(1, Math.round(p.n)));
const num = (v: number, lang: Lang = "en") => (v === Infinity ? labels[lang].over : v.toLocaleString("en-US"));
const SUP: Record<string, string> = { "-": "⁻", "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹" };
const sup = (s: string) => s.replace(/./g, (ch) => SUP[ch] ?? ch);
// Two significant figures; below 0.001 as 3.2×10⁻⁴ with a real superscript.
function pr(v: number): string {
  if (v >= 0.995) return "1";
  if (v === 0) return "0";
  if (v >= 1e-3) return sig(v, 2);
  const [m, e] = v.toExponential(1).split("e");
  return `${Number(m)}×10${sup(String(Number(e)))}`;
}

function describe(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const n = nOf(p);
  const c = at(p, n);
  const nd = needed(p);
  return tpl(p.union === 1 ? L.describe : L.describeT, {
    p: sig(p.mean, 2), q: sig(p.mean + p.dev, 2), n, T: p.union,
    e: pr(c.exact), m: pr(c.markov), c: pr(c.chebyshev), h: pr(c.hoeffding),
    ne: num(nd.exact, lang), nc: num(nd.cheb, lang), nh: num(nd.hoef, lang),
    nm: nd.markovOk ? tpl(L.markovFrom, { nm: 1 }) : L.markovNever,
  });
}

const yFmt = (v: number) => (v >= 0.01 ? String(v) : `10${sup(String(Math.round(Math.log10(v))))}`);

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 480;
  const n = nOf(p);
  const T = p.union;
  const parts: string[] = [];
  const COL = { exact: C.ink, markov: C.c8, chebyshev: C.c7, hoeffding: C.c6 };

  const lg = legend(narrow ? 8 : 58, 16, st.w - (narrow ? 16 : 74), [
    { kind: "line", color: COL.exact, label: T === 1 ? L.exact : L.exactT },
    { kind: "dash", color: COL.markov, label: L.markov },
    { kind: "dash", color: COL.chebyshev, label: L.chebyshev },
    { kind: "dash", color: COL.hoeffding, label: L.hoeffding },
  ]);
  parts.push(lg.svg);

  const left = narrow ? 46 : 58, right = st.w - 14;
  const top = 16 + lg.height + (narrow ? 18 : 4), h = narrow ? 190 : 240;
  const x = logScale([1, NMAX], [left, right]);
  const y = logScale([FLOOR, 1], [top + h, top]);
  parts.push(axis({ scale: y, orient: "left", at: left, span: [left, right], title: narrow ? undefined : L.y, format: yFmt }));
  parts.push(axis({ scale: x, orient: "bottom", at: top + h, span: [top, top + h], title: L.x, ticks: [1, 10, 100, 1000], format: (v) => String(v) }));
  if (narrow) parts.push(text(left - 30, top - 8, L.y, { "font-size": TYPE.small, class: "fig-t-muted" }));

  // The 0.05 reference level.
  parts.push(el("line", { x1: left, x2: right, y1: y(LEVEL), y2: y(LEVEL), stroke: C.ink3, "stroke-width": 1, "stroke-dasharray": "2 3" }));
  parts.push(text(left + 4, y(LEVEL) + 13, L.level, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo fig-t-num" }));

  const cid = `${st.uid}-clip`;
  parts.push(el("defs", {}, el("clipPath", { id: cid }, el("rect", { x: left, y: top - 2, width: right - left, height: h + 4 }))));
  const ns = Array.from({ length: NMAX }, (_, i) => i + 1);
  const series = ns.map((k) => at(p, k));
  const yy = (v: number) => y(Math.max(FLOOR / 10, v));
  const path = (key: keyof Curves) => ns.map((k, i) => `${i ? "L" : "M"}${x(k).toFixed(1)},${yy(series[i][key]).toFixed(1)}`).join("");
  // The exact probability zigzags with n (the number of heads is a whole
  // number). Where several n share a pixel column, which happens from a few
  // dozen flips on, drawing every n would fill a band; the curve keeps the
  // largest value in each column instead, so it traces the top of the zigzag.
  const cols = new Map<number, { px: number; v: number }>();
  for (let i = 0; i < NMAX; i++) {
    const px = Math.round(x(ns[i]));
    const v = series[i].exact;
    const c0 = cols.get(px);
    if (!c0) cols.set(px, { px: x(ns[i]), v });
    else if (v > c0.v) cols.set(px, { px: c0.px, v });
  }
  const exactPath = [...cols.values()].map((c0, i) => `${i ? "L" : "M"}${c0.px.toFixed(1)},${yy(c0.v).toFixed(1)}`).join("");
  parts.push(g({ "clip-path": `url(#${cid})` },
    el("path", { d: path("markov"), fill: "none", stroke: COL.markov, "stroke-width": 2, "stroke-dasharray": "6 4" }),
    el("path", { d: path("chebyshev"), fill: "none", stroke: COL.chebyshev, "stroke-width": 2, "stroke-dasharray": "6 4" }),
    el("path", { d: path("hoeffding"), fill: "none", stroke: COL.hoeffding, "stroke-width": 2, "stroke-dasharray": "6 4" }),
    el("path", { d: exactPath, fill: "none", stroke: COL.exact, "stroke-width": 2, "stroke-linejoin": "round" }),
  ));

  // The marked n.
  const c = series[n - 1];
  const mx = x(n);
  parts.push(el("line", { x1: mx, x2: mx, y1: top, y2: top + h, stroke: C.ink3, "stroke-width": 1.2, "stroke-dasharray": "4 3" }));
  for (const key of ["markov", "chebyshev", "hoeffding", "exact"] as const) {
    if (c[key] < FLOOR) continue;
    parts.push(el("circle", { cx: mx, cy: y(c[key]), r: 3.6, fill: COL[key], stroke: C.paper, "stroke-width": 1.4 }));
  }

  // Readout.
  const nd = needed(p);
  const vals = { n, T, tt: T === 1 ? "" : lang === "zh" ? `，T = ${T}` : `, T = ${T}`, e: pr(c.exact), m: pr(c.markov), c: pr(c.chebyshev), h: pr(c.hoeffding) };
  const ry = top + h + 48;
  const lines: string[] = narrow
    ? [tpl(L.atShort1, vals), tpl(L.atShort2, vals), tpl(L.needShort1, { e: num(nd.exact, lang) }), tpl(L.needShort2, { c: num(nd.cheb, lang), h: num(nd.hoef, lang) })]
    : [tpl(T === 1 ? L.at : L.atT, vals), tpl(nd.markovOk ? L.needMarkov : L.need, { e: num(nd.exact, lang), c: num(nd.cheb, lang), h: num(nd.hoef, lang), m: 1 })];
  lines.forEach((s, i) => parts.push(text(left, ry + 16 * i, s, { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" })));
  const H = ry + 16 * (lines.length - 1) + 10;
  return svg(st.w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "regret-concentration",
  title: { en: "The tail probability of an average of coin flips against Markov's, Chebyshev's, and Hoeffding's bounds", zh: "抛硬币平均值的尾概率，与 Markov、Chebyshev 和 Hoeffding 三个界的比较" },
  labels,
  params,
  hint: { en: "The curves are one-sided: the chance that the average is at least θ + a. The union over T multiplies every bound by T.", zh: "各曲线都是单侧的：平均值不小于 θ + a 的概率。对 T 个平均值取联合界时，每个界都乘以 T。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  // States the chapter quotes beyond the defaults.
  snapshots: {
    n1000: (p) => ({ ...p, n: 1000 }),
    union1000: (p) => ({ ...p, union: 1000 }),
    mean01: (p) => ({ ...p, mean: 0.1 }),
  },
});
