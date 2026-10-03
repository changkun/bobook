// A comparison as a random utility. Each time a person looks at options A
// and B, they perceive a utility for each: the true utility plus noise. They
// choose the option that seems better on that occasion. The top panel shows
// the two noise distributions and the latest occasion; the middle panel shows
// the probability of choosing A as a function of the utility difference, with
// the share of A choices among the simulated occasions so far. Gaussian noise
// gives the probit link of Thurstone's Case V; Gumbel noise with the same
// standard deviation gives the logistic link of Bradley and Terry. An
// optional bottom panel shows what each link charges, in negative log
// probability, for an answer that contradicts the utility difference.

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid } from "./lib/scale.ts";
import { memo, normal, rng } from "./lib/random.ts";
import { Phi, logPhi, normalPdf, sigmoid } from "./lib/stats.ts";
import { curve, frame, frameAxes, clip } from "./lib/plot.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    perceived: "perceived utility on one occasion",
    density: "density",
    prob: "P(A chosen)",
    delta: "utility difference g(A) − g(B)",
    cost: "−log P(B chosen)",
    probit: "Gaussian noise (probit)",
    logit: "Gumbel noise (logit)",
    thisTime: "occasion {i}: {who} seems better",
    noDraws: "press play to simulate occasions",
    readout: "P(A) = {p} · A chosen {k} of {n}",
    readoutNone: "P(A) = {p}",
    costTitle: "cost of choosing B, by link",
    describe: "With a utility difference of {d} and noise sd {s} per option, the {link} link gives P(A chosen) = {p}. Over {n} simulated occasions, A was chosen {k} times.",
    linkProbit: "probit",
    linkLogit: "logit",
    kf0: "before any answer",
    kf1: "a single answer",
    kf2: "a few answers",
    kf3: "the share settles",
  },
  zh: {
    perceived: "某一次感知到的效用",
    density: "密度",
    prob: "P(选择 A)",
    delta: "效用差 g(A) − g(B)",
    cost: "−log P(选择 B)",
    probit: "高斯噪声（概率单位链接）",
    logit: "Gumbel 噪声（逻辑链接）",
    thisTime: "第 {i} 次：{who} 看起来更好",
    noDraws: "按播放键逐次模拟",
    readout: "P(A) = {p} · {n} 次中选择 A {k} 次",
    readoutNone: "P(A) = {p}",
    costTitle: "各链接下选择 B 的代价",
    describe: "效用差为 {d}、每个选项的噪声标准差为 {s} 时，{link}链接给出 P(选择 A) = {p}。在 {n} 次模拟中，A 被选中 {k} 次。",
    linkProbit: "概率单位",
    linkLogit: "逻辑",
    kf0: "尚无回答",
    kf1: "一次回答",
    kf2: "几次回答",
    kf3: "比例趋于稳定",
  },
};

const params = {
  delta: { kind: "range", label: { en: "Difference g(A) − g(B)", zh: "效用差 g(A) − g(B)" }, min: -3, max: 3, default: 0.8, step: 0.05 },
  sigma: { kind: "range", label: { en: "Noise sd σ", zh: "噪声标准差 σ" }, min: 0.1, max: 2, default: 0.6, step: 0.05 },
  link: { kind: "choice", label: { en: "Noise", zh: "噪声" }, options: [{ value: "probit", label: { en: "Gaussian (probit)", zh: "高斯（概率单位）" } }, { value: "logit", label: { en: "Gumbel (logit)", zh: "Gumbel（逻辑）" } }], default: "probit" },
  cost: { kind: "toggle", label: { en: "Show the cost of a surprising answer", zh: "显示意外回答的代价" }, default: false },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 5, step: 1, control: false },
} as const;

type P = { delta: number; sigma: number; link: "probit" | "logit"; cost: boolean; seed: number };

const EULER = 0.5772156649015329;
const N_DRAWS = 60;
const DX = grid(-3, 3, 241);

// Gumbel scale with the same standard deviation as a Gaussian of sd sigma:
// Var = pi^2 beta^2 / 6.
const beta = (s: number) => (s * Math.sqrt(6)) / Math.PI;

// P(A chosen) for a utility difference d and per-option noise sd s.
function pA(link: string, d: number, s: number): number {
  return link === "probit" ? Phi(d / (Math.SQRT2 * s)) : sigmoid(d / beta(s));
}

// -log P(B chosen) = -log(1 - P(A chosen)), computed stably.
function costB(link: string, d: number, s: number): number {
  if (link === "probit") return -logPhi(-d / (Math.SQRT2 * s));
  const z = d / beta(s);
  return z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z));
}

// Density of the perceived utility of an option with mean m.
function density(link: string, x: number, m: number, s: number): number {
  if (link === "probit") return normalPdf(x, m, s);
  const b = beta(s);
  const z = (x - (m - EULER * b)) / b;
  return Math.exp(-(z + Math.exp(-z))) / b;
}

// The simulated occasions, drawn in sequence from one seeded generator:
// perceived utilities of A and B on each occasion.
const occasions = memo((delta: number, sigma: number, link: string, seed: number): Array<[number, number]> => {
  const r = rng(seed * 7919 + 13);
  const z = normal(r);
  const b = beta(sigma);
  const gum = () => { let u = r(); while (u <= 1e-12) u = r(); return -b * Math.log(-Math.log(u)) - EULER * b; };
  const out: Array<[number, number]> = [];
  for (let i = 0; i < N_DRAWS; i++) {
    const ea = link === "probit" ? sigma * z() : gum();
    const eb = link === "probit" ? sigma * z() : gum();
    out.push([delta / 2 + ea, -delta / 2 + eb]);
  }
  return out;
}, 16);

const draw = (p: P, i: number) => occasions(p.delta, p.sigma, p.link, p.seed)[i - 1];

function tally(p: P, n: number): number {
  const xs = occasions(p.delta, p.sigma, p.link, p.seed);
  let k = 0;
  for (let i = 0; i < n; i++) if (xs[i][0] > xs[i][1]) k++;
  return k;
}

function describe(st: State<P>): string {
  const p = st.p;
  const n = Math.round(st.t);
  const L = labels[st.lang ?? "en"];
  return tpl(L.describe, { d: fixed(p.delta, 2), s: fixed(p.sigma, 2), link: p.link === "probit" ? L.linkProbit : L.linkLogit, p: fixed(pA(p.link, p.delta, p.sigma), 2), n, k: tally(p, n) });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const n = Math.round(st.t);
  const parts: string[] = [];
  const other = p.link === "probit" ? "logit" : "probit";

  // 1. The two perceived-utility distributions and the latest occasion.
  const f1 = frame({ w: st.w, top: narrow ? 38 : 22, height: narrow ? 84 : 96, xDomain: [-4.5, 4.5], yDomain: [0, 1], yTitle: "" });
  const XS = grid(-4.5, 4.5, 241);
  const dA = XS.map((x) => density(p.link, x, p.delta / 2, p.sigma));
  const dB = XS.map((x) => density(p.link, x, -p.delta / 2, p.sigma));
  const top = Math.max(...dA, ...dB) * 1.12;
  f1.y.domain[1] = top;
  const y1 = (v: number) => f1.bottom - (v / top) * (f1.bottom - f1.top);
  const areaPath = (ds: number[]) => `M${f1.x(XS[0])},${f1.bottom}` + XS.map((x, i) => `L${f1.x(x)},${y1(ds[i])}`).join("") + `L${f1.x(XS[XS.length - 1])},${f1.bottom}Z`;
  parts.push(
    text(f1.left, 12, L.perceived, { "font-size": TYPE.small, class: "fig-t-muted" }),
    el("line", { x1: f1.left, x2: f1.right, y1: f1.bottom, y2: f1.bottom, stroke: C.rule }),
    el("path", { d: areaPath(dB), fill: C.c7, opacity: 0.16 }),
    el("path", { d: areaPath(dA), fill: C.acq, opacity: 0.16 }),
    el("path", { d: areaPath(dB).replace(/Z$/, ""), fill: "none", stroke: C.c7, "stroke-width": 1.8 }),
    el("path", { d: areaPath(dA).replace(/Z$/, ""), fill: "none", stroke: C.acq, "stroke-width": 1.8 }),
  );
  // option labels at the means
  const lab = (m: number, s: string, color: string) => text(f1.x(m), y1(density(p.link, m, m, p.sigma)) - 5, s, { "text-anchor": "middle", "font-size": TYPE.label, class: "fig-t-strong fig-t-halo", style: `fill:${color}` });
  parts.push(lab(-p.delta / 2, "B", C.c7), lab(p.delta / 2, "A", C.acq));
  if (n >= 1) {
    const [a, b] = draw(p, n);
    const mark = (v: number, color: string, win: boolean) => {
      const x = Math.max(f1.left, Math.min(f1.right, f1.x(v)));
      return el("path", { d: `M${x},${f1.bottom + 2}l-5,9h10z`, fill: win ? color : C.paper, stroke: color, "stroke-width": 1.6 });
    };
    parts.push(mark(b, C.c7, b > a), mark(a, C.acq, a >= b));
    parts.push(text(f1.right, narrow ? 28 : 12, tpl(L.thisTime, { i: n, who: a > b ? "A" : "B" }), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-strong" }));
  } else {
    parts.push(text(f1.right, narrow ? 28 : 12, L.noDraws, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-faint" }));
  }

  // 2. The choice curve.
  const f2top = f1.bottom + 40;
  const f2 = frame({ w: st.w, top: f2top, height: narrow ? 150 : 170, xDomain: [-3, 3], yDomain: [0, 1], yTitle: L.prob });
  const cid = `${st.uid}-c2`;
  const pCur = pA(p.link, p.delta, p.sigma);
  parts.push(
    el("defs", {}, clip(f2, cid)),
    frameAxes(f2, { xTitle: L.delta, yTitle: L.prob, yCount: 4 }),
    g({ "clip-path": `url(#${cid})` },
      curve(f2, DX, DX.map((d) => pA(other, d, p.sigma)), { stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "5 4" }),
      curve(f2, DX, DX.map((d) => pA(p.link, d, p.sigma)), { stroke: C.model, "stroke-width": 2.2 }),
    ),
    el("line", { x1: f2.x(p.delta), x2: f2.x(p.delta), y1: f2.top, y2: f2.bottom, stroke: C.acq, "stroke-width": 1.3, "stroke-dasharray": "4 3" }),
    el("circle", { cx: f2.x(p.delta), cy: f2.y(pCur), r: 4.5, fill: C.model, stroke: C.paper, "stroke-width": 1.5 }),
  );
  let readout = tpl(L.readoutNone, { p: fixed(pCur, 3) });
  if (n >= 1) {
    const k = tally(p, n);
    const ph = k / n;
    const se = Math.sqrt(Math.max(ph * (1 - ph), 0.25 / n) / n);
    const lo = Math.max(0, ph - 1.96 * se), hi = Math.min(1, ph + 1.96 * se);
    const xo = f2.x(p.delta) + 9;
    parts.push(
      el("line", { x1: xo, x2: xo, y1: f2.y(lo), y2: f2.y(hi), stroke: C.ink, "stroke-width": 1.4 }),
      el("circle", { cx: xo, cy: f2.y(ph), r: 4, fill: C.paper, stroke: C.ink, "stroke-width": 1.6 }),
    );
    readout = tpl(L.readout, { p: fixed(pCur, 3), k, n });
  }
  // legend and readout, above the plot
  const legY = f2top - 10;
  const items: Array<[string, string, boolean]> = [[p.link === "probit" ? L.probit : L.logit, C.model, false], [p.link === "probit" ? L.logit : L.probit, C.ink3, true]];
  let lx = f2.left;
  if (!narrow) {
    for (const [name, color, dash] of items) {
      parts.push(el("line", { x1: lx, x2: lx + 18, y1: legY - 4, y2: legY - 4, stroke: color, "stroke-width": 2, "stroke-dasharray": dash ? "5 4" : undefined }), text(lx + 23, legY, name, { "font-size": TYPE.small, class: "fig-t-muted" }));
      lx += labelWidth(name, 6.3) + 40;
    }
    parts.push(text(f2.right, legY, readout, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-num fig-t-strong" }));
  } else {
    parts.push(text(f2.left, legY, readout, { "font-size": TYPE.small, class: "fig-t-num fig-t-strong" }));
  }
  let H = f2.bottom + 44;
  if (narrow) {
    H += 8;
    // legend below the axis title on a phone
    let ly = H;
    for (const [name, color, dash] of items) {
      parts.push(el("line", { x1: f2.left, x2: f2.left + 18, y1: ly - 4, y2: ly - 4, stroke: color, "stroke-width": 2, "stroke-dasharray": dash ? "5 4" : undefined }), text(f2.left + 23, ly, name, { "font-size": TYPE.small, class: "fig-t-muted" }));
      ly += 16;
    }
    H = ly + 2;
  }

  // 3. Optional: the cost of an answer that goes against the difference.
  if (p.cost) {
    const f3top = H + 22;
    const ymax = Math.max(4, costB("logit", 3, p.sigma) * 1.8);
    const f3 = frame({ w: st.w, top: f3top, height: narrow ? 120 : 140, xDomain: [-3, 3], yDomain: [0, ymax], yTitle: L.cost });
    const cid3 = `${st.uid}-c3`;
    parts.push(
      text(f3.left, f3top - 8, L.costTitle, { "font-size": TYPE.small, class: "fig-t-muted" }),
      el("defs", {}, clip(f3, cid3)),
      frameAxes(f3, { xTitle: L.delta, yTitle: L.cost, yCount: 4 }),
      g({ "clip-path": `url(#${cid3})` },
        curve(f3, DX, DX.map((d) => costB(other, d, p.sigma)), { stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "5 4" }),
        curve(f3, DX, DX.map((d) => costB(p.link, d, p.sigma)), { stroke: C.model, "stroke-width": 2.2 }),
      ),
      el("line", { x1: f3.x(p.delta), x2: f3.x(p.delta), y1: f3.top, y2: f3.bottom, stroke: C.acq, "stroke-width": 1.3, "stroke-dasharray": "4 3" }),
    );
    H = f3.bottom + 44;
  }
  return svg(st.w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "cmp-choice-curve",
  title: { en: "Choice probability against utility difference, under Gaussian and Gumbel noise", zh: "高斯噪声与 Gumbel 噪声下，选择概率随效用差的变化" },
  labels,
  params,
  hint: { en: "Press play to simulate occasions one by one; move the sliders to change the difference and the noise.", zh: "按播放键逐次模拟；拖动滑块改变效用差与噪声。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  timeline: {
    duration: () => N_DRAWS,
    rate: 6,
    discrete: true,
    unit: { symbol: { en: "occasions", zh: "次" }, value: (t) => String(Math.round(t)) },
    keyframes: (_p, lang) => [{ t: 0, label: labels[lang].kf0 }, { t: 1, label: labels[lang].kf1 }, { t: 10, label: labels[lang].kf2 }, { t: N_DRAWS, label: labels[lang].kf3 }],
    poster: () => N_DRAWS,
  },
});
