// Entropy and the two directions of the KL divergence. The distribution p is
// a fixed-shape mixture of two Gaussian bumps whose separation the reader
// sets; q is a single Gaussian whose mean and standard deviation the reader
// sets, or fits with one of two buttons. Minimizing KL(p || q) over Gaussians
// matches the mean and variance of p and spreads q over both bumps;
// minimizing KL(q || p) is done within the basin of the current q (the side of
// the midpoint its mean is on) and settles on one bump. The lower panel draws the integrand of the chosen
// direction, so the reader sees where the divergence is accumulated. All
// integrals are trapezoid sums on a grid, with log densities computed
// analytically so that tails never produce log 0.

import { defineFigure, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { clamp, grid } from "./lib/scale.ts";
import { clip, frame, frameAxes, legend } from "./lib/plot.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "x",
    dens: "density",
    integ: "integrand",
    p: "p: two bumps",
    q: "q: one Gaussian",
    fwd: "p log(p/q), whose area is KL(p‖q)",
    rev: "q log(q/p), whose area is KL(q‖p)",
    hp: "H(p) = {h} nats",
    hq: "H(q) = {h} nats",
    kpq: "KL(p‖q) = {k} nats",
    kqp: "KL(q‖p) = {k} nats",
    describe: "p has two bumps {d} apart; q is a Gaussian with mean {m} and sd {s}. KL(p‖q) = {f} nats and KL(q‖p) = {r} nats; the entropies are {hp} and {hq} nats.",
  },
  zh: {
    x: "x",
    dens: "密度",
    integ: "被积函数",
    p: "p：两个峰",
    q: "q：单个高斯分布",
    fwd: "p log(p/q)，其面积为 KL(p‖q)",
    rev: "q log(q/p)，其面积为 KL(q‖p)",
    hp: "H(p) = {h} 奈特",
    hq: "H(q) = {h} 奈特",
    kpq: "KL(p‖q) = {k} 奈特",
    kqp: "KL(q‖p) = {k} 奈特",
    describe: "p 有两个相距 {d} 的峰；q 是均值为 {m}、标准差为 {s} 的高斯分布。KL(p‖q) = {f} 奈特，KL(q‖p) = {r} 奈特；两者的熵分别为 {hp} 和 {hq} 奈特。",
  },
};

const params = {
  sep: { kind: "range", label: { en: "Bump separation", zh: "两峰间距" }, min: 0, max: 5, default: 4, step: 0.1 },
  m: { kind: "range", label: { en: "Mean of q", zh: "q 的均值" }, min: -4, max: 4, default: 0.5, step: 0.05 },
  s: { kind: "range", label: { en: "Sd of q", zh: "q 的标准差" }, min: 0.2, max: 3, default: 1, step: 0.01 },
  show: { kind: "choice", label: { en: "Integrand", zh: "被积函数" }, options: [{ value: "fwd", label: { en: "KL(p‖q)", zh: "KL(p‖q)" } }, { value: "rev", label: { en: "KL(q‖p)", zh: "KL(q‖p)" } }], default: "fwd", control: "buttons" },
} as const;

type P = { sep: number; m: number; s: number; show: "fwd" | "rev" };

const XS = grid(-8, 8, 801);
const DX = XS[1] - XS[0];
const SC = 0.6; // sd of each bump of p
const W1 = 0.5; // weight of the left bump; the right one has the rest
const LOG2PI = Math.log(2 * Math.PI);

const logN = (x: number, m: number, s: number) => -0.5 * ((x - m) / s) ** 2 - Math.log(s) - 0.5 * LOG2PI;
function logP(x: number, sep: number): number {
  const a = Math.log(W1) + logN(x, -sep / 2, SC), b = Math.log(1 - W1) + logN(x, sep / 2, SC);
  const mx = Math.max(a, b);
  return mx + Math.log(Math.exp(a - mx) + Math.exp(b - mx));
}

function trap(ys: number[]): number {
  let s = 0;
  for (let i = 1; i < ys.length; i++) s += 0.5 * (ys[i] + ys[i - 1]) * DX;
  return s;
}

function stats(sep: number, m: number, s: number) {
  const lp = XS.map((x) => logP(x, sep));
  const lq = XS.map((x) => logN(x, m, s));
  const p = lp.map(Math.exp), q = lq.map(Math.exp);
  const fwdI = p.map((v, i) => v * (lp[i] - lq[i]));
  const revI = q.map((v, i) => v * (lq[i] - lp[i]));
  return {
    p, q, fwdI, revI,
    hp: trap(p.map((v, i) => -v * lp[i])),
    hq: 0.5 * Math.log(2 * Math.PI * Math.E * s * s),
    kpq: trap(fwdI), kqp: trap(revI),
  };
}

// Mean and sd of p: the minimizer of KL(p || q) over Gaussians q.
function momentMatch(sep: number): { m: number; s: number } {
  const m1 = -sep / 2, m2 = sep / 2;
  const mean = W1 * m1 + (1 - W1) * m2;
  const v = W1 * (SC * SC + m1 * m1) + (1 - W1) * (SC * SC + m2 * m2) - mean * mean;
  return { m: mean, s: Math.sqrt(v) };
}

// Minimizer of KL(q || p) in the basin of the current q. The reverse KL has
// one local minimum per well-separated bump of p, and which one an optimizer
// finds depends on where it starts. So: search a coarse grid restricted to
// the side of the midpoint where q's mean now is, then refine by pattern
// search. If the restricted optimum sits on the midpoint (p is effectively
// unimodal), search both sides instead.
function fitReverse(sep: number, m0: number, s0: number): { m: number; s: number } {
  const mid = 0;
  const side = m0 > mid ? 1 : -1;
  const coarse = (lo: number, hi: number) => {
    let best = Infinity, bm = 0, bs = 1;
    for (let m = lo; m <= hi + 1e-9; m += 0.1) for (let s = 0.2; s <= 3 + 1e-9; s += 0.1) {
      const v = stats(sep, m, s).kqp;
      if (v < best) { best = v; bm = m; bs = s; }
    }
    return { m: bm, s: bs };
  };
  let start = side > 0 ? coarse(mid, 4) : coarse(-4, mid);
  if (Math.abs(start.m - mid) < 0.05) start = coarse(-4, 4);
  let m = start.m, s = start.s, best = stats(sep, m, s).kqp;
  void s0;
  let step = 0.1;
  while (step > 0.004) {
    let moved = false;
    for (const [dm, ds] of [[step, 0], [-step, 0], [0, step / 2], [0, -step / 2]]) {
      const mm = clamp(m + dm, -4, 4), ss = clamp(s + ds, 0.2, 3);
      const v = stats(sep, mm, ss).kqp;
      if (v < best - 1e-9) { best = v; m = mm; s = ss; moved = true; }
    }
    if (!moved) step /= 2;
  }
  return { m, s };
}

const round = (v: number, q: number) => Math.round(v / q) * q;

function describe(st: State<P>): string {
  const p = st.p, r = stats(p.sep, p.m, p.s);
  return tpl(labels[st.lang ?? "en"].describe, { d: fixed(p.sep, 1), m: fixed(p.m, 2), s: fixed(p.s, 2), f: fixed(r.kpq, 2), r: fixed(r.kqp, 2), hp: fixed(r.hp, 2), hq: fixed(r.hq, 2) });
}

function render(st: State<P>): string {
  const p = st.p, L = labels[st.lang ?? "en"], w = st.w;
  const narrow = w < 480;
  const r = stats(p.sep, p.m, p.s);
  const lg = legend(narrow ? 8 : 52, 14, w - 16, [
    { kind: "band", color: C.c5, label: L.p },
    { kind: "line", color: C.model, label: L.q },
  ]);
  const top = 14 + lg.height + 6;
  const ymax = Math.max(...r.p, ...r.q) * 1.08;
  const f = frame({ w, top, height: narrow ? 150 : 180, xDomain: [-6, 6], yDomain: [0, ymax], yTitle: L.dens });
  const integ = p.show === "fwd" ? r.fwdI : r.revI;
  const imax = Math.max(0.05, ...integ.map(Math.abs)) * 1.1;
  const imin = Math.min(0, ...integ) * 1.1;
  const f2 = frame({ w, top: f.bottom + 30, height: narrow ? 80 : 96, xDomain: [-6, 6], yDomain: [Math.max(imin, -imax), imax], yTitle: narrow ? "" : L.integ });
  const c1 = `${st.uid}-c1`, c2 = `${st.uid}-c2`;
  const pathOf = (fr: typeof f, ys: number[]) => linePath(XS.map((x, i) => [fr.x(x), fr.y(ys[i])]));
  const area = (fr: typeof f, ys: number[], color: string, op: number) => el("path", {
    d: pathOf(fr, ys) + `L${fr.x(8)},${fr.y(0)}L${fr.x(-8)},${fr.y(0)}Z`, fill: color, "fill-opacity": op, stroke: color, "stroke-width": 1.6,
  });
  const icolor = p.show === "fwd" ? C.c5 : C.model;
  const parts: string[] = [
    el("defs", {}, clip(f, c1), clip(f2, c2)),
    lg.svg,
    frameAxes(f, { yTitle: L.dens, xTicks: false, yCount: 3 }),
    g({ "clip-path": `url(#${c1})` },
      area(f, r.p, C.c5, 0.18),
      el("path", { d: pathOf(f, r.q), fill: "none", stroke: C.model, "stroke-width": 2.4 }),
    ),
    frameAxes(f2, { yTitle: narrow ? "" : L.integ, xTitle: L.x, yCount: 2 }),
    g({ "clip-path": `url(#${c2})` }, area(f2, integ, icolor, 0.25)),
    text(f2.left + 6, f2.top + 10, p.show === "fwd" ? L.fwd : L.rev, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
  ];
  // readout
  let y = f2.bottom + 50;
  const cols = narrow ? 1 : 2;
  const x0 = narrow ? 8 : f.left;
  const colW = (w - x0 - 10) / cols;
  const items: Array<[string, string]> = [
    [tpl(L.hp, { h: fixed(r.hp, 2) }), "fig-t-muted"],
    [tpl(L.hq, { h: fixed(r.hq, 2) }), "fig-t-muted"],
    [tpl(L.kpq, { k: fixed(r.kpq, 2) }), p.show === "fwd" ? "fig-t-strong" : "fig-t-muted"],
    [tpl(L.kqp, { k: fixed(r.kqp, 2) }), p.show === "rev" ? "fig-t-strong" : "fig-t-muted"],
  ];
  items.forEach(([s, cls], i) => {
    const col = cols === 2 ? i % 2 : 0, row = cols === 2 ? Math.floor(i / 2) : i;
    parts.push(text(x0 + col * colW, y + row * 18, s, { "font-size": TYPE.body, class: `${cls} fig-t-num` }));
  });
  y += (cols === 2 ? 2 : 4) * 18;
  return svg(w, y - 8, describe(st), ...parts);
}

export default defineFigure({
  name: "info-kl",
  title: { en: "Entropy and the two directions of the KL divergence", zh: "熵与 KL 散度的两个方向" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    {
      label: { en: "Fit q by minimizing KL(p‖q)", zh: "最小化 KL(p‖q) 来拟合 q" }, primary: true,
      run: (p) => { const r = momentMatch(p.sep); return { ...p, m: round(clamp(r.m, -4, 4), 0.05), s: round(clamp(r.s, 0.2, 3), 0.01), show: "fwd" }; },
    },
    {
      label: { en: "Fit q by minimizing KL(q‖p)", zh: "最小化 KL(q‖p) 来拟合 q" }, primary: true,
      run: (p) => { const r = fitReverse(p.sep, p.m, p.s); return { ...p, m: round(r.m, 0.05), s: round(r.s, 0.01), show: "rev" }; },
    },
  ],
});
