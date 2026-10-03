// The drift-diffusion model of a two-option choice. Evidence for option A
// over option B accumulates with drift v = k·Δu (Δu is the utility
// difference) plus unit Gaussian noise, until it reaches +a (choose A) or -a
// (choose B). The top panel shows sample paths and, in strips above and
// below, the distribution of decision times for each answer; the simulation
// stops at TMAX = 3 s, so decisions still undecided then are left out of the
// strips (213 of 600 at a = 2, Δu = 0.4). The lower panels
// show the closed forms for the choice probability, a logistic function of
// Δu, and the mean response time, which peaks at Δu = 0. An optional overall
// value effect multiplies the drift by (1 + γV), so pairs of good options are
// decided faster and more accurately at the same utility difference.

import { defineFigure, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { memo, normal, rng } from "./lib/random.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    evidence: "evidence for A",
    time: "decision time (s)",
    chooseA: "choose A",
    chooseB: "choose B",
    pA: "P(choose A)",
    rt: "mean response time (s)",
    du: "utility difference Δu",
    duShort: "Δu",
    atV0: "same pair at V = 0",
    readout: "P(choose A) = {p} · mean response time {t} s",
    misread: "A model that ignores overall value reads this response time as Δu = {d}.",
    describe: "Utility difference {du}, boundary {a}, overall value {v}: the person chooses A with probability {p} and responds in {t} seconds on average.",
  },
  zh: {
    evidence: "支持 A 的证据",
    time: "决策时间（秒）",
    chooseA: "选择 A",
    chooseB: "选择 B",
    pA: "P(选择 A)",
    rt: "平均反应时（秒）",
    du: "效用差 Δu",
    duShort: "Δu",
    atV0: "V = 0 时的同一对",
    readout: "P(选择 A) = {p} · 平均反应时 {t} 秒",
    misread: "忽略总体价值的模型会把这个反应时解读为 Δu = {d}。",
    describe: "效用差为 {du}、边界为 {a}、总体价值为 {v} 时，这个人以 {p} 的概率选择 A，平均反应时为 {t} 秒。",
  },
};

const params = {
  du: { kind: "range", label: { en: "Utility difference Δu", zh: "效用差 Δu" }, min: -1.5, max: 1.5, default: 0.4, step: 0.05 },
  a: { kind: "range", label: { en: "Caution (boundary a)", zh: "谨慎程度（边界 a）" }, min: 0.5, max: 2.5, default: 1, step: 0.05 },
  value: { kind: "range", label: { en: "Overall value of the pair V", zh: "这一对的总体价值 V" }, min: 0, max: 1, default: 0, step: 0.05 },
  gain: { kind: "toggle", label: { en: "Overall value speeds decisions", zh: "总体价值加快决策" }, default: false },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 21, step: 1, control: false },
} as const;

type P = { du: number; a: number; value: number; gain: boolean; seed: number };

const K = 1.5; // drift per unit utility difference, per second
const T0 = 0.3; // non-decision time, seconds
const GAMMA = 1.5; // strength of the overall-value effect when switched on
const TMAX = 3;
const DT = 0.005;

const drift = (p: P, du = p.du) => K * du * (p.gain ? 1 + GAMMA * p.value : 1);
const sigm = (x: number) => 1 / (1 + Math.exp(-x));
// Closed forms for the symmetric DDM with unit noise (Bogacz et al. 2006).
const probA = (v: number, a: number) => sigm(2 * a * v);
const meanDT = (v: number, a: number) => (Math.abs(v) < 1e-9 ? a * a : (a / v) * Math.tanh(a * v));

interface Sim { paths: Array<{ pts: Array<[number, number]>; hit: 1 | -1 | 0 }>; timesA: number[]; timesB: number[] }

const simulate = memo((v: number, a: number, seed: number): Sim => {
  const z = normal(rng(seed));
  const sq = Math.sqrt(DT);
  const paths: Sim["paths"] = [];
  const timesA: number[] = [], timesB: number[] = [];
  for (let i = 0; i < 600; i++) {
    let x = 0, t = 0;
    const keep = i < 9;
    const pts: Array<[number, number]> = keep ? [[0, 0]] : [];
    let hit: 1 | -1 | 0 = 0;
    while (t < TMAX) {
      x += v * DT + sq * z();
      t += DT;
      if (keep && Math.round(t / DT) % 4 === 0) pts.push([t, Math.max(-a, Math.min(a, x))]);
      if (x >= a) { hit = 1; break; }
      if (x <= -a) { hit = -1; break; }
    }
    if (hit === 1) timesA.push(t); else if (hit === -1) timesB.push(t);
    if (keep) { pts.push([t, Math.max(-a, Math.min(a, x))]); paths.push({ pts, hit }); }
  }
  return { paths, timesA, timesB };
}, 12);

function describe(st: State<P>): string {
  const p = st.p, v = drift(p);
  return tpl(labels[st.lang ?? "en"].describe, { du: fixed(p.du, 2), a: fixed(p.a, 2), v: fixed(p.value, 2), p: fixed(probA(v, p.a), 2), t: fixed(meanDT(v, p.a) + T0, 2) });
}

function histogram(times: number[], total: number, x: (t: number) => number, y0: number, h: number, up: boolean, color: string): string {
  const bins = 30, bw = TMAX / bins;
  const counts = new Array(bins).fill(0);
  for (const t of times) counts[Math.min(bins - 1, Math.floor(t / bw))]++;
  const peak = Math.max(1, ...counts) / total;
  const scale = h / Math.max(peak, 0.06);
  return g({}, ...counts.map((c, i) => {
    const hh = (c / total) * scale;
    return el("rect", { x: x(i * bw) + 0.5, y: up ? y0 - hh : y0, width: Math.max(0.5, x(bw) - x(0) - 1), height: hh, fill: color, opacity: 0.7 });
  }));
}

function render(st: State<P>): string {
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const p = st.p, w = st.w, narrow = w < 480;
  const v = drift(p);
  const sim = simulate(v, p.a, p.seed);
  const parts: string[] = [];
  const left = narrow ? 40 : 52, right = w - 12;
  // 1. sample paths with decision-time strips above and below
  const stripH = 30;
  const top = 16 + stripH + 6;
  const ph = narrow ? 150 : 170;
  const x = linear([0, TMAX], [left, right]);
  const yMax = 2.6;
  const y = linear([-yMax, yMax], [top + ph, top]);
  parts.push(axis({ scale: y, orient: "left", at: left, span: [left, right], title: L.evidence, count: 4 }));
  parts.push(el("line", { x1: left, x2: right, y1: y(p.a), y2: y(p.a), stroke: C.c1, "stroke-width": 2 }), el("line", { x1: left, x2: right, y1: y(-p.a), y2: y(-p.a), stroke: C.c2, "stroke-width": 2 }));
  parts.push(text(right - 4, y(p.a) - 5, L.chooseA, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-strong fig-t-halo", style: `fill:${C.c1}` }), text(right - 4, y(-p.a) + 14, L.chooseB, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-strong fig-t-halo", style: `fill:${C.c2}` }));
  for (const path of sim.paths) {
    const col = path.hit === 1 ? C.c1 : path.hit === -1 ? C.c2 : C.ink3;
    parts.push(el("path", { d: linePath(path.pts.map(([t, e]) => [x(t), y(e)])), fill: "none", stroke: col, "stroke-width": 1.1, opacity: 0.75 }));
    const end = path.pts[path.pts.length - 1];
    if (path.hit) parts.push(el("circle", { cx: x(end[0]), cy: y(end[1]), r: 2.8, fill: col }));
  }
  const total = sim.timesA.length + sim.timesB.length || 1;
  parts.push(histogram(sim.timesA, total, x, top - 4, stripH, true, C.c1));
  parts.push(histogram(sim.timesB, total, x, top + ph + 4, stripH, false, C.c2));
  const axisY = top + ph + stripH + 10;
  parts.push(axis({ scale: x, orient: "bottom", at: axisY, title: L.time, count: narrow ? 4 : 6 }));
  // 2. closed-form curves against Δu
  const curveTop = axisY + 58;
  const ch = narrow ? 120 : 130;
  const dus = grid(-1.5, 1.5, 121);
  const panels = narrow
    ? [{ l: left, r: right, t: curveTop }, { l: left, r: right, t: curveTop + ch + 64 }]
    : [{ l: left, r: (left + right) / 2 - 24, t: curveTop }, { l: (left + right) / 2 + 40, r: right, t: curveTop }];
  const showRef = p.gain && p.value > 0;
  const refP: P = { ...p, value: 0 };
  // choice probability
  {
    const q = panels[0];
    const xs = linear([-1.5, 1.5], [q.l, q.r]), ys = linear([0, 1], [q.t + ch, q.t]);
    parts.push(axis({ scale: ys, orient: "left", at: q.l, span: [q.l, q.r], title: L.pA, count: 4 }), axis({ scale: xs, orient: "bottom", at: q.t + ch, span: [q.t, q.t + ch], title: L.du, count: 4 }));
    if (showRef) parts.push(el("path", { d: linePath(dus.map((d) => [xs(d), ys(probA(drift(refP, d), p.a))])), fill: "none", stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "4 3" }));
    parts.push(el("path", { d: linePath(dus.map((d) => [xs(d), ys(probA(drift(p, d), p.a))])), fill: "none", stroke: C.model, "stroke-width": 2.2 }));
    parts.push(el("circle", { cx: xs(p.du), cy: ys(probA(v, p.a)), r: 5, fill: C.acq, stroke: C.paper, "stroke-width": 1.5 }));
  }
  // mean response time
  {
    const q = panels[1];
    const rtMax = Math.max(1.5, T0 + p.a * p.a) * 1.08;
    const xs = linear([-1.5, 1.5], [q.l, q.r]), ys = linear([0, rtMax], [q.t + ch, q.t]);
    parts.push(axis({ scale: ys, orient: "left", at: q.l, span: [q.l, q.r], title: L.rt, count: 4 }), axis({ scale: xs, orient: "bottom", at: q.t + ch, span: [q.t, q.t + ch], title: L.du, count: 4 }));
    if (showRef) parts.push(el("path", { d: linePath(dus.map((d) => [xs(d), ys(T0 + meanDT(drift(refP, d), p.a))])), fill: "none", stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "4 3" }));
    parts.push(el("path", { d: linePath(dus.map((d) => [xs(d), ys(T0 + meanDT(drift(p, d), p.a))])), fill: "none", stroke: C.model, "stroke-width": 2.2 }));
    parts.push(el("circle", { cx: xs(p.du), cy: ys(T0 + meanDT(v, p.a)), r: 5, fill: C.acq, stroke: C.paper, "stroke-width": 1.5 }));
  }
  let H = panels[panels.length - 1].t + ch + 52;
  parts.push(text(narrow ? 8 : left, H, tpl(L.readout, { p: fixed(probA(v, p.a), 2), t: fixed(meanDT(v, p.a) + T0, 2) }), { "font-size": TYPE.body, class: "fig-t-strong fig-t-num" }));
  if (showRef) {
    const msg = tpl(L.misread, { d: fixed(p.du * (1 + GAMMA * p.value), 2) });
    const zhCut = msg.indexOf("解读为");
    const lines = narrow ? (lang === "zh" ? [msg.slice(0, zhCut), msg.slice(zhCut)] : [msg.slice(0, msg.indexOf(" reads") ), msg.slice(msg.indexOf(" reads") + 1)]) : [msg];
    for (const ln of lines) { H += 16; parts.push(text(narrow ? 8 : left, H, ln, { "font-size": TYPE.small, class: "fig-t-muted" })); }
    H += 18;
    parts.push(el("line", { x1: narrow ? 8 : left, x2: (narrow ? 8 : left) + 18, y1: H - 4, y2: H - 4, stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "4 3" }), text((narrow ? 8 : left) + 24, H, L.atV0, { "font-size": TYPE.small, class: "fig-t-muted" }));
  }
  return svg(w, H + 10, describe(st), ...parts);
}

export default defineFigure({
  name: "neuro-ddm",
  title: { en: "The drift-diffusion model: how a utility difference shapes choice and response time", zh: "漂移扩散模型：效用差如何决定选择与反应时" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
