// Who moved: the person or the system? A simulated person has an ideal design
// on [0, 1] and answers incumbent-versus-challenger comparisons with probit
// noise. After every answer, the person's ideal moves a fraction eta toward
// the design they chose, a one-dimensional version of the assumption in Dean
// and Morgenstern (2022) that preferences move toward content that is
// consumed and liked. The top panel shows one session: the utility before and
// after, where the system started, and where it ended. The bottom panel
// averages many sessions and scores the same recommendations two ways:
// against the person's preferences at that moment (what a benchmark reports)
// and against the preferences the person came in with. With eta = 0 the two
// coincide; as eta grows, the benchmark score improves even though the
// recommendation drifts away from what the person originally wanted.

import { defineFigure, type State } from "./types.ts";
import { arrowMarker, el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid } from "./lib/scale.ts";
import { memo, rng } from "./lib/random.ts";
import { Phi } from "./lib/stats.ts";
import { clip, curve, frame, frameAxes, legend } from "./lib/plot.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    design: "design x",
    utility: "utility",
    queries: "comparisons answered",
    regret: "regret",
    before: "preferences before the session",
    after: "after {t} comparisons",
    pick: "system's final pick",
    start: "start",
    personMoved: "person moved {d}",
    systemMoved: "system moved {d}",
    stayed: "barely moved",
    fixedPerson: "preferences fixed",
    now: "judged by the person's preferences now",
    then: "judged by the preferences they started with",
    fixedRef: "same optimizer, fixed preferences",
    oneSession: "one simulated session",
    average: "mean of {r} sessions",
    readout: "After {t} comparisons: regret {now} by current preferences, {then} by starting preferences.",
    readout2: "On average the person's ideal moved {p}; the system's pick moved {s}.",
    describe: "Pull η = {eta}, challengers drawn {pol}, system starting at {x0}. Averaged over {r} simulated sessions of {t} comparisons, the final regret is {now} judged by the person's current preferences and {then} judged by the preferences they started with; with fixed preferences the same optimizer reaches {fix}. The person's ideal moved {p} on average, the system's pick {s}.",
    near: "near the incumbent",
    anywhere: "anywhere",
    personStayed: "person {stayed}",
    narrow1: "After {t} comparisons, regret is",
    narrow2: "{now} by current preferences,",
    narrow3: "{then} by starting preferences.",
    narrow4: "Person moved {p}; system {s}.",
  },
  zh: {
    design: "设计 x",
    utility: "效用",
    queries: "已回答的比较",
    regret: "遗憾",
    before: "会话前的偏好",
    after: "{t} 次比较之后",
    pick: "系统的最终选择",
    start: "起点",
    personMoved: "人移动了 {d}",
    systemMoved: "系统移动了 {d}",
    stayed: "几乎没动",
    fixedPerson: "偏好固定",
    now: "按此刻的偏好评判",
    then: "按最初的偏好评判",
    fixedRef: "同一优化器，偏好固定",
    oneSession: "一次模拟会话",
    average: "{r} 次会话的平均",
    readout: "{t} 次比较之后：按当前偏好的遗憾为 {now}，按最初偏好的遗憾为 {then}。",
    readout2: "平均而言，人的理想点移动了 {p}；系统的选择移动了 {s}。",
    describe: "每次选择的牵引 η = {eta}，挑战者{pol}抽取，系统从 {x0} 出发。在 {r} 次各含 {t} 次比较的模拟会话上平均，按人当前的偏好评判，最终遗憾为 {now}；按人最初的偏好评判，为 {then}；偏好固定时，同一优化器达到 {fix}。平均而言，人的理想点移动了 {p}，系统的选择移动了 {s}。",
    near: "在当前最优点附近",
    anywhere: "在任意位置",
    personStayed: "人{stayed}",
    narrow1: "{t} 次比较之后，遗憾为",
    narrow2: "按当前偏好：{now}，",
    narrow3: "按最初偏好：{then}。",
    narrow4: "人移动了 {p}；系统移动了 {s}。",
  },
};

const params = {
  eta: { kind: "range", label: { en: "Pull per choice η", zh: "每次选择的牵引 η" }, min: 0, max: 0.3, default: 0.1, step: 0.01 },
  policy: {
    kind: "choice", label: { en: "Challenger", zh: "挑战者" }, control: "buttons",
    options: [{ value: "near", label: { en: "Nearby", zh: "附近" } }, { value: "anywhere", label: { en: "Anywhere", zh: "任意位置" } }],
    default: "near",
  },
  start: { kind: "range", label: { en: "System start", zh: "系统起点" }, min: 0, max: 1, default: 0.15, step: 0.01 },
  seed: { kind: "range", label: { en: "Session", zh: "会话" }, min: 1, max: 9999, default: 3, step: 1, control: false },
} as const;

type P = { eta: number; policy: "near" | "anywhere"; start: number; seed: number };

const THETA0 = 0.75; // the person's ideal design when the session begins
const WIDTH = 0.2; // width of the utility bump
const SIGMA = 0.05; // answer noise on the utility scale (probit)
const STEP = 0.1; // half-width of a "near" challenger's neighborhood
const T = 40; // comparisons per session
const RUNS = 200; // sessions averaged in the lower panel
const XS = grid(0, 1, 161);

const u = (x: number, th: number) => Math.exp(-((x - th) ** 2) / (2 * WIDTH * WIDTH));

interface Session { th: number[]; x: number[] }

// One session of the incumbent-versus-challenger loop. The winner of each
// comparison becomes the incumbent, and the person's ideal moves toward it.
function session(eta: number, policy: string, x0: number, seed: number): Session {
  const r = rng(seed);
  let x = x0, th = THETA0;
  const ths = [th], xs = [x];
  for (let t = 0; t < T; t++) {
    let c = policy === "near" ? x + (2 * r() - 1) * STEP : r();
    c = Math.min(1, Math.max(0, c));
    const pc = Phi((u(c, th) - u(x, th)) / (Math.SQRT2 * SIGMA));
    const win = r() < pc ? c : x;
    x = win;
    th += eta * (win - th);
    ths.push(th);
    xs.push(x);
  }
  return { th: ths, x: xs };
}

interface Avg { now: number[]; then: number[]; moved: number; sysMoved: number }

const average = memo((eta: number, policy: string, x0: number): Avg => {
  const now = new Array(T + 1).fill(0), then = new Array(T + 1).fill(0);
  let moved = 0, sysMoved = 0;
  for (let s = 0; s < RUNS; s++) {
    const ss = session(eta, policy, x0, 7919 * s + 101);
    for (let t = 0; t <= T; t++) {
      now[t] += (1 - u(ss.x[t], ss.th[t])) / RUNS;
      then[t] += (1 - u(ss.x[t], THETA0)) / RUNS;
    }
    moved += Math.abs(ss.th[T] - THETA0) / RUNS;
    sysMoved += Math.abs(ss.x[T] - x0) / RUNS;
  }
  return { now, then, moved, sysMoved };
}, 48);

// Two decimals, or three for small regrets so a near-zero value does not read as 0.00.
const fr = (v: number) => (v < 0.095 ? fixed(v, 3) : fixed(v, 2));

function compute(p: P) {
  const eta = Math.round(p.eta * 100) / 100;
  const avg = average(eta, p.policy, p.start);
  const ref = average(0, p.policy, p.start);
  const one = session(eta, p.policy, p.start, p.seed * 104729 + 17);
  return { avg, ref, one };
}

function describe(st: State<P>): string {
  const p = st.p;
  const { avg, ref } = compute(p);
  const L = labels[st.lang ?? "en"];
  return tpl(L.describe, {
    eta: fixed(p.eta, 2), pol: p.policy === "near" ? L.near : L.anywhere, x0: fixed(p.start, 2), r: RUNS, t: T,
    now: fr(avg.now[T]), then: fr(avg.then[T]), fix: fr(ref.now[T]), p: fixed(avg.moved, 2), s: fixed(avg.sysMoved, 2),
  });
}

function star(cx: number, cy: number, r: number): string {
  let d = "";
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    const rr = i % 2 ? r * 0.45 : r;
    d += `${i ? "L" : "M"}${(cx + rr * Math.cos(a)).toFixed(1)},${(cy + rr * Math.sin(a)).toFixed(1)}`;
  }
  return d + "Z";
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const { avg, ref, one } = compute(p);
  const thT = one.th[T], xT = one.x[T];
  const parts: string[] = [];
  const aidP = `${st.uid}-arr-p`, aidS = `${st.uid}-arr-s`;
  const cidA = `${st.uid}-clip-a`, cidB = `${st.uid}-clip-b`;

  // Panel A: one session.
  parts.push(text(narrow ? 8 : 40, 14, L.oneSession, { "font-size": TYPE.label, class: "fig-t-strong" }));
  const lgA = legend(narrow ? 8 : 40, 34, st.w - 16, [
    { kind: "dash", color: C.truth, label: L.before },
    { kind: "line", color: C.truth, label: tpl(L.after, { t: T }) },
    { kind: "dot", color: C.acq, label: L.pick },
  ]);
  parts.push(lgA.svg);
  const topA = 34 + lgA.height + 4;
  const fa = frame({ w: st.w, top: topA, height: narrow ? 150 : 170, yDomain: [0, 1.32], yTitle: L.utility });
  parts.push(el("defs", {}, clip(fa, cidA), arrowMarker(aidP, C.truth), arrowMarker(aidS, C.acq)));
  parts.push(frameAxes(fa, { xTitle: L.design, yTitle: L.utility, yCount: 3 }));
  parts.push(g({ "clip-path": `url(#${cidA})` },
    curve(fa, XS, XS.map((x) => u(x, THETA0)), { stroke: C.truth, "stroke-width": 1.6, "stroke-dasharray": "5 4" }),
    curve(fa, XS, XS.map((x) => u(x, thT)), { stroke: C.truth, "stroke-width": 2.4 }),
  ));
  // The person's move, drawn above the peaks.
  const ay = fa.y(1.14);
  const dP = Math.abs(thT - THETA0);
  if (dP > 0.025) {
    parts.push(el("line", { x1: fa.x(THETA0), x2: fa.x(thT), y1: ay, y2: ay, stroke: C.truth, "stroke-width": 1.6, "marker-end": `url(#${aidP})` }));
  }
  const pLabel = dP > 0.025 ? tpl(L.personMoved, { d: fixed(dP, 2) }) : p.eta === 0 ? L.fixedPerson : tpl(L.personStayed, { stayed: L.stayed });
  const pMid = dP > 0.025 ? (fa.x(THETA0) + fa.x(thT)) / 2 : fa.x(THETA0);
  parts.push(text(clampX(pMid, st.w, pLabel), ay - 6, pLabel, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));
  // The system's move, drawn near the baseline.
  const sy = fa.y(0.16);
  const dS = Math.abs(xT - p.start);
  parts.push(el("circle", { cx: fa.x(p.start), cy: fa.y(0), r: 4.5, fill: C.paper, stroke: C.acq, "stroke-width": 2 }));
  // The arrow starts at the hollow circle, so the circle needs its own label
  // only when the system did not move.
  if (dS <= 0.025) parts.push(text(fa.x(p.start), fa.y(0) - 8, L.start, { "text-anchor": p.start < 0.08 ? "start" : p.start > 0.92 ? "end" : "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));
  if (dS > 0.025) {
    parts.push(el("line", { x1: fa.x(p.start), x2: fa.x(p.start), y1: fa.y(0) - 5, y2: sy, stroke: C.acq, "stroke-width": 1, "stroke-dasharray": "2 2" }));
    parts.push(el("line", { x1: fa.x(p.start), x2: fa.x(xT), y1: sy, y2: sy, stroke: C.acq, "stroke-width": 1.6, "marker-end": `url(#${aidS})` }));
    const sLabel = tpl(L.systemMoved, { d: fixed(dS, 2) });
    parts.push(text(clampX((fa.x(p.start) + fa.x(xT)) / 2, st.w, sLabel), sy - 6, sLabel, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));
  }
  parts.push(el("path", { d: star(fa.x(xT), fa.y(u(xT, thT)), 8), fill: C.acq, stroke: C.paper, "stroke-width": 1.2 }));

  // Panel B: the two scoreboards, averaged over sessions.
  const tB = fa.bottom + 52;
  parts.push(text(narrow ? 8 : 40, tB, tpl(L.average, { r: RUNS }), { "font-size": TYPE.label, class: "fig-t-strong" }));
  const lgB = legend(narrow ? 8 : 40, tB + 20, st.w - 16, [
    { kind: "line", color: C.c1, label: L.now },
    { kind: "line", color: C.c5, label: L.then },
    { kind: "dash", color: C.ink3, label: L.fixedRef },
  ]);
  parts.push(lgB.svg);
  const topB = tB + 20 + lgB.height + 4;
  const ts = grid(0, T, T + 1);
  const fb = frame({ w: st.w, top: topB, height: narrow ? 130 : 150, yDomain: [0, 1], xDomain: [0, T], yTitle: L.regret });
  parts.push(el("defs", {}, clip(fb, cidB)));
  parts.push(frameAxes(fb, { xTitle: L.queries, yTitle: L.regret, yCount: 4 }));
  parts.push(g({ "clip-path": `url(#${cidB})` },
    curve(fb, ts, ref.now, { stroke: C.ink3, "stroke-width": 1.5, "stroke-dasharray": "5 4" }),
    curve(fb, ts, avg.then, { stroke: C.c5, "stroke-width": 2.2 }),
    curve(fb, ts, avg.now, { stroke: C.c1, "stroke-width": 2.2 }),
  ));
  const ry = fb.bottom + 50;
  const r1 = tpl(L.readout, { t: T, now: fr(avg.now[T]), then: fr(avg.then[T]) });
  const r2 = tpl(L.readout2, { p: fixed(avg.moved, 2), s: fixed(avg.sysMoved, 2) });
  if (narrow) {
    parts.push(text(8, ry, tpl(L.narrow1, { t: T }), { "font-size": TYPE.small, class: "fig-t-muted" }));
    parts.push(text(8, ry + 15, tpl(L.narrow2, { now: fr(avg.now[T]) }), { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
    parts.push(text(8, ry + 30, tpl(L.narrow3, { then: fr(avg.then[T]) }), { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
    parts.push(text(8, ry + 45, tpl(L.narrow4, { p: fixed(avg.moved, 2), s: fixed(avg.sysMoved, 2) }), { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
  } else {
    parts.push(text(fb.left, ry, r1, { "font-size": TYPE.small, class: "fig-t-muted" }));
    parts.push(text(fb.left, ry + 16, r2, { "font-size": TYPE.small, class: "fig-t-muted" }));
  }
  const H = ry + (narrow ? 52 : 24);
  return svg(st.w, H, describe(st), ...parts);
}

// Keep a centered label inside the figure.
function clampX(x: number, w: number, s: string): number {
  const half = labelWidth(s, 3.1, 6) + 4;
  return Math.max(half + 4, Math.min(w - half - 4, x));
}

export default defineFigure({
  name: "soc-moving-target",
  title: { en: "Who moved: the person or the system?", zh: "谁在移动：人还是系统？" },
  labels,
  params,
  hint: { en: "Drag the pull η to make the simulated person more or less malleable. The lower panel averages 200 sessions.", zh: "拖动牵引 η，让模拟的人更容易或更不容易被改变。下方面板是 200 次会话的平均。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "Another session", zh: "再来一次会话" }, primary: true, run: (p) => ({ ...p, seed: (Number(p.seed) % 9999) + 1 }) },
    { label: { en: "Fixed preferences", zh: "偏好固定" }, run: (p) => ({ ...p, eta: 0 }), enabled: (p) => p.eta !== 0 },
  ],
});
