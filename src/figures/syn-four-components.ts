// What one comparison measures, as a simulation of a model (not data). A
// simulated person answers 30 comparisons between designs on a line. Each
// answer mixes four components whose sizes the reader sets:
//
//   stable preference     a fixed utility with two peaks, scaled by `stable`
//   structured noise      probit noise whose scale grows when the two designs
//                         are similar: sd = 0.15 + noise * exp(-d^2 / 0.02)
//   query-induced change  every choice raises the chosen design by `induced`
//                         and lowers the rejected one by half that, spread
//                         over nearby designs; most of it fades within a few
//                         answers (by 15% per answer), a share `lasting` stays
//   incomplete answers    a share of comparisons with no preference behind
//                         them: a coin flip under forced choice, or "can't
//                         compare" when that answer is offered
//
// A Gaussian process preference model (RBF kernel, probit, Laplace) fits the
// answers, and the next pair comes from EUBO or uniformly at random. At the
// end of the session and a week later, the person is asked again, without the
// system, whether they prefer the final design to each design they rejected
// in the first 10 answers. The retest panel averages 24 simulated people, each
// run under both query orders with the same random draws; running one person
// under both orders is a luxury of simulation, since a real experiment
// randomizes people between orders and compares groups.

import { defineFigure, tr, type Lang, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { fitPreference, kernel, pointIndex, predictPreference, type Duel, type PrefFit, type X } from "./lib/gp.ts";
import { argmax, expectedMax2 } from "./lib/acq.ts";
import { memo, rng } from "./lib/random.ts";
import { Phi } from "./lib/stats.ts";
import { clip, curve, frame, legend } from "./lib/plot.ts";
import { tpl } from "./lib/format.ts";

const labels = {
  en: {
    stable: "stable preference",
    now: "the person now",
    fit: "model's fitted utility",
    u: "utility",
    x: "design",
    eubo: "Acquisition order (EUBO)",
    random: "Randomized order",
    euboShort: "Acquisition (EUBO)",
    randomShort: "Randomized",
    answer: "answer {t} of {n}",
    ended: "session over: {n} answers",
    week: "one week later",
    early: "early favorite",
    final: "final design",
    rejected: "rejected early",
    raster: "Answers, first at top",
    first: "first {k}",
    key: "● chosen · ○ rejected · grey: no preference behind it",
    keyCc: "dashed: “can't compare” ({k} of {n})",
    retest: "Retest without the system: does the person still prefer the final design?",
    retest1: "Retest without the system:",
    retest2: "still prefers the final design?",
    retestSub: "over designs rejected in the first {k} answers · bar: mean of {g} simulated people · dot: the person above",
    retestSubShort: "bar: mean of {g} people · dot: person above",
    atEnd: "at session end",
    atWeek: "a week later",
    atEndShort: "session end",
    atWeekShort: "week later",
    pending: "after answer {n}",
    waiting: "after a week",
    chance: "chance",
    k0: "Before the first answer",
    k10: "Answer {k}: the early favorite",
    kEnd: "Session ends: first retest",
    kWeek: "One week later: second retest",
    phaseWeek: "a week after the session",
    phaseEnd: "at the end of the session",
    phaseAfter: "after {t} of {n} answers",
    unitWeek: "+1 week",
    describe: "Simulation of a model, not data. {order}, {phase}. Across {g} simulated people, the final design is still preferred over early-rejected designs {ae}% of the time at the end of the session and {aw}% a week later under acquisition order, against {re}% and {rw}% under randomized order.",
  },
  zh: {
    stable: "稳定偏好",
    now: "此刻的效用",
    fit: "模型拟合的效用",
    u: "效用",
    x: "设计",
    eubo: "采集顺序（EUBO）",
    random: "随机顺序",
    euboShort: "采集（EUBO）",
    randomShort: "随机",
    answer: "第 {t} 个回答（共 {n} 个）",
    ended: "会话结束：共 {n} 个回答",
    week: "一周后",
    early: "早期首选",
    final: "最终设计",
    rejected: "早期被拒",
    raster: "回答（最早的在上）",
    first: "前 {k}",
    key: "● 选中 · ○ 被拒 · 灰色：背后没有偏好",
    keyCc: "虚线：“无法比较”（{n} 个中有 {k} 个）",
    retest: "不借助系统的重测：此人是否仍更喜欢最终设计？",
    retest1: "不借助系统的重测：",
    retest2: "仍更喜欢最终设计吗？",
    retestSub: "对比前 {k} 个回答中被拒的设计 · 条：{g} 个模拟人的均值 · 点：上方这个人",
    retestSubShort: "条：{g} 人均值 · 点：上方这个人",
    atEnd: "会话结束时",
    atWeek: "一周后",
    atEndShort: "会话结束",
    atWeekShort: "一周后",
    pending: "第 {n} 个回答之后",
    waiting: "一周之后",
    chance: "随机水平",
    k0: "第一个回答之前",
    k10: "第 {k} 个回答：早期首选",
    kEnd: "会话结束：第一次重测",
    kWeek: "一周后：第二次重测",
    phaseWeek: "会话结束一周后",
    phaseEnd: "会话结束时",
    phaseAfter: "第 {t} 个回答之后（共 {n} 个）",
    unitWeek: "+1 周",
    describe: "这是模型的模拟，不是数据。{order}，{phase}。在 {g} 个模拟人中，采集顺序下最终设计在会话结束时有 {ae}% 的时候仍比早期被拒的设计更受偏好，一周后为 {aw}%；随机顺序下为 {re}% 与 {rw}%。",
  },
};

const ORDERS = [
  { value: "eubo", label: { en: "Acquisition order (EUBO)", zh: "采集顺序（EUBO）" } },
  { value: "random", label: { en: "Randomized order", zh: "随机顺序" } },
] as const;

const params = {
  order: { kind: "choice", label: { en: "Session shown", zh: "显示的会话" }, options: ORDERS, default: "eubo", control: "buttons" },
  stable: { kind: "range", label: { en: "Stable preference", zh: "稳定偏好" }, min: 0, max: 1, default: 0.4, step: 0.05 },
  noise: { kind: "range", label: { en: "Structured noise", zh: "结构化噪声" }, min: 0, max: 1, default: 0.3, step: 0.05 },
  induced: { kind: "range", label: { en: "Induced change", zh: "查询引起的改变" }, min: 0, max: 0.12, default: 0.05, step: 0.01 },
  incomplete: { kind: "range", label: { en: "Incomplete answers", zh: "不完备回答" }, min: 0, max: 0.6, default: 0.1, step: 0.05 },
  lasting: { kind: "range", label: { en: "Lasting share", zh: "持久部分" }, min: 0, max: 1, default: 0.1, step: 0.05 },
  cantCompare: { kind: "toggle", label: { en: "Offer “can't compare”", zh: "提供“无法比较”" }, default: false },
  person: { kind: "range", label: { en: "Person shown", zh: "显示的人" }, min: 1, max: 24, default: 1, step: 1, control: false },
} as const;

type Order = "eubo" | "random";
type P = { order: Order; stable: number; noise: number; induced: number; incomplete: number; lasting: number; cantCompare: boolean; person: number };
interface Q { A: number; s: number; c: number; rho: number; pi: number; cc: boolean }

const N = 30; // answers per session
const EARLY = 10; // the early phase: early favorite and early-rejected designs
const GROUP = 24; // simulated people per query order
const CAND = grid(0.02, 0.98, 21); // designs the system can show
const XS = grid(0, 1, 121); // display grid
const K = kernel("rbf", 0.12, 0.36); // prior sd 0.6, on the scale of the person's utilities
const SIGMA_M = 0.15; // the model's fixed noise scale: the person's base level
const BASE = 0.15; // noise sd for very different designs
const FADE = 0.85; // transient part of a choice's effect, per later answer
const SPREAD = 0.04; // how far a choice's effect spreads to similar designs

// The person's stable utility: two peaks of heights 1 and 0.8 at positions
// that differ from person to person.
const stableOf = memo((person: number) => {
  const r = rng(person * 101 + 3);
  const m1 = 0.62 + 0.2 * r(), m2 = 0.18 + 0.16 * r();
  const swap = r() < 0.5;
  const hi = swap ? m2 : m1, lo = swap ? m1 : m2;
  return (x: number) => Math.exp(-((x - hi) ** 2) / (2 * 0.1 ** 2)) + 0.8 * Math.exp(-((x - lo) ** 2) / (2 * 0.09 ** 2));
}, 64);

const bump = (x: number, w: number) => Math.exp(-((x - w) ** 2) / (2 * SPREAD * SPREAD));
const noiseSd = (s: number, a: number, b: number) => BASE + s * Math.exp(-((a - b) ** 2) / (2 * 0.1 * 0.1));

interface Answer { a: number; b: number; ans: "a" | "b" | "cc"; noPref: boolean }
interface Choice { w: number; l: number; at: number }
interface Run {
  answers: Answer[];
  choices: Choice[];
  early: number;
  fin: number;
  rejected: number[];
  end: number; // P(final design preferred over an early-rejected one), at session end
  week: number; // the same a week later
  frames?: Array<{ mean: number[] }>; // the model after t answers, t = 0..N
}

// Change caused by the queries so far, at design x. After `t` answers each
// earlier choice keeps its lasting share pi plus a transient part that fades;
// a week later only the lasting share remains.
function inducedAt(x: number, choices: Choice[], q: Q, t: number, week: boolean): number {
  let s = 0;
  for (const ch of choices) {
    if (ch.at >= t) break;
    const wgt = week ? q.pi : q.pi + (1 - q.pi) * FADE ** (t - 1 - ch.at);
    s += wgt * q.c * (bump(x, ch.w) - 0.5 * bump(x, ch.l));
  }
  return s;
}

function fitAnswers(answers: Answer[]): PrefFit {
  const xs: X[] = [];
  const duels: Duel[] = [];
  for (const a of answers) {
    if (a.ans === "cc") continue;
    const w = a.ans === "a" ? a.a : a.b, l = a.ans === "a" ? a.b : a.a;
    duels.push({ winner: pointIndex(xs, w), loser: pointIndex(xs, l) });
  }
  return fitPreference(K, xs, duels, SIGMA_M);
}

// EUBO over the candidate pairs not shown before. Without that rule the
// Laplace posterior, whose curvature vanishes once the model is sure of an
// answer, can ask the same pair over and over.
function nextEubo(fit: PrefFit, asked: Set<string>): [number, number] {
  const cp = predictPreference(fit, CAND, true);
  let bi = 0, bj = 1, bv = -Infinity;
  for (let i = 0; i < CAND.length; i++) for (let j = i + 1; j < CAND.length; j++) {
    if (asked.has(`${i},${j}`)) continue;
    const v = expectedMax2(cp.mean[i], cp.cov![i][i], cp.mean[j], cp.cov![j][j], cp.cov![i][j]);
    if (v > bv) { bv = v; bi = i; bj = j; }
  }
  return [CAND[bi], CAND[bj]];
}

function simulate(q: Q, person: number, order: Order, display: boolean): Run {
  const u0 = stableOf(person);
  // The same draws for both orders: the answer at step t uses the same two
  // uniforms whichever pair is shown, so the orders differ only by the queries.
  const rAns = rng(person * 7919 + 17);
  const rPair = rng(person * 104729 + 3);
  const answers: Answer[] = [];
  const choices: Choice[] = [];
  const frames: Array<{ mean: number[] }> = [];
  const asked = new Set<string>();
  let early = NaN;
  for (let t = 0; t < N; t++) {
    const needFit = order === "eubo" || display || t === EARLY;
    const fit = needFit ? fitAnswers(answers) : undefined;
    if (fit && (display || t === EARLY)) {
      const post = predictPreference(fit, XS);
      if (display) frames.push({ mean: post.mean });
      if (t === EARLY) early = XS[argmax(post.mean)];
    }
    let pair: [number, number];
    if (t === 0) pair = [CAND[6], CAND[14]];
    else if (order === "random") {
      const i = Math.floor(rPair() * CAND.length);
      let j = Math.floor(rPair() * (CAND.length - 1));
      if (j >= i) j++;
      pair = [CAND[i], CAND[j]];
    } else pair = nextEubo(fit!, asked);
    const [a, b] = pair;
    const ia = CAND.indexOf(a), ib = CAND.indexOf(b);
    asked.add(`${Math.min(ia, ib)},${Math.max(ia, ib)}`);
    const r1 = rAns(), r2 = rAns();
    const noPref = r1 < q.rho;
    let ans: Answer["ans"];
    if (noPref) ans = q.cc ? "cc" : r2 < 0.5 ? "a" : "b";
    else {
      const ua = q.A * u0(a) + inducedAt(a, choices, q, t, false);
      const ub = q.A * u0(b) + inducedAt(b, choices, q, t, false);
      ans = r2 < Phi((ua - ub) / (Math.SQRT2 * noiseSd(q.s, a, b))) ? "a" : "b";
    }
    answers.push({ a, b, ans, noPref });
    if (ans !== "cc") choices.push({ w: ans === "a" ? a : b, l: ans === "a" ? b : a, at: t });
  }
  const post = predictPreference(fitAnswers(answers), XS);
  if (display) frames.push({ mean: post.mean });
  const fin = XS[argmax(post.mean)];
  // The retest opponents: designs rejected in the early phase, not too close
  // to the final design; if there are none, any other early loser.
  const losers = answers.slice(0, EARLY).filter((x) => x.ans !== "cc").map((x) => (x.ans === "a" ? x.b : x.a));
  let rejected = [...new Set(losers)].filter((r) => Math.abs(r - fin) > 0.1);
  if (!rejected.length) rejected = [...new Set(losers)].filter((r) => Math.abs(r - fin) > 1e-6);
  if (!rejected.length) rejected = [CAND[0], CAND[CAND.length - 1]].filter((r) => Math.abs(r - fin) > 0.1);
  const agree = (week: boolean) => {
    if (!rejected.length) return 0.5;
    let s = 0;
    for (const r of rejected) {
      const d = q.A * (u0(fin) - u0(r)) + inducedAt(fin, choices, q, N, week) - inducedAt(r, choices, q, N, week);
      const pr = Phi(d / (Math.SQRT2 * noiseSd(q.s, fin, r)));
      s += q.cc ? pr : (1 - q.rho) * pr + q.rho * 0.5;
    }
    return s / rejected.length;
  };
  return { answers, choices, early, fin, rejected, end: agree(false), week: agree(true), frames: display ? frames : undefined };
}

const qOf = (p: P): Q => ({ A: p.stable, s: p.noise, c: p.induced, rho: p.incomplete, pi: p.lasting, cc: p.cantCompare });

// The retest for a group of simulated people under both orders.
const group = memo((A: number, s: number, c: number, rho: number, pi: number, cc: boolean) => {
  const q = { A, s, c, rho, pi, cc };
  const out = { eubo: [] as Array<{ end: number; week: number }>, random: [] as Array<{ end: number; week: number }> };
  for (let k = 1; k <= GROUP; k++) for (const o of ["eubo", "random"] as const) {
    const r = simulate(q, k, o, false);
    out[o].push({ end: r.end, week: r.week });
  }
  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
  return {
    people: out,
    eubo: { end: mean(out.eubo.map((r) => r.end)), week: mean(out.eubo.map((r) => r.week)) },
    random: { end: mean(out.random.map((r) => r.end)), week: mean(out.random.map((r) => r.week)) },
  };
}, 12);

const shown = memo((A: number, s: number, c: number, rho: number, pi: number, cc: boolean, person: number, order: Order) =>
  simulate({ A, s, c, rho, pi, cc }, person, order, true), 12);

const groupOf = (p: P) => group(p.stable, p.noise, p.induced, p.incomplete, p.lasting, p.cantCompare);
const shownOf = (p: P) => shown(p.stable, p.noise, p.induced, p.incomplete, p.lasting, p.cantCompare, p.person, p.order);

const pc = (v: number) => `${Math.round(v * 100)}%`;
const centered = (ys: number[]) => { const m = ys.reduce((a, b) => a + b, 0) / ys.length; return ys.map((y) => y - m); };

function phaseText(t: number, lang: Lang): string {
  const L = labels[lang];
  return t > N ? L.week : t === N ? tpl(L.ended, { n: N }) : tpl(L.answer, { t, n: N });
}

function describe(st: State<P>): string {
  const p = st.p;
  const G = groupOf(p);
  const t = Math.round(st.t);
  const lang = st.lang ?? "en";
  const L = labels[lang];
  return tpl(L.describe, {
    order: tr(ORDERS.find((o) => o.value === p.order)!.label, lang),
    phase: t > N ? L.phaseWeek : t === N ? L.phaseEnd : tpl(L.phaseAfter, { t, n: N }),
    g: GROUP, ae: Math.round(G.eubo.end * 100), aw: Math.round(G.eubo.week * 100), re: Math.round(G.random.end * 100), rw: Math.round(G.random.week * 100),
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
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 480;
  const q = qOf(p);
  const t = Math.max(0, Math.min(N + 1, Math.round(st.t)));
  const run = shownOf(p);
  const G = groupOf(p);
  const u0 = stableOf(p.person);
  const parts: string[] = [];

  // Legend.
  const lg = legend(narrow ? 8 : 40, 14, st.w - 16, [
    { kind: "dash", color: C.truth, label: L.stable },
    { kind: "line", color: C.c5, label: L.now },
    { kind: "line", color: C.model, label: L.fit },
  ]);
  parts.push(lg.svg);

  // Panel A: utilities. Every curve is centered, because comparisons fix a
  // utility only up to an added constant.
  const fr = run.frames![Math.min(t, N)];
  const stableC = centered(XS.map((x) => q.A * u0(x)));
  const nowC = centered(XS.map((x) => q.A * u0(x) + inducedAt(x, run.choices, q, Math.min(t, N), t > N)));
  const meanRaw = fr.mean;
  const mOff = meanRaw.reduce((a, b) => a + b, 0) / meanRaw.length;
  const meanC = meanRaw.map((v) => v - mOff);
  // A y range that holds still while the timeline plays: the largest curve
  // over the whole session.
  let ymax = 0.25;
  const fN = run.frames![N];
  const mN = fN.mean.reduce((a, b) => a + b, 0) / fN.mean.length;
  for (const v of fN.mean) ymax = Math.max(ymax, Math.abs(v - mN));
  for (let tt = 0; tt <= N + 1; tt += 3) for (const v of centered(XS.map((x) => q.A * u0(x) + inducedAt(x, run.choices, q, Math.min(tt, N), tt > N)))) ymax = Math.max(ymax, Math.abs(v));
  for (const tt of [N, N + 1]) for (const v of centered(XS.map((x) => q.A * u0(x) + inducedAt(x, run.choices, q, N, tt > N)))) ymax = Math.max(ymax, Math.abs(v));
  ymax = Math.ceil(ymax * 1.3 * 10) / 10;
  const titleY = 14 + lg.height + 10;
  const fa = frame({ w: st.w, top: titleY + 12, height: narrow ? 140 : 170, yDomain: [-ymax, ymax], yTitle: L.u });
  const orderName = p.order === "eubo" ? L.eubo : L.random;
  parts.push(text(fa.left, titleY, `${orderName} · ${phaseText(t, lang)}`, { "font-size": TYPE.label, class: "fig-t-strong" }));
  const cid = `${st.uid}-a`;
  parts.push(el("defs", {}, clip(fa, cid)));
  parts.push(axis({ scale: fa.y, orient: "left", at: fa.left, span: [fa.left, fa.right], title: L.u, count: 4 }));
  parts.push(el("line", { x1: fa.left, x2: fa.right, y1: fa.bottom, y2: fa.bottom, stroke: C.rule }));
  // No credible band: a preference model pins utilities down only relative
  // to each other, so the marginal band of f(x) mostly shows the unknown
  // overall level, not what the answers have taught the model.
  parts.push(g({ "clip-path": `url(#${cid})` },
    curve(fa, XS, stableC, { stroke: C.truth, "stroke-dasharray": "5 4", "stroke-width": 1.6 }),
    curve(fa, XS, nowC, { stroke: C.c5, "stroke-width": 1.6 }),
    curve(fa, XS, meanC),
  ));
  // The pair being asked now.
  if (t < N) {
    const a = run.answers[t];
    for (const [k, x] of [["A", a.a], ["B", a.b]] as const) {
      parts.push(el("line", { x1: fa.x(x), x2: fa.x(x), y1: fa.top, y2: fa.bottom, stroke: C.acq, "stroke-width": 1.4, "stroke-dasharray": "4 3" }),
        // Inside the plot's top edge: above it they would collide with the panel title.
        text(fa.x(x) + 4, fa.top + 11, k, { "font-size": TYPE.small, class: "fig-t-strong fig-t-halo" }));
    }
  }
  // The early favorite and, once the session is over, the final design and
  // the early-rejected designs the retest uses.
  const iAt = (x: number) => Math.round(x * (XS.length - 1));
  if (t >= EARLY && Number.isFinite(run.early)) {
    const x = fa.x(run.early);
    parts.push(el("path", { d: `M${x},${fa.bottom - 1}l-5,8h10z`, fill: C.paper, stroke: C.ink2, "stroke-width": 1.4 }));
  }
  if (t >= N) {
    for (const r of run.rejected) {
      const x = fa.x(r), y = fa.bottom + 4;
      parts.push(el("path", { d: `M${x - 3.5},${y - 3.5 + 4}l7,7M${x + 3.5},${y - 3.5 + 4}l-7,7`, stroke: C.ink3, "stroke-width": 1.6 }));
    }
    const yf = fa.y(Math.max(-ymax, Math.min(ymax, meanC[iAt(run.fin)])));
    parts.push(el("path", { d: star(fa.x(run.fin), yf, 7.5), fill: C.c4, stroke: C.paper, "stroke-width": 1.2 }));
  }
  // Marker key under panel A.
  const mkY = fa.bottom + 24;
  const mk: string[] = [];
  let mx = fa.left, my = mkY;
  const keyItem = (sym: (x: number, y: number) => string, label: string) => {
    const wItem = 14 + labelWidth(label, 6.1, 12) + 16;
    if (mx > fa.left && mx + wItem > fa.right) { mx = fa.left; my += 17; }
    mk.push(sym(mx, my), text(mx + 14, my + 4, label, { "font-size": TYPE.small, class: "fig-t-muted" }));
    mx += wItem;
  };
  keyItem((x, y) => el("path", { d: `M${x},${y + 4}l5,-8l5,8z`, fill: C.paper, stroke: C.ink2, "stroke-width": 1.4 }), L.early);
  if (t >= N) {
    keyItem((x, y) => el("path", { d: star(x + 5, y, 6), fill: C.c4, stroke: C.paper, "stroke-width": 1 }), L.final);
    keyItem((x, y) => el("path", { d: `M${x + 1.5},${y - 3.5}l7,7M${x + 8.5},${y - 3.5}l-7,7`, stroke: C.ink3, "stroke-width": 1.6 }), L.rejected);
  }
  parts.push(g({}, ...mk));

  // Panel B: the answers, one row per comparison.
  const rowH = narrow ? 3.8 : 4.4;
  const rbTitle = my + 26;
  const rTop = rbTitle + 8;
  const rBot = rTop + N * rowH;
  const nCc = run.answers.filter((a) => a.ans === "cc").length;
  parts.push(text(fa.left, rbTitle, L.raster, { "font-size": TYPE.small, class: "fig-t-strong" }));
  if (!narrow) parts.push(text(fa.right, rbTitle, p.cantCompare ? `${L.key} · ${tpl(L.keyCc, { k: nCc, n: N })}` : L.key, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted" }));
  parts.push(el("rect", { x: fa.left, y: rTop, width: fa.right - fa.left, height: EARLY * rowH, fill: C.panel }));
  parts.push(text(fa.left - 4, rTop + EARLY * rowH / 2 + 4, tpl(L.first, { k: EARLY }), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-faint" }));
  const shownRows = Math.min(t, N);
  for (let i = 0; i < shownRows; i++) {
    const a = run.answers[i];
    const y = rTop + (i + 0.5) * rowH;
    const muted = a.noPref;
    const col = muted ? C.ink3 : C.ink;
    parts.push(el("line", { x1: fa.x(a.a), x2: fa.x(a.b), y1: y, y2: y, stroke: muted ? C.ink3 : C.rule, "stroke-width": 1, "stroke-dasharray": a.ans === "cc" ? "2 2" : undefined }));
    if (a.ans === "cc") {
      for (const x of [a.a, a.b]) parts.push(el("circle", { cx: fa.x(x), cy: y, r: 1.9, fill: C.paper, stroke: C.ink3, "stroke-width": 1 }));
    } else {
      const w = a.ans === "a" ? a.a : a.b, l = a.ans === "a" ? a.b : a.a;
      parts.push(el("circle", { cx: fa.x(l), cy: y, r: 1.9, fill: C.paper, stroke: col, "stroke-width": 1 }), el("circle", { cx: fa.x(w), cy: y, r: 2.2, fill: col }));
    }
  }
  if (t < N) {
    const a = run.answers[t];
    const y = rTop + (t + 0.5) * rowH;
    for (const x of [a.a, a.b]) parts.push(el("circle", { cx: fa.x(x), cy: y, r: 2.6, fill: "none", stroke: C.acq, "stroke-width": 1.4 }));
  }
  parts.push(axis({ scale: fa.x, orient: "bottom", at: rBot + 4, title: L.x, count: narrow ? 4 : 5, grid: false }));
  if (narrow) parts.push(text(fa.left, rBot + 52, p.cantCompare ? tpl(L.keyCc, { k: nCc, n: N }) : L.key, { "font-size": TYPE.small, class: "fig-t-muted" }));

  // Panel C: the retest, both orders.
  const cTop = rBot + (narrow ? 76 : 58);
  if (narrow) parts.push(text(fa.left, cTop, L.retest1, { "font-size": TYPE.label, class: "fig-t-strong" }), text(fa.left, cTop + 16, L.retest2, { "font-size": TYPE.label, class: "fig-t-strong" }));
  else parts.push(text(fa.left, cTop, L.retest, { "font-size": TYPE.label, class: "fig-t-strong" }));
  const cSub = narrow ? cTop + 16 : cTop;
  parts.push(text(fa.left, cSub + 16, narrow ? tpl(L.retestSubShort, { g: GROUP }) : tpl(L.retestSub, { k: EARLY, g: GROUP }), { "font-size": TYPE.small, class: "fig-t-muted" }));
  const labW = narrow ? 84 : 132;
  const bx0 = fa.left + labW, bx1 = fa.right - (narrow ? 34 : 40);
  const bs = linear([0, 1], [bx0, bx1]);
  const rowsTop = cSub + 32;
  const rowStep = 17, groupGap = 8;
  const out: string[] = [];
  const groupsH = 2 * (rowStep + 2 * rowStep) + groupGap;
  // chance line and ticks
  out.push(el("line", { x1: bs(0.5), x2: bs(0.5), y1: rowsTop - 2, y2: rowsTop + groupsH, stroke: C.rule, "stroke-dasharray": "3 3" }));
  for (const v of [0, 0.5, 1]) out.push(text(bs(v), rowsTop + groupsH + 13, v === 0.5 ? `50% ${L.chance}` : pc(v), { "font-size": TYPE.small, "text-anchor": v === 0 ? "start" : v === 1 ? "end" : "middle", class: "fig-t-muted fig-t-num" }));
  let y = rowsTop;
  for (const o of ["eubo", "random"] as const) {
    const sel = p.order === o;
    out.push(text(fa.left, y + 10, narrow ? (o === "eubo" ? L.euboShort : L.randomShort) : (o === "eubo" ? L.eubo : L.random), { "font-size": TYPE.small, class: sel ? "fig-t-strong" : "fig-t-muted" }));
    y += rowStep;
    const me = G.people[o][p.person - 1];
    for (const [key, color, ready, lab, val, dot] of [
      ["end", C.c5, t >= N, narrow ? L.atEndShort : L.atEnd, G[o].end, me.end],
      ["week", C.c6, t > N, narrow ? L.atWeekShort : L.atWeek, G[o].week, me.week],
    ] as const) {
      out.push(text(bx0 - 6, y + 9, lab, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted" }));
      out.push(el("rect", { x: bx0, y: y + 1, width: bx1 - bx0, height: 11, rx: 2, fill: C.grid }));
      if (ready) {
        out.push(el("rect", { x: bx0, y: y + 1, width: Math.max(0, bs(val) - bx0), height: 11, rx: 2, fill: color, opacity: sel ? 1 : 0.75 }));
        out.push(el("circle", { cx: bs(dot), cy: y + 6.5, r: 3.2, fill: C.ink, stroke: C.paper, "stroke-width": 1.4 }));
        out.push(text(bx1 + 4, y + 10, pc(val), { "font-size": TYPE.small, class: "fig-t-num fig-t-strong" }));
      } else {
        out.push(text(bx0 + 6, y + 10, tpl(key === "end" ? L.pending : L.waiting, { n: N }), { "font-size": TYPE.small, class: "fig-t-faint" }));
      }
      y += rowStep;
    }
    y += groupGap;
  }
  parts.push(g({}, ...out));
  const H = rowsTop + groupsH + 22;
  return svg(st.w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "syn-four-components",
  title: { en: "Four components of a comparison, and the experiment that separates them (a simulation)", zh: "一次比较的四个成分，以及区分它们的实验（模拟）" },
  labels,
  params,
  hint: { en: "A simulation of a model, not data. Play the session or drag the timeline; change the four components and compare how the retest bars of the two query orders move.", zh: "这是模型的模拟，不是数据。播放会话或拖动时间轴；改变四个成分，比较两种查询顺序的重测条形如何变化。" },
  timeline: {
    rate: 3,
    discrete: true,
    duration: () => N + 1,
    keyframes: (_p, lang) => [
      { t: 0, label: labels[lang].k0 },
      { t: EARLY, label: tpl(labels[lang].k10, { k: EARLY }) },
      { t: N, label: labels[lang].kEnd },
      { t: N + 1, label: labels[lang].kWeek },
    ],
    poster: () => N + 1,
    unit: { symbol: { en: "", zh: "" }, value: (t, _p, lang) => (Math.round(t) > N ? labels[lang].unitWeek : `${Math.round(t)}/${N}`) },
  },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "Next person", zh: "下一个人" }, run: (p) => ({ ...p, person: (Number(p.person) % GROUP) + 1 }) },
  ],
});
