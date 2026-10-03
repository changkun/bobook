// Answers that depend on where an option is shown. A simulated person
// compares two designs on [0, 1], one in a left slot and one in a right slot,
// and answers by the Bradley-Terry model with a preference b for whatever is
// in the left slot: P(left wins) = sigmoid(u(left) - u(right) + b). The
// optimizer keeps an incumbent (the compared design with the highest fitted
// utility) and pairs it with an optimistic challenger (largest fitted
// improvement over the incumbent plus two standard deviations of that
// difference). Three interfaces are compared:
//
//   same slot  the incumbent is always shown on the left, so position and
//              quality are confounded and the model reads the bias as quality
//   random     the incumbent's side is drawn at random each round, so the
//              bias becomes symmetric noise
//   modeled    random sides and an extra bias parameter in the likelihood,
//              which turns the bias into a measured quantity
//
// The surrogate is a Gaussian prior on 11 radial basis functions (a small
// stand-in for a Gaussian process), fitted by Newton's method with a Laplace
// covariance. The top panel replays one simulated session round by round;
// the bottom panel averages the regret of the incumbent over 24 simulated
// people for all three interfaces.

import { defineFigure, tr, type Lang, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { cholesky, cholSolve, spdInverse } from "./lib/linalg.ts";
import { memo, rng } from "./lib/random.ts";
import { sigmoid } from "./lib/stats.ts";
import { band, clip, curve, frame, frameAxes } from "./lib/plot.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "design x",
    y: "utility",
    regret: "regret",
    rounds: "answers",
    truth: "the person's utility",
    mean: "fitted utility",
    band: "95% band",
    incumbent: "incumbent",
    best: "true best",
    same: "incumbent always left",
    random: "random side",
    modeled: "random side, bias modeled",
    avg: "mean regret of the incumbent over {s} simulated people",
    status: "After {t} answer{t:/s}: incumbent x = {x} (true best {best})",
    share: "Answers for the left slot: {p}%",
    bhat: " · estimated bias {b} (true {tb})",
    none: "No answers yet: the session starts from the design at x = 0.25.",
    describe: "{pol}, left-slot preference b = {b}. After {t} answers the incumbent is x = {x} against a true best of {best}, and {p}% of answers went to the left slot{bh}. Averaged over {s} simulated people, regret after {T} answers is {r1} with the incumbent always on the left, {r2} with random sides, and {r3} with random sides and the bias modeled.",
    bhatD: "; the estimated bias is {b}",
    kf0: "start of the session",
    kf1: "early",
    kf2: "midway",
    kf3: "end of the session",
  },
  zh: {
    x: "设计 x",
    y: "效用",
    regret: "遗憾",
    rounds: "回答数",
    truth: "此人的效用",
    mean: "拟合的效用",
    band: "95% 区间",
    incumbent: "当前最优点",
    best: "真实最优",
    same: "当前最优点总在左侧",
    random: "随机左右",
    modeled: "随机左右，并对偏差建模",
    avg: "当前最优点的遗憾，在 {s} 个模拟的人上取平均",
    status: "{t} 个回答之后：当前最优点 x = {x}（真实最优为 {best}）",
    share: "选左侧位置的回答：{p}%",
    bhat: " · 估计偏差 {b}（真值 {tb}）",
    none: "还没有回答：会话从 x = 0.25 处的设计开始。",
    describe: "{pol}，对左侧位置的偏好 b = {b}。{t} 个回答之后，当前最优点为 x = {x}，真实最优为 {best}，{p}% 的回答选了左侧位置{bh}。在 {s} 个模拟的人上取平均，{T} 个回答之后的遗憾为：当前最优点总在左侧时 {r1}，随机左右时 {r2}，随机左右并对偏差建模时 {r3}。",
    bhatD: "；估计的偏差为 {b}",
    kf0: "会话开始",
    kf1: "早期",
    kf2: "中途",
    kf3: "会话结束",
  },
};

const POLICIES = [
  { value: "same", label: { en: "Incumbent always left", zh: "当前最优点总在左侧" } },
  { value: "random", label: { en: "Random side", zh: "随机左右" } },
  { value: "modeled", label: { en: "Random side, bias modeled", zh: "随机左右，并对偏差建模" } },
] as const;

const params = {
  bias: { kind: "range", label: { en: "Preference for the left slot b", zh: "对左侧位置的偏好 b" }, min: 0, max: 2, default: 1, step: 0.25 },
  policy: { kind: "choice", label: { en: "Interface", zh: "界面" }, options: POLICIES, default: "same", control: "buttons" },
  seed: { kind: "range", label: { en: "Person", zh: "人" }, min: 1, max: 999, default: 4, step: 1, control: false },
} as const;

type P = { bias: number; policy: string; seed: number };

const T = 60; // answers per session
const PEOPLE = 24; // simulated people in the average
const G = 41;
const XS = grid(0, 1, G);
const FINE = grid(0, 1, 121);
const K = 11;
const CENTERS = grid(0, 1, K);
const ELL = 0.1;
const NP = K + 1; // basis weights plus the bias parameter
const KAPPA = 2;
const START = Math.round(0.25 * (G - 1));
const basis = (x: number) => CENTERS.map((c) => Math.exp(-((x - c) ** 2) / (2 * ELL * ELL)));
const PHI = XS.map(basis);
const PHI_FINE = FINE.map(basis);
const MEAN_PHI = CENTERS.map((_, k) => PHI_FINE.reduce((a, f) => a + f[k], 0) / FINE.length);

// Each simulated person has a clear favorite and a weaker second peak; the
// session starts near the second peak.
function person(seed: number) {
  const r = rng(seed * 31 + 5);
  const a = 0.62 + 0.2 * r(), b = 0.12 + 0.18 * r();
  const u = (x: number) => 4 * Math.exp(-((x - a) ** 2) / (2 * 0.1 ** 2)) + 3 * Math.exp(-((x - b) ** 2) / (2 * 0.09 ** 2));
  const U = XS.map(u);
  let bi = 0;
  for (let i = 1; i < G; i++) if (U[i] > U[bi]) bi = i;
  return { u, U, best: bi };
}

interface Comp { L: number; R: number; leftWon: boolean }

function fit(comps: Comp[], modelBias: boolean, w0: number[]): { w: number[]; cov: number[][] } {
  const w = w0.slice();
  let H: number[][] = [];
  for (let it = 0; it < 30; it++) {
    const grad = w.map((v) => -v);
    H = Array.from({ length: NP }, (_, a) => Array.from({ length: NP }, (_, b) => (a === b ? 1 : 0)));
    for (const c of comps) {
      const d = PHI[c.L].map((v, k) => v - PHI[c.R][k]);
      d.push(modelBias ? 1 : 0);
      let z = 0;
      for (let k = 0; k < NP; k++) z += d[k] * w[k];
      const s = sigmoid(z), y = c.leftWon ? 1 : 0, h = s * (1 - s);
      for (let a = 0; a < NP; a++) {
        grad[a] += (y - s) * d[a];
        for (let b = 0; b <= a; b++) H[a][b] += h * d[a] * d[b];
      }
    }
    for (let a = 0; a < NP; a++) for (let b = a + 1; b < NP; b++) H[a][b] = H[b][a];
    const step = cholSolve(cholesky(H), grad);
    let big = 0;
    for (let k = 0; k < NP; k++) { w[k] += step[k]; big = Math.max(big, Math.abs(step[k])); }
    if (big < 1e-7) break;
  }
  return { w, cov: spdInverse(H) };
}

const quad = (d: number[], cov: number[][]) => {
  let s = 0;
  for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) s += d[a] * cov[a][b] * d[b];
  return Math.max(0, s);
};

interface Snapshot { inc: number; pair?: [number, number]; leftWon?: boolean; w: number[]; cov: number[][] | null; left: number }

// One session: returns the incumbent's regret after every answer and, when
// asked, a snapshot of the model after every answer.
function session(policy: string, bias: number, seed: number, keep: boolean): { regret: number[]; snaps: Snapshot[] } {
  const { U, best } = person(seed);
  const r = rng(seed * 977 + Math.round(bias * 8) * 101 + (policy === "same" ? 1 : policy === "random" ? 2 : 3));
  const comps: Comp[] = [];
  let w = new Array<number>(NP).fill(0);
  let cov: number[][] | null = null;
  let inc = START, left = 0;
  let meanG = XS.map(() => 0);
  const regret: number[] = [U[best] - U[inc]];
  const snaps: Snapshot[] = keep ? [{ inc, w: w.slice(), cov: null, left: 0 }] : [];
  for (let t = 0; t < T; t++) {
    // the optimistic challenger: largest fitted improvement over the incumbent
    // plus KAPPA standard deviations of that improvement
    let ch = -1, bv = -Infinity;
    for (let i = 0; i < G; i++) {
      if (Math.abs(i - inc) < 2) continue;
      const sdd = cov ? Math.sqrt(quad(PHI[i].map((q, k) => q - PHI[inc][k]), cov)) : 1;
      const v = meanG[i] - meanG[inc] + KAPPA * sdd + 1e-9 * r();
      if (v > bv) { bv = v; ch = i; }
    }
    const incLeft = policy === "same" ? true : r() < 0.5;
    const L = incLeft ? inc : ch, R = incLeft ? ch : inc;
    const leftWon = r() < sigmoid(U[L] - U[R] + bias);
    if (leftWon) left++;
    comps.push({ L, R, leftWon });
    const f = fit(comps, policy === "modeled", w);
    w = f.w;
    cov = f.cov;
    meanG = PHI.map((q) => q.reduce((a, v, k) => a + v * w[k], 0));
    let bi = inc, bm = -Infinity;
    for (const c of comps) for (const i of [c.L, c.R]) if (meanG[i] > bm) { bm = meanG[i]; bi = i; }
    inc = bi;
    regret.push(U[best] - U[inc]);
    if (keep) snaps.push({ inc, pair: [L, R], leftWon, w: w.slice(), cov, left });
  }
  return { regret, snaps };
}

const averages = memo((bias: number): Record<string, number[]> => {
  const out: Record<string, number[]> = {};
  for (const pol of POLICIES) {
    const avg = new Array<number>(T + 1).fill(0);
    for (let s = 1; s <= PEOPLE; s++) session(pol.value, bias, s, false).regret.forEach((v, i) => (avg[i] += v / PEOPLE));
    out[pol.value] = avg;
  }
  return out;
}, 16);

const single = memo((policy: string, bias: number, seed: number) => {
  const p = person(seed + 1000);
  const run = session(policy, bias, seed + 1000, true);
  const tbar = FINE.reduce((a, x) => a + p.u(x), 0) / FINE.length;
  return { truth: FINE.map((x) => p.u(x) - tbar), best: XS[p.best], run };
}, 24);

function view(snap: Snapshot) {
  const mean = PHI_FINE.map((q) => q.reduce((a, v, k) => a + (v - MEAN_PHI[k]) * snap.w[k], 0));
  const sd = PHI_FINE.map((q) => (snap.cov ? Math.sqrt(quad(q.map((v, k) => v - MEAN_PHI[k]), snap.cov)) : Math.sqrt(quad(q.map((v, k) => v - MEAN_PHI[k]), IDENT))));
  return { mean, sd };
}
const IDENT = Array.from({ length: K }, (_, a) => Array.from({ length: K }, (_, b) => (a === b ? 1 : 0)));

const policyLabel = (v: string, lang: Lang) => tr(POLICIES.find((q) => q.value === v)!.label, lang);
const COLORS: Record<string, string> = { same: C.c5, random: C.c6, modeled: C.c7 };

function stateAt(st: State<P>) {
  const p = st.p;
  const t = Math.max(0, Math.min(T, Math.round(st.t)));
  const S = single(p.policy, p.bias, p.seed);
  const snap = S.run.snaps[t];
  return { t, S, snap };
}

function describe(st: State<P>): string {
  const p = st.p, lang = st.lang ?? "en", L = labels[lang];
  const { t, S, snap } = stateAt(st);
  const A = averages(p.bias);
  return tpl(L.describe, {
    pol: policyLabel(p.policy, lang), b: fixed(p.bias, 2), t, x: fixed(XS[snap.inc], 2), best: fixed(S.best, 2),
    p: t ? Math.round((100 * snap.left) / t) : 0,
    bh: p.policy === "modeled" && t ? tpl(L.bhatD, { b: fixed(snap.w[K], 2) }) : "",
    s: PEOPLE, T, r1: fixed(A.same[T], 2), r2: fixed(A.random[T], 2), r3: fixed(A.modeled[T], 2),
  });
}

function render(st: State<P>): string {
  const p = st.p, L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const { t, S, snap } = stateAt(st);
  const A = averages(p.bias);
  const V = view(snap);
  const x0 = narrow ? 8 : 40;
  const parts: string[] = [];
  // legend for the top panel
  const items: Array<{ kind: "line" | "dash" | "band"; color: string; label: string }> = [
    { kind: "dash", color: C.truth, label: L.truth },
    { kind: "line", color: C.model, label: L.mean },
    { kind: "band", color: C.band, label: L.band },
  ];
  let lx = x0, ly = 14;
  for (const it of items) {
    const tw = labelWidth(it.label, 6.4) + 34;
    if (lx + tw > st.w - 8 && lx > x0) { lx = x0; ly += 18; }
    parts.push(it.kind === "band"
      ? el("rect", { x: lx, y: ly - 6, width: 18, height: 10, fill: it.color, rx: 2 })
      : el("line", { x1: lx, x2: lx + 18, y1: ly - 1, y2: ly - 1, stroke: it.color, "stroke-width": 2, "stroke-dasharray": it.kind === "dash" ? "5 4" : undefined }));
    parts.push(text(lx + 24, ly + 3, it.label, { "font-size": TYPE.small, class: "fig-t-muted" }));
    lx += tw;
  }
  const top = ly + 18;
  const mainH = narrow ? 170 : 200;
  const f = frame({ w: st.w, top, height: mainH, yDomain: [-3, 3.5], yTitle: L.y });
  const cid = `${st.uid}-clip`;
  const lo = V.mean.map((m, i) => m - 1.96 * V.sd[i]);
  const hi = V.mean.map((m, i) => m + 1.96 * V.sd[i]);
  parts.push(
    el("defs", {}, clip(f, cid)),
    frameAxes(f, { yTitle: L.y, xTitle: narrow ? "" : L.x }),
    g({ "clip-path": `url(#${cid})` },
      band(f, FINE, lo, hi),
      curve(f, FINE, S.truth, { stroke: C.truth, "stroke-dasharray": "5 4", "stroke-width": 1.8 }),
      curve(f, FINE, V.mean),
    ),
  );
  // true best (tick on the axis) and the incumbent (orange line)
  const bx = f.x(S.best);
  parts.push(
    el("path", { d: `M${bx},${f.bottom} l-5,8 h10 z`, fill: C.truth }),
    el("line", { x1: f.x(XS[snap.inc]), x2: f.x(XS[snap.inc]), y1: f.top, y2: f.bottom, stroke: C.acq, "stroke-width": 1.6, "stroke-dasharray": "4 3" }),
    text(f.x(XS[snap.inc]) + (XS[snap.inc] > 0.8 ? -5 : 5), f.top + 12, L.incumbent, { "font-size": TYPE.small, "text-anchor": XS[snap.inc] > 0.8 ? "end" : "start", class: "fig-t-muted fig-t-halo" }),
  );
  // the latest pair, labeled by slot; the winner is filled
  if (snap.pair) {
    const [Li, Ri] = snap.pair;
    const yAt = (i: number) => {
      const j = Math.round((XS[i] * (FINE.length - 1)));
      return Math.max(f.top + 10, Math.min(f.bottom - 10, f.y(V.mean[j])));
    };
    for (const [i, lab, won] of [[Li, "L", snap.leftWon], [Ri, "R", !snap.leftWon]] as const) {
      const cx = f.x(XS[i]), cy = yAt(i);
      parts.push(
        el("circle", { cx, cy, r: 9, fill: won ? C.ink : C.paper, stroke: C.ink, "stroke-width": 1.5 }),
        text(cx, cy + 4, lab, { "font-size": TYPE.small, "text-anchor": "middle", fill: won ? C.paper : C.ink, class: "fig-t-strong" }),
      );
    }
  }
  // status lines
  let ry = f.bottom + (narrow ? 26 : 50);
  if (narrow) parts.push(text((f.left + f.right) / 2, f.bottom + 30, L.x, { "font-size": TYPE.body, "text-anchor": "middle", class: "fig-t-muted" }));
  if (narrow) ry += 22;
  const status = t
    ? tpl(L.status, { t, x: fixed(XS[snap.inc], 2), best: fixed(S.best, 2) })
    : L.none;
  parts.push(text(x0, ry, status, { "font-size": TYPE.small, class: "fig-t-num fig-t-strong" }));
  ry += 17;
  if (t) {
    parts.push(text(x0, ry, tpl(L.share, { p: Math.round((100 * snap.left) / t) }) + (p.policy === "modeled" ? tpl(L.bhat, { b: fixed(snap.w[K], 2), tb: fixed(p.bias, 2) }) : ""), { "font-size": TYPE.small, class: "fig-t-num fig-t-strong" }));
  }
  ry += 26;
  // bottom panel: mean regret over rounds, all three interfaces
  parts.push(text(x0, ry, tpl(L.avg, { s: PEOPLE }), { "font-size": TYPE.small, class: "fig-t-muted" }));
  ry += 8;
  let lx2 = x0, ly2 = ry + 12;
  for (const pol of POLICIES) {
    const lab = L[pol.value];
    const tw = labelWidth(lab, 6.4) + 34;
    if (lx2 + tw > st.w - 8 && lx2 > x0) { lx2 = x0; ly2 += 18; }
    const on = pol.value === p.policy;
    parts.push(
      el("line", { x1: lx2, x2: lx2 + 18, y1: ly2 - 1, y2: ly2 - 1, stroke: COLORS[pol.value], "stroke-width": on ? 3 : 1.6 }),
      text(lx2 + 24, ly2 + 3, lab, { "font-size": TYPE.small, class: on ? "fig-t-strong" : "fig-t-muted" }),
    );
    lx2 += tw;
  }
  const rTop = ly2 + 14;
  const rH = narrow ? 110 : 130;
  const maxR = Math.max(1.2, ...A.same, ...A.random, ...A.modeled) * 1.05;
  const fr = frame({ w: st.w, top: rTop, height: rH, yDomain: [0, maxR], xDomain: [0, T], yTitle: L.regret });
  const rx = linear([0, T], [fr.left, fr.right]);
  parts.push(
    axis({ scale: fr.y, orient: "left", at: fr.left, span: [fr.left, fr.right], title: L.regret, count: 3 }),
    axis({ scale: rx, orient: "bottom", at: fr.bottom, span: [fr.top, fr.bottom], title: L.rounds, count: narrow ? 4 : 6 }),
  );
  const order = (POLICIES.map((q) => q.value) as string[]).filter((v) => v !== p.policy).concat([p.policy]);
  for (const v of order) {
    const on = v === p.policy;
    parts.push(el("path", {
      d: A[v].map((y, i) => `${i ? "L" : "M"}${rx(i).toFixed(1)},${fr.y(y).toFixed(1)}`).join(""),
      fill: "none", stroke: COLORS[v], "stroke-width": on ? 2.8 : 1.5, opacity: on ? 1 : 0.7, "stroke-linejoin": "round",
    }));
  }
  parts.push(el("line", { x1: rx(t), x2: rx(t), y1: fr.top, y2: fr.bottom, stroke: C.ink3, "stroke-width": 1, "stroke-dasharray": "3 3" }));
  return svg(st.w, fr.bottom + 44, describe(st), g({}, ...parts));
}

export default defineFigure({
  name: "adj-position-bias",
  title: { en: "Answers that depend on where an option is shown", zh: "取决于选项展示位置的回答" },
  labels,
  params,
  hint: { en: "Play or scrub through one session; switch the interface to compare.", zh: "播放或拖动浏览一次会话；切换界面加以比较。" },
  timeline: {
    rate: 6,
    discrete: true,
    duration: () => T,
    unit: { symbol: { en: "answers", zh: "个回答" }, value: (t) => String(Math.round(t)) },
    keyframes: (_p, lang) => [
      { t: 0, label: labels[lang].kf0 },
      { t: 10, label: labels[lang].kf1 },
      { t: 30, label: labels[lang].kf2 },
      { t: T, label: labels[lang].kf3 },
    ],
    poster: () => T,
  },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [{ label: { en: "New person", zh: "换一个人" }, run: (p) => ({ ...p, seed: (p.seed % 999) + 1 }) }],
});
