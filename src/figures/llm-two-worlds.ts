// One Bradley-Terry likelihood, two worlds. A utility (or reward) r on a
// one-dimensional option axis is fitted to simulated comparisons with the
// logistic Bradley-Terry likelihood P(a beats b) = sigmoid(r(a) - r(b)), a
// Gaussian prior on the weights of 15 radial basis functions (a small stand-in
// for a Gaussian process), the maximum a posteriori fit found by Newton's
// method, and a Laplace band. The reader switches between two presets:
//
//   one person: 50 comparisons, one utility, pairs drawn anywhere in the domain
//   alignment:  a million comparisons, two annotator groups, pairs drawn from
//               a reference policy that covers only part of the domain
//
// and can then change one factor at a time (sample size, the second group's
// share, where the pairs come from) to see which difference does what. The
// bottom strip shows where the comparisons fall and what each world returns:
// one design (the argmax of the fitted mean) or a distribution (the reference
// policy tilted by exp(r / beta), the optimum of KL-regularized reward
// maximization).
//
// Comparisons are simulated, not looped over: up to 2,000 they are drawn one
// by one; above that, options are binned on a 61-point grid, the count of each
// pair is drawn from a Poisson (normal for large means) and its wins from a
// binomial (normal for large counts), so the cost is constant in N.

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { cholesky, cholSolve, spdInverse } from "./lib/linalg.ts";
import { memo, normal, rng, type Rng } from "./lib/random.ts";
import { sigmoid } from "./lib/stats.ts";
import { axis } from "./lib/axis.ts";
import { band, clip, curve, frame, frameAxes } from "./lib/plot.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    pboTitle: "One person · {n} comparisons",
    alignTitle: "Many annotators · {n} comparisons",
    anywhere: "pairs from anywhere",
    policy: "pairs from the policy",
    oneUtility: "one utility",
    groups: "two groups ({a}% / {b}%)",
    lik: "One likelihood in both worlds: P(a ≻ b) = sigmoid(r(a) − r(b))",
    x: "option: a design, or a response on one axis",
    xShort: "option",
    y: "utility or reward",
    yShort: "reward",
    mean: "fitted reward",
    band: "95% band",
    person: "the person's utility",
    groupA: "group A's utility",
    groupB: "group B's utility",
    share: "where comparisons fall",
    pick: "returned design",
    tilt: "π_ref tilted by exp(r/β)",
    sdIn: "Reward sd where compared: {a}",
    sdOut: " · elsewhere: {b}",
    err1: "Error against the person's utility, where compared: {a}",
    err1Short: "Error against the person's utility: {a}",
    err2: "Error where compared, against group A: {a} · against group B: {b}",
    err2a: "Error where compared, against group A: {a}",
    err2b: "Error where compared, against group B: {b}",
    retPbo: "Returns one design: x = {x} ({who}: {b})",
    bestOne: "the person's best",
    bestA: "group A's best",
    never: "the policy never produces these options",
    retAlign: "Returns a distribution, β = 1: mean x = {x}",
    describe: "{world}: {n} comparisons from {who}, {q}; posterior sd of the reward where compared {sd}{out}, error against {ref} {err}. {ret}",
    dPbo: "It returns the design x = {x}.",
    dAlign: "It returns the reference policy tilted by exp(r/β), with mean x = {x}.",
    wPbo: "One person",
    wAlign: "Many annotators (alignment)",
    outD: ", and {b} where no comparison falls",
    refA: "group A",
    refOne: "the person's utility",
  },
  zh: {
    pboTitle: "一个人 · {n} 次比较",
    alignTitle: "许多标注者 · {n} 次比较",
    anywhere: "选项对来自任意位置",
    policy: "选项对来自策略",
    oneUtility: "一个效用",
    groups: "两组（{a}% / {b}%）",
    lik: "两个世界用同一个似然：P(a ≻ b) = sigmoid(r(a) − r(b))",
    x: "选项：一个设计，或一维轴上的一个回复",
    xShort: "选项",
    y: "效用或奖励",
    yShort: "奖励",
    mean: "拟合的奖励",
    band: "95% 区间",
    person: "此人的效用",
    groupA: "A 组的效用",
    groupB: "B 组的效用",
    share: "比较落在何处",
    pick: "返回的设计",
    tilt: "按 exp(r/β) 倾斜的 π_ref",
    sdIn: "比较过的位置上奖励的标准差：{a}",
    sdOut: " · 其他位置：{b}",
    err1: "比较过的位置上，与此人效用的误差：{a}",
    err1Short: "与此人效用的误差：{a}",
    err2: "比较过的位置上的误差，相对 A 组：{a} · 相对 B 组：{b}",
    err2a: "比较过的位置上相对 A 组的误差：{a}",
    err2b: "比较过的位置上相对 B 组的误差：{b}",
    retPbo: "返回一个设计：x = {x}（{who}：{b}）",
    bestOne: "此人的最佳",
    bestA: "A 组的最佳",
    never: "策略从不产生这些选项",
    retAlign: "返回一个分布，β = 1：均值 x = {x}",
    describe: "{world}：{n} 次比较，来自{who}，{q}；比较过的位置上奖励的后验标准差为 {sd}{out}，与{ref}之间的误差为 {err}。{ret}",
    dPbo: "它返回设计 x = {x}。",
    dAlign: "它返回按 exp(r/β) 倾斜的参考策略，均值 x = {x}。",
    wPbo: "一个人",
    wAlign: "许多标注者（对齐）",
    outD: "，没有比较的位置上为 {b}",
    refA: " A 组",
    refOne: "此人的效用",
  },
};

const WORLDS = [
  { value: "pbo", label: { en: "One person (PBO)", zh: "一个人（PBO）" } },
  { value: "align", label: { en: "Many annotators (alignment)", zh: "许多标注者（对齐）" } },
] as const;

const QUERIES = [
  { value: "anywhere", label: { en: "Anywhere", zh: "任意位置" } },
  { value: "policy", label: { en: "From the policy", zh: "来自策略" } },
] as const;

const params = {
  world: { kind: "choice", label: { en: "World", zh: "世界" }, options: WORLDS, default: "pbo", control: "buttons" },
  n: { kind: "range", label: { en: "Comparisons N", zh: "比较次数 N" }, min: 10, max: 1_000_000, default: 50, scale: "log" },
  mix: { kind: "range", label: { en: "Second group's share", zh: "第二组的占比" }, min: 0, max: 0.5, default: 0, step: 0.05 },
  queries: { kind: "choice", label: { en: "Where pairs come from", zh: "选项对的来源" }, options: QUERIES, default: "anywhere", control: "buttons" },
  seed: { kind: "range", label: { en: "Seed", zh: "种子" }, min: 1, max: 999, default: 7, step: 1, control: false },
} as const;

type P = { world: string; n: number; mix: number; queries: string; seed: number };

const PRESETS: Record<string, Pick<P, "n" | "mix" | "queries">> = {
  pbo: { n: 50, mix: 0, queries: "anywhere" },
  align: { n: 1_000_000, mix: 0.4, queries: "policy" },
};

// The option grid, the basis, and the prior.
const G = 61;
const XS = grid(0, 1, G);
const K = 15;
const CENTERS = grid(0, 1, K);
const ELL = 0.09;
const PRIOR_VAR = 1;
const BETA = 1;
const PHI = XS.map((x) => CENTERS.map((c) => Math.exp(-((x - c) ** 2) / (2 * ELL * ELL))));
const FINE = grid(0, 1, 201);
const PHI_FINE = FINE.map((x) => CENTERS.map((c) => Math.exp(-((x - c) ** 2) / (2 * ELL * ELL))));

// Group A is also "the person" of the PBO world; group B is a second group of
// annotators whose favorite lies elsewhere.
const uA = (x: number) => 3 * Math.exp(-((x - 0.75) ** 2) / (2 * 0.12 ** 2));
const uB = (x: number) => 3 * Math.exp(-((x - 0.33) ** 2) / (2 * 0.1 ** 2));

// Where pairs come from: uniform over the grid, or a reference policy that
// puts no mass outside about [0.2, 0.6].
const Q_ANY = XS.map(() => 1 / G);
const Q_POL = (() => {
  const raw = XS.map((x) => { const z = (x - 0.4) / 0.09; return Math.abs(z) <= 2.2 ? Math.exp(-z * z / 2) : 0; });
  const s = raw.reduce((a, b) => a + b, 0);
  return raw.map((v) => v / s);
})();

interface Cell { i: number; j: number; n: number; w: number }

function poisson(mu: number, r: Rng): number {
  const L = Math.exp(-mu);
  let k = 0, p = 1;
  do { k++; p *= r(); } while (p > L);
  return k - 1;
}

function simulate(n: number, mix: number, q: number[], seed: number): Cell[] {
  const r = rng(seed * 7919 + Math.round(n) + Math.round(mix * 100) * 13);
  const z = normal(r);
  const pref = (i: number, j: number) => (1 - mix) * sigmoid(uA(XS[i]) - uA(XS[j])) + mix * sigmoid(uB(XS[i]) - uB(XS[j]));
  if (n <= 2000) {
    const cdf: number[] = [];
    let acc = 0;
    for (const v of q) { acc += v; cdf.push(acc); }
    const pick = () => { const u = r() * acc; let k = 0; while (k < G - 1 && cdf[k] < u) k++; return k; };
    const map = new Map<number, Cell>();
    for (let t = 0; t < n; t++) {
      let i = pick(), j = pick(), guard = 0;
      while (i === j && guard++ < 100) j = pick();
      if (i === j) continue;
      if (i > j) [i, j] = [j, i];
      const key = i * G + j;
      let c = map.get(key);
      if (!c) { c = { i, j, n: 0, w: 0 }; map.set(key, c); }
      c.n++;
      if (r() < pref(i, j)) c.w++;
    }
    return [...map.values()];
  }
  const sq = q.reduce((a, b) => a + b * b, 0);
  const cells: Cell[] = [];
  for (let i = 0; i < G; i++) {
    for (let j = i + 1; j < G; j++) {
      const share = (2 * q[i] * q[j]) / (1 - sq);
      if (share <= 0) continue;
      const mu = n * share;
      const c = mu < 30 ? poisson(mu, r) : Math.max(0, Math.round(mu + Math.sqrt(mu) * z()));
      if (!c) continue;
      const pp = pref(i, j);
      let w = 0;
      if (c < 40) { for (let t = 0; t < c; t++) if (r() < pp) w++; }
      else w = Math.min(c, Math.max(0, Math.round(c * pp + Math.sqrt(c * pp * (1 - pp)) * z())));
      cells.push({ i, j, n: c, w });
    }
  }
  return cells;
}

// Maximum a posteriori weights under the Bradley-Terry likelihood and a
// N(0, PRIOR_VAR I) prior, by Newton's method; the Laplace covariance is the
// inverse of the negative Hessian at the mode.
function fitBT(cells: Cell[]): { w: number[]; cov: number[][] } {
  const D = cells.map((c) => PHI[c.i].map((v, k) => v - PHI[c.j][k]));
  let w = new Array<number>(K).fill(0);
  let H: number[][] = [];
  for (let it = 0; it < 50; it++) {
    const grad = w.map((v) => -v / PRIOR_VAR);
    H = Array.from({ length: K }, (_, a) => Array.from({ length: K }, (_, b) => (a === b ? 1 / PRIOR_VAR : 0)));
    for (let m = 0; m < cells.length; m++) {
      const c = cells[m], d = D[m];
      let delta = 0;
      for (let k = 0; k < K; k++) delta += d[k] * w[k];
      const s = sigmoid(delta);
      const gr = c.w - c.n * s;
      const h = c.n * s * (1 - s);
      for (let a = 0; a < K; a++) {
        grad[a] += gr * d[a];
        for (let b = 0; b <= a; b++) H[a][b] += h * d[a] * d[b];
      }
    }
    for (let a = 0; a < K; a++) for (let b = a + 1; b < K; b++) H[a][b] = H[b][a];
    const step = cholSolve(cholesky(H), grad);
    let big = 0;
    for (let k = 0; k < K; k++) { w[k] += step[k]; big = Math.max(big, Math.abs(step[k])); }
    if (big < 1e-9) break;
  }
  return { w, cov: spdInverse(H) };
}

interface Result {
  q: number[];
  share: number[]; // share of comparison endpoints at each grid option
  mean: number[]; // on FINE, centered by the query-weighted mean
  sd: number[]; // on FINE
  ta: number[]; // group A's utility on FINE, centered the same way
  tb: number[];
  sdIn: number;
  sdOut: number; // NaN when every option is queried
  errA: number;
  errB: number;
  argmax: number;
  tilt: number[]; // on the grid, sums to 1
  tiltMean: number;
  total: number;
}

const compute = memo((n0: number, mix: number, queries: string, seed: number): Result => {
  const n = Math.round(n0);
  const q = queries === "policy" ? Q_POL : Q_ANY;
  const cells = simulate(n, mix, q, seed);
  const { w, cov } = fitBT(cells);
  const phibar = CENTERS.map((_, k) => q.reduce((a, v, i) => a + v * PHI[i][k], 0));
  const centredMean = (phi: number[]) => phi.reduce((a, v, k) => a + (v - phibar[k]) * w[k], 0);
  const sdAt = (phi: number[]) => {
    const d = phi.map((v, k) => v - phibar[k]);
    let s = 0;
    for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) s += d[a] * cov[a][b] * d[b];
    return Math.sqrt(Math.max(0, s));
  };
  const abar = q.reduce((a, v, i) => a + v * uA(XS[i]), 0);
  const bbar = q.reduce((a, v, i) => a + v * uB(XS[i]), 0);
  const mean = PHI_FINE.map(centredMean);
  const sd = PHI_FINE.map(sdAt);
  const meanG = PHI.map(centredMean);
  const sdG = PHI.map(sdAt);
  const sdIn = sdG.reduce((a, v, i) => a + q[i] * v, 0);
  const out = XS.map((_, i) => i).filter((i) => q[i] === 0);
  const sdOut = out.length ? out.reduce((a, i) => a + sdG[i], 0) / out.length : NaN;
  const errA = Math.sqrt(q.reduce((a, v, i) => a + v * (meanG[i] - (uA(XS[i]) - abar)) ** 2, 0));
  const errB = Math.sqrt(q.reduce((a, v, i) => a + v * (meanG[i] - (uB(XS[i]) - bbar)) ** 2, 0));
  // The returned design is the best option among those that can be shown:
  // anywhere in the domain, or only within the policy's support.
  const lo = XS.find((_, i) => q[i] > 0)! - 0.5 / (G - 1), hi = XS.findLast((_, i) => q[i] > 0)! + 0.5 / (G - 1);
  let am = -1;
  for (let i = 0; i < FINE.length; i++) if (FINE[i] >= lo && FINE[i] <= hi && (am < 0 || mean[i] > mean[am])) am = i;
  const tiltRaw = q.map((v, i) => v * Math.exp(meanG[i] / BETA));
  const tz = tiltRaw.reduce((a, b) => a + b, 0);
  const tilt = tiltRaw.map((v) => v / tz);
  const ends = new Array<number>(G).fill(0);
  let total = 0;
  for (const c of cells) { ends[c.i] += c.n; ends[c.j] += c.n; total += c.n; }
  return {
    q, share: ends.map((v) => (total ? v / (2 * total) : 0)), mean, sd,
    ta: FINE.map((x) => uA(x) - abar), tb: FINE.map((x) => uB(x) - bbar),
    sdIn, sdOut, errA, errB, argmax: FINE[am], tilt, tiltMean: tilt.reduce((a, v, i) => a + v * XS[i], 0), total,
  };
}, 48);

const fmtN = (n: number) => Math.round(n).toLocaleString("en-US");
const fmtSd = (v: number) => (v < 0.095 ? v.toFixed(3) : v.toFixed(2));
const BEST_A = 0.75;

function describe(st: State<P>): string {
  const p = st.p, L = labels[st.lang ?? "en"];
  const R = compute(p.n, p.mix, p.queries, p.seed);
  const who = p.mix > 0 ? tpl(L.groups, { a: Math.round((1 - p.mix) * 100), b: Math.round(p.mix * 100) }) : L.oneUtility;
  return tpl(L.describe, {
    world: p.world === "pbo" ? L.wPbo : L.wAlign,
    n: fmtN(p.n), who, q: p.queries === "policy" ? L.policy : L.anywhere,
    sd: fmtSd(R.sdIn), out: Number.isNaN(R.sdOut) ? "" : tpl(L.outD, { b: fmtSd(R.sdOut) }),
    ref: p.mix > 0 ? L.refA : L.refOne, err: fmtSd(R.errA),
    ret: p.world === "pbo" ? tpl(L.dPbo, { x: fixed(R.argmax, 2) }) : tpl(L.dAlign, { x: fixed(R.tiltMean, 2) }),
  });
}

// A legend row like lib/plot.ts legend, with a wider per-character estimate
// so labels with digits and symbols ("95% band") never touch the next swatch.
function legendRow(x: number, y: number, w: number, items: Swatch[]): { svg: string; height: number } {
  const parts: string[] = [];
  let cx = x, cy = y;
  const rowH = 18;
  for (const it of items) {
    const tw = labelWidth(it.label, 6.9) + 38;
    if (cx + tw > x + w && cx > x) { cx = x; cy += rowH; }
    const sw = it.kind === "band"
      ? el("rect", { x: cx, y: cy - 6, width: 18, height: 10, fill: it.color, rx: 2, opacity: it.opacity ?? 1 })
      : el("line", { x1: cx, x2: cx + 18, y1: cy - 1, y2: cy - 1, stroke: it.color, "stroke-width": 2, "stroke-dasharray": it.kind === "dash" ? "5 4" : undefined });
    parts.push(sw, text(cx + 24, cy + 3, it.label, { "font-size": TYPE.small, class: "fig-t-muted" }));
    cx += tw;
  }
  return { svg: g({}, ...parts), height: cy - y + rowH };
}

interface Swatch { kind: "line" | "dash" | "band"; color: string; label: string; opacity?: number }

function render(st: State<P>): string {
  const p = st.p, L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const R = compute(p.n, p.mix, p.queries, p.seed);
  const het = p.mix > 0;
  const qLabel = p.queries === "policy" ? L.policy : L.anywhere;
  const head = tpl(p.world === "pbo" ? L.pboTitle : L.alignTitle, { n: fmtN(p.n) });
  const who = het ? tpl(L.groups, { a: Math.round((1 - p.mix) * 100), b: Math.round(p.mix * 100) }) : L.oneUtility;
  const x0 = narrow ? 8 : 40;
  const parts: string[] = [];
  let y = 16;
  if (narrow) {
    parts.push(text(x0, y, head, { "font-size": TYPE.label, class: "fig-t-strong" }));
    y += 17;
    parts.push(text(x0, y, `${who} · ${qLabel}`, { "font-size": TYPE.label, class: "fig-t-strong" }));
  } else {
    parts.push(text(x0, y, `${head} · ${who} · ${qLabel}`, { "font-size": TYPE.label, class: "fig-t-strong" }));
  }
  y += 17;
  parts.push(text(x0, y, L.lik, { "font-size": TYPE.small, class: "fig-t-muted" }));
  const lg = legendRow(x0, y + 22, st.w - x0 - 8, [
    { kind: "line", color: C.model, label: L.mean },
    { kind: "band", color: C.band, label: L.band },
    { kind: "dash", color: C.truth, label: het ? L.groupA : L.person },
    ...(het ? [{ kind: "dash" as const, color: C.c5, label: L.groupB }] : []),
  ]);
  parts.push(lg.svg);
  const top = y + 22 + lg.height;
  const mainH = narrow ? 180 : 220;
  const yDom: [number, number] = [-3, 3.5];
  const f = frame({ w: st.w, top, height: mainH, yDomain: yDom, yTitle: narrow ? L.yShort : L.y });
  const cid = `${st.uid}-clip`;
  const lo = R.mean.map((m, i) => m - 1.96 * R.sd[i]);
  const hi = R.mean.map((m, i) => m + 1.96 * R.sd[i]);
  // the strip below: where comparisons fall, and what each world returns
  const sl = legendRow(f.left + 4, f.bottom + 22, f.right - f.left - 8, [
    { kind: "band", color: C.ink3, label: L.share, opacity: 0.5 },
    { kind: "line", color: C.acq, label: p.world === "align" ? L.tilt : L.pick },
  ]);
  const stripTop = f.bottom + 22 + sl.height - 6;
  const stripH = narrow ? 60 : 70;
  // shade the part of the axis the policy never reaches, in both panels
  const shade: string[] = [];
  if (p.queries === "policy") {
    const sup = XS.filter((_, i) => R.q[i] > 0);
    const a = sup[0] - 0.5 / (G - 1), b = sup[sup.length - 1] + 0.5 / (G - 1);
    for (const [y0, y1] of [[f.top, f.bottom], [stripTop, stripTop + stripH]] as const) {
      shade.push(
        el("rect", { x: f.x(0), y: y0, width: f.x(a) - f.x(0), height: y1 - y0, fill: C.panel }),
        el("rect", { x: f.x(b), y: y0, width: f.x(1) - f.x(b), height: y1 - y0, fill: C.panel }),
      );
    }
    if (!narrow) {
      parts.push(text(f.x(1) - 4, f.top + 13, L.never, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo" }));
    }
  }
  parts.unshift(g({}, ...shade));
  parts.push(
    el("defs", {}, clip(f, cid)),
    frameAxes(f, { yTitle: narrow ? L.yShort : L.y, xTicks: false }),
    g({ "clip-path": `url(#${cid})` },
      band(f, FINE, lo, hi),
      curve(f, FINE, R.ta, { stroke: C.truth, "stroke-dasharray": "5 4", "stroke-width": 1.8 }),
      het ? curve(f, FINE, R.tb, { stroke: C.c5, "stroke-dasharray": "5 4", "stroke-width": 1.8 }) : "",
      curve(f, FINE, R.mean),
    ),
    sl.svg,
  );
  const maxShare = Math.max(0.03, ...R.share, ...(p.world === "align" ? R.tilt : [])) * 1.08;
  const sy = linear([0, maxShare], [stripTop + stripH, stripTop]);
  const bw = Math.max(1.5, ((f.x(1) - f.x(0)) / (G - 1)) * 0.72);
  const bars = XS.map((x, i) => (R.share[i] > 0
    ? el("rect", { x: f.x(x) - bw / 2, y: sy(R.share[i]), width: bw, height: stripTop + stripH - sy(R.share[i]), fill: C.ink3, opacity: 0.5 })
    : "")).join("");
  parts.push(
    axis({ scale: f.x, orient: "bottom", at: stripTop + stripH, span: [stripTop, stripTop + stripH], title: narrow ? L.xShort : L.x, count: narrow ? 4 : 5 }),
    bars,
  );
  if (p.world === "align") {
    parts.push(el("path", {
      d: XS.map((x, i) => `${i ? "L" : "M"}${f.x(x).toFixed(1)},${sy(R.tilt[i]).toFixed(1)}`).join(""),
      fill: "none", stroke: C.acq, "stroke-width": 2, "stroke-linejoin": "round",
    }));
  } else {
    const px = f.x(R.argmax);
    parts.push(
      el("line", { x1: px, x2: px, y1: f.top, y2: f.bottom, stroke: C.acq, "stroke-width": 1.6, "stroke-dasharray": "4 3" }),
      el("line", { x1: px, x2: px, y1: stripTop, y2: stripTop + stripH, stroke: C.acq, "stroke-width": 1.6, "stroke-dasharray": "4 3" }),
      el("path", { d: `M${px},${f.top + 2} l5,-9 h-10 z`, fill: C.acq }),
    );
  }
  // readouts
  let ry = stripTop + stripH + 50;
  const line = (s: string) => {
    parts.push(text(x0, ry, s, { "font-size": TYPE.small, class: "fig-t-num fig-t-strong" }));
    ry += 17;
  };
  line(tpl(L.sdIn, { a: fmtSd(R.sdIn) }) + (Number.isNaN(R.sdOut) ? "" : tpl(L.sdOut, { b: fmtSd(R.sdOut) })));
  if (!het) line(narrow ? tpl(L.err1Short, { a: fmtSd(R.errA) }) : tpl(L.err1, { a: fmtSd(R.errA) }));
  else if (narrow) {
    line(tpl(L.err2a, { a: fmtSd(R.errA) }));
    line(tpl(L.err2b, { b: fmtSd(R.errB) }));
  } else line(tpl(L.err2, { a: fmtSd(R.errA), b: fmtSd(R.errB) }));
  line(p.world === "pbo"
    ? tpl(L.retPbo, { x: fixed(R.argmax, 2), b: fixed(BEST_A, 2), who: het ? L.bestA : L.bestOne })
    : tpl(L.retAlign, { x: fixed(R.tiltMean, 2) }));
  return svg(st.w, ry - 4, describe(st), g({}, ...parts));
}

export default defineFigure({
  name: "llm-two-worlds",
  title: { en: "One Bradley-Terry likelihood fitted in two worlds", zh: "同一个 Bradley-Terry 似然在两个世界中的拟合" },
  labels,
  params,
  hint: { en: "Switch worlds, then change one factor at a time.", zh: "切换世界，然后每次只改变一个因素。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  update(p, key) {
    if (key === "world") return { ...p, ...PRESETS[p.world] };
    return p;
  },
  actions: [{ label: { en: "New draw", zh: "重新抽取" }, run: (p) => ({ ...p, seed: (p.seed % 999) + 1 }) }],
});
