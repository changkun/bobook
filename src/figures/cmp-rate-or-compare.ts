// Rate or compare? The reader judges gray patches in two ways. In the rating
// task a single patch appears on a dark or a light surround, and the reader
// gives it a number from 1 (darkest) to 9 (lightest). In the comparison task
// two patches appear side by side on one shared surround, and the reader says
// which is lighter. The results score both tasks on the same question: for two
// patches one or two levels apart, how often is the lighter one ranked higher?
// Ratings answer it across trials, made at different moments on different
// surrounds; a comparison answers it within one trial, where anything that
// shifts both patches equally cancels. A simulated observer with illustrative
// noise shows the pattern the chapter predicts.

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { normal, rng } from "./lib/random.ts";
import { tpl } from "./lib/format.ts";

const labels = {
  en: {
    rateQ: "How light is the patch? 1 = darkest, 9 = lightest",
    cmpQ: "Which patch is lighter?",
    simRate: "The simulated observer rates this patch:",
    simCmp: "The simulated observer compares these patches:",
    darkest: "darkest",
    lightest: "lightest",
    progress: "{r} rating{r:/s} · {c} comparison{c:/s}",
    simNote: "simulated answers, illustrative noise",
    agreeTitle: "Lighter patch ranked higher",
    gap1: "1 level apart",
    gap2: "2 levels apart",
    ratings: "ratings",
    comparisons: "comparisons",
    pairs: "n = {n}",
    scatterTitle: "Your ratings against the true level",
    simScatterTitle: "Ratings against the true level",
    trueLevel: "true level",
    rating: "rating",
    onDark: "on dark surround",
    onLight: "on light surround",
    shift: "dark surround adds {v} points",
    shiftNone: "surround effect: needs 3+ ratings on each",
    empty1: "Answer some trials of each kind;",
    empty2: "the scores appear here.",
    describe: "{r} rating{r:/s} and {c} comparison{c:/s}. For patches one level apart, ratings ranked the lighter one higher in {a1} of pairs; comparisons picked it in {b1}.",
    describeEmpty: "No answers yet. The current trial asks you to {task}.",
    taskRate: "rate one patch",
    taskCmp: "pick the lighter of two patches",
    left: "Left",
    right: "Right",
  },
  zh: {
    rateQ: "这块色块有多亮？1 = 最暗，9 = 最亮",
    cmpQ: "哪块色块更亮？",
    simRate: "模拟观察者为这块色块评分：",
    simCmp: "模拟观察者比较这两块色块：",
    darkest: "最暗",
    lightest: "最亮",
    progress: "{r} 次评分 · {c} 次比较",
    simNote: "模拟回答，噪声为示意值",
    agreeTitle: "较亮色块排得更高的比例",
    gap1: "相差 1 级",
    gap2: "相差 2 级",
    ratings: "评分",
    comparisons: "比较",
    pairs: "n = {n}",
    scatterTitle: "你的评分与真实等级",
    simScatterTitle: "评分与真实等级",
    trueLevel: "真实等级",
    rating: "评分",
    onDark: "深色背景上",
    onLight: "浅色背景上",
    shift: "深色背景使评分偏移 {v} 分",
    shiftNone: "背景效应：每种背景至少需要 3 次评分",
    empty1: "两种试次各回答一些，",
    empty2: "得分会显示在这里。",
    describe: "{r} 次评分，{c} 次比较。对相差 1 级的色块，评分在 {a1} 的色块对中把较亮的一块排得更高；比较在 {b1} 的色块对中选中了它。",
    describeEmpty: "尚无回答。当前试次请你{task}。",
    taskRate: "为一块色块评分",
    taskCmp: "从两块色块中选出较亮的一块",
    left: "左",
    right: "右",
  },
};

const params = {
  task: { kind: "choice", label: { en: "Task", zh: "任务" }, options: [{ value: "rate", label: { en: "Rate one patch", zh: "为一块评分" } }, { value: "compare", label: { en: "Compare two", zh: "比较两块" } }], default: "rate", control: "buttons" },
  who: { kind: "choice", label: { en: "Who answers", zh: "回答者" }, options: [{ value: "you", label: { en: "You", zh: "你" } }, { value: "sim", label: { en: "Simulated observer", zh: "模拟观察者" } }], default: "you" },
  ratings: { kind: "data", label: { en: "Ratings", zh: "评分" }, default: "", validate: (v: string) => (v === "" || /^[1-9][dl]:[1-9](,[1-9][dl]:[1-9])*$/.test(v) ? undefined : "expected entries like 5d:6 (level, surround d/l, rating)") },
  compares: { kind: "data", label: { en: "Comparisons", zh: "比较" }, default: "", validate: (v: string) => (v === "" || /^[1-9][1-9][dl]:[LR](,[1-9][1-9][dl]:[LR])*$/.test(v) ? undefined : "expected entries like 46d:R (left level, right level, surround, pick)") },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 35, step: 1, control: false },
} as const;

type P = { task: "rate" | "compare"; who: "you" | "sim"; ratings: string; compares: string; seed: number };
type Surround = "d" | "l";
interface Rated { level: number; s: Surround; r: number }
interface Compared { left: number; right: number; s: Surround; pick: "L" | "R" }

const LEVELS = 9;
const MAX = 60; // answers kept per task
const SIM_N = 18;
// Lightness of each level (percent) and of the two surrounds. The stimuli are
// data, not chrome, so they are fixed grays rather than theme tokens.
const light = (level: number) => 20 + (level - 1) * 7;
const gray = (l: number) => `hsl(0, 0%, ${l}%)`;
const SURROUND: Record<Surround, number> = { d: 9, l: 91 };

const parseRated = (s: string): Rated[] => (s ? s.split(",").map((t) => ({ level: Number(t[0]), s: t[1] as Surround, r: Number(t[3]) })) : []);
const fmtRated = (xs: Rated[]) => xs.slice(-MAX).map((x) => `${x.level}${x.s}:${x.r}`).join(",");
const parseCompared = (s: string): Compared[] => (s ? s.split(",").map((t) => ({ left: Number(t[0]), right: Number(t[1]), s: t[2] as Surround, pick: t[4] as "L" | "R" })) : []);
const fmtCompared = (xs: Compared[]) => xs.slice(-MAX).map((x) => `${x.left}${x.right}${x.s}:${x.pick}`).join(",");

// The i-th rating trial: levels in shuffled blocks of nine, so every level
// appears equally often; the surround is a fair coin.
function rateTrial(seed: number, i: number): { level: number; s: Surround } {
  const block = Math.floor(i / LEVELS);
  const r = rng(seed * 7919 + block * 104729 + 1);
  const perm = Array.from({ length: LEVELS }, (_, k) => k + 1);
  for (let k = perm.length - 1; k > 0; k--) { const j = Math.floor(r() * (k + 1)); [perm[k], perm[j]] = [perm[j], perm[k]]; }
  const rs = rng(seed * 31 + i * 977 + 3);
  rs();
  return { level: perm[i % LEVELS], s: rs() < 0.5 ? "d" : "l" };
}

// The i-th comparison trial: two levels one or two apart, sides and surround
// at random.
function cmpTrial(seed: number, i: number): { left: number; right: number; s: Surround } {
  const r = rng(seed * 104723 + i * 7907 + 5);
  r();
  const gap = r() < 0.55 ? 1 : 2;
  const lo = 1 + Math.floor(r() * (LEVELS - gap));
  const hi = lo + gap;
  const lighterLeft = r() < 0.5;
  const s: Surround = r() < 0.5 ? "d" : "l";
  return { left: lighterLeft ? hi : lo, right: lighterLeft ? lo : hi, s };
}

// The simulated observer (illustrative parameters). A rating reads the patch
// on its own: the true level, plus a shift from the surround (a gray looks
// lighter on a dark surround), plus a slow drift in how the scale is used,
// plus noise, rounded to the scale. A comparison reads the difference of two
// patches seen together: the surround and the drift shift both equally and
// cancel, leaving a smaller noise.
const SIM = { context: 0.6, drift: 0.25, rateNoise: 0.8, cmpNoise: 0.5 };

function simRating(seed: number, i: number, level: number, s: Surround): number {
  let drift = 0;
  const zd = normal(rng(seed * 613 + 17));
  for (let k = 0; k <= i; k++) drift += SIM.drift * zd();
  const z = normal(rng(seed * 4241 + i * 131 + 11))();
  const v = level + (s === "d" ? SIM.context : -SIM.context) + drift + SIM.rateNoise * z;
  return Math.min(LEVELS, Math.max(1, Math.round(v)));
}

function simPick(seed: number, i: number, left: number, right: number): "L" | "R" {
  const z = normal(rng(seed * 2909 + i * 173 + 29))();
  return left - right + SIM.cmpNoise * z > 0 ? "L" : "R";
}

function simulate(p: P): P {
  const rs = parseRated(p.ratings), cs = parseCompared(p.compares);
  for (let k = 0; k < SIM_N; k++) {
    const i = rs.length;
    const t = rateTrial(p.seed, i);
    rs.push({ ...t, r: simRating(p.seed, i, t.level, t.s) });
    const j = cs.length;
    const c = cmpTrial(p.seed, j);
    cs.push({ ...c, pick: simPick(p.seed, j, c.left, c.right) });
  }
  return { ...p, ratings: fmtRated(rs), compares: fmtCompared(cs) };
}

function rate(p: P, r: number): P {
  const rs = parseRated(p.ratings);
  const t = rateTrial(p.seed, rs.length);
  rs.push({ ...t, r });
  return { ...p, ratings: fmtRated(rs) };
}

function pick(p: P, side: "L" | "R"): P {
  const cs = parseCompared(p.compares);
  const t = cmpTrial(p.seed, cs.length);
  cs.push({ ...t, pick: side });
  return { ...p, compares: fmtCompared(cs) };
}

interface Tally { k: number; n: number }

// For every pair of rated patches whose levels differ by `gap`, did the
// lighter one get the higher rating? Ties count half.
function scoreRatings(rs: Rated[]): { g: [Tally, Tally]; shift: number | null } {
  const g: [Tally, Tally] = [{ k: 0, n: 0 }, { k: 0, n: 0 }];
  for (let i = 0; i < rs.length; i++) for (let j = i + 1; j < rs.length; j++) {
    const d = rs[i].level - rs[j].level;
    const gap = Math.abs(d);
    if (gap !== 1 && gap !== 2) continue;
    const t = g[gap - 1];
    t.n++;
    const agree = (rs[i].r - rs[j].r) * d;
    t.k += agree > 0 ? 1 : agree === 0 ? 0.5 : 0;
  }
  // Surround effect: least squares of rating on [1, level, dark].
  const nd = rs.filter((x) => x.s === "d").length, nl = rs.length - nd;
  let shift: number | null = null;
  if (nd >= 3 && nl >= 3) {
    const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b = [0, 0, 0];
    for (const x of rs) {
      const row = [1, x.level, x.s === "d" ? 1 : 0];
      for (let a = 0; a < 3; a++) { b[a] += row[a] * x.r; for (let c = 0; c < 3; c++) A[a][c] += row[a] * row[c]; }
    }
    const sol = solve3(A, b);
    if (sol) shift = sol[2];
  }
  return { g, shift };
}

function solve3(A: number[][], b: number[]): number[] | null {
  const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < 3; c++) {
    let p = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-9) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < 3; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k <= 3; k++) M[r][k] -= f * M[c][k]; }
  }
  return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]];
}

function scoreCompares(cs: Compared[]): [Tally, Tally] {
  const g: [Tally, Tally] = [{ k: 0, n: 0 }, { k: 0, n: 0 }];
  for (const c of cs) {
    const gap = Math.abs(c.left - c.right);
    if (gap !== 1 && gap !== 2) continue;
    g[gap - 1].n++;
    if ((c.pick === "L") === (c.left > c.right)) g[gap - 1].k++;
  }
  return g;
}

const pctOf = (t: Tally) => (t.n ? `${Math.round((100 * t.k) / t.n)}%` : "–");

function describe(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const rs = parseRated(p.ratings), cs = parseCompared(p.compares);
  if (!rs.length && !cs.length) return tpl(L.describeEmpty, { task: p.task === "rate" ? L.taskRate : L.taskCmp });
  const sr = scoreRatings(rs), sc = scoreCompares(cs);
  return tpl(L.describe, { r: rs.length, c: cs.length, a1: pctOf(sr.g[0]), b1: pctOf(sc[0]) });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const w = st.w;
  const narrow = w < 480;
  const you = p.who === "you";
  const rs = parseRated(p.ratings), cs = parseCompared(p.compares);
  const parts: string[] = [];

  // 1. The trial.
  const qText = p.task === "rate" ? (you ? L.rateQ : L.simRate) : (you ? L.cmpQ : L.simCmp);
  parts.push(text(w / 2, 20, qText, { "text-anchor": "middle", "font-size": narrow ? TYPE.body : TYPE.label, class: "fig-t-strong" }));
  const boxTop = 32;
  const boxH = narrow ? 104 : 120;
  const patch = narrow ? 50 : 60;
  if (p.task === "rate") {
    const t = rateTrial(p.seed, rs.length);
    const bw = Math.min(260, w - 32);
    const bx = (w - bw) / 2;
    parts.push(
      el("rect", { x: bx, y: boxTop, width: bw, height: boxH, rx: 8, fill: gray(SURROUND[t.s]), stroke: C.rule, "stroke-width": 1 }),
      el("rect", { x: w / 2 - patch / 2, y: boxTop + boxH / 2 - patch / 2, width: patch, height: patch, fill: gray(light(t.level)) }),
    );
    // the rating scale
    const cell = Math.min(36, (w - 24) / LEVELS - 4);
    const sx = w / 2 - (LEVELS * (cell + 4) - 4) / 2;
    const sy = boxTop + boxH + 10;
    for (let k = 1; k <= LEVELS; k++) {
      const x = sx + (k - 1) * (cell + 4);
      parts.push(g({ "data-fig-hit": you ? `r-${k}` : undefined },
        el("rect", { x, y: sy, width: cell, height: 28, rx: 5, fill: C.panel, stroke: C.rule, "stroke-width": 1 }),
        text(x + cell / 2, sy + 18, k, { "text-anchor": "middle", "font-size": TYPE.label, class: "fig-t-num" }),
      ));
    }
    parts.push(
      text(sx, sy + 42, L.darkest, { "font-size": TYPE.small, class: "fig-t-muted" }),
      text(sx + LEVELS * (cell + 4) - 4, sy + 42, L.lightest, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted" }),
    );
  } else {
    const t = cmpTrial(p.seed, cs.length);
    const bw = Math.min(420, w - 32);
    const bx = (w - bw) / 2;
    parts.push(el("rect", { x: bx, y: boxTop, width: bw, height: boxH, rx: 8, fill: gray(SURROUND[t.s]), stroke: C.rule, "stroke-width": 1 }));
    const py = boxTop + boxH / 2 - patch / 2;
    const cxL = bx + bw / 3, cxR = bx + (2 * bw) / 3;
    parts.push(
      el("rect", { x: cxL - patch / 2, y: py, width: patch, height: patch, fill: gray(light(t.left)), "data-fig-hit": you ? "L" : undefined }),
      el("rect", { x: cxR - patch / 2, y: py, width: patch, height: patch, fill: gray(light(t.right)), "data-fig-hit": you ? "R" : undefined }),
      text(cxL, boxTop + boxH + 18, L.left, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }),
      text(cxR, boxTop + boxH + 18, L.right, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }),
    );
  }
  let progY = boxTop + boxH + (p.task === "rate" ? 72 : 44);
  const prog = tpl(L.progress, { r: rs.length, c: cs.length });
  if (you || !narrow) {
    parts.push(text(w / 2, progY, prog + (you ? "" : ` · ${L.simNote}`), { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  } else {
    parts.push(
      text(w / 2, progY, prog, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }),
      text(w / 2, progY + 15, L.simNote, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }),
    );
    progY += 15;
  }

  // 2. The scores.
  const resTop = progY + 22;
  parts.push(el("line", { x1: 8, x2: w - 8, y1: resTop - 8, y2: resTop - 8, stroke: C.grid }));
  const sr = scoreRatings(rs), sc = scoreCompares(cs);
  const panelH = narrow ? 150 : 170;
  const gapX = 28;
  const leftW = narrow ? w : (w - gapX) * 0.5;
  // 2a. Bars: the share of pairs ranked correctly, ratings against comparisons.
  const bLeft = narrow ? 44 : 50, bRight = leftW - 10;
  const bTop = resTop + 26, bBot = bTop + panelH - 40;
  const y = linear([0, 1], [bBot, bTop]);
  parts.push(
    text(bLeft - 34, resTop + 8, L.agreeTitle, { "font-size": TYPE.body, class: "fig-t-strong" }),
    axis({ scale: y, orient: "left", at: bLeft, span: [bLeft, bRight], count: 4, format: (v) => `${Math.round(v * 100)}%` }),
    el("line", { x1: bLeft, x2: bRight, y1: bBot, y2: bBot, stroke: C.rule }),
  );
  // the chance line: a coin flip ranks half the pairs correctly
  parts.push(el("line", { x1: bLeft, x2: bRight, y1: y(0.5), y2: y(0.5), stroke: C.ink3, "stroke-dasharray": "3 3" }));
  const groupW = (bRight - bLeft) / 2;
  const barW = Math.min(36, groupW / 3.2);
  ([0, 1] as const).forEach((gi) => {
    const gx = bLeft + groupW * (gi + 0.5);
    const series: Array<[Tally, string]> = [[sr.g[gi], C.c5], [sc[gi], C.c6]];
    series.forEach(([t, color], si) => {
      const x = gx + (si === 0 ? -barW - 3 : 3);
      if (t.n) {
        const v = t.k / t.n;
        parts.push(el("rect", { x, y: y(v), width: barW, height: bBot - y(v), rx: 2, fill: color, opacity: 0.88 }));
        parts.push(text(x + barW / 2, y(v) - 4, pctOf(t), { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-num fig-t-halo" }));
      }
      parts.push(text(x + barW / 2, bBot + 13, tpl(L.pairs, { n: t.n }), { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-faint fig-t-num" }));
    });
    parts.push(text(gx, bBot + 28, gi === 0 ? L.gap1 : L.gap2, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }));
  });
  // legend for the bars
  const lgY = bBot + 46;
  parts.push(
    el("rect", { x: bLeft, y: lgY - 9, width: 12, height: 10, rx: 2, fill: C.c5 }),
    text(bLeft + 17, lgY, L.ratings, { "font-size": TYPE.small, class: "fig-t-muted" }),
    el("rect", { x: bLeft + 80, y: lgY - 9, width: 12, height: 10, rx: 2, fill: C.c6 }),
    text(bLeft + 97, lgY, L.comparisons, { "font-size": TYPE.small, class: "fig-t-muted" }),
  );
  if (!rs.length && !cs.length) {
    parts.push(text((bLeft + bRight) / 2 + 6, y(0.78), L.empty1, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-faint fig-t-halo" }));
    parts.push(text((bLeft + bRight) / 2 + 6, y(0.78) + 15, L.empty2, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-faint fig-t-halo" }));
  }

  // 2b. Scatter: each rating against the patch's true level.
  const sTopBase = narrow ? lgY + 34 : resTop;
  const sx0 = narrow ? 44 : leftW + gapX + 40;
  const sx1 = w - 12;
  const sTop = sTopBase + 26, sBot = sTop + panelH - 40;
  const xs = linear([0.5, LEVELS + 0.5], [sx0, sx1]);
  const ys = linear([0.5, LEVELS + 0.5], [sBot, sTop]);
  const ticks = [1, 3, 5, 7, 9];
  parts.push(
    text(sx0 - 34, sTopBase + 8, you ? L.scatterTitle : L.simScatterTitle, { "font-size": TYPE.body, class: "fig-t-strong" }),
    axis({ scale: ys, orient: "left", at: sx0, span: [sx0, sx1], ticks, title: L.rating }),
    axis({ scale: xs, orient: "bottom", at: sBot, span: [sTop, sBot], ticks }),
    text((sx0 + sx1) / 2, sBot + 30, L.trueLevel, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }),
    el("line", { x1: xs(0.5), y1: ys(0.5), x2: xs(LEVELS + 0.5), y2: ys(LEVELS + 0.5), stroke: C.ink3, "stroke-dasharray": "4 3" }),
  );
  // jitter repeated (level, rating) cells so every answer stays visible
  const seen = new Map<string, number>();
  for (const x of rs) {
    const key = `${x.level}:${x.r}`;
    const k = seen.get(key) ?? 0;
    seen.set(key, k + 1);
    const ang = k * 2.4, rad = k ? 3 + k * 1.2 : 0;
    const cx = xs(x.level) + rad * Math.cos(ang), cy = ys(x.r) + rad * Math.sin(ang);
    parts.push(x.s === "d"
      ? el("circle", { cx, cy, r: 3.6, fill: C.ink, stroke: C.paper, "stroke-width": 1.2 })
      : el("circle", { cx, cy, r: 3.4, fill: C.paper, stroke: C.ink, "stroke-width": 1.4 }));
  }
  const lg2 = sBot + 46;
  parts.push(
    el("circle", { cx: sx0 + 4, cy: lg2 - 4, r: 3.6, fill: C.ink }),
    text(sx0 + 12, lg2, L.onDark, { "font-size": TYPE.small, class: "fig-t-muted" }),
    el("circle", { cx: sx0 + 122, cy: lg2 - 4, r: 3.4, fill: C.paper, stroke: C.ink, "stroke-width": 1.4 }),
    text(sx0 + 130, lg2, L.onLight, { "font-size": TYPE.small, class: "fig-t-muted" }),
  );
  const shiftText = sr.shift === null ? L.shiftNone : tpl(L.shift, { v: `${sr.shift >= 0 ? "+" : "−"}${Math.abs(sr.shift).toFixed(1)}` });
  parts.push(text(sx0, lg2 + 17, shiftText, { "font-size": TYPE.small, class: sr.shift === null ? "fig-t-faint" : "fig-t-strong fig-t-num" }));
  const H = lg2 + 26;
  return svg(w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "cmp-rate-or-compare",
  title: { en: "Rate or compare: two ways to judge the same gray patches", zh: "评分还是比较：判断同一组灰色色块的两种方式" },
  labels,
  params,
  hint: { en: "Click a number to rate, or click the lighter patch; the buttons do the same. Do a dozen or more of each task.", zh: "点击数字评分，或点击较亮的色块；按钮的作用相同。每种任务各做十几次或更多。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click" || p.who !== "you") return null;
    const q = p as P;
    if (q.task === "rate" && e.target?.startsWith("r-")) return rate(q, Number(e.target.slice(2)));
    if (q.task === "compare" && (e.target === "L" || e.target === "R")) return pick(q, e.target);
    return null;
  },
  actions: [
    ...Array.from({ length: LEVELS }, (_, k) => ({
      label: { en: String(k + 1), zh: String(k + 1) },
      run: (p: P) => rate(p, k + 1),
      enabled: (p: P) => p.who === "you" && p.task === "rate",
    })),
    { label: { en: "Left is lighter", zh: "左边更亮" }, primary: true, run: (p) => pick(p as P, "L"), enabled: (p) => p.who === "you" && p.task === "compare" },
    { label: { en: "Right is lighter", zh: "右边更亮" }, primary: true, run: (p) => pick(p as P, "R"), enabled: (p) => p.who === "you" && p.task === "compare" },
    { label: { en: "Simulate 18 of each", zh: "各模拟 18 次" }, primary: true, run: (p) => simulate(p as P), enabled: (p) => p.who === "sim" },
    {
      label: { en: "Undo", zh: "撤销" },
      run: (p) => {
        const q = p as P;
        if (q.task === "rate") return { ...q, ratings: fmtRated(parseRated(q.ratings).slice(0, -1)) };
        return { ...q, compares: fmtCompared(parseCompared(q.compares).slice(0, -1)) };
      },
      enabled: (p) => p.who === "you" && ((p as P).task === "rate" ? p.ratings !== "" : p.compares !== ""),
    },
  ],
  update(p, key) {
    // Switching who answers starts over: the answers belong to one observer.
    if (key === "who") return { ...p, ratings: "", compares: "" };
    return p;
  },
});
