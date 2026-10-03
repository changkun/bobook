// A Bernoulli multi-armed bandit, played by three algorithms or by the
// reader. Each arm pays 1 with a fixed hidden probability. The algorithms
// (epsilon-greedy, UCB1, Thompson sampling) are run many times on the same
// arms, and the plot shows their cumulative pseudo-regret, the sum over
// rounds of the gap between the best arm's mean and the mean of the arm
// pulled, averaged over runs with a band for the spread between runs. The
// Lai-Robbins rate c* ln t is drawn dashed: it is an asymptotic statement
// about the slope on a log time axis, not a floor at every finite t. In the
// "you" mode the reader pulls arms by clicking, against the same reward
// sequences the algorithms' first run faced, and the regret curve is hidden
// until the arms are revealed, because its slope would give the best arm away.
//
// Reward sequences are "tapes": the n-th pull of arm i in run r pays
// hash(seed, r, i, n) < mu_i, so every algorithm in a run faces the same
// luck, and the comparison between them is not blurred by different draws.

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear, log as logScale, type Scale } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import type { Frame, Swatch } from "./lib/plot.ts";
import { memo, normal, rng, type Rng } from "./lib/random.ts";
import { fixed, tpl } from "./lib/format.ts";
import { parseNumbers, validNumbers } from "./lib/params.ts";

const labels = {
  en: {
    t: "round t",
    regret: "cumulative regret",
    eps: "ε-greedy",
    ucb: "UCB1",
    ts: "Thompson",
    you: "you",
    lr: "Lai-Robbins rate",
    guar: "UCB1 guarantee",
    arm: "Arm {i}",
    mean: "mean θ",
    share: "share of pulls",
    hidden: "?",
    yourStatus: "Pull {n} of {b} · total reward {r}",
    yourDone: "Done: {n} pulls, total reward {r}. The best arm pays {m} on average.",
    pulls: "{n} pull{n:/s}",
    wins: "{s} won",
    clickHint: "click an arm to pull it",
    hiddenCurve: "your regret appears when the arms are revealed",
    readout: "at T = {T}:",
    guarAt: "UCB1 guarantee: {v} at T = {T}",
    worst: "always the worst arm: {v}",
    after: "after {n} pull{n:/s}:",
    describe: "{K} Bernoulli arms with best mean {m} and gap {g} to the second best. After {T} rounds, averaged over {R} run{R:/s}, cumulative regret is {e} for ε-greedy with ε = {eps}, {u} for UCB1, and {s} for Thompson sampling.",
    describeYou: "You have pulled {n} of {b} times and won {r}. {rev}",
    revYou: "Your cumulative regret is {y}; on the same rounds UCB1 averages {u} and Thompson sampling {s}.",
    hidYou: "The arms' means are hidden.",
  },
  zh: {
    t: "轮次 t",
    regret: "累积遗憾",
    eps: "ε-贪心",
    ucb: "UCB1",
    ts: "Thompson",
    you: "你",
    lr: "Lai-Robbins 速率",
    guar: "UCB1 保证",
    arm: "臂 {i}",
    mean: "均值 θ",
    share: "拉动占比",
    hidden: "？",
    yourStatus: "已拉动 {n}/{b} 次 · 总奖励 {r}",
    yourDone: "完成：共拉动 {n} 次，总奖励 {r}。最优臂的平均奖励为 {m}。",
    pulls: "拉动 {n} 次",
    wins: "赢 {s} 次",
    clickHint: "点击一条臂即可拉动",
    hiddenCurve: "揭晓各臂之后才显示你的遗憾",
    readout: "T = {T} 时：",
    guarAt: "UCB1 保证：T = {T} 时为 {v}",
    worst: "始终拉动最差的臂：{v}",
    after: "拉动 {n} 次后：",
    describe: "{K} 条 Bernoulli 臂，最优均值为 {m}，与次优臂的差距为 {g}。{T} 轮之后，在 {R} 次运行上取平均，累积遗憾分别为：ε = {eps} 的 ε-贪心 {e}，UCB1 {u}，Thompson 采样 {s}。",
    describeYou: "你已拉动 {n} 次（共 {b} 次），总奖励为 {r}。{rev}",
    revYou: "你的累积遗憾为 {y}；在同样的轮数内，UCB1 平均为 {u}，Thompson 采样平均为 {s}。",
    hidYou: "各臂的均值尚未揭晓。",
  },
};

const params = {
  mode: { kind: "choice", label: { en: "Who pulls", zh: "由谁拉动" }, options: [{ value: "algos", label: { en: "Algorithms", zh: "算法" } }, { value: "you", label: { en: "You", zh: "你" } }], default: "algos", control: "buttons" },
  arms: { kind: "range", label: { en: "Arms K", zh: "臂数 K" }, min: 2, max: 8, default: 5, step: 1 },
  gap: { kind: "range", label: { en: "Gap Δ", zh: "差距 Δ" }, min: 0.02, max: 0.3, default: 0.1, step: 0.01 },
  eps: { kind: "range", label: { en: "ε-greedy ε", zh: "ε-贪心的 ε" }, min: 0, max: 0.5, default: 0.1, step: 0.01 },
  horizon: { kind: "range", label: { en: "Rounds T", zh: "轮数 T" }, min: 100, max: 5000, default: 1000, step: 100 },
  runs: { kind: "range", label: { en: "Runs", zh: "运行次数" }, min: 1, max: 40, default: 20, step: 1 },
  logx: { kind: "toggle", label: { en: "Log time axis", zh: "对数时间轴" }, default: false },
  guarantee: { kind: "toggle", label: { en: "Show UCB1 guarantee", zh: "显示 UCB1 保证" }, default: false },
  reveal: { kind: "toggle", label: { en: "Reveal the arms", zh: "揭晓各臂" }, default: false },
  pulls: { kind: "data", label: { en: "Your pulls", zh: "你的拉动记录" }, default: "", validate: validNumbers },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 4, step: 1, control: false },
} as const;

type P = {
  mode: "algos" | "you"; arms: number; gap: number; eps: number; horizon: number; runs: number;
  logx: boolean; guarantee: boolean; reveal: boolean; pulls: string; seed: number;
};

const BUDGET = 50; // pulls in the "you" mode
const MAX_ARMS = 8;
const ALGS = ["eps", "ucb", "ts"] as const;
type Alg = typeof ALGS[number];
const COLOR: Record<Alg | "you", string> = { eps: C.c5, ucb: C.c2, ts: C.c6, you: C.ink };

// ---------------------------------------------------------------------------
// The bandit

// Arm means: the best pays 0.6, the second best 0.6 - gap, the rest evenly
// down to 0.15. Positions are shuffled by the seed, so the best arm is not
// always the first one.
function armMeans(K: number, gap: number, seed: number): number[] {
  const top = 0.6, second = top - gap, low = 0.15;
  const sorted = [top];
  for (let j = 1; j < K; j++) sorted.push(K === 2 ? second : second - ((j - 1) / (K - 2)) * (second - low));
  const r = rng(seed * 7 + 3);
  const perm = Array.from({ length: K }, (_, i) => i);
  for (let i = K - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
  const out = new Array<number>(K);
  perm.forEach((pos, rank) => { out[pos] = sorted[rank]; });
  return out;
}

// A 32-bit integer mix (the MurmurHash3 finalizer), chained over the key
// parts, turned into a uniform number on [0, 1).
function fmix(h: number): number {
  h ^= h >>> 16; h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13; h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}
function u01(a: number, b: number, c: number, d: number): number {
  let h = fmix((a * 0x9e3779b1) >>> 0);
  h = fmix((h ^ Math.imul(b + 0x632be5ab, 0x85ebca77)) >>> 0);
  h = fmix((h ^ Math.imul(c + 0x2f2a3c1d, 0xc2b2ae3d)) >>> 0);
  h = fmix((h ^ Math.imul(d + 0x165667b1, 0x27d4eb2f)) >>> 0);
  return h / 4294967296;
}
const reward = (seed: number, run: number, arm: number, n: number, mu: number) => (u01(seed, run, arm, n) < mu ? 1 : 0);

// Gamma(a) for a >= 1 by Marsaglia and Tsang (2000); Beta(a, b) from two Gammas.
function gamma(a: number, r: Rng, z: () => number): number {
  const d = a - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    const x = z();
    let v = 1 + c * x;
    if (v <= 0) continue;
    v = v * v * v;
    const u = r();
    if (u < 1 - 0.0331 * x ** 4) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}
function beta(a: number, b: number, r: Rng, z: () => number): number {
  const x = gamma(a, r, z), y = gamma(b, r, z);
  return x / (x + y);
}

// Bernoulli KL divergence kl(p, q), for the Lai-Robbins constant.
function kl(p: number, q: number): number {
  const t = (a: number, b: number) => (a <= 0 ? 0 : a * Math.log(a / b));
  return t(p, q) + t(1 - p, 1 - q);
}

function constants(mu: number[]) {
  const best = Math.max(...mu);
  let lr = 0, g8 = 0, gsum = 0;
  for (const m of mu) {
    const d = best - m;
    gsum += d;
    if (d > 1e-12) { lr += d / kl(m, best); g8 += 8 / d; }
  }
  return { best, lr, ucbLog: g8, ucbConst: (1 + Math.PI ** 2 / 3) * gsum };
}

// ---------------------------------------------------------------------------
// The algorithms

function argmaxTie(v: number[], r: Rng): number {
  let best = -Infinity, idx: number[] = [];
  for (let i = 0; i < v.length; i++) {
    if (v[i] > best + 1e-12) { best = v[i]; idx = [i]; } else if (Math.abs(v[i] - best) <= 1e-12) idx.push(i);
  }
  return idx.length === 1 ? idx[0] : idx[Math.floor(r() * idx.length)];
}

// One run of one algorithm: the arm pulled in each round.
function play(alg: Alg, mu: number[], T: number, eps: number, seed: number, run: number): Int8Array {
  const K = mu.length;
  const n = new Array<number>(K).fill(0), s = new Array<number>(K).fill(0);
  const r = rng(seed * 1009 + run * 31 + ALGS.indexOf(alg) + 1);
  const z = normal(rng(seed * 2003 + run * 17 + 5));
  const arms = new Int8Array(T);
  for (let t = 0; t < T; t++) {
    let a: number;
    if (alg !== "ts" && t < K) a = t; // epsilon-greedy and UCB1 start by pulling each arm once
    else if (alg === "eps") a = r() < eps ? Math.floor(r() * K) : argmaxTie(n.map((c, i) => s[i] / c), r);
    else if (alg === "ucb") a = argmaxTie(n.map((c, i) => s[i] / c + Math.sqrt((2 * Math.log(t)) / c)), r);
    else a = argmaxTie(n.map((c, i) => beta(1 + s[i], 1 + c - s[i], r, z)), r);
    s[a] += reward(seed, run, a, n[a], mu[a]);
    n[a] += 1;
    arms[t] = a;
  }
  return arms;
}

interface Curves {
  ts: number[]; // the rounds at which curves are sampled
  mean: Record<Alg, number[]>;
  lo: Record<Alg, number[]>;
  hi: Record<Alg, number[]>;
  share: Record<Alg, number[]>; // average share of pulls per arm
  final: Record<Alg, number>;
}

function sampleTimes(T: number): number[] {
  // Every round for short horizons; otherwise about 240 points, denser early
  // so that a log time axis is smooth too.
  if (T <= 240) return Array.from({ length: T }, (_, i) => i + 1);
  const set = new Set<number>();
  for (let i = 0; i < 120; i++) set.add(Math.max(1, Math.round(T ** (i / 119))));
  for (let i = 0; i <= 120; i++) set.add(Math.max(1, Math.round((T * i) / 120)));
  return [...set].sort((a, b) => a - b);
}

const simulate = memo((K: number, gap: number, eps: number, T: number, R: number, seed: number): Curves => {
  const mu = armMeans(K, gap, seed);
  const best = Math.max(...mu);
  const ts = sampleTimes(T);
  const mk = () => ({ eps: [] as number[], ucb: [] as number[], ts: [] as number[] });
  const mean = mk(), lo = mk(), hi = mk();
  const share = { eps: new Array<number>(K).fill(0), ucb: new Array<number>(K).fill(0), ts: new Array<number>(K).fill(0) };
  const final = { eps: 0, ucb: 0, ts: 0 };
  for (const alg of ALGS) {
    const runs: number[][] = [];
    for (let run = 0; run < R; run++) {
      const arms = play(alg, mu, T, eps, seed, run);
      const cum: number[] = [];
      let c = 0, j = 0;
      for (let t = 0; t < T; t++) {
        c += best - mu[arms[t]];
        share[alg][arms[t]] += 1 / (T * R);
        if (t + 1 === ts[j]) { cum.push(c); j++; }
      }
      runs.push(cum);
    }
    for (let j = 0; j < ts.length; j++) {
      const col = runs.map((c) => c[j]).sort((a, b) => a - b);
      mean[alg].push(col.reduce((a, b) => a + b, 0) / R);
      lo[alg].push(col[Math.floor(0.1 * (R - 1))]);
      hi[alg].push(col[Math.ceil(0.9 * (R - 1))]);
    }
    final[alg] = mean[alg][ts.length - 1];
  }
  return { ts, mean, lo, hi, share, final };
}, 12);

// The reader's pulls, against the tapes of run 0.
function yours(p: P) {
  const K = p.arms;
  const mu = armMeans(K, p.gap, p.seed);
  const best = Math.max(...mu);
  const pulls = parseNumbers(p.pulls).filter((a) => Number.isInteger(a) && a >= 0 && a < K).slice(0, BUDGET);
  const n = new Array<number>(K).fill(0), s = new Array<number>(K).fill(0);
  const cum: number[] = [];
  let c = 0, total = 0;
  for (const a of pulls) {
    const w = reward(p.seed, 0, a, n[a], mu[a]);
    s[a] += w; n[a] += 1; total += w;
    c += best - mu[a];
    cum.push(c);
  }
  return { mu, best, pulls, n, s, cum, total };
}

// ---------------------------------------------------------------------------
// Rendering

const algLabel = (a: Alg, L: typeof labels.en = labels.en) => L[a];

// Characters drawn at full width (CJK and full-width punctuation): about 11 px
// at this font size, against 6.6 px for Latin. Zero for an English label.
const wide = (s: string) => (s.match(/[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]/g) ?? []).length;

// A legend row like lib/plot.ts legend, with more room per entry so that a
// label never runs into the next swatch.
function legend(x: number, y: number, w: number, items: Swatch[]): { svg: string; height: number } {
  const parts: string[] = [];
  let cx = x, cy = y;
  const rowH = 18;
  for (const it of items) {
    const tw = it.label.length * 6.6 + wide(it.label) * 4.6 + 40;
    if (cx + tw > x + w && cx > x) { cx = x; cy += rowH; }
    parts.push(
      el("line", { x1: cx, x2: cx + 18, y1: cy - 1, y2: cy - 1, stroke: it.color, "stroke-width": 2, "stroke-dasharray": it.kind === "dash" ? "4 3" : undefined }),
      text(cx + 24, cy + 3, it.label, { "font-size": TYPE.small, class: "fig-t-muted" }),
    );
    cx += tw;
  }
  return { svg: g({}, ...parts), height: cy - y + rowH };
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const you = p.mode === "you";
  const K = p.arms;
  const T = you ? BUDGET : p.horizon;
  const R = p.runs;
  const sim = simulate(K, p.gap, p.eps, T, R, p.seed);
  const mine = yours(p);
  const done = mine.pulls.length >= BUDGET;
  const shown = !you || p.reveal || done;
  const k = constants(mine.mu);
  const parts: string[] = [];

  // Legend.
  const items: Swatch[] = [
    ...(you ? [{ kind: "line" as const, color: COLOR.you, label: L.you }] : []),
    ...ALGS.map((a) => ({ kind: "line" as const, color: COLOR[a], label: algLabel(a, L) })),
    ...(you ? [] : [{ kind: "dash" as const, color: C.ink3, label: L.lr }]),
    ...(p.guarantee && !you ? [{ kind: "dash" as const, color: C.c2, label: L.guar }] : []),
  ];
  const lg = legend(narrow ? 8 : 40, 14, st.w - 16, items);
  parts.push(lg.svg);

  // The arms table.
  const tTop = 14 + lg.height + 4;
  const rowH = narrow ? 22 : 24;
  const left = narrow ? 8 : 40;
  const right = st.w - 10;
  const nameW = narrow ? 42 : 52;
  const meanW = narrow ? 70 : 110;
  const x0 = left + nameW, x1 = x0 + meanW + 8;
  const colW = (right - x1) / 3;
  // headers
  parts.push(text(x0, tTop + 10, L.mean, { "font-size": TYPE.small, class: "fig-t-muted" }));
  if (!you) {
    parts.push(text(x1, tTop + 10, L.share, { "font-size": TYPE.small, class: "fig-t-muted" }));
  } else {
    parts.push(text(x1, tTop + 10, done ? tpl(L.yourStatus, { n: mine.pulls.length, b: BUDGET, r: mine.total }) : tpl(L.yourStatus, { n: mine.pulls.length, b: BUDGET, r: mine.total }), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  }
  const rTop = tTop + 18;
  const bestIdx = mine.mu.indexOf(k.best);
  for (let i = 0; i < K; i++) {
    const y = rTop + i * rowH;
    const cy = y + rowH / 2;
    const isBest = i === bestIdx && shown;
    if (you) {
      parts.push(el("rect", { x: left - 4, y: y + 1, width: right - left + 8, height: rowH - 2, rx: 5, fill: C.panel, stroke: C.rule, "stroke-width": 1, "data-fig-hit": `arm-${i}` }));
    } else if (i % 2 === 0) {
      parts.push(el("rect", { x: left - 4, y: y + 1, width: right - left + 8, height: rowH - 2, rx: 4, fill: C.panel }));
    }
    parts.push(text(left, cy + 4, tpl(L.arm, { i: i + 1 }), { "font-size": TYPE.small, class: isBest ? "fig-t-strong" : "", "pointer-events": "none" }));
    // the mean bar
    const bw = meanW - 34;
    parts.push(el("rect", { x: x0, y: cy - 4, width: bw, height: 8, rx: 2, fill: C.grid, "pointer-events": "none" }));
    if (shown) {
      parts.push(el("rect", { x: x0, y: cy - 4, width: bw * mine.mu[i], height: 8, rx: 2, fill: isBest ? C.ink : C.ink3, "pointer-events": "none" }));
      parts.push(text(x0 + bw + 4, cy + 4, fixed(mine.mu[i], 2), { "font-size": TYPE.small, class: "fig-t-num fig-t-muted", "pointer-events": "none" }));
    } else {
      parts.push(text(x0 + bw / 2, cy + 4, L.hidden, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-faint", "pointer-events": "none" }));
    }
    if (!you) {
      ALGS.forEach((a, j) => {
        const sx = x1 + j * colW;
        const sw = colW - (narrow ? 26 : 40);
        const v = sim.share[a][i];
        parts.push(el("rect", { x: sx, y: cy - 4, width: Math.max(1, sw * v), height: 8, rx: 2, fill: COLOR[a] }));
        parts.push(text(sx + sw * v + 3, cy + 4, `${Math.round(v * 100)}%`, { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
      });
    } else {
      const n = mine.n[i], s = mine.s[i];
      const tx = x1;
      parts.push(text(tx, cy + 4, `${tpl(L.pulls, { n })} · ${tpl(L.wins, { s })}`, { "font-size": TYPE.small, class: "fig-t-num", "pointer-events": "none" }));
      // a row of small squares for the outcomes of the last pulls of this arm
      const maxSq = narrow ? 8 : 16;
      const outcomes: number[] = [];
      { const cnt = new Array<number>(K).fill(0); for (const a of mine.pulls) { if (a === i) outcomes.push(reward(p.seed, 0, a, cnt[a], mine.mu[a])); cnt[a]++; } }
      const show = outcomes.slice(-maxSq);
      const sq0 = right - show.length * 9;
      show.forEach((w, q) => parts.push(el("rect", { x: sq0 + q * 9, y: cy - 3.5, width: 7, height: 7, rx: 1.5, fill: w ? C.c6 : "none", stroke: w ? "none" : C.ink3, "stroke-width": 1, "pointer-events": "none" })));
    }
  }

  // The regret plot.
  const pTop = rTop + K * rowH + (narrow ? 26 : 30);
  const pH = narrow ? 170 : 220;
  const fl = narrow ? 40 : 52;
  const xs: Scale = p.logx ? logScale([1, T], [fl, st.w - 14]) : linear([0, T], [fl, st.w - 14]);
  // y range: the largest plotted curve, including the reader's.
  let ymax = 1;
  for (const a of ALGS) ymax = Math.max(ymax, ...sim.hi[a]);
  if (you && shown && mine.cum.length) ymax = Math.max(ymax, ...mine.cum);
  if (!you) ymax = Math.max(ymax, k.lr * Math.log(T));
  // With the guarantee shown, the scale stretches to hold it, next to the
  // regret of the worst possible play (always the worst arm).
  const showG = p.guarantee && !you;
  const gval = (t: number) => k.ucbLog * Math.log(t) + k.ucbConst;
  const worstGap = Math.max(...mine.mu.map((m) => k.best - m));
  if (showG) ymax = Math.max(ymax, gval(T), worstGap * T);
  ymax *= 1.08;
  const ys = linear([0, ymax], [pTop + pH, pTop]);
  const f: Frame = { x: xs, y: ys, left: fl, right: st.w - 14, top: pTop, bottom: pTop + pH, w: st.w };
  parts.push(
    axis({ scale: ys, orient: "left", at: f.left, span: [f.left, f.right], title: narrow ? undefined : L.regret, count: 4 }),
    axis({ scale: xs, orient: "bottom", at: f.bottom, span: [f.top, f.bottom], title: L.t, count: narrow ? 4 : 6 }),
  );
  if (narrow) parts.push(text(f.left + 4, f.top + 10, L.regret, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));
  const cid = `${st.uid}-clip`;
  parts.push(el("defs", {}, el("clipPath", { id: cid }, el("rect", { x: f.left, y: f.top - 2, width: f.right - f.left, height: f.bottom - f.top + 4 }))));
  const xOf = (t: number) => xs(p.logx ? Math.max(1, t) : t);
  const path = (tt: number[], vv: number[]) => tt.map((t, i) => `${i ? "L" : "M"}${xOf(t).toFixed(1)},${ys(vv[i]).toFixed(1)}`).join("");
  const inner: string[] = [];
  // the asymptotic rate and the guarantee, sampled on the same times
  const lrT = sim.ts.filter((t) => t >= 2);
  if (!you) inner.push(el("path", { d: path(lrT, lrT.map((t) => k.lr * Math.log(t))), fill: "none", stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "5 4" }));
  if (showG) {
    inner.push(el("path", { d: path(lrT, lrT.map(gval)), fill: "none", stroke: C.c2, "stroke-width": 1.6, "stroke-dasharray": "2 3" }));
    inner.push(el("path", { d: path([1, ...lrT], [worstGap, ...lrT.map((t) => worstGap * t)]), fill: "none", stroke: C.ink2, "stroke-width": 1, "stroke-dasharray": "1 3" }));
  }
  for (const a of ALGS) {
    if (R > 1) {
      const up = sim.ts.map((t, i) => `${i ? "L" : "M"}${xOf(t).toFixed(1)},${ys(sim.hi[a][i]).toFixed(1)}`).join("");
      const dn = sim.ts.slice().reverse().map((t, i) => `L${xOf(t).toFixed(1)},${ys(sim.lo[a][sim.ts.length - 1 - i]).toFixed(1)}`).join("");
      inner.push(el("path", { d: up + dn + "Z", fill: COLOR[a], opacity: 0.12, stroke: "none" }));
    }
  }
  for (const a of ALGS) inner.push(el("path", { d: path(sim.ts, sim.mean[a]), fill: "none", stroke: COLOR[a], "stroke-width": 2, "stroke-linejoin": "round" }));
  if (you && shown && mine.cum.length) {
    const tt = mine.cum.map((_, i) => i + 1);
    inner.push(el("path", { d: path([0, ...tt], [0, ...mine.cum]), fill: "none", stroke: COLOR.you, "stroke-width": 2.6, "stroke-linejoin": "round" }));
  }
  parts.push(g({ "clip-path": `url(#${cid})` }, ...inner));
  // labels for the guarantee and the worst possible play
  if (showG) {
    const gt = p.logx ? Math.sqrt(T) : T * 0.12;
    const gx = xOf(gt);
    parts.push(text(gx, ys(gval(gt)) + 16, tpl(L.guarAt, { v: fixed(gval(T), 0), T }), { "font-size": TYPE.small, "text-anchor": "start", class: "fig-t-muted fig-t-halo" }));
    const wt = p.logx ? T * 0.5 : T * 0.72;
    parts.push(text(xOf(wt), ys(worstGap * wt) - 6, tpl(L.worst, { v: fixed(worstGap * T, 0) }), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo" }));
  }
  // readout at the right end
  if (!you) {
    const items2 = ALGS.map((a) => ({ a, v: sim.final[a] })).sort((u, v) => v.v - u.v);
    const ry = f.top + 12;
    parts.push(text(f.right - 4, ry, tpl(L.readout, { T }), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo" }));
    items2.forEach(({ a, v }, i) => parts.push(text(f.right - 4, ry + 14 * (i + 1), `${algLabel(a, L)} ${fixed(v, 0)}`, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-num fig-t-halo", style: `fill:${COLOR[a]}` })));
  } else if (!shown) {
    parts.push(text((f.left + f.right) / 2, f.top + (narrow ? 34 : 16), L.hiddenCurve, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted fig-t-halo" }));
  } else if (mine.cum.length) {
    const n = mine.cum.length;
    const ry = f.top + (narrow ? 26 : 12);
    const rows: Array<[string, number, string]> = [[L.you, mine.cum[n - 1], COLOR.you], ...ALGS.map((a) => [algLabel(a, L), sim.mean[a][Math.min(n, sim.ts.length) - 1], COLOR[a]] as [string, number, string])];
    parts.push(text(f.right - 4, ry, tpl(L.after, { n }), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo" }));
    rows.forEach(([name, v, col], i) => parts.push(text(f.right - 4, ry + 14 * (i + 1), `${name} ${fixed(v, 1)}`, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-num fig-t-halo", style: `fill:${col}` })));
  }
  const H = f.bottom + 42;
  return svg(st.w, H, describe(st), ...parts);
}

function describe(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  if (p.mode === "you") {
    const mine = yours(p);
    const done = mine.pulls.length >= BUDGET;
    const sim = simulate(p.arms, p.gap, p.eps, BUDGET, p.runs, p.seed);
    const n = mine.pulls.length;
    const at = (a: Alg) => (n ? sim.mean[a][Math.min(n, sim.ts.length) - 1] : 0);
    const rev = p.reveal || done ? tpl(L.revYou, { y: fixed(n ? mine.cum[n - 1] : 0, 1), u: fixed(at("ucb"), 1), s: fixed(at("ts"), 1) }) : L.hidYou;
    return tpl(L.describeYou, { n, b: BUDGET, r: mine.total, rev });
  }
  const sim = simulate(p.arms, p.gap, p.eps, p.horizon, p.runs, p.seed);
  return tpl(L.describe, {
    K: p.arms, m: "0.6", g: fixed(p.gap, 2), T: p.horizon, R: p.runs, eps: fixed(p.eps, 2),
    e: fixed(sim.final.eps, 0), u: fixed(sim.final.ucb, 0), s: fixed(sim.final.ts, 0),
  });
}

function pull(p: P, i: number): P | null {
  if (p.mode !== "you" || i >= p.arms) return null;
  const xs = parseNumbers(p.pulls);
  if (xs.length >= BUDGET) return null;
  xs.push(i);
  return { ...p, pulls: xs.join(",") };
}

export default defineFigure({
  name: "regret-bandits",
  title: { en: "Multi-armed bandit: regret of epsilon-greedy, UCB1, and Thompson sampling, or of your own pulls", zh: "多臂赌博机：ε-贪心、UCB1 与 Thompson 采样的遗憾，或你自己拉动的遗憾" },
  labels,
  params,
  hint: { en: "In the “You” mode, click an arm (or its button) to pull it; each pull pays 1 or 0.", zh: "在“你”模式下，点击一条臂（或对应的按钮）即可拉动它；每次拉动得 1 或 0。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click" || !e.target?.startsWith("arm-")) return null;
    return pull(p as P, Number(e.target.slice(4)));
  },
  update(p, key) {
    if (key === "mode" || key === "arms" || key === "gap") return { ...p, pulls: "", reveal: false };
    return p;
  },
  actions: [
    ...Array.from({ length: MAX_ARMS }, (_, i) => ({
      label: { en: `Pull ${i + 1}`, zh: `拉动臂 ${i + 1}` },
      run: (p: P) => pull(p, i) ?? p,
      enabled: (p: P) => p.mode === "you" && i < p.arms && parseNumbers(p.pulls).length < BUDGET,
    })),
    { label: { en: "Undo", zh: "撤销" }, run: (p) => ({ ...p, pulls: parseNumbers(p.pulls).slice(0, -1).join(",") }), enabled: (p) => p.mode === "you" && p.pulls !== "" },
    { label: { en: "New arms", zh: "换一组臂" }, run: (p) => ({ ...p, seed: (p.seed % 9999) + 1, pulls: "", reveal: false }) },
  ],
});
