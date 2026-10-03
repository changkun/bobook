// The metric decides the winner. Three strategies for choosing the next duel
// run on the same simulated user: EUBO (the expected utility of the better
// option), an incumbent-versus-challenger rule (the current best guess against
// the most optimistic other point), and random pairs. Each run starts from the
// same random first duel and asks 15 more. The reader switches between the
// three regret definitions used in the literature and sees a different
// strategy come out ahead; changing the number of repetitions, or drawing a
// new batch of runs, shows how much a ranking from few repetitions moves.
//
// Model: a Gaussian process utility on a grid of 41 candidates in [0, 1]
// (RBF kernel, lengthscale 0.1, unit variance), the probit likelihood with
// noise 0.1, and the Laplace approximation computed on the whole grid. The
// simulated user answers with Thurstone noise of the chosen size on the
// book's running objective. Everything is seeded and computed in the browser.

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { kernel } from "./lib/gp.ts";
import { expectedMax2 } from "./lib/acq.ts";
import { Phi, millsInv } from "./lib/stats.ts";
import { memo, rng } from "./lib/random.ts";
import { running } from "./lib/objectives.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "comparisons answered",
    yRec: "regret of the recommended point",
    yBest: "regret of the best queried point",
    yCum: "cumulative regret",
    rank: "After {n} comparisons, mean ± standard error over {r} runs:",
    describe: "With {metric} as the measure and simulated noise {s}, after {n} comparisons averaged over {r} runs: {order}.",
    eubo: "EUBO",
    inc: "incumbent vs. challenger",
    rand: "random pairs",
    mRec: "the regret of the recommended point",
    mBest: "the regret of the best queried point",
    mCum: "cumulative regret",
  },
  zh: {
    x: "已回答的比较次数",
    yRec: "推荐点的遗憾",
    yBest: "最佳已查询点的遗憾",
    yCum: "累积遗憾",
    rank: "{n} 次比较后，{r} 次运行的均值 ± 标准误：",
    describe: "以{metric}为度量、模拟噪声为 {s} 时，{n} 次比较后在 {r} 次运行上的平均结果：{order}。",
    eubo: "EUBO",
    inc: "当前最优点对挑战者",
    rand: "随机点对",
    mRec: "推荐点的遗憾",
    mBest: "最佳已查询点的遗憾",
    mCum: "累积遗憾",
  },
};

const STRATS = [
  { key: "eubo", name: "EUBO", color: C.c2, dash: undefined as string | undefined },
  { key: "inc", name: "incumbent vs. challenger", color: C.c1, dash: undefined as string | undefined },
  { key: "rand", name: "random pairs", color: C.ink3, dash: "5 4" },
];

const METRICS = [
  { value: "rec", label: { en: "Recommended", zh: "推荐点" } },
  { value: "best", label: { en: "Best queried", zh: "最佳已查询点" } },
  { value: "cum", label: { en: "Cumulative", zh: "累积" } },
] as const;

type Lang = "en" | "zh";
const METRIC_KEYS: Record<string, "mRec" | "mBest" | "mCum"> = { rec: "mRec", best: "mBest", cum: "mCum" };
// A strategy's name in the edition's language.
const nameOf = (lang: Lang, key: string) => labels[lang][key as "eubo" | "inc" | "rand"];

const params = {
  metric: { kind: "choice", label: { en: "Metric", zh: "度量" }, options: METRICS, default: "rec", control: "buttons" },
  noise: {
    kind: "choice", label: { en: "User noise σ", zh: "用户噪声 σ" }, default: 0.1, control: "buttons",
    options: [{ value: 0.05, label: { en: "0.05", zh: "0.05" } }, { value: 0.1, label: { en: "0.1", zh: "0.1" } }, { value: 0.2, label: { en: "0.2", zh: "0.2" } }],
  },
  reps: {
    kind: "choice", label: { en: "Runs", zh: "运行次数" }, default: 30, control: "buttons",
    options: [{ value: 5, label: { en: "5", zh: "5" } }, { value: 10, label: { en: "10", zh: "10" } }, { value: 30, label: { en: "30", zh: "30" } }],
  },
  batch: { kind: "range", label: { en: "Batch", zh: "批次" }, min: 0, max: 99, default: 0, step: 1, control: false },
} as const;

type P = { metric: string; noise: number; reps: number; batch: number };

// --- the simulation --------------------------------------------------------

const M = 41;
const CAND = Array.from({ length: M }, (_, i) => i / (M - 1));
const FV = CAND.map(running.f);
const FSTAR = Math.max(...FV);
const KERN = kernel("rbf", 0.1, 1);
const K = CAND.map((a, i) => CAND.map((b, j) => KERN(a, b) + (i === j ? 1e-6 : 0)));
const SIGMA_MODEL = 0.1;
const STEPS = 15; // duels chosen by the strategy, after one random first duel

interface LU { a: number[][]; p: number[] }
function luFactor(A: number[][]): LU {
  const n = A.length, a = A.map((r) => r.slice()), p = Array.from({ length: n }, (_, i) => i);
  for (let c = 0; c < n; c++) {
    let q = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[q][c])) q = r;
    if (q !== c) { const t = a[c]; a[c] = a[q]; a[q] = t; const u = p[c]; p[c] = p[q]; p[q] = u; }
    const piv = a[c][c] || 1e-14;
    for (let r = c + 1; r < n; r++) {
      const f = (a[r][c] /= piv);
      if (f !== 0) for (let j = c + 1; j < n; j++) a[r][j] -= f * a[c][j];
    }
  }
  return { a, p };
}
function luSolve(F: LU, b: number[]): number[] {
  const n = b.length, y = F.p.map((i) => b[i]);
  for (let i = 0; i < n; i++) for (let j = 0; j < i; j++) y[i] -= F.a[i][j] * y[j];
  for (let i = n - 1; i >= 0; i--) { for (let j = i + 1; j < n; j++) y[i] -= F.a[i][j] * y[j]; y[i] /= F.a[i][i] || 1e-14; }
  return y;
}

// Laplace approximation of the latent utility on the whole grid, warm-started
// from f. Returns the mode and the posterior covariance (K^-1 + W)^-1.
function laplace(duels: Array<[number, number]>, f0: number[]): { f: number[]; S: number[][] } {
  const s = Math.SQRT2 * SIGMA_MODEL;
  let f = f0.slice();
  for (let it = 0; it < 30; it++) {
    const gr = new Array<number>(M).fill(0);
    const W = CAND.map(() => new Array<number>(M).fill(0));
    for (const [i, j] of duels) {
      const z = (f[i] - f[j]) / s, r = millsInv(z), w = (r * (z + r)) / (s * s);
      gr[i] += r / s; gr[j] -= r / s;
      W[i][i] += w; W[j][j] += w; W[i][j] -= w; W[j][i] -= w;
    }
    // (I + K W) f_new = K (W f + grad)
    const A = K.map((row, i) => CAND.map((_, j) => { let v = i === j ? 1 : 0; for (let q = 0; q < M; q++) v += row[q] * W[q][j]; return v; }));
    const F = luFactor(A);
    const rhs = W.map((row, i) => row.reduce((acc, v, j) => acc + v * f[j], 0) + gr[i]);
    const fn = luSolve(F, K.map((row) => row.reduce((acc, v, j) => acc + v * rhs[j], 0)));
    const delta = Math.max(...fn.map((v, i) => Math.abs(v - f[i])));
    f = fn.every(Number.isFinite) ? fn : f;
    if (delta < 1e-6 || it === 29) {
      const cols = CAND.map((_, j) => luSolve(F, K.map((row) => row[j])));
      return { f, S: CAND.map((_, i) => CAND.map((__, j) => cols[j][i])) };
    }
  }
  return { f, S: K };
}

interface Run { rec: number[]; best: number[]; cum: number[] }

const run = memo((strat: string, noise: number, seed: number): Run => {
  const r0 = rng(seed * 7919 + 1), ru = rng(seed * 104729 + 3), rs = rng(seed * 31 + 11);
  const duels: Array<[number, number]> = [];
  const ask = (a: number, b: number) => {
    const pa = Phi((FV[a] - FV[b]) / (Math.SQRT2 * noise));
    duels.push(ru() < pa ? [a, b] : [b, a]);
  };
  const a0 = Math.floor(r0() * M);
  let b0 = Math.floor(r0() * (M - 1));
  if (b0 >= a0) b0++;
  ask(a0, b0);
  let cum = (2 * FSTAR - FV[a0] - FV[b0]) / 2, bestQ = Math.max(FV[a0], FV[b0]);
  let f = new Array<number>(M).fill(0);
  const out: Run = { rec: [], best: [], cum: [] };
  for (let t = 0; t <= STEPS; t++) {
    const post = laplace(duels, f);
    f = post.f;
    let inc = 0;
    for (let i = 1; i < M; i++) if (f[i] > f[inc]) inc = i;
    out.rec.push(FSTAR - FV[inc]); out.best.push(FSTAR - bestQ); out.cum.push(cum);
    if (t === STEPS) break;
    let a = 0, b = 1;
    if (strat === "eubo") {
      let bv = -Infinity;
      for (let i = 0; i < M; i++) for (let j = i + 1; j < M; j++) {
        const v = expectedMax2(f[i], post.S[i][i], f[j], post.S[j][j], post.S[i][j]);
        if (v > bv) { bv = v; a = i; b = j; }
      }
    } else if (strat === "inc") {
      a = inc;
      let bv = -Infinity;
      for (let j = 0; j < M; j++) if (j !== inc) { const v = f[j] + 2 * Math.sqrt(Math.max(0, post.S[j][j])); if (v > bv) { bv = v; b = j; } }
    } else {
      a = Math.floor(rs() * M);
      b = Math.floor(rs() * (M - 1));
      if (b >= a) b++;
    }
    ask(a, b);
    cum += (2 * FSTAR - FV[a] - FV[b]) / 2;
    bestQ = Math.max(bestQ, FV[a], FV[b]);
  }
  return out;
}, 400);

interface Summary { mean: number[]; se: number[] }

function summarize(p: P, strat: string): Summary {
  const runs: Run[] = [];
  for (let i = 1; i <= p.reps; i++) runs.push(run(strat, p.noise, p.batch * 30 + i));
  const key = p.metric as keyof Run;
  const mean: number[] = [], se: number[] = [];
  for (let t = 0; t <= STEPS; t++) {
    const v = runs.map((r) => r[key][t]);
    const m = v.reduce((s, x) => s + x, 0) / v.length;
    const sd = Math.sqrt(v.reduce((s, x) => s + (x - m) ** 2, 0) / Math.max(1, v.length - 1));
    mean.push(m); se.push(sd / Math.sqrt(v.length));
  }
  return { mean, se };
}

function compute(p: P) {
  const sums = STRATS.map((s) => ({ ...s, ...summarize(p, s.key) }));
  const order = sums.slice().sort((u, v) => u.mean[STEPS] - v.mean[STEPS]);
  return { sums, order };
}

function describe(st: State<P>): string {
  const { order } = compute(st.p);
  const lang = st.lang ?? "en";
  const L = labels[lang];
  return tpl(L.describe, {
    metric: L[METRIC_KEYS[st.p.metric]], s: st.p.noise, n: STEPS + 1, r: st.p.reps,
    order: lang === "zh"
      ? order.map((o, i) => `${i + 1}. ${nameOf(lang, o.key)}（${fixed(o.mean[STEPS], 2)}）`).join("，")
      : order.map((o, i) => `${i + 1}. ${o.name} (${fixed(o.mean[STEPS], 2)})`).join(", "),
  });
}

const yLabel = (metric: string, lang: Lang = "en") => ({ rec: labels[lang].yRec, best: labels[lang].yBest, cum: labels[lang].yCum } as Record<string, string>)[metric];

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 480;
  const { sums, order } = compute(p);
  const left = narrow ? 44 : 58, right = st.w - (narrow ? 12 : 20);
  // Legend: one row of swatches with generous spacing, wrapping on a phone.
  const lgParts: string[] = [];
  let lx = narrow ? 8 : left, ly = 14;
  for (const s of STRATS) {
    const name = nameOf(lang, s.key);
    const wTxt = labelWidth(name, 5.8);
    if (lx + 24 + wTxt > st.w - 8) { lx = narrow ? 8 : left; ly += 18; }
    lgParts.push(
      el("line", { x1: lx, x2: lx + 18, y1: ly - 1, y2: ly - 1, stroke: s.color, "stroke-width": 2.5, "stroke-dasharray": s.dash }),
      text(lx + 24, ly + 3, name, { "font-size": TYPE.small, class: "fig-t-muted" }),
    );
    lx += 24 + wTxt + 20;
  }
  const lg = { svg: g({}, ...lgParts), height: ly - 14 + 18 };
  const top = 14 + lg.height + 16, h = narrow ? 180 : 230;
  const ymax = p.metric === "cum" ? 12 : 0.7;
  const x = linear([1, STEPS + 1], [left, right]);
  const y = linear([0, ymax], [top + h, top]);
  const parts: string[] = [lg.svg];
  parts.push(
    axis({ scale: y, orient: "left", at: left, span: [left, right], title: narrow ? undefined : yLabel(p.metric, lang), count: 4 }),
    axis({ scale: x, orient: "bottom", at: top + h, span: [top, top + h], title: L.x, ticks: [1, 4, 8, 12, 16] }),
  );
  if (narrow) parts.push(text(left, top - 6, yLabel(p.metric, lang), { "font-size": TYPE.small, class: "fig-t-muted" }));
  const cid = `${st.uid}-mclip`;
  parts.push(el("defs", {}, el("clipPath", { id: cid }, el("rect", { x: left, y: top - 2, width: right - left, height: h + 4 }))));
  const ns = Array.from({ length: STEPS + 1 }, (_, i) => i + 1);
  const layer: string[] = [];
  for (const s of sums) {
    const upper = ns.map((n, i) => `${x(n).toFixed(1)},${y(s.mean[i] + s.se[i]).toFixed(1)}`);
    const lower = ns.map((n, i) => `${x(n).toFixed(1)},${y(Math.max(0, s.mean[i] - s.se[i])).toFixed(1)}`).reverse();
    layer.push(el("path", { d: `M${upper.join("L")}L${lower.join("L")}Z`, fill: `color-mix(in srgb, ${s.color} 16%, transparent)` }));
  }
  for (const s of sums) {
    layer.push(el("path", { d: "M" + ns.map((n, i) => `${x(n).toFixed(1)},${y(s.mean[i]).toFixed(1)}`).join("L"), fill: "none", stroke: s.color, "stroke-width": 2, "stroke-dasharray": s.dash, "stroke-linejoin": "round" }));
    layer.push(el("circle", { cx: x(STEPS + 1), cy: y(s.mean[STEPS]), r: 4, fill: s.color, stroke: C.paper, "stroke-width": 1.6 }));
  }
  parts.push(g({ "clip-path": `url(#${cid})` }, ...layer));

  // Ranking readout.
  let ry = top + h + 52;
  parts.push(text(narrow ? 8 : left, ry, tpl(L.rank, { n: STEPS + 1, r: p.reps }), { "font-size": TYPE.small, class: "fig-t-muted" }));
  ry += 18;
  order.forEach((o, i) => {
    const tx = (narrow ? 8 : left);
    parts.push(
      el("line", { x1: tx, x2: tx + 16, y1: ry - 4, y2: ry - 4, stroke: o.color, "stroke-width": 2.5, "stroke-dasharray": o.dash }),
      text(tx + 22, ry, `${i + 1}. ${nameOf(lang, o.key)}${lang === "zh" ? "：" : ": "}${fixed(o.mean[STEPS], p.metric === "cum" ? 1 : 3)} ± ${fixed(o.se[STEPS], p.metric === "cum" ? 1 : 3)}`, { "font-size": TYPE.body, class: i === 0 ? "fig-t-num fig-t-strong" : "fig-t-num" }),
    );
    ry += 18;
  });
  return svg(st.w, ry - 4, describe(st), g({}, ...parts));
}

export default defineFigure({
  name: "sw-metrics",
  title: { en: "Three regret definitions applied to the same simulated runs", zh: "三种遗憾定义作用于同一组模拟运行" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [{ label: { en: "Another batch of runs", zh: "再运行一批" }, primary: true, run: (p) => ({ ...p, batch: (p.batch + 1) % 100 }) }],
});
