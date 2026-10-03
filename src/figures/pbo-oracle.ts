// You are the oracle. Two colors are shown; pick the one you like better.
// A Gaussian process over hue (with a periodic kernel, since hue wraps
// around) learns a latent utility from your comparisons with the probit
// likelihood and the Laplace approximation, and EUBO, the expected utility of
// the better option, picks the next pair. In the simulated mode a hidden
// utility answers instead, with probit noise, so the loop can be watched
// against a known answer.

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { fitPreference, kernel, pointIndex, predictPreference, type Duel, type X } from "./lib/gp.ts";
import { argmax, expectedMax2 } from "./lib/acq.ts";
import { fmtDuels, parseDuels, validDuels } from "./lib/params.ts";
import { memo, rng } from "./lib/random.ts";
import { Phi } from "./lib/stats.ts";
import { band, curve, frame, frameAxes } from "./lib/plot.ts";
import { tpl } from "./lib/format.ts";

const labels = {
  en: {
    a: "A",
    b: "B",
    pick: "Which do you prefer?",
    simPick: "The simulated person is asked:",
    hue: "hue",
    u: "utility",
    best: "best guess",
    favorite: "true favorite",
    history: "your answers",
    simHistory: "answers so far",
    none: "No answers yet. The first pair is fixed; after that, EUBO chooses.",
    noneShort: "No answers yet.",
    status: "{n} comparison{n:/s} · best guess {h}°",
    describe: "After {n} comparison{n:/s}, the model's best guess is a hue of {h} degrees. The next pair is {a} degrees against {b} degrees.",
    undecided: "undecided",
  },
  zh: {
    a: "A",
    b: "B",
    pick: "你更喜欢哪一个？",
    simPick: "模拟用户被问到：",
    hue: "色相",
    u: "效用",
    best: "最佳猜测",
    favorite: "真实最爱",
    history: "你的回答",
    simHistory: "已有回答",
    none: "还没有回答。第一对是固定的，之后由 EUBO 选择。",
    noneShort: "还没有回答。",
    status: "{n} 次比较 · 最佳猜测 {h}°",
    describe: "经过 {n} 次比较，模型对色相的最佳猜测是 {h} 度。下一对是 {a} 度对 {b} 度。",
    undecided: "未定",
  },
};

const params = {
  mode: { kind: "choice", label: { en: "Who answers", zh: "由谁回答" }, options: [{ value: "you", label: { en: "You", zh: "你" } }, { value: "sim", label: { en: "Simulated person", zh: "模拟用户" } }], default: "you" },
  acq: { kind: "choice", label: { en: "Next pair", zh: "下一对" }, options: [{ value: "eubo", label: { en: "EUBO", zh: "EUBO" } }, { value: "random", label: { en: "Random", zh: "随机" } }], default: "eubo" },
  noise: { kind: "range", label: { en: "Simulated noise σ", zh: "模拟噪声 σ" }, min: 0.02, max: 0.6, default: 0.12, step: 0.01 },
  truth: { kind: "toggle", label: { en: "Reveal simulated favorite", zh: "显示模拟用户的最爱" }, default: false },
  duels: { kind: "data", label: { en: "Answers", zh: "回答" }, default: "", validate: validDuels },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 11, step: 1, control: false },
} as const;

type P = { mode: "you" | "sim"; acq: "eubo" | "random"; noise: number; truth: boolean; duels: string; seed: number };

const CAND = grid(0, 1, 37).slice(0, 36); // every 10 degrees
const XS = grid(0, 1, 145);
const MAX_DUELS = 40;
const K = kernel("periodic", 0.7, 1, 1);
const SIGMA = 0.15; // the model's assumed noise; deliberately not the person's

const hsl = (h: number) => `hsl(${Math.round(((h % 1) + 1) % 1 * 360)}, 62%, 56%)`;
const deg = (h: number) => Math.round(((h % 1) + 1) % 1 * 360);

// The simulated person's hidden utility over hue: a smooth periodic function
// with one clear favorite and a weaker second peak.
function hidden(seed: number) {
  const r = rng(seed * 101 + 7);
  const h0 = r(), h1 = r();
  const u = (h: number) => Math.cos(2 * Math.PI * (h - h0)) + 0.45 * Math.cos(4 * Math.PI * (h - h1));
  let best = 0, bv = -Infinity;
  for (const h of grid(0, 1, 721)) { const v = u(h); if (v > bv) { bv = v; best = h; } }
  return { u, best };
}

interface Model { xs: X[]; mean: number[]; sd: number[]; best: number; pair: [number, number] }

const model = memo((duelStr: string, acq: string, seed: number): Model => {
  const ds = parseDuels(duelStr);
  const xs: X[] = [];
  const duels: Duel[] = ds.map(([w, l]) => ({ winner: pointIndex(xs, w), loser: pointIndex(xs, l) }));
  const fitted = fitPreference(K, xs, duels, SIGMA);
  const post = predictPreference(fitted, XS);
  const sd = post.var.map(Math.sqrt);
  const best = ds.length ? XS[argmax(post.mean)] : NaN;
  let pair: [number, number];
  if (!ds.length) pair = [CAND[3], CAND[21]];
  else if (acq === "random") {
    const r = rng(seed * 13 + ds.length);
    const i = Math.floor(r() * CAND.length);
    let j = Math.floor(r() * (CAND.length - 1));
    if (j >= i) j++;
    pair = [CAND[i], CAND[j]];
  } else {
    const cp = predictPreference(fitted, CAND, true);
    let bi = 0, bj = 1, bv = -Infinity;
    for (let i = 0; i < CAND.length; i++) for (let j = i + 1; j < CAND.length; j++) {
      const v = expectedMax2(cp.mean[i], cp.cov![i][i], cp.mean[j], cp.cov![j][j], cp.cov![i][j]);
      if (v > bv) { bv = v; bi = i; bj = j; }
    }
    pair = [CAND[bi], CAND[bj]];
  }
  return { xs, mean: post.mean, sd, best, pair };
}, 24);

function simAnswer(p: P): string {
  const ds = parseDuels(p.duels);
  const m = model(p.duels, p.acq, p.seed);
  const { u } = hidden(p.seed);
  const [a, b] = m.pair;
  const r = rng(p.seed * 977 + ds.length * 31 + 5);
  const pa = Phi((u(a) - u(b)) / (Math.SQRT2 * p.noise));
  ds.push(r() < pa ? [a, b] : [b, a]);
  return fmtDuels(ds.slice(-MAX_DUELS));
}

function answer(p: P, choice: "A" | "B"): P {
  const m = model(p.duels, p.acq, p.seed);
  const [a, b] = m.pair;
  const ds = parseDuels(p.duels);
  ds.push(choice === "A" ? [a, b] : [b, a]);
  return { ...p, duels: fmtDuels(ds.slice(-MAX_DUELS)) };
}

function describe(st: State<P>): string {
  const m = model(st.p.duels, st.p.acq, st.p.seed);
  const n = parseDuels(st.p.duels).length;
  const L = labels[st.lang ?? "en"];
  return tpl(L.describe, { n, h: n ? deg(m.best) : L.undecided, a: deg(m.pair[0]), b: deg(m.pair[1]) });
}

function card(x: number, y: number, w: number, h: number, hue: number, name: string): string {
  return g({ "data-fig-hit": name, class: "fig-card" },
    el("rect", { x, y, width: w, height: h, rx: 12, fill: hsl(hue), stroke: "var(--fig-rule)", "stroke-width": 1 }),
    // a little abstract poster: a sun and two text bars, so the color reads as a design, not a swatch
    el("circle", { cx: x + w * 0.72, cy: y + h * 0.36, r: Math.min(w, h) * 0.16, fill: "rgba(255,255,255,0.85)" }),
    el("rect", { x: x + w * 0.12, y: y + h * 0.66, width: w * 0.5, height: 8, rx: 4, fill: "rgba(255,255,255,0.9)" }),
    el("rect", { x: x + w * 0.12, y: y + h * 0.66 + 14, width: w * 0.32, height: 6, rx: 3, fill: "rgba(255,255,255,0.65)" }),
    text(x + 12, y + 22, name, { "font-size": TYPE.title, class: "fig-t-strong", style: "fill:rgba(255,255,255,.95)" }),
    text(x + w - 10, y + h - 10, `${deg(hue)}°`, { "font-size": TYPE.small, "text-anchor": "end", style: "fill:rgba(255,255,255,.9)", class: "fig-t-num" }),
  );
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const m = model(p.duels, p.acq, p.seed);
  const ds = parseDuels(p.duels);
  const hid = hidden(p.seed);
  const parts: string[] = [];
  // 1. The question.
  const qy = 8;
  parts.push(text(st.w / 2, qy + 14, p.mode === "you" ? L.pick : L.simPick, { "text-anchor": "middle", "font-size": TYPE.label, class: "fig-t-strong" }));
  const cw = narrow ? (st.w - 36) / 2 : Math.min(200, (st.w - 80) / 2);
  const ch = narrow ? 92 : 116;
  const gap = narrow ? 12 : 28;
  const cx0 = st.w / 2 - cw - gap / 2;
  const cy = qy + 26;
  parts.push(card(cx0, cy, cw, ch, m.pair[0], L.a), card(st.w / 2 + gap / 2, cy, cw, ch, m.pair[1], L.b));
  // 2. The utility posterior over hue.
  const top = cy + ch + 34;
  const lo = m.mean.map((v, i) => v - 1.96 * m.sd[i]);
  const hi = m.mean.map((v, i) => v + 1.96 * m.sd[i]);
  const ymax = Math.max(1.5, ...hi.map(Math.abs)) * 1.05;
  const f = frame({ w: st.w, top, height: narrow ? 130 : 150, yDomain: [-Math.min(ymax, 4), Math.min(ymax, 4)], yTitle: L.u });
  parts.push(frameAxes(f, { yTitle: L.u, xTicks: false, yCount: 3 }));
  parts.push(band(f, XS, lo, hi), curve(f, XS, m.mean));
  // the hue strip under the plot doubles as the x axis
  const sy = f.bottom + 6;
  const sx = linear([0, 1], [f.left, f.right]);
  const n = 72;
  for (let i = 0; i < n; i++) parts.push(el("rect", { x: sx(i / n), y: sy, width: (f.right - f.left) / n + 0.6, height: 10, fill: hsl(i / n) }));
  for (const t of [0, 0.25, 0.5, 0.75, 1]) parts.push(text(sx(t), sy + 24, `${Math.round(t * 360)}°`, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  // compared points, the current pair, the best guess
  for (const x of m.xs as number[]) {
    const i = Math.round(x * (XS.length - 1));
    parts.push(el("circle", { cx: f.x(x), cy: f.y(m.mean[i]), r: 3.2, fill: C.ink, stroke: C.paper, "stroke-width": 1.4 }));
  }
  for (const [k, x] of [[L.a, m.pair[0]], [L.b, m.pair[1]]] as const) {
    parts.push(el("line", { x1: f.x(x), x2: f.x(x), y1: f.top, y2: f.bottom, stroke: C.acq, "stroke-width": 1.4, "stroke-dasharray": "4 3" }),
      text(f.x(x), f.top - 4, k, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-strong" }));
  }
  if (ds.length) {
    const bi = Math.round(m.best * (XS.length - 1));
    parts.push(el("path", { d: star(f.x(m.best), f.y(m.mean[bi]), 7), fill: C.c4, stroke: C.paper, "stroke-width": 1.2 }));
  }
  if (p.mode === "sim" && p.truth) {
    parts.push(el("line", { x1: f.x(hid.best), x2: f.x(hid.best), y1: f.top, y2: sy + 10, stroke: C.truth, "stroke-width": 2 }),
      text(f.x(hid.best) + 5, f.bottom - 6, L.favorite, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));
  }
  // 3. History: recent answers as small pairs, the winner on the left with a ring.
  const hy = sy + 44;
  parts.push(text(f.left, hy, `${p.mode === "you" ? L.history : L.simHistory} · ${tpl(L.status, { n: ds.length, h: ds.length ? deg(m.best) : "–" })}`, { "font-size": TYPE.small, class: "fig-t-muted" }));
  const recent = ds.slice(narrow ? -8 : -14);
  const pw = 34, ph = 16;
  if (!recent.length) parts.push(text(f.left, hy + 22, narrow ? L.noneShort : L.none, { "font-size": TYPE.small, class: "fig-t-faint" }));
  recent.forEach(([w, l], i) => {
    const x = f.left + i * (pw + 6);
    parts.push(
      el("rect", { x, y: hy + 10, width: pw / 2, height: ph, rx: 3, fill: hsl(w), stroke: C.ink, "stroke-width": 1.5 }),
      el("rect", { x: x + pw / 2, y: hy + 10, width: pw / 2, height: ph, rx: 3, fill: hsl(l), opacity: 0.55 }),
    );
  });
  const H = hy + 10 + ph + 10;
  return svg(st.w, H, describe(st), ...parts);
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

export default defineFigure({
  name: "pbo-oracle",
  title: { en: "You are the oracle: preferential Bayesian optimization over color", zh: "你就是预言机：在颜色上做偏好贝叶斯优化" },
  labels,
  params,
  hint: { en: "Click the color you prefer, or use the buttons. The star marks the model's best guess.", zh: "点击你更喜欢的颜色，或使用按钮。星形标出模型的最佳猜测。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click" || p.mode !== "you") return null;
    if (e.target === "A" || e.target === "B") return answer(p as P, e.target);
    return null;
  },
  actions: [
    { label: { en: "Prefer A", zh: "更喜欢 A" }, primary: true, run: (p) => answer(p as P, "A"), enabled: (p) => p.mode === "you" },
    { label: { en: "Prefer B", zh: "更喜欢 B" }, primary: true, run: (p) => answer(p as P, "B"), enabled: (p) => p.mode === "you" },
    { label: { en: "Simulate one answer", zh: "模拟一次回答" }, run: (p) => ({ ...p, duels: simAnswer(p as P) }), enabled: (p) => p.mode === "sim" },
    { label: { en: "Simulate ten", zh: "模拟十次" }, run: (p) => { let q = p as P; for (let i = 0; i < 10; i++) q = { ...q, duels: simAnswer(q) }; return q; }, enabled: (p) => p.mode === "sim" },
    { label: { en: "Undo", zh: "撤销" }, run: (p) => ({ ...p, duels: fmtDuels(parseDuels(p.duels).slice(0, -1)) }), enabled: (p) => p.duels !== "" },
  ],
  update(p, key) {
    // Switching who answers starts over: the answers belong to one oracle.
    if (key === "mode") return { ...p, duels: "" };
    return p;
  },
});
