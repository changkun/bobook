// Sequential line search with you on the slider (Koyama et al. 2017). The
// design is a small poster landscape with two parameters, warmth (cool to
// warm) and vividness (muted to vivid). Each query is one slider: a line
// segment in the design space from the current best design x+ to the design
// with the highest expected improvement x^EI. The reader moves the slider,
// watches the preview, and submits the position that looks best. The answer
// is recorded the way the 2017 method records it, as a choice from a set of
// three: the chosen design beats both ends of the slider, with the
// Bradley-Terry-Luce (softmax) likelihood. A Gaussian process over the two
// parameters is fitted with the Laplace approximation (lib/query-choice.ts),
// and the next slider is drawn from the new x+ to the new x^EI.
//
// In the simulated mode a hidden utility answers instead: the simulated
// person picks a position on the whole slider by a softmax over 41 positions,
// so the model's three-option record is the same approximation the method
// makes with real answers.

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { clamp, linear } from "./lib/scale.ts";
import { kernel, pointIndex, type X } from "./lib/gp.ts";
import { fitChoice, predictChoice, type Choice } from "./lib/query-choice.ts";
import { argmax, ei } from "./lib/acq.ts";
import { memo, rng } from "./lib/random.ts";
import { tpl } from "./lib/format.ts";

const labels = {
  en: {
    you: "Move the slider until the picture looks best, then choose it.",
    youShort: "Slide until it looks best, then choose.",
    sim: "The simulated person searches this slider.",
    warmth: "warmth",
    vivid: "vividness",
    cool: "cool",
    warm: "warm",
    muted: "muted",
    vividEnd: "vivid",
    startEnd: "start",
    otherEnd: "end",
    bestEnd: "current best",
    eiEnd: "highest EI",
    best: "best so far",
    favorite: "hidden favorite",
    settings: "warmth {w} · vividness {v}",
    status: "{n} answer{n:/s} · each recorded as chosen ≻ both ends",
    none: "No answers yet. The first slider is fixed.",
    describe: "After {n} slider answer{n:/s}, the best design so far has warmth {bw} and vividness {bv}. The current slider runs from warmth {aw}, vividness {av} to warmth {ew}, vividness {ev}, and is at position {t}.",
  },
  zh: {
    you: "移动滑块，直到画面看起来最好，然后选定它。",
    youShort: "滑到最好看的位置，然后选定。",
    sim: "模拟用户在这条滑块上搜索。",
    warmth: "暖度",
    vivid: "鲜艳度",
    cool: "冷",
    warm: "暖",
    muted: "素淡",
    vividEnd: "鲜艳",
    startEnd: "起点",
    otherEnd: "终点",
    bestEnd: "当前最优",
    eiEnd: "期望改进最高",
    best: "目前最优",
    favorite: "隐藏的最爱",
    settings: "暖度 {w} · 鲜艳度 {v}",
    status: "{n} 次回答 · 每次记录为“选中点 ≻ 两端”",
    none: "还没有回答。第一条滑块是固定的。",
    describe: "经过 {n} 次滑块回答，目前最优设计的暖度为 {bw}、鲜艳度为 {bv}。当前滑块从暖度 {aw}、鲜艳度 {av} 延伸到暖度 {ew}、鲜艳度 {ev}，滑块位置为 {t}。",
  },
};

const params = {
  mode: { kind: "choice", label: { en: "Who answers", zh: "由谁回答" }, options: [{ value: "you", label: { en: "You", zh: "你" } }, { value: "sim", label: { en: "Simulated person", zh: "模拟用户" } }], default: "you" },
  pos: { kind: "range", label: { en: "Slider", zh: "滑块" }, min: 0, max: 1, default: 0, step: 0.01 },
  noise: { kind: "range", label: { en: "Simulated noise τ", zh: "模拟噪声 τ" }, min: 0.01, max: 0.3, default: 0.04, step: 0.01 },
  truth: { kind: "toggle", label: { en: "Reveal simulated favorite", zh: "显示模拟用户的最爱" }, default: false },
  answers: { kind: "data", label: { en: "Answers", zh: "回答" }, default: "", validate: validAnswers },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 5, step: 1, control: false },
} as const;

type P = { mode: "you" | "sim"; pos: number; noise: number; truth: boolean; answers: string; seed: number };
type V2 = [number, number];
interface Answer { a: V2; b: V2; t: number }

// ---------------------------------------------------------------------------
// Answers are stored as "ax:ay:bx:by:t", one per slider: the two ends and the
// chosen position t in [0, 1] (t = 0 is the first end).

function parseAnswers(s: string): Answer[] {
  if (!s.trim()) return [];
  return s.split(",").map((tok) => {
    const v = tok.split(":").map(Number);
    return { a: [v[0], v[1]], b: [v[2], v[3]], t: v[4] };
  });
}

function validAnswers(s: string): string | undefined {
  const ok = parseAnswers(s).every((a) => [...a.a, ...a.b, a.t].every((v) => Number.isFinite(v) && v >= 0 && v <= 1));
  return ok ? undefined : "expected ax:ay:bx:by:t entries in [0, 1], separated by commas";
}

const r3 = (v: number) => String(Math.round(v * 1000) / 1000);
const fmtAnswers = (as: Answer[]) => as.map((q) => [q.a[0], q.a[1], q.b[0], q.b[1], q.t].map(r3).join(":")).join(",");
const lerp = (a: V2, b: V2, t: number): V2 => [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];

// ---------------------------------------------------------------------------
// The model.

const G = 24; // map and acquisition grid, cell centers
const CELLS: V2[] = [];
for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) CELLS.push([(i + 0.5) / G, (j + 0.5) / G]);
const K = kernel("rbf", 0.22, 1);
const TAU = 0.15; // the model's choice temperature
const INIT: { a: V2; b: V2 } = { a: [0.2, 0.72], b: [0.8, 0.3] };
const MAX_ANSWERS = 20;
const MIN_LEN = 0.12; // the EI end must be at least this far from the best end

interface Model { mean: number[]; sd: number[]; best: V2 | null; line: { a: V2; b: V2 }; chosen: V2[] }

const model = memo((answers: string): Model => {
  const as = parseAnswers(answers);
  const xs: X[] = [];
  const choices: Choice[] = [];
  const chosen: V2[] = [];
  for (const q of as) {
    const c = lerp(q.a, q.b, q.t);
    chosen.push(c);
    const ia = pointIndex(xs, q.a, 2e-3), ib = pointIndex(xs, q.b, 2e-3), ic = pointIndex(xs, c, 2e-3);
    choices.push({ chosen: ic, set: [...new Set([ic, ia, ib])] });
  }
  const fitted = fitChoice(K, xs, choices, TAU);
  const post = predictChoice(fitted, CELLS);
  const sd = post.var.map(Math.sqrt);
  if (!as.length) return { mean: post.mean, sd, best: null, line: INIT, chosen };
  // x+: the observed design with the highest posterior mean (Koyama et al.).
  const atObs = predictChoice(fitted, xs);
  const bi = argmax(atObs.mean);
  const best = xs[bi] as V2;
  const incumbent = atObs.mean[bi];
  // x^EI: the grid cell with the largest expected improvement over x+.
  const score = CELLS.map((c, i) => (Math.hypot(c[0] - best[0], c[1] - best[1]) < MIN_LEN ? -1 : ei(post.mean[i], sd[i], incumbent)));
  const e = CELLS[argmax(score)];
  return { mean: post.mean, sd, best, line: { a: best, b: e }, chosen };
}, 32);

// The simulated person's hidden utility: one favorite design and a gentle
// slope toward it, so a slider far from the favorite still has a direction.
function hidden(seed: number) {
  const r = rng(seed * 131 + 17);
  const star: V2 = [0.15 + 0.7 * r(), 0.15 + 0.7 * r()];
  const u = (x: V2) => {
    const d = Math.hypot(x[0] - star[0], x[1] - star[1]);
    return 0.7 * Math.exp(-(d * d) / (2 * 0.18 * 0.18)) + 0.3 * (1 - d / 1.42);
  };
  return { star, u };
}

const SLIDER_STEPS = 41;

function simAnswer(p: P): P {
  const as = parseAnswers(p.answers);
  if (as.length >= MAX_ANSWERS) return p;
  const { line } = model(p.answers);
  const { u } = hidden(p.seed);
  const ts = Array.from({ length: SLIDER_STEPS }, (_, i) => i / (SLIDER_STEPS - 1));
  const us = ts.map((t) => u(lerp(line.a, line.b, t)) / p.noise);
  const m = Math.max(...us);
  const w = us.map((v) => Math.exp(v - m));
  const tot = w.reduce((a, b) => a + b, 0);
  let x = rng(p.seed * 7919 + as.length * 31 + 3)() * tot;
  let k = 0;
  while (k < w.length - 1 && x > w[k]) { x -= w[k]; k++; }
  as.push({ a: line.a, b: line.b, t: ts[k] });
  return { ...p, answers: fmtAnswers(as), pos: 0 };
}

function choose(p: P): P {
  const as = parseAnswers(p.answers);
  if (as.length >= MAX_ANSWERS) return p;
  const { line } = model(p.answers);
  as.push({ a: line.a, b: line.b, t: clamp(p.pos, 0, 1) });
  return { ...p, answers: fmtAnswers(as), pos: 0 };
}

// ---------------------------------------------------------------------------
// The design: a flat-shaded poster landscape. Warmth shifts every color
// toward orange or blue; vividness scales its distance from gray.

const warmthOf = (x: number) => -1 + 2 * x; // -1 cool .. +1 warm
const vividOf = (y: number) => 0.1 + 1.8 * y; // 0.1 muted .. 1.9 vivid

type RGB = [number, number, number];
const BASE: Record<string, RGB> = {
  sky: [96, 150, 205], glow: [240, 196, 150], sun: [255, 226, 150], far: [104, 100, 142],
  lake: [62, 112, 160], shine: [250, 215, 150], hills: [58, 104, 72], near: [40, 78, 56],
};

function adjust(c: RGB, d: V2): string {
  const w = warmthOf(d[0]), s = vividOf(d[1]);
  const y = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
  const v = c.map((ch) => y + s * (ch - y));
  v[0] += 50 * w; v[1] += 6 * w; v[2] -= 50 * w;
  const q = v.map((ch) => Math.round(clamp(ch, 0, 255)));
  return `rgb(${q[0]},${q[1]},${q[2]})`;
}

function scene(x: number, y: number, w: number, h: number, d: V2, id: string, rx: number): string {
  const col = (k: string) => adjust(BASE[k], d);
  const P = (pts: number[][]) => pts.map(([u, v]) => `${(x + u * w).toFixed(1)},${(y + v * h).toFixed(1)}`).join(" ");
  const sunR = Math.min(w, h) * 0.11;
  return g({},
    el("clipPath", { id }, el("rect", { x, y, width: w, height: h, rx })),
    g({ "clip-path": `url(#${id})` },
      el("rect", { x, y, width: w, height: h, fill: col("sky") }),
      el("rect", { x, y: y + h * 0.44, width: w, height: h * 0.2, fill: col("glow"), opacity: 0.85 }),
      el("circle", { cx: x + w * 0.68, cy: y + h * 0.44, r: sunR, fill: col("sun") }),
      el("polygon", { points: P([[0, 0.62], [0.12, 0.4], [0.25, 0.55], [0.38, 0.35], [0.52, 0.58], [0.66, 0.46], [0.8, 0.6], [0.92, 0.42], [1, 0.52], [1, 0.67], [0, 0.67]]), fill: col("far") }),
      el("rect", { x, y: y + h * 0.66, width: w, height: h * 0.34, fill: col("lake") }),
      el("rect", { x: x + w * 0.62, y: y + h * 0.72, width: w * 0.12, height: Math.max(1.5, h * 0.025), rx: 1, fill: col("shine") }),
      el("rect", { x: x + w * 0.645, y: y + h * 0.79, width: w * 0.07, height: Math.max(1.5, h * 0.02), rx: 1, fill: col("shine"), opacity: 0.8 }),
      el("polygon", { points: P([[0, 0.74], [0.16, 0.63], [0.36, 0.78], [0.42, 1], [0, 1]]), fill: col("hills") }),
      el("polygon", { points: P([[0.6, 1], [0.76, 0.74], [0.9, 0.66], [1, 0.72], [1, 1]]), fill: col("near") }),
    ),
    el("rect", { x, y, width: w, height: h, rx, fill: "none", stroke: "var(--fig-rule)", "stroke-width": 1 }),
  );
}

// ---------------------------------------------------------------------------

const signed = (v: number) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(2);
const fmtW = (x: number) => signed(warmthOf(x));
const fmtV = (y: number) => vividOf(y).toFixed(2);

function describe(st: State<P>): string {
  const p = st.p;
  const m = model(p.answers);
  const n = parseAnswers(p.answers).length;
  const b = m.best ?? m.line.a;
  return tpl(labels[st.lang ?? "en"].describe, {
    n, bw: fmtW(b[0]), bv: fmtV(b[1]), aw: fmtW(m.line.a[0]), av: fmtV(m.line.a[1]),
    ew: fmtW(m.line.b[0]), ev: fmtV(m.line.b[1]), t: p.pos.toFixed(2),
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

// Layout shared by render and pointer: where the slider track sits.
function layout(w: number) {
  const narrow = w < 480;
  const pad = 10;
  const leftW = narrow ? w - 2 * pad : Math.min(340, Math.round(w * 0.54));
  const top = 30;
  const pvH = Math.round(leftW * (narrow ? 0.56 : 0.58));
  const thumbs = narrow ? 5 : 7;
  const gap = 6;
  const tw = (leftW - (thumbs - 1) * gap) / thumbs;
  const th = Math.round(tw * 0.62);
  const filmY = top + pvH + 10;
  const trackY = filmY + th + 16;
  const t0 = pad + tw / 2, t1 = pad + leftW - tw / 2;
  return { narrow, pad, leftW, top, pvH, thumbs, gap, tw, th, filmY, trackY, t0, t1 };
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const colon = st.lang === "zh" ? "：" : ": ";
  const m = model(p.answers);
  const as = parseAnswers(p.answers);
  const lay = layout(st.w);
  const { narrow, pad, leftW, top, pvH, thumbs, gap, tw, th, filmY, trackY, t0, t1 } = lay;
  const parts: string[] = [];
  const cur = lerp(m.line.a, m.line.b, p.pos);
  // 1. The instruction and the preview.
  parts.push(text(pad, 16, p.mode === "you" ? (narrow ? L.youShort : L.you) : L.sim, { "font-size": TYPE.label, class: "fig-t-strong" }));
  parts.push(scene(pad, top, leftW, pvH, cur, `${st.uid}-pv`, 10));
  parts.push(text(pad + leftW - 8, top + pvH - 8, tpl(L.settings, { w: fmtW(cur[0]), v: fmtV(cur[1]) }), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-num", style: "fill:rgba(255,255,255,.95)" }));
  // 2. The filmstrip along the slider, and the slider itself.
  for (let i = 0; i < thumbs; i++) {
    const t = i / (thumbs - 1);
    parts.push(scene(pad + i * (tw + gap), filmY, tw, th, lerp(m.line.a, m.line.b, t), `${st.uid}-th${i}`, 4));
  }
  const kx = t0 + p.pos * (t1 - t0);
  parts.push(
    el("line", { x1: t0, x2: t1, y1: trackY, y2: trackY, stroke: C.rule, "stroke-width": 4, "stroke-linecap": "round" }),
    el("line", { x1: t0, x2: kx, y1: trackY, y2: trackY, stroke: C.acq, "stroke-width": 4, "stroke-linecap": "round" }),
    el("circle", { cx: kx, cy: trackY, r: 8, fill: C.paper, stroke: C.acq, "stroke-width": 2.5 }),
    text(t0 - tw / 2, trackY + 22, as.length ? L.bestEnd : L.startEnd, { "font-size": TYPE.small, class: "fig-t-muted" }),
    text(t1 + tw / 2, trackY + 22, as.length ? L.eiEnd : L.otherEnd, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted" }),
    // the hit area covers the filmstrip and the track; it maps x to t in [0, 1]
    el("rect", {
      x: t0 - tw / 2, y: filmY, width: t1 - t0 + tw, height: trackY + 12 - filmY, fill: "transparent", "data-fig-hit": "slider",
      "data-fig-map": [t0, t1, 0, 1, trackY, trackY - 1, 0, 1].join(",")
    }),
  );
  // 3. The map of the design space: posterior mean, sliders so far, choices.
  const mapLeft = narrow ? pad + 22 : pad + leftW + 46;
  const side = narrow ? Math.min(st.w - mapLeft - pad, 250) : Math.min(st.w - mapLeft - pad, trackY - top + 10);
  const mapTop = narrow ? trackY + 48 : top;
  const sx = linear([0, 1], [mapLeft, mapLeft + side]);
  const sy = linear([0, 1], [mapTop + side, mapTop]);
  const lo = Math.min(...m.mean), hi = Math.max(...m.mean);
  // One path per shade level keeps the markup small (576 cells, 16 levels).
  const cell = side / G;
  const LEVELS = 16;
  const byLevel: string[][] = Array.from({ length: LEVELS }, () => []);
  CELLS.forEach((c, i) => {
    const t = hi - lo > 1e-6 ? (m.mean[i] - lo) / (hi - lo) : 0;
    const k = Math.min(LEVELS - 1, Math.floor(t * LEVELS));
    byLevel[k].push(`M${(sx(c[0]) - cell / 2).toFixed(1)},${(sy(c[1]) - cell / 2).toFixed(1)}h${(cell + 0.4).toFixed(1)}v${(cell + 0.4).toFixed(1)}h-${(cell + 0.4).toFixed(1)}z`);
  });
  byLevel.forEach((d, k) => {
    if (d.length) parts.push(el("path", { d: d.join(""), fill: C.model, opacity: (0.05 + 0.85 * ((k + 0.5) / LEVELS) ** 1.4).toFixed(3) }));
  });
  parts.push(el("rect", { x: mapLeft, y: mapTop, width: side, height: side, fill: "none", stroke: C.rule }));
  // axis words instead of numbers: the parameters are design knobs
  parts.push(
    text(mapLeft, mapTop + side + 15, L.cool, { "font-size": TYPE.small, class: "fig-t-muted" }),
    text(mapLeft + side, mapTop + side + 15, L.warm, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted" }),
    text(mapLeft + side / 2, mapTop + side + 15, L.warmth, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-strong" }),
    text(0, 0, L.muted, { "font-size": TYPE.small, class: "fig-t-muted", transform: `translate(${mapLeft - 7},${mapTop + side}) rotate(-90)` }),
    text(0, 0, L.vividEnd, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted", transform: `translate(${mapLeft - 7},${mapTop}) rotate(-90)` }),
    text(0, 0, L.vivid, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-strong", transform: `translate(${mapLeft - 7},${mapTop + side / 2}) rotate(-90)` }),
  );
  for (const q of as) parts.push(el("line", { x1: sx(q.a[0]), y1: sy(q.a[1]), x2: sx(q.b[0]), y2: sy(q.b[1]), stroke: C.ink3, "stroke-width": 1, opacity: 0.55 }));
  for (const c of m.chosen) parts.push(el("circle", { cx: sx(c[0]), cy: sy(c[1]), r: 3, fill: C.ink, stroke: C.paper, "stroke-width": 1.2 }));
  const { a, b } = m.line;
  parts.push(
    el("line", { x1: sx(a[0]), y1: sy(a[1]), x2: sx(b[0]), y2: sy(b[1]), stroke: C.acq, "stroke-width": 2.5 }),
    el("circle", { cx: sx(a[0]), cy: sy(a[1]), r: 4.5, fill: C.paper, stroke: C.acq, "stroke-width": 2 }),
    el("rect", { x: sx(b[0]) - 4.5, y: sy(b[1]) - 4.5, width: 9, height: 9, fill: C.acq, stroke: C.paper, "stroke-width": 1.2, transform: `rotate(45 ${sx(b[0])} ${sy(b[1])})` }),
    el("circle", { cx: sx(cur[0]), cy: sy(cur[1]), r: 6, fill: "none", stroke: C.ink, "stroke-width": 2 }),
  );
  if (m.best) parts.push(el("path", { d: star(sx(m.best[0]), sy(m.best[1]), 7.5), fill: C.c4, stroke: C.ink, "stroke-width": 0.8 }));
  if (p.mode === "sim" && p.truth) {
    const h = hidden(p.seed).star;
    const hx = sx(h[0]), hy = sy(h[1]);
    parts.push(
      el("path", { d: `M${hx - 6},${hy - 6}L${hx + 6},${hy + 6}M${hx - 6},${hy + 6}L${hx + 6},${hy - 6}`, stroke: C.truth, "stroke-width": 2.5 }),
      text(hx, hy - 14, L.favorite, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-strong fig-t-halo" }),
    );
  }
  // 4. Status.
  const sy0 = narrow ? mapTop + side + 36 : Math.max(trackY + 44, mapTop + side + 36);
  const bestTxt = m.best ? ` · ${L.best}${colon}${tpl(L.settings, { w: fmtW(m.best[0]), v: fmtV(m.best[1]) })}` : "";
  if (narrow) {
    parts.push(text(pad, sy0, as.length ? tpl(L.status, { n: as.length }) : L.none, { "font-size": TYPE.small, class: "fig-t-muted" }));
    if (m.best) parts.push(text(pad, sy0 + 16, `${L.best}${colon}${tpl(L.settings, { w: fmtW(m.best[0]), v: fmtV(m.best[1]) })}`, { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  } else {
    parts.push(text(pad, sy0, as.length ? tpl(L.status, { n: as.length }) + bestTxt : L.none, { "font-size": TYPE.small, class: "fig-t-muted" }));
  }
  const H = sy0 + (narrow && m.best ? 24 : 10);
  return svg(st.w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "query-line-search",
  title: { en: "Sequential line search: you search along a slider the model chooses", zh: "序列线搜索：你沿着模型选定的滑块搜索" },
  labels,
  params,
  hint: { en: "Drag along the strip or move the Slider control, then press Choose. The star is the best design so far; the orange segment on the map is the current slider.", zh: "沿缩略图条拖动，或移动“滑块”控件，然后按“就选这个”。星形是目前最优的设计；图上的橙色线段是当前滑块。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.target !== "slider" || !e.data) return null;
    return { ...p, pos: Math.round(clamp(e.data.x, 0, 1) * 100) / 100 };
  },
  actions: [
    { label: { en: "Choose this one", zh: "就选这个" }, primary: true, run: (p) => choose(p as P), enabled: (p) => p.mode === "you" && parseAnswers(p.answers).length < MAX_ANSWERS },
    { label: { en: "Simulate one", zh: "模拟一次" }, run: (p) => simAnswer(p as P), enabled: (p) => p.mode === "sim" && parseAnswers(p.answers).length < MAX_ANSWERS },
    { label: { en: "Simulate five", zh: "模拟五次" }, run: (p) => { let q = p as P; for (let i = 0; i < 5; i++) q = simAnswer(q); return q; }, enabled: (p) => p.mode === "sim" && parseAnswers(p.answers).length < MAX_ANSWERS },
    { label: { en: "Undo", zh: "撤销" }, run: (p) => ({ ...p, answers: fmtAnswers(parseAnswers(p.answers).slice(0, -1)), pos: 0 }), enabled: (p) => p.answers !== "" },
  ],
  update(p, key) {
    if (key === "mode") return { ...p, answers: "", pos: 0 };
    return p;
  },
});
