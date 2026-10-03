// Who leads the search? A hidden score over a two-parameter design stands in
// for the measured speed and accuracy of the interactive design studies. The
// reader runs the same budget of tries three times: choosing every design
// (You lead), letting a Gaussian process with expected improvement choose
// (Optimizer leads), or letting it suggest while the reader accepts or
// overrides (Cooperative). After each run the reader rates how much they
// controlled where the search went, and the scoreboard sets the best score of
// each run beside the rating and beside the medians Mo et al. (ACM TiiS 2024,
// N = 18) measured for a similar statement. The landscape is illustrative; the
// figure shows the shape of the trade-off the studies report, it does not
// measure it.

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { fit, kernel, predict } from "./lib/gp.ts";
import { argmax, ei } from "./lib/acq.ts";
import { fmtPoints, parsePoints, validPoints } from "./lib/params.ts";
import { memo, normal, rng } from "./lib/random.ts";
import { hitArea, type Frame } from "./lib/plot.ts";
import { tpl } from "./lib/format.ts";

const labels = {
  en: {
    you: "You lead",
    coop: "Cooperative",
    opt: "Optimizer leads",
    p1: "parameter 1",
    p2: "parameter 2",
    status: "{mode} · try {n} of {B} · best {b}",
    statusEmpty: "{mode} · no tries yet",
    statusDone: "{mode} · done · best {b} · now rate your control",
    statusRated: "{mode} · done · best {b} · your control {r} of 7",
    hintYou: "Click the map to try a design",
    hintOpt: "The optimizer chooses; press Next",
    suggestion: "suggestion",
    next: "next",
    max: "max",
    landscape: "hidden landscape",
    scores: "score of each try",
    runningBest: "best so far",
    board: "runs compared",
    best: "best score",
    control: "your control",
    notRun: "not run",
    yours: "your rating",
    mo: "Mo et al. median (N = 18)",
    describe: "{mode}: {n} of {B} tries, best score {b}. Best scores so far: you lead {by}, cooperative {bc}, optimizer leads {bo}. Your control ratings: {ry}, {rc}, {ro}, against medians of 6.0, 5.5, and 1.5 in Mo et al.",
    notRated: "not rated",
  },
  zh: {
    you: "你主导",
    coop: "协作",
    opt: "优化器主导",
    p1: "参数 1",
    p2: "参数 2",
    status: "{mode} · 已尝试 {n}/{B} 次 · 最佳 {b}",
    statusEmpty: "{mode} · 尚未尝试",
    statusDone: "{mode} · 完成 · 最佳 {b} · 请为掌控程度打分",
    statusRated: "{mode} · 完成 · 最佳 {b} · 掌控程度 {r}/7",
    hintYou: "点击地图，尝试一个设计",
    hintOpt: "由优化器选择；按“下一步”",
    suggestion: "建议",
    next: "下一个",
    max: "最大值",
    landscape: "隐藏的地形",
    scores: "每次尝试的得分",
    runningBest: "目前最佳",
    board: "各次运行对比",
    best: "最佳得分",
    control: "你的掌控程度",
    notRun: "未运行",
    yours: "你的评分",
    mo: "Mo 等的中位数（N = 18）",
    describe: "{mode}：已尝试 {n}/{B} 次，最佳得分 {b}。目前的最佳得分：你主导 {by}，协作 {bc}，优化器主导 {bo}。你的掌控程度评分：{ry}、{rc}、{ro}；Mo 等人测得的中位数为 6.0、5.5 和 1.5。",
    notRated: "未评分",
  },
};

const MODES = [
  { value: "you", label: { en: "You", zh: "你" } },
  { value: "coop", label: { en: "Cooperative", zh: "协作" } },
  { value: "opt", label: { en: "Optimizer", zh: "优化器" } },
] as const;

const RATES = [
  { value: 0, label: { en: "–", zh: "–" } },
  ...[1, 2, 3, 4, 5, 6, 7].map((v) => ({ value: v, label: { en: String(v), zh: String(v) } })),
];

const params = {
  mode: { kind: "choice", label: { en: "Who leads", zh: "由谁主导" }, options: MODES, default: "you" },
  px: { kind: "range", label: { en: "Parameter 1", zh: "参数 1" }, min: 0, max: 1, default: 0.5, step: 0.01 },
  py: { kind: "range", label: { en: "Parameter 2", zh: "参数 2" }, min: 0, max: 1, default: 0.5, step: 0.01 },
  rate: { kind: "choice", label: { en: "Your control, 1 to 7", zh: "你的掌控程度（1 至 7）" }, options: RATES, default: 0, control: "buttons" },
  reveal: { kind: "toggle", label: { en: "Reveal the hidden landscape", zh: "揭示隐藏的地形" }, default: false },
  you: { kind: "data", label: { en: "Tries when you lead", zh: "你主导时的尝试" }, default: "", validate: validPoints },
  coop: { kind: "data", label: { en: "Tries in cooperation", zh: "协作时的尝试" }, default: "", validate: validPoints },
  opt: { kind: "data", label: { en: "Tries when the optimizer leads", zh: "优化器主导时的尝试" }, default: "", validate: validPoints },
  rYou: { kind: "range", label: { en: "Rating, you lead", zh: "评分：你主导" }, min: 0, max: 7, default: 0, step: 1, control: false },
  rCoop: { kind: "range", label: { en: "Rating, cooperative", zh: "评分：协作" }, min: 0, max: 7, default: 0, step: 1, control: false },
  rOpt: { kind: "range", label: { en: "Rating, optimizer leads", zh: "评分：优化器主导" }, min: 0, max: 7, default: 0, step: 1, control: false },
  seed: { kind: "range", label: { en: "Landscape", zh: "地形" }, min: 1, max: 999, default: 1, step: 1, control: false },
} as const;

type Mode = "you" | "coop" | "opt";
type P = {
  mode: Mode; px: number; py: number; rate: number; reveal: boolean;
  you: string; coop: string; opt: string; rYou: number; rCoop: number; rOpt: number; seed: number;
};
type Pt = [number, number];

const B = 10; // tries per run
const MO: Record<Mode, number> = { you: 6.0, coop: 5.5, opt: 1.5 }; // Mo et al. 2024, medians
const RKEY: Record<Mode, "rYou" | "rCoop" | "rOpt"> = { you: "rYou", coop: "rCoop", opt: "rOpt" };
const INIT: Pt[] = [[0.3, 0.3], [0.7, 0.7]]; // the optimizer's first two tries
const LS = 0.15; // the optimizer's lengthscale on the unit square
const CAND: Pt[] = [];
for (let i = 0; i <= 40; i++) for (let j = 0; j <= 40; j++) CAND.push([i / 40, j / 40]);

// The hidden landscape: a broad hill, a narrower and taller peak, and a gentle
// slope, rescaled so the minimum is 0 and the maximum 100. The peak is the
// one a search that settles on the first hill it finds tends to miss.
const landscape = memo((seed: number) => {
  const r = rng(seed * 7919 + 17);
  let a: Pt = [0.5, 0.5], b: Pt = [0.5, 0.5];
  for (let t = 0; t < 200; t++) {
    a = [0.15 + 0.7 * r(), 0.15 + 0.7 * r()];
    b = [0.12 + 0.76 * r(), 0.12 + 0.76 * r()];
    if (Math.hypot(a[0] - b[0], a[1] - b[1]) > 0.42) break;
  }
  const raw = (x: number, y: number) =>
    0.72 * Math.exp(-((x - a[0]) ** 2 + (y - a[1]) ** 2) / (2 * 0.24 * 0.24)) +
    Math.exp(-((x - b[0]) ** 2 + (y - b[1]) ** 2) / (2 * 0.1 * 0.1)) +
    0.06 * (x + y);
  let mx = -Infinity, mn = Infinity, top: Pt = [0, 0];
  for (let i = 0; i <= 100; i++) for (let j = 0; j <= 100; j++) {
    const v = raw(i / 100, j / 100);
    if (v > mx) { mx = v; top = [i / 100, j / 100]; }
    if (v < mn) mn = v;
  }
  const f = (x: number, y: number) => (100 * (raw(x, y) - mn)) / (mx - mn);
  return { f, top };
}, 8);

// One measured try: the landscape plus a little measurement noise, fixed by
// the design and the landscape so that repeating a try repeats its score.
function measure(seed: number, x: number, y: number): number {
  const z = normal(rng(seed * 1009 + Math.round(x * 1000) * 7 + Math.round(y * 1000) * 7919 + 3))();
  return Math.max(0, Math.min(100, landscape(seed).f(x, y) + 1.5 * z));
}

interface Run { tries: Pt[]; ys: number[]; best: number; bestAt: number; next: Pt }

// The optimizer: a GP on standardized scores and expected improvement over a
// 41 by 41 grid. It proposes the next try whether or not the reader takes it.
const run = memo((list: string, seed: number): Run => {
  const tries = parsePoints(list).slice(0, B) as Pt[];
  const ys = tries.map(([x, y]) => measure(seed, x, y));
  const bestAt = ys.length ? argmax(ys) : -1;
  const best = ys.length ? ys[bestAt] : NaN;
  let next: Pt;
  if (tries.length < INIT.length) next = INIT[tries.length];
  else {
    const m = ys.reduce((s, v) => s + v, 0) / ys.length;
    const sd = Math.max(12, Math.sqrt(ys.reduce((s, v) => s + (v - m) ** 2, 0) / ys.length));
    const yn = ys.map((v) => (v - m) / sd);
    const gp = fit(kernel("rbf", LS, 1), tries, yn, 0.005, 0);
    const post = predict(gp, CAND);
    const yb = Math.max(...yn);
    next = CAND[argmax(post.mean.map((mu, i) => ei(mu, Math.sqrt(post.var[i]), yb, 0.01)))];
  }
  return { tries, ys, best, bestAt, next };
}, 24);

const listOf = (p: P, m: Mode) => p[m];
const count = (s: string) => parsePoints(s).length;
const add = (p: P, m: Mode, pt: Pt): P => {
  const pts = parsePoints(p[m]);
  if (pts.length >= B) return p;
  pts.push([Math.round(pt[0] * 1000) / 1000, Math.round(pt[1] * 1000) / 1000]);
  return { ...p, [m]: fmtPoints(pts) };
};
const fmtScore = (v: number) => (Number.isFinite(v) ? String(Math.round(v)) : "–");

function describe(st: State<P>): string {
  const p = st.p;
  const cur = run(listOf(p, p.mode), p.seed);
  const L = labels[st.lang ?? "en"];
  const r = (k: Mode) => (p[RKEY[k]] ? String(p[RKEY[k]]) : L.notRated);
  return tpl(L.describe, {
    mode: L[p.mode], n: cur.tries.length, B, b: fmtScore(cur.best),
    by: fmtScore(run(p.you, p.seed).best), bc: fmtScore(run(p.coop, p.seed).best), bo: fmtScore(run(p.opt, p.seed).best),
    ry: r("you"), rc: r("coop"), ro: r("opt"),
  });
}

function star(cx: number, cy: number, rr: number): string {
  let d = "";
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    const q = i % 2 ? rr * 0.45 : rr;
    d += `${i ? "L" : "M"}${(cx + q * Math.cos(a)).toFixed(1)},${(cy + q * Math.sin(a)).toFixed(1)}`;
  }
  return d + "Z";
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 560;
  const mode = p.mode;
  const cur = run(listOf(p, mode), p.seed);
  const land = landscape(p.seed);
  const parts: string[] = [];

  // Status line across the top.
  const n = cur.tries.length;
  const rating = p[RKEY[mode]];
  const status = n === 0 ? tpl(L.statusEmpty, { mode: L[mode] })
    : n < B ? tpl(L.status, { mode: L[mode], n, B, b: fmtScore(cur.best) })
      : rating ? tpl(L.statusRated, { mode: L[mode], b: fmtScore(cur.best), r: rating })
        : tpl(L.statusDone, { mode: L[mode], b: fmtScore(cur.best) });
  parts.push(text(narrow ? 8 : 12, 16, status, { "font-size": TYPE.label, class: "fig-t-strong" }));
  if (n >= B && !rating) parts.push(el("rect", { x: narrow ? 4 : 8, y: 3, width: Math.min(st.w - 12, labelWidth(status, 7.2, 13) + 10), height: 18, rx: 4, fill: "none", stroke: C.c5, "stroke-width": 1.4 }));

  // 1. The design map.
  const x0 = narrow ? 34 : 44;
  const y0 = 34;
  const S = narrow ? Math.min(st.w - x0 - 10, 300) : Math.min(280, Math.floor((st.w - x0 - 30) * 0.48));
  const sx = linear([0, 1], [x0, x0 + S]);
  const sy = linear([0, 1], [y0 + S, y0]);
  const f: Frame = { x: sx, y: sy, left: x0, right: x0 + S, top: y0, bottom: y0 + S, w: st.w };
  parts.push(el("rect", { x: x0, y: y0, width: S, height: S, fill: C.panel, stroke: C.rule, rx: 2 }));
  if (p.reveal) {
    const cells = 30, cw = S / cells;
    for (let i = 0; i < cells; i++) for (let j = 0; j < cells; j++) {
      const v = land.f((i + 0.5) / cells, (j + 0.5) / cells) / 100;
      parts.push(el("rect", { x: x0 + i * cw, y: y0 + S - (j + 1) * cw, width: cw + 0.4, height: cw + 0.4, fill: C.c1, opacity: (0.04 + 0.8 * v * v).toFixed(3) }));
    }
    const [tx, ty] = land.top;
    parts.push(
      el("path", { d: `M${sx(tx) - 5},${sy(ty) - 5}L${sx(tx) + 5},${sy(ty) + 5}M${sx(tx) - 5},${sy(ty) + 5}L${sx(tx) + 5},${sy(ty) - 5}`, stroke: C.truth, "stroke-width": 2.2 }),
      text(x0 + S - 6, y0 + 14, L.landscape, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo" }),
    );
  }
  for (const t of [0.25, 0.5, 0.75]) {
    parts.push(
      el("line", { x1: sx(t), x2: sx(t), y1: y0, y2: y0 + S, stroke: C.grid, "stroke-width": 1 }),
      el("line", { x1: x0, x2: x0 + S, y1: sy(t), y2: sy(t), stroke: C.grid, "stroke-width": 1 }),
    );
  }
  for (const t of [0, 0.5, 1]) parts.push(text(sx(t), y0 + S + 14, String(t), { "font-size": TYPE.small, "text-anchor": t === 0 ? "start" : t === 1 ? "end" : "middle", class: "fig-t-muted fig-t-num" }));
  for (const t of [0, 1]) parts.push(text(x0 - 5, sy(t) + (t === 1 ? 9 : -2), String(t), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-num" }));
  parts.push(
    text(x0 + S / 2, y0 + S + 28, L.p1, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted" }),
    text(0, 0, L.p2, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted", transform: `translate(${x0 - (narrow ? 22 : 26)},${y0 + S / 2}) rotate(-90)` }),
  );
  // The search path, the tries (darker means a higher score), and the best.
  if (n > 1) parts.push(el("path", { d: cur.tries.map(([x, y], i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(""), fill: "none", stroke: C.ink3, "stroke-width": 1, opacity: 0.6 }));
  cur.tries.forEach(([x, y], i) => {
    parts.push(el("circle", { cx: sx(x), cy: sy(y), r: 5.5, fill: C.ink, opacity: (0.18 + 0.82 * cur.ys[i] / 100).toFixed(2), stroke: "none" }),
      el("circle", { cx: sx(x), cy: sy(y), r: 5.5, fill: "none", stroke: C.ink, "stroke-width": 1.2 }));
  });
  if (n) {
    const [bx, by] = cur.tries[cur.bestAt];
    parts.push(el("path", { d: star(sx(bx), sy(by), 8), fill: C.c4, stroke: C.paper, "stroke-width": 1.2 }));
    const right = sx(bx) < x0 + S - 40;
    parts.push(text(sx(bx) + (right ? 11 : -11), sy(by) + 4, fmtScore(cur.best), { "font-size": TYPE.small, "text-anchor": right ? "start" : "end", class: "fig-t-strong fig-t-halo fig-t-num" }));
    const last = n - 1;
    if (last !== cur.bestAt) {
      const [lx, ly] = cur.tries[last];
      const r2 = sx(lx) < x0 + S - 40;
      const below = sy(ly) < y0 + 18;
      parts.push(text(sx(lx) + (r2 ? 9 : -9), sy(ly) + (below ? 17 : -7), fmtScore(cur.ys[last]), { "font-size": TYPE.small, "text-anchor": r2 ? "start" : "end", class: "fig-t-muted fig-t-halo fig-t-num" }));
    }
  }
  // The reader's cursor (where Try this design would go) and the optimizer's proposal.
  if (mode !== "opt" && n < B) {
    const cx = sx(p.px), cy = sy(p.py);
    parts.push(el("path", { d: `M${cx - 8},${cy}L${cx - 3},${cy}M${cx + 3},${cy}L${cx + 8},${cy}M${cx},${cy - 8}L${cx},${cy - 3}M${cx},${cy + 3}L${cx},${cy + 8}`, stroke: C.ink2, "stroke-width": 1.6 }));
  }
  if (mode !== "you" && n < B) {
    const [nx, ny] = cur.next;
    const qx = sx(nx), qy = sy(ny);
    parts.push(el("circle", { cx: qx, cy: qy, r: 9, fill: "none", stroke: C.acq, "stroke-width": 2.2 }));
    // Put the label above the ring, unless that is off the map or onto the best try's label.
    let above = qy > y0 + 22;
    if (n) {
      const [bx, by] = cur.tries[cur.bestAt];
      if (Math.abs(sx(bx) - qx) < 70 && Math.abs(sy(by) - qy) < 40) above = sy(by) > qy ? true : qy > y0 + S - 22 ? true : false;
      if (above && qy < y0 + 22) above = false;
    }
    const lw = labelWidth(mode === "coop" ? L.suggestion : L.next, 3.3, 5.5);
    const lxp = Math.min(x0 + S - lw - 2, Math.max(x0 + lw + 2, qx));
    parts.push(text(lxp, qy + (above ? -13 : 22), mode === "coop" ? L.suggestion : L.next, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-strong fig-t-halo" }));
  }
  if (n === 0 && !p.reveal) {
    parts.push(text(x0 + S / 2, y0 + S / 2 + 26, mode === "opt" ? L.hintOpt : L.hintYou, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-faint" }));
  }
  parts.push(hitArea(f, "map"));

  // 2. Right column (or below, when narrow): scores per try, then the scoreboard.
  const px0 = narrow ? x0 : x0 + S + 34;
  const pw = narrow ? st.w - px0 - 10 : st.w - px0 - 10;
  let py0 = narrow ? y0 + S + 48 : y0;

  // 2a. Score of each try, with the running best.
  const ch = narrow ? 92 : 112;
  const cx0 = px0 + 24, cx1 = px0 + pw - 6;
  const tx = linear([1, B], [cx0 + 6, cx1 - 6]);
  const ty = linear([0, 100], [py0 + 14 + ch, py0 + 14]);
  parts.push(text(px0, py0 + 4, L.scores, { "font-size": TYPE.small, class: "fig-t-muted" }));
  parts.push(el("line", { x1: cx0, x2: cx1, y1: ty(0), y2: ty(0), stroke: C.rule }));
  for (const v of [0, 50, 100]) parts.push(text(cx0 - 4, ty(v) + 4, String(v), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-num" }));
  parts.push(el("line", { x1: cx0, x2: cx1, y1: ty(100), y2: ty(100), stroke: C.truth, "stroke-width": 1.4, "stroke-dasharray": "4 3" }),
    text(cx1, ty(100) - 4, L.max, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted" }));
  for (const k of [1, 5, 10]) parts.push(text(tx(k), ty(0) + 14, String(k), { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted fig-t-num" }));
  if (n) {
    let rb = -Infinity;
    const steps: string[] = [];
    cur.ys.forEach((v, i) => {
      rb = Math.max(rb, v);
      steps.push(`${i ? "L" : "M"}${tx(i + 1).toFixed(1)},${ty(rb).toFixed(1)}`);
      if (i < n - 1) steps.push(`L${tx(i + 2).toFixed(1)},${ty(rb).toFixed(1)}`);
    });
    parts.push(el("path", { d: steps.join(""), fill: "none", stroke: C.model, "stroke-width": 2 }));
    cur.ys.forEach((v, i) => parts.push(el("circle", { cx: tx(i + 1), cy: ty(v), r: 3.6, fill: C.ink, stroke: C.paper, "stroke-width": 1.2 })));
  }
  const lgY = ty(0) + 30;
  parts.push(el("line", { x1: px0, x2: px0 + 18, y1: lgY - 4, y2: lgY - 4, stroke: C.model, "stroke-width": 2 }),
    text(px0 + 24, lgY, L.runningBest, { "font-size": TYPE.small, class: "fig-t-muted" }));

  // 2b. Scoreboard: best score and the reader's control rating per run.
  py0 = lgY + 26;
  parts.push(text(px0, py0, L.board, { "font-size": TYPE.label, class: "fig-t-strong" }));
  const labW = narrow ? 100 : 96;
  const barW = narrow ? 56 : 64;
  const cA = px0 + labW, cB = cA + barW + 26;
  const scaleW = Math.max(90, px0 + pw - cB - 8);
  const rx = linear([1, 7], [cB + 6, cB + scaleW - 6]);
  const hy = py0 + 18;
  parts.push(text(cA, hy, L.best, { "font-size": TYPE.small, class: "fig-t-muted" }),
    text(cB, hy, L.control, { "font-size": TYPE.small, class: "fig-t-muted" }));
  (["you", "coop", "opt"] as Mode[]).forEach((m, i) => {
    const ry = hy + 22 + i * 30;
    const rr = run(listOf(p, m), p.seed);
    const active = m === mode;
    if (active) parts.push(el("rect", { x: px0 - 4, y: ry - 14, width: pw + 2, height: 26, rx: 4, fill: C.panel, stroke: C.rule }));
    parts.push(text(px0, ry + 4, L[m], { "font-size": TYPE.small, class: active ? "fig-t-strong" : "fig-t-muted" }));
    if (rr.tries.length) {
      const w = (barW * rr.best) / 100;
      parts.push(el("rect", { x: cA, y: ry - 6, width: barW, height: 10, rx: 2, fill: C.grid }),
        el("rect", { x: cA, y: ry - 6, width: w, height: 10, rx: 2, fill: C.model, opacity: rr.tries.length < B ? 0.45 : 1 }),
        text(cA + barW + 4, ry + 4, fmtScore(rr.best), { "font-size": TYPE.small, class: "fig-t-strong fig-t-num" }));
    } else {
      parts.push(text(cA, ry + 4, L.notRun, { "font-size": TYPE.small, class: "fig-t-faint" }));
    }
    parts.push(el("line", { x1: rx(1), x2: rx(7), y1: ry, y2: ry, stroke: C.rule }));
    for (let k = 1; k <= 7; k++) parts.push(el("line", { x1: rx(k), x2: rx(k), y1: ry - 2, y2: ry + 2, stroke: C.rule }));
    const mine = p[RKEY[m]];
    if (mine) parts.push(el("circle", { cx: rx(mine), cy: ry, r: 5.5, fill: C.c5, stroke: C.paper, "stroke-width": 1.4 }));
    parts.push(el("line", { x1: rx(MO[m]), x2: rx(MO[m]), y1: ry - 8, y2: ry + 8, stroke: C.ink2, "stroke-width": 2 }));
  });
  const fy = hy + 22 + 2 * 30 + 42;
  parts.push(el("circle", { cx: px0 + 5, cy: fy - 4, r: 5, fill: C.c5 }), text(px0 + 14, fy, L.yours, { "font-size": TYPE.small, class: "fig-t-muted" }));
  const mx = px0 + (narrow ? 96 : 104);
  parts.push(el("line", { x1: mx, x2: mx, y1: fy - 11, y2: fy + 3, stroke: C.ink2, "stroke-width": 2 }), text(mx + 7, fy, L.mo, { "font-size": TYPE.small, class: "fig-t-muted" }));
  for (const k of [1, 4, 7]) parts.push(text(rx(k), hy + 22 + 2 * 30 + 25, String(k), { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-faint fig-t-num" }));

  const H = Math.max(y0 + S + 36, fy + 10);
  return svg(st.w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "hci-agency",
  title: { en: "Who leads the search: you, the optimizer, or both", zh: "由谁主导搜索：你、优化器，还是两者协作" },
  labels,
  params,
  hint: { en: "Run each mode once with 10 tries, then rate your control. Darker dots scored higher; the star is the best try so far.", zh: "每种模式各运行一次，每次 10 次尝试，然后为你的掌控程度打分。圆点越深，得分越高；星形是目前最好的一次尝试。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  update(p, key) {
    const q = p as P;
    if (key === "mode") return { ...q, rate: q[RKEY[q.mode]] } as typeof p;
    if (key === "rate") return { ...q, [RKEY[q.mode]]: q.rate } as typeof p;
    return p;
  },
  pointer(p, e) {
    const q = p as P;
    if (e.phase !== "click" || e.target !== "map" || !e.data || q.mode === "opt") return null;
    if (count(q[q.mode]) >= B) return null;
    const x = Math.min(1, Math.max(0, e.data.x)), y = Math.min(1, Math.max(0, e.data.y));
    return { ...add(q, q.mode, [x, y]), px: Math.round(x * 100) / 100, py: Math.round(y * 100) / 100 };
  },
  actions: [
    {
      label: { en: "Try this design", zh: "试这个设计" }, primary: true,
      run: (p) => add(p as P, (p as P).mode, [(p as P).px, (p as P).py]),
      enabled: (p) => (p as P).mode !== "opt" && count((p as P)[(p as P).mode]) < B,
    },
    {
      label: { en: "Accept suggestion", zh: "接受建议" }, primary: true,
      run: (p) => add(p as P, "coop", run((p as P).coop, (p as P).seed).next),
      enabled: (p) => (p as P).mode === "coop" && count((p as P).coop) < B,
    },
    {
      label: { en: "Next", zh: "下一步" }, primary: true,
      run: (p) => add(p as P, "opt", run((p as P).opt, (p as P).seed).next),
      enabled: (p) => (p as P).mode === "opt" && count((p as P).opt) < B,
    },
    {
      label: { en: "Run to the end", zh: "运行到底" },
      run: (p) => { let q = p as P; while (count(q.opt) < B) q = add(q, "opt", run(q.opt, q.seed).next); return q; },
      enabled: (p) => (p as P).mode === "opt" && count((p as P).opt) < B,
    },
    {
      label: { en: "Undo", zh: "撤销" },
      run: (p) => { const q = p as P; return { ...q, [q.mode]: fmtPoints(parsePoints(q[q.mode]).slice(0, -1)) }; },
      enabled: (p) => (p as P)[(p as P).mode] !== "",
    },
    {
      label: { en: "New landscape", zh: "新地形" },
      run: (p) => ({ ...(p as P), seed: ((p as P).seed % 999) + 1, you: "", coop: "", opt: "", rYou: 0, rCoop: 0, rOpt: 0, rate: 0, reveal: false }),
    },
  ],
});
