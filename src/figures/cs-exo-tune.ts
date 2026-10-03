// Tune a simulated walker's exoskeleton yourself, then see what three
// optimizers do with the same walking time. The walker (lib/cs-exo.ts) is a
// simulation calibrated to published numbers, not data.
//
// The reader sets the peak torque and the peak time, lets the walker try the
// setting (18.7 seconds each, the per-setting time of Schäfer et al., 2026),
// and is told only what the walker feels compared with the previous setting:
// easier, harder, or about the same. When the reader stops, the figure
// reveals the metabolic landscape and runs, for the same walking time,
// Bayesian optimization and CMA-ES on two-minute metabolic estimates and
// preferential Bayesian optimization on felt comparisons, each on a fresh
// copy of the same walker. The table reports the true metabolic reduction of
// every final setting at the end of the session and, for a novice, once the
// walker has fully adapted.

import { defineFigure, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { memo } from "./lib/random.ts";
import { fmtPoints, parsePoints, validPoints } from "./lib/params.ts";
import { tpl } from "./lib/format.ts";
import {
  G, GENERIC, GRID, MIN_PER_FEEL, MIN_PER_METABOLIC, MIN_PER_PAIR, PEAK_TIME, feel, runBO, runCMA, runPBO, runSelf, walker,
  type Feel, type Run, type U,
} from "./lib/cs-exo.ts";

const labels = {
  en: {
    xTitle: "peak torque (N·m per kg of body mass)",
    xShort: "peak torque",
    yTitle: "peak time (% of gait cycle)",
    clock: "walking time used: {t} of {b} min · {n} setting{n:/s} tried",
    start: "Press Walk to try the starting setting.",
    startShort: "Press Walk to begin.",
    first: "#1 · the starting point",
    easier: "easier",
    harder: "harder",
    same: "about the same",
    line: "#{i} · torque {a}, peak {b}% · {f} than #{j}",
    lineSame: "#{i} · torque {a}, peak {b}% · {f} as #{j}",
    you: "you",
    simPerson: "simulated person (by feel)",
    bo: "Bayesian opt. (metabolic)",
    cma: "CMA-ES (metabolic)",
    pbo: "preferential BO (felt)",
    generic: "generic setting",
    best: "best possible",
    hMethod: "tuner",
    hTried: "tried",
    hNow: "now",
    hLater: "adapted",
    table: "true metabolic reduction after {t} min of walking",
    tableShort: "metabolic reduction after {t} min",
    optNow: "optimum now",
    optLater: "optimum once adapted",
    outOfTime: "The session's time is used up. Press I'm done to compare.",
    describe: "A simulated walker, {k}. {n} setting{n:/s} tried in {t} minutes.",
    describeDone: " Final setting: torque {a}, peak time {b}%, a true metabolic reduction of {r}%; Bayesian optimization reached {bo}, CMA-ES {cma}, preferential optimization {pbo}, the generic setting {gen}%, and the best possible is {best}%.",
    novice: "new to the device",
    expert: "already adapted",
    nothing: "nothing (no evaluation fit in the time)",
    optBoth: "{now}; faint: {later}",
  },
  zh: {
    xTitle: "峰值力矩（N·m，按每千克体重计）",
    xShort: "峰值力矩",
    yTitle: "峰值时刻（步态周期的 %）",
    clock: "已用行走时间：{t} / {b} 分钟 · 已试 {n} 个设置",
    start: "按“行走”试一试起始设置。",
    startShort: "按“行走”开始。",
    first: "#1 · 起始点",
    easier: "更轻松",
    harder: "更吃力",
    same: "差不多",
    line: "#{i} · 力矩 {a}，峰值 {b}% · 比 #{j} {f}",
    lineSame: "#{i} · 力矩 {a}，峰值 {b}% · 与 #{j} {f}",
    you: "你",
    simPerson: "模拟的人（凭感觉）",
    bo: "贝叶斯优化（代谢）",
    cma: "CMA-ES（代谢）",
    pbo: "偏好贝叶斯优化（感觉）",
    generic: "通用设置",
    best: "可能的最优值",
    hMethod: "调节方式",
    hTried: "已试",
    hNow: "现在",
    hLater: "适应后",
    table: "行走 {t} 分钟后的真实代谢降幅",
    tableShort: "{t} 分钟后的代谢降幅",
    optNow: "此刻的最优设置",
    optLater: "适应后的最优设置",
    outOfTime: "本次会话的时间已用完。按“完成”进行比较。",
    describe: "一位模拟的行走者，{k}。在 {t} 分钟内试了 {n} 个设置。",
    describeDone: "最终设置：力矩 {a}，峰值时刻 {b}%，真实代谢降幅 {r}%。贝叶斯优化：{bo}；CMA-ES：{cma}；偏好贝叶斯优化：{pbo}；通用设置：{gen}%；可能的最优值：{best}%。",
    novice: "初次使用这个装置",
    expert: "已经适应",
    nothing: "没有结果（时间内来不及完成一次评估）",
    optBoth: "{now}；浅色：{later}",
  },
};

const params = {
  walker: { kind: "choice", label: { en: "Walker", zh: "行走者" }, options: [{ value: "novice", label: { en: "New to the device", zh: "初次使用装置" } }, { value: "expert", label: { en: "Already adapted", zh: "已经适应" } }], default: "novice" },
  budget: { kind: "choice", label: { en: "Session (min)", zh: "会话时长（分钟）" }, options: [{ value: 12, label: { en: "12", zh: "12" } }, { value: 24, label: { en: "24", zh: "24" } }, { value: 48, label: { en: "48", zh: "48" } }], default: 24, control: "buttons" },
  torque: { kind: "range", label: { en: "Peak torque", zh: "峰值力矩" }, min: 0, max: 1, default: 0.6, step: 0.05 },
  time: { kind: "range", label: { en: "Peak time (% gait cycle)", zh: "峰值时刻（步态周期的 %）" }, min: 35, max: 55, default: 45, step: 1 },
  trail: { kind: "choice", label: { en: "Show trail of", zh: "显示谁的轨迹" }, options: [{ value: "you", label: { en: "You", zh: "你" } }, { value: "bo", label: { en: "Bayesian opt.", zh: "贝叶斯优化" } }, { value: "cma", label: { en: "CMA-ES", zh: "CMA-ES" } }, { value: "pbo", label: { en: "Preferences", zh: "偏好" } }], default: "you" },
  reveal: { kind: "toggle", label: { en: "Reveal the landscape", zh: "显示地形" }, default: false },
  log: { kind: "data", label: { en: "Settings tried", zh: "已试的设置" }, default: "", validate: validPoints },
  done: { kind: "toggle", label: { en: "Finished", zh: "已完成" }, default: false, control: false },
  sim: { kind: "toggle", label: { en: "Tuned by the simulated person", zh: "由模拟的人调节" }, default: false, control: false },
  seed: { kind: "range", label: { en: "Walker number", zh: "行走者编号" }, min: 1, max: 9999, default: 12, step: 1, control: false },
} as const;

type P = { walker: "novice" | "expert"; budget: 12 | 24 | 48; torque: number; time: number; trail: "you" | "bo" | "cma" | "pbo"; reveal: boolean; log: string; done: boolean; sim: boolean; seed: number };

const toU = (p: P): U => [Math.round(p.torque * 100) / 100, Math.round(((p.time - 35) / 20) * 100) / 100];
const tried = (p: P) => parsePoints(p.log) as U[];
const maxSettings = (p: P) => Math.floor(p.budget / MIN_PER_FEEL + 1e-9);
const minutesUsed = (n: number) => n * MIN_PER_FEEL;

// What the walker feels at each step, relative to the previous setting.
const feels = memo((log: string, seed: number, novice: boolean): Array<Feel | null> => {
  const w = walker(seed, novice);
  const us = parsePoints(log) as U[];
  return us.map((u, i) => (i === 0 ? null : feel(w, us[i - 1], u, minutesUsed(i + 1), i)));
}, 8);

// The three optimizers, given the same walking time as the reader used.
const rivals = memo((seed: number, novice: boolean, minutes: number): { bo: Run; cma: Run; pbo: Run } => {
  const w = walker(seed, novice);
  return { bo: runBO(w, minutes, seed), cma: runCMA(w, minutes, seed), pbo: runPBO(w, minutes, seed) };
}, 4);

function walk(p: P): P {
  const us = tried(p);
  if (p.done || us.length >= maxSettings(p)) return p;
  us.push(toU(p));
  return { ...p, log: fmtPoints(us) };
}

function finish(p: P): P {
  if (!tried(p).length) return p;
  return { ...p, done: true, reveal: true };
}

function selfTuner(p: P): P {
  const w = walker(p.seed, p.walker === "novice");
  const n = Math.min(maxSettings(p), 35);
  const run = runSelf(w, n * MIN_PER_FEEL, p.seed);
  const us = run.evals.map((e) => e.u.map((v) => Math.round(v * 100) / 100) as U);
  // end on the setting the simulated person kept
  const rec = run.rec.map((v) => Math.round(v * 100) / 100) as U;
  return { ...p, log: fmtPoints(us), torque: rec[0], time: Math.round(PEAK_TIME(rec[1])), done: true, reveal: true, sim: true };
}

const pctOf = (v: number) => (v * 100).toFixed(1);
const fmtU = (u: U) => ({ a: u[0].toFixed(2), b: PEAK_TIME(u[1]).toFixed(0) });

function summary(p: P) {
  const novice = p.walker === "novice";
  const w = walker(p.seed, novice);
  const us = tried(p);
  const T = minutesUsed(us.length);
  const final = toU(p);
  const rv = rivals(p.seed, novice, T);
  const row = (u: U) => ({ now: w.reduction(u, T), later: w.reduction(u, 1e6) });
  return { w, us, T, final, rv, you: row(final), bo: row(rv.bo.rec), cma: row(rv.cma.rec), pbo: row(rv.pbo.rec), generic: row(GENERIC), best: { now: w.best(T), later: w.best(1e6) } };
}

function describe(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const us = tried(p);
  let s = tpl(L.describe, { k: p.walker === "novice" ? L.novice : L.expert, n: us.length, t: minutesUsed(us.length).toFixed(1) });
  if (p.done) {
    const m = summary(p);
    const orNone = (run: Run, v: number) => (run.evals.length ? `${pctOf(v)}%` : L.nothing);
    s += tpl(L.describeDone, { ...fmtU(m.final), r: pctOf(m.you.now), bo: orNone(m.rv.bo, m.bo.now), cma: orNone(m.rv.cma, m.cma.now), pbo: orNone(m.rv.pbo, m.pbo.now), gen: pctOf(m.generic.now), best: pctOf(m.best.now) });
  }
  return s;
}

const FEEL_COLOR: Record<Feel, string> = { easier: C.good, harder: C.bad, same: C.ink3 };

function star(cx: number, cy: number, r: number): string {
  let d = "";
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    const rr = i % 2 ? r * 0.45 : r;
    d += `${i ? "L" : "M"}${(cx + rr * Math.cos(a)).toFixed(1)},${(cy + rr * Math.sin(a)).toFixed(1)}`;
  }
  return d + "Z";
}

function layout(w: number) {
  const narrow = w < 480;
  const left = 44, top = 12;
  const side = narrow ? w - left - 12 : Math.min(300, Math.round(w * 0.46));
  return { narrow, left, top, side, panelX: narrow ? 8 : left + side + 28, panelTop: narrow ? top + side + 52 : top };
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const w = st.w;
  const lay = layout(w);
  const { narrow, left, top, side } = lay;
  const novice = p.walker === "novice";
  const wk = walker(p.seed, novice);
  const us = tried(p);
  const fl = feels(p.log, p.seed, novice);
  const T = minutesUsed(us.length);
  const parts: string[] = [];
  const sx = linear([0, 1], [left, left + side]);
  const sy = linear([0, 1], [top + side, top]);

  // 1. The map of settings.
  parts.push(el("rect", { x: left, y: top, width: side, height: side, fill: C.panel, stroke: C.rule }));
  if (p.reveal) {
    const cell = side / (G - 1);
    const vals = GRID.map((u) => wk.reduction(u, T));
    const hi = Math.max(...vals), lo = Math.min(hi - 0.25, Math.min(...vals));
    const LEVELS = 12;
    const by: string[][] = Array.from({ length: LEVELS }, () => []);
    GRID.forEach((u, i) => {
      const f = Math.max(0, (vals[i] - lo) / (hi - lo || 1));
      const k = Math.min(LEVELS - 1, Math.floor(f * LEVELS));
      const x0 = Math.max(left, sx(u[0]) - cell / 2), x1 = Math.min(left + side, sx(u[0]) + cell / 2);
      const y0 = Math.max(top, sy(u[1]) - cell / 2), y1 = Math.min(top + side, sy(u[1]) + cell / 2);
      by[k].push(`M${x0.toFixed(1)},${y0.toFixed(1)}H${(x1 + 0.3).toFixed(1)}V${(y1 + 0.3).toFixed(1)}H${x0.toFixed(1)}Z`);
    });
    by.forEach((d, k) => { if (d.length) parts.push(el("path", { d: d.join(""), fill: C.truth, opacity: (0.04 + 0.6 * ((k + 0.5) / LEVELS) ** 1.6).toFixed(3) })); });
  }
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    parts.push(
      text(sx(v), top + side + 15, v.toFixed(2).replace(/0$/, "").replace(/\.$/, ""), { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted fig-t-num" }),
      text(left - 6, sy(v) + 4, String(Math.round(PEAK_TIME(v))), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-num" }),
    );
    if (v > 0 && v < 1) parts.push(el("line", { x1: sx(v), x2: sx(v), y1: top, y2: top + side, stroke: C.grid }), el("line", { x1: left, x2: left + side, y1: sy(v), y2: sy(v), stroke: C.grid }));
  }
  parts.push(
    text(left + side / 2, top + side + 32, narrow ? L.xShort : L.xTitle, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-strong" }),
    text(0, 0, L.yTitle, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-strong", transform: `translate(12,${top + side / 2}) rotate(-90)` }),
  );
  const m = p.done ? summary(p) : null;
  // trails
  if (p.trail === "you" || !m) {
    if (us.length > 1) parts.push(el("path", { d: linePath(us.map((u) => [sx(u[0]), sy(u[1])])), fill: "none", stroke: C.ink3, "stroke-width": 1, opacity: 0.6 }));
    us.forEach((u, i) => {
      const f = fl[i];
      parts.push(el("circle", { cx: sx(u[0]), cy: sy(u[1]), r: i === us.length - 1 ? 5 : 3.6, fill: f ? FEEL_COLOR[f] : C.ink, stroke: C.paper, "stroke-width": 1.2 }));
    });
  } else {
    const run = m.rv[p.trail];
    const col = p.trail === "bo" ? C.acq : p.trail === "cma" ? C.c7 : C.c5;
    run.evals.forEach((e, i) => parts.push(el("circle", { cx: sx(e.u[0]), cy: sy(e.u[1]), r: 3.4, fill: col, stroke: C.paper, "stroke-width": 1, opacity: 0.35 + 0.65 * ((i + 1) / run.evals.length) })));
  }
  // the optimum, when revealed
  if (p.reveal) {
    const o = wk.optimum(T), oe = wk.optimum(1e6);
    const cross = (u: U, a: Record<string, string | number>) => { const x = sx(u[0]), y = sy(u[1]); return el("path", { d: `M${x - 6},${y - 6}L${x + 6},${y + 6}M${x - 6},${y + 6}L${x + 6},${y - 6}`, stroke: C.truth, "stroke-width": 2.6, fill: "none", ...a }); };
    if (novice) parts.push(el("line", { x1: sx(o[0]), y1: sy(o[1]), x2: sx(oe[0]), y2: sy(oe[1]), stroke: C.truth, "stroke-width": 1.2, "stroke-dasharray": "3 3" }), cross(oe, { opacity: 0.45 }));
    parts.push(cross(o, {}));
  }
  // final settings of all tuners
  if (m) {
    const mark = (u: U, col: string, shape: "sq" | "tri" | "dia") => {
      const x = sx(u[0]), y = sy(u[1]);
      if (shape === "sq") return el("rect", { x: x - 5, y: y - 5, width: 10, height: 10, fill: col, stroke: C.ink, "stroke-width": 1 });
      if (shape === "tri") return el("path", { d: `M${x},${y - 6.5}L${x + 6},${y + 5}L${x - 6},${y + 5}Z`, fill: col, stroke: C.ink, "stroke-width": 1 });
      return el("rect", { x: x - 4.5, y: y - 4.5, width: 9, height: 9, fill: col, stroke: C.ink, "stroke-width": 1, transform: `rotate(45 ${x} ${y})` });
    };
    if (m.rv.bo.evals.length) parts.push(mark(m.rv.bo.rec, C.acq, "sq"));
    if (m.rv.cma.evals.length) parts.push(mark(m.rv.cma.rec, C.c7, "tri"));
    if (m.rv.pbo.evals.length) parts.push(mark(m.rv.pbo.rec, C.c5, "dia"));
    parts.push(el("circle", { cx: sx(GENERIC[0]), cy: sy(GENERIC[1]), r: 4.5, fill: "none", stroke: C.ink, "stroke-width": 1.6, "stroke-dasharray": "2 2" }));
    parts.push(el("path", { d: star(sx(m.final[0]), sy(m.final[1]), 8), fill: C.c4, stroke: C.ink, "stroke-width": 0.9 }));
  } else {
    // the setting on the sliders: where the next walk will be
    const cu = toU(p);
    parts.push(el("circle", { cx: sx(cu[0]), cy: sy(cu[1]), r: 8, fill: "none", stroke: C.acq, "stroke-width": 2.2 }));
    parts.push(el("rect", { x: left, y: top, width: side, height: side, fill: "transparent", "data-fig-hit": "map", "data-fig-map": [left, left + side, 0, 1, top + side, top, 0, 1].join(",") }));
  }

  // 2. The panel: the clock and the feed, or the results.
  const px = lay.panelX, pw = w - px - 8;
  let y = lay.panelTop + 10;
  parts.push(text(px, y, tpl(L.clock, { t: T.toFixed(1), b: p.budget, n: us.length }), { "font-size": TYPE.small, class: "fig-t-strong" }));
  y += 8;
  parts.push(
    el("rect", { x: px, y, width: pw, height: 7, rx: 3.5, fill: C.grid }),
    el("rect", { x: px, y, width: Math.max(0, Math.min(1, T / p.budget)) * pw, height: 7, rx: 3.5, fill: C.acq }),
  );
  y += 26;
  if (!m) {
    if (!us.length) parts.push(text(px, y, narrow ? L.startShort : L.start, { "font-size": TYPE.small, class: "fig-t-faint" }));
    const recent = us.map((u, i) => ({ u, i })).slice(-8);
    recent.forEach(({ u, i }, k) => {
      const f = fl[i];
      const yy = y + k * 18;
      const s = f ? tpl(f === "same" ? L.lineSame : L.line, { i: i + 1, ...fmtU(u), f: L[f], j: i }) : L.first;
      parts.push(el("circle", { cx: px + 5, cy: yy - 4, r: 4, fill: f ? FEEL_COLOR[f] : C.ink }), text(px + 16, yy, s, { "font-size": TYPE.small, class: "fig-t-muted" }));
    });
    y += Math.max(1, recent.length) * 18;
    if (us.length >= maxSettings(p)) { parts.push(text(px, y + 4, L.outOfTime, { "font-size": TYPE.small, class: "fig-t-strong" })); y += 18; }
  } else {
    parts.push(text(px, y, tpl(narrow ? L.tableShort : L.table, { t: T.toFixed(1) }), { "font-size": TYPE.small, class: "fig-t-muted" }));
    y += 20;
    const c1 = px + 18, c2 = px + pw - (novice ? 108 : 52), c3 = px + pw - 54, c4 = px + pw;
    parts.push(
      text(c1, y, L.hMethod, { "font-size": TYPE.small, class: "fig-t-faint" }),
      text(c2, y, L.hTried, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-faint" }),
      text(novice ? c3 : c4, y, L.hNow, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-faint" }),
      novice ? text(c4, y, L.hLater, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-faint" }) : "",
      el("line", { x1: px, x2: px + pw, y1: y + 5, y2: y + 5, stroke: C.rule }),
    );
    y += 20;
    const nMet = Math.floor(T / MIN_PER_METABOLIC + 1e-9), nPair = Math.floor(T / MIN_PER_PAIR + 1e-9);
    const rows: Array<{ glyph: string; label: string; n: string; v: { now: number; later: number }; strong?: boolean; none?: boolean }> = [
      { glyph: el("path", { d: star(0, 0, 6.5), fill: C.c4, stroke: C.ink, "stroke-width": 0.8 }), label: p.sim ? L.simPerson : L.you, n: String(us.length), v: m.you, strong: true },
      { glyph: el("rect", { x: -4.5, y: -4.5, width: 9, height: 9, fill: C.acq, stroke: C.ink, "stroke-width": 0.8 }), label: L.bo, n: String(nMet), v: m.bo, none: nMet === 0 },
      { glyph: el("path", { d: "M0,-6L5.5,4.5L-5.5,4.5Z", fill: C.c7, stroke: C.ink, "stroke-width": 0.8 }), label: L.cma, n: String(nMet), v: m.cma, none: nMet === 0 },
      { glyph: el("rect", { x: -4, y: -4, width: 8, height: 8, fill: C.c5, stroke: C.ink, "stroke-width": 0.8, transform: "rotate(45)" }), label: L.pbo, n: `${nPair}×2`, v: m.pbo, none: nPair === 0 },
      { glyph: el("circle", { cx: 0, cy: 0, r: 4.5, fill: "none", stroke: C.ink, "stroke-width": 1.5, "stroke-dasharray": "2 2" }), label: L.generic, n: "0", v: m.generic },
      { glyph: el("path", { d: "M-5,-5L5,5M-5,5L5,-5", stroke: C.truth, "stroke-width": 2.4, fill: "none" }), label: L.best, n: "", v: m.best },
    ];
    rows.forEach((r) => {
      parts.push(
        g({ transform: `translate(${px + 7},${y - 4})` }, r.glyph),
        text(c1, y, r.label, { "font-size": TYPE.small, class: r.strong ? "fig-t-strong" : "fig-t-muted" }),
        text(c2, y, r.n, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-num" }),
        text(novice ? c3 : c4, y, r.none ? "–" : `${pctOf(r.v.now)}%`, { "font-size": TYPE.small, "text-anchor": "end", class: `fig-t-num ${r.strong ? "fig-t-strong" : ""}` }),
        novice ? text(c4, y, r.none ? "–" : `${pctOf(r.v.later)}%`, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-num fig-t-muted" }) : "",
      );
      y += 19;
    });
    if (p.reveal) {
      y += 4;
      parts.push(
        el("path", { d: `M${px + 2},${y - 9}l10,10m-10,0l10,-10`, stroke: C.truth, "stroke-width": 2.4, fill: "none" }),
        text(px + 18, y, novice ? tpl(L.optBoth, { now: L.optNow, later: L.optLater }) : L.optNow, { "font-size": TYPE.small, class: "fig-t-faint" }),
      );
      y += 8;
    }
  }
  const H = Math.max(narrow ? 0 : top + side + 40, y + 8);
  return svg(w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "cs-exo-tune",
  title: { en: "Tune a simulated walker's exoskeleton yourself, then compare with three optimizers (calibrated simulation)", zh: "亲手调节一位模拟行走者的外骨骼，再与三种优化器比较（校准过的模拟）" },
  labels,
  params,
  hint: { en: "Set the torque and the timing (or click the map), then press Walk. You are told only how the walker feels compared with the previous setting. Leave the sliders on the setting you want to keep and press I'm done.", zh: "设定力矩和时机（或点击设置图），然后按“行走”。你只会被告知行走者觉得这个设置与上一个相比如何。把滑块停在想保留的设置上，再按“完成”。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p0, e) {
    const p = p0 as P;
    if (p.done || e.target !== "map" || !e.data) return null;
    if (e.phase !== "click" && e.phase !== "move" && e.phase !== "down") return null;
    const tq = Math.round(Math.min(1, Math.max(0, e.data.x)) * 20) / 20;
    const tm = Math.round(35 + 20 * Math.min(1, Math.max(0, e.data.y)));
    return { ...p, torque: tq, time: tm };
  },
  actions: [
    { label: { en: "Walk on this setting (19 s)", zh: "行走：试这个设置（19 秒）" }, primary: true, run: (p) => walk(p as P), enabled: (p) => !p.done && tried(p as P).length < maxSettings(p as P) },
    { label: { en: "I'm done: keep this setting", zh: "完成：保留这个设置" }, run: (p) => finish(p as P), enabled: (p) => !p.done && p.log !== "" },
    { label: { en: "Let a simulated person tune", zh: "让模拟的人来调" }, run: (p) => selfTuner(p as P), enabled: (p) => !p.done && p.log === "" },
    { label: { en: "Undo", zh: "撤销" }, run: (p) => ({ ...p, log: fmtPoints(tried(p as P).slice(0, -1)) }), enabled: (p) => !p.done && p.log !== "" },
    { label: { en: "Another walker", zh: "换一位行走者" }, run: (p) => ({ ...p, seed: (p.seed % 9998) + 1, log: "", done: false, sim: false, reveal: false, torque: 0.6, time: 45, trail: "you" as const }) },
  ],
  update(p, key) {
    // A different walker or session length starts a new session.
    if (key === "walker" || key === "budget") return { ...p, log: "", done: false, sim: false, reveal: false, torque: 0.6, time: 45, trail: "you" };
    return p;
  },
});
