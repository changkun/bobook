// Expected information gain on the smallest interesting problem: locating an
// unknown threshold theta in [0, 1] with yes/no questions "is theta below x?",
// answered with noise. The belief is a grid of 128 cells, so the uniform
// prior has an entropy of exactly 7 bits. Each answer follows a probit curve,
// P(yes | theta, x) = Phi((x - theta) / s). The lower panel is the expected
// information gain of asking at x, computed in the outcome-space form
// H[answer | x] - E_theta H[answer | theta, x], in bits. The reader asks by
// clicking either panel or with the buttons; a hidden threshold drawn from
// the seed answers. This is the logic of Bayesian adaptive methods in
// psychophysics, which choose each stimulus to maximize the expected
// information about a perceptual threshold.

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { clamp, grid } from "./lib/scale.ts";
import { Phi } from "./lib/stats.ts";
import { rng } from "./lib/random.ts";
import { fmtNumbers, parseNumbers, validNumbers } from "./lib/params.ts";
import { frame, frameAxes, hitArea, legend } from "./lib/plot.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    theta: "threshold θ",
    belief: "belief",
    x: "question: is θ below x?",
    eig: "expected bits",
    post: "belief about θ",
    yes: "answered yes",
    no: "answered no",
    truth: "hidden θ",
    next: "best next x",
    status: "{n} question{n:/s} · belief entropy {h} bits (started at 7) · learned {l} bits",
    statusA: "{n} question{n:/s} · belief entropy {h} bits",
    statusB: "started at 7 bits · learned {l} bits",
    best: "best next question: x = {x}, expected gain {e} bits",
    ceiling: "1 bit: the most a yes/no answer can carry",
    describe: "After {n} question{n:/s}, the belief about the threshold has entropy {h} bits, down from 7. The most informative next question is at x = {x}, with an expected gain of {e} bits.",
  },
  zh: {
    theta: "阈值 θ",
    belief: "信念",
    x: "问题：θ 是否低于 x？",
    eig: "期望比特数",
    post: "关于 θ 的信念",
    yes: "回答“是”",
    no: "回答“否”",
    truth: "隐藏的 θ",
    next: "最佳的下一个 x",
    status: "{n} 个问题 · 信念的熵 {h} 比特（初始为 7）· 已学到 {l} 比特",
    statusA: "{n} 个问题 · 信念的熵 {h} 比特",
    statusB: "初始为 7 比特 · 已学到 {l} 比特",
    best: "最佳的下一个问题：x = {x}，期望增益 {e} 比特",
    ceiling: "1 比特：一个是非回答最多能携带的信息",
    describe: "问了 {n} 个问题之后，关于阈值的信念的熵为 {h} 比特，最初为 7 比特。信息量最大的下一个问题在 x = {x}，期望增益为 {e} 比特。",
  },
};

const params = {
  noise: { kind: "range", label: { en: "Answer noise s", zh: "回答噪声 s" }, min: 0.002, max: 0.3, default: 0.04, scale: "log" },
  reveal: { kind: "toggle", label: { en: "Reveal hidden θ", zh: "显示隐藏的 θ" }, default: false },
  asked: { kind: "data", label: { en: "Questions asked", zh: "已问的问题" }, default: "", validate: validNumbers },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 6, step: 1, control: false },
} as const;

type P = { noise: number; reveal: boolean; asked: string; seed: number };

const NC = 128;
const TH = Array.from({ length: NC }, (_, k) => (k + 0.5) / NC);
const XQ = grid(0, 1, 201);
const MAXQ = 40;

const h2 = (p: number) => (p <= 1e-12 || p >= 1 - 1e-12 ? 0 : -p * Math.log2(p) - (1 - p) * Math.log2(1 - p));
const hidden = (seed: number) => 0.06 + 0.88 * rng(seed * 104729 + 3)();
const answerOf = (seed: number, i: number, x: number, s: number) => rng(seed * 7919 + i * 31 + 1)() < Phi((x - hidden(seed)) / s);

function compute(p: P) {
  const xs = parseNumbers(p.asked);
  const ans = xs.map((x, i) => answerOf(p.seed, i, x, p.noise));
  const logb = TH.map(() => 0);
  xs.forEach((x, i) => {
    for (let k = 0; k < NC; k++) {
      const py = Phi((x - TH[k]) / p.noise);
      logb[k] += Math.log(Math.max(1e-300, ans[i] ? py : 1 - py));
    }
  });
  const mx = Math.max(...logb);
  const un = logb.map((v) => Math.exp(v - mx));
  const z = un.reduce((a, b) => a + b, 0);
  const b = un.map((v) => v / z);
  const H = -b.reduce((s, v) => s + (v > 0 ? v * Math.log2(v) : 0), 0);
  const eig = XQ.map((x) => {
    let py = 0, cond = 0;
    for (let k = 0; k < NC; k++) {
      if (b[k] < 1e-14) continue;
      const q = Phi((x - TH[k]) / p.noise);
      py += b[k] * q;
      cond += b[k] * h2(q);
    }
    return Math.max(0, h2(py) - cond);
  });
  let bi = 0;
  for (let i = 1; i < XQ.length; i++) if (eig[i] > eig[bi] + 1e-12) bi = i;
  return { xs, ans, b, H, eig, best: XQ[bi], bestE: eig[bi] };
}

function describe(st: State<P>): string {
  const c = compute(st.p);
  return tpl(labels[st.lang ?? "en"].describe, { n: c.xs.length, h: fixed(c.H, 2), x: fixed(c.best, 3), e: fixed(c.bestE, 2) });
}

function ask(p: P, x: number): P {
  const xs = parseNumbers(p.asked);
  if (xs.length >= MAXQ) return p;
  xs.push(Math.round(clamp(x, 0, 1) * 1000) / 1000);
  return { ...p, asked: fmtNumbers(xs) };
}

function render(st: State<P>): string {
  const p = st.p, L = labels[st.lang ?? "en"], w = st.w;
  const narrow = w < 480;
  const c = compute(p);
  const lg = legend(narrow ? 8 : 52, 14, w - 16, [
    { kind: "band", color: C.model, label: L.post },
    { kind: "dot", color: C.c6, label: L.yes },
    { kind: "dot", color: C.c8, label: L.no },
    ...(p.reveal ? [{ kind: "dash" as const, color: C.truth, label: L.truth }] : []),
  ]);
  const top = 14 + lg.height + 6;
  const bmax = Math.max(4 / NC, ...c.b) * 1.1;
  const f = frame({ w, top, height: narrow ? 120 : 150, yDomain: [0, bmax], yTitle: L.belief });
  const f2 = frame({ w, top: f.bottom + 34, height: narrow ? 90 : 110, yDomain: [0, 1.08], yTitle: L.eig });
  const parts: string[] = [lg.svg, frameAxes(f, { yTitle: L.belief, xTicks: false, yCount: 2 })];
  // belief bars
  const bw = (f.right - f.left) / NC;
  c.b.forEach((v, k) => {
    if (v * (f.bottom - f.top) / bmax < 0.3) return;
    parts.push(el("rect", { x: f.left + k * bw, y: f.y(v), width: Math.max(0.8, bw - 0.4), height: f.bottom - f.y(v), fill: C.model, opacity: 0.85 }));
  });
  if (p.reveal) {
    const hx = f.x(hidden(p.seed));
    parts.push(el("line", { x1: hx, x2: hx, y1: f.top, y2: f2.bottom, stroke: C.truth, "stroke-width": 1.8, "stroke-dasharray": "5 3" }));
  }
  // asked questions as dots under the belief panel; the latest ringed
  const rowY = f.bottom + 12;
  c.xs.forEach((x, i) => {
    const last = i === c.xs.length - 1;
    parts.push(el("circle", { cx: f.x(x), cy: rowY, r: last ? 5 : 3.6, fill: c.ans[i] ? C.c6 : C.c8, stroke: last ? C.ink : C.paper, "stroke-width": last ? 1.6 : 1 }));
  });
  // expected information gain
  parts.push(frameAxes(f2, { yTitle: L.eig, xTitle: L.x, yCount: 2 }));
  parts.push(el("line", { x1: f2.left, x2: f2.right, y1: f2.y(1), y2: f2.y(1), stroke: C.ink3, "stroke-width": 1, "stroke-dasharray": "4 3" }));
  parts.push(text(f2.right - 4, f2.y(1) - 4, L.ceiling, { "text-anchor": "end", "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));
  const path = XQ.map((x, i) => `${i ? "L" : "M"}${f2.x(x).toFixed(1)},${f2.y(c.eig[i]).toFixed(1)}`).join("");
  parts.push(el("path", { d: path + `L${f2.x(1)},${f2.y(0)}L${f2.x(0)},${f2.y(0)}Z`, fill: C.acqFill, stroke: "none" }));
  parts.push(el("path", { d: path, fill: "none", stroke: C.acq, "stroke-width": 2 }));
  const bx = f2.x(c.best);
  parts.push(
    el("line", { x1: bx, x2: bx, y1: f2.y(c.bestE), y2: f2.bottom, stroke: C.acq, "stroke-width": 1.4, "stroke-dasharray": "3 2" }),
    el("circle", { cx: bx, cy: f2.y(c.bestE), r: 4.5, fill: C.acq, stroke: C.paper, "stroke-width": 1.5 }),
  );
  parts.push(hitArea(f, "ask-top"), hitArea(f2, "ask"));
  let y = f2.bottom + 50;
  const x0 = narrow ? 8 : f.left;
  const size = narrow ? TYPE.small : TYPE.body;
  if (narrow) {
    parts.push(text(x0, y, tpl(L.statusA, { n: c.xs.length, h: fixed(c.H, 2) }), { "font-size": size, class: "fig-t-strong fig-t-num" }));
    y += 16;
    parts.push(text(x0, y, tpl(L.statusB, { l: fixed(7 - c.H, 2) }), { "font-size": size, class: "fig-t-muted fig-t-num" }));
    y += 16;
  } else {
    parts.push(text(x0, y, tpl(L.status, { n: c.xs.length, h: fixed(c.H, 2), l: fixed(7 - c.H, 2) }), { "font-size": size, class: "fig-t-strong fig-t-num" }));
    y += 18;
  }
  parts.push(text(x0, y, tpl(L.best, { x: fixed(c.best, 3), e: fixed(c.bestE, 2) }), { "font-size": size, class: "fig-t-muted fig-t-num" }));
  return svg(w, y + 8, describe(st), ...parts);
}

export default defineFigure({
  name: "info-eig",
  title: { en: "Expected information gain: locating a threshold with noisy yes/no questions", zh: "期望信息增益：用带噪声的是非问题定位阈值" },
  labels,
  params,
  hint: { en: "Click either panel to ask “is θ below x?” at that x, or use the buttons.", zh: "点击任一面板，在该 x 处提问“θ 是否低于 x？”；也可以使用按钮。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click" || !e.data || (e.target !== "ask" && e.target !== "ask-top")) return null;
    return ask(p as P, e.data.x);
  },
  update(p, key) {
    // The answers were given under the old noise level; start over.
    if (key === "noise") return { ...p, asked: "" };
    return p;
  },
  actions: [
    { label: { en: "Ask at the best x", zh: "在最佳 x 处提问" }, primary: true, run: (p) => ask(p as P, compute(p as P).best), enabled: (p) => parseNumbers(p.asked).length < MAXQ },
    { label: { en: "Ask at a random x", zh: "在随机 x 处提问" }, run: (p) => ask(p as P, rng(p.seed * 13 + parseNumbers(p.asked).length * 7 + 5)()), enabled: (p) => parseNumbers(p.asked).length < MAXQ },
    { label: { en: "Undo", zh: "撤销" }, run: (p) => ({ ...p, asked: fmtNumbers(parseNumbers(p.asked).slice(0, -1)) }), enabled: (p) => p.asked !== "" },
    { label: { en: "New threshold", zh: "换一个阈值" }, run: (p) => ({ ...p, asked: "", seed: (p.seed % 999) + 1 }) },
  ],
});
