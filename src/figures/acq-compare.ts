// Six acquisition functions on one posterior. The reader evaluates the
// running objective by clicking the plot (the value comes from the hidden
// objective, so the reader plays the optimizer) and sees, in six strips, how
// probability of improvement, expected improvement, the upper confidence
// bound, Thompson sampling, the knowledge gradient, and max-value entropy
// search score every input. Each strip is rescaled to fill its height: only
// its shape and the location of its maximum matter for the decision. The
// selected rule is drawn in the acquisition color, its maximum is marked on
// the posterior, and "Evaluate its maximum" runs one step of the loop with it.
// Under the posterior, a histogram of the maximizers of 64 posterior samples
// estimates where the maximum may be, the distribution entropy search
// reasons about and Thompson sampling draws from.
//
// The GP settings follow bo-loop: RBF kernel with lengthscale 0.08 and signal
// variance 0.45, constant mean equal to the data average, near-exact
// observations, and the margin xi = 0.01 for PI and EI by default.

import { defineFigure, tr, type Lang, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { fit, kernel, predict } from "./lib/gp.ts";
import { argmax, ei, pi, ucb } from "./lib/acq.ts";
import { memo, rng } from "./lib/random.ts";
import { running } from "./lib/objectives.ts";
import { drawMaxima, knowledgeGradient, mes } from "./lib/acq-lookahead.ts";
import { fmtNumbers, parseNumbers, validNumbers } from "./lib/params.ts";
import { band, clip, curve, dots, frame, frameAxes, hitArea, legend, type Frame } from "./lib/plot.ts";
import { axis } from "./lib/axis.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "input x",
    f: "f(x)",
    truth: "hidden objective",
    mean: "posterior mean",
    band: "95% band",
    sample: "Thompson sample",
    pstar: "p(x*): where the maximum may be",
    next: "{name} would evaluate x = {x}",
    describe: "After {n} evaluations, the six acquisition functions would evaluate next at: {list}. The selected rule, {name}, picks x = {x}.",
  },
  zh: {
    x: "输入 x",
    f: "f(x)",
    truth: "隐藏的目标函数",
    mean: "后验均值",
    band: "95% 区间",
    sample: "Thompson 样本",
    pstar: "p(x*)：最大值可能的位置",
    next: "{name}：下一步评估 x = {x}",
    describe: "{n} 次评估之后，六个采集函数下一步分别会评估：{list}。所选规则（{name}）选择 x = {x}。",
  },
};

const ACQ = [
  { value: "pi", label: { en: "PI", zh: "PI" } },
  { value: "ei", label: { en: "EI", zh: "EI" } },
  { value: "ucb", label: { en: "UCB", zh: "UCB" } },
  { value: "ts", label: { en: "Thompson", zh: "Thompson" } },
  { value: "kg", label: { en: "KG", zh: "KG" } },
  { value: "mes", label: { en: "MES", zh: "MES" } },
] as const;

const params = {
  acq: { kind: "choice", label: { en: "Acquisition", zh: "采集函数" }, options: ACQ, default: "ei", control: "buttons" },
  xi: { kind: "range", label: { en: "Margin ξ (PI, EI)", zh: "裕量 ξ（PI、EI）" }, min: 0, max: 0.5, default: 0.01, step: 0.01 },
  beta: { kind: "range", label: { en: "UCB weight √β", zh: "UCB 权重 √β" }, min: 0, max: 5, default: 2, step: 0.1 },
  truth: { kind: "toggle", label: { en: "Show hidden objective", zh: "显示隐藏的目标函数" }, default: true },
  points: { kind: "data", label: { en: "Evaluated inputs", zh: "已评估的输入" }, default: "0.1,0.25,0.45,0.92", validate: validNumbers },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 4, step: 1, control: false },
} as const;

type P = { acq: string; xi: number; beta: number; truth: boolean; points: string; seed: number };

const XS = grid(0, 1, 121);
const K = kernel("rbf", 0.08, 0.45);
const NOISE = 1e-5;
const NSAMP = 64;

// Everything that does not depend on xi or beta, memoized by the data and seed.
const base = memo((pts: string, seed: number) => {
  const xs = parseNumbers(pts);
  const ys = xs.map(running.f);
  const ym = ys.length ? ys.reduce((a, b) => a + b, 0) / ys.length : 0;
  const gp = fit(K, xs, ys, NOISE, ym);
  const post = predict(gp, XS, true);
  const sd = post.var.map(Math.sqrt);
  const draws = drawMaxima(post, rng(seed), NSAMP);
  const kg = knowledgeGradient(post, NOISE);
  const ms = mes(post.mean, sd, draws.max);
  const hist = new Array<number>(XS.length).fill(0);
  for (const k of draws.argmax) hist[k]++;
  return { xs, ys, post, sd, draws, kg, ms, hist, best: ys.length ? Math.max(...ys) : 0 };
}, 48);

function curves(p: P) {
  const b = base(p.points, p.seed);
  const m = b.post.mean, sd = b.sd;
  const out: Record<string, number[]> = {
    pi: m.map((mu, i) => pi(mu, sd[i], b.best, p.xi)),
    ei: m.map((mu, i) => ei(mu, sd[i], b.best, p.xi)),
    ucb: m.map((mu, i) => ucb(mu, sd[i], p.beta)),
    ts: b.draws.samples[0],
    kg: b.kg,
    mes: b.ms,
  };
  return { b, out };
}

const nameOf = (v: string, lang: Lang = "en") => tr(ACQ.find((a) => a.value === v)!.label, lang);

function describe(st: State<P>): string {
  const lang = st.lang ?? "en";
  const { b, out } = curves(st.p);
  const list = ACQ.map((a) => `${tr(a.label, lang)} ${fixed(XS[argmax(out[a.value])], 2)}`).join(lang === "zh" ? "，" : ", ");
  return tpl(labels[lang].describe, { n: b.xs.length, list, name: nameOf(st.p.acq, lang), x: fixed(XS[argmax(out[st.p.acq])], 2) });
}

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 480;
  const { b, out } = curves(p);
  const sel = out[p.acq];
  const xnext = XS[argmax(sel)];

  const lg = legend(narrow ? 8 : 40, 14, st.w - 16, [
    ...(p.truth ? [{ kind: "dash" as const, color: C.truth, label: L.truth }] : []),
    { kind: "line", color: C.model, label: L.mean },
    { kind: "band", color: C.band, label: L.band },
    ...(p.acq === "ts" ? [{ kind: "line" as const, color: C.c7, label: L.sample }] : []),
    { kind: "band", color: C.c7, label: L.pstar },
  ]);
  const top = 14 + lg.height + 22;
  const f = frame({ w: st.w, top, height: narrow ? 170 : 210, yDomain: running.range, yTitle: L.f });
  const cid = `${st.uid}-c`;
  const lo = b.post.mean.map((m, i) => m - 1.96 * b.sd[i]);
  const hi = b.post.mean.map((m, i) => m + 1.96 * b.sd[i]);
  // p(x*) as bars standing on the bottom of the plot, at most 40 px tall.
  const hmax = Math.max(...b.hist, 1);
  const bw = (f.right - f.left) / (XS.length - 1);
  const bars = b.hist.map((h, i) => (h ? el("rect", { x: f.x(XS[i]) - bw / 2, y: f.bottom - (40 * h) / hmax, width: bw, height: (40 * h) / hmax, fill: C.c7, opacity: 0.45 }) : "")).join("");

  const parts: string[] = [
    lg.svg,
    el("defs", {}, clip(f, cid)),
    text(f.left, top - 8, tpl(L.next, { name: nameOf(p.acq, lang), x: fixed(xnext, 2) }), { "font-size": TYPE.small, class: "fig-t-num fig-t-strong" }),
    frameAxes(f, { yTitle: L.f, xTicks: false }),
    hitArea(f),
    g({ "clip-path": `url(#${cid})` },
      band(f, XS, lo, hi),
      bars,
      p.truth ? curve(f, XS, XS.map(running.f), { stroke: C.truth, "stroke-dasharray": "5 4", "stroke-width": 1.5 }) : "",
      p.acq === "ts" ? curve(f, XS, b.draws.samples[0], { stroke: C.c7, "stroke-width": 1.4 }) : "",
      curve(f, XS, b.post.mean),
    ),
    el("line", { x1: f.x(xnext), x2: f.x(xnext), y1: f.top, y2: f.bottom, stroke: C.acq, "stroke-width": 1.6, "stroke-dasharray": "4 3" }),
    dots(f, b.xs.map((x, i) => [x, b.ys[i]] as [number, number]), { hit: (i) => `pt-${i}` }),
  ];

  // Six strips, one per acquisition function.
  const sH = narrow ? 34 : 38;
  const gap = 6;
  let y = f.bottom + 14;
  for (const a of ACQ) {
    const v = out[a.value];
    const lo2 = Math.min(...v), hi2 = Math.max(...v);
    const span = hi2 - lo2 || 1;
    const fs: Frame = { ...f, y: linear([lo2, hi2 + span * 0.08], [y + sH, y]), top: y, bottom: y + sH };
    const on = a.value === p.acq;
    const k = argmax(v);
    const path = `M${fs.x(0)},${fs.bottom}` + XS.map((x, i) => `L${fs.x(x)},${fs.y(v[i])}`).join("") + `L${fs.x(1)},${fs.bottom}Z`;
    parts.push(
      el("rect", { x: fs.left, y: fs.top, width: fs.right - fs.left, height: sH, fill: on ? C.panel : "none", stroke: C.rule, "stroke-width": 1, "data-fig-hit": `acq-${a.value}` }),
      el("path", { d: path, fill: on ? C.acqFill : C.grid, stroke: on ? C.acq : C.ink3, "stroke-width": on ? 1.8 : 1.2, "pointer-events": "none" }),
      el("circle", { cx: fs.x(XS[k]), cy: fs.y(v[k]), r: on ? 4 : 3, fill: on ? C.acq : C.ink3, stroke: C.paper, "stroke-width": 1.2 }),
      text(fs.left - 6, y + sH / 2 + 4, narrow && a.value === "ts" ? "TS" : tr(a.label, lang), { "font-size": TYPE.small, "text-anchor": "end", class: on ? "fig-t-strong" : "fig-t-muted" }),
    );
    y += sH + gap;
  }
  // x axis under the strips.
  parts.push(axis({ scale: f.x, orient: "bottom", at: y - gap, title: L.x, count: narrow ? 4 : 5, grid: false }));
  return svg(st.w, y - gap + 42, describe(st), ...parts);
}

export default defineFigure({
  name: "acq-compare",
  title: { en: "Six acquisition functions on one posterior", zh: "同一后验上的六个采集函数" },
  labels,
  params,
  hint: { en: "Click the plot to evaluate the objective there; click a point to remove it; click a strip to select that rule.", zh: "点击图中某处，即在该处评估目标函数；点击某个点可将其删除；点击某条横条可选中对应的规则。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click") return null;
    const xs = parseNumbers(p.points);
    if (e.target?.startsWith("pt-")) {
      xs.splice(Number(e.target.slice(3)), 1);
      return { ...p, points: fmtNumbers(xs) };
    }
    if (e.target?.startsWith("acq-")) return { ...p, acq: e.target.slice(4) };
    if (e.target !== "plot" || !e.data) return null;
    if (xs.length >= 20) xs.shift();
    xs.push(Math.round(Math.min(1, Math.max(0, e.data.x)) * 1000) / 1000);
    return { ...p, points: fmtNumbers(xs) };
  },
  actions: [
    {
      label: { en: "Evaluate its maximum", zh: "评估其最大值处" }, primary: true,
      run: (p) => {
        const x = XS[argmax(curves(p).out[p.acq])];
        const xs = parseNumbers(p.points);
        if (xs.length >= 20) xs.shift();
        xs.push(x);
        return { ...p, points: fmtNumbers(xs), seed: (p.seed % 9999) + 1 };
      },
    },
    { label: { en: "New samples", zh: "新样本" }, run: (p) => ({ ...p, seed: (p.seed % 9999) + 1 }) },
  ],
});
