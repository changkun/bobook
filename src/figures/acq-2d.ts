// Acquisition functions in two dimensions, on the Branin function (rescaled to
// the unit square and negated, so larger is better; three global maxima).
// Four heatmaps share one grid: the hidden objective, the posterior mean, the
// posterior standard deviation, and the selected acquisition function, with
// the evaluations so far as dots and the next query as an orange ring on
// every panel. The reader clicks any panel to evaluate the objective there,
// or presses "Evaluate the maximum" to let the selected rule choose.
//
// Observations are standardized (mean zero, unit variance) before the GP is
// fitted, as libraries do; the kernel is RBF with lengthscale 0.18 in both
// inputs. Everything is computed on a 31 x 31 grid, which is also the
// candidate set over which each acquisition function is maximized.

import { defineFigure, tr, type Lang, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { fit, kernel, predict } from "./lib/gp.ts";
import { argmax, ei, pi, ucb } from "./lib/acq.ts";
import { memo, rng } from "./lib/random.ts";
import { branin01 } from "./lib/objectives.ts";
import { drawMaxima, knowledgeGradient, mes } from "./lib/acq-lookahead.ts";
import { fmtPoints, parsePoints, validPoints } from "./lib/params.ts";
import { hitArea, type Frame } from "./lib/plot.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    truth: "hidden objective (3 maxima, +)",
    mean: "posterior mean",
    sd: "posterior sd",
    acq: "{name}",
    tsName: "Thompson: one posterior sample",
    next: "next query ({x1}, {x2})",
    readout: "{n} evaluations; best value {b} (max {m}); {name} picks ({x1}, {x2})",
    describe: "Branin in two dimensions after {n} evaluations. The best value found is {b}; the maximum is {m}. {name} would evaluate next at ({x1}, {x2}), where the posterior mean is {mu} and the standard deviation is {sd}.",
    hidden: "hidden",
  },
  zh: {
    truth: "隐藏的目标函数（3 个最大值点，+）",
    mean: "后验均值",
    sd: "后验标准差",
    acq: "{name}",
    tsName: "Thompson：一个后验样本",
    next: "下一个查询点 ({x1}, {x2})",
    readout: "{n} 次评估；最优值 {b}（最大值 {m}）；{name}：选择 ({x1}, {x2})",
    describe: "二维 Branin 函数，已评估 {n} 次。找到的最优值为 {b}，最大值为 {m}。所选规则（{name}）下一步将评估 ({x1}, {x2})，该处的后验均值为 {mu}，标准差为 {sd}。",
    hidden: "已隐藏",
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
  beta: { kind: "range", label: { en: "UCB weight √β", zh: "UCB 权重 √β" }, min: 0, max: 5, default: 2, step: 0.1 },
  truth: { kind: "toggle", label: { en: "Show hidden objective", zh: "显示隐藏的目标函数" }, default: true },
  points: { kind: "data", label: { en: "Evaluated inputs", zh: "已评估的输入" }, default: "0.12:0.31,0.38:0.66,0.62:0.94,0.88:0.47,0.29:0.06,0.71:0.19", validate: validPoints },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 2, step: 1, control: false },
} as const;

type P = { acq: string; beta: number; truth: boolean; points: string; seed: number };

const G = 31;
const GX: number[][] = [];
for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) GX.push([i / (G - 1), j / (G - 1)]); // index = j * G + i
const TRUTH = GX.map(([u, v]) => branin01(u, v));
const TMAX = 0.99263; // value at each of the three maxima
const MAXIMA: Array<[number, number]> = [[0.1239, 0.8183], [0.5428, 0.1517], [0.9617, 0.165]];
const K = kernel("rbf", [0.18, 0.18], 1);
const NOISE = 1e-6;
const XI = 0.01;

const base = memo((pts: string, seed: number) => {
  const obs = parsePoints(pts);
  const ys = obs.map(([u, v]) => branin01(u, v));
  const n = ys.length;
  const ym = n ? ys.reduce((a, b) => a + b, 0) / n : 0;
  const ysd = n > 1 ? Math.sqrt(ys.reduce((a, b) => a + (b - ym) ** 2, 0) / (n - 1)) || 1 : 1;
  const z = ys.map((y) => (y - ym) / ysd); // standardized
  const gp = fit(K, obs.map((o) => [o[0], o[1]]), z, NOISE, 0);
  const post = predict(gp, GX, true);
  const sd = post.var.map(Math.sqrt);
  const best = n ? Math.max(...z) : 0;
  const draws = drawMaxima(post, rng(seed), 32);
  const kg = knowledgeGradient(post, NOISE);
  const ms = mes(post.mean, sd, draws.max);
  return { obs, ys, ym, ysd, post, sd, best, draws, kg, ms };
}, 24);

function acqValues(p: P) {
  const b = base(p.points, p.seed);
  const m = b.post.mean, sd = b.sd;
  let v: number[];
  switch (p.acq) {
    case "pi": v = m.map((mu, i) => pi(mu, sd[i], b.best, XI)); break;
    case "ucb": v = m.map((mu, i) => ucb(mu, sd[i], p.beta)); break;
    case "ts": v = b.draws.samples[0]; break;
    case "kg": v = b.kg; break;
    case "mes": v = b.ms; break;
    default: v = m.map((mu, i) => ei(mu, sd[i], b.best, XI));
  }
  return { b, v, k: argmax(v) };
}

const nameOf = (v: string, lang: Lang = "en") => tr(ACQ.find((a) => a.value === v)!.label, lang);

function describe(st: State<P>): string {
  const lang = st.lang ?? "en";
  const { b, k } = acqValues(st.p);
  return tpl(labels[lang].describe, {
    n: b.obs.length, b: fixed(b.ys.length ? Math.max(...b.ys) : NaN, 2), m: fixed(TMAX, 2), name: nameOf(st.p.acq, lang),
    x1: fixed(GX[k][0], 2), x2: fixed(GX[k][1], 2), mu: fixed(b.ym + b.ysd * b.post.mean[k], 2), sd: fixed(b.ysd * b.sd[k], 2),
  });
}

// Cells snap to whole pixels and do not overlap, so translucent cells leave
// no darker seams where neighbours meet.
function heat(f: Frame, vals: number[], color: string, lo: number, hi: number, gamma = 1): string {
  const xe = Array.from({ length: G + 1 }, (_, i) => Math.round(f.left + (i * (f.right - f.left)) / G));
  const ye = Array.from({ length: G + 1 }, (_, j) => Math.round(f.bottom - (j * (f.bottom - f.top)) / G));
  const parts: string[] = [];
  for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) {
    const t = Math.max(0, Math.min(1, (vals[j * G + i] - lo) / (hi - lo || 1))) ** gamma;
    parts.push(el("rect", { x: xe[i], y: ye[j + 1], width: xe[i + 1] - xe[i], height: ye[j] - ye[j + 1], fill: color, opacity: (0.03 + 0.92 * t).toFixed(3) }));
  }
  return g({ "pointer-events": "none", "shape-rendering": "crispEdges" }, ...parts);
}

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 480;
  const { b, v, k } = acqValues(p);
  const parts: string[] = [];

  // 2 x 2 panels of equal squares.
  const pad = narrow ? 6 : 14;
  const side = Math.floor((st.w - 3 * pad - (narrow ? 6 : 20)) / 2);
  const x0 = narrow ? 6 : 20;
  const top = 26;
  const titleH = 18;
  const frames: Frame[] = [0, 1, 2, 3].map((q) => {
    const left = x0 + (q % 2) * (side + pad);
    const t = top + titleH + Math.floor(q / 2) * (side + titleH + pad + 4);
    return { x: linear([0, 1], [left, left + side]), y: linear([0, 1], [t + side, t]), left, right: left + side, top: t, bottom: t + side, w: st.w };
  });
  parts.push(text(x0, 16, tpl(L.readout, {
    n: b.obs.length, b: fixed(b.ys.length ? Math.max(...b.ys) : NaN, 2), m: fixed(TMAX, 2), name: nameOf(p.acq, lang), x1: fixed(GX[k][0], 2), x2: fixed(GX[k][1], 2),
  }), { "font-size": TYPE.small, class: "fig-t-num fig-t-strong" }));

  const meanRaw = b.post.mean.map((m) => b.ym + b.ysd * m);
  const tLo = -1.5; // Branin falls to about -4.7 in one corner; clip so the structure shows
  const panels: Array<{ title: string; vals: number[]; color: string; lo: number; hi: number; gamma?: number; hidden?: boolean }> = [
    { title: L.truth, vals: TRUTH, color: C.truth, lo: tLo, hi: TMAX, gamma: 1.6, hidden: !p.truth },
    { title: L.mean, vals: meanRaw, color: C.model, lo: tLo, hi: TMAX, gamma: 1.6 },
    { title: L.sd, vals: b.sd, color: C.c7, lo: 0, hi: Math.max(...b.sd) },
    { title: p.acq === "ts" ? L.tsName : nameOf(p.acq, lang), vals: v, color: C.acq, lo: Math.min(...v), hi: Math.max(...v), gamma: p.acq === "ts" ? 1.6 : 1 },
  ];
  panels.forEach((pn, q) => {
    const f = frames[q];
    parts.push(
      text(f.left, f.top - 6, pn.title, { "font-size": TYPE.small, class: q === 3 ? "fig-t-strong" : "fig-t-muted" }),
      el("rect", { x: f.left, y: f.top, width: side, height: side, fill: C.paper, stroke: C.rule }),
      pn.hidden ? text(f.left + side / 2, f.top + side / 2, L.hidden, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted" }) : heat(f, pn.vals, pn.color, pn.lo, pn.hi, pn.gamma),
      hitArea(f, `panel-${q}`),
    );
    if (q === 0 && !pn.hidden) {
      for (const [u, w] of MAXIMA) parts.push(text(f.x(u), f.y(w) + 4, "+", { "font-size": TYPE.label, "text-anchor": "middle", class: "fig-t-strong fig-t-halo", "pointer-events": "none" }));
    }
    parts.push(g({ "pointer-events": "none" },
      ...b.obs.map(([u, w]) => el("circle", { cx: f.x(u), cy: f.y(w), r: narrow ? 2.8 : 3.4, fill: C.ink, stroke: C.paper, "stroke-width": 1.2 })),
      el("circle", { cx: f.x(GX[k][0]), cy: f.y(GX[k][1]), r: narrow ? 5.5 : 7, fill: "none", stroke: C.paper, "stroke-width": 4.5 }),
      el("circle", { cx: f.x(GX[k][0]), cy: f.y(GX[k][1]), r: narrow ? 5.5 : 7, fill: "none", stroke: C.acq, "stroke-width": 2.2 }),
    ));
  });
  const H = frames[3].bottom + 12;
  return svg(st.w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "acq-2d",
  title: { en: "Acquisition functions on a two-dimensional problem", zh: "二维问题上的采集函数" },
  labels,
  params,
  hint: { en: "Click any panel to evaluate the objective there; the orange ring is the selected rule's next query.", zh: "点击任一面板，即在该处评估目标函数；橙色圆环是所选规则的下一个查询点。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click" || !e.target?.startsWith("panel-") || !e.data) return null;
    const pts = parsePoints(p.points);
    if (pts.length >= 40) pts.shift();
    pts.push([Math.min(1, Math.max(0, e.data.x)), Math.min(1, Math.max(0, e.data.y))]);
    return { ...p, points: fmtPoints(pts) };
  },
  actions: [
    {
      label: { en: "Evaluate the maximum", zh: "评估最大值处" }, primary: true,
      run: (p) => {
        const { k } = acqValues(p);
        const pts = parsePoints(p.points);
        if (pts.length >= 40) pts.shift();
        pts.push([GX[k][0], GX[k][1]]);
        return { ...p, points: fmtPoints(pts), seed: (p.seed % 9999) + 1 };
      },
    },
    { label: { en: "New samples", zh: "新样本" }, run: (p) => ({ ...p, seed: (p.seed % 9999) + 1 }) },
  ],
});
