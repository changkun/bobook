// Choosing a batch of q evaluations at once. The posterior is fitted to the
// observations so far on the running objective; the figure then picks q
// points by one of four rules: the q largest values of expected improvement
// (the naive choice, which lands on neighbouring inputs), the kriging
// believer and the constant liar of Ginsbourger et al. (2010), which pick one
// point at a time and pretend its outcome is known before picking the next,
// and Thompson sampling with q independent posterior draws. The bottom panel
// shows the acquisition function after each fantasy. The readout estimates
// q-EI, the expected improvement of the best point in the batch, by Monte
// Carlo with fixed base samples, so batches from different rules can be
// compared on the same footing. "Evaluate the batch" adds the q outcomes.

import { defineFigure, tr, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid } from "./lib/scale.ts";
import { fit, kernel, predict, samplePosterior } from "./lib/gp.ts";
import { argmax, ei } from "./lib/acq.ts";
import { cholesky } from "./lib/linalg.ts";
import { memo, normals, rng } from "./lib/random.ts";
import { running } from "./lib/objectives.ts";
import { band, clip, curve, dots, frame, frameAxes, legend } from "./lib/plot.ts";
import { fmtPoints, parsePoints, validPoints } from "./lib/params.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "input x",
    f: "f(x)",
    ei: "expected improvement",
    truth: "hidden objective",
    mean: "posterior mean",
    band: "95% band",
    batch: "batch",
    fantasy: "fantasized outcome",
    samples: "posterior draws",
    tsNote: "Thompson sampling ignores this curve; its picks are the maxima of the draws above",
    qei: "q-EI of this batch ≈ {v}",
    qeiBest: "q-EI of this batch ≈ {v} · the best single point alone: {s}",
    spread: "{d} distinct region{d:/s}",
    describe: "A batch of {q} point{q:/s} chosen by {s} after {n} observations: x = {xs}. Its Monte Carlo q-EI is about {v}.",
  },
  zh: {
    x: "输入 x",
    f: "f(x)",
    ei: "期望改进",
    truth: "隐藏的目标函数",
    mean: "后验均值",
    band: "95% 区间",
    batch: "批量",
    fantasy: "虚拟观测",
    samples: "后验样本",
    tsNote: "Thompson 采样不使用这条曲线；它选的点是上方各样本的最大值点",
    qei: "本批的 q-EI ≈ {v}",
    qeiBest: "本批的 q-EI ≈ {v} · 单个最优点：{s}",
    spread: "{d} 个不同区域",
    describe: "在 {n} 个观测之后，按“{s}”规则选出的 {q} 个点：x = {xs}。其蒙特卡洛 q-EI 约为 {v}。",
  },
};

const STRATS = [
  { value: "top", label: { en: "Top q of EI", zh: "EI 前 q 名" } },
  { value: "kb", label: { en: "Kriging believer", zh: "克里金信念" } },
  { value: "cl", label: { en: "Constant liar", zh: "常数说谎者" } },
  { value: "ts", label: { en: "Thompson", zh: "Thompson 采样" } },
] as const;

const X0 = [0.06, 0.38, 0.5, 0.94];
const DEFAULT_OBS = fmtPoints(X0.map((x) => [x, running.f(x)]));

const params = {
  strategy: { kind: "choice", label: { en: "Batch rule", zh: "批量规则" }, options: STRATS, default: "kb" },
  q: { kind: "range", label: { en: "Batch size q", zh: "批量大小 q" }, min: 1, max: 6, default: 4, step: 1 },
  truth: { kind: "toggle", label: { en: "Show hidden objective", zh: "显示隐藏的目标函数" }, default: true },
  obs: { kind: "data", label: { en: "Observations", zh: "观测" }, default: DEFAULT_OBS, validate: validPoints },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 3, step: 1, control: false },
} as const;

type P = { strategy: string; q: number; truth: boolean; obs: string; seed: number };

const XS = grid(0, 1, 201);
const LS = 0.08, SF2 = 0.45, NOISE = 1e-5;
const K = kernel("rbf", LS, SF2);

interface Batch {
  obs: Array<[number, number]>;
  mean: number[]; sd: number[];
  picks: number[]; // indices into XS, in order
  fantasies: Array<[number, number]>; // (x, pretended y) for kb / cl
  acqs: number[][]; // acquisition curve before each pick (one curve for top and ts)
  draws: number[][]; // Thompson draws
  qei: number; single: number; regions: number;
}

const build = memo((strategy: string, q: number, obsStr: string, seed: number): Batch => {
  const obs = parsePoints(obsStr);
  const ym = obs.length ? obs.reduce((a, o) => a + o[1], 0) / obs.length : 0;
  const best = obs.length ? Math.max(...obs.map((o) => o[1])) : 0;
  const gp0 = fit(K, obs.map((o) => o[0]), obs.map((o) => o[1]), NOISE, ym);
  const post0 = predict(gp0, XS, strategy === "ts");
  const sd0 = post0.var.map(Math.sqrt);
  const ei0 = post0.mean.map((m, i) => ei(m, sd0[i], best, 0.001));
  const picks: number[] = [], fantasies: Array<[number, number]> = [], acqs: number[][] = [];
  let draws: number[][] = [];
  if (strategy === "top") {
    const order = ei0.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
    for (let j = 0; j < q; j++) picks.push(order[j][1]);
    acqs.push(ei0);
  } else if (strategy === "ts") {
    draws = samplePosterior(post0, rng(seed * 131 + q), q);
    for (const d of draws) picks.push(argmax(d));
    acqs.push(ei0);
  } else {
    const lie = Math.min(...obs.map((o) => o[1]));
    const xs = obs.map((o) => o[0]), ys = obs.map((o) => o[1]);
    for (let j = 0; j < q; j++) {
      const gp = fit(K, xs, ys, NOISE, ym);
      const post = j === 0 ? post0 : predict(gp, XS);
      const sd = post.var.map(Math.sqrt);
      const bestNow = Math.max(...ys);
      const a = post.mean.map((m, i) => ei(m, sd[i], bestNow, 0.001));
      acqs.push(a);
      const i = argmax(a);
      picks.push(i);
      const yFake = strategy === "kb" ? post.mean[i] : lie;
      fantasies.push([XS[i], yFake]);
      xs.push(XS[i]); ys.push(yFake);
    }
  }
  // q-EI of the chosen batch by Monte Carlo on the joint posterior, with
  // fixed base samples so that batches are compared on the same draws.
  const bx = picks.map((i) => XS[i]);
  const pb = predict(gp0, bx, true);
  const L = cholesky(pb.cov!, 1e-9);
  const z = rng(77);
  const S = 4000;
  let acc = 0;
  for (let s = 0; s < S; s++) {
    const e = normals(z, q);
    let mx = -Infinity;
    for (let a = 0; a < q; a++) {
      let v = pb.mean[a];
      for (let b = 0; b <= a; b++) v += L[a][b] * e[b];
      if (v > mx) mx = v;
    }
    acc += Math.max(mx - best, 0);
  }
  const sorted = bx.slice().sort((a, b) => a - b);
  let regions = sorted.length ? 1 : 0;
  for (let i = 1; i < sorted.length; i++) if (sorted[i] - sorted[i - 1] > LS) regions++;
  return { obs, mean: post0.mean, sd: sd0, picks, fantasies, acqs, draws, qei: acc / S, single: Math.max(...ei0), regions };
}, 24);

const run = (p: P) => build(p.strategy, p.q, p.obs, p.seed);

function describe(st: State<P>): string {
  const p = st.p;
  const b = run(p);
  const lang = st.lang ?? "en";
  return tpl(labels[lang].describe, {
    q: p.q, s: tr(STRATS.find((s) => s.value === p.strategy)!.label, lang), n: b.obs.length,
    xs: b.picks.map((i) => fixed(XS[i], 2)).join(", "), v: fixed(b.qei, 3),
  });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const b = run(p);
  const items = [
    ...(p.truth ? [{ kind: "dash" as const, color: C.truth, label: L.truth }] : []),
    { kind: "line" as const, color: C.model, label: L.mean },
    { kind: "band" as const, color: C.band, label: L.band },
    { kind: "line" as const, color: C.acq, label: L.batch },
    ...(p.strategy === "kb" || p.strategy === "cl" ? [{ kind: "dot" as const, color: C.acq, label: L.fantasy }] : []),
    ...(p.strategy === "ts" ? [{ kind: "line" as const, color: C.c7, label: L.samples }] : []),
  ];
  const lg = legend(narrow ? 8 : 40, 14, st.w - 16, items);
  const top = 14 + lg.height + 14;
  const f = frame({ w: st.w, top, height: narrow ? 170 : 210, yDomain: running.range, yTitle: L.f });
  const aTop = f.bottom + 18;
  const aH = narrow ? 80 : 96;
  const amax = Math.max(1e-9, ...b.acqs.flat());
  const fa = frame({ w: st.w, top: aTop, height: aH, yDomain: [0, amax * 1.12], yTitle: L.f });
  const cid = `${st.uid}-c`, cid2 = `${st.uid}-c2`;
  const lo = b.mean.map((m, i) => m - 1.96 * b.sd[i]);
  const hi = b.mean.map((m, i) => m + 1.96 * b.sd[i]);
  const parts: string[] = [el("defs", {}, clip(f, cid), clip(fa, cid2)), lg.svg, frameAxes(f, { yTitle: L.f, xTicks: false })];
  parts.push(g({ "clip-path": `url(#${cid})` },
    band(f, XS, lo, hi),
    p.truth ? curve(f, XS, XS.map(running.f), { stroke: C.truth, "stroke-dasharray": "5 4", "stroke-width": 1.6 }) : "",
    ...b.draws.map((d) => curve(f, XS, d, { stroke: C.c7, "stroke-width": 1.1, opacity: 0.7 })),
    curve(f, XS, b.mean),
  ));
  // batch markers with their order; picks closer than 14 px share one label
  const order = b.picks.map((i, j) => ({ x: f.x(XS[i]), i, j })).sort((a, c) => a.x - c.x);
  const groups: Array<{ xs: number[]; js: number[] }> = [];
  for (const o of order) {
    const gr = groups[groups.length - 1];
    if (gr && o.x - gr.xs[gr.xs.length - 1] < 14) { gr.xs.push(o.x); gr.js.push(o.j + 1); } else groups.push({ xs: [o.x], js: [o.j + 1] });
  }
  for (const o of order) {
    parts.push(el("line", { x1: o.x, x2: o.x, y1: f.top, y2: f.bottom, stroke: C.acq, "stroke-width": 1.4, "stroke-dasharray": "4 3" }));
    if (b.draws.length) parts.push(el("circle", { cx: o.x, cy: f.y(b.draws[o.j][o.i]), r: 3.5, fill: C.c7, stroke: C.paper, "stroke-width": 1.2 }));
  }
  for (const gr of groups) {
    const js = gr.js.slice().sort((a, c) => a - c);
    const contiguous = js.length > 2 && js[js.length - 1] - js[0] === js.length - 1;
    const label = contiguous ? `${js[0]}–${js[js.length - 1]}` : js.join(",");
    const cx = Math.min(f.right - 14, Math.max(f.left + 10, (gr.xs[0] + gr.xs[gr.xs.length - 1]) / 2));
    parts.push(text(cx, f.top - 4, label, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-strong fig-t-halo" }));
  }
  parts.push(dots(f, b.obs));
  for (const [x, y] of b.fantasies) parts.push(el("circle", { cx: f.x(x), cy: f.y(y), r: 4.4, fill: C.acq, stroke: C.paper, "stroke-width": 1.6 }));
  // acquisition panel
  parts.push(frameAxes(fa, { xTitle: L.x, yCount: 2 }));
  const fill = (a: number[]) => `M${fa.x(0)},${fa.y(0)}` + XS.map((x, i) => `L${fa.x(x).toFixed(1)},${fa.y(a[i]).toFixed(1)}`).join("") + `L${fa.x(1)},${fa.y(0)}Z`;
  const inner: string[] = [];
  b.acqs.forEach((a, j) => {
    if (j === 0) inner.push(el("path", { d: fill(a), fill: C.acqFill, stroke: C.acq, "stroke-width": 1.6, opacity: p.strategy === "ts" ? 0.45 : 1 }));
    else inner.push(curve(fa, XS, a, { stroke: C.acq, "stroke-width": 1.2, opacity: Math.max(0.35, 1 - j * 0.15) }));
  });
  parts.push(g({ "clip-path": `url(#${cid2})` }, ...inner));
  b.picks.forEach((i, j) => {
    const a = b.acqs[p.strategy === "top" || p.strategy === "ts" ? 0 : j];
    if (p.strategy !== "ts") parts.push(el("circle", { cx: fa.x(XS[i]), cy: fa.y(a[i]), r: 3.6, fill: C.acq, stroke: C.paper, "stroke-width": 1.2 }));
  });
  parts.push(text(fa.right - 6, fa.top + 12, p.strategy === "ts" ? (narrow ? L.ei : L.tsNote) : L.ei, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo" }));
  const ry = fa.bottom + 44;
  parts.push(text(fa.left, ry, `${tpl(narrow ? L.qei : L.qeiBest, { v: fixed(b.qei, 3), s: fixed(b.single, 3) })} · ${tpl(L.spread, { d: b.regions })}`, { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
  const H = ry + 10;
  return svg(st.w, H, describe(st), ...parts);
}
function evaluate(p: P): P {
  const b = run(p);
  const obs = b.obs.slice();
  for (const i of b.picks) if (!obs.some((o) => Math.abs(o[0] - XS[i]) < 1e-9)) obs.push([XS[i], running.f(XS[i])]);
  return { ...p, obs: fmtPoints(obs.slice(-24)), seed: (p.seed % 9999) + 1 };
}

export default defineFigure({
  name: "practice-batch",
  title: { en: "Choosing a batch of evaluations: naive top-q, kriging believer, constant liar, and Thompson sampling", zh: "选择一批评估：朴素的前 q 名、克里金信念、常数说谎者与 Thompson 采样" },
  labels,
  params,
  hint: { en: "Change the rule and the batch size; press “Evaluate the batch” to run the q evaluations and choose the next batch.", zh: "更改规则与批量大小；按“评估这一批”运行这 q 次评估，并选择下一批。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "Evaluate the batch", zh: "评估这一批" }, primary: true, run: (p) => evaluate(p as P) },
  ],
});
