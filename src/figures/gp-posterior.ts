// Gaussian process regression on points the reader places. Click the plot to
// add an observation at that position; click a point to remove it. The main
// panel shows the posterior mean, a 95% credible band, and optional sample
// paths; the strip below shows the posterior standard deviation, which
// collapses at observed inputs and recovers over about one lengthscale.

import { defineFigure, tr, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid } from "./lib/scale.ts";
import { fit, kernel, predict, samplePosterior, type KernelName } from "./lib/gp.ts";
import { fmtPoints, parsePoints, validPoints } from "./lib/params.ts";
import { rng } from "./lib/random.ts";
import { band, clip, curve, dots, frame, frameAxes, hitArea, legend } from "./lib/plot.ts";
import { sig, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "input x",
    y: "f(x)",
    sd: "posterior sd",
    mean: "posterior mean",
    band: "95% credible band",
    samples: "posterior samples",
    bumps: "weighted kernels α_i k(x, x_i)",
    obs: "observations",
    describe: "Gaussian process posterior with {n} observation{n:/s}, {k} kernel, lengthscale {l}, noise sd {s}. The posterior standard deviation ranges from {lo} at the data to {hi} far from it.",
    empty: "With no observations the posterior is the prior: mean zero, the same uncertainty everywhere.",
  },
  zh: {
    x: "输入 x",
    y: "f(x)",
    sd: "后验标准差",
    mean: "后验均值",
    band: "95% 可信区间",
    samples: "后验样本",
    bumps: "加权核函数 α_i k(x, x_i)",
    obs: "观测",
    describe: "高斯过程后验：{n} 个观测，{k} 核，长度尺度 {l}，噪声标准差 {s}。后验标准差从数据处的 {lo} 变化到远离数据处的 {hi}。",
    empty: "没有观测时，后验就是先验：均值为零，处处具有相同的不确定性。",
  },
};

const KERNELS = [
  { value: "rbf", label: { en: "RBF", zh: "RBF" } },
  { value: "matern52", label: { en: "Matérn 5/2", zh: "Matérn 5/2" } },
  { value: "matern12", label: { en: "Matérn 1/2", zh: "Matérn 1/2" } },
] as const;

const params = {
  kernel: { kind: "choice", label: { en: "Kernel", zh: "核函数" }, options: KERNELS, default: "rbf" },
  lengthscale: { kind: "range", label: { en: "Lengthscale ℓ", zh: "长度尺度 ℓ" }, min: 0.02, max: 0.6, default: 0.12, scale: "log" },
  noise: { kind: "range", label: { en: "Noise sd σ_n", zh: "噪声标准差 σ_n" }, min: 0, max: 0.4, default: 0.02, step: 0.01 },
  samples: { kind: "toggle", label: { en: "Show samples", zh: "显示样本" }, default: false },
  bumps: { kind: "toggle", label: { en: "Show kernel bumps", zh: "显示核函数鼓包" }, default: false },
  points: { kind: "data", label: { en: "Observations", zh: "观测" }, default: "0.1:-0.4,0.32:0.55,0.45:0.1,0.78:0.85", validate: validPoints },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 7, step: 1, control: false },
} as const;

type P = { kernel: string; lengthscale: number; noise: number; samples: boolean; bumps: boolean; points: string; seed: number };

const N = 160;
const XS = grid(0, 1, N);
const YDOM: [number, number] = [-2.6, 2.6];

function compute(p: P) {
  const pts = parsePoints(p.points);
  const k = kernel(p.kernel as KernelName, p.lengthscale, 1);
  const gp = fit(k, pts.map((q) => q[0]), pts.map((q) => q[1]), Math.max(p.noise * p.noise, 1e-6));
  const post = predict(gp, XS, p.samples);
  const sd = post.var.map(Math.sqrt);
  // The mean as a weighted sum of kernels centred on the data: alpha_i k(x, x_i).
  const bumps = p.bumps ? pts.map((q, i) => XS.map((x) => gp.alpha[i] * k(x, q[0]))) : [];
  return { pts, post, sd, bumps };
}

function describe(st: State<P>): string {
  const lang = st.lang ?? "en";
  const { pts, sd } = compute(st.p);
  if (!pts.length) return labels[lang].empty;
  return tpl(labels[lang].describe, {
    n: pts.length, k: tr(KERNELS.find((k) => k.value === st.p.kernel)!.label, lang), l: sig(st.p.lengthscale, 2), s: sig(st.p.noise, 2),
    lo: sig(Math.min(...sd), 2), hi: sig(Math.max(...sd), 2),
  });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const { pts, post, sd, bumps } = compute(p);
  const lg = legend(narrow ? 8 : 40, 14, st.w - 16, [
    { kind: "line", color: C.model, label: L.mean },
    { kind: "band", color: C.band, label: L.band },
    ...(p.samples ? [{ kind: "line" as const, color: C.c7, label: L.samples }] : []),
    ...(p.bumps ? [{ kind: "dash" as const, color: C.c5, label: L.bumps }] : []),
    { kind: "dot", color: C.ink, label: L.obs },
  ]);
  const top = 14 + lg.height + 6;
  const mainH = narrow ? 200 : 250;
  const f = frame({ w: st.w, top, height: mainH, yDomain: YDOM, yTitle: L.y });
  const stripTop = f.bottom + 22;
  const stripH = narrow ? 56 : 64;
  const fs = frame({ w: st.w, top: stripTop, height: stripH, yDomain: [0, 1.05], yTitle: narrow ? "" : "sd" });
  fs.x.range[0] = f.left;
  const H = stripTop + stripH + 40;
  const lo = post.mean.map((m, i) => m - 1.96 * sd[i]);
  const hi = post.mean.map((m, i) => m + 1.96 * sd[i]);
  const cid = `${st.uid}-clip`;
  const sampleLines = p.samples && post.cov
    ? samplePosterior(post, rng(p.seed), 3).map((s) => curve(f, XS, s, { stroke: C.c7, "stroke-width": 1.2, opacity: 0.75 })).join("")
    : "";
  const hit = hitArea(f);
  return svg(st.w, H, describe(st),
    el("defs", {}, clip(f, cid)),
    lg.svg,
    frameAxes(f, { yTitle: L.y, xTicks: false }),
    hit,
    g({ "clip-path": `url(#${cid})` },
      band(f, XS, lo, hi),
      sampleLines,
      bumps.map((b) => curve(f, XS, b, { stroke: C.c5, "stroke-width": 1.2, "stroke-dasharray": "4 3" })).join(""),
      curve(f, XS, post.mean),
    ),
    dots(f, pts, { hit: (i) => `pt-${i}` }),
    // the standard-deviation strip
    frameAxes({ ...fs, x: f.x, left: f.left }, { xTitle: L.x, yCount: 2 }),
    el("path", {
      d: `M${f.x(0)},${fs.y(0)}` + XS.map((x, i) => `L${f.x(x)},${fs.y(sd[i])}`).join("") + `L${f.x(1)},${fs.y(0)}Z`,
      fill: C.band, stroke: C.model, "stroke-width": 1.2,
    }),
    narrow ? text(f.left + 4, stripTop + 10, L.sd, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }) : text(f.right, stripTop + 10, L.sd, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo" }),
  );
}

export default defineFigure({
  name: "gp-posterior",
  title: { en: "Gaussian process regression on points you place", zh: "在你放置的点上做高斯过程回归" },
  labels,
  params,
  hint: { en: "Click the plot to add an observation; click a point to remove it.", zh: "点击图像添加一个观测；点击一个点将其删除。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click") return null;
    const pts = parsePoints(p.points);
    if (e.target?.startsWith("pt-")) {
      const i = Number(e.target.slice(3));
      pts.splice(i, 1);
      return { ...p, points: fmtPoints(pts) };
    }
    if (e.target !== "plot" || !e.data) return null;
    const x = Math.min(1, Math.max(0, e.data.x));
    const y = Math.min(YDOM[1], Math.max(YDOM[0], e.data.y));
    if (pts.length >= 25) pts.shift();
    pts.push([x, y]);
    return { ...p, points: fmtPoints(pts) };
  },
  actions: [
    { label: { en: "Clear", zh: "清除" }, run: (p) => ({ ...p, points: "" }), enabled: (p) => p.points !== "" },
  ],
});
