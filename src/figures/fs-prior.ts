// Functions drawn from a Gaussian process prior. The reader changes the
// kernel, its lengthscale, its amplitude, and (for the periodic kernel) its
// period, and the same three draws of the underlying standard normal vector z
// are pushed through each new Cholesky factor, so the functions deform
// continuously instead of jumping. The strip below shows the correlation
// between f(x0) and f(x), which is the kernel itself up to the amplitude. A
// readout compares the expected number of upward zero crossings on [0, 1]
// (Rasmussen and Williams 2006, eq. 4.3) with the count in the drawn samples.

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid } from "./lib/scale.ts";
import { kernel, samplePrior, type KernelName } from "./lib/gp.ts";
import { memo, rng } from "./lib/random.ts";
import { band, clip, curve, frame, frameAxes, hitArea, legend } from "./lib/plot.ts";
import { sig, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "input x",
    y: "f(x)",
    corr: "corr",
    corrLong: "correlation of f(x) with f(x₀)",
    samples: "functions drawn from the prior",
    band: "95% prior band (±1.96 σ_f)",
    expected: "Upward zero crossings on [0, 1]: {e} expected; {c} in these three draws",
    expectedShort1: "Upward zero crossings on [0, 1]:",
    expectedShort2: "{e} expected; {c} in these draws",
    roughLong: "Upward zero crossings on [0, 1]: no finite expectation (rough paths); {c} counted on this grid",
    roughShort2: "unbounded for rough paths; {c} on this grid",
    describe: "Three functions drawn from a Gaussian process prior with {k} kernel, lengthscale {l}, amplitude {a}{p}. {e} The correlation between f at x₀ = {x0} and f at distance {d} is {r}.",
    expectedSentence: "On average such a function crosses zero upward {e} times on [0, 1].",
    roughSentence: "Its paths are continuous but not differentiable, so they cross zero infinitely often near every crossing.",
    kRbf: "an RBF",
    kMatern52: "a Matérn 5/2",
    kMatern12: "a Matérn 1/2",
    kPeriodic: "a periodic",
    periodPart: ", period {p}",
  },
  zh: {
    x: "输入 x",
    y: "f(x)",
    corr: "相关",
    corrLong: "f(x) 与 f(x₀) 的相关系数",
    samples: "从先验中抽取的函数",
    band: "95% 先验区间（±1.96 σ_f）",
    expected: "[0, 1] 上向上穿零的次数：期望 {e} 次；这三个样本中为 {c}",
    expectedShort1: "[0, 1] 上向上穿零的次数：",
    expectedShort2: "期望 {e} 次；样本中为 {c}",
    roughLong: "[0, 1] 上向上穿零的次数：期望值不是有限的（路径粗糙）；在这个网格上计得 {c}",
    roughShort2: "路径粗糙，次数无界；在这个网格上为 {c}",
    describe: "从使用{k}的高斯过程先验中抽取的三个函数，长度尺度 {l}，幅度 {a}{p}。{e}x₀ = {x0} 处的 f 与相距 {d} 处的 f 的相关系数为 {r}。",
    expectedSentence: "这样的函数在 [0, 1] 上平均向上穿过零点 {e} 次。",
    roughSentence: "它的路径连续但不可微，所以在每次穿过零点的附近都会无穷多次穿过零点。",
    kRbf: "径向基函数核",
    kMatern52: " Matérn 5/2 核",
    kMatern12: " Matérn 1/2 核",
    kPeriodic: "周期核",
    periodPart: "，周期 {p}",
  },
};

const KERNELS = [
  { value: "rbf", label: { en: "RBF", zh: "RBF" } },
  { value: "matern52", label: { en: "Matérn 5/2", zh: "Matérn 5/2" } },
  { value: "matern12", label: { en: "Matérn 1/2", zh: "Matérn 1/2" } },
  { value: "periodic", label: { en: "Periodic", zh: "周期核" } },
] as const;

const params = {
  kernel: { kind: "choice", label: { en: "Kernel", zh: "核函数" }, options: KERNELS, default: "rbf" },
  lengthscale: { kind: "range", label: { en: "Lengthscale ℓ", zh: "长度尺度 ℓ" }, min: 0.02, max: 2, default: 0.1, scale: "log" },
  amplitude: { kind: "range", label: { en: "Amplitude σ_f", zh: "幅度 σ_f" }, min: 0.25, max: 2, default: 1, step: 0.05 },
  period: { kind: "range", label: { en: "Period p", zh: "周期 p" }, min: 0.15, max: 1, default: 0.4, step: 0.01 },
  x0: { kind: "range", label: { en: "Reference input x₀", zh: "参考输入 x₀" }, min: 0, max: 1, default: 0.3, step: 0.01 },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 46, step: 1, control: false },
  // The kernel before the last change, so switching out of the periodic
  // kernel can restore a lengthscale that suits the others.
  lastKernel: { kind: "data", label: { en: "Previous kernel", zh: "上一个核函数" }, default: "rbf" },
} as const;

type P = { kernel: string; lengthscale: number; amplitude: number; period: number; x0: number; seed: number; lastKernel: string };

const N = 201;
const XS = grid(0, 1, N);
const YDOM: [number, number] = [-4.2, 4.2];
const COLORS = [C.c7, C.c5, C.c6];

// Draws with unit amplitude; the amplitude only scales them, so changing it
// does not refactor the matrix.
const draws = memo((k: string, ell: number, period: number, seed: number) => {
  const kern = kernel(k as KernelName, ell, 1, period);
  return samplePrior(kern, XS, rng(seed), 3);
}, 24);

// Expected upward crossings of zero on the unit interval for a zero-mean
// stationary process: sqrt(-k''(0) / k(0)) / (2 pi), Rasmussen and Williams
// (2006), eq. (4.3). Matérn 1/2 has no second derivative at zero.
function expectedCrossings(k: string, ell: number, period: number): number | undefined {
  switch (k) {
    case "rbf": return 1 / (2 * Math.PI * ell);
    case "matern52": return Math.sqrt(5 / 3) / (2 * Math.PI * ell);
    case "periodic": return 1 / (period * ell);
    default: return undefined;
  }
}

function crossings(s: number[]): number {
  let n = 0;
  for (let i = 1; i < s.length; i++) if (s[i - 1] < 0 && s[i] >= 0) n++;
  return n;
}

function fmtE(e: number): string {
  return e >= 100 ? String(Math.round(e)) : sig(e, 2);
}

function describe(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const kern = kernel(p.kernel as KernelName, p.lengthscale, 1, p.period);
  const e = expectedCrossings(p.kernel, p.lengthscale, p.period);
  const d = Math.min(p.lengthscale, 0.5);
  return tpl(L.describe, {
    k: { rbf: L.kRbf, matern52: L.kMatern52, matern12: L.kMatern12, periodic: L.kPeriodic }[p.kernel] ?? p.kernel,
    l: sig(p.lengthscale, 2), a: sig(p.amplitude, 2),
    p: p.kernel === "periodic" ? tpl(L.periodPart, { p: sig(p.period, 2) }) : "",
    e: e === undefined ? L.roughSentence : tpl(L.expectedSentence, { e: fmtE(e) }),
    x0: sig(p.x0, 2), d: sig(d, 2), r: sig(kern(0, d), 2),
  });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const S = draws(p.kernel, p.lengthscale, p.period, p.seed).map((s) => s.map((v) => v * p.amplitude));
  const kern = kernel(p.kernel as KernelName, p.lengthscale, 1, p.period);
  const corr = XS.map((x) => kern(p.x0, x));
  const lg = legend(narrow ? 8 : 40, 14, st.w - 16, [
    { kind: "line", color: COLORS[0], label: L.samples },
    { kind: "band", color: C.band, label: L.band },
  ]);
  const top = 14 + lg.height + 6;
  const mainH = narrow ? 200 : 240;
  const f = frame({ w: st.w, top, height: mainH, yDomain: YDOM, yTitle: L.y });
  const stripTop = f.bottom + 30;
  const stripH = narrow ? 60 : 72;
  const fs = frame({ w: st.w, top: stripTop, height: stripH, yDomain: [0, 1.05], yTitle: L.corr });
  const e = expectedCrossings(p.kernel, p.lengthscale, p.period);
  const counts = S.map(crossings).join(st.lang === "zh" ? "、" : ", ");
  const lines = e === undefined
    ? (narrow ? [L.expectedShort1, tpl(L.roughShort2, { c: counts })] : [tpl(L.roughLong, { c: counts })])
    : (narrow ? [L.expectedShort1, tpl(L.expectedShort2, { e: fmtE(e), c: counts })] : [tpl(L.expected, { e: fmtE(e), c: counts })]);
  const readTop = fs.bottom + 56;
  const H = readTop + lines.length * 16 + 4;
  const cid = `${st.uid}-clip`;
  const x0px = f.x(p.x0);
  const i0 = Math.round(p.x0 * (N - 1));
  const sf = 1.96 * p.amplitude;
  return svg(st.w, H, describe(st),
    el("defs", {}, clip(f, cid)),
    lg.svg,
    frameAxes(f, { yTitle: L.y, xTicks: false }),
    hitArea(f, "plot"),
    g({ "clip-path": `url(#${cid})` },
      band(f, XS, XS.map(() => -sf), XS.map(() => sf)),
      el("line", { x1: f.left, x2: f.right, y1: f.y(0), y2: f.y(0), stroke: C.ink3, "stroke-width": 1 }),
      S.map((s, i) => curve(f, XS, s, { stroke: COLORS[i], "stroke-width": 1.6, opacity: 0.95 })).join(""),
    ),
    el("line", { x1: x0px, x2: x0px, y1: f.top, y2: fs.bottom, stroke: C.ink3, "stroke-width": 1, "stroke-dasharray": "2 3" }),
    S.map((s, i) => el("circle", { cx: x0px, cy: f.y(Math.max(YDOM[0], Math.min(YDOM[1], s[i0]))), r: 3.4, fill: COLORS[i], stroke: C.paper, "stroke-width": 1.4 })).join(""),
    text(x0px, f.top - 4, "x_0", { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
    // the correlation strip
    frameAxes(fs, { xTitle: L.x, yTitle: narrow ? "" : L.corr, yCount: 2 }),
    hitArea(fs, "strip"),
    el("path", {
      d: `M${fs.x(0)},${fs.y(0)}` + XS.map((x, i) => `L${fs.x(x)},${fs.y(Math.max(0, corr[i]))}`).join("") + `L${fs.x(1)},${fs.y(0)}Z`,
      fill: C.band, stroke: C.model, "stroke-width": 1.4,
    }),
    text(narrow ? fs.left + 4 : fs.right, stripTop - 6, L.corrLong, { "font-size": TYPE.small, "text-anchor": narrow ? "start" : "end", class: "fig-t-muted fig-t-halo" }),
    ...lines.map((s, i) => text(narrow ? 8 : f.left, readTop + i * 16, s, { "font-size": TYPE.small, class: "fig-t-num" })),
  );
}

export default defineFigure({
  name: "fs-prior",
  title: { en: "Functions drawn from a Gaussian process prior", zh: "从高斯过程先验中抽取的函数" },
  labels,
  params,
  hint: { en: "Click or drag on the plot to move x₀; the strip shows how strongly f at every other input moves with f(x₀).", zh: "在图上点击或拖动可以移动 x₀；条带显示其他每个输入处的 f 与 f(x₀) 共同变化的程度。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  // The periodic kernel's lengthscale is measured against the period, so a
  // value that suits the RBF makes it far too wiggly, and the reverse.
  update(p, key) {
    if (key !== "kernel") return p;
    const from = p.lastKernel;
    const q = { ...p, lastKernel: p.kernel };
    if (p.kernel === "periodic" && from !== "periodic" && p.lengthscale < 0.4) return { ...q, lengthscale: 0.8 };
    if (p.kernel !== "periodic" && from === "periodic" && p.lengthscale > 0.4) return { ...q, lengthscale: 0.1 };
    return q;
  },
  pointer(p, e) {
    if ((e.target !== "plot" && e.target !== "strip") || e.data === undefined) return null;
    return { ...p, x0: Math.round(Math.min(1, Math.max(0, e.data.x)) * 100) / 100 };
  },
  actions: [
    { label: { en: "Draw again", zh: "重新抽取" }, primary: true, run: (p) => ({ ...p, seed: (p.seed % 9999) + 1 }) },
  ],
});
