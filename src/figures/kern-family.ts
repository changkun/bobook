// The kernel family side by side. Six small panels, each a kernel; every
// panel pushes the same two standard normal vectors through its own Cholesky
// factor, so differences between panels come from the kernel and not from
// the random numbers. Under each panel is the kernel itself as a function of
// x for a fixed x0 = 0.7. One set shows the building blocks (the Matérn
// ladder up to the RBF, the periodic kernel, the linear kernel); the other
// shows sums and products of them.

import { defineFigure, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { cholesky } from "./lib/linalg.ts";
import { memo, normals, rng } from "./lib/random.ts";
import { sig, tpl } from "./lib/format.ts";

const labels = {
  en: {
    kernelRow: "Under each panel: the kernel k(0.7, x) as a function of x, scaled to the panel.",
    kernelRowShort: "Under each panel: the kernel k(0.7, x).",
    describeBase: "Six kernels with lengthscale {l}, each shown by two functions drawn with the same random numbers: Matérn 1/2, 3/2, and 5/2 grow smoother toward the RBF; the periodic kernel repeats with period {p}; the linear kernel draws straight lines.",
    describeCombo: "Six kernels built from sums and products with lengthscale {l}: a slow trend plus a fast wiggle, a trend plus a periodic part, a periodic kernel times an RBF (repetition that drifts), a line plus a periodic part, a product of two linear kernels (parabolas), and a linear kernel times a periodic one (oscillations that grow).",
  },
  zh: {
    kernelRow: "每个面板下方：核函数 k(0.7, x) 随 x 的变化，已按面板缩放。",
    kernelRowShort: "每个面板下方：核函数 k(0.7, x)。",
    describeBase: "六个长度尺度为 {l} 的核函数，每个都用同一组随机数抽取的两个函数来展示：Matérn 1/2、3/2 和 5/2 越来越光滑，趋近径向基函数核；周期核以周期 {p} 重复；线性核抽出直线。",
    describeCombo: "由和与积构造、长度尺度为 {l} 的六个核函数：缓慢的趋势加快速的波动，趋势加周期部分，周期核乘以径向基函数核（会漂移的重复），直线加周期部分，两个线性核之积（抛物线），以及线性核乘以周期核（不断增大的振荡）。",
  },
};

type K1 = (a: number, b: number) => number;

const PERIOD = 0.25;
const rbf = (l: number): K1 => (a, b) => Math.exp(-((a - b) ** 2) / (2 * l * l));
const m12 = (l: number): K1 => (a, b) => Math.exp(-Math.abs(a - b) / l);
const m32 = (l: number): K1 => (a, b) => { const r = (Math.sqrt(3) * Math.abs(a - b)) / l; return (1 + r) * Math.exp(-r); };
const m52 = (l: number): K1 => (a, b) => { const r = (Math.sqrt(5) * Math.abs(a - b)) / l; return (1 + r + (r * r) / 3) * Math.exp(-r); };
const per = (l: number, p = PERIOD): K1 => (a, b) => Math.exp((-2 * Math.sin((Math.PI * (a - b)) / p) ** 2) / (l * l));
// Linear kernel centered on the middle of the domain: an offset variance plus
// a slope variance times (x - c)(x' - c).
const lin = (sb = 0.6, sv = 1.8, c = 0.5): K1 => (a, b) => sb * sb + sv * sv * (a - c) * (b - c);
const add = (k1: K1, k2: K1, w1 = 1, w2 = 1): K1 => (a, b) => w1 * k1(a, b) + w2 * k2(a, b);
const mul = (k1: K1, k2: K1): K1 => (a, b) => k1(a, b) * k2(a, b);

interface Panel { title: string; zh: string; k: (l: number) => K1 }

const SETS: Record<string, Panel[]> = {
  base: [
    { title: "Matérn 1/2", zh: "Matérn 1/2", k: m12 },
    { title: "Matérn 3/2", zh: "Matérn 3/2", k: m32 },
    { title: "Matérn 5/2", zh: "Matérn 5/2", k: m52 },
    { title: "RBF", zh: "RBF", k: rbf },
    { title: "Periodic", zh: "周期核", k: () => per(1) },
    { title: "Linear", zh: "线性核", k: () => lin() },
  ],
  combo: [
    { title: "RBF (long) + RBF (short)", zh: "RBF（长）+ RBF（短）", k: (l) => add(rbf(Math.min(1.5, 4 * l)), rbf(l / 3), 0.8, 0.1) },
    { title: "RBF + Periodic", zh: "RBF + 周期核", k: (l) => add(rbf(Math.min(1.5, 3 * l)), per(1), 0.7, 0.3) },
    { title: "RBF × Periodic", zh: "RBF × 周期核", k: (l) => mul(rbf(Math.min(1.5, 2.5 * l)), per(1)) },
    { title: "Linear + Periodic", zh: "线性核 + 周期核", k: () => add(lin(0.4, 2.2), per(1), 0.7, 0.2) },
    { title: "Linear × Linear", zh: "线性核 × 线性核", k: () => mul(lin(0.5, 2), lin(0.5, 2)) },
    { title: "Linear × Periodic", zh: "线性核 × 周期核", k: () => mul(lin(0.15, 2.6), per(1)) },
  ],
};

const params = {
  set: { kind: "choice", label: { en: "Kernels", zh: "核函数" }, options: [{ value: "base", label: { en: "Building blocks", zh: "基本构件" } }, { value: "combo", label: { en: "Sums and products", zh: "和与积" } }], default: "base" },
  lengthscale: { kind: "range", label: { en: "Lengthscale ℓ", zh: "长度尺度 ℓ" }, min: 0.04, max: 0.5, default: 0.12, scale: "log" },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 4, step: 1, control: false },
} as const;

type P = { set: string; lengthscale: number; seed: number };

const N = 121;
const XS = grid(0, 1, N);
const YD: [number, number] = [-3, 3];
const COLORS = [C.c7, C.c6];

const compute = memo((set: string, ell: number, seed: number) => {
  const r = rng(seed);
  const Z = [normals(r, N), normals(r, N)];
  return SETS[set].map((panel) => {
    const k = panel.k(ell);
    const K = XS.map((a) => XS.map((b) => k(a, b)));
    const L = cholesky(K, 1e-8);
    const draws = Z.map((z) => XS.map((_, i) => { let s = 0; for (let j = 0; j <= i; j++) s += L[i][j] * z[j]; return s; }));
    const kern = XS.map((x) => k(0.7, x));
    return { title: panel.title, zh: panel.zh, draws, kern };
  });
}, 16);

function describe(st: State<P>): string {
  const L = labels[st.lang ?? "en"];
  return tpl(st.p.set === "base" ? L.describeBase : L.describeCombo, { l: sig(st.p.lengthscale, 2), p: PERIOD });
}

function render(st: State<P>): string {
  const p = st.p;
  const narrow = st.w < 480;
  const cols = narrow ? 2 : 3;
  const panels = compute(p.set, p.lengthscale, p.seed);
  const gap = narrow ? 8 : 14;
  const left = 6, right = st.w - 6;
  const pw = (right - left - gap * (cols - 1)) / cols;
  const drawH = narrow ? 84 : 104, kernH = narrow ? 26 : 30, titleH = 18, rowGap = 14;
  const cellH = titleH + drawH + 6 + kernH + rowGap;
  const parts: string[] = [];
  panels.forEach((panel, idx) => {
    const cx = left + (idx % cols) * (pw + gap);
    const cy = 6 + Math.floor(idx / cols) * cellH;
    const sx = linear([0, 1], [cx + 2, cx + pw - 2]);
    const sy = linear(YD, [cy + titleH + drawH, cy + titleH]);
    const cid = `${st.uid}-c${idx}`;
    const kmax = Math.max(1e-9, ...panel.kern.map(Math.abs));
    const kTop = cy + titleH + drawH + 6;
    const ky = linear([Math.min(0, ...panel.kern) / kmax, 1], [kTop + kernH, kTop + 2]);
    parts.push(
      el("defs", {}, el("clipPath", { id: cid }, el("rect", { x: cx, y: cy + titleH, width: pw, height: drawH }))),
      text(cx + 2, cy + 12, st.lang === "zh" ? panel.zh : panel.title, { "font-size": narrow ? TYPE.small : TYPE.body, class: "fig-t-strong" }),
      el("rect", { x: cx, y: cy + titleH, width: pw, height: drawH, fill: C.panel, stroke: C.grid, rx: 3 }),
      el("line", { x1: cx, x2: cx + pw, y1: sy(0), y2: sy(0), stroke: C.rule, "stroke-width": 1 }),
      g({ "clip-path": `url(#${cid})` },
        ...panel.draws.map((d, i) => el("path", { d: linePath(XS.map((x, j) => [sx(x), sy(d[j])])), fill: "none", stroke: COLORS[i], "stroke-width": 1.5, "stroke-linejoin": "round" }))),
      // the kernel slice
      el("line", { x1: cx, x2: cx + pw, y1: ky(0), y2: ky(0), stroke: C.rule, "stroke-width": 1 }),
      el("path", { d: linePath(XS.map((x, j) => [sx(x), ky(panel.kern[j] / kmax)])), fill: "none", stroke: C.model, "stroke-width": 1.5 }),
    );
  });
  const rows = Math.ceil(panels.length / cols);
  const noteY = 6 + rows * cellH - rowGap + 16;
  const L = labels[st.lang ?? "en"];
  parts.push(text(left + 2, noteY, narrow ? L.kernelRowShort : L.kernelRow, { "font-size": TYPE.small, class: "fig-t-muted" }));
  const H = noteY + 8;
  return svg(st.w, H, describe(st), g({}, ...parts));
}

export default defineFigure({
  name: "kern-family",
  title: { en: "Kernels side by side: the same random numbers through six kernels", zh: "并排比较核函数：同一组随机数经过六个核函数" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "Draw again", zh: "重新抽取" }, primary: true, run: (p) => ({ ...p, seed: (p.seed % 9999) + 1 }) },
  ],
});
