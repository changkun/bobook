// How much T noisy evaluations teach, kernel by kernel. For T evaluations
// spread evenly over [0, 1] (at the midpoints (t - 1/2)/T), the information
// about f is (1/2) log det(I + K_T / σ_n²) (foundations, eq. info-gp-gain),
// computed exactly. K_T is a symmetric Toeplitz matrix because the inputs are
// equally spaced, so its log determinant comes from the Levinson-Durbin
// recursion in O(T²) steps: the product of the one-step prediction error
// variances. The value is a lower bound on the maximum information gain γ_T,
// which maximizes over all designs. On log axes a power law T^a is a straight
// line of slope a; the dotted segments show the slopes 1/(2ν + 1) that the
// eigenvalue decay of each Matérn kernel predicts for large T in one
// dimension, and the dashed line is the information of T evaluations that are
// each completely new.

import { defineFigure, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { log } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { kernel, type KernelName } from "./lib/gp.ts";
import { memo } from "./lib/random.ts";
import { legend } from "./lib/plot.ts";
import { fixed, sig, tpl } from "./lib/format.ts";

const labels = {
  en: {
    title: "Information from T evaluations spread evenly over [0, 1], in nats",
    T: "evaluations T",
    allNew: "every evaluation new",
    slope: "slope 1/(2ν+1)",
    row: "{k}: {i} nats at T = {t}; local slope {s}; large-T slope {a}",
    rowRbf: "{k}: {i} nats at T = {t}; local slope {s}; slope tends to 0",
    rowA: "{k}: {i} nats at T = {t};",
    rowB: "local slope {s}; large-T slope {a}",
    rowBRbf: "local slope {s}; slope tends to 0",
    describe: "With lengthscale {l} and noise standard deviation {n}, T = {t} evaluations spread evenly over [0, 1] gather {r} nats under the RBF kernel, {a} under Matérn 5/2, {b} under Matérn 3/2, and {c} under Matérn 1/2.",
  },
  zh: {
    title: "均匀分布在 [0, 1] 上的 T 次评估所获得的信息（奈特）",
    T: "评估次数 T",
    allNew: "每次评估都是全新的",
    slope: "斜率 1/(2ν+1)",
    row: "{k}：T = {t} 时为 {i} 奈特；局部斜率 {s}；大 T 极限斜率 {a}",
    rowRbf: "{k}：T = {t} 时为 {i} 奈特；局部斜率 {s}；斜率趋于 0",
    rowA: "{k}：T = {t} 时为 {i} 奈特；",
    rowB: "局部斜率 {s}；大 T 极限斜率 {a}",
    rowBRbf: "局部斜率 {s}；斜率趋于 0",
    describe: "长度尺度为 {l}、噪声标准差为 {n} 时，均匀分布在 [0, 1] 上的 T = {t} 次评估所获得的信息为：径向基函数核 {r} 奈特，Matérn 5/2 {a} 奈特，Matérn 3/2 {b} 奈特，Matérn 1/2 {c} 奈特。",
  },
};

const KERNELS = [
  { value: "rbf", label: "RBF", nu: Infinity },
  { value: "matern52", label: "Matérn 5/2", nu: 2.5 },
  { value: "matern32", label: "Matérn 3/2", nu: 1.5 },
  { value: "matern12", label: "Matérn 1/2", nu: 0.5 },
] as const;
const COLOR: Record<string, string> = { rbf: C.c7, matern52: C.c6, matern32: C.c8, matern12: C.c5 };
const TMAX = 1000;

const params = {
  lengthscale: { kind: "range", label: { en: "Lengthscale ℓ", zh: "长度尺度 ℓ" }, min: 0.03, max: 0.5, default: 0.1, scale: "log" },
  noise: { kind: "range", label: { en: "Noise sd σ_n", zh: "噪声标准差 σ_n" }, min: 0.01, max: 1, default: 0.1, scale: "log" },
  evals: { kind: "range", label: { en: "Evaluations T", zh: "评估次数 T" }, min: 1, max: TMAX, default: 100, scale: "log" },
} as const;

type P = { lengthscale: number; noise: number; evals: number };

// log det of the symmetric positive definite Toeplitz matrix with first row r,
// by the Durbin recursion: the product of the prediction error variances.
export function logdetToeplitz(r: Float64Array): number {
  const n = r.length;
  let E = r[0];
  let ld = Math.log(E);
  let a = new Float64Array(n), next = new Float64Array(n);
  for (let k = 1; k < n; k++) {
    let acc = r[k];
    for (let j = 1; j < k; j++) acc -= a[j] * r[k - j];
    const kap = acc / E;
    next[k] = kap;
    for (let j = 1; j < k; j++) next[j] = a[j] - kap * a[k - j];
    [a, next] = [next, a];
    E *= 1 - kap * kap;
    ld += Math.log(E);
  }
  return ld;
}

// Information of T evenly spread evaluations.
export function infoEven(kname: string, ls: number, sn: number, T: number): number {
  const k = kernel(kname as KernelName, ls, 1);
  const s2 = sn * sn;
  const r = new Float64Array(T);
  for (let j = 0; j < T; j++) r[j] = k(0, j / T) / s2 + (j === 0 ? 1 : 0);
  return 0.5 * logdetToeplitz(r);
}

const TS = (() => {
  const out = new Set<number>();
  for (let i = 0; i <= 60; i++) out.add(Math.round(10 ** ((3 * i) / 60)));
  return [...out].sort((a, b) => a - b);
})();

const curves = memo((kname: string, ls: number, sn: number) => TS.map((T) => infoEven(kname, ls, sn, T)), 16);
const at = memo((kname: string, ls: number, sn: number, T: number) => infoEven(kname, ls, sn, T), 64);

export function compute(p: P) {
  const T = Math.max(1, Math.min(TMAX, Math.round(p.evals)));
  const T1 = Math.max(1, Math.round(T / 2)), T2 = Math.min(TMAX, T * 2);
  const rows = KERNELS.map((k) => {
    const ys = curves(k.value, p.lengthscale, p.noise);
    const iT = at(k.value, p.lengthscale, p.noise, T);
    const slope = T2 > T1 ? Math.log(at(k.value, p.lengthscale, p.noise, T2) / at(k.value, p.lengthscale, p.noise, T1)) / Math.log(T2 / T1) : 1;
    return { k, ys, iT, slope };
  });
  return { T, rows, one: 0.5 * Math.log(1 + 1 / (p.noise * p.noise)) };
}

function describe(st: State<P>): string {
  const p = st.p;
  const c = compute(p);
  const v = (i: number) => fixed(c.rows[i].iT, 1);
  return tpl(labels[st.lang ?? "en"].describe, { l: sig(p.lengthscale, 2), n: sig(p.noise, 2), t: c.T, r: v(0), a: v(1), b: v(2), c: v(3) });
}

const SUP: Record<string, string> = { "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴" };

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const c = compute(p);
  const parts: string[] = [];
  const lg = legend(4, 30, st.w - 8, [
    ...KERNELS.map((k) => ({ kind: "line" as const, color: COLOR[k.value], label: k.label })),
    { kind: "dash" as const, color: C.ink3, label: L.allNew },
  ]);
  parts.push(text(4, 14, L.title, { "font-size": TYPE.small, class: "fig-t-strong" }), lg.svg);
  const top = 30 + lg.height;
  const h = narrow ? 210 : 250;
  const left = narrow ? 44 : 54, right = st.w - 12;
  const xs = log([1, TMAX], [left, right]);
  const ymin = c.one * 0.5;
  const ymax = Math.max(...c.rows.map((r) => r.ys[r.ys.length - 1])) * 3;
  const ys = log([ymin, ymax], [top + h, top]);
  const yTicks = [0.1, 1, 10, 100, 1000, 10000, 100000].filter((v) => v >= ymin && v <= ymax);
  parts.push(
    axis({ scale: ys, orient: "left", at: left, span: [left, right], ticks: yTicks, format: (v) => (v >= 1000 ? `${v / 1000}k` : String(v)) }),
    axis({ scale: xs, orient: "bottom", at: top + h, span: [top, top + h], ticks: [1, 10, 100, 1000], title: L.T, format: (v) => `10${SUP[String(Math.round(Math.log10(v)))]}` }),
  );
  const cid = `${st.uid}-clip`;
  const inner: string[] = [];
  // every evaluation new: T times the information of one
  inner.push(el("path", { d: linePath([[xs(1), ys(c.one)], [xs(TMAX), ys(c.one * TMAX)]]), fill: "none", stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "5 4" }));
  for (const r of c.rows) {
    inner.push(el("path", { d: linePath(TS.map((T, i) => [xs(T), ys(r.ys[i])])), fill: "none", stroke: COLOR[r.k.value], "stroke-width": 2.2 }));
    if (Number.isFinite(r.k.nu)) {
      // dotted slope segment through the curve's value at T = 1000, back to T = 100
      const a = 1 / (2 * r.k.nu + 1);
      const yEnd = r.ys[r.ys.length - 1];
      inner.push(el("path", { d: linePath([[xs(100), ys(yEnd * Math.pow(0.1, a))], [xs(TMAX), ys(yEnd)]]), fill: "none", stroke: COLOR[r.k.value], "stroke-width": 1.4, "stroke-dasharray": "1.5 3", opacity: 0.9 }));
    }
  }
  inner.push(el("line", { x1: xs(c.T), x2: xs(c.T), y1: top, y2: top + h, stroke: C.ink2, "stroke-width": 1, "stroke-dasharray": "3 3" }));
  for (const r of c.rows) inner.push(el("circle", { cx: xs(c.T), cy: ys(r.iT), r: 3.6, fill: COLOR[r.k.value], stroke: C.paper, "stroke-width": 1.4 }));
  parts.push(el("defs", {}, el("clipPath", { id: cid }, el("rect", { x: left, y: top - 4, width: right - left, height: h + 8 }))), g({ "clip-path": `url(#${cid})` }, ...inner));
  parts.push(text(right - 4, top + 14, `···  ${L.slope}`, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo" }));

  let ry = top + h + 50;
  for (const r of c.rows) {
    const vals = { k: r.k.label, i: fixed(r.iT, 1), t: c.T, s: fixed(r.slope, 2), a: Number.isFinite(r.k.nu) ? `1/${2 * r.k.nu + 1} ≈ ${fixed(1 / (2 * r.k.nu + 1), 2)}` : "" };
    const fin = Number.isFinite(r.k.nu);
    parts.push(el("circle", { cx: 9, cy: ry - 4, r: 3.6, fill: COLOR[r.k.value] }));
    if (narrow) {
      parts.push(text(18, ry, tpl(L.rowA, vals), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
      ry += 15;
      parts.push(text(18, ry, tpl(fin ? L.rowB : L.rowBRbf, vals), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
      ry += 19;
    } else {
      parts.push(text(18, ry, tpl(fin ? L.row : L.rowRbf, vals), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
      ry += 16;
    }
  }
  return svg(st.w, ry, describe(st), ...parts);
}

export default defineFigure({
  name: "ka-infogain",
  title: { en: "Information gathered by evenly spread evaluations under four kernels", zh: "四种核函数下均匀分布的评估所获得的信息" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  snapshots: {
    "T = 1000": (p) => ({ ...p, evals: 1000 }),
    "lengthscale 0.3": (p) => ({ ...p, lengthscale: 0.3, evals: 1000 }),
  },
});
