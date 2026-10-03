// How much T evaluations can teach a Gaussian process, by input dimension.
// For d = 1, 2, 3, 6, and 10, a fixed Latin hypercube of 800 candidate points
// in the unit cube stands in for the domain. On each, the greedy rule
// (evaluate the candidate with the largest posterior variance) is run for 100
// steps, and the information gained, the sum of (1/2) log(1 + var/s^2), is
// plotted against T. Greedy reaches at least 1 - 1/e of the maximum over all
// sets of T candidates, so each curve is a lower estimate of gamma_T on that
// candidate set, the same estimator as the regret chapter's figure. The
// dashed line is T times the gain of one evaluation at prior variance: every
// evaluation completely new. The lower panel shows the largest posterior
// standard deviation left on the candidates. Optionally the lengthscale is
// multiplied by sqrt(d), which keeps the correlation between two typical
// random points roughly fixed as d grows.
//
// The posterior covariance with the candidates is maintained by a rank-one
// update per evaluation: with u_j the scaled covariance vectors of earlier
// evaluations, the covariance of candidate i with the new point x is
// k(i, x) - sum_j u_j[i] u_j[x], and the new u is that vector divided by
// sqrt(var(x) + s^2).

import { defineFigure, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, CATEGORICAL, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { memo, rng } from "./lib/random.ts";
import { latinHypercube, type Pt } from "./lib/nd.ts";
import { fixed, sig, tpl } from "./lib/format.ts";

const labels = {
  en: {
    T: "evaluations T",
    info: "information, nats",
    sd: "largest sd left",
    indep: "every evaluation new",
    dim: "d = {d}",
    cover: "evaluations until no candidate has sd above 0.5:",
    coverItem: "d = {d}: {t}",
    never: "> 100",
    lsLine: "lengthscale {l}{scaled}, noise sd {s}",
    scaledNote: " × √d",
    describe: "Greedy information gain after 100 evaluations with lengthscale {l}{scaled} and noise sd {s}: {list} nats for d = 1, 2, 3, 6, 10.",
  },
  zh: {
    T: "评估次数 T",
    info: "信息量（奈特）",
    sd: "剩余的最大标准差",
    indep: "每次评估都是全新的",
    dim: "d = {d}",
    cover: "直到没有候选点的标准差高于 0.5 所需的评估次数：",
    coverItem: "d = {d}: {t}",
    never: "超过 100",
    lsLine: "长度尺度 {l}{scaled}，噪声标准差 {s}",
    scaledNote: " × √d",
    describe: "长度尺度为 {l}{scaled}、噪声标准差为 {s} 时，贪心规则在 100 次评估后获得的信息增益：d = 1、2、3、6、10 时分别为 {list} 奈特。",
  },
};

const params = {
  kernel: { kind: "choice", label: { en: "Kernel", zh: "核函数" }, options: [{ value: "rbf", label: { en: "RBF", zh: "径向基函数核" } }, { value: "matern52", label: { en: "Matérn 5/2", zh: "Matérn 5/2 核" } }], default: "rbf", control: "buttons" },
  ls: { kind: "range", label: { en: "Lengthscale ℓ", zh: "长度尺度 ℓ" }, min: 0.05, max: 1, default: 0.2, scale: "log" },
  noise: { kind: "range", label: { en: "Noise sd σₙ", zh: "噪声标准差 σₙ" }, min: 0.01, max: 1, default: 0.1, scale: "log" },
  scaled: { kind: "toggle", label: { en: "Scale ℓ by √d", zh: "ℓ 乘以 √d" }, default: false },
} as const;

type P = { kernel: "rbf" | "matern52"; ls: number; noise: number; scaled: boolean };

const DIMS = [1, 2, 3, 6, 10];
const N = 800;
const TMAX = 100;

const domain = memo((d: number): Pt[] => latinHypercube(rng(9100 + d), N, d), 8);

function kfun(kind: string, ell: number) {
  return kind === "rbf"
    ? (r2: number) => Math.exp(-0.5 * r2 / (ell * ell))
    : (r2: number) => { const r = Math.sqrt(5 * r2) / ell; return (1 + r + (r * r) / 3) * Math.exp(-r); };
}

const run = memo((kind: string, ell: number, noise: number, d: number) => {
  const X = domain(d);
  const k = kfun(kind, ell);
  const s2 = noise * noise;
  const v = new Float64Array(N).fill(1);
  const U: Float64Array[] = [];
  const info: number[] = [], maxSd: number[] = [];
  let sum = 0;
  for (let t = 0; t < TMAX; t++) {
    let bi = 0;
    for (let i = 1; i < N; i++) if (v[i] > v[bi]) bi = i;
    const vb = Math.max(v[bi], 0);
    sum += 0.5 * Math.log(1 + vb / s2);
    info.push(sum);
    const xb = X[bi];
    const c = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const xi = X[i];
      let r2 = 0;
      for (let q = 0; q < d; q++) { const dd = xi[q] - xb[q]; r2 += dd * dd; }
      let cv = k(r2);
      for (const u of U) cv -= u[i] * u[bi];
      c[i] = cv;
    }
    const den = Math.sqrt(vb + s2);
    const u = new Float64Array(N);
    for (let i = 0; i < N; i++) { u[i] = c[i] / den; v[i] -= u[i] * u[i]; }
    U.push(u);
    let m = 0;
    for (let i = 0; i < N; i++) if (v[i] > m) m = v[i];
    maxSd.push(Math.sqrt(Math.max(0, m)));
  }
  return { info, maxSd };
}, 40);

function all(p: P) {
  return DIMS.map((d) => run(p.kernel, p.ls * (p.scaled ? Math.sqrt(d) : 1), p.noise, d));
}

function coverT(maxSd: number[]): number {
  const i = maxSd.findIndex((s) => s <= 0.5);
  return i < 0 ? -1 : i + 1;
}

function describe(st: State<P>): string {
  const p = st.p, r = all(p), L = labels[st.lang ?? "en"];
  return tpl(L.describe, { l: sig(p.ls, 2), scaled: p.scaled ? L.scaledNote : "", s: sig(p.noise, 2), list: r.map((x) => fixed(x.info[TMAX - 1], 0)).join(st.lang === "zh" ? "、" : ", ") });
}

function render(st: State<P>): string {
  const p = st.p, L = labels[st.lang ?? "en"], w = st.w;
  const narrow = w < 480;
  const r = all(p);
  const colors = [CATEGORICAL[0], CATEGORICAL[1], CATEGORICAL[2], CATEGORICAL[4], CATEGORICAL[6]];
  const left = narrow ? 44 : 56, right = w - (narrow ? 12 : 90);
  const top = 16;
  const h1 = narrow ? 170 : 210, h2 = narrow ? 80 : 96;
  const gain1 = 0.5 * Math.log(1 + 1 / (p.noise * p.noise));
  const imax = Math.max(...r.map((x) => x.info[TMAX - 1])) * 1.08;
  const x = linear([0, TMAX], [left, right]);
  const y1 = linear([0, imax], [top + h1, top]);
  const top2 = top + h1 + 30;
  const y2 = linear([0, 1.05], [top2 + h2, top2]);
  const c1 = `${st.uid}-c1`;
  const Ts = Array.from({ length: TMAX }, (_, i) => i + 1);
  const parts: string[] = [
    el("defs", {}, el("clipPath", { id: c1 }, el("rect", { x: left, y: top - 2, width: right - left, height: h1 + 4 }))),
    axis({ scale: y1, orient: "left", at: left, span: [left, right], title: L.info, count: 4 }),
    el("line", { x1: left, x2: right, y1: top + h1, y2: top + h1, stroke: C.rule }),
    g({ "clip-path": `url(#${c1})` },
      el("path", { d: linePath(Ts.map((t) => [x(t), y1(t * gain1)])), fill: "none", stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "5 4" }),
      ...r.map((res, k) => el("path", { d: linePath(Ts.map((t, i) => [x(t), y1(res.info[i])])), fill: "none", stroke: colors[k], "stroke-width": 2.2 })),
    ),
    axis({ scale: y2, orient: "left", at: left, span: [left, right], title: narrow ? "" : L.sd, count: 2 }),
    axis({ scale: x, orient: "bottom", at: top2 + h2, span: [top2, top2 + h2], title: L.T, count: 5 }),
    ...r.map((res, k) => el("path", { d: linePath(Ts.map((t, i) => [x(t), y2(res.maxSd[i])])), fill: "none", stroke: colors[k], "stroke-width": 1.8 })),
    el("line", { x1: left, x2: right, y1: y2(0.5), y2: y2(0.5), stroke: C.ink3, "stroke-width": 1, "stroke-dasharray": "3 3" }),
  ];
  if (narrow) parts.push(text(left, top2 - 8, L.sd, { "font-size": TYPE.small, class: "fig-t-muted" }));
  // the "every evaluation new" label, placed where the dashed line leaves the plot or at its end
  const tEnd = Math.min(TMAX, imax / gain1);
  const lx = Math.min(x(tEnd), right) - 4, ly = y1(Math.min(tEnd * gain1, imax)) + (tEnd < TMAX ? 14 : -6);
  parts.push(text(lx, ly, L.indep, { "text-anchor": "end", "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));
  // labels at line ends (wide) or a legend row (narrow)
  if (!narrow) {
    const ends = r.map((res, k) => ({ k, y: y1(res.info[TMAX - 1]) })).sort((a, b) => a.y - b.y);
    let last = -Infinity;
    for (const e of ends) {
      const yy = Math.max(e.y + 4, last + 13);
      last = yy;
      parts.push(text(right + 6, yy, tpl(L.dim, { d: DIMS[e.k] }), { "font-size": TYPE.small, class: "fig-t-strong", style: `fill:${colors[e.k]}` }));
    }
  }
  let y = top2 + h2 + 50;
  const x0 = narrow ? 8 : left;
  if (narrow) {
    DIMS.forEach((d, k) => {
      const cx = x0 + (k % 3) * 110, cy = y + Math.floor(k / 3) * 18;
      parts.push(el("line", { x1: cx, x2: cx + 16, y1: cy - 4, y2: cy - 4, stroke: colors[k], "stroke-width": 2.2 }), text(cx + 22, cy, tpl(L.dim, { d }), { "font-size": TYPE.small, class: "fig-t-muted" }));
    });
    y += 42;
  }
  parts.push(text(x0, y, L.cover, { "font-size": TYPE.small, class: "fig-t-muted" }));
  y += 17;
  const items = r.map((res, k) => tpl(L.coverItem, { d: DIMS[k], t: coverT(res.maxSd) < 0 ? L.never : String(coverT(res.maxSd)) }));
  const perRow = narrow ? 3 : 5, colW = narrow ? 110 : Math.min(120, (w - x0 - 10) / 5);
  items.forEach((s, k) => parts.push(text(x0 + (k % perRow) * colW, y + Math.floor(k / perRow) * 17, s, { "font-size": TYPE.small, class: "fig-t-num fig-t-strong", style: `fill:${colors[k]}` })));
  y += Math.ceil(items.length / perRow) * 17;
  return svg(w, y, describe(st), ...parts);
}

export default defineFigure({
  name: "info-gain-dim",
  title: { en: "Information gained by T evaluations, in 1 to 10 dimensions", zh: "1 至 10 维中 T 次评估获得的信息" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
