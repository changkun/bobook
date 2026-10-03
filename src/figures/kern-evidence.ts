// The log marginal likelihood as a surface. For a fixed set of observations
// and an RBF kernel with unit amplitude, every pair (lengthscale, noise sd) is
// one Gaussian process model of the data. Three linked panels:
//   1. the log marginal likelihood over a grid of pairs (log axes), darker is
//      higher, with every local maximum marked;
//   2. the regression fit that the selected pair implies;
//   3. a slice of the surface along the lengthscale at the selected noise,
//      split into its data-fit and complexity terms.
// The reader drags a point on the surface (or moves the two sliders) and
// watches the fit and the two terms change; clicking the fit panel adds or
// removes observations, which reshapes the surface. An optional overlay draws
// each observation's leave-one-out prediction.
//
// The amplitude is fixed at sigma_f = 1, as in Rasmussen and Williams (2006),
// figures 5.4 and 5.5; the preset data are standardized to match.

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { fit, kernel, predict } from "./lib/gp.ts";
import { cholesky, cholSolve, logDetFromChol, spdInverse } from "./lib/linalg.ts";
import { fmtPoints, parsePoints, validPoints } from "./lib/params.ts";
import { memo } from "./lib/random.ts";
import { band, clip, curve, dots, frame, frameAxes, hitArea, legend } from "./lib/plot.ts";
import { fixed, sig, tpl } from "./lib/format.ts";

const labels = {
  en: {
    ell: "lengthscale ℓ",
    noise: "noise sd σ_n",
    surface: "log marginal likelihood",
    x: "input x",
    y: "y",
    mean: "posterior mean",
    band: "95% band for f",
    obs: "observations",
    loo: "leave-one-out 95% intervals",
    fitTerm: "data fit",
    cxTerm: "complexity",
    total: "log marginal likelihood",
    sliceX: "lengthscale ℓ (at the selected noise)",
    best: "best",
    readout: "log p(y) = {t}: data fit {f}, complexity {c}, constant {k}",
    readoutShort1: "log p(y) = {t}",
    readoutShort2: "data fit {f}, complexity {c}, constant {k}",
    gap: "{d} below the best pair (ℓ = {l}, σ_n = {s})",
    atBest: "this is the best pair on the surface",
    looRead: "leave-one-out: {k} of {n} outside their 95% interval; log predictive {v}",
    looShort: "LOO: {k} of {n} outside; log pred. {v}",
    few: "Add at least two observations to compare models.",
    describe: "With {n} observations, lengthscale {l} and noise sd {s} give a log marginal likelihood of {t} (data fit {f}, complexity {c}). The surface has {m} local maxim{m:um/a}; the highest is {b} at lengthscale {bl} and noise sd {bs}.",
  },
  zh: {
    ell: "长度尺度 ℓ",
    noise: "噪声标准差 σ_n",
    surface: "对数边际似然",
    x: "输入 x",
    y: "y",
    mean: "后验均值",
    band: "f 的 95% 区间",
    obs: "观测",
    loo: "留一 95% 区间",
    fitTerm: "数据拟合项",
    cxTerm: "复杂度项",
    total: "对数边际似然",
    sliceX: "长度尺度 ℓ（在所选噪声水平下）",
    best: "最佳",
    readout: "log p(y) = {t}：数据拟合 {f}，复杂度 {c}，常数 {k}",
    readoutShort1: "log p(y) = {t}",
    readoutShort2: "数据拟合 {f}，复杂度 {c}，常数 {k}",
    gap: "比最佳参数对（ℓ = {l}，σ_n = {s}）低 {d}",
    atBest: "这就是曲面上的最佳参数对",
    looRead: "留一检验：{n} 个观测中有 {k} 个落在各自的 95% 区间之外；对数预测密度 {v}",
    looShort: "留一：{n} 个中 {k} 个在区间外；对数预测 {v}",
    few: "至少添加两个观测，才能比较模型。",
    describe: "有 {n} 个观测时，长度尺度 {l} 与噪声标准差 {s} 给出的对数边际似然为 {t}（数据拟合 {f}，复杂度 {c}）。曲面有 {m} 个局部极大值；最高的为 {b}，位于长度尺度 {bl}、噪声标准差 {bs} 处。",
  },
};

const PRESETS = {
  two: "0.05:0.69,0.15:1.42,0.21:0.57,0.28:0.12,0.41:-0.74,0.81:-0.16,0.9:-1.9",
  ten: "0.06:-0.68,0.14:-1.27,0.19:-0.94,0.23:-1.54,0.38:0.12,0.44:1.3,0.51:1.45,0.6:0.69,0.76:0.33,0.83:0.53",
  three: "0.2:0.9,0.5:-0.6,0.8:0.4",
};

const LR: [number, number] = [0.02, 2];
const SR: [number, number] = [0.01, 2];

const params = {
  lengthscale: { kind: "range", label: { en: "Lengthscale ℓ", zh: "长度尺度 ℓ" }, min: LR[0], max: LR[1], default: 0.44, scale: "log" },
  noise: { kind: "range", label: { en: "Noise sd σ_n", zh: "噪声标准差 σ_n" }, min: SR[0], max: SR[1], default: 0.74, scale: "log" },
  loo: { kind: "toggle", label: { en: "Leave-one-out check", zh: "留一检验" }, default: false },
  points: { kind: "data", label: { en: "Observations", zh: "观测" }, default: PRESETS.two, validate: validPoints },
} as const;

type P = { lengthscale: number; noise: number; loo: boolean; points: string };

const G = 44; // surface resolution per axis
const NS = 90; // slice resolution
const XS = grid(0, 1, 121);
const YDOM: [number, number] = [-3, 3];
const lgrid = (r: [number, number], n: number) => grid(Math.log10(r[0]), Math.log10(r[1]), n).map((v) => 10 ** v);
const LG = lgrid(LR, G), SG = lgrid(SR, G), LS = lgrid(LR, NS);

interface Terms { fit: number; cx: number; k: number; total: number }

// The three terms of the log marginal likelihood for an RBF kernel with unit
// amplitude: -y^T K_y^-1 y / 2, -log|K_y| / 2, and -n log(2 pi) / 2.
function terms(xs: number[], ys: number[], ell: number, sn: number): Terms {
  const n = xs.length;
  const K = xs.map((a, i) => xs.map((b, j) => Math.exp(-((a - b) ** 2) / (2 * ell * ell)) + (i === j ? sn * sn : 0)));
  const L = cholesky(K, 1e-12);
  const alpha = cholSolve(L, ys);
  const f = -0.5 * ys.reduce((s, y, i) => s + y * alpha[i], 0);
  const cx = -0.5 * logDetFromChol(L);
  const k = -0.5 * n * Math.log(2 * Math.PI);
  return { fit: f, cx, k, total: f + cx + k };
}

// The surface and its local maxima depend only on the data.
const surface = memo((pts: string) => {
  const P = parsePoints(pts);
  const xs = P.map((q) => q[0]), ys = P.map((q) => q[1]);
  const S = LG.map((l) => SG.map((s) => terms(xs, ys, l, s).total));
  // Local maxima of the grid, each refined by a shrinking pattern search in
  // log10 coordinates and kept inside the plotted ranges.
  const f = (a: number, b: number) => terms(xs, ys, 10 ** a, 10 ** b).total;
  const lo = [Math.log10(LR[0]), Math.log10(SR[0])], hi = [Math.log10(LR[1]), Math.log10(SR[1])];
  const maxima: Array<{ l: number; s: number; v: number; edge: boolean }> = [];
  for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
    let top = true;
    for (let di = -1; di <= 1 && top; di++) for (let dj = -1; dj <= 1; dj++) {
      const a = i + di, b = j + dj;
      if ((di || dj) && a >= 0 && b >= 0 && a < G && b < G && S[a][b] > S[i][j]) { top = false; break; }
    }
    if (!top) continue;
    let p = [Math.log10(LG[i]), Math.log10(SG[j])], v = S[i][j];
    for (let step = 0.04; step > 1e-4; step /= 2) {
      for (let moved = true, it = 0; moved && it < 60; it++) {
        moved = false;
        for (const [da, db] of [[step, 0], [-step, 0], [0, step], [0, -step]]) {
          const q = [Math.min(hi[0], Math.max(lo[0], p[0] + da)), Math.min(hi[1], Math.max(lo[1], p[1] + db))];
          const w = f(q[0], q[1]);
          if (w > v + 1e-12) { p = q; v = w; moved = true; }
        }
      }
    }
    const edge = p[0] <= lo[0] + 1e-6 || p[0] >= hi[0] - 1e-6 || p[1] <= lo[1] + 1e-6 || p[1] >= hi[1] - 1e-6;
    if (!maxima.some((m) => Math.abs(Math.log10(m.l) - p[0]) < 0.05 && Math.abs(Math.log10(m.s) - p[1]) < 0.05)) maxima.push({ l: 10 ** p[0], s: 10 ** p[1], v, edge });
  }
  maxima.sort((a, b) => b.v - a.v);
  // Keep interior maxima, and the best one wherever it is.
  const kept = maxima.filter((m, i) => i === 0 || (!m.edge && maxima[0].v - m.v < 12));
  return { S, maxima: kept, vmax: kept[0].v };
}, 12);

function describe(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const pts = parsePoints(p.points);
  if (pts.length < 2) return L.few;
  const t = terms(pts.map((q) => q[0]), pts.map((q) => q[1]), p.lengthscale, p.noise);
  const s = surface(p.points);
  return tpl(L.describe, {
    n: pts.length, l: sig(p.lengthscale, 2), s: sig(p.noise, 2), t: fixed(t.total, 2), f: fixed(t.fit, 2), c: fixed(t.cx, 2),
    m: s.maxima.length, b: fixed(s.vmax, 2), bl: sig(s.maxima[0].l, 2), bs: sig(s.maxima[0].s, 2),
  });
}

// How far below the maximum a cell is, as an opacity: bands one, two, four,
// ... units of log probability wide, starting at a quarter, so the picture
// reads like a contour map.
function shade(delta: number): number {
  const band = delta < 0.25 ? 0 : delta < 0.5 ? 1 : delta < 1 ? 2 : delta < 2 ? 3 : delta < 4 ? 4 : delta < 8 ? 5 : delta < 16 ? 6 : 7;
  return [0.95, 0.8, 0.66, 0.52, 0.38, 0.25, 0.13, 0.05][band];
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 560;
  const pts = parsePoints(p.points);
  const xs = pts.map((q) => q[0]), ys = pts.map((q) => q[1]);
  const ok = pts.length >= 2;
  const parts: string[] = [];

  // --- layout -------------------------------------------------------------
  const sx0 = narrow ? 48 : 52;
  const cell = Math.max(3, Math.floor((narrow ? Math.min(st.w - sx0 - 12, 264) : 264) / G));
  const size = cell * G;
  const sy0 = 30;
  const fitLeftW = narrow ? st.w : st.w - (sx0 + size + 18);
  const fitX = narrow ? 0 : sx0 + size + 18;
  const lgFit = legend((narrow ? 8 : fitX + 40), narrow ? sy0 + size + 62 : 14, (narrow ? st.w : fitLeftW) - 16, [
    { kind: "line", color: C.model, label: L.mean },
    { kind: "band", color: C.band, label: L.band },
    ...(p.loo ? [{ kind: "line" as const, color: C.c5, label: L.loo }] : []),
  ]);
  const fitTop = (narrow ? sy0 + size + 62 : 14) + lgFit.height + 4;
  const fitH = narrow ? 170 : sy0 + size - fitTop;
  const f0 = frame({ w: fitLeftW, top: fitTop, height: fitH, yDomain: YDOM, yTitle: L.y });
  // shift the fit frame to its column
  const f = { ...f0, left: f0.left + fitX, right: f0.right + fitX, x: linear([0, 1], [f0.left + fitX, f0.right + fitX]) };

  // --- 1. surface ---------------------------------------------------------
  const lx = linear([Math.log10(LR[0]), Math.log10(LR[1])], [sx0, sx0 + size]);
  const ly = linear([Math.log10(SR[0]), Math.log10(SR[1])], [sy0 + size, sy0]);
  if (ok) {
    const s = surface(p.points);
    const cells: string[] = [];
    for (let i = 0; i < G; i++) for (let j = 0; j < G; j++) {
      cells.push(el("rect", { x: sx0 + i * cell, y: sy0 + size - (j + 1) * cell, width: cell, height: cell, fill: C.model, opacity: shade(s.vmax - s.S[i][j]) }));
    }
    parts.push(g({ "shape-rendering": "crispEdges" }, ...cells));
    s.maxima.forEach((m, i) => {
      const cx = lx(Math.log10(m.l)), cy = ly(Math.log10(m.s));
      parts.push(el("path", { d: `M${cx - 5},${cy}h10M${cx},${cy - 5}v10`, stroke: C.paper, "stroke-width": 4, fill: "none" }),
        el("path", { d: `M${cx - 5},${cy}h10M${cx},${cy - 5}v10`, stroke: C.ink, "stroke-width": 1.6, fill: "none" }));
      if (i === 0) parts.push(text(cx + 8, cy - 7, L.best, { "font-size": TYPE.small, class: "fig-t-halo" }));
    });
  }
  const tickFmt = (v: number) => String(Number((10 ** v).toPrecision(1)));
  const lticks = [0.03, 0.1, 0.3, 1].map(Math.log10), sticks = [0.01, 0.03, 0.1, 0.3, 1].map(Math.log10);
  parts.push(
    axis({ scale: lx, orient: "bottom", at: sy0 + size, title: L.ell, ticks: lticks, format: tickFmt, grid: false }),
    axis({ scale: ly, orient: "left", at: sx0, title: L.noise, ticks: sticks, format: tickFmt, grid: false }),
    text(sx0, sy0 - 9, L.surface, { "font-size": TYPE.small, class: "fig-t-muted" }),
    el("rect", { x: sx0, y: sy0, width: size, height: size, fill: "none", stroke: C.rule }),
  );
  // selected pair
  const selX = lx(Math.log10(p.lengthscale)), selY = ly(Math.log10(p.noise));
  parts.push(
    el("line", { x1: selX, x2: selX, y1: sy0, y2: sy0 + size, stroke: C.ink3, "stroke-width": 1, "stroke-dasharray": "2 3" }),
    el("line", { x1: sx0, x2: sx0 + size, y1: selY, y2: selY, stroke: C.ink3, "stroke-width": 1, "stroke-dasharray": "2 3" }),
    el("circle", { cx: selX, cy: selY, r: 7.5, fill: C.paper }),
    el("circle", { cx: selX, cy: selY, r: 5, fill: C.paper, stroke: C.ink, "stroke-width": 2.5 }),
    el("rect", {
      x: sx0, y: sy0, width: size, height: size, fill: "transparent", "data-fig-hit": "surface",
      "data-fig-map": [sx0, sx0 + size, Math.log10(LR[0]), Math.log10(LR[1]), sy0 + size, sy0, Math.log10(SR[0]), Math.log10(SR[1])].join(","),
    }),
  );

  // --- 2. the fit this pair implies --------------------------------------
  const cid = `${st.uid}-clip`;
  parts.push(el("defs", {}, clip(f, cid)), lgFit.svg, frameAxes(f, { yTitle: L.y, xTitle: L.x, yCount: 3 }), hitArea(f, "plot"));
  const gp = fit(kernel("rbf", p.lengthscale, 1), xs, ys, p.noise * p.noise);
  const post = predict(gp, XS);
  const sd = post.var.map(Math.sqrt);
  parts.push(g({ "clip-path": `url(#${cid})` },
    band(f, XS, post.mean.map((m, i) => m - 1.96 * sd[i]), post.mean.map((m, i) => m + 1.96 * sd[i])),
    curve(f, XS, post.mean),
  ));
  let looLine = "";
  if (p.loo && ok) {
    const Ky = xs.map((a, i) => xs.map((b, j) => Math.exp(-((a - b) ** 2) / (2 * p.lengthscale ** 2)) + (i === j ? p.noise ** 2 : 0)));
    const Ki = spdInverse(Ky);
    let out = 0, lp = 0;
    const bars = xs.map((x, i) => {
      const a = Ki[i].reduce((s, v, j) => s + v * ys[j], 0);
      const mu = ys[i] - a / Ki[i][i], s2 = 1 / Ki[i][i], s = Math.sqrt(s2);
      const miss = Math.abs(ys[i] - mu) > 1.96 * s;
      if (miss) out++;
      lp += -0.5 * Math.log(2 * Math.PI * s2) - (ys[i] - mu) ** 2 / (2 * s2);
      const px = f.x(x), cl = (v: number) => f.y(Math.max(YDOM[0], Math.min(YDOM[1], v)));
      return el("line", { x1: px, x2: px, y1: cl(mu - 1.96 * s), y2: cl(mu + 1.96 * s), stroke: miss ? C.bad : C.c5, "stroke-width": miss ? 3 : 2, opacity: 0.85 })
        + el("line", { x1: px - 4, x2: px + 4, y1: cl(mu), y2: cl(mu), stroke: miss ? C.bad : C.c5, "stroke-width": 2 });
    }).join("");
    parts.push(g({ "clip-path": `url(#${cid})` }, bars));
    looLine = tpl(narrow ? L.looShort : L.looRead, { k: out, n: xs.length, v: fixed(lp, 2) });
  }
  parts.push(dots(f, pts, { hit: (i) => `pt-${i}` }));

  // --- 3. the slice: data fit and complexity against lengthscale ----------
  const sliceTop0 = (narrow ? f.bottom : sy0 + size) + 52;
  const lgS = legend(narrow ? 8 : sx0, sliceTop0, st.w - 16, [
    { kind: "line", color: C.model, label: L.total },
    { kind: "dash", color: C.c5, label: L.fitTerm },
    { kind: "dash", color: C.c6, label: L.cxTerm },
  ]);
  const sTop = sliceTop0 + lgS.height + 2, sH = narrow ? 120 : 130;
  const sLeft = sx0, sRight = st.w - 10;
  const slx = linear([Math.log10(LR[0]), Math.log10(LR[1])], [sLeft, sRight]);
  let H = sTop + sH + 44;
  if (ok) {
    const T = LS.map((l) => terms(xs, ys, l, p.noise));
    const here = terms(xs, ys, p.lengthscale, p.noise);
    const s = surface(p.points);
    const hiV = Math.max(4, Math.min(14, Math.max(...T.map((t) => t.cx)) + 2));
    const loV = Math.max(-26, Math.min(...T.map((t) => Math.min(t.fit, t.total))) - 2);
    const sly = linear([loV, hiV], [sTop + sH, sTop]);
    const cidS = `${st.uid}-clipS`;
    const path = (get: (t: Terms) => number) => "M" + T.map((t, i) => `${slx(Math.log10(LS[i])).toFixed(1)},${sly(Math.max(loV - 5, Math.min(hiV + 5, get(t)))).toFixed(1)}`).join("L");
    const hx = slx(Math.log10(p.lengthscale));
    parts.push(
      el("defs", {}, el("clipPath", { id: cidS }, el("rect", { x: sLeft, y: sTop, width: sRight - sLeft, height: sH }))),
      lgS.svg,
      axis({ scale: sly, orient: "left", at: sLeft, span: [sLeft, sRight], count: 4 }),
      axis({ scale: slx, orient: "bottom", at: sTop + sH, title: L.sliceX, ticks: lticks, format: tickFmt, grid: false }),
      el("line", { x1: sLeft, x2: sRight, y1: sly(0), y2: sly(0), stroke: C.rule }),
      g({ "clip-path": `url(#${cidS})` },
        el("path", { d: path((t) => t.fit), fill: "none", stroke: C.c5, "stroke-width": 1.6, "stroke-dasharray": "5 3" }),
        el("path", { d: path((t) => t.cx), fill: "none", stroke: C.c6, "stroke-width": 1.6, "stroke-dasharray": "5 3" }),
        el("path", { d: path((t) => t.total), fill: "none", stroke: C.model, "stroke-width": 2.2 }),
      ),
      el("line", { x1: hx, x2: hx, y1: sTop, y2: sTop + sH, stroke: C.ink3, "stroke-width": 1, "stroke-dasharray": "2 3" }),
      ...[[here.fit, C.c5], [here.cx, C.c6], [here.total, C.model]].map(([v, c]) => el("circle", { cx: hx, cy: sly(Math.max(loV, Math.min(hiV, v as number))), r: c === C.model ? 4.6 : 3.4, fill: c as string, stroke: C.paper, "stroke-width": 1.5 })),
    );
    const readTop = sTop + sH + 50;
    const gap = s.vmax - here.total;
    const lines = [
      ...(narrow
        ? [tpl(L.readoutShort1, { t: fixed(here.total, 2) }), tpl(L.readoutShort2, { f: fixed(here.fit, 2), c: fixed(here.cx, 2), k: fixed(here.k, 2) })]
        : [tpl(L.readout, { t: fixed(here.total, 2), f: fixed(here.fit, 2), c: fixed(here.cx, 2), k: fixed(here.k, 2) })]),
      gap < 0.005 ? L.atBest : tpl(L.gap, { d: fixed(gap, 2), l: sig(s.maxima[0].l, 2), s: sig(s.maxima[0].s, 2) }),
      ...(looLine ? [looLine] : []),
    ];
    lines.forEach((s2, i) => parts.push(text(narrow ? 8 : sLeft, readTop + i * 16, s2, { "font-size": TYPE.small, class: "fig-t-num" })));
    H = readTop + lines.length * 16;
  } else {
    parts.push(text(narrow ? 8 : sLeft, sTop + 20, L.few, { "font-size": TYPE.body, class: "fig-t-muted" }));
    H = sTop + 40;
  }
  return svg(st.w, H, describe(st), g({}, ...parts));
}

const clampTo = (v: number, r: [number, number]) => Number(Math.min(r[1], Math.max(r[0], v)).toPrecision(3));

function best(p: P, which: number): P {
  if (parsePoints(p.points).length < 2) return p;
  const m = surface(p.points).maxima;
  const pick = m[Math.min(which, m.length - 1)];
  return { ...p, lengthscale: clampTo(pick.l, LR), noise: clampTo(pick.s, SR) };
}

export default defineFigure({
  name: "kern-evidence",
  title: { en: "The log marginal likelihood over lengthscale and noise, and the fit each pair implies", zh: "长度尺度与噪声上的对数边际似然，以及每个参数对所对应的拟合" },
  labels,
  params,
  hint: { en: "Drag on the surface to choose a lengthscale and a noise level. Click the fit panel to add an observation; click a point to remove it.", zh: "在曲面上拖动，选择长度尺度与噪声水平。点击拟合面板添加一个观测；点击一个点将其删除。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.target === "surface" && e.data) {
      return { ...p, lengthscale: clampTo(10 ** e.data.x, LR), noise: clampTo(10 ** e.data.y, SR) };
    }
    if (e.phase !== "click") return null;
    const pts = parsePoints(p.points);
    if (e.target?.startsWith("pt-")) {
      pts.splice(Number(e.target.slice(3)), 1);
      return { ...p, points: fmtPoints(pts) };
    }
    if (e.target !== "plot" || !e.data) return null;
    if (pts.length >= 16) pts.shift();
    pts.push([Math.min(1, Math.max(0, e.data.x)), Math.min(YDOM[1], Math.max(YDOM[0], e.data.y))]);
    return { ...p, points: fmtPoints(pts) };
  },
  actions: [
    { label: { en: "Go to the best pair", zh: "前往最佳参数对" }, primary: true, run: (p) => best(p, 0), enabled: (p) => parsePoints(p.points).length >= 2 },
    { label: { en: "Go to the second optimum", zh: "前往第二个极大值" }, run: (p) => best(p, 1), enabled: (p) => parsePoints(p.points).length >= 2 && surface(p.points).maxima.length > 1 },
    { label: { en: "Data: two explanations", zh: "数据：两种解释" }, run: (p) => ({ ...p, points: PRESETS.two }) },
    { label: { en: "Data: ten points", zh: "数据：十个点" }, run: (p) => ({ ...p, points: PRESETS.ten }) },
    { label: { en: "Data: three points", zh: "数据：三个点" }, run: (p) => ({ ...p, points: PRESETS.three }) },
  ],
});
