// A 3x3 covariance matrix as a shape. Three standard deviations and three
// correlations set Σ; the figure draws the ellipsoid x^T Σ^{-1} x = 1 in a
// rotatable cube, with its principal axes (the eigenvectors, half-lengths
// √λ_i), its shadows on the three back walls, and beside it the three
// pairwise 2-D views, which are the ellipses of the 2x2 blocks of Σ. When
// the correlations are each between −1 and 1 but cannot hold together, Σ is
// not positive definite: the pairwise views still look fine, but no
// ellipsoid exists, one eigenvalue is negative, and the Cholesky
// factorization fails. Optional samples x = L z show the Cholesky factor
// turning independent normal draws into correlated ones.

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { clamp, linear } from "./lib/scale.ts";
import { cubeEdges, project } from "./lib/nd.ts";
import { eig2 } from "./lib/linalg.ts";
import { choleskyStrict, symEig } from "./lib/linalg-eigen.ts";
import { memo, normals, rng } from "./lib/random.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    sigma: "Σ =",
    eig: "eigenvalues {l1}, {l2}, {l3}",
    det: "det Σ = λ_1 λ_2 λ_3 = {d}",
    chol: "Cholesky diagonal L_11, L_22, L_33 = {a}, {b}, {c}",
    cholFail: "Cholesky fails at column {k}: pivot {v} ≤ 0",
    pd: "positive definite: the ellipsoid exists",
    notPd: "not positive definite: no ellipsoid exists",
    negVar: "a direction with variance {v}",
    pairs: "pairwise views (2×2 blocks of Σ)",
    rho: "ρ = {r}",
    samples: "{n} samples x = L z",
    describe: "A three-dimensional covariance with standard deviations {s1}, {s2}, {s3} and correlations {r12}, {r13}, {r23}. Its eigenvalues are {l1}, {l2}, and {l3}; {status}",
    statusPd: "it is positive definite, so the ellipsoid exists.",
    statusNot: "it is not positive definite, so no Gaussian has these correlations, although each pair on its own is valid.",
  },
  zh: {
    sigma: "Σ =",
    eig: "特征值 {l1}、{l2}、{l3}",
    det: "det Σ = λ_1 λ_2 λ_3 = {d}",
    chol: "Cholesky 对角元 L_11, L_22, L_33 = {a}, {b}, {c}",
    cholFail: "Cholesky 分解在第 {k} 列失败：主元 {v} ≤ 0",
    pd: "正定：椭球存在",
    notPd: "非正定：不存在椭球",
    negVar: "方差为 {v} 的方向",
    pairs: "成对视图（Σ 的 2×2 子块）",
    rho: "ρ = {r}",
    samples: "{n} 个样本 x = L z",
    describe: "一个三维协方差，标准差为 {s1}、{s2}、{s3}，相关系数为 {r12}、{r13}、{r23}。它的特征值为 {l1}、{l2} 和 {l3}；{status}",
    statusPd: "它是正定的，因此椭球存在。",
    statusNot: "它不是正定的，因此没有任何高斯分布具有这些相关系数，尽管每一对单独来看都是有效的。",
  },
};

const sd = (l: string, d: number) => ({ kind: "range", label: { en: l, zh: l }, min: 0.3, max: 1.4, default: d, step: 0.05 }) as const;
const cor = (l: string, d: number) => ({ kind: "range", label: { en: l, zh: l }, min: -0.95, max: 0.95, default: d, step: 0.01 }) as const;

const params = {
  r12: cor("ρ₁₂", 0.7),
  r13: cor("ρ₁₃", 0.4),
  r23: cor("ρ₂₃", 0.2),
  s1: sd("σ₁", 1.2),
  s2: sd("σ₂", 0.9),
  s3: sd("σ₃", 0.7),
  yaw: { kind: "range", label: { en: "Turn", zh: "转动" }, min: -3.14, max: 3.14, default: 0.55, step: 0.01 },
  pitch: { kind: "range", label: { en: "Tilt", zh: "倾斜" }, min: -0.2, max: 1.2, default: 0.38, step: 0.01, control: false },
  samples: { kind: "toggle", label: { en: "Show samples", zh: "显示样本" }, default: false },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 3, step: 1, control: false },
} as const;

type P = { r12: number; r13: number; r23: number; s1: number; s2: number; s3: number; yaw: number; pitch: number; samples: boolean; seed: number };

const R = 1.9; // the cube spans [-R, R] in each coordinate
const NS = 160;
const PAIRS: Array<[number, number]> = [[0, 1], [0, 2], [1, 2]];

function cov(p: P): number[][] {
  const s = [p.s1, p.s2, p.s3];
  const r = [[1, p.r12, p.r13], [p.r12, 1, p.r23], [p.r13, p.r23, 1]];
  return r.map((row, i) => row.map((v, j) => v * s[i] * s[j]));
}

function analyze(p: P) {
  const S = cov(p);
  const e = symEig(S);
  const ch = choleskyStrict(S);
  return { S, e, ch, pd: !!ch.L && e.values[2] > 1e-9 };
}

const draws = memo((seed: number) => {
  const z = normals(rng(seed * 31 + 7), 3 * NS);
  return Array.from({ length: NS }, (_, i) => [z[3 * i], z[3 * i + 1], z[3 * i + 2]]);
}, 8);

function describe(st: State<P>): string {
  const p = st.p;
  const a = analyze(p);
  const L = labels[st.lang ?? "en"];
  return tpl(L.describe, {
    s1: fixed(p.s1, 2), s2: fixed(p.s2, 2), s3: fixed(p.s3, 2), r12: fixed(p.r12, 2), r13: fixed(p.r13, 2), r23: fixed(p.r23, 2),
    l1: fixed(a.e.values[0], 2), l2: fixed(a.e.values[1], 2), l3: fixed(a.e.values[2], 2),
    status: a.pd ? L.statusPd : L.statusNot,
  });
}

interface View { cx: number; cy: number; s: number; yaw: number; pitch: number }

// Data coordinates in [-R, R]^3 to the unit cube that lib/nd.ts projects.
const toUnit = (v: number[]) => v.map((c) => c / (2 * R) + 0.5);
const proj = (vw: View, v: number[]) => project(toUnit(v), vw.yaw, vw.pitch, vw.cx, vw.cy, vw.s);

// The outline of the ellipsoid {x : x^T Σ^{-1} x <= 1} seen from the camera is
// the ellipse of the projected covariance P Σ P^T, P the 2x3 screen map.
function silhouette(vw: View, S: number[][]): string {
  const k = vw.s / (2 * R);
  const cyw = Math.cos(vw.yaw), syw = Math.sin(vw.yaw), cp = Math.cos(vw.pitch), sp = Math.sin(vw.pitch);
  const Px = [k * cyw, k * syw, 0];
  const Pu = [k * sp * syw, -k * sp * cyw, k * cp];
  const q = (a: number[], b: number[]) => a.reduce((s, ai, i) => s + ai * b.reduce((t, bj, j) => t + S[i][j] * bj, 0), 0);
  const m11 = q(Px, Px), m12 = q(Px, Pu), m22 = q(Pu, Pu);
  const e = eig2(m11, m12, m22);
  const pts: Array<[number, number]> = [];
  for (let i = 0; i <= 96; i++) {
    const t = (2 * Math.PI * i) / 96;
    const a = Math.sqrt(Math.max(0, e.values[0])) * Math.cos(t), b = Math.sqrt(Math.max(0, e.values[1])) * Math.sin(t);
    const x = a * e.vectors[0][0] + b * e.vectors[1][0], u = a * e.vectors[0][1] + b * e.vectors[1][1];
    pts.push([vw.cx + x, vw.cy - u]);
  }
  return linePath(pts) + "Z";
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const a = analyze(p);
  const parts: string[] = [];

  // 1. The 3-D view.
  const viewW = narrow ? st.w : 330;
  const cs = narrow ? Math.min(158, st.w * 0.48) : 168;
  const statusY = 16;
  const vw: View = { cx: viewW / 2, cy: statusY + 22 + cs * 0.86, s: cs, yaw: p.yaw, pitch: p.pitch };
  const viewH = statusY + 22 + cs * 1.72 + 10;
  parts.push(text(viewW / 2, statusY, a.pd ? L.pd : L.notPd, { "text-anchor": "middle", "font-size": TYPE.body, class: "fig-t-strong", fill: a.pd ? C.ink : C.bad }));
  parts.push(el("rect", { x: 0, y: statusY + 6, width: viewW, height: viewH - statusY - 6, fill: "transparent", "data-fig-hit": "cube" }));
  parts.push(cubeEdges(p.yaw, p.pitch, vw.cx, vw.cy, vw.s, ["x_1", "x_2", "x_3"]));
  const center = proj(vw, [0, 0, 0]);
  // shadows: for each coordinate, the 2x2 block of the other two drawn on the
  // farther of the two walls perpendicular to it
  const T = Array.from({ length: 73 }, (_, i) => (2 * Math.PI * i) / 72);
  PAIRS.forEach(([i, j]) => {
    const k = 3 - i - j;
    const plus = [0, 0, 0], minus = [0, 0, 0];
    plus[k] = R; minus[k] = -R;
    const wall = proj(vw, plus).z < proj(vw, minus).z ? R : -R;
    const b11 = a.S[i][i], b12 = a.S[i][j], b22 = a.S[j][j];
    const l11 = Math.sqrt(b11), l21 = b12 / l11, l22 = Math.sqrt(Math.max(0, b22 - l21 * l21));
    const pts = T.map((t) => {
      const v = [0, 0, 0];
      v[i] = l11 * Math.cos(t); v[j] = l21 * Math.cos(t) + l22 * Math.sin(t); v[k] = wall;
      const q = proj(vw, v);
      return [q.x, q.y] as [number, number];
    });
    parts.push(el("path", { d: linePath(pts) + "Z", fill: C.ink3, opacity: 0.16, stroke: C.ink3, "stroke-width": 1 }));
  });
  // samples behind the ellipsoid first, then the ellipsoid, then those in front
  const sampleDots = (front: boolean) => {
    if (!p.samples || !a.ch.L) return "";
    const Lm = a.ch.L;
    const out: Array<{ x: number; y: number; z: number }> = [];
    for (const z of draws(p.seed)) {
      const v = [Lm[0][0] * z[0], Lm[1][0] * z[0] + Lm[1][1] * z[1], Lm[2][0] * z[0] + Lm[2][1] * z[1] + Lm[2][2] * z[2]];
      if (v.some((c) => Math.abs(c) > R)) continue;
      const q = proj(vw, v);
      if ((q.z >= center.z) === front) out.push(q);
    }
    out.sort((u, w) => u.z - w.z);
    return g({}, ...out.map((q) => el("circle", { cx: q.x, cy: q.y, r: 2.2, fill: C.model, opacity: front ? 0.75 : 0.4 })));
  };
  parts.push(sampleDots(false));
  // principal axes and principal ellipses
  const axisPts = (v: number[], len: number) => [proj(vw, v.map((c) => -c * len)), proj(vw, v.map((c) => c * len))];
  const wire: string[] = [];
  if (a.pd) {
    parts.push(el("path", { d: silhouette(vw, a.S), fill: C.band, stroke: C.model, "stroke-width": 1.8 }));
    const ax = a.e.values.map((l) => Math.sqrt(l));
    for (const [i, j] of PAIRS) {
      const seg = T.map((t) => {
        const v = [0, 1, 2].map((c) => ax[i] * Math.cos(t) * a.e.vectors[i][c] + ax[j] * Math.sin(t) * a.e.vectors[j][c]);
        return proj(vw, v);
      });
      for (let s = 0; s < seg.length - 1; s++) {
        const back = (seg[s].z + seg[s + 1].z) / 2 < center.z;
        wire.push(el("line", { x1: seg[s].x, y1: seg[s].y, x2: seg[s + 1].x, y2: seg[s + 1].y, stroke: C.model, "stroke-width": back ? 0.8 : 1.1, opacity: back ? 0.35 : 0.7 }));
      }
    }
  }
  parts.push(g({}, ...wire));
  a.e.values.forEach((l, i) => {
    const v = a.e.vectors[i];
    if (l > 1e-9) {
      const [u, w] = axisPts(v, Math.sqrt(l));
      parts.push(el("line", { x1: u.x, y1: u.y, x2: w.x, y2: w.y, stroke: C.ink, "stroke-width": 1.5 }));
      // label the end that sits farther from the center on screen, pushed outward
      const tip = Math.hypot(w.x - center.x, w.y - center.y) >= Math.hypot(u.x - center.x, u.y - center.y) ? w : u;
      const dx = tip.x - center.x, dy = tip.y - center.y, dn = Math.hypot(dx, dy) || 1;
      parts.push(text(tip.x + (dx / dn) * 12, tip.y + (dy / dn) * 12 + 4, `√λ_${i + 1}`, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-strong fig-t-halo" }));
    } else {
      // the direction along which the "variance" would be negative, with a
      // key under the status line rather than a label that could collide
      const [u, w] = axisPts(v, R * 0.95);
      parts.push(el("line", { x1: u.x, y1: u.y, x2: w.x, y2: w.y, stroke: C.bad, "stroke-width": 2.4, "stroke-dasharray": "6 3" }));
      const lbl = tpl(L.negVar, { v: fixed(l, 2) });
      const kx = viewW / 2 - (labelWidth(lbl, 6, 11) + 26) / 2;
      parts.push(
        el("line", { x1: kx, x2: kx + 18, y1: statusY + 17, y2: statusY + 17, stroke: C.bad, "stroke-width": 2.4, "stroke-dasharray": "6 3" }),
        text(kx + 24, statusY + 21, lbl, { "font-size": TYPE.small, class: "fig-t-strong", fill: C.bad }),
      );
    }
  });
  parts.push(sampleDots(true));

  // 2. The three pairwise views.
  const px0 = narrow ? 10 : viewW + 16;
  const pw = narrow ? st.w - 20 : st.w - px0 - 6;
  const gap = 12;
  const ps = Math.min(100, (pw - 2 * gap) / 3);
  let y = narrow ? viewH + 12 : statusY + 2;
  parts.push(text(px0, y, L.pairs, { "font-size": TYPE.small, class: "fig-t-muted" }));
  y += 8;
  const rhos = [p.r12, p.r13, p.r23];
  PAIRS.forEach(([i, j], n) => {
    const x0 = px0 + n * (ps + gap);
    const sx = linear([-R, R], [x0, x0 + ps]), sy = linear([-R, R], [y + ps, y]);
    const b11 = a.S[i][i], b12 = a.S[i][j], b22 = a.S[j][j];
    const l11 = Math.sqrt(b11), l21 = b12 / l11, l22 = Math.sqrt(Math.max(0, b22 - l21 * l21));
    const pts = T.map((t) => [sx(l11 * Math.cos(t)), sy(l21 * Math.cos(t) + l22 * Math.sin(t))] as [number, number]);
    const dots: string[] = [];
    if (p.samples && a.ch.L) {
      const Lm = a.ch.L;
      for (const z of draws(p.seed).slice(0, 80)) {
        const v = [Lm[0][0] * z[0], Lm[1][0] * z[0] + Lm[1][1] * z[1], Lm[2][0] * z[0] + Lm[2][1] * z[1] + Lm[2][2] * z[2]];
        if (Math.abs(v[i]) < R && Math.abs(v[j]) < R) dots.push(el("circle", { cx: sx(v[i]), cy: sy(v[j]), r: 1.5, fill: C.model, opacity: 0.5 }));
      }
    }
    parts.push(
      el("rect", { x: x0, y, width: ps, height: ps, fill: C.panel, stroke: C.rule }),
      el("line", { x1: sx(0), x2: sx(0), y1: y, y2: y + ps, stroke: C.grid }),
      el("line", { y1: sy(0), y2: sy(0), x1: x0, x2: x0 + ps, stroke: C.grid }),
      el("path", { d: linePath(pts) + "Z", fill: C.band, stroke: C.model, "stroke-width": 1.5 }),
      g({}, ...dots),
      text(x0 + ps / 2, y + ps + 14, `x_${i + 1}, x_${j + 1}`, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }),
      text(x0 + ps / 2, y + ps + 29, tpl(L.rho, { r: fixed(rhos[n], 2) }), { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-strong fig-t-num" }),
    );
  });
  y += ps + 50;

  // 3. Readouts: the matrix, its eigenvalues, determinant, and Cholesky factor.
  const rx = px0;
  parts.push(text(rx, y + 20, L.sigma, { "font-size": TYPE.label, class: "fig-t-strong" }));
  const bx = rx + 30, cw = 52;
  parts.push(
    el("path", { d: `M${bx + 6},${y}H${bx}V${y + 52}H${bx + 6}`, fill: "none", stroke: C.ink2, "stroke-width": 1.2 }),
    el("path", { d: `M${bx + 3 * cw + 6},${y}H${bx + 3 * cw + 12}V${y + 52}H${bx + 3 * cw + 6}`, fill: "none", stroke: C.ink2, "stroke-width": 1.2 }),
  );
  a.S.forEach((row, i) => row.forEach((v, j) => {
    parts.push(text(bx + cw * (j + 1) - 4, y + 14 + i * 16, fixed(v, 2), { "text-anchor": "end", "font-size": TYPE.body, class: i === j ? "fig-t-num fig-t-strong" : "fig-t-num" }));
  }));
  y += 74;
  const lines: Array<[string, string, string?]> = [
    [tpl(L.eig, { l1: fixed(a.e.values[0], 2), l2: fixed(a.e.values[1], 2), l3: fixed(a.e.values[2], 2) }), "fig-t-num fig-t-strong"],
    [tpl(L.det, { d: fixed(a.e.values[0] * a.e.values[1] * a.e.values[2], 3) }), "fig-t-num"],
    a.ch.L
      ? [tpl(L.chol, { a: fixed(a.ch.L[0][0], 2), b: fixed(a.ch.L[1][1], 2), c: fixed(a.ch.L[2][2], 2) }), "fig-t-num"]
      : [tpl(L.cholFail, { k: (a.ch.failedAt ?? 0) + 1, v: fixed(a.ch.pivot ?? 0, 2) }), "fig-t-num fig-t-strong", C.bad],
  ];
  if (p.samples && a.ch.L) lines.push([tpl(L.samples, { n: NS }), "fig-t-muted"]);
  for (const [s, cls, fill] of lines) {
    parts.push(text(rx, y, s, { "font-size": TYPE.small, class: cls, fill }));
    y += 18;
  }
  const H = Math.max(viewH, y) + 6;
  return svg(st.w, H, describe(st), ...parts);
}

let drag: { x: number; y: number; yaw: number; pitch: number } | undefined;

export default defineFigure({
  name: "linalg-ellipsoid",
  title: { en: "A 3×3 covariance matrix as an ellipsoid, with its pairwise views", zh: "3×3 协方差矩阵的椭球及其成对视图" },
  labels,
  params,
  hint: { en: "Drag the cube to turn it. Set the three correlations; the pairwise views are always valid, the ellipsoid only sometimes.", zh: "拖动立方体可以转动它。设定三个相关系数：成对视图总是有效的，椭球只是有时存在。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase === "down" && e.target === "cube") { drag = { x: e.x, y: e.y, yaw: p.yaw, pitch: p.pitch }; return null; }
    if (e.phase === "move" && drag) {
      const yaw = clamp(drag.yaw + (e.x - drag.x) * 0.012, -3.14, 3.14);
      const pitch = clamp(drag.pitch + (e.y - drag.y) * 0.01, -0.2, 1.2);
      return { ...p, yaw, pitch };
    }
    if (e.phase === "up") drag = undefined;
    return null;
  },
  actions: [
    { label: { en: "Independent", zh: "独立" }, run: (p) => ({ ...p, r12: 0, r13: 0, r23: 0 }) },
    { label: { en: "A chain", zh: "链式" }, run: (p) => ({ ...p, r12: 0.8, r23: 0.8, r13: 0.64 }) },
    { label: { en: "An impossible triple", zh: "不可能的三元组" }, run: (p) => ({ ...p, r12: 0.8, r13: 0.8, r23: 0, s1: 1, s2: 1, s3: 1 }) },
    { label: { en: "New samples", zh: "新样本" }, run: (p) => ({ ...p, seed: (p.seed % 999) + 1 }), enabled: (p) => p.samples },
  ],
});
