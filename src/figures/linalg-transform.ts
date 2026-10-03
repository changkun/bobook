// A 2x2 matrix as a map of the plane. The columns of A are where the two
// basis vectors land; the unit circle lands on an ellipse, the unit square on
// a parallelogram whose area is |det A|, and the faint grid shows how every
// line moves. A probe vector x on the unit circle is drawn with its image
// A x; where x and A x line up, x is an eigenvector. The reader drags the tips
// of the two columns or the probe, or sets the four entries with sliders.

import { defineFigure, type State } from "./types.ts";
import { arrowMarker, el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { clamp, linear, type Scale } from "./lib/scale.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    col1: "Ae₁",
    col2: "Ae₂",
    x: "x",
    ax: "Ax",
    matrix: "A =",
    det: "det A = ad − bc = {d}",
    area: "The unit square's area is multiplied by {a}.",
    flip: "The sign is negative: the map flips orientation.",
    singular: "det A = 0: the plane collapses onto a line.",
    eigReal: "eigenvalues λ_1 = {l1}, λ_2 = {l2}",
    eigRepeated: "one repeated eigenvalue λ = {l1}",
    eigComplex: "no real eigenvalues: every direction turns",
    sym: "symmetric: the eigenvectors are perpendicular",
    angle: "angle between x and Ax: {t}°",
    aligned: "x is (nearly) an eigenvector: Ax = {l} x",
    legendCircle: "unit circle → its image",
    legendSquare: "unit square → parallelogram",
    legendEig: "eigenvector directions",
    describe: "The matrix with rows ({a}, {b}) and ({c}, {d}) maps the unit circle to an ellipse and multiplies areas by {det}. {e}",
  },
  // "Ae₁", "Ax", and "ad − bc" separate their letters with a thin space
  // (U+2009), so that each letter reads as its own symbol.
  zh: {
    col1: "A e₁",
    col2: "A e₂",
    x: "x",
    ax: "A x",
    matrix: "A =",
    det: "det A = a d − b c = {d}",
    area: "单位正方形的面积乘以 {a}。",
    flip: "符号为负：这个映射翻转了方向。",
    singular: "det A = 0：平面被压到一条直线上。",
    eigReal: "特征值 λ_1 = {l1}，λ_2 = {l2}",
    eigRepeated: "一个重特征值 λ = {l1}",
    eigComplex: "没有实特征值：每个方向都被转动",
    sym: "对称：特征向量互相垂直",
    angle: "x 与 A x 的夹角：{t}°",
    aligned: "x（近似）是特征向量：A x = {l} x",
    legendCircle: "单位圆 → 它的像",
    legendSquare: "单位正方形 → 平行四边形",
    legendEig: "特征向量方向",
    describe: "第一行为 ({a}, {b})、第二行为 ({c}, {d}) 的矩阵把单位圆映射为椭圆，并把面积乘以 {det}。{e}",
  },
};

const PRESETS = [
  { value: "custom", label: { en: "Custom", zh: "自定义" } },
  { value: "stretch", label: { en: "Stretch", zh: "拉伸" } },
  { value: "rotation", label: { en: "Rotation", zh: "旋转" } },
  { value: "shear", label: { en: "Shear", zh: "剪切" } },
  { value: "symmetric", label: { en: "Symmetric", zh: "对称" } },
  { value: "singular", label: { en: "Singular", zh: "奇异" } },
  { value: "flip", label: { en: "Reflection", zh: "反射" } },
] as const;

const PRESET_VALUES: Record<string, [number, number, number, number]> = {
  stretch: [1.6, 0, 0, 0.6],
  rotation: [0.85, -0.5, 0.5, 0.85],
  shear: [1, 0.9, 0, 1],
  symmetric: [1.3, 0.6, 0.6, 0.8],
  singular: [1, 0.5, 1.4, 0.7],
  flip: [0.4, 1.1, 1.1, 0.4],
};

const entry = (l: string) => ({ kind: "range", label: { en: l, zh: l }, min: -2, max: 2, default: 0, step: 0.05 }) as const;

const params = {
  preset: { kind: "choice", label: { en: "Matrix", zh: "矩阵" }, options: PRESETS, default: "custom", control: "select" },
  a: { ...entry("a"), default: 1.2 },
  b: { ...entry("b"), default: 0.6 },
  c: { ...entry("c"), default: 0.2 },
  d: { ...entry("d"), default: 0.8 },
  probe: { kind: "range", label: { en: "Angle of x", zh: "x 的角度" }, min: 0, max: 360, default: 125, step: 1, unit: { en: "°", zh: "°" } },
  eig: { kind: "toggle", label: { en: "Show eigenvectors", zh: "显示特征向量" }, default: false },
} as const;

type P = { preset: string; a: number; b: number; c: number; d: number; probe: number; eig: boolean };

const R = 2.4; // the plot shows [-R, R] in both directions

function eigen(p: P) {
  const tr = p.a + p.d, det = p.a * p.d - p.b * p.c;
  const disc = (tr * tr) / 4 - det;
  if (disc < -1e-9) return { real: false as const, det };
  const s = Math.sqrt(Math.max(0, disc));
  const l1 = tr / 2 + s, l2 = tr / 2 - s;
  const vec = (l: number): [number, number] => {
    // (A − λI) v = 0: take a row that is not zero
    let v: [number, number];
    if (Math.abs(p.b) > 1e-9 || Math.abs(p.a - l) > 1e-9) v = [p.b, l - p.a];
    else if (Math.abs(p.c) > 1e-9 || Math.abs(p.d - l) > 1e-9) v = [l - p.d, p.c];
    else v = [1, 0];
    const n = Math.hypot(v[0], v[1]) || 1;
    return [v[0] / n, v[1] / n];
  };
  let v1 = vec(l1), v2 = vec(l2);
  if (Math.abs(v1[0] * v2[1] - v1[1] * v2[0]) < 1e-6) {
    // repeated eigenvalue: a multiple of the identity has every direction as an eigenvector
    if (Math.abs(p.b) < 1e-9 && Math.abs(p.c) < 1e-9) { v1 = [1, 0]; v2 = [0, 1]; }
  }
  return { real: true as const, det, l1, l2, v1, v2, repeated: Math.abs(l1 - l2) < 1e-6 };
}

interface Layout { sx: Scale; sy: Scale; x0: number; y0: number; size: number; panelX: number; panelY: number; narrow: boolean }

function layout(w: number): Layout {
  const narrow = w < 480;
  const size = narrow ? w - 20 : 320;
  const x0 = 10, y0 = 10;
  const sx = linear([-R, R], [x0, x0 + size]);
  const sy = linear([-R, R], [y0 + size, y0]);
  return { sx, sy, x0, y0, size, narrow, panelX: narrow ? 10 : x0 + size + 24, panelY: narrow ? y0 + size + 22 : y0 + 8 };
}

const apply = (p: P, x: number, y: number): [number, number] => [p.a * x + p.b * y, p.c * x + p.d * y];

function describe(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const e = eigen(p);
  const es = !e.real ? L.eigComplex : e.repeated ? tpl(L.eigRepeated, { l1: fixed(e.l1, 2) }) : tpl(L.eigReal, { l1: fixed(e.l1, 2), l2: fixed(e.l2, 2) });
  return tpl(L.describe, { a: fixed(p.a, 2), b: fixed(p.b, 2), c: fixed(p.c, 2), d: fixed(p.d, 2), det: fixed(Math.abs(e.det), 2), e: es.replace(/_/g, "") + (st.lang === "zh" ? "。" : ".") });
}

function arrow(lx: Layout, x: number, y: number, color: string, marker: string, width = 2, dash?: string): string {
  return el("line", { x1: lx.sx(0), y1: lx.sy(0), x2: lx.sx(x), y2: lx.sy(y), stroke: color, "stroke-width": width, "stroke-dasharray": dash, "marker-end": `url(#${marker})` });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const lx = layout(st.w);
  const { sx, sy, x0, y0, size } = lx;
  const e = eigen(p);
  const parts: string[] = [];
  const clipId = `${st.uid}-clip`;
  const mk = (n: string) => `${st.uid}-${n}`;
  parts.push(el("defs", {},
    el("clipPath", { id: clipId }, el("rect", { x: x0, y: y0, width: size, height: size })),
    arrowMarker(mk("a1"), C.c5), arrowMarker(mk("a2"), C.c6), arrowMarker(mk("ax"), C.ink), arrowMarker(mk("e"), C.ink3),
  ));
  // background grid and axes
  const grid: string[] = [];
  for (let k = -2; k <= 2; k++) {
    grid.push(el("line", { x1: sx(k), x2: sx(k), y1: y0, y2: y0 + size, stroke: C.grid }));
    grid.push(el("line", { y1: sy(k), y2: sy(k), x1: x0, x2: x0 + size, stroke: C.grid }));
  }
  grid.push(el("line", { x1: sx(0), x2: sx(0), y1: y0, y2: y0 + size, stroke: C.rule }));
  grid.push(el("line", { y1: sy(0), y2: sy(0), x1: x0, x2: x0 + size, stroke: C.rule }));
  parts.push(el("rect", { x: x0, y: y0, width: size, height: size, fill: "none", stroke: C.rule }), g({}, ...grid));

  const inner: string[] = [];
  // the transformed grid: images of the lines x = k and y = k
  for (let k = -4; k <= 4; k++) {
    const a1 = apply(p, k, -6), a2 = apply(p, k, 6), b1 = apply(p, -6, k), b2 = apply(p, 6, k);
    inner.push(el("line", { x1: sx(a1[0]), y1: sy(a1[1]), x2: sx(a2[0]), y2: sy(a2[1]), stroke: C.model, "stroke-width": 1, opacity: 0.22 }));
    inner.push(el("line", { x1: sx(b1[0]), y1: sy(b1[1]), x2: sx(b2[0]), y2: sy(b2[1]), stroke: C.model, "stroke-width": 1, opacity: 0.22 }));
  }
  // unit square and its image
  const sq: Array<[number, number]> = [[0, 0], [1, 0], [1, 1], [0, 1]];
  inner.push(el("path", { d: linePath(sq.map(([x, y]) => [sx(x), sy(y)])) + "Z", fill: "none", stroke: C.ink3, "stroke-width": 1, "stroke-dasharray": "3 3" }));
  inner.push(el("path", { d: linePath(sq.map(([x, y]) => { const q = apply(p, x, y); return [sx(q[0]), sy(q[1])]; })) + "Z", fill: C.c4, opacity: 0.28, stroke: C.c4, "stroke-width": 1.2 }));
  // unit circle and its image
  const T = Array.from({ length: 121 }, (_, i) => (2 * Math.PI * i) / 120);
  inner.push(el("path", { d: linePath(T.map((t) => [sx(Math.cos(t)), sy(Math.sin(t))])) + "Z", fill: "none", stroke: C.ink3, "stroke-width": 1.2, "stroke-dasharray": "4 3" }));
  inner.push(el("path", { d: linePath(T.map((t) => { const q = apply(p, Math.cos(t), Math.sin(t)); return [sx(q[0]), sy(q[1])]; })) + "Z", fill: C.band, stroke: C.model, "stroke-width": 2 }));
  // eigenvector directions
  if (p.eig && e.real) {
    const vs = e.repeated && Math.abs(p.b) < 1e-9 && Math.abs(p.c) < 1e-9 ? [] : e.repeated ? [e.v1] : [e.v1, e.v2];
    vs.forEach((v) => inner.push(el("line", { x1: sx(-R * 2 * v[0]), y1: sy(-R * 2 * v[1]), x2: sx(R * 2 * v[0]), y2: sy(R * 2 * v[1]), stroke: C.ink2, "stroke-width": 1.3, "stroke-dasharray": "7 4" })));
  }
  parts.push(g({ "clip-path": `url(#${clipId})` }, ...inner));

  // basis vectors, columns, probe
  const c1 = apply(p, 1, 0), c2 = apply(p, 0, 1);
  const th = (p.probe * Math.PI) / 180;
  const px = Math.cos(th), py = Math.sin(th);
  const ax = apply(p, px, py);
  parts.push(
    arrow(lx, 1, 0, C.ink3, mk("e"), 1.2), arrow(lx, 0, 1, C.ink3, mk("e"), 1.2),
    arrow(lx, c1[0], c1[1], C.c5, mk("a1"), 2.4), arrow(lx, c2[0], c2[1], C.c6, mk("a2"), 2.4),
    arrow(lx, ax[0], ax[1], C.ink, mk("ax"), 1.8),
    arrow(lx, px, py, C.ink, mk("ax"), 1.4, "3 2"),
  );
  const tag = (x: number, y: number, s: string, cls = "fig-t-strong") => {
    const n = Math.hypot(x, y) || 1;
    const tx = clamp(sx(x) + (x / n) * 12, x0 + 12, x0 + size - 12), ty = clamp(sy(y) - (y / n) * 12 + 4, y0 + 12, y0 + size - 4);
    return text(tx, ty, s, { "text-anchor": "middle", "font-size": TYPE.small, class: `${cls} fig-t-halo` });
  };
  parts.push(tag(c1[0], c1[1], L.col1), tag(c2[0], c2[1], L.col2), tag(ax[0], ax[1], L.ax), tag(px, py, L.x, "fig-t-muted"));
  // drag handles at the tips
  const handle = (x: number, y: number, id: string, color: string) => el("circle", {
    cx: clamp(sx(x), x0, x0 + size), cy: clamp(sy(y), y0, y0 + size), r: 9, fill: color, opacity: 0.18, stroke: color, "stroke-width": 1, "data-fig-hit": id,
  });
  parts.push(handle(c1[0], c1[1], "col1", C.c5), handle(c2[0], c2[1], "col2", C.c6), handle(px, py, "probe", C.ink));

  // readouts
  const rp: string[] = [];
  let ry = lx.panelY + 14;
  const rx = lx.panelX;
  // the matrix, with its columns colored
  const mw = 46;
  rp.push(text(rx, ry + 12, L.matrix, { "font-size": TYPE.label, class: "fig-t-strong" }));
  const bx = rx + 30;
  rp.push(el("path", { d: `M${bx + 6},${ry - 6}H${bx}V${ry + 30}H${bx + 6}`, fill: "none", stroke: C.ink2, "stroke-width": 1.2 }));
  rp.push(el("path", { d: `M${bx + 2 * mw + 10},${ry - 6}H${bx + 2 * mw + 16}V${ry + 30}H${bx + 2 * mw + 10}`, fill: "none", stroke: C.ink2, "stroke-width": 1.2 }));
  [[p.a, p.b], [p.c, p.d]].forEach((row, i) => row.forEach((v, j) => {
    rp.push(text(bx + 8 + mw * (j + 1) - 8, ry + 8 + i * 18, fixed(v, 2), { "text-anchor": "end", "font-size": TYPE.label, class: "fig-t-num fig-t-strong", fill: j === 0 ? C.c5 : C.c6 }));
  }));
  ry += 50;
  const line = (s: string, cls = "fig-t-muted", size: number = TYPE.small) => { rp.push(text(rx, ry, s, { "font-size": size, class: cls })); ry += size + 7; };
  line(tpl(L.det, { d: fixed(e.det, 2) }), "fig-t-strong fig-t-num", TYPE.body);
  if (Math.abs(e.det) < 0.005) line(L.singular);
  else {
    line(tpl(L.area, { a: fixed(Math.abs(e.det), 2) }));
    if (e.det < 0) line(L.flip);
  }
  ry += 4;
  if (!e.real) line(L.eigComplex, "fig-t-strong", TYPE.body);
  else if (e.repeated) line(tpl(L.eigRepeated, { l1: fixed(e.l1, 2) }), "fig-t-strong fig-t-num", TYPE.body);
  else line(tpl(L.eigReal, { l1: fixed(e.l1, 2), l2: fixed(e.l2, 2) }), "fig-t-strong fig-t-num", TYPE.body);
  if (Math.abs(p.b - p.c) < 1e-9 && e.real && !e.repeated) line(L.sym);
  ry += 4;
  const cross = px * ax[1] - py * ax[0], dotp = px * ax[0] + py * ax[1];
  const ang = (Math.atan2(cross, dotp) * 180) / Math.PI;
  const nAx = Math.hypot(ax[0], ax[1]);
  if (nAx > 1e-6 && (Math.abs(ang) < 1.5 || Math.abs(Math.abs(ang) - 180) < 1.5)) line(tpl(L.aligned, { l: fixed(Math.sign(dotp) * nAx, 2) }), "fig-t-strong fig-t-num");
  else line(tpl(L.angle, { t: nAx > 1e-6 ? Math.round(Math.abs(ang)) : "–" }), "fig-t-num");
  ry += 8;
  // legend
  const sw = (kind: "circle" | "square" | "eig", label: string) => {
    const y = ry - 4;
    if (kind === "circle") rp.push(el("ellipse", { cx: rx + 9, cy: y, rx: 9, ry: 5, fill: C.band, stroke: C.model, "stroke-width": 1.5 }));
    else if (kind === "square") rp.push(el("rect", { x: rx, y: y - 5, width: 18, height: 10, fill: C.c4, opacity: 0.4, stroke: C.c4 }));
    else rp.push(el("line", { x1: rx, x2: rx + 18, y1: y, y2: y, stroke: C.ink2, "stroke-width": 1.3, "stroke-dasharray": "7 4" }));
    rp.push(text(rx + 26, ry, label, { "font-size": TYPE.small, class: "fig-t-muted" }));
    ry += 18;
  };
  sw("circle", L.legendCircle);
  sw("square", L.legendSquare);
  if (p.eig) sw("eig", L.legendEig);
  parts.push(g({}, ...rp));
  const H = Math.max(y0 + size + 10, ry + 4);
  return svg(st.w, H, describe(st), ...parts);
}

const snap = (v: number) => clamp(Math.round(v * 20) / 20, -2, 2);

export default defineFigure({
  name: "linalg-transform",
  title: { en: "A 2×2 matrix as a map of the plane", zh: "把 2×2 矩阵看作平面的映射" },
  labels,
  params,
  hint: { en: "Drag the tips of Ae₁ and Ae₂ to change the columns of A, or drag x around the circle.", zh: "拖动 A e₁ 与 A e₂ 的箭头尖端来改变 A 的列，或沿圆周拖动 x。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  update(p, key) {
    if (key === "preset") {
      const v = PRESET_VALUES[p.preset as string];
      return v ? { ...p, a: v[0], b: v[1], c: v[2], d: v[3] } : p;
    }
    if (key === "a" || key === "b" || key === "c" || key === "d") return { ...p, preset: "custom" };
    return p;
  },
  pointer(p, e) {
    if (e.phase !== "down" && e.phase !== "move") return null;
    if (e.target !== "col1" && e.target !== "col2" && e.target !== "probe") return null;
    const lx = layout(e.w);
    const x = lx.sx.invert(e.x), y = lx.sy.invert(e.y);
    if (e.target === "col1") return { ...p, a: snap(x), c: snap(y), preset: "custom" };
    if (e.target === "col2") return { ...p, b: snap(x), d: snap(y), preset: "custom" };
    const deg = Math.round(((Math.atan2(y, x) * 180) / Math.PI + 360) % 360);
    return { ...p, probe: deg };
  },
});
