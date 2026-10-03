// Initial designs in the unit square. The reader picks a design (uniform
// random, a regular grid, a Latin hypercube, or a Sobol sequence) and a number
// of points. The square shows the points and joins the closest pair; the
// strips along the bottom and the left show the points projected onto each
// axis, cut into as many equal bins as there are points, with empty bins
// shaded. The projections are what matter when only one input turns out to
// affect the objective: a grid collapses onto a few values, random points
// clump and leave gaps, and the stratified designs put exactly one point in
// every bin.

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { rng } from "./lib/random.ts";
import { axis } from "./lib/axis.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x1: "first input",
    x2: "second input",
    points: "{m} points",
    gridNote: "a {k} × {k} grid: {m} points",
    emptyX: "empty bins, first input: {e} of {m}",
    emptyY: "empty bins, second input: {e} of {m}",
    closest: "closest pair: {d} apart",
    emptyBin: "empty bin",
    projX: "projection onto the first input",
    describe: "A {name} design with {m} points in the unit square. Projected onto the first input, {ex} of {m} equal bins are empty; onto the second, {ey}. The closest two points are {d} apart.",
  },
  zh: {
    x1: "第一个输入",
    x2: "第二个输入",
    points: "{m} 个点",
    gridNote: "{k} × {k} 网格：{m} 个点",
    emptyX: "第一个输入的空格子：{m} 个中有 {e} 个",
    emptyY: "第二个输入的空格子：{m} 个中有 {e} 个",
    closest: "最近点对：相距 {d}",
    emptyBin: "空格子",
    projX: "在第一个输入上的投影",
    describe: "单位正方形中包含 {m} 个点的{name}设计。投影到第一个输入上，{m} 个等宽格子中有 {ex} 个是空的；投影到第二个输入上，有 {ey} 个是空的。最近的两个点相距 {d}。",
  },
};

const DESIGNS = [
  { value: "random", label: { en: "Random", zh: "随机" } },
  { value: "grid", label: { en: "Grid", zh: "网格" } },
  { value: "lhs", label: { en: "Latin hypercube", zh: "拉丁超立方" } },
  { value: "sobol", label: { en: "Sobol", zh: "Sobol" } },
] as const;

const params = {
  design: { kind: "choice", label: { en: "Design", zh: "设计" }, options: DESIGNS, default: "random", control: "buttons" },
  n: { kind: "range", label: { en: "Points", zh: "点数" }, min: 4, max: 64, default: 16, step: 1 },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 5, step: 1, control: false },
} as const;

type P = { design: string; n: number; seed: number };
const NAMES: Record<string, string> = { random: "random", grid: "grid", lhs: "Latin hypercube", sobol: "Sobol" };
// In the Chinese description the name sits between Chinese characters, so a
// Latin name carries its own spaces.
const NAMES_ZH: Record<string, string> = { random: "随机", grid: "网格", lhs: "拉丁超立方", sobol: " Sobol " };
type Pt = [number, number];

// Sobol's sequence in two dimensions. The first coordinate uses direction
// numbers m_k = 1 (the van der Corput sequence in base 2); the second uses the
// primitive polynomial x + 1, whose recurrence is m_k = 2 m_{k-1} xor m_{k-1}
// with m_1 = 1. A random digital shift (xor with a fixed random integer per
// coordinate) gives a different draw for each seed and keeps the
// stratification of the first 2^k points.
const BITS = 30;
const V1: number[] = [];
const V2: number[] = [];
{
  let m = 1;
  for (let k = 1; k <= BITS; k++) {
    if (k > 1) m = (2 * m) ^ m;
    V1.push(1 << (BITS - k));
    V2.push(m << (BITS - k));
  }
}

function sobol(n: number, seed: number): Pt[] {
  const r = rng(seed);
  const s1 = Math.floor(r() * (1 << BITS)), s2 = Math.floor(r() * (1 << BITS));
  const out: Pt[] = [];
  for (let i = 0; i < n; i++) {
    let a = 0, b = 0;
    for (let j = 0, k = i; k; j++, k >>>= 1) if (k & 1) { a ^= V1[j]; b ^= V2[j]; }
    out.push([((a ^ s1) + 0.5) / (1 << BITS), ((b ^ s2) + 0.5) / (1 << BITS)]);
  }
  return out;
}

function permutation(n: number, r: () => number): number[] {
  const p = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; }
  return p;
}

function design(p: P): { pts: Pt[]; k?: number } {
  const r = rng(p.seed * 7 + p.n);
  switch (p.design) {
    case "grid": {
      const k = Math.max(2, Math.round(Math.sqrt(p.n)));
      const pts: Pt[] = [];
      for (let i = 0; i < k; i++) for (let j = 0; j < k; j++) pts.push([(i + 0.5) / k, (j + 0.5) / k]);
      return { pts, k };
    }
    case "lhs": {
      const px = permutation(p.n, r), py = permutation(p.n, r);
      return { pts: px.map((a, i) => [(a + r()) / p.n, (py[i] + r()) / p.n] as Pt) };
    }
    case "sobol":
      return { pts: sobol(p.n, p.seed) };
    default:
      return { pts: Array.from({ length: p.n }, () => [r(), r()] as Pt) };
  }
}

function stats(pts: Pt[]) {
  const m = pts.length;
  const hx = new Array<number>(m).fill(0), hy = new Array<number>(m).fill(0);
  for (const [x, y] of pts) { hx[Math.min(m - 1, Math.floor(x * m))]++; hy[Math.min(m - 1, Math.floor(y * m))]++; }
  let d = Infinity, pair: [number, number] = [0, 1];
  for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) {
    const v = Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
    if (v < d) { d = v; pair = [i, j]; }
  }
  return { m, hx, hy, ex: hx.filter((c) => c === 0).length, ey: hy.filter((c) => c === 0).length, d, pair };
}

function describe(st: State<P>): string {
  const { pts } = design(st.p);
  const s = stats(pts);
  const zh = st.lang === "zh";
  return tpl(labels[st.lang ?? "en"].describe, {
    name: (zh ? NAMES_ZH : NAMES)[st.p.design] ?? (zh ? NAMES_ZH : NAMES).random, m: s.m, ex: s.ex, ey: s.ey, d: fixed(s.d, 3),
  });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const { pts, k } = design(p);
  const s = stats(pts);
  const rug = 16; // the projection strips
  const gap = 6;
  const left = narrow ? 58 : 74;
  const top = 12;
  const size = narrow ? Math.min(st.w - left - 12, 300) : Math.min(300, st.w - left - 260);
  const sx = linear([0, 1], [left, left + size]);
  const sy = linear([0, 1], [top + size, top]);
  const parts: string[] = [];

  // Bin boundaries inside the square, when there are few enough to read.
  if (s.m <= 32) {
    for (let i = 1; i < s.m; i++) {
      const t = i / s.m;
      parts.push(
        el("line", { x1: sx(t), x2: sx(t), y1: sy(0), y2: sy(1), stroke: C.grid, "stroke-width": 1 }),
        el("line", { x1: sx(0), x2: sx(1), y1: sy(t), y2: sy(t), stroke: C.grid, "stroke-width": 1 }),
      );
    }
  }
  parts.push(el("rect", { x: sx(0), y: sy(1), width: size, height: size, fill: "none", stroke: C.rule, "stroke-width": 1 }));

  // Projection strips: bottom for the first input, left for the second.
  const by = top + size + gap;
  const lx = left - gap - rug;
  const bw = size / s.m;
  for (let i = 0; i < s.m; i++) {
    if (s.hx[i] === 0) parts.push(el("rect", { x: sx(0) + i * bw, y: by, width: bw, height: rug, fill: C.warn, opacity: 0.35 }));
    if (s.hy[i] === 0) parts.push(el("rect", { x: lx, y: sy(0) - (i + 1) * bw, width: rug, height: bw, fill: C.warn, opacity: 0.35 }));
  }
  parts.push(
    el("rect", { x: sx(0), y: by, width: size, height: rug, fill: "none", stroke: C.rule }),
    el("rect", { x: lx, y: sy(1), width: rug, height: size, fill: "none", stroke: C.rule }),
    ...pts.map(([x]) => el("line", { x1: sx(x), x2: sx(x), y1: by + 2, y2: by + rug - 2, stroke: C.ink, "stroke-width": 1.4 })),
    ...pts.map(([, y]) => el("line", { x1: lx + 2, x2: lx + rug - 2, y1: sy(y), y2: sy(y), stroke: C.ink, "stroke-width": 1.4 })),
  );

  // Axes outside the strips.
  const ax = linear([0, 1], [left, left + size]);
  const ay = linear([0, 1], [top + size, top]);
  parts.push(
    axis({ scale: ax, orient: "bottom", at: by + rug, title: L.x1, ticks: [0, 0.5, 1], grid: false }),
    axis({ scale: ay, orient: "left", at: lx, title: L.x2, ticks: [0, 0.5, 1], grid: false }),
  );

  // The closest pair, then the points on top.
  const [i, j] = s.pair;
  parts.push(
    el("line", { x1: sx(pts[i][0]), y1: sy(pts[i][1]), x2: sx(pts[j][0]), y2: sy(pts[j][1]), stroke: C.acq, "stroke-width": 2.2 }),
    g({}, ...pts.map(([x, y], q) => el("circle", { cx: sx(x), cy: sy(y), r: s.m > 40 ? 3.2 : 4, fill: q === i || q === j ? C.acq : C.ink, stroke: C.paper, "stroke-width": 1.4 }))),
  );

  // Readouts: to the right when wide, below when narrow.
  const lines = [
    k ? tpl(L.gridNote, { k, m: s.m }) : tpl(L.points, { m: s.m }),
    tpl(L.emptyX, { e: s.ex, m: s.m }),
    tpl(L.emptyY, { e: s.ey, m: s.m }),
    tpl(L.closest, { d: fixed(s.d, 3) }),
  ];
  const rx = narrow ? 12 : left + size + 36;
  const ry = narrow ? by + rug + 56 : top + 26;
  lines.forEach((t, q) => parts.push(text(rx, ry + q * 22, t, { "font-size": TYPE.body, class: q === 0 ? "fig-t-strong fig-t-num" : "fig-t-num" })));
  const ly = ry + lines.length * 22 + 2;
  parts.push(
    el("rect", { x: rx, y: ly - 9, width: 18, height: 12, fill: C.warn, opacity: 0.35, stroke: C.rule }),
    text(rx + 24, ly + 1, L.emptyBin, { "font-size": TYPE.small, class: "fig-t-muted" }),
    el("line", { x1: rx, x2: rx + 18, y1: ly + 19, y2: ly + 19, stroke: C.acq, "stroke-width": 2.2 }),
    text(rx + 24, ly + 23, L.closest.replace(/[:：].*$/, ""), { "font-size": TYPE.small, class: "fig-t-muted" }),
  );
  const H = Math.max(by + rug + 44, ly + 34);
  return svg(st.w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "loop-designs",
  title: { en: "Four ways to choose initial points", zh: "选择初始点的四种方法" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "New draw", zh: "重新抽取" }, primary: true, run: (p) => ({ ...p, seed: (p.seed % 999) + 1 }), enabled: (p) => p.design !== "grid" },
  ],
});
