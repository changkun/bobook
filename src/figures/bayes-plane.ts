// Bayesian linear regression with two inputs: a plane f(x) = w0 + w1 x1 +
// w2 x2 over the square [-1, 1]^2. The reader evaluates a hidden plane (with
// a little noise) by clicking the top-down map. The 3-D view shows the
// observations as dots on stems, the posterior mean plane, and planes drawn
// from the posterior; the map shades the posterior standard deviation of f.
// When every input lies on one line, the drawn planes still swing about that
// line, because the data say nothing about the tilt across it.

import { defineFigure, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { clamp, grid, linear } from "./lib/scale.ts";
import { eig2 } from "./lib/linalg.ts";
import { normals, rng } from "./lib/random.ts";
import { blrPosterior, quad, sampleWeights } from "./lib/bayes-linear.ts";
import { cubeEdges, project } from "./lib/nd.ts";
import { fmtPoints, parsePoints, validPoints } from "./lib/params.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    axes: "x_1,x_2,f",
    mapTitle: "inputs (x_1, x_2); more color = less certain f",
    n: "{n} evaluation{n:/s} of a hidden plane, noise sd 0.1",
    span0: "no data: every tilt is as uncertain as the prior",
    span1: "inputs span 1 of 2 directions: the plane can tilt about them",
    span2: "inputs span both directions: the plane is pinned down",
    tilt: "least certain slope: sd {s} along ({a}, {b})",
    sdRange: "sd of f: {lo} to {hi} over the square",
    describe: "A plane fitted to {n} evaluation{n:/s}. {span} The posterior standard deviation of f ranges from {lo} to {hi} over the square.",
  },
  zh: {
    axes: "x_1,x_2,f",
    mapTitle: "输入 (x_1, x_2)；颜色越深，f 越不确定",
    n: "对隐藏平面的 {n} 次评估，噪声标准差 0.1",
    span0: "没有数据：每个倾斜方向都和先验一样不确定",
    span1: "输入只张成 2 个方向中的 1 个：平面可以绕它们倾斜",
    span2: "输入张成了两个方向：平面被确定下来",
    tilt: "最不确定的斜率：沿 ({a}, {b}) 方向，标准差 {s}",
    sdRange: "正方形上 f 的标准差：{lo} 至 {hi}",
    describe: "对 {n} 次评估拟合的平面。{span}在整个正方形上，f 的后验标准差在 {lo} 至 {hi} 之间。",
  },
};

const params = {
  prior: { kind: "range", label: { en: "Prior sd of weights σₚ", zh: "权重的先验标准差 σₚ" }, min: 0.2, max: 3, default: 1, scale: "log" },
  points: { kind: "data", label: { en: "Evaluated inputs", zh: "已评估的输入" }, default: "-0.7:-0.7,0:0,0.7:0.7", validate: validPoints },
  yaw: { kind: "range", label: { en: "View angle", zh: "视角" }, min: -180, max: 180, default: -35, step: 1, control: false },
  grabYaw: { kind: "range", label: { en: "Drag start angle", zh: "拖动起始角度" }, min: -180, max: 180, default: 0, step: 0.1, control: false },
  grabX: { kind: "range", label: { en: "Drag start x", zh: "拖动起始 x" }, min: -2000, max: 4000, default: 0, step: 0.1, control: false },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 9, step: 1, control: false },
} as const;

type P = { prior: number; points: string; yaw: number; grabYaw: number; grabX: number; seed: number };

const NOISE = 0.1;
const YR = 3; // the value axis spans [-YR, YR]
const truth = (x1: number, x2: number) => 0.2 + 0.9 * x1 - 0.6 * x2;
const Z = normals(rng(4242), 64);
const G = grid(-1, 1, 21);
const PITCH = 0.42;

function compute(p: P) {
  const xs = parsePoints(p.points);
  const ys = xs.map(([a, b], i) => truth(a, b) + NOISE * Z[i % Z.length]);
  const post = blrPosterior(xs.map(([a, b]) => [1, a, b]), ys, p.prior * p.prior, NOISE * NOISE, 3);
  const sd = G.map((b) => G.map((a) => Math.sqrt(quad(post.cov, [1, a, b]))));
  const flat = sd.flat();
  // directions spanned by the inputs: rank of their centered scatter matrix
  let span = 0;
  if (xs.length >= 2) {
    const m1 = xs.reduce((s, q) => s + q[0], 0) / xs.length, m2 = xs.reduce((s, q) => s + q[1], 0) / xs.length;
    let a = 0, b = 0, d = 0;
    for (const [u, v] of xs) { a += (u - m1) ** 2; b += (u - m1) * (v - m2); d += (v - m2) ** 2; }
    const ev = eig2(a, b, d).values;
    span = (ev[0] > 1e-3 ? 1 : 0) + (ev[1] > 1e-3 ? 1 : 0);
  }
  const slope = eig2(post.cov[1][1], post.cov[1][2], post.cov[2][2]);
  return { xs, ys, post, sd, lo: Math.min(...flat), hi: Math.max(...flat), span, slope };
}

function spanText(n: number, span: number, L: typeof labels.en): string {
  return n === 0 ? L.span0 : span < 2 ? L.span1 : L.span2;
}

function describe(st: State<P>): string {
  const c = compute(st.p);
  const L = labels[st.lang ?? "en"];
  const t = spanText(c.xs.length, c.span, L);
  return tpl(L.describe, { n: c.xs.length, span: t.charAt(0).toUpperCase() + t.slice(1) + (st.lang === "zh" ? "。" : "."), lo: fixed(c.lo, 2), hi: fixed(c.hi, 2) });
}

function render(st: State<P>): string {
  const p = st.p, L = labels[st.lang ?? "en"], w = st.w;
  const narrow = w < 480;
  const c = compute(p);
  const yaw = (p.yaw * Math.PI) / 180;
  // 3-D panel
  const pw = narrow ? w : Math.round(w * 0.6);
  const ph = narrow ? 240 : 290;
  // the projected cube spans about 1.5 s vertically at this pitch; leave room
  // for the axis labels above and below
  const s = Math.min(pw * 0.5, (ph - 36) / 1.5);
  const cx = pw / 2, cy = 14 + ph / 2 + 4;
  const U = (x1: number, x2: number, y: number) => project([(x1 + 1) / 2, (x2 + 1) / 2, (y + YR) / (2 * YR)], yaw, PITCH, cx, cy, s);
  const parts: string[] = [];
  const cid = `${st.uid}-c3`;
  parts.push(el("defs", {}, el("clipPath", { id: cid }, el("rect", { x: 0, y: 8, width: pw, height: ph + 12 }))));
  const inner: string[] = [];
  // floor grid
  for (const t of grid(-1, 1, 5)) {
    const a = U(t, -1, -YR), b = U(t, 1, -YR), c2 = U(-1, t, -YR), d = U(1, t, -YR);
    inner.push(el("line", { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: C.grid }), el("line", { x1: c2.x, y1: c2.y, x2: d.x, y2: d.y, stroke: C.grid }));
  }
  const axisLabels = L.axes.split(",");
  inner.push(cubeEdges(yaw, PITCH, cx, cy, s, axisLabels));
  const quadPath = (wv: number[]) => {
    const pts = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([a, b]) => U(a, b, wv[0] + wv[1] * a + wv[2] * b));
    return linePath(pts.map((q) => [q.x, q.y])) + "Z";
  };
  const ws = sampleWeights(c.post, rng(p.seed * 31 + c.xs.length), 5);
  for (const wv of ws) inner.push(el("path", { d: quadPath(wv), fill: C.c7, "fill-opacity": 0.08, stroke: C.c7, "stroke-width": 1, "stroke-opacity": 0.75 }));
  inner.push(el("path", { d: quadPath(c.post.mean), fill: C.model, "fill-opacity": 0.16, stroke: C.model, "stroke-width": 1.8 }));
  // observations: stems to the floor, then dots, far ones first
  const obs = c.xs.map(([a, b], i) => ({ top: U(a, b, c.ys[i]), foot: U(a, b, -YR) })).sort((u, v) => v.top.z - u.top.z);
  for (const o of obs) {
    inner.push(
      el("line", { x1: o.foot.x, y1: o.foot.y, x2: o.top.x, y2: o.top.y, stroke: C.ink3, "stroke-width": 1, "stroke-dasharray": "3 2" }),
      el("circle", { cx: o.foot.x, cy: o.foot.y, r: 2, fill: C.ink3 }),
      el("circle", { cx: o.top.x, cy: o.top.y, r: 4.2, fill: C.ink, stroke: C.paper, "stroke-width": 1.6 }),
    );
  }
  parts.push(g({ "clip-path": `url(#${cid})` }, ...inner));
  // drag surface for rotating the view
  parts.push(el("rect", { x: 0, y: 8, width: pw, height: ph + 12, fill: "transparent", "data-fig-hit": "cube" }));

  // top-down map of the posterior sd
  const M = narrow ? Math.min(190, w - 60) : Math.min(190, w - pw - 40);
  const mx0 = narrow ? 34 : pw + 24;
  const my0 = narrow ? 14 + ph + 40 : 40;
  const sx = linear([-1, 1], [mx0, mx0 + M]);
  const sy = linear([-1, 1], [my0 + M, my0]);
  const cell = M / (G.length - 1);
  const ref = Math.sqrt(3) * p.prior; // prior sd of f at a corner
  parts.push(text(mx0, my0 - 10, L.mapTitle, { "font-size": TYPE.small, class: "fig-t-muted" }));
  for (let j = 0; j < G.length - 1; j++) for (let i = 0; i < G.length - 1; i++) {
    const v = (c.sd[j][i] + c.sd[j][i + 1] + c.sd[j + 1][i] + c.sd[j + 1][i + 1]) / 4;
    parts.push(el("rect", { x: sx(G[i]), y: sy(G[j + 1]), width: cell + 0.5, height: cell + 0.5, fill: C.model, "fill-opacity": 0.04 + 0.8 * Math.min(1, v / ref) }));
  }
  parts.push(el("rect", { x: mx0, y: my0, width: M, height: M, fill: "none", stroke: C.rule }));
  for (const [t, lab] of [[-1, "−1"], [0, "0"], [1, "1"]] as const) {
    parts.push(text(sx(t), my0 + M + 14, lab, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
    parts.push(text(mx0 - 6, sy(t) + 4, lab, { "text-anchor": "end", "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  }
  const map = [sx.range[0], sx.range[1], sx.domain[0], sx.domain[1], sy.range[0], sy.range[1], sy.domain[0], sy.domain[1]].join(",");
  parts.push(el("rect", { x: mx0, y: my0, width: M, height: M, fill: "transparent", "data-fig-hit": "map", "data-fig-map": map }));
  c.xs.forEach(([a, b], i) => parts.push(el("circle", { cx: sx(a), cy: sy(b), r: 4.5, fill: C.ink, stroke: C.paper, "stroke-width": 1.6, "data-fig-hit": `pt-${i}` })));

  // readout
  let y = my0 + M + 36;
  const x0 = narrow ? 8 : mx0 - 4;
  const line = (s2: string, cls = "fig-t-muted") => { parts.push(text(x0, y, s2, { "font-size": TYPE.small, class: `${cls} fig-t-num` })); y += 16; };
  line(tpl(L.n, { n: c.xs.length }), "fig-t-strong");
  // the span sentence is long: split it after the colon
  const st2 = spanText(c.xs.length, c.span, L), cut = st2.search(/[:：]/);
  if (cut > 0) { line(st2.slice(0, cut + 1)); line("  " + st2.slice(cut + (st2[cut] === "：" ? 1 : 2))); } else line(st2);
  const v = c.slope.vectors[0];
  line(tpl(L.tilt, { s: fixed(Math.sqrt(c.slope.values[0]), 2), a: fixed(v[0], 2), b: fixed(v[1], 2) }));
  line(tpl(L.sdRange, { lo: fixed(c.lo, 2), hi: fixed(c.hi, 2) }));
  const H = Math.max(14 + ph + 20, y);
  return svg(w, H, describe(st), ...parts);
}

const rotate = (p: P, d: number): P => {
  let a = p.yaw + d;
  while (a > 180) a -= 360;
  while (a < -180) a += 360;
  return { ...p, yaw: a };
};

export default defineFigure({
  name: "bayes-plane",
  title: { en: "Bayesian linear regression with two inputs: a posterior over planes", zh: "两个输入的贝叶斯线性回归：平面的后验分布" },
  labels,
  params,
  hint: { en: "Click the map to evaluate the hidden plane there; click a dot to remove it. Drag the 3-D view to turn it.", zh: "点击俯视图，在该处评估隐藏的平面；点击一个点将其删除。拖动三维视图可以旋转它。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    const q = p as P;
    if (e.target === "cube") {
      if (e.phase === "down") return { ...q, grabYaw: q.yaw, grabX: clamp(e.x, -2000, 4000) };
      if (e.phase === "move") return rotate({ ...q, yaw: q.grabYaw }, (e.x - q.grabX) * 0.6);
      return null;
    }
    if (e.phase !== "click") return null;
    const xs = parsePoints(q.points);
    if (e.target?.startsWith("pt-")) {
      xs.splice(Number(e.target.slice(3)), 1);
      return { ...q, points: fmtPoints(xs) };
    }
    if (e.target !== "map" || !e.data) return null;
    if (xs.length >= 30) xs.shift();
    xs.push([clamp(e.data.x, -1, 1), clamp(e.data.y, -1, 1)]);
    return { ...q, points: fmtPoints(xs) };
  },
  actions: [
    {
      label: { en: "Evaluate a random input", zh: "评估一个随机输入" }, primary: true,
      run: (p) => {
        const xs = parsePoints(p.points);
        const r = rng(p.seed * 7 + xs.length * 13 + 1);
        xs.push([Math.round((r() * 2 - 1) * 100) / 100, Math.round((r() * 2 - 1) * 100) / 100]);
        return { ...p, points: fmtPoints(xs.slice(-30)) };
      },
    },
    { label: { en: "◀ Turn", zh: "◀ 旋转" }, run: (p) => rotate(p as P, -20) },
    { label: { en: "Turn ▶", zh: "旋转 ▶" }, run: (p) => rotate(p as P, 20) },
    { label: { en: "Clear", zh: "清空" }, run: (p) => ({ ...p, points: "" }), enabled: (p) => p.points !== "" },
  ],
});
