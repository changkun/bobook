// How well do the answers tie the utilities together? Twelve compared designs
// sit on a circle; each answered comparison is an edge. Four query designs add
// edges in different orders: disjoint pairs (queries that never share an
// input, as EUBO tends to choose), incumbent versus challenger (a star around
// the current best), a chain (each design against the previous one), and
// random pairs. The left panel shades each design by the standard deviation of
// its utility difference to the incumbent: with no prior this is the square
// root of the effective resistance between the two nodes (unit curvature per
// comparison), infinite when no path of comparisons connects them; with a
// Gaussian process prior over designs in d dimensions it is the posterior sd
// under the Gaussian approximation, so nearby designs are partly tied together
// by the kernel. The right panel traces the algebraic connectivity (the
// second-smallest Laplacian eigenvalue) of the comparison graph as edges are
// added, for all four designs.

import { defineFigure, tr, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { cholesky, solveLower, symEigenvalues, zeros, type Mat } from "./lib/linalg.ts";
import { kernel } from "./lib/gp.ts";
import { memo, rng } from "./lib/random.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    graphTitle: "compared designs and answered pairs",
    specTitle: "algebraic connectivity λ₂ of the comparison graph",
    m: "comparisons answered",
    lam: "λ₂",
    incumbent: "incumbent",
    sdKey: "sd of utility difference to the incumbent",
    free: "not tied",
    status: "{m} comparison{m:/s} · {k} component{k:/s} · λ₂ = {lam}",
    worst: "worst-tied design: sd {sd}",
    worstInf: "{u} design{u:/s} not tied to the incumbent at all",
    worstPrior: "{u} design{u:/s} tied only through the prior",
    describe: "{design}, {m} comparisons among 12 designs: {k} connected component{k:/s}, algebraic connectivity {lam}. {prior} The largest sd of a utility difference to the incumbent is {sd}.",
    priorNone: "With no prior, a design outside the incumbent's component has an undetermined utility difference.",
    priorGP: "With a Gaussian process prior over designs in {d} dimension{d:/s}, kernel correlations tie nearby designs together.",
    unbounded: "unbounded",
  },
  zh: {
    graphTitle: "已比较的设计与已回答的对",
    specTitle: "比较图的代数连通度 λ₂",
    m: "已回答的比较",
    lam: "λ₂",
    incumbent: "当前最优点",
    sdKey: "与当前最优点的效用差的标准差",
    free: "未关联",
    status: "{m} 次比较 · {k} 个连通分量 · λ₂ = {lam}",
    worst: "关联最弱的设计：标准差 {sd}",
    worstInf: "{u} 个设计与当前最优点完全没有关联",
    worstPrior: "{u} 个设计只通过先验关联",
    describe: "{design}，12 个设计之间的 {m} 次比较：{k} 个连通分量，代数连通度 {lam}。{prior}与当前最优点的效用差，最大标准差：{sd}。",
    priorNone: "没有先验时，当前最优点所在连通分量之外的设计，其效用差无法确定。",
    priorGP: "在 {d} 维设计上使用高斯过程先验时，核函数的相关性把相近的设计关联在一起。",
    unbounded: "无穷大",
  },
};

const DESIGNS = [
  { value: "disjoint", label: { en: "Disjoint pairs", zh: "不相交的对" } },
  { value: "star", label: { en: "Incumbent vs challenger", zh: "当前最优点对挑战者" } },
  { value: "chain", label: { en: "Chain", zh: "链" } },
  { value: "random", label: { en: "Random pairs", zh: "随机对" } },
] as const;

const PRIORS = [
  { value: 0, label: { en: "None", zh: "无" } },
  { value: 1, label: { en: "GP, d = 1", zh: "高斯过程，d = 1" } },
  { value: 3, label: { en: "GP, d = 3", zh: "高斯过程，d = 3" } },
  { value: 10, label: { en: "GP, d = 10", zh: "高斯过程，d = 10" } },
  { value: 30, label: { en: "GP, d = 30", zh: "高斯过程，d = 30" } },
] as const;

const params = {
  design: { kind: "choice", label: { en: "Query design", zh: "查询设计" }, options: DESIGNS, default: "disjoint", control: "select" },
  m: { kind: "range", label: { en: "Comparisons", zh: "比较次数" }, min: 1, max: 24, default: 11, step: 1 },
  prior: { kind: "choice", label: { en: "Prior over designs", zh: "设计上的先验" }, options: PRIORS, default: 0, control: "select" },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 5, step: 1, control: false },
} as const;

type P = { design: string; m: number; prior: number; seed: number };

const N = 12;
const MAXM = 24;
const LS = 0.5; // RBF lengthscale on the unit cube, near the 0.52 mode of BoTorch's Gamma(2.4, 2.7) prior
const SD_MAX = 1.6; // sd at which the shading reaches its palest

// The order in which each design adds comparisons (pairs of node indices).
const sequence = memo((design: string, seed: number): Array<[number, number]> => {
  const out: Array<[number, number]> = [];
  if (design === "random") {
    const pairs: Array<[number, number]> = [];
    for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++) pairs.push([i, j]);
    const r = rng(seed);
    for (let i = pairs.length - 1; i > 0; i--) { const k = Math.floor(r() * (i + 1)); [pairs[i], pairs[k]] = [pairs[k], pairs[i]]; }
    return pairs.slice(0, MAXM);
  }
  for (let t = 0; out.length < MAXM; t++) {
    if (design === "star") out.push([0, (t % (N - 1)) + 1]);
    else if (design === "chain") { const k = t % (N - 1); out.push([k, k + 1]); }
    else { const k = t % (N / 2); out.push([2 * k, 2 * k + 1]); }
  }
  return out;
}, 16);

function laplacian(edges: Array<[number, number]>): Mat {
  const L = zeros(N);
  for (const [a, b] of edges) { L[a][a]++; L[b][b]++; L[a][b]--; L[b][a]--; }
  return L;
}

function components(edges: Array<[number, number]>): number[] {
  const parent = Array.from({ length: N }, (_, i) => i);
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  for (const [a, b] of edges) parent[find(a)] = find(b);
  return Array.from({ length: N }, (_, i) => find(i));
}

const lambda2 = (edges: Array<[number, number]>) => Math.max(0, symEigenvalues(laplacian(edges))[1]);

// Every design's lambda_2 curve over m = 0..MAXM.
const curves = memo((seed: number) => {
  const out: Record<string, number[]> = {};
  for (const d of DESIGNS) {
    const s = sequence(d.value, seed);
    out[d.value] = Array.from({ length: MAXM + 1 }, (_, m) => (m === 0 ? 0 : lambda2(s.slice(0, m))));
  }
  return out;
}, 8);

// Designs as points in [0, 1]^D for the Gaussian process prior.
const points = memo((d: number, seed: number) => {
  const r = rng(seed * 31 + d);
  return Array.from({ length: N }, () => Array.from({ length: d }, () => r()));
}, 16);

// sd of f(x_j) - f(x_0) for every j. Without a prior: sqrt of the effective
// resistance in the incumbent's component (Infinity elsewhere). With a GP
// prior K and comparison incidence B (W = B B^T, unit curvature):
// Sigma = K - K B (I + B^T K B)^{-1} B^T K, which avoids inverting K.
const anchoring = memo((design: string, m: number, prior: number, seed: number): number[] => {
  const edges = sequence(design, seed).slice(0, m);
  if (!prior) {
    const comp = components(edges);
    const idx = Array.from({ length: N }, (_, i) => i).filter((i) => i !== 0 && comp[i] === comp[0]);
    const sd = new Array<number>(N).fill(Infinity);
    sd[0] = 0;
    if (!idx.length) return sd;
    const L = laplacian(edges);
    const red = idx.map((i) => idx.map((j) => L[i][j]));
    const Lc = cholesky(red);
    idx.forEach((node, a) => {
      const e = new Array<number>(idx.length).fill(0);
      e[a] = 1;
      const y = solveLower(Lc, e);
      sd[node] = Math.sqrt(y.reduce((s, v) => s + v * v, 0));
    });
    return sd;
  }
  const xs = points(prior, seed);
  const k = kernel("rbf", LS, 1);
  const K = xs.map((a) => xs.map((b) => k(a, b)));
  const ne = edges.length;
  // KB[i][e] = (K B)_{i,e} = K[i][a] - K[i][b] for edge e = (a, b)
  const KB = K.map((row) => edges.map(([a, b]) => row[a] - row[b]));
  const M = zeros(ne);
  for (let p = 0; p < ne; p++) for (let q = 0; q < ne; q++) {
    const [a] = edges[p], b = edges[p][1];
    M[p][q] = (p === q ? 1 : 0) + KB[a][q] - KB[b][q];
  }
  const Lm = cholesky(M);
  return Array.from({ length: N }, (_, j) => {
    if (j === 0) return 0;
    const prior2 = K[j][j] + K[0][0] - 2 * K[0][j];
    const v = edges.map((_, e) => KB[j][e] - KB[0][e]); // B^T K d_j
    const y = solveLower(Lm, v);
    return Math.sqrt(Math.max(0, prior2 - y.reduce((s, t) => s + t * t, 0)));
  });
}, 64);

function compute(p: P) {
  const edges = sequence(p.design, p.seed).slice(0, p.m);
  const comp = components(edges);
  const k = new Set(comp).size;
  const lam = curves(p.seed)[p.design][p.m];
  const sd = anchoring(p.design, p.m, p.prior, p.seed);
  const untied = comp.filter((c) => c !== comp[0]).length;
  const finite = sd.filter(Number.isFinite);
  return { edges, comp, k, lam, sd, untied, worst: Math.max(...finite) };
}

function describe(st: State<P>): string {
  const p = st.p;
  const c = compute(p);
  const lang = st.lang ?? "en";
  const L = labels[lang];
  return tpl(L.describe, {
    design: tr(DESIGNS.find((d) => d.value === p.design)!.label, lang), m: p.m, k: c.k, lam: fixed(c.lam, 2),
    prior: p.prior ? tpl(L.priorGP, { d: p.prior }) : L.priorNone,
    sd: c.untied && !p.prior ? L.unbounded : fixed(c.worst, 2),
  });
}

function nodeFill(sd: number): number {
  return Math.max(0.1, Math.min(1, 1 - sd / SD_MAX));
}

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 480;
  const c = compute(p);
  const parts: string[] = [];

  // Left (or top): the comparison graph.
  const gw = narrow ? st.w : 290;
  const cx = gw / 2, cy = 36 + (narrow ? 112 : 118);
  const R = narrow ? 92 : 100;
  parts.push(text(narrow ? 8 : 12, 16, L.graphTitle, { "font-size": TYPE.label, class: "fig-t-strong" }));
  const pos = Array.from({ length: N }, (_, i) => {
    const a = -Math.PI / 2 + (2 * Math.PI * i) / N;
    return [cx + R * Math.cos(a), cy + R * Math.sin(a)] as [number, number];
  });
  // Edge multiplicities: repeated comparisons draw thicker.
  const mult = new Map<string, number>();
  for (const [a, b] of c.edges) { const key = `${Math.min(a, b)}-${Math.max(a, b)}`; mult.set(key, (mult.get(key) ?? 0) + 1); }
  for (const [key, w] of mult) {
    const [a, b] = key.split("-").map(Number);
    parts.push(el("line", { x1: pos[a][0], y1: pos[a][1], x2: pos[b][0], y2: pos[b][1], stroke: C.ink2, "stroke-width": 1.2 + 1.4 * (w - 1), "stroke-linecap": "round", opacity: 0.8 }));
  }
  const last = c.edges[c.edges.length - 1];
  if (last) parts.push(el("line", { x1: pos[last[0]][0], y1: pos[last[0]][1], x2: pos[last[1]][0], y2: pos[last[1]][1], stroke: C.acq, "stroke-width": 2.6, "stroke-linecap": "round" }));
  pos.forEach(([x, y], i) => {
    const sd = c.sd[i];
    if (i === 0) {
      parts.push(el("circle", { cx: x, cy: y, r: 11, fill: C.model, stroke: C.paper, "stroke-width": 2 }));
      parts.push(text(x, y - 16, L.incumbent, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));
      return;
    }
    if (!Number.isFinite(sd)) {
      parts.push(el("circle", { cx: x, cy: y, r: 10, fill: C.paper, stroke: C.ink3, "stroke-width": 1.5, "stroke-dasharray": "3 2" }));
    } else {
      parts.push(el("circle", { cx: x, cy: y, r: 10, fill: C.paper, stroke: "none" }));
      parts.push(el("circle", { cx: x, cy: y, r: 10, fill: C.model, opacity: nodeFill(sd), stroke: C.model, "stroke-width": 1.2 }));
    }
    if (!narrow) {
      const a = -Math.PI / 2 + (2 * Math.PI * i) / N;
      const lx = cx + (R + 22) * Math.cos(a), ly = cy + (R + 22) * Math.sin(a) + 4;
      parts.push(text(lx, ly, Number.isFinite(sd) ? fixed(sd, 2) : "∞", { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
    }
  });
  // Shading key.
  const ky = cy + R + (narrow ? 26 : 40);
  const kx = narrow ? 8 : 24;
  const kw = narrow ? 120 : 130;
  parts.push(text(kx, ky, L.sdKey, { "font-size": TYPE.small, class: "fig-t-muted" }));
  const steps = 8;
  for (let i = 0; i < steps; i++) {
    const sd = (SD_MAX * i) / (steps - 1);
    parts.push(el("rect", { x: kx + (kw / steps) * i, y: ky + 8, width: kw / steps + 0.5, height: 10, fill: C.model, opacity: nodeFill(sd) }));
  }
  parts.push(text(kx, ky + 31, "0", { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
  parts.push(text(kx + kw, ky + 31, fixed(SD_MAX, 1), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-num fig-t-muted" }));
  parts.push(el("circle", { cx: kx + kw + 22, cy: ky + 13, r: 6, fill: C.paper, stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "3 2" }));
  parts.push(text(kx + kw + 32, ky + 17, L.free, { "font-size": TYPE.small, class: "fig-t-muted" }));
  const graphBottom = ky + 40;

  // Right (or bottom): lambda_2 against m for every design.
  const sx0 = narrow ? 44 : gw + 46;
  const sTop = narrow ? graphBottom + 34 : 40;
  const sH = narrow ? 150 : 210;
  const sx1 = st.w - 12;
  parts.push(text(narrow ? 8 : gw + 12, sTop - 22, L.specTitle, { "font-size": TYPE.label, class: "fig-t-strong" }));
  const x = linear([0, MAXM], [sx0, sx1]);
  const y = linear([0, 2.1], [sTop + sH, sTop]);
  parts.push(axis({ scale: y, orient: "left", at: sx0, span: [sx0, sx1], count: 4, title: L.lam }));
  parts.push(axis({ scale: x, orient: "bottom", at: sTop + sH, span: [sTop, sTop + sH], count: narrow ? 4 : 6, title: L.m }));
  const cv = curves(p.seed);
  const colors: Record<string, string> = { disjoint: C.c8, star: C.c1, chain: C.c6, random: C.c7 };
  const order = (DESIGNS.map((d) => d.value) as string[]).filter((d) => d !== p.design).concat([p.design]);
  for (const d of order) {
    const ys = cv[d];
    const on = d === p.design;
    // A step curve: lambda_2 changes only when an edge arrives.
    let path = `M${x(0)},${y(ys[0])}`;
    for (let m = 1; m <= MAXM; m++) path += `H${x(m)}V${y(ys[m])}`;
    parts.push(el("path", { d: path, fill: "none", stroke: colors[d], "stroke-width": on ? 2.6 : 1.4, opacity: on ? 1 : 0.55, "stroke-linejoin": "round" }));
  }
  parts.push(el("circle", { cx: x(p.m), cy: y(cv[p.design][p.m]), r: 5, fill: colors[p.design], stroke: C.paper, "stroke-width": 1.6 }));
  // Legend for the curves.
  let ly = sTop + sH + 50;
  let lx = sx0 - (narrow ? 36 : 30);
  for (const d of DESIGNS) {
    const label = tr(d.label, lang);
    const w = labelWidth(label) + 30;
    if (lx + w > st.w - 4) { lx = sx0 - (narrow ? 36 : 30); ly += 18; }
    parts.push(el("line", { x1: lx, x2: lx + 18, y1: ly - 4, y2: ly - 4, stroke: colors[d.value], "stroke-width": d.value === p.design ? 2.6 : 1.6 }));
    parts.push(text(lx + 24, ly, label, { "font-size": TYPE.small, class: d.value === p.design ? "fig-t-strong" : "fig-t-muted" }));
    lx += w;
  }
  // Status lines.
  const s1 = tpl(L.status, { m: p.m, k: c.k, lam: fixed(c.lam, 2) });
  const s2 = c.untied && !p.prior ? tpl(L.worstInf, { u: c.untied }) : tpl(L.worst, { sd: fixed(c.worst, 2) });
  const sy = Math.max(ly + 26, narrow ? 0 : graphBottom + 6);
  parts.push(text(narrow ? 8 : 12, sy, s1, { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
  parts.push(text(narrow ? 8 : 12, sy + 16, s2, { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
  return svg(st.w, sy + 26, describe(st), ...parts);
}

export default defineFigure({
  name: "nat-anchoring",
  title: { en: "How comparisons tie utilities together", zh: "比较如何把效用关联在一起" },
  labels,
  params,
  hint: { en: "Choose a query design and drag the number of comparisons. Each node shows how uncertain its utility is relative to the incumbent.", zh: "选择一种查询设计，拖动比较次数。每个节点显示其效用相对当前最优点有多不确定。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "Add a comparison", zh: "增加一次比较" }, primary: true, run: (p) => ({ ...p, m: Math.min(MAXM, Number(p.m) + 1) }), enabled: (p) => p.m < MAXM },
    { label: { en: "Remove one", zh: "减少一次" }, run: (p) => ({ ...p, m: Math.max(1, Number(p.m) - 1) }), enabled: (p) => p.m > 1 },
  ],
});
