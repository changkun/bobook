// What the comparison graph lets the answers determine. The same number of
// comparisons is arranged in four designs: isolated pairs (what EUBO tends to
// produce), a chain, a star, and random pairs. The top panel draws the graph
// on the input line, colored by connected component. The bottom panel shows
// the eigenvalues of the likelihood Hessian W, the graph Laplacian of the
// comparisons (unit curvature per comparison). Each zero eigenvalue is a
// direction of the utility vector that no answer constrains: one of them is
// the global shift, which never matters; every further one is the relative
// offset between two groups of inputs that were never compared, which only
// the prior can fill in.

import { defineFigure, tr, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, CATEGORICAL, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { symEigenvalues, zeros } from "./lib/linalg.ts";
import { memo, rng } from "./lib/random.ts";
import { tpl } from "./lib/format.ts";

const labels = {
  en: {
    inputs: "compared inputs on [0, 1]",
    eig: "eigenvalues of the likelihood Hessian W",
    summary: "{n} comparisons · {m} inputs · {k} component{k:/s}",
    shift: "global shift (harmless)",
    offsets: "{u} unmeasured offset{u:/s} between groups",
    none: "every relative utility is constrained by some answer",
    describe: "{d}: {n} comparisons among {m} inputs form {k} connected component{k:/s}, so the likelihood Hessian has {k} zero eigenvalue{k:/s}: the global shift and {u} offset{u:/s} between groups that no answer constrains.",
  },
  zh: {
    inputs: "[0, 1] 上被比较的输入",
    eig: "似然 Hessian 矩阵 W 的特征值",
    summary: "{n} 次比较 · {m} 个输入 · {k} 个连通分量",
    shift: "全局平移（无害）",
    offsets: "组间有 {u} 个未被测量的偏移",
    none: "每个相对效用都受到某个回答的约束",
    describe: "{d}：{m} 个输入之间的 {n} 次比较构成 {k} 个连通分量，因此似然 Hessian 矩阵有 {k} 个零特征值：全局平移，以及没有任何回答约束的 {u} 个组间偏移。",
  },
};

const DESIGNS = [
  { value: "isolated", label: { en: "Isolated pairs", zh: "孤立对" } },
  { value: "chain", label: { en: "Chain", zh: "链" } },
  { value: "star", label: { en: "Star", zh: "星形" } },
  { value: "random", label: { en: "Random pairs", zh: "随机对" } },
] as const;

const params = {
  design: { kind: "choice", label: { en: "Design", zh: "布局" }, options: DESIGNS, default: "isolated" },
  comparisons: { kind: "range", label: { en: "Comparisons", zh: "比较次数" }, min: 2, max: 10, default: 6, step: 1 },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 4, step: 1, control: false },
} as const;

type P = { design: string; comparisons: number; seed: number };

interface Graph { xs: number[]; edges: Array<[number, number]>; comp: number[]; k: number }

const build = memo((design: string, n: number, seed: number): Graph => {
  let xs: number[] = [];
  const edges: Array<[number, number]> = [];
  if (design === "isolated") {
    xs = grid(0.04, 0.96, 2 * n);
    for (let i = 0; i < n; i++) edges.push([i, i + n]);
  } else if (design === "chain") {
    xs = grid(0.04, 0.96, n + 1);
    for (let i = 0; i < n; i++) edges.push([i, i + 1]);
  } else if (design === "star") {
    xs = grid(0.04, 0.96, n + 1);
    const c = Math.floor(n / 2);
    for (let i = 0; i <= n; i++) if (i !== c) edges.push([c, i]);
  } else {
    const m = Math.min(2 * n, Math.max(4, n + 2));
    xs = grid(0.04, 0.96, m);
    const r = rng(seed);
    let guard = 0;
    while (edges.length < n && guard++ < 1000) {
      const a = Math.floor(r() * m), b = Math.floor(r() * m);
      if (a !== b && !edges.some(([p, q]) => (p === a && q === b) || (p === b && q === a))) edges.push([a, b]);
    }
  }
  const parent = xs.map((_, i) => i);
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  for (const [a, b] of edges) parent[find(a)] = find(b);
  const roots = new Map<number, number>();
  const comp = xs.map((_, i) => { const r0 = find(i); if (!roots.has(r0)) roots.set(r0, roots.size); return roots.get(r0)!; });
  return { xs, edges, comp, k: roots.size };
}, 64);

const spectrum = memo((design: string, n: number, seed: number): number[] => {
  const G = build(design, n, seed);
  const W = zeros(G.xs.length);
  for (const [a, b] of G.edges) { W[a][a] += 1; W[b][b] += 1; W[a][b] -= 1; W[b][a] -= 1; }
  return symEigenvalues(W).map((v) => (Math.abs(v) < 1e-9 ? 0 : v));
}, 64);

function describe(st: State<P>): string {
  const p = st.p;
  const G = build(p.design, p.comparisons, p.seed);
  return tpl(labels[st.lang ?? "en"].describe, { d: tr(DESIGNS.find((d) => d.value === p.design)!.label, st.lang ?? "en"), n: G.edges.length, m: G.xs.length, k: G.k, u: G.k - 1 });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const G = build(p.design, p.comparisons, p.seed);
  const ev = spectrum(p.design, p.comparisons, p.seed);
  const parts: string[] = [];
  const left = narrow ? 36 : 52, right = st.w - 14;
  // graph
  const gy = narrow ? 92 : 108;
  const x = linear([0, 1], [left, right]);
  parts.push(el("line", { x1: left, x2: right, y1: gy, y2: gy, stroke: C.rule }));
  const colorOf = (c: number) => (c < CATEGORICAL.length ? CATEGORICAL[c] : C.ink3);
  for (const [a, b] of G.edges) {
    const xa = x(G.xs[a]), xb = x(G.xs[b]);
    const h = Math.min(gy - 18, Math.abs(xb - xa) * 0.45 + 10);
    parts.push(el("path", { d: `M${xa},${gy} C${xa},${gy - h} ${xb},${gy - h} ${xb},${gy}`, fill: "none", stroke: colorOf(G.comp[a]), "stroke-width": 2 }));
  }
  G.xs.forEach((v, i) => parts.push(el("circle", { cx: x(v), cy: gy, r: 5, fill: colorOf(G.comp[i]), stroke: C.paper, "stroke-width": 1.5 })));
  parts.push(
    text(left, gy + 20, L.inputs, { "font-size": TYPE.small, class: "fig-t-muted" }),
    text(right, gy + 20, tpl(L.summary, { n: G.edges.length, m: G.xs.length, k: G.k }), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-num fig-t-strong" }),
  );
  // spectrum: one bar per eigenvalue, zeros as hollow markers at the baseline
  const top = gy + 52, h = narrow ? 120 : 140;
  const maxEv = Math.max(4, ...ev);
  const y = linear([0, maxEv * 1.05], [top + h, top]);
  const bw = (right - left) / ev.length;
  parts.push(
    axis({ scale: y, orient: "left", at: left, span: [left, right], count: 3 }),
    text(left + 6, top + 10, L.eig, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
  );
  ev.forEach((v, i) => {
    const cx = left + bw * (i + 0.5);
    if (v === 0) {
      const isShift = i === 0;
      parts.push(el("rect", { x: cx - Math.min(bw * 0.35, 9), y: top + h - 6, width: Math.min(bw * 0.7, 18), height: 6, rx: 2, fill: isShift ? C.ink3 : C.bad }));
    } else {
      parts.push(el("rect", { x: cx - Math.min(bw * 0.35, 9), y: y(v), width: Math.min(bw * 0.7, 18), height: top + h - y(v), rx: 2, fill: C.model, opacity: 0.85 }));
    }
  });
  const zeros0 = ev.filter((v) => v === 0).length;
  const ly = top + h + 22;
  parts.push(el("rect", { x: left, y: ly - 9, width: 12, height: 6, rx: 2, fill: C.ink3 }), text(left + 18, ly - 3, L.shift, { "font-size": TYPE.small, class: "fig-t-muted" }));
  const second = narrow ? { x: left, y: ly + 16 } : { x: left + 190, y: ly };
  parts.push(
    zeros0 > 1 ? el("rect", { x: second.x, y: second.y - 9, width: 12, height: 6, rx: 2, fill: C.bad }) : "",
    text(second.x + (zeros0 > 1 ? 18 : 0), second.y - 3, zeros0 > 1 ? tpl(L.offsets, { u: zeros0 - 1 }) : L.none, { "font-size": TYPE.small, class: zeros0 > 1 ? "fig-t-strong" : "fig-t-muted" }),
  );
  return svg(st.w, second.y + 14, describe(st), g({}, ...parts));
}

export default defineFigure({
  name: "comparison-graph",
  title: { en: "The comparison graph and what the answers can determine", zh: "比较图与回答能确定的量" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [{ label: { en: "New random pairs", zh: "换一组随机对" }, run: (p) => ({ ...p, seed: (p.seed % 999) + 1 }), enabled: (p) => p.design === "random" }],
});
