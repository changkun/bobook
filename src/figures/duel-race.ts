// Three duel-choosing rules race on the five-armed preference matrices of
// duel-winners: uniformly random pairs, RUCB (built for a Condorcet winner),
// and Double Thompson Sampling (built for Copeland winners). The plot shows
// the cumulative Copeland regret, averaged over independent runs, against the
// number of duels. With a Condorcet winner, both learning rules flatten out;
// add a rock-paper-scissors cycle and RUCB, which assumes a Condorcet winner,
// starts to grow linearly, while D-TS keeps flattening.
//
// Colors follow regret-bandits: optimism (UCB, RUCB) orange, Thompson
// sampling green.

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { ARMS, concepts, scenario } from "./lib/duel-matrix.ts";
import { race, type Algo } from "./lib/duel-algos.ts";
import { memo, rng } from "./lib/random.ts";
import { tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "duels t",
    y: "cumulative Copeland regret",
    uniform: "random pairs",
    rucb: "RUCB (simplified)",
    dts: "Double Thompson Sampling",
    uniformShort: "random pairs",
    rucbShort: "RUCB",
    dtsShort: "D-TS",
    winners: "Condorcet winner: {cw} · Copeland winner{cs:/s}: {cp} · mean of {runs} runs",
    describe: "After {T} duels, mean cumulative Copeland regret is {u} for random pairs, {r} for RUCB, and {d} for Double Thompson Sampling. Condorcet winner: {cw}.",
    none: "none",
    sep: ", ",
    colon: ": ",
  },
  zh: {
    x: "对决次数 t",
    y: "累积 Copeland 遗憾",
    uniform: "随机配对",
    rucb: "RUCB（简化版）",
    dts: "双重 Thompson 采样",
    uniformShort: "随机配对",
    rucbShort: "RUCB",
    dtsShort: "D-TS",
    winners: "Condorcet 赢家：{cw} · Copeland 赢家：{cp} · {runs} 次运行的均值",
    describe: "{T} 次对决之后，平均累积 Copeland 遗憾为：随机配对 {u}，RUCB {r}，双重 Thompson 采样 {d}。Condorcet 赢家：{cw}。",
    none: "无",
    sep: "、",
    colon: "：",
  },
};

const params = {
  base: { kind: "choice", label: { en: "Start from", zh: "起点" }, options: [{ value: "utility", label: { en: "A utility", zh: "效用" } }, { value: "narrow", label: { en: "A narrow champion", zh: "险胜的冠军" } }], default: "utility" },
  cycle: { kind: "range", label: { en: "Cycle strength c", zh: "循环强度 c" }, min: 0, max: 3, default: 0, step: 0.1 },
  horizon: { kind: "choice", label: { en: "Duels T", zh: "对决次数 T" }, options: [{ value: 1000, label: { en: "1000", zh: "1000" } }, { value: 2000, label: { en: "2000", zh: "2000" } }, { value: 4000, label: { en: "4000", zh: "4000" } }], default: 2000 },
  runs: { kind: "range", label: { en: "Runs", zh: "运行次数" }, min: 1, max: 30, default: 10, step: 1, control: false },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 3, step: 1, control: false },
} as const;

type P = { base: string; cycle: number; horizon: number; runs: number; seed: number };

const ALGOS: Array<{ id: Algo; color: string; dash?: string }> = [
  { id: "uniform", color: C.ink3, dash: "5 4" },
  { id: "rucb", color: C.c2 },
  { id: "dts", color: C.c6 },
];

const POINTS = 80;

const compute = memo((base: string, cycle: number, T: number, runs: number, seed: number) => {
  const M = scenario(base, cycle, "");
  const k = concepts(M);
  const curves = Object.fromEntries(ALGOS.map(({ id }, a) => [id, race(M, id, T, runs, (run) => rng(seed * 7919 + run * 104729 + a * 13), POINTS)])) as Record<Algo, number[]>;
  return { k, curves };
}, 16);

const fmt = (v: number) => (v >= 100 ? Math.round(v).toString() : v.toFixed(1));

function describe(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const { k, curves } = compute(p.base, p.cycle, p.horizon, p.runs, p.seed);
  return tpl(L.describe, {
    T: p.horizon, u: fmt(curves.uniform[POINTS - 1]), r: fmt(curves.rucb[POINTS - 1]), d: fmt(curves.dts[POINTS - 1]),
    cw: k.condorcet >= 0 ? ARMS[k.condorcet] : L.none,
  });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const { k, curves } = compute(p.base, p.cycle, p.horizon, p.runs, p.seed);
  const left = narrow ? 44 : 58, right = st.w - (narrow ? 12 : 112), top = 16, h = narrow ? 190 : 220;
  const ts = Array.from({ length: POINTS }, (_, i) => Math.round(((i + 1) * p.horizon) / POINTS));
  // The uniform rule grows fastest; clip the y axis so the learning rules stay
  // readable, and say where the uniform line leaves the plot.
  const learnMax = Math.max(...curves.rucb, ...curves.dts);
  const yMax = Math.max(10, learnMax * 1.6);
  const x = linear([0, p.horizon], [left, right]);
  const y = linear([0, yMax], [top + h, top]);
  const parts: string[] = [];
  parts.push(
    axis({ scale: y, orient: "left", at: left, span: [left, right], title: narrow ? "" : L.y, count: 4 }),
    axis({ scale: x, orient: "bottom", at: top + h, span: [top, top + h], title: L.x, count: narrow ? 3 : 4 }),
  );
  if (narrow) parts.push(text(left + 4, top + 10, L.y, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));
  const cid = `${st.uid}-clip`;
  parts.push(el("defs", {}, el("clipPath", { id: cid }, el("rect", { x: left, y: top - 2, width: right - left, height: h + 4 }))));
  const lines: string[] = [];
  const ends: Array<{ id: Algo; yEnd: number; color: string; v: number }> = [];
  for (const a of ALGOS) {
    const ys = curves[a.id];
    const d = "M" + x(0) + "," + y(0) + ys.map((v, i) => `L${x(ts[i]).toFixed(1)},${y(v).toFixed(1)}`).join("");
    lines.push(el("path", { d, fill: "none", stroke: a.color, "stroke-width": a.id === "uniform" ? 1.6 : 2.2, "stroke-dasharray": a.dash }));
    ends.push({ id: a.id, yEnd: y(Math.min(ys[POINTS - 1], yMax)), color: a.color, v: ys[POINTS - 1] });
  }
  parts.push(g({ "clip-path": `url(#${cid})` }, ...lines));
  // end labels (wide) or a legend row (narrow)
  if (!narrow) {
    ends.sort((a, b) => a.yEnd - b.yEnd);
    let last = -Infinity;
    for (const e of ends) {
      const ly = Math.max(e.yEnd + 4, last + 15);
      last = ly;
      parts.push(text(right + 6, ly, `${L[`${e.id}Short` as const]} ${fmt(e.v)}`, { "font-size": TYPE.small, class: "fig-t-num", style: `fill:${e.color}` }));
    }
  }
  let yy = top + h + 52;
  if (narrow) {
    for (const e of ALGOS) {
      const v = curves[e.id][POINTS - 1];
      parts.push(
        el("line", { x1: 8, x2: 26, y1: yy - 4, y2: yy - 4, stroke: e.color, "stroke-width": 2, "stroke-dasharray": e.dash }),
        text(32, yy, `${L[e.id]}${L.colon}${fmt(v)}`, { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }),
      );
      yy += 16;
    }
  }
  const info = tpl(L.winners, {
    cw: k.condorcet >= 0 ? ARMS[k.condorcet] : L.none,
    cs: k.copelandWinners.length, cp: k.copelandWinners.map((i) => ARMS[i]).join(L.sep), runs: p.runs,
  });
  if (narrow) {
    const cut = info.lastIndexOf(" · ");
    parts.push(text(8, yy + 4, info.slice(0, cut), { "font-size": TYPE.small, class: "fig-t-muted" }), text(8, yy + 20, info.slice(cut + 3), { "font-size": TYPE.small, class: "fig-t-muted" }));
    yy += 26;
  } else {
    parts.push(text(left, yy, info, { "font-size": TYPE.small, class: "fig-t-muted" }));
  }
  return svg(st.w, yy + 8, describe(st), ...parts);
}

export default defineFigure({
  name: "duel-race",
  title: { en: "Dueling bandit algorithms with and without a Condorcet winner", zh: "有无 Condorcet 赢家时的对决赌博机算法" },
  labels,
  params,
  hint: { en: "Move the cycle strength c past about 1 to remove the Condorcet winner.", zh: "把循环强度 c 调到约 1 以上，Condorcet 赢家就不复存在。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "New runs", zh: "重新运行" }, run: (p) => ({ ...p, seed: (p.seed % 9999) + 1 }) },
  ],
  update(p, key) {
    if (key === "base") return { ...p, cycle: 0 };
    return p;
  },
});
