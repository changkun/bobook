// An inverted U that nobody has. Simulated viewers rate images of increasing
// complexity. In one group liking falls with complexity, in the other it
// rises; every individual is monotone, with their own turning point. The
// average over the population is an inverted U whose peak lies where neither
// group is best served. The reader sets the share of each group and sees what
// a design aimed at the population's peak gives each group, compared with
// what each would choose for itself.

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid } from "./lib/scale.ts";
import { memo, rng, normal } from "./lib/random.ts";
import { sigmoid } from "./lib/stats.ts";
import { clip, curve, frame, frameAxes, legend, vline } from "./lib/plot.ts";
import { fixed, pct, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "image complexity",
    y: "liking",
    falls: "individuals whose liking falls",
    rises: "individuals whose liking rises",
    mean: "population average",
    peak: "population peak",
    readout1: "Population peak at complexity {c}.",
    readout2: "There, the 'falls' group gets {a} of the liking it would get at its own best, and the 'rises' group {b}.",
    describe: "With {p} of viewers whose liking rises with complexity, the population average peaks at complexity {c}, although every simulated individual is monotone. At that peak the falling group gets {a} and the rising group {b} of the liking each would get at its own best.",
    narrow1: "There the 'falls' group gets {a} of its best,",
    narrow2: "and the 'rises' group {b} of its best.",
    na: "n/a",
  },
  zh: {
    x: "图像复杂度",
    y: "喜爱度",
    falls: "喜爱度下降的个体",
    rises: "喜爱度上升的个体",
    mean: "总体平均",
    peak: "总体峰值",
    readout1: "总体峰值位于复杂度 {c}。",
    readout2: "在那里，“下降”组得到的喜爱度是它在自身最佳处的 {a}，“上升”组是 {b}。",
    describe: "喜爱度随复杂度上升的观看者占 {p} 时，总体平均在复杂度 {c} 处达到峰值，尽管每个模拟个体都是单调的。在这一峰值处，下降组得到的喜爱度是它在自身最佳处的 {a}，上升组是 {b}。",
    narrow1: "在那里，“下降”组得到其最佳的 {a}，",
    narrow2: "“上升”组得到其最佳的 {b}。",
    na: "无",
  },
};

const params = {
  share: { kind: "range", label: { en: "Share whose liking rises", zh: "喜爱度上升者的比例" }, min: 0, max: 1, default: 0.5, step: 0.05 },
  individuals: { kind: "toggle", label: { en: "Show individuals", zh: "显示个体" }, default: true },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 3, step: 1, control: false },
} as const;

type P = { share: number; individuals: boolean; seed: number };

const XS = grid(0, 1, 121);
const M = 40; // simulated viewers

interface Viewer { rises: boolean; at: number; width: number }

// Each viewer is a smooth step: down (liking falls past their turning point)
// or up (liking rises past it). Turning points vary between viewers.
const viewers = memo((share: number, seed: number): Viewer[] => {
  const r = rng(seed * 811 + 3);
  const z = normal(r);
  const nUp = Math.round(share * M);
  return Array.from({ length: M }, (_, i) => {
    const rises = i < nUp;
    return { rises, at: Math.min(0.85, Math.max(0.15, (rises ? 0.4 : 0.6) + 0.1 * z())), width: 0.07 + 0.03 * r() };
  });
}, 16);

const liking = (v: Viewer, x: number) => (v.rises ? sigmoid((x - v.at) / v.width) : 1 - sigmoid((x - v.at) / v.width));

function compute(p: P) {
  const vs = viewers(p.share, p.seed);
  const mean = XS.map((x) => vs.reduce((s, v) => s + liking(v, x), 0) / vs.length);
  let bi = 0;
  mean.forEach((m, i) => { if (m > mean[bi] + 1e-12) bi = i; });
  const peak = XS[bi];
  // Each group's average liking at the population peak, relative to the best
  // the group could get from one design aimed at it alone.
  const groupAt = (rises: boolean) => {
    const gs = vs.filter((v) => v.rises === rises);
    if (!gs.length) return NaN;
    const curveG = XS.map((x) => gs.reduce((s, v) => s + liking(v, x), 0) / gs.length);
    const best = Math.max(...curveG);
    return curveG[bi] / best;
  };
  return { vs, mean, peak, a: groupAt(false), b: groupAt(true) };
}

const show = (v: number, na = labels.en.na) => (Number.isFinite(v) ? pct(v) : na);

function describe(st: State<P>): string {
  const c = compute(st.p);
  const L = labels[st.lang ?? "en"];
  return tpl(L.describe, { p: pct(st.p.share), c: fixed(c.peak, 2), a: show(c.a, L.na), b: show(c.b, L.na) });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const c = compute(p);
  const lg = legend(narrow ? 8 : 40, 14, st.w - 16, [
    { kind: "line", color: C.c5, label: L.falls },
    { kind: "line", color: C.c6, label: L.rises },
    { kind: "line", color: C.model, label: L.mean },
  ]);
  const top = 14 + lg.height + 14;
  const f = frame({ w: st.w, top, height: narrow ? 190 : 230, yDomain: [0, 1.05], yTitle: L.y });
  const cid = `${st.uid}-clip`;
  const lines = p.individuals
    ? c.vs.map((v) => curve(f, XS, XS.map((x) => liking(v, x)), { stroke: v.rises ? C.c6 : C.c5, "stroke-width": 1, opacity: 0.35 })).join("")
    : "";
  const ry = f.bottom + 52;
  return svg(st.w, ry + (narrow ? 50 : 24), describe(st),
    el("defs", {}, clip(f, cid)),
    lg.svg,
    frameAxes(f, { xTitle: L.x, yTitle: L.y, yCount: 4 }),
    g({ "clip-path": `url(#${cid})` }, lines, curve(f, XS, c.mean, { stroke: C.model, "stroke-width": 3 })),
    vline(f, c.peak, C.acq, L.peak),
    text(narrow ? 8 : f.left, ry, tpl(L.readout1, { c: fixed(c.peak, 2) }), { "font-size": TYPE.small, class: "fig-t-muted" }),
    ...(narrow
      ? [
        text(8, ry + 16, tpl(L.narrow1, { a: show(c.a, L.na) }), { "font-size": TYPE.small, class: "fig-t-muted" }),
        text(8, ry + 32, tpl(L.narrow2, { b: show(c.b, L.na) }), { "font-size": TYPE.small, class: "fig-t-muted" }),
      ]
      : [text(f.left, ry + 16, tpl(L.readout2, { a: show(c.a, L.na), b: show(c.b, L.na) }), { "font-size": TYPE.small, class: "fig-t-muted" })]),
  );
}

export default defineFigure({
  name: "ds-mixture-u",
  title: { en: "An inverted U that nobody has", zh: "一个谁都没有的倒 U 形" },
  labels,
  params,
  hint: { en: "Move the share of viewers whose liking rises with complexity.", zh: "调节喜爱度随复杂度上升的观看者所占的比例。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
