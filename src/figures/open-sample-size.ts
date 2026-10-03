// How many people the decisive experiment needs: a planning approximation.
// Each participant is retested on k pairs (final design against an early-
// rejected design) at the end of the session and again a week later; their
// drop is the share of retest choices favoring the final design at the end
// minus the share a week later. The primary contrast is the mean drop in the
// acquisition arm minus the mean drop in the randomized arm.
//
// With agreement around p at both retests, a person's measured drop has
// variance  s^2 = sb^2 + 2 p (1 - p) / k  (true differences between people,
// plus binomial noise of the two retests), and a two-sided z test at
// alpha = 0.05 with n people per arm has power
//   Phi( delta / sqrt(2 s^2 / n) - 1.96 ).
// The figure plots that power against n and marks the n that reaches 80%.

import { defineFigure, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear, log as logScale } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { Phi } from "./lib/stats.ts";
import { tpl } from "./lib/format.ts";

const labels = {
  en: {
    y: "power",
    x: "participants per arm",
    target: "80% power",
    need: "{n} per arm",
    needTotal: "{n} per arm · {t} in three arms",
    breakdown: "spread of one person's measured drop: SD {s} = retest noise {b} and differences between people {w}",
    breakdownShort: "SD of one person's drop: {s}",
    over: "> 3000 per arm",
    describe: "A planning approximation, not data. To detect a difference of {d} percentage points in the drop of retest agreement between the acquisition and randomized arms, with {k} retest pairs per person and agreement near {p}%, a two-sided test at the 5% level reaches 80% power with about {n} participants per arm, {t} in three arms.",
  },
  zh: {
    y: "检验效能",
    x: "每组被试数",
    target: "80% 检验效能",
    need: "每组 {n} 人",
    needTotal: "每组 {n} 人 · 三组共 {t} 人",
    breakdown: "单人测得下降幅度的离散：标准差 {s}，由重测噪声 {b} 与人际差异 {w} 合成",
    breakdownShort: "单人下降幅度的标准差：{s}",
    over: "每组超过 3,000 人",
    describe: "这是规划用的近似，不是数据。要检测采集组与随机组之间重测一致率下降幅度相差 {d} 个百分点，在每人 {k} 个重测对、一致率约 {p}% 的条件下，5% 水平的双侧检验要达到 80% 检验效能，每组约需 {n} 名被试，三组共 {t} 名。",
  },
};

const params = {
  diff: { kind: "range", label: { en: "Difference in drop", zh: "下降幅度之差" }, min: 0.02, max: 0.25, default: 0.07, step: 0.01 },
  pairs: { kind: "range", label: { en: "Retest pairs", zh: "重测对数" }, min: 2, max: 24, default: 8, step: 1 },
  agree: { kind: "range", label: { en: "Agreement", zh: "一致率" }, min: 0.55, max: 0.95, default: 0.8, step: 0.05 },
  spread: { kind: "range", label: { en: "Between people", zh: "人际差异" }, min: 0, max: 0.3, default: 0.1, step: 0.01 },
} as const;

type P = { diff: number; pairs: number; agree: number; spread: number };

const Z_ALPHA = 1.959964;
const Z_POWER = 0.841621;

function sd(p: P) {
  const noise = Math.sqrt((2 * p.agree * (1 - p.agree)) / p.pairs);
  return { noise, total: Math.sqrt(p.spread * p.spread + noise * noise) };
}

const power = (p: P, n: number) => Phi(p.diff / Math.sqrt((2 * sd(p).total ** 2) / n) - Z_ALPHA);
const needed = (p: P) => Math.ceil((2 * (Z_ALPHA + Z_POWER) ** 2 * sd(p).total ** 2) / (p.diff * p.diff));

function describe(st: State<P>): string {
  const p = st.p, n = needed(p);
  return tpl(labels[st.lang ?? "en"].describe, { d: Math.round(p.diff * 100), k: p.pairs, p: Math.round(p.agree * 100), n, t: 3 * n });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const left = narrow ? 46 : 58, right = st.w - 16;
  const top = 26, h = narrow ? 170 : 210;
  const x = logScale([10, 3000], [left, right]);
  const y = linear([0, 1], [top + h, top]);
  const parts: string[] = [];
  parts.push(axis({ scale: y, orient: "left", at: left, span: [left, right], title: L.y, count: 5, format: (v) => v.toFixed(1) }));
  parts.push(axis({ scale: x, orient: "bottom", at: top + h, span: [top, top + h], title: L.x, ticks: [10, 30, 100, 300, 1000, 3000], format: (v) => String(v) }));
  parts.push(el("line", { x1: left, x2: right, y1: y(0.8), y2: y(0.8), stroke: C.rule, "stroke-dasharray": "4 3" }));
  parts.push(text(left + 6, y(0.8) - 5, L.target, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));
  const ns = Array.from({ length: 120 }, (_, i) => 10 * Math.pow(300, i / 119));
  parts.push(el("path", { d: linePath(ns.map((n) => [x(n), y(power(p, n))])), fill: "none", stroke: C.model, "stroke-width": 2.2 }));
  const n80 = needed(p);
  if (n80 <= 3000) {
    const cx = x(Math.max(10, n80));
    parts.push(
      el("line", { x1: cx, x2: cx, y1: y(0.8), y2: top + h, stroke: C.acq, "stroke-width": 1.5, "stroke-dasharray": "4 3" }),
      el("circle", { cx, cy: y(0.8), r: 4.5, fill: C.acq, stroke: C.paper, "stroke-width": 1.6 }),
    );
    const lab = narrow ? tpl(L.need, { n: n80 }) : tpl(L.needTotal, { n: n80, t: 3 * n80 });
    const anchorEnd = cx > (left + right) / 2;
    parts.push(text(cx + (anchorEnd ? -8 : 8), y(0.8) + 18, lab, { "font-size": TYPE.label, "text-anchor": anchorEnd ? "end" : "start", class: "fig-t-strong fig-t-halo fig-t-num" }));
  } else {
    parts.push(text(right - 4, y(0.8) + 18, L.over, { "font-size": TYPE.label, "text-anchor": "end", class: "fig-t-strong fig-t-halo" }));
  }
  const s = sd(p);
  const by = top + h + 52;
  parts.push(text(left, by, narrow
    ? tpl(L.breakdownShort, { s: s.total.toFixed(2) })
    : tpl(L.breakdown, { s: s.total.toFixed(2), b: s.noise.toFixed(2), w: p.spread.toFixed(2) }), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  return svg(st.w, by + 10, describe(st), g({}, ...parts));
}

export default defineFigure({
  name: "open-sample-size",
  title: { en: "How many participants the decisive experiment needs (a planning approximation)", zh: "判定实验需要多少被试（规划用近似）" },
  labels,
  params,
  hint: { en: "A normal approximation for planning, not a result. Set the effect you expect and the number of retest pairs per person.", zh: "这是用于规划的正态近似，不是结果。设定你预期的效应，以及每人的重测对数。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
