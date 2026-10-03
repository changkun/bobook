// What faded and what held: effect sizes on one scale, with a panel that
// shows what a given Cohen's d means for two groups of people. Every row is a
// number stated in the chapters with its source:
//
// set "judgment" (perspectives/01-judgment-and-psychophysics):
//   Many Labs 2, 28 findings: median original d = 0.60, median replication
//     d = 0.15 [klein2018many]
//   Ego depletion, 23 labs, registered (N = 2,141): d = 0.04
//     [hagger2016multilab]
//   Ego depletion, 36 labs, preregistered (N = 3,531): d = 0.06
//     [vohs2021multisite]
//   Ego depletion, 12 labs (N = 1,775): d = 0.10, or 0.16 after exclusions
//     [dang2021multilab]
//   Free choice, four studies addressing the artifact: d = 0.26
//     [izuma2013choice]
//   Free choice, 43 artifact-free studies (N = 2,191): d = 0.40, 95% CI
//     [0.32, 0.49] [enisman2021choice]
//
// set "social" (perspectives/02-social-psychology):
//   Moral reminders and cheating: original d = 0.48; primary analysis of 25
//     direct replications d = -0.04 [verschuere2018registered]
//   Moral licensing, 115 experiments: uncorrected g = 0.21; bias-corrected g
//     between -0.08 and -0.02 [rotella2026observation]
//   Licensing when observed g = 0.65; when not observed g = 0.13
//     [rotella2026observation]
//   Morning morality, meta-analysis (2024 preprint): d = 0.04
//     [zickfeld2024investigating]
//   Choice-induced change, 43 artifact-free studies: d = 0.40, 95% CI
//     [0.32, 0.49] [enisman2021choice]
//
// The lower panel is a property of the normal model, not data: two normal
// distributions with equal standard deviation whose means differ by d
// overlap by 2 Φ(-|d|/2) (the overlapping coefficient), and a draw from the
// one with the higher mean exceeds a draw from the other with probability
// Φ(|d|/√2) (the probability of superiority).

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { Phi, normalPdf } from "./lib/stats.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    judgment: "Effect sizes from judgment and decision research",
    social: "Effect sizes from social and moral psychology",
    orig: "original or uncorrected",
    later: "later test or corrected",
    interval: "reported interval",
    small: "small",
    medium: "medium",
    large: "large",
    axis: "effect size d (or g)",
    first: "group 1",
    second: "group 2",
    overlap: "overlap {o}",
    readout: "d = {d}: the distributions overlap {o}; the higher group outscores the lower {p} of the time.",
    describe: "{set}, {n} rows. At d = {d}, two normal distributions of equal spread overlap {o}, and a person drawn from the group with the higher mean outscores one drawn from the other {p} of the time.",
  },
  zh: {
    judgment: "判断与决策研究中的效应量",
    social: "社会心理学与道德心理学中的效应量",
    orig: "原始估计或未校正估计",
    later: "后续检验或校正后估计",
    interval: "报告的区间",
    small: "小",
    medium: "中",
    large: "大",
    axis: "效应量 d（或 g）",
    first: "第 1 组",
    second: "第 2 组",
    overlap: "重叠 {o}",
    readout: "d = {d}：两个分布重叠 {o}；均值较高的一组在 {p} 的情况下胜过较低的一组。",
    describe: "{set}，共 {n} 行。d = {d} 时，两个离散程度相同的正态分布重叠 {o}，从均值较高的一组中抽取的人在 {p} 的情况下得分高于从另一组中抽取的人。",
  },
};

interface Row { label: string; zh: string; orig?: number; later?: number; lo?: number; hi?: number }

const ROWS: Record<string, Row[]> = {
  judgment: [
    { label: "Many Labs 2, 28 findings (medians)", zh: "Many Labs 2，28 项发现（中位数）", orig: 0.6, later: 0.15 },
    { label: "Ego depletion, 23 labs", zh: "自我损耗，23 个实验室", later: 0.04 },
    { label: "Ego depletion, 36 labs", zh: "自我损耗，36 个实验室", later: 0.06 },
    { label: "Ego depletion, 12 labs (0.16 after exclusions)", zh: "自我损耗，12 个实验室（排除后为 0.16）", later: 0.1 },
    { label: "Free choice, 4 studies addressing the artifact", zh: "自由选择，4 项针对伪影的研究", later: 0.26 },
    { label: "Free choice, 43 artifact-free studies", zh: "自由选择，43 项无伪影研究", later: 0.4, lo: 0.32, hi: 0.49 },
  ],
  social: [
    { label: "Moral reminders and cheating, 25 replications", zh: "道德提醒与作弊，25 项复现", orig: 0.48, later: -0.04 },
    { label: "Moral licensing, 115 experiments", zh: "道德许可，115 项实验", orig: 0.21, lo: -0.08, hi: -0.02 },
    { label: "Licensing when observed", zh: "被观察时的道德许可", later: 0.65 },
    { label: "Licensing when not observed", zh: "未被观察时的道德许可", later: 0.13 },
    { label: "Morning morality (preprint)", zh: "早晨道德效应（预印本）", later: 0.04 },
    { label: "Choice-induced change, 43 studies", zh: "选择引起的改变，43 项研究", later: 0.4, lo: 0.32, hi: 0.49 },
  ],
};

const params = {
  set: { kind: "choice", label: { en: "Effects", zh: "效应" }, options: [{ value: "judgment", label: { en: "Judgment", zh: "判断" } }, { value: "social", label: { en: "Social", zh: "社会" } }], default: "judgment", control: false },
  d: { kind: "range", label: { en: "Effect size d", zh: "效应量 d" }, min: -0.2, max: 0.9, default: 0.06, step: 0.01 },
} as const;

type P = { set: "judgment" | "social"; d: number };

const D_DOM: [number, number] = [-0.2, 0.9];
const overlap = (d: number) => 2 * Phi(-Math.abs(d) / 2);
const superiority = (d: number) => Phi(Math.abs(d) / Math.SQRT2);
const pc = (v: number) => `${Math.round(v * 100)}%`;

function describe(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  return tpl(L.describe, { set: L[p.set], n: ROWS[p.set].length, d: fixed(p.d, 2), o: pc(overlap(p.d)), p: pc(superiority(p.d)) });
}

function layout(w: number) {
  const narrow = w < 480;
  const labelW = narrow ? 0 : 236;
  const left = narrow ? 14 : labelW + 12, right = w - (narrow ? 14 : 18);
  const rowH = narrow ? 38 : 26;
  return { narrow, labelW, left, right, rowH };
}

function render(st: State<P>): string {
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const p = st.p, w = st.w;
  const { narrow, left, right, rowH } = layout(w);
  const rows = ROWS[p.set];
  const parts: string[] = [];
  // legend
  let lx = narrow ? 8 : left, ly = 14;
  const items: Array<[string, string]> = [["orig", L.orig], ["later", L.later], ["int", L.interval]];
  for (const [k, lab] of items) {
    const tw = labelWidth(lab, 6.1) + 26;
    if (lx + tw > w - 8 && lx > 8) { lx = narrow ? 8 : left; ly += 18; }
    const sw = k === "orig"
      ? el("circle", { cx: lx + 6, cy: ly - 1, r: 4.2, fill: C.paper, stroke: C.ink2, "stroke-width": 1.6 })
      : k === "later"
        ? el("circle", { cx: lx + 6, cy: ly - 1, r: 4.2, fill: C.c1 })
        : el("line", { x1: lx, x2: lx + 14, y1: ly - 1, y2: ly - 1, stroke: C.c1, "stroke-width": 2.2 });
    parts.push(sw, text(lx + 18, ly + 3, lab, { "font-size": TYPE.small, class: "fig-t-muted" }));
    lx += tw;
  }
  const top = ly + 30;
  const x = linear(D_DOM, [left, right]);
  const bottom = top + rows.length * rowH;
  const ry = linear([0, rows.length], [top, bottom]);
  // Cohen's conventions and zero
  for (const [v, lab] of [[0.2, L.small], [0.5, L.medium], [0.8, L.large]] as const) {
    parts.push(el("line", { x1: x(v), x2: x(v), y1: top - 6, y2: bottom, stroke: C.grid, "stroke-width": 1.2, "stroke-dasharray": "3 3" }));
    parts.push(text(x(v), top - 10, lab, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-faint" }));
  }
  parts.push(el("line", { x1: x(0), x2: x(0), y1: top - 6, y2: bottom, stroke: C.rule, "stroke-width": 1.2 }));
  // rows
  rows.forEach((r, i) => {
    const y0 = ry(i);
    const cy = narrow ? y0 + 27 : y0 + rowH / 2;
    if (i % 2 === 0) parts.push(el("rect", { x: narrow ? 4 : 4, y: y0, width: w - 8, height: rowH, fill: C.panel, opacity: 0.6 }));
    parts.push(text(narrow ? 10 : 10, narrow ? y0 + 13 : cy + 4, lang === "zh" ? r.zh : r.label, { "font-size": TYPE.small, class: "fig-t-muted" }));
    if (r.lo !== undefined && r.hi !== undefined) {
      parts.push(el("line", { x1: x(r.lo), x2: x(r.hi), y1: cy, y2: cy, stroke: C.c1, "stroke-width": 2.4 }));
      for (const v of [r.lo, r.hi]) parts.push(el("line", { x1: x(v), x2: x(v), y1: cy - 4, y2: cy + 4, stroke: C.c1, "stroke-width": 1.6 }));
    }
    if (r.orig !== undefined) {
      const to = r.later ?? (r.lo !== undefined && r.hi !== undefined ? (r.lo + r.hi) / 2 : undefined);
      if (to !== undefined) parts.push(el("line", { x1: x(r.orig), x2: x(to), y1: cy, y2: cy, stroke: C.ink3, "stroke-width": 1, "stroke-dasharray": "2 3" }));
      parts.push(el("circle", { cx: x(r.orig), cy, r: 4.6, fill: C.paper, stroke: C.ink2, "stroke-width": 1.6 }));
    }
    if (r.later !== undefined) parts.push(el("circle", { cx: x(r.later), cy, r: 4.6, fill: C.c1, stroke: C.paper, "stroke-width": 1.2 }));
  });
  // the selected d
  parts.push(el("line", { x1: x(p.d), x2: x(p.d), y1: top - 4, y2: bottom + 2, stroke: C.acq, "stroke-width": 1.6, "stroke-dasharray": "5 3" }));
  parts.push(axis({ scale: x, orient: "bottom", at: bottom + 4, span: [bottom + 4, bottom + 4], title: L.axis, count: narrow ? 4 : 6, grid: false }));
  // hit area: rows, with data x in d units and data y in row index
  const map = [x.range[0], x.range[1], x.domain[0], x.domain[1], ry.range[0], ry.range[1], ry.domain[0], ry.domain[1]].join(",");
  parts.push(el("rect", { x: 0, y: top, width: w, height: bottom - top, fill: "transparent", "data-fig-hit": "rows", "data-fig-map": map }));
  // lower panel: two normal distributions whose means differ by d
  const pTop = bottom + 56, pH = narrow ? 92 : 104, pBot = pTop + pH;
  const zx = linear([-3.6, 4.5], [narrow ? 14 : 40, right]);
  const zy = linear([0, 0.42], [pBot, pTop]);
  const zs = grid(-3.6, 4.5, 163);
  const f1 = (z: number) => normalPdf(z, 0, 1), f2 = (z: number) => normalPdf(z, p.d, 1);
  const area = linePath([[zx(-3.6), zy(0)], ...zs.map((z) => [zx(z), zy(Math.min(f1(z), f2(z)))] as [number, number]), [zx(4.5), zy(0)]]) + "Z";
  parts.push(el("path", { d: area, fill: C.band, stroke: "none" }));
  parts.push(el("path", { d: linePath(zs.map((z) => [zx(z), zy(f1(z))])), fill: "none", stroke: C.ink2, "stroke-width": 1.8 }));
  parts.push(el("path", { d: linePath(zs.map((z) => [zx(z), zy(f2(z))])), fill: "none", stroke: C.acq, "stroke-width": 2.2 }));
  parts.push(el("line", { x1: zx.range[0], x2: zx.range[1], y1: pBot, y2: pBot, stroke: C.rule }));
  // label each curve on its outer side, so the labels never collide
  const lowX = zx(Math.min(0, p.d)) - 30, highX = zx(Math.max(0, p.d)) + 30, ly2 = zy(0.3);
  const [lowLab, highLab, lowCol, highCol] = p.d >= 0 ? [L.first, L.second, "", C.acq] : [L.second, L.first, C.acq, ""];
  parts.push(text(lowX, ly2, lowLab, { "font-size": TYPE.small, "text-anchor": "end", class: lowCol ? "fig-t-halo" : "fig-t-muted fig-t-halo", style: lowCol ? `fill:${lowCol}` : undefined }));
  parts.push(text(highX, ly2, highLab, { "font-size": TYPE.small, class: highCol ? "fig-t-halo" : "fig-t-muted fig-t-halo", style: highCol ? `fill:${highCol}` : undefined }));
  parts.push(text(zx(p.d / 2), zy(0.06), tpl(L.overlap, { o: pc(overlap(p.d)) }), { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-strong fig-t-halo" }));
  // readout (wraps on a phone)
  const read = tpl(L.readout, { d: fixed(p.d, 2), o: pc(overlap(p.d)), p: pc(superiority(p.d)) });
  let y = pBot + 22;
  const cut = read.indexOf(lang === "zh" ? "；" : ";");
  parts.push(text(10, y, read.slice(0, cut + 1), { "font-size": TYPE.body }));
  y += 17;
  parts.push(text(10, y, read.slice(cut + (lang === "zh" ? 1 : 2)), { "font-size": TYPE.body }));
  return svg(w, y + 12, describe(st), ...parts);
}

export default defineFigure({
  name: "jdm-effect-sizes",
  title: { en: "Effect sizes on one scale, and what each means", zh: "同一标尺上的效应量，以及每个效应量的含义" },
  labels,
  params,
  hint: { en: "Click a dot or a row to show its effect size below, or use the slider.", zh: "点击一个点或一行，在下方显示它的效应量，或使用滑块。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click" || e.target !== "rows" || !e.data) return null;
    const rows = ROWS[p.set];
    const i = Math.floor(e.data.y);
    if (i < 0 || i >= rows.length) return null;
    const r = rows[i];
    const marks = [r.orig, r.later, r.lo, r.hi].filter((v): v is number => v !== undefined);
    if (!marks.length) return null;
    const cx = e.data.x;
    let best = marks[0];
    for (const v of marks) if (Math.abs(v - cx) < Math.abs(best - cx)) best = v;
    // A click on the label (left of the axis) or far from every mark picks the
    // row's later estimate, or the original when there is none.
    if (cx < D_DOM[0] || Math.abs(best - cx) > 0.12) best = r.later ?? r.orig ?? best;
    return { ...p, d: best };
  },
  snapshots: {
    "artifact-free": (p) => ({ ...p, d: 0.4 }),
    "many labs original": (p) => ({ ...p, d: 0.6 }),
    "many labs replication": (p) => ({ ...p, d: 0.15 }),
    "social, moral reminders": (p) => ({ ...p, set: "social", d: 0.48 }),
    "social, replication": (p) => ({ ...p, set: "social", d: -0.04 }),
    "social, observed": (p) => ({ ...p, set: "social", d: 0.65 }),
    "social, not observed": (p) => ({ ...p, set: "social", d: 0.13 }),
  },
});
