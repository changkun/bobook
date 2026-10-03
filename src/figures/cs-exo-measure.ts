// What one evaluation of an exoskeleton setting costs: a simulated bout of
// walking in which the metabolic rate, measured breath by breath, moves from
// its old steady state toward the new one as a first-order response with a
// time constant of 42 seconds (Selinger and Donelan, 2014), and a fit of that
// response estimates where it is heading. The breath-to-breath scatter is set
// so that a two-minute estimate has a standard deviation of 4.6% of the
// zero-torque rate, the value Kutulakos and Slade (2024) report for the
// estimates of Zhang et al. (2017). Everything in the figure is simulated.

import { defineFigure, type State } from "./types.ts";
import { el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { legend } from "./lib/plot.ts";
import { normal, rng } from "./lib/random.ts";
import { tpl } from "./lib/format.ts";

const labels = {
  en: {
    yTitle: "metabolic rate (% of walking with zero torque)",
    yShort: "metabolic rate (%)",
    xTitle: "seconds of walking at the new setting",
    breath: "one breath",
    truth: "true response",
    fitted: "fitted response",
    est: "estimate {e}%",
    target: "true value {v}%",
    repeats: "the estimate from 60 repeated bouts of this length",
    repeatsShort: "60 repeated bouts",
    sd: "standard deviation {s} points",
    sdTheory: "theory {s}",
    describe: "A simulated {d}-minute bout: the true steady-state rate is {v}% of zero-torque walking, this bout's estimate is {e}%, and repeated bouts of this length give estimates with a standard deviation of {s} percentage points.",
  },
  zh: {
    yTitle: "代谢率（零力矩行走时的 %）",
    yShort: "代谢率（%）",
    xTitle: "在新设置下行走的秒数",
    breath: "一次呼吸",
    truth: "真实响应",
    fitted: "拟合的响应",
    est: "估计值 {e}%",
    target: "真实值 {v}%",
    repeats: "同样时长的行走重复 60 次所得的估计值",
    repeatsShort: "重复 60 次行走",
    sd: "标准差 {s} 个百分点",
    sdTheory: "理论值 {s}",
    describe: "一次模拟的 {d} 分钟行走：真实的稳态代谢率是零力矩行走时的 {v}%，这次行走的估计值为 {e}%，同样时长的重复行走所得估计值的标准差为 {s} 个百分点。",
  },
};

const params = {
  minutes: { kind: "range", label: { en: "Minutes walked", zh: "行走分钟数" }, min: 1, max: 6, default: 2, step: 0.5 },
  effect: { kind: "range", label: { en: "True change", zh: "真实变化" }, min: -30, max: 10, default: -15, step: 1, unit: { en: "%", zh: "%" } },
  seed: { kind: "range", label: { en: "Bout", zh: "行走编号" }, min: 1, max: 9999, default: 3, step: 1, control: false },
} as const;

type P = { minutes: number; effect: number; seed: number };

const TAU = 42; // seconds
const DT = 3; // seconds between breaths
const TARGET_SD = 0.046; // of a 2-minute estimate

// Least squares for y = a (1 - e^{-t/tau}) + b e^{-t/tau}: a is the steady
// state the response is heading to, b the rate at the start of the bout.
function design(dur: number) {
  const ts: number[] = [];
  for (let t = DT; t <= dur + 1e-9; t += DT) ts.push(t);
  let saa = 0, sab = 0, sbb = 0;
  for (const t of ts) { const e = Math.exp(-t / TAU); saa += (1 - e) ** 2; sab += (1 - e) * e; sbb += e * e; }
  const det = saa * sbb - sab * sab;
  return { ts, saa, sab, sbb, det, sdFactor: Math.sqrt(sbb / det) };
}

const SIGMA_BREATH = TARGET_SD / design(120).sdFactor; // about 0.18

function bout(dur: number, target: number, seed: number) {
  const d = design(dur);
  const z = normal(rng(seed * 2221 + Math.round(dur) * 7 + 1));
  const ys = d.ts.map((t) => { const e = Math.exp(-t / TAU); return target * (1 - e) + 1 * e + SIGMA_BREATH * z(); });
  let ya = 0, yb = 0;
  d.ts.forEach((t, i) => { const e = Math.exp(-t / TAU); ya += ys[i] * (1 - e); yb += ys[i] * e; });
  const a = (d.sbb * ya - d.sab * yb) / d.det;
  const b = (d.saa * yb - d.sab * ya) / d.det;
  return { ts: d.ts, ys, a, b };
}

function describe(st: State<P>): string {
  const p = st.p;
  const target = 1 + p.effect / 100;
  const bt = bout(p.minutes * 60, target, p.seed);
  return tpl(labels[st.lang ?? "en"].describe, { d: p.minutes, v: (target * 100).toFixed(0), e: (bt.a * 100).toFixed(1), s: (design(p.minutes * 60).sdFactor * SIGMA_BREATH * 100).toFixed(1) });
}

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const w = st.w;
  const narrow = w < 480;
  const dur = p.minutes * 60;
  const target = 1 + p.effect / 100;
  const bt = bout(dur, target, p.seed);
  const parts: string[] = [];
  const left = narrow ? 40 : 50, right = w - 12;
  const top = 26, height = narrow ? 170 : 200;
  const xs = linear([0, 360], [left, right]);
  const ys = linear([0.4, 1.6], [top + height, top]);
  parts.push(text(left - (narrow ? 36 : 44), 14, narrow ? L.yShort : L.yTitle, { "font-size": TYPE.small, class: "fig-t-strong" }));
  parts.push(axis({ scale: ys, orient: "left", at: left, span: [left, right], ticks: [0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6], format: (v) => String(Math.round(v * 100)) }));
  parts.push(axis({ scale: xs, orient: "bottom", at: top + height, span: [top, top + height], ticks: [0, 60, 120, 180, 240, 300, 360], title: L.xTitle }));
  // the part of the axis not walked
  parts.push(el("rect", { x: xs(dur), y: top, width: right - xs(dur), height, fill: C.panel, opacity: 0.6 }));
  // breaths
  const inner: string[] = [];
  bt.ts.forEach((t, i) => inner.push(el("circle", { cx: xs(t), cy: ys(Math.min(1.6, Math.max(0.4, bt.ys[i]))), r: 2.3, fill: C.ink3, opacity: 0.75 })));
  const curveT = Array.from({ length: 121 }, (_, i) => (i / 120) * 360);
  inner.push(el("path", { d: linePath(curveT.map((t) => [xs(t), ys(target * (1 - Math.exp(-t / TAU)) + Math.exp(-t / TAU))])), fill: "none", stroke: C.truth, "stroke-width": 2, "stroke-dasharray": "5 4" }));
  const fitT = curveT.filter((t) => t <= dur);
  inner.push(el("path", { d: linePath(fitT.map((t) => [xs(t), ys(bt.a * (1 - Math.exp(-t / TAU)) + bt.b * Math.exp(-t / TAU))])), fill: "none", stroke: C.model, "stroke-width": 2.2 }));
  // the estimate and the true value as levels at the right edge of the bout
  const ey = ys(Math.min(1.6, Math.max(0.4, bt.a))), ty = ys(target);
  inner.push(
    el("line", { x1: xs(dur), x2: right, y1: ey, y2: ey, stroke: C.model, "stroke-width": 1.5 }),
    el("line", { x1: xs(dur), x2: right, y1: ty, y2: ty, stroke: C.truth, "stroke-width": 1.5, "stroke-dasharray": "5 4" }),
  );
  parts.push(g({}, ...inner));
  const above = ey < ty;
  parts.push(
    text(right - 4, ey + (above ? -6 : 14), tpl(L.est, { e: (bt.a * 100).toFixed(1) }), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-strong fig-t-halo", style: `fill:${C.model}` }),
    text(right - 4, ty + (above ? 14 : -6), tpl(L.target, { v: (target * 100).toFixed(0) }), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo" }),
  );
  const lg = legend(left, top + height + 46, right - left, [
    { kind: "dot", color: C.ink3, label: L.breath },
    { kind: "dash", color: C.truth, label: L.truth },
    { kind: "line", color: C.model, label: L.fitted },
  ]);
  parts.push(lg.svg);
  // repeated bouts: a strip of 60 estimates
  const t2 = top + height + 46 + lg.height + 20;
  const sd = design(dur).sdFactor * SIGMA_BREATH;
  parts.push(text(left - (narrow ? 36 : 44), t2, narrow ? L.repeatsShort : L.repeats, { "font-size": TYPE.small, class: "fig-t-strong" }));
  const sx = linear([target - 0.3, target + 0.3], [left, right]);
  const sy = t2 + 34;
  parts.push(el("line", { x1: left, x2: right, y1: sy + 14, y2: sy + 14, stroke: C.rule }));
  for (const dv of [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]) {
    parts.push(
      el("line", { x1: sx(target + dv), x2: sx(target + dv), y1: sy + 14, y2: sy + 18, stroke: C.rule }),
      text(sx(target + dv), sy + 30, `${Math.round((target + dv) * 100)}`, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted fig-t-num" }),
    );
  }
  parts.push(el("line", { x1: sx(target), x2: sx(target), y1: sy - 20, y2: sy + 14, stroke: C.truth, "stroke-width": 2, "stroke-dasharray": "5 4" }));
  const jit = rng(77);
  let s1 = 0, s2 = 0;
  for (let k = 0; k < 60; k++) {
    const a = bout(dur, target, 1000 + k).a;
    s1 += a; s2 += a * a;
    const x = Math.min(right, Math.max(left, sx(a)));
    parts.push(el("circle", { cx: x, cy: sy - 14 + jit() * 22, r: 2.6, fill: C.model, opacity: 0.6 }));
  }
  const esd = Math.sqrt(Math.max(0, s2 / 60 - (s1 / 60) ** 2));
  parts.push(el("circle", { cx: Math.min(right, Math.max(left, sx(bt.a))), cy: sy + 2, r: 4.4, fill: "none", stroke: C.ink, "stroke-width": 1.8 }));
  const sdA = tpl(L.sd, { s: (esd * 100).toFixed(1) }), sdB = tpl(L.sdTheory, { s: (sd * 100).toFixed(1) });
  parts.push(text(right, t2, lang === "zh" ? `${sdA}（${sdB}）` : `${sdA} (${sdB})`, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-num" }));
  return svg(w, sy + 38, describe(st), ...parts);
}

export default defineFigure({
  name: "cs-exo-measure",
  title: { en: "One evaluation: estimating a metabolic rate from a few minutes of breaths (simulated)", zh: "一次评估：从几分钟的呼吸数据估计代谢率（模拟）" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "Walk another bout", zh: "再走一次" }, primary: true, run: (p) => ({ ...p, seed: (p.seed % 9998) + 1 }) },
  ],
});
