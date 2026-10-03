// Noise or change? A person answers the same pair twice. The utility
// difference between the two options is Δ in the long run; in any one session
// it is shifted by a session effect δ ~ N(0, τ²), and each answer adds
// response noise (probit, σ per option). Asked twice within one session, the
// answers share δ; asked in two sessions, they do not. The main panel plots
// the probability that the two answers differ against the long-run gap Δ; the
// side panel averages over a mix of pairs and sets the averages beside the
// 6% to 20% range reported across sessions in kidney-allocation studies.
// Within-session repeats are treated as independent given δ, which ignores
// memory of the first answer.

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { Phi } from "./lib/stats.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "long-run utility gap Δ between the two options",
    y: "P(the two answers differ)",
    within: "asked twice in one session",
    across: "asked in two sessions",
    avg: "average over pairs",
    avgSub: "gaps 0 to {m}",
    band: "reported across sessions",
    bandShort: "reported",
    readout: "Average over pairs with gaps spread evenly from 0 to {m}: {w} within a session, {a} across sessions.",
    describe: "With response noise σ = {s} and session-to-session change τ = {t}, a pair asked twice in one session gives different answers {w} of the time on average, and asked in two sessions {a}.",
    xShort: "utility gap Δ",
    bandRange: "6% to 20%",
  },
  zh: {
    x: "两个选项之间的长期效用差距 Δ",
    y: "P(两次回答不同)",
    within: "在一次会话中问两次",
    across: "在两次会话中各问一次",
    avg: "对各对取平均",
    avgSub: "差距 0 至 {m}",
    band: "跨会话的报告值",
    bandShort: "报告值",
    readout: "对差距在 0 至 {m} 之间均匀分布的各对取平均：一次会话内为 {w}，跨会话为 {a}。",
    describe: "反应噪声 σ = {s}、会话之间的改变 τ = {t} 时，同一对在一次会话中问两次，平均有 {w} 的时候回答不同；在两次会话中各问一次，则为 {a}。",
    xShort: "效用差距 Δ",
    bandRange: "6% 至 20%",
  },
};

const params = {
  sigma: { kind: "range", label: { en: "Response noise σ", zh: "反应噪声 σ" }, min: 0.05, max: 1.5, default: 0.3, step: 0.05 },
  tau: { kind: "range", label: { en: "Change between sessions τ", zh: "会话之间的改变 τ" }, min: 0, max: 1.5, default: 0, step: 0.05 },
  maxGap: { kind: "range", label: { en: "Pairs: gaps up to", zh: "各对的差距上限" }, min: 0.5, max: 3, default: 1.5, step: 0.1 },
} as const;

type P = { sigma: number; tau: number; maxGap: number };

// Expectation over δ ~ N(0, τ²) by a fine grid on ±5 standard deviations,
// weighted by the normal density and renormalized.
const QZ = grid(-5, 5, 81);
const QW = (() => { const w = QZ.map((z) => Math.exp(-0.5 * z * z)); const s = w.reduce((a, b) => a + b, 0); return w.map((v) => v / s); })();

function flipWithin(d: number, s: number, t: number): number {
  let out = 0;
  for (let k = 0; k < QZ.length; k++) {
    const p = Phi((d + t * QZ[k]) / (Math.SQRT2 * s));
    out += QW[k] * 2 * p * (1 - p);
  }
  return out;
}

function flipAcross(d: number, s: number, t: number): number {
  const p = Phi(d / Math.sqrt(2 * s * s + t * t));
  return 2 * p * (1 - p);
}

const XS = grid(0, 3, 121);

function averages(p: P) {
  const gs = grid(0, p.maxGap, 61);
  const w = gs.reduce((a, d) => a + flipWithin(d, p.sigma, p.tau), 0) / gs.length;
  const a = gs.reduce((acc, d) => acc + flipAcross(d, p.sigma, p.tau), 0) / gs.length;
  return { w, a };
}

const pc = (v: number) => `${Math.round(v * 100)}%`;

function describe(st: State<P>): string {
  const { w, a } = averages(st.p);
  return tpl(labels[st.lang ?? "en"].describe, { s: fixed(st.p.sigma, 2), t: fixed(st.p.tau, 2), w: pc(w), a: pc(a) });
}

function render(st: State<P>): string {
  const L = labels[st.lang ?? "en"];
  const p = st.p, w = st.w, narrow = w < 480;
  const parts: string[] = [];
  // legend
  const lgY = 14;
  let lx = narrow ? 8 : 52;
  let ly = lgY;
  for (const [lab, col, dash] of [[L.within, C.c1, ""], [L.across, C.c2, "6 4"]] as const) {
    const tw = labelWidth(lab, 6.3) + 30;
    if (lx + tw > w - 8 && lx > 52) { lx = narrow ? 8 : 52; ly += 18; }
    parts.push(el("line", { x1: lx, x2: lx + 18, y1: ly - 1, y2: ly - 1, stroke: col, "stroke-width": 2.4, "stroke-dasharray": dash || undefined }), text(lx + 24, ly + 3, lab, { "font-size": TYPE.small, class: "fig-t-muted" }));
    lx += tw;
  }
  if (narrow) { ly += 20; parts.push(text(8, ly, L.y, { "font-size": TYPE.small, class: "fig-t-muted" })); }
  const top = ly + 22;
  const sideW = narrow ? 92 : 130;
  const left = narrow ? 44 : 56, right = w - sideW - (narrow ? 14 : 24);
  const H0 = narrow ? 180 : 210;
  const bottom = top + H0;
  const x = linear([0, 3], [left, right]);
  const y = linear([0, 0.5], [bottom, top]);
  parts.push(axis({ scale: y, orient: "left", at: left, span: [left, right], title: narrow ? undefined : L.y, count: 5, format: (v) => `${Math.round(v * 100)}%` }));
  parts.push(axis({ scale: x, orient: "bottom", at: bottom, span: [top, bottom], title: narrow ? L.xShort : L.x, count: narrow ? 3 : 6 }));
  // the range of pairs that the averages use
  parts.push(el("rect", { x: x(0), y: top, width: x(p.maxGap) - x(0), height: bottom - top, fill: C.acqFill, opacity: 0.6 }));
  const path = (f: (d: number) => number) => linePath(XS.map((d) => [x(d), y(f(d))]));
  parts.push(el("path", { d: path((d) => flipWithin(d, p.sigma, p.tau)), fill: "none", stroke: C.c1, "stroke-width": 2.4 }));
  parts.push(el("path", { d: path((d) => flipAcross(d, p.sigma, p.tau)), fill: "none", stroke: C.c2, "stroke-width": 2.4, "stroke-dasharray": "6 4" }));
  // side panel: averages against the reported band
  const sx0 = right + (narrow ? 14 : 24), sx1 = w - (narrow ? 6 : 10);
  const { w: aw, a: aa } = averages(p);
  parts.push(
    el("rect", { x: sx0, y: y(0.2), width: sx1 - sx0, height: y(0.06) - y(0.2), fill: C.grid, stroke: C.rule, "stroke-dasharray": "3 3" }),
    text((sx0 + sx1) / 2, y(0.2) - 20, narrow ? L.bandShort : L.band, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-faint" }),
    text((sx0 + sx1) / 2, y(0.2) - 6, L.bandRange, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-faint fig-t-num" }),
    el("line", { x1: sx0, x2: sx0, y1: top, y2: bottom, stroke: C.rule }),
  );
  const mx = (sx0 + sx1) / 2;
  const dotLab = (v: number, col: string, dx: number) => g({},
    el("circle", { cx: mx + dx, cy: y(Math.min(v, 0.5)), r: 5.5, fill: col, stroke: C.paper, "stroke-width": 1.5 }),
    text(mx + dx + (dx < 0 ? -9 : 9), y(Math.min(v, 0.5)) + 4, pc(v), { "font-size": TYPE.small, "text-anchor": dx < 0 ? "end" : "start", class: "fig-t-num fig-t-halo" }),
  );
  parts.push(dotLab(aw, C.c1, -10), dotLab(aa, C.c2, 10));
  parts.push(text(mx, bottom + 16, L.avg, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted" }), text(mx, bottom + 30, tpl(L.avgSub, { m: fixed(p.maxGap, 1) }), { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-faint" }));
  const H = bottom + (narrow ? 44 : 50);
  return svg(w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "sp-flip",
  title: { en: "Noise or change: answers that differ within and across sessions", zh: "噪声还是改变：会话内与跨会话的回答差异" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
