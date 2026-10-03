// What an "about the same" answer is worth. A person compares a and b, whose
// utilities differ by Delta = g(a) - g(b), with Gaussian comparison noise of
// unit standard deviation and an indifference threshold delta: they say "a"
// when the perceived difference exceeds delta, "b" when it is below -delta,
// and "about the same" in between (a Thurstone model with a
// just-noticeable-difference band). Top: the three answer probabilities
// against Delta, and the forced-choice probability of "a" when the same
// person must flip a coin instead of saying "about the same". Bottom: the
// mutual information between the answer and Delta, in bits, when the model's
// belief about Delta is N(m, spread^2), for the three-answer interface and
// for the forced choice. The forced choice is a garbling of the three-answer
// response, so by the data-processing inequality it never carries more.

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid } from "./lib/scale.ts";
import { Phi, normalPdf } from "./lib/stats.ts";
import { curve, frame, frameAxes, hitArea, legend } from "./lib/plot.ts";
import { memo } from "./lib/random.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    pa: "P(says a)",
    pb: "P(says b)",
    ps: "P(says about the same)",
    pf: "P(a) if forced to choose",
    belief: "model's belief about Δ",
    band: "indifference band",
    xTop: "utility difference Δ = g(a) − g(b), in noise units",
    yTop: "probability",
    xBot: "the model's best guess m of Δ",
    yBot: "bits per answer",
    three: "three answers",
    forced: "forced choice",
    readout: "At m = {m}: three answers carry {i3} bits, a forced choice {i2} bits ({pct} of it).",
    describe: "With an indifference threshold of {d} noise units and a belief about Δ with spread {s}, an answer at m = {m} carries {i3} bits with an about-the-same option and {i2} bits as a forced choice.",
    xTopShort: "Δ, in noise units",
    oneBit: "1 bit",
  },
  zh: {
    pa: "P(回答 a)",
    pb: "P(回答 b)",
    ps: "P(回答差不多)",
    pf: "强制选择时的 P(a)",
    belief: "模型关于 Δ 的信念",
    band: "无差异带",
    xTop: "效用差 Δ = g(a) − g(b)，以噪声为单位",
    yTop: "概率",
    xBot: "模型对 Δ 的最佳猜测 m",
    yBot: "每个回答的比特数",
    three: "三种回答",
    forced: "强制选择",
    readout: "在 m = {m} 处：三种回答携带 {i3} 比特，强制选择携带 {i2} 比特（为前者的 {pct}）。",
    describe: "无差异阈值为 {d} 个噪声单位、关于 Δ 的信念的分散程度为 {s} 时，m = {m} 处的一个回答在有“差不多”选项时携带 {i3} 比特，在强制选择时携带 {i2} 比特。",
    xTopShort: "Δ，以噪声为单位",
    oneBit: "1 比特",
  },
};

const params = {
  delta: { kind: "range", label: { en: "Indifference threshold δ", zh: "无差异阈值 δ" }, min: 0, max: 2, default: 0.75, step: 0.05 },
  spread: { kind: "range", label: { en: "Uncertainty about Δ", zh: "关于 Δ 的不确定性" }, min: 0.25, max: 3, default: 1.5, step: 0.05 },
  m: { kind: "range", label: { en: "Model's best guess m", zh: "模型的最佳猜测 m" }, min: -3, max: 3, default: 0, step: 0.1 },
} as const;

type P = { delta: number; spread: number; m: number };

const DX = grid(-4, 4, 161);
const MX = grid(-3.5, 3.5, 71);

function probs(d: number, delta: number): [number, number, number] {
  const pa = Phi(d - delta), pb = Phi(-d - delta);
  return [pa, pb, Math.max(0, 1 - pa - pb)];
}

const H = (ps: number[]) => ps.reduce((s, p) => (p > 1e-15 ? s - p * Math.log2(p) : s), 0);

// I(answer; Delta) for Delta ~ N(m, s^2), by quadrature on m +- 6 s.
function info(m: number, s: number, delta: number): [number, number] {
  const n = 121;
  const zs = grid(-6, 6, n);
  const ws = zs.map((z) => normalPdf(z, 0, 1));
  const wsum = ws.reduce((a, b) => a + b, 0);
  const avg3 = [0, 0, 0], avg2 = [0, 0];
  let h3 = 0, h2 = 0;
  zs.forEach((z, i) => {
    const w = ws[i] / wsum;
    const [pa, pb, ps] = probs(m + s * z, delta);
    const fa = pa + ps / 2, fb = pb + ps / 2;
    avg3[0] += w * pa; avg3[1] += w * pb; avg3[2] += w * ps;
    avg2[0] += w * fa; avg2[1] += w * fb;
    h3 += w * H([pa, pb, ps]);
    h2 += w * H([fa, fb]);
  });
  return [Math.max(0, H(avg3) - h3), Math.max(0, H(avg2) - h2)];
}

const curves = memo((delta: number, spread: number) => {
  const i3: number[] = [], i2: number[] = [];
  for (const m of MX) { const [a, b] = info(m, spread, delta); i3.push(a); i2.push(b); }
  return { i3, i2 };
}, 32);

function readout(p: P) {
  const [i3, i2] = info(p.m, p.spread, p.delta);
  return { i3, i2, pct: i3 > 1e-9 ? `${Math.round((100 * i2) / i3)}%` : "–" };
}

function describe(st: State<P>): string {
  const p = st.p;
  const r = readout(p);
  return tpl(labels[st.lang ?? "en"].describe, { d: fixed(p.delta, 2), s: fixed(p.spread, 2), m: fixed(p.m, 1), i3: fixed(r.i3, 2), i2: fixed(r.i2, 2) });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const parts: string[] = [];
  // Top panel: answer probabilities.
  const lg = legend(narrow ? 8 : 52, 14, st.w - 16, [
    { kind: "line", color: C.c1, label: L.pa },
    { kind: "line", color: C.c7, label: L.pb },
    { kind: "line", color: C.c6, label: L.ps },
    { kind: "dash", color: C.c1, label: L.pf },
    { kind: "band", color: C.rule, label: L.belief },
  ]);
  parts.push(lg.svg);
  const top = 14 + lg.height + 8;
  const f = frame({ w: st.w, top, height: narrow ? 130 : 150, yDomain: [0, 1.02], xDomain: [-4, 4], yTitle: L.yTop });
  // indifference band
  parts.push(el("rect", { x: f.x(-p.delta), y: f.top, width: Math.max(0, f.x(p.delta) - f.x(-p.delta)), height: f.bottom - f.top, fill: C.c6, opacity: 0.1 }));
  // the model's belief about Delta, drawn as a faint density along the bottom
  const peak = normalPdf(p.m, p.m, p.spread);
  const dens = DX.map((d) => 0.35 * normalPdf(d, p.m, p.spread) / peak);
  parts.push(el("path", {
    d: `M${f.x(DX[0])},${f.y(0)}` + DX.map((d, i) => `L${f.x(d).toFixed(1)},${f.y(dens[i]).toFixed(1)}`).join("") + `L${f.x(DX[DX.length - 1])},${f.y(0)}Z`,
    fill: C.ink3, opacity: 0.18, stroke: "none",
  }));
  parts.push(frameAxes(f, { yTitle: L.yTop, xTitle: narrow ? L.xTopShort : L.xTop, yCount: 3 }));
  const P3 = DX.map((d) => probs(d, p.delta));
  parts.push(
    curve(f, DX, P3.map((q) => q[0]), { stroke: C.c1 }),
    curve(f, DX, P3.map((q) => q[1]), { stroke: C.c7 }),
    curve(f, DX, P3.map((q) => q[2]), { stroke: C.c6 }),
    curve(f, DX, P3.map((q) => q[0] + q[2] / 2), { stroke: C.c1, "stroke-dasharray": "5 4", "stroke-width": 1.6 }),
    hitArea(f, "top"),
  );
  if (p.delta > 0.05) parts.push(text(f.x(0), f.top + 12, L.band, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted fig-t-halo" }));
  // Bottom panel: bits per answer.
  const btop = f.bottom + (narrow ? 58 : 62);
  const fb = frame({ w: st.w, top: btop, height: narrow ? 120 : 140, yDomain: [0, Math.log2(3) * 1.02], xDomain: [-3.5, 3.5], yTitle: L.yBot });
  const { i3, i2 } = curves(p.delta, p.spread);
  parts.push(
    frameAxes(fb, { yTitle: L.yBot, xTitle: L.xBot, yCount: 3 }),
    el("line", { x1: fb.left, x2: fb.right, y1: fb.y(1), y2: fb.y(1), stroke: C.rule, "stroke-dasharray": "2 3" }),
    text(fb.right - 2, fb.y(1) - 4, L.oneBit, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo" }),
    el("path", { d: MX.map((m, i) => `${i ? "L" : "M"}${fb.x(m).toFixed(1)},${fb.y(i3[i]).toFixed(1)}`).join("") + MX.slice().reverse().map((m, j) => `L${fb.x(m).toFixed(1)},${fb.y(i2[MX.length - 1 - j]).toFixed(1)}`).join("") + "Z", fill: C.c6, opacity: 0.16 }),
    curve(fb, MX, i3, { stroke: C.c6 }),
    curve(fb, MX, i2, { stroke: C.c1, "stroke-dasharray": "5 4" }),
    hitArea(fb, "bottom"),
  );
  const r = readout(p);
  const mx = fb.x(p.m);
  parts.push(
    el("line", { x1: mx, x2: mx, y1: fb.top, y2: fb.bottom, stroke: C.ink, "stroke-width": 1.2, "stroke-dasharray": "3 3" }),
    el("circle", { cx: mx, cy: fb.y(r.i3), r: 4.2, fill: C.c6, stroke: C.paper, "stroke-width": 1.5 }),
    el("circle", { cx: mx, cy: fb.y(r.i2), r: 4.2, fill: C.c1, stroke: C.paper, "stroke-width": 1.5 }),
  );
  const lx = narrow ? fb.left : fb.left + 8;
  parts.push(
    el("line", { x1: lx, x2: lx + 18, y1: fb.top + 10, y2: fb.top + 10, stroke: C.c6, "stroke-width": 2 }),
    text(lx + 24, fb.top + 14, L.three, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
    el("line", { x1: lx, x2: lx + 18, y1: fb.top + 26, y2: fb.top + 26, stroke: C.c1, "stroke-width": 2, "stroke-dasharray": "5 4" }),
    text(lx + 24, fb.top + 30, L.forced, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
  );
  const ry = fb.bottom + 50;
  const msg = tpl(L.readout, { m: fixed(p.m, 1), i3: fixed(r.i3, 2), i2: fixed(r.i2, 2), pct: r.pct });
  if (narrow) {
    const zh = st.lang === "zh";
    const cut = msg.indexOf(zh ? "，强制" : ", a forced");
    parts.push(
      text(fb.left - 30, ry, msg.slice(0, cut + 1), { "font-size": TYPE.small, class: "fig-t-strong fig-t-num" }),
      text(fb.left - 30, ry + 16, msg.slice(cut + (zh ? 1 : 2)), { "font-size": TYPE.small, class: "fig-t-strong fig-t-num" }),
    );
  } else {
    parts.push(text(fb.left, ry, msg, { "font-size": TYPE.small, class: "fig-t-strong fig-t-num" }));
  }
  return svg(st.w, ry + (narrow ? 24 : 10), describe(st), g({}, ...parts));
}

export default defineFigure({
  name: "query-ties",
  title: { en: "What an about-the-same answer is worth", zh: "“差不多”这个回答值多少" },
  labels,
  params,
  hint: { en: "Click either plot to move the model's best guess m, or use the controls.", zh: "点击任一幅图来移动模型的最佳猜测 m，或使用控件。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (!e.data || (e.target !== "top" && e.target !== "bottom")) return null;
    const m = Math.max(-3, Math.min(3, Math.round(e.data.x * 10) / 10));
    return { ...p, m };
  },
});
