// What one comparison can teach. The model's belief about the utility
// difference Δ = g(A) − g(B) is Gaussian with mean m and standard deviation v;
// the answer follows the probit link with noise sd σ per option. The expected
// information the answer carries about Δ is the entropy of the predicted answer
// minus the entropy the answer would still have if Δ were known (the noise).
// The top panel shows the belief against the choice curve; the bottom panel
// shows the two entropies and their gap as the belief's mean moves.

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid } from "./lib/scale.ts";
import { Phi, phi } from "./lib/stats.ts";
import { band, clip, curve, frame, frameAxes } from "./lib/plot.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    belief: "belief about Δ (scaled)",
    choice: "P(A chosen | Δ)",
    delta: "utility difference Δ",
    mean: "mean m of the belief about Δ",
    bits: "bits",
    total: "uncertainty about the answer",
    noise: "noise left if Δ were known",
    info: "information about Δ",
    readout: "{t} − {z} = {i} bits",
    describe: "The belief about the difference has mean {m} and sd {v}; with noise sd {s} per option the answer is uncertain by {t} bits, of which {z} bits are noise, so it carries {i} bits about the difference.",
    prob: "prob.",
    oneBit: "1 bit",
  },
  zh: {
    belief: "关于 Δ 的信念（已缩放）",
    choice: "P(选择 A | Δ)",
    delta: "效用差 Δ",
    mean: "关于 Δ 的信念的均值 m",
    bits: "比特",
    total: "回答的不确定性",
    noise: "已知 Δ 时剩下的噪声",
    info: "关于 Δ 的信息",
    readout: "{t} − {z} = {i} 比特",
    describe: "关于效用差的信念均值为 {m}、标准差为 {v}；每个选项的噪声标准差为 {s} 时，回答的不确定性为 {t} 比特，其中 {z} 比特是噪声，因此回答携带 {i} 比特关于效用差的信息。",
    prob: "概率",
    oneBit: "1 比特",
  },
};

const params = {
  mean: { kind: "range", label: { en: "Belief mean m", zh: "信念均值 m" }, min: -4, max: 4, default: 0.5, step: 0.05 },
  spread: { kind: "range", label: { en: "Belief sd v", zh: "信念标准差 v" }, min: 0.05, max: 3, default: 1, step: 0.05 },
  sigma: { kind: "range", label: { en: "Noise sd σ", zh: "噪声标准差 σ" }, min: 0.05, max: 1.5, default: 0.3, step: 0.05 },
} as const;

type P = { mean: number; spread: number; sigma: number };

const ZS = grid(-6, 6, 121);
const WZ = (() => { const w = ZS.map(phi); const s = w.reduce((a, b) => a + b, 0); return w.map((v) => v / s); })();
const MS = grid(-4, 4, 161);

const h2 = (p: number) => (p <= 1e-12 || p >= 1 - 1e-12 ? 0 : -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p)));

// Entropy of the predicted answer, expected entropy given Δ, and their gap.
function parts(m: number, v: number, sigma: number) {
  const s = Math.SQRT2 * sigma;
  const total = h2(Phi(m / Math.sqrt(s * s + v * v)));
  let noise = 0;
  for (let i = 0; i < ZS.length; i++) noise += WZ[i] * h2(Phi((m + v * ZS[i]) / s));
  return { total, noise, info: Math.max(0, total - noise) };
}

function describe(st: State<P>): string {
  const p = st.p;
  const r = parts(p.mean, p.spread, p.sigma);
  return tpl(labels[st.lang ?? "en"].describe, { m: fixed(p.mean, 2), v: fixed(p.spread, 2), s: fixed(p.sigma, 2), t: fixed(r.total, 2), z: fixed(r.noise, 2), i: fixed(r.info, 2) });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const out: string[] = [];
  const s = Math.SQRT2 * p.sigma;

  // 1. The belief about Δ and the choice curve.
  const f1 = frame({ w: st.w, top: narrow ? 40 : 26, height: narrow ? 100 : 116, xDomain: [-4, 4], yDomain: [0, 1.08], yTitle: L.prob });
  const XS = grid(-4, 4, 241);
  const dens = XS.map((x) => phi((x - p.mean) / p.spread));
  const peak = Math.max(...dens);
  const scaled = dens.map((d) => (0.95 * d) / peak);
  const cid1 = `${st.uid}-c1`;
  out.push(
    el("defs", {}, clip(f1, cid1)),
    frameAxes(f1, { yTitle: L.prob, xTitle: L.delta, yCount: 2 }),
    g({ "clip-path": `url(#${cid1})` },
      band(f1, XS, XS.map(() => 0), scaled),
      curve(f1, XS, scaled, { "stroke-width": 1.6 }),
      curve(f1, XS, XS.map((x) => Phi(x / s)), { stroke: C.acq, "stroke-width": 2 }),
    ),
  );
  // legend above
  const ly = 14;
  out.push(
    el("rect", { x: f1.left, y: ly - 9, width: 16, height: 10, rx: 2, fill: C.band, stroke: C.model }),
    text(f1.left + 21, ly, L.belief, { "font-size": TYPE.small, class: "fig-t-muted" }),
    el("line", { x1: narrow ? f1.left : f1.left + 170, x2: narrow ? f1.left + 16 : f1.left + 186, y1: (narrow ? ly + 15 : ly) - 4, y2: (narrow ? ly + 15 : ly) - 4, stroke: C.acq, "stroke-width": 2 }),
    text(narrow ? f1.left + 21 : f1.left + 191, narrow ? ly + 15 : ly, L.choice, { "font-size": TYPE.small, class: "fig-t-muted" }),
  );
  // legend for the bottom panel, laid out before the panel so it never overlaps
  const items: Array<[string, string, string | undefined, number]> = [[L.total, C.ink2, "5 3", 1.5], [L.noise, C.ink3, "1.5 3", 1.5], [L.info, C.acq, undefined, 2.2]];
  const legendParts: string[] = [];
  let lx = f1.left, lyy = f1.bottom + 52;
  for (const [name, color, dash, sw] of items) {
    const tw = labelWidth(name, 6.1) + 34;
    if (lx + tw > st.w - 8) { lx = f1.left; lyy += 15; }
    legendParts.push(el("line", { x1: lx, x2: lx + 18, y1: lyy - 4, y2: lyy - 4, stroke: color, "stroke-width": sw, "stroke-dasharray": dash }), text(lx + 23, lyy, name, { "font-size": TYPE.small, class: "fig-t-muted" }));
    lx += tw;
  }
  out.push(...legendParts);
  const readY = lyy + 18;

  // 2. Entropies against the belief mean.
  const f2top = readY + 12;
  const f2 = frame({ w: st.w, top: f2top, height: narrow ? 150 : 170, xDomain: [-4, 4], yDomain: [0, 1.05], yTitle: L.bits });
  const rows = MS.map((m) => parts(m, p.spread, p.sigma));
  const tot = rows.map((r) => r.total), noi = rows.map((r) => r.noise), inf = rows.map((r) => r.info);
  const cur = parts(p.mean, p.spread, p.sigma);
  const cid2 = `${st.uid}-c2`;
  out.push(
    el("defs", {}, clip(f2, cid2)),
    frameAxes(f2, { xTitle: L.mean, yTitle: L.bits, yCount: 4 }),
    el("line", { x1: f2.left, x2: f2.right, y1: f2.y(1), y2: f2.y(1), stroke: C.ink3, "stroke-dasharray": "2 3" }),
    g({ "clip-path": `url(#${cid2})` },
      band(f2, MS, noi, tot, C.acqFill),
      curve(f2, MS, tot, { stroke: C.ink2, "stroke-width": 1.5, "stroke-dasharray": "5 3" }),
      curve(f2, MS, noi, { stroke: C.ink3, "stroke-width": 1.5, "stroke-dasharray": "1.5 3" }),
      curve(f2, MS, inf, { stroke: C.acq, "stroke-width": 2.2 }),
    ),
    el("line", { x1: f2.x(p.mean), x2: f2.x(p.mean), y1: f2.top, y2: f2.bottom, stroke: C.model, "stroke-width": 1.3, "stroke-dasharray": "4 3" }),
    el("circle", { cx: f2.x(p.mean), cy: f2.y(cur.info), r: 4.5, fill: C.acq, stroke: C.paper, "stroke-width": 1.5 }),
    text(f2.left + 6, f2.y(1) - 5, L.oneBit, { "font-size": TYPE.small, class: "fig-t-faint fig-t-halo" }),
  );
  out.push(text(f2.right, readY, tpl(L.readout, { t: fixed(cur.total, 2), z: fixed(cur.noise, 2), i: fixed(cur.info, 2) }), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-num fig-t-strong fig-t-halo" }));
  return svg(st.w, f2.bottom + 44, describe(st), ...out);
}

export default defineFigure({
  name: "cmp-information",
  title: { en: "How much one comparison can teach about a utility difference", zh: "一次比较能告诉我们多少关于效用差的信息" },
  labels,
  params,
  hint: { en: "Move the belief's mean and spread, and the noise, and watch the gap between the two entropies.", zh: "调整信念的均值与标准差以及噪声，观察两个熵之间的差距。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
