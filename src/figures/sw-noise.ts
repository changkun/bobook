// How a simulated user errs. PBO papers simulate the person answering a
// comparison with one of several noise models (Software, Evaluation, and the
// Research Community: "Noise models chosen paper by paper"): a fixed
// probability of flipping the answer (10% in BOPE, Lin et al. 2022), logistic
// noise on the utility difference (the Bradley-Terry link; qEUBO, Astudillo et
// al. 2023, calibrates its scale to target error rates), and Gaussian noise on
// each utility (the probit link; Takeno et al. 2023, Siivola et al. 2021, the
// BoTorch tutorial). This module is an illustration of the three models, not
// data: each is matched so that a comparison with utility gap Δ₀ is answered
// wrongly with probability ε, and the curves show the error probability at
// every other gap Δ > 0:
//   flip      ε
//   logistic  sigmoid(-(Δ/Δ₀) · ln((1 - ε)/ε))
//   probit    Φ(-(Δ/Δ₀) · Φ⁻¹(1 - ε))
// Exercise exr-sw-flip computes the probit case by hand (σ ≈ 0.055 for
// Δ₀ = 0.1, ε = 10%).

import { defineFigure, type State } from "./types.ts";
import { el, labelWidth, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear, log } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { Phi, PhiInv, sigmoid } from "./lib/stats.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    x: "utility gap Δ between the two options",
    xShort: "utility gap Δ",
    y: "P(the worse option is chosen)",
    flip: "fixed flip rate",
    logistic: "logistic noise (Bradley-Terry)",
    probit: "Gaussian noise (probit)",
    logisticShort: "logistic",
    probitShort: "probit",
    match: "matched here",
    readout: "At a gap of {g}: fixed flip {f}, logistic {l}, probit {p}.",
    describe: "Matched to answer a utility gap of {g0} wrongly {e} of the time, the three simulated users differ at a gap of {g}: a fixed flip rate errs {f} of the time, logistic noise {l}, and probit noise {p}.",
  },
  zh: {
    x: "两个选项之间的效用差 Δ",
    xShort: "效用差 Δ",
    y: "选中较差选项的概率",
    flip: "固定翻转率",
    logistic: "逻辑噪声（Bradley-Terry）",
    probit: "高斯噪声（概率单位）",
    logisticShort: "逻辑",
    probitShort: "概率单位",
    match: "在此匹配",
    readout: "效用差为 {g} 时：固定翻转 {f}，逻辑 {l}，概率单位 {p}。",
    describe: "三个模拟用户都匹配为在效用差 {g0} 处有 {e} 的回答出错，而在效用差 {g} 处各不相同：固定翻转率出错 {f}，逻辑噪声出错 {l}，概率单位噪声出错 {p}。",
  },
};

const params = {
  matchGap: { kind: "range", label: { en: "Matched at a gap of", zh: "匹配处的效用差" }, min: 0.05, max: 0.25, default: 0.1, step: 0.01 },
  matchErr: { kind: "range", label: { en: "Error rate at that gap", zh: "匹配处的错误率" }, min: 2, max: 30, default: 10, step: 1, unit: { en: "%", zh: "%" } },
  readGap: { kind: "range", label: { en: "Read at a gap of", zh: "读数处的效用差" }, min: 0.01, max: 0.5, default: 0.3, step: 0.01 },
} as const;

type P = { matchGap: number; matchErr: number; readGap: number };

const FLOOR = 1e-6;
const XMAX = 0.5;

function models(p: P) {
  const e = p.matchErr / 100;
  const zP = PhiInv(1 - e);
  const zL = Math.log((1 - e) / e);
  return {
    flip: (_d: number) => e,
    logistic: (d: number) => sigmoid(-(d / p.matchGap) * zL),
    probit: (d: number) => Phi(-(d / p.matchGap) * zP),
  };
}

// Two significant figures, as a percentage.
function pc(v: number): string {
  if (v < FLOOR) return "<0.0001%";
  const x = v * 100;
  return `${x >= 10 ? Math.round(x) : Number(x.toPrecision(2)).toString()}%`;
}

function readouts(p: P) {
  const m = models(p);
  return { f: pc(m.flip(p.readGap)), l: pc(m.logistic(p.readGap)), p: pc(m.probit(p.readGap)) };
}

function describe(st: State<P>): string {
  const p = st.p, r = readouts(p);
  return tpl(labels[st.lang ?? "en"].describe, { g0: fixed(p.matchGap, 2), e: `${p.matchErr}%`, g: fixed(p.readGap, 2), ...r });
}

const SERIES = [
  { key: "flip", color: C.c6, dash: "6 4" },
  { key: "logistic", color: C.c7, dash: "" },
  { key: "probit", color: C.c1, dash: "" },
] as const;

function render(st: State<P>): string {
  const L = labels[st.lang ?? "en"];
  const p = st.p, w = st.w, narrow = w < 480;
  const m = models(p);
  const parts: string[] = [];
  // legend
  let lx = narrow ? 8 : 12, ly = 14;
  for (const s of SERIES) {
    const lab = narrow && s.key !== "flip" ? L[`${s.key}Short` as "logisticShort" | "probitShort"] : L[s.key];
    const tw = labelWidth(lab) + 30;
    if (lx + tw > w - 8 && lx > 12) { lx = narrow ? 8 : 12; ly += 18; }
    parts.push(
      el("line", { x1: lx, x2: lx + 18, y1: ly - 1, y2: ly - 1, stroke: s.color, "stroke-width": 2.4, "stroke-dasharray": s.dash || undefined }),
      text(lx + 24, ly + 3, lab, { "font-size": TYPE.small, class: "fig-t-muted" }),
    );
    lx += tw;
  }
  ly += 20;
  parts.push(text(narrow ? 8 : 12, ly, L.y, { "font-size": narrow ? TYPE.small : TYPE.body, class: "fig-t-muted" }));
  const top = ly + 20;
  const left = narrow ? 62 : 66, right = w - (narrow ? 12 : 24);
  const bottom = top + (narrow ? 210 : 250);
  const x = linear([0, XMAX], [left, right]);
  const y = log([FLOOR, 1], [bottom, top]);
  parts.push(axis({
    scale: y, orient: "left", at: left, span: [left, right],
    format: (v) => pc(v).replace("<", ""),
  }));
  parts.push(axis({ scale: x, orient: "bottom", at: bottom, span: [top, bottom], title: narrow ? L.xShort : L.x, count: narrow ? 4 : 5 }));
  // the matched gap
  const ex = x(p.matchGap), ey = y(p.matchErr / 100);
  parts.push(el("line", { x1: ex, x2: ex, y1: top, y2: bottom, stroke: C.ink3, "stroke-dasharray": "2 3" }));
  parts.push(text(ex + 5, top + 12, L.match, { "font-size": TYPE.small, class: "fig-t-faint" }));
  // curves, clipped at the floor
  const xs = grid(0.0005, XMAX, 240);
  for (const s of SERIES) {
    const f = m[s.key];
    const pts: Array<[number, number]> = [];
    for (const d of xs) { const v = f(d); if (v < FLOOR) break; pts.push([x(d), y(v)]); }
    if (pts.length > 1) parts.push(el("path", { d: linePath(pts), fill: "none", stroke: s.color, "stroke-width": 2.4, "stroke-dasharray": s.dash || undefined }));
  }
  parts.push(el("circle", { cx: ex, cy: ey, r: 4.5, fill: C.ink, stroke: C.paper, "stroke-width": 1.5 }));
  // reading line
  const rx = x(p.readGap);
  parts.push(el("line", { x1: rx, x2: rx, y1: top, y2: bottom, stroke: C.acq, "stroke-width": 1.4 }));
  for (const s of SERIES) {
    const v = m[s.key](p.readGap);
    if (v < FLOOR) continue;
    parts.push(el("circle", { cx: rx, cy: y(v), r: 4, fill: s.color, stroke: C.paper, "stroke-width": 1.5 }));
  }
  const r = readouts(p);
  const ry = bottom + (narrow ? 50 : 54);
  const readout = tpl(L.readout, { g: fixed(p.readGap, 2), ...r });
  const ra = { "font-size": narrow ? TYPE.small : TYPE.body, class: "fig-t-num" };
  // A Chinese readout is wider than its Latin estimate; on a phone, break it after the colon.
  const cut = st.lang === "zh" && narrow ? readout.indexOf("：") + 1 : 0;
  if (cut > 0) parts.push(text(8, ry, readout.slice(0, cut), ra), text(8, ry + 16, readout.slice(cut), ra));
  else parts.push(text(narrow ? 8 : left, ry, readout, ra));
  const H = ry + 12 + (cut > 0 ? 16 : 0);
  return svg(w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "sw-noise",
  title: { en: "How a simulated user errs: three noise models matched at one gap", zh: "模拟用户如何出错：在同一效用差处匹配的三种噪声模型" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  snapshots: {
    "close call": (p) => ({ ...p, readGap: 0.05 }),
    "high noise": (p) => ({ ...p, matchErr: 30 }),
  },
});
