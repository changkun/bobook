// Bayes' rule as bookkeeping, one coin flip at a time. The unknown bias of a
// coin is one of eleven values 0, 0.1, ..., 1, and the reader's belief is a
// table of eleven probabilities. Each flip multiplies the table by the
// likelihood of that flip (θ for heads, 1 − θ for tails) and rescales it to
// sum to one. Three panels show the step: the belief before the last flip,
// the likelihood of that flip, and the belief after, with the unscaled
// product drawn as an outline. The reader calls the flips, or flips a
// mystery coin whose bias is hidden until revealed. The continuous version of
// the same update, with a Beta prior, is the beta-binomial model of the
// Bayesian inference chapter.

import { defineFigure, tr, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear, type Scale } from "./lib/scale.ts";
import { rng } from "./lib/random.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    flips: "Flips",
    noFlips: "No flips yet. Call heads or tails, or flip the mystery coin.",
    count: "{n} flip{n:/s} · {h} H, {t} T",
    earlier: "+{k} earlier",
    prior: "Prior belief",
    before: "Before the last flip",
    likH: "Likelihood of H: θ",
    likT: "Likelihood of T: 1 − θ",
    likNone: "Likelihood (no flip yet)",
    after: "After: belief now",
    afterNone: "Belief now (the prior)",
    theta: "bias θ",
    product: "product",
    divide: "÷ {e}",
    unnorm: "product before rescaling",
    hidden: "mystery bias {b}",
    next: "P(next flip is H) = {p}",
    above: "P(θ > 0.5) = {p}",
    evidence: "The last flip ({s}) had been predicted with probability {e}; dividing by {e} rescales the product to sum to one.",
    evidenceShort: "Last flip ({s}) predicted with probability {e}: the normalizer.",
    describe: "After {n} flip{n:/s} ({h} head{h:/s}, {t} tail{t:/s}) under the {prior} prior, the most probable bias is {mode} and the probability that the next flip lands heads is {next}.",
  },
  zh: {
    flips: "抛掷",
    noFlips: "还没有抛掷。点“正面”或“反面”，或者抛神秘硬币。",
    count: "{n} 次 · {h} 次 H，{t} 次 T",
    earlier: "另有 {k} 次更早",
    prior: "先验信念",
    before: "最近一次抛掷之前",
    likH: "H（正面）的似然：θ",
    likT: "T（反面）的似然：1 − θ",
    likNone: "似然（尚未抛掷）",
    after: "之后：当前信念",
    afterNone: "当前信念（即先验）",
    theta: "偏差 θ",
    product: "乘积",
    divide: "÷ {e}",
    unnorm: "重新缩放之前的乘积",
    hidden: "神秘硬币的偏差 {b}",
    next: "下一次为 H 的概率 = {p}",
    above: "θ 大于 0.5 的概率 = {p}",
    evidence: "最近一次抛掷（{s}）此前被预测的概率为 {e}；除以 {e}，乘积的总和就重新缩放为一。",
    evidenceShort: "最近一次（{s}）的预测概率为 {e}：即归一化因子。",
    describe: "先验为“{prior}”，抛掷 {n} 次（正面 {h} 次，反面 {t} 次）后，最可能的偏差是 {mode}，下一次抛出正面的概率为 {next}。",
  },
};

const PRIORS = [
  { value: "flat", label: { en: "Flat", zh: "平坦" } },
  { value: "fair", label: { en: "Probably fair", zh: "多半公平" } },
  { value: "trick", label: { en: "Maybe a trick coin", zh: "也许是作弊硬币" } },
  { value: "sure", label: { en: "Certain it is fair", zh: "确信公平" } },
] as const;

const MAX_FLIPS = 200;
const AXIS_W = 34; // room for a panel's tick labels

const params = {
  prior: { kind: "choice", label: { en: "Prior", zh: "先验" }, options: PRIORS, default: "flat", control: "select" },
  reveal: { kind: "toggle", label: { en: "Reveal the mystery coin", zh: "揭晓神秘硬币" }, default: false },
  flips: {
    kind: "data", label: { en: "Flips", zh: "抛掷记录" }, default: "HHTH",
    validate: (s: string) => (/^[HT]*$/.test(s) && s.length <= MAX_FLIPS ? undefined : `expected a string of H and T, at most ${MAX_FLIPS} long`),
  },
  seed: { kind: "range", label: { en: "Mystery coin", zh: "神秘硬币" }, min: 1, max: 999, default: 5, step: 1, control: false },
} as const;

type P = { prior: string; reveal: boolean; flips: string; seed: number };

// The eleven hypotheses about the coin's bias.
const TH = Array.from({ length: 11 }, (_, i) => i / 10);

function normalize(w: number[]): number[] {
  const s = w.reduce((a, b) => a + b, 0);
  return w.map((v) => v / s);
}

function priorTable(kind: string): number[] {
  switch (kind) {
    case "fair": return normalize(TH.map((t) => Math.exp(-((t - 0.5) ** 2) / (2 * 0.1 ** 2)) + 0.01));
    case "trick": return normalize(TH.map((t) => (t === 0 || t === 1 ? 0.3 : 0.4 / 9)));
    case "sure": return TH.map((t) => (Math.abs(t - 0.5) < 1e-9 ? 1 : 0));
    default: return TH.map(() => 1 / TH.length);
  }
}

const lik = (flip: string) => TH.map((t) => (flip === "H" ? t : 1 - t));

// Belief after each prefix of the flips, rescaled at every step so that long
// runs never underflow. The normalizer of a step is the probability the
// belief before it gave to the flip that happened.
function run(prior: string, flips: string) {
  let b = priorTable(prior);
  let before = b;
  let product: number[] | undefined;
  let evidence = NaN;
  for (const f of flips) {
    before = b;
    product = before.map((v, i) => v * lik(f)[i]);
    evidence = product.reduce((a, c) => a + c, 0);
    b = product.map((v) => v / evidence);
  }
  return { before: flips.length ? before : b, after: b, product, evidence };
}

function mysteryBias(seed: number): number {
  const opts = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9];
  return opts[Math.floor(rng(seed * 101 + 3)() * opts.length)];
}

function flipMystery(p: P): string {
  if (p.flips.length >= MAX_FLIPS) return p.flips;
  const r = rng(p.seed * 7919 + p.flips.length * 31 + 17);
  return p.flips + (r() < mysteryBias(p.seed) ? "H" : "T");
}

function stats(p: P) {
  const { after } = run(p.prior, p.flips);
  const next = after.reduce((a, v, i) => a + v * TH[i], 0);
  const above = after.reduce((a, v, i) => a + (TH[i] > 0.5 ? v : 0), 0);
  let mode = 0;
  after.forEach((v, i) => { if (v > after[mode] + 1e-12) mode = i; });
  const h = [...p.flips].filter((c) => c === "H").length;
  return { next, above, mode: TH[mode], h, t: p.flips.length - h };
}

function describe(st: State<P>): string {
  const p = st.p;
  const s = stats(p);
  const lang = st.lang ?? "en";
  const pl = PRIORS.find((q) => q.value === p.prior)!.label;
  return tpl(labels[lang].describe, {
    n: p.flips.length, h: s.h, t: s.t, prior: lang === "zh" ? tr(pl, lang) : pl.en.toLowerCase(),
    mode: fixed(s.mode, 1), next: fixed(s.next, 2),
  });
}

// A shared y-scale ceiling for the belief panels, from a short list of round values.
function ceiling(v: number): number {
  for (const c of [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1]) if (v <= c + 1e-9) return c;
  return 1;
}

interface Panel { x0: number; x1: number; top: number; h: number }

function bars(pn: Panel, y: Scale, vals: number[], a: { fill?: string; stroke?: string; dash?: string; opacity?: number }): string {
  const x = linear([-0.05, 1.05], [pn.x0, pn.x1]);
  const bw = (x(0.1) - x(0)) * 0.72;
  return g({}, ...vals.map((v, i) => {
    const h = Math.max(0, y(0) - y(v));
    return el("rect", {
      x: x(TH[i]) - bw / 2, y: y(0) - h, width: bw, height: h, rx: 1.5,
      fill: a.fill ?? "none", stroke: a.stroke, "stroke-width": a.stroke ? 1.3 : undefined,
      "stroke-dasharray": a.dash, opacity: a.opacity,
    });
  }));
}

function panelAxes(pn: Panel, y: Scale, ticks: number[], narrow: boolean): string {
  const x = linear([-0.05, 1.05], [pn.x0, pn.x1]);
  const parts: string[] = [];
  parts.push(el("line", { x1: pn.x0, x2: pn.x1, y1: y(0), y2: y(0), stroke: C.rule }));
  for (const v of ticks) {
    if (v > 0) parts.push(el("line", { x1: pn.x0, x2: pn.x1, y1: y(v), y2: y(v), stroke: C.grid }));
    parts.push(text(pn.x0 - 4, y(v) + 4, v === 0 ? "0" : String(Number(v.toFixed(2))), { "text-anchor": "end", "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
  }
  for (const t of narrow ? [0, 0.5, 1] : [0, 0.5, 1]) {
    parts.push(text(x(t), y(0) + 14, t === 0.5 ? "0.5" : String(t), { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
  }
  return g({}, ...parts);
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const n = p.flips.length;
  const last = n ? p.flips[n - 1] : "";
  const { before, after, product, evidence } = run(p.prior, p.flips);
  const s = stats(p);
  const parts: string[] = [];

  // 1. The flips so far, newest at the right and ringed.
  const chipR = 9, chipGap = 22;
  const rowY = 18;
  const countStr = tpl(L.count, { n, h: s.h, t: s.t });
  const chipsX0 = narrow ? 10 : 56;
  const chipsRight = narrow ? st.w - 10 : st.w - 10 - labelWidth(countStr, 6.4, 11) - 12;
  if (!narrow) parts.push(text(10, rowY + 4, L.flips, { "font-size": TYPE.small, class: "fig-t-muted" }));
  if (!n) {
    parts.push(text(chipsX0, rowY + 4, L.noFlips, { "font-size": TYPE.small, class: "fig-t-faint" }));
  } else {
    let x0 = chipsX0;
    let room = Math.max(1, Math.floor((chipsRight - x0) / chipGap));
    if (n > room) {
      // Reserve space for a "+k earlier" note, then show the newest flips.
      const reserve = 84;
      room = Math.max(1, Math.floor((chipsRight - x0 - reserve) / chipGap));
      parts.push(text(x0, rowY + 4, tpl(L.earlier, { k: n - room }), { "font-size": TYPE.small, class: "fig-t-faint" }));
      x0 += reserve;
    }
    const vis = p.flips.slice(-room);
    [...vis].forEach((c, i) => {
      const cx = x0 + chipR + i * chipGap;
      const isLast = i === vis.length - 1;
      parts.push(
        el("circle", { cx, cy: rowY, r: chipR, fill: c === "H" ? C.c4 : C.paper, stroke: isLast ? C.ink : C.ink3, "stroke-width": isLast ? 2 : 1.2 }),
        text(cx, rowY + 4, c, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-strong" }),
      );
    });
  }
  if (narrow) parts.push(text(10, rowY + 30, countStr, { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  else parts.push(text(st.w - 10, rowY + 4, countStr, { "text-anchor": "end", "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));

  // 2. Three panels: before × likelihood ∝ after.
  const yMax = ceiling(Math.max(...before, ...after) * 1.05);
  const beliefTicks = [0, yMax / 2, yMax];
  const titles = [n ? L.before : L.prior, n ? (last === "H" ? L.likH : L.likT) : L.likNone, n ? L.after : L.afterNone];
  const panels: Panel[] = [];
  let bottom = 0;
  if (narrow) {
    const x0 = 40, x1 = st.w - 10, h = 64;
    let top = rowY + 62;
    for (let i = 0; i < 3; i++) {
      panels.push({ x0, x1, top, h });
      top += h + 52;
    }
    bottom = panels[2].top + panels[2].h + 18;
  } else {
    const opW = 44, axisW = AXIS_W;
    const pw = (st.w - 10 - 2 * opW - 3 * axisW) / 3;
    const top = rowY + 50, h = 150;
    for (let i = 0; i < 3; i++) {
      const x0 = 10 + axisW + i * (pw + axisW + opW);
      panels.push({ x0, x1: x0 + pw, top, h });
    }
    bottom = top + h + 20;
  }
  const ys = panels.map((pn, i) => linear([0, i === 1 ? 1 : yMax], [pn.top + pn.h, pn.top]));
  const rescale = n ? tpl(L.divide, { e: fixed(evidence, 3) }) : "";
  panels.forEach((pn, i) => {
    const title = narrow && i === 2 && n ? `${titles[i]} (${L.product} ${rescale})` : titles[i];
    parts.push(text(pn.x0, pn.top - 10, title, { "font-size": TYPE.small, class: i === 2 ? "fig-t-strong" : "fig-t-muted" }));
    parts.push(panelAxes(pn, ys[i], i === 1 ? [0, 0.5, 1] : beliefTicks, narrow));
  });
  // The operators of Bayes' rule between the panels: before × likelihood ∝ after.
  const opStyle = { "text-anchor": "middle", "font-size": 22, class: "fig-t-glyph" };
  if (narrow) {
    for (const i of [1, 2]) parts.push(text(16, panels[i].top - 8, i === 1 ? "×" : "∝", opStyle));
  } else {
    for (const i of [0, 1]) {
      const xm = (panels[i].x1 + panels[i + 1].x0 - AXIS_W) / 2;
      const ym = panels[i].top + panels[i].h / 2;
      parts.push(text(xm, ym + 7, i === 0 ? "×" : "∝", opStyle));
      if (i === 1 && n) parts.push(text(xm, ym + 26, rescale.replace(" ", ""), { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
    }
  }
  // before
  parts.push(bars(panels[0], ys[0], before, { fill: C.model, opacity: 0.6 }));
  // likelihood
  if (n) parts.push(bars(panels[1], ys[1], lik(last), { fill: C.c7, opacity: 0.8 }));
  else parts.push(text((panels[1].x0 + panels[1].x1) / 2, panels[1].top + panels[1].h / 2, "–", { "text-anchor": "middle", class: "fig-t-faint" }));
  // after, with the unscaled product as an outline
  parts.push(bars(panels[2], ys[2], after, { fill: C.model, opacity: 0.9 }));
  if (product) parts.push(bars(panels[2], ys[2], product, { stroke: C.ink, dash: "3 2" }));
  // the mystery coin's bias
  if (p.reveal) {
    const b = mysteryBias(p.seed);
    const pn = panels[2];
    const x = linear([-0.05, 1.05], [pn.x0, pn.x1]);
    parts.push(
      el("line", { x1: x(b), x2: x(b), y1: pn.top - 2, y2: pn.top + pn.h, stroke: C.truth, "stroke-width": 2, "stroke-dasharray": "5 3" }),
      text(b > 0.5 ? x(b) - 5 : x(b) + 5, pn.top + 10, tpl(L.hidden, { b: fixed(b, 1) }), { "text-anchor": b > 0.5 ? "end" : "start", "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }),
    );
  }
  // x-axis title under the panels
  const xt = bottom + 14;
  parts.push(text((panels[2].x0 + panels[2].x1) / 2, xt, L.theta, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }));
  if (!narrow) for (const i of [0, 1]) parts.push(text((panels[i].x0 + panels[i].x1) / 2, xt, L.theta, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }));

  // 3. Readouts.
  let ry = xt + 22;
  const readA = tpl(L.next, { p: fixed(s.next, 3) });
  const readB = tpl(L.above, { p: fixed(s.above, 3) });
  if (narrow) {
    parts.push(text(10, ry, readA, { "font-size": TYPE.body, class: "fig-t-strong fig-t-num" }));
    ry += 18;
    parts.push(text(10, ry, readB, { "font-size": TYPE.body, class: "fig-t-num" }));
  } else {
    parts.push(text(10, ry, readA, { "font-size": TYPE.body, class: "fig-t-strong fig-t-num" }));
    parts.push(text(10 + labelWidth(readA, 7, 12) + 28, ry, readB, { "font-size": TYPE.body, class: "fig-t-num" }));
  }
  // legend for the outline
  if (product) {
    ry += 20;
    const lx = 10;
    parts.push(
      el("rect", { x: lx, y: ry - 9, width: 12, height: 10, fill: "none", stroke: C.ink, "stroke-width": 1.3, "stroke-dasharray": "3 2" }),
      text(lx + 18, ry, L.unnorm, { "font-size": TYPE.small, class: "fig-t-muted" }),
    );
    ry += 18;
    const ev = tpl(narrow ? L.evidenceShort : L.evidence, { s: last, e: fixed(evidence, 3) });
    if (narrow) {
      // two lines at phone width
      const cut = ev.search(/[:：]/);
      parts.push(text(10, ry, ev.slice(0, cut + 1), { "font-size": TYPE.small, class: "fig-t-muted" }));
      ry += 15;
      parts.push(text(10, ry, ev.slice(cut + 1).trimStart(), { "font-size": TYPE.small, class: "fig-t-muted" }));
    } else {
      parts.push(text(10, ry, ev, { "font-size": TYPE.small, class: "fig-t-muted" }));
    }
  }
  const H = ry + 10;
  return svg(st.w, H, describe(st), ...parts);
}

const add = (p: P, f: string): P => (p.flips.length >= MAX_FLIPS ? p : { ...p, flips: p.flips + f });

export default defineFigure({
  name: "prob-coin-update",
  title: { en: "Updating a belief about a coin's bias, one flip at a time", zh: "逐次抛掷，更新对硬币偏差的信念" },
  labels,
  params,
  hint: { en: "Call each flip yourself, or flip the mystery coin and try to guess its bias before revealing it.", zh: "自己指定每次抛掷的结果，或者抛神秘硬币，并在揭晓之前猜一猜它的偏差。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "Heads", zh: "正面" }, primary: true, run: (p) => add(p as P, "H"), enabled: (p) => p.flips.length < MAX_FLIPS },
    { label: { en: "Tails", zh: "反面" }, primary: true, run: (p) => add(p as P, "T"), enabled: (p) => p.flips.length < MAX_FLIPS },
    { label: { en: "Flip the mystery coin", zh: "抛神秘硬币" }, run: (p) => ({ ...p, flips: flipMystery(p as P) }), enabled: (p) => p.flips.length < MAX_FLIPS },
    {
      label: { en: "Flip it ten times", zh: "抛十次" },
      run: (p) => { let q = p as P; for (let i = 0; i < 10; i++) q = { ...q, flips: flipMystery(q) }; return q; },
      enabled: (p) => p.flips.length < MAX_FLIPS,
    },
    { label: { en: "Undo", zh: "撤销" }, run: (p) => ({ ...p, flips: p.flips.slice(0, -1) }), enabled: (p) => p.flips !== "" },
    { label: { en: "New coin", zh: "换一枚硬币" }, run: (p) => ({ ...p, flips: "", reveal: false, seed: (p.seed % 999) + 1 }) },
  ],
});
