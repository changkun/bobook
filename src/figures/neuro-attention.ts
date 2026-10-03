// How much attention moves choice. Data as quoted in the neuroscience chapter
// (en/perspectives/03-neuroscience.md, section sec-neuro-ddm-evidence):
//   Bhatnagar and Orquin 2022, meta-analysis of experiments that manipulated
//   attention in two-option preferential choice: probability of choosing the
//   option attention was steered toward,
//     total looking time increased   P = .541, 95% CI [.523, .560]
//     last fixation controlled        P = .532, 95% CI [.518, .547]
//     first fixation manipulated      P = .507, 95% CI [.497, .516], p = .18
//   (slight publication bias lowers these values a little).
//   Krajbich et al. 2010, attentional drift-diffusion model: the discount on
//   the unattended option was 0.3 as the best fit to the pooled data of 39
//   participants; fits to each participant had mean 0.52 and SD 0.3.
// The session readout is arithmetic on the pooled values: (P - 0.5) x n
// answers out of n comparisons that all carried the manipulation, with the
// same arithmetic on the interval ends. No other data.

import { defineFigure, type Lang, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    panelA: "Probability of choosing the option attention was steered toward",
    axisA: "probability (dashed line at 0.5: no effect)",
    axisAShort: "probability (dashed: no effect)",
    panelB: "Discount on the option not being looked at",
    axisB: "discount (1: the unattended option counts fully)",
    axisBShort: "discount (1: counts fully)",
    pooled: "pooled fit, 39 participants",
    indiv: "each participant, mean ± 1 SD",
    session: "In {n} comparisons that all carried this manipulation, the option attention was steered toward would win about {d} more answers ({lo} to {hi}) than with no effect.",
    describe: "{name}: P = {p} (95% CI {lo} to {hi}). {session} The pooled discount on the unattended option is 0.3; per-participant fits average 0.52 with SD 0.3.",
  },
  zh: {
    panelA: "注意被引向某个选项时，选择该选项的概率",
    axisA: "概率（0.5 处的虚线：无效应）",
    axisAShort: "概率（虚线：无效应）",
    panelB: "对未被注视选项的折扣",
    axisB: "折扣（1：未被注视的选项完全计入）",
    axisBShort: "折扣（1：完全计入）",
    pooled: "合并拟合，39 名被试",
    indiv: "每名被试，均值 ± 1 个标准差",
    session: "在 {n} 次比较都带有这一操纵的会话中，注意被引向的选项会比无效应时约多赢得 {d} 个回答（{lo} 至 {hi}）。",
    describe: "{name}：P = {p}（95% 置信区间 {lo} 至 {hi}）。{session}对未被注视选项的合并折扣为 0.3；对每名被试分别拟合，平均值为 0.52，标准差为 0.3。",
  },
};

interface Row { id: string; name: string; zh: string; p: number; lo: number; hi: number; note?: string }

const ROWS: Row[] = [
  { id: "total", name: "Total looking time", zh: "总注视时间", p: 0.541, lo: 0.523, hi: 0.56 },
  { id: "last", name: "Last fixation", zh: "最后一次注视", p: 0.532, lo: 0.518, hi: 0.547 },
  { id: "first", name: "First fixation", zh: "首次注视", p: 0.507, lo: 0.497, hi: 0.516, note: "p = .18" },
];

const params = {
  manip: { kind: "choice", label: { en: "Manipulation", zh: "操纵" }, options: ROWS.map((r) => ({ value: r.id, label: { en: r.name, zh: r.zh } })), default: "total", control: "buttons" },
  n: { kind: "range", label: { en: "Comparisons in a session", zh: "一次会话中的比较次数" }, min: 10, max: 100, default: 40, step: 5 },
} as const;

type P = { manip: string; n: number };

const pick = (id: string) => ROWS.find((r) => r.id === id) ?? ROWS[0];
const p3 = (v: number) => fixed(v, 3).replace(/^0\./, ".");
const nameOf = (r: Row, lang: Lang) => (lang === "zh" ? r.zh : r.name);

function sessionText(p: P, lang: Lang = "en"): string {
  const r = pick(p.manip);
  const f = (v: number) => fixed((v - 0.5) * p.n, 1);
  return tpl(labels[lang].session, { n: p.n, d: f(r.p), lo: f(r.lo), hi: f(r.hi) });
}

function describe(st: State<P>): string {
  const lang = st.lang ?? "en";
  const r = pick(st.p.manip);
  return tpl(labels[lang].describe, { name: nameOf(r, lang), p: p3(r.p), lo: p3(r.lo), hi: p3(r.hi), session: sessionText(st.p, lang) });
}

function wrapEn(s: string, width: number, px = 6.6): string[] {
  const max = Math.max(12, Math.floor(width / px));
  const out: string[] = [];
  let line = "";
  for (const w of s.split(" ")) {
    if (line && (line + " " + w).length > max) { out.push(line); line = w; } else line = line ? `${line} ${w}` : w;
  }
  if (line) out.push(line);
  return out;
}

// Wrap Chinese text, which has no spaces between words, to lines that fit
// `width` pixels: a Chinese character counts 1.9 Latin letters of `px`
// pixels, a run of Latin letters, digits, and symbols is kept whole, and a
// line never starts with closing punctuation.
function wrapZh(s: string, width: number, px = 6.6): string[] {
  const max = Math.max(12, Math.floor(width / px));
  const toks = s.match(/[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]|[^\s\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]+|\s+/g) ?? [];
  const tw = (t: string) => labelWidth(t, 1, 1.9);
  const lines: string[] = [];
  let line = "", w = 0;
  for (const t of toks) {
    const k = tw(t);
    if (line && w + k > max && !/^[，。；：、）？！”]$/.test(t) && !/^\s+$/.test(t)) { lines.push(line.trimEnd()); line = ""; w = 0; }
    if (!line && /^\s+$/.test(t)) continue;
    line += t; w += k;
  }
  if (line.trim()) lines.push(line.trimEnd());
  return lines;
}

function render(st: State<P>): string {
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const wrap = lang === "zh" ? wrapZh : wrapEn;
  const p = st.p, w = st.w, narrow = w < 480;
  const sel = pick(p.manip);
  const parts: string[] = [];
  const lx = narrow ? 8 : 12;
  const labelW = narrow ? 0 : 170;
  const left = lx + labelW + (narrow ? 6 : 12);
  const right = w - 16;

  // Panel A: the three manipulations.
  for (const [i, ln] of wrap(L.panelA, w - 2 * lx, 7).entries()) parts.push(text(lx, 16 + i * 17, ln, { "font-size": TYPE.label, class: "fig-t-strong" }));
  const titleA = wrap(L.panelA, w - 2 * lx, 7).length;
  const top = 16 + titleA * 17 + 6;
  const rowH = narrow ? 36 : 28;
  const x = linear([0.48, 0.58], [left, right]);
  const bottom = top + rowH * ROWS.length;
  parts.push(axis({ scale: x, orient: "bottom", at: bottom, span: [top - 4, bottom], title: narrow ? L.axisAShort : L.axisA, count: narrow ? 4 : 5, format: (v) => p3(v) }));
  parts.push(el("line", { x1: x(0.5), x2: x(0.5), y1: top - 4, y2: bottom, stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "4 3" }));
  ROWS.forEach((r, i) => {
    const y0 = top + i * rowH;
    const yl = narrow ? y0 + rowH - 10 : y0 + rowH / 2;
    const isSel = r.id === sel.id;
    const col = isSel ? C.c1 : C.ink2;
    const items: string[] = [];
    if (isSel) items.push(el("rect", { x: lx - 4, y: y0 + 1, width: w - 2 * lx + 8, height: rowH - 2, rx: 4, fill: C.panel, stroke: C.rule }));
    items.push(el("rect", { x: lx - 4, y: y0, width: w - 2 * lx + 8, height: rowH, fill: "transparent" }));
    const label = r.note ? (lang === "zh" ? `${r.zh}（${r.note}）` : `${r.name} (${r.note})`) : nameOf(r, lang);
    if (narrow) items.push(text(lx + 2, y0 + 12, label, { "font-size": TYPE.small, class: (isSel ? "fig-t-strong" : "fig-t-muted") + " fig-t-halo" }));
    else items.push(text(lx + labelW, yl + 4, nameOf(r, lang), { "font-size": TYPE.small, "text-anchor": "end", class: isSel ? "fig-t-strong" : "fig-t-muted" }));
    items.push(el("line", { x1: x(r.lo), x2: x(r.hi), y1: yl, y2: yl, stroke: col, "stroke-width": 3, "stroke-linecap": "round" }));
    items.push(el("circle", { cx: x(r.p), cy: yl, r: 5, fill: col, stroke: C.paper, "stroke-width": 1.4 }));
    if (!narrow) items.push(text(x(r.hi) + 8, yl + 4, `${p3(r.p)} [${p3(r.lo)}, ${p3(r.hi)}]${r.note ? `, ${r.note}` : ""}`, { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
    parts.push(g({ "data-fig-set": `manip=${r.id}`, style: "cursor:pointer" }, ...items));
  });

  // Session readout.
  let y = bottom + 50;
  for (const ln of wrap(sessionText(p, lang), w - 2 * lx, 6.9)) { parts.push(text(lx, y, ln, { "font-size": TYPE.body, class: "fig-t-strong" })); y += 17; }

  // Panel B: the discount on the unattended option.
  y += 18;
  for (const ln of wrap(L.panelB, w - 2 * lx, 7)) { parts.push(text(lx, y, ln, { "font-size": TYPE.label, class: "fig-t-strong" })); y += 17; }
  const topB = y + 4;
  const xb = linear([0, 1], [left, right]);
  const rowB = narrow ? 36 : 28;
  const bottomB = topB + 2 * rowB;
  parts.push(axis({ scale: xb, orient: "bottom", at: bottomB, span: [topB - 4, bottomB], title: narrow ? L.axisBShort : L.axisB, count: 5 }));
  const yb1 = narrow ? topB + rowB - 10 : topB + rowB / 2;
  const yb2 = narrow ? topB + 2 * rowB - 10 : topB + rowB + rowB / 2;
  if (narrow) {
    parts.push(text(lx + 2, topB + 12, L.pooled, { "font-size": TYPE.small, class: "fig-t-muted" }));
    parts.push(text(lx + 2, topB + rowB + 12, L.indiv, { "font-size": TYPE.small, class: "fig-t-muted" }));
  } else {
    parts.push(text(lx + labelW, yb1 + 4, L.pooled, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted" }));
    parts.push(text(lx + labelW, yb2 + 4, L.indiv, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted" }));
  }
  parts.push(el("circle", { cx: xb(0.3), cy: yb1, r: 5, fill: C.c2, stroke: C.paper, "stroke-width": 1.4 }));
  parts.push(text(xb(0.3) + 9, yb1 + 4, "0.3", { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
  parts.push(el("line", { x1: xb(0.22), x2: xb(0.82), y1: yb2, y2: yb2, stroke: C.c2, "stroke-width": 3, "stroke-linecap": "round" }));
  parts.push(el("circle", { cx: xb(0.52), cy: yb2, r: 5, fill: C.c2, stroke: C.paper, "stroke-width": 1.4 }));
  parts.push(text(xb(0.82) + 8, yb2 + 4, "0.52 ± 0.3", { "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
  return svg(w, bottomB + 42, describe(st), ...parts);
}

export default defineFigure({
  name: "neuro-attention",
  title: { en: "How much steering attention moves a two-option choice", zh: "引导注意能在多大程度上改变二选一的选择" },
  labels,
  params,
  hint: { en: "Choose a manipulation, or click a row; the slider sets the length of a session.", zh: "选择一种操纵，或点击一行；滑块设定一次会话的长度。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
