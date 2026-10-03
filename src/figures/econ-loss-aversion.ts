// How large is loss aversion? Each row is an estimate of the loss aversion
// coefficient (the weight of a loss relative to an equal gain) quoted in the
// economics chapter, with its 95% interval where the chapter gives one:
//   Tversky and Kahneman 1992: about 2.25 (no interval quoted).
//   Brown et al. 2024, meta-analysis of 607 estimates from 150 articles: mean
//     1.955, 95% interval 1.820 to 2.102.
//   Walasek et al. 2024, cumulative prospect theory fitted to individual
//     choices between mixed gambles: 1.31, 95% CI 1.10 to 1.53; heterogeneity
//     between studies 91.6% of the total variation.
//   Yechiam and Zeif 2025, reanalysis of 163 estimates from 84 papers
//     (n = 149,218): about 2.33 when losses are smaller than gains and items
//     are presented in order of size; about 1.07, not significantly different
//     from 1, when gains and losses are symmetric and unordered (no intervals
//     quoted).
// A vertical line marks a coefficient that a preference likelihood might fix;
// the readout says which quoted intervals contain it. No other data.

import { defineFigure, type Lang, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    axis: "loss aversion coefficient (dashed line at 1: no loss aversion)",
    axisShort: "loss aversion (dashed: none)",
    fixed: "fixed at {l}",
    noCI: "hollow: point estimate, no interval quoted",
    inside: "Inside the 95% interval of: {list}.",
    insideNone: "Inside neither quoted 95% interval.",
    and: " and ",
    describe: "A coefficient fixed at {l} lies {where}. Selected: {name}, {est}.",
    whereNone: "outside both quoted 95% intervals, 1.820 to 2.102 (Brown et al. 2024) and 1.10 to 1.53 (Walasek et al. 2024)",
    whereIn: "inside the 95% interval of {list}",
    sel: "{name}: {est}.",
  },
  zh: {
    axis: "损失厌恶系数（1 处的虚线：没有损失厌恶）",
    axisShort: "损失厌恶系数（虚线：无）",
    fixed: "固定为 {l}",
    noCI: "空心：点估计，未给出区间",
    inside: "落在以下估计的 95% 区间内：{list}。",
    insideNone: "不在给出的任何一个 95% 区间内。",
    and: " 与 ",
    describe: "固定为 {l} 的系数{where}。选中的估计：{name}，{est}。",
    whereNone: "位于给出的两个 95% 区间之外，即 1.820 至 2.102（Brown 等 2024）与 1.10 至 1.53（Walasek 等 2024）",
    whereIn: "位于 {list} 的 95% 区间内",
    sel: "{name}：{est}。",
  },
};

interface Row { id: string; name: string; short: string; est: number; lo?: number; hi?: number; estText: string; detail: string }

// Values as quoted in en/perspectives/04-economics.md (section sec-econ-behavioral).
const ROWS: Row[] = [
  { id: "tk", name: "Tversky and Kahneman 1992", short: "Tversky and Kahneman 1992", est: 2.25, estText: "about 2.25", detail: "The original estimate from cumulative prospect theory." },
  { id: "brown", name: "Brown et al. 2024, meta-analysis", short: "Brown et al. 2024", est: 1.955, lo: 1.82, hi: 2.102, estText: "mean 1.955 (1.820 to 2.102)", detail: "607 estimates from 150 articles." },
  { id: "walasek", name: "Walasek et al. 2024, individual fits", short: "Walasek et al. 2024", est: 1.31, lo: 1.1, hi: 1.53, estText: "1.31 (1.10 to 1.53)", detail: "Cumulative prospect theory fitted to individual choices between mixed gambles; heterogeneity between studies is 91.6% of the total variation." },
  { id: "yzOrd", name: "Yechiam and Zeif 2025, losses smaller, ordered", short: "Yechiam and Zeif, ordered", est: 2.33, estText: "about 2.33", detail: "Losses smaller than gains, items presented in order of size; reanalysis of 163 estimates from 84 papers (n = 149,218)." },
  { id: "yzSym", name: "Yechiam and Zeif 2025, symmetric, unordered", short: "Yechiam and Zeif, symmetric", est: 1.07, estText: "about 1.07, not significantly different from 1", detail: "Gains and losses symmetric and unordered, the condition closest to two designs shown side by side." },
];

// The same rows in the Chinese edition.
const ROWS_ZH: Record<string, Pick<Row, "name" | "short" | "estText" | "detail">> = {
  tk: { name: "Tversky 与 Kahneman 1992", short: "Tversky 与 Kahneman 1992", estText: "约 2.25", detail: "累积前景理论给出的原始估计。" },
  brown: { name: "Brown 等 2024，元分析", short: "Brown 等 2024", estText: "均值 1.955（1.820 至 2.102）", detail: "来自 150 篇文章的 607 个估计。" },
  walasek: { name: "Walasek 等 2024，个体拟合", short: "Walasek 等 2024", estText: "1.31（1.10 至 1.53）", detail: "把累积前景理论拟合到个体在混合赌局之间的选择上；研究间的异质性占总变异的 91.6%。" },
  yzOrd: { name: "Yechiam 与 Zeif 2025，损失较小、有序", short: "Yechiam 与 Zeif，有序", estText: "约 2.33", detail: "损失小于收益，各项按大小顺序呈现；对 84 篇论文中 163 个估计的再分析（n = 149,218）。" },
  yzSym: { name: "Yechiam 与 Zeif 2025，对称、无序", short: "Yechiam 与 Zeif，对称", estText: "约 1.07，与 1 无显著差异", detail: "收益与损失对称且无序，这是最接近并排展示两个设计的情形。" },
};
const rowText = (r: Row, lang: Lang) => (lang === "zh" ? ROWS_ZH[r.id] : r);

const params = {
  lambda: { kind: "range", label: { en: "Coefficient fixed in the likelihood", zh: "似然中固定的系数" }, min: 0.8, max: 2.6, default: 2.25, step: 0.01 },
  row: { kind: "choice", label: { en: "Estimate", zh: "估计" }, options: ROWS.map((r) => ({ value: r.id, label: { en: r.short, zh: ROWS_ZH[r.id].short } })), default: "brown", control: "select" },
} as const;

type P = { lambda: number; row: string };

const pick = (id: string) => ROWS.find((r) => r.id === id) ?? ROWS[1];
const contains = (r: Row, l: number) => r.lo !== undefined && r.hi !== undefined && l >= r.lo - 1e-9 && l <= r.hi + 1e-9;

function insideList(l: number, lang: Lang = "en"): string[] {
  return ROWS.filter((r) => contains(r, l)).map((r) => rowText(r, lang).short);
}

function describe(st: State<P>): string {
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const l = st.p.lambda;
  const ins = insideList(l, lang);
  const where = ins.length ? tpl(L.whereIn, { list: ins.join(L.and) }) : L.whereNone;
  const r = rowText(pick(st.p.row), lang);
  return tpl(L.describe, { l: fixed(l, 2), where, name: r.name, est: r.estText });
}

function wrap(s: string, width: number, px = 6.4): string[] {
  const max = Math.max(12, Math.floor(width / px));
  const out: string[] = [];
  let line = "";
  for (const w of s.split(" ")) {
    if (line && (line + " " + w).length > max) { out.push(line); line = w; } else line = line ? `${line} ${w}` : w;
  }
  if (line) out.push(line);
  return out;
}

// Chinese has no spaces between words: wrap by estimated width instead, with
// a Chinese character a full em of `em` pixels, a run of Latin letters and
// digits kept whole, and no line starting with closing punctuation.
function wrapZh(s: string, width: number, px = 6.4, em = 12): string[] {
  const toks = s.match(/[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef“”‘’]|[^\s\u3000-\u303f\u4e00-\u9fff\uff00-\uffef“”‘’]+|\s+/g) ?? [];
  const lines: string[] = [];
  let line = "";
  for (const t of toks) {
    if (line && labelWidth((line + t).trimEnd(), px, em) > width && !/^[，。；：、）？！”]$/.test(t) && !/^\s+$/.test(t)) { lines.push(line.trimEnd()); line = ""; }
    if (!line && /^\s+$/.test(t)) continue;
    line += t;
  }
  if (line.trim()) lines.push(line.trimEnd());
  return lines;
}

function render(st: State<P>): string {
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const wr = (s: string, width: number, px: number, em: number) => (lang === "zh" ? wrapZh(s, width, px, em) : wrap(s, width, px));
  const p = st.p, w = st.w, narrow = w < 480;
  const sel = pick(p.row);
  const selT = rowText(sel, lang);
  const parts: string[] = [];
  const lx = narrow ? 8 : 12;
  const labelW = narrow ? 0 : 200;
  const left = lx + labelW + (narrow ? 6 : 12);
  const right = w - 14;
  const top = 26;
  const rowH = narrow ? 36 : 28;
  const x = linear([0.8, 2.6], [left, right]);
  const bottom = top + rowH * ROWS.length;

  // Fixed coefficient label above the plot.
  const lxp = x(p.lambda);
  const anchor = lxp > right - 50 ? "end" : lxp < left + 50 ? "start" : "middle";
  parts.push(text(lxp, 14, tpl(L.fixed, { l: fixed(p.lambda, 2) }), { "font-size": TYPE.small, "text-anchor": anchor, class: "fig-t-strong", style: `fill:${C.acq}` }));

  parts.push(axis({ scale: x, orient: "bottom", at: bottom, span: [top - 4, bottom], title: narrow ? L.axisShort : L.axis, count: narrow ? 4 : 9 }));
  // No loss aversion at 1.
  parts.push(el("line", { x1: x(1), x2: x(1), y1: top - 4, y2: bottom, stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "4 3" }));

  ROWS.forEach((r, i) => {
    const y0 = top + i * rowH;
    const yl = narrow ? y0 + rowH - 10 : y0 + rowH / 2;
    const isSel = r.id === sel.id;
    const inside = contains(r, p.lambda);
    const col = r.lo === undefined ? C.ink2 : inside ? C.c1 : C.ink2;
    const items: string[] = [];
    if (isSel) items.push(el("rect", { x: lx - 4, y: y0 + 1, width: w - 2 * lx + 8, height: rowH - 2, rx: 4, fill: C.panel, stroke: C.rule }));
    items.push(el("rect", { x: lx - 4, y: y0, width: w - 2 * lx + 8, height: rowH, fill: "transparent" }));
    if (narrow) items.push(text(lx + 2, y0 + 12, rowText(r, lang).short, { "font-size": TYPE.small, class: (isSel ? "fig-t-strong" : "fig-t-muted") + " fig-t-halo" }));
    else items.push(text(lx + labelW, yl + 4, rowText(r, lang).short, { "font-size": TYPE.small, "text-anchor": "end", class: isSel ? "fig-t-strong" : "fig-t-muted" }));
    if (r.lo !== undefined && r.hi !== undefined) {
      items.push(el("line", { x1: x(r.lo), x2: x(r.hi), y1: yl, y2: yl, stroke: col, "stroke-width": 3, "stroke-linecap": "round" }));
      items.push(el("circle", { cx: x(r.est), cy: yl, r: 5, fill: col, stroke: C.paper, "stroke-width": 1.4 }));
    } else {
      items.push(el("circle", { cx: x(r.est), cy: yl, r: 4.6, fill: C.paper, stroke: col, "stroke-width": 2 }));
    }
    parts.push(g({ "data-fig-set": `row=${r.id}`, style: "cursor:pointer" }, ...items));
  });

  // The fixed coefficient, drawn over the rows.
  parts.push(el("line", { x1: lxp, x2: lxp, y1: 20, y2: bottom, stroke: C.acq, "stroke-width": 2 }));
  // Clicks on the plot move the line.
  parts.push(el("rect", { x: left, y: top - 4, width: right - left, height: bottom - top + 4, fill: "transparent", "data-fig-hit": "plot", "data-fig-map": [left, right, 0.8, 2.6, bottom, top, 0, 1].join(",") }));

  // Readout and the selected estimate.
  const ins = insideList(p.lambda, lang);
  const pw = w - 2 * lx;
  // Legend for the hollow markers.
  parts.push(el("circle", { cx: lx + 6, cy: bottom + 46, r: 4.2, fill: C.paper, stroke: C.ink2, "stroke-width": 1.8 }));
  parts.push(text(lx + 16, bottom + 50, L.noCI, { "font-size": TYPE.small, class: "fig-t-muted" }));
  let y = bottom + 76;
  const read = ins.length ? tpl(L.inside, { list: ins.join(L.and) }) : L.insideNone;
  for (const ln of wr(read, pw, 6.9, TYPE.label)) { parts.push(text(lx, y, ln, { "font-size": TYPE.label, class: "fig-t-strong" })); y += 18; }
  y += 6;
  const lines = [
    ...wr(tpl(L.sel, { name: selT.name, est: selT.estText }), pw - 16, 7, TYPE.body).map((t) => ({ t, cls: "fig-t-strong" })),
    ...wr(selT.detail, pw - 16, 6.6, TYPE.body).map((t) => ({ t, cls: "" })),
  ];
  const ph = 12 + lines.length * 17;
  parts.push(el("rect", { x: lx - 4, y: y - 4, width: pw + 8, height: ph, rx: 6, fill: C.panel, stroke: C.rule }));
  lines.forEach((ln, i) => parts.push(text(lx + 6, y + 13 + i * 17, ln.t, { "font-size": TYPE.body, class: ln.cls || undefined })));
  return svg(w, y - 4 + ph + 6, describe(st), ...parts);
}

export default defineFigure({
  name: "econ-loss-aversion",
  title: { en: "Estimates of the loss aversion coefficient, against a coefficient fixed in a likelihood", zh: "损失厌恶系数的各项估计，与似然中固定的系数对照" },
  labels,
  params,
  hint: { en: "Drag the slider or click the plot to move the fixed coefficient; click a row to read its source.", zh: "拖动滑块或点击图中任意位置来移动固定的系数；点击一行查看它的来源。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.target !== "plot" || !e.data) return null;
    const l = Math.round(Math.min(2.6, Math.max(0.8, e.data.x)) * 100) / 100;
    return { ...p, lambda: l };
  },
  snapshots: {
    "between the pooled intervals": (p) => ({ ...p, lambda: 1.7 }),
    "at 1": (p) => ({ ...p, lambda: 1, row: "yzSym" }),
  },
});
