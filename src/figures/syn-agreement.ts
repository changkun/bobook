// How consistent human answers are: the agreement and reliability statistics
// cited in the synthesis chapter (en/synthesis/01, @sec-syn-observation), each
// drawn on its own scale between "chance" and "perfect". No value here is
// computed or simulated; every number is stated in the chapter with its source:
//   designers   Krippendorff's alpha 0.25; 20 people with design training,
//               600 pairs of generated interfaces [peng2026efficient, arXiv
//               preprint 2026]
//   chemists    Fleiss' kappa 0.40 and 0.32 in two rounds; 35 chemists
//               [choung2023extracting, Nature Communications 2023]
//   flips       on average 6% to 20% of answers to the same moral pairwise
//               questions flipped within and across sessions
//               [keswani2026moral, AAAI 2026]
//   agents      sighted participants and the simulated agents meant to stand
//               in for them agreed on about 50% of choices
//               [schoinas2025evaluating, EMBC 2025]
//   selfreport  reliability 0.61 for self-reported risk taking, and
//   behavioral  0.25 for behavioral measures such as lottery choices
//               [bagaini2025systematic, Nature Human Behaviour 2025]
// The ends of each scale are reference points, not data: 0 for chance-corrected
// agreement and reliability, a 50% flip rate for a person answering a two-way
// question by coin flip, and 50% agreement for two coin flips between two
// options.

import { defineFigure, tr, type Lang, type State, type Text } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { tpl } from "./lib/format.ts";

const labels = {
  en: {
    describe: "{name}: {stat} {value}, {where} of the way from {left} to {right}. {sample}",
    fromTo: "from {left} to {right}",
    range: "{lo} to {hi}",
    chanceEnd: "chance, or no stable signal",
    perfectEnd: "perfect consistency",
    head: "{name}: {stat}, {value}",
    scale: "Scale {fromTo}; the value sits {where} of the way along it.",
  },
  zh: {
    describe: "{name}：{stat} {value}，位于从 {left}到 {right}的 {where} 处。{sample}",
    fromTo: "从 {left}到 {right}",
    range: "{lo} 至 {hi}",
    chanceEnd: "随机水平，或无稳定信号",
    perfectEnd: "完全一致",
    head: "{name}：{stat}，{value}",
    scale: "量表{fromTo}；该值位于 {where} 处。",
  },
};

type Group = "between" | "within" | "sim" | "measure";
const GROUP_LABEL: Record<Group, Text> = {
  between: { en: "Different people, same pairs", zh: "不同的人，相同的配对" },
  within: { en: "Same person, same question again", zh: "同一个人，再问一次同一个问题" },
  sim: { en: "Simulated stand-in against the person", zh: "模拟替身与真人对比" },
  measure: { en: "Same people measured twice (risk preference)", zh: "同一批人测两次（风险偏好）" },
};
const GROUP_COLOR: Record<Group, string> = { between: C.c7, within: C.c6, sim: C.c5, measure: C.c4 };

interface Row {
  id: string;
  group: Group;
  short: string; // row label
  stat: string; // the statistic, as named in the chapter
  value: string; // the reported value, as text
  tag?: string; // a shorter form for the label on the track
  // Positions in [0, 1] on the stretched scale: 0 is the left (chance) end.
  dots?: number[];
  span?: [number, number];
  left: string; // native label of the left end
  right: string; // native label of the right end
  sample: string; // who, what, and where it was reported
}

// Positions: alpha, kappa, and reliability are already 0 at chance and 1 at
// perfect. A flip rate f sits at (0.5 - f) / 0.5; an agreement rate a between
// two options at (a - 0.5) / 0.5.
const flipPos = (f: number) => (0.5 - f) / 0.5;
const agreePos = (a: number) => (a - 0.5) / 0.5;

const ROWS: Row[] = [
  { id: "designers", group: "between", short: "Designers, generated interfaces", stat: "Krippendorff's α", value: "0.25", dots: [0.25], left: "0 (chance)", right: "1 (perfect)", sample: "20 people with design training judged the same 600 pairs of generated interfaces (Peng and colleagues, a 2026 preprint)." },
  { id: "chemists", group: "between", short: "Chemists, molecules", stat: "Fleiss' κ", value: "0.40 and 0.32 in two rounds", tag: "0.32 and 0.40", dots: [0.40, 0.32], left: "0 (chance)", right: "1 (perfect)", sample: "35 chemists compared pairs of molecules (Choung and colleagues, 2023)." },
  { id: "flips", group: "within", short: "Moral questions, asked again", stat: "share of answers that flipped", value: "6% to 20%", span: [flipPos(0.20), flipPos(0.06)], left: "50% (coin flip)", right: "0% (never flips)", sample: "The same moral pairwise questions were repeated within and across sessions; on average 6% to 20% of answers flipped (Keswani and colleagues, 2026)." },
  { id: "agents", group: "sim", short: "Retinal implant, simulated agents", stat: "agreement on choices", value: "about 50%", dots: [agreePos(0.5)], left: "50% (chance)", right: "100% (always)", sample: "Sighted participants and the simulated agents meant to stand in for them agreed on about 50% of choices (Schoinas and colleagues, 2025)." },
  { id: "selfreport", group: "measure", short: "Self-reported risk taking", stat: "reliability", value: "0.61", dots: [0.61], left: "0 (no stable signal)", right: "1 (perfect)", sample: "A meta-analysis of the stability of risk preference: self-reported propensity to take risks (Bagaïni and colleagues, 2025)." },
  { id: "behavioral", group: "measure", short: "Behavioral choices (lotteries)", stat: "reliability", value: "0.25", dots: [0.25], left: "0 (no stable signal)", right: "1 (perfect)", sample: "The same meta-analysis: behavioral measures, standardized tasks such as choices between lotteries (Bagaïni and colleagues, 2025)." },
];

// The Chinese text of each row, keyed by id; the numbers and positions stay
// in ROWS.
type RowText = Pick<Row, "short" | "stat" | "value" | "tag" | "left" | "right" | "sample">;
const ZH: Record<string, RowText> = {
  designers: { short: "设计者，生成的界面", stat: "Krippendorff α", value: "0.25", left: "0（随机水平）", right: "1（完全一致）", sample: "20 名受过设计训练的人评判了相同的 600 对生成界面（Peng 等，2026 年的预印本）。" },
  chemists: { short: "化学家，分子", stat: "Fleiss κ", value: "两轮分别为 0.40 与 0.32", tag: "0.32 与 0.40", left: "0（随机水平）", right: "1（完全一致）", sample: "35 名化学家比较了成对的分子（Choung 等，2023 年）。" },
  flips: { short: "道德问题，再问一次", stat: "翻转的回答比例", value: "6% 至 20%", left: "50%（掷硬币）", right: "0%（从不翻转）", sample: "相同的道德成对问题在会话内与会话间重复提出；平均有 6% 至 20% 的回答翻转（Keswani 等，2026 年）。" },
  agents: { short: "视网膜植入物，模拟智能体", stat: "选择的一致率", value: "约 50%", left: "50%（随机水平）", right: "100%（总是一致）", sample: "视力正常的被试与本应代替他们的模拟智能体只在约 50% 的选择上一致（Schoinas 等，2025 年）。" },
  selfreport: { short: "自我报告的冒险倾向", stat: "信度", value: "0.61", left: "0（无稳定信号）", right: "1（完全一致）", sample: "一项关于风险偏好稳定性的元分析：自我报告的冒险倾向（Bagaïni 等，2025 年）。" },
  behavioral: { short: "行为选择（彩票）", stat: "信度", value: "0.25", left: "0（无稳定信号）", right: "1（完全一致）", sample: "同一项元分析：行为测量，即彩票选择这类标准化任务（Bagaïni 等，2025 年）。" },
};
const loc = (r: Row, lang: Lang): Row => (lang === "zh" && ZH[r.id] ? { ...r, ...ZH[r.id] } : r);

const params = {
  row: { kind: "choice", label: { en: "Study", zh: "研究" }, options: ROWS.map((r) => ({ value: r.id, label: { en: r.short, zh: ZH[r.id].short } })), default: "designers", control: "select" },
} as const;

type P = { row: string };
const pick = (id: string) => ROWS.find((r) => r.id === id) ?? ROWS[0];

function where(r: Row, lang: Lang): string {
  const lo = r.span ? r.span[0] : Math.min(...r.dots!);
  const hi = r.span ? r.span[1] : Math.max(...r.dots!);
  const q = (v: number) => `${Math.round(v * 100)}%`;
  return lo === hi ? q(lo) : tpl(labels[lang].range, { lo: q(lo), hi: q(hi) });
}

function describe(st: State<P>): string {
  const lang = st.lang ?? "en";
  const r = loc(pick(st.p.row), lang);
  return tpl(labels[lang].describe, { name: r.short, stat: r.stat, value: r.value, where: where(r, lang), left: r.left, right: r.right, sample: r.sample });
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

// Chinese has no spaces to break at: break between characters (and between
// Latin runs), measuring width with labelWidth; never start a line with
// closing punctuation.
function wrapZh(s: string, width: number, latin: number, em: number): string[] {
  const toks = s.match(/[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef“”]|[^\s\u3000-\u303f\u4e00-\u9fff\uff00-\uffef“”]+|\s+/g) ?? [];
  const out: string[] = [];
  let line = "", w = 0;
  for (const t of toks) {
    const tw = labelWidth(t, latin, em);
    if (line && w + tw > width && !/^[，。；：、）？！”]$/.test(t) && !/^\s+$/.test(t)) { out.push(line.trimEnd()); line = ""; w = 0; }
    if (!line && /^\s+$/.test(t)) continue;
    line += t; w += tw;
  }
  if (line.trim()) out.push(line.trimEnd());
  return out;
}

function render(st: State<P>): string {
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 480;
  const sel = loc(pick(st.p.row), lang);
  const parts: string[] = [];
  const lx = narrow ? 8 : 12;
  const labelW = narrow ? 0 : 206;
  const left = lx + labelW + (narrow ? 6 : 14);
  const right = st.w - (narrow ? 14 : 22);
  const x = (v: number) => left + v * (right - left);

  // Header: what the two ends of every track mean.
  let y = 16;
  parts.push(text(left, y, L.chanceEnd, { "font-size": TYPE.small, class: "fig-t-muted" }));
  parts.push(text(right, y, L.perfectEnd, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted" }));
  y += 10;

  const groupH = 22;
  const rowH = narrow ? 66 : 44;
  let lastGroup: Group | undefined;
  for (const r0 of ROWS) {
    const r = loc(r0, lang);
    if (r.group !== lastGroup) {
      y += groupH;
      parts.push(text(lx, y - 6, tr(GROUP_LABEL[r.group], lang), { "font-size": TYPE.small, class: "fig-t-strong" }));
      lastGroup = r.group;
    }
    const isSel = r.id === sel.id;
    const col = GROUP_COLOR[r.group];
    const y0 = y;
    const ty = narrow ? y0 + 40 : y0 + 18; // track baseline
    const items: string[] = [];
    if (isSel) items.push(el("rect", { x: lx - 4, y: y0 + 1, width: st.w - 2 * lx + 8, height: rowH - 2, rx: 4, fill: C.panel, stroke: C.rule }));
    items.push(el("rect", { x: lx - 4, y: y0, width: st.w - 2 * lx + 8, height: rowH, fill: "transparent" }));
    if (narrow) items.push(text(lx + 2, y0 + 13, r.short, { "font-size": TYPE.small, class: isSel ? "fig-t-strong" : "fig-t-muted" }));
    else items.push(text(lx + labelW, ty + 4, r.short, { "font-size": TYPE.small, "text-anchor": "end", class: isSel ? "fig-t-strong" : "fig-t-muted" }));
    // The track and its two native end labels.
    items.push(el("line", { x1: x(0), x2: x(1), y1: ty, y2: ty, stroke: C.grid, "stroke-width": 3, "stroke-linecap": "round" }));
    items.push(el("line", { x1: x(0), x2: x(0), y1: ty - 5, y2: ty + 5, stroke: C.rule }));
    items.push(el("line", { x1: x(1), x2: x(1), y1: ty - 5, y2: ty + 5, stroke: C.rule }));
    items.push(text(x(0), ty + 17, r.left, { "font-size": TYPE.small, class: "fig-t-faint" }));
    items.push(text(x(1), ty + 17, r.right, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-faint" }));
    // The reported value: a bar for a range, dots for single values.
    let labelX = 0;
    if (r.span) {
      const [a, b] = r.span;
      items.push(el("rect", { x: x(a), y: ty - 5, width: Math.max(4, x(b) - x(a)), height: 10, rx: 5, fill: col }));
      labelX = (x(a) + x(b)) / 2;
    }
    for (const v of r.dots ?? []) items.push(el("circle", { cx: x(v), cy: ty, r: 6, fill: col, stroke: C.paper, "stroke-width": 1.5 }));
    if (r.dots) labelX = (x(Math.min(...r.dots)) + x(Math.max(...r.dots))) / 2;
    const anchor = labelX < x(0) + 40 ? "start" : labelX > x(1) - 40 ? "end" : "middle";
    const lxv = anchor === "start" ? x(0) - 4 : labelX;
    items.push(text(lxv, ty - 10, r.tag ?? r.value, { "font-size": TYPE.small, "text-anchor": anchor, class: "fig-t-num" }));
    parts.push(g({ "data-fig-set": `row=${r.id}`, style: "cursor:pointer" }, ...items));
    y += rowH;
  }

  // Detail panel for the chosen row.
  const py = y + 18;
  const pw = st.w - 2 * lx;
  const wr = (s: string, px: number, size: number) => (lang === "zh" ? wrapZh(s, pw - 16, px, size) : wrap(s, pw - 16, px));
  const lines = [
    ...wr(tpl(L.head, { name: sel.short, stat: sel.stat, value: sel.value }), 7.4, TYPE.label).map((t) => ({ t, cls: "fig-t-strong", size: TYPE.label })),
    ...wr(tpl(L.scale, { fromTo: tpl(L.fromTo, { left: sel.left, right: sel.right }), where: where(sel, lang) }), 6.1, TYPE.small).map((t) => ({ t, cls: "fig-t-muted", size: TYPE.small })),
    ...wr(sel.sample, 6.6, TYPE.body).map((t) => ({ t, cls: "", size: TYPE.body })),
  ];
  const ph = 12 + lines.length * 17;
  parts.push(el("rect", { x: lx - 4, y: py - 4, width: pw + 8, height: ph, rx: 6, fill: C.panel, stroke: C.rule }));
  lines.forEach((ln, i) => parts.push(text(lx + 6, py + 13 + i * 17, ln.t, { "font-size": ln.size, class: ln.cls || undefined })));
  return svg(st.w, py + ph + 6, describe(st), ...parts);
}

export default defineFigure({
  name: "syn-agreement",
  title: { en: "How consistent human answers are", zh: "人的回答有多一致" },
  labels,
  params,
  hint: { en: "Click a row, or choose a study, to read what was measured.", zh: "点击一行或选择一项研究，查看测量了什么。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
