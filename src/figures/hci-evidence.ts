// An evidence map of the HCI studies of interactive Bayesian optimization:
// the number of participants (log scale) against the strength of the
// comparison condition, colored by the kind of feedback the person gave, with
// the marker showing what the comparison found. Choosing a study (from the
// control or by clicking its marker) shows its comparison and result. The
// strength grading follows the chapter's rule and is the chapter's judgment;
// the data are the numbers reported in the chapter's tables.

import { defineFigure, type Lang, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { log } from "./lib/scale.ts";
import { tpl } from "./lib/format.ts";

const labels = {
  en: {
    n: "participants (log scale)",
    none: "none",
    weak: "weak",
    medium: "medium",
    strong: "strong",
    strength: "strength of comparison",
    grading: "The strength of each comparison is this chapter's grading (inference), not the authors'.",
    better: "better outcome",
    cheaper: "same outcome, lower cost",
    same: "no difference",
    nocomp: "no comparison",
    preference: "preference",
    rating: "rating",
    performance: "performance",
    implicit: "implicit (edits)",
    vs: "compared with: {c} (graded {s} by this chapter)",
    meta: "{venue} {year} · N = {n} · {fb} feedback",
    describe: "Evidence map of {k} studies. Selected: {name}, {venue} {year}, N = {n}, {fb} feedback, compared with {c} (graded {s} by this chapter): {r}.",
    ringed: " (ringed on the map)",
  },
  zh: {
    n: "被试人数（对数刻度）",
    none: "无",
    weak: "弱",
    medium: "中",
    strong: "强",
    strength: "对照条件的强度",
    grading: "每项对照的强度是本章的分级（推断），不是作者的分级。",
    better: "结果更好",
    cheaper: "结果相同，成本更低",
    same: "无差异",
    nocomp: "无对照",
    preference: "偏好",
    rating: "评分",
    performance: "性能",
    implicit: "隐式（编辑）",
    vs: "对照条件：{c}（本章分级：{s}）",
    meta: "{venue} {year} · N = {n} · {fb}反馈",
    describe: "{k} 项研究的证据图。已选：{name}，{venue} {year}，N = {n}，{fb}反馈，对照条件为{c}（本章分级：{s}）：{r}。",
    ringed: "（图中带圈者）",
  },
};

type Fb = "preference" | "rating" | "performance" | "implicit";
type Strength = "none" | "weak" | "medium" | "strong";
type Res = "better" | "cheaper" | "same" | "nocomp";

interface Study { id: string; name: string; venue: string; year: number; n: number; nNote?: string; fb: Fb; s: Strength; sNote?: string; res: Res; vs: string; r: string }

// Numbers as reported in the chapter's tables; see the chapter for sources.
const STUDIES: Study[] = [
  { id: "dudley", name: "Dudley et al.", venue: "CHI", year: 2019, n: 200, nNote: "200 crowd workers in experiment 1", fb: "performance", s: "medium", res: "better", vs: "baseline interface", r: "no difference in batch 1 (p = 0.88); from batch 2, median task time fell from about 34 s to about 21 s" },
  { id: "gallery", name: "Sequential Gallery", venue: "SIGGRAPH", year: 2020, n: 6, fb: "preference", s: "none", res: "nocomp", vs: "no control condition with users (line search only in simulation)", r: "satisfied after 5.36 iterations on average; plane search beat line search in simulation" },
  { id: "zhou", name: "Zhou et al.", venue: "IUI", year: 2021, n: 12, fb: "preference", s: "strong", res: "better", vs: "automatic expected improvement", r: "Creativity Support Index total and several factors higher; 11 of 12 preferred balancing by hand" },
  { id: "chan", name: "Chan et al.", venue: "CHI", year: 2022, n: 40, fb: "performance", s: "medium", res: "better", vs: "designer-led design", r: "better spatial error; lower agency, ownership, and expressiveness" },
  { id: "ou22", name: "Ou et al.", venue: "Mensch und Computer", year: 2022, n: 22, nNote: "2 artists in the field, 20 in the lab", fb: "rating", s: "none", res: "nocomp", vs: "none (observational)", r: "415 of 549 field sequences stopped at iteration 1; satisfied 16 of 134 (field), 97 of 200 (lab)" },
  { id: "shen", name: "Shen et al.", venue: "ISMAR", year: 2022, n: 12, fb: "performance", s: "medium", res: "better", vs: "baseline keyboard", r: "speed +14.4%, accuracy +13.8%, learning effects ruled out" },
  { id: "ou23", name: "Ou et al.", venue: "IUI", year: 2023, n: 60, fb: "preference", s: "none", res: "nocomp", vs: "levels of expertise, no system comparison", r: "novices reach expert-level quality; experts iterate more and are less satisfied" },
  { id: "liao23", name: "Liao et al.", venue: "IEEE Pervasive Computing", year: 2023, n: 8, fb: "performance", s: "medium", res: "cheaper", vs: "designers' own strategies", r: "very similar design performance; lower perceived workload" },
  { id: "liao24", name: "Liao et al.", venue: "CHI", year: 2024, n: 11, nNote: "11 for evaluation, after 14 for the model", fb: "performance", s: "strong", res: "better", vs: "standard Bayesian optimization; manual calibration", r: "absolute pointing improved 22.92% over standard Bayesian optimization and 21.35% over manual calibration" },
  { id: "mo", name: "Mo et al.", venue: "ACM TiiS", year: 2024, n: 18, fb: "performance", s: "strong", res: "cheaper", vs: "optimizer-led design", r: "no difference in hypervolume with fewer formal evaluations, but fewer Pareto designs" },
  { id: "fontcraft", name: "FontCraft", venue: "CHI", year: 2025, n: 10, fb: "preference", s: "weak", res: "better", vs: "single-slider Bayesian optimization", r: "closer to target fonts and more consistent styles; no inferential statistics seen" },
  { id: "cpbo", name: "Constrained PBO", venue: "IJCAI", year: 2025, n: 11, fb: "preference", s: "none", res: "nocomp", vs: "no control system", r: "4.2 s per choice; positive attitudes among professional ad designers" },
  { id: "opticarvis", name: "OptiCarVis", venue: "CHI", year: 2025, n: 117, fb: "rating", s: "strong", sNote: "weak to strong", res: "better", vs: "no visualization; user-customized design; expert design", r: "personalized designs rated better on perceived safety, predictability, and trust" },
  { id: "metapo", name: "Meta-PO", venue: "UIST", year: 2025, n: 36, fb: "preference", s: "strong", res: "cheaper", vs: "the same method without transfer", r: "iterations to satisfaction fell from 9.54 to 5.86 (same theme) and 7.41 (other theme)" },
  { id: "niwa", name: "Niwa et al., study 2", venue: "UIST", year: 2025, n: 12, fb: "performance", s: "strong", res: "cheaper", vs: "cooperation through explicit constraints", r: "same performance (p = 0.204); lower task load, but less agency" },
  { id: "song", name: "Song et al.", venue: "UIST", year: 2025, n: 12, fb: "implicit", s: "medium", sNote: "medium to strong", res: "better", vs: "manual adaptation; ParetoSelect", r: "fewer elements moved, and layouts ranked above ParetoSelect's; no difference in hypervolume, overall quality, or experience" },
  { id: "rombo", name: "ROMBO", venue: "PLOS ONE", year: 2025, n: 16, fb: "rating", s: "weak", res: "better", vs: "random queries", r: "new favorite tracks found 40% more often and 16% faster; 18% less time on disliked tracks" },
  { id: "gimmbo", name: "GimmBO", venue: "SIGGRAPH", year: 2026, n: 12, fb: "preference", s: "medium", res: "better", vs: "sliders; Sequential Gallery", r: "ranking better on similarity and success rate, but 50.5 s against 34.7 s per step" },
  { id: "appo", name: "APPO", venue: "CHI", year: 2026, n: 16, fb: "preference", s: "strong", res: "cheaper", vs: "PromptCharm; DSPy; a clarifying-question baseline", r: "satisfied in fewer than 4 iterations against more than 6; less expressive" },
  { id: "cost", name: "Cost-aware BO", venue: "CHI", year: 2026, n: 12, fb: "performance", s: "strong", res: "cheaper", vs: "Bayesian optimization ignoring cost", r: "same performance at about 67% of the cost; no difference in final quality (p = .77)" },
  { id: "homi", name: "HOMI", venue: "CHI", year: 2026, n: 12, fb: "performance", s: "strong", res: "cheaper", vs: "transfer acquisition function; continual Bayesian optimization", r: "better only at iterations 2 and 3; all methods equal from iteration 6" },
  { id: "tanaka", name: "Tanaka et al.", venue: "CAADRIA", year: 2026, n: 40, fb: "rating", s: "medium", res: "better", vs: "sliders", r: "62.5% preferred the optimized results (p = 0.025); 85% endorsed their diversity" },
  { id: "multi", name: "Multi-session study", venue: "AutomotiveUI", year: 2026, n: 74, fb: "rating", s: "medium", sNote: "medium to strong", res: "better", vs: "the day-one design kept unchanged on days two and three", r: "continued optimization rated better on cognitive load, trust, predictability, and perceived safety; authors also report shortcomings for subjective measures" },
];

// The Chinese text of each study: name, notes, comparison, and result.
const ZH: Record<string, { name: string; nNote?: string; sNote?: string; vs: string; r: string }> = {
  dudley: { name: "Dudley 等", nNote: "实验 1 中 200 名众包工作者", vs: "基线界面", r: "第 1 批无差异（p = 0.88）；从第 2 批起，任务用时中位数从约 34 秒降到约 21 秒" },
  gallery: { name: "序列画廊", vs: "没有用户参与的对照条件（线搜索只在模拟中）", r: "平均 5.36 次迭代后满意；模拟中平面搜索胜过线搜索" },
  zhou: { name: "Zhou 等", vs: "自动期望改进", r: "创造力支持指数总分及若干因素更高；12 人中 11 人偏好手动平衡" },
  chan: { name: "Chan 等", vs: "设计者主导设计", r: "空间误差更好；能动感、归属感和表达力更低" },
  ou22: { name: "Ou 等", nNote: "现场 2 名美术师，实验室 20 名", vs: "无（观察性）", r: "549 个现场序列中有 415 个停在第 1 次迭代；以满意结束的有 134 个中的 16 个（现场）、200 个中的 97 个（实验室）" },
  shen: { name: "Shen 等", vs: "基线键盘", r: "速度 +14.4%，准确率 +13.8%，排除了学习效应" },
  ou23: { name: "Ou 等", vs: "不同专业水平，没有系统之间的比较", r: "新手达到专家级质量；专家迭代更多，满意度更低" },
  liao23: { name: "Liao 等", vs: "设计师自己的策略", r: "设计性能非常相似；感知工作负荷更低" },
  liao24: { name: "Liao 等", nNote: "11 名用于评估，此前 14 名用于建模", vs: "标准贝叶斯优化；手动校准", r: "绝对指点比标准贝叶斯优化改善 22.92%，比手动校准改善 21.35%" },
  mo: { name: "Mo 等", vs: "优化器主导设计", r: "超体积无差异，正式评估更少，但帕累托设计也更少" },
  fontcraft: { name: "FontCraft", vs: "单滑块贝叶斯优化", r: "更接近目标字体，风格更一致；未见推断统计" },
  cpbo: { name: "约束偏好贝叶斯优化", vs: "无对照系统", r: "每次选择 4.2 秒；专业广告设计师态度积极" },
  opticarvis: { name: "OptiCarVis", sNote: "弱至强", vs: "无可视化；用户自定义设计；专家设计", r: "个性化设计在感知安全、可预测性和信任上评分更好" },
  metapo: { name: "Meta-PO", vs: "不迁移的同一方法", r: "达到满意的迭代次数从 9.54 降到 5.86（同一主题）和 7.41（其他主题）" },
  niwa: { name: "Niwa 等，研究 2", vs: "通过显式约束协作", r: "性能相同（p = 0.204）；任务负荷更低，但能动感也更少" },
  song: { name: "Song 等", sNote: "中至强", vs: "手动调整；ParetoSelect", r: "移动的元素更少，布局排名高于 ParetoSelect 的；超体积、整体质量或体验无差异" },
  rombo: { name: "ROMBO", vs: "随机查询", r: "找到新的喜爱曲目的频率高 40%、速度快 16%；花在不喜欢的曲目上的时间少 18%" },
  gimmbo: { name: "GimmBO", vs: "滑块；序列画廊", r: "排序在相似度和成功率上更好，但每步 50.5 秒对 34.7 秒" },
  appo: { name: "APPO", vs: "PromptCharm；DSPy；一个澄清式提问基线", r: "不到 4 次迭代即满意，对照需要 6 次以上；表达力较低" },
  cost: { name: "成本感知贝叶斯优化", vs: "忽略成本的贝叶斯优化", r: "以约 67% 的成本达到相同性能；最终质量无差异（p = .77）" },
  homi: { name: "HOMI", vs: "迁移采集函数；持续贝叶斯优化", r: "只在第 2 和第 3 次迭代更好；从第 6 次迭代起所有方法相同" },
  tanaka: { name: "Tanaka 等", vs: "滑块", r: "62.5% 偏好优化结果（p = 0.025）；85% 认可其多样性" },
  multi: { name: "多会话研究", sNote: "中至强", vs: "第二、三天保持第一天的设计不变", r: "持续优化在认知负荷、信任、可预测性和感知安全上评分更好；作者也报告了主观测量方面的不足" },
};

// A study's text in the edition's language.
function textOf(s: Study, lang: Lang) {
  if (lang !== "zh") return { name: s.name, nNote: s.nNote, vs: s.vs, r: s.r, fb: s.fb as string, s: s.sNote ?? s.s };
  const z = ZH[s.id], L = labels.zh;
  return { name: z.name, nNote: z.nNote, vs: z.vs, r: z.r, fb: L[s.fb], s: z.sNote ?? L[s.s] };
}

const FB_COLOR: Record<Fb, string> = { preference: C.c1, rating: C.c2, performance: C.c6, implicit: C.c7 };
const ROWS: Strength[] = ["none", "weak", "medium", "strong"];

const FILTERS = [
  { value: "all", label: { en: "All", zh: "全部" } },
  { value: "preference", label: { en: "Preference", zh: "偏好" } },
  { value: "rating", label: { en: "Rating", zh: "评分" } },
  { value: "performance", label: { en: "Performance", zh: "性能" } },
] as const;

const params = {
  study: { kind: "choice", label: { en: "Study", zh: "研究" }, options: STUDIES.map((s) => ({ value: s.id, label: { en: `${s.name} ${s.year}`, zh: `${ZH[s.id].name} ${s.year}` } })), default: "chan", control: "select" },
  only: { kind: "choice", label: { en: "Feedback", zh: "反馈" }, options: FILTERS, default: "all" },
} as const;

type P = { study: string; only: string };

const pick = (id: string) => STUDIES.find((s) => s.id === id) ?? STUDIES[0];

function describe(st: State<P>): string {
  const s = pick(st.p.study);
  const lang = st.lang ?? "en";
  const t = textOf(s, lang);
  return tpl(labels[lang].describe, { k: STUDIES.length, name: t.name, venue: s.venue, year: s.year, n: s.n, fb: t.fb, c: t.vs, s: t.s, r: t.r });
}

function marker(kind: Res, x: number, y: number, color: string, r: number, a: Record<string, string | number> = {}): string {
  if (kind === "better") return el("circle", { cx: x, cy: y, r, fill: color, stroke: C.paper, "stroke-width": 1.2, ...a });
  if (kind === "cheaper") return el("path", { d: `M${x},${y - r * 1.25}L${x + r * 1.25},${y}L${x},${y + r * 1.25}L${x - r * 1.25},${y}Z`, fill: color, stroke: C.paper, "stroke-width": 1.2, ...a });
  if (kind === "same") return el("circle", { cx: x, cy: y, r: r - 0.6, fill: C.paper, stroke: color, "stroke-width": 2.2, ...a });
  return el("rect", { x: x - r + 0.8, y: y - r + 0.8, width: 2 * r - 1.6, height: 2 * r - 1.6, fill: C.paper, stroke: color, "stroke-width": 2, ...a });
}

// Greedy word wrap by an approximate character width.
function wrap(s: string, width: number, px = 6.1): string[] {
  const max = Math.max(12, Math.floor(width / px));
  const out: string[] = [];
  let line = "";
  for (const w of s.split(" ")) {
    if (line && (line + " " + w).length > max) { out.push(line); line = w; } else line = line ? `${line} ${w}` : w;
  }
  if (line) out.push(line);
  return out;
}

// Wrap Chinese text, which has no spaces between words: a Chinese character
// or full-width mark counts `em` pixels, a run of Latin letters, digits, and
// symbols `px` per character and is kept whole, and a line never starts with
// closing punctuation.
function wrapZh(s: string, width: number, px: number, em: number): string[] {
  const toks = s.match(/[　-〿一-鿿＀-￯“”‘’]|[^\s　-〿一-鿿＀-￯“”‘’]+|\s+/g) ?? [];
  const lines: string[] = [];
  let line = "", w = 0;
  for (const t of toks) {
    const tw = labelWidth(t, px, em);
    if (line && w + tw > width && !/^[，。；：、）？！”]$/.test(t) && !/^\s+$/.test(t)) { lines.push(line.trimEnd()); line = ""; w = 0; }
    if (!line && /^\s+$/.test(t)) continue;
    line += t; w += tw;
  }
  if (line.trim()) lines.push(line.trimEnd());
  return lines;
}

const wrapIn = (lang: Lang, s: string, width: number, px: number, em: number) => (lang === "zh" ? wrapZh(s, width, px, em) : wrap(s, width, px));

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 480;
  const sel = pick(p.study);
  const tx = textOf(sel, lang);
  const parts: string[] = [];

  // Legend: feedback colors, then result markers.
  const lx = narrow ? 8 : 12;
  let lxx = lx, ly = 14;
  const legendRow = (items: Array<{ sw: (cx: number, cy: number) => string; label: string }>) => {
    for (const it of items) {
      const wpx = labelWidth(it.label, 6.2) + 28;
      if (lxx + wpx > st.w - 8 && lxx > lx) { lxx = lx; ly += 18; }
      parts.push(it.sw(lxx + 6, ly - 4), text(lxx + 15, ly, it.label, { "font-size": TYPE.small, class: "fig-t-muted" }));
      lxx += wpx;
    }
    lxx = lx; ly += 18;
  };
  legendRow((["preference", "rating", "performance", "implicit"] as Fb[]).map((fb) => ({ sw: (cx, cy) => el("circle", { cx, cy, r: 5, fill: FB_COLOR[fb] }), label: L[fb] })));
  legendRow((["better", "cheaper", "same", "nocomp"] as Res[]).filter((res) => STUDIES.some((s) => s.res === res)).map((res) => ({ sw: (cx, cy) => marker(res, cx, cy, C.ink2, 5), label: L[res] })));
  ly -= 18;

  // The map.
  const left = narrow ? 78 : 104;
  const right = st.w - 14;
  const top = ly + 18;
  const rowH = narrow ? 50 : 58;
  const bottom = top + rowH * ROWS.length;
  const x = log([5, 300], [left + 8, right - 8]);
  const rowY = (s: Strength) => bottom - rowH * (ROWS.indexOf(s) + 0.5);
  ROWS.forEach((s, i) => {
    const y0 = bottom - rowH * (i + 1);
    if (i % 2 === 1) parts.push(el("rect", { x: left, y: y0, width: right - left, height: rowH, fill: C.panel }));
    parts.push(text(left - 8, y0 + rowH / 2 + 4, L[s], { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted" }));
  });
  parts.push(text(0, 0, L.strength, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-faint", transform: `translate(${narrow ? 10 : 14},${(top + bottom) / 2}) rotate(-90)` }));
  for (const t of [5, 10, 20, 50, 100, 200]) {
    parts.push(el("line", { x1: x(t), x2: x(t), y1: top, y2: bottom, stroke: C.grid }),
      text(x(t), bottom + 15, String(t), { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted fig-t-num" }));
  }
  parts.push(el("line", { x1: left, x2: right, y1: bottom, y2: bottom, stroke: C.rule }),
    text((left + right) / 2, bottom + 31, L.n, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted" }));
  // Say on the figure itself that the grading is the chapter's, not the sources'.
  const note = wrapIn(lang, L.grading, st.w - 2 * lx, 6.1, 11);
  note.forEach((t, i) => parts.push(text(lx, bottom + 49 + i * 15, t, { "font-size": TYPE.small, class: "fig-t-faint" })));

  // Place markers in lanes within each row so neighbors do not overlap.
  const LANES = narrow ? [0, -11, 11, -22, 22] : [0, -13, 13, -26, 26];
  const placed: Array<{ s: Study; cx: number; cy: number }> = [];
  for (const row of ROWS) {
    const lastX = LANES.map(() => -Infinity);
    for (const s of STUDIES.filter((q) => q.s === row).sort((a, b) => a.n - b.n || a.year - b.year)) {
      const cx = x(s.n);
      let lane = LANES.findIndex((_, i) => cx - lastX[i] >= (narrow ? 11 : 13));
      if (lane < 0) lane = 0;
      lastX[lane] = cx;
      placed.push({ s, cx, cy: rowY(row) + LANES[lane] });
    }
  }
  for (const { s, cx, cy } of placed) {
    const dim = p.only !== "all" && s.fb !== p.only;
    const isSel = s.id === sel.id;
    parts.push(g({ "data-fig-set": `study=${s.id}`, opacity: dim ? 0.18 : 1, style: "cursor:pointer" },
      el("circle", { cx, cy, r: 10, fill: "transparent" }),
      marker(s.res, cx, cy, FB_COLOR[s.fb], isSel ? 6.5 : 5.2),
      isSel ? el("circle", { cx, cy, r: 10.5, fill: "none", stroke: C.ink, "stroke-width": 1.6 }) : "",
    ));
  }

  // Detail panel for the selected study.
  const py = bottom + 50 + note.length * 15 + 4;
  const pw = st.w - 2 * lx;
  const lines = [
    { t: `${tx.name}${L.ringed}`, cls: "fig-t-strong", size: TYPE.label },
    { t: tpl(L.meta, { venue: sel.venue, year: sel.year, n: sel.n, fb: tx.fb }) + (tx.nNote ? (lang === "zh" ? `（${tx.nNote}）` : ` (${tx.nNote})`) : ""), cls: "fig-t-muted", size: TYPE.small },
    ...wrapIn(lang, tpl(L.vs, { c: tx.vs, s: tx.s }), pw - 16, 6.1, 11).map((t) => ({ t, cls: "fig-t-muted", size: TYPE.small })),
    ...wrapIn(lang, tx.r, pw - 16, 6.5, 12).map((t) => ({ t, cls: "", size: TYPE.body })),
  ];
  const ph = 12 + lines.length * 17;
  parts.push(el("rect", { x: lx - 4, y: py - 4, width: pw + 8, height: ph, rx: 6, fill: C.panel, stroke: C.rule }));
  lines.forEach((ln, i) => parts.push(text(lx + 6, py + 13 + i * 17, ln.t, { "font-size": ln.size, class: ln.cls || undefined })));

  return svg(st.w, py + ph + 4, describe(st), ...parts);
}

export default defineFigure({
  name: "hci-evidence",
  title: { en: "Evidence map of interactive design studies", zh: "交互式设计研究的证据图" },
  labels,
  params,
  hint: { en: "Click a marker, or choose a study, to read its comparison and result.", zh: "点击标记或选择一项研究，查看它的对照与结果。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
