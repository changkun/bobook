// The human evidence for preferential optimization, by application domain.
// Each row is a domain from the chapter's table: a thin line spans the number
// of real people per study, a thick segment the typical range, and dots mark
// single studies. Color says where the judgments came from. Choosing a domain
// shows its study count and main limitation. All values come from the table
// and are approximate where the table says so.

import { defineFigure, type Lang, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { log } from "./lib/scale.ts";
import { tpl } from "./lib/format.ts";

const labels = {
  en: {
    axis: "real people per study (log scale)",
    none: "no real people",
    describe: "{name}: {studies}; typical sample {typical}; {source}. Main limitation: {limit}.",
    missing: "No study found:",
    typicalLine: "Typical sample: {t}.",
    limitLine: "Main limitation: {t}.",
    missingLine: "{m} {list}.",
  },
  zh: {
    axis: "每项研究的真人人数（对数刻度）",
    none: "没有真人",
    describe: "{name}：{studies}；典型样本 {typical}；{source}。主要局限：{limit}。",
    missing: "未找到研究的领域：",
    typicalLine: "典型样本：{t}。",
    limitLine: "主要局限：{t}。",
    missingLine: "{m}{list}。",
  },
};

type Source = "real" | "mixed" | "sim" | "nohuman";
const SOURCE_LABEL: Record<Source, string> = { real: "real people", mixed: "real and simulated", sim: "mostly simulated", nohuman: "no human preferences" };
const SOURCE_LABEL_ZH: Record<Source, string> = { real: "真人", mixed: "真人与模拟", sim: "主要为模拟", nohuman: "没有人的偏好" };
const sourceLabel = (s: Source, lang: Lang) => (lang === "zh" ? SOURCE_LABEL_ZH : SOURCE_LABEL)[s];
const SOURCE_COLOR: Record<Source, string> = { real: C.c1, mixed: C.c7, sim: C.c2, nohuman: C.ink3 };

interface Domain {
  id: string; name: string; short: string; source: Source; studies: string; typical: string; limit: string;
  lo?: number; hi?: number; tLo?: number; tHi?: number; dots?: Array<{ v: number; hollow?: boolean }>; sec: string;
}

// From the chapter's table of the evidence by domain.
const DOMAINS: Domain[] = [
  { id: "design", name: "Visual, interface, and generative design", short: "Visual and generative design", source: "real", studies: "about 24 studies", typical: "mostly 10 to 40 (3 to 60, excluding crowdsourcing)", limit: "mostly weak comparisons; mostly novices; no test of drift within a session", lo: 3, hi: 60, tLo: 10, tHi: 40, sec: "design" },
  { id: "rating", name: "Rating or performance-based optimization with people", short: "Rating or performance HITL", source: "real", studies: "about 19 studies", typical: "12 to 40 (8 to 200)", limit: "not preference feedback; against strong comparisons final quality often no different", lo: 8, hi: 200, tLo: 12, tHi: 40, sec: "design" },
  { id: "exo", name: "Lower-limb exoskeletons", short: "Exoskeletons", source: "real", studies: "13 studies plus 2 self-tuning baselines", typical: "2 to 15, median about 5", limit: "only 2 participants with paraplegia; internal validation; almost no comparison with manual tuning", lo: 2, hi: 15, dots: [{ v: 5 }], sec: "health" },
  { id: "prosth", name: "Prostheses", short: "Prostheses", source: "real", studies: "1 PBO study (plus 3 related)", typical: "2 to 3 amputees", limit: "tiny samples; preferences inconsistent across trials; no clear biomechanical difference", lo: 2, hi: 3, tLo: 2, tHi: 3, sec: "health" },
  { id: "audio", name: "Hearing aids and audio", short: "Hearing aids and audio", source: "mixed", studies: "5 studies (plus 3 related)", typical: "20 to 35", limit: "no improvement in speech clarity; evaluations mostly involve the manufacturer", lo: 20, hi: 35, tLo: 20, tHi: 35, sec: "health" },
  { id: "visual", name: "Visual prostheses", short: "Visual prostheses", source: "mixed", studies: "3 studies", typical: "17 sighted people (one study's sample unknown)", limit: "no blind users; about 50% agreement between people and the simulated agent", dots: [{ v: 17 }], sec: "health" },
  { id: "scs", name: "Spinal cord stimulation", short: "Spinal cord stimulation", source: "real", studies: "3 studies", typical: "1 to 5 patients", limit: "comparison with physicians qualitative; assumes responses do not change over time", lo: 1, hi: 5, tLo: 1, tHi: 5, sec: "health" },
  { id: "build", name: "Thermal comfort and daylight", short: "Buildings", source: "sim", studies: "8 studies", typical: "no real people since 2024 (earlier samples not obtained)", limit: "no trial of PBO with real occupants", sec: "here" },
  { id: "vehicle", name: "Driving style and controller calibration", short: "Vehicles and controllers", source: "sim", studies: "7 studies (plus 1 non-optimization)", typical: "one study with 29 drivers; otherwise authors or synthetic decision makers", limit: "no study on real roads; preferences vary with the scenario", dots: [{ v: 1 }, { v: 29 }], sec: "here" },
  { id: "robots", name: "Legged robots and controller tuning", short: "Robot controller tuning", source: "real", studies: "about 14 papers, with method and tool papers", typical: "1 expert or a few lab members", limit: "an expert's own cost function disagrees with their choices; effort savings not quantified", dots: [{ v: 1 }], sec: "health" },
  { id: "reward", name: "Active preference-based reward learning", short: "Preference reward learning", source: "mixed", studies: "8 studies", typical: "about 10 users", limit: "mostly linear rewards; Gaussian process versions grow costly with dimension", dots: [{ v: 10 }], sec: "health" },
  { id: "materials", name: "Materials and physical sciences", short: "Materials and physics", source: "sim", studies: "7 studies", typical: "1 to 4 experts", limit: "experts and non-experts diverge; over-trust in the model", lo: 1, hi: 4, tLo: 1, tHi: 4, sec: "here" },
  { id: "chem", name: "Chemistry and drug discovery", short: "Chemistry and drugs", source: "sim", studies: "4 studies", typical: "35 chemists in MolSkill (not BO); otherwise 1 or simulated", limit: "only moderate agreement between chemists (kappa 0.32 to 0.40)", dots: [{ v: 1 }, { v: 35, hollow: true }], sec: "here" },
  { id: "protein", name: "Protein design", short: "Protein design", source: "nohuman", studies: "1 study", typical: "none", limit: "the 'preference' comes from measured fitness", sec: "here" },
  { id: "industry", name: "Industrial platforms and A/B testing", short: "Industry and A/B testing", source: "sim", studies: "6 studies", typical: "no public deployment data", limit: "no public report quantifying production deployment", sec: "here" },
];

const WHERE: Record<string, string> = { design: "see the interactive design chapter", health: "see the wearables and health chapter", here: "covered in this chapter" };
const WHERE_ZH: Record<string, string> = { design: "见交互式设计一章", health: "见可穿戴机器人与健康一章", here: "在本章中介绍" };

// The Chinese text of each domain, by id.
const DOMAIN_ZH: Record<string, { name: string; short: string; studies: string; typical: string; limit: string }> = {
  design: { name: "视觉、界面与生成式设计", short: "视觉与生成式设计", studies: "约 24 项研究", typical: "多为 10 至 40（3 至 60，不含众包）", limit: "对照条件大多较弱；被试大多是新手；没有检验会话内的漂移" },
  rating: { name: "基于评分或性能的人在回路优化", short: "评分或性能的人在回路优化", studies: "约 19 项研究", typical: "12 至 40（8 至 200）", limit: "不是偏好反馈；面对强对照时，最终质量往往没有差别" },
  exo: { name: "下肢外骨骼", short: "外骨骼", studies: "13 项研究，另加 2 项自调基线", typical: "2 至 15，中位数约 5", limit: "只有 2 名截瘫被试；内部验证；几乎没有与手动调节的比较" },
  prosth: { name: "假肢", short: "假肢", studies: "1 项偏好贝叶斯优化研究（另有 3 项相关）", typical: "2 至 3 名截肢者", limit: "样本极小；偏好在各次试验之间不一致；没有明显的生物力学差异" },
  audio: { name: "助听器与音频", short: "助听器与音频", studies: "5 项研究（另有 3 项相关）", typical: "20 至 35", limit: "言语清晰度没有改善；评估大多有制造商参与" },
  visual: { name: "视觉假体", short: "视觉假体", studies: "3 项研究", typical: "17 名视力正常者（一项研究的样本未知）", limit: "没有盲人用户；人与模拟智能体之间的一致率约 50%" },
  scs: { name: "脊髓刺激", short: "脊髓刺激", studies: "3 项研究", typical: "1 至 5 名患者", limit: "与医生的比较是定性的；假设反应不随时间改变" },
  build: { name: "热舒适与天然采光", short: "建筑", studies: "8 项研究", typical: "2024 年以来没有真人（更早研究的样本未能获得）", limit: "没有用真实居住者进行的偏好贝叶斯优化试验" },
  vehicle: { name: "驾驶风格与控制器标定", short: "车辆与控制器", studies: "7 项研究（另有 1 项非优化研究）", typical: "一项研究有 29 名驾驶员；其余为作者或合成决策者", limit: "没有在真实道路上的研究；偏好随场景而变" },
  robots: { name: "腿式机器人与控制器调节", short: "机器人控制器调节", studies: "约 14 篇论文，含方法与工具论文", typical: "1 位专家或几名实验室成员", limit: "专家自己的代价函数与其选择不一致；节省的工作量未经量化" },
  reward: { name: "主动的基于偏好的奖励学习", short: "偏好奖励学习", studies: "8 项研究", typical: "约 10 名用户", limit: "大多是线性奖励；高斯过程版本的代价随维度增长而升高" },
  materials: { name: "材料与物理科学", short: "材料与物理", studies: "7 项研究", typical: "1 至 4 位专家", limit: "专家与非专家结果分化；对模型过度信任" },
  chem: { name: "化学与药物发现", short: "化学与药物", studies: "4 项研究", typical: "MolSkill 中有 35 位化学家（不是贝叶斯优化）；其余为 1 人或模拟", limit: "化学家之间的一致性只是中等（κ 为 0.32 至 0.40）" },
  protein: { name: "蛋白质设计", short: "蛋白质设计", studies: "1 项研究", typical: "无", limit: "“偏好”来自测得的适应度" },
  industry: { name: "工业平台与 A/B 测试", short: "工业与 A/B 测试", studies: "6 项研究", typical: "没有公开的部署数据", limit: "没有量化生产部署的公开报告" },
};

// A domain's text in the edition's language.
function textOf(d: Domain, lang: Lang): { name: string; short: string; studies: string; typical: string; limit: string } {
  return lang === "zh" ? DOMAIN_ZH[d.id] : d;
}

const MISSING = "agriculture; food and flavor with sensory panels; motion sickness; cochlear implants; deep brain stimulation; functional electrical stimulation; comfort of sockets, seats, and clothing; drones before 2026; driving on real roads; daylight after 2020; protein design from human preferences; production A/B testing";
const MISSING_ZH = "农业；借助感官评价小组的食品与风味；晕动症；人工耳蜗；脑深部电刺激；功能性电刺激；接受腔、座椅与服装的舒适度；2026 年以前的无人机；真实道路上的驾驶；2020 年以后的天然采光；基于人的偏好的蛋白质设计；生产环境中的 A/B 测试";

const params = {
  domain: { kind: "choice", label: { en: "Domain", zh: "领域" }, options: DOMAINS.map((d) => ({ value: d.id, label: { en: d.short, zh: DOMAIN_ZH[d.id].short } })), default: "exo", control: "select" },
} as const;

type P = { domain: string };
const pick = (id: string) => DOMAINS.find((d) => d.id === id) ?? DOMAINS[0];

function describe(st: State<P>): string {
  const lang = st.lang ?? "en";
  const d = pick(st.p.domain);
  const t = textOf(d, lang);
  return tpl(labels[lang].describe, { name: t.name, studies: t.studies, typical: t.typical, source: sourceLabel(d.source, lang), limit: t.limit });
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

// Wrap Chinese text, which has no spaces between words, to lines at most
// `width` pixels wide: a Chinese character or full-width mark counts `em`, a
// run of Latin letters, digits, and symbols is kept whole at `px` per
// character, and a line never starts with closing punctuation.
function wrapZh(s: string, width: number, px = 6.4, em = 12): string[] {
  const toks = s.match(/[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef“”‘’]|[^\s\u3000-\u303f\u4e00-\u9fff\uff00-\uffef“”‘’]+|\s+/g) ?? [];
  const lines: string[] = [];
  let line = "", w = 0;
  for (const t of toks) {
    const tw = labelWidth(t, px, em);
    if (line && w + tw > width && !/^[，。；：、）？！”’]$/.test(t) && !/^\s+$/.test(t)) { lines.push(line.trimEnd()); line = ""; w = 0; }
    if (!line && /^\s+$/.test(t)) continue;
    line += t; w += tw;
  }
  if (line.trim()) lines.push(line.trimEnd());
  return lines;
}

const wrapIn = (lang: Lang, s: string, width: number, px: number, em: number) => (lang === "zh" ? wrapZh(s, width, px, em) : wrap(s, width, px));

function render(st: State<P>): string {
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 480;
  const sel = pick(st.p.domain);
  const selT = textOf(sel, lang);
  const parts: string[] = [];
  const lx = narrow ? 8 : 12;

  // Legend.
  let cx = lx, ly = 14;
  for (const s of ["real", "mixed", "sim", "nohuman"] as Source[]) {
    const wpx = labelWidth(sourceLabel(s, lang), 6.1) + 26;
    if (cx + wpx > st.w - 8 && cx > lx) { cx = lx; ly += 18; }
    parts.push(el("rect", { x: cx, y: ly - 8, width: 14, height: 8, rx: 2, fill: SOURCE_COLOR[s] }), text(cx + 19, ly, sourceLabel(s, lang), { "font-size": TYPE.small, class: "fig-t-muted" }));
    cx += wpx;
  }

  // Rows. Wide: label column on the left. Narrow: label above each row's line.
  const labelW = narrow ? 0 : 196;
  const left = lx + labelW + (narrow ? 4 : 8);
  const right = st.w - 14;
  const top = ly + 22;
  const rowH = narrow ? 34 : 24;
  const x = log([1, 300], [left + 6, right - 6]);
  const bottom = top + rowH * DOMAINS.length;
  for (const v of [1, 3, 10, 30, 100, 300]) {
    parts.push(el("line", { x1: x(v), x2: x(v), y1: top - 6, y2: bottom, stroke: C.grid }),
      text(x(v), bottom + 15, String(v), { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted fig-t-num" }));
  }
  parts.push(el("line", { x1: left, x2: right, y1: bottom, y2: bottom, stroke: C.rule }),
    text((left + right) / 2, bottom + 31, L.axis, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted" }));

  DOMAINS.forEach((d, i) => {
    const y0 = top + i * rowH;
    const yl = narrow ? y0 + rowH - 10 : y0 + rowH / 2;
    const isSel = d.id === sel.id;
    const col = SOURCE_COLOR[d.source];
    const items: string[] = [];
    if (isSel) items.push(el("rect", { x: lx - 4, y: y0 + 1, width: st.w - 2 * lx + 8, height: rowH - 2, rx: 4, fill: C.panel, stroke: C.rule }));
    items.push(el("rect", { x: lx - 4, y: y0, width: st.w - 2 * lx + 8, height: rowH, fill: "transparent" }));
    const name = textOf(d, lang).short;
    if (narrow) items.push(text(lx + 2, y0 + 12, name, { "font-size": TYPE.small, class: isSel ? "fig-t-strong" : "fig-t-muted" }));
    else items.push(text(lx + labelW, yl + 4, name, { "font-size": TYPE.small, "text-anchor": "end", class: isSel ? "fig-t-strong" : "fig-t-muted" }));
    if (d.lo !== undefined) items.push(el("line", { x1: x(d.lo), x2: x(d.hi!), y1: yl, y2: yl, stroke: col, "stroke-width": 2, "stroke-linecap": "round" }));
    if (d.tLo !== undefined) {
      const w = Math.max(6, x(d.tHi!) - x(d.tLo));
      const cxm = (x(d.tLo) + x(d.tHi!)) / 2;
      items.push(el("rect", { x: cxm - w / 2, y: yl - 4.5, width: w, height: 9, rx: 4.5, fill: col }));
    }
    for (const dt of d.dots ?? []) items.push(el("circle", { cx: x(dt.v), cy: yl, r: 5, fill: dt.hollow ? C.paper : col, stroke: dt.hollow ? col : C.paper, "stroke-width": dt.hollow ? 2 : 1.2 }));
    if (d.lo === undefined && !d.dots) items.push(text(x(1) + 2, yl + 4, L.none, { "font-size": TYPE.small, class: "fig-t-faint", style: "font-style:italic" }));
    parts.push(g({ "data-fig-set": `domain=${d.id}`, style: "cursor:pointer" }, ...items));
  });

  // Detail panel, then the list of domains with no study.
  const py = bottom + 48;
  const pw = st.w - 2 * lx;
  const lines = [
    { t: selT.name, cls: "fig-t-strong", size: TYPE.label },
    ...wrapIn(lang, `${selT.studies} · ${sourceLabel(sel.source, lang)} · ${(lang === "zh" ? WHERE_ZH : WHERE)[sel.sec]}`, pw - 16, 6.1, TYPE.small).map((t) => ({ t, cls: "fig-t-muted", size: TYPE.small })),
    ...wrapIn(lang, tpl(L.typicalLine, { t: selT.typical }), pw - 16, 6.6, TYPE.body).map((t) => ({ t, cls: "", size: TYPE.body })),
    ...wrapIn(lang, tpl(L.limitLine, { t: selT.limit }), pw - 16, 6.6, TYPE.body).map((t) => ({ t, cls: "", size: TYPE.body })),
  ];
  const ph = 12 + lines.length * 17;
  parts.push(el("rect", { x: lx - 4, y: py - 4, width: pw + 8, height: ph, rx: 6, fill: C.panel, stroke: C.rule }));
  lines.forEach((ln, i) => parts.push(text(lx + 6, py + 13 + i * 17, ln.t, { "font-size": ln.size, class: ln.cls || undefined })));
  const my = py + ph + 20;
  const ml = wrapIn(lang, tpl(L.missingLine, { m: L.missing, list: lang === "zh" ? MISSING_ZH : MISSING }), pw, 6.1, TYPE.small);
  ml.forEach((t, i) => parts.push(text(lx, my + i * 16, t, { "font-size": TYPE.small, class: i === 0 ? "fig-t-muted" : "fig-t-muted" })));
  return svg(st.w, my + ml.length * 16, describe(st), ...parts);
}

export default defineFigure({
  name: "sci-evidence",
  title: { en: "The human evidence by application domain", zh: "按应用领域列出的人类证据" },
  labels,
  params,
  hint: { en: "Click a row, or choose a domain, to read its study count and main limitation.", zh: "点击一行或选择一个领域，查看它的研究数量和主要局限。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
