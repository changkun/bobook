// A map of the book: every chapter as a numbered node, one row per part, and
// the prerequisite links between them. Choose a part to highlight it; click a
// chapter to see what it builds on (above it) and what builds on it (below),
// with its title and a link to open it.

import { defineFigure, type Lang, type State } from "./types.ts";
import { el, esc, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";

interface Ch { n: number; title: string; zh: string; href: string; part: number }

const PARTS = [
  ["foundations", "Foundations"], ["gp", "Gaussian Processes"], ["bo", "Bayesian Optimization"],
  ["preferences", "Learning from Comparisons"], ["cases", "Case Studies"], ["frontier", "The Research Frontier"],
  ["humans", "People in the Loop"], ["neighbors", "Neighbors in Computing"], ["perspectives", "What Is a Preference?"],
  ["synthesis", "Synthesis"],
] as const;

// The Chinese part and chapter titles (GLOSSARY.zh.md §5.2 and §5.3), in the
// order of PARTS and FILES.
const PARTS_ZH = ["基础", "高斯过程", "贝叶斯优化", "从比较中学习", "案例研究", "研究前沿", "人在回路", "相邻计算领域", "偏好是什么？", "综合"];
const ZH_NUM = ["一", "二", "三", "四", "五", "六", "七", "八", "九", "十"];
const TITLES_ZH: Record<string, string> = {
  "foundations/01-optimizing-the-unknown": "优化写不出公式的函数",
  "foundations/02-probability": "概率：为不确定性记账",
  "foundations/03-linear-algebra": "不确定性的线性代数",
  "foundations/04-gaussian": "高斯分布",
  "foundations/05-bayesian-inference": "贝叶斯推断",
  "foundations/06-information": "度量信息",
  "gp/01-distributions-over-functions": "函数上的分布",
  "gp/02-gp-regression": "高斯过程回归",
  "gp/03-kernels-and-hyperparameters": "核函数与超参数",
  "gp/04-analysis-of-kernels": "核函数背后的分析",
  "bo/01-the-loop": "贝叶斯优化循环",
  "bo/02-acquisition-functions": "采集函数",
  "bo/03-regret-and-bandits": "遗憾、赌博机与理论保证",
  "bo/04-bo-in-practice": "贝叶斯优化实践",
  "bo/05-applications": "贝叶斯优化的用武之地",
  "preferences/01-why-comparisons": "为什么请人做比较",
  "preferences/02-approximate-inference": "后验不是高斯分布时",
  "preferences/03-gp-preference-learning": "高斯过程偏好学习",
  "preferences/04-preferential-bo": "偏好贝叶斯优化",
  "preferences/05-query-design": "设计提问",
  "preferences/06-dueling-bandits": "对决赌博机与比较的理论",
  "cases/01-tuning-a-classifier": "为分类器调参",
  "cases/02-chemical-reaction": "优化化学反应",
  "cases/03-exoskeleton": "人在回路中调节外骨骼",
  "cases/04-photo-enhancement": "通过比较增强照片",
  "frontier/01-a-decade-of-pbo": "偏好贝叶斯优化的十年",
  "frontier/02-observation-models": "观测模型、代理模型与推断",
  "frontier/03-acquisition-frontier": "采集函数、查询形式与问题扩展",
  "frontier/04-theory": "理论：从对决赌博机到核化偏好优化",
  "frontier/05-high-dimensions": "高维问题与贝叶斯优化格局的变化",
  "frontier/06-software-evaluation": "软件、评测方法与研究社区",
  "humans/01-interactive-design": "交互式设计与人机交互",
  "humans/02-body-and-health": "可穿戴机器人、健康与辅助技术",
  "humans/03-environments-science-industry": "建成环境、科学与工业",
  "neighbors/01-language-models": "偏好与大语言模型",
  "neighbors/02-adjacent-fields": "奖励学习、推荐、排序与自动化科学",
  "perspectives/01-judgment-and-psychophysics": "判断、决策与心理物理学",
  "perspectives/02-social-psychology": "社会、情感与发展心理学",
  "perspectives/03-neuroscience": "神经科学与计算认知科学",
  "perspectives/04-economics": "经济学、决策理论与运筹学",
  "perspectives/05-philosophy": "哲学与宗教传统",
  "perspectives/06-social-sciences": "社会科学与人文学科",
  "perspectives/07-natural-sciences": "自然科学与形式科学",
  "perspectives/08-design-and-senses": "设计、感官科学、艺术与健康",
  "synthesis/01-what-a-comparison-measures": "一次比较测量什么",
  "synthesis/02-recommendations": "构建与评估偏好优化系统",
  "synthesis/03-open-problems": "未解决问题与判定实验",
};

// [part index, file slug, title]. Chapters are numbered in this order, which
// must match en/book.yml; prerequisite links below use the slugs, so adding a
// chapter never silently rewires the map.
const FILES: Array<[number, string, string]> = [
  [0, "01-optimizing-the-unknown", "Optimizing What You Cannot Write Down"],
  [0, "02-probability", "Probability as Bookkeeping for Uncertainty"],
  [0, "03-linear-algebra", "The Linear Algebra of Uncertainty"],
  [0, "04-gaussian", "The Gaussian Distribution"],
  [0, "05-bayesian-inference", "Bayesian Inference"],
  [0, "06-information", "Measuring Information"],
  [1, "01-distributions-over-functions", "Distributions over Functions"],
  [1, "02-gp-regression", "Gaussian Process Regression"],
  [1, "03-kernels-and-hyperparameters", "Kernels and Hyperparameters"],
  [1, "04-analysis-of-kernels", "The Analysis Behind Kernels"],
  [2, "01-the-loop", "The Bayesian Optimization Loop"],
  [2, "02-acquisition-functions", "Acquisition Functions"],
  [2, "03-regret-and-bandits", "Regret, Bandits, and Guarantees"],
  [2, "04-bo-in-practice", "Bayesian Optimization in Practice"],
  [2, "05-applications", "Where Bayesian Optimization Works"],
  [3, "01-why-comparisons", "Why Ask for Comparisons"],
  [3, "02-approximate-inference", "When the Posterior Is Not Gaussian"],
  [3, "03-gp-preference-learning", "Gaussian Process Preference Learning"],
  [3, "04-preferential-bo", "Preferential Bayesian Optimization"],
  [3, "05-query-design", "Designing the Question"],
  [3, "06-dueling-bandits", "Dueling Bandits and the Theory of Comparisons"],
  [4, "01-tuning-a-classifier", "Tuning a Classifier"],
  [4, "02-chemical-reaction", "Optimizing a Chemical Reaction"],
  [4, "03-exoskeleton", "Tuning an Exoskeleton with a Person in the Loop"],
  [4, "04-photo-enhancement", "Enhancing a Photo by Comparison"],
  [5, "01-a-decade-of-pbo", "A Decade of Preferential Bayesian Optimization"],
  [5, "02-observation-models", "Observation Models, Surrogates, and Inference"],
  [5, "03-acquisition-frontier", "Acquisition, Query Forms, and Problem Extensions"],
  [5, "04-theory", "Theory: From Dueling Bandits to Kernelized Preference Optimization"],
  [5, "05-high-dimensions", "High Dimensions and the Changing Landscape of BO"],
  [5, "06-software-evaluation", "Software, Evaluation, and the Research Community"],
  [6, "01-interactive-design", "Interactive Design and Human-Computer Interaction"],
  [6, "02-body-and-health", "Wearable Robots, Health, and Assistive Technology"],
  [6, "03-environments-science-industry", "Built Environments, Science, and Industry"],
  [7, "01-language-models", "Preferences and Large Language Models"],
  [7, "02-adjacent-fields", "Reward Learning, Recommendation, Ranking, and Automated Science"],
  [8, "01-judgment-and-psychophysics", "Judgment, Decision, and Psychophysics"],
  [8, "02-social-psychology", "Social, Affective, and Developmental Psychology"],
  [8, "03-neuroscience", "Neuroscience and Computational Cognitive Science"],
  [8, "04-economics", "Economics, Decision Theory, and Operations Research"],
  [8, "05-philosophy", "Philosophy and Religious Traditions"],
  [8, "06-social-sciences", "Social Sciences and the Humanities"],
  [8, "07-natural-sciences", "Natural and Formal Sciences"],
  [8, "08-design-and-senses", "Design, Sensory Science, Art, and Health"],
  [9, "01-what-a-comparison-measures", "What a Comparison Measures"],
  [9, "02-recommendations", "Building and Evaluating a Preferential Optimization System"],
  [9, "03-open-problems", "Open Problems and the Decisive Experiment"],
];

const CHAPTERS: Ch[] = FILES.map(([part, file, title], i) => ({ n: i + 1, title, zh: TITLES_ZH[`${PARTS[part][0]}/${file}`] ?? title, part, href: `en/${PARTS[part][0]}/${file}.html` }));
// A chapter's title and page in the edition's language.
const titleOf = (c: Ch, lang: Lang) => (lang === "zh" ? c.zh : c.title);
const hrefOf = (c: Ch, lang: Lang) => (lang === "zh" ? c.href.replace(/^en\//, "zh/") : c.href);
const BY_SLUG = new Map(FILES.map(([part, file], i) => [`${PARTS[part][0]}/${file}`, i + 1]));
const n = (slug: string) => { const v = BY_SLUG.get(slug); if (!v) throw new Error(`book-map: unknown chapter ${slug}`); return v; };

// Prerequisites: [chapter, builds on], by part/slug.
const LINKS: Array<[string, string]> = [
  ["foundations/02-probability", "foundations/01-optimizing-the-unknown"], ["foundations/03-linear-algebra", "foundations/01-optimizing-the-unknown"],
  ["foundations/04-gaussian", "foundations/02-probability"], ["foundations/04-gaussian", "foundations/03-linear-algebra"],
  ["foundations/05-bayesian-inference", "foundations/04-gaussian"], ["foundations/06-information", "foundations/02-probability"],
  ["gp/01-distributions-over-functions", "foundations/05-bayesian-inference"], ["gp/02-gp-regression", "gp/01-distributions-over-functions"],
  ["gp/03-kernels-and-hyperparameters", "gp/02-gp-regression"],
  ["gp/04-analysis-of-kernels", "gp/03-kernels-and-hyperparameters"], ["gp/04-analysis-of-kernels", "foundations/03-linear-algebra"],
  ["bo/01-the-loop", "gp/02-gp-regression"], ["bo/02-acquisition-functions", "bo/01-the-loop"],
  ["bo/03-regret-and-bandits", "bo/02-acquisition-functions"], ["bo/03-regret-and-bandits", "foundations/06-information"],
  ["bo/03-regret-and-bandits", "gp/04-analysis-of-kernels"],
  ["bo/04-bo-in-practice", "bo/02-acquisition-functions"], ["bo/04-bo-in-practice", "gp/03-kernels-and-hyperparameters"],
  ["bo/05-applications", "bo/04-bo-in-practice"],
  ["preferences/01-why-comparisons", "foundations/02-probability"], ["preferences/02-approximate-inference", "foundations/04-gaussian"],
  ["preferences/03-gp-preference-learning", "preferences/01-why-comparisons"], ["preferences/03-gp-preference-learning", "preferences/02-approximate-inference"],
  ["preferences/03-gp-preference-learning", "gp/02-gp-regression"], ["preferences/04-preferential-bo", "preferences/03-gp-preference-learning"],
  ["preferences/04-preferential-bo", "bo/02-acquisition-functions"], ["preferences/05-query-design", "preferences/04-preferential-bo"],
  ["preferences/06-dueling-bandits", "preferences/04-preferential-bo"], ["preferences/06-dueling-bandits", "bo/03-regret-and-bandits"],
  ["cases/01-tuning-a-classifier", "bo/04-bo-in-practice"], ["cases/02-chemical-reaction", "bo/04-bo-in-practice"],
  ["cases/03-exoskeleton", "bo/04-bo-in-practice"], ["cases/03-exoskeleton", "preferences/04-preferential-bo"],
  ["cases/04-photo-enhancement", "preferences/05-query-design"],
  ["frontier/01-a-decade-of-pbo", "preferences/04-preferential-bo"], ["frontier/02-observation-models", "preferences/03-gp-preference-learning"],
  ["frontier/03-acquisition-frontier", "preferences/04-preferential-bo"], ["frontier/03-acquisition-frontier", "preferences/05-query-design"],
  ["frontier/04-theory", "preferences/06-dueling-bandits"], ["frontier/05-high-dimensions", "gp/03-kernels-and-hyperparameters"],
  ["frontier/05-high-dimensions", "bo/04-bo-in-practice"], ["frontier/06-software-evaluation", "bo/04-bo-in-practice"],
  ["frontier/06-software-evaluation", "preferences/04-preferential-bo"],
  ["humans/01-interactive-design", "preferences/04-preferential-bo"], ["humans/01-interactive-design", "preferences/05-query-design"],
  ["humans/02-body-and-health", "preferences/04-preferential-bo"], ["humans/02-body-and-health", "cases/03-exoskeleton"],
  ["humans/03-environments-science-industry", "preferences/04-preferential-bo"],
  ["neighbors/01-language-models", "preferences/01-why-comparisons"], ["neighbors/01-language-models", "preferences/03-gp-preference-learning"],
  ["neighbors/02-adjacent-fields", "bo/03-regret-and-bandits"], ["neighbors/02-adjacent-fields", "neighbors/01-language-models"],
  ["perspectives/01-judgment-and-psychophysics", "preferences/01-why-comparisons"],
  ["perspectives/02-social-psychology", "perspectives/01-judgment-and-psychophysics"], ["perspectives/03-neuroscience", "perspectives/01-judgment-and-psychophysics"],
  ["perspectives/04-economics", "perspectives/01-judgment-and-psychophysics"], ["perspectives/05-philosophy", "perspectives/01-judgment-and-psychophysics"],
  ["perspectives/06-social-sciences", "perspectives/01-judgment-and-psychophysics"], ["perspectives/07-natural-sciences", "preferences/01-why-comparisons"],
  ["perspectives/08-design-and-senses", "perspectives/01-judgment-and-psychophysics"],
  ["synthesis/01-what-a-comparison-measures", "frontier/02-observation-models"], ["synthesis/01-what-a-comparison-measures", "humans/01-interactive-design"],
  ["synthesis/01-what-a-comparison-measures", "perspectives/01-judgment-and-psychophysics"], ["synthesis/01-what-a-comparison-measures", "perspectives/04-economics"],
  ["synthesis/02-recommendations", "synthesis/01-what-a-comparison-measures"], ["synthesis/02-recommendations", "frontier/06-software-evaluation"],
  ["synthesis/03-open-problems", "synthesis/02-recommendations"],
];
const EDGES: Array<[number, number]> = LINKS.map(([a, b]) => [n(a), n(b)]);

const labels = {
  en: {
    open: "Open chapter {n} →",
    prompt: "Click a chapter to see what it builds on and what builds on it.",
    before: "builds on: {list}",
    after: "leads to: {list}",
    none: "nothing earlier",
    describe: "A map of the book's {n} chapters in ten parts, with prerequisite links.",
    describeSel: "Chapter {n}, {t}, builds on chapters {b} and leads to chapters {a}.",
    nil: "none",
  },
  zh: {
    open: "打开第 {n} 章 →",
    prompt: "点击某一章，查看它以哪些章为基础，又是哪些章的基础。",
    before: "前置：{list}",
    after: "后续：{list}",
    none: "无",
    describe: "全书 {n} 章、十个部分的地图，标出各章之间的前置关系。",
    describeSel: "第 {n} 章“{t}”的前置章节：{b}；后续章节：{a}。",
    nil: "无",
  },
};

const params = {
  part: { kind: "choice", label: { en: "Highlight", zh: "突出显示" }, options: [{ value: -1, label: { en: "Whole book", zh: "全书" } }, ...PARTS.map(([, t], i) => ({ value: i, label: { en: `Part ${["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"][i]}`, zh: `第${ZH_NUM[i]}部分` } }))], default: -1, control: "select" },
  sel: { kind: "range", label: { en: "Selected chapter", zh: "选中的章" }, min: 0, max: 47, default: 0, step: 1, control: false },
} as const;

type P = { part: number; sel: number };

const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"];

function describe(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const sep = lang === "zh" ? "、" : ", ";
  if (!p.sel) return L.describe.replace("{n}", String(CHAPTERS.length));
  const c = CHAPTERS[p.sel - 1];
  const b = EDGES.filter(([a]) => a === c.n).map(([, x]) => x);
  const a = EDGES.filter(([, x]) => x === c.n).map(([y]) => y);
  return L.describeSel.replace("{n}", String(c.n)).replace("{t}", titleOf(c, lang)).replace("{b}", b.join(sep) || L.nil).replace("{a}", a.join(sep) || L.nil);
}

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const sep = lang === "zh" ? "、" : ", ";
  const narrow = st.w < 520;
  const rowH = narrow ? 44 : 46;
  const left = narrow ? 34 : 196;
  const right = st.w - 12;
  const top = 10;
  const maxPer = 8;
  const pos = new Map<number, [number, number]>();
  const parts: string[] = [];
  const sel = p.sel ? CHAPTERS[p.sel - 1] : undefined;
  const pre = new Set(sel ? EDGES.filter(([a]) => a === sel.n).map(([, b]) => b) : []);
  const post = new Set(sel ? EDGES.filter(([, b]) => b === sel.n).map(([a]) => a) : []);
  PARTS.forEach(([, title], pi) => {
    const y = top + pi * rowH + rowH / 2;
    const chs = CHAPTERS.filter((c) => c.part === pi);
    const span = right - left;
    const step = span / maxPer;
    chs.forEach((c, i) => pos.set(c.n, [left + step * (i + 0.5), y]));
    const on = p.part === -1 || p.part === pi;
    const row = lang === "zh" ? (narrow ? ZH_NUM[pi] : `第${ZH_NUM[pi]}部分 ${PARTS_ZH[pi]}`) : narrow ? ROMAN[pi] : `${ROMAN[pi]}  ${title}`;
    parts.push(text(narrow ? 4 : 8, y + 4, row, { "font-size": TYPE.small, class: on ? "fig-t-strong" : "fig-t-faint" }));
  });
  // edges
  for (const [a, b] of EDGES) {
    const [xa, ya] = pos.get(a)!, [xb, yb] = pos.get(b)!;
    const hot = sel && (a === sel.n || b === sel.n);
    // Draw only what answers a question: the selected chapter's links, or
    // the links touching the highlighted part. The whole web at once is noise.
    const touchesPart = p.part !== -1 && (CHAPTERS[a - 1].part === p.part || CHAPTERS[b - 1].part === p.part);
    if (!hot && !(touchesPart && !sel)) continue;
    const dim = false;
    const my = (ya + yb) / 2;
    parts.push(el("path", {
      d: ya === yb ? `M${xb},${yb} Q${(xa + xb) / 2},${yb - 16} ${xa},${ya}` : `M${xb},${yb} C${xb},${my} ${xa},${my} ${xa},${ya}`,
      fill: "none", stroke: hot ? C.acq : C.ink3, "stroke-width": hot ? 1.8 : 0.9, opacity: dim ? 0.15 : hot ? 1 : 0.45,
    }));
  }
  // nodes
  for (const c of CHAPTERS) {
    const [x, y] = pos.get(c.n)!;
    const inPart = p.part === -1 || p.part === c.part;
    const isSel = sel?.n === c.n;
    const rel = pre.has(c.n) || post.has(c.n);
    const fill = isSel ? C.acq : rel ? C.model : inPart ? C.panel : C.paper;
    const stroke = isSel ? C.acq : rel ? C.model : inPart ? C.ink2 : C.rule;
    parts.push(g({ "data-fig-hit": `ch-${c.n}`, "data-fig-set": `sel=${c.n}` },
      el("circle", { cx: x, cy: y, r: narrow ? 12 : 13, fill, stroke, "stroke-width": isSel || rel ? 2 : 1.2 }),
      text(x, y + 4, c.n, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-num", style: isSel || rel ? "fill:var(--fig-paper);font-weight:600" : inPart ? "" : "fill:var(--fig-ink-3)" }),
      el("title", {}, esc(`${c.n}. ${titleOf(c, lang)}`)),
    ));
  }
  // readout
  const ry = top + PARTS.length * rowH + 18;
  if (sel) {
    const b = [...pre].sort((x, y) => x - y), a = [...post].sort((x, y) => x - y);
    parts.push(
      text(8, ry, `${sel.n}. ${titleOf(sel, lang)}`, { "font-size": TYPE.label, class: "fig-t-strong" }),
      text(8, ry + 18, L.before.replace("{list}", b.length ? b.join(sep) : L.none), { "font-size": TYPE.small, class: "fig-t-muted" }),
      text(narrow ? 8 : st.w / 2, ry + (narrow ? 34 : 18), L.after.replace("{list}", a.length ? a.join(sep) : "–"), { "font-size": TYPE.small, class: "fig-t-muted" }),
      el("a", { href: `@@ROOT@@${hrefOf(sel, lang)}` }, text(8, ry + (narrow ? 54 : 40), L.open.replace("{n}", String(sel.n)), { "font-size": TYPE.label, style: "fill:var(--fig-c1);font-weight:600" })),
    );
  } else {
    parts.push(text(8, ry, L.prompt, { "font-size": TYPE.small, class: "fig-t-muted" }));
  }
  return svg(st.w, ry + (sel ? (narrow ? 64 : 50) : 12), describe(st), ...parts);
}

export default defineFigure({
  name: "book-map",
  title: { en: "Map of the book", zh: "全书地图" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  actions: [
    { label: { en: "Previous chapter", zh: "上一章" }, run: (p) => ({ ...p, sel: Math.max(1, (p.sel || 2) - 1) }) },
    { label: { en: "Next chapter", zh: "下一章" }, run: (p) => ({ ...p, sel: Math.min(CHAPTERS.length, p.sel + 1) }) },
  ],
});
