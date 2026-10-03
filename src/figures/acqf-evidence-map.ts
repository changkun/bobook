// Where each finding about an acquisition rule was observed. Every published
// comparison of preferential acquisition functions ran at particular input
// dimensions and under a particular noise model; this map draws each study as
// a row, placed by its noise model (groups) and spanning the dimensions it
// tested (log axis), with tasks whose dimension the study does not give in a
// separate column. Choosing a rule highlights the studies that report on it,
// colored by whether the finding was favorable, mixed, or unfavorable for that
// rule, and lists the findings with their conditions. The point it makes is
// that rankings were observed in largely disjoint regions of dimension and
// noise, so they cannot simply be pooled; the Thompson sampling view shows the
// dimension dependence most clearly.

import { defineFigure, type Lang, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { log } from "./lib/scale.ts";
import { tpl } from "./lib/format.ts";

type Verdict = "good" | "mixed" | "bad";

interface Study {
  id: string;
  name: string; // full name for the list
  short: string; // row label
  group: number;
  dims: number[]; // dimensions the study names
  lo?: number; // "up to": lower dimensions also tested, values not given
  ns?: string; // tasks whose dimension is not given, or real-data tasks
  span?: boolean; // dims are the two ends of a tested range
  noise: string;
  budget: string;
}

const GROUPS = [
  "Noise-free or nearly",
  "Gaussian noise on the utility",
  "Probit or logistic answers",
  "A share of wrong answers",
  "Noise model not stated",
];

const STUDIES: Study[] = [
  { id: "pabbo", name: "Zhang et al. 2025 (PABBO)", short: "Zhang 2025", group: 0, dims: [1, 2, 6], ns: "HPO-B, Candy, Sushi", noise: "noise-free comparisons", budget: "30 repetitions" },
  { id: "takeno", name: "Takeno et al. 2023", short: "Takeno 2023", group: 0, dims: [4, 6], lo: 1, noise: "noise variance 10⁻⁴", budget: "12 functions up to 6-D; 3d initial duels; 10 repetitions" },
  { id: "mikkola", name: "Mikkola et al. 2020", short: "Mikkola 2020", group: 1, dims: [2, 6, 10, 20], noise: "small Gaussian noise", budget: "100 queries; 25 initializations" },
  { id: "siivola", name: "Siivola et al. 2021", short: "Siivola 2021", group: 1, dims: [4], lo: 1, noise: "utility noise sd 0.05", budget: "6 functions and real data, at most 4-D; batch sizes 2 to 6; 10 repetitions" },
  { id: "menn", name: "Menn et al. 2026 (local PBO, preprint)", short: "Menn 2026", group: 1, dims: [33, 96, 102], lo: 1, noise: "Gaussian, 10% of the value range", budget: "GP samples up to 96-D, Hopper 33-D, Walker2D 102-D; about 10d comparisons" },
  { id: "fauvel", name: "Fauvel and Chalk 2021 (preprint)", short: "Fauvel 2021", group: 2, dims: [], ns: "34 functions", noise: "probit, unit variance after normalization", budget: "80 iterations; 40 repetitions" },
  { id: "wu", name: "Wu and Gardner 2026 (preprint)", short: "Wu 2026", group: 2, dims: [2], noise: "probit", budget: "a 2-D Levy case study" },
  { id: "astudillo", name: "Astudillo et al. 2023 (qEUBO)", short: "Astudillo 2023", group: 2, dims: [4, 7], span: true, noise: "logistic, calibrated to 10%, 20%, 30% errors on the top 1% of pairs", budget: "4d initial + 150 queries; 50 or 100 repetitions" },
  { id: "xu", name: "Xu et al. 2024 (POP-BO)", short: "Xu 2024", group: 2, dims: [6], ns: "GP samples", noise: "logistic", budget: "6-D Ackley and GP samples; budget not stated" },
  { id: "lazzaro", name: "Lazzaro et al. 2026 (PF-TS)", short: "Lazzaro 2026", group: 2, dims: [1, 3], noise: "logistic", budget: "1-D Ackley, 3-D catalyst; T = 300; 30 repetitions" },
  { id: "lin", name: "Lin et al. 2022 (BOPE)", short: "Lin 2022", group: 3, dims: [], ns: "multi-outcome problems", noise: "10% wrong choices", budget: "75 comparisons in 3 stages; 30 repetitions" },
  { id: "gonzalez", name: "González et al. 2017", short: "González 2017", group: 4, dims: [1, 2], noise: "not stated", budget: "33 grid points per dimension; 5 + 200 duels; 20 repetitions" },
  { id: "nguyen", name: "Nguyen et al. 2021 (MPES)", short: "Nguyen 2021", group: 4, dims: [1, 3], span: true, ns: "CIFAR-10 embedding, SUSHI", noise: "not stated", budget: "budget not stated" },
  { id: "koyama", name: "Koyama et al. 2020 (Sequential Gallery)", short: "Koyama 2020", group: 4, dims: [5, 20], span: true, noise: "not stated", budget: "simulated functions, 5-D to 20-D; 50 trials" },
];

interface Finding { s: string; v: Verdict; t: string }
interface Rule { id: string; label: string; f: Finding[] }

const RULES: Rule[] = [
  { id: "ts", label: "Thompson sampling (DTS, qTS, PF-TS)", f: [
    { s: "gonzalez", v: "good", t: "DTS was consistently the best strategy." },
    { s: "lazzaro", v: "good", t: "PF-TS had significantly lower cumulative regret than MR-LPF and POP-BO." },
    { s: "nguyen", v: "bad", t: "DTS was behind MPES." },
    { s: "siivola", v: "mixed", t: "Batch Thompson sampling and batch EI showed no clear difference." },
    { s: "takeno", v: "bad", t: "Thompson sampling over-explored on the 4-D and 6-D Hartmann functions." },
    { s: "astudillo", v: "bad", t: "qTS lost to qEUBO, which was best on every problem except Car cab (q = 2)." },
    { s: "mikkola", v: "mixed", t: "The pairwise DTS variant lost to every projective variant, a contrast of query forms more than of rules." },
    { s: "fauvel", v: "mixed", t: "DTS ranked fourth of nine rules; Thompson-based rules performed only modestly." },
  ] },
  { id: "ei", label: "Expected improvement (Brochu EI, CEI, qEI)", f: [
    { s: "gonzalez", v: "bad", t: "CEI over-exploited and was too costly to run beyond the Forrester function; the interactive BO of Brochu et al. did poorly." },
    { s: "nguyen", v: "bad", t: "EI was behind MPES." },
    { s: "siivola", v: "mixed", t: "Batch EI and batch Thompson sampling showed no clear difference." },
    { s: "takeno", v: "bad", t: "Expectation propagation with EI often stalled from over-exploitation." },
    { s: "astudillo", v: "bad", t: "qEI stalled late in runs (7-D Alpine1) and is provably not consistent." },
    { s: "fauvel", v: "bad", t: "Brochu et al.'s EI ranked eighth of nine; only random was worse." },
  ] },
  { id: "muc", label: "Maximally uncertain challenge (MUC)", f: [
    { s: "takeno", v: "mixed", t: "Behind the hallucination believer overall, and relatively poor when combined with it." },
    { s: "fauvel", v: "good", t: "Tied first by Borda rank over 34 functions." },
  ] },
  { id: "ducb", label: "Dueling UCB and EIIG", f: [
    { s: "fauvel", v: "good", t: "Dueling UCB tied first by Borda rank; EIIG ranked seventh." },
  ] },
  { id: "eubo", label: "EUBO and qEUBO", f: [
    { s: "wu", v: "bad", t: "EUBO's queries collapsed toward the estimated maximum." },
    { s: "pabbo", v: "mixed", t: "PABBO ranked first or second on most tasks against qEUBO and other GP baselines." },
    { s: "astudillo", v: "good", t: "qEUBO was best on every problem except Car cab (q = 2)." },
    { s: "xu", v: "mixed", t: "qEUBO's reported solution was slightly better, but its cumulative regret was more than 2.5 times higher." },
    { s: "menn", v: "mixed", t: "Local methods did better than qEUBO on steep optima." },
    { s: "lin", v: "good", t: "EUBO variants were best." },
  ] },
  { id: "hb", label: "Hallucination believer (HB)", f: [
    { s: "takeno", v: "good", t: "Best overall across 12 functions." },
    { s: "xu", v: "bad", t: "Stuck in local optima under logistic noise." },
    { s: "menn", v: "mixed", t: "HB with EI was a baseline; local methods did better on steep optima." },
  ] },
  { id: "mpes", label: "Multinomial predictive entropy search (MPES)", f: [
    { s: "nguyen", v: "good", t: "Consistently best." },
    { s: "pabbo", v: "mixed", t: "A baseline; PABBO ranked first or second on most tasks." },
    { s: "astudillo", v: "bad", t: "Lost to qEUBO and was slowest: 12.7 to 24.8 s per iteration against about 7 to 12 s." },
  ] },
  { id: "forms", label: "Projections and galleries", f: [
    { s: "mikkola", v: "good", t: "Every projective variant clearly beat every pairwise variant." },
    { s: "koyama", v: "good", t: "The gallery beat line search and random plane construction." },
  ] },
  { id: "regret", label: "Regret-bound algorithms (POP-BO, PF-TS)", f: [
    { s: "lazzaro", v: "good", t: "PF-TS had significantly lower cumulative regret than MR-LPF and POP-BO." },
    { s: "xu", v: "good", t: "qEUBO's cumulative regret was more than 2.5 times POP-BO's." },
  ] },
  { id: "pabbo", label: "PABBO (amortized)", f: [
    { s: "pabbo", v: "good", t: "First or second on most tasks; weaker on 6-D Hartmann." },
  ] },
  { id: "local", label: "Local PBO", f: [
    { s: "menn", v: "good", t: "Better on steep optima; the authors call it unreliable from poor starting points." },
  ] },
  { id: "random", label: "Random queries", f: [
    { s: "pabbo", v: "good", t: "Random often beat some of the GP baselines." },
    { s: "siivola", v: "good", t: "On real data with a low signal-to-noise ratio, every method only barely beat the baseline." },
    { s: "mikkola", v: "bad", t: "Pairwise random lost to every projective variant." },
    { s: "astudillo", v: "bad", t: "Behind qEUBO." },
    { s: "fauvel", v: "bad", t: "Ranked last of nine." },
    { s: "lin", v: "bad", t: "Behind the EUBO variants." },
  ] },
];

// The Chinese edition: group headings, each study's list name, noise, and
// unnamed-dimension tasks, and each rule's label and findings (by study id).
// Row labels (`short`) are names and years and stay as they are.
const GROUPS_ZH = [
  "无噪声或近乎无噪声",
  "效用上的高斯噪声",
  "概率单位或逻辑链接的回答",
  "一定比例的错误回答",
  "未说明噪声模型",
];

const STUDY_ZH: Record<string, { name: string; noise: string; ns?: string }> = {
  pabbo: { name: "Zhang 等 2025（PABBO）", noise: "无噪声的比较", ns: "HPO-B、Candy、Sushi" },
  takeno: { name: "Takeno 等 2023", noise: "噪声方差 10⁻⁴" },
  mikkola: { name: "Mikkola 等 2020", noise: "较小的高斯噪声" },
  siivola: { name: "Siivola 等 2021", noise: "效用噪声标准差 0.05" },
  menn: { name: "Menn 等 2026（局部偏好贝叶斯优化，预印本）", noise: "高斯噪声，取值范围的 10%" },
  fauvel: { name: "Fauvel 与 Chalk 2021（预印本）", noise: "概率单位噪声，归一化后方差为 1", ns: "34 个函数" },
  wu: { name: "Wu 与 Gardner 2026（预印本）", noise: "概率单位噪声" },
  astudillo: { name: "Astudillo 等 2023（qEUBO）", noise: "逻辑噪声，按最优的 1% 点对上 10%、20%、30% 的错误率校准" },
  xu: { name: "Xu 等 2024（POP-BO）", noise: "逻辑噪声", ns: "高斯过程样本" },
  lazzaro: { name: "Lazzaro 等 2026（PF-TS）", noise: "逻辑噪声" },
  lin: { name: "Lin 等 2022（BOPE）", noise: "10% 的错误选择", ns: "多结果问题" },
  gonzalez: { name: "González 等 2017", noise: "未说明" },
  nguyen: { name: "Nguyen 等 2021（MPES）", noise: "未说明", ns: "CIFAR-10 嵌入、SUSHI" },
  koyama: { name: "Koyama 等 2020（序列画廊）", noise: "未说明" },
};

const RULE_ZH: Record<string, { label: string; f: Record<string, string> }> = {
  ts: { label: "汤普森采样（DTS、qTS、PF-TS）", f: {
    gonzalez: "DTS 始终是最好的策略。",
    lazzaro: "PF-TS 的累积遗憾显著低于 MR-LPF 与 POP-BO。",
    nguyen: "DTS 落后于 MPES。",
    siivola: "批量汤普森采样与批量期望改进没有明显差异。",
    takeno: "汤普森采样在 4 维与 6 维 Hartmann 函数上过度探索。",
    astudillo: "qTS 输给了 qEUBO；除汽车驾驶室设计问题外，qEUBO 在每个问题上都最好（q = 2）。",
    mikkola: "成对的 DTS 变体输给了每一个投影变体，这更多是查询形式之间的对比，而不是规则之间的对比。",
    fauvel: "DTS 在九个规则中排第四；基于汤普森采样的规则表现平平。",
  } },
  ei: { label: "期望改进（Brochu EI、CEI、qEI）", f: {
    gonzalez: "CEI 过度利用，且计算代价过高，除 Forrester 函数外无法运行；Brochu 等人的交互式贝叶斯优化表现不佳。",
    nguyen: "期望改进落后于 MPES。",
    siivola: "批量期望改进与批量汤普森采样没有明显差异。",
    takeno: "期望传播加期望改进常因过度利用而停滞。",
    astudillo: "qEI 在运行后期停滞（7 维 Alpine1 函数），且可证明不具一致性。",
    fauvel: "Brochu 等人的期望改进在九个规则中排第八；只有随机查询更差。",
  } },
  muc: { label: "最大不确定挑战（MUC）", f: {
    takeno: "总体落后于幻觉信念，与之结合时表现相对较差。",
    fauvel: "在 34 个函数上按 Borda 排名并列第一。",
  } },
  ducb: { label: "对决上置信界与 EIIG", f: {
    fauvel: "对决上置信界按 Borda 排名并列第一；EIIG 排第七。",
  } },
  eubo: { label: "EUBO 与 qEUBO", f: {
    wu: "EUBO 的查询向估计的最大值处坍缩。",
    pabbo: "与 qEUBO 及其他高斯过程基线相比，PABBO 在多数任务上排第一或第二。",
    astudillo: "除汽车驾驶室设计问题外，qEUBO 在每个问题上都最好（q = 2）。",
    xu: "qEUBO 报告的解略好，但累积遗憾是 POP-BO 的 2.5 倍以上。",
    menn: "在陡峭的最优点上，局部方法优于 qEUBO。",
    lin: "EUBO 的各个变体最好。",
  } },
  hb: { label: "幻觉信念（HB）", f: {
    takeno: "在 12 个函数上总体最好。",
    xu: "在逻辑噪声下陷入局部最优。",
    menn: "幻觉信念加期望改进是一个基线；局部方法在陡峭的最优点上更好。",
  } },
  mpes: { label: "多项预测熵搜索（MPES）", f: {
    nguyen: "始终最好。",
    pabbo: "作为基线；PABBO 在多数任务上排第一或第二。",
    astudillo: "输给 qEUBO，且最慢：每次迭代 12.7 至 24.8 秒，而 qEUBO 约为 7 至 12 秒。",
  } },
  forms: { label: "投影与画廊", f: {
    mikkola: "每一个投影变体都明显胜过每一个成对变体。",
    koyama: "画廊胜过线搜索与随机构造平面。",
  } },
  regret: { label: "有遗憾界的算法（POP-BO、PF-TS）", f: {
    lazzaro: "PF-TS 的累积遗憾显著低于 MR-LPF 与 POP-BO。",
    xu: "qEUBO 的累积遗憾是 POP-BO 的 2.5 倍以上。",
  } },
  pabbo: { label: "PABBO（摊销）", f: {
    pabbo: "在多数任务上排第一或第二；在 6 维 Hartmann 函数上较弱。",
  } },
  local: { label: "局部偏好贝叶斯优化", f: {
    menn: "在陡峭的最优点上更好；作者称它从较差的起点出发时并不可靠。",
  } },
  random: { label: "随机查询", f: {
    pabbo: "随机查询常常胜过某些高斯过程基线。",
    siivola: "在信噪比低的真实数据上，每种方法都只是勉强胜过基线。",
    mikkola: "成对的随机查询输给了每一个投影变体。",
    astudillo: "落后于 qEUBO。",
    fauvel: "在九个规则中排最后。",
    lin: "落后于 EUBO 的各个变体。",
  } },
};

const labels = {
  en: {
    dim: "input dimension (log scale)",
    ns: "not given",
    good: "favorable",
    mixed: "mixed",
    bad: "unfavorable",
    tested: "tested here",
    upto: "lower dimensions also tested",
    describe: "Evidence map for {rule}: {n} stud{n:y/ies} report on it, {g} favorable, {m} mixed, {b} unfavorable. Each study is placed by its noise model and the input dimensions it tested.",
  },
  zh: {
    dim: "输入维度（对数刻度）",
    ns: "未给出",
    good: "有利",
    mixed: "好坏参半",
    bad: "不利",
    tested: "在此测试",
    upto: "也测试了更低的维度",
    describe: "证据图：{rule}。共 {n} 项研究报告了这一规则，其中有利 {g} 项、好坏参半 {m} 项、不利 {b} 项。每项研究按其噪声模型与所测试的输入维度放置。",
  },
};

const params = {
  rule: { kind: "choice", label: { en: "Rule", zh: "规则" }, control: "select", default: "ts", options: RULES.map((r) => ({ value: r.id, label: { en: r.label, zh: RULE_ZH[r.id].label } })) },
} as const;

type P = { rule: string };

const ruleOf = (id: string) => RULES.find((r) => r.id === id) ?? RULES[0];
const VCOL: Record<Verdict, string> = { good: C.good, mixed: C.warn, bad: C.bad };

// Order findings by the largest dimension the study names, studies that give
// no dimension last, so the list reads from low to high dimension.
const maxDim = (s: Study) => (s.dims.length ? Math.max(...s.dims) : Infinity);
function ordered(r: Rule): Array<Finding & { st: Study }> {
  return r.f.map((f) => ({ ...f, st: STUDIES.find((s) => s.id === f.s)! })).sort((a, b) => maxDim(a.st) - maxDim(b.st));
}

function wrap(s: string, max: number): string[] {
  const out: string[] = [];
  let line = "";
  for (const word of s.split(/\s+/)) {
    if (line && (line + " " + word).length > max) { out.push(line); line = word; }
    else line = line ? `${line} ${word}` : word;
  }
  if (line) out.push(line);
  return out;
}

// Wrap Chinese text, which has no spaces between words, to lines of at most
// `max` Latin-letter widths: a Chinese character counts 1.9, a run of Latin
// letters, digits, and symbols is kept whole, and a line never starts with
// closing punctuation.
function wrapZh(s: string, max: number): string[] {
  const toks = s.match(/[　-〿一-鿿＀-￯]|[^\s　-〿一-鿿＀-￯]+|\s+/g) ?? [];
  const width = (t: string) => labelWidth(t, 1, 1.9);
  const lines: string[] = [];
  let line = "", w = 0;
  for (const t of toks) {
    const tw = width(t);
    if (line && w + tw > max && !/^[，。；：、）？！”]$/.test(t) && !/^\s+$/.test(t)) { lines.push(line.trimEnd()); line = ""; w = 0; }
    if (!line && /^\s+$/.test(t)) continue;
    line += t; w += tw;
  }
  if (line.trim()) lines.push(line.trimEnd());
  return lines;
}

const wrapIn = (lang: Lang, s: string, max: number) => (lang === "zh" ? wrapZh(s, max) : wrap(s, max));

// The list line of a study: its name, the dimensions it tested, and its noise.
function headOf(st: Study, lang: Lang): string {
  if (lang === "zh") {
    const z = STUDY_ZH[st.id];
    const dimsText = st.dims.length
      ? `${st.lo != null ? `至多 ${Math.max(...st.dims)}` : st.dims.join(st.span ? " 至 " : "、")} 维${z.ns ? `，另有 ${z.ns}` : ""}`
      : `${z.ns}，未给出维度`;
    return `${z.name} · ${dimsText} · ${z.noise}`;
  }
  const dimsText = st.dims.length
    ? `${st.lo != null ? `up to ${Math.max(...st.dims)}` : st.dims.join(st.span ? " to " : ", ")}-D${st.ns ? `, plus ${st.ns}` : ""}`
    : `${st.ns}, dimension not given`;
  return `${st.name} · ${dimsText} · ${st.noise}`;
}

function describe(st: State<P>): string {
  const lang = st.lang ?? "en";
  const r = ruleOf(st.p.rule);
  const c = (v: Verdict) => r.f.filter((f) => f.v === v).length;
  return tpl(labels[lang].describe, { rule: lang === "zh" ? RULE_ZH[r.id].label : r.label, n: r.f.length, g: c("good"), m: c("mixed"), b: c("bad") });
}

function render(st: State<P>): string {
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 480;
  const rule = ruleOf(st.p.rule);
  const list = ordered(rule);
  const badge = new Map(list.map((f, i) => [f.s, { n: i + 1, v: f.v }]));
  const parts: string[] = [];

  const labelW = narrow ? 84 : 112;
  const nsW = narrow ? 46 : 66;
  const left = labelW + 8;
  const right = st.w - nsW - 14;
  const nsX = st.w - nsW / 2 - 6;
  const x = log([1, 120], [left + 8, right - 6]);
  const rowH = 18, headH = 18;
  let y = 12;

  // column header for the "not given" column
  parts.push(text(nsX, y + 6, L.ns, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }));
  y += 8;
  const top = y;
  const rowsY = new Map<string, number>();
  const heads: string[] = [];
  GROUPS.forEach((gname, gi) => {
    y += headH;
    heads.push(text(4, y - 5, lang === "zh" ? GROUPS_ZH[gi] : gname, { "font-size": TYPE.small, class: "fig-t-strong fig-t-halo" }));
    for (const s of STUDIES.filter((q) => q.group === gi)) {
      y += rowH;
      rowsY.set(s.id, y - rowH / 2);
    }
  });
  const bottom = y + 4;

  // gridlines and the not-given column
  for (const d of [1, 2, 5, 10, 20, 50, 100]) parts.push(el("line", { x1: x(d), x2: x(d), y1: top, y2: bottom, stroke: C.grid }));
  parts.push(el("rect", { x: st.w - nsW - 8, y: top, width: nsW, height: bottom - top, fill: C.panel, rx: 3 }));
  parts.push(...heads);

  // study rows
  for (const s of STUDIES) {
    const cy = rowsY.get(s.id)!;
    const b = badge.get(s.id);
    const col = b ? VCOL[b.v] : C.ink3;
    const op = b ? 1 : 0.45;
    parts.push(text(narrow ? 10 : 14, cy + 4, s.short, { "font-size": TYPE.small, class: b ? "fig-t-strong" : "fig-t-muted" }));
    const xs = s.dims.map(x);
    if (s.lo != null && s.dims.length) {
      parts.push(el("line", { x1: x(s.lo), x2: Math.min(...xs), y1: cy, y2: cy, stroke: col, "stroke-width": b ? 2 : 1.4, "stroke-dasharray": "3 3", opacity: op }));
    }
    if (xs.length > 1) parts.push(el("line", { x1: Math.min(...xs), x2: Math.max(...xs), y1: cy, y2: cy, stroke: col, "stroke-width": b ? 4 : 2, "stroke-linecap": "round", opacity: op }));
    for (const px of xs) parts.push(el("circle", { cx: px, cy, r: b ? 4 : 3, fill: col, stroke: C.paper, "stroke-width": 1.2, opacity: op }));
    if (s.ns) parts.push(el("rect", { x: nsX - 5, y: cy - 5, width: 10, height: 10, rx: 2, fill: col, opacity: op }));
    if (b) {
      const bx = s.dims.length ? Math.max(...xs) + 13 : nsX + 15;
      parts.push(el("circle", { cx: bx, cy, r: 7.5, fill: C.paper, stroke: col, "stroke-width": 1.6 }), text(bx, cy + 4, b.n, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-strong fig-t-num" }));
    }
  }

  // axis
  parts.push(el("line", { x1: left, x2: right, y1: bottom, y2: bottom, stroke: C.rule }));
  for (const d of [1, 2, 5, 10, 20, 50, 100]) {
    parts.push(el("line", { x1: x(d), x2: x(d), y1: bottom, y2: bottom + 4, stroke: C.rule }));
    parts.push(text(x(d), bottom + 16, d, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  }
  parts.push(text((left + right) / 2, bottom + 31, L.dim, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }));

  // legend
  let ly = bottom + 52;
  let lx = 4;
  const items: Array<[string, string, "dot" | "dash"]> = [[C.good, L.good, "dot"], [C.warn, L.mixed, "dot"], [C.bad, L.bad, "dot"], [C.ink3, L.upto, "dash"]];
  for (const [c, lab, kind] of items) {
    const wv = labelWidth(lab, 6) + 28;
    if (lx + wv > st.w - 4 && lx > 4) { lx = 4; ly += 18; }
    parts.push(kind === "dot"
      ? el("circle", { cx: lx + 6, cy: ly - 4, r: 4.5, fill: c })
      : el("line", { x1: lx, x2: lx + 16, y1: ly - 4, y2: ly - 4, stroke: c, "stroke-width": 2, "stroke-dasharray": "3 3" }));
    parts.push(text(lx + (kind === "dot" ? 15 : 22), ly, lab, { "font-size": TYPE.small, class: "fig-t-muted" }));
    lx += wv;
  }

  // findings list
  let py = ly + 14;
  parts.push(el("line", { x1: 4, x2: st.w - 4, y1: py, y2: py, stroke: C.rule }));
  py += 6;
  const textX = 26;
  const maxChars = Math.floor((st.w - textX - 6) / 6.3);
  list.forEach((f, i) => {
    py += 18;
    parts.push(el("circle", { cx: 12, cy: py - 4, r: 7.5, fill: C.paper, stroke: VCOL[f.v], "stroke-width": 1.6 }), text(12, py, i + 1, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-strong fig-t-num" }));
    const head = headOf(f.st, lang);
    wrapIn(lang, head, maxChars).forEach((line, k) => { if (k) py += 15; parts.push(text(textX, py, line, { "font-size": TYPE.small, class: "fig-t-muted" })); });
    const finding = lang === "zh" ? RULE_ZH[rule.id].f[f.s] : f.t;
    for (const line of wrapIn(lang, finding, Math.floor((st.w - textX - 6) / 6.6))) { py += 16; parts.push(text(textX, py, line, { "font-size": TYPE.body })); }
  });

  return svg(st.w, py + 10, describe(st), g({}, ...parts));
}

export default defineFigure({
  name: "acqf-evidence-map",
  title: { en: "Where each finding about an acquisition rule was observed", zh: "关于采集规则的每项发现是在什么条件下观察到的" },
  labels,
  params,
  hint: { en: "Choose a rule to see which studies report on it, at which dimensions and under which noise.", zh: "选择一个规则，查看哪些研究报告了它，以及是在哪些维度、何种噪声下。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
