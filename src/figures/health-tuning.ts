// Exoskeleton and prosthesis tuning studies side by side. Two views share a
// study selector and a detail panel. "Time and benefit" plots the minutes of
// walking spent tuning (log scale) against the reported metabolic reduction,
// with error bars as reported, colored by what judged each setting; a strip
// below holds results with a time but no comparable metabolic number. "Who
// took part" plots participants against evaluations per person for the
// preference and self-tuning studies. All numbers are the ones reported in the
// chapter; the baselines differ, which the detail panel states for each study.

import { defineFigure, type Lang, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear, log } from "./lib/scale.ts";
import { tpl } from "./lib/format.ts";

const labels = {
  en: {
    time: "minutes of walking spent tuning (log scale)",
    reduction: "metabolic reduction (%)",
    timeOnly: "time only",
    evals: "evaluations per person (log scale)",
    people: "participants",
    clinical: "hollow: includes clinical participants",
    describeTime: "Time and benefit. Selected: {name}: {detail}",
    describeWho: "Who took part. Selected: {name}: {detail}",
    notIn: "not shown in this view",
    sladeNarrow: "Slade lab: 32 vs 128 min",
    sladeWide: "Slade 2022, lab: sensors (dot) vs respirometry (square)",
    mohilboNarrow: "MO-HILBO",
    mohilboWide: "MO-HILBO, per person",
  },
  zh: {
    time: "调节所用的行走时间（分钟，对数刻度）",
    reduction: "代谢降幅（%）",
    timeOnly: "仅有时间",
    evals: "每人的评估次数（对数刻度）",
    people: "被试人数",
    clinical: "空心：包含临床被试",
    describeTime: "时间与收益。已选：{name}：{detail}",
    describeWho: "参与者。已选：{name}：{detail}",
    notIn: "本视图中未显示",
    sladeNarrow: "Slade 实验室：32 对 128 分钟",
    sladeWide: "Slade 2022，实验室：传感器（圆点）对呼吸测量（方块）",
    mohilboNarrow: "MO-HILBO",
    mohilboWide: "MO-HILBO，每人",
  },
};

type Method = "self" | "pref" | "prefOther" | "metab" | "sensor";
const METHOD_LABEL: Record<Method, string> = {
  self: "the wearer, by hand",
  pref: "the wearer's choices (GP model)",
  prefOther: "the wearer's choices (other model)",
  metab: "measured metabolic cost",
  sensor: "wearable-sensor model",
};
const METHOD_LABEL_ZH: Record<Method, string> = {
  self: "穿戴者手动调节",
  pref: "穿戴者的选择（高斯过程模型）",
  prefOther: "穿戴者的选择（其他模型）",
  metab: "测得的代谢消耗",
  sensor: "可穿戴传感器模型",
};
const methodLabel = (m: Method, lang: Lang) => (lang === "zh" ? METHOD_LABEL_ZH : METHOD_LABEL)[m];
const METHOD_COLOR: Record<Method, string> = { self: C.c2, pref: C.c1, prefOther: C.c7, metab: C.c6, sensor: C.c5 };

interface Study {
  id: string; name: string; method: Method; head: string; detail: string;
  // Time and benefit
  t?: number; tErr?: number; y?: number; yErr?: number; yLo?: number; yHi?: number;
  timeOnly?: Array<{ t: number; t2?: number; label: string }>;
  // Who took part
  n?: number; ev?: number; evLo?: number; evHi?: number; clinical?: boolean;
}

// Numbers as reported in the chapter (see its text for sources and caveats).
const STUDIES: Study[] = [
  { id: "schafer", name: "Schäfer et al. 2026", method: "self", t: 10.9, tErr: 0.9, y: 16.6, yErr: 1.1, n: 11, ev: 30.5, evLo: 16, evHi: 111,
    head: "hip exoskeleton, 4 timing parameters, 11 healthy adults",
    detail: "Self-tuned with a thumbstick remote control in 10.9 ± 0.9 min (at most 16.2), median 30.5 settings (16 to 111). Metabolic cost 16.6 ± 1.1% lower than walking with the exoskeleton at zero torque; ±8% timing changes made no significant difference." },
  { id: "liu", name: "Liu et al. 2026 (preprint)", method: "pref", t: 20.6, tErr: 4.6, yLo: 14.5, yHi: 15.4, n: 5, ev: 20,
    head: "hip exoskeleton, 6 parameters, 5 healthy adults",
    detail: "Pairwise preference BO, 20 iterations, 90.7% validation agreement. Sessions of 20.6 ± 4.6 min including validation. Metabolic rate 14.5% to 15.4% lower than with assistance off (8.6% to 13.0% lower than without the exoskeleton), measured on 2 to 3 of the participants." },
  { id: "ding", name: "Ding et al. 2018", method: "metab", t: 21.4, tErr: 1.0, y: 17.4, yErr: 3.2,
    head: "soft hip exosuit, 2 timing parameters, 8 healthy adults",
    detail: "Bayesian optimization of measured metabolic cost. Optimum found in 21.4 ± 1.0 min, a convergence time computed afterward from a 40-minute optimization; metabolic cost 17.4 ± 3.2% lower than walking without the device (mean ± standard error)." },
  { id: "slade", name: "Slade et al. 2022", method: "sensor", t: 60, y: 23, yErr: 8, timeOnly: [{ t: 32, t2: 128, label: "lab: 32 min with sensors, 128 min with respirometry (n = 9)" }],
    head: "ankle exoskeleton, data-driven optimization from wearable sensors, 10 adults",
    detail: "After 1 h of naturalistic walking outdoors, metabolic cost on a treadmill at 1.5 m/s was 23 ± 8% lower than in normal shoes. In the lab, the sensor-based method found parameters within 5% of metabolic optimization in 32 instead of 128 min of walking (n = 9)." },
  { id: "mohilbo", name: "MO-HILBO 2026 (preprint)", method: "pref", timeOnly: [{ t: 180, t2: 240, label: "3 to 4 h per participant, with validation" }],
    head: "hip-knee exoskeleton, 3 parameters, 3 participants",
    detail: "Metabolic cost and ordinal comfort optimized together; 94% (17 of 18) of Pareto-front rankings predicted. No single metabolic reduction is reported for comparison." },
  { id: "cospar", name: "CoSpar 2020", method: "pref", n: 3, ev: 20,
    head: "Atalante exoskeleton, step length (1 parameter, also 2), 3 able-bodied",
    detail: "Pairwise preferences plus suggested improvements, 20 gait trials; blind rankings of 3 gaits matched the posterior. No objective outcome measured." },
  { id: "linecospar", name: "LineCoSpar 2020", method: "pref", n: 6, ev: 36,
    head: "Atalante exoskeleton, 6 gait parameters, 6 able-bodied novices",
    detail: "30 trials plus 6 for validation; agreement on 4 validation preferences was 75%, 100%, 100%, 25%, 100%, 100%." },
  { id: "roial", name: "ROIAL 2021", method: "pref", n: 3, ev: 40,
    head: "Atalante exoskeleton, 4 parameters, 3 able-bodied",
    detail: "Pairwise plus ordinal labels, 40 trials (30 training, 10 validation); most ordinal predictions within one level; explored under 2% of the space." },
  { id: "tucker", name: "Tucker thesis 2023", method: "pref", n: 2, ev: 35, evLo: 30, evHi: 40, clinical: true,
    head: "Atalante exoskeleton, 3 parameters, 2 people with complete paraplegia",
    detail: "ROIAL for 15 iterations, then LineCoSpar the next day for 15 iterations (one participant) and 25 (the other), 30 and 40 in all; one evaluation each, both rated the best gait good. 8 able-bodied participants are reported separately." },
  { id: "wander", name: "WANDER 2024 (GLISp)", method: "prefOther", n: 12, ev: 15,
    head: "walking-aid robot, 2 admittance parameters, 12 healthy adults",
    detail: "Radial basis function surrogate, up to 15 iterations. Against two literature settings, linear energy fell 17.61% and 13.93%; other measures partly significant." },
  { id: "ramella", name: "Ramella et al. 2025", method: "prefOther", n: 8, ev: 12,
    head: "hip exoskeleton, 6 torque features, 8 healthy",
    detail: "Linear reward with a sampled posterior, 12 pairwise comparisons; most kept their preference against perturbed versions." },
  { id: "arens", name: "Arens et al. 2025", method: "pref", n: 15, ev: 45,
    head: "soft back exosuit, 2 parameters, 15 healthy",
    detail: "3 blocks of 15 iterations with JND-aware UCB; ICC 0.80 (lowering) and 0.91 (lifting); preferred assistance rose 15% and 29% across blocks (not significant)." },
  { id: "taddei", name: "Taddei et al. 2026 (preprint)", method: "pref", n: 4, ev: 10.3, evHi: 35, clinical: true,
    head: "active prosthesis, 4 parameters, 2 transfemoral amputees and 2 able-bodied",
    detail: "Discrete version 10.3 ± 2.5 iterations (93% validation); continuous version 35.0 ± 6.2 (67%). One amputee's gait asymmetry improved." },
  { id: "diaz", name: "Díaz et al. 2026", method: "self", n: 3, evLo: 8, evHi: 14, ev: 11, clinical: true,
    head: "powered knee-ankle prosthesis, 2 parameters, 3 transfemoral amputees",
    detail: "Blind self-exploration on a touchscreen grid, 8 to 14 settings per trial; 2 of 3 chose inconsistent settings across trials of the same day." },
];

// The Chinese edition's name, setting, and result of each study.
const ZH: Record<string, { name: string; head: string; detail: string }> = {
  schafer: { name: "Schäfer 等 2026", head: "髋外骨骼，4 个时机参数，11 名健康成人",
    detail: "用拇指杆遥控器自调，用时 10.9 ± 0.9 分钟（最长 16.2 分钟），尝试设置数的中位数为 30.5 个（16 至 111）。代谢消耗比穿着外骨骼以零力矩行走时低 16.6 ± 1.1%；±8% 的时机变化没有造成显著差异。" },
  liu: { name: "Liu 等 2026（预印本）", head: "髋外骨骼，6 个参数，5 名健康成人",
    detail: "成对偏好贝叶斯优化，20 次迭代，验证一致率 90.7%。会话用时 20.6 ± 4.6 分钟（含验证）。代谢率比助力关闭时低 14.5% 至 15.4%（比不穿外骨骼时低 8.6% 至 13.0%），在其中 2 至 3 名被试身上测得。" },
  ding: { name: "Ding 等 2018", head: "柔性髋部外骨骼服，2 个时机参数，8 名健康成人",
    detail: "以测得的代谢消耗为目标的贝叶斯优化。在 21.4 ± 1.0 分钟内找到最优点，这一收敛时间是事后根据一次 40 分钟的优化计算的；代谢消耗比不穿装置行走时低 17.4 ± 3.2%（均值 ± 标准误）。" },
  slade: { name: "Slade 等 2022", head: "踝外骨骼，基于可穿戴传感器的数据驱动优化，10 名成人",
    detail: "在户外自然行走 1 小时后，在跑步机上以 1.5 m/s 行走的代谢消耗比穿普通鞋时低 23 ± 8%。在实验室中，基于传感器的方法用 32 分钟而不是 128 分钟的行走，就找到了与代谢优化相差 5% 以内的参数（n = 9）。" },
  mohilbo: { name: "MO-HILBO 2026（预印本）", head: "髋膝外骨骼，3 个参数，3 名被试",
    detail: "同时优化代谢消耗与序数舒适度；预测对了 94%（18 个中的 17 个）帕累托前沿排序。没有报告可供比较的单一代谢降幅。" },
  cospar: { name: "CoSpar 2020", head: "Atalante 外骨骼，步长（1 个参数，也有 2 个），3 名身体健全者",
    detail: "成对偏好加上用户建议的改进，20 次步态试验；对 3 种步态的盲排序与后验一致。未测量客观结果。" },
  linecospar: { name: "LineCoSpar 2020", head: "Atalante 外骨骼，6 个步态参数，6 名身体健全的新手",
    detail: "30 次试验，另加 6 次验证；在 4 个验证偏好上的一致率为 75%、100%、100%、25%、100%、100%。" },
  roial: { name: "ROIAL 2021", head: "Atalante 外骨骼，4 个参数，3 名身体健全者",
    detail: "成对偏好加序数标签，40 次试验（30 次训练，10 次验证）；大多数序数预测相差不超过一级；探索了不到 2% 的空间。" },
  tucker: { name: "Tucker 学位论文 2023", head: "Atalante 外骨骼，3 个参数，2 名完全性截瘫患者",
    detail: "先用 ROIAL 进行 15 次迭代，次日再用 LineCoSpar 进行 15 次（一名被试）和 25 次（另一名）迭代，合计 30 次和 40 次；每人一次评估，两人都把最佳步态评为好。另有 8 名身体健全的被试单独报告。" },
  wander: { name: "WANDER 2024（GLISp）", head: "助行机器人，2 个导纳参数，12 名健康成人",
    detail: "径向基函数代理模型，至多 15 次迭代。与文献中的两种设置相比，线性能量降低 17.61% 和 13.93%；其他指标部分显著。" },
  ramella: { name: "Ramella 等 2025", head: "髋外骨骼，6 个力矩特征，8 名健康者",
    detail: "带采样后验的线性奖励，12 次成对比较；面对扰动后的版本，大多数人保持了原偏好。" },
  arens: { name: "Arens 等 2025", head: "柔性背部外骨骼服，2 个参数，15 名健康者",
    detail: "3 个区组，每组 15 次迭代，使用考虑最小可觉差的上置信界；ICC 为 0.80（放下）和 0.91（抬举）；偏好的助力在各区组间上升 15% 和 29%（不显著）。" },
  taddei: { name: "Taddei 等 2026（预印本）", head: "主动假肢，4 个参数，2 名经股骨截肢者和 2 名身体健全者",
    detail: "离散版本 10.3 ± 2.5 次迭代（验证率 93%）；连续版本 35.0 ± 6.2 次（67%）。一名截肢者的步态不对称性得到改善。" },
  diaz: { name: "Díaz 等 2026", head: "动力膝踝假肢，2 个参数，3 名经股骨截肢者",
    detail: "在触摸屏网格上不知道数值地自主探索，每次试验 8 至 14 个设置；3 人中有 2 人在同一天的不同试验中选择的设置不一致。" },
};
const nameOf = (s: Study, lang: Lang) => (lang === "zh" ? ZH[s.id].name : s.name);
const headOf = (s: Study, lang: Lang) => (lang === "zh" ? ZH[s.id].head : s.head);
const detailOf = (s: Study, lang: Lang) => (lang === "zh" ? ZH[s.id].detail : s.detail);

const VIEWS = [
  { value: "time", label: { en: "Time and benefit", zh: "时间与收益" } },
  { value: "who", label: { en: "Who took part", zh: "参与者" } },
] as const;

const params = {
  view: { kind: "choice", label: { en: "View", zh: "视图" }, options: VIEWS, default: "time" },
  study: { kind: "choice", label: { en: "Study", zh: "研究" }, options: STUDIES.map((s) => ({ value: s.id, label: { en: s.name, zh: ZH[s.id].name } })), default: "schafer", control: "select" },
} as const;

type P = { view: "time" | "who"; study: string };
const pick = (id: string) => STUDIES.find((s) => s.id === id) ?? STUDIES[0];
const inTime = (s: Study) => s.t !== undefined || !!s.timeOnly;
const inWho = (s: Study) => s.n !== undefined;

function describe(st: State<P>): string {
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const s = pick(st.p.study);
  const detail = lang === "zh" ? `${headOf(s, lang)}。${detailOf(s, lang)}` : `${s.head}. ${s.detail}`;
  return tpl(st.p.view === "time" ? L.describeTime : L.describeWho, { name: nameOf(s, lang), detail });
}

function wrap(s: string, width: number, px = 6.2): string[] {
  const max = Math.max(12, Math.floor(width / px));
  const out: string[] = [];
  let line = "";
  for (const w of s.split(" ")) {
    if (line && (line + " " + w).length > max) { out.push(line); line = w; } else line = line ? `${line} ${w}` : w;
  }
  if (line) out.push(line);
  return out;
}

// Wrap Chinese text, which has no spaces between words, to `width` pixels: a
// Chinese character or full-width mark is `em` wide, a run of Latin letters,
// digits, and symbols (with a "± error" after it) is kept whole at `px` per
// character, and a line never starts with closing punctuation.
function wrapZh(s: string, width: number, px: number, em: number): string[] {
  const toks = s.match(/[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef“”‘’]|[^\s\u3000-\u303f\u4e00-\u9fff\uff00-\uffef“”‘’]+(?: ± [^\s\u3000-\u303f\u4e00-\u9fff\uff00-\uffef“”‘’]+)?|\s+/g) ?? [];
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

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 480;
  const sel = pick(p.study);
  const parts: string[] = [];
  const lx = narrow ? 8 : 12;

  // Legend: what judged each setting.
  let cx = lx, ly = 14;
  const methods: Method[] = p.view === "time" ? ["self", "pref", "metab", "sensor"] : ["self", "pref", "prefOther"];
  for (const m of methods) {
    const wpx = labelWidth(methodLabel(m, lang), 6.1) + 26;
    if (cx + wpx > st.w - 8 && cx > lx) { cx = lx; ly += 18; }
    parts.push(el("circle", { cx: cx + 6, cy: ly - 4, r: 5, fill: METHOD_COLOR[m] }), text(cx + 15, ly, methodLabel(m, lang), { "font-size": TYPE.small, class: "fig-t-muted" }));
    cx += wpx;
  }
  if (p.view === "who") { ly += 18; parts.push(el("circle", { cx: lx + 6, cy: ly - 4, r: 4.2, fill: C.paper, stroke: C.ink2, "stroke-width": 2 }), text(lx + 15, ly, L.clinical, { "font-size": TYPE.small, class: "fig-t-muted" })); }

  const left = narrow ? 44 : 56;
  const right = st.w - 14;
  const top = ly + 20;
  const ph = narrow ? 200 : 230;
  const bottom = top + ph;
  const hits: string[] = [];
  const marks: string[] = [];

  const dot = (s: Study, x: number, y: number, hollow = false) => {
    const isSel = s.id === sel.id;
    const col = METHOD_COLOR[s.method];
    marks.push(el("circle", { cx: x, cy: y, r: isSel ? 6.5 : 5.2, fill: hollow ? C.paper : col, stroke: hollow ? col : C.paper, "stroke-width": hollow ? 2.2 : 1.2 }));
    if (isSel) marks.push(el("circle", { cx: x, cy: y, r: 10.5, fill: "none", stroke: C.ink, "stroke-width": 1.6 }));
    hits.push(g({ "data-fig-set": `study=${s.id}`, style: "cursor:pointer" }, el("circle", { cx: x, cy: y, r: 11, fill: "transparent" })));
  };

  let H: number;
  if (p.view === "time") {
    const x = log([5, 300], [left + 6, right - 8]);
    const y = linear([0, 32], [bottom, top]);
    for (const v of [0, 10, 20, 30]) parts.push(el("line", { x1: left, x2: right, y1: y(v), y2: y(v), stroke: C.grid }), text(left - 6, y(v) + 4, String(v), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-num" }));
    for (const v of [5, 10, 20, 50, 100, 200]) parts.push(el("line", { x1: x(v), x2: x(v), y1: top, y2: bottom, stroke: C.grid }));
    parts.push(el("line", { x1: left, x2: right, y1: bottom, y2: bottom, stroke: C.rule }));
    parts.push(text(0, 0, L.reduction, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted", transform: `translate(${narrow ? 12 : 16},${(top + bottom) / 2}) rotate(-90)` }));
    for (const s of STUDIES.filter((q) => q.t !== undefined)) {
      const col = METHOD_COLOR[s.method];
      const px = x(s.t!);
      if (s.tErr) marks.push(el("line", { x1: x(s.t! - s.tErr), x2: x(s.t! + s.tErr), y1: y(s.y ?? (s.yLo! + s.yHi!) / 2), y2: y(s.y ?? (s.yLo! + s.yHi!) / 2), stroke: col, "stroke-width": 1.6 }));
      if (s.yErr !== undefined) marks.push(el("line", { x1: px, x2: px, y1: y(s.y! - s.yErr), y2: y(s.y! + s.yErr), stroke: col, "stroke-width": 1.6 }));
      if (s.yLo !== undefined) marks.push(el("rect", { x: px - 4, y: y(s.yHi!), width: 8, height: y(s.yLo!) - y(s.yHi!), rx: 3, fill: col, opacity: 0.9 }));
      dot(s, px, y(s.y ?? (s.yLo! + s.yHi!) / 2));
      const lab = s.name.replace(/ \(preprint\)/, "").replace(/ et al\./, "");
      // Hand-placed labels: [dx, dy, anchor] for each point.
      const place: Record<string, [number, number, string]> = narrow
        ? { schafer: [0, -14, "middle"], ding: [0, -26, "middle"], liu: [10, 18, "start"], slade: [10, 4, "start"] }
        : { schafer: [0, -14, "middle"], ding: [10, -10, "start"], liu: [10, 18, "start"], slade: [-12, -6, "end"] };
      const [dx, dy, anchor] = place[s.id] ?? [10, -8, "start"];
      const cy = y(s.y ?? (s.yLo! + s.yHi!) / 2);
      marks.push(text(px + dx, cy + dy, lab, { "font-size": TYPE.small, "text-anchor": anchor, class: s.id === sel.id ? "fig-t-strong fig-t-halo" : "fig-t-muted fig-t-halo" }));
    }
    // Time-only strip.
    const sy0 = bottom + 26;
    const rows = STUDIES.filter((q) => q.timeOnly);
    parts.push(text(left, sy0 - 8, L.timeOnly, { "font-size": TYPE.small, class: "fig-t-faint" }));
    rows.forEach((s, i) => {
      const ry = sy0 + 14 + i * 24;
      const col = METHOD_COLOR[s.method];
      for (const seg of s.timeOnly!) {
        if (s.id === "slade") {
          // two separate marks: sensor-based (32) and respirometry (128)
          marks.push(el("line", { x1: x(seg.t), x2: x(seg.t2!), y1: ry, y2: ry, stroke: C.rule, "stroke-dasharray": "3 3" }));
          marks.push(el("rect", { x: x(seg.t2!) - 4, y: ry - 4, width: 8, height: 8, fill: METHOD_COLOR.metab }));
          dot(s, x(seg.t), ry);
        } else {
          marks.push(el("rect", { x: x(seg.t), y: ry - 4, width: x(seg.t2!) - x(seg.t), height: 8, rx: 4, fill: col, opacity: 0.85 }));
          hits.push(g({ "data-fig-set": `study=${s.id}`, style: "cursor:pointer" }, el("rect", { x: x(seg.t) - 4, y: ry - 9, width: x(seg.t2!) - x(seg.t) + 8, height: 18, fill: "transparent" })));
          if (s.id === sel.id) { marks.push(el("rect", { x: x(seg.t) - 4, y: ry - 8, width: x(seg.t2!) - x(seg.t) + 8, height: 16, rx: 7, fill: "none", stroke: C.ink, "stroke-width": 1.6 })); }
        }
        const cls = s.id === sel.id ? "fig-t-strong fig-t-halo" : "fig-t-muted fig-t-halo";
        if (s.id === "slade") marks.push(text((x(seg.t) + x(seg.t2!)) / 2, ry - 7, narrow ? L.sladeNarrow : L.sladeWide, { "font-size": TYPE.small, "text-anchor": "middle", class: cls }));
        else marks.push(text(x(seg.t) - 8, ry + 4, narrow ? L.mohilboNarrow : L.mohilboWide, { "font-size": TYPE.small, "text-anchor": "end", class: cls }));
      }
    });
    const axY = sy0 + 14 + rows.length * 24 - 2;
    parts.push(el("line", { x1: left, x2: right, y1: axY - 8, y2: axY - 8, stroke: C.rule }));
    for (const v of [5, 10, 20, 50, 100, 200]) parts.push(text(x(v), axY + 8, String(v), { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted fig-t-num" }));
    parts.push(text((left + right) / 2, axY + 24, L.time, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted" }));
    H = axY + 34;
  } else {
    const x = log([5, 120], [left + 6, right - 8]);
    const y = linear([0, 16], [bottom, top]);
    for (const v of [0, 5, 10, 15]) parts.push(el("line", { x1: left, x2: right, y1: y(v), y2: y(v), stroke: C.grid }), text(left - 6, y(v) + 4, String(v), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-num" }));
    for (const v of [5, 10, 20, 50, 100]) parts.push(el("line", { x1: x(v), x2: x(v), y1: top, y2: bottom, stroke: C.grid }), text(x(v), bottom + 16, String(v), { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted fig-t-num" }));
    parts.push(el("line", { x1: left, x2: right, y1: bottom, y2: bottom, stroke: C.rule }));
    parts.push(text(0, 0, L.people, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted", transform: `translate(${narrow ? 12 : 16},${(top + bottom) / 2}) rotate(-90)` }));
    parts.push(text((left + right) / 2, bottom + 32, L.evals, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted" }));
    // Every point is labeled; offsets are placed by hand so labels clear the
    // range bars and each other at both widths.
    const place: Record<string, [number, number, string]> = {
      cospar: [9, 4, "start"], linecospar: [9, 4, "start"], roial: [9, 4, "start"], tucker: [9, 4, "start"],
      wander: narrow ? [-9, 4, "end"] : [9, 4, "start"], ramella: [9, 4, "start"], arens: [-9, 4, "end"], liu: [9, 4, "start"],
      taddei: [0, -9, "start"], schafer: narrow ? [0, 17, "middle"] : [0, -9, "middle"], diaz: [0, 16, "middle"],
    };
    const short: Record<string, string> = {
      cospar: "CoSpar", linecospar: "LineCoSpar", roial: "ROIAL", tucker: "Tucker thesis", wander: "WANDER", ramella: "Ramella",
      arens: "Arens", liu: "Liu", taddei: "Taddei", schafer: "Schäfer", diaz: "Díaz",
    };
    if (lang === "zh") short.tucker = "Tucker 学位论文";
    for (const s of STUDIES.filter(inWho)) {
      const py = y(s.n!);
      const px = x(s.ev!);
      const col = METHOD_COLOR[s.method];
      if (s.evLo !== undefined || s.evHi !== undefined) marks.push(el("line", { x1: x(s.evLo ?? s.ev!), x2: x(s.evHi ?? s.ev!), y1: py, y2: py, stroke: col, "stroke-width": 1.6, opacity: 0.8 }));
      dot(s, px, py, !!s.clinical);
      const [dx, dy, anchor] = place[s.id] ?? [9, 4, "start"];
      marks.push(text(px + dx, py + dy, short[s.id] ?? nameOf(s, lang), { "font-size": TYPE.small, "text-anchor": anchor, class: s.id === sel.id ? "fig-t-strong fig-t-halo" : "fig-t-muted fig-t-halo" }));
    }
    H = bottom + 42;
  }
  parts.push(...marks, ...hits);


  // Detail panel.
  const py = H + 6;
  const pw = st.w - 2 * lx;
  const shown = p.view === "time" ? inTime(sel) : inWho(sel);
  const wr = (s: string, width: number, px: number, em: number) => (lang === "zh" ? wrapZh(s, width, px, em) : wrap(s, width, px));
  const note = shown ? "" : lang === "zh" ? `（${L.notIn}）` : ` (${L.notIn})`;
  const lines = [
    { t: nameOf(sel, lang), cls: "fig-t-strong", size: TYPE.label },
    ...wr(headOf(sel, lang) + note, pw - 16, 6.1, 11).map((t) => ({ t, cls: "fig-t-muted", size: TYPE.small })),
    ...wr(detailOf(sel, lang), pw - 16, 6.6, 12).map((t) => ({ t, cls: "", size: TYPE.body })),
  ];
  const ph2 = 12 + lines.length * 17;
  parts.push(el("rect", { x: lx - 4, y: py - 4, width: pw + 8, height: ph2, rx: 6, fill: C.panel, stroke: C.rule }));
  lines.forEach((ln, i) => parts.push(text(lx + 6, py + 13 + i * 17, ln.t, { "font-size": ln.size, class: ln.cls || undefined })));
  return svg(st.w, py + ph2 + 4, describe(st), g({}, ...parts));
}

export default defineFigure({
  name: "health-tuning",
  title: { en: "Exoskeleton and prosthesis tuning studies side by side", zh: "并列比较外骨骼与假肢的调节研究" },
  labels,
  params,
  hint: { en: "Switch views, and click a marker or choose a study to read its protocol and result.", zh: "切换视图，点击标记或选择一项研究，查看它的方案与结果。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
