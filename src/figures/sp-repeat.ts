// The same moral question, asked twice. The reader answers eight hypothetical
// kidney-allocation questions (which of two patients should receive the one
// available kidney), then the same eight again in a new order with the sides
// swapped. The summary counts how many answers changed and sets the count
// beside the rates that studies found when the repeats were days apart. With
// eight questions minutes apart, and memory of round 1, this is a
// demonstration, not a measurement.

import { defineFigure, type Lang, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { rng } from "./lib/random.ts";
import { tpl } from "./lib/format.ts";

const labels = {
  en: {
    round: "Round {r} of 2 · question {i} of 8",
    prompt: "One kidney is available. Which patient should receive it?",
    note1: "Hypothetical patients. Both are medically compatible.",
    note2: "Round 2 asks the same eight questions again. Answer as you would now.",
    patient: "Patient {k}",
    age: "Age",
    deps: "Dependent children",
    years: "Life-years gained",
    ageV: "{v} years",
    yearsV: "about {v} years",
    resultTitle: "Your two rounds",
    changed: "You changed {n} of your 8 answers ({p}).",
    ref: "Studies that repeated such questions days apart found people changing about 10% to 18% of their answers to controversial cases over ten sessions in two weeks, and about 6% to 20% across three to five sessions.",
    caveat: "Here the repeat came minutes later and you may remember round 1, so a lower rate is expected. Eight questions are a demonstration, not a measurement.",
    colPair: "the two patients (age, children, life-years), numbered as listed here",
    colPairShort: "patients, as listed here",
    colR1: "round 1",
    colR2: "round 2",
    same: "same",
    flip: "changed",
    describePlay: "Round {r}, question {i} of 8. Patient 1: {a}. Patient 2: {b}.",
    describeDone: "Finished. You changed {n} of 8 answers between the two rounds.",
    patientV: "{age} y, {deps} ch., {years} yrs",
    rowPair: "1: {a}   vs   2: {b}",
    rowOne: "{k}: {v}",
  },
  zh: {
    round: "第 {r} 轮（共 2 轮）· 第 {i} 题（共 8 题）",
    prompt: "现有一个肾脏可供移植。应该给哪一位患者？",
    note1: "患者是假设的。两人在医学上都匹配。",
    note2: "第 2 轮再问一遍同样的八个问题。按你现在的想法回答。",
    patient: "患者 {k}",
    age: "年龄",
    deps: "需抚养的子女",
    years: "可获得的生命年",
    ageV: "{v} 岁",
    yearsV: "约 {v} 年",
    resultTitle: "你的两轮回答",
    changed: "你的 8 个回答中有 {n} 个改变了（{p}）。",
    ref: "在相隔数天重复这类问题的研究中，人们在两周内的十次会话里，对有争议情形的回答约有 10% 至 18% 发生改变；在三至五次会话之间，约有 6% 至 20% 发生改变。",
    caveat: "这里的重复只隔了几分钟，你可能还记得第 1 轮的回答，因此改变的比例预计会更低。八个问题只是演示，不是测量。",
    colPair: "两位患者（年龄、子女、生命年），按此处所列编号",
    colPairShort: "患者，按此处所列",
    colR1: "第 1 轮",
    colR2: "第 2 轮",
    same: "相同",
    flip: "改变",
    describePlay: "第 {r} 轮，第 {i} 题（共 8 题）。患者 1：{a}。患者 2：{b}。",
    describeDone: "已完成。两轮之间，你的 8 个回答中有 {n} 个改变了。",
    patientV: "{age} 岁，{deps} 个子女，{years} 年",
    rowPair: "1：{a}　对　2：{b}",
    rowOne: "{k}：{v}",
  },
};

interface Patient { age: number; deps: number; years: number }
// Eight pairs, most of them trade-offs between age, dependents, and benefit.
const PAIRS: Array<[Patient, Patient]> = [
  [{ age: 30, deps: 0, years: 12 }, { age: 55, deps: 2, years: 9 }],
  [{ age: 45, deps: 1, years: 10 }, { age: 45, deps: 3, years: 7 }],
  [{ age: 25, deps: 0, years: 15 }, { age: 62, deps: 0, years: 6 }],
  [{ age: 38, deps: 2, years: 8 }, { age: 52, deps: 1, years: 11 }],
  [{ age: 66, deps: 3, years: 5 }, { age: 33, deps: 0, years: 14 }],
  [{ age: 28, deps: 1, years: 13 }, { age: 41, deps: 2, years: 12 }],
  [{ age: 50, deps: 0, years: 9 }, { age: 50, deps: 0, years: 8 }],
  [{ age: 35, deps: 3, years: 6 }, { age: 44, deps: 1, years: 10 }],
];
const N = PAIRS.length;
const ORDER2 = [5, 2, 7, 0, 3, 6, 1, 4];

type Id = "A" | "B"; // A is the first patient of the pair as listed in PAIRS
type P = { answers: string; seed: number };

const params = {
  answers: {
    kind: "data", label: { en: "Answers", zh: "回答" }, default: "",
    validate: (s: string) => (parse(s).length <= 2 * N && parse(s).every((x) => x === "A" || x === "B") ? undefined : "up to 16 answers, each A or B"),
  },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 9, step: 1, control: false },
} as const;

function parse(s: string): Id[] {
  return s.trim() ? (s.split(",").map((t) => t.trim()) as Id[]) : [];
}

interface Question { round: 1 | 2; i: number; pair: number; left: Id }

// Round 1 shows each pair with a seeded side; round 2 swaps the sides.
function sideOf(p: P, pair: number): Id {
  return rng(p.seed * 31 + pair)() < 0.5 ? "A" : "B";
}

function question(p: P): Question | undefined {
  const a = parse(p.answers);
  if (a.length >= 2 * N) return undefined;
  if (a.length < N) return { round: 1, i: a.length, pair: a.length, left: sideOf(p, a.length) };
  const k = a.length - N, pair = ORDER2[k];
  return { round: 2, i: k, pair, left: sideOf(p, pair) === "A" ? "B" : "A" };
}

function answer(p: P, pos: 0 | 1): P {
  const q = question(p);
  if (!q) return p;
  const id: Id = pos === 0 ? q.left : q.left === "A" ? "B" : "A";
  return { ...p, answers: [...parse(p.answers), id].join(",") };
}

function results(p: P) {
  const a = parse(p.answers);
  const r2: Array<Id | undefined> = Array(N).fill(undefined);
  for (let k = 0; k < Math.max(0, a.length - N); k++) r2[ORDER2[k]] = a[N + k];
  const rows = PAIRS.map((_, i) => ({ i, r1: a[i], r2: r2[i] }));
  const n = rows.filter((r) => r.r1 && r.r2 && r.r1 !== r.r2).length;
  return { rows, n };
}

const fmt = (pt: Patient, lang: Lang = "en") => tpl(labels[lang].patientV, { age: pt.age, deps: pt.deps, years: pt.years });

function describe(st: State<P>): string {
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const q = question(st.p);
  if (!q) return tpl(L.describeDone, { n: results(st.p).n });
  const [a, b] = PAIRS[q.pair];
  const left = q.left === "A" ? a : b, right = q.left === "A" ? b : a;
  return tpl(L.describePlay, { r: q.round, i: q.i + 1, a: fmt(left, lang), b: fmt(right, lang) });
}

function wrap(s: string, max: number): string[] {
  const out: string[] = [];
  let line = "";
  for (const w of s.split(" ")) {
    if (line && (line + " " + w).length > max) { out.push(line); line = w; } else line = line ? `${line} ${w}` : w;
  }
  if (line) out.push(line);
  return out;
}

// Wrap Chinese text, which has no spaces between words, to lines of at most
// `max` Latin-letter widths: a Chinese character counts 2.1, a run of Latin
// letters, digits, and symbols is kept whole, and a line never starts with
// closing punctuation.
function wrapZh(s: string, max: number): string[] {
  const toks = s.match(/[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]|[^\s\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]+|\s+/g) ?? [];
  const width = (t: string) => labelWidth(t, 1, 2.1);
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

function renderPlay(st: State<P>, q: Question): string {
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const w = st.w, narrow = w < 480, pad = narrow ? 8 : 16;
  const parts: string[] = [];
  parts.push(text(pad, 16, tpl(L.round, { r: q.round, i: q.i + 1 }), { "font-size": TYPE.small, class: "fig-t-muted" }));
  const pl = wrapIn(lang, L.prompt, narrow ? 44 : 90);
  pl.forEach((ln, i) => parts.push(text(pad, 38 + i * 18, ln, { "font-size": TYPE.label, class: "fig-t-strong" })));
  const top = 38 + (pl.length - 1) * 18 + 16;
  const [a, b] = PAIRS[q.pair];
  const shown = q.left === "A" ? [a, b] : [b, a];
  const gap = narrow ? 10 : 24;
  const cw = narrow ? (w - 2 * pad - gap) / 2 : Math.min(220, (w - 2 * pad - gap) / 2);
  const ch = narrow ? 150 : 140;
  const x0 = (w - 2 * cw - gap) / 2;
  shown.forEach((pt, k) => {
    const x = x0 + k * (cw + gap);
    const rows: Array<[string, string]> = [[L.age, tpl(L.ageV, { v: pt.age })], [L.deps, String(pt.deps)], [L.years, tpl(L.yearsV, { v: pt.years })]];
    const items: string[] = [
      el("rect", { x, y: top, width: cw, height: ch, rx: 10, fill: C.panel, stroke: C.rule }),
      text(x + 12, top + 22, tpl(L.patient, { k: k + 1 }), { "font-size": TYPE.label, class: "fig-t-strong" }),
    ];
    rows.forEach(([name, v], r) => {
      const y = top + (narrow ? 46 : 46) + r * (narrow ? 34 : 31);
      items.push(text(x + 12, y, name, { "font-size": TYPE.small, class: "fig-t-muted" }), text(x + 12, y + 15, v, { "font-size": TYPE.body, class: "fig-t-num" }));
    });
    parts.push(g({ "data-fig-hit": `pos-${k}` }, ...items));
  });
  let y = top + ch + 20;
  for (const ln of wrapIn(lang, q.round === 1 ? L.note1 : L.note2, narrow ? 52 : 96)) { parts.push(text(pad, y, ln, { "font-size": TYPE.small, class: "fig-t-faint" })); y += 15; }
  return svg(w, y, describe(st), ...parts);
}

function renderDone(st: State<P>): string {
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const w = st.w, narrow = w < 480, pad = narrow ? 8 : 16;
  const { rows, n } = results(st.p);
  const parts: string[] = [];
  parts.push(text(pad, 18, L.resultTitle, { "font-size": TYPE.label, class: "fig-t-strong" }));
  let y = 40;
  for (const ln of wrapIn(lang, tpl(L.changed, { n, p: `${Math.round((100 * n) / N)}%` }), narrow ? 46 : 90)) { parts.push(text(pad, y, ln, { "font-size": TYPE.label, class: "fig-t-strong" })); y += 18; }
  y += 6;
  // the table of the eight questions
  const cR1 = narrow ? w - 128 : w - 220, cR2 = narrow ? w - 76 : w - 150, cF = w - pad;
  parts.push(text(pad, y, narrow ? L.colPairShort : L.colPair, { "font-size": TYPE.small, class: "fig-t-muted" }),
    text(cR1, y, L.colR1, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted" }),
    text(cR2, y, L.colR2, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted" }));
  y += 8;
  const rh = narrow ? 36 : 24;
  for (const r of rows) {
    const [a, b] = PAIRS[r.i];
    const flipped = r.r1 !== r.r2;
    parts.push(el("line", { x1: pad, x2: w - pad, y1: y, y2: y, stroke: C.grid }));
    if (narrow) {
      parts.push(text(pad, y + 15, tpl(L.rowOne, { k: 1, v: fmt(a, lang) }), { "font-size": TYPE.small, class: "fig-t-num" }), text(pad, y + 29, tpl(L.rowOne, { k: 2, v: fmt(b, lang) }), { "font-size": TYPE.small, class: "fig-t-num" }));
    } else {
      parts.push(text(pad, y + 16, tpl(L.rowPair, { a: fmt(a, lang), b: fmt(b, lang) }), { "font-size": TYPE.body, class: "fig-t-num" }));
    }
    const cy = y + rh / 2 + 4;
    const name = (id?: Id) => (id ? (id === "A" ? "1" : "2") : "–");
    parts.push(text(cR1, cy, name(r.r1), { "font-size": TYPE.body, "text-anchor": "middle", class: "fig-t-strong" }), text(cR2, cy, name(r.r2), { "font-size": TYPE.body, "text-anchor": "middle", class: "fig-t-strong" }));
    parts.push(text(cF, cy, flipped ? L.flip : L.same, { "font-size": TYPE.small, "text-anchor": "end", class: flipped ? "fig-t-strong" : "fig-t-faint", style: flipped ? `fill:${C.acq}` : undefined }));
    y += rh;
  }
  parts.push(el("line", { x1: pad, x2: w - pad, y1: y, y2: y, stroke: C.grid }));
  y += 22;
  for (const [s, cls] of [[L.ref, ""], [L.caveat, "fig-t-muted"]] as const) {
    for (const ln of wrapIn(lang, s, narrow ? 52 : 96)) { parts.push(text(pad, y, ln, { "font-size": TYPE.body, class: cls })); y += 16; }
    y += 4;
  }
  return svg(w, y, describe(st), ...parts);
}

function render(st: State<P>): string {
  const q = question(st.p);
  return q ? renderPlay(st, q) : renderDone(st);
}

export default defineFigure({
  name: "sp-repeat",
  title: { en: "The same allocation question, asked twice", zh: "同一个分配问题，问两次" },
  labels,
  params,
  hint: { en: "Click the patient you would choose, or use the buttons. Answer as you would; there is no right answer.", zh: "点击你会选择的患者，或使用按钮。按你自己的想法回答，没有正确答案。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click" || (e.target !== "pos-0" && e.target !== "pos-1")) return null;
    return answer(p as P, e.target === "pos-0" ? 0 : 1);
  },
  actions: [
    { label: { en: "Patient 1", zh: "患者 1" }, primary: true, run: (p) => answer(p as P, 0), enabled: (p) => !!question(p as P) },
    { label: { en: "Patient 2", zh: "患者 2" }, primary: true, run: (p) => answer(p as P, 1), enabled: (p) => !!question(p as P) },
    { label: { en: "Undo", zh: "撤销" }, run: (p) => ({ ...p, answers: parse(p.answers).slice(0, -1).join(",") }), enabled: (p) => p.answers !== "" },
  ],
});
