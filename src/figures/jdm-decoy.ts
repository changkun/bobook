// The attraction (decoy) effect, tried on the reader's own choices. Round 1
// asks six two-option choices. Round 2 asks the same six situations again,
// shuffled, each with a third option that is worse than one of the two on
// both attributes (an asymmetrically dominated decoy). In three situations the
// decoy is built to favor the option the reader passed over in round 1; in the
// other three it favors the option the reader chose, so switches there
// measure the plain inconsistency of repeated choice. The summary compares the
// two switch counts and maps every situation in attribute space. Three
// answers per condition make this a demonstration, not a measurement.

import { defineFigure, type Lang, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { rng } from "./lib/random.ts";
import { tpl } from "./lib/format.ts";

const labels = {
  en: {
    round: "Round {r} of 2 · choice {i} of 6",
    prompt: "You are choosing {topic}. Which do you pick?",
    note1: "Round 2 returns to the same situations. Answer each time as you would now.",
    note2: "The same situations again. Answer as you would now.",
    option: "Option {k}",
    doneTitle: "Your choices, round 1 and round 2",
    against: "When the third option favored the one you had passed over, you switched to it in {x} of 3.",
    for: "When the third option favored the one you had chosen, you switched away in {y} of 3.",
    more: "You moved toward the decoy's target more often, as the attraction effect predicts.",
    same: "The two switch counts are equal: no sign of the attraction effect in these answers.",
    less: "You moved away from the decoy's targets more often, the opposite direction.",
    decoyPicked: "You picked a decoy {d} time{d:/s}; it is worse than its target on both attributes.",
    caveat: "Three answers per condition is a demonstration, not a measurement.",
    tagAgainst: "decoy favors rejected",
    tagFor: "decoy favors chosen",
    lgOpt: "original two",
    lgDecoy: "decoy",
    lgR1: "round 1 pick",
    lgR2: "round 2 pick",
    better: "better",
    describePlay: "Round {r}, choice {i} of 6: {topic}. {opts}",
    describeDone: "Finished. {a} {b}",
    magenta: "Magenta marks the option each decoy was built to favor.",
  },
  zh: {
    round: "第 {r} 轮（共 2 轮）· 第 {i} 题（共 6 题）",
    prompt: "你在挑选{topic}。你会选哪一个？",
    note1: "第 2 轮会回到同样的情境。每次都按你当下的想法回答。",
    note2: "又是同样的情境。按你当下的想法回答。",
    option: "选项 {k}",
    doneTitle: "你在第 1 轮和第 2 轮的选择",
    against: "当第三个选项衬托的是你先前放弃的那个选项时，3 次中有 {x} 次你改选了它。",
    for: "当第三个选项衬托的是你先前选中的那个选项时，3 次中有 {y} 次你改选了别的选项。",
    more: "你更常转向诱饵所衬托的选项，这与吸引效应的预测一致。",
    same: "两种改选的次数相同：这些回答中看不到吸引效应的迹象。",
    less: "你更常离开诱饵所衬托的选项，方向与预测相反。",
    decoyPicked: "你选了诱饵 {d} 次；它在两个属性上都不如它所衬托的选项。",
    caveat: "每种条件只有三个回答，这只是演示，不是测量。",
    tagAgainst: "诱饵衬托被放弃者",
    tagFor: "诱饵衬托被选中者",
    lgOpt: "原来的两个选项",
    lgDecoy: "诱饵",
    lgR1: "第 1 轮的选择",
    lgR2: "第 2 轮的选择",
    better: "更好",
    describePlay: "第 {r} 轮，第 {i} 题（共 6 题）：{topic}。{opts}",
    describeDone: "已完成。{a}{b}",
    magenta: "品红色标出每个诱饵被设计来衬托的选项。",
  },
};

interface Attr { name: string; fmt: (v: number) => string; higher: boolean }
interface Scenario { topic: string; short: string; a1: Attr; a2: Attr; A: [number, number]; B: [number, number]; dA: [number, number]; dB: [number, number] }
// The Chinese edition's wording of each situation, in the same order.
interface ScenarioZh { topic: string; short: string; a1: string; a2: string; f1: (v: number) => string; f2: (v: number) => string }

const thousands = (v: number) => String(v).replace(/\B(?=(\d{3})+(?!\d))/g, ",");

// A is better on the first attribute, B on the second. dA is worse than A on
// both attributes but better than B on the first; dB mirrors it for B.
const SCENARIOS: Scenario[] = [
  {
    topic: "a cloud storage plan", short: "Storage plan",
    a1: { name: "Storage", fmt: (v) => `${thousands(v)} GB`, higher: true },
    a2: { name: "Price", fmt: (v) => `$${v} / month`, higher: false },
    A: [2000, 10], B: [500, 4], dA: [1600, 11], dB: [400, 5],
  },
  {
    topic: "a laptop", short: "Laptop",
    a1: { name: "Battery life", fmt: (v) => `${v} hours`, higher: true },
    a2: { name: "Weight", fmt: (v) => `${v.toFixed(1)} kg`, higher: false },
    A: [16, 1.9], B: [9, 1.1], dA: [14, 2.1], dB: [8, 1.3],
  },
  {
    topic: "an apartment", short: "Apartment",
    a1: { name: "Floor area", fmt: (v) => `${v} m²`, higher: true },
    a2: { name: "Commute", fmt: (v) => `${v} min`, higher: false },
    A: [72, 45], B: [48, 15], dA: [66, 50], dB: [44, 20],
  },
  {
    topic: "headphones", short: "Headphones",
    a1: { name: "Noise canceling", fmt: (v) => `${v.toFixed(1)} / 10`, higher: true },
    a2: { name: "Price", fmt: (v) => `$${v}`, higher: false },
    A: [9.1, 320], B: [7.2, 140], dA: [8.6, 350], dB: [6.8, 165],
  },
  {
    topic: "a flight", short: "Flight",
    a1: { name: "Fare", fmt: (v) => `$${v}`, higher: false },
    a2: { name: "Travel time", fmt: (v) => `${v} hours`, higher: false },
    A: [420, 13], B: [690, 8], dA: [450, 14], dB: [720, 9],
  },
  {
    topic: "a restaurant for tonight", short: "Restaurant",
    a1: { name: "Rating", fmt: (v) => `${v.toFixed(1)} stars`, higher: true },
    a2: { name: "Wait", fmt: (v) => `${v} min`, higher: false },
    A: [4.7, 40], B: [4.1, 5], dA: [4.5, 45], dB: [3.9, 10],
  },
];

const SCENARIOS_ZH: ScenarioZh[] = [
  { topic: "云存储套餐", short: "存储套餐", a1: "存储空间", a2: "价格", f1: (v) => `${thousands(v)} GB`, f2: (v) => `每月 ${v} 美元` },
  { topic: "笔记本电脑", short: "笔记本电脑", a1: "续航时间", a2: "重量", f1: (v) => `${v} 小时`, f2: (v) => `${v.toFixed(1)} kg` },
  { topic: "公寓", short: "公寓", a1: "面积", a2: "通勤时间", f1: (v) => `${v} m²`, f2: (v) => `${v} 分钟` },
  { topic: "耳机", short: "耳机", a1: "降噪效果", a2: "价格", f1: (v) => `${v.toFixed(1)} / 10`, f2: (v) => `${v} 美元` },
  { topic: "航班", short: "航班", a1: "票价", a2: "旅行时间", f1: (v) => `${v} 美元`, f2: (v) => `${v} 小时` },
  { topic: "今晚去的餐厅", short: "餐厅", a1: "评分", a2: "等位时间", f1: (v) => `${v.toFixed(1)} 星`, f2: (v) => `${v} 分钟` },
];

// A situation in the edition's language.
function view(s: number, lang: Lang): { topic: string; short: string; a1: Attr; a2: Attr } {
  const sc = SCENARIOS[s];
  if (lang !== "zh") return sc;
  const z = SCENARIOS_ZH[s];
  return { topic: z.topic, short: z.short, a1: { ...sc.a1, name: z.a1, fmt: z.f1 }, a2: { ...sc.a2, name: z.a2, fmt: z.f2 } };
}

const N = SCENARIOS.length;
const ORDER2 = [3, 0, 4, 1, 5, 2]; // round 2 visits the situations in this order
const AGAINST = new Set([0, 2, 4]); // decoy favors the option passed over in round 1

type Id = "A" | "B" | "D";
type P = { answers: string; seed: number };

const params = {
  answers: {
    kind: "data", label: { en: "Answers", zh: "回答" }, default: "",
    validate: (s: string) => {
      const a = parse(s);
      if (a.length > 2 * N) return "at most 12 answers";
      if (a.some((x, i) => !["A", "B", "D"].includes(x) || (i < N && x === "D"))) return "answers are A or B in round 1, A, B, or D in round 2";
      return undefined;
    },
  },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 5, step: 1, control: false },
} as const;

function parse(s: string): Id[] {
  return s.trim() ? (s.split(",").map((t) => t.trim()) as Id[]) : [];
}

const other = (x: Id): Id => (x === "A" ? "B" : "A");

// Display order of the options for the current question, from the seed.
function shuffled<T>(xs: T[], seed: number): T[] {
  const r = rng(seed);
  const a = xs.slice();
  for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

interface Question { round: 1 | 2; i: number; s: number; ids: Id[]; target?: Id }

function question(p: P): Question | undefined {
  const a = parse(p.answers);
  if (a.length >= 2 * N) return undefined;
  if (a.length < N) {
    const s = a.length;
    return { round: 1, i: s, s, ids: shuffled<Id>(["A", "B"], p.seed * 31 + s) };
  }
  const k = a.length - N;
  const s = ORDER2[k];
  const first = a[s];
  const target = AGAINST.has(s) ? other(first) : first;
  return { round: 2, i: k, s, ids: shuffled<Id>(["A", "B", "D"], p.seed * 57 + k), target };
}

function values(sc: Scenario, id: Id, target?: Id): [number, number] {
  if (id === "A") return sc.A;
  if (id === "B") return sc.B;
  return target === "A" ? sc.dA : sc.dB;
}

function answer(p: P, pos: number): P {
  const q = question(p);
  if (!q || pos >= q.ids.length) return p;
  const a = parse(p.answers);
  a.push(q.ids[pos]);
  return { ...p, answers: a.join(",") };
}

interface Outcome { s: number; first: Id; second: Id; target: Id; against: boolean }

function outcomes(p: P): Outcome[] {
  const a = parse(p.answers);
  const out: Outcome[] = [];
  for (let k = 0; k < Math.max(0, a.length - N); k++) {
    const s = ORDER2[k];
    const first = a[s];
    const against = AGAINST.has(s);
    out.push({ s, first, second: a[N + k], target: against ? other(first) : first, against });
  }
  return out.sort((u, v) => u.s - v.s);
}

function tally(p: P) {
  const o = outcomes(p);
  const x = o.filter((u) => u.against && u.second === u.target).length;
  const y = o.filter((u) => !u.against && u.second === other(u.target)).length;
  const d = o.filter((u) => u.second === "D").length;
  return { x, y, d };
}

function summary(p: P, lang: Lang = "en"): string[] {
  const L = labels[lang];
  const { x, y, d } = tally(p);
  const lines = [tpl(L.against, { x }), tpl(L.for, { y }), x > y ? L.more : x === y ? L.same : L.less];
  if (d) lines.push(tpl(L.decoyPicked, { d }));
  lines.push(L.caveat);
  return lines;
}

function describe(st: State<P>): string {
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const q = question(st.p);
  if (!q) { const s = summary(st.p, lang); return tpl(L.describeDone, { a: s[0], b: s[1] }); }
  const sc = SCENARIOS[q.s], vw = view(q.s, lang);
  const opts = q.ids.map((id, k) => {
    const v = values(sc, id, q.target);
    return lang === "zh"
      ? `${tpl(L.option, { k: k + 1 })}：${vw.a1.name} ${vw.a1.fmt(v[0])}，${vw.a2.name} ${vw.a2.fmt(v[1])}。`
      : `${tpl(L.option, { k: k + 1 })}: ${sc.a1.name} ${sc.a1.fmt(v[0])}, ${sc.a2.name} ${sc.a2.fmt(v[1])}.`;
  }).join(lang === "zh" ? "" : " ");
  return tpl(L.describePlay, { r: q.round, i: q.i + 1, topic: vw.topic, opts });
}

// Wrap a sentence into lines of at most `max` characters.
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
  const w = st.w;
  const narrow = w < 480;
  const sc = { ...SCENARIOS[q.s], ...view(q.s, lang) };
  const parts: string[] = [];
  const pad = narrow ? 8 : 16;
  parts.push(text(pad, 16, tpl(L.round, { r: q.round, i: q.i + 1 }), { "font-size": TYPE.small, class: "fig-t-muted" }));
  const promptLines = wrapIn(lang, tpl(L.prompt, { topic: sc.topic }), narrow ? 44 : 80);
  promptLines.forEach((ln, i) => parts.push(text(pad, 38 + i * 18, ln, { "font-size": TYPE.label, class: "fig-t-strong" })));
  let y = 38 + (promptLines.length - 1) * 18 + 16;
  const n = q.ids.length;
  if (!narrow) {
    const gap = 16;
    const cw = Math.min(190, (w - 2 * pad - (n - 1) * gap) / n);
    const ch = 118;
    const x0 = (w - (n * cw + (n - 1) * gap)) / 2;
    q.ids.forEach((id, k) => {
      const v = values(sc, id, q.target);
      const x = x0 + k * (cw + gap);
      parts.push(g({ "data-fig-hit": `pos-${k}` },
        el("rect", { x, y, width: cw, height: ch, rx: 10, fill: C.panel, stroke: C.rule, "stroke-width": 1 }),
        text(x + 14, y + 24, tpl(L.option, { k: k + 1 }), { "font-size": TYPE.label, class: "fig-t-strong" }),
        text(x + 14, y + 50, sc.a1.name, { "font-size": TYPE.small, class: "fig-t-muted" }),
        text(x + 14, y + 67, sc.a1.fmt(v[0]), { "font-size": TYPE.title, class: "fig-t-num" }),
        text(x + 14, y + 89, sc.a2.name, { "font-size": TYPE.small, class: "fig-t-muted" }),
        text(x + 14, y + 106, sc.a2.fmt(v[1]), { "font-size": TYPE.title, class: "fig-t-num" }),
      ));
    });
    y += ch + 22;
  } else {
    // A small table: one row per option, so three options fit a phone column.
    const c1 = pad + 78, c2 = pad + 78 + (w - 2 * pad - 78) / 2;
    parts.push(text(c1, y + 8, sc.a1.name, { "font-size": TYPE.small, class: "fig-t-muted" }), text(c2, y + 8, sc.a2.name, { "font-size": TYPE.small, class: "fig-t-muted" }));
    y += 16;
    const rh = 40;
    q.ids.forEach((id, k) => {
      const v = values(sc, id, q.target);
      const ry = y + k * (rh + 8);
      parts.push(g({ "data-fig-hit": `pos-${k}` },
        el("rect", { x: pad, y: ry, width: w - 2 * pad, height: rh, rx: 8, fill: C.panel, stroke: C.rule, "stroke-width": 1 }),
        text(pad + 10, ry + 25, tpl(L.option, { k: k + 1 }), { "font-size": TYPE.label, class: "fig-t-strong" }),
        text(c1, ry + 25, sc.a1.fmt(v[0]), { "font-size": TYPE.label, class: "fig-t-num" }),
        text(c2, ry + 25, sc.a2.fmt(v[1]), { "font-size": TYPE.label, class: "fig-t-num" }),
      ));
    });
    y += n * (rh + 8) + 14;
  }
  const note = wrapIn(lang, q.round === 1 ? L.note1 : L.note2, narrow ? 52 : 96);
  note.forEach((ln, i) => parts.push(text(pad, y + i * 15, ln, { "font-size": TYPE.small, class: "fig-t-faint" })));
  y += note.length * 15 + 4;
  return svg(w, y, describe(st), ...parts);
}

// One small attribute map per situation: the first attribute across, the
// second up, both oriented so that better is to the right and up.
function miniMap(x: number, y: number, pw: number, ph: number, o: Outcome, lang: Lang = "en"): string {
  const L = labels[lang];
  const sc = { ...SCENARIOS[o.s], ...view(o.s, lang) };
  const parts: string[] = [];
  const pts: Array<{ id: Id; v: [number, number] }> = [
    { id: "A", v: sc.A }, { id: "B", v: sc.B }, { id: "D", v: values(sc, "D", o.target) },
  ];
  const all = [sc.A, sc.B, sc.dA, sc.dB];
  const ext = (i: 0 | 1) => {
    const vs = all.map((v) => v[i]);
    const lo = Math.min(...vs), hi = Math.max(...vs), m = (hi - lo) * 0.14;
    return [lo - m, hi + m] as [number, number];
  };
  const left = x + 14, right = x + pw - 6, top = y + 32, bottom = y + ph - 16;
  const [x0, x1] = ext(0), [y0, y1] = ext(1);
  const sx = linear(sc.a1.higher ? [x0, x1] : [x1, x0], [left, right]);
  const sy = linear(sc.a2.higher ? [y0, y1] : [y1, y0], [bottom, top]);
  parts.push(
    el("rect", { x, y, width: pw, height: ph, rx: 8, fill: C.panel, stroke: C.grid }),
    text(x + 8, y + 15, sc.short, { "font-size": TYPE.small, class: "fig-t-strong" }),
    text(x + 8, y + 27, o.against ? L.tagAgainst : L.tagFor, { "font-size": TYPE.small, class: "fig-t-faint" }),
    el("line", { x1: left, x2: right, y1: bottom, y2: bottom, stroke: C.rule }),
    el("line", { x1: left, x2: left, y1: top, y2: bottom, stroke: C.rule }),
    text(right, bottom + 12, sc.a1.name, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-faint" }),
    text(0, 0, sc.a2.name, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-faint", transform: `translate(${left - 4},${top}) rotate(-90)` }),
    // a corner arrow: better is up and to the right on both axes
    el("path", { d: `M${right - 30},${top + 22}L${right - 8},${top}`, stroke: C.ink3, "stroke-width": 1, fill: "none" }),
    el("path", { d: `M${right - 14},${top}L${right - 8},${top}L${right - 8},${top + 6}`, stroke: C.ink3, "stroke-width": 1, fill: "none" }),
    text(right - 32, top + 28, L.better, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-faint" }),
  );
  for (const pt of pts) {
    const cx = sx(pt.v[0]), cy = sy(pt.v[1]);
    const isTarget = pt.id === o.target;
    if (pt.id === "D") {
      parts.push(el("path", { d: `M${cx},${cy - 6}L${cx + 6},${cy}L${cx},${cy + 6}L${cx - 6},${cy}Z`, fill: C.paper, stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "2 2" }));
    } else {
      parts.push(el("circle", { cx, cy, r: 5, fill: isTarget ? C.c5 : C.c1, stroke: C.paper, "stroke-width": 1.2 }));
    }
    if (pt.id === o.first) parts.push(el("circle", { cx, cy, r: 9.5, fill: "none", stroke: C.ink, "stroke-width": 1.3 }));
    if (pt.id === o.second) parts.push(el("circle", { cx, cy, r: 13, fill: "none", stroke: C.acq, "stroke-width": 2 }));
  }
  // a faint line from the decoy to its target shows which option it flatters
  const t = pts.find((q) => q.id === o.target)!, d = pts.find((q) => q.id === "D")!;
  parts.push(el("line", { x1: sx(d.v[0]), y1: sy(d.v[1]), x2: sx(t.v[0]), y2: sy(t.v[1]), stroke: C.ink3, "stroke-width": 0.8, "stroke-dasharray": "1 3" }));
  return g({}, ...parts);
}

function renderDone(st: State<P>): string {
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const w = st.w;
  const narrow = w < 480;
  const pad = narrow ? 8 : 16;
  const parts: string[] = [];
  parts.push(text(pad, 18, L.doneTitle, { "font-size": TYPE.label, class: "fig-t-strong" }));
  let y = 34;
  for (const line of summary(st.p, lang)) {
    for (const ln of wrapIn(lang, line, narrow ? 50 : 92)) { parts.push(text(pad, y + 4, ln, { "font-size": TYPE.body })); y += 17; }
  }
  y += 8;
  // legend
  const items: Array<[string, string]> = [["opt", L.lgOpt], ["decoy", L.lgDecoy], ["r1", L.lgR1], ["r2", L.lgR2]];
  let lx = pad, ly = y + 6;
  for (const [kind, lab] of items) {
    const tw = labelWidth(lab, 6.2) + 30;
    if (lx + tw > w - pad) { lx = pad; ly += 20; }
    const cx = lx + 8;
    if (kind === "opt") parts.push(el("circle", { cx, cy: ly, r: 5, fill: C.c1 }), el("circle", { cx: cx + 7, cy: ly, r: 5, fill: C.c5 }));
    if (kind === "decoy") parts.push(el("path", { d: `M${cx},${ly - 6}L${cx + 6},${ly}L${cx},${ly + 6}L${cx - 6},${ly}Z`, fill: C.paper, stroke: C.ink3, "stroke-width": 1.4, "stroke-dasharray": "2 2" }));
    if (kind === "r1") parts.push(el("circle", { cx, cy: ly, r: 6.5, fill: "none", stroke: C.ink, "stroke-width": 1.3 }));
    if (kind === "r2") parts.push(el("circle", { cx, cy: ly, r: 7, fill: "none", stroke: C.acq, "stroke-width": 2 }));
    parts.push(text(lx + (kind === "opt" ? 28 : 20), ly + 4, lab, { "font-size": TYPE.small, class: "fig-t-muted" }));
    lx += tw + (kind === "opt" ? 8 : 0);
  }
  parts.push(text(pad, ly + 22, L.magenta, { "font-size": TYPE.small, class: "fig-t-faint" }));
  y = ly + 34;
  const cols = narrow ? 2 : 3;
  const gap = 10;
  const pw = (w - 2 * pad - (cols - 1) * gap) / cols;
  const ph = narrow ? 128 : 140;
  outcomes(st.p).forEach((o, i) => {
    const cx = pad + (i % cols) * (pw + gap);
    const cy = y + Math.floor(i / cols) * (ph + gap);
    parts.push(miniMap(cx, cy, pw, ph, o, lang));
  });
  y += Math.ceil(N / cols) * (ph + gap) + 4;
  return svg(w, y, describe(st), ...parts);
}

function render(st: State<P>): string {
  const q = question(st.p);
  return q ? renderPlay(st, q) : renderDone(st);
}

const pick = (k: number) => ({
  label: { en: `Option ${k + 1}`, zh: `选项 ${k + 1}` },
  primary: true,
  run: (p: P) => answer(p, k),
  enabled: (p: P) => { const q = question(p); return !!q && k < q.ids.length; },
});

export default defineFigure({
  name: "jdm-decoy",
  title: { en: "A decoy added to your own choices", zh: "在你自己的选择中加入诱饵" },
  labels,
  params,
  hint: { en: "Click the option you would pick, or use the buttons. There are no right answers; answer quickly.", zh: "点击你会选的选项，或使用按钮。没有正确答案，尽快作答。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click" || !e.target?.startsWith("pos-")) return null;
    return answer(p as P, Number(e.target.slice(4)));
  },
  actions: [
    pick(0), pick(1), pick(2),
    { label: { en: "Undo", zh: "撤销" }, run: (p) => ({ ...p, answers: parse(p.answers).slice(0, -1).join(",") }), enabled: (p) => p.answers !== "" },
  ],
});
