// Can't compare: how a forced binary likelihood reads answers that are not
// preferences. The reader is shown pairs of flats that trade living space
// against commute time and may answer A, B, "can't compare", or "flip a coin
// for me". Two deliberately small models read the same answers.
//
//   Forced choice. One weight w on space (1 - w on commute), a linear
//   utility, and a probit likelihood with a small lapse. A forced binary
//   interface has no third answer, so every "can't compare" or coin request
//   reaches this model as the coin's outcome, which is what the person would
//   have been made to click.
//
//   Incomplete preferences (Bewley's multi-utility idea). The person has a
//   set of admissible weights [lo, hi]. One flat is preferred only when every
//   admissible weight agrees; otherwise the pair is unresolved. The same
//   probit noise blurs the edges of the set, and the same lapse applies. In
//   this sketch "can't compare" and "flip a coin" mean the same thing to the
//   model; experiments separate them, which a fuller model would need.
//
// Each pair of flats ties at one weight t (space wins when w > t), so every
// answer is a statement about the weight axis, and both posteriors are exact
// on a grid. Below the axis, the shortlist of seven flats: each is the best
// flat on one interval of weights. The forced model reports the probability
// that each flat is best; the incomplete model reports the probability that
// each flat is not beaten by another under every admissible weight.

import { defineFigure, type Lang, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { grid, linear } from "./lib/scale.ts";
import { Phi } from "./lib/stats.ts";
import { memo, rng, normal } from "./lib/random.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    ask: "Which flat would you rather rent?",
    askSim: "The simulated person is asked:",
    pairOf: "pair {k} of {n}",
    allAnswered: "{n} of {n} answered",
    done: "All {n} pairs answered. Undo to change an answer, or reset to start again.",
    flat: "Flat {s}",
    space: "{v} m² living space",
    commute: "{v} min to work",
    strip: "your answers, placed at the weight where the two flats tie",
    stripSim: "answers, placed at the weight where the two flats tie",
    stripShort: "answers, at the weight where the flats tie",
    forced: "Forced choice: one weight",
    forcedCI: "90% of belief in {a}–{b}",
    incomplete: "Allows “can’t compare”: a set of weights",
    incompleteSet: "even odds or better: {a}–{b}",
    incompletePoint: "even odds or better: only near {a}",
    incompleteNone: "below even odds everywhere: a narrow set near {w}",
    incompleteNoneShort: "a narrow set near {w}",
    probIn: "P(weight is one of yours)",
    axisLeft: "← commute matters more",
    axisRight: "space matters more →",
    axisMid: "weight on living space",
    shortlist: "Shortlist: the flat that is best at each weight",
    pBest: "forced: P(best)",
    pUndom: "set: P(not beaten)",
    legS: "space won",
    legC: "commute won",
    legN: "can’t compare",
    legR: "coin requested",
    legF: "what forced choice recorded",
    truth: "simulated person’s weights",
    truthShort: "simulated weights",
    none: "No answers yet.",
    describe0: "No answers yet. Both models start from a flat prior over the weight on living space.",
    describe: "After {n} answer{n:/s} ({u} of them “can’t compare” or a coin request), the forced-choice model puts 90% of its belief on weights {fa} to {fb} and calls flat {fb1} best with probability {fp}; the model that allows incomparability {ia} and keeps {iset} as not beaten.",
    doneShort: "All {n} pairs answered.",
    flat1: "flat {a}",
    flat2: "flats {a} and {b}",
    flatN: "flats {list}, and {last}",
    listSep: ", ",
    noFlat: "no flat",
    noFlatYet: "no flat yet",
    rangePoint: "only weights near {a}",
    rangeSet: "weights {a} to {b}",
    iaNarrow: "judges the person’s set of weights narrow, most likely near {w},",
    iaSet: "gives {r} at least even odds of being among the person’s weights",
  },
  zh: {
    ask: "你更愿意租哪套公寓？",
    askSim: "向模拟的人提问：",
    pairOf: "第 {k} 对，共 {n} 对",
    allAnswered: "已回答 {n} 对，共 {n} 对",
    done: "{n} 对都已回答。撤销可以修改回答，重置可以重新开始。",
    flat: "公寓 {s}",
    space: "居住面积 {v} m²",
    commute: "通勤 {v} 分钟",
    strip: "你的回答，放在两套公寓打平的权重处",
    stripSim: "回答，放在两套公寓打平的权重处",
    stripShort: "回答，放在两套公寓打平的权重处",
    forced: "强制选择：一个权重",
    forcedCI: "90% 的信念在 {a} 至 {b}",
    incomplete: "允许“无法比较”：一组权重",
    incompleteSet: "概率不低于一半：{a} 至 {b}",
    incompletePoint: "概率不低于一半：仅在 {a} 附近",
    incompleteNone: "处处低于一半：{w} 附近的一个窄集合",
    incompleteNoneShort: "{w} 附近的窄集合",
    probIn: "P(权重属于此人)",
    axisLeft: "← 通勤更重要",
    axisRight: "面积更重要 →",
    axisMid: "居住面积的权重",
    shortlist: "候选清单：每个权重下最好的公寓",
    pBest: "强制：P(最好)",
    pUndom: "集合：P(未被击败)",
    legS: "面积胜出",
    legC: "通勤胜出",
    legN: "无法比较",
    legR: "请求掷硬币",
    legF: "强制选择记录下的回答",
    truth: "模拟的人的权重",
    truthShort: "模拟权重",
    none: "还没有回答。",
    describe0: "还没有回答。两个模型都从居住面积权重上的均匀先验出发。",
    describe: "{n} 个回答之后（其中 {u} 个是“无法比较”或请求掷硬币），强制选择模型把 90% 的信念放在权重 {fa} 至 {fb} 上，并以 {fp} 的概率认为公寓 {fb1} 最好；允许不可比的模型{ia}，并认为未被击败的有：{iset}。",
    doneShort: "{n} 对都已回答。",
    flat1: "公寓 {a}",
    flat2: "公寓 {a} 与 {b}",
    flatN: "公寓 {list} 与 {last}",
    listSep: "、",
    noFlat: "没有公寓",
    noFlatYet: "暂无",
    rangePoint: "仅 {a} 附近的权重",
    rangeSet: "介于 {a} 至 {b} 之间的权重",
    iaNarrow: "判断此人的权重集合很窄，最可能在 {w} 附近",
    iaSet: "认为{r}至少有一半的概率属于此人",
  },
};

const params = {
  mode: { kind: "choice", label: { en: "Who answers", zh: "谁来回答" }, options: [{ value: "you", label: { en: "You", zh: "你" } }, { value: "sim", label: { en: "Simulated person", zh: "模拟的人" } }], default: "you" },
  truth: { kind: "toggle", label: { en: "Reveal simulated weights", zh: "揭晓模拟的权重" }, default: false },
  answers: { kind: "data", label: { en: "Answers", zh: "回答" }, default: "", validate: (s: string) => (/^([abnr](,[abnr])*)?$/.test(s.trim()) ? undefined : "expected a comma-separated list of a, b, n, r") },
  lo: { kind: "range", label: { en: "Simulated lowest weight", zh: "模拟的最低权重" }, min: 0.05, max: 0.95, default: 0.38, step: 0.01, control: false },
  hi: { kind: "range", label: { en: "Simulated highest weight", zh: "模拟的最高权重" }, min: 0.05, max: 0.95, default: 0.62, step: 0.01, control: false },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 3, step: 1, control: false },
} as const;

type P = { mode: "you" | "sim"; truth: boolean; answers: string; lo: number; hi: number; seed: number };

const TAU = 0.05; // probit noise on the weight scale, shared by both models
const EPS = 0.05; // lapse rate, shared by both models

// The questions. Each pair trades dSpace square meters against dCommute
// minutes; on a common scale of 50 units for both attributes, the two flats
// tie at weight t = dCommute / (dSpace + dCommute).
interface Pair { space: [number, number]; commute: [number, number]; t: number; spaciousIsA: boolean }
const SPEC: Array<[number, number, number, number]> = [
  // [tie weight target, total difference, base space, base commute]
  [0.5, 30, 48, 14], [0.25, 32, 52, 18], [0.75, 32, 46, 12], [0.4, 30, 55, 20],
  [0.6, 30, 44, 15], [0.15, 40, 47, 22], [0.85, 40, 45, 10], [0.33, 36, 50, 16],
  [0.67, 36, 49, 13], [0.45, 40, 43, 17], [0.55, 40, 46, 11], [0.2, 30, 58, 19],
  [0.8, 30, 51, 12], [0.3, 40, 45, 15], [0.7, 40, 47, 14], [0.5, 40, 44, 12],
];
const PAIRS: Pair[] = SPEC.map(([t0, tot, s0, c0], k) => {
  const dc = Math.round(t0 * tot), ds = tot - dc;
  const spaciousIsA = rng(k * 7919 + 13)() < 0.5;
  const spacious: [number, number] = [s0 + ds, c0 + dc];
  const close: [number, number] = [s0, c0];
  const [a, b] = spaciousIsA ? [spacious, close] : [close, spacious];
  return { space: [a[0], b[0]], commute: [a[1], b[1]], t: dc / tot, spaciousIsA };
});
const N = PAIRS.length;

// The shortlist: seven flats on a concave trade-off front. Flat j+1 beats
// flat j when the weight exceeds dCommute / (dSpace + dCommute).
const FLATS: Array<[number, number]> = [[50, 12], [56, 14], [62, 17], [68, 22], [74, 29], [80, 40], [86, 58]];
const CUTS = FLATS.slice(1).map(([s, c], j) => (c - FLATS[j][1]) / ((s - FLATS[j][0]) + (c - FLATS[j][1])));
const SEG: Array<[number, number]> = FLATS.map((_, j) => [j === 0 ? 0 : CUTS[j - 1], j === FLATS.length - 1 ? 1 : CUTS[j]]);

type Outcome = "S" | "C" | "U";
interface Read { t: number; out: Outcome; forced: "S" | "C"; token: string }

function readAnswers(s: string, seed: number): Read[] {
  const toks = s.trim() ? s.trim().split(",") : [];
  return toks.slice(0, N).map((tok, k) => {
    const p = PAIRS[k];
    let out: Outcome;
    if (tok === "a") out = p.spaciousIsA ? "S" : "C";
    else if (tok === "b") out = p.spaciousIsA ? "C" : "S";
    else out = "U";
    const forced = out === "U" ? (rng(seed * 104729 + k * 7907 + 1)() < 0.5 ? "S" : "C") : out;
    return { t: p.t, out, forced, token: tok };
  });
}

const WG = grid(0, 1, 201); // weight grid for the forced model and for display
const BG = grid(0, 1, 81); // grid for each end of the admissible set

interface Fit {
  dens: number[]; ciLo: number; ciHi: number; pBest: number[];
  pin: number[]; setLo: number; setHi: number; peak: number; pUndom: number[];
}

const fitAll = memo((s: string, seed: number): Fit => {
  const rs = readAnswers(s, seed);
  // Forced choice: exact posterior over one weight on a grid.
  const lw = WG.map((w) => {
    let l = 0;
    for (const r of rs) {
      const z = (w - r.t) / TAU;
      const p = Phi(r.forced === "S" ? z : -z);
      l += Math.log((1 - EPS) * p + EPS / 2);
    }
    return l;
  });
  const mx = Math.max(...lw);
  const pw = lw.map((v) => Math.exp(v - mx));
  const z0 = pw.reduce((a, b) => a + b, 0);
  const post = pw.map((v) => v / z0);
  let acc = 0, ciLo = 0, ciHi = 1;
  let gotLo = false;
  for (let i = 0; i < WG.length; i++) {
    acc += post[i];
    if (!gotLo && acc >= 0.05) { ciLo = WG[i]; gotLo = true; }
    if (acc >= 0.95) { ciHi = WG[i]; break; }
  }
  const pBest = SEG.map(([a, b]) => WG.reduce((t, w, i) => t + ((w >= a && (w < b || (b === 1 && w <= 1))) ? post[i] : 0), 0));
  // Display scale: the flat prior sits at 1/8 of the panel, and a posterior
  // fills the panel once its peak is eight times the prior density.
  const dmax = Math.max(8, ...post.map((v) => v * WG.length));
  const dens = post.map((v) => (v * WG.length) / dmax);
  // Incomplete preferences: exact posterior over the set [lo, hi], lo <= hi,
  // uniform prior on the triangle.
  const cells: Array<[number, number, number]> = [];
  let lmax = -Infinity;
  for (let i = 0; i < BG.length; i++) for (let j = i; j < BG.length; j++) {
    const lo = BG[i], hi = BG[j];
    let l = 0;
    for (const r of rs) {
      const pS = Phi((lo - r.t) / TAU);
      const pC = Phi((r.t - hi) / TAU);
      const pU = Math.max(0, 1 - pS - pC);
      const p = r.out === "S" ? pS : r.out === "C" ? pC : pU;
      l += Math.log((1 - EPS) * p + EPS / 3);
    }
    cells.push([lo, hi, l]);
    if (l > lmax) lmax = l;
  }
  let zc = 0;
  for (const c of cells) { c[2] = Math.exp(c[2] - lmax); zc += c[2]; }
  for (const c of cells) c[2] /= zc;
  const pin = WG.map((w) => cells.reduce((t, [lo, hi, p]) => t + (lo <= w && w <= hi ? p : 0), 0));
  let setLo = NaN, setHi = NaN;
  WG.forEach((w, i) => { if (pin[i] >= 0.5) { if (Number.isNaN(setLo)) setLo = w; setHi = w; } });
  let pk = 0;
  pin.forEach((v, i) => { if (v > pin[pk]) pk = i; });
  const peak = WG[pk];
  const pUndom = SEG.map(([a, b]) => cells.reduce((t, [lo, hi, p]) => t + (a <= hi && b >= lo ? p : 0), 0));
  return { dens, ciLo, ciHi, pBest, pin, setLo, setHi, peak, pUndom };
}, 24);

function simulateAll(p: P): P {
  let q = p;
  for (let i = 0; i < N; i++) q = { ...q, answers: simAnswer(q) };
  return q;
}

function simAnswer(p: P): string {
  const toks = p.answers.trim() ? p.answers.trim().split(",") : [];
  const k = toks.length;
  if (k >= N) return p.answers;
  const pr = PAIRS[k];
  const r = rng(p.seed * 7717 + k * 131 + 3);
  const z = normal(r);
  let tok: string;
  if (r() < 0.03) tok = "abnr"[Math.floor(r() * 4)];
  else {
    const lo = Math.min(p.lo, p.hi) + 0.02 * z(), hi = Math.max(p.lo, p.hi) + 0.02 * z();
    if (pr.t < lo) tok = pr.spaciousIsA ? "a" : "b";
    else if (pr.t > hi) tok = pr.spaciousIsA ? "b" : "a";
    else tok = r() < 0.4 ? "r" : "n";
  }
  return [...toks, tok].join(",");
}

function add(p: P, tok: string): P {
  const toks = p.answers.trim() ? p.answers.trim().split(",") : [];
  if (toks.length >= N) return p;
  return { ...p, answers: [...toks, tok].join(",") };
}

const count = (p: P) => (p.answers.trim() ? p.answers.trim().split(",").length : 0);

function flatsList(xs: number[], lang: Lang = "en"): string {
  const L = labels[lang];
  const names = xs.map((j) => String(j + 1));
  if (names.length <= 1) return names.length ? tpl(L.flat1, { a: names[0] }) : L.noFlat;
  if (names.length === 2) return tpl(L.flat2, { a: names[0], b: names[1] });
  return tpl(L.flatN, { list: names.slice(0, -1).join(L.listSep), last: names[names.length - 1] });
}

// The weights the set model gives at least even odds. On the edge of the grid
// the range can shrink to one grid point; say so instead of "0.07 to 0.07".
function setRange(lo: number, hi: number, lang: Lang = "en"): string {
  const L = labels[lang];
  return fixed(lo, 2) === fixed(hi, 2) ? tpl(L.rangePoint, { a: fixed(lo, 2) }) : tpl(L.rangeSet, { a: fixed(lo, 2), b: fixed(hi, 2) });
}

function describe(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const n = count(p);
  if (!n) return L.describe0;
  const f = fitAll(p.answers, p.seed);
  const rs = readAnswers(p.answers, p.seed);
  const u = rs.filter((r) => r.out === "U").length;
  let jb = 0;
  f.pBest.forEach((v, j) => { if (v > f.pBest[jb]) jb = j; });
  const keep = f.pUndom.map((v, j) => [v, j] as const).filter(([v]) => v >= 0.5).map(([, j]) => j);
  return tpl(L.describe, {
    n, u, fa: fixed(f.ciLo, 2), fb: fixed(f.ciHi, 2), fb1: jb + 1, fp: fixed(f.pBest[jb], 2),
    ia: Number.isNaN(f.setLo) ? tpl(L.iaNarrow, { w: fixed(f.peak, 2) }) : tpl(L.iaSet, { r: setRange(f.setLo, f.setHi, lang) }),
    iset: keep.length ? flatsList(keep, lang) : L.noFlatYet,
  });
}

function card(x: number, y: number, w: number, h: number, name: string, space: number, commute: number, narrow: boolean, lang: Lang = "en"): string {
  const L = labels[lang];
  const bx = x + 12, bw = w - 24;
  const sx = linear([40, 90], [0, bw]);
  const cx = linear([0, 60], [0, bw]);
  const fs = narrow ? TYPE.small : TYPE.body;
  return g({ "data-fig-hit": name, class: "fig-card", style: "cursor:pointer" },
    el("rect", { x, y, width: w, height: h, rx: 10, fill: C.panel, stroke: C.rule, "stroke-width": 1 }),
    text(x + 12, y + 20, tpl(L.flat, { s: name }), { "font-size": TYPE.title, class: "fig-t-strong" }),
    text(bx, y + 40, tpl(L.space, { v: space }), { "font-size": fs, class: "fig-t-num" }),
    el("rect", { x: bx, y: y + 45, width: bw, height: 7, rx: 3.5, fill: C.grid }),
    el("rect", { x: bx, y: y + 45, width: Math.max(4, sx(space)), height: 7, rx: 3.5, fill: C.c6 }),
    text(bx, y + 70, tpl(L.commute, { v: commute }), { "font-size": fs, class: "fig-t-num" }),
    el("rect", { x: bx, y: y + 75, width: bw, height: 7, rx: 3.5, fill: C.grid }),
    el("rect", { x: bx, y: y + 75, width: Math.max(4, cx(commute)), height: 7, rx: 3.5, fill: C.c7 }),
  );
}

// Glyphs for answers on the strip.
function glyph(kind: string, x: number, y: number, color: string, s = 5): string {
  if (kind === "S") return el("path", { d: `M${x - s * 0.8},${y - s}L${x + s},${y}L${x - s * 0.8},${y + s}Z`, fill: color });
  if (kind === "C") return el("path", { d: `M${x + s * 0.8},${y - s}L${x - s},${y}L${x + s * 0.8},${y + s}Z`, fill: color });
  if (kind === "n") return el("circle", { cx: x, cy: y, r: s * 0.85, fill: C.paper, stroke: color, "stroke-width": 1.6 });
  return el("path", { d: `M${x},${y - s}L${x + s},${y}L${x},${y + s}L${x - s},${y}Z`, fill: C.paper, stroke: color, "stroke-width": 1.6 });
}

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 480;
  const W = st.w;
  const n = count(p);
  const rs = readAnswers(p.answers, p.seed);
  const f = fitAll(p.answers, p.seed);
  const parts: string[] = [];
  const left = narrow ? 14 : 24, right = W - (narrow ? 14 : 24);
  const x = linear([0, 1], [left, right]);

  // 1. The question.
  let y = 18;
  parts.push(text(left, y, p.mode === "you" ? L.ask : L.askSim, { "font-size": TYPE.label, class: "fig-t-strong" }));
  parts.push(text(right, y, n < N ? tpl(L.pairOf, { k: n + 1, n: N }) : tpl(L.allAnswered, { n: N }), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-num" }));
  y += 10;
  const ch = 94;
  if (n < N) {
    const pr = PAIRS[n];
    const gap = narrow ? 10 : 24;
    const cw = narrow ? (right - left - gap) / 2 : Math.min(240, (right - left - gap) / 2);
    const x0 = (left + right) / 2 - cw - gap / 2;
    parts.push(card(x0, y, cw, ch, "A", pr.space[0], pr.commute[0], narrow, lang), card(x0 + cw + gap, y, cw, ch, "B", pr.space[1], pr.commute[1], narrow, lang));
  } else {
    parts.push(el("rect", { x: left, y, width: right - left, height: ch, rx: 10, fill: C.panel, stroke: C.rule }),
      text((left + right) / 2, y + ch / 2 + 4, narrow ? tpl(L.doneShort, { n: N }) : tpl(L.done, { n: N }), { "font-size": TYPE.body, "text-anchor": "middle", class: "fig-t-muted" }));
  }
  y += ch + 24;

  // 2. The answer strip: answers stack upward above the line, and what a
  // forced interface would have recorded stacks downward below it.
  parts.push(text(left, y, narrow ? L.stripShort : p.mode === "you" ? L.strip : L.stripSim, { "font-size": TYPE.small, class: "fig-t-muted" }));
  const placed: Array<[number, number]> = [];
  const levels = rs.map((r) => {
    const px = x(r.t);
    let lvl = 0;
    while (placed.some(([q, l]) => Math.abs(q - px) < 11 && l === lvl)) lvl++;
    placed.push([px, lvl]);
    return lvl;
  });
  const up = Math.max(0, ...levels);
  const stripY = y + 18 + up * 12;
  parts.push(el("line", { x1: left, x2: right, y1: stripY, y2: stripY, stroke: C.rule }));
  const below: Array<[number, number]> = [];
  let down = -1;
  rs.forEach((r, k) => {
    const px = x(r.t);
    const kind = r.out === "U" ? r.token : r.out;
    parts.push(glyph(kind, px, stripY - levels[k] * 12, C.ink));
    if (r.out === "U") {
      let lvl = 0;
      while (below.some(([q, l]) => Math.abs(q - px) < 9 && l === lvl)) lvl++;
      below.push([px, lvl]);
      down = Math.max(down, lvl);
      parts.push(glyph(r.forced, px, stripY + 11 + lvl * 9, C.c5, 3.6));
    }
  });
  if (!rs.length) parts.push(text((left + right) / 2, stripY - 4, L.none, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-faint" }));
  y = stripY + 16 + (down + 1) * 9;
  // legend for the glyphs
  const leg: Array<[string, string, string]> = [["S", L.legS, C.ink], ["C", L.legC, C.ink], ["n", L.legN, C.ink], ["r", L.legR, C.ink], ["S", L.legF, C.c5]];
  let lx = left, ly = y + 6;
  for (const [k, lab, col] of leg) {
    const tw = (lang === "zh" ? labelWidth(lab, 6.2, 11) : lab.length * 6.2) + 30;
    if (lx + tw > right && lx > left) { lx = left; ly += 16; }
    parts.push(glyph(k, lx + 5, ly - 4, col, k === "S" && col === C.c5 ? 3.6 : 4.5), text(lx + 14, ly, lab, { "font-size": TYPE.small, class: "fig-t-muted" }));
    lx += tw;
  }
  y = ly + 22;

  // Panel headers: title and readout on one line when wide, two lines when narrow.
  const panelH = narrow ? 58 : 64;
  const head = narrow ? 30 : 14;
  const topF = y + head, topI = topF + panelH + 22 + head;
  const header = (top: number, title: string, readout: string, color: string) => narrow
    ? [text(left, top - 20, title, { "font-size": TYPE.body, class: "fig-t-strong" }),
      text(left, top - 6, readout, { "font-size": TYPE.small, class: "fig-t-num", style: `fill:${color}` })]
    : [text(left, top - 6, title, { "font-size": TYPE.body, class: "fig-t-strong" }),
      text(right, top - 6, readout, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-num", style: `fill:${color}` })];

  // Truth band (simulation only), drawn behind both panels.
  if (p.mode === "sim" && p.truth) {
    const a = x(Math.min(p.lo, p.hi)), b = x(Math.max(p.lo, p.hi));
    for (const top of [topF, topI]) {
      parts.push(el("rect", { x: a, y: top, width: Math.max(1, b - a), height: panelH, fill: C.truth, opacity: 0.12 }),
        el("line", { x1: a, x2: a, y1: top, y2: top + panelH, stroke: C.truth, "stroke-width": 1.5, "stroke-dasharray": "4 3" }),
        el("line", { x1: b, x2: b, y1: top, y2: top + panelH, stroke: C.truth, "stroke-width": 1.5, "stroke-dasharray": "4 3" }));
    }
    const lab = narrow ? L.truthShort : L.truth;
    const fitsRight = b + 6 + (lang === "zh" ? labelWidth(lab, 6.2, 11) : lab.length * 6.2) < right;
    parts.push(text(fitsRight ? b + 6 : a - 6, topF + 14, lab, { "font-size": TYPE.small, class: "fig-t-muted fig-t-halo", "text-anchor": fitsRight ? "start" : "end" }));
  }

  // 3. Forced-choice posterior.
  const area = (vals: number[], top: number, color: string): string => {
    const yy = linear([0, 1], [top + panelH, top + 4]);
    const pts = WG.map((w, i) => `${x(w).toFixed(1)},${yy(vals[i]).toFixed(1)}`);
    return g({},
      el("path", { d: `M${x(0)},${top + panelH}L${pts.join("L")}L${x(1)},${top + panelH}Z`, fill: color, opacity: 0.22 }),
      el("path", { d: `M${pts.join("L")}`, fill: "none", stroke: color, "stroke-width": 2 }),
    );
  };
  parts.push(...header(topF, L.forced, n ? tpl(L.forcedCI, { a: fixed(f.ciLo, 2), b: fixed(f.ciHi, 2) }) : "", C.c5));
  parts.push(el("line", { x1: left, x2: right, y1: topF + panelH, y2: topF + panelH, stroke: C.rule }));
  parts.push(area(f.dens, topF, C.c5));
  if (n) {
    const by = topF + panelH + 7;
    parts.push(el("path", { d: `M${x(f.ciLo)},${by - 4}L${x(f.ciLo)},${by}L${x(f.ciHi)},${by}L${x(f.ciHi)},${by - 4}`, fill: "none", stroke: C.c5, "stroke-width": 1.6 }));
  }

  // 4. Incomplete-preference posterior: P(w is admissible).
  parts.push(...header(topI, L.incomplete, n ? (Number.isNaN(f.setLo) ? tpl(narrow ? L.incompleteNoneShort : L.incompleteNone, { w: fixed(f.peak, 2) }) : (fixed(f.setLo, 2) === fixed(f.setHi, 2) ? tpl(L.incompletePoint, { a: fixed(f.setLo, 2) }) : tpl(L.incompleteSet, { a: fixed(f.setLo, 2), b: fixed(f.setHi, 2) }))) : "", C.model));
  const yI = linear([0, 1], [topI + panelH, topI + 4]);
  parts.push(el("line", { x1: left, x2: right, y1: yI(0.5), y2: yI(0.5), stroke: C.grid, "stroke-dasharray": "3 3" }),
    text(left + 2, yI(0.5) - 3, "0.5", { "font-size": TYPE.small, class: "fig-t-faint fig-t-num" }),
    text(left + 2, yI(1) + 7, "1", { "font-size": TYPE.small, class: "fig-t-faint fig-t-num" }));
  parts.push(el("line", { x1: left, x2: right, y1: topI + panelH, y2: topI + panelH, stroke: C.rule }));
  parts.push(area(f.pin, topI, C.model));
  if (n && !Number.isNaN(f.setLo)) {
    const by = topI + panelH + 7;
    parts.push(el("path", { d: `M${x(f.setLo)},${by - 4}L${x(f.setLo)},${by}L${x(f.setHi)},${by}L${x(f.setHi)},${by - 4}`, fill: "none", stroke: C.model, "stroke-width": 1.6 }));
  }
  if (!narrow) parts.push(text(right, topI + 14, L.probIn, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo" }));

  // 5. The weight axis.
  let ay = topI + panelH + 22;
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    parts.push(el("line", { x1: x(v), x2: x(v), y1: ay - 6, y2: ay - 2, stroke: C.rule }),
      text(x(v), ay + 9, fixed(v, v === 0 || v === 1 ? 0 : 2), { "font-size": TYPE.small, "text-anchor": v === 0 ? "start" : v === 1 ? "end" : "middle", class: "fig-t-muted fig-t-num" }));
  }
  ay += 26;
  parts.push(text(left, ay, L.axisLeft, { "font-size": TYPE.small, class: "fig-t-muted" }),
    text(right, ay, L.axisRight, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted" }));
  if (!narrow) parts.push(text((left + right) / 2, ay, L.axisMid, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-strong" }));

  // 6. The shortlist: segments of the weight axis, with both models' verdicts.
  let sy = ay + 26;
  parts.push(text(left, sy, L.shortlist, { "font-size": TYPE.body, class: "fig-t-strong" }));
  sy += 18;
  const sw = (cx: number, col: string, lab: string) => [el("rect", { x: cx, y: sy - 9, width: 12, height: 10, rx: 2, fill: col, opacity: 0.85 }), text(cx + 18, sy, lab, { "font-size": TYPE.small, class: "fig-t-muted" })];
  parts.push(...sw(left, C.c5, L.pBest), ...sw(left + (narrow ? 130 : 170), C.model, L.pUndom));
  sy += 10;
  const barH = narrow ? 26 : 30;
  const rowF = sy, rowI = sy + barH + 4, segY = rowI + barH + 4;
  SEG.forEach(([a, b], j) => {
    const xa = x(a) + 1, xb = x(b) - 1, wd = xb - xa;
    parts.push(
      el("rect", { x: xa, y: rowF, width: wd, height: barH, fill: C.grid, opacity: 0.5 }),
      el("rect", { x: xa, y: rowF + barH * (1 - f.pBest[j]), width: wd, height: barH * f.pBest[j], fill: C.c5, opacity: 0.85 }),
      el("rect", { x: xa, y: rowI, width: wd, height: barH, fill: C.grid, opacity: 0.5 }),
      el("rect", { x: xa, y: rowI + barH * (1 - f.pUndom[j]), width: wd, height: barH * f.pUndom[j], fill: C.model, opacity: 0.85 }),
      el("rect", { x: xa, y: segY, width: wd, height: 18, rx: 3, fill: C.panel, stroke: C.rule }),
      text((xa + xb) / 2, segY + 13, String(j + 1), { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-strong fig-t-num" }),
    );
  });
  const H = segY + 18 + 8;
  return svg(W, H, describe(st), ...parts);
}

export default defineFigure({
  name: "econ-incomplete",
  title: { en: "Can't compare: what a forced binary likelihood does with answers that are not preferences", zh: "无法比较：强制二元似然如何处理不是偏好的回答" },
  labels,
  params,
  hint: { en: "Pick the flat you would rather rent. If the two feel impossible to rank, say so, or ask for a coin flip.", zh: "选出你更愿意租的公寓。如果觉得两套无法排出先后，就直接说出来，或者请求掷硬币。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  // "Things to try" quotes the simulated person's finished session.
  snapshots: { "simulated session": (p) => simulateAll({ ...(p as P), mode: "sim" }) },
  pointer(p, e) {
    if (e.phase !== "click" || (p as P).mode !== "you") return null;
    if (e.target === "A") return add(p as P, "a");
    if (e.target === "B") return add(p as P, "b");
    return null;
  },
  actions: [
    { label: { en: "Flat A", zh: "公寓 A" }, primary: true, run: (p) => add(p as P, "a"), enabled: (p) => (p as P).mode === "you" && count(p as P) < N },
    { label: { en: "Flat B", zh: "公寓 B" }, primary: true, run: (p) => add(p as P, "b"), enabled: (p) => (p as P).mode === "you" && count(p as P) < N },
    { label: { en: "Can't compare", zh: "无法比较" }, run: (p) => add(p as P, "n"), enabled: (p) => (p as P).mode === "you" && count(p as P) < N },
    { label: { en: "Flip a coin for me", zh: "替我掷硬币" }, run: (p) => add(p as P, "r"), enabled: (p) => (p as P).mode === "you" && count(p as P) < N },
    { label: { en: "Simulate one answer", zh: "模拟一个回答" }, run: (p) => ({ ...p, answers: simAnswer(p as P) }), enabled: (p) => (p as P).mode === "sim" && count(p as P) < N },
    { label: { en: "Simulate all", zh: "全部模拟" }, run: (p) => simulateAll(p as P), enabled: (p) => (p as P).mode === "sim" && count(p as P) < N },
    { label: { en: "Undo", zh: "撤销" }, run: (p) => { const t = (p as P).answers.trim().split(","); t.pop(); return { ...p, answers: t.join(",") }; }, enabled: (p) => (p as P).answers.trim() !== "" },
  ],
  update(p, key) {
    if (key === "mode") return { ...p, answers: "" };
    return p;
  },
});
