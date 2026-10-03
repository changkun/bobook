// The free-choice paradigm, run on the reader: rate six posters, choose
// between two pairs of posters you rated alike, then rate all six again. The
// result plots each poster's first and second rating, grouped into chosen,
// rejected, and never offered in a choice, and reports the "spread of
// alternatives": how much more the chosen posters gained than the rejected
// ones. One person's six posters cannot separate a real choice-induced change
// from the measurement artifact that the companion simulation shows; the
// figure says so.

import { defineFigure, type Lang, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { rng } from "./lib/random.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    step1: "Step 1 of 3 · rate poster {i} of 6",
    step2: "Step 2 of 3 · choice {i} of 2",
    step3: "Step 3 of 3 · rate poster {i} of 6 again",
    rateQ: "How much do you like this poster?",
    chooseQ: "Which of these two would you rather hang on your wall?",
    rate2Note: "Rate how you feel now; there is no need to match your earlier ratings.",
    low: "not at all",
    high: "very much",
    chosen: "chosen",
    rejected: "rejected",
    unpaired: "not offered",
    axis: "rating",
    first: "first rating",
    second: "second rating",
    resultTitle: "Your ratings before and after choosing",
    change: "average change: chosen {c}, rejected {r}, not offered {u}",
    spread: "Spread of alternatives: {s} points (chosen gain minus rejected gain).",
    caveat: "With two choices this is a demonstration. Part of any spread is an artifact of noisy first ratings; the next figure shows how much.",
    describeRate: "Step {k}: rate poster {i} of 6, {name}, on a scale from 1 to 9.",
    describeChoose: "Step 2: choose between {a} and {b}.",
    describeDone: "Spread of alternatives {s} points. Chosen posters changed by {c} on average, rejected by {r}, posters not offered by {u}.",
  },
  zh: {
    step1: "第 1 步（共 3 步）· 为第 {i} 张海报评分（共 6 张）",
    step2: "第 2 步（共 3 步）· 第 {i} 次选择（共 2 次）",
    step3: "第 3 步（共 3 步）· 再次为第 {i} 张海报评分（共 6 张）",
    rateQ: "你有多喜欢这张海报？",
    chooseQ: "这两张海报中，你更愿意把哪一张挂在墙上？",
    rate2Note: "按你现在的感受评分，不必与先前的评分一致。",
    low: "完全不喜欢",
    high: "非常喜欢",
    chosen: "被选中",
    rejected: "被拒绝",
    unpaired: "未参与选择",
    axis: "评分",
    first: "第一次评分",
    second: "第二次评分",
    resultTitle: "选择前后你的评分",
    change: "平均变化：被选中 {c}，被拒绝 {r}，未参与选择 {u}",
    spread: "选项分化：{s} 分（被选中者的增幅减去被拒绝者的增幅）。",
    caveat: "只有两次选择，这只是演示。任何分化中都有一部分是带噪声的第一次评分造成的伪影；下一张图显示这一部分有多大。",
    describeRate: "第 {k} 步：为第 {i} 张海报（共 6 张）“{name}”评分，范围为 1 至 9。",
    describeChoose: "第 2 步：在“{a}”与“{b}”之间选择。",
    describeDone: "选项分化为 {s} 分。被选中的海报平均变化 {c}，被拒绝的 {r}，未参与选择的 {u}。",
  },
};

interface Poster { name: string; zh: string; hue: number; motif: "sun" | "stripes" | "peak" | "rings" | "dots" | "band" }

const POSTERS: Poster[] = [
  { name: "the blue sun", zh: "蓝色太阳", hue: 208, motif: "sun" },
  { name: "the orange stripes", zh: "橙色条纹", hue: 24, motif: "stripes" },
  { name: "the green peak", zh: "绿色山峰", hue: 150, motif: "peak" },
  { name: "the pink rings", zh: "粉色圆环", hue: 335, motif: "rings" },
  { name: "the yellow dots", zh: "黄色圆点", hue: 46, motif: "dots" },
  { name: "the violet band", zh: "紫色斜带", hue: 265, motif: "band" },
];
const posterName = (i: number, lang: Lang) => (lang === "zh" ? POSTERS[i].zh : POSTERS[i].name);
const N = POSTERS.length;

type P = { answers: string; seed: number };

const params = {
  answers: {
    kind: "data", label: { en: "Answers", zh: "回答" }, default: "",
    validate: (s: string) => {
      const a = parse(s);
      if (a.length > 2 * N + 2) return "at most 14 answers";
      for (let i = 0; i < a.length; i++) {
        const ok = i === N || i === N + 1 ? a[i] === "L" || a[i] === "R" : /^[1-9]$/.test(a[i]);
        if (!ok) return "ratings are 1 to 9; the two choices are L or R";
      }
      return undefined;
    },
  },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 999, default: 3, step: 1, control: false },
} as const;

function parse(s: string): string[] {
  return s.trim() ? s.split(",").map((t) => t.trim()) : [];
}

function perm(n: number, seed: number): number[] {
  const r = rng(seed);
  const a = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
  return a;
}

interface Derived {
  order1: number[]; order2: number[];
  r1: Array<number | undefined>; r2: Array<number | undefined>;
  pairs: Array<[number, number]>; // display order: [left, right]
  chosen: number[]; rejected: number[]; unpaired: number[];
}

function derive(p: P): Derived {
  const a = parse(p.answers);
  const order1 = perm(N, p.seed * 7 + 1), order2 = perm(N, p.seed * 7 + 5);
  const r1: Array<number | undefined> = Array(N).fill(undefined), r2: Array<number | undefined> = Array(N).fill(undefined);
  for (let i = 0; i < Math.min(N, a.length); i++) r1[order1[i]] = Number(a[i]);
  for (let i = N + 2; i < a.length; i++) r2[order2[i - N - 2]] = Number(a[i]);
  // Pair posters with neighboring first ratings, as the paradigm does: ranks 2
  // and 3, and ranks 4 and 5. The highest and lowest are never offered.
  const pairs: Array<[number, number]> = [];
  const chosen: number[] = [], rejected: number[] = [], unpaired: number[] = [];
  if (a.length >= N) {
    const rank = Array.from({ length: N }, (_, i) => i).sort((i, j) => (r1[j]! - r1[i]!) || (i - j));
    const r = rng(p.seed * 11 + 3);
    for (const [u, v] of [[rank[1], rank[2]], [rank[3], rank[4]]] as Array<[number, number]>) pairs.push(r() < 0.5 ? [u, v] : [v, u]);
    unpaired.push(rank[0], rank[5]);
    for (let k = 0; k < 2 && N + k < a.length; k++) {
      const [L, R] = pairs[k];
      chosen.push(a[N + k] === "L" ? L : R);
      rejected.push(a[N + k] === "L" ? R : L);
    }
  }
  return { order1, order2, r1, r2, pairs, chosen, rejected, unpaired };
}

type Stage = { kind: "rate1"; i: number; item: number } | { kind: "choose"; i: number; pair: [number, number] } | { kind: "rate2"; i: number; item: number } | { kind: "done" };

function stage(p: P): Stage {
  const a = parse(p.answers);
  const d = derive(p);
  if (a.length < N) return { kind: "rate1", i: a.length, item: d.order1[a.length] };
  if (a.length < N + 2) return { kind: "choose", i: a.length - N, pair: d.pairs[a.length - N] };
  if (a.length < 2 * N + 2) return { kind: "rate2", i: a.length - N - 2, item: d.order2[a.length - N - 2] };
  return { kind: "done" };
}

const push = (p: P, tok: string): P => ({ ...p, answers: [...parse(p.answers), tok].join(",") });

function rate(p: P, v: number): P {
  const s = stage(p);
  return s.kind === "rate1" || s.kind === "rate2" ? push(p, String(v)) : p;
}
function choose(p: P, side: "L" | "R"): P {
  return stage(p).kind === "choose" ? push(p, side) : p;
}

const mean = (xs: number[]) => (xs.length ? xs.reduce((s, v) => s + v, 0) / xs.length : NaN);

function changes(d: Derived) {
  const ch = (ids: number[]) => mean(ids.map((i) => d.r2[i]! - d.r1[i]!));
  const c = ch(d.chosen), r = ch(d.rejected), u = ch(d.unpaired);
  return { c, r, u, s: c - r };
}

const signed = (v: number) => (v > 0 ? "+" : "") + fixed(v, 1);

function describe(st: State<P>): string {
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const s = stage(st.p);
  if (s.kind === "rate1" || s.kind === "rate2") return tpl(L.describeRate, { k: s.kind === "rate1" ? 1 : 3, i: s.i + 1, name: posterName(s.item, lang) });
  if (s.kind === "choose") return tpl(L.describeChoose, { a: posterName(s.pair[0], lang), b: posterName(s.pair[1], lang) });
  const c = changes(derive(st.p));
  return tpl(L.describeDone, { s: signed(c.s), c: signed(c.c), r: signed(c.r), u: signed(c.u) });
}

const fillOf = (h: number, l = 56) => `hsl(${h}, 58%, ${l}%)`;

// A small abstract poster, so the reader rates a design rather than a swatch.
function poster(uid: string, x: number, y: number, w: number, h: number, ps: Poster, hit?: string): string {
  const cid = `${uid}-fc-${ps.motif}-${Math.round(x)}`;
  const ink = "rgba(255,255,255,0.88)";
  const m: string[] = [];
  const cx = x + w / 2, cy = y + h / 2, s = Math.min(w, h);
  switch (ps.motif) {
    case "sun": m.push(el("circle", { cx: x + w * 0.66, cy: y + h * 0.4, r: s * 0.2, fill: ink }), el("rect", { x, y: y + h * 0.72, width: w, height: h * 0.28, fill: "rgba(0,0,0,0.12)" })); break;
    case "stripes": for (let i = 0; i < 5; i++) m.push(el("rect", { x: x + w * (0.12 + i * 0.16), y: y + h * 0.15, width: w * 0.07, height: h * 0.7, rx: 2, fill: ink, opacity: 0.55 + 0.09 * i })); break;
    case "peak": m.push(el("path", { d: `M${x + w * 0.1},${y + h * 0.82}L${x + w * 0.45},${y + h * 0.22}L${x + w * 0.62},${y + h * 0.52}L${x + w * 0.72},${y + h * 0.4}L${x + w * 0.92},${y + h * 0.82}Z`, fill: ink })); break;
    case "rings": for (let i = 1; i <= 3; i++) m.push(el("circle", { cx, cy, r: s * 0.12 * i, fill: "none", stroke: ink, "stroke-width": 3 })); break;
    case "dots": for (let i = 0; i < 4; i++) for (let j = 0; j < 3; j++) m.push(el("circle", { cx: x + w * (0.2 + i * 0.2), cy: y + h * (0.27 + j * 0.23), r: s * 0.05, fill: ink })); break;
    case "band": m.push(el("path", { d: `M${x},${y + h * 0.75}L${x + w},${y + h * 0.2}L${x + w},${y + h * 0.42}L${x},${y + h * 0.97}Z`, fill: ink })); break;
  }
  return g(hit ? { "data-fig-hit": hit } : {},
    el("rect", { x, y, width: w, height: h, rx: 8, fill: fillOf(ps.hue), stroke: C.rule, "stroke-width": 1 }),
    el("clipPath", { id: cid }, el("rect", { x, y, width: w, height: h, rx: 8 })),
    g({ "clip-path": `url(#${cid})` }, ...m),
  );
}

function renderRate(st: State<P>, s: Extract<Stage, { kind: "rate1" | "rate2" }>): string {
  const L = labels[st.lang ?? "en"];
  const w = st.w, narrow = w < 480, pad = narrow ? 8 : 16;
  const parts: string[] = [];
  parts.push(text(pad, 16, tpl(s.kind === "rate1" ? L.step1 : L.step3, { i: s.i + 1 }), { "font-size": TYPE.small, class: "fig-t-muted" }));
  parts.push(text(pad, 38, L.rateQ, { "font-size": TYPE.label, class: "fig-t-strong" }));
  const pw = narrow ? 170 : 210, ph = narrow ? 118 : 140;
  const py = 52;
  parts.push(poster(st.uid, (w - pw) / 2, py, pw, ph, POSTERS[s.item]));
  // the 1 to 9 scale
  const bw = Math.min(40, (w - 2 * pad - 8 * 5) / 9), gap = narrow ? 4 : 6;
  const total = 9 * bw + 8 * gap, x0 = (w - total) / 2, by = py + ph + 20;
  for (let k = 1; k <= 9; k++) {
    const x = x0 + (k - 1) * (bw + gap);
    parts.push(g({ "data-fig-hit": `rate-${k}` },
      el("rect", { x, y: by, width: bw, height: 32, rx: 6, fill: C.panel, stroke: C.rule }),
      text(x + bw / 2, by + 21, k, { "text-anchor": "middle", "font-size": TYPE.label, class: "fig-t-num fig-t-strong" }),
    ));
  }
  parts.push(text(x0, by + 48, L.low, { "font-size": TYPE.small, class: "fig-t-muted" }), text(x0 + total, by + 48, L.high, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted" }));
  let H = by + 56;
  if (s.kind === "rate2") { parts.push(text(pad, H + 12, L.rate2Note, { "font-size": TYPE.small, class: "fig-t-faint" })); H += 22; }
  return svg(w, H, describe(st), ...parts);
}

function renderChoose(st: State<P>, s: Extract<Stage, { kind: "choose" }>): string {
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const w = st.w, narrow = w < 480, pad = narrow ? 8 : 16;
  const parts: string[] = [];
  parts.push(text(pad, 16, tpl(L.step2, { i: s.i + 1 }), { "font-size": TYPE.small, class: "fig-t-muted" }));
  const q = narrow ? (lang === "zh" ? wrapZh(L.chooseQ, 40) : ["Which of these two would you rather", "hang on your wall?"]) : [L.chooseQ];
  q.forEach((ln, i) => parts.push(text(pad, 38 + i * 17, ln, { "font-size": TYPE.label, class: "fig-t-strong" })));
  const top = 38 + (q.length - 1) * 17 + 16;
  const gap = narrow ? 12 : 28;
  const pw = narrow ? (w - 2 * pad - gap) / 2 : Math.min(210, (w - 2 * pad - gap) / 2), ph = narrow ? 100 : 140;
  const x0 = (w - 2 * pw - gap) / 2;
  parts.push(poster(st.uid, x0, top, pw, ph, POSTERS[s.pair[0]], "left"), poster(st.uid, x0 + pw + gap, top, pw, ph, POSTERS[s.pair[1]], "right"));
  return svg(w, top + ph + 12, describe(st), ...parts);
}

function renderDone(st: State<P>): string {
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const w = st.w, narrow = w < 480, pad = narrow ? 8 : 16;
  const d = derive(st.p);
  const c = changes(d);
  const parts: string[] = [];
  parts.push(text(pad, 18, L.resultTitle, { "font-size": TYPE.label, class: "fig-t-strong" }));
  const left = narrow ? 34 : 48, right = w - pad, top = 40, bottom = top + (narrow ? 170 : 190);
  const y = linear([1, 9], [bottom, top]);
  for (let k = 1; k <= 9; k += 2) {
    parts.push(el("line", { x1: left, x2: right, y1: y(k), y2: y(k), stroke: C.grid }), text(left - 8, y(k) + 4, k, { "text-anchor": "end", "font-size": TYPE.small, class: "fig-t-num fig-t-muted" }));
  }
  parts.push(text(0, 0, L.axis, { "text-anchor": "middle", "font-size": TYPE.body, class: "fig-t-muted", transform: `translate(${narrow ? 10 : 14},${(top + bottom) / 2}) rotate(-90)` }));
  const groups: Array<[string, number[], string]> = [[L.chosen, d.chosen, C.c6], [L.rejected, d.rejected, C.c8], [L.unpaired, d.unpaired, C.ink3]];
  const gw = (right - left) / 3;
  groups.forEach(([name, ids, color], gi) => {
    const gx = left + gi * gw;
    parts.push(text(gx + gw / 2, bottom + 16, name, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-strong" }));
    ids.forEach((id, k) => {
      const x = gx + gw * (0.33 + 0.34 * k);
      const a = d.r1[id]!, b = d.r2[id]!;
      parts.push(
        el("line", { x1: x, x2: x, y1: y(a), y2: y(b), stroke: color, "stroke-width": 2.2 }),
        el("circle", { cx: x, cy: y(a), r: 4.5, fill: C.paper, stroke: color, "stroke-width": 2 }),
        el("circle", { cx: x, cy: y(b), r: 4.5, fill: color, stroke: C.paper, "stroke-width": 1.2 }),
        el("rect", { x: x - 9, y: bottom + 24, width: 18, height: 12, rx: 2, fill: fillOf(POSTERS[id].hue) }),
      );
    });
  });
  // legend for the dot kinds
  const ly = bottom + 54;
  parts.push(
    el("circle", { cx: pad + 6, cy: ly - 4, r: 4.5, fill: C.paper, stroke: C.ink2, "stroke-width": 2 }), text(pad + 16, ly, L.first, { "font-size": TYPE.small, class: "fig-t-muted" }),
    el("circle", { cx: pad + 112, cy: ly - 4, r: 4.5, fill: C.ink2 }), text(pad + 122, ly, L.second, { "font-size": TYPE.small, class: "fig-t-muted" }),
  );
  let ty = ly + 22;
  const lines = [tpl(L.change, { c: signed(c.c), r: signed(c.r), u: signed(c.u) }), tpl(L.spread, { s: signed(c.s) }), L.caveat];
  lines.forEach((line, i) => {
    for (const ln of (lang === "zh" ? wrapZh : wrapText)(line, narrow ? 50 : 92)) { parts.push(text(pad, ty, ln, { "font-size": i === 1 ? TYPE.label : TYPE.body, class: i === 1 ? "fig-t-strong" : i === 2 ? "fig-t-muted" : "" })); ty += 17; }
  });
  return svg(w, ty, describe(st), ...parts);
}

function wrapText(s: string, max: number): string[] {
  const out: string[] = [];
  let line = "";
  for (const wd of s.split(" ")) {
    if (line && (line + " " + wd).length > max) { out.push(line); line = wd; } else line = line ? `${line} ${wd}` : wd;
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

function render(st: State<P>): string {
  const s = stage(st.p);
  if (s.kind === "rate1" || s.kind === "rate2") return renderRate(st, s);
  if (s.kind === "choose") return renderChoose(st, s);
  return renderDone(st);
}

const rateAction = (k: number) => ({
  label: { en: String(k), zh: String(k) },
  run: (p: P) => rate(p, k),
  enabled: (p: P) => { const s = stage(p).kind; return s === "rate1" || s === "rate2"; },
});

export default defineFigure({
  name: "jdm-free-choice",
  title: { en: "Rate, choose, rate again: the free-choice paradigm on your own answers", zh: "评分、选择、再评分：用你自己的回答做自由选择范式" },
  labels,
  params,
  hint: { en: "Click a number to rate, click a poster to choose, or use the buttons. Answer quickly.", zh: "点击数字评分，点击海报做选择，或使用按钮。尽快作答。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click" || !e.target) return null;
    if (e.target.startsWith("rate-")) return rate(p as P, Number(e.target.slice(5)));
    if (e.target === "left") return choose(p as P, "L");
    if (e.target === "right") return choose(p as P, "R");
    return null;
  },
  actions: [
    ...[1, 2, 3, 4, 5, 6, 7, 8, 9].map(rateAction),
    { label: { en: "Left poster", zh: "左边的海报" }, primary: true, run: (p) => choose(p as P, "L"), enabled: (p) => stage(p as P).kind === "choose" },
    { label: { en: "Right poster", zh: "右边的海报" }, primary: true, run: (p) => choose(p as P, "R"), enabled: (p) => stage(p as P).kind === "choose" },
    { label: { en: "Undo", zh: "撤销" }, run: (p) => ({ ...p, answers: parse(p.answers).slice(0, -1).join(",") }), enabled: (p) => p.answers !== "" },
  ],
});
