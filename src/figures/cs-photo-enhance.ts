// Enhancing a real photograph by comparison. Six adjustments (exposure,
// contrast, saturation, temperature, tint, shadows), each in [0, 1] with the
// original at 0.5, are applied to the photo with an SVG filter chain
// (lib/cs-photo.ts). The reader answers one of two kinds of question:
//
//   pairs   two versions, pick the better one; the next pair maximizes EUBO
//           over a random candidate pool (all pairs in the pool);
//   slider  one slider along the line from the best design so far to the
//           design with the highest expected improvement (sequential line
//           search); the choice is recorded as comparisons against both ends
//           and two interior points of the slider.
//
// A Gaussian process utility is learned from all answers with the probit
// likelihood and the Laplace approximation (lib/gp.ts). The figure shows the
// question, the best guess beside the original, the learned utility's effect
// curves through the best guess, and the history of choices. A pair can be
// asked again with the sides swapped, to measure consistency. In the
// simulated mode a hidden taste answers instead, so the result can be checked
// against a known favorite.

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { clamp, linear } from "./lib/scale.ts";
import { bandPath, linePath } from "./lib/svg.ts";
import { memo, rng } from "./lib/random.ts";
import { tpl } from "./lib/format.ts";
import {
  KNOBS, ORIGINAL, PHOTO_ASPECT, fitLog, fmtLog, gap, knobText, lerp, nextPair, nextSlider, parseLog, photo, round3, settingsText,
  simPair, simSlider, slices, taste, validLog, type Entry, type Pt,
} from "./lib/cs-photo.ts";

const labels = {
  en: {
    pickYou: "Which version of the photo do you prefer?",
    pickYouShort: "Which do you prefer?",
    pickSim: "The simulated taste is asked:",
    slideYou: "Slide until the photo looks best, then choose it.",
    slideShort: "Slide to the best, then choose.",
    slideSim: "The simulated taste searches this slider.",
    bestEnd: "best so far",
    startEnd: "original",
    eiEnd: "most promising",
    otherEnd: "random edit",
    best: "best guess",
    original: "original",
    favorite: "hidden favorite",
    effects: "learned utility along each adjustment, others held at the best guess",
    effectsShort: "learned utility per adjustment",
    choices: "chosen versions, oldest first",
    choicesRecent: "recent choices",
    none: "No answers yet. The first question shows the original against a random edit.",
    noneShort: "No answers yet.",
    status: "{n} answer{n:/s}",
    reps: " · asked again {k}, same answer {a}",
    gapTxt: " · utility gap {g} (original {g0})",
    describe: "After {n} answer{n:/s}, the best guess sets {settings}.",
    describeQ: " The current question compares A ({a}) with B ({b}).",
    describeS: " The current slider runs from {a} to {b} and is at {t}.",
    nothingYet: "nothing yet (the original)",
    bestIsOriginal: "best guess: original",
    same: "same",
    flipped: "flipped",
    limit: " · limit reached",
  },
  zh: {
    pickYou: "你更喜欢照片的哪个版本？",
    pickYouShort: "你更喜欢哪个？",
    pickSim: "向模拟品味提问：",
    slideYou: "拖动滑块，直到照片看起来最好，然后选定它。",
    slideShort: "滑到最好处，然后选定。",
    slideSim: "模拟品味在这条滑块上搜索。",
    bestEnd: "目前最好",
    startEnd: "原图",
    eiEnd: "最有希望",
    otherEnd: "随机编辑",
    best: "最佳猜测",
    original: "原图",
    favorite: "隐藏的最爱",
    effects: "沿每项调整学到的效用，其余调整固定在最佳猜测",
    effectsShort: "每项调整上学到的效用",
    choices: "选中的版本，从早到晚",
    choicesRecent: "最近的选择",
    none: "还没有回答。第一个问题把原图与一个随机编辑放在一起比较。",
    noneShort: "还没有回答。",
    status: "{n} 个回答",
    reps: " · 重问 {k} 次，答案相同 {a} 次",
    gapTxt: " · 效用差距 {g}（原图 {g0}）",
    describe: "{n} 个回答之后，最佳猜测的设置为：{settings}。",
    describeQ: "当前问题比较 A（{a}）与 B（{b}）。",
    describeS: "当前滑块从（{a}）到（{b}），位于 {t}。",
    nothingYet: "尚无（即原图）",
    bestIsOriginal: "最佳猜测：原图",
    same: "相同",
    flipped: "反转",
    limit: " · 已达上限",
  },
};

const params = {
  who: { kind: "choice", label: { en: "Who answers", zh: "谁来回答" }, options: [{ value: "you", label: { en: "You", zh: "你" } }, { value: "sim", label: { en: "Simulated taste", zh: "模拟品味" } }], default: "you" },
  form: { kind: "choice", label: { en: "Question", zh: "问题形式" }, options: [{ value: "pairs", label: { en: "Pick one of two", zh: "二选一" } }, { value: "slider", label: { en: "Slide along a line", zh: "沿直线滑动" } }], default: "pairs" },
  noise: { kind: "range", label: { en: "Simulated noise σ", zh: "模拟噪声 σ" }, min: 0.02, max: 0.4, default: 0.1, step: 0.01 },
  truth: { kind: "toggle", label: { en: "Reveal simulated favorite", zh: "显示模拟的最爱" }, default: false },
  log: { kind: "data", label: { en: "Answers", zh: "回答" }, default: "", validate: validLog },
  pos: { kind: "range", label: { en: "Slider position", zh: "滑块位置" }, min: -1, max: 1, default: -1, step: 0.01, control: false }, // -1: not moved yet
  rep: { kind: "range", label: { en: "Pair asked again", zh: "重问的一对" }, min: -1, max: 99, default: -1, step: 1, control: false },
  seed: { kind: "range", label: { en: "Seed", zh: "随机种子" }, min: 1, max: 9999, default: 7, step: 1, control: false },
} as const;

type P = { who: "you" | "sim"; form: "pairs" | "slider"; noise: number; truth: boolean; log: string; pos: number; rep: number; seed: number };

const MAX_ANSWERS = 24;

// ---------------------------------------------------------------------------
// The model and the next question, memoized on exactly what they depend on.

const model = memo((log: string) => fitLog(parseLog(log)), 16);

interface Question { kind: "pair" | "slider"; a: Pt; b: Pt; repeatOf?: number; tBest?: number; tTarget?: number }

const question = memo((log: string, form: string, seed: number, rep: number): Question => {
  const es = parseLog(log);
  if (form === "pairs" && rep >= 0 && rep < es.length && es[rep].kind === "pair") {
    const e = es[rep] as Extract<Entry, { kind: "pair" }>;
    return { kind: "pair", a: e.l, b: e.w, repeatOf: rep }; // sides swapped: the earlier loser is now A
  }
  const m = model(log);
  if (form === "slider") {
    const sl = nextSlider(m, seed, es.length);
    return { kind: "slider", a: sl.a, b: sl.b, tBest: sl.tBest, tTarget: sl.tTarget };
  }
  const [x, y] = nextPair(m, seed, es.length);
  // randomize sides, so the better-looking option is not always on one side
  return rng(seed * 31 + es.length * 7 + 1)() < 0.5 ? { kind: "pair", a: x, b: y } : { kind: "pair", a: y, b: x };
}, 16);

// The slider's position: where the reader put it, or, before they move it,
// a random position, as in Koyama et al. (2017), who set the initial slider
// position randomly "to reduce cognitive bias".
function slidePos(p: P): number {
  if (p.pos >= 0) return p.pos;
  const n = parseLog(p.log).length;
  return Math.round(rng(p.seed * 53 + n * 7 + 2)() * 100) / 100;
}

const sliceMemo = memo((log: string) => { const m = model(log); return slices(m, m.best); }, 8);

// Repeats: which earlier pair to ask again, and how the repeats came out.
function repeatCandidates(es: Entry[]): number[] {
  const asked = new Set<string>();
  es.forEach((e) => { if (e.kind === "pair" && e.rep) asked.add(key(e.w, e.l)); });
  const out: number[] = [];
  es.forEach((e, i) => { if (e.kind === "pair" && !e.rep && i < es.length - 2 && !asked.has(key(e.w, e.l))) out.push(i); });
  return out;
}
const key = (a: Pt, b: Pt) => [a.join("/"), b.join("/")].sort().join("|");

function repeatStats(es: Entry[]): { asked: number; same: number; marks: Map<number, boolean> } {
  const first = new Map<string, Pt>();
  const marks = new Map<number, boolean>();
  let asked = 0, same = 0;
  es.forEach((e, i) => {
    if (e.kind !== "pair") return;
    const k = key(e.w, e.l);
    if (!e.rep) { if (!first.has(k)) first.set(k, e.w); return; }
    const w0 = first.get(k);
    if (!w0) return;
    asked++;
    const ok = w0.join("/") === e.w.join("/");
    if (ok) same++;
    marks.set(i, ok);
  });
  return { asked, same, marks };
}

// ---------------------------------------------------------------------------
// Answers.

function answerPair(p: P, choice: "A" | "B"): P {
  const es = parseLog(p.log);
  if (es.length >= MAX_ANSWERS) return p;
  const q = question(p.log, p.form, p.seed, p.rep);
  const [w, l] = choice === "A" ? [q.a, q.b] : [q.b, q.a];
  es.push({ kind: "pair", w, l, rep: q.repeatOf != null });
  return { ...p, log: fmtLog(es), rep: -1, pos: -1 };
}

function answerSlider(p: P, t: number): P {
  const es = parseLog(p.log);
  if (es.length >= MAX_ANSWERS) return p;
  const q = question(p.log, "slider", p.seed, -1);
  es.push({ kind: "slider", a: q.a, b: q.b, t: Math.round(clamp(t, 0, 1) * 100) / 100 });
  return { ...p, log: fmtLog(es), rep: -1, pos: -1 };
}

function simulate(p: P): P {
  const es = parseLog(p.log);
  if (es.length >= MAX_ANSWERS) return p;
  const tz = taste(p.seed);
  const r = rng(p.seed * 977 + es.length * 31 + 5);
  const q = question(p.log, p.form, p.seed, p.rep);
  if (q.kind === "slider") return answerSlider(p, simSlider(tz, q.a, q.b, p.noise, r, es.length));
  return answerPair(p, simPair(tz, q.a, q.b, p.noise, r, es.length) ? "A" : "B");
}

function askAgain(p: P): P {
  const es = parseLog(p.log);
  const cand = repeatCandidates(es);
  if (!cand.length) return p;
  const r = rng(p.seed * 271 + es.length * 13 + 9);
  return { ...p, rep: cand[Math.floor(r() * cand.length)] };
}

const canAskAgain = (p: P) => p.form === "pairs" && p.rep < 0 && parseLog(p.log).length < MAX_ANSWERS && repeatCandidates(parseLog(p.log)).length > 0;

// ---------------------------------------------------------------------------
// Text helpers.

function describe(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const es = parseLog(p.log);
  const m = model(p.log);
  const q = question(p.log, p.form, p.seed, p.rep);
  let s = tpl(L.describe, { n: es.length, settings: es.length ? settingsText(m.best, lang) : L.nothingYet });
  if (q.kind === "pair") s += tpl(L.describeQ, { a: settingsText(q.a, lang), b: settingsText(q.b, lang) });
  else s += tpl(L.describeS, { a: settingsText(q.a, lang), b: settingsText(q.b, lang), t: slidePos(p).toFixed(2) });
  return s;
}

function star(cx: number, cy: number, r: number): string {
  let d = "";
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + (i * Math.PI) / 5;
    const rr = i % 2 ? r * 0.45 : r;
    d += `${i ? "L" : "M"}${(cx + rr * Math.cos(a)).toFixed(1)},${(cy + rr * Math.sin(a)).toFixed(1)}`;
  }
  return d + "Z";
}

function badge(x: number, y: number, s: string, fill: string = "rgba(0,0,0,.55)"): string {
  const w = s.length * 7.2 + 12;
  return g({},
    el("rect", { x, y, width: w, height: 18, rx: 9, fill }),
    text(x + w / 2, y + 13, s, { "text-anchor": "middle", "font-size": TYPE.small, style: "fill:#fff;font-weight:600" }),
  );
}

// ---------------------------------------------------------------------------
// Layout. The slider's hit area is shared by render and pointer.

function sliderLayout(w: number) {
  const narrow = w < 480;
  const pad = 8;
  const pvW = narrow ? w - 2 * pad : Math.min(440, w - 2 * pad);
  const pvH = Math.round(pvW / PHOTO_ASPECT);
  const px = (w - pvW) / 2;
  const top = 26;
  const thumbs = narrow ? 5 : 7;
  const gap = 5;
  const tw = (pvW - (thumbs - 1) * gap) / thumbs;
  const th = Math.round(tw / PHOTO_ASPECT);
  const filmY = top + pvH + 8;
  const trackY = filmY + th + 14;
  const t0 = px + tw / 2, t1 = px + pvW - tw / 2;
  return { narrow, pad, pvW, pvH, px, top, thumbs, gap, tw, th, filmY, trackY, t0, t1, bottom: trackY + 26 };
}

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const w = st.w;
  const narrow = w < 480;
  const pad = 8;
  const es = parseLog(p.log);
  const m = model(p.log);
  const q = question(p.log, p.form, p.seed, p.rep);
  const tz = taste(p.seed);
  const parts: string[] = [];
  const id = (s: string) => `${st.uid}-${s}`;
  let y = 0;

  // 1. The question.
  if (q.kind === "pair") {
    // A pair asked again looks like any other question: announcing the
    // repeat would invite the reader to answer for consistency.
    const title = p.who === "sim" ? L.pickSim : narrow ? L.pickYouShort : L.pickYou;
    parts.push(text(w / 2, 16, title, { "text-anchor": "middle", "font-size": TYPE.label, class: "fig-t-strong" }));
    const gap = narrow ? 8 : 12;
    const cw = (w - 2 * pad - gap) / 2;
    const ch = Math.round(cw / PHOTO_ASPECT);
    const top = 26;
    parts.push(
      photo(pad, top, cw, ch, q.a, id("A"), { hit: "A", rx: 8 }),
      photo(pad + cw + gap, top, cw, ch, q.b, id("B"), { hit: "B", rx: 8 }),
      badge(pad + 8, top + 8, "A"), badge(pad + cw + gap + 8, top + 8, "B"),
    );
    y = top + ch;
  } else {
    const lay = sliderLayout(w);
    const title = p.who === "sim" ? L.slideSim : narrow ? L.slideShort : L.slideYou;
    parts.push(text(w / 2, 16, title, { "text-anchor": "middle", "font-size": TYPE.label, class: "fig-t-strong" }));
    const pos = slidePos(p);
    const cur = round3(lerp(q.a, q.b, pos));
    parts.push(photo(lay.px, lay.top, lay.pvW, lay.pvH, cur, id("pv"), { rx: 8 }));
    for (let i = 0; i < lay.thumbs; i++) {
      const t = i / (lay.thumbs - 1);
      parts.push(photo(lay.px + i * (lay.tw + lay.gap), lay.filmY, lay.tw, lay.th, round3(lerp(q.a, q.b, t)), id(`f${i}`), { rx: 3 }));
    }
    const kx = lay.t0 + pos * (lay.t1 - lay.t0);
    const tx = (t: number) => lay.t0 + t * (lay.t1 - lay.t0);
    const bx = tx(q.tBest ?? 0), ex = tx(q.tTarget ?? 1);
    const leftFirst = bx <= ex;
    parts.push(
      el("line", { x1: lay.t0, x2: lay.t1, y1: lay.trackY, y2: lay.trackY, stroke: C.rule, "stroke-width": 4, "stroke-linecap": "round" }),
      el("line", { x1: lay.t0, x2: kx, y1: lay.trackY, y2: lay.trackY, stroke: C.acq, "stroke-width": 4, "stroke-linecap": "round" }),
      el("circle", { cx: kx, cy: lay.trackY, r: 8, fill: C.paper, stroke: C.acq, "stroke-width": 2.5 }),
      el("path", { d: star(bx, lay.trackY, 6), fill: C.c4, stroke: C.ink, "stroke-width": 0.7 }),
      el("rect", { x: ex - 4, y: lay.trackY - 4, width: 8, height: 8, fill: C.acq, stroke: C.paper, "stroke-width": 1, transform: `rotate(45 ${ex} ${lay.trackY})` }),
      text(bx, lay.trackY + 22, es.length ? L.bestEnd : L.startEnd, { "font-size": TYPE.small, "text-anchor": leftFirst ? "end" : "start", class: "fig-t-muted" }),
      text(ex, lay.trackY + 22, es.length ? L.eiEnd : L.otherEnd, { "font-size": TYPE.small, "text-anchor": leftFirst ? "start" : "end", class: "fig-t-muted" }),
      el("rect", {
        x: lay.px, y: lay.filmY, width: lay.pvW, height: lay.trackY + 12 - lay.filmY, fill: "transparent", "data-fig-hit": "slider",
        "data-fig-map": [lay.t0, lay.t1, 0, 1, lay.trackY, lay.trackY - 1, 0, 1].join(","),
      }),
    );
    y = lay.bottom;
  }

  // 2. The best guess beside the original (or the hidden favorite), and the
  // effect curves of the learned utility through the best guess.
  const reveal = p.who === "sim" && p.truth;
  const second = reveal ? tz.favorite(0) : ORIGINAL;
  const secondLabel = reveal ? L.favorite : L.original;
  const sl = sliceMemo(p.log);
  const top2 = y + 26;
  let curvesX: number, curvesTop: number, curvesW: number, cols: number;
  let smallBottom: number;
  if (narrow) {
    const gap = 8;
    const sw = (w - 2 * pad - gap) / 2;
    const sh = Math.round(sw / PHOTO_ASPECT);
    parts.push(
      text(pad, top2 - 6, es.length ? L.best : L.bestIsOriginal, { "font-size": TYPE.small, class: "fig-t-strong" }),
      text(pad + sw + gap, top2 - 6, secondLabel, { "font-size": TYPE.small, class: "fig-t-strong" }),
      photo(pad, top2, sw, sh, m.best, id("best"), { ring: C.c4, ringWidth: 2.5 }),
      photo(pad + sw + gap, top2, sw, sh, second, id("second"), { ring: reveal ? C.truth : undefined, ringWidth: reveal ? 2.5 : 1 }),
    );
    smallBottom = top2 + sh;
    curvesX = pad; curvesTop = smallBottom + 30; curvesW = w - 2 * pad; cols = 2;
  } else {
    const sw = 150;
    const sh = Math.round(sw / PHOTO_ASPECT);
    parts.push(
      text(pad, top2 - 6, es.length ? L.best : L.bestIsOriginal, { "font-size": TYPE.small, class: "fig-t-strong" }),
      photo(pad, top2, sw, sh, m.best, id("best"), { ring: C.c4, ringWidth: 2.5 }),
      text(pad, top2 + sh + 20, secondLabel, { "font-size": TYPE.small, class: "fig-t-strong" }),
      photo(pad, top2 + sh + 26, sw, sh, second, id("second"), { ring: reveal ? C.truth : undefined, ringWidth: reveal ? 2.5 : 1 }),
    );
    smallBottom = top2 + 2 * sh + 26;
    curvesX = pad + sw + 26; curvesTop = top2 + 10; curvesW = w - pad - curvesX; cols = 3;
  }
  parts.push(text(curvesX, curvesTop - 16, narrow ? L.effectsShort : L.effects, { "font-size": TYPE.small, class: "fig-t-muted" }));
  // shared vertical scale, so the panels show which adjustment matters most
  let lo = Infinity, hi = -Infinity;
  for (const s of sl) s.mean.forEach((v, i) => { lo = Math.min(lo, v - 1.96 * s.sd[i]); hi = Math.max(hi, v + 1.96 * s.sd[i]); });
  const rows = Math.ceil(KNOBS.length / cols);
  const cg = narrow ? 14 : 18;
  const pw = (curvesW - (cols - 1) * cg) / cols;
  const ph = narrow ? 46 : 52;
  const rowH = ph + 36;
  KNOBS.forEach((k, d) => {
    const cx = curvesX + (d % cols) * (pw + cg);
    const cy = curvesTop + Math.floor(d / cols) * rowH;
    const sx = linear([0, 1], [cx, cx + pw]);
    const sy = linear([lo, hi], [cy + 14 + ph, cy + 14]);
    const s = sl[d];
    parts.push(
      text(cx, cy + 8, knobText(d, lang).label, { "font-size": TYPE.small, class: "fig-t-strong" }),
      text(cx + pw, cy + 8, k.fmt(m.best[d]), { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-num" }),
      el("rect", { x: cx, y: cy + 14, width: pw, height: ph, fill: "none", stroke: C.grid }),
      el("path", { d: bandPath(s.xs.map((x, i) => [sx(x), sy(s.mean[i] + 1.96 * s.sd[i])]), s.xs.map((x, i) => [sx(x), sy(s.mean[i] - 1.96 * s.sd[i])])), fill: C.band }),
      el("path", { d: linePath(s.xs.map((x, i) => [sx(x), sy(s.mean[i])])), fill: "none", stroke: C.model, "stroke-width": 1.8 }),
      // the original's setting (0.5) as a faint tick
      el("line", { x1: sx(0.5), x2: sx(0.5), y1: cy + 14 + ph, y2: cy + 14 + ph + 4, stroke: C.ink3, "stroke-width": 1.2 }),
      text(cx, cy + ph + 27, knobText(d, lang).lo, { "font-size": TYPE.small, class: "fig-t-faint" }),
      text(cx + pw, cy + ph + 27, knobText(d, lang).hi, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-faint" }),
    );
    // where the current question sits on this adjustment
    if (q.kind === "pair") {
      const xa = sx(q.a[d]), xb = sx(q.b[d]);
      const marks: Array<[string, number]> = Math.abs(xa - xb) < 13 ? [["AB", (xa + xb) / 2]] : [["A", xa], ["B", xb]];
      for (const [lab, x] of marks) {
        parts.push(text(x, cy + 14 + ph - 3, lab, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-strong fig-t-halo", style: `fill:${C.acq}` }));
      }
    } else {
      const a = q.a[d], b = q.b[d];
      parts.push(
        el("line", { x1: sx(Math.min(a, b)), x2: sx(Math.max(a, b)) + 0.01, y1: cy + 14 + ph - 3, y2: cy + 14 + ph - 3, stroke: C.acq, "stroke-width": 3, "stroke-linecap": "round", opacity: 0.8 }),
        el("circle", { cx: sx(a + slidePos(p) * (b - a)), cy: cy + 14 + ph - 3, r: 3.5, fill: C.paper, stroke: C.acq, "stroke-width": 1.8 }),
      );
    }
    if (reveal) {
      const f = second[d];
      parts.push(el("line", { x1: sx(f), x2: sx(f), y1: cy + 14, y2: cy + 14 + ph, stroke: C.truth, "stroke-width": 2, "stroke-dasharray": "4 3" }));
    }
    if (es.length) {
      const i = Math.round(m.best[d] * (s.xs.length - 1));
      parts.push(el("path", { d: star(sx(m.best[d]), sy(s.mean[i]), 6), fill: C.c4, stroke: C.ink, "stroke-width": 0.7 }));
    }
  });
  const curvesBottom = curvesTop + rows * rowH - 6;
  y = Math.max(smallBottom, curvesBottom);

  // 3. History: the chosen versions, with repeats marked.
  const stats = repeatStats(es);
  const hy = y + 22;
  const maxThumbs = narrow ? 6 : 12;
  const hist = es.map((e, i) => ({ i, u: e.kind === "pair" ? e.w : round3(lerp(e.a, e.b, e.t)), mark: stats.marks.get(i) }));
  const shown = hist.slice(-maxThumbs);
  parts.push(text(pad, hy, es.length > maxThumbs ? L.choicesRecent : L.choices, { "font-size": TYPE.small, class: "fig-t-muted" }));
  const tw = narrow ? (w - 2 * pad - 5 * 6) / 6 : 46;
  const th = Math.round(tw / PHOTO_ASPECT);
  if (!shown.length) parts.push(text(pad, hy + 20, narrow ? L.noneShort : L.none, { "font-size": TYPE.small, class: "fig-t-faint" }));
  shown.forEach((h, j) => {
    const x = pad + j * (tw + 6);
    const ring = h.mark === undefined ? undefined : h.mark ? C.good : C.bad;
    parts.push(photo(x, hy + 8, tw, th, h.u, id(`h${h.i}`), { rx: 3, ring, ringWidth: ring ? 2.5 : 1 }));
    if (h.mark !== undefined) parts.push(text(x + tw / 2, hy + 8 + th + 12, h.mark ? L.same : L.flipped, { "text-anchor": "middle", "font-size": TYPE.small, style: `fill:${h.mark ? C.good : C.bad}` }));
  });
  const anyMark = shown.some((h) => h.mark !== undefined);
  const sy0 = hy + 8 + (shown.length ? th : 6) + (anyMark ? 30 : 18);
  let status = es.length ? tpl(L.status, { n: es.length }) : "";
  if (stats.asked) status += tpl(L.reps, { k: stats.asked, a: stats.same });
  if (es.length >= MAX_ANSWERS) status += L.limit;
  const gapLine = reveal && es.length ? tpl(L.gapTxt, { g: gap(tz, m.best).toFixed(2), g0: gap(tz, ORIGINAL).toFixed(2) }) : "";
  const lines = (narrow ? [status, gapLine.replace(/^ · /, "")] : [status + gapLine]).filter(Boolean);
  lines.forEach((ln, i) => parts.push(text(pad, sy0 + i * 16, ln, { "font-size": TYPE.small, class: "fig-t-muted" })));
  return svg(w, sy0 + 8 + Math.max(0, lines.length - 1) * 16 - (lines.length ? 0 : 14), describe(st), ...parts);
}

export default defineFigure({
  name: "cs-photo-enhance",
  title: { en: "Enhancing a real photograph by comparison: pairs chosen by EUBO, or a slider along a line", zh: "通过比较增强一张真实照片：由 EUBO 选出的一对，或沿直线的滑块" },
  labels,
  params,
  hint: { en: "Click the version you prefer, or drag along the strip and press Choose. The star marks the model's best guess.", zh: "点击你更喜欢的版本，或沿缩略图条拖动后按“就选这个”。星形标出模型的最佳猜测。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p0, e) {
    const p = p0 as P;
    if (p.who !== "you") return null;
    if (p.form === "pairs") {
      if (e.phase === "click" && (e.target === "A" || e.target === "B")) return answerPair(p, e.target);
      return null;
    }
    if (e.target !== "slider" || !e.data) return null;
    return { ...p, pos: Math.round(clamp(e.data.x, 0, 1) * 100) / 100 };
  },
  actions: [
    { label: { en: "Prefer A", zh: "选 A" }, primary: true, run: (p) => answerPair(p as P, "A"), enabled: (p) => p.who === "you" && p.form === "pairs" && parseLog(p.log).length < MAX_ANSWERS },
    { label: { en: "Prefer B", zh: "选 B" }, primary: true, run: (p) => answerPair(p as P, "B"), enabled: (p) => p.who === "you" && p.form === "pairs" && parseLog(p.log).length < MAX_ANSWERS },
    { label: { en: "◀ Slide", zh: "◀ 滑动" }, run: (p) => ({ ...p, pos: Math.max(0, Math.round((slidePos(p as P) - 0.05) * 100) / 100) }), enabled: (p) => p.who === "you" && p.form === "slider" },
    { label: { en: "Slide ▶", zh: "滑动 ▶" }, run: (p) => ({ ...p, pos: Math.min(1, Math.round((slidePos(p as P) + 0.05) * 100) / 100) }), enabled: (p) => p.who === "you" && p.form === "slider" },
    { label: { en: "Choose this one", zh: "就选这个" }, primary: true, run: (p) => answerSlider(p as P, slidePos(p as P)), enabled: (p) => p.who === "you" && p.form === "slider" && parseLog(p.log).length < MAX_ANSWERS },
    { label: { en: "Ask an earlier pair again", zh: "重问之前的一对" }, run: (p) => askAgain(p as P), enabled: (p) => canAskAgain(p as P) },
    { label: { en: "Simulate one", zh: "模拟一次" }, primary: true, run: (p) => simulate(p as P), enabled: (p) => p.who === "sim" && parseLog(p.log).length < MAX_ANSWERS },
    { label: { en: "Simulate five", zh: "模拟五次" }, run: (p) => { let q = p as P; for (let i = 0; i < 5; i++) q = simulate(q); return q; }, enabled: (p) => p.who === "sim" && parseLog(p.log).length < MAX_ANSWERS },
    { label: { en: "Undo", zh: "撤销" }, run: (p) => ({ ...p, log: fmtLog(parseLog(p.log).slice(0, -1)), rep: -1, pos: -1 }), enabled: (p) => p.log !== "" },
  ],
  update(p, key) {
    // The answers belong to one oracle: switching who answers starts over.
    if (key === "who") return { ...p, log: "", rep: -1, pos: -1 };
    if (key === "form") return { ...p, rep: -1, pos: -1 };
    return p;
  },
});
