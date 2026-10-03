// A paired preference test with placebo pairs, taken by the reader. Twelve
// pairs of small color posters are shown one at a time; the reader prefers
// the left one, the right one, or neither. Some pairs are identical (the
// placebo pairs of consumer testing); the others differ slightly and each
// appears twice with the sides swapped. After the last answer the figure
// reveals which pairs were which and reports the reader's own false-preference
// rate, side bias, and consistency, and the width of the "no preference" band
// that a Thurstonian model with a no-preference option (the 2-AC model) would
// infer from the placebo answers.

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { memo, rng } from "./lib/random.ts";
import { PhiInv } from "./lib/stats.ts";
import { fixed, tpl } from "./lib/format.ts";

const labels = {
  en: {
    question: "Which poster do you prefer?",
    progress: "pair {k} of {n}",
    left: "left",
    right: "right",
    pending: "Your results appear after the last pair.",
    done: "Your answers, revealed",
    same: "identical",
    diff: "different",
    placebo: "Identical pairs: you stated a preference on {k} of {n} ({p}).",
    side: "Of those preferences, {l} went to the left poster and {r} to the right.",
    consistent: "Repeated different pairs: same choice both times on {k} of {n}.",
    tauInf: "Thurstone reading: you never preferred between identical posters, so the model's 'no preference' band is wide (more than ±{t} noise units).",
    tauZero: "Thurstone reading: every identical pair got a preference, so the model sees no 'no preference' band at all.",
    tau: "Thurstone reading: your 'no preference' band is about ±{t} noise units (from only {n} pairs).",
    forced: "A forced-choice interface would have recorded all {n} identical pairs as preferences.",
    describe: "Paired preference test, {a} of {n} pairs answered.",
    describeDone: "Paired preference test finished. On identical pairs the reader stated a preference {k} of {n} times; on repeated different pairs they were consistent {c} of {m} times.",
  },
  zh: {
    question: "你更喜欢哪张海报？",
    progress: "第 {k} 对，共 {n} 对",
    left: "左",
    right: "右",
    pending: "回答完最后一对后显示你的结果。",
    done: "揭晓你的回答",
    same: "相同",
    diff: "不同",
    placebo: "相同的对：你在 {n} 对中的 {k} 对上表达了偏好（{p}）。",
    side: "这些偏好中，{l} 个给了左边的海报，{r} 个给了右边的海报。",
    consistent: "重复出现的不同对：{n} 对中有 {k} 对两次选择相同。",
    tauInf: "Thurstone 解读：你从未在相同的海报之间表达偏好，所以模型的“无偏好”区间很宽（超过 ±{t} 个噪声单位）。",
    tauZero: "Thurstone 解读：每个相同的对都得到了偏好，所以模型完全看不到“无偏好”区间。",
    tau: "Thurstone 解读：你的“无偏好”区间约为 ±{t} 个噪声单位（仅依据 {n} 对）。",
    forced: "强制选择界面会把全部 {n} 个相同的对都记录为偏好。",
    describe: "成对偏好测试，已回答 {n} 对中的 {a} 对。",
    describeDone: "成对偏好测试已完成。在相同的对上，读者 {n} 次中有 {k} 次表达了偏好；在重复出现的不同对上，{m} 次中有 {c} 次前后一致。",
  },
};

const N_SAME = 6;
const N_DIFF = 3; // each shown twice
const N = N_SAME + 2 * N_DIFF;

const params = {
  answers: {
    kind: "data", label: { en: "Answers", zh: "回答" }, default: "",
    validate: (s: string) => (s === "" || /^[LNR](,[LNR])*$/.test(s) ? undefined : "expected L, N, or R separated by commas"),
  },
  seed: { kind: "range", label: { en: "Session", zh: "会话" }, min: 1, max: 9999, default: 7, step: 1, control: false },
} as const;

type P = { answers: string; seed: number };

interface Trial { kind: "same" | "diff"; left: string; right: string; pair: number; swapped: boolean }

// Stimulus colors are data, not roles: each poster is an HSL color.
const BASES: Array<[number, number, number]> = [
  [200, 55, 52], [24, 72, 56], [146, 42, 46], [332, 48, 58], [262, 42, 58], [44, 74, 52],
  [180, 45, 44], [8, 60, 56], [96, 40, 48],
];
const hsl = ([h, s, l]: [number, number, number]) => `hsl(${h}, ${s}%, ${l}%)`;

const trials = memo((seed: number): Trial[] => {
  const r = rng(seed * 2654435761 + 11);
  const order = BASES.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) { const k = Math.floor(r() * (i + 1)); [order[i], order[k]] = [order[k], order[i]]; }
  const out: Trial[] = [];
  for (let i = 0; i < N_SAME; i++) {
    const c = hsl(BASES[order[i]]);
    out.push({ kind: "same", left: c, right: c, pair: -1, swapped: false });
  }
  for (let j = 0; j < N_DIFF; j++) {
    const [h, s, l] = BASES[order[N_SAME + j]];
    // A small difference: a few degrees of hue and a little lightness.
    const a = hsl([h - 2, s, l - 1]);
    const b = hsl([h + 2, s, l + 1]);
    out.push({ kind: "diff", left: a, right: b, pair: j, swapped: false });
    out.push({ kind: "diff", left: b, right: a, pair: j, swapped: true });
  }
  // Shuffle, keeping the two showings of a different pair apart.
  for (let attempt = 0; attempt < 200; attempt++) {
    for (let i = out.length - 1; i > 0; i--) { const k = Math.floor(r() * (i + 1)); [out[i], out[k]] = [out[k], out[i]]; }
    let ok = true;
    for (let i = 1; i < out.length; i++) if (out[i].pair >= 0 && out[i].pair === out[i - 1].pair) ok = false;
    if (ok) break;
  }
  return out;
}, 8);

const parse = (s: string) => (s ? s.split(",") : []) as Array<"L" | "N" | "R">;

function results(p: P) {
  const ts = trials(p.seed);
  const ans = parse(p.answers);
  let falsePref = 0, left = 0, right = 0;
  ts.forEach((t, i) => {
    if (t.kind !== "same" || !ans[i]) return;
    if (ans[i] !== "N") falsePref++;
    if (ans[i] === "L") left++;
    if (ans[i] === "R") right++;
  });
  // For each different pair: did the reader pick the same color both times?
  let consistent = 0;
  for (let j = 0; j < N_DIFF; j++) {
    const picks = ts.map((t, i) => ({ t, a: ans[i] })).filter((x) => x.t.pair === j && x.a);
    if (picks.length < 2) continue;
    const chosen = picks.map(({ t, a }) => (a === "N" ? "none" : (a === "L") !== t.swapped ? "a" : "b"));
    if (chosen[0] === chosen[1] && chosen[0] !== "none") consistent++;
  }
  const r = falsePref / N_SAME;
  // 2-AC Thurstonian model: identical stimuli give a perceived difference
  // D ~ N(0, 2 sigma^2); "no preference" when |D| <= tau, so
  // r = 2 Phi(-tau / (sqrt(2) sigma)) and tau / sigma = -sqrt(2) PhiInv(r / 2).
  const tauOf = (q: number) => -Math.SQRT2 * PhiInv(q / 2);
  const tau = r > 0 && r < 1 ? tauOf(r) : r === 0 ? tauOf(0.5 / N_SAME) : 0;
  return { ts, ans, falsePref, left, right, consistent, r, tau };
}

function describe(st: State<P>): string {
  const res = results(st.p);
  const L = labels[st.lang ?? "en"];
  if (res.ans.length < N) return tpl(L.describe, { a: res.ans.length, n: N });
  return tpl(L.describeDone, { k: res.falsePref, n: N_SAME, c: res.consistent, m: N_DIFF });
}

function poster(x: number, y: number, w: number, h: number, color: string, name: string, side: string): string {
  return g({ "data-fig-hit": name },
    el("rect", { x, y, width: w, height: h, rx: 12, fill: color, stroke: C.rule, "stroke-width": 1 }),
    el("circle", { cx: x + w * 0.7, cy: y + h * 0.36, r: Math.min(w, h) * 0.15, fill: "rgba(255,255,255,0.85)" }),
    el("rect", { x: x + w * 0.12, y: y + h * 0.64, width: w * 0.52, height: 8, rx: 4, fill: "rgba(255,255,255,0.9)" }),
    el("rect", { x: x + w * 0.12, y: y + h * 0.64 + 14, width: w * 0.32, height: 6, rx: 3, fill: "rgba(255,255,255,0.65)" }),
    text(x + w / 2, y + h + 16, side, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }),
  );
}

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 480;
  const res = results(p);
  const k = res.ans.length;
  const parts: string[] = [];
  const pad = narrow ? 8 : 16;

  if (k < N) {
    // The current pair.
    const t = res.ts[k];
    parts.push(text(st.w / 2, 18, L.question, { "text-anchor": "middle", "font-size": TYPE.title, class: "fig-t-strong" }));
    parts.push(text(st.w / 2, 36, tpl(L.progress, { k: k + 1, n: N }), { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
    const cw = narrow ? (st.w - 3 * pad) / 2 : Math.min(210, (st.w - 90) / 2);
    const chh = narrow ? 100 : 128;
    const gap = narrow ? pad : 36;
    const x0 = st.w / 2 - cw - gap / 2;
    parts.push(poster(x0, 48, cw, chh, t.left, "L", L.left), poster(st.w / 2 + gap / 2, 48, cw, chh, t.right, "R", L.right));
    // Progress strip.
    const sy = 48 + chh + 36;
    const bw = Math.min(26, (st.w - 2 * pad) / N - 4);
    const sx0 = st.w / 2 - (N * (bw + 4) - 4) / 2;
    for (let i = 0; i < N; i++) {
      parts.push(el("rect", {
        x: sx0 + i * (bw + 4), y: sy, width: bw, height: 8, rx: 3,
        fill: i < k ? C.ink2 : C.paper, stroke: i === k ? C.acq : C.rule, "stroke-width": i === k ? 2 : 1,
      }));
    }
    parts.push(text(st.w / 2, sy + 30, L.pending, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }));
    return svg(st.w, sy + 42, describe(st), ...parts);
  }

  // Finished: reveal every pair with its answer, then the summary.
  parts.push(text(pad, 18, L.done, { "font-size": TYPE.title, class: "fig-t-strong" }));
  const cols = narrow ? 3 : 6;
  const cellW = (st.w - 2 * pad) / cols;
  const sw = Math.min(30, cellW / 2 - 10);
  const rowH = sw + 40;
  res.ts.forEach((t, i) => {
    const cx = pad + (i % cols) * cellW + cellW / 2;
    const cy = 34 + Math.floor(i / cols) * rowH;
    const a = res.ans[i];
    const same = t.kind === "same";
    // Identical pairs with a stated preference are the false preferences.
    const flagged = same && a !== "N";
    parts.push(el("rect", { x: cx - sw - 3, y: cy, width: sw, height: sw, rx: 4, fill: t.left, stroke: a === "L" ? C.ink : "none", "stroke-width": 2.5 }));
    parts.push(el("rect", { x: cx + 3, y: cy, width: sw, height: sw, rx: 4, fill: t.right, stroke: a === "R" ? C.ink : "none", "stroke-width": 2.5 }));
    if (a === "N") parts.push(text(cx, cy + sw / 2 + 5, "=", { "text-anchor": "middle", "font-size": TYPE.title, class: "fig-t-strong fig-t-glyph" }));
    parts.push(text(cx, cy + sw + 15, same ? L.same : L.diff, {
      "text-anchor": "middle", "font-size": TYPE.small, class: flagged ? "fig-t-strong" : "fig-t-muted",
      fill: flagged ? C.bad : undefined,
    }));
  });
  let y = 34 + Math.ceil(N / cols) * rowH + 6;
  const lines = [
    tpl(L.placebo, { k: res.falsePref, n: N_SAME, p: `${Math.round(res.r * 100)}%` }),
    res.falsePref ? tpl(L.side, { l: res.left, r: res.right }) : "",
    tpl(L.consistent, { k: res.consistent, n: N_DIFF }),
    res.r === 0 ? tpl(L.tauInf, { t: fixed(res.tau, 1) }) : res.r === 1 ? L.tauZero : tpl(L.tau, { t: fixed(res.tau, 1), n: N_SAME }),
    tpl(L.forced, { n: N_SAME }),
  ].filter(Boolean);
  // Wrap long lines at phone width.
  const maxChars = Math.floor((st.w - 2 * pad) / 6.2);
  for (const line of lines) {
    for (const seg of lang === "zh" ? wrapWidth(line, st.w - 2 * pad) : wrap(line, maxChars)) {
      parts.push(text(pad, y, seg, { "font-size": TYPE.small, class: "fig-t-muted" }));
      y += 15;
    }
    y += 3;
  }
  return svg(st.w, y + 4, describe(st), ...parts);
}

function wrap(s: string, n: number): string[] {
  const out: string[] = [];
  let line = "";
  for (const word of s.split(" ")) {
    if ((line + " " + word).trim().length > n && line) { out.push(line); line = word; } else line = (line + " " + word).trim();
  }
  if (line) out.push(line);
  return out;
}

// Wrap by estimated width, for text without spaces between words (Chinese).
// Runs of Latin letters, digits, and their punctuation are kept whole, and a
// line never starts with closing Chinese punctuation.
function wrapWidth(s: string, w: number): string[] {
  const out: string[] = [];
  let line = "";
  for (const tok of s.match(/[A-Za-z0-9.,±%'()+\-]+ ?|[^]/gu) ?? []) {
    if (line && !/^[，。；：）”、！？]/.test(tok) && labelWidth((line + tok).trimEnd(), 6.2, 11.5) > w) { out.push(line.trimEnd()); line = tok.trimStart(); } else line += tok;
  }
  if (line.trim()) out.push(line.trimEnd());
  return out;
}

const answer = (p: P, a: "L" | "N" | "R"): P => {
  const ans = parse(p.answers);
  if (ans.length >= N) return p;
  ans.push(a);
  return { ...p, answers: ans.join(",") };
};

export default defineFigure({
  name: "ds-placebo",
  title: { en: "A paired preference test with placebo pairs", zh: "带安慰剂对的成对偏好测试" },
  labels,
  params,
  hint: { en: "Click the poster you prefer, or use the buttons. Choose No preference if you have none.", zh: "点击你更喜欢的海报，或使用按钮。没有偏好时选“无偏好”。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click") return null;
    if (e.target === "L" || e.target === "R") return answer(p as P, e.target);
    return null;
  },
  actions: [
    { label: { en: "Prefer left", zh: "选左边" }, primary: true, run: (p) => answer(p as P, "L"), enabled: (p) => parse(p.answers).length < N },
    { label: { en: "No preference", zh: "无偏好" }, primary: true, run: (p) => answer(p as P, "N"), enabled: (p) => parse(p.answers).length < N },
    { label: { en: "Prefer right", zh: "选右边" }, primary: true, run: (p) => answer(p as P, "R"), enabled: (p) => parse(p.answers).length < N },
    { label: { en: "Undo", zh: "撤销" }, run: (p) => ({ ...p, answers: parse(p.answers).slice(0, -1).join(",") }), enabled: (p) => p.answers !== "" },
    { label: { en: "Start over", zh: "重新开始" }, run: (p) => ({ ...p, answers: "", seed: (Number(p.seed) % 9999) + 1 }), enabled: (p) => p.answers !== "" },
  ],
});
