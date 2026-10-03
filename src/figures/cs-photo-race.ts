// Four ways of asking, replayed against a known (simulated) taste. The
// sessions are precomputed by tools/figure-data/cs-photo-race.ts with the
// same model and question rules as cs-photo-enhance, against the simulated
// taste with seed 7 (that figure's default): 16 sessions per questioner,
// noise level, and stationary or drifting taste, 20 answers each.
//
// Top: the utility gap of the recommendation (the simulated taste's utility
// at its favorite minus at the recommendation; 0 is perfect) after each
// answer, median and interquartile range over the 16 sessions, against the
// number of answers or against seconds, with times per answer borrowed from
// two other studies (assumptions, not measurements of this task).
// Bottom: where the 16 sessions ended, for each adjustment, for EUBO pairs
// and the line slider, with the hidden favorite (and, if the taste drifts,
// where the favorite started and where it ended).

import { defineFigure, type State } from "./types.ts";
import { el, g, linePath, bandPath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { legend } from "./lib/plot.ts";
import { tpl } from "./lib/format.ts";
import { KNOBS, knobText } from "./lib/cs-photo.ts";
import { RACE } from "./lib/cs-photo-race-data.ts";

const labels = {
  en: {
    gapTitle: "utility gap of the recommendation (lower is better)",
    gapShort: "utility gap",
    answers: "answers",
    seconds: "seconds, assuming 4.2 s per pair and 17.2 s per slider answer (timings from other tasks)",
    secondsShort: "seconds (timings borrowed)",
    original: "original photo",
    eubo: "pairs, EUBO",
    random: "pairs, random",
    segment: "slider, published segment",
    line: "slider, full line",
    finals: "where the 16 sessions ended (after 20 answers)",
    finalsShort: "where the sessions ended",
    fav: "hidden favorite",
    favStart: "favorite at start",
    favEnd: "at end",
    describe: "With noise σ = {s}{d}, after 20 answers the median utility gap is {e} for EUBO pairs, {r} for random pairs, {g} for the published slider, and {l} for the full-line slider; the original photo's gap is {o}.",
    drifting: " and a drifting taste",
    mid: "0.5 (original)",
  },
  zh: {
    gapTitle: "推荐结果的效用差距（越低越好）",
    gapShort: "效用差距",
    answers: "回答数",
    seconds: "秒数，假设每对 4.2 秒、每次滑块回答 17.2 秒（时间取自其他任务）",
    secondsShort: "秒数（借用的时间）",
    original: "原图",
    eubo: "配对，EUBO",
    random: "配对，随机",
    segment: "滑块，已发表的线段",
    line: "滑块，整条直线",
    finals: "16 次会话的终点（20 个回答之后）",
    finalsShort: "各次会话的终点",
    fav: "隐藏的最爱",
    favStart: "起始时的最爱",
    favEnd: "结束时",
    describe: "噪声 σ = {s}{d}，20 个回答之后效用差距的中位数：EUBO 配对 {e}，随机配对 {r}，已发表的滑块 {g}，整条直线的滑块 {l}；原图的差距为 {o}。",
    drifting: "，品味漂移",
    mid: "0.5（原图）",
  },
};

const params = {
  noise: { kind: "choice", label: { en: "Simulated noise σ", zh: "模拟噪声 σ" }, options: [{ value: 0.05, label: { en: "0.05", zh: "0.05" } }, { value: 0.1, label: { en: "0.1", zh: "0.1" } }, { value: 0.25, label: { en: "0.25", zh: "0.25" } }], default: 0.1, control: "buttons" },
  axis: { kind: "choice", label: { en: "Horizontal axis", zh: "横轴" }, options: [{ value: "answers", label: { en: "Answers", zh: "回答数" } }, { value: "seconds", label: { en: "Seconds", zh: "秒数" } }], default: "answers" },
  drift: { kind: "toggle", label: { en: "Taste drifts during the session", zh: "品味在会话中漂移" }, default: false },
} as const;

type P = { noise: 0.05 | 0.1 | 0.25; axis: "answers" | "seconds"; drift: boolean };
type Form = "eubo" | "random" | "segment" | "line";
const FORMS: Form[] = ["eubo", "random", "segment", "line"];
const COLOR: Record<Form, string> = { eubo: C.acq, random: C.c7, segment: C.c5, line: C.c6 };
const DASH: Record<Form, string | undefined> = { eubo: undefined, random: undefined, segment: "5 3", line: undefined };
const SECONDS: Record<Form, number> = { eubo: 4.2, random: 4.2, segment: 17.2, line: 17.2 };
const T_MAX = 120;

type Sessions = Record<string, { gaps: readonly (readonly number[])[]; final: readonly (readonly number[])[] }>;
const sessions = (p: P, f: Form) => (RACE.sessions as unknown as Sessions)[`${f}|${p.noise}|${p.drift ? 1 : 0}`];

function quantile(v: number[], q: number): number {
  const s = [...v].sort((a, b) => a - b);
  const i = (s.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
  return s[lo] + (s[hi] - s[lo]) * (i - lo);
}

function summary(p: P, f: Form) {
  const s = sessions(p, f);
  const steps = RACE.steps;
  const at = (i: number) => s.gaps.map((g) => g[i]);
  return Array.from({ length: steps }, (_, i) => ({ n: i + 1, med: quantile(at(i), 0.5), q1: quantile(at(i), 0.25), q3: quantile(at(i), 0.75) }));
}

function describe(st: State<P>): string {
  const p = st.p;
  const last = (f: Form) => summary(p, f)[RACE.steps - 1].med.toFixed(2);
  const L = labels[st.lang ?? "en"];
  return tpl(L.describe, { s: p.noise, d: p.drift ? L.drifting : "", e: last("eubo"), r: last("random"), g: last("segment"), l: last("line"), o: RACE.gapOriginal.toFixed(2) });
}

function render(st: State<P>): string {
  const p = st.p;
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const w = st.w;
  const narrow = w < 480;
  const parts: string[] = [];
  // 1. Gap curves.
  const left = narrow ? 40 : 48, right = w - 12;
  const top = 26, height = narrow ? 150 : 180;
  const yMax = 1.4;
  const xs = p.axis === "answers" ? linear([0, RACE.steps], [left, right]) : linear([0, T_MAX], [left, right]);
  const ys = linear([0, yMax], [top + height, top]);
  parts.push(text(left - (narrow ? 34 : 40), 14, narrow ? L.gapShort : L.gapTitle, { "font-size": TYPE.small, class: "fig-t-strong" }));
  parts.push(axis({ scale: ys, orient: "left", at: left, span: [left, right], count: 4 }));
  parts.push(axis({ scale: xs, orient: "bottom", at: top + height, span: [top, top + height], count: narrow ? 4 : 6, title: p.axis === "answers" ? L.answers : narrow ? L.secondsShort : L.seconds }));
  const clipId = `${st.uid}-clip`;
  parts.push(el("clipPath", { id: clipId }, el("rect", { x: left, y: top - 2, width: right - left, height: height + 4 })));
  const inner: string[] = [];
  // the original photo's gap, the bar to beat
  const oy = ys(RACE.gapOriginal);
  inner.push(el("line", { x1: left, x2: right, y1: oy, y2: oy, stroke: C.ink3, "stroke-width": 1.2, "stroke-dasharray": "2 3" }));
  for (const f of FORMS) {
    const sm = summary(p, f);
    const xOf = (n: number) => (p.axis === "answers" ? xs(n) : xs(n * SECONDS[f]));
    const pts = [{ n: 0, med: RACE.gapOriginal, q1: RACE.gapOriginal, q3: RACE.gapOriginal }, ...sm];
    if (f !== "random") inner.push(el("path", { d: bandPath(pts.map((s) => [xOf(s.n), ys(Math.min(yMax, s.q3))]), pts.map((s) => [xOf(s.n), ys(Math.min(yMax, s.q1))])), fill: COLOR[f], opacity: 0.12 }));
    inner.push(el("path", { d: linePath(pts.map((s) => [xOf(s.n), ys(Math.min(yMax, s.med))])), fill: "none", stroke: COLOR[f], "stroke-width": 2.2, "stroke-dasharray": DASH[f], "stroke-linejoin": "round" }));
  }
  parts.push(g({ "clip-path": `url(#${clipId})` }, ...inner));
  parts.push(text(right - 4, oy - 5, L.original, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-halo" }));
  const lg = legend(left, top + height + 46, right - left, FORMS.map((f) => ({ kind: f === "segment" ? "dash" as const : "line" as const, color: COLOR[f], label: L[f] })));
  parts.push(lg.svg);
  // 2. Where the sessions ended.
  const t2 = top + height + 46 + lg.height + 22;
  parts.push(text(left - (narrow ? 34 : 40), t2, narrow ? L.finalsShort : L.finals, { "font-size": TYPE.small, class: "fig-t-strong" }));
  const labW = narrow ? 70 : 84;
  const sx = linear([0, 1], [left - (narrow ? 34 : 40) + labW, right - 6]);
  const rowH = 30;
  const r0 = t2 + 16;
  const fav0 = RACE.fav0 as readonly number[], fav1 = RACE.fav1 as readonly number[];
  KNOBS.forEach((k, d) => {
    const y = r0 + d * rowH;
    parts.push(
      text(left - (narrow ? 34 : 40), y + 13, knobText(d, lang).label, { "font-size": TYPE.small, class: "fig-t-muted" }),
      el("line", { x1: sx(0), x2: sx(1), y1: y + 9, y2: y + 9, stroke: C.grid, "stroke-width": 14, "stroke-linecap": "butt" }),
      el("line", { x1: sx(0.5), x2: sx(0.5), y1: y, y2: y + 18, stroke: C.ink3, "stroke-width": 1 }),
    );
    // favorites
    const f0 = sx(fav0[d]);
    parts.push(el("line", { x1: f0, x2: f0, y1: y - 2, y2: y + 20, stroke: C.truth, "stroke-width": 2.2 }));
    if (p.drift && Math.abs(fav1[d] - fav0[d]) > 1e-6) {
      const f1 = sx(fav1[d]);
      parts.push(
        el("line", { x1: f1, x2: f1, y1: y - 2, y2: y + 20, stroke: C.truth, "stroke-width": 2.2, "stroke-dasharray": "3 2" }),
        el("line", { x1: f0, x2: f1, y1: y - 2, y2: y - 2, stroke: C.truth, "stroke-width": 1.2 }),
      );
    }
    // final recommendations: EUBO pairs above, line slider below
    (["eubo", "line"] as Form[]).forEach((f, j) => {
      for (const fin of sessions(p, f).final) {
        parts.push(el("circle", { cx: sx(fin[d]), cy: y + 5 + j * 8, r: 2.6, fill: COLOR[f], opacity: 0.75 }));
      }
    });
  });
  // a scale under the last row, in the adjustments' own words
  const yb = r0 + KNOBS.length * rowH;
  parts.push(
    text(sx(0), yb + 6, "0", { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted fig-t-num" }),
    text(sx(0.5), yb + 6, L.mid, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted fig-t-num" }),
    text(sx(1), yb + 6, "1", { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted fig-t-num" }),
  );
  const lg2 = legend(left - (narrow ? 34 : 40), yb + 30, w - 20, [
    { kind: "dot", color: COLOR.eubo, label: L.eubo },
    { kind: "dot", color: COLOR.line, label: L.line },
    { kind: "line", color: C.truth, label: p.drift ? L.favStart : L.fav },
    ...(p.drift ? [{ kind: "dash" as const, color: C.truth, label: L.favEnd }] : []),
  ]);
  parts.push(lg2.svg);
  const H = yb + 30 + lg2.height;
  return svg(w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "cs-photo-race",
  title: { en: "Four ways of asking, replayed against a simulated taste", zh: "四种提问方式，以模拟品味回放" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
