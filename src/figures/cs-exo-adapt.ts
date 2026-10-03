// What a walker's adaptation does to the tuners. Precomputed simulated
// sessions (tools/figure-data/cs-exo-race.ts) on 40 simulated walkers from
// lib/cs-exo.ts, each as an already adapted walker and as a novice whose
// metabolic landscape changes during the session. The lines show, minute by
// minute, the true metabolic reduction of each tuner's current
// recommendation: on the landscape of that moment, or on the landscape the
// walker will have once fully adapted. Everything is simulated; the walker is
// calibrated to published numbers (see lib/cs-exo.ts).

import { defineFigure, type State } from "./types.ts";
import { bandPath, el, g, linePath, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { legend, type Swatch } from "./lib/plot.ts";
import { tpl } from "./lib/format.ts";
import { EXO_RACE } from "./lib/cs-exo-race-data.ts";

const labels = {
  en: {
    yNow: "metabolic reduction of the current recommendation (%, on the walker as they are at that minute)",
    yLater: "metabolic reduction of the current recommendation (%, on the walker once fully adapted)",
    yShortNow: "metabolic reduction (%), walker as they are now",
    yShortLater: "metabolic reduction (%), walker once adapted",
    x: "minutes of assisted walking",
    best: "best possible",
    generic: "generic setting",
    bo: "Bayesian opt. (metabolic)",
    boTime: "Bayesian opt. with time as an input",
    boExploit: "Bayesian opt., less exploration",
    cma: "CMA-ES (metabolic)",
    self: "simulated self-tuner",
    pbo: "preferential BO (felt)",
    describe: "For {k} simulated walkers, after {t} minutes the median reduction is {bo}% for Bayesian optimization, {cma}% for CMA-ES, {self}% for the simulated self-tuner, and {gen}% for the generic setting; the best possible is {best}%.",
  },
  zh: {
    yNow: "当前推荐的代谢降幅（%，按行走者在该分钟的状态评判）",
    yLater: "当前推荐的代谢降幅（%，按完全适应后的行走者评判）",
    yShortNow: "代谢降幅（%），行走者此刻的状态",
    yShortLater: "代谢降幅（%），适应后的行走者",
    x: "辅助行走的分钟数",
    best: "可能的最优值",
    generic: "通用设置",
    bo: "贝叶斯优化（代谢）",
    boTime: "以时间为输入的贝叶斯优化",
    boExploit: "贝叶斯优化，探索较少",
    cma: "CMA-ES（代谢）",
    self: "模拟的自调者",
    pbo: "偏好贝叶斯优化（感觉）",
    describe: "对 {k} 位模拟行走者，{t} 分钟后代谢降幅的中位数为：贝叶斯优化 {bo}%，CMA-ES {cma}%，模拟的自调者 {self}%，通用设置 {gen}%；可能的最优值为 {best}%。",
  },
};

const params = {
  walker: { kind: "choice", label: { en: "Walkers", zh: "行走者" }, options: [{ value: "novice", label: { en: "New to the device", zh: "初次使用装置" } }, { value: "expert", label: { en: "Already adapted", zh: "已经适应" } }], default: "novice" },
  judge: { kind: "choice", label: { en: "Judge each setting on", zh: "评判设置所用的行走者" }, options: [{ value: "now", label: { en: "The walker now", zh: "此刻的行走者" } }, { value: "later", label: { en: "The walker once adapted", zh: "适应后的行走者" } }], default: "now" },
  variants: { kind: "toggle", label: { en: "Show Bayesian optimization variants", zh: "显示贝叶斯优化的变体" }, default: false },
} as const;

type P = { walker: "novice" | "expert"; judge: "now" | "later"; variants: boolean };
type Q = readonly (readonly number[])[];
type Method = "bo" | "boTime" | "boExploit" | "cma" | "self" | "pbo";
interface Cond { best: Q; generic: Q; bo: { now: Q; later: Q }; boTime: { now: Q; later: Q }; boExploit: { now: Q; later: Q }; cma: { now: Q; later: Q }; self: { now: Q; later: Q }; pbo: { now: Q; later: Q } }

const COLOR: Record<Method, string> = { bo: C.acq, boTime: C.c6, boExploit: C.c8, cma: C.c7, self: C.c4, pbo: C.c5 };
const DASH: Partial<Record<Method, string>> = { boTime: "6 3", boExploit: "2 3" };
const TIMES = EXO_RACE.times as readonly number[];

const cond = (p: P) => (EXO_RACE as unknown as Record<string, Cond>)[p.walker];
const shown = (p: P): Method[] => (p.variants ? ["bo", "boTime", "boExploit", "cma", "self", "pbo"] : ["bo", "cma", "self", "pbo"]);

function describe(st: State<P>): string {
  const p = st.p, c = cond(p);
  const i = TIMES.indexOf(60);
  const v = (q: Q) => (q[i][1] * 100).toFixed(1);
  const expertLater = p.judge === "later" ? (EXO_RACE as unknown as Record<string, Cond>).expert : c;
  return tpl(labels[st.lang ?? "en"].describe, { k: EXO_RACE.walkers, t: 60, bo: v(c.bo[p.judge]), cma: v(c.cma[p.judge]), self: v(c.self[p.judge]), gen: v(p.judge === "later" ? expertLater.generic : c.generic), best: v(p.judge === "later" ? expertLater.best : c.best) });
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const w = st.w;
  const narrow = w < 480;
  const c = cond(p);
  // judged on the adapted walker, the best and the generic setting are the adapted walker's
  const ref = p.judge === "later" ? (EXO_RACE as unknown as Record<string, Cond>).expert : c;
  const parts: string[] = [];
  const left = narrow ? 34 : 44, right = w - 12;
  const top = 28, height = narrow ? 200 : 250;
  const xs = linear([0, 120], [left, right]);
  const ys = linear([0, 0.42], [top + height, top]);
  parts.push(text(left - (narrow ? 30 : 38), 14, narrow ? (p.judge === "now" ? L.yShortNow : L.yShortLater) : (p.judge === "now" ? L.yNow : L.yLater), { "font-size": TYPE.small, class: "fig-t-strong" }));
  parts.push(axis({ scale: ys, orient: "left", at: left, span: [left, right], ticks: [0, 0.1, 0.2, 0.3, 0.4], format: (v) => String(Math.round(v * 100)) }));
  parts.push(axis({ scale: xs, orient: "bottom", at: top + height, span: [top, top + height], ticks: narrow ? [0, 30, 60, 90, 120] : [0, 20, 40, 60, 80, 100, 120], title: L.x }));
  const clipId = `${st.uid}-clip`;
  parts.push(el("clipPath", { id: clipId }, el("rect", { x: left, y: top - 2, width: right - left, height: height + 4 })));
  const inner: string[] = [];
  const pts = (q: Q, k: number) => q.map((v, i) => [xs(TIMES[i]), ys(Math.max(0, v[k]))] as [number, number]);
  inner.push(el("path", { d: linePath(pts(ref.best, 1)), fill: "none", stroke: C.truth, "stroke-width": 2, "stroke-dasharray": "5 4" }));
  inner.push(el("path", { d: linePath(pts(ref.generic, 1)), fill: "none", stroke: C.ink3, "stroke-width": 1.6, "stroke-dasharray": "2 3" }));
  for (const m of shown(p)) {
    const q = c[m][p.judge];
    if (m === "bo" || m === "self") inner.push(el("path", { d: bandPath(pts(q, 2), pts(q, 0)), fill: COLOR[m], opacity: 0.12 }));
    inner.push(el("path", { d: linePath(pts(q, 1)), fill: "none", stroke: COLOR[m], "stroke-width": 2.1, "stroke-dasharray": DASH[m], "stroke-linejoin": "round" }));
  }
  parts.push(g({ "clip-path": `url(#${clipId})` }, ...inner));
  const items: Swatch[] = [
    { kind: "dash", color: C.truth, label: L.best },
    { kind: "dash", color: C.ink3, label: L.generic },
    ...shown(p).map((m) => ({ kind: (DASH[m] ? "dash" : "line") as "dash" | "line", color: COLOR[m], label: L[m] })),
  ];
  const lg = legend(left, top + height + 46, right - left, items);
  parts.push(lg.svg);
  return svg(w, top + height + 46 + lg.height, describe(st), ...parts);
}

export default defineFigure({
  name: "cs-exo-adapt",
  title: { en: "What a walker's adaptation does to four tuners (calibrated simulation)", zh: "行走者的适应对四种调节方式的影响（校准过的模拟）" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
