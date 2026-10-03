// Right about the population, wrong about the person. Numbers from PRISM-X
// (Kirk, Leqi, Zeng, Davidson, Vidgen, Summerfield, and Hale, arXiv 2605.13307,
// a 2026 preprint, @kirk2026prism), as reported in the chapter's table of
// judge studies: 530 people each ranked four models; GPT-4o as a simulated
// user either ranked the person's transcripts only or also held the
// conversations. Model scores fitted to simulated and to human rankings
// correlate at r = 0.98 to 0.99 (the two conditions); the mean Kendall's tau
// between a simulated ranking and the person's own is 0.22 (ranking only) and
// 0.11 (conversing), against 0.57 for the consistency of people's ratings with
// their own rankings. The model shown in the first of four positions was
// ranked best in 33.8% (ranking only), 44.9% (conversing), and 24.1% (people)
// of trials; the one in the fourth position in 15.0%, 7.2%, and 28.6%.

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { tpl } from "./lib/format.ts";

const AGG = { lo: 0.98, hi: 0.99 };
const TAU = { rank: 0.22, converse: 0.11 };
const PEOPLE_SELF = 0.57;
const POS = {
  people: { first: 24.1, fourth: 28.6 },
  rank: { first: 33.8, fourth: 15.0 },
  converse: { first: 44.9, fourth: 7.2 },
};

const labels = {
  en: {
    left: "Agreement with people",
    right: "Ranked best, by position shown",
    agg: "model scores, simulated vs human (r)",
    ind: "each simulated ranking vs the person's own (τ)",
    self: "people's ratings vs their own rankings (τ)",
    both: "both conditions",
    first: "shown 1st",
    fourth: "shown 4th",
    people: "people",
    rank: "ranks transcripts",
    converse: "holds conversations",
    chance: "25%, no position effect",
    describe: "PRISM-X, simulated user that {cond}: model scores agree with those fitted to human rankings at r = 0.98 to 0.99, but each simulated ranking agrees with the person's own at a mean Kendall's tau of {tau}, against 0.57 for people's own consistency. It ranked the model shown first best in {f}% of trials and the one shown fourth in {l}%; people did so in 24.1% and 28.6%.",
    condRank: "ranks the person's transcripts only",
    condConverse: "also holds the conversations",
    aggVal: "0.98 to 0.99",
    faint: "{a} (faint: {b})",
  },
  zh: {
    left: "与人的一致性",
    right: "被排为最佳的比例（按展示位置）",
    agg: "模型得分：模拟排序与人类排序（r）",
    ind: "每个模拟排序与此人自己的排序（τ）",
    self: "人的评分与其自己的排序（τ）",
    both: "两种条件",
    first: "展示于第 1 位",
    fourth: "展示于第 4 位",
    people: "人",
    rank: "对对话记录排序",
    converse: "亲自进行对话",
    chance: "25%，无位置效应",
    describe: "PRISM-X 中{cond}的模拟用户：模型得分与根据人类排序拟合的得分相关，r = 0.98 至 0.99，但每个模拟排序与此人自己的排序之间的平均 Kendall τ 只有 {tau}，而人自身的一致性为 0.57。它把展示在第一位的模型排为最佳的试次占 {f}%，展示在第四位的占 {l}%；人的这两个比例为 24.1% 与 28.6%。",
    condRank: "只对此人的对话记录排序",
    condConverse: "还亲自进行对话",
    aggVal: "0.98 至 0.99",
    faint: "{a}（浅色：{b}）",
  },
};

const params = {
  sim: {
    kind: "choice",
    label: { en: "Simulated user", zh: "模拟用户" },
    options: [
      { value: "rank", label: { en: "Ranks transcripts only", zh: "只对对话记录排序" } },
      { value: "converse", label: { en: "Also holds the conversations", zh: "还亲自进行对话" } },
    ],
    default: "rank",
    control: "buttons",
  },
} as const;

type P = { sim: string };
type Sim = "rank" | "converse";
const cur = (p: P): Sim => (p.sim === "converse" ? "converse" : "rank");
const SIM_COLOR: Record<Sim, string> = { rank: C.c7, converse: C.c5 };

function describe(st: State<P>): string {
  const s = cur(st.p);
  const L = labels[st.lang ?? "en"];
  return tpl(L.describe, {
    cond: s === "rank" ? L.condRank : L.condConverse,
    tau: TAU[s].toFixed(2),
    f: POS[s].first.toFixed(1),
    l: POS[s].fourth.toFixed(1),
  });
}

function leftPanel(x0: number, y0: number, w: number, s: Sim, L: Labels): { svg: string; h: number } {
  const x = linear([0, 1], [x0, x0 + w]);
  const parts: string[] = [text(x0, y0 + 12, L.left, { "font-size": TYPE.label, class: "fig-t-strong" })];
  const rowH = 44;
  const top = y0 + 26;
  const bar = (row: number, v: number, color: string, opacity = 1) =>
    el("rect", { x: x(0), y: top + row * rowH + 16, width: Math.max(1, x(v) - x(0)), height: 12, rx: 2, fill: color, opacity });
  const label = (row: number, t: string) => text(x0, top + row * rowH + 10, t, { "font-size": TYPE.small, class: "fig-t-muted" });
  const val = (row: number, v: string, at: number) => text(Math.min(at + 5, x0 + w - 2), top + row * rowH + 26, v, { "font-size": TYPE.small, class: "fig-t-num", "text-anchor": at + 40 > x0 + w ? "end" : "start" });

  // Gridlines.
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    parts.push(el("line", { x1: x(v), x2: x(v), y1: top + 12, y2: top + 3 * rowH, stroke: C.grid }),
      text(x(v), top + 3 * rowH + 13, v === 0 || v === 1 ? String(v) : v.toFixed(2), { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted fig-t-num" }));
  }
  // Row 0: aggregate, both conditions.
  parts.push(label(0, L.agg), bar(0, AGG.lo, C.ink2),
    el("line", { x1: x(AGG.lo), x2: x(AGG.hi), y1: top + 22, y2: top + 22, stroke: C.ink2, "stroke-width": 3 }));
  parts.push(text(x(AGG.lo) - 4, top + 26, L.aggVal, { "font-size": TYPE.small, class: "fig-t-num", "text-anchor": "end", fill: C.paper }));
  // Row 1: per person, selected condition, the other faint.
  const other: Sim = s === "rank" ? "converse" : "rank";
  parts.push(label(1, L.ind),
    g({ "data-fig-set": `sim=${other}`, style: "cursor:pointer" }, el("rect", { x: x(0), y: top + rowH + 16, width: x(TAU[other]) - x(0), height: 12, rx: 2, fill: SIM_COLOR[other], opacity: 0.3 })),
    bar(1, TAU[s], SIM_COLOR[s]),
    TAU[other] > TAU[s]
      ? val(1, tpl(L.faint, { a: TAU[s].toFixed(2), b: TAU[other].toFixed(2) }), x(TAU[other]))
      : val(1, TAU[s].toFixed(2), x(TAU[s])));
  // Row 2: people's own consistency.
  parts.push(label(2, L.self), bar(2, PEOPLE_SELF, C.ink3), val(2, PEOPLE_SELF.toFixed(2), x(PEOPLE_SELF)));
  return { svg: parts.join(""), h: 26 + 3 * rowH + 18 };
}

function rightPanel(x0: number, y0: number, w: number, s: Sim, L: Labels): { svg: string; h: number } {
  const parts: string[] = [text(x0, y0 + 12, L.right, { "font-size": TYPE.label, class: "fig-t-strong" })];
  const top = y0 + 30, plotH = 120;
  const left = x0 + 26;
  const y = linear([0, 50], [top + plotH, top]);
  for (const v of [0, 10, 20, 30, 40, 50]) {
    parts.push(el("line", { x1: left, x2: x0 + w, y1: y(v), y2: y(v), stroke: C.grid }),
      text(left - 4, y(v) + 4, `${v}%`, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-num" }));
  }
  const groups: Array<{ key: "people" | Sim; name: string; color: string }> = [
    { key: "people", name: L.people, color: C.ink3 },
    { key: "rank", name: L.rank, color: SIM_COLOR.rank },
    { key: "converse", name: L.converse, color: SIM_COLOR.converse },
  ];
  const slotW = (x0 + w - left) / 2;
  const bw = Math.min(26, (slotW - 16) / 3);
  (["first", "fourth"] as const).forEach((pos, pi) => {
    const cx = left + slotW * pi + slotW / 2;
    groups.forEach((gr, gi) => {
      const v = POS[gr.key][pos];
      const bx = cx + (gi - 1.5) * bw + 1;
      const dim = gr.key !== "people" && gr.key !== s;
      const item = g(gr.key === "people" ? {} : { "data-fig-set": `sim=${gr.key}`, style: "cursor:pointer" },
        el("rect", { x: bx, y: y(v), width: bw - 2, height: y(0) - y(v), rx: 2, fill: gr.color, opacity: dim ? 0.3 : 1 }),
        dim ? "" : text(bx + (bw - 2) / 2, y(v) - 4, v.toFixed(1), { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-num" }));
      parts.push(item);
    });
    parts.push(text(cx, top + plotH + 15, pos === "first" ? L.first : L.fourth, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted" }));
  });
  parts.push(el("line", { x1: left, x2: x0 + w, y1: y(25), y2: y(25), stroke: C.ink2, "stroke-dasharray": "4 3" }));
  // Legend under the plot, with the dashed reference line as its last entry.
  let lx = x0, ly = top + plotH + 34;
  for (const gr of groups) {
    const wpx = labelWidth(gr.name, 6.1) + 22;
    if (lx + wpx > x0 + w && lx > x0) { lx = x0; ly += 16; }
    parts.push(el("rect", { x: lx, y: ly - 9, width: 10, height: 10, rx: 2, fill: gr.color, opacity: gr.key !== "people" && gr.key !== s ? 0.3 : 1 }),
      text(lx + 14, ly, gr.name, { "font-size": TYPE.small, class: "fig-t-muted" }));
    lx += wpx;
  }
  const cw = labelWidth(L.chance, 6.1) + 30;
  if (lx + cw > x0 + w && lx > x0) { lx = x0; ly += 16; }
  parts.push(el("line", { x1: lx, x2: lx + 16, y1: ly - 4, y2: ly - 4, stroke: C.ink2, "stroke-dasharray": "4 3" }),
    text(lx + 20, ly, L.chance, { "font-size": TYPE.small, class: "fig-t-muted" }));
  return { svg: parts.join(""), h: ly - y0 + 8 };
}

type Labels = typeof labels.en;

function render(st: State<P>): string {
  const s = cur(st.p);
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const pad = narrow ? 8 : 12;
  if (narrow) {
    const a = leftPanel(pad, 4, st.w - 2 * pad, s, L);
    const b = rightPanel(pad, a.h + 18, st.w - 2 * pad, s, L);
    return svg(st.w, a.h + 18 + b.h + 6, describe(st), a.svg, b.svg);
  }
  const gap = 32;
  const lw = (st.w - 2 * pad - gap) * 0.55, rw = st.w - 2 * pad - gap - lw;
  const a = leftPanel(pad, 4, lw, s, L);
  const b = rightPanel(pad + lw + gap, 4, rw, s, L);
  return svg(st.w, Math.max(a.h, b.h) + 10, describe(st), a.svg, b.svg);
}

export default defineFigure({
  name: "llm-judge-fidelity",
  title: { en: "A simulated user, right in aggregate and wrong for the person", zh: "总体上正确、对具体的人却出错的模拟用户" },
  labels,
  params,
  hint: { en: "Choose a simulator condition, or click its bars, to highlight it.", zh: "选择一种模拟条件，或点击它的条形，使之突出显示。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  snapshots: { converse: (p) => ({ ...p, sim: "converse" }) },
});
