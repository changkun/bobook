// Every evaluation sequence of the mesh-simplification study of Ou, Buschek,
// Mayer, and Butz (Mensch und Computer 2022, @ou2022human), one square per
// sequence, as reported in the chapter: in the 3-month field deployment with
// two professional 3D artists, 415 of 549 sequences stopped at the first
// iteration, the other 134 asked for optimization, and 16 of those ended
// satisfied; in the lab study with 20 participants, 97 of 200 sequences ended
// satisfied (the paper's lab numbers are not split by where a sequence
// stopped). The control changes only the denominator of the field percentage.

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { tpl } from "./lib/format.ts";

const FIELD = { total: 549, stopped: 415, asked: 134, satisfied: 16 };
const LAB = { total: 200, satisfied: 97 };

const labels = {
  en: {
    fieldTitle: "Field: 3 months, 2 professional artists",
    labTitle: "Lab: 20 participants",
    sat: "ended satisfied",
    asked: "asked for optimization, not satisfied",
    stopped: "stopped at the first iteration",
    other: "lab: other endings (not split)",
    fieldAll: "{s} of {n} sequences = {p}",
    fieldAsked: "{s} of {n} that asked = {p}",
    lab: "{s} of {n} sequences = {p}",
    describeAll: "Field deployment: 549 sequences; 415 stopped at the first iteration, 134 asked for optimization, and 16 ended satisfied, 2.9% of all sequences. Lab study: 97 of 200 sequences (48.5%) ended satisfied.",
    describeAsked: "Field deployment, counting only the 134 sequences that asked for optimization: 16 ended satisfied, 11.9%; the 415 that stopped at the first iteration are left out of the count. Lab study: 97 of 200 sequences (48.5%) ended satisfied.",
  },
  zh: {
    fieldTitle: "现场：3 个月，2 名专业美术师",
    labTitle: "实验室：20 名被试",
    sat: "以满意结束",
    asked: "请求了优化，未满意",
    stopped: "停在第一次迭代",
    other: "实验室：其他结局（未细分）",
    fieldAll: "{n} 个序列中的 {s} 个 = {p}",
    fieldAsked: "请求了优化的 {n} 个中的 {s} 个 = {p}",
    lab: "{n} 个序列中的 {s} 个 = {p}",
    describeAll: "现场部署：549 个序列；415 个停在第一次迭代，134 个请求了优化，16 个以满意结束，占全部序列的 2.9%。实验室研究：200 个序列中有 97 个（48.5%）以满意结束。",
    describeAsked: "现场部署，只计请求了优化的 134 个序列：16 个以满意结束，占 11.9%；停在第一次迭代的 415 个序列不计入。实验室研究：200 个序列中有 97 个（48.5%）以满意结束。",
  },
};

const params = {
  count: {
    kind: "choice",
    label: { en: "Count against", zh: "计数基准" },
    options: [
      { value: "all", label: { en: "All sequences", zh: "全部序列" } },
      { value: "asked", label: { en: "Sequences that asked for optimization", zh: "请求了优化的序列" } },
    ],
    default: "all",
    control: "buttons",
  },
} as const;

type P = { count: string };

const pctText = (a: number, b: number) => `${((100 * a) / b).toFixed(1)}%`;

function describe(st: State<P>): string {
  const L = labels[st.lang ?? "en"];
  return st.p.count === "asked" ? L.describeAsked : L.describeAll;
}

type Kind = "sat" | "asked" | "stopped" | "other";

function fill(kind: Kind, faded: boolean): Record<string, string | number> {
  if (kind === "sat") return { fill: C.c6 };
  if (kind === "asked") return { fill: C.c1 };
  if (kind === "stopped") return faded ? { fill: "none", stroke: C.rule, "stroke-width": 0.8 } : { fill: C.ink3, opacity: 0.55 };
  return { fill: C.panel, stroke: C.rule, "stroke-width": 0.8 };
}

// A grid of squares filled in reading order from a list of kinds.
function grid(x0: number, y0: number, cols: number, cell: number, kinds: Kind[], faded: boolean): string {
  const s = cell - 2;
  const parts: string[] = [];
  kinds.forEach((k, i) => {
    const cx = x0 + (i % cols) * cell, cy = y0 + Math.floor(i / cols) * cell;
    parts.push(el("rect", { x: cx, y: cy, width: s, height: s, rx: 1.5, ...fill(k, faded) }));
  });
  return parts.join("");
}

function render(st: State<P>): string {
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const asked = st.p.count === "asked";
  const pad = narrow ? 8 : 12;
  const parts: string[] = [];

  // Legend, wrapping as needed.
  const items: Array<[Kind, string]> = [["sat", L.sat], ["asked", L.asked], ["stopped", L.stopped], ["other", L.other]];
  let lx = pad, ly = 14;
  for (const [k, lab] of items) {
    const wpx = labelWidth(lab, 6.1) + 22;
    if (lx + wpx > st.w - pad && lx > pad) { lx = pad; ly += 18; }
    parts.push(el("rect", { x: lx, y: ly - 9, width: 10, height: 10, rx: 1.5, ...fill(k, false) }),
      text(lx + 15, ly, lab, { "font-size": TYPE.small, class: "fig-t-muted" }));
    lx += wpx;
  }

  const fieldCols = 30, labCols = 20;
  const fieldRows = Math.ceil(FIELD.total / fieldCols), labRows = Math.ceil(LAB.total / labCols);
  const inner = st.w - 2 * pad;
  const gap = 28;
  const cell = narrow ? Math.min(12, inner / fieldCols) : Math.min(13, (inner - gap) / (fieldCols + labCols));

  const fieldKinds: Kind[] = [
    ...Array<Kind>(FIELD.satisfied).fill("sat"),
    ...Array<Kind>(FIELD.asked - FIELD.satisfied).fill("asked"),
    ...Array<Kind>(FIELD.stopped).fill("stopped"),
  ];
  const labKinds: Kind[] = [...Array<Kind>(LAB.satisfied).fill("sat"), ...Array<Kind>(LAB.total - LAB.satisfied).fill("other")];

  const top = ly + 26;
  const fx = pad, fy = top + 40;
  const lxp = narrow ? pad : pad + fieldCols * cell + gap;
  const lyp = narrow ? fy + fieldRows * cell + 64 : fy;

  const fieldRead = asked
    ? tpl(L.fieldAsked, { s: FIELD.satisfied, n: FIELD.asked, p: pctText(FIELD.satisfied, FIELD.asked) })
    : tpl(L.fieldAll, { s: FIELD.satisfied, n: FIELD.total, p: pctText(FIELD.satisfied, FIELD.total) });
  const labRead = tpl(L.lab, { s: LAB.satisfied, n: LAB.total, p: pctText(LAB.satisfied, LAB.total) });

  parts.push(g({ "data-fig-set": `count=${asked ? "all" : "asked"}`, style: "cursor:pointer" },
    el("rect", { x: fx - 4, y: top - 4, width: fieldCols * cell + 6, height: fy - top + fieldRows * cell + 6, fill: "transparent" }),
    text(fx, top + 10, L.fieldTitle, { "font-size": TYPE.label, class: "fig-t-strong" }),
    text(fx, top + 28, fieldRead, { "font-size": TYPE.body, class: "fig-t-num" }),
    grid(fx, fy, fieldCols, cell, fieldKinds, asked),
  ));
  parts.push(
    text(lxp, lyp - 30, L.labTitle, { "font-size": TYPE.label, class: "fig-t-strong" }),
    text(lxp, lyp - 12, labRead, { "font-size": TYPE.body, class: "fig-t-num" }),
    grid(lxp, lyp, labCols, cell, labKinds, false),
  );

  const h = Math.max(fy + fieldRows * cell, lyp + labRows * cell) + 10;
  return svg(st.w, h, describe(st), ...parts);
}

export default defineFigure({
  name: "hci-field-funnel",
  title: { en: "Every sequence of a PBO field deployment and lab study", zh: "一次偏好贝叶斯优化现场部署与实验室研究中的每个序列" },
  labels,
  params,
  hint: { en: "Switch the denominator, or click the field panel, to see how the field percentage changes.", zh: "切换分母，或点击现场面板，看现场百分比如何变化。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  snapshots: { asked: (p) => ({ ...p, count: "asked" }) },
});
