// What a stated choice predicts about real behavior. A hundred people either
// take up a health option in real life or refuse it; each is marked by what a
// discrete choice experiment (DCE) predicted for them. Sensitivity (the share
// of actual take-ups the DCE predicted) and specificity (the share of actual
// refusals it predicted) are the pooled estimates of two meta-analyses that
// compared stated with actual choices, as the chapter states them:
//
//   Quaife et al. 2018 (European Journal of Health Economics, bib key
//     quaife2018well): 8 studies, 6 in the meta-analysis; sensitivity 88%,
//     specificity 34%, area under the ROC curve 0.60.
//   Zhang et al. 2025 (eClinicalMedicine, bib key zhang2025prediction): 14
//     studies, 10 in the meta-analysis; sensitivity 89%, specificity 52%,
//     area under the curve 0.81.
//
// These are the numbers en/perspectives/08-design-and-senses.md states in
// @sec-ds-health.
//
// The share of people who actually take the option up is an illustrative
// parameter, not from either source. Counts are rounded to whole people, and
// the share of predicted take-ups that are real follows from the rounded
// counts.

import { defineFigure, type Lang, type State } from "./types.ts";
import { el, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { fixed, pct, tpl } from "./lib/format.ts";

const SOURCES = {
  quaife: { sens: 0.88, spec: 0.34, auc: 0.6, k: 6, cite: "Quaife et al. 2018", citeZh: "Quaife 等人（2018）" },
  zhang: { sens: 0.89, spec: 0.52, auc: 0.81, k: 10, cite: "Zhang et al. 2025", citeZh: "Zhang 等人（2025）" },
} as const;

const citeOf = (src: (typeof SOURCES)[keyof typeof SOURCES], lang: Lang) => (lang === "zh" ? src.citeZh : src.cite);

const labels = {
  en: {
    took: "Took it up in real life ({n})",
    refused: "Refused in real life ({n})",
    sens: "DCE predicted take-up for {a}: sensitivity {s}",
    spec: "DCE predicted refusal for {b}: specificity {s}",
    filled: "filled: DCE predicted take-up",
    hollow: "hollow: DCE predicted refusal",
    ppv: "Predicted to take it up: {p}. Of them, {a} did ({r}).",
    auc: "{cite}, {k} studies pooled; area under the ROC curve {auc} (0.5 is chance).",
    describe: "With the pooled estimates of {cite} (sensitivity {s}, specificity {sp}) and {nt} of 100 people actually taking up the option, the stated-choice experiment predicts that {p} will take it up, and {a} of them do ({r}).",
  },
  zh: {
    took: "现实中接受了（{n}）",
    refused: "现实中拒绝了（{n}）",
    sens: "DCE 预测其中 {a} 人接受：灵敏度 {s}",
    spec: "DCE 预测其中 {b} 人拒绝：特异度 {s}",
    filled: "实心：DCE 预测接受",
    hollow: "空心：DCE 预测拒绝",
    ppv: "预测会接受：{p} 人。其中 {a} 人确实接受（{r}）。",
    auc: "{cite}，合并 {k} 项研究；ROC 曲线下面积 {auc}（0.5 为随机水平）。",
    describe: "采用 {cite}的合并估计（灵敏度 {s}，特异度 {sp}），且 100 人中有 {nt} 人实际接受该选项时，陈述选择实验预测 {p} 人会接受，其中 {a} 人确实接受（{r}）。",
  },
};

const params = {
  source: {
    kind: "choice",
    label: { en: "Meta-analysis", zh: "元分析" },
    options: [
      { value: "quaife", label: { en: "Quaife et al. 2018 (6 studies)", zh: "Quaife 等人（2018），6 项研究" } },
      { value: "zhang", label: { en: "Zhang et al. 2025 (10 studies)", zh: "Zhang 等人（2025），10 项研究" } },
    ],
    default: "quaife",
    control: "buttons",
  },
  uptake: {
    kind: "range",
    label: { en: "Share who actually take it up (illustrative)", zh: "实际接受者的比例（示意）" },
    min: 0.1, max: 0.9, default: 0.5, step: 0.05,
  },
} as const;

type P = { source: string; uptake: number };

function compute(p: P) {
  const src = SOURCES[p.source as keyof typeof SOURCES] ?? SOURCES.quaife;
  const nt = Math.round(p.uptake * 100);
  const nr = 100 - nt;
  const a = Math.round(src.sens * nt); // takers predicted to take up
  const b = Math.round(src.spec * nr); // refusers predicted to refuse
  const fy = nr - b; // refusers predicted to take up
  const predicted = a + fy;
  return { src, nt, nr, a, b, fy, predicted, ppv: predicted ? a / predicted : 0 };
}

function describe(st: State<P>): string {
  const c = compute(st.p);
  const lang = st.lang ?? "en";
  return tpl(labels[lang].describe, {
    cite: citeOf(c.src, lang), s: pct(c.src.sens), sp: pct(c.src.spec), nt: c.nt,
    p: c.predicted, a: c.a, r: pct(c.ppv),
  });
}

// A block of n people, 10 per row; the first `filled` are filled dots.
function block(x: number, y: number, n: number, filled: number, color: string, gap: number): string {
  const out: string[] = [];
  const r = gap * 0.34;
  for (let i = 0; i < n; i++) {
    const cx = x + (i % 10) * gap + gap / 2;
    const cy = y + Math.floor(i / 10) * gap + gap / 2;
    out.push(i < filled
      ? el("circle", { cx, cy, r, fill: color })
      : el("circle", { cx, cy, r: r - 0.8, fill: C.paper, stroke: color, "stroke-width": 1.6 }));
  }
  return out.join("");
}

function render(st: State<P>): string {
  const lang = st.lang ?? "en";
  const L = labels[lang];
  const narrow = st.w < 480;
  const c = compute(st.p);
  const gap = narrow ? 15 : 16;
  const rows = (n: number) => Math.ceil(n / 10);
  const parts: string[] = [];
  const x0 = narrow ? 8 : 16;
  const x1 = narrow ? 8 : st.w / 2 + 8;
  let y = 18;
  const head = (x: number, yy: number, title: string, sub: string, color: string) => {
    parts.push(el("circle", { cx: x + 5, cy: yy - 4, r: 5, fill: color }));
    parts.push(text(x + 15, yy, title, { "font-size": TYPE.label, class: "fig-t-strong" }));
    parts.push(text(x, yy + 16, sub, { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  };
  const takerTitle = tpl(L.took, { n: c.nt });
  const takerSub = tpl(L.sens, { a: c.a, s: pct(c.src.sens) });
  const refTitle = tpl(L.refused, { n: c.nr });
  const refSub = tpl(L.spec, { b: c.b, s: pct(c.src.spec) });
  if (narrow) {
    head(x0, y, takerTitle, takerSub, C.c1);
    parts.push(block(x0, y + 24, c.nt, c.a, C.c1, gap));
    y += 24 + rows(c.nt) * gap + 22;
    head(x0, y, refTitle, refSub, C.c5);
    // Refusers: the DCE's false "take up" predictions are the filled dots.
    parts.push(block(x0, y + 24, c.nr, c.fy, C.c5, gap));
    y += 24 + rows(c.nr) * gap + 20;
  } else {
    head(x0, y, takerTitle, takerSub, C.c1);
    head(x1, y, refTitle, refSub, C.c5);
    parts.push(block(x0, y + 24, c.nt, c.a, C.c1, gap));
    parts.push(block(x1, y + 24, c.nr, c.fy, C.c5, gap));
    y += 24 + Math.max(rows(c.nt), rows(c.nr)) * gap + 20;
  }
  // Key for filled and hollow dots.
  parts.push(el("circle", { cx: x0 + 5, cy: y - 4, r: 5, fill: C.ink2 }));
  parts.push(text(x0 + 15, y, L.filled, { "font-size": TYPE.small, class: "fig-t-muted" }));
  const hx = narrow ? x0 : x0 + 230;
  const hy = narrow ? y + 16 : y;
  parts.push(el("circle", { cx: hx + 5, cy: hy - 4, r: 4.2, fill: C.paper, stroke: C.ink2, "stroke-width": 1.6 }));
  parts.push(text(hx + 15, hy, L.hollow, { "font-size": TYPE.small, class: "fig-t-muted" }));
  y = hy + 26;
  const ppvLine = tpl(L.ppv, { p: c.predicted, a: c.a, r: pct(c.ppv) });
  if (narrow) {
    const sep = lang === "zh" ? "。" : ". ";
    const cut = ppvLine.indexOf(sep);
    parts.push(text(x0, y, ppvLine.slice(0, cut + 1), { "font-size": TYPE.label, class: "fig-t-strong fig-t-num" }));
    y += 17;
    parts.push(text(x0, y, ppvLine.slice(cut + sep.length), { "font-size": TYPE.label, class: "fig-t-strong fig-t-num" }));
  } else {
    parts.push(text(x0, y, ppvLine, { "font-size": TYPE.label, class: "fig-t-strong fig-t-num" }));
  }
  y += 18;
  const aucLine = tpl(L.auc, { cite: citeOf(c.src, lang), k: c.src.k, auc: fixed(c.src.auc, 2) });
  if (narrow) {
    const sep = lang === "zh" ? "；" : "; ";
    const cut = aucLine.indexOf(sep);
    parts.push(text(x0, y, aucLine.slice(0, cut + 1), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
    y += 15;
    parts.push(text(x0, y, aucLine.slice(cut + sep.length), { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  } else {
    parts.push(text(x0, y, aucLine, { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
  }
  return svg(st.w, y + 10, describe(st), ...parts);
}

export default defineFigure({
  name: "des-validity",
  title: { en: "What a stated choice predicts about real behavior", zh: "陈述的选择对真实行为预测了什么" },
  labels,
  params,
  hint: { en: "Switch the meta-analysis, and move the share of people who actually take the option up.", zh: "切换元分析，并调节实际接受该选项的人所占的比例。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  snapshots: {
    zhang: (p) => ({ ...p, source: "zhang" }),
    low: (p) => ({ ...p, uptake: 0.2 }),
    "zhang low": (p) => ({ ...p, source: "zhang", uptake: 0.2 }),
  },
});
