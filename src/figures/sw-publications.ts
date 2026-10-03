// How much preferential Bayesian optimization is published each year,
// according to four searches run in September 2026: two over arXiv abstracts
// and two over Semantic Scholar, each in a narrow form (the exact phrase) and
// a broad form (preference words together with "Bayesian optimization"). The
// reader switches between searches on a fixed vertical axis, so the size of
// the base, the growth after 2023, and the disagreement between searches are
// all visible. Years a search did not list are marked, not drawn as zero, and
// 2026 covers January to September only.

import { defineFigure, type State } from "./types.ts";
import { el, g, hatch, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { linear } from "./lib/scale.ts";
import { axis } from "./lib/axis.ts";
import { tpl } from "./lib/format.ts";

const YEARS = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026];

// Yearly counts; null means the search did not list the year.
const DATA: Record<string, { name: string; total: number; counts: Array<number | null> }> = {
  arxivPhrase: { name: "arXiv abstracts with preferential, Bayesian, and optimization or optimisation", total: 46, counts: [1, 0, 0, 6, 2, 2, 4, 5, 13, 12] },
  arxivBroad: { name: "arXiv abstracts with a preference word (preference, pairwise, dueling, ...) and \"Bayesian optimization\"", total: 162, counts: [1, 5, 5, 10, 7, 9, 16, 23, 39, 38] },
  s2Phrase: { name: "Semantic Scholar, exact phrase \"preferential bayesian optimization\"", total: 43, counts: [1, null, null, 2, 2, 3, 3, 9, 13, 10] },
  s2Broad: { name: "Semantic Scholar, broad Boolean search", total: 135, counts: [5, 4, 3, 7, 11, 10, 10, 19, 34, 32] },
};

const labels = {
  en: {
    y: "entries per year",
    partial: "Jan to Sep",
    missing: "not listed",
    ratio: "2025 against 2023: {r} times. First nine months of 2026: {a}, against {b} in all of 2025.",
    describe: "{name}: {list}. 2026 covers January to September.",
    notListed: "not listed",
  },
  zh: {
    y: "每年条目数",
    partial: "1 至 9 月",
    missing: "未列出",
    ratio: "2025 年与 2023 年之比：{r} 倍。2026 年前九个月：{a} 条，2025 年全年：{b} 条。",
    describe: "{name}：{list}。2026 年只含 1 至 9 月。",
    notListed: "未列出",
  },
};

// The search descriptions in Chinese; the query words stay as they were typed.
const NAMES_ZH: Record<string, string> = {
  arxivPhrase: "arXiv 摘要，同时含 preferential、Bayesian，以及 optimization 或 optimisation",
  arxivBroad: "arXiv 摘要，含一个偏好类词语（preference、pairwise、dueling 等）以及“Bayesian optimization”",
  s2Phrase: "Semantic Scholar，精确短语“preferential bayesian optimization”",
  s2Broad: "Semantic Scholar，宽泛的布尔检索",
};

const params = {
  search: {
    kind: "choice", label: { en: "Search", zh: "检索" }, default: "arxivPhrase", control: "select",
    options: [
      { value: "arxivPhrase", label: { en: "arXiv, phrase", zh: "arXiv，短语" } },
      { value: "arxivBroad", label: { en: "arXiv, broad", zh: "arXiv，宽泛" } },
      { value: "s2Phrase", label: { en: "Semantic Scholar, phrase", zh: "Semantic Scholar，短语" } },
      { value: "s2Broad", label: { en: "Semantic Scholar, broad", zh: "Semantic Scholar，宽泛" } },
    ],
  },
} as const;

type P = { search: string };

function describe(st: State<P>): string {
  const d = DATA[st.p.search];
  if (st.lang === "zh") {
    const L = labels.zh;
    return tpl(L.describe, { name: NAMES_ZH[st.p.search], list: YEARS.map((y, i) => (d.counts[i] === null ? `${y} 年${L.notListed}` : `${y} 年 ${d.counts[i]} 条`)).join("，") });
  }
  return tpl(labels.en.describe, { name: d.name, list: YEARS.map((y, i) => `${y}: ${d.counts[i] ?? "not listed"}`).join(", ") });
}

// Break a long line into pieces of at most n characters at spaces.
function wrap(s: string, n: number): string[] {
  const out: string[] = [];
  let cur = "";
  for (const w of s.split(" ")) {
    if (cur && cur.length + 1 + w.length > n) { out.push(cur); cur = w; } else cur = cur ? `${cur} ${w}` : w;
  }
  if (cur) out.push(cur);
  return out;
}

// Chinese has no spaces between words: break into lines of at most n
// Latin-letter widths (a Chinese character counts 1.8), keeping runs of
// Latin letters and digits whole and never starting a line with closing
// punctuation.
function wrapZh(s: string, n: number): string[] {
  const toks = s.match(/[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef“”]|[^\s\u3000-\u303f\u4e00-\u9fff\uff00-\uffef“”]+|\s+/g) ?? [];
  const lines: string[] = [];
  let line = "", w = 0;
  for (const t of toks) {
    const tw = labelWidth(t, 1, 1.8);
    if (line && w + tw > n && !/^[，。；：、）”]$/.test(t) && !/^\s+$/.test(t)) { lines.push(line.trimEnd()); line = ""; w = 0; }
    if (!line && /^\s+$/.test(t)) continue;
    line += t; w += tw;
  }
  if (line.trim()) lines.push(line.trimEnd());
  return lines;
}

function render(st: State<P>): string {
  const p = st.p;
  const zh = st.lang === "zh";
  const L = labels[zh ? "zh" : "en"];
  const wr = zh ? wrapZh : wrap;
  const narrow = st.w < 480;
  const d = DATA[p.search];
  const left = narrow ? 34 : 46, right = st.w - (narrow ? 8 : 14);
  const parts: string[] = [];
  const hid = `${st.uid}-hatch`;
  parts.push(el("defs", {}, hatch(hid, C.c1, 5, 1.4)));
  const titleLines = wr(zh ? NAMES_ZH[p.search] : d.name, narrow ? 52 : 96);
  titleLines.forEach((ln, i) => parts.push(text(narrow ? 8 : left, 16 + i * 15, ln, { "font-size": TYPE.small, class: "fig-t-muted" })));
  const top = 16 + titleLines.length * 15 + 14, h = narrow ? 170 : 210;
  const y = linear([0, 40], [top + h, top]);
  const band = (right - left) / YEARS.length;
  parts.push(axis({ scale: y, orient: "left", at: left, span: [left, right], title: narrow ? undefined : L.y, ticks: [0, 10, 20, 30, 40] }));
  parts.push(el("line", { x1: left, x2: right, y1: top + h, y2: top + h, stroke: C.rule }));
  YEARS.forEach((yr, i) => {
    const cx = left + band * (i + 0.5);
    const bw = Math.min(band * 0.62, 34);
    const v = d.counts[i];
    const label = narrow ? `'${String(yr).slice(2)}` : String(yr);
    parts.push(text(cx, top + h + 16, label, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-num fig-t-muted" }));
    if (v === null) {
      parts.push(text(cx, top + h - 8, L.missing, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted", transform: `rotate(-90 ${cx} ${top + h - 8})`, dx: 34 }));
      return;
    }
    const partial = yr === 2026;
    const y0 = y(0), y1 = y(v);
    if (v > 0) {
      // Rounded top, square base: a path with 3px corners at the data end.
      const rr = Math.min(3, (y0 - y1) / 2, bw / 2);
      const x0 = cx - bw / 2, x1 = cx + bw / 2;
      parts.push(el("path", {
        d: `M${x0},${y0}L${x0},${y1 + rr}Q${x0},${y1} ${x0 + rr},${y1}L${x1 - rr},${y1}Q${x1},${y1} ${x1},${y1 + rr}L${x1},${y0}Z`,
        fill: partial ? `url(#${hid})` : C.c1, stroke: partial ? C.c1 : "none", "stroke-width": partial ? 1.4 : 0,
      }));
    }
    parts.push(text(cx, y1 - 5, String(v), { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-num" }));
    if (partial) parts.push(narrow
      ? text(Math.min(cx + bw / 2 + 2, st.w - 4), top + h + 30, L.partial, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted" })
      : text(cx, top + h + 30, L.partial, { "font-size": TYPE.small, "text-anchor": "middle", class: "fig-t-muted" }));
  });
  const i23 = YEARS.indexOf(2023), i25 = YEARS.indexOf(2025), i26 = YEARS.indexOf(2026);
  const r = (d.counts[i25]! / d.counts[i23]!).toFixed(1);
  const lines = wr(tpl(L.ratio, { r, a: d.counts[i26]!, b: d.counts[i25]! }), narrow ? 52 : 100);
  let ry = top + h + 52;
  for (const ln of lines) { parts.push(text(narrow ? 8 : left, ry, ln, { "font-size": TYPE.small, class: "fig-t-num" })); ry += 15; }
  return svg(st.w, ry, describe(st), g({}, ...parts));
}

export default defineFigure({
  name: "sw-publications",
  title: { en: "Yearly counts of preferential Bayesian optimization papers in four searches", zh: "四次检索中偏好贝叶斯优化论文的年度数量" },
  labels,
  params,
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
});
