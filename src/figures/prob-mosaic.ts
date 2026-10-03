// A joint distribution of two yes/no quantities drawn as areas. A commit is
// broken or fine, and a test run on it fails or passes. The unit square is
// split into columns whose widths are P(commit) and each column is split by
// P(test | commit), so every region's area is the joint probability
// P(commit, test): the product rule, drawn. The table beside it holds the same
// numbers with their row and column sums, the marginals. Conditioning keeps
// one row or column and rescales it to sum to one; choosing "fail" gives
// Bayes' rule for a failing test.

import { defineFigure, type State } from "./types.ts";
import { el, g, labelWidth, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { tpl } from "./lib/format.ts";

const labels = {
  en: {
    broken: "broken",
    fine: "fine",
    fail: "fail",
    pass: "pass",
    total: "total",
    key1: "width = P(commit)",
    key2: "height = P(test | commit)",
    key3: "area = P(commit, test)",
    table: "joint probabilities and their sums",
    none: "Before any test is run:",
    givenFail: "Keep only the runs where the test failed:",
    givenPass: "Keep only the runs where the test passed:",
    givenBroken: "Keep only the broken commits:",
    givenFine: "Keep only the fine commits:",
    describe: "Commits are broken with probability {b}; the test fails on a broken commit with probability {s} and on a fine one with probability {f}. {r}",
    describeNone: "The test fails on {pf} of all runs.",
    describeGiven: "Conditioned on {g}, {q} = {v}.",
    gFail: "a failing test",
    gPass: "a passing test",
    gBroken: "a broken commit",
    gFine: "a fine commit",
  },
  zh: {
    broken: "有缺陷",
    fine: "正常",
    fail: "失败",
    pass: "通过",
    total: "合计",
    key1: "宽 = P(提交)",
    key2: "高 = P(测试 | 提交)",
    key3: "面积 = P(提交, 测试)",
    table: "联合概率及其行列之和",
    none: "运行任何测试之前：",
    givenFail: "只保留测试失败的运行：",
    givenPass: "只保留测试通过的运行：",
    givenBroken: "只保留有缺陷的提交：",
    givenFine: "只保留正常的提交：",
    describe: "提交有缺陷的概率为 {b}；测试在有缺陷的提交上失败的概率为 {s}，在正常的提交上失败的概率为 {f}。{r}",
    describeNone: "在全部运行中，测试失败的比例为 {pf}。",
    describeGiven: "以{g}为条件，{q} = {v}。",
    gFail: "测试失败",
    gPass: "测试通过",
    gBroken: "提交有缺陷",
    gFine: "提交正常",
  },
};

type Labels = typeof labels.en;

const GIVEN = [
  { value: "none", label: { en: "nothing", zh: "无" } },
  { value: "fail", label: { en: "fail", zh: "失败" } },
  { value: "pass", label: { en: "pass", zh: "通过" } },
  { value: "broken", label: { en: "broken", zh: "有缺陷" } },
  { value: "fine", label: { en: "fine", zh: "正常" } },
] as const;

const params = {
  base: { kind: "range", label: { en: "P(broken)", zh: "P(有缺陷)" }, min: 0.005, max: 0.5, default: 0.1, scale: "log" },
  sens: { kind: "range", label: { en: "P(fail | broken)", zh: "P(失败 | 有缺陷)" }, min: 0.5, max: 1, default: 0.9, step: 0.01 },
  fpr: { kind: "range", label: { en: "P(fail | fine)", zh: "P(失败 | 正常)" }, min: 0, max: 0.5, default: 0.05, step: 0.01 },
  given: { kind: "choice", label: { en: "Condition on", zh: "条件" }, options: GIVEN, default: "none", control: "buttons" },
} as const;

type P = { base: number; sens: number; fpr: number; given: string };

const COL = { broken: C.c8, fine: C.c7 };

function joint(p: P) {
  const bf = p.base * p.sens, bp = p.base * (1 - p.sens);
  const ff = (1 - p.base) * p.fpr, fp = (1 - p.base) * (1 - p.fpr);
  return { bf, bp, ff, fp, b: p.base, f: 1 - p.base, fail: bf + ff, pass: bp + fp };
}

// Probabilities keep three decimals, or four when some cell of the table is
// below 0.01, so that the numbers in one table line up and add up.
function num(v: number, d = 3): string {
  if (Math.abs(v) < 1e-12) return "0";
  if (Math.abs(v - 1) < 1e-12) return "1";
  return v.toFixed(v < 0.00995 ? Math.max(d, 4) : d);
}

function places(p: P): number {
  const j = joint(p);
  const small = [j.bf, j.bp, j.ff, j.fp].filter((v) => v > 1e-12);
  return Math.min(...small) < 0.00995 ? 4 : 3;
}

function pct(v: number): string {
  const x = Math.round(v * 1000) / 10;
  return `${Number.isInteger(x) ? x.toFixed(0) : x.toFixed(1)}%`;
}

// The conditional the current choice asks for: which cells survive, the
// numerator and denominator, and the two-part split it produces.
function conditional(p: P, L: Labels = labels.en) {
  const j = joint(p);
  switch (p.given) {
    case "fail": return { title: L.givenFail, q: `P(${L.broken} | ${L.fail})`, num: j.bf, den: j.fail, split: [j.bf / j.fail, j.ff / j.fail], names: [L.broken, L.fine], hue: [COL.broken, COL.fine], solid: [true, true] };
    case "pass": return { title: L.givenPass, q: `P(${L.broken} | ${L.pass})`, num: j.bp, den: j.pass, split: [j.bp / j.pass, j.fp / j.pass], names: [L.broken, L.fine], hue: [COL.broken, COL.fine], solid: [false, false] };
    case "broken": return { title: L.givenBroken, q: `P(${L.fail} | ${L.broken})`, num: j.bf, den: j.b, split: [p.sens, 1 - p.sens], names: [L.fail, L.pass], hue: [COL.broken, COL.broken], solid: [true, false] };
    case "fine": return { title: L.givenFine, q: `P(${L.fail} | ${L.fine})`, num: j.ff, den: j.f, split: [p.fpr, 1 - p.fpr], names: [L.fail, L.pass], hue: [COL.fine, COL.fine], solid: [true, false] };
    default: return { title: L.none, q: `P(${L.broken})`, num: j.b, den: 1, split: [j.b, j.f], names: [L.broken, L.fine], hue: [COL.broken, COL.fine], solid: [true, true] };
  }
}

function describe(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const j = joint(p);
  const c = conditional(p, L);
  const r = p.given === "none"
    ? tpl(L.describeNone, { pf: num(j.fail) })
    : tpl(L.describeGiven, { g: { fail: L.gFail, pass: L.gPass, broken: L.gBroken, fine: L.gFine }[p.given as "fail"] ?? p.given, q: c.q, v: num(c.split[0]) });
  return tpl(L.describe, { b: num(p.base), s: num(p.sens), f: num(p.fpr), r });
}

// Push apart labels that would overlap, keeping their mean position.
function spread(ys: number[], gap: number): number[] {
  if (ys.length !== 2 || Math.abs(ys[1] - ys[0]) >= gap) return ys;
  const m = (ys[0] + ys[1]) / 2;
  return [m - gap / 2, m + gap / 2];
}

const FILL_SOLID = 0.88, FILL_LIGHT = 0.3, FADED = 0.08;

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const j = joint(p);
  const c = conditional(p, L);
  const d = places(p);
  const parts: string[] = [];

  // 1. The mosaic.
  const side = narrow ? 60 : 64;
  const mx0 = 10 + side;
  const mw = narrow ? st.w - 20 - 2 * side : 220;
  const mx1 = mx0 + mw;
  const my0 = 44, mh = narrow ? 150 : 200;
  const wb = mw * p.base;
  const keepTest = p.given === "fail" || p.given === "pass" ? p.given : undefined;
  const keepState = p.given === "broken" || p.given === "fine" ? p.given : undefined;
  const cells = [
    { state: "broken", test: "fail", x: mx0, w: wb, y: my0, h: mh * p.sens },
    { state: "broken", test: "pass", x: mx0, w: wb, y: my0 + mh * p.sens, h: mh * (1 - p.sens) },
    { state: "fine", test: "fail", x: mx0 + wb, w: mw - wb, y: my0, h: mh * p.fpr },
    { state: "fine", test: "pass", x: mx0 + wb, w: mw - wb, y: my0 + mh * p.fpr, h: mh * (1 - p.fpr) },
  ];
  for (const cl of cells) {
    const kept = (!keepTest || cl.test === keepTest) && (!keepState || cl.state === keepState);
    const base = cl.test === "fail" ? FILL_SOLID : FILL_LIGHT;
    parts.push(el("rect", {
      x: cl.x, y: cl.y, width: Math.max(0, cl.w), height: Math.max(0, cl.h),
      fill: COL[cl.state as "broken" | "fine"], opacity: kept ? base : base * FADED,
      "data-fig-hit": cl.test,
    }));
  }
  // cell borders and the frame
  parts.push(
    el("line", { x1: mx0 + wb, x2: mx0 + wb, y1: my0, y2: my0 + mh, stroke: C.paper, "stroke-width": 1.5 }),
    el("line", { x1: mx0, x2: mx0 + wb, y1: my0 + mh * p.sens, y2: my0 + mh * p.sens, stroke: C.paper, "stroke-width": 1.5 }),
    el("line", { x1: mx0 + wb, x2: mx1, y1: my0 + mh * p.fpr, y2: my0 + mh * p.fpr, stroke: C.paper, "stroke-width": 1.5 }),
    el("rect", { x: mx0, y: my0, width: mw, height: mh, fill: "none", stroke: C.rule }),
  );
  // outline what conditioning keeps
  if (keepTest || keepState) {
    for (const cl of cells) {
      const kept = (!keepTest || cl.test === keepTest) && (!keepState || cl.state === keepState);
      if (kept && cl.w > 0 && cl.h > 0) parts.push(el("rect", { x: cl.x, y: cl.y, width: cl.w, height: cl.h, fill: "none", stroke: C.ink, "stroke-width": 1.6 }));
    }
  }
  // column labels: the marginal of the commit's state
  const colLab = [
    { name: "broken", cx: mx0 + wb / 2, v: p.base },
    { name: "fine", cx: mx0 + wb + (mw - wb) / 2, v: 1 - p.base },
  ];
  const lw = (s: string) => labelWidth(s, 6.3, 11);
  const labelsX = colLab.map((cl) => cl.cx);
  const s0 = `${L.broken} ${pct(colLab[0].v)}`, s1 = `${L.fine} ${pct(colLab[1].v)}`;
  // keep the broken label from colliding with the fine one or the left edge
  labelsX[0] = Math.max(10 + lw(s0) / 2, Math.min(labelsX[0], labelsX[1] - lw(s1) / 2 - lw(s0) / 2 - 10));
  colLab.forEach((cl, i) => {
    const s = i ? s1 : s0;
    const faded = keepState && keepState !== cl.name;
    parts.push(g({ "data-fig-hit": cl.name },
      el("rect", { x: labelsX[i] - lw(s) / 2 - 4, y: my0 - 30, width: lw(s) + 8, height: 20, fill: "transparent" }),
      text(labelsX[i], my0 - 16, s, { "text-anchor": "middle", "font-size": TYPE.small, class: faded ? "fig-t-faint" : "fig-t-strong" }),
    ));
    parts.push(el("line", { x1: labelsX[i], x2: cl.cx, y1: my0 - 11, y2: my0 - 2, stroke: C.ink3, "stroke-width": 1 }));
  });
  // side labels: P(test | commit) for each column, broken on the left, fine on the right
  const sideLab = (state: "broken" | "fine", x: number, anchor: string) => {
    const col = cells.filter((cl) => cl.state === state);
    const ys = spread(col.map((cl) => cl.y + cl.h / 2 + 4), 13);
    const vals = state === "broken" ? [p.sens, 1 - p.sens] : [p.fpr, 1 - p.fpr];
    col.forEach((cl, i) => {
      const kept = (!keepTest || cl.test === keepTest) && (!keepState || cl.state === keepState);
      parts.push(text(x, Math.min(my0 + mh + 2, Math.max(my0 + 8, ys[i])), `${L[cl.test as "fail" | "pass"]} ${pct(vals[i])}`, { "text-anchor": anchor, "font-size": TYPE.small, class: kept ? "fig-t-muted fig-t-num" : "fig-t-faint fig-t-num" }));
    });
  };
  sideLab("broken", mx0 - 6, "end");
  sideLab("fine", mx1 + 6, "start");
  // key
  let ky = my0 + mh + 22;
  if (narrow) {
    parts.push(text(mx0 + mw / 2, ky, `${L.key1} · ${L.key2}`, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }));
    ky += 15;
    parts.push(text(mx0 + mw / 2, ky, L.key3, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }));
  } else {
    parts.push(text(mx0 + mw / 2, ky, `${L.key1} · ${L.key2}`, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }));
    ky += 15;
    parts.push(text(mx0 + mw / 2, ky, L.key3, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted" }));
  }

  // 2. The table of joint probabilities with margins.
  const tx0 = narrow ? 10 : mx1 + side + 20;
  const tx1 = st.w - 10;
  const ty0 = narrow ? ky + 34 : my0 - 16;
  const rowH = 24;
  const labW = 64;
  const colW = (tx1 - tx0 - labW) / 3;
  const colX = (k: number) => tx0 + labW + colW * (k + 1) - 6; // right edge of numeric column k
  const heads = [L.fail, L.pass, L.total];
  const keyOf = ["fail", "pass", "total"];
  // highlights
  const hl = (x0: number, y0: number, w: number, h: number) => el("rect", { x: x0, y: y0, width: w, height: h, rx: 3, fill: C.panel });
  if (p.given === "fail" || p.given === "pass" || p.given === "none") {
    const k = p.given === "fail" ? 0 : p.given === "pass" ? 1 : 2;
    parts.push(hl(tx0 + labW + colW * k + 4, ty0 + 6, colW - 4, 22 + rowH * 3));
  } else {
    const i = p.given === "broken" ? 0 : 1;
    parts.push(hl(tx0 - 4, ty0 + 29 + rowH * i, tx1 - tx0 + 4, rowH - 2));
  }
  parts.push(text(tx0, ty0 - 2, L.table, { "font-size": TYPE.small, class: "fig-t-muted" }));
  heads.forEach((h, k) => {
    const hit = k < 2 ? keyOf[k] : undefined;
    parts.push(g(hit ? { "data-fig-hit": hit } : {},
      hit ? el("rect", { x: colX(k) - colW + 8, y: ty0 + 6, width: colW - 4, height: rowH - 2, fill: "transparent" }) : "",
      text(colX(k), ty0 + 22, h, { "text-anchor": "end", "font-size": TYPE.small, class: "fig-t-strong" }),
    ));
  });
  parts.push(el("line", { x1: tx0, x2: tx1, y1: ty0 + 28, y2: ty0 + 28, stroke: C.rule }));
  const rows = [
    { name: "broken", label: L.broken, vals: [j.bf, j.bp, j.b] },
    { name: "fine", label: L.fine, vals: [j.ff, j.fp, j.f] },
    { name: "total", label: L.total, vals: [j.fail, j.pass, 1] },
  ];
  rows.forEach((r, i) => {
    const y = ty0 + 28 + rowH * (i + 1) - 7;
    const isState = r.name !== "total";
    parts.push(g(isState ? { "data-fig-hit": r.name } : {},
      isState ? el("rect", { x: tx0, y: y - 14, width: labW, height: rowH - 2, fill: "transparent" }) : "",
      isState ? el("rect", { x: tx0, y: y - 9, width: 10, height: 10, rx: 2, fill: COL[r.name as "broken" | "fine"] }) : "",
      text(tx0 + (isState ? 16 : 0), y, r.label, { "font-size": TYPE.small, class: isState ? "fig-t-strong" : "fig-t-muted" }),
    ));
    r.vals.forEach((v, k) => {
      const inSel = p.given === "none" ? k === 2
        : p.given === "fail" ? k === 0 : p.given === "pass" ? k === 1
        : p.given === r.name;
      parts.push(text(colX(k), y, num(v, d), { "text-anchor": "end", "font-size": TYPE.body, class: inSel ? "fig-t-num fig-t-strong" : "fig-t-num fig-t-muted" }));
    });
    if (i === 1) parts.push(el("line", { x1: tx0, x2: tx1, y1: y + 7, y2: y + 7, stroke: C.rule }));
  });
  const tBottom = ty0 + 28 + rowH * 3 + 4;

  // 3. The conditional: keep the selected cells and rescale them to sum to one.
  let ry = tBottom + 26;
  parts.push(text(tx0, ry, c.title, { "font-size": TYPE.small, class: "fig-t-muted" }));
  ry += 20;
  const formula = p.given === "none" ? `${c.q} = ${num(c.num, d)}` : `${c.q} = ${num(c.num, d)} / ${num(c.den, d)} = ${num(c.split[0])}`;
  parts.push(text(tx0, ry, formula, { "font-size": TYPE.label, class: "fig-t-strong fig-t-num" }));
  ry += 14;
  const bw = tx1 - tx0, bh = 18;
  const w0 = bw * c.split[0];
  parts.push(
    el("rect", { x: tx0, y: ry, width: Math.max(0, w0), height: bh, fill: c.hue[0], opacity: c.solid[0] ? FILL_SOLID : FILL_LIGHT }),
    el("rect", { x: tx0 + w0, y: ry, width: Math.max(0, bw - w0), height: bh, fill: c.hue[1], opacity: c.solid[1] ? FILL_SOLID : FILL_LIGHT }),
    el("rect", { x: tx0, y: ry, width: bw, height: bh, fill: "none", stroke: C.rule }),
  );
  ry += bh + 15;
  parts.push(
    text(tx0, ry, `${c.names[0]} ${pct(c.split[0])}`, { "font-size": TYPE.small, class: "fig-t-strong fig-t-num" }),
    text(tx1, ry, `${c.names[1]} ${pct(c.split[1])}`, { "text-anchor": "end", "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }),
  );
  const H = Math.max(ry, ky) + 12;
  return svg(st.w, H, describe(st), ...parts);
}

export default defineFigure({
  name: "prob-mosaic",
  title: { en: "A joint distribution as areas: commits, tests, and conditioning", zh: "用面积表示联合分布：提交、测试与条件化" },
  labels,
  params,
  hint: { en: "Click a fail or pass region (or a column of the table) to condition on the test result; click broken or fine to condition on the commit. Click again to undo.", zh: "点击失败或通过的区域（或表格的一列），以测试结果为条件；点击有缺陷或正常，以提交的状态为条件。再点一次即可撤销。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click" || !e.target) return null;
    if (!["fail", "pass", "broken", "fine"].includes(e.target)) return null;
    return { ...p, given: p.given === e.target ? "none" : e.target };
  },
});
