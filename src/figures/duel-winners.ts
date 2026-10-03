// Who is the best arm when preferences go in a circle? Five arms A to E and
// their preference matrix P (P[i][j] = probability that i beats j). A slider
// adds a rock-paper-scissors component among A, B, and C; a preset starts from
// a matrix with a narrow champion; and any pair can be flipped by clicking
// its cell. Left: the tournament, an arrow from the winner of each pair to the
// loser, thicker for more lopsided pairs, with three-cycles highlighted.
// Right: the matrix. Below: the four solution concepts of the chapter, with
// the winner of each marked: Condorcet (beats everyone), Copeland (beats the
// most), Borda (highest average win probability), and von Neumann (a mixture
// that no arm beats on average).

import { defineFigure, type State } from "./types.ts";
import { el, g, svg, text } from "./lib/svg.ts";
import { C, TYPE } from "./lib/theme.ts";
import { ARMS, K, PAIRS, concepts, pairKey, scenario, toggleFlip, validFlips, parseFlips, type Mat } from "./lib/duel-matrix.ts";
import { memo } from "./lib/random.ts";
import { tpl } from "./lib/format.ts";

const labels = {
  en: {
    tournament: "who beats whom",
    matrix: "P(row beats column)",
    condorcet: "Condorcet winner",
    none: "none",
    cycleNote: "{c} form a cycle",
    copeland: "Copeland: arms beaten",
    borda: "Borda: average win probability",
    vonNeumann: "von Neumann: weight in the mixture",
    copelandShort: "Copeland",
    bordaShort: "Borda",
    vonNeumannShort: "von Neumann",
    winner: "winner marked",
    describe: "Condorcet winner: {cw}. Copeland winner{cs:/s}: {cp}. Borda winner: {bw}. The von Neumann winner puts weight {vn}.",
    vnItem: "{v} on {a}",
    sep: ", ",
    colon: ": ",
    semi: "; ",
  },
  zh: {
    tournament: "谁胜过谁",
    matrix: "P(行胜过列)",
    condorcet: "Condorcet 赢家",
    none: "无",
    cycleNote: "{c} 构成循环",
    copeland: "Copeland：击败的臂数",
    borda: "Borda：平均获胜概率",
    vonNeumann: "von Neumann：混合中的权重",
    copelandShort: "Copeland",
    bordaShort: "Borda",
    vonNeumannShort: "von Neumann",
    winner: "已标出赢家",
    describe: "Condorcet 赢家：{cw}。Copeland 赢家：{cp}。Borda 赢家：{bw}。von Neumann 赢家的权重：{vn}。",
    vnItem: "{a} 占 {v}",
    sep: "、",
    colon: "：",
    semi: "；",
  },
};

const PAIR_OPTS = PAIRS.map(([i, j]) => ({ value: pairKey(i, j), label: { en: `${ARMS[i]} vs ${ARMS[j]}`, zh: `${ARMS[i]} 对 ${ARMS[j]}` } }));

const params = {
  base: { kind: "choice", label: { en: "Start from", zh: "起点" }, options: [{ value: "utility", label: { en: "A utility", zh: "效用" } }, { value: "narrow", label: { en: "A narrow champion", zh: "险胜的冠军" } }], default: "utility" },
  cycle: { kind: "range", label: { en: "Cycle strength c", zh: "循环强度 c" }, min: 0, max: 3, default: 1.5, step: 0.1 },
  pair: { kind: "choice", label: { en: "Pair", zh: "臂对" }, options: PAIR_OPTS, default: "AB", control: "select" },
  flips: { kind: "data", label: { en: "Flipped pairs", zh: "已翻转的臂对" }, default: "", validate: validFlips },
} as const;

type P = { base: string; cycle: number; pair: string; flips: string };

const model = memo((base: string, cycle: number, flips: string) => {
  const M = scenario(base, cycle, flips);
  const k = concepts(M);
  const cycles: Array<[number, number, number]> = [];
  for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) for (let c = 0; c < K; c++) {
    if (a < b && a < c && b !== c && M[a][b] > 0.5 && M[b][c] > 0.5 && M[c][a] > 0.5) cycles.push([a, b, c]);
  }
  return { M, k, cycles };
}, 32);

const two = (v: number) => (v >= 0.995 ? "1.00" : v.toFixed(2).replace(/^0/, ""));

function describe(st: State<P>): string {
  const L = labels[st.lang ?? "en"];
  const { k } = model(st.p.base, st.p.cycle, st.p.flips);
  const vn = k.vonNeumann.map((v, i) => [v, i] as const).filter(([v]) => v > 0.005).map(([v, i]) => tpl(L.vnItem, { v: v.toFixed(2), a: ARMS[i] })).join(L.sep);
  return tpl(L.describe, {
    cw: k.condorcet >= 0 ? ARMS[k.condorcet] : L.none,
    cs: k.copelandWinners.length, cp: k.copelandWinners.map((i) => ARMS[i]).join(L.sep),
    bw: ARMS[k.bordaWinner], vn,
  });
}

function edgeInCycle(cycles: Array<[number, number, number]>, i: number, j: number): boolean {
  return cycles.some(([a, b, c]) => (a === i && b === j) || (b === i && c === j) || (c === i && a === j));
}

function tournament(cx: number, cy: number, R: number, M: Mat, k: ReturnType<typeof concepts>, cycles: Array<[number, number, number]>, uid: string): string {
  const pos = ARMS.map((_, i) => {
    const a = -Math.PI / 2 + (2 * Math.PI * i) / K;
    return [cx + R * Math.cos(a), cy + R * Math.sin(a)] as const;
  });
  const parts: string[] = [];
  const nodeR = 13;
  for (const [i, j] of PAIRS) {
    const p = M[i][j];
    if (Math.abs(p - 0.5) < 1e-9) continue;
    const [w, l] = p > 0.5 ? [i, j] : [j, i];
    const [x1, y1] = pos[w], [x2, y2] = pos[l];
    const dx = x2 - x1, dy = y2 - y1, d = Math.hypot(dx, dy);
    const ux = dx / d, uy = dy / d;
    const inCycle = edgeInCycle(cycles, w, l);
    const col = inCycle ? C.c5 : C.ink3;
    const width = 1 + 7 * Math.abs(p - 0.5);
    parts.push(el("line", {
      x1: x1 + ux * nodeR, y1: y1 + uy * nodeR, x2: x2 - ux * (nodeR + 3), y2: y2 - uy * (nodeR + 3),
      stroke: col, "stroke-width": width.toFixed(1), "marker-end": `url(#${uid}-${inCycle ? "ac" : "an"})`, opacity: inCycle ? 0.95 : 0.7,
    }));
  }
  ARMS.forEach((name, i) => {
    const [x, y] = pos[i];
    const isC = k.condorcet === i;
    parts.push(
      el("circle", { cx: x, cy: y, r: nodeR, fill: isC ? C.c4 : C.panel, stroke: C.ink, "stroke-width": 1.4 }),
      text(x, y + 4.5, name, { "text-anchor": "middle", "font-size": TYPE.label, class: "fig-t-strong" }),
    );
  });
  return g({}, ...parts);
}

function render(st: State<P>): string {
  const p = st.p;
  const L = labels[st.lang ?? "en"];
  const narrow = st.w < 480;
  const { M, k, cycles } = model(p.base, p.cycle, p.flips);
  const parts: string[] = [];
  const uid = st.uid;
  const flipped = new Set(parseFlips(p.flips));
  // arrowheads
  parts.push(el("defs", {},
    el("marker", { id: `${uid}-an`, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto", markerUnits: "userSpaceOnUse" }, el("path", { d: "M0,0L10,5L0,10z", fill: C.ink3 })),
    el("marker", { id: `${uid}-ac`, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 5, markerHeight: 5, orient: "auto", markerUnits: "userSpaceOnUse" }, el("path", { d: "M0,0L10,5L0,10z", fill: C.c5 })),
  ));
  // 1. tournament
  const pad = 8;
  const gW = narrow ? 138 : 210;
  const top = 22;
  const R = narrow ? 50 : 76;
  const gcx = pad + gW / 2, gcy = top + R + 18;
  parts.push(text(pad, 14, L.tournament, { "font-size": TYPE.small, class: "fig-t-muted" }));
  parts.push(tournament(gcx, gcy, R, M, k, cycles, uid));
  // 2. matrix
  const cell = narrow ? 30 : 38;
  const mx0 = narrow ? pad + gW + 22 : Math.max(pad + gW + 60, st.w - pad - 5 * cell - 20);
  const my0 = top + 20;
  parts.push(text(mx0 - 16, 14, L.matrix, { "font-size": TYPE.small, class: "fig-t-muted" }));
  ARMS.forEach((name, j) => parts.push(text(mx0 + j * cell + cell / 2, my0 - 6, name, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-strong" })));
  ARMS.forEach((name, i) => parts.push(text(mx0 - 8, my0 + i * cell + cell / 2 + 4, name, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-strong" })));
  for (let i = 0; i < K; i++) for (let j = 0; j < K; j++) {
    const x = mx0 + j * cell, y = my0 + i * cell;
    if (i === j) {
      parts.push(el("rect", { x: x + 1, y: y + 1, width: cell - 2, height: cell - 2, rx: 3, fill: C.grid }));
      continue;
    }
    const v = M[i][j];
    const key = pairKey(i, j);
    parts.push(
      el("rect", { x: x + 1, y: y + 1, width: cell - 2, height: cell - 2, rx: 3, fill: v > 0.5 ? C.c6 : C.c5, opacity: (0.08 + 1.5 * Math.abs(v - 0.5)).toFixed(2), "data-fig-hit": `cell-${key}` }),
      text(x + cell / 2, y + cell / 2 + 4, two(v), { "text-anchor": "middle", "font-size": TYPE.small, class: v > 0.5 ? "fig-t-num fig-t-strong" : "fig-t-num fig-t-muted", "pointer-events": "none" }),
    );
    if (flipped.has(key)) parts.push(el("rect", { x: x + 1.5, y: y + 1.5, width: cell - 3, height: cell - 3, rx: 3, fill: "none", stroke: C.ink, "stroke-width": 1.2, "stroke-dasharray": "3 2", "pointer-events": "none" }));
  }
  const gridBottom = Math.max(gcy + R + 20, my0 + 5 * cell + 8);
  // 3. the solution concepts
  let y = gridBottom + 18;
  const cyc = cycles.length ? tpl(L.cycleNote, { c: [...new Set(cycles.flat())].sort().map((i) => ARMS[i]).join(L.sep) }) : "";
  parts.push(text(pad, y, `${L.condorcet}${L.colon}${k.condorcet >= 0 ? ARMS[k.condorcet] : `${L.none}${cyc ? `${L.semi}${cyc}` : ""}`}`, { "font-size": TYPE.label, class: "fig-t-strong" }));
  y += 12;
  const labW = narrow ? 84 : 230;
  const colW = (st.w - pad * 2 - labW) / K;
  const rowH = 24;
  ARMS.forEach((name, i) => parts.push(text(pad + labW + colW * i + colW / 2, y + 14, name, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-strong" })));
  y += 20;
  const rows: Array<{ label: string; vals: string[]; win: (i: number) => boolean }> = [
    { label: narrow ? L.copelandShort : L.copeland, vals: k.copeland.map((c) => (Number.isInteger(c) ? String(c) : c.toFixed(1))), win: (i) => k.copelandWinners.includes(i) },
    { label: narrow ? L.bordaShort : L.borda, vals: k.borda.map(two), win: (i) => i === k.bordaWinner },
    { label: narrow ? L.vonNeumannShort : L.vonNeumann, vals: k.vonNeumann.map(two), win: (i) => k.vonNeumann[i] > 0.005 },
  ];
  for (const row of rows) {
    parts.push(el("line", { x1: pad, x2: st.w - pad, y1: y, y2: y, stroke: C.rule }));
    parts.push(text(pad, y + 16, row.label, { "font-size": TYPE.small, class: "fig-t-muted" }));
    row.vals.forEach((v, i) => {
      const cx = pad + labW + colW * i + colW / 2;
      if (row.win(i)) parts.push(el("rect", { x: cx - 19, y: y + 4, width: 38, height: 17, rx: 8, fill: C.c4, opacity: 0.45 }));
      parts.push(text(cx, y + 16, v, { "text-anchor": "middle", "font-size": TYPE.small, class: row.win(i) ? "fig-t-num fig-t-strong" : "fig-t-num fig-t-muted" }));
    });
    y += rowH;
  }
  parts.push(el("line", { x1: pad, x2: st.w - pad, y1: y, y2: y, stroke: C.rule }));
  return svg(st.w, y + 6, describe(st), ...parts);
}

export default defineFigure({
  name: "duel-winners",
  title: { en: "Four ways to name the best arm when preferences go in a circle", zh: "偏好成环时认定最优臂的四种方式" },
  labels,
  params,
  hint: { en: "Click a cell of the matrix, or pick a pair and press Flip, to reverse who wins that pair. Highlighted values are each concept's winner.", zh: "点击矩阵中的一格，或选好一对臂后按“翻转这一对”，即可颠倒这一对的胜负。高亮的数值标出各个概念的赢家。" },
  render: (st) => render(st as State<P>),
  describe: (st) => describe(st as State<P>),
  pointer(p, e) {
    if (e.phase !== "click" || !e.target?.startsWith("cell-")) return null;
    const key = e.target.slice(5);
    return { ...p, pair: key, flips: toggleFlip(p.flips, key) };
  },
  actions: [
    { label: { en: "Flip the pair", zh: "翻转这一对" }, primary: true, run: (p) => ({ ...p, flips: toggleFlip(p.flips, p.pair) }) },
    { label: { en: "Unflip all", zh: "全部还原" }, run: (p) => ({ ...p, flips: "" }), enabled: (p) => p.flips !== "" },
  ],
  update(p, key) {
    if (key === "base") return { ...p, flips: "", cycle: p.base === "narrow" ? 0 : 1.5 };
    return p;
  },
});
