// Drawing helpers shared by the chemistry case-study figures: the 1,728
// reactions of the direct arylation benchmark as a 48 x 36 map (rows: 12
// ligands x 4 bases; columns: 4 solvents x 3 temperatures x 3
// concentrations), and the best-yield-so-far panel with the 50 recorded
// chemists and the paper's 50 optimizer runs. Data and provenance:
// lib/cs-chem.ts.

import type { Lang } from "../types.ts";
import { el, g, linePath, text } from "./svg.ts";
import { C, TYPE } from "./theme.ts";
import { linear, type Scale } from "./scale.ts";
import { axis } from "./axis.ts";
import { BASES, BO_RANDOM, LIGANDS, PLAYERS, SOLVENTS, YIELD, bestSoFar, bestSoFarYields, parts, solventName } from "./cs-chem.ts";

export const ROWS = 48, COLS = 36;

export function rowCol(i: number): [number, number] {
  const [l, b, s, c, t] = parts(i);
  return [l * 4 + b, s * 9 + t * 3 + c];
}

export function cellAt(row: number, col: number): number {
  const l = Math.floor(row / 4), b = row % 4, s = Math.floor(col / 9), t = Math.floor((col % 9) / 3), c = col % 3;
  return (((l * 4 + b) * 4 + s) * 3 + c) * 3 + t;
}

export interface MapGeom { x0: number; y0: number; cw: number; ch: number; w: number; h: number; labelW: number }

export function mapGeom(left: number, top: number, width: number, narrow: boolean): MapGeom {
  const labelW = narrow ? 90 : 94;
  const cw = Math.max(5, Math.floor(((width - labelW) / COLS) * 10) / 10);
  const ch = narrow ? 4.6 : 5.6;
  return { x0: left + labelW, y0: top + 16, cw, ch, w: cw * COLS, h: ch * ROWS, labelW };
}

const shade = (y: number) => 0.05 + 0.9 * Math.min(1, Math.max(0, y / 100)) ** 1.3;

// The map: cells with a value are filled by yield; others show the panel.
export function yieldMap(m: MapGeom, value: (i: number) => number | undefined): string {
  const parts2: string[] = [el("rect", { x: m.x0, y: m.y0, width: m.w, height: m.h, fill: C.panel })];
  for (let r = 0; r < ROWS; r++) {
    let c = 0;
    while (c < COLS) {
      const v = value(cellAt(r, c));
      if (v === undefined) { c++; continue; }
      const q = Math.round(shade(v) * 30) / 30;
      let k = c + 1;
      while (k < COLS) { const w = value(cellAt(r, k)); if (w === undefined || Math.round(shade(w) * 30) / 30 !== q) break; k++; }
      parts2.push(el("rect", { x: m.x0 + c * m.cw, y: m.y0 + r * m.ch, width: (k - c) * m.cw, height: m.ch, fill: C.truth, opacity: q.toFixed(3) }));
      c = k;
    }
  }
  const lines: string[] = [];
  for (let l = 1; l < 12; l++) lines.push(el("line", { x1: m.x0, x2: m.x0 + m.w, y1: m.y0 + l * 4 * m.ch, y2: m.y0 + l * 4 * m.ch, stroke: C.paper, "stroke-width": 1 }));
  for (let s = 1; s < 4; s++) lines.push(el("line", { x1: m.x0 + s * 9 * m.cw, x2: m.x0 + s * 9 * m.cw, y1: m.y0, y2: m.y0 + m.h, stroke: C.paper, "stroke-width": 1.6 }));
  return g({}, g({ "shape-rendering": "crispEdges" }, ...parts2), ...lines, el("rect", { x: m.x0, y: m.y0, width: m.w, height: m.h, fill: "none", stroke: C.rule, "stroke-width": 0.8 }));
}

export function mapLabels(m: MapGeom, _narrow: boolean, lang: Lang = "en"): string {
  const out: string[] = [];
  LIGANDS.forEach((name, l) => {
    out.push(text(m.x0 - 5, m.y0 + (l * 4 + 2) * m.ch + 4, name, { "text-anchor": "end", "font-size": TYPE.small, class: "fig-t-muted" }));
  });
  SOLVENTS.forEach((_name, s) => {
    out.push(text(m.x0 + (s * 9 + 4.5) * m.cw, m.y0 - 6, solventName(s, lang), { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-strong" }));
  });
  return g({}, ...out);
}

export function cellCenter(m: MapGeom, i: number): [number, number] {
  const [r, c] = rowCol(i);
  return [m.x0 + (c + 0.5) * m.cw, m.y0 + (r + 0.5) * m.ch];
}

// Map from a click (in data coordinates of a hit area spanning the map, with
// x in [0, COLS) and y in [0, ROWS)) to a reaction.
export function hitMap(m: MapGeom, name: string): string {
  const map = [m.x0, m.x0 + m.w, 0, COLS, m.y0, m.y0 + m.h, 0, ROWS].join(",");
  return el("rect", { x: m.x0, y: m.y0, width: m.w, height: m.h, fill: "transparent", "data-fig-hit": name, "data-fig-map": map });
}

// Under the map: how rows and columns are ordered, and the color key.
// Returns the markup and the y below it.
const FOOTER: Record<Lang, { narrow: string[]; wide: string[]; key: string }> = {
  en: {
    narrow: ["Rows in each ligand: KOAc, KOPiv, CsOAc, CsOPiv.", "Columns in each solvent: 90, 105, 120 °C,", "each at 0.057, 0.1, 0.153 M."],
    wide: ["Rows in each ligand: bases KOAc, KOPiv, CsOAc, CsOPiv.", "Columns in each solvent: 90, 105, 120 °C, each at 0.057, 0.1, 0.153 M."],
    key: "measured yield",
  },
  zh: {
    narrow: ["每个配体内的行：KOAc、KOPiv、CsOAc、CsOPiv。", "每个溶剂内的列：90、105、120 °C，", "每个温度下依次为 0.057、0.1、0.153 M。"],
    wide: ["每个配体内的行：碱 KOAc、KOPiv、CsOAc、CsOPiv。", "每个溶剂内的列：90、105、120 °C，每个温度下依次为 0.057、0.1、0.153 M。"],
    key: "实测产率",
  },
};

export function mapFooter(m: MapGeom, narrow: boolean, keyLabel: string, lang: Lang = "en"): { svg: string; y: number } {
  const y = m.y0 + m.h + 14;
  const tx = m.x0 - m.labelW + 4;
  const notes = narrow ? FOOTER[lang].narrow : FOOTER[lang].wide;
  const out = notes.map((s, k) => text(tx, y + k * 13, s, { "font-size": TYPE.small, class: "fig-t-faint" }));
  const kw = narrow ? 90 : 100;
  const ky = y + notes.length * 13 + 2;
  out.push(shadeKey(tx + (narrow ? 0 : 92), ky, kw, narrow ? "" : keyLabel, lang));
  return { svg: g({}, ...out), y: ky + 30 };
}

export function shadeKey(x: number, y: number, w: number, label: string, lang: Lang = "en"): string {
  const out: string[] = [];
  for (let s = 0; s < 20; s++) out.push(el("rect", { x: x + (s * w) / 20, y, width: w / 20 + 0.3, height: 8, fill: C.truth, opacity: shade((s / 19) * 100).toFixed(3) }));
  out.push(el("rect", { x, y, width: w, height: 8, fill: "none", stroke: C.rule, "stroke-width": 0.6 }),
    text(x, y + 19, "0%", { "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }),
    text(x + w, y + 19, "100%", { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted fig-t-num" }));
  if (label) out.push(text(x - 6, y + 8, label, { "font-size": TYPE.small, "text-anchor": "end", class: "fig-t-muted" }));
  else out.push(text(x + w + 8, y + 8, FOOTER[lang].key, { "font-size": TYPE.small, class: "fig-t-muted" }));
  return g({}, ...out);
}

// Recorded reference curves, computed once.
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
export const HUMAN_CURVES = PLAYERS.map((p) => bestSoFar(p.picks));
export const PAPER_BO_MEAN = Array.from({ length: 100 }, (_, k) => mean(BO_RANDOM.map((r) => bestSoFarYields(r)[k])));
// The chemists' mean best yield after n experiments. A player who stopped
// before n keeps the best yield they had found (the paper's "lower bound",
// which it reports is close to the raw average up to batch 11); `count` is
// how many were still playing.
export function humanMean(n: number): { mean: number; count: number } {
  return { mean: mean(HUMAN_CURVES.map((c) => c[Math.min(n, c.length) - 1])), count: HUMAN_CURVES.filter((c) => c.length >= n).length };
}

export interface CurveOpts { x0: number; x1: number; y0: number; h: number; budget: number; narrow: boolean; title: string; xTitle: string }

export function curveFrame(o: CurveOpts): { xs: Scale; ys: Scale; svg: string } {
  const xs = linear([1, o.budget], [o.x0, o.x1]);
  const ys = linear([0, 100], [o.y0 + o.h, o.y0]);
  const svg = g({},
    axis({ scale: xs, orient: "bottom", at: o.y0 + o.h, span: [o.y0, o.y0 + o.h], title: o.xTitle, count: o.narrow ? 4 : 5 }),
    axis({ scale: ys, orient: "left", at: o.x0, span: [o.x0, o.x1], title: o.narrow ? "" : o.title, ticks: [0, 25, 50, 75, 100] }),
    o.narrow ? text(o.x0 + 4, o.y0 - 6, o.title, { "font-size": TYPE.small, class: "fig-t-muted" }) : "",
  );
  return { xs, ys, svg };
}

export function stepLine(xs: Scale, ys: Scale, ys0: number[], n: number, a: Record<string, string | number>): string {
  const pts = ys0.slice(0, n).map((v, i): [number, number] => [xs(i + 1), ys(v)]);
  if (!pts.length) return "";
  return el("path", { d: linePath(pts), fill: "none", "stroke-linejoin": "round", ...a });
}

// The recorded chemists: faint individual curves and the mean of those still
// playing; the paper's optimizer: the mean of its 50 runs.
export function references(xs: Scale, ys: Scale, budget: number, showPlayers: boolean): string {
  const out: string[] = [];
  if (showPlayers) for (const c of HUMAN_CURVES) out.push(stepLine(xs, ys, c, Math.min(budget, c.length), { stroke: C.ink3, "stroke-width": 0.8, opacity: 0.35 }));
  const hm = Array.from({ length: budget }, (_, k) => humanMean(k + 1).mean);
  out.push(stepLine(xs, ys, hm, hm.length, { stroke: C.ink, "stroke-width": 2.2 }));
  out.push(stepLine(xs, ys, PAPER_BO_MEAN, budget, { stroke: C.acq, "stroke-width": 2, "stroke-dasharray": "5 4" }));
  return g({}, ...out);
}

export const BEST_MEASURED = Math.max(...YIELD);
export { BASES };
