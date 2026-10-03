// Helpers for figures in more than two dimensions: space-filling and random
// points in the unit cube, candidate sets for maximizing an acquisition
// function, an orthographic 3-D projection, and parallel coordinates.

import { el, g, linePath, text } from "./svg.ts";
import { C, TYPE } from "./theme.ts";
import { normal, type Rng } from "./random.ts";

export type Pt = number[];

export function uniformPoints(r: Rng, n: number, d: number): Pt[] {
  return Array.from({ length: n }, () => Array.from({ length: d }, () => r()));
}

// Latin hypercube: each coordinate's values fall in distinct strata.
export function latinHypercube(r: Rng, n: number, d: number): Pt[] {
  const cols = Array.from({ length: d }, () => {
    const perm = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [perm[i], perm[j]] = [perm[j], perm[i]]; }
    return perm.map((k) => (k + r()) / n);
  });
  return Array.from({ length: n }, (_, i) => cols.map((c) => c[i]));
}

// Candidates for maximizing an acquisition function in many dimensions: a
// uniform sample of the whole cube plus Gaussian perturbations of the best
// points so far, the strategy most practical implementations use in some form.
// In many dimensions, perturbing every coordinate at once moves a point far
// from its parent; `maskProb` perturbs each coordinate only with that
// probability (at least one), as trust-region methods such as TuRBO do.
export function candidates(r: Rng, d: number, nGlobal: number, around: Pt[], nLocal: number, sd = 0.08, maskProb = 1): Pt[] {
  const z = normal(r);
  const out = uniformPoints(r, nGlobal, d);
  for (let i = 0; i < nLocal && around.length; i++) {
    const c = around[i % around.length];
    const forced = Math.floor(r() * d);
    out.push(c.map((v, j) => (j === forced || r() < maskProb ? Math.min(1, Math.max(0, v + sd * z())) : v)));
  }
  return out;
}

export function dist(a: Pt, b: Pt): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s);
}

// Orthographic projection of the unit cube, centered, rotated by yaw (about
// the vertical axis) and pitch (about the horizontal axis). Returns screen
// coordinates and a depth for painter's ordering.
export function project(p: Pt, yaw: number, pitch: number, cx: number, cy: number, s: number): { x: number; y: number; z: number } {
  const x = p[0] - 0.5, y = p[1] - 0.5, z = p[2] - 0.5;
  const cyw = Math.cos(yaw), syw = Math.sin(yaw);
  const x1 = cyw * x + syw * y, y1 = -syw * x + cyw * y;
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const z2 = cp * z - sp * y1, y2 = sp * z + cp * y1;
  return { x: cx + s * x1, y: cy - s * z2, z: y2 };
}

export function cubeEdges(yaw: number, pitch: number, cx: number, cy: number, s: number, labels: string[]): string {
  const corners: Pt[] = [];
  for (let i = 0; i < 8; i++) corners.push([i & 1, (i >> 1) & 1, (i >> 2) & 1]);
  const parts: string[] = [];
  for (let i = 0; i < 8; i++) for (let j = i + 1; j < 8; j++) {
    const diff = i ^ j;
    if (diff !== 1 && diff !== 2 && diff !== 4) continue;
    const a = project(corners[i], yaw, pitch, cx, cy, s), b = project(corners[j], yaw, pitch, cx, cy, s);
    const axisEdge = i === 0;
    parts.push(el("line", { x1: a.x, y1: a.y, x2: b.x, y2: b.y, stroke: axisEdge ? C.ink3 : C.rule, "stroke-width": axisEdge ? 1.4 : 1 }));
  }
  // axis labels at the far end of the three edges from the origin corner
  [[1, 0, 0], [0, 1, 0], [0, 0, 1]].forEach((e, k) => {
    const q = project(e.map((v) => v * 1.12) as Pt, yaw, pitch, cx, cy, s);
    parts.push(text(q.x, q.y + 4, labels[k], { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }));
  });
  return g({}, ...parts);
}

// Parallel coordinates: one vertical axis per dimension, one polyline per
// point. `color(i)` and `width(i)` style each line; axes listed in `axes`.
export function parallel(points: Pt[], axes: number[], x0: number, x1: number, y0: number, y1: number,
  style: (i: number) => { stroke: string; width: number; opacity: number }, axisLabel: (d: number) => string): string {
  const parts: string[] = [];
  const step = axes.length > 1 ? (x1 - x0) / (axes.length - 1) : 0;
  axes.forEach((d, k) => {
    const x = x0 + k * step;
    parts.push(el("line", { x1: x, x2: x, y1: y0, y2: y1, stroke: C.rule }));
    if (axes.length <= 24 || k % Math.ceil(axes.length / 24) === 0) {
      parts.push(text(x, y1 + 14, axisLabel(d), { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-num" }));
    }
  });
  points.forEach((p, i) => {
    const s = style(i);
    parts.push(el("path", { d: linePath(axes.map((d, k) => [x0 + k * step, y1 - (y1 - y0) * p[d]])), fill: "none", stroke: s.stroke, "stroke-width": s.width, opacity: s.opacity }));
  });
  return g({}, ...parts);
}
