// A generated design with up to ten parameters, drawn as an abstract poster,
// for figures in which the reader judges designs. Every parameter lies in
// [0, 1]; parameters beyond the active dimension take neutral values. The order puts
// the most visible properties first, so a 3-parameter design is still easy
// to judge and a 10-parameter one differs in many small ways.

import { el, g } from "./svg.ts";

export const DESIGN_PARAMS = [
  "background hue", "corner roundness", "shape size", "saturation", "stripes",
  "rotation", "lightness", "accent hue", "shape position", "shape count",
] as const;

// Values for parameters beyond the active dimension: neutral, so that a
// design with three parameters varies in exactly three visible ways.
const NEUTRAL = [0.5, 0.5, 0.5, 0.6, 0, 0, 0.5, 0.5, 0.5, 0];

export function full(x: number[]): number[] {
  return DESIGN_PARAMS.map((_, i) => (i < x.length ? x[i] : NEUTRAL[i]));
}

export function designCard(xIn: number[], x0: number, y0: number, w: number, h: number, uid: string, hit?: string): string {
  const x = full(xIn);
  const hue = Math.round(x[0] * 360);
  const sat = Math.round(25 + 60 * x[3]);
  const light = Math.round(38 + 32 * x[6]);
  const bg = `hsl(${hue}, ${sat}%, ${light}%)`;
  const accent = `hsl(${Math.round(hue + 60 + x[7] * 240) % 360}, 75%, ${light > 55 ? 30 : 82}%)`;
  const stripes = Math.round(x[4] * 7);
  const parts: string[] = [];
  const clip = `${uid}-card-${Math.round(x0)}-${Math.round(y0)}`;
  parts.push(el("clipPath", { id: clip }, el("rect", { x: x0, y: y0, width: w, height: h, rx: 10 })));
  parts.push(el("rect", { x: x0, y: y0, width: w, height: h, rx: 10, fill: bg, stroke: "var(--fig-rule)", "stroke-width": 1 }));
  const sg: string[] = [];
  for (let i = 0; i < stripes; i++) {
    const yy = y0 + ((i + 0.5) / stripes) * h;
    sg.push(el("rect", { x: x0 - w, y: yy - h * 0.03, width: w * 3, height: h * 0.06, fill: "rgba(255,255,255,0.18)", transform: `rotate(-20 ${x0 + w / 2} ${yy})` }));
  }
  parts.push(g({ "clip-path": `url(#${clip})` }, ...sg));
  const count = 1 + Math.round(x[9] * 3);
  const size = Math.min(w, h) * (0.18 + 0.32 * x[2]) / Math.sqrt(count);
  const cx0 = x0 + w * (0.25 + 0.5 * x[8]);
  for (let k = 0; k < count; k++) {
    const off = (k - (count - 1) / 2) * size * 1.25;
    const cx = cx0 + off * 0.6, cy = y0 + h * 0.45 + off * 0.5;
    const rx = (size / 2) * x[1];
    parts.push(el("rect", { x: cx - size / 2, y: cy - size / 2, width: size, height: size, rx, fill: accent, transform: `rotate(${Math.round(x[5] * 45)} ${cx} ${cy})`, opacity: 0.95 }));
  }
  parts.push(el("rect", { x: x0 + w * 0.1, y: y0 + h * 0.8, width: w * 0.45, height: Math.max(4, h * 0.05), rx: 2, fill: "rgba(255,255,255,0.85)" }));
  return g(hit ? { "data-fig-hit": hit, style: "cursor:pointer" } : {}, ...parts);
}
