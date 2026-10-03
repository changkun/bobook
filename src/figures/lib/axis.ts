// Axes with gridlines and a titled quantity. Tick labels are written in data
// units and thinned so they never collide at phone width.

import { el, g, text } from "./svg.ts";
import { C, TYPE } from "./theme.ts";
import type { Scale } from "./scale.ts";
import { tick as fmtTick } from "./format.ts";

export interface AxisOpts {
  scale: Scale;
  orient: "bottom" | "left";
  at: number; // y of a bottom axis, x of a left axis
  span?: [number, number]; // gridline extent across the plot (y range for bottom, x range for left)
  title?: string;
  count?: number;
  format?(v: number): string;
  grid?: boolean;
  ticks?: number[];
}

export function axis(o: AxisOpts): string {
  const ticks = o.ticks ?? o.scale.ticks(o.count ?? 5);
  const step = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1;
  const fmt = o.format ?? ((v: number) => (o.scale.kind === "log" ? String(v) : fmtTick(v, step)));
  const parts: string[] = [];
  const [r0, r1] = o.scale.range;
  if (o.orient === "bottom") {
    parts.push(el("line", { x1: Math.min(r0, r1), x2: Math.max(r0, r1), y1: o.at, y2: o.at, stroke: C.rule, "stroke-width": 1 }));
    // Thin labels so neighbours are at least ~36 px apart.
    const px = ticks.map((v) => o.scale(v));
    const minGap = 36;
    let last = -Infinity;
    ticks.forEach((v, i) => {
      const x = px[i];
      if (o.grid !== false && o.span) parts.push(el("line", { x1: x, x2: x, y1: o.span[0], y2: o.span[1], stroke: C.grid, "stroke-width": 1 }));
      parts.push(el("line", { x1: x, x2: x, y1: o.at, y2: o.at + 4, stroke: C.rule }));
      if (x - last >= minGap) {
        parts.push(text(x, o.at + 16, fmt(v), { "text-anchor": "middle", class: "fig-t-num fig-t-muted", "font-size": TYPE.small }));
        last = x;
      }
    });
    if (o.title) parts.push(text((r0 + r1) / 2, o.at + 32, o.title, { "text-anchor": "middle", class: "fig-t-muted", "font-size": TYPE.body }));
  } else {
    parts.push(el("line", { y1: Math.min(r0, r1), y2: Math.max(r0, r1), x1: o.at, x2: o.at, stroke: C.rule, "stroke-width": 1 }));
    let last = Infinity;
    for (const v of ticks) {
      const y = o.scale(v);
      if (o.grid !== false && o.span) parts.push(el("line", { y1: y, y2: y, x1: o.span[0], x2: o.span[1], stroke: C.grid, "stroke-width": 1 }));
      parts.push(el("line", { y1: y, y2: y, x1: o.at - 4, x2: o.at, stroke: C.rule }));
      if (Math.abs(last - y) >= 16) {
        parts.push(text(o.at - 7, y + 4, fmt(v), { "text-anchor": "end", class: "fig-t-num fig-t-muted", "font-size": TYPE.small }));
        last = y;
      }
    }
    if (o.title) {
      const cy = (r0 + r1) / 2;
      parts.push(text(0, 0, o.title, { "text-anchor": "middle", class: "fig-t-muted", "font-size": TYPE.body, transform: `translate(${o.at - 34},${cy}) rotate(-90)` }));
    }
  }
  return g({ class: "fig-axis" }, ...parts);
}
