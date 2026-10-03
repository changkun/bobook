// A one-dimensional function plot frame shared by the GP and BO figures: the
// x domain [0, 1] on the bottom, a value axis on the left, and helpers to draw
// a curve, a credible band, observation dots, and a vertical marker. Keeping
// the frame in one place keeps the margins, colors, and roles identical in
// every chapter, so the reader learns the picture once.

import { bandPath, el, g, labelWidth, linePath, text } from "./svg.ts";
import { C, TYPE } from "./theme.ts";
import { linear, type Scale } from "./scale.ts";
import { axis } from "./axis.ts";

export interface Frame {
  x: Scale;
  y: Scale;
  left: number;
  right: number;
  top: number;
  bottom: number;
  w: number;
}

export interface FrameOpts {
  w: number;
  top: number; // y of the plot's top edge within the SVG
  height: number; // plot height
  yDomain: [number, number];
  xDomain?: [number, number];
  xTitle?: string;
  yTitle?: string;
  xTicks?: boolean;
}

export function frame(o: FrameOpts): Frame {
  const narrow = o.w < 480;
  const left = o.yTitle ? (narrow ? 40 : 52) : 34;
  const right = o.w - 10;
  const x = linear(o.xDomain ?? [0, 1], [left, right]);
  const y = linear(o.yDomain, [o.top + o.height, o.top]);
  return { x, y, left, right, top: o.top, bottom: o.top + o.height, w: o.w };
}

export function frameAxes(f: Frame, o: { xTitle?: string; yTitle?: string; xTicks?: boolean; yCount?: number }): string {
  return g({},
    axis({ scale: f.y, orient: "left", at: f.left, span: [f.left, f.right], title: o.yTitle, count: o.yCount ?? 4 }),
    o.xTicks === false
      ? el("line", { x1: f.left, x2: f.right, y1: f.bottom, y2: f.bottom, stroke: C.rule })
      : axis({ scale: f.x, orient: "bottom", at: f.bottom, span: [f.top, f.bottom], title: o.xTitle, count: f.w < 480 ? 4 : 5 }),
  );
}

// A clip path so curves and bands never spill over the axes.
export function clip(f: Frame, id: string): string {
  return el("clipPath", { id }, el("rect", { x: f.left, y: f.top - 2, width: f.right - f.left, height: f.bottom - f.top + 4 }));
}

export function curve(f: Frame, xs: number[], ys: number[], a: Record<string, string | number> = {}): string {
  return el("path", { d: linePath(xs.map((x, i) => [f.x(x), f.y(ys[i])])), fill: "none", stroke: C.model, "stroke-width": 2, "stroke-linejoin": "round", ...a });
}

export function band(f: Frame, xs: number[], lo: number[], hi: number[], fill: string = C.band): string {
  return el("path", { d: bandPath(xs.map((x, i) => [f.x(x), f.y(hi[i])]), xs.map((x, i) => [f.x(x), f.y(lo[i])])), fill, stroke: "none" });
}

export function dots(f: Frame, pts: Array<[number, number]>, a: { r?: number; fill?: string; hit?: (i: number) => string } = {}): string {
  return g({}, ...pts.map(([x, y], i) => el("circle", {
    cx: f.x(x), cy: f.y(y), r: a.r ?? 4.2, fill: a.fill ?? C.ink, stroke: C.paper, "stroke-width": 1.8,
    "data-fig-hit": a.hit?.(i),
  })));
}

export function vline(f: Frame, x: number, color: string, label?: string, dash = "4 3"): string {
  const px = f.x(x);
  return g({},
    el("line", { x1: px, x2: px, y1: f.top, y2: f.bottom, stroke: color, "stroke-width": 1.5, "stroke-dasharray": dash }),
    label ? text(px, f.top - 5, label, { "text-anchor": "middle", "font-size": TYPE.small, class: "fig-t-muted fig-t-halo" }) : "",
  );
}

// A transparent rectangle over the plot that receives pointer input and
// tells the client how to turn a pixel position into data coordinates.
export function hitArea(f: Frame, name = "plot"): string {
  const map = [f.x.range[0], f.x.range[1], f.x.domain[0], f.x.domain[1], f.y.range[0], f.y.range[1], f.y.domain[0], f.y.domain[1]].join(",");
  return el("rect", { x: f.left, y: f.top, width: f.right - f.left, height: f.bottom - f.top, fill: "transparent", "data-fig-hit": name, "data-fig-map": map });
}

// A row of legend entries: line, band, dot, or dashed swatches with labels.
export type Swatch = { kind: "line" | "dash" | "band" | "dot"; color: string; label: string };

export function legend(x: number, y: number, w: number, items: Swatch[]): { svg: string; height: number } {
  const parts: string[] = [];
  let cx = x, cy = y;
  const rowH = 18;
  for (const it of items) {
    const tw = labelWidth(it.label) + 30;
    if (cx + tw > x + w && cx > x) { cx = x; cy += rowH; }
    const sw = it.kind === "band"
      ? el("rect", { x: cx, y: cy - 6, width: 18, height: 10, fill: it.color, rx: 2 })
      : it.kind === "dot"
        ? el("circle", { cx: cx + 9, cy: cy - 1, r: 4, fill: it.color, stroke: C.paper, "stroke-width": 1.5 })
        : el("line", { x1: cx, x2: cx + 18, y1: cy - 1, y2: cy - 1, stroke: it.color, "stroke-width": 2, "stroke-dasharray": it.kind === "dash" ? "4 3" : undefined });
    parts.push(sw, text(cx + 24, cy + 3, it.label, { "font-size": TYPE.small, class: "fig-t-muted" }));
    cx += tw;
  }
  return { svg: g({}, ...parts), height: cy - y + rowH };
}
