// Build-time rendering of a figure's opening state. The static markup is a
// real figure, not a placeholder: it is what a reader sees without script, in
// print, and in search indexes. Two layouts are emitted, one for the desktop
// column and one for a phone column; a container query in book.css shows the
// one that fits. The client replaces both with a single live render at the
// measured width.

import type { AnyFigure, Lang } from "./types.ts";
import { esc } from "./lib/svg.ts";
import { overrides, type ParamRecord } from "./lib/params.ts";

export const STATIC_WIDTHS = { wide: 660, narrow: 330 } as const;

export function openingTime(fig: AnyFigure, p: ParamRecord, t?: number): number {
  if (!fig.timeline) return 0;
  const d = fig.timeline.duration(p);
  return Math.min(t ?? fig.timeline.poster(p), d);
}

export function renderStatic(fig: AnyFigure, p: ParamRecord, lang: Lang, id: string, t?: number): string {
  const t0 = openingTime(fig, p, t);
  const wide = fig.render({ p, t: t0, w: STATIC_WIDTHS.wide, uid: `${id}-w`, lang }, lang);
  const narrow = fig.render({ p, t: t0, w: STATIC_WIDTHS.narrow, uid: `${id}-n`, lang }, lang);
  const params = JSON.stringify(overrides(fig, p));
  const tAttr = t != null ? ` data-t="${t}"` : "";
  return `<div class="fig" data-figure="${esc(fig.name)}" data-lang="${lang}" data-params="${esc(params)}"${tAttr}>`
    + `<div class="fig-static"><div class="fig-v fig-v-wide">${wide}</div><div class="fig-v fig-v-narrow">${narrow}</div></div>`
    + `</div>`;
}
