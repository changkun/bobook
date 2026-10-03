// Parameter defaults, parsing, and validation. A chapter block sets
// parameters as `key: value` lines; values are checked against the figure's
// declaration so a typo or an out-of-range value fails the build instead of
// shipping a figure that silently ignores it.

import type { AnyFigure, ParamSpec } from "../types.ts";

export type ParamRecord = Record<string, number | string | boolean>;

export function defaults(fig: AnyFigure): ParamRecord {
  const out: ParamRecord = {};
  for (const [k, spec] of Object.entries(fig.params as Record<string, ParamSpec>)) out[k] = spec.default;
  return out;
}

export function coerce(fig: AnyFigure, key: string, raw: string | number | boolean): number | string | boolean {
  const spec = (fig.params as Record<string, ParamSpec>)[key];
  if (!spec) throw new Error(`figure "${fig.name}" has no parameter "${key}" (known: ${Object.keys(fig.params).join(", ")})`);
  const s = String(raw).trim();
  switch (spec.kind) {
    case "range": {
      const v = Number(s);
      if (!Number.isFinite(v)) throw new Error(`figure "${fig.name}": ${key} must be a number, got "${s}"`);
      if (v < spec.min - 1e-12 || v > spec.max + 1e-12) throw new Error(`figure "${fig.name}": ${key} = ${v} is outside [${spec.min}, ${spec.max}]`);
      return v;
    }
    case "toggle":
      if (s !== "true" && s !== "false") throw new Error(`figure "${fig.name}": ${key} must be true or false, got "${s}"`);
      return s === "true";
    case "choice": {
      const hit = spec.options.find((o) => String(o.value) === s);
      if (!hit) throw new Error(`figure "${fig.name}": ${key} = "${s}" is not one of ${spec.options.map((o) => o.value).join(", ")}`);
      return hit.value;
    }
    case "data": {
      const v = s.replace(/^"(.*)"$/, "$1");
      const err = spec.validate?.(v);
      if (err) throw new Error(`figure "${fig.name}": ${key} = "${v}": ${err}`);
      return v;
    }
  }
}

export function resolve(fig: AnyFigure, entries: Record<string, string>): { p: ParamRecord; t?: number } {
  const p = defaults(fig);
  let t: number | undefined;
  for (const [k, v] of Object.entries(entries)) {
    if (k === "t") {
      if (!fig.timeline) throw new Error(`figure "${fig.name}" has no timeline, so "t" cannot be set`);
      t = Number(v);
      if (!Number.isFinite(t) || t < 0) throw new Error(`figure "${fig.name}": t must be a non-negative number, got "${v}"`);
      continue;
    }
    p[k] = coerce(fig, k, v);
  }
  return { p, t };
}

export function overrides(fig: AnyFigure, p: ParamRecord): ParamRecord {
  const d = defaults(fig);
  const out: ParamRecord = {};
  for (const k of Object.keys(p)) if (p[k] !== d[k]) out[k] = p[k];
  return out;
}

// Serialized lists for data parameters.
//   numbers  "0.1,0.4,0.85"
//   points   "0.1:0.3,0.5:-0.2"        (x:y pairs)
//   duels    "0.2>0.7,0.7>0.9"         (winner>loser, by x value)
export function parseNumbers(s: string): number[] {
  return s.trim() ? s.split(",").map((t) => Number(t.trim())) : [];
}

export function parsePoints(s: string): Array<[number, number]> {
  return s.trim() ? s.split(",").map((t) => t.split(":").map(Number) as [number, number]) : [];
}

export function parseDuels(s: string): Array<[number, number]> {
  return s.trim() ? s.split(",").map((t) => t.split(">").map(Number) as [number, number]) : [];
}

const r4 = (v: number) => String(Math.round(v * 1e4) / 1e4);
export const fmtNumbers = (xs: number[]) => xs.map(r4).join(",");
export const fmtPoints = (ps: Array<[number, number]>) => ps.map(([x, y]) => `${r4(x)}:${r4(y)}`).join(",");
export const fmtDuels = (ds: Array<[number, number]>) => ds.map(([a, b]) => `${r4(a)}>${r4(b)}`).join(",");

export const validNumbers = (s: string) => (parseNumbers(s).every(Number.isFinite) ? undefined : "expected comma-separated numbers");
export const validPoints = (s: string) => (parsePoints(s).every((p) => p.length === 2 && p.every(Number.isFinite)) ? undefined : "expected x:y pairs separated by commas");
export const validDuels = (s: string) => (parseDuels(s).every((p) => p.length === 2 && p.every(Number.isFinite)) ? undefined : "expected winner>loser pairs separated by commas");
