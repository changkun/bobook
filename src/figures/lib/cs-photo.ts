// Shared code for the photo-enhancement case study: the six adjustments and
// the SVG filter chain that applies them to a real photograph, the
// preference model (lib/gp.ts fitPreference, with EUBO from lib/acq.ts over a
// random candidate pool), the two query forms (a pair, and a slider along a
// line from the best design to the design with the highest expected
// improvement), and a simulated taste with a hidden favorite.
//
// Every design is a point u in [0, 1]^6; u = 0.5 in every coordinate is the
// original photograph. The figures and the offline script that precomputes
// simulated sessions (tools/figure-data/cs-photo-race.ts) import this file, so
// the figure and the precomputed runs use the same model.

import type { Lang } from "../types.ts";
import { el, g } from "./svg.ts";
import { fitPreference, kernel, pointIndex, predictPreference, type Duel, type Kernel, type PrefFit, type X } from "./gp.ts";
import { argmax, ei, expectedMax2 } from "./acq.ts";
import { normal, rng } from "./random.ts";
import { Phi } from "./stats.ts";

export type Pt = number[];
export const D = 6;
export const ORIGINAL: Pt = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5];

// The photograph: Zelenci nature reserve, Slovenia, 19 October 2014, by Aleš
// Krivec, CC0 1.0 (Wikimedia Commons, File:Zelenci_2014_(2).jpg). The stored
// copy is 720 by 477 pixels. The build and the client replace @@ROOT@@.
export const PHOTO_HREF = "@@ROOT@@en/images/cs-photo-zelenci.jpg";
export const PHOTO_ASPECT = 720 / 477;

// ---------------------------------------------------------------------------
// The six adjustments. Each maps u in [0, 1] to a physical setting, with the
// identity at u = 0.5.

const signed = (v: number, d = 1) => (v >= 0 ? "+" : "−") + Math.abs(v).toFixed(d);

export interface Knob { key: string; label: string; map(u: number): number; fmt(u: number): string; lo: string; hi: string }

export const KNOBS: Knob[] = [
  { key: "exposure", label: "exposure", map: (u) => (u - 0.5) * 2, fmt: (u) => `${signed((u - 0.5) * 2)} EV`, lo: "darker", hi: "brighter" },
  { key: "contrast", label: "contrast", map: (u) => 2 ** ((u - 0.5) * 1.2), fmt: (u) => `×${(2 ** ((u - 0.5) * 1.2)).toFixed(2)}`, lo: "flat", hi: "punchy" },
  { key: "saturation", label: "saturation", map: (u) => 2 ** ((u - 0.5) * 2), fmt: (u) => `×${(2 ** ((u - 0.5) * 2)).toFixed(2)}`, lo: "muted", hi: "vivid" },
  { key: "temperature", label: "temperature", map: (u) => (u - 0.5) * 2, fmt: (u) => signed((u - 0.5) * 2, 2), lo: "cool", hi: "warm" },
  { key: "tint", label: "tint", map: (u) => (u - 0.5) * 2, fmt: (u) => signed((u - 0.5) * 2, 2), lo: "green", hi: "magenta" },
  { key: "shadows", label: "shadows", map: (u) => (u - 0.5) * 2, fmt: (u) => signed((u - 0.5) * 2, 2), lo: "deeper", hi: "lifted" },
];

// The names of the adjustments and of the two ends of their ranges, in the
// edition's language (the English ones are those of KNOBS).
const KNOB_ZH: Array<{ label: string; lo: string; hi: string }> = [
  { label: "曝光", lo: "更暗", hi: "更亮" },
  { label: "对比度", lo: "平淡", hi: "强烈" },
  { label: "饱和度", lo: "素淡", hi: "鲜艳" },
  { label: "色温", lo: "偏冷", hi: "偏暖" },
  { label: "色调", lo: "偏绿", hi: "偏品红" },
  { label: "阴影", lo: "更深", hi: "提亮" },
];
export function knobText(i: number, lang: Lang = "en"): { label: string; lo: string; hi: string } {
  return lang === "zh" ? KNOB_ZH[i] : { label: KNOBS[i].label, lo: KNOBS[i].lo, hi: KNOBS[i].hi };
}

// The six settings of a design as text: "exposure +0.3 EV, contrast ×1.06, ...".
export function settingsText(u: Pt, lang: Lang = "en"): string {
  return KNOBS.map((k, i) => `${knobText(i, lang).label} ${k.fmt(u[i])}`).join(lang === "zh" ? "，" : ", ");
}

// The tone curve applied to each of R, G, B (in sRGB values in [0, 1]):
// exposure as an endpoint-preserving curve y = kx / (1 + (k - 1)x) with
// k = 2^EV, then a shadow lift or crush that peaks at x = 1/3, then a linear
// contrast about mid-gray, clamped.
export function tone(x: number, u: Pt): number {
  const k = 2 ** KNOBS[0].map(u[0]);
  let y = (k * x) / (1 + (k - 1) * x);
  const a = 0.14 * KNOBS[5].map(u[5]);
  y = y + (a * y * (1 - y) * (1 - y)) / (4 / 27);
  const c = KNOBS[1].map(u[1]);
  y = 0.5 + c * (y - 0.5);
  return Math.min(1, Math.max(0, y));
}

// The color matrix: saturation about Rec. 709 luma, then per-channel gains
// for temperature (red against blue) and tint (green against magenta).
export function colorMatrix(u: Pt): number[] {
  const s = KNOBS[2].map(u[2]);
  const t = KNOBS[3].map(u[3]);
  const m = KNOBS[4].map(u[4]);
  const lum = [0.2126, 0.7152, 0.0722];
  const gain = [1 + 0.11 * t + 0.03 * m, 1 + 0.015 * t - 0.07 * m, 1 - 0.13 * t + 0.03 * m];
  const out: number[] = [];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) out.push(gain[r] * ((1 - s) * lum[c] + (r === c ? s : 0)));
    out.push(0, 0);
  }
  out.push(0, 0, 0, 1, 0);
  return out;
}

const n4 = (v: number) => String(Math.round(v * 1e4) / 1e4);

export function filterDef(id: string, u: Pt): string {
  const table = Array.from({ length: 17 }, (_, i) => n4(tone(i / 16, u))).join(" ");
  return el("filter", { id, x: 0, y: 0, width: 1, height: 1, "color-interpolation-filters": "sRGB" },
    el("feComponentTransfer", {},
      el("feFuncR", { type: "table", tableValues: table }),
      el("feFuncG", { type: "table", tableValues: table }),
      el("feFuncB", { type: "table", tableValues: table }),
    ),
    el("feColorMatrix", { type: "matrix", values: colorMatrix(u).map(n4).join(" ") }),
  );
}

// The photograph at (x, y, w, h) with the adjustments u applied. `id` must be
// unique in the page (include the render uid). `hit` names a click target.
export function photo(x: number, y: number, w: number, h: number, u: Pt, id: string, o: { rx?: number; hit?: string; ring?: string; ringWidth?: number; opacity?: number } = {}): string {
  const rx = o.rx ?? 6;
  return g({ "data-fig-hit": o.hit, class: o.hit ? "fig-card" : undefined, opacity: o.opacity },
    el("defs", {}, filterDef(`${id}-f`, u), el("clipPath", { id: `${id}-c` }, el("rect", { x, y, width: w, height: h, rx }))),
    g({ "clip-path": `url(#${id}-c)` },
      el("rect", { x, y, width: w, height: h, fill: "var(--fig-panel)" }),
      el("image", { href: PHOTO_HREF, x, y, width: w, height: h, preserveAspectRatio: "xMidYMid slice", filter: `url(#${id}-f)` }),
    ),
    el("rect", { x, y, width: w, height: h, rx, fill: "none", stroke: o.ring ?? "var(--fig-rule)", "stroke-width": o.ringWidth ?? 1 }),
  );
}

// ---------------------------------------------------------------------------
// The log of answers. A pair is stored winner > loser; a slider as its two
// ends and the chosen position t in [0, 1]. A pair asked again as a
// consistency check carries a flag. Coordinates are rounded to three
// decimals, so a log recomputes to exactly the same model.

export type Entry =
  | { kind: "pair"; w: Pt; l: Pt; rep: boolean }
  | { kind: "slider"; a: Pt; b: Pt; t: number };

const r3 = (v: number) => Math.round(v * 1000) / 1000;
export const round3 = (p: Pt): Pt => p.map(r3);
const fmtPt = (p: Pt) => p.map((v) => String(r3(v))).join("/");
const parsePt = (s: string): Pt => s.split("/").map(Number);

export function parseLog(s: string): Entry[] {
  if (!s.trim()) return [];
  return s.split(",").map((tok) => {
    if (tok[0] === "S") {
      const [ab, t] = tok.slice(1).split("@");
      const [a, b] = ab.split("|");
      return { kind: "slider", a: parsePt(a), b: parsePt(b), t: Number(t) };
    }
    const rep = tok.endsWith("r");
    const body = tok.slice(1, rep ? -1 : undefined);
    const [w, l] = body.split(">");
    return { kind: "pair", w: parsePt(w), l: parsePt(l), rep };
  });
}

export function fmtLog(es: Entry[]): string {
  return es.map((e) => e.kind === "slider"
    ? `S${fmtPt(e.a)}|${fmtPt(e.b)}@${r3(e.t)}`
    : `P${fmtPt(e.w)}>${fmtPt(e.l)}${e.rep ? "r" : ""}`).join(",");
}

const okPt = (p: Pt) => p.length === D && p.every((v) => Number.isFinite(v) && v >= 0 && v <= 1);
export function validLog(s: string): string | undefined {
  try {
    const es = parseLog(s);
    const ok = es.every((e) => e.kind === "slider" ? okPt(e.a) && okPt(e.b) && e.t >= 0 && e.t <= 1 : okPt(e.w) && okPt(e.l));
    return ok ? undefined : "expected P<w>><l> or S<a>|<b>@<t> entries with six coordinates in [0, 1]";
  } catch {
    return "malformed answer log";
  }
}

export const lerp = (a: Pt, b: Pt, t: number): Pt => a.map((v, i) => v + t * (b[i] - v));
export const dist = (a: Pt, b: Pt) => Math.sqrt(a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0));

// A slider answer becomes several pairwise comparisons: the chosen design
// beats both ends and two interior points of the slider, skipping any point
// closer than 0.15 (in slider units) to the choice. They share one judgment,
// so this overstates the evidence somewhat (the chapter says so).
export const SLIDER_LOSERS = [0, 1 / 3, 2 / 3, 1];
export function sliderDuels(e: { a: Pt; b: Pt; t: number }): Array<[Pt, Pt]> {
  const c = round3(lerp(e.a, e.b, e.t));
  return SLIDER_LOSERS.filter((s) => Math.abs(s - e.t) >= 0.15).map((s) => [c, round3(lerp(e.a, e.b, s))]);
}

// ---------------------------------------------------------------------------
// The model: a Gaussian process utility over [0, 1]^6 with a Matérn 5/2
// kernel (lengthscale 0.4 in every coordinate, prior variance 1), learned
// from the comparisons with the probit likelihood (noise 0.5 in the model's
// units) and the Laplace approximation. Both values were chosen by running
// simulated sessions (tools/figure-data/cs-photo-race.ts); they are fixed,
// not fitted to the reader's answers.

export const MODEL_SIGMA = 0.5;
export const KERNEL = kernel("matern52", 0.4, 1);

export interface Model {
  fit: PrefFit;
  xs: Pt[]; // distinct compared designs
  best: Pt; // the compared design with the highest posterior mean (the recommendation)
  bestMean: number;
}

export function fitLog(es: Entry[], k: Kernel = KERNEL, sigma = MODEL_SIGMA): Model {
  const xs: X[] = [];
  const duels: Duel[] = [];
  const add = (w: Pt, l: Pt) => duels.push({ winner: pointIndex(xs, w, 1e-4), loser: pointIndex(xs, l, 1e-4) });
  for (const e of es) {
    if (e.kind === "pair") add(e.w, e.l);
    else for (const [w, l] of sliderDuels(e)) add(w, l);
  }
  const fit = fitPreference(k, xs, duels, sigma);
  const pts = xs as Pt[];
  if (!pts.length) return { fit, xs: pts, best: ORIGINAL, bestMean: 0 };
  const at = predictPreference(fit, pts);
  const bi = argmax(at.mean);
  return { fit, xs: pts, best: pts[bi], bestMean: at.mean[bi] };
}

// A candidate pool for maximizing an acquisition function in six dimensions:
// uniform points, Gaussian perturbations of the best design, and every
// design compared so far. Seeded by the number of answers, so the pool, and
// with it the next query, is a pure function of the log.
export function candidatePool(m: Model, seed: number, step: number, nGlobal = 160, nLocal = 60): Pt[] {
  const r = rng(seed * 7717 + step * 131 + 3);
  const z = normal(r);
  const out: Pt[] = [];
  for (let i = 0; i < nGlobal; i++) out.push(round3(Array.from({ length: D }, () => r())));
  for (let i = 0; i < nLocal; i++) out.push(round3(m.best.map((v) => Math.min(1, Math.max(0, v + 0.1 * z())))));
  for (const x of m.xs) out.push(x);
  return out;
}

// The first query, before any answer: the original photograph against a
// random edit. Both query forms start from it.
export function firstOther(seed: number): Pt {
  const r = rng(seed * 104729 + 11);
  // keep the first edit clearly different from the original
  return round3(Array.from({ length: D }, () => { const v = r(); return v < 0.5 ? 0.1 + 0.3 * v : 0.6 + 0.3 * v; }));
}

// The pair with the highest EUBO among all pairs in the pool whose two
// versions are at least MIN_PAIR_DIST apart, so that a person can see a
// difference between them.
export const MIN_PAIR_DIST = 0.2;
export function nextPair(m: Model, seed: number, step: number, minDist = MIN_PAIR_DIST): [Pt, Pt] {
  if (!m.xs.length) return [ORIGINAL, firstOther(seed)];
  const pool = candidatePool(m, seed, step);
  const post = predictPreference(m.fit, pool, true);
  const cov = post.cov!;
  let bi = 0, bj = 1, bv = -Infinity;
  for (let i = 0; i < pool.length; i++) for (let j = i + 1; j < pool.length; j++) {
    if (dist(pool[i], pool[j]) < minDist) continue;
    const v = expectedMax2(post.mean[i], cov[i][i], post.mean[j], cov[j][j], cov[i][j]);
    if (v > bv) { bv = v; bi = i; bj = j; }
  }
  return [pool[bi], pool[bj]];
}

// A random pair from the pool, for comparison with EUBO.
export function randomPair(m: Model, seed: number, step: number): [Pt, Pt] {
  if (!m.xs.length) return [ORIGINAL, firstOther(seed)];
  const r = rng(seed * 3571 + step * 17 + 1);
  return [round3(Array.from({ length: D }, () => r())), round3(Array.from({ length: D }, () => r()))];
}

// The slider of sequential line search. Koyama et al. (2017) run it from the
// best design so far, x+, to the design with the highest expected
// improvement over it, x^EI ("segment"). We extend that segment to the edges
// of the box in both directions ("line", the default), so the slider also
// offers less of the change as well as more, and covers a range of looks wide
// enough to see; in simulated sessions this reached a given utility gap in
// fewer answers. Before any answer, the line runs through the original and a
// random edit. Returns the ends a, b and where x+ and x^EI sit on [0, 1].

export interface Slider { a: Pt; b: Pt; tBest: number; tTarget: number }

export function nextSlider(m: Model, seed: number, step: number, extend: "segment" | "ray" | "line" = "line"): Slider {
  let anchor: Pt, target: Pt;
  if (!m.xs.length) { anchor = ORIGINAL; target = firstOther(seed); }
  else {
    const pool = candidatePool(m, seed, step);
    const post = predictPreference(m.fit, pool);
    let bi = -1, bv = -Infinity;
    pool.forEach((x, i) => {
      if (dist(x, m.best) < 0.25) return;
      const v = ei(post.mean[i], Math.sqrt(post.var[i]), m.bestMean);
      if (v > bv) { bv = v; bi = i; }
    });
    anchor = m.best;
    target = bi >= 0 ? pool[bi] : firstOther(seed + step);
  }
  if (extend === "segment") return { a: anchor, b: target, tBest: 0, tTarget: 1 };
  const dir = target.map((v, i) => v - anchor[i]);
  const reach = (sgn: number) => {
    let tmax = Infinity;
    dir.forEach((dv, i) => {
      const d = sgn * dv;
      if (d > 1e-9) tmax = Math.min(tmax, (1 - anchor[i]) / d);
      else if (d < -1e-9) tmax = Math.min(tmax, (0 - anchor[i]) / d);
    });
    return Number.isFinite(tmax) ? tmax : 0;
  };
  const back = extend === "ray" ? 0 : reach(-1), fwd = reach(1);
  const a = round3(anchor.map((v, i) => Math.min(1, Math.max(0, v - back * dir[i]))));
  const b = round3(anchor.map((v, i) => Math.min(1, Math.max(0, v + fwd * dir[i]))));
  const span = back + fwd || 1;
  return { a, b, tBest: back / span, tTarget: (back + 1) / span };
}

// Posterior slices through a design: for each parameter, the utility's mean
// and standard deviation as that one parameter moves and the others stay.
export function slices(m: Model, at: Pt, n = 33): Array<{ xs: number[]; mean: number[]; sd: number[] }> {
  const ts = Array.from({ length: n }, (_, i) => i / (n - 1));
  return KNOBS.map((_, d) => {
    const pts = ts.map((t) => { const p = at.slice(); p[d] = t; return p; });
    const post = predictPreference(m.fit, pts);
    return { xs: ts, mean: post.mean, sd: post.var.map(Math.sqrt) };
  });
}

// ---------------------------------------------------------------------------
// The simulated taste. A hidden favorite c, and a utility that falls off
// quadratically from it, with per-parameter weights (exposure and
// temperature matter most, tint least) and one interaction: exposure and
// shadow lift both brighten the picture, so too much of one can be offset by
// less of the other. With `drift`, the favorite moves steadily during the
// session: by `answers` answers it has moved by the vector `drift`.

const W = [3.2, 2.0, 2.2, 2.8, 1.4, 1.6];

export interface Taste { c0: Pt; drift: Pt; u(x: Pt, step?: number): number; favorite(step?: number): Pt }

export function taste(seed: number, drifting = false, horizon = 20): Taste {
  const r = rng(seed * 6007 + 29);
  const c0 = round3(Array.from({ length: D }, () => 0.5 + 0.56 * (r() - 0.5)));
  // the drift: saturation and warmth rise, contrast falls, by about 0.3 in all
  const drift = drifting ? [0.04, -0.12, 0.2, 0.18, 0, 0.04] : [0, 0, 0, 0, 0, 0];
  const favorite = (step = 0) => {
    const f = Math.min(1, step / horizon);
    return c0.map((v, i) => Math.min(0.97, Math.max(0.03, v + f * drift[i])));
  };
  const u = (x: Pt, step = 0) => {
    const c = favorite(step);
    const dlt = x.map((v, i) => v - c[i]);
    let s = 0;
    for (let i = 0; i < D; i++) s += W[i] * dlt[i] * dlt[i];
    s += 2.0 * (dlt[0] + 0.6 * dlt[5]) ** 2;
    return -s;
  };
  return { c0, drift, u, favorite };
}

// Simulated answers. Each design's perceived utility is its utility plus
// independent Gaussian noise of standard deviation `noise` (Thurstone's
// model), so a pair is answered with probability Phi(diff / (sqrt 2 noise)),
// and a slider position is the best of 41 noisy positions along the line.
export function simPair(tz: Taste, a: Pt, b: Pt, noise: number, r: () => number, step: number): boolean {
  return r() < Phi((tz.u(a, step) - tz.u(b, step)) / (Math.SQRT2 * noise));
}

export function simSlider(tz: Taste, a: Pt, b: Pt, noise: number, r: () => number, step: number): number {
  const z = normal(r);
  let bt = 0, bv = -Infinity;
  for (let k = 0; k <= 40; k++) {
    const t = k / 40;
    const v = tz.u(lerp(a, b, t), step) + noise * z();
    if (v > bv) { bv = v; bt = t; }
  }
  return bt;
}

// The gap between the hidden favorite's utility (zero) and the
// recommendation's, at a given step: the simple regret of the session.
export function gap(tz: Taste, x: Pt, step = 0): number {
  return -tz.u(x, step);
}
