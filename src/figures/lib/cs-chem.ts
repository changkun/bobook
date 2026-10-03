// The direct arylation benchmark of Shields et al. (Nature 2021) and a small
// Bayesian optimizer that replays it. Shared by the figures cs-chem-replay
// and cs-chem-game and by tools/figure-data/cs-chem-race.ts, which runs the
// same optimizer over many seeds for the numbers in the chapter.
//
// Data: src/figures/data/cs-chem-arylation.json, written by
// tools/figure-data/cs-chem-arylation.py from github.com/b-shields/edbo and
// github.com/b-shields/EvML at pinned commits (MIT License; the JSON carries
// the copyright and permission notice). It holds the measured yield of every
// one of the 1,728 reactions (high-throughput experiments, one measurement
// each), the choices of the 50 players of the paper's reaction optimization
// game, and the yields found by the paper's 50 Bayesian optimization runs.
// Nothing in the data is simulated; an "experiment" in the figures looks up
// the measured yield of the chosen reaction.
//
// The optimizer is a simplified re-implementation, not the paper's code: a
// Gaussian process with a Matérn 5/2 kernel over an encoding of the five
// choices, one lengthscale per choice picked by marginal likelihood from a
// small grid with a weak preference for long lengthscales, a noise variance
// from a small grid, standardized
// yields, expected improvement, and batches by the Kriging believer.

import chem from "../data/cs-chem-arylation.json" with { type: "json" };
import type { Lang } from "../types.ts";
import { cholesky, cholSolve, logDetFromChol, solveLower, type Mat } from "./linalg.ts";
import { Phi, phi } from "./stats.ts";
import { rng } from "./random.ts";

export const LIGANDS: string[] = chem.ligands;
export const BASES: string[] = chem.bases;
export const SOLVENTS: string[] = chem.solvents;
export const CONC: number[] = chem.conc;
export const TEMP: number[] = chem.temp;
export const YIELD: number[] = chem.yield;
export const N = YIELD.length; // 1,728
export const BEST_YIELD = Math.max(...YIELD);

export interface Player { area: string; expertise: string; experience: string; picks: number[] }
export const PLAYERS = chem.players as Player[];
export const BO_RANDOM = chem.boRandom as number[][];
export const BO_HUMAN = chem.boHuman as number[][];

// index = (((ligand * 4 + base) * 4 + solvent) * 3 + conc) * 3 + temp
export function parts(i: number): [number, number, number, number, number] {
  const t = i % 3, c = Math.floor(i / 3) % 3, s = Math.floor(i / 9) % 4, b = Math.floor(i / 36) % 4, l = Math.floor(i / 144);
  return [l, b, s, c, t];
}
export function reaction(l: number, b: number, s: number, c: number, t: number): number {
  return (((l * 4 + b) * 4 + s) * 3 + c) * 3 + t;
}
// Reagent names stay in Latin script in every edition, except the one
// solvent with a common Chinese name.
const SOLVENT_ZH: Record<string, string> = { "p-Xylene": "对二甲苯" };
export function solventName(s: number, lang: Lang = "en"): string {
  return (lang === "zh" ? SOLVENT_ZH[SOLVENTS[s]] : undefined) ?? SOLVENTS[s];
}
export function describeReaction(i: number, lang: Lang = "en"): string {
  const [l, b, s, c, t] = parts(i);
  if (lang === "zh") return `${LIGANDS[l]}、${BASES[b]}、${solventName(s, lang)}、${CONC[c]} M、${TEMP[t]} °C`;
  return `${LIGANDS[l]}, ${BASES[b]}, ${SOLVENTS[s]}, ${CONC[c]} M, ${TEMP[t]} °C`;
}

export type Encoding = "onehot" | "desc";
const GROUPS = 5;

// Each reaction as a vector, with a group label per coordinate so each of the
// five choices gets its own lengthscale. One-hot coordinates are scaled by
// 1/sqrt(2) so that two different ligands (or bases, or solvents) are at
// distance 1, the same as the two ends of the concentration or temperature
// range. Descriptor coordinates are principal components of the published
// DFT descriptors, scaled to [-0.5, 0.5].
function encode(kind: Encoding): { X: number[][]; group: number[] } {
  const group: number[] = [];
  const oh = (n: number, g: number) => { for (let k = 0; k < n; k++) group.push(g); };
  const D = chem.desc as Record<string, number[][]>;
  if (kind === "onehot") { oh(12, 0); oh(4, 1); oh(4, 2); } else { oh(D.ligand[0].length, 0); oh(D.base[0].length, 1); oh(D.solvent[0].length, 2); }
  group.push(3, 4);
  const X = Array.from({ length: N }, (_, i) => {
    const [l, b, s, c, t] = parts(i);
    const v: number[] = [];
    if (kind === "onehot") {
      const h = (n: number, k: number) => { for (let j = 0; j < n; j++) v.push(j === k ? Math.SQRT1_2 : 0); };
      h(12, l); h(4, b); h(4, s);
    } else {
      v.push(...D.ligand[l].map((z) => z / 2), ...D.base[b].map((z) => z / 2), ...D.solvent[s].map((z) => z / 2));
    }
    v.push(c / 2, t / 2);
    return v;
  });
  return { X, group };
}
const ENC: Record<Encoding, { X: number[][]; group: number[] }> = { onehot: encode("onehot"), desc: encode("desc") };
export const encodingDims = (e: Encoding) => ENC[e].X[0].length;

function m52(a: number[], b: number[], inv: number[]): number {
  let r2 = 0;
  for (let k = 0; k < a.length; k++) { const d = (a[k] - b[k]) * inv[k]; r2 += d * d; }
  const r = Math.sqrt(5 * r2);
  return (1 + r + (r * r) / 3) * Math.exp(-r);
}

// Settings of the model fit; exported so the race script can report them.
export const OPT = { lsGrid: [0.5, 1, 2, 4, 8], noiseGrid: [1e-2, 0.05, 0.2, 0.5], priorCenter: 4, priorWidth: 1, sweeps: 2, start: 4, startNoise: 0.2 };

export interface Fit {
  enc: Encoding;
  idx: number[];
  y: number[]; // standardized
  ls: number[]; // per group
  inv: number[]; // per coordinate, 1 / lengthscale
  noise: number;
  L: Mat;
  alpha: number[];
  mu: number;
  sd: number;
}

function invOf(enc: Encoding, ls: number[]): number[] {
  return ENC[enc].group.map((gi) => 1 / ls[gi]);
}

function gpFit(enc: Encoding, idx: number[], y: number[], ls: number[], noise: number): { L: Mat; alpha: number[]; lml: number; inv: number[] } {
  const X = ENC[enc].X, inv = invOf(enc, ls), n = idx.length;
  const K: Mat = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) K[i][j] = K[j][i] = m52(X[idx[i]], X[idx[j]], inv) + (i === j ? noise : 0);
  const L = cholesky(K, 1e-8);
  const alpha = cholSolve(L, y);
  let q = 0;
  for (let i = 0; i < n; i++) q += y[i] * alpha[i];
  return { L, alpha, inv, lml: -0.5 * q - 0.5 * logDetFromChol(L) - 0.5 * n * Math.log(2 * Math.PI) };
}

// A weak log-normal preference for long lengthscales (centered at 4, one
// e-fold wide): with a handful of experiments, the data alone cannot tell
// which choices matter. The settings were chosen by trying a few on this
// data set (tools/figure-data/cs-chem-race.ts reports the result), which is
// itself a small act of tuning; the paper tuned its priors on other reactions.
const logPrior = (ls: number[]) => ls.reduce((s, l) => s - 0.5 * (Math.log(l / OPT.priorCenter) / OPT.priorWidth) ** 2, 0);

export function fit(enc: Encoding, idx: number[], yields: number[]): Fit {
  const mu = yields.reduce((a, b) => a + b, 0) / yields.length;
  const sd = Math.sqrt(yields.reduce((a, b) => a + (b - mu) ** 2, 0) / yields.length) || 1;
  const y = yields.map((v) => (v - mu) / sd);
  let ls = new Array<number>(GROUPS).fill(OPT.start);
  let noise = OPT.startNoise;
  let best = gpFit(enc, idx, y, ls, noise);
  let bestScore = best.lml + logPrior(ls);
  // Coordinate search: two sweeps over the five lengthscales and the noise.
  for (let sweep = 0; sweep < OPT.sweeps; sweep++) {
    for (let g = 0; g <= GROUPS; g++) {
      const grid = g < GROUPS ? OPT.lsGrid : OPT.noiseGrid;
      for (const v of grid) {
        const ls2 = ls.slice();
        let nz = noise;
        if (g < GROUPS) ls2[g] = v; else nz = v;
        const f = gpFit(enc, idx, y, ls2, nz);
        const sc = f.lml + logPrior(ls2);
        if (sc > bestScore + 1e-9) { bestScore = sc; best = f; ls = ls2; noise = nz; }
      }
    }
  }
  return { enc, idx: idx.slice(), y, ls, inv: best.inv, noise, L: best.L, alpha: best.alpha, mu, sd };
}

// Posterior mean and standard deviation (standardized scale) at reactions `cand`.
export function posterior(f: Fit, cand: number[]): { mean: number[]; sdev: number[] } {
  const X = ENC[f.enc].X;
  const mean: number[] = [], sdev: number[] = [];
  for (const c of cand) {
    const k = f.idx.map((i) => m52(X[c], X[i], f.inv));
    let m = 0;
    for (let i = 0; i < k.length; i++) m += k[i] * f.alpha[i];
    const v = solveLower(f.L, k);
    let s = 1;
    for (const x of v) s -= x * x;
    mean.push(m);
    sdev.push(Math.sqrt(Math.max(s, 1e-12)));
  }
  return { mean, sdev };
}

// Condition the same GP (same hyperparameters) on extra pseudo-observations:
// the Kriging believer's "take the prediction on faith".
function believe(f: Fit, idx: number, value: number): Fit {
  const idx2 = [...f.idx, idx], y = [...f.y, value];
  const g = gpFit(f.enc, idx2, y, f.ls, f.noise);
  return { ...f, idx: idx2, y, L: g.L, alpha: g.alpha };
}

export const XI = 0.01;

export function ei(m: number, s: number, best: number): number {
  const d = m - best - XI;
  const z = d / s;
  return d * Phi(z) + s * phi(z);
}

// The next batch: maximize EI over untried reactions, then take the
// prediction at the chosen reaction as if it had been measured, and repeat.
export function nextBatch(f: Fit, tried: Set<number>, size: number): number[] {
  const out: number[] = [];
  let g = f;
  const best = Math.max(...f.y);
  const cand = Array.from({ length: N }, (_, i) => i).filter((i) => !tried.has(i));
  for (let k = 0; k < size && cand.length; k++) {
    const p = posterior(g, cand);
    let bi = 0, bv = -Infinity;
    for (let i = 0; i < cand.length; i++) { const v = ei(p.mean[i], p.sdev[i], best); if (v > bv) { bv = v; bi = i; } }
    const c = cand[bi];
    out.push(c);
    cand.splice(bi, 1);
    if (k < size - 1) g = believe(g, c, p.mean[bi]);
  }
  return out;
}

export function randomPicks(n: number, seed: number): number[] {
  const r = rng(seed * 104729 + 7);
  const all = Array.from({ length: N }, (_, i) => i);
  for (let i = N - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); [all[i], all[j]] = [all[j], all[i]]; }
  return all.slice(0, n);
}

export type Init = "random" | "chemist";

export interface BORun { picks: number[]; fits: Fit[] } // fits[k]: the model used to choose batch k + 1

// Bayesian optimization for `budget` experiments in batches of `batch`. The
// first batch is random (the first reactions random search would try with
// the same seed) or the first experiments of a player of the game.
export function boRun(enc: Encoding, batch: number, init: Init, budget: number, seed: number): BORun {
  const first = init === "chemist" ? PLAYERS[(seed - 1) % PLAYERS.length].picks.slice(0, batch) : randomPicks(batch, seed);
  const picks = first.slice(0, budget);
  const fits: Fit[] = [];
  while (picks.length < budget) {
    const f = fit(enc, picks, picks.map((i) => YIELD[i]));
    fits.push(f);
    const nb = nextBatch(f, new Set(picks), Math.min(batch, budget - picks.length));
    picks.push(...nb);
  }
  return { picks, fits };
}

export function bestSoFar(picks: number[]): number[] {
  let b = -Infinity;
  return picks.map((i) => (b = Math.max(b, YIELD[i])));
}

export function bestSoFarYields(ys: number[]): number[] {
  let b = -Infinity;
  return ys.map((v) => (b = Math.max(b, v)));
}
