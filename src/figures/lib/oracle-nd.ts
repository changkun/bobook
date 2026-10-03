// The model behind dim-oracle-nd, shared by the figure and the script that
// records its reference curves: a Gaussian process preference model over a
// D-dimensional design, EUBO over a candidate pool, and a simulated person
// with a separable hidden utility.

import { fitPreference, kernel, pointIndex, predictPreference, type Duel, type X } from "./gp.ts";
import { argmax, expectedMax2 } from "./acq.ts";
import { Phi } from "./stats.ts";
import { rng } from "./random.ts";
import { candidates } from "./nd.ts";

export const SIGMA = 0.12; // the model's assumed answer noise
// Lengthscale grows with dimension (see the high-dimensions chapter).
export const lengthscale = (D: number) => 0.18 * Math.sqrt(D) + 0.08;

export interface Session { xs: X[]; duels: Duel[] }

export function hidden(D: number, seed: number) {
  const r = rng(seed * 7919 + D);
  const c = Array.from({ length: D }, () => 0.15 + 0.7 * r());
  const w = Array.from({ length: D }, (_, i) => 1 / (1 + 0.25 * i)); // earlier parameters matter more
  const W = w.reduce((a, b) => a + b, 0);
  const u = (x: number[]) => x.reduce((s, v, i) => s + (w[i] / W) * Math.exp(-((v - c[i]) ** 2) / (2 * 0.2 ** 2)), 0);
  // E[u] under uniform x, per coordinate: integral of the bump over [0, 1]
  let mean = 0;
  for (let i = 0; i < D; i++) { let s = 0; for (let k = 0; k < 200; k++) s += Math.exp(-((((k + 0.5) / 200) - c[i]) ** 2) / (2 * 0.04)); mean += (w[i] / W) * s / 200; }
  return { c, u, max: u(c), mean };
}

export function model(D: number, pairs: Array<[number[], number[]]>) {
  const xs: X[] = [];
  const duels: Duel[] = pairs.map(([a, b]) => ({ winner: pointIndex(xs, a), loser: pointIndex(xs, b) }));
  return { fitted: fitPreference(kernel("rbf", lengthscale(D), 1), xs, duels, SIGMA), xs };
}

// Candidate pool, best guess, and the EUBO pair.
export function step(D: number, pairs: Array<[number[], number[]]>, seed: number, acq: "eubo" | "random" = "eubo") {
  const r = rng(seed * 104729 + pairs.length * 31 + D);
  const { fitted, xs } = model(D, pairs);
  const base = xs.length ? (xs as number[][]) : [];
  let pool: number[][];
  if (!pairs.length) pool = candidates(r, D, 120, [], 0);
  else {
    const m0 = predictPreference(fitted, base).mean;
    const order = base.map((_, i) => i).sort((a, b) => m0[b] - m0[a]);
    pool = [...candidates(r, D, 120, order.slice(0, 3).map((i) => base[i]), 90, 0.15, Math.min(1, 3 / D)), ...base];
  }
  const post = predictPreference(fitted, pool, true);
  const bestIdx = argmax(post.mean);
  let pair: [number[], number[]];
  if (!pairs.length || acq === "random") {
    const i = Math.floor(r() * pool.length);
    let j = Math.floor(r() * (pool.length - 1));
    if (j >= i) j++;
    pair = [pool[i], pool[j]];
  } else {
    let bi = 0, bj = 1, bv = -Infinity;
    for (let i = 0; i < pool.length; i++) for (let j = i + 1; j < pool.length; j++) {
      const v = expectedMax2(post.mean[i], post.cov![i][i], post.mean[j], post.cov![j][j], post.cov![i][j]);
      if (v > bv) { bv = v; bi = i; bj = j; }
    }
    pair = [pool[bi], pool[bj]];
  }
  return { fitted, best: pairs.length ? pool[bestIdx] : new Array(D).fill(0.5), pair };
}

export function simulatedAnswer(D: number, seed: number, noise: number, pair: [number[], number[]], k: number): [number[], number[]] {
  const { u } = hidden(D, seed);
  const r = rng(seed * 977 + k * 131 + 7);
  const p = Phi((u(pair[0]) - u(pair[1])) / (Math.SQRT2 * noise));
  return r() < p ? [pair[0], pair[1]] : [pair[1], pair[0]];
}

// Regret of the best guess as a fraction of the gap between the favorite and
// a random design: 1 is no better than chance, 0 is the favorite.
export function normRegret(D: number, seed: number, x: number[]): number {
  const h = hidden(D, seed);
  return Math.max(0, (h.max - h.u(x)) / (h.max - h.mean));
}
