// Three ways to choose duels on a K-armed preference matrix, for the race
// figure of the dueling-bandits chapter. Each takes the win counts so far and
// returns the next pair; the environment then draws the winner from P.
//
//   uniform   a uniformly random pair of distinct arms (no learning).
//   rucb      Relative Upper Confidence Bound (Zoghi et al. 2014), in a
//             simplified form: the champion is drawn uniformly from the arms
//             that beat every other arm under optimistic estimates (the
//             original adds a hypothesis set B that biases this draw), and the
//             challenger is the arm with the highest optimistic chance of
//             beating the champion.
//   dts       Double Thompson Sampling (Wu and Liu 2016), Copeland version:
//             the first arm maximizes a sampled Copeland score among the arms
//             with the highest optimistic Copeland score, and the second is
//             the arm most likely to beat it under a second posterior sample,
//             among arms not yet known to lose to it.
//
// Regret is Copeland regret (Zoghi et al. 2015; Wu and Liu 2016): with
// normalized Copeland scores z_i = (number of arms i beats) / (K - 1), a duel
// (a, b) costs max_i z_i - (z_a + z_b) / 2. When a Condorcet winner exists,
// max_i z_i = 1.

import { normal, type Rng } from "./random.ts";

export type Algo = "uniform" | "rucb" | "dts";

const ALPHA = 0.51;

function gamma(r: Rng, z: () => number, shape: number): number {
  // Marsaglia and Tsang (2000); shapes here are always >= 1.
  const d = shape - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x = 0, v = 0;
    do { x = z(); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = r();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function beta(r: Rng, z: () => number, a: number, b: number): number {
  const x = gamma(r, z, a), y = gamma(r, z, b);
  return x / (x + y);
}

function pickRandom(r: Rng, xs: number[]): number {
  return xs[Math.floor(r() * xs.length)];
}

function argmaxTies(r: Rng, idx: number[], score: (i: number) => number): number {
  let best = -Infinity;
  let arg: number[] = [];
  for (const i of idx) {
    const s = score(i);
    if (s > best + 1e-12) { best = s; arg = [i]; } else if (Math.abs(s - best) <= 1e-12) arg.push(i);
  }
  return pickRandom(r, arg);
}

// Optimistic and pessimistic estimates of P[i][j] from win counts W.
function bounds(W: number[][], t: number, K: number) {
  const U: number[][] = [], L: number[][] = [];
  for (let i = 0; i < K; i++) {
    U.push([]); L.push([]);
    for (let j = 0; j < K; j++) {
      if (i === j) { U[i].push(0.5); L[i].push(0.5); continue; }
      const n = W[i][j] + W[j][i];
      if (n === 0) { U[i].push(1); L[i].push(0); continue; }
      const m = W[i][j] / n, w = Math.sqrt((ALPHA * Math.log(t)) / n);
      U[i].push(m + w); L[i].push(m - w);
    }
  }
  return { U, L };
}

export function choose(algo: Algo, W: number[][], t: number, r: Rng, z: () => number): [number, number] {
  const K = W.length;
  const all = Array.from({ length: K }, (_, i) => i);
  if (algo === "uniform") {
    const a = Math.floor(r() * K);
    let b = Math.floor(r() * (K - 1));
    if (b >= a) b++;
    return [a, b];
  }
  const { U, L } = bounds(W, t, K);
  if (algo === "rucb") {
    const cand = all.filter((c) => all.every((j) => U[c][j] >= 0.5));
    const c = cand.length ? pickRandom(r, cand) : pickRandom(r, all);
    const d = argmaxTies(r, all, (j) => U[j][c]);
    return [c, d];
  }
  // Double Thompson Sampling
  const upperCope = all.map((i) => all.reduce((s, j) => (j !== i && U[i][j] > 0.5 ? s + 1 : s), 0));
  const top = Math.max(...upperCope);
  const C = all.filter((i) => upperCope[i] === top);
  const th: number[][] = all.map(() => new Array<number>(K).fill(0.5));
  for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) {
    const v = beta(r, z, W[i][j] + 1, W[j][i] + 1);
    th[i][j] = v; th[j][i] = 1 - v;
  }
  const a = argmaxTies(r, C, (i) => all.reduce((s, j) => (j !== i && th[i][j] > 0.5 ? s + 1 : s), 0));
  const cand2 = all.filter((i) => i === a || L[i][a] <= 0.5);
  const b = argmaxTies(r, cand2, (i) => (i === a ? 0.5 : beta(r, z, W[i][a] + 1, W[a][i] + 1)));
  return [a, b];
}

// Mean cumulative Copeland regret over `runs` independent runs, sampled at
// `points` evenly spaced rounds (the last is T).
export function race(P: number[][], algo: Algo, T: number, runs: number, rngFor: (run: number) => Rng, points = 80): number[] {
  const K = P.length;
  const cope = P.map((row, i) => row.reduce((s, p, j) => (j === i ? s : s + (p > 0.5 ? 1 : p === 0.5 ? 0.5 : 0)), 0) / (K - 1));
  const best = Math.max(...cope);
  const every = T / points;
  const out = new Array<number>(points).fill(0);
  for (let run = 0; run < runs; run++) {
    const r = rngFor(run);
    const z = normal(r);
    const W = Array.from({ length: K }, () => new Array<number>(K).fill(0));
    let cum = 0, k = 0;
    for (let t = 1; t <= T; t++) {
      const [a, b] = choose(algo, W, t, r, z);
      cum += best - (cope[a] + cope[b]) / 2;
      if (a !== b) { if (r() < P[a][b]) W[a][b]++; else W[b][a]++; }
      while (k < points && t >= Math.round((k + 1) * every)) { out[k] += cum / runs; k++; }
    }
  }
  return out;
}
