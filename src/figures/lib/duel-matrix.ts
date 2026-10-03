// Five-armed preference matrices for the dueling-bandits chapter, and the
// solution concepts computed from them. P[i][j] is the probability that arm i
// beats arm j, with P[j][i] = 1 - P[i][j] and P[i][i] = 1/2.
//
// A scenario starts from a base matrix and adds a rock-paper-scissors
// component of strength c among the first three arms, on the logit scale:
//
//   P[i][j] = sigmoid(logit(P0[i][j]) + c * R[i][j]),
//
// where R is skew-symmetric with R[B][A] = R[C][B] = R[A][C] = 1, so that a
// large c makes B beat A, C beat B, and A beat C. Individual pairs can then be
// flipped (P[i][j] and P[j][i] swapped) to explore by hand.

export const ARMS = ["A", "B", "C", "D", "E"] as const;
export const K = ARMS.length;

export type Mat = number[][];

const sig = (z: number) => 1 / (1 + Math.exp(-z));
const logit = (p: number) => Math.log(p / (1 - p));

function fromUpper(upper: Record<string, number>): Mat {
  const P: Mat = Array.from({ length: K }, () => new Array<number>(K).fill(0.5));
  for (const [key, p] of Object.entries(upper)) {
    const [i, j] = key.split("").map((ch) => ARMS.indexOf(ch as (typeof ARMS)[number]));
    P[i][j] = p; P[j][i] = 1 - p;
  }
  return P;
}

// Base matrices.
//   utility: P = sigmoid(3 (u_i - u_j)) with u = (1, 0.7, 0.45, 0.2, 0): a
//            Bradley-Terry model, so every solution concept agrees.
//   narrow:  A beats everyone, but only just (0.55); B beats C, D, E by a
//            wide margin. The Condorcet winner is A; the Borda winner is B.
const UTIL = [1, 0.7, 0.45, 0.2, 0];
export const BASES: Record<string, Mat> = {
  utility: (() => {
    const P: Mat = Array.from({ length: K }, (_, i) => Array.from({ length: K }, (_, j) => (i === j ? 0.5 : sig(3 * (UTIL[i] - UTIL[j])))));
    return P;
  })(),
  narrow: fromUpper({ AB: 0.55, AC: 0.55, AD: 0.55, AE: 0.55, BC: 0.9, BD: 0.9, BE: 0.9, CD: 0.7, CE: 0.75, DE: 0.65 }),
};

const R: Mat = (() => {
  const M: Mat = Array.from({ length: K }, () => new Array<number>(K).fill(0));
  const set = (w: number, l: number) => { M[w][l] = 1; M[l][w] = -1; };
  set(1, 0); set(2, 1); set(0, 2);
  return M;
})();

export const PAIRS: Array<[number, number]> = [];
for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) PAIRS.push([i, j]);
export const pairKey = (i: number, j: number) => `${ARMS[Math.min(i, j)]}${ARMS[Math.max(i, j)]}`;

export function parseFlips(s: string): string[] {
  return s.trim() ? s.split(",").map((t) => t.trim()).filter(Boolean) : [];
}
export function validFlips(s: string): string | undefined {
  return parseFlips(s).every((k) => PAIRS.some(([i, j]) => pairKey(i, j) === k)) ? undefined : "expected pairs such as AB,CE";
}
export function toggleFlip(s: string, key: string): string {
  const fl = parseFlips(s);
  const i = fl.indexOf(key);
  if (i >= 0) fl.splice(i, 1); else fl.push(key);
  return fl.join(",");
}

export function scenario(base: string, c: number, flips: string): Mat {
  const P0 = BASES[base] ?? BASES.utility;
  const P: Mat = P0.map((row, i) => row.map((p, j) => (i === j ? 0.5 : sig(logit(p) + c * R[i][j]))));
  for (const key of parseFlips(flips)) {
    const i = ARMS.indexOf(key[0] as (typeof ARMS)[number]), j = ARMS.indexOf(key[1] as (typeof ARMS)[number]);
    const t = P[i][j]; P[i][j] = P[j][i]; P[j][i] = t;
  }
  return P;
}

export interface Concepts {
  condorcet: number; // index, or -1 when none exists
  copeland: number[]; // number of other arms each arm beats (ties count 1/2)
  borda: number[]; // average probability of beating another arm, over the K - 1 others
  vonNeumann: number[]; // a maximin mixed strategy of the symmetric zero-sum game
  copelandWinners: number[];
  bordaWinner: number;
}

export function concepts(P: Mat): Concepts {
  const copeland = P.map((row, i) => row.reduce((s, p, j) => (j === i ? s : s + (p > 0.5 + 1e-12 ? 1 : Math.abs(p - 0.5) <= 1e-12 ? 0.5 : 0)), 0));
  const borda = P.map((row, i) => row.reduce((s, p, j) => (j === i ? s : s + p), 0) / (K - 1));
  const condorcet = copeland.findIndex((c) => c === K - 1);
  const top = Math.max(...copeland);
  const copelandWinners = copeland.map((c, i) => (c === top ? i : -1)).filter((i) => i >= 0);
  let bordaWinner = 0;
  for (let i = 1; i < K; i++) if (borda[i] > borda[bordaWinner]) bordaWinner = i;
  return { condorcet, copeland, borda, vonNeumann: vonNeumann(P), copelandWinners, bordaWinner };
}

// The von Neumann winner: a distribution pi over arms with
// sum_i pi_i (P[i][j] - 1/2) >= 0 for every j (Dudik et al. 2015). The game
// with payoff M = P - 1/2 is symmetric and zero-sum with value 0, so pi is an
// optimal strategy of that game. With five arms we can find one exactly by
// support enumeration: for a candidate support S, the optimal strategy makes
// every column in S an exact tie (pi^T M[:, j] = 0 for j in S); we solve that
// linear system, and accept the solution if it is a distribution and no column
// outside S beats it. Generic antisymmetric games have supports of odd size.
export function vonNeumann(P: Mat): number[] {
  const M = P.map((row) => row.map((p) => p - 0.5));
  const tol = 1e-9;
  const check = (pi: number[]) => pi.every((v) => v >= -tol) && Array.from({ length: K }, (_, j) => pi.reduce((s, v, i) => s + v * M[i][j], 0)).every((v) => v >= -1e-7);
  const subsets: number[][] = [];
  for (let mask = 1; mask < 1 << K; mask++) {
    const S = Array.from({ length: K }, (_, i) => i).filter((i) => mask & (1 << i));
    subsets.push(S);
  }
  subsets.sort((a, b) => a.length - b.length);
  for (const S of subsets) {
    const n = S.length;
    // unknowns pi_S; equations: sum_i pi_i M[i][j] = 0 for j in S (one is
    // redundant for odd n), plus sum pi = 1. Solve the n x n system that
    // replaces the last tie equation by the normalization.
    const A: number[][] = [];
    const b: number[] = [];
    for (let r = 0; r < n - 1; r++) { A.push(S.map((i) => M[i][S[r]])); b.push(0); }
    A.push(S.map(() => 1)); b.push(1);
    const x = solve(A, b);
    if (!x) continue;
    const pi = new Array<number>(K).fill(0);
    S.forEach((i, k) => { pi[i] = x[k]; });
    // the dropped equation must hold too
    if (Math.abs(pi.reduce((s, v, i) => s + v * M[i][S[n - 1]], 0)) > 1e-7) continue;
    if (check(pi)) return pi.map((v) => Math.max(0, v));
  }
  return new Array<number>(K).fill(1 / K); // unreachable for valid matrices
}

function solve(A: number[][], b: number[]): number[] | null {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-12) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let j = c; j <= n; j++) M[r][j] -= f * M[c][j];
    }
  }
  return M.map((row, i) => row[n] / row[i]);
}
