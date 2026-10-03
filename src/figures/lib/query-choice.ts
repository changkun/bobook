// A Gaussian process utility learned from choices among sets of options, for
// the query-design chapter. Each observation says "option c was chosen from
// the set S", with the Bradley-Terry-Luce (softmax) likelihood
//
//   P(c | S) = exp(f(c) / tau) / sum_{j in S} exp(f(j) / tau),
//
// which is Luce's choice model with temperature tau; a pair is the special
// case |S| = 2 (Bradley-Terry with a logistic link). The sequential line
// search of Koyama et al. (2017) records a slider answer as "the chosen point
// beats both ends of the slider", a choice from a set of three.
//
// The posterior over f is approximated with the Laplace method exactly as in
// lib/gp.ts (fitPreference): Newton iterations to the mode, then a Gaussian
// whose precision is the curvature there. The curvature of one choice is
// (diag(p) - p p^T) / tau^2 on the options in its set, the covariance of a
// one-hot draw from p, which is positive semidefinite but not diagonal.

import { gram, cross, type Kernel, type X } from "./gp.ts";
import { zeros, type Mat, type Vec } from "./linalg.ts";

export interface Choice { chosen: number; set: number[] } // indices into the point list; set includes chosen

export interface ChoiceFit {
  xs: X[];
  k: Kernel;
  tau: number;
  fhat: Vec; // posterior mode of f at xs
  grad: Vec; // gradient of the log-likelihood at the mode (= K^-1 fhat)
  M: Mat; // (I + W K)^-1 W; latent variance at t is k(t,t) - k_t^T M k_t
  iters: number;
}

function terms(f: Vec, choices: Choice[], tau: number) {
  const n = f.length;
  const grad = new Array<number>(n).fill(0);
  const W = zeros(n);
  let ll = 0;
  for (const { chosen, set } of choices) {
    const z = set.map((j) => f[j] / tau);
    const zmax = Math.max(...z);
    const e = z.map((v) => Math.exp(v - zmax));
    const sum = e.reduce((a, b) => a + b, 0);
    const p = e.map((v) => v / sum);
    ll += f[chosen] / tau - (zmax + Math.log(sum));
    set.forEach((j, a) => {
      grad[j] += ((j === chosen ? 1 : 0) - p[a]) / tau;
      set.forEach((i, b) => {
        W[j][i] += ((a === b ? p[a] : 0) - p[a] * p[b]) / (tau * tau);
      });
    });
  }
  return { grad, W, ll };
}

// Solve A x = b by Gaussian elimination with partial pivoting (A small, not symmetric).
function solve(A: Mat, b: Vec): Vec {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const piv = M[c][c] || 1e-12;
    for (let r = c + 1; r < n; r++) {
      const q = M[r][c] / piv;
      if (q === 0) continue;
      for (let j = c; j <= n; j++) M[r][j] -= q * M[c][j];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = M[i][n];
    for (let j = i + 1; j < n; j++) s -= M[i][j] * x[j];
    x[i] = s / (M[i][i] || 1e-12);
  }
  return x;
}

function iwk(W: Mat, K: Mat): Mat {
  const n = W.length;
  const A = zeros(n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    let s = i === j ? 1 : 0;
    for (let p = 0; p < n; p++) s += W[i][p] * K[p][j];
    A[i][j] = s;
  }
  return A;
}

// Laplace fit. The iterate is tracked through a = K^-1 f, so the prior term
// f^T K^-1 f = f^T a needs no extra solve, and a damped step is linear in a.
export function fitChoice(k: Kernel, xs: X[], choices: Choice[], tau = 0.15, maxIter = 40): ChoiceFit {
  const n = xs.length;
  const K = gram(k, xs, 1e-8);
  let f = new Array<number>(n).fill(0);
  let a = new Array<number>(n).fill(0);
  let t = terms(f, choices, tau);
  let obj = t.ll;
  let it = 0;
  for (; it < maxIter && n; it++) {
    const b = t.W.map((row, i) => row.reduce((s, v, j) => s + v * f[j], 0) + t.grad[i]);
    const aFull = solve(iwk(t.W, K), b); // K^-1 f_new
    let step = 1;
    let aNew = aFull;
    let fNew = K.map((row) => row.reduce((s, v, j) => s + v * aNew[j], 0));
    let tNew = terms(fNew, choices, tau);
    let oNew = tNew.ll - 0.5 * fNew.reduce((s, v, i) => s + v * aNew[i], 0);
    while (oNew < obj - 1e-10 && step > 1e-3) {
      step /= 2;
      aNew = a.map((v, i) => v + step * (aFull[i] - v));
      fNew = K.map((row) => row.reduce((s, v, j) => s + v * aNew[j], 0));
      tNew = terms(fNew, choices, tau);
      oNew = tNew.ll - 0.5 * fNew.reduce((s, v, i) => s + v * aNew[i], 0);
    }
    const delta = Math.max(...fNew.map((v, i) => Math.abs(v - f[i])));
    f = fNew; a = aNew; t = tNew; obj = oNew;
    if (delta < 1e-7) { it++; break; }
  }
  const A = iwk(t.W, K);
  const M = zeros(n);
  for (let j = 0; j < n; j++) {
    const col = solve(A, t.W.map((row) => row[j]));
    for (let i = 0; i < n; i++) M[i][j] = col[i];
  }
  return { xs, k, tau, fhat: f, grad: t.grad, M, iters: it };
}

// Latent posterior mean and variance at test points.
export function predictChoice(g: ChoiceFit, ts: X[]): { mean: number[]; var: number[] } {
  const { k, xs, grad, M } = g;
  if (!xs.length) return { mean: ts.map(() => 0), var: ts.map((t) => k(t, t)) };
  const Ks = cross(k, ts, xs);
  const mean = Ks.map((row) => row.reduce((s, v, i) => s + v * grad[i], 0));
  const vars = Ks.map((row, i) => {
    let q = 0;
    for (let p = 0; p < row.length; p++) {
      let s = 0;
      for (let j = 0; j < row.length; j++) s += M[p][j] * row[j];
      q += row[p] * s;
    }
    return Math.max(1e-12, k(ts[i], ts[i]) - q);
  });
  return { mean, var: vars };
}
