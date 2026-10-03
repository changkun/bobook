// Gaussian processes for figures: kernels, exact regression, samples, the log
// marginal likelihood, and the Laplace-approximated preference model of Chu
// and Ghahramani (2005). Inputs are numbers (1-D) or number arrays (n-D); every
// function takes a kernel, so a figure can swap RBF for Matérn with a control.
//
// The algorithms follow Rasmussen and Williams (2006): Algorithm 2.1 for
// regression and Section 3.4 for the Laplace approximation, adapted to the
// pairwise likelihood whose Hessian is not diagonal.

import { cholesky, cholSolve, logDetFromChol, luFactor, luSolve, solveLower, zeros, type Mat, type Vec } from "./linalg.ts";
import { logPhi, millsInv } from "./stats.ts";
import { normals, type Rng } from "./random.ts";

export type X = number | number[];

export interface Kernel {
  (a: X, b: X): number;
  variance: number; // k(x, x), the prior variance of f at any point
}

function sqdist(a: X, b: X, ls: number | number[]): number {
  if (typeof a === "number" && typeof b === "number") {
    const l = typeof ls === "number" ? ls : ls[0];
    return ((a - b) / l) ** 2;
  }
  const av = a as number[], bv = b as number[];
  let s = 0;
  for (let i = 0; i < av.length; i++) {
    const l = typeof ls === "number" ? ls : ls[i];
    s += ((av[i] - bv[i]) / l) ** 2;
  }
  return s;
}

export type KernelName = "rbf" | "matern52" | "matern32" | "matern12" | "periodic" | "linear";

// Kernel by name. `ls` is the lengthscale (one per dimension for ARD), `sf2`
// the signal variance, `period` the period of the periodic kernel.
export function kernel(name: KernelName, ls: number | number[] = 0.2, sf2 = 1, period = 1): Kernel {
  let k: (a: X, b: X) => number;
  switch (name) {
    case "rbf": k = (a, b) => sf2 * Math.exp(-0.5 * sqdist(a, b, ls)); break;
    case "matern52": k = (a, b) => { const r = Math.sqrt(5 * sqdist(a, b, ls)); return sf2 * (1 + r + (r * r) / 3) * Math.exp(-r); }; break;
    case "matern32": k = (a, b) => { const r = Math.sqrt(3 * sqdist(a, b, ls)); return sf2 * (1 + r) * Math.exp(-r); }; break;
    case "matern12": k = (a, b) => sf2 * Math.exp(-Math.sqrt(sqdist(a, b, ls))); break;
    case "periodic": k = (a, b) => {
      const d = Math.abs((a as number) - (b as number));
      const l = typeof ls === "number" ? ls : ls[0];
      return sf2 * Math.exp((-2 * Math.sin((Math.PI * d) / period) ** 2) / (l * l));
    }; break;
    case "linear": k = (a, b) => sf2 * (a as number) * (b as number); break;
  }
  const out = k as Kernel;
  out.variance = name === "linear" ? NaN : sf2;
  return out;
}

export function gram(k: Kernel, xs: X[], noise = 0): Mat {
  const n = xs.length;
  const K = zeros(n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) K[i][j] = K[j][i] = k(xs[i], xs[j]);
    K[i][i] += noise;
  }
  return K;
}

export function cross(k: Kernel, xs: X[], ys: X[]): Mat {
  return xs.map((a) => ys.map((b) => k(a, b)));
}

// Draw `count` functions from GP(m, k) evaluated at `xs`.
export function samplePrior(k: Kernel, xs: X[], r: Rng, count = 1, mean: (x: X) => number = () => 0): number[][] {
  const L = cholesky(gram(k, xs), 1e-8);
  const m = xs.map(mean);
  return Array.from({ length: count }, () => {
    const z = normals(r, xs.length);
    return L.map((row, i) => m[i] + row.reduce((s, v, j) => (j <= i ? s + v * z[j] : s), 0));
  });
}

export interface Posterior {
  mean: number[];
  var: number[]; // latent variance (no observation noise)
  cov?: Mat; // full covariance over the test points, when requested
}

export interface GPFit {
  xs: X[];
  ys: number[];
  k: Kernel;
  noise: number;
  L: Mat; // Cholesky of K + noise I
  alpha: Vec; // (K + noise I)^-1 (y - m)
  mean0: number; // constant prior mean
  logml: number; // log marginal likelihood
}

// Exact GP regression with Gaussian noise of variance `noise` and a constant
// prior mean. Rasmussen and Williams, Algorithm 2.1.
export function fit(k: Kernel, xs: X[], ys: number[], noise = 1e-6, mean0 = 0): GPFit {
  const n = xs.length;
  const L = n ? cholesky(gram(k, xs, noise), 1e-10) : [];
  const yc = ys.map((y) => y - mean0);
  const alpha = n ? cholSolve(L, yc) : [];
  const logml = n ? -0.5 * yc.reduce((s, y, i) => s + y * alpha[i], 0) - 0.5 * logDetFromChol(L) - 0.5 * n * Math.log(2 * Math.PI) : 0;
  return { xs, ys, k, noise, L, alpha, mean0, logml };
}

export function predict(g: GPFit, ts: X[], full = false): Posterior {
  const { k, xs, L, alpha, mean0 } = g;
  if (!xs.length) {
    const v = ts.map((t) => k(t, t));
    return { mean: ts.map(() => mean0), var: v, cov: full ? gram(k, ts) : undefined };
  }
  const Ks = cross(k, ts, xs); // m x n
  const mean = Ks.map((row) => mean0 + row.reduce((s, v, i) => s + v * alpha[i], 0));
  const V = Ks.map((row) => solveLower(L, row)); // each row: L^-1 k*
  const vars = ts.map((t, i) => Math.max(1e-12, k(t, t) - V[i].reduce((s, v) => s + v * v, 0)));
  let cov: Mat | undefined;
  if (full) {
    cov = zeros(ts.length);
    for (let i = 0; i < ts.length; i++) for (let j = 0; j <= i; j++) {
      let s = k(ts[i], ts[j]);
      for (let p = 0; p < V[i].length; p++) s -= V[i][p] * V[j][p];
      cov[i][j] = cov[j][i] = s;
    }
  }
  return { mean, var: vars, cov };
}

// Joint samples of f over `ts` from a posterior with full covariance.
export function samplePosterior(post: Posterior, r: Rng, count = 1): number[][] {
  if (!post.cov) throw new Error("samplePosterior needs predict(..., full = true)");
  const L = cholesky(post.cov, 1e-8);
  return Array.from({ length: count }, () => {
    const z = normals(r, post.mean.length);
    return post.mean.map((m, i) => {
      let s = m;
      for (let j = 0; j <= i; j++) s += L[i][j] * z[j];
      return s;
    });
  });
}

// ---------------------------------------------------------------------------
// Preference learning (Chu and Ghahramani, 2005).
//
// The latent utility f is a GP. A duel "a beats b" is observed with
// probability Phi((f(a) - f(b)) / (sqrt(2) * sigma)): Thurstone's comparative
// judgment with Gaussian noise of variance sigma^2 on each option's utility.
// The posterior over f at the dueled points is not Gaussian; the Laplace
// approximation replaces it by the Gaussian at its mode whose precision is the
// curvature there.

export interface Duel { winner: number; loser: number } // indices into the point list

export interface PrefFit {
  xs: X[];
  duels: Duel[];
  k: Kernel;
  sigma: number;
  fhat: Vec; // posterior mode of f at xs
  grad: Vec; // gradient of the log-likelihood at the mode; equals K^-1 fhat
  M: Mat; // (I + W K)^-1 W, so latent variance at t is k(t,t) - k_t^T M k_t
  W: Mat; // negative Hessian of the log-likelihood at the mode
  iters: number;
  logLik: number;
}

// Solve A x = b by Gaussian elimination with partial pivoting (A is small and
// not symmetric here).
function solveGeneral(A: Mat, b: Vec): Vec {
  const n = A.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const piv = M[c][c] || 1e-12;
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / piv;
      if (f === 0) continue;
      for (let j = c; j <= n; j++) M[r][j] -= f * M[c][j];
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

function likTerms(f: Vec, duels: Duel[], sigma: number) {
  const n = f.length;
  const s = Math.SQRT2 * sigma;
  const grad = new Array<number>(n).fill(0);
  const W = zeros(n);
  let ll = 0;
  for (const { winner: i, loser: j } of duels) {
    const z = (f[i] - f[j]) / s;
    ll += logPhi(z);
    const r = millsInv(z); // phi(z) / Phi(z)
    grad[i] += r / s;
    grad[j] -= r / s;
    const w = (r * (z + r)) / (s * s); // -d^2/df^2 log Phi(z), positive
    W[i][i] += w; W[j][j] += w; W[i][j] -= w; W[j][i] -= w;
  }
  return { grad, W, ll };
}

export function fitPreference(k: Kernel, xs: X[], duels: Duel[], sigma = 0.1, maxIter = 50): PrefFit {
  const n = xs.length;
  const K = gram(k, xs, 1e-8);
  let f = new Array<number>(n).fill(0);
  let it = 0;
  let terms = likTerms(f, duels, sigma);
  let objective = terms.ll; // log-likelihood - 0.5 f^T K^-1 f, with f = 0 at start
  let Lk: Mat | undefined; // Cholesky of K, computed only if a step needs damping
  for (; it < maxIter && n; it++) {
    // Newton step: f_new = (K^-1 + W)^-1 (W f + grad) = K (I + W K)^-1 (W f + grad)
    const b = terms.W.map((row, i) => row.reduce((s, v, j) => s + v * f[j], 0) + terms.grad[i]);
    const IWK = zeros(n);
    for (let i = 0; i < n; i++) {
      const Wi = terms.W[i], row = IWK[i];
      row[i] = 1;
      for (let p = 0; p < n; p++) {
        const w = Wi[p];
        if (w === 0) continue;
        const Kp = K[p];
        for (let j = 0; j < n; j++) row[j] += w * Kp[j];
      }
    }
    const a = solveGeneral(IWK, b); // a = K^-1 f_new
    let fnew = K.map((row) => row.reduce((s, v, j) => s + v * a[j], 0));
    // Damp the step if the objective did not improve.
    let step = 1;
    let next = likTerms(fnew, duels, sigma);
    let obj = next.ll - 0.5 * fnew.reduce((s, v, i) => s + v * a[i], 0);
    while (obj < objective - 1e-9 && step > 1e-3) {
      step /= 2;
      fnew = f.map((v, i) => v + step * (fnew[i] - v));
      next = likTerms(fnew, duels, sigma);
      Lk ??= cholesky(K, 1e-8);
      const af = cholSolve(Lk, fnew);
      obj = next.ll - 0.5 * fnew.reduce((s, v, i) => s + v * af[i], 0);
    }
    const delta = Math.max(...fnew.map((v, i) => Math.abs(v - f[i])));
    f = fnew;
    terms = next;
    objective = obj;
    if (delta < 1e-7) { it++; break; }
  }
  // M = (I + W K)^-1 W: factor once, solve one column of W at a time.
  const IWK = zeros(n);
  for (let i = 0; i < n; i++) {
    const Wi = terms.W[i], row = IWK[i];
    row[i] = 1;
    for (let p = 0; p < n; p++) {
      const w = Wi[p];
      if (w === 0) continue;
      const Kp = K[p];
      for (let j = 0; j < n; j++) row[j] += w * Kp[j];
    }
  }
  const M = zeros(n);
  if (n) {
    const f = luFactor(IWK);
    for (let j = 0; j < n; j++) {
      const col = luSolve(f, terms.W.map((row) => row[j]));
      for (let i = 0; i < n; i++) M[i][j] = col[i];
    }
  }
  return { xs, duels, k, sigma, fhat: f, grad: terms.grad, M, W: terms.W, iters: it, logLik: terms.ll };
}

// Latent posterior of f at test points under the Laplace approximation.
export function predictPreference(g: PrefFit, ts: X[], full = false): Posterior {
  const { k, xs, grad, M } = g;
  if (!xs.length) return { mean: ts.map(() => 0), var: ts.map((t) => k(t, t)), cov: full ? gram(k, ts) : undefined };
  const Ks = cross(k, ts, xs);
  const mean = Ks.map((row) => row.reduce((s, v, i) => s + v * grad[i], 0));
  const KM = Ks.map((row) => xs.map((_, j) => row.reduce((s, v, p) => s + v * M[p][j], 0)));
  const vars = ts.map((t, i) => Math.max(1e-12, k(t, t) - KM[i].reduce((s, v, j) => s + v * Ks[i][j], 0)));
  let cov: Mat | undefined;
  if (full) {
    cov = zeros(ts.length);
    for (let i = 0; i < ts.length; i++) for (let j = 0; j <= i; j++) {
      cov[i][j] = cov[j][i] = k(ts[i], ts[j]) - KM[i].reduce((s, v, p) => s + v * Ks[j][p], 0);
    }
  }
  return { mean, var: vars, cov };
}

// Index of a value in a point list, appending it if new: duels are stored by
// position, and the latent vector holds one entry per distinct point.
export function pointIndex(xs: X[], x: X, tol = 1e-9): number {
  const i = xs.findIndex((p) => (typeof p === "number" ? Math.abs(p - (x as number)) < tol : (p as number[]).every((v, j) => Math.abs(v - (x as number[])[j]) < tol)));
  if (i >= 0) return i;
  xs.push(x);
  return xs.length - 1;
}
