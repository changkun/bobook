// Inference for a single latent utility difference d under a Gaussian prior
// and probit comparisons, used by the figures of the approximate-inference
// chapter. The prior is N(0, v0). Each comparison contributes Phi(y d / s),
// where y = +1 if A won and -1 if B won, and s = sqrt(2) sigma is the noise
// on the difference. One latent dimension keeps every method exact enough to
// compare: the true posterior on a fine grid, the Laplace approximation by
// Newton's method, expectation propagation with one site per comparison, a
// Gaussian variational approximation by coordinate ascent, and elliptical
// slice sampling.

import { Phi, logPhi, millsInv, phi } from "./stats.ts";
import { normal, rng } from "./random.ts";

export interface Gauss1 { mean: number; sd: number }

export interface Problem { v0: number; wins: number; losses: number; s: number }

// The log likelihood of all comparisons at d.
export function logLik(pr: Problem, d: number): number {
  const z = d / pr.s;
  return (pr.wins ? pr.wins * logPhi(z) : 0) + (pr.losses ? pr.losses * logPhi(-z) : 0);
}

export interface Exact { xs: number[]; dens: number[]; mean: number; sd: number; mode: number; pNeg: number; logZ: number }

// The exact posterior on an even grid over [lo, hi], normalized numerically.
export function exactGrid(pr: Problem, lo: number, hi: number, n = 2001): Exact {
  const h = (hi - lo) / (n - 1);
  const xs = Array.from({ length: n }, (_, i) => lo + i * h);
  const lp = xs.map((d) => -0.5 * (d * d) / pr.v0 + logLik(pr, d));
  const m = Math.max(...lp);
  const un = lp.map((v) => Math.exp(v - m));
  let Z = 0;
  for (let i = 0; i < n; i++) Z += un[i] * (i === 0 || i === n - 1 ? 0.5 : 1);
  Z *= h;
  const dens = un.map((v) => v / Z);
  let mean = 0, m2 = 0, pNeg = 0, mode = xs[0], best = -Infinity;
  for (let i = 0; i < n; i++) {
    const w = dens[i] * h * (i === 0 || i === n - 1 ? 0.5 : 1);
    mean += w * xs[i];
    m2 += w * xs[i] * xs[i];
    if (xs[i] < 0) pNeg += w;
    if (lp[i] > best) { best = lp[i]; mode = xs[i]; }
  }
  // log of the normalizing constant, the evidence: prior density is
  // N(d; 0, v0), so add its normalizer back.
  const logZ = m + Math.log(Z) - 0.5 * Math.log(2 * Math.PI * pr.v0);
  return { xs, dens, mean, sd: Math.sqrt(Math.max(0, m2 - mean * mean)), mode, pNeg, logZ };
}

// Derivatives of the log posterior at d.
function derivs(pr: Problem, d: number): { g: number; h: number } {
  const z = d / pr.s;
  let g = -d / pr.v0, h = -1 / pr.v0;
  if (pr.wins) { const r = millsInv(z); g += (pr.wins * r) / pr.s; h -= (pr.wins * r * (z + r)) / (pr.s * pr.s); }
  if (pr.losses) { const r = millsInv(-z); g -= (pr.losses * r) / pr.s; h -= (pr.losses * r * (-z + r)) / (pr.s * pr.s); }
  return { g, h };
}

// Laplace: Newton's method to the mode, then the curvature there. Also
// returns the iterates, so a figure can show the steps.
export function laplace(pr: Problem): Gauss1 & { iterates: number[] } {
  let d = 0;
  const iterates = [d];
  const obj = (x: number) => -0.5 * (x * x) / pr.v0 + logLik(pr, x);
  for (let it = 0; it < 100; it++) {
    const { g, h } = derivs(pr, d);
    let step = -g / h;
    let next = d + step;
    while (obj(next) < obj(d) - 1e-12 && Math.abs(step) > 1e-12) { step /= 2; next = d + step; }
    iterates.push(next);
    if (Math.abs(next - d) < 1e-10) { d = next; break; }
    d = next;
  }
  const { h } = derivs(pr, d);
  return { mean: d, sd: Math.sqrt(-1 / h), iterates };
}

// Expectation propagation with one Gaussian site per comparison (Rasmussen
// and Williams 2006, Algorithm 3.5, in one dimension and with noise s).
export function ep(pr: Problem, sweeps = 200): Gauss1 {
  const ys = [...Array(pr.wins).fill(1), ...Array(pr.losses).fill(-1)] as number[];
  const n = ys.length;
  const tau = new Array<number>(n).fill(0), nu = new Array<number>(n).fill(0);
  let P = 1 / pr.v0, Q = 0; // posterior precision and precision-times-mean
  for (let sw = 0; sw < sweeps; sw++) {
    let change = 0;
    for (let i = 0; i < n; i++) {
      const tc = P - tau[i], nc = Q - nu[i];
      const vc = 1 / tc, mc = nc / tc;
      const den = Math.sqrt(pr.s * pr.s + vc);
      const z = (ys[i] * mc) / den;
      const r = millsInv(z);
      const mh = mc + (ys[i] * vc * r) / den;
      const vh = vc - (vc * vc * r * (z + r)) / (den * den);
      const tNew = Math.max(1e-12, 1 / vh - tc);
      const nNew = mh / vh - nc;
      const damp = 0.7;
      const t2 = damp * tNew + (1 - damp) * tau[i];
      const n2 = damp * nNew + (1 - damp) * nu[i];
      change = Math.max(change, Math.abs(t2 - tau[i]), Math.abs(n2 - nu[i]));
      P += t2 - tau[i];
      Q += n2 - nu[i];
      tau[i] = t2;
      nu[i] = n2;
    }
    if (change < 1e-10) break;
  }
  return { mean: Q / P, sd: Math.sqrt(1 / P) };
}

// Gauss-Hermite-like quadrature on a fixed grid of standard normal points.
const QZ = Array.from({ length: 241 }, (_, i) => -6 + (12 * i) / 240);
const QW = (() => { const w = QZ.map(phi); const t = w.reduce((a, b) => a + b, 0); return w.map((v) => v / t); })();

// The evidence lower bound of N(m, sd^2): expected log likelihood minus the
// KL divergence from the prior.
export function elbo(pr: Problem, m: number, sd: number): number {
  let e = 0;
  for (let i = 0; i < QZ.length; i++) e += QW[i] * logLik(pr, m + sd * QZ[i]);
  const kl = 0.5 * ((sd * sd + m * m) / pr.v0 - 1 - Math.log((sd * sd) / pr.v0));
  return e - kl;
}

function golden(f: (x: number) => number, a: number, b: number, iters = 50): number {
  const g = (Math.sqrt(5) - 1) / 2;
  let c = b - g * (b - a), d = a + g * (b - a);
  let fc = f(c), fd = f(d);
  for (let i = 0; i < iters; i++) {
    if (fc > fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = f(c); }
    else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = f(d); }
  }
  return (a + b) / 2;
}

// Gaussian variational inference: maximize the ELBO over the mean and the log
// standard deviation by alternating one-dimensional searches. For a
// log-concave likelihood the objective is concave, so this converges.
export function vi(pr: Problem, start: Gauss1): Gauss1 {
  let m = start.mean, ls = Math.log(Math.max(1e-3, start.sd));
  const span = 4 * Math.sqrt(pr.v0);
  for (let round = 0; round < 30; round++) {
    const m0 = m, l0 = ls;
    m = golden((x) => elbo(pr, x, Math.exp(ls)), m - span, m + span, 40);
    ls = golden((x) => elbo(pr, m, Math.exp(x)), Math.log(1e-4), Math.log(3 * Math.sqrt(pr.v0)), 40);
    if (Math.abs(m - m0) < 1e-7 && Math.abs(ls - l0) < 1e-7) break;
  }
  return { mean: m, sd: Math.exp(ls) };
}

// Elliptical slice sampling (Murray, Adams and MacKay 2010) for the prior
// N(0, v0) and the comparison likelihood. Returns `count` samples after a
// short burn-in.
export function ess(pr: Problem, count: number, seed: number, burn = 50): number[] {
  const r = rng(seed * 2654435761 % 4294967296 + 7);
  const z = normal(r);
  const sd0 = Math.sqrt(pr.v0);
  let d = Math.abs(sd0 * z()) * (pr.wins >= pr.losses ? 1 : -1);
  let ll = logLik(pr, d);
  const out: number[] = [];
  for (let it = 0; it < count + burn; it++) {
    const nu = sd0 * z();
    const logy = ll + Math.log(r() || 1e-300);
    let theta = 2 * Math.PI * r();
    let lo = theta - 2 * Math.PI, hi = theta;
    for (let guard = 0; guard < 200; guard++) {
      const prop = d * Math.cos(theta) + nu * Math.sin(theta);
      const lp = logLik(pr, prop);
      if (lp > logy) { d = prop; ll = lp; break; }
      if (theta < 0) lo = theta; else hi = theta;
      theta = lo + (hi - lo) * r();
    }
    if (it >= burn) out.push(d);
  }
  return out;
}

// Probability that d < 0 under a Gaussian.
export const pNegGauss = (g: Gauss1) => Phi(-g.mean / g.sd);
