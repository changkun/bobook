// Recorded runs for the dim-bo-race figure.
//
//   node tools/figure-data/dim-bo-race.ts      (a few minutes)
//
// Objective: the six-dimensional Hartmann function embedded in D = 6, 10,
// 20, and 50 dimensions; only the first six inputs matter. All methods share
// each seed's initial Latin hypercube design of 10 points and a budget of 80
// evaluations.
//
// Bayesian optimization uses a GP with a squared-exponential kernel and one
// lengthscale per input (ARD), refitted every few evaluations by maximizing
// the log marginal likelihood plus the log prior, starting from the prior's
// mode, as libraries do. Two priors:
//   fixed    Gamma(2.4, 2.7) on every lengthscale, mode 0.52 in any dimension
//            (BoTorch's PairwiseGP default; fixed-scale priors were the
//            scalar default before BoTorch 0.12.0)
//   scaled   LogNormal(√2 + ln(D)/2, √3), mode 0.205·√D (Hvarfner et al.,
//            2024; BoTorch's default since 0.12.0)
// The optimizer is gradient ascent in log-lengthscale that stops when the
// gradient norm falls below 1e-5, like the gradient tolerance of L-BFGS-B, so
// vanishing gradients stop the fit where they would in a library.
//
// Expected improvement is maximized in one of two ways:
//   global   2,000 uniform random candidates
//   local    1,000 uniform candidates plus 1,000 perturbations of the five
//            best points (each coordinate perturbed with probability
//            min(1, 10/D)), the axis-aligned subspace perturbation that
//            Papenmeier et al. (2025) identify as the source of success
// Outputs are standardized; noise variance 1e-3.

import { writeFileSync } from "node:fs";
import { fit, kernel, predict } from "../../src/figures/lib/gp.ts";
import { argmax, ei } from "../../src/figures/lib/acq.ts";
import { cholesky, cholSolve, zeros } from "../../src/figures/lib/linalg.ts";
import { rng } from "../../src/figures/lib/random.ts";
import { candidates, latinHypercube, uniformPoints, type Pt } from "../../src/figures/lib/nd.ts";
import { embeddedHartmann6, hartmann6 } from "../../src/figures/lib/objectives.ts";

const DIMS = [6, 10, 20, 50];
const SEEDS = 6;
const INIT = 10;
const BUDGET = 80;
const REFIT = 5;
const NOISE = 1e-3;

type Prior = "fixed" | "scaled";

function priorMode(prior: Prior, D: number): number {
  return prior === "fixed" ? (2.4 - 1) / 2.7 : Math.exp(Math.SQRT2 + 0.5 * Math.log(D) - 3);
}

// d/du of log prior, u = log lengthscale (includes the Jacobian of u).
function priorGrad(prior: Prior, D: number, u: number): number {
  if (prior === "fixed") return (2.4 - 1) + 1 - 2.7 * Math.exp(u); // Gamma(a, b) in u: a*u - b*e^u
  const mu = Math.SQRT2 + 0.5 * Math.log(D), s2 = 3;
  return -(u - mu) / s2; // log-normal density in u is normal
}

function fitARD(X: Pt[], y: number[], prior: Prior): number[] {
  const n = X.length, D = X[0].length;
  const d2: Float64Array[] = Array.from({ length: D }, () => new Float64Array(n * n));
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let d = 0; d < D; d++) d2[d][i * n + j] = (X[i][d] - X[j][d]) ** 2;
  let u = new Array(D).fill(Math.log(priorMode(prior, D)));
  for (let it = 0; it < 150; it++) {
    const ls2 = u.map((v) => Math.exp(2 * v));
    const K = zeros(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let s = 0;
      for (let d = 0; d < D; d++) s += d2[d][i * n + j] / ls2[d];
      K[i][j] = Math.exp(-0.5 * s) + (i === j ? NOISE : 0);
    }
    const L = cholesky(K, 1e-8);
    const alpha = cholSolve(L, y);
    // W = alpha alpha^T - K^-1
    const Kinv = zeros(n);
    for (let j = 0; j < n; j++) { const e = new Array(n).fill(0); e[j] = 1; const c = cholSolve(L, e); for (let i = 0; i < n; i++) Kinv[i][j] = c[i]; }
    const g = new Array(D).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const w = (alpha[i] * alpha[j] - Kinv[i][j]) * (K[i][j] - (i === j ? NOISE : 0));
      if (w === 0) continue;
      for (let d = 0; d < D; d++) g[d] += 0.5 * w * d2[d][i * n + j] / ls2[d];
    }
    for (let d = 0; d < D; d++) g[d] += priorGrad(prior, D, u[d]);
    const norm = Math.sqrt(g.reduce((a, b) => a + b * b, 0));
    if (norm < 1e-5) break;
    u = u.map((v, d) => Math.min(Math.log(100), Math.max(Math.log(0.025), v + Math.max(-0.3, Math.min(0.3, 0.05 * g[d])))));
  }
  return u.map(Math.exp);
}

function runBO(D: number, seed: number, prior: Prior, local: boolean): { X: Pt[]; ls: number[] } {
  const r = rng(seed * 1009 + D);
  const X: Pt[] = latinHypercube(rng(seed * 31 + D), INIT, D);
  let ls = new Array(D).fill(priorMode(prior, D));
  while (X.length < BUDGET) {
    const ys = X.map(embeddedHartmann6);
    const m = ys.reduce((a, b) => a + b, 0) / ys.length;
    const sd = Math.sqrt(ys.reduce((a, b) => a + (b - m) ** 2, 0) / ys.length) || 1;
    const yz = ys.map((y) => (y - m) / sd);
    if ((X.length - INIT) % REFIT === 0) ls = fitARD(X, yz, prior);
    const model = fit(kernel("rbf", ls, 1), X, yz, NOISE);
    const order = X.map((_, i) => i).sort((a, b) => ys[b] - ys[a]);
    const cand = local
      ? candidates(r, D, 1000, order.slice(0, 5).map((i) => X[i]), 1000, 0.1, Math.min(1, 10 / D))
      : uniformPoints(r, 2000, D);
    const post = predict(model, cand);
    const best = Math.max(...yz);
    const a = post.mean.map((mu, i) => ei(mu, Math.sqrt(post.var[i]), best, 0));
    X.push(cand[argmax(a)]);
  }
  const ys = X.map(embeddedHartmann6);
  const m = ys.reduce((a, b) => a + b, 0) / ys.length;
  const sd = Math.sqrt(ys.reduce((a, b) => a + (b - m) ** 2, 0) / ys.length) || 1;
  return { X, ls: fitARD(X, ys.map((y) => (y - m) / sd), prior) };
}

function runRandom(D: number, seed: number): Pt[] {
  const X = latinHypercube(rng(seed * 31 + D), INIT, D);
  return X.concat(uniformPoints(rng(seed * 7 + D), BUDGET - INIT, D));
}

const r3 = (v: number) => Math.round(v * 1000) / 1000;
const METHODS = ["random", "fixed-global", "scaled-global", "fixed-local", "scaled-local"] as const;
const out = {
  meta: {
    dims: DIMS, seeds: SEEDS, init: INIT, budget: BUDGET, max: hartmann6.max, methods: METHODS,
    priorMode: Object.fromEntries(DIMS.map((D) => [D, { fixed: r3(priorMode("fixed", D)), scaled: r3(priorMode("scaled", D)) }])),
  },
  runs: {} as Record<number, Record<string, { regret: number[][]; points: number[][]; values: number[]; lengthscales?: number[] }>>,
};
const t0 = Date.now();
for (const D of DIMS) {
  out.runs[D] = {};
  for (const method of METHODS) {
    const regret: number[][] = [];
    let points: number[][] = [], values: number[] = [], lengthscales: number[] | undefined;
    for (let s = 0; s < SEEDS; s++) {
      let X: Pt[], ls: number[] | undefined;
      if (method === "random") X = runRandom(D, s);
      else { const res = runBO(D, s, method.startsWith("fixed") ? "fixed" : "scaled", method.endsWith("local")); X = res.X; ls = res.ls; }
      const ys = X.map(embeddedHartmann6);
      let b = -Infinity;
      regret.push(ys.map((y) => r3(hartmann6.max - (b = Math.max(b, y)))));
      if (s === 0) { points = X.map((p) => p.map((v) => Math.round(v * 100) / 100)); values = ys.map(r3); lengthscales = ls?.map(r3); }
    }
    out.runs[D][method] = { regret, points, values, ...(lengthscales ? { lengthscales } : {}) };
    const finals = regret.map((c) => c[c.length - 1]).sort((a, b) => a - b);
    const med = SEEDS % 2 ? finals[(SEEDS - 1) / 2] : (finals[SEEDS / 2 - 1] + finals[SEEDS / 2]) / 2;
    console.log(`D=${D} ${method}: median final regret ${r3(med)}${lengthscales ? `, seed-0 lengthscales relevant ${lengthscales.slice(0, 6).join(" ")} | irrelevant mean ${r3(lengthscales.slice(6).reduce((a, b) => a + b, 0) / Math.max(1, D - 6))}` : ""} (${Math.round((Date.now() - t0) / 1000)} s)`);
  }
}
writeFileSync("src/figures/data/dim-bo-race.json", JSON.stringify(out));
console.log("wrote src/figures/data/dim-bo-race.json");
