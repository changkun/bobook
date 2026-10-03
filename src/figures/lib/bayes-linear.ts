// Bayesian linear regression for the figures of the Bayesian inference
// chapter: a Gaussian prior N(0, s_p^2 I) on the weights of a linear model
// y = phi(x)^T w + noise, with noise N(0, s_n^2). The posterior over the
// weights is Gaussian with precision A = Phi^T Phi / s_n^2 + I / s_p^2 and
// mean A^{-1} Phi^T y / s_n^2; the log evidence is log N(y; 0, s_p^2 Phi
// Phi^T + s_n^2 I). The models here have at most a few weights, so dense
// inverses of M x M matrices are fine.

import { cholesky, cholSolve, logDetFromChol, matVec, spdInverse, zeros, type Mat, type Vec } from "./linalg.ts";
import { normals, type Rng } from "./random.ts";

export interface BlrPosterior {
  mean: Vec;
  cov: Mat;
  prec: Mat;
  logEvidence: number;
}

export function blrPosterior(Phi: Mat, y: Vec, priorVar: number, noiseVar: number, M: number): BlrPosterior {
  const prec = zeros(M);
  for (let i = 0; i < M; i++) prec[i][i] = 1 / priorVar;
  const b = new Array<number>(M).fill(0);
  Phi.forEach((row, n) => {
    for (let i = 0; i < M; i++) {
      b[i] += (row[i] * y[n]) / noiseVar;
      for (let j = 0; j < M; j++) prec[i][j] += (row[i] * row[j]) / noiseVar;
    }
  });
  const cov = spdInverse(prec);
  const mean = matVec(cov, b);
  // log evidence: y ~ N(0, priorVar Phi Phi^T + noiseVar I)
  let logEvidence = 0;
  const n = y.length;
  if (n) {
    const K = zeros(n);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < M; k++) s += Phi[i][k] * Phi[j][k];
      K[i][j] = priorVar * s + (i === j ? noiseVar : 0);
    }
    const L = cholesky(K);
    const alpha = cholSolve(L, y);
    logEvidence = -0.5 * y.reduce((s, v, i) => s + v * alpha[i], 0) - 0.5 * logDetFromChol(L) - 0.5 * n * Math.log(2 * Math.PI);
  }
  return { mean, cov, prec, logEvidence };
}

// Variance of phi^T w under the posterior.
export function quad(cov: Mat, phi: Vec): number {
  let s = 0;
  for (let i = 0; i < phi.length; i++) for (let j = 0; j < phi.length; j++) s += phi[i] * cov[i][j] * phi[j];
  return Math.max(0, s);
}

// Weight vectors drawn from the posterior, mean + L z.
export function sampleWeights(post: BlrPosterior, r: Rng, k: number): Vec[] {
  const L = cholesky(post.cov);
  const M = post.mean.length;
  const out: Vec[] = [];
  for (let s = 0; s < k; s++) {
    const z = normals(r, M);
    out.push(post.mean.map((m, i) => m + L[i].reduce((acc, v, j) => acc + v * z[j], 0)));
  }
  return out;
}
