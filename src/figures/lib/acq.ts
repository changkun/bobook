// Acquisition functions in closed form, for maximization. Each takes the
// posterior mean and standard deviation at a candidate and the incumbent
// value; they are the formulas derived in the acquisition-functions chapter.

import { Phi, phi } from "./stats.ts";

// Probability of improvement over `best` by at least `xi`.
export function pi(mu: number, sd: number, best: number, xi = 0): number {
  if (sd <= 1e-12) return mu > best + xi ? 1 : 0;
  return Phi((mu - best - xi) / sd);
}

// Expected improvement: E[max(f - best - xi, 0)] = (mu - best - xi) Phi(z) + sd phi(z).
export function ei(mu: number, sd: number, best: number, xi = 0): number {
  const d = mu - best - xi;
  if (sd <= 1e-12) return Math.max(d, 0);
  const z = d / sd;
  return d * Phi(z) + sd * phi(z);
}

// Upper confidence bound mu + beta * sd.
export function ucb(mu: number, sd: number, beta = 2): number {
  return mu + beta * sd;
}

// Expected maximum of two jointly Gaussian values (Clark, 1961):
//   E[max(A, B)] = mu_a Phi(d / s) + mu_b Phi(-d / s) + s phi(d / s),
// with d = mu_a - mu_b and s^2 = var_a + var_b - 2 cov_ab. For a duel between
// two options under a GP posterior on the utility, this is EUBO, the expected
// utility of the better option (Lin et al., 2022; Astudillo et al., 2023).
export function expectedMax2(ma: number, va: number, mb: number, vb: number, cab: number): number {
  const s2 = Math.max(va + vb - 2 * cab, 0);
  const d = ma - mb;
  if (s2 < 1e-14) return Math.max(ma, mb);
  const s = Math.sqrt(s2);
  return ma * Phi(d / s) + mb * Phi(-d / s) + s * phi(d / s);
}

export function argmax(xs: ArrayLike<number>): number {
  let bi = 0;
  for (let i = 1; i < xs.length; i++) if (xs[i] > xs[bi]) bi = i;
  return bi;
}
