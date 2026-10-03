// Acquisition functions that look beyond a single input: the knowledge
// gradient on a discrete set (computed exactly, following Frazier, Powell and
// Dayanik 2009), max-value entropy search from sampled maxima (Wang and
// Jegelka 2017, their Eq. 6), and the samples both Thompson sampling and
// max-value entropy search need. Used by the acquisition-functions chapter's
// figures; the closed-form PI, EI, and UCB stay in acq.ts.

import { Phi, phi, logPhi, millsInv } from "./stats.ts";
import { samplePosterior, type Posterior } from "./gp.ts";
import { argmax } from "./acq.ts";
import type { Rng } from "./random.ts";

// E[max_j (a_j + b_j Z)] for Z ~ N(0, 1), exactly. The maximum of the lines
// a_j + b_j z is piecewise linear in z; sort the lines by slope, keep those on
// the upper envelope with the z-intervals where each is on top, and integrate
// each piece against the normal density:
//   int_c^d (a + b z) phi(z) dz = a (Phi(d) - Phi(c)) + b (phi(c) - phi(d)).
export function expectedMaxLinear(a: number[], b: number[]): number {
  const idx = a.map((_, i) => i).sort((i, j) => b[i] - b[j] || a[i] - a[j]);
  const A: number[] = [], B: number[] = [], C: number[] = []; // envelope lines and their left breakpoints
  for (const i of idx) {
    // Equal slopes: the later one has the larger intercept (sort order), so it replaces.
    if (B.length && Math.abs(b[i] - B[B.length - 1]) < 1e-12) { A.pop(); B.pop(); C.pop(); }
    let c = -Infinity;
    while (A.length) {
      const k = A.length - 1;
      c = (A[k] - a[i]) / (b[i] - B[k]); // where line i overtakes the last envelope line
      if (c <= C[k]) { A.pop(); B.pop(); C.pop(); c = -Infinity; continue; }
      break;
    }
    A.push(a[i]); B.push(b[i]); C.push(A.length === 1 ? -Infinity : c);
  }
  let s = 0;
  for (let k = 0; k < A.length; k++) {
    const lo = C[k], hi = k + 1 < A.length ? C[k + 1] : Infinity;
    const Plo = lo === -Infinity ? 0 : Phi(lo), Phi_hi = hi === Infinity ? 1 : Phi(hi);
    const plo = lo === -Infinity ? 0 : phi(lo), phi_hi = hi === Infinity ? 0 : phi(hi);
    s += A[k] * (Phi_hi - Plo) + B[k] * (plo - phi_hi);
  }
  return s;
}

// The knowledge gradient at every point of a discrete set X, for an
// observation y = f(x) + noise of variance `noiseVar` at x in X. After that
// observation the posterior mean at x' becomes mu(x') + sigmaTilde(x', x) Z,
// with sigmaTilde(x', x) = cov(x', x) / sqrt(var(x) + noiseVar) and Z
// standard normal, so KG(x) = E[max_x' (mu(x') + sigmaTilde(x', x) Z)] - max mu.
export function knowledgeGradient(post: Posterior, noiseVar: number): number[] {
  const cov = post.cov;
  if (!cov) throw new Error("knowledgeGradient needs predict(..., full = true)");
  const m = post.mean;
  const best = Math.max(...m);
  return m.map((_, i) => {
    const s = Math.sqrt(Math.max(cov[i][i], 0) + noiseVar);
    const b = cov.map((row) => row[i] / s);
    return Math.max(0, expectedMaxLinear(m, b) - best);
  });
}

// The fantasized posterior mean after observing y = mu(x) + s Z at x: one
// curve per value of z, for drawing the lookahead.
export function fantasyMean(post: Posterior, i: number, noiseVar: number, z: number): number[] {
  const cov = post.cov!;
  const s = Math.sqrt(Math.max(cov[i][i], 0) + noiseVar);
  return post.mean.map((mu, j) => mu + (cov[j][i] / s) * z);
}

export interface Draws {
  samples: number[][]; // joint posterior samples over the set
  argmax: number[]; // index of each sample's maximum
  max: number[]; // each sample's maximum value
}

export function drawMaxima(post: Posterior, r: Rng, count: number): Draws {
  const samples = samplePosterior(post, r, count);
  const am = samples.map((s) => argmax(s));
  return { samples, argmax: am, max: am.map((k, i) => samples[i][k]) };
}

// Max-value entropy search: the expected reduction in the entropy of f(x)
// from learning the maximum value y*, averaged over sampled maxima. With
// gamma = (y* - mu) / sd, the Gaussian f(x) truncated above at y* has entropy
// lower by gamma phi(gamma) / (2 Phi(gamma)) - log Phi(gamma).
export function mes(mean: number[], sd: number[], ystar: number[]): number[] {
  return mean.map((mu, i) => {
    if (sd[i] < 1e-6) return 0;
    let s = 0;
    for (const y of ystar) {
      const g = (y - mu) / sd[i];
      s += (g * millsInv(g)) / 2 - logPhi(g);
    }
    return Math.max(0, s / ystar.length);
  });
}
