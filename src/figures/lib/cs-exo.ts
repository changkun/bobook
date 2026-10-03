// A simulated walker for the exoskeleton case study, and four ways to tune
// the device for it. Nothing here is measured data: it is a model calibrated
// to published numbers, named where they are used.
//
// The device is an ankle exoskeleton with two assistance parameters, taken
// from the four-parameter torque profile of Zhang et al. (2017) and Poggensee
// and Collins (2021): the peak torque, 0 to 1 N·m per kilogram of body mass
// (the range in Slade et al., 2022), and the peak time, 35% to 55% of the
// gait cycle (the range reported by Kutulakos and Slade, 2024, for the data
// of Poggensee and Collins). A setting is a point u in [0, 1]^2: u[0] the
// peak torque in N·m per kg, u[1] the peak time mapped from 35% to 55%.
//
// The walker's metabolic rate with a setting, relative to walking in the
// device with zero torque, is 1 - R(u, t). R is the reduction: a smooth bump
// with a person-specific optimum. The bump changes as the person adapts to
// the device over t minutes of assisted walking:
//   - a generic setting gives 10% before training and 31% after, and the
//     person's own optimum 39% after (Poggensee and Collins, 2021);
//   - becoming an expert takes about 109 minutes of assisted walking, and the
//     best peak torque grows more slowly than that (same study);
//   - a two-minute metabolic estimate has noise of standard deviation 4.6% of
//     the zero-torque rate (Kutulakos and Slade, 2024, from Zhang et al.).
//
// What the wearer feels is a different signal: the same landscape plus a
// dislike of high torque that fades with adaptation (naive users preferred
// more torque as the experiment went on: Ingraham et al., 2022), read with
// noise, and available after about 18.7 seconds per setting (Schäfer et al.,
// 2026). The noise and the dislike are assumptions; no study has measured
// them in these units.

import { fit, fitPreference, kernel, pointIndex, predict, predictPreference, type Duel, type X } from "./gp.ts";
import { argmax, expectedMax2 } from "./acq.ts";
import { normal, rng, type Rng } from "./random.ts";

export type U = [number, number];

export const TORQUE = (u: number) => u; // N·m per kg of body mass
export const PEAK_TIME = (u: number) => 35 + 20 * u; // % of the gait cycle

export const NOISE_METABOLIC = 0.046; // sd of a 2-minute estimate, as a fraction of the zero-torque rate
export const MIN_PER_METABOLIC = 2; // minutes of walking per metabolic estimate
export const MIN_PER_FEEL = 18.7 / 60; // minutes per self-tuned setting (Schäfer et al., 2026)
export const MIN_PER_PAIR = 2 * MIN_PER_FEEL; // a comparison needs both settings walked
export const NOISE_FEEL = 0.03; // sd of one felt reading (assumption)
export const JND_FEEL = 0.025; // smaller felt differences read as "about the same" (assumption)
export const GENERIC: U = [0.6, 0.5]; // a generic, one-size-fits-all setting
export const AMP_NOVICE = 0.135; // the novice's best possible reduction (assumption, set so the generic setting gives 10%)
export const TAU_ADAPT = 36; // minutes: 95% adapted after about 109 minutes
export const TAU_TORQUE = 72; // the best torque adapts half as fast

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

export interface Walker {
  seed: number;
  novice: boolean; // false: already adapted, the landscape does not change
  optimum(t: number): U; // the metabolic optimum after t minutes of assisted walking
  reduction(u: U, t: number): number; // true metabolic reduction against zero torque
  felt(u: U, t: number): number; // perceived effort (lower is better), noise-free
  best(t: number): number; // reduction at the optimum
}

export function walker(seed: number, novice: boolean): Walker {
  const r = rng(seed * 9973 + 41);
  // The adapted person's optimum: high torque, a person-specific peak time.
  const torqueE = 0.78 + 0.16 * r();
  const timeE = 0.12 + 0.8 * r();
  // The novice's optimum: much less torque, a somewhat different timing.
  const torqueN = torqueE - 0.38 - 0.08 * r();
  const timeN = clamp01(timeE + (r() < 0.5 ? -1 : 1) * (0.12 + 0.12 * r()));
  const ampE = 0.39, ampN = AMP_NOVICE;
  const level = (t: number, tau: number) => (novice ? 1 - Math.exp(-t / tau) : 1);
  const optimum = (t: number): U => {
    const a = level(t, TAU_ADAPT), aT = level(t, TAU_TORQUE);
    return [torqueN + (torqueE - torqueN) * aT, timeN + (timeE - timeN) * a];
  };
  const reduction = (u: U, t: number) => {
    const a = level(t, TAU_ADAPT);
    const amp = ampN + (ampE - ampN) * a;
    const [T, tm] = optimum(t);
    // torque: zero benefit at zero torque, a maximum at T, a loss beyond it
    const s = u[0] <= T ? 1 - ((u[0] - T) / T) ** 2 : 1 - ((u[0] - T) / 0.5) ** 2;
    const h = Math.exp(-((u[1] - tm) ** 2) / (2 * 0.45 * 0.45));
    return amp * s * h;
  };
  const felt = (u: U, t: number) => {
    const aT = level(t, TAU_TORQUE);
    return 1 - reduction(u, t) + 0.1 * (1 - aT) * u[0] * u[0];
  };
  return { seed, novice, optimum, reduction, felt, best: (t) => reduction(optimum(t), t) };
}

// ---------------------------------------------------------------------------
// The grid of settings used for maps, acquisition, and recommendations.

export const G = 21;
export const GRID: U[] = [];
for (let j = 0; j < G; j++) for (let i = 0; i < G; i++) GRID.push([i / (G - 1), j / (G - 1)]);

export interface Eval { u: U; t: number; y?: number } // setting, minute at which it finished, measurement
export interface Run { evals: Eval[]; rec: U; recs: Array<{ t: number; u: U }> } // rec: final recommendation

// ---------------------------------------------------------------------------
// Bayesian optimization on metabolic estimates (Ding et al., 2018; Kutulakos
// and Slade, 2024): a Gaussian process on the measured rate, a lower
// confidence bound with exploration constant beta, one estimate every two
// minutes. With `timeScale`, time is a third input, so old measurements count
// less for predictions about now.

const K2 = kernel("matern52", [0.35, 0.35], 0.15 * 0.15);

export function runBO(w: Walker, budget: number, seed: number, beta = 2.6, timeScale = 0): Run {
  const r = normal(rng(seed * 733 + 5));
  const k = timeScale > 0 ? kernel("matern52", [0.35, 0.35, 1], 0.15 * 0.15) : K2;
  const X = (u: U, t: number): X => (timeScale > 0 ? [u[0], u[1], t / timeScale] : [u[0], u[1]]);
  const n = Math.floor(budget / MIN_PER_METABOLIC + 1e-9);
  const init: U[] = [[0.5, 0.5], [0.2, 0.2], [0.8, 0.8], [0.2, 0.8]]; // a small spread before the model takes over
  const evals: Eval[] = [];
  const recs: Array<{ t: number; u: U }> = [];
  let rec: U = [0.5, 0.5];
  for (let i = 0; i < n; i++) {
    const tNow = (i + 1) * MIN_PER_METABOLIC;
    let u: U;
    if (i < init.length) u = init[i];
    else {
      const g = fit(k, evals.map((e) => X(e.u, e.t)), evals.map((e) => e.y!), NOISE_METABOLIC ** 2, 0.85);
      const post = predict(g, GRID.map((p) => X(p, tNow)));
      // minimize the rate: the lower confidence bound mean - beta * sd
      u = GRID[argmax(post.mean.map((m, j) => -(m - beta * Math.sqrt(post.var[j]))))];
    }
    evals.push({ u, t: tNow, y: 1 - w.reduction(u, tNow) + NOISE_METABOLIC * r() });
    const g = fit(k, evals.map((e) => X(e.u, e.t)), evals.map((e) => e.y!), NOISE_METABOLIC ** 2, 0.85);
    const post = predict(g, GRID.map((p) => X(p, tNow)));
    rec = GRID[argmax(post.mean.map((m) => -m))];
    recs.push({ t: tNow, u: rec });
  }
  return { evals, rec, recs };
}

// ---------------------------------------------------------------------------
// A covariance matrix adaptation evolution strategy (CMA-ES), the optimizer
// of Zhang et al. (2017): 4 + floor(3 ln N) = 6 settings per generation in two
// dimensions, the mean starting at the center of the range with a step size of
// 30% of it (the settings reported by Kutulakos and Slade, 2024). This is a
// compact (mu/mu_w, lambda) implementation with rank-mu and rank-one updates
// and cumulative step-size adaptation.

export function runCMA(w: Walker, budget: number, seed: number): Run {
  const r = normal(rng(seed * 577 + 9));
  const N = 2, lambda = 6, mu = 3;
  const wRaw = Array.from({ length: mu }, (_, i) => Math.log(mu + 0.5) - Math.log(i + 1));
  const wSum = wRaw.reduce((a, b) => a + b, 0);
  const wt = wRaw.map((v) => v / wSum);
  const muEff = 1 / wt.reduce((a, b) => a + b * b, 0);
  const cs = (muEff + 2) / (N + muEff + 5);
  const ds = 1 + 2 * Math.max(0, Math.sqrt((muEff - 1) / (N + 1)) - 1) + cs;
  const cc = (4 + muEff / N) / (N + 4 + (2 * muEff) / N);
  const c1 = 2 / ((N + 1.3) ** 2 + muEff);
  const cmu = Math.min(1 - c1, (2 * (muEff - 2 + 1 / muEff)) / ((N + 2) ** 2 + muEff));
  const chiN = Math.sqrt(N) * (1 - 1 / (4 * N) + 1 / (21 * N * N));
  let mean: U = [0.5, 0.5];
  let sigma = 0.3;
  let C = [[1, 0], [0, 1]];
  let ps = [0, 0], pc = [0, 0];
  const n = Math.floor(budget / MIN_PER_METABOLIC + 1e-9);
  const evals: Eval[] = [];
  const recs: Array<{ t: number; u: U }> = [];
  let pop: Array<{ z: number[]; yv: number[]; u: U; y: number }> = [];
  for (let i = 0; i < n; i++) {
    const tNow = (i + 1) * MIN_PER_METABOLIC;
    // sample from N(mean, sigma^2 C) with the 2 by 2 Cholesky factor of C
    const l11 = Math.sqrt(C[0][0]), l21 = C[1][0] / l11, l22 = Math.sqrt(Math.max(1e-12, C[1][1] - l21 * l21));
    const z = [r(), r()];
    const yv = [l11 * z[0], l21 * z[0] + l22 * z[1]];
    const u: U = [clamp01(mean[0] + sigma * yv[0]), clamp01(mean[1] + sigma * yv[1])];
    const y = 1 - w.reduction(u, tNow) + NOISE_METABOLIC * r();
    evals.push({ u, t: tNow, y });
    pop.push({ z, yv, u, y });
    if (pop.length === lambda) {
      pop.sort((a, b) => a.y - b.y);
      const yw = [0, 0], zw = [0, 0];
      for (let k = 0; k < mu; k++) for (let d = 0; d < N; d++) { yw[d] += wt[k] * pop[k].yv[d]; zw[d] += wt[k] * pop[k].z[d]; }
      mean = [clamp01(mean[0] + sigma * yw[0]), clamp01(mean[1] + sigma * yw[1])];
      ps = ps.map((v, d) => (1 - cs) * v + Math.sqrt(cs * (2 - cs) * muEff) * zw[d]);
      const psNorm = Math.hypot(ps[0], ps[1]);
      pc = pc.map((v, d) => (1 - cc) * v + Math.sqrt(cc * (2 - cc) * muEff) * yw[d]);
      const Cn = [[0, 0], [0, 0]];
      for (let a = 0; a < N; a++) for (let b = 0; b < N; b++) {
        let rankMu = 0;
        for (let k = 0; k < mu; k++) rankMu += wt[k] * pop[k].yv[a] * pop[k].yv[b];
        Cn[a][b] = (1 - c1 - cmu) * C[a][b] + c1 * pc[a] * pc[b] + cmu * rankMu;
      }
      C = Cn;
      sigma = Math.min(0.5, sigma * Math.exp((cs / ds) * (psNorm / chiN - 1)));
      pop = [];
    }
    recs.push({ t: tNow, u: mean });
  }
  return { evals, rec: mean, recs };
}

// ---------------------------------------------------------------------------
// Preferential Bayesian optimization on what the wearer feels: pairs chosen
// by EUBO over a coarse grid, answered by comparing the felt effort of the
// two settings (noise on each), one comparison per 37 seconds.

const KP = kernel("matern52", [0.35, 0.35], 1);
const PGRID: U[] = [];
for (let j = 0; j < 11; j++) for (let i = 0; i < 11; i++) PGRID.push([i / 10, j / 10]);

export function runPBO(w: Walker, budget: number, seed: number): Run {
  const r = normal(rng(seed * 311 + 3));
  const n = Math.floor(budget / MIN_PER_PAIR + 1e-9);
  const xs: X[] = [];
  const duels: Duel[] = [];
  const evals: Eval[] = [];
  const recs: Array<{ t: number; u: U }> = [];
  let rec: U = [0.5, 0.5];
  for (let i = 0; i < n; i++) {
    const tNow = (i + 1) * MIN_PER_PAIR;
    let a: U, b: U;
    if (i === 0) { a = [0.3, 0.3]; b = [0.7, 0.7]; }
    else {
      const f = fitPreference(KP, xs, duels, 0.3);
      const post = predictPreference(f, PGRID, true);
      let bi = 0, bj = 1, bv = -Infinity;
      for (let p = 0; p < PGRID.length; p++) for (let q = p + 1; q < PGRID.length; q++) {
        const v = expectedMax2(post.mean[p], post.cov![p][p], post.mean[q], post.cov![q][q], post.cov![p][q]);
        if (v > bv) { bv = v; bi = p; bj = q; }
      }
      a = PGRID[bi]; b = PGRID[bj];
    }
    const fa = w.felt(a, tNow) + NOISE_FEEL * r(), fb = w.felt(b, tNow) + NOISE_FEEL * r();
    const [win, lose] = fa < fb ? [a, b] : [b, a];
    duels.push({ winner: pointIndex(xs, win, 1e-6), loser: pointIndex(xs, lose, 1e-6) });
    evals.push({ u: a, t: tNow }, { u: b, t: tNow });
    const f = fitPreference(KP, xs, duels, 0.3);
    const at = predictPreference(f, xs);
    rec = xs[argmax(at.mean)] as U;
    recs.push({ t: tNow, u: rec });
  }
  return { evals, rec, recs };
}

// ---------------------------------------------------------------------------
// What the wearer feels when they switch from one setting to another: easier,
// harder, or about the same. Used for the reader's own tuning (the answer is
// a pure function of the walker, the two settings, the time, and the step
// number) and for the simulated self-tuner below.

export type Feel = "easier" | "harder" | "same";

export function feel(w: Walker, prev: U, next: U, t: number, step: number): Feel {
  const r = normal(rng(w.seed * 1543 + step * 67 + 13));
  const d = w.felt(prev, t) + NOISE_FEEL * r() - (w.felt(next, t) + NOISE_FEEL * r());
  return d > JND_FEEL ? "easier" : d < -JND_FEEL ? "harder" : "same";
}

// A simulated self-tuner, for the comparison across many walkers and for
// readers who would rather watch. It starts from the generic setting and
// changes one parameter at a time (as 97.5% of the adjustments in Schäfer et
// al., 2026 did). Each tried setting is felt against the one before it, as in
// the reader's own session. A change that feels easier is kept; one that
// feels harder or about the same is undone, which costs another setting.
// After two such failures the tuner turns to the other parameter, and after
// four it halves its step. It is a model of a careful person, not data.
export function runSelf(w: Walker, budget: number, _seed = 0): Run {
  const n = Math.floor(budget / MIN_PER_FEEL + 1e-9);
  let cur: U = [...GENERIC];
  const evals: Eval[] = [];
  const recs: Array<{ t: number; u: U }> = [];
  const push = (u: U) => { evals.push({ u, t: evals.length * MIN_PER_FEEL + MIN_PER_FEEL }); recs.push({ t: evals.length * MIN_PER_FEEL, u: cur }); };
  if (n > 0) push(cur);
  let dim = 0, dir = 1, step = 0.2, fails = 0;
  const fail = () => {
    fails++;
    dir = -dir;
    if (fails % 2 === 0) { dim = 1 - dim; if (fails % 4 === 0) step = Math.max(0.05, step / 2); }
  };
  while (evals.length < n) {
    const next: U = [...cur];
    next[dim] = Math.round(clamp01(next[dim] + dir * step) * 100) / 100;
    if (next[dim] === cur[dim]) { fail(); if (fails > 400) break; continue; }
    const i = evals.length;
    const f = feel(w, cur, next, (i + 1) * MIN_PER_FEEL, i);
    if (f === "easier") {
      cur = next;
      push(next);
      fails = 0;
    } else {
      push(next); // tried it,
      if (evals.length < n) push(cur); // and went back
      fail();
    }
  }
  return { evals, rec: cur, recs };
}

// The recommendation in force at minute t of a run (before the first
// recommendation: the generic setting).
export function recAt(run: Run, t: number): U {
  let u: U = GENERIC;
  for (const q of run.recs) { if (q.t <= t + 1e-9) u = q.u; else break; }
  return u;
}

export type { Rng };
