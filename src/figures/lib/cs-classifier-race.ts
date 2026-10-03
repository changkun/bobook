// Grid search, random search, and Bayesian optimization on a measured
// landscape: the validation error of an RBF-kernel support vector machine on
// handwritten digits over a 41 x 37 grid of (log2 C, log2 gamma). Shared by
// the figure cs-classifier-landscape and the script
// tools/figure-data/cs-classifier-race.ts, which runs the same procedures over
// many seeds for the numbers quoted in the chapter, so the figure and the
// prose come from one implementation.
//
// Data: src/figures/data/cs-classifier-svm.json, written by
// tools/figure-data/cs-classifier-svm.py (run it with the venv described in
// its header). The images are scikit-learn's copy of the test set of the UCI
// "Optical Recognition of Handwritten Digits" data (Alpaydin and Kaynak 1998,
// doi:10.24432/C50P49, CC BY 4.0). Every value in the file is measured: for
// each setting, the number of misclassified images (of 1,797) under 5-fold
// cross-validation, repeated with five different random splits.
//
// Evaluating a setting returns one of its five measured repeats, so the
// noise a strategy sees is real cross-validation noise, not a simulation.
// The k-th visit to a cell reads repeat (h + k) mod 5 for a hash h of the cell
// and the strategy's stream, so revisiting a cell gives a new measurement.

import svm from "../data/cs-classifier-svm.json" with { type: "json" };
import { fit, kernel, predict, type GPFit } from "./gp.ts";
import { argmax, ei } from "./acq.ts";
import { rng } from "./random.ts";

export const LOG2C: number[] = svm.log2C;
export const LOG2G: number[] = svm.log2G;
export const NC = LOG2C.length;
export const NG = LOG2G.length;
export const NCELL = NC * NG;
export const NIMG: number = svm.n;
export const REPEATS: number = svm.repeats;
const ERR: number[][] = svm.err;

export const cellOf = (i: number, j: number) => i * NG + j;
export const ijOf = (c: number): [number, number] => [Math.floor(c / NG), c % NG];

// The five-split average error of every cell: the best estimate of a
// setting's error that the data allow, used to score recommendations.
export const MEAN: number[] = ERR.map((r) => r.reduce((a, b) => a + b, 0) / r.length / NIMG);
export const BEST_CELL = argmin(MEAN);
export const BEST = MEAN[BEST_CELL];
export const WORST = Math.max(...MEAN);

export function argmin(xs: ArrayLike<number>): number {
  let bi = 0;
  for (let i = 1; i < xs.length; i++) if (xs[i] < xs[bi]) bi = i;
  return bi;
}

function hash(a: number, b: number): number {
  let h = Math.imul(a ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul(b + 0x632be5ab, 0xc2b2ae35);
  h ^= h >>> 15;
  h = Math.imul(h, 0x2c1b3c6d);
  h ^= h >>> 12;
  return h >>> 0;
}

// One evaluation: the measured error of cell c on its next unused split.
export class Evaluator {
  private visits = new Map<number, number>();
  private stream: number;
  constructor(stream: number) { this.stream = stream; }
  eval(c: number): number {
    const k = this.visits.get(c) ?? 0;
    this.visits.set(c, k + 1);
    return ERR[c][(hash(c, this.stream) + k) % REPEATS] / NIMG;
  }
}

export interface Run {
  cells: number[];
  obs: number[]; // the measured value each evaluation returned
  rec: number[]; // after k + 1 evaluations: the five-split error of the cell with the lowest measured value so far
}

export function score(cells: number[], obs: number[]): number[] {
  const rec: number[] = [];
  let b = -1;
  for (let k = 0; k < cells.length; k++) {
    if (b < 0 || obs[k] < obs[b]) b = k;
    rec.push(MEAN[cells[b]]);
  }
  return rec;
}

// The grid shape for a budget: the largest a x b <= budget with a >= b and
// a - b <= 2, more values of C than of gamma as in Hsu, Chang, and Lin's guide.
export function gridShape(budget: number): [number, number] {
  let best: [number, number] = [1, 1];
  for (let b = 1; b * b <= budget; b++) for (let a = b; a <= b + 2; a++) {
    if (a * b <= budget && a * b >= best[0] * best[1]) best = [a, b];
  }
  return best;
}

const spread = (k: number, n: number) => Array.from({ length: k }, (_, t) => (k === 1 ? Math.floor((n - 1) / 2) : Math.round((t * (n - 1)) / (k - 1))));

export function gridCells(budget: number): number[] {
  const [a, b] = gridShape(budget);
  const out: number[] = [];
  for (const i of spread(a, NC)) for (const j of spread(b, NG)) out.push(cellOf(i, j));
  return out;
}

export function gridRun(budget: number, seed: number): Run {
  const ev = new Evaluator(seed * 7 + 1);
  const cells = gridCells(budget);
  const obs = cells.map((c) => ev.eval(c));
  return { cells, obs, rec: score(cells, obs) };
}

export function randomCells(n: number, seed: number): number[] {
  const r = rng(seed * 7919 + 13);
  return Array.from({ length: n }, () => Math.floor(r() * NCELL));
}

export function randomRun(budget: number, seed: number): Run {
  const ev = new Evaluator(seed * 7 + 2);
  const cells = randomCells(budget, seed);
  const obs = cells.map((c) => ev.eval(c));
  return { cells, obs, rec: score(cells, obs) };
}

// Inputs on the unit square, so lengthscales are fractions of each axis.
export const UNIT: number[][] = Array.from({ length: NCELL }, (_, c) => { const [i, j] = ijOf(c); return [i / (NC - 1), j / (NG - 1)]; });

const LS = [0.06, 0.1, 0.16, 0.25, 0.4];
const NOISE = [1e-3, 1e-2, 0.05];
export const N_INIT = 3;

export interface Model {
  gp: GPFit;
  ls: [number, number];
  noise: number;
  mu: number; // mean and sd used to standardize -ln(error)
  sd: number;
}

// A Gaussian process on y = -ln(measured error), standardized, with a Matérn
// 5/2 kernel and one lengthscale per axis. The two lengthscales and the noise
// variance are chosen from a small grid by the log marginal likelihood, a
// discrete version of what libraries do with gradient-based fitting.
export function model(cells: number[], obs: number[]): Model {
  const y = obs.map((e) => -Math.log(e));
  const mu = y.reduce((a, b) => a + b, 0) / y.length;
  const sd = Math.sqrt(y.reduce((a, b) => a + (b - mu) ** 2, 0) / y.length) || 1;
  const z = y.map((v) => (v - mu) / sd);
  const xs = cells.map((c) => UNIT[c]);
  let best: Model | undefined;
  for (const l1 of LS) for (const l2 of LS) for (const nz of NOISE) {
    const gp = fit(kernel("matern52", [l1, l2], 1), xs, z, nz, 0);
    if (!best || gp.logml > best.gp.logml) best = { gp, ls: [l1, l2], noise: nz, mu, sd };
  }
  return best!;
}

// Expected improvement over the best posterior mean at an evaluated cell (the
// usual plug-in for noisy observations), maximized over every cell.
export function nextCell(m: Model): { cell: number; mean: number[]; sdev: number[]; acq: number[] } {
  const post = predict(m.gp, UNIT);
  const sdev = post.var.map(Math.sqrt);
  const inc = Math.max(...predict(m.gp, m.gp.xs).mean);
  const acq = post.mean.map((v, i) => ei(v, sdev[i], inc, 0));
  return { cell: argmax(acq), mean: post.mean, sdev, acq };
}

// The posterior mean of the error itself, for drawing what the model believes.
export function modelError(m: Model, mean: number[]): number[] {
  return mean.map((v) => Math.exp(-(v * m.sd + m.mu)));
}

export function boRun(budget: number, seed: number): Run {
  const ev = new Evaluator(seed * 7 + 3);
  // The first points are random search's first points, so the two differ
  // only in what they do with what they have seen.
  const cells = randomCells(Math.min(N_INIT, budget), seed);
  const obs = cells.map((c) => ev.eval(c));
  while (cells.length < budget) {
    const c = nextCell(model(cells, obs)).cell;
    cells.push(c);
    obs.push(ev.eval(c));
  }
  return { cells, obs, rec: score(cells, obs) };
}

// The reader's own evaluations read a fixed stream, so starting a new random
// run for the strategies never changes what the reader has already seen.
export function readerRun(cells: number[]): Run {
  const ev = new Evaluator(0);
  const obs = cells.map((c) => ev.eval(c));
  return { cells, obs, rec: score(cells, obs) };
}
