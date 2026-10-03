// Objective functions the book's figures optimize. One running example keeps
// its shape from chapter to chapter, so a reader recognizes the landscape when
// the method around it changes: two bumps, the taller one narrower and to the
// right, so a greedy search that finds the wide bump first gets stuck there.

export interface Objective1D {
  name: string;
  f(x: number): number; // maximize on [0, 1]
  argmax: number;
  max: number;
  range: [number, number]; // a y-range that frames the function with room for bands
}

function finish(name: string, f: (x: number) => number, range: [number, number]): Objective1D {
  let argmax = 0, max = -Infinity;
  for (let i = 0; i <= 4000; i++) {
    const x = i / 4000, v = f(x);
    if (v > max) { max = v; argmax = x; }
  }
  return { name, f, argmax, max, range };
}

// The running example: a wide local bump near x = 0.25 and the global bump
// near x = 0.73, on a gently sloping, slightly wavy base.
export const running = finish(
  "running",
  (x) => 0.62 * Math.exp(-((x - 0.25) ** 2) / (2 * 0.1 ** 2))
    + 1.0 * Math.exp(-((x - 0.73) ** 2) / (2 * 0.055 ** 2))
    + 0.1 * Math.sin(11 * x + 0.6)
    - 0.35 * x,
  [-1.2, 1.6],
);

// Forrester et al. (2008), negated so that larger is better; a standard 1-D
// test function with a deceptive local optimum.
export const forrester = finish(
  "forrester",
  (x) => -((6 * x - 2) ** 2 * Math.sin(12 * x - 4)) / 10,
  [-1.8, 1.2],
);

// A smooth, single-peaked utility, for preference examples where the point is
// the comparison model rather than the search.
export const gentle = finish(
  "gentle",
  (x) => Math.exp(-((x - 0.62) ** 2) / (2 * 0.18 ** 2)) - 0.5,
  [-1.4, 1.4],
);

export const OBJECTIVES: Record<string, Objective1D> = { running, forrester, gentle };

// Two-dimensional Branin, rescaled to [0, 1]^2 and negated and normalized so
// that larger is better and values lie roughly in [-1, 1]. Three global optima.
export function branin01(u: number, v: number): number {
  const x = 15 * u - 5, y = 15 * v;
  const a = 1, b = 5.1 / (4 * Math.PI ** 2), c = 5 / Math.PI, r = 6, s = 10, t = 1 / (8 * Math.PI);
  const f = a * (y - b * x * x + c * x - r) ** 2 + s * (1 - t) * Math.cos(x) + s;
  return -(f - 54) / 54;
}

// ---------------------------------------------------------------------------
// Test functions in more dimensions, on the unit cube, negated so that larger
// is better. Hartmann functions are the standard 3-D and 6-D benchmarks of
// the Bayesian optimization literature (Dixon and Szegő, 1978).

const H_ALPHA = [1.0, 1.2, 3.0, 3.2];
const H3_A = [[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]];
const H3_P = [[0.3689, 0.117, 0.2673], [0.4699, 0.4387, 0.747], [0.1091, 0.8732, 0.5547], [0.03815, 0.5743, 0.8828]];
const H6_A = [[10, 3, 17, 3.5, 1.7, 8], [0.05, 10, 17, 0.1, 8, 14], [3, 3.5, 1.7, 10, 17, 8], [17, 8, 0.05, 10, 0.1, 14]];
const H6_P = [
  [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886], [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
  [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.665], [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
];

function hartmann(x: number[], A: number[][], P: number[][]): number {
  let s = 0;
  for (let i = 0; i < 4; i++) {
    let e = 0;
    for (let j = 0; j < x.length; j++) e += A[i][j] * (x[j] - P[i][j]) ** 2;
    s += H_ALPHA[i] * Math.exp(-e);
  }
  return s; // the negated Hartmann function: maximum 3.86278 (3-D), 3.32237 (6-D)
}

export const hartmann3 = {
  name: "hartmann3", dim: 3,
  f: (x: number[]) => hartmann(x, H3_A, H3_P),
  max: 3.86278, argmax: [0.114614, 0.555649, 0.852547],
};

export const hartmann6 = {
  name: "hartmann6", dim: 6,
  f: (x: number[]) => hartmann(x, H6_A, H6_P),
  max: 3.32237, argmax: [0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573],
};

// Hartmann-6 hidden in D >= 6 dimensions: only the first six inputs matter,
// the rest are irrelevant. Real tuning problems often look like this: many
// knobs, few that matter, and nobody tells you which.
export function embeddedHartmann6(x: number[]): number {
  return hartmann6.f(x.slice(0, 6));
}
