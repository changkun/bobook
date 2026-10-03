// Dense linear algebra for the small systems figures solve: a few dozen
// observations at most. Matrices are arrays of rows. The workhorse is the
// Cholesky factorization, the "square root" of a covariance matrix, which the
// book introduces in the linear algebra chapter and uses everywhere after.

export type Mat = number[][];
export type Vec = number[];

export function zeros(n: number, m = n): Mat {
  return Array.from({ length: n }, () => new Array<number>(m).fill(0));
}

export function eye(n: number): Mat {
  const a = zeros(n);
  for (let i = 0; i < n; i++) a[i][i] = 1;
  return a;
}

export function matVec(a: Mat, v: Vec): Vec {
  return a.map((row) => row.reduce((s, x, j) => s + x * v[j], 0));
}

export function matMul(a: Mat, b: Mat): Mat {
  const n = a.length, m = b[0]?.length ?? 0, k = b.length;
  const out = zeros(n, m);
  for (let i = 0; i < n; i++) for (let p = 0; p < k; p++) {
    const aip = a[i][p];
    if (aip === 0) continue;
    for (let j = 0; j < m; j++) out[i][j] += aip * b[p][j];
  }
  return out;
}

export function transpose(a: Mat): Mat {
  return a[0] ? a[0].map((_, j) => a.map((row) => row[j])) : [];
}

export function dot(a: Vec, b: Vec): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

// Lower-triangular L with L Lᵀ = A. Adds jitter on the diagonal, growing
// tenfold, until the factorization succeeds: covariance matrices of nearby
// points are positive definite in exact arithmetic but not always in floats.
export function cholesky(a: Mat, jitter = 1e-10): Mat {
  const n = a.length;
  let j = 0;
  for (let attempt = 0; attempt < 8; attempt++) {
    const L = zeros(n);
    let ok = true;
    outer: for (let i = 0; i < n; i++) {
      for (let k = 0; k <= i; k++) {
        let s = a[i][k] + (i === k ? j : 0);
        for (let p = 0; p < k; p++) s -= L[i][p] * L[k][p];
        if (i === k) {
          if (s <= 0) { ok = false; break outer; }
          L[i][i] = Math.sqrt(s);
        } else {
          L[i][k] = s / L[k][k];
        }
      }
    }
    if (ok) return L;
    j = j ? j * 10 : jitter;
  }
  throw new Error("cholesky: matrix is not positive definite");
}

// Solve L x = b for lower-triangular L.
export function solveLower(L: Mat, b: Vec): Vec {
  const n = L.length, x = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i][k] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

// Solve Lᵀ x = b for lower-triangular L.
export function solveUpperT(L: Mat, b: Vec): Vec {
  const n = L.length, x = new Array<number>(n);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
    x[i] = s / L[i][i];
  }
  return x;
}

// Solve A x = b given the Cholesky factor of A.
export function cholSolve(L: Mat, b: Vec): Vec {
  return solveUpperT(L, solveLower(L, b));
}

export function logDetFromChol(L: Mat): number {
  let s = 0;
  for (let i = 0; i < L.length; i++) s += Math.log(L[i][i]);
  return 2 * s;
}

// Eigen-decomposition of a symmetric 2x2 matrix: the axes of a covariance
// ellipse. Returns eigenvalues (descending) and unit eigenvectors.
export function eig2(a: number, b: number, d: number): { values: [number, number]; vectors: [[number, number], [number, number]] } {
  // [[a, b], [b, d]]
  const tr = a + d, det = a * d - b * b;
  const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
  let v1: [number, number];
  if (Math.abs(b) > 1e-12) v1 = [l1 - d, b];
  else v1 = a >= d ? [1, 0] : [0, 1];
  const n1 = Math.hypot(v1[0], v1[1]);
  v1 = [v1[0] / n1, v1[1] / n1];
  const v2: [number, number] = [-v1[1], v1[0]];
  return { values: [l1, l2], vectors: [v1, v2] };
}

// Inverse of a symmetric positive-definite matrix via its Cholesky factor.
export function spdInverse(a: Mat): Mat {
  const L = cholesky(a);
  const n = a.length;
  const inv = zeros(n);
  for (let j = 0; j < n; j++) {
    const e = new Array<number>(n).fill(0);
    e[j] = 1;
    const col = cholSolve(L, e);
    for (let i = 0; i < n; i++) inv[i][j] = col[i];
  }
  return inv;
}

// Eigenvalues of a symmetric matrix by cyclic Jacobi rotations; fine for the
// tens of rows figures use. Returned in ascending order.
export function symEigenvalues(a: Mat, sweeps = 60): number[] {
  const n = a.length;
  const A = a.map((r) => r.slice());
  for (let s = 0; s < sweeps; s++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(A[p][q]) < 1e-300) continue;
      const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), sn = t * c;
      for (let k = 0; k < n; k++) {
        const akp = A[k][p], akq = A[k][q];
        A[k][p] = c * akp - sn * akq;
        A[k][q] = sn * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = A[p][k], aqk = A[q][k];
        A[p][k] = c * apk - sn * aqk;
        A[q][k] = sn * apk + c * aqk;
      }
    }
  }
  return A.map((r, i) => r[i]).sort((x, y) => x - y);
}

// LU factorization with partial pivoting, for small non-symmetric systems
// solved with many right-hand sides (the Laplace preference model's
// (I + W K)^-1 W). Factor once, then solve each column in O(n^2).
export interface LU { lu: Mat; piv: number[] }

export function luFactor(a: Mat): LU {
  const n = a.length;
  const lu = a.map((r) => r.slice());
  const piv = Array.from({ length: n }, (_, i) => i);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(lu[r][c]) > Math.abs(lu[p][c])) p = r;
    if (p !== c) { [lu[c], lu[p]] = [lu[p], lu[c]]; [piv[c], piv[p]] = [piv[p], piv[c]]; }
    const d = lu[c][c] || 1e-12;
    for (let r = c + 1; r < n; r++) {
      const f = (lu[r][c] /= d);
      if (f === 0) continue;
      const row = lu[r], top = lu[c];
      for (let j = c + 1; j < n; j++) row[j] -= f * top[j];
    }
  }
  return { lu, piv };
}

export function luSolve({ lu, piv }: LU, b: Vec): Vec {
  const n = lu.length;
  const x = piv.map((i) => b[i]);
  for (let i = 0; i < n; i++) { let s = x[i]; const row = lu[i]; for (let k = 0; k < i; k++) s -= row[k] * x[k]; x[i] = s; }
  for (let i = n - 1; i >= 0; i--) { let s = x[i]; const row = lu[i]; for (let k = i + 1; k < n; k++) s -= row[k] * x[k]; x[i] = s / (row[i] || 1e-12); }
  return x;
}
