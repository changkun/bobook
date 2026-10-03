// Eigenvalues and eigenvectors of a small symmetric matrix by cyclic Jacobi
// rotations, for the linear algebra chapter's figures. lib/linalg.ts has the
// eigenvalues only (symEigenvalues) and the 2x2 case (eig2); the 3-D
// covariance figure also needs the eigenvectors, the axes of the ellipsoid.

import type { Mat } from "./linalg.ts";

// Returns eigenvalues in descending order and unit eigenvectors as rows
// (vectors[i] belongs to values[i]).
export function symEig(a: Mat, sweeps = 60): { values: number[]; vectors: number[][] } {
  const n = a.length;
  const A = a.map((r) => r.slice());
  const V: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  for (let s = 0; s < sweeps; s++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
    if (off < 1e-24) break;
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
      for (let k = 0; k < n; k++) {
        const vkp = V[k][p], vkq = V[k][q];
        V[k][p] = c * vkp - sn * vkq;
        V[k][q] = sn * vkp + c * vkq;
      }
    }
  }
  const order = A.map((_, i) => i).sort((i, j) => A[j][j] - A[i][i]);
  return {
    values: order.map((i) => A[i][i]),
    // column i of V is the eigenvector for A[i][i]; fix the sign so the
    // largest component is positive, which keeps the drawing stable
    vectors: order.map((i) => {
      const v = V.map((row) => row[i]);
      let m = 0;
      for (let k = 1; k < n; k++) if (Math.abs(v[k]) > Math.abs(v[m])) m = k;
      return v[m] < 0 ? v.map((x) => -x) : v;
    }),
  };
}

// Cholesky factorization without jitter, reporting where it fails: the
// column whose pivot is not positive. Figures use it to show that the
// factorization exists exactly when the matrix is positive definite.
export function choleskyStrict(a: Mat): { L?: Mat; failedAt?: number; pivot?: number } {
  const n = a.length;
  const L = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let j = 0; j < n; j++) {
    let s = a[j][j];
    for (let k = 0; k < j; k++) s -= L[j][k] * L[j][k];
    if (!(s > 1e-12)) return { failedAt: j, pivot: s };
    L[j][j] = Math.sqrt(s);
    for (let i = j + 1; i < n; i++) {
      let t = a[i][j];
      for (let k = 0; k < j; k++) t -= L[i][k] * L[j][k];
      L[i][j] = t / L[j][j];
    }
  }
  return { L };
}
