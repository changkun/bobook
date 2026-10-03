// The standard normal and friends. Phi and phi appear in every closed-form
// acquisition function and in the probit preference likelihood.

const SQRT2 = Math.SQRT2;
const INV_SQRT_2PI = 1 / Math.sqrt(2 * Math.PI);

// Density of the standard normal.
export function phi(z: number): number {
  return INV_SQRT_2PI * Math.exp(-0.5 * z * z);
}

// Density of N(mu, s^2) at x.
export function normalPdf(x: number, mu: number, s: number): number {
  return phi((x - mu) / s) / s;
}

// erf with a maximum absolute error near 1.2e-7 (Numerical Recipes erfc).
export function erf(x: number): number {
  const t = 1 / (1 + 0.5 * Math.abs(x));
  const y = t * Math.exp(-x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
    + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587
    + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? 1 - y : y - 1;
}

// Cumulative distribution of the standard normal.
export function Phi(z: number): number {
  return 0.5 * (1 + erf(z / SQRT2));
}

// log Phi, accurate far into the lower tail where Phi underflows. Used by the
// probit likelihood, whose log is summed over many comparisons.
export function logPhi(z: number): number {
  if (z > -5) return Math.log(Phi(z));
  // Asymptotic expansion of the Mills ratio.
  const z2 = z * z;
  return -0.5 * z2 - Math.log(-z) - 0.5 * Math.log(2 * Math.PI) + Math.log(1 - 1 / z2 + 3 / (z2 * z2));
}

// phi(z) / Phi(z), the inverse Mills ratio, stable for very negative z.
export function millsInv(z: number): number {
  if (z > -5) return phi(z) / Phi(z);
  const z2 = z * z;
  return -z / (1 - 1 / z2 + 3 / (z2 * z2));
}

// Inverse of Phi (Acklam's rational approximation, relative error < 1.2e-9).
export function PhiInv(p: number): number {
  if (p <= 0) return -Infinity;
  if (p >= 1) return Infinity;
  const a = [-3.969683028665376e1, 2.209460984245205e2, -2.759285104469687e2, 1.38357751867269e2, -3.066479806614716e1, 2.506628277459239];
  const b = [-5.447609879822406e1, 1.615858368580409e2, -1.556989798598866e2, 6.680131188771972e1, -1.328068155288572e1];
  const c = [-7.784894002430293e-3, -3.223964580411365e-1, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425, hi = 1 - lo;
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > hi) {
    const q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  const q = p - 0.5, r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

// The logistic sigmoid, the Bradley-Terry link.
export function sigmoid(z: number): number {
  return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
}

export function mean(xs: number[]): number {
  return xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
}

export function variance(xs: number[]): number {
  const m = mean(xs);
  return xs.reduce((a, b) => a + (b - m) ** 2, 0) / Math.max(1, xs.length - 1);
}

// Beta function via log-gamma, for the Beta density in the coin examples.
export function logGamma(z: number): number {
  // Lanczos approximation, g = 7.
  const g = 7;
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (z < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  z -= 1;
  let x = c[0];
  for (let i = 1; i < g + 2; i++) x += c[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

export function betaPdf(x: number, a: number, b: number): number {
  if (x <= 0 || x >= 1) return 0;
  const lb = logGamma(a) + logGamma(b) - logGamma(a + b);
  return Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) - lb);
}
