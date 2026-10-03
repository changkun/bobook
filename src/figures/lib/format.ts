// Number formatting for labels and readouts.

// Significant figures, without trailing zeros or exponent noise.
export function sig(v: number, n = 3): string {
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (a >= 1e5 || a < 1e-3) {
    const [m, e] = v.toExponential(n - 1).split("e");
    return `${Number(m)}×10^${Number(e)}`;
  }
  return String(Number(v.toPrecision(n))).replace("-", "−");
}

export function fixed(v: number, d = 2): string {
  return v.toFixed(d).replace(/^-(?=0\.0*$)/, "").replace("-", "−");
}

export function pct(v: number, d = 0): string {
  return `${(v * 100).toFixed(d)}%`;
}

// Tick labels: as few digits as the step needs.
export function tick(v: number, step: number): string {
  const d = Math.max(0, -Math.floor(Math.log10(Math.abs(step) || 1) + 1e-9));
  return fixed(v, Math.min(d, 6));
}

// "{n} observation{n:s}" style templates: {key} substitutes, {key:one/many}
// picks a plural form from the numeric value of key.
export function tpl(s: string, vars: Record<string, string | number>): string {
  return s.replace(/\{(\w+)(?::([^}/]*)\/([^}]*))?\}/g, (_, k, one, many) => {
    const v = vars[k];
    if (one !== undefined) return Number(v) === 1 ? one : many;
    return v === undefined ? `{${k}}` : String(v);
  });
}
