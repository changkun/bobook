// Scales map data values to pixels. Each scale knows its domain, its range, an
// inverse (for pointer input), and how to produce readable ticks: steps of 1,
// 2, or 5 times a power of ten for linear scales, decades for log scales.

export interface Scale {
  (v: number): number;
  invert(px: number): number;
  domain: [number, number];
  range: [number, number];
  ticks(count?: number): number[];
  kind: "linear" | "log";
}

export function niceStep(span: number, count: number): number {
  const raw = Math.abs(span) / Math.max(1, count);
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
}

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain, [r0, r1] = range;
  const k = (r1 - r0) / (d1 - d0 || 1);
  const s = ((v: number) => r0 + (v - d0) * k) as Scale;
  s.invert = (px) => d0 + (px - r0) / k;
  s.domain = domain;
  s.range = range;
  s.kind = "linear";
  s.ticks = (count = 5) => {
    const lo = Math.min(d0, d1), hi = Math.max(d0, d1);
    const step = niceStep(hi - lo, count);
    const out: number[] = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : Number(v.toPrecision(12)));
    return out;
  };
  return s;
}

export function log(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain.map(Math.log10) as [number, number];
  const inner = linear([d0, d1], range);
  const s = ((v: number) => inner(Math.log10(v))) as Scale;
  s.invert = (px) => 10 ** inner.invert(px);
  s.domain = domain;
  s.range = range;
  s.kind = "log";
  s.ticks = () => {
    const out: number[] = [];
    for (let e = Math.ceil(Math.min(d0, d1) - 1e-9); e <= Math.max(d0, d1) + 1e-9; e++) out.push(10 ** e);
    return out;
  };
  return s;
}

// Evenly spaced sample points over [a, b], inclusive.
export function grid(a: number, b: number, n: number): number[] {
  if (n <= 1) return [a];
  return Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}
