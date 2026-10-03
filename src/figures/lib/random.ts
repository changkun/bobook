// Seeded randomness. Render functions are pure, so every random draw comes
// from a generator built from a seed parameter: the same seed gives the same
// picture at build time, in the browser, and in a screenshot.

export type Rng = () => number; // uniform on [0, 1)

// mulberry32: small, fast, and good enough for figures.
export function rng(seed: number): Rng {
  let a = (Math.floor(seed) >>> 0) || 0x9e3779b9;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Standard normal draws by Box-Muller, cached in pairs.
export function normal(r: Rng): () => number {
  let spare: number | undefined;
  return () => {
    if (spare !== undefined) { const s = spare; spare = undefined; return s; }
    let u = 0, v = 0;
    while (u <= 1e-12) u = r();
    v = r();
    const m = Math.sqrt(-2 * Math.log(u));
    spare = m * Math.sin(2 * Math.PI * v);
    return m * Math.cos(2 * Math.PI * v);
  };
}

export function normals(r: Rng, n: number): number[] {
  const z = normal(r);
  return Array.from({ length: n }, () => z());
}

export function intBetween(r: Rng, lo: number, hi: number): number {
  return lo + Math.floor(r() * (hi - lo + 1));
}

// Memoize a pure function of a parameter tuple: simulations that render many
// frames of the same run compute the run once.
export function memo<A extends unknown[], R>(f: (...a: A) => R, size = 32): (...a: A) => R {
  const cache = new Map<string, R>();
  return (...a: A) => {
    const k = JSON.stringify(a);
    const hit = cache.get(k);
    if (hit !== undefined) return hit;
    const v = f(...a);
    cache.set(k, v);
    if (cache.size > size) cache.delete(cache.keys().next().value as string);
    return v;
  };
}
