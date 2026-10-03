// Reference curves for the dim-oracle-nd figure.
//
//   node tools/figure-data/dim-oracle-nd.ts
//
// For D = 3, 6, and 10 design parameters, a simulated person with a hidden
// separable utility (lib/oracle-nd.ts) answers 40 comparisons with probit
// noise 0.1; pairs are chosen by EUBO or at random. For each comparison
// count the file stores the quartiles, over 24 simulated people (seeds), of
// the normalized regret of the model's best guess: 1 is no better than a
// random design, 0 is the person's favorite.

import { writeFileSync } from "node:fs";
import { normRegret, simulatedAnswer, step } from "../../src/figures/lib/oracle-nd.ts";

const DIMS = [3, 6, 10], SEEDS = 24, N = 40, NOISE = 0.1;
const out: Record<string, Record<string, number[][]>> = {};
const t0 = Date.now();
for (const D of DIMS) {
  out[D] = {};
  for (const acq of ["eubo", "random"] as const) {
    const curves: number[][] = [];
    for (let s = 1; s <= SEEDS; s++) {
      const pairs: Array<[number[], number[]]> = [];
      const curve: number[] = [];
      for (let k = 0; k <= N; k++) {
        const st = step(D, pairs, s, acq);
        curve.push(normRegret(D, s, st.best));
        if (k < N) pairs.push(simulatedAnswer(D, s, NOISE, st.pair, k));
      }
      curves.push(curve);
    }
    const q = (v: number[], f: number) => { const s = v.slice().sort((a, b) => a - b); const i = f * (s.length - 1); const lo = Math.floor(i); return s[lo] + (s[Math.ceil(i)] - s[lo]) * (i - lo); };
    out[D][acq] = curves[0].map((_, k) => [0.25, 0.5, 0.75].map((f) => Math.round(q(curves.map((c) => c[k]), f) * 1000) / 1000));
    console.log(`D=${D} ${acq}: median after 10/20/40 → ${[10, 20, 40].map((k) => out[D][acq][k][1]).join(" ")} (${Math.round((Date.now() - t0) / 1000)} s)`);
  }
}
writeFileSync("src/figures/data/dim-oracle-nd.json", JSON.stringify({ meta: { dims: DIMS, seeds: SEEDS, comparisons: N, noise: NOISE }, quartiles: out }));
console.log("wrote src/figures/data/dim-oracle-nd.json");
