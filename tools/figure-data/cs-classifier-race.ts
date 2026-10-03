// Grid search, random search, and Bayesian optimization on the measured SVM
// landscape, over many seeds: the reliability numbers quoted in the chapter
// "Tuning a Classifier". It runs the same procedures as the figure
// cs-classifier-landscape (src/figures/lib/cs-classifier-race.ts).
//
//   node tools/figure-data/cs-classifier-race.ts [seeds=100]
//
// Data: src/figures/data/cs-classifier-svm.json (measured; see
// tools/figure-data/cs-classifier-svm.py for its provenance and license).

import { BEST, MEAN, NCELL, boRun, gridCells, gridRun, gridShape, randomRun, type Run } from "../../src/figures/lib/cs-classifier-race.ts";

const SEEDS = Number(process.argv[2] ?? 100);
const pct = (v: number) => `${(100 * v).toFixed(2)}%`;
const q = (xs: number[], p: number) => { const s = [...xs].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };

console.log(`best setting: ${pct(BEST)}; settings within 0.1 points of it: ${MEAN.filter((m) => m < BEST + 0.001).length} of ${NCELL}`);
for (const budget of [12, 15, 20, 30]) {
  const [a, b] = gridShape(budget);
  const gc = gridCells(budget);
  console.log(`\nbudget ${budget} (grid ${a} x ${b} = ${a * b}; best grid cell ${pct(Math.min(...gc.map((c) => MEAN[c])))})`);
  for (const [name, f] of [["grid", gridRun], ["random", randomRun], ["bo", boRun]] as Array<[string, (b: number, s: number) => Run]>) {
    const t0 = performance.now();
    const fin: number[] = [];
    const hit: number[] = [];
    for (let s = 1; s <= SEEDS; s++) {
      const r = f(budget, s);
      fin.push(r.rec[r.rec.length - 1]);
      const k = r.rec.findIndex((v) => v < 0.01);
      hit.push(k < 0 ? Infinity : k + 1);
    }
    console.log(`  ${name.padEnd(6)} median ${pct(q(fin, 0.5))}  p10 ${pct(q(fin, 0.1))}  p90 ${pct(q(fin, 0.9))}  P(<1.0%) ${(fin.filter((v) => v < 0.01).length / SEEDS).toFixed(2)}  P(<1.1%) ${(fin.filter((v) => v < 0.011).length / SEEDS).toFixed(2)}  median evals to <1.0% ${q(hit, 0.5)}  (${((performance.now() - t0) / SEEDS).toFixed(0)} ms/run)`);
  }
}
