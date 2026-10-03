// Bayesian optimization, random selection, and the recorded chemists on the
// measured direct arylation data of Shields et al. (Nature 2021), over many
// seeds: the numbers quoted in the chapter "Optimizing a Chemical Reaction".
// It runs the same optimizer as the figure cs-chem-replay
// (src/figures/lib/cs-chem.ts).
//
//   node tools/figure-data/cs-chem-race.ts [runs=50]
//
// Data: src/figures/data/cs-chem-arylation.json (measured yields, MIT License;
// provenance in tools/figure-data/cs-chem-arylation.py). The "published" rows
// are the paper's own 50 optimizer runs as recorded in its EvML repository;
// the "ours" rows are re-runs of our simplified optimizer.

import { BO_HUMAN, BO_RANDOM, PLAYERS, bestSoFar, bestSoFarYields, boRun, randomPicks, type Encoding, type Init } from "../../src/figures/lib/cs-chem.ts";

const RUNS = Number(process.argv[2] ?? 50);
const CHECK = [5, 10, 15, 20, 25, 30, 50];
// The chapter's table counts runs that reached 99% within the first 50
// experiments, so the recorded curves (the chemists played up to 100) are cut
// there before counting.
const HORIZON = 50;
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
const median = (xs: number[]) => { const s = [...xs].sort((a, b) => a - b); return s.length % 2 ? s[s.length >> 1] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2; };

// `carry`: a run that stopped early keeps its last best value.
function report(name: string, curves: number[][], carry = true) {
  const at = CHECK.map((n) => {
    const vals = carry ? curves.map((c) => c[Math.min(n, c.length) - 1]) : curves.filter((c) => c.length >= n).map((c) => c[n - 1]);
    return mean(vals).toFixed(1);
  });
  const reach = curves.map((c) => c.slice(0, HORIZON).findIndex((v) => v >= 99));
  const hit = reach.filter((k) => k >= 0).map((k) => k + 1);
  console.log(`${name.padEnd(38)} mean best after ${CHECK.join("/")}: ${at.join(" / ")}   reached >= 99% within ${HORIZON}: ${hit.length}/${curves.length}${hit.length ? `, median experiments ${median(hit)}` : ""}`);
  return hit;
}

report("published BO, random start", BO_RANDOM.map(bestSoFarYields));
report("published BO, chemist's start", BO_HUMAN.map(bestSoFarYields));
report("chemists (carried forward)", PLAYERS.map((p) => bestSoFar(p.picks)));
report("chemists (only those still playing)", PLAYERS.map((p) => bestSoFar(p.picks)), false);
report("random selection", Array.from({ length: RUNS }, (_, s) => bestSoFar(randomPicks(50, s + 1))));
const configs: Array<[Encoding, number, Init]> = [["desc", 5, "random"], ["onehot", 5, "random"], ["desc", 1, "random"], ["desc", 10, "random"], ["desc", 5, "chemist"]];
const NAMES = ["ligand", "base", "solvent", "conc.", "temp."];
for (const [enc, b, init] of configs) {
  const t0 = performance.now();
  const runs = Array.from({ length: RUNS }, (_, s) => boRun(enc, b, init, 50, s + 1));
  const hit = report(`ours: ${enc}, batch ${b}, ${init}`, runs.map((r) => bestSoFar(r.picks)));
  const rounds = hit.map((k) => Math.ceil(k / b));
  const k = Math.min(3, runs[0].fits.length - 1);
  const ls = NAMES.map((_, gi) => median(runs.map((r) => r.fits[k].ls[gi])));
  console.log(`   rounds to >= 99%: median ${median(rounds)}; median lengthscales of the model fitted after ${(k + 1) * b} experiments: ${NAMES.map((n, gi) => `${n} ${ls[gi]}`).join(", ")}  (${((performance.now() - t0) / RUNS).toFixed(0)} ms per run)`);
}
