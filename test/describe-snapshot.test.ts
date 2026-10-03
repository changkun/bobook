// Chapters quote numbers that their figures show at the default settings
// ("the regret falls to about 0.2"). This test pins each figure's description
// at its defaults, so changing a default, a seed, or the shared math fails
// here and sends the author back to the prose that quotes it. A figure's
// `snapshots` add further named states, such as a finished simulated session.
//
//   UPDATE=1 npm test      rewrite the snapshot after checking the prose

import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { loadFigures } from "../src/figures/registry.ts";
import { defaults } from "../src/figures/lib/params.ts";

const FILE = "test/snapshots/describe.json";
const figures = await loadFigures();
const current: Record<string, string> = {};
for (const [name, fig] of [...figures].sort(([a], [b]) => a.localeCompare(b))) {
  const p = defaults(fig);
  const t = fig.timeline ? fig.timeline.poster(p) : 0;
  current[name] = fig.describe({ p, t, w: 660, uid: "s" }, "en");
  for (const [k, reach] of Object.entries(fig.snapshots ?? {})) {
    const q = reach(p);
    current[`${name} (${k})`] = fig.describe({ p: q, t: fig.timeline ? fig.timeline.poster(q) : 0, w: 660, uid: "s" }, "en");
  }
}

if (process.env.UPDATE || !existsSync(FILE)) writeFileSync(FILE, JSON.stringify(current, null, 2) + "\n");
const saved = JSON.parse(readFileSync(FILE, "utf8")) as Record<string, string>;

for (const name of Object.keys(current)) {
  test(`description of ${name} is unchanged`, () => {
    assert.equal(current[name], saved[name], `${name}: the state changed; check the prose that quotes this figure, then run UPDATE=1 npm test`);
  });
}
