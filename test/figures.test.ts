// Generic contract tests for every figure module. They do not assert what a
// figure draws; they assert that it can be drawn: every module renders its
// opening state and its last timeline position at both static widths, returns
// an SVG with a viewBox and no fixed width, describes itself in a sentence,
// and keeps em dashes out of every label.

import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import { parseBib, people } from "../src/pipeline/bib.ts";
import { loadFigures } from "../src/figures/registry.ts";
import { defaults } from "../src/figures/lib/params.ts";
import { STATIC_WIDTHS } from "../src/figures/static.ts";
import type { AnyFigure } from "../src/figures/types.ts";

const errors: string[] = [];
const figures = await loadFigures(errors);

test("every figure module loads", () => {
  assert.deepEqual(errors, []);
  assert.ok(figures.size > 0);
});

function texts(fig: AnyFigure): string[] {
  const out: string[] = [fig.title.en];
  for (const v of Object.values(fig.labels.en)) out.push(String(v));
  for (const spec of Object.values(fig.params) as Array<{ label: { en: string }; options?: Array<{ label: { en: string } }> }>) {
    out.push(spec.label.en);
    for (const o of spec.options ?? []) out.push(o.label.en);
  }
  for (const a of fig.actions ?? []) out.push(a.label.en);
  if (fig.hint) out.push(fig.hint.en);
  return out;
}

for (const [name, fig] of figures) {
  test(`figure ${name}`, () => {
    const p = defaults(fig);
    const times = fig.timeline ? [0, fig.timeline.poster(p), fig.timeline.duration(p)] : [0];
    for (const w of [STATIC_WIDTHS.wide, STATIC_WIDTHS.narrow]) {
      for (const t of times) {
        const out = fig.render({ p, t, w, uid: "t" }, "en");
        assert.match(out, /^<svg class="fig-svg" viewBox="0 0 [\d.]+ [\d.]+"/, `${name} at w=${w}, t=${t} must start with an svg root with a viewBox`);
        assert.doesNotMatch(out.slice(0, 200), /\swidth="/, `${name}: the root svg must not have a fixed width`);
        assert.doesNotMatch(out, /NaN|undefined|Infinity/, `${name} at w=${w}, t=${t} renders NaN, undefined, or Infinity`);
        const vb = /viewBox="0 0 ([\d.]+) ([\d.]+)"/.exec(out)!;
        assert.ok(Number(vb[1]) <= w + 1, `${name}: viewBox width ${vb[1]} exceeds layout width ${w}`);
      }
      const d = fig.describe({ p, t: times[1] ?? 0, w, uid: "t" }, "en");
      assert.ok(d.length > 20 && /[.]$/.test(d.trim()), `${name}: describe() should be a sentence, got "${d}"`);
    }
    for (const s of texts(fig)) assert.ok(!s.includes("—"), `${name}: em dash in label "${s}"`);
  });
}

// The Chinese edition. A module that has Chinese labels must have all of them
// (the same keys as the English, and a Chinese form of every title, control,
// option, action, and hint), must render and describe itself in Chinese, and
// must not leave English words in the drawing beyond names that stay in
// English (GLOSSARY.zh.md). Modules without Chinese labels are skipped until
// the translation reaches them.
const KEEP = /^(Phos|fur|EUBO|qEUBO|EI|PI|UCB|KG|MES|PES|ES|TS|BO|PBO|GP|RBF|ARD|SE|EP|VI|MCMC|DPO|RLHF|LLM|CMA|ES|BoTorch|PairwiseGP|GPyTorch|optuna|dashboard|Thompson|Laplace|Gaussian|Matérn|Hartmann|Branin|Condorcet|Copeland|Borda|Kendall|Spearman|Pearson|Bradley|Terry|Thurstone|Plackett|Luce|Elo|Cholesky|Hessian|Bernoulli|Beta|Dirichlet|Gumbel|Weber|Fechner|Stevens|Hodge|Fiedler|Kemeny|Arrow|Pareto|Shannon|Kullback|Leibler|Fisher|Bayes|Monte|Carlo|Markov|Newton|Hodges|log|exp|max|min|sin|cos|argmax|argmin|sigmoid|probit|logit|softmax|det|tr|var|std|mean|sd|nats|bits|bit|dB|Hz|ms|min|kg|cm|mm|px|JND|DDM|WTA|WTP|CI|SD|SE|AUC|ROC|RMSE|MSE|MAE|PDF|CDF|OK|vs|et|al|[A-Z]{1,6}\d*|[a-z]|[A-Za-z]\d+)$/;
// Plus every Latin word in GLOSSARY.zh.md §3, the names that stay in English.
const GLOSSARY_KEEP = (() => {
  try {
    const g = readFileSync("GLOSSARY.zh.md", "utf8");
    const sec = /^## 3\.[^\n]*\n([\s\S]*?)^## 4\./m.exec(g)?.[1] ?? "";
    return new Set(sec.match(/[A-Za-zÀ-ÿ]{2,}/g) ?? []);
  } catch { return new Set<string>(); }
})();
// Person names stay in Latin script: every surname in the bibliography.
const SURNAMES = new Set<string>();
for (const f of readdirSync("refs").filter((f) => f.endsWith(".bib"))) {
  for (const e of parseBib(readFileSync(`refs/${f}`, "utf8"), f)) {
    for (const p of people(e.fields.author ?? e.fields.editor).list) for (const w of p.last.split(/[^A-Za-zÀ-ÿ]+/)) if (w) SURNAMES.add(w);
  }
}
// Names with capitals inside (BrettPhos, tBPh, PairwiseGP) are product,
// reagent, or method names, never ordinary English words.
const kept = (w: string) => KEEP.test(w) || GLOSSARY_KEEP.has(w) || SURNAMES.has(w) || /[a-z][A-Z]|[A-Z]{2}/.test(w);
for (const fig of figures.values()) {
  if (!fig.labels.zh) continue;
  test(`figure ${fig.name} has complete Chinese labels`, () => {
    assert.deepEqual(Object.keys(fig.labels.zh!).sort(), Object.keys(fig.labels.en).sort());
    assert.ok(fig.title.zh, "title");
    for (const [k, spec] of Object.entries(fig.params) as Array<[string, { label: { zh?: string }; options?: Array<{ label: { zh?: string } }> }]>) {
      assert.ok(spec.label.zh, `param ${k} label`);
      for (const o of spec.options ?? []) assert.ok(o.label.zh, `param ${k} option`);
    }
    for (const a of fig.actions ?? []) assert.ok(a.label.zh, "action");
    if (fig.hint) assert.ok(fig.hint.zh, "hint");
  });
  test(`figure ${fig.name} draws in Chinese`, () => {
    const p = defaults(fig);
    const t = fig.timeline ? fig.timeline.poster(p) : 0;
    const svg = fig.render({ p, t, w: STATIC_WIDTHS.wide, uid: "z", lang: "zh" }, "zh");
    const d = fig.describe({ p, t, w: STATIC_WIDTHS.wide, uid: "z", lang: "zh" }, "zh");
    assert.match(d, /[一-鿿]/, "the description is in Chinese");
    const decode = (t: string) => t.replace(/&(lt|gt|amp|quot|apos|#\d+|#x[0-9a-f]+);/gi, " ");
    const words = [...svg.matchAll(/<text[^>]*>([^<]*)</g)].flatMap((m) => decode(m[1]).split(/[^A-Za-zÀ-ÿ]+/)).filter((w) => w.length > 1 && !kept(w));
    assert.deepEqual([...new Set(words)], [], "English words in the Chinese drawing");
  });
}
