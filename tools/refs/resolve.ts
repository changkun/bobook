// Rewrite URL citations in chapter sources to bibliography keys.
//
//   node tools/refs/resolve.ts en/frontier/02-observation-models.md [...]
//
// While drafting, a work can be cited by its URL:
//   [@<https://arxiv.org/abs/2008.06677>]      or narratively   @<https://...>
// This tool replaces each with the key of the entry whose URL matches (after
// normalization: scheme, www, arXiv abs/html/pdf and versions), leaves
// unmatched ones in place, and lists them so they can be added to
// refs/additions.bib. It exits non-zero if any remain.

import { readFileSync, writeFileSync } from "node:fs";
import { loadAll, normUrl } from "./lookup.ts";

const files = process.argv.slice(2);
if (!files.length) { console.error("usage: node tools/refs/resolve.ts <file.md ...>"); process.exit(2); }

const byUrl = new Map<string, string>();
for (const e of loadAll()) {
  for (const u of [e.fields.url, ...(e.fields.urlalt ?? "").split(/\s+/)].filter(Boolean)) {
    const n = normUrl(u!);
    if (!byUrl.has(n)) byUrl.set(n, e.key);
  }
  if (e.fields.doi) byUrl.set(normUrl(`https://doi.org/${e.fields.doi}`), byUrl.get(normUrl(`https://doi.org/${e.fields.doi}`)) ?? e.key);
  if (e.fields.eprint) byUrl.set(normUrl(`https://arxiv.org/abs/${e.fields.eprint}`), byUrl.get(normUrl(`https://arxiv.org/abs/${e.fields.eprint}`)) ?? e.key);
}

let missing = 0;
for (const f of files) {
  const src = readFileSync(f, "utf8");
  let n = 0;
  const miss: string[] = [];
  const out = src.replace(/@<(https?:\/\/[^>\s]+)>/g, (whole, url: string) => {
    const key = byUrl.get(normUrl(url));
    if (!key) { miss.push(url); return whole; }
    n++;
    return `@${key}`;
  });
  if (out !== src) writeFileSync(f, out);
  console.log(`${f}: ${n} resolved${miss.length ? `, ${miss.length} unresolved` : ""}`);
  for (const u of [...new Set(miss)]) console.log(`  unresolved: ${u}`);
  missing += miss.length;
}
if (missing) process.exit(1);
