// Find citation keys for a URL, an arXiv id, a DOI, or words of a title.
//
//   node tools/refs/lookup.ts https://arxiv.org/abs/2303.15746
//   node tools/refs/lookup.ts 2303.15746
//   node tools/refs/lookup.ts "qEUBO"
//
// URLs are matched after normalization (scheme, www, trailing slash, arXiv
// abs/html/pdf and version suffix), so a link in a draft and its bib entry meet
// even when they spell the URL differently.

import { readFileSync, readdirSync } from "node:fs";
import { parseBib, type BibEntry } from "../../src/pipeline/bib.ts";

export function loadAll(dir = "refs"): BibEntry[] {
  return readdirSync(dir).filter((f) => f.endsWith(".bib")).flatMap((f) => parseBib(readFileSync(`${dir}/${f}`, "utf8"), `${dir}/${f}`));
}

export function normUrl(u: string): string {
  let s = u.trim().replace(/^https?:\/\//, "").replace(/^www\./, "").replace(/[)）。.,，]+$/, "").replace(/\/$/, "");
  s = s.replace(/^arxiv\.org\/(?:abs|html|pdf)\/([^\s?#]+?)(?:v\d+)?(?:\.pdf)?$/i, "arxiv.org/abs/$1");
  s = s.replace(/^dx\.doi\.org\//, "doi.org/");
  try { s = decodeURIComponent(s); } catch { /* keep */ }
  return s.toLowerCase();
}

function main() {
  const q = process.argv.slice(2).join(" ").trim();
  if (!q) { console.error("usage: node tools/refs/lookup.ts <url | arXiv id | DOI | title words>"); process.exit(2); }
  const all = loadAll();
  let hits: BibEntry[];
  if (/^https?:\/\//.test(q)) {
    const n = normUrl(q);
    const urls = (e: { fields: Record<string, string> }) => [e.fields.url, ...(e.fields.urlalt ?? "").split(/\s+/)].filter(Boolean) as string[];
    hits = all.filter((e) => urls(e).some((u) => normUrl(u) === n));
  } else if (/^\d{4}\.\d{4,5}(v\d+)?$/.test(q)) {
    const id = q.replace(/v\d+$/, "");
    hits = all.filter((e) => e.fields.eprint === id || (e.fields.url ?? "").includes(id));
  } else if (/^10\.\d{4,}\//.test(q)) {
    hits = all.filter((e) => (e.fields.doi ?? "").toLowerCase() === q.toLowerCase());
  } else {
    const words = q.toLowerCase().split(/\s+/);
    hits = all.filter((e) => {
      const hay = `${e.key} ${e.fields.title ?? ""} ${e.fields.author ?? ""}`.toLowerCase();
      return words.every((w) => hay.includes(w));
    });
  }
  if (!hits.length) { console.log("no match"); process.exit(1); }
  for (const e of hits.slice(0, 20)) {
    const venue = e.fields.journal ?? e.fields.booktitle ?? e.fields.howpublished ?? e.fields.publisher ?? "";
    console.log(`${e.key}\t${e.fields.year ?? ""}\t${e.fields.evidence ?? ""}\t${(e.fields.title ?? "").slice(0, 90)}\t${venue}`);
  }
}

if (import.meta.url === `file://${process.argv[1]}`) main();
