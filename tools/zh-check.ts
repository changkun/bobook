// Check the Chinese edition against the English one, file by file.
//
//   node tools/zh-check.ts                 every zh/ file
//   node tools/zh-check.ts zh/gp/02-gp-regression.md ...
//
// A translation must keep the structure byte for byte: ids, cross-references,
// citation keys, math, figure blocks and their parameters, code, and the
// number of headings and callouts. It must also follow GLOSSARY.zh.md: no
// forbidden term variants, a space between Chinese and Latin letters or digits,
// Chinese punctuation in Chinese text, and no English sentences left behind.
// Exit status 1 if anything is wrong.

import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const walk = (d: string): string[] => readdirSync(d, { withFileTypes: true }).flatMap((e) => e.isDirectory() ? walk(join(d, e.name)) : e.name.endsWith(".md") ? [join(d, e.name)] : []);
const files = process.argv.slice(2).length ? process.argv.slice(2) : existsSync("zh") ? walk("zh") : [];

// Forbidden variants: rows "| forbidden | required |" of the mechanical
// checklist in GLOSSARY.zh.md (§4.1, "机械检查清单").
function forbidden(): Array<[string, string]> {
  if (!existsSync("GLOSSARY.zh.md")) return [];
  const g = readFileSync("GLOSSARY.zh.md", "utf8");
  const m = /^#+[^\n]*(机械检查|Forbidden variants)[^\n]*\n([\s\S]*?)(?=^#+ )/im.exec(g + "\n# end");
  if (!m) return [];
  const rows: Array<[string, string]> = [];
  for (const line of m[2].split("\n")) {
    const c = line.split("|").map((x) => x.trim()).filter((x) => x !== "");
    if (c.length < 2 || /^:?-+:?$/.test(c[0]) || /forbidden|禁用写法/i.test(c[0])) continue;
    const bad = c[0].replace(/[`*]/g, "").trim();
    if (bad && /[一-鿿]/.test(bad)) rows.push([bad, c[1].replace(/[`*]/g, "").trim()]);
  }
  return rows;
}
const FORBIDDEN = forbidden();

// Parts of a Markdown file that must survive translation unchanged.
function skeleton(src: string) {
  const body = src.replace(/^---\n[\s\S]*?\n---\n/, "");
  const fences = [...body.matchAll(/^```[^\n]*\n[\s\S]*?^```/gm)].map((m) => m[0]);
  const figures = fences.filter((f) => f.startsWith("```{figure}")).map((f) => f.split("\n").filter((l) => !/^\/\/\| fig-cap:/.test(l) && !/^\/\/\| (alt|fig-alt):/.test(l)).join("\n"));
  const code = fences.filter((f) => !f.startsWith("```{figure}")).map((f) => f.split("\n").map((l) => l.replace(/#.*$|\/\/.*$/, "").trimEnd()).join("\n"));
  const noFence = body.replace(/^```[^\n]*\n[\s\S]*?^```/gm, "");
  const displayMath = [...noFence.matchAll(/\$\$[\s\S]*?\$\$/g)].map((m) => m[0].replace(/\s+/g, " "));
  const noDisplay = noFence.replace(/\$\$[\s\S]*?\$\$/g, "");
  // Inline math may wrap across source lines; compare it with whitespace
  // normalized. Inline code is removed first so a $ inside it cannot pair.
  // As in the parser, an opening $ is not followed by a space and a closing $
  // is not preceded by one or followed by a digit, so "$1,000 ... $500" is
  // money, not math.
  const inlineMath = [...noDisplay.replace(/`[^`\n]*`/g, "").matchAll(/(?<![\\$])\$(?![\s$])([^$]+?)(?<!\s)\$(?!\d)/g)].map((m) => m[1].replace(/\s+/g, " ").trim()).sort();
  const ids = [...body.matchAll(/\{[^}\n]*#([a-z]+-[\w:.-]+)[^}\n]*\}/g)].map((m) => m[1]).sort();
  const xrefs = [...noFence.matchAll(/@((?:sec|fig|eq|tbl|def|thm|lem|prop|cor|ex|exr|alg)-[\w-]+)/g)].map((m) => m[1]).sort();
  const cites = [...noFence.matchAll(/(?<![\w.])@([a-z][\w:.+-]*\d{4}[\w:.+-]*)/g)].map((m) => m[1].replace(/[.:]+$/, "")).sort();
  const headings = (noFence.match(/^#{1,4} /gm) ?? []).length;
  const callouts = (noFence.match(/^:::+ *\{/gm) ?? []).length;
  return { figures, code, displayMath, inlineMath, ids, xrefs, cites, headings, callouts, noFence: noDisplay };
}

function diffList(name: string, a: string[], b: string[], out: string[]) {
  const count = (xs: string[]) => xs.reduce((m, x) => m.set(x, (m.get(x) ?? 0) + 1), new Map<string, number>());
  const ca = count(a), cb = count(b);
  const missing = [...ca].filter(([k, n]) => (cb.get(k) ?? 0) < n).map(([k]) => k);
  const extra = [...cb].filter(([k, n]) => (ca.get(k) ?? 0) < n).map(([k]) => k);
  if (missing.length) out.push(`${name} missing: ${missing.slice(0, 6).join(" | ")}${missing.length > 6 ? ` (+${missing.length - 6})` : ""}`);
  if (extra.length) out.push(`${name} added: ${extra.slice(0, 6).join(" | ")}${extra.length > 6 ? ` (+${extra.length - 6})` : ""}`);
}

let bad = 0;
for (const zf of files) {
  const ef = zf.replace(/^zh\//, "en/");
  const out: string[] = [];
  if (!existsSync(ef)) { console.log(`✗ ${zf}: no English counterpart ${ef}`); bad++; continue; }
  const e = skeleton(readFileSync(ef, "utf8"));
  const zsrc = readFileSync(zf, "utf8");
  const z = skeleton(zsrc);
  diffList("ids", e.ids, z.ids, out);
  diffList("cross-references", e.xrefs, z.xrefs, out);
  diffList("citations", e.cites, z.cites, out);
  diffList("inline math", e.inlineMath, z.inlineMath, out);
  diffList("display math", e.displayMath, z.displayMath, out);
  diffList("figure blocks", e.figures, z.figures, out);
  diffList("code (comments ignored)", e.code, z.code, out);
  if (e.headings !== z.headings) out.push(`headings: ${e.headings} in English, ${z.headings} in Chinese`);
  if (e.callouts !== z.callouts) out.push(`callouts: ${e.callouts} in English, ${z.callouts} in Chinese`);

  // Style, on prose only (no math, code, links, ids, or citations).
  const prose = z.noFence
    .replace(/^---\n[\s\S]*?\n---\n/, "")
    .replace(/`[^`\n]*`/g, " ")
    .replace(/\$[^$\n]*\$/g, " M ")
    .replace(/\]\([^)]*\)/g, "]")
    .replace(/\{[^}\n]*\}/g, " ")
    .replace(/\[[-+]?@[^\]]*\]/g, " C ")
    .replace(/@[\w:.-]+/g, " C ")
    .replace(/<[^>]+>/g, " ")
    .replace(/https?:\/\/\S+/g, " ");
  const lines = prose.split("\n");
  lines.forEach((l, i) => {
    if (/^\s*(\/\/\||\||:::)/.test(l)) return; // figure options, tables are checked as text below
    for (const [f, r] of FORBIDDEN) if (l.includes(f)) out.push(`line ~${i + 1}: forbidden "${f}" (use "${r}")`);
    if (/[一-鿿][A-Za-z0-9]|[A-Za-z0-9][一-鿿]/.test(l.replace(/\b[CM]\b/g, ""))) {
      const m = /.{0,12}([一-鿿][A-Za-z0-9]|[A-Za-z0-9][一-鿿]).{0,12}/.exec(l)!;
      out.push(`line ~${i + 1}: no space between Chinese and Latin: "${m[0]}"`);
    }
    if (/[一-鿿][,;:?!]|[一-鿿] ?\((?![A-Za-z])/.test(l)) {
      const m = /.{0,10}[一-鿿][,;:?!(].{0,10}/.exec(l);
      if (m) out.push(`line ~${i + 1}: half-width punctuation after Chinese: "${m[0]}"`);
    }
    // An English sentence left untranslated: eight or more Latin words in a row.
    // English kept on purpose in full-width parentheses after a translated
    // quotation or title does not count.
    const run = /(?:\b[A-Za-z][a-z'-]+\b[ ,]+){8,}/.exec(l.replace(/（[^（）]*）/g, " "));
    // A proper name in title case (a workshop, a law, a project) is kept on purpose.
    const words = run ? run[0].split(/[ ,]+/).filter(Boolean) : [];
    const titleCase = words.filter((w) => /^[A-Z]/.test(w)).length >= 0.6 * words.length;
    if (run && !titleCase && !/[一-鿿]/.test(run[0]) && !/“[^”]*$/.test(l.slice(0, run.index))) out.push(`line ~${i + 1}: English left untranslated? "${run[0].slice(0, 60)}"`);
  });
  // Tables and figure captions: forbidden terms only.
  for (const [f, r] of FORBIDDEN) if (zsrc.split("\n").some((l) => /^\s*(\||\/\/\| fig-cap)/.test(l) && l.includes(f))) out.push(`table or caption: forbidden "${f}" (use "${r}")`);

  const shown = [...new Set(out)];
  if (shown.length) { bad++; console.log(`✗ ${zf}\n    ${shown.slice(0, 25).join("\n    ")}${shown.length > 25 ? `\n    ... ${shown.length - 25} more` : ""}`); }
  else console.log(`✓ ${zf}`);
}
if (!files.length) console.log("no zh/ files to check");
process.exit(bad ? 1 : 0);
