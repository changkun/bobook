// Citation formatting: author-year labels for inline citations and full
// entries for reference lists. Labels are disambiguated across the whole book
// (Lin et al., 2022a / 2022b) so a label means the same work on every page.

import katex from "katex";
import { people, plain, type BibEntry } from "./bib.ts";
import { UI, type Lang } from "./i18n.ts";

export type Bib = Map<string, BibEntry>;

export const EVIDENCE_LABEL: Record<string, string> = UI.en.evidence;

// The author part of a label, kept as structure so each edition can join the
// names in its own words ("and" / "与", "et al." / "等人").
export interface Authors { names: string[]; etal: boolean }

function surnameParts(e: BibEntry): Authors {
  const { list, etal } = people(e.fields.author ?? e.fields.editor);
  if (!list.length) return { names: [plain(e.fields.howpublished ?? e.fields.publisher ?? e.key)], etal: false };
  const last = list.map((p) => plain(p.last));
  if (list.length === 1 && !etal) return { names: last, etal: false };
  if (list.length === 2 && !etal) return { names: last, etal: false };
  return { names: [last[0]], etal: true };
}

export function joinAuthors(a: Authors, lang: Lang = "en", paren = false): string {
  const ui = UI[lang];
  if (a.etal) return `${a.names[0]} ${paren ? ui.etAlParen : ui.etAl}`;
  return a.names.join(` ${ui.and} `);
}

function surnames(e: BibEntry): string {
  return joinAuthors(surnameParts(e));
}

export function year(e: BibEntry): string {
  const y = e.fields.year ?? "";
  return /^\d{4}$/.test(y) ? y : y ? plain(y) : "n.d.";
}

// Short labels for every key that will be cited, with a/b suffixes where two
// works share author text and year.
export function citeLabels(bib: Bib, keys: Iterable<string>): Map<string, { authors: string; year: string; parts: Authors }> {
  const out = new Map<string, { authors: string; year: string; parts: Authors }>();
  const groups = new Map<string, string[]>();
  for (const k of new Set(keys)) {
    const e = bib.get(k);
    if (!e) continue;
    const a = surnames(e), y = year(e);
    out.set(k, { authors: a, year: y, parts: surnameParts(e) });
    const g = `${a}|${y}`;
    groups.set(g, [...(groups.get(g) ?? []), k]);
  }
  for (const ks of groups.values()) {
    if (ks.length < 2) continue;
    ks.sort((a, b) => plain(bib.get(a)!.fields.title).localeCompare(plain(bib.get(b)!.fields.title)));
    ks.forEach((k, i) => { const l = out.get(k)!; out.set(k, { ...l, year: l.year + String.fromCharCode(97 + i) }); });
  }
  return out;
}

const escHtml = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

// Titles may contain inline math ("Top-$k$ Ranking"): render it with KaTeX
// and escape the rest.
function titleHtml(t: string): string {
  return t.split(/(\$[^$]+\$)/).map((part) => {
    if (/^\$[^$]+\$$/.test(part)) {
      try { return katex.renderToString(part.slice(1, -1), { throwOnError: true, output: "html" }); } catch { return escHtml(part); }
    }
    return escHtml(part);
  }).join("");
}

function nameList(e: BibEntry): string {
  const { list, etal } = people(e.fields.author ?? e.fields.editor);
  if (!list.length) return "";
  const fmt = list.map((p) => (p.corporate ? plain(p.last) : `${plain(p.last)}${p.first ? `, ${initials(plain(p.first))}` : ""}`));
  let s: string;
  if (fmt.length > 8) s = `${fmt.slice(0, 6).join(", ")}, … ${fmt[fmt.length - 1]}`;
  else if (fmt.length === 1) s = fmt[0];
  else s = `${fmt.slice(0, -1).join(", ")}, and ${fmt[fmt.length - 1]}`;
  return etal ? `${s}, et al.` : s;
}

function initials(first: string): string {
  return first.split(/\s+/).map((w) => (/^[A-Z]\.?$/.test(w) ? w.replace(/\.?$/, ".") : w.split("-").map((p) => (p ? `${p[0]}.` : "")).join("-"))).join(" ");
}

export function venue(e: BibEntry): string {
  const f = e.fields;
  return plain(f.journal ?? f.booktitle ?? f.howpublished ?? f.publisher ?? f.school ?? f.institution ?? f.venue ?? "");
}

export function bestLink(e: BibEntry): string | undefined {
  const f = e.fields;
  if (f.url) return f.url;
  if (f.doi) return `https://doi.org/${f.doi}`;
  if (f.eprint) return `https://arxiv.org/abs/${f.eprint}`;
  return undefined;
}

// One reference-list entry as HTML (inside an <li>).
export function entryHtml(e: BibEntry, label?: { year: string }, lang: Lang = "en"): string {
  const f = e.fields;
  const names = nameList(e);
  const y = label?.year ?? year(e);
  const title = plain(f.title);
  const link = bestLink(e);
  const v = venue(e);
  const ev = UI[lang].evidence[f.evidence ?? ""];
  let s = "";
  if (names) s += `<span class="ref-authors">${escHtml(names)}</span> `;
  s += `<span class="ref-year">(${escHtml(y)})</span>. `;
  const t = titleHtml(title.replace(/[.?!]$/, "")) + (/[?!]$/.test(title) ? title.slice(-1) : "");
  s += link ? `<a class="ref-title" href="${escHtml(link)}" rel="noopener">${t}</a>` : `<span class="ref-title">${t}</span>`;
  s += /[?!]$/.test(title) ? " " : ". ";
  if (v) s += `<span class="ref-venue">${escHtml(v)}</span>. `;
  if (f.doi && !(link ?? "").includes(f.doi)) s += `<a class="ref-doi" href="https://doi.org/${escHtml(f.doi)}">doi:${escHtml(f.doi)}</a>. `;
  if (ev) s += `<span class="ref-badge ref-badge-${escHtml(f.evidence ?? "")}">${escHtml(ev)}</span>`;
  return s.trim();
}

// The inline label: "Chu and Ghahramani, 2005" or, narratively, "Chu and
// Ghahramani (2005)"; in Chinese "Lin 等，2022" inside parentheses and
// "Lin 等人（2022）" in running text, with "与" between two authors.
export function inlineLabel(l: { authors: string; year: string; parts?: Authors }, mode: "paren" | "narrative" | "year", lang: Lang = "en"): string {
  if (mode === "year") return l.year;
  const who = l.parts ? joinAuthors(l.parts, lang, mode === "paren") : l.authors;
  if (lang === "zh") return mode === "narrative" ? `${who}（${l.year}）` : `${who}，${l.year}`;
  if (mode === "narrative") return `${who} (${l.year})`;
  return `${who}, ${l.year}`;
}
