// A small BibTeX reader and writer. The book's bibliography is BibTeX so it
// moves in and out of reference managers, but the entries are written by the
// tools in tools/refs and by hand in a narrow, regular style: one field per
// line, values in braces, UTF-8 text instead of LaTeX escapes. This parser
// accepts that style plus quoted values and bare numbers, and fails loudly on
// anything else rather than guessing.

export interface BibEntry {
  type: string; // lower case: article, inproceedings, misc, book, ...
  key: string;
  fields: Record<string, string>;
  file?: string; // the .bib file the entry came from, for error messages
}

export function parseBib(src: string, file = "<bib>"): BibEntry[] {
  const out: BibEntry[] = [];
  let i = 0;
  const n = src.length;
  const fail = (msg: string): never => {
    const line = src.slice(0, i).split("\n").length;
    throw new Error(`${file}:${line}: ${msg}`);
  };
  while (i < n) {
    const at = src.indexOf("@", i);
    if (at < 0) break;
    i = at + 1;
    const typeMatch = /^[A-Za-z]+/.exec(src.slice(i));
    if (!typeMatch) fail("expected an entry type after @");
    const type = typeMatch![0].toLowerCase();
    i += type.length;
    while (/\s/.test(src[i])) i++;
    if (src[i] !== "{") fail(`expected { after @${type}`);
    i++;
    if (type === "comment" || type === "preamble" || type === "string") {
      // Skip the balanced body.
      let depth = 1;
      while (i < n && depth) { if (src[i] === "{") depth++; else if (src[i] === "}") depth--; i++; }
      continue;
    }
    const comma = src.indexOf(",", i);
    if (comma < 0) fail("entry has no key");
    const key = src.slice(i, comma).trim();
    if (!/^[A-Za-z0-9_:.\-+]+$/.test(key)) fail(`bad key "${key}"`);
    i = comma + 1;
    const fields: Record<string, string> = {};
    for (;;) {
      while (i < n && /[\s,]/.test(src[i])) i++;
      if (src[i] === "}") { i++; break; }
      const nameMatch = /^[A-Za-z][A-Za-z0-9_\-]*/.exec(src.slice(i));
      if (!nameMatch) fail(`expected a field name in ${key}`);
      const name = nameMatch![0].toLowerCase();
      i += name.length;
      while (/\s/.test(src[i])) i++;
      if (src[i] !== "=") fail(`expected = after ${name} in ${key}`);
      i++;
      while (/\s/.test(src[i])) i++;
      let value = "";
      if (src[i] === "{") {
        let depth = 1;
        const start = ++i;
        while (i < n && depth) {
          if (src[i] === "\\") { i += 2; continue; }
          if (src[i] === "{") depth++;
          else if (src[i] === "}") depth--;
          i++;
        }
        if (depth) fail(`unbalanced braces in ${key}.${name}`);
        value = src.slice(start, i - 1);
      } else if (src[i] === '"') {
        const start = ++i;
        while (i < n && src[i] !== '"') { if (src[i] === "\\") i++; i++; }
        value = src.slice(start, i);
        i++;
      } else {
        const m = /^[A-Za-z0-9]+/.exec(src.slice(i));
        if (!m) fail(`expected a value for ${key}.${name}`);
        value = m![0];
        i += value.length;
      }
      fields[name] = value.replace(/\s+/g, " ").trim();
    }
    out.push({ type, key, fields, file });
  }
  return out;
}

// Field order for writing: identity first, then where it appeared, then links,
// then the book's own annotations.
const ORDER = [
  "author", "editor", "title", "year", "journal", "booktitle", "publisher", "school", "institution",
  "volume", "number", "pages", "venue", "howpublished", "doi", "eprint", "archiveprefix", "url",
  "evidence", "note", "tldr", "tldr-zh", "source",
];

export function formatEntry(e: BibEntry): string {
  const keys = Object.keys(e.fields).filter((k) => e.fields[k] !== undefined && e.fields[k] !== "");
  keys.sort((a, b) => {
    const ia = ORDER.indexOf(a), ib = ORDER.indexOf(b);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || a.localeCompare(b);
  });
  const width = Math.max(...keys.map((k) => k.length));
  const lines = keys.map((k) => `  ${k.padEnd(width)} = {${e.fields[k]}}`);
  return `@${e.type}{${e.key},\n${lines.join(",\n")}\n}\n`;
}

export function formatBib(entries: BibEntry[], header = ""): string {
  return (header ? header.trimEnd() + "\n\n" : "") + entries.map(formatEntry).join("\n");
}

// Split a BibTeX author field into display names. "Last, First" and
// "First Last" both work; "others" becomes an et al. marker.
export interface Person { last: string; first: string; corporate?: boolean }

export function people(field: string | undefined): { list: Person[]; etal: boolean } {
  if (!field) return { list: [], etal: false };
  const parts = splitAnd(field);
  let etal = false;
  const list: Person[] = [];
  for (const raw of parts) {
    const p = raw.trim();
    if (!p) continue;
    if (p === "others") { etal = true; continue; }
    if (p.startsWith("{") && p.endsWith("}")) { list.push({ last: p.slice(1, -1), first: "", corporate: true }); continue; }
    if (p.includes(",")) {
      const [last, ...rest] = p.split(",");
      list.push({ last: last.trim(), first: rest.join(",").trim() });
    } else {
      const words = p.split(/\s+/);
      // Particles stay with the surname: "de Las Casas", "van den Driessche".
      let j = words.length - 1;
      while (j > 0 && /^(van|von|der|den|de|del|della|di|da|la|le|du|dos|das|ter|ten)$/i.test(words[j - 1])) j--;
      list.push({ last: words.slice(j).join(" "), first: words.slice(0, j).join(" ") });
    }
  }
  return { list, etal };
}

function splitAnd(field: string): string[] {
  const out: string[] = [];
  let depth = 0, cur = "";
  const tokens = field.split(/(\s+and\s+|[{}])/);
  for (const t of tokens) {
    if (t === "{") { depth++; cur += t; continue; }
    if (t === "}") { depth--; cur += t; continue; }
    if (depth === 0 && /^\s+and\s+$/.test(t)) { out.push(cur); cur = ""; continue; }
    cur += t;
  }
  out.push(cur);
  return out;
}

// Strip the braces BibTeX uses to protect capitalization.
// Math segments ($...$) keep their braces for KaTeX.
export function plain(s: string | undefined): string {
  return (s ?? "").split(/(\$[^$]+\$)/).map((part) => (/^\$[^$]+\$$/.test(part) ? part
    : part.replace(/[{}]/g, "").replace(/\\&/g, "&").replace(/\\%/g, "%").replace(/\\_/g, "_").replace(/--/g, "–"))).join("");
}
