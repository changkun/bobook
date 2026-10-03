// The book's structure: <lang>/book.yml lists the front matter, the parts with
// their chapters, and the appendices. This module reads it, numbers the
// chapters (1, 2, ... across parts; A, B, ... for appendices), and gives every
// page its output path. Each chapter file may open with YAML front matter:
//
//   ---
//   status: draft            # outline | draft | review | done
//   synopsis: One or two sentences shown on the part page.
//   sources: [Rasmussen and Williams 2006]   # where the material comes from, for editors
//   ---

import { readFileSync, existsSync } from "node:fs";
import { parse as parseYaml } from "yaml";

export interface Page {
  file: string; // source path, relative to the repo root
  href: string; // output path, root-relative ("en/foundations/02-probability.html")
  kind: "chapter" | "appendix" | "front" | "part" | "index";
  num: string; // "3", "A", "I" for parts, "" otherwise
  part?: PartInfo;
  meta: { status?: string; synopsis?: string; sources?: string[]; title?: string };
  src: string; // markdown without front matter
  untranslated?: boolean; // a translated edition fell back to the English source
}

export interface PartInfo {
  id: string;
  num: string; // roman numeral
  title: string;
  intro?: Page;
  chapters: Page[];
}

export interface Book {
  lang: "en" | "zh";
  title: string;
  subtitle: string;
  author: string;
  repo?: string;
  url?: string; // the published site's root, ending in /
  front: Page[];
  parts: PartInfo[];
  appendices: Page[];
  back: Page[];
  pages: Page[]; // reading order
}

const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"];

export function splitFrontMatter(raw: string): { meta: Page["meta"]; src: string } {
  const m = /^---\n([\s\S]*?)\n---\n?/.exec(raw);
  if (!m) return { meta: {}, src: raw };
  const meta = (parseYaml(m[1]) ?? {}) as Page["meta"];
  return { meta, src: raw.slice(m[0].length) };
}

// A translated edition lists the same files as the English one. While a page
// is not translated yet, it is built from the English source and marked, so
// numbering and cross-references stay complete; a strict build refuses that.
function load(lang: string, file: string, kind: Page["kind"], num: string, part?: PartInfo): Page {
  let path = `${lang}/${file}`;
  let untranslated = false;
  if (!existsSync(path) && lang !== "en" && existsSync(`en/${file}`)) { path = `en/${file}`; untranslated = true; }
  if (!existsSync(path)) throw new Error(`book.yml lists ${path}, which does not exist`);
  const { meta, src } = splitFrontMatter(readFileSync(path, "utf8"));
  const href = `${lang}/${file.replace(/\.md$/, ".html")}`;
  return { file: path, href, kind, num, part, meta, src, untranslated };
}

interface BookYml {
  title: string;
  subtitle?: string;
  author?: string;
  repo?: string;
  url?: string;
  front?: string[];
  parts?: Array<{ id: string; title: string; intro?: string; chapters: string[] }>;
  appendices?: string[];
  back?: string[];
}

export function loadBook(lang: "en" | "zh" = "en"): Book {
  const y = parseYaml(readFileSync(`${lang}/book.yml`, "utf8")) as BookYml;
  const front = (y.front ?? []).map((f) => load(lang, f, f === "index.md" ? "index" : "front", ""));
  let n = 0;
  const parts: PartInfo[] = (y.parts ?? []).map((p, i) => {
    const part: PartInfo = { id: p.id, num: ROMAN[i], title: p.title, chapters: [] };
    if (p.intro) part.intro = load(lang, p.intro, "part", ROMAN[i], part);
    part.chapters = p.chapters.map((c) => load(lang, c, "chapter", String(++n), part));
    return part;
  });
  const appendices = (y.appendices ?? []).map((f, i) => load(lang, f, "appendix", String.fromCharCode(65 + i)));
  const back = (y.back ?? []).map((f) => load(lang, f, "front", ""));
  const pages: Page[] = [...front];
  for (const p of parts) {
    if (p.intro) pages.push(p.intro);
    pages.push(...p.chapters);
  }
  pages.push(...appendices, ...back);
  return { lang, title: y.title, subtitle: y.subtitle ?? "", author: y.author ?? "", repo: y.repo, url: y.url, front, parts, appendices, back, pages };
}
