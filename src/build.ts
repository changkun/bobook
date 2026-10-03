// Build the book into _site/.
//
//   node src/build.ts            full build, fails on any error
//   node src/build.ts --lenient  report errors but still write the site
//   node src/build.ts --only en  build one edition
//
// Every edition whose <lang>/book.yml exists is built (en, and zh once it
// exists). Steps for each: read its book.yml and every chapter; parse all chapters so labels,
// numbers, and citations are known book-wide; render each page with
// cross-references, per-section and per-chapter reference lists, and static
// figure SVG; write the pages with relative links; bundle the client script
// with one lazy chunk per figure; copy KaTeX and the stylesheet.

import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, relative } from "node:path";
import { createHash } from "node:crypto";
import * as esbuild from "esbuild";
import { loadBook, type Page } from "./pipeline/book.ts";
import { parseBib, type BibEntry } from "./pipeline/bib.ts";
import { citeLabels, entryHtml, inlineLabel, type Bib } from "./pipeline/cite.ts";
import { UI, partName, type Lang } from "./pipeline/i18n.ts";
import { tightenCjkSpaces } from "./pipeline/cjk.ts";
import { createMarkdown, newEnv, type ChapterEnv, type Label } from "./pipeline/markdown.ts";
import { page as shell } from "./pipeline/layout.ts";
import { figureFiles, loadFigures, registrySource } from "./figures/registry.ts";
import type { AnyFigure } from "./figures/types.ts";
import { coverArt } from "./cover.ts";

// --out <dir> builds elsewhere, so several people (or agents) can build and
// check the book at the same time without deleting each other's output.
const outIdx = process.argv.indexOf("--out");
const OUT = outIdx > 0 ? process.argv[outIdx + 1] : "_site";
const lenient = process.argv.includes("--lenient");
const t0 = performance.now();

const esc = (s: string) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
const stripTags = (s: string) => s.replace(/<annotation[\s\S]*?<\/annotation>/g, "").replace(/<span class="katex-mathml">[\s\S]*?<\/span>(?=<span class="katex-html")/g, "").replace(/<[^>]+>/g, "").replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/\s+/g, " ").trim();

function loadBibs(): Bib {
  const bib: Bib = new Map();
  for (const f of readdirSync("refs").filter((f) => f.endsWith(".bib")).sort()) {
    for (const e of parseBib(readFileSync(`refs/${f}`, "utf8"), `refs/${f}`)) {
      if (bib.has(e.key)) throw new Error(`duplicate bib key ${e.key} in refs/${f} and ${bib.get(e.key)!.file}`);
      bib.set(e.key, e);
    }
  }
  // Renamed and merged keys (written by tools/refs/enrich.ts) stay citable:
  // the old key resolves to the entry under its new key.
  if (existsSync("refs/rekey.json")) {
    const rekey = JSON.parse(readFileSync("refs/rekey.json", "utf8")) as Record<string, string>;
    for (const [from, to] of Object.entries(rekey)) {
      let target = to, hops = 0;
      while (!bib.has(target) && rekey[target] && hops++ < 10) target = rekey[target];
      const e = bib.get(target);
      if (e && !bib.has(from)) bib.set(from, e);
    }
  }
  return bib;
}

// One edition (en, zh): parse every page so labels, numbers, and citations
// are known edition-wide, then render each page with its reference lists, and
// the part pages, contents, and bibliography that the build writes itself.
function renderEdition(lang: Lang, bib: Bib, figures: Map<string, AnyFigure>) {
  const ui = UI[lang];
  const book = loadBook(lang);
  // --gallery adds a page that embeds every figure module with its defaults,
  // for looking at figures before (or regardless of) the chapters that use them.
  if (lang === "en" && process.argv.includes("--gallery")) {
    const names = figureFiles().map((f) => f.name);
    const src = "# Figure gallery {#sec-gallery}\n\nEvery figure module with its default parameters. Development only.\n\n"
      + names.map((n) => "## " + n + " {.unnumbered}\n\n```{figure}\n//| figure: " + n + "\n//| fig-cap: \"" + n + "\"\n```\n").join("\n");
    const g = { file: "(gallery)", href: "en/_gallery.html", kind: "front" as const, num: "", meta: {}, src };
    book.back.push(g);
    book.pages.push(g);
  }
  const md = createMarkdown({ bib, figures, lang });
  const labels = new Map<string, Label>();
  const errors: string[] = [];
  if (lang !== "en") for (const p of book.pages) if (p.untranslated) errors.push(`${lang}/${p.file.replace(/^en\//, "")}: not translated yet (built from ${p.file})`);

  // Phase 1: parse every page.
  const parsed = book.pages.map((p) => {
    const kind = p.kind === "appendix" ? "appendix" : p.kind === "chapter" ? "chapter" : p.kind === "part" ? "part" : "front";
    const env = newEnv({ num: kind === "front" ? "" : p.num, kind, href: p.href, file: p.file }, labels);
    const tokens = md.parse(p.src, env);
    return { p, env, tokens };
  });
  const allCites = parsed.flatMap((x) => x.env.cites);
  const labelsByKey = citeLabels(bib, allCites);
  const titles = new Map<string, string>();
  for (const { p, env } of parsed) {
    const html = env.title ? md.renderer.renderInline(env.title.children, md.options, env) : esc(p.meta.title ?? p.file);
    titles.set(p.href, html);
  }
  const byAuthor = (a: string, b: string) => labelsByKey.get(a)!.authors.localeCompare(labelsByKey.get(b)!.authors) || labelsByKey.get(a)!.year.localeCompare(labelsByKey.get(b)!.year);
  const short = (k: string) => inlineLabel(labelsByKey.get(k)!, "narrative", lang);
  // Where each work is cited, for the bibliography's backlinks and part pages.
  const citedIn = new Map<string, Array<{ href: string; sec: string; num: string; page: string }>>();

  // Phase 2: render.
  const rendered = new Map<string, { body: string; env: ChapterEnv }>();
  for (const { p, env, tokens } of parsed) {
    env.citeLabels = labelsByKey;
    let body = md.renderer.render(tokens, md.options, env);
    if (env.section !== undefined) body += `<!--SECREFS:${env.section}-->\n`;
    // Per-section reference lists.
    const secNum = new Map(env.sectionOrder.map((s) => [s.id, s]));
    body = body.replace(/<!--SECREFS:([^>]*?)-->/g, (_: string, id: string) => {
      const keys = env.sectionCites.get(id);
      const sec = secNum.get(id);
      for (const k of keys ?? []) {
        const list = citedIn.get(k) ?? [];
        list.push({ href: p.href, sec: id === "_intro" ? "" : id, num: sec?.num ?? "", page: p.href });
        citedIn.set(k, list);
      }
      if (!keys?.length || p.kind === "index") return "";
      if (/^(further-reading|references)$/.test(id)) return "";
      const items = keys.map((k) => {
        const e = bib.get(k);
        if (!e) return "";
        return `<li><a href="#ref-${esc(k)}">${esc(short(k))}</a> ${esc((e.fields.title ?? "").replace(/[{}]/g, ""))}</li>`;
      }).join("");
      const where = sec?.num ? ui.section(sec.num) : id === "_intro" ? ui.theIntroduction : ui.thisSection;
      return `<details class="sec-refs"><summary>${esc(ui.citedInSection(where))} <span class="sec-refs-n">${keys.length}</span></summary><ol>${items}</ol></details>\n`;
    });
    // Per-chapter reference list.
    const keys = [...new Set(env.cites)].filter((k) => bib.has(k));
    if (keys.length && p.kind !== "index") {
      keys.sort(byAuthor);
      const secs = (k: string) => [...env.sectionCites.entries()].filter(([, ks]) => ks.includes(k)).map(([id]) => secNum.get(id)).filter((s) => s?.num);
      const items = keys.map((k) => {
        const back = secs(k).map((s) => `<a href="#${esc(s!.id)}">§${esc(s!.num)}</a>`).join(" ");
        return `<li id="ref-${esc(k)}">${entryHtml(bib.get(k)!, labelsByKey.get(k), lang)}${back ? ` <span class="ref-back">${esc(ui.citedIn)} ${back}</span>` : ""}</li>`;
      }).join("\n");
      body += `<section class="references"><h2 id="references" class="unnumbered">${esc(ui.references)}</h2><ol class="ref-list">\n${items}\n</ol></section>\n`;
      env.toc.push({ level: 2, id: "references", num: "", inline: null });
    }
    if (lang === "zh") body = tightenCjkSpaces(body);
    rendered.set(p.href, { body, env });
  }

  // Part pages: the intro text, then the chapters with synopses, then every
  // work the part's chapters cite.
  for (const part of book.parts) {
    if (!part.intro) continue;
    const r = rendered.get(part.intro.href)!;
    const list = part.chapters.map((c) => {
      const st = c.meta.status === "outline" ? ` <span class="badge">${esc(ui.notWritten)}</span>` : "";
      return `<li><a href="@@ROOT@@${esc(c.href)}"><span class="part-ch-num">${esc(c.num)}</span> <span class="part-ch-title">${titles.get(c.href)}</span></a>${st}${c.meta.synopsis ? `<p>${esc(c.meta.synopsis)}</p>` : ""}</li>`;
    }).join("\n");
    let extra = `<h2 id="chapters" class="unnumbered">${esc(ui.chaptersInPart)}</h2><ol class="part-chapters">${list}</ol>\n`;
    const partKeys = new Set(part.chapters.flatMap((c) => rendered.get(c.href)!.env.cites).filter((k) => bib.has(k)));
    if (partKeys.size) {
      const items = [...partKeys].sort(byAuthor).map((k) => {
        const where = [...new Set((citedIn.get(k) ?? []).filter((c) => part.chapters.some((ch) => ch.href === c.href)).map((c) => c.href))]
          .map((h) => `<a href="@@ROOT@@${esc(h)}#ref-${esc(k)}">${esc(ui.chapterShort(book.pages.find((x) => x.href === h)!.num))}</a>`).join(" ");
        return `<li id="ref-${esc(k)}">${entryHtml(bib.get(k)!, labelsByKey.get(k), lang)} <span class="ref-back">${where}</span></li>`;
      }).join("\n");
      extra += `<section class="references"><h2 id="references" class="unnumbered">${esc(ui.referencesForPart(partName(lang, part.num)))}</h2><p class="refs-note">${esc(ui.worksInPart(partKeys.size))}</p><ol class="ref-list">\n${items}\n</ol></section>\n`;
    }
    r.body += extra;
    r.env.toc.push({ level: 2, id: "chapters", num: "", inline: null });
  }

  // The landing page carries the whole table of contents: every part with
  // its chapters, set in columns so the book's shape is visible at a glance.
  for (const home of book.front.filter((p) => p.kind === "index")) {
    const r = rendered.get(home.href)!;
    const li = (c: Page) => `<li><a href="@@ROOT@@${esc(c.href)}"><span class="bt-n">${esc(c.num)}</span><span>${titles.get(c.href)}</span></a></li>`;
    const sections = book.parts.map((part) => {
      const head = `<span class="bt-num">${esc(partName(lang, part.num))}</span>${esc(part.title)}`;
      return `<section class="bt-part"><h3>${part.intro ? `<a href="@@ROOT@@${esc(part.intro.href)}">${head}</a>` : head}</h3><ol>${part.chapters.map(li).join("")}</ol></section>`;
    });
    if (book.appendices.length) sections.push(`<section class="bt-part"><h3><span class="bt-num">${esc(ui.appendices)}</span></h3><ol>${book.appendices.map(li).join("")}</ol></section>`);
    const tocBlock = `<nav class="book-toc" aria-label="${esc(ui.toc)}">${sections.join("\n")}</nav>\n`;
    r.body = r.body.includes("<!--CONTENTS-->") ? r.body.replace("<!--CONTENTS-->", tocBlock) : r.body + `<h2 id="contents" class="unnumbered">${esc(ui.contents)}</h2>${tocBlock}`;
  }

  // The bibliography page lists every cited work with links to where it is cited.
  const biblio = book.back.find((b) => /bibliography\.html$/.test(b.href));
  if (biblio) {
    const r = rendered.get(biblio.href)!;
    const keys = [...citedIn.keys()].filter((k) => bib.has(k)).sort(byAuthor);
    const item = (k: string) => {
      const seen = new Set<string>();
      const where = (citedIn.get(k) ?? []).filter((c) => { const id = `${c.href}#${c.sec}`; if (seen.has(id)) return false; seen.add(id); return true; })
        .map((c) => { const pg = book.pages.find((x) => x.href === c.href)!; const lab = c.num ? `§${c.num}` : pg.num ? ui.chapterShort(pg.num) : stripTags(titles.get(pg.href) ?? ""); return `<a href="@@ROOT@@${esc(c.href)}${c.sec ? `#${esc(c.sec)}` : ""}">${esc(lab)}</a>`; }).join(" ");
      return `<li id="ref-${esc(k)}">${entryHtml(bib.get(k)!, labelsByKey.get(k), lang)} <span class="ref-back">${where}</span></li>`;
    };
    // Grouped by first letter, with a jump bar and a filter box.
    const letter = (k: string) => { const c = labelsByKey.get(k)!.authors.normalize("NFKD").replace(/[^A-Za-z]/g, "")[0]?.toUpperCase(); return c ?? "#"; };
    const groups = new Map<string, string[]>();
    for (const k of keys) groups.set(letter(k), [...(groups.get(letter(k)) ?? []), k]);
    const letters = [...groups.keys()].sort();
    const az = letters.map((l) => `<a href="#bib-${l}">${l}</a>`).join("");
    const lists = letters.map((l) => `<section class="bib-group"><h2 id="bib-${l}" class="unnumbered bib-letter">${l}</h2><ol class="ref-list ref-list-all">\n${groups.get(l)!.map(item).join("\n")}\n</ol></section>`).join("\n");
    r.body += `<div class="bib-tools"><input class="bib-filter" type="search" placeholder="${esc(ui.filterBib(keys.length))}" aria-label="${esc(ui.filterBibLabel)}"><nav class="bib-az" aria-label="${esc(ui.jumpToLetter)}">${az}</nav><span class="bib-count" aria-live="polite"></span></div>\n${lists}\n`;
  }

  errors.push(...parsed.flatMap((x) => x.env.errors));
  return { lang, book, md, rendered, titles, labelsByKey, errors };
}

type Edition = ReturnType<typeof renderEdition>;

// Write one edition's pages and its search index. `alt` maps a page's path
// to the same page in the other edition, when that one exists and is translated.
function writeEdition(ed: Edition, version: string, alt: (href: string) => string | undefined) {
  const { book, md, rendered, titles } = ed;
  const ui = UI[ed.lang];
  const order = book.pages;
  const search: Array<{ h: string; t: string; s: string; x: string }> = [];
  order.forEach((p, i) => {
    const r = rendered.get(p.href)!;
    const toc = r.env.toc.map((t) => ({ level: t.level, id: t.id, num: t.num, html: t.inline ? md.renderer.renderInline(t.inline.children, md.options, r.env) : t.id === "references" ? esc(ui.references) : esc(ui.chaptersInPart) }));
    const label = (q: Page) => ui.pagerLabel(q.kind, q.num, titles.get(q.href) ?? "", q.kind === "part" ? q.num : q.part?.num);
    const html = shell({
      book, page: p, title: stripTags(titles.get(p.href) ?? ""), titleHtml: titles.get(p.href) ?? "", body: r.body, toc,
      prev: i > 0 ? { href: order[i - 1].href, label: label(order[i - 1]) } : undefined,
      next: i < order.length - 1 ? { href: order[i + 1].href, label: label(order[i + 1]) } : undefined,
      chapterTitles: titles, description: p.meta.synopsis, alt: p.untranslated ? undefined : alt(p.href), untranslated: p.untranslated,
    });
    const depth = p.href.split("/").length - 1;
    const root = depth ? "../".repeat(depth) : "./";
    const out = join(OUT, p.href);
    mkdirSync(dirname(out), { recursive: true });
    writeFileSync(out, html.replaceAll("@@ROOT@@", root).replaceAll("@@VERSION@@", version));
    // Search records: one per h2/h3 section.
    const pageTitle = stripTags(titles.get(p.href) ?? "");
    const chunks = r.body.split(/(?=<h[23] id=")/);
    for (const c of chunks) {
      const m = /^<h([23]) id="([^"]+)"[^>]*>([\s\S]*?)<\/h\1>/.exec(c);
      const text = stripTags(c.replace(/<details class="sec-refs">[\s\S]*?<\/details>/g, "").replace(/<figure[\s\S]*?<\/figure>/g, "")).slice(0, 700);
      search.push({ h: p.href + (m ? `#${m[2]}` : ""), t: pageTitle, s: m ? stripTags(m[3]).replace(/#$/, "").trim() : "", x: text });
    }
  });
  writeFileSync(`${OUT}/assets/search-${ed.lang}.json`, JSON.stringify(search));
  if (existsSync(`${ed.lang}/images`)) cpSync(`${ed.lang}/images`, `${OUT}/${ed.lang}/images`, { recursive: true });
}

async function main() {
  const bib = loadBibs();
  const figErrors: string[] = [];
  const figures = await loadFigures(figErrors);
  // Every edition whose book.yml exists; --only <lang> builds just one.
  const onlyIdx = process.argv.indexOf("--only");
  const only = onlyIdx > 0 ? process.argv[onlyIdx + 1] : undefined;
  const langs = (["en", "zh"] as Lang[]).filter((l) => existsSync(`${l}/book.yml`) && (!only || l === only));
  const editions = langs.map((l) => renderEdition(l, bib, figures));

  // Errors.
  const errors = [...figErrors, ...editions.flatMap((e) => e.errors)];
  if (errors.length) {
    console.error(`\n${errors.length} error(s):`);
    for (const e of [...new Set(errors)]) console.error("  " + e);
    if (!lenient) process.exit(1);
  }

  // Write pages.
  rmSync(OUT, { recursive: true, force: true });
  mkdirSync(`${OUT}/assets`, { recursive: true });
  const version = createHash("sha1").update(String(Date.now())).digest("hex").slice(0, 8);
  const translated = new Map(editions.map((e) => [e.lang, new Set(e.book.pages.filter((p) => !p.untranslated).map((p) => p.href))]));
  for (const ed of editions) {
    const other = editions.find((e) => e.lang !== ed.lang);
    writeEdition(ed, version, (href) => {
      if (!other) return undefined;
      const h = href.replace(/^[a-z]+\//, `${other.lang}/`);
      return translated.get(other.lang)!.has(h) ? h : undefined;
    });
  }
  const first = editions[0].book;
  writeFileSync(`${OUT}/assets/cover-art.svg`, coverArt());
  writeFileSync(`${OUT}/index.html`, `<!doctype html><meta charset="utf-8"><title>${esc(first.title)}</title><meta http-equiv="refresh" content="0; url=${first.lang}/index.html"><link rel="canonical" href="${first.lang}/index.html"><a href="${first.lang}/index.html">${esc(first.title)}</a>\n`);

  // Assets.
  cpSync("node_modules/katex/dist/katex.min.css", `${OUT}/assets/katex/katex.min.css`);
  cpSync("node_modules/katex/dist/fonts", `${OUT}/assets/katex/fonts`, { recursive: true });
  cpSync("src/styles/book.css", `${OUT}/assets/book.css`);
  if (existsSync("src/styles/favicon.svg")) cpSync("src/styles/favicon.svg", `${OUT}/assets/favicon.svg`);
  for (const ed of editions) if (existsSync(`src/styles/og-${ed.lang}.png`)) cpSync(`src/styles/og-${ed.lang}.png`, `${OUT}/assets/og-${ed.lang}.png`);
  await esbuild.build({
    entryPoints: { main: "src/client/main.ts" },
    bundle: true,
    splitting: true,
    format: "esm",
    outdir: `${OUT}/assets`,
    chunkNames: "chunks/[name]-[hash]",
    minify: !process.argv.includes("--dev"),
    sourcemap: process.argv.includes("--dev") ? "inline" : false,
    target: "es2022",
    logLevel: "warning",
    plugins: [{
      name: "figure-registry",
      setup(b) {
        b.onResolve({ filter: /^virtual:figure-registry$/ }, () => ({ path: "figure-registry", namespace: "virtual" }));
        b.onLoad({ filter: /.*/, namespace: "virtual" }, () => ({ contents: registrySource(new Set(figures.keys())), loader: "ts", resolveDir: "src/figures" }));
      },
    }],
  });

  for (const ed of editions) {
    const { book } = ed;
    const nCh = book.pages.filter((p) => p.kind === "chapter").length;
    const status = book.pages.reduce<Record<string, number>>((a, p) => { if (p.kind === "chapter") a[p.meta.status ?? "done"] = (a[p.meta.status ?? "done"] ?? 0) + 1; return a; }, {});
    const untr = book.pages.filter((p) => p.untranslated).length;
    console.log(`built ${ed.lang}: ${book.pages.length} pages (${nCh} chapters: ${Object.entries(status).map(([k, v]) => `${v} ${k}`).join(", ")}${untr ? `; ${untr} pages not translated yet` : ""}), ${ed.labelsByKey.size} cited works`);
  }
  console.log(`${figures.size} figure modules, in ${Math.round(performance.now() - t0)} ms -> ${relative(".", OUT)}/`);
}

main().catch((e) => { console.error(e); process.exit(1); });
