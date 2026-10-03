// The page shell: header, the book's table of contents, the article, the
// page's own outline, and the pager. Every link is written root-relative with
// the @@ROOT@@ marker, which the build replaces by the right number of "../"
// for each page, so the site works from any path prefix (changkun.de/bobook/,
// a local file server, a preview deploy).

import type { Book, Page } from "./book.ts";
import { UI, partName } from "./i18n.ts";

const esc = (s: string) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

export interface Shell {
  book: Book;
  page: Page;
  title: string; // plain text title for <title>
  titleHtml: string; // rendered h1 content (for the sidebar and pager)
  body: string; // article HTML
  toc: Array<{ level: number; id: string; num: string; html: string }>;
  prev?: { href: string; label: string };
  next?: { href: string; label: string };
  chapterTitles: Map<string, string>; // href -> title html, for the sidebar
  description?: string;
  alt?: string; // href of the same page in the other edition, if it exists
  untranslated?: boolean; // a translated edition shows the original for this page
}

function sidebar(s: Shell): string {
  const { book, page } = s;
  const item = (p: Page, label: string) => {
    const cur = p.href === page.href ? ` aria-current="page"` : "";
    // Only unwritten chapters are marked; a mark on every draft is noise.
    const st = p.meta.status === "outline" ? `<span class="nav-status" title="not written yet"></span>` : "";
    return `<li><a href="@@ROOT@@${esc(p.href)}"${cur}>${label}${st}</a></li>`;
  };
  const chap = (p: Page) => item(p, `${p.num ? `<span class="nav-num">${esc(p.num)}</span>` : ""}<span class="nav-title">${s.chapterTitles.get(p.href) ?? esc(p.file)}</span>`);
  let html = `<ol class="nav-list nav-front">${book.front.filter((p) => p.kind !== "index").map(chap).join("")}</ol>`;
  for (const part of book.parts) {
    const open = part.chapters.some((c) => c.href === page.href) || part.intro?.href === page.href;
    const head = part.intro
      ? `<a class="nav-part-link" href="@@ROOT@@${esc(part.intro.href)}"${part.intro.href === page.href ? ` aria-current="page"` : ""}><span class="nav-part-num">${esc(part.num)}</span><span class="nav-part-title">${esc(part.title)}</span></a>`
      : `<span class="nav-part-link"><span class="nav-part-num">${esc(part.num)}</span><span class="nav-part-title">${esc(part.title)}</span></span>`;
    html += `<details class="nav-part"${open ? " open" : ""}><summary>${head}</summary><ol class="nav-list">${part.chapters.map(chap).join("")}</ol></details>`;
  }
  if (book.appendices.length) {
    const open = book.appendices.some((c) => c.href === page.href);
    html += `<details class="nav-part"${open ? " open" : ""}><summary><span class="nav-part-link"><span class="nav-part-num"></span><span class="nav-part-title">${esc(UI[book.lang].appendices)}</span></span></summary><ol class="nav-list">${book.appendices.map(chap).join("")}</ol></details>`;
  }
  if (book.back.length) html += `<ol class="nav-list nav-back">${book.back.map(chap).join("")}</ol>`;
  return html;
}

// A new GitHub issue, prefilled with the page it is about.
function issueUrl(book: Book, p: Page, title: string): string {
  const where = book.url ? `${book.url}${p.href}` : p.href;
  const t = `${book.lang === "zh" ? "[zh] " : ""}${title}: `;
  const body = `Page: ${where}\n\nWhat is wrong (a quote of the passage helps):\n\n`;
  return `${book.repo}/issues/new?title=${encodeURIComponent(t)}&body=${encodeURIComponent(body)}`;
}

export function page(s: Shell): string {
  const { book, page: p } = s;
  const ui = UI[book.lang];
  const tocHtml = s.toc.length && p.kind !== "index" && p.kind !== "part"
    ? `<nav class="toc" aria-label="${esc(ui.onThisPage)}"><div class="toc-head">${esc(ui.onThisPage)}</div><ol>${s.toc.map((t) => `<li class="toc-l${t.level}"><a href="#${esc(t.id)}">${t.num ? `<span class="toc-num">${esc(t.num)}</span> ` : ""}${t.html}</a></li>`).join("")}</ol></nav>`
    : "";
  const crumbs = p.part ? `<a href="@@ROOT@@${esc(p.part.intro?.href ?? p.href)}">${esc(partName(book.lang, p.part.num))}${book.lang === "zh" ? "：" : ": "}${esc(p.part.title)}</a>` : "";
  const note = p.meta.status ? ui.status[p.meta.status] : undefined;
  const status = (note ? `<p class="status-line status-${esc(p.meta.status!)}" role="note"><span class="status-pill">${esc(note[0])}</span> ${esc(note[1])}</p>` : "")
    + (s.untranslated ? `<p class="status-line status-untranslated" role="note">${esc(ui.untranslated)}</p>` : "");
  const pager = `<nav class="pager" aria-label="${esc(ui.chapterNav)}">${s.prev ? `<a class="pager-prev" href="@@ROOT@@${esc(s.prev.href)}"><span class="pager-dir" aria-hidden="true">←</span><span class="pager-label">${s.prev.label}</span></a>` : "<span></span>"}${s.next ? `<a class="pager-next" href="@@ROOT@@${esc(s.next.href)}"><span class="pager-label">${s.next.label}</span><span class="pager-dir" aria-hidden="true">→</span></a>` : "<span></span>"}</nav>`;
  const desc = s.description ?? book.subtitle;
  return `<!doctype html>
<html lang="${book.lang}" data-page-kind="${esc(p.kind)}"${tocHtml ? "" : " data-no-toc"}>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(s.title)}${p.kind === "index" ? "" : ` · ${esc(book.title)}`}</title>
<meta name="description" content="${esc(desc)}">
<meta property="og:title" content="${esc(s.title)}">
<meta property="og:description" content="${esc(desc)}">
${book.url ? `<meta property="og:type" content="website">
<meta property="og:site_name" content="${esc(book.title)}">
<meta property="og:locale" content="${book.lang === "zh" ? "zh_CN" : "en_US"}">
<meta property="og:url" content="${esc(book.url + p.href.replace(/index\.html$/, ""))}">
<meta property="og:image" content="${esc(`${book.url}assets/og-${book.lang}.png`)}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="${esc(`${book.title}${book.lang === "zh" ? "：" : ": "}${book.subtitle}`)}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="${esc(`${book.url}assets/og-${book.lang}.png`)}">
${s.alt ? `<link rel="alternate" hreflang="${book.lang === "en" ? "zh" : "en"}" href="${esc(book.url + s.alt)}">` : ""}` : ""}
<meta name="author" content="${esc(book.author)}">
<link rel="icon" href="@@ROOT@@assets/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400;1,8..60,600&display=swap">
${book.lang === "zh" ? `<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;600;700&family=Noto+Serif+SC:wght@400;600;700&display=swap">\n` : ""}<link rel="stylesheet" href="@@ROOT@@assets/katex/katex.min.css">
<link rel="stylesheet" href="@@ROOT@@assets/book.css?v=@@VERSION@@">
<script>try{var t=localStorage.getItem("bobook-theme");if(t)document.documentElement.dataset.theme=t}catch(e){}</script>
<script type="module" src="@@ROOT@@assets/main.js?v=@@VERSION@@"></script>
</head>
<body>
<a class="skip" href="#content">${esc(ui.skip)}</a>
<header class="topbar">
  <button class="menu-btn" type="button" aria-label="${esc(ui.toc)}" aria-controls="sidebar" aria-expanded="false"><span></span></button>
  <a class="brand" href="@@ROOT@@${esc(book.front[0]?.href ?? "")}">${esc(book.title)}</a>
  <div class="crumbs">${crumbs}</div>
  <div class="topbar-tools">
    <button class="search-btn" type="button" aria-label="${esc(ui.search)}"><svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true"><circle cx="8.5" cy="8.5" r="5.5" fill="none" stroke="currentColor" stroke-width="1.8"/><path d="M13 13l4.5 4.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg></button>
    ${s.alt ? `<a class="lang-btn" href="@@ROOT@@${esc(s.alt)}" hreflang="${book.lang === "en" ? "zh" : "en"}" title="${esc(ui.otherEdition.title)}">${esc(ui.otherEdition.label)}</a>` : ""}
    <button class="theme-btn" type="button" aria-label="${esc(ui.theme)}"><svg viewBox="0 0 20 20" width="18" height="18" aria-hidden="true"><path d="M10 2.5a7.5 7.5 0 1 0 0 15z" fill="currentColor"/><circle cx="10" cy="10" r="7.5" fill="none" stroke="currentColor" stroke-width="1.6"/></svg></button>
  </div>
</header>
<div class="layout">
<nav id="sidebar" class="sidebar" aria-label="${esc(ui.bookContents)}">${sidebar(s)}</nav>
<main id="content" class="main">
<article class="article page-${esc(p.kind)}">
${status}
${s.body}
</article>
${pager}
<footer class="site-footer">${esc(book.title)}${book.lang === "zh" ? "：" : ": "}${esc(book.subtitle)} · © 2026 ${esc(book.author)} · <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/" rel="license">CC BY-NC-SA 4.0</a> · <a href="@@ROOT@@${esc(book.back.find((b) => /bibliography/.test(b.href))?.href ?? "")}">${esc(ui.bibliography)}</a>${book.repo ? ` · <a href="${esc(book.repo)}">${esc(ui.source)}</a> · <a href="${esc(issueUrl(book, p, s.title))}">${esc(ui.reportIssue)}</a>` : ""}<span class="page-stat"> · PV/UV <span id="urlstat-page-pv"></span>/<span id="urlstat-page-uv"></span></span></footer>
</main>
${tocHtml}
</div>
<div class="search-panel" hidden><div class="search-box"><input type="search" placeholder="${esc(ui.search)}" aria-label="${esc(ui.search)}"><ol class="search-results"></ol></div></div>
</body>
</html>
`;
}
