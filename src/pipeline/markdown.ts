// The book's Markdown dialect, built on markdown-it. It follows Pandoc and
// Quarto syntax where they have one, so a chapter reads naturally as source:
//
//   # Chapter title {#sec-gaussian}          numbered headings, stable ids
//   ## Section {#sec-conditioning}
//   $x$  and  $$ ... $$ {#eq-bayes}          KaTeX, numbered display equations
//   [@chu2005preference]  [@a; @b, p. 3]     parenthetical citations
//   @chu2005preference                       narrative citation
//   [-@chu2005preference]                    year only
//   @sec-conditioning  @fig-gp-prior  @eq-bayes  @def-gp   cross-references
//   ::: {.definition #def-gp title="Gaussian process"}  ... :::   callouts
//   ```{figure}  //| figure: gp-prior ...   interactive figure modules
//
// Parsing runs for every chapter first, so numbers, labels, and the set of
// cited works are known before any page is rendered; rendering then resolves
// cross-references across chapters and builds the per-section reference lists.

import MarkdownIt from "markdown-it";
import attrs from "markdown-it-attrs";
import katex from "katex";
import hljs from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import typescript from "highlight.js/lib/languages/typescript";
import bash from "highlight.js/lib/languages/bash";
import yamlLang from "highlight.js/lib/languages/yaml";
import json from "highlight.js/lib/languages/json";

hljs.registerLanguage("python", python);
hljs.registerLanguage("typescript", typescript);
hljs.registerLanguage("ts", typescript);
hljs.registerLanguage("bash", bash);
hljs.registerLanguage("sh", bash);
hljs.registerLanguage("yaml", yamlLang);
hljs.registerLanguage("json", json);
import { MACROS } from "./macros.ts";
import type { Bib } from "./cite.ts";
import { inlineLabel } from "./cite.ts";
import { UI, partName } from "./i18n.ts";
import type { AnyFigure } from "../figures/types.ts";
import { resolve } from "../figures/lib/params.ts";
import { renderStatic } from "../figures/static.ts";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface Label {
  kind: string; // sec, fig, eq, tbl, def, thm, ex, exr, alg, ...
  num: string; // "3.2", "A.1", "" for unnumbered
  href: string; // root-relative: "en/foundations/03-gaussian.html#sec-x"
  level?: number; // headings
  chapterKind?: "chapter" | "appendix" | "front" | "part";
  title?: string;
}

export interface ChapterCtx {
  num: string; // "3", "A", or "" for unnumbered front matter
  kind: "chapter" | "appendix" | "front" | "part";
  href: string; // root-relative page path
  file: string;
}

export interface TocItem { level: number; id: string; num: string; inline: Any }

export interface ChapterEnv {
  ch: ChapterCtx;
  labels: Map<string, Label>; // shared across the book
  cites: string[]; // keys in order of appearance (parse phase)
  errors: string[];
  toc: TocItem[];
  title?: Any; // inline token of the h1
  titleText?: string;
  counters: Record<string, number>;
  // render phase
  section?: string;
  sectionCites: Map<string, string[]>;
  sectionOrder: Array<{ id: string; num: string; text: string }>;
  citeLabels?: Map<string, { authors: string; year: string; parts?: import("./cite.ts").Authors }>;
}

export function newEnv(ch: ChapterCtx, labels: Map<string, Label>): ChapterEnv {
  return { ch, labels, cites: [], errors: [], toc: [], counters: {}, sectionCites: new Map(), sectionOrder: [] };
}

const FIGURE_INFO = "@figure";
const XREF_PREFIX = /^(sec|fig|eq|tbl|def|thm|lem|prop|cor|ex|exr|alg)-/;

const CALLOUTS: Record<string, { label: string; prefix?: string; numbered?: boolean }> = {
  definition: { label: "Definition", prefix: "def", numbered: true },
  theorem: { label: "Theorem", prefix: "thm", numbered: true },
  lemma: { label: "Lemma", prefix: "lem", numbered: true },
  proposition: { label: "Proposition", prefix: "prop", numbered: true },
  corollary: { label: "Corollary", prefix: "cor", numbered: true },
  example: { label: "Example", prefix: "ex", numbered: true },
  exercise: { label: "Exercise", prefix: "exr", numbered: true },
  algorithm: { label: "Algorithm", prefix: "alg", numbered: true },
  table: { label: "Table", prefix: "tbl", numbered: true },
  solution: { label: "Solution" },
  proof: { label: "Proof" },
  derivation: { label: "Derivation" },
  note: { label: "Note" },
  aside: { label: "Going deeper" },
  pitfall: { label: "Pitfall" },
  keyidea: { label: "Key idea" },
  frontier: { label: "Research status" },
  recap: { label: "Recap" },
  code: { label: "In code" },
};

const escHtml = (s: string) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

export function slugify(s: string): string {
  return s.toLowerCase().replace(/<[^>]+>/g, "").replace(/\$[^$]*\$/g, "").replace(/[^\p{L}\p{N}\s-]/gu, "").trim().replace(/\s+/g, "-").replace(/-+/g, "-").slice(0, 60) || "section";
}

function parseInfo(params: string): { kind: string; id?: string; classes: string[]; kv: Record<string, string> } | undefined {
  const s = params.trim();
  if (/^\w[\w-]*$/.test(s)) return { kind: s, classes: [s], kv: {} };
  const m = /^\{(.*)\}$/.exec(s);
  if (!m) return undefined;
  const body = m[1];
  const classes: string[] = [];
  let id: string | undefined;
  const kv: Record<string, string> = {};
  const re = /\.([\w-]+)|#([\w:.-]+)|([\w-]+)=("([^"]*)"|'([^']*)'|(\S+))/g;
  let mm: RegExpExecArray | null;
  while ((mm = re.exec(body))) {
    if (mm[1]) classes.push(mm[1]);
    else if (mm[2]) id = mm[2];
    else if (mm[3]) kv[mm[3]] = mm[5] ?? mm[6] ?? mm[7] ?? "";
  }
  const kind = classes.find((c) => c in CALLOUTS) ?? classes[0] ?? "note";
  return { kind, id, classes, kv };
}

export interface MdOptions {
  bib: Bib;
  figures: Map<string, AnyFigure>;
  lang: "en" | "zh";
}

export function createMarkdown(o: MdOptions): Any {
  const ui = UI[o.lang];
  // Code is highlighted at build time, so pages carry no highlighter script.
  const md = new MarkdownIt({
    html: true, linkify: false, typographer: false,
    highlight: (code: string, lang: string) => {
      if (lang && hljs.getLanguage(lang)) return hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
      return "";
    },
  });
  // A line break inside a paragraph renders as a space. Chinese has no spaces
  // between words, so in the Chinese edition a break between two Chinese
  // characters, or next to full-width punctuation, is dropped; next to Latin
  // letters or digits it stays a space, as the style rules want. Translators
  // can then wrap source lines freely.
  if (o.lang === "zh") {
    const HAN = /[\u4e00-\u9fff]/, PUNCT = /[\u3000-\u303f\uff00-\uffef“”‘’]/;
    md.renderer.rules.softbreak = (tokens: Any, idx: number) => {
      const b = (tokens[idx - 1]?.content ?? "").slice(-1), a = (tokens[idx + 1]?.content ?? "").slice(0, 1);
      if (PUNCT.test(b) || PUNCT.test(a) || (HAN.test(b) && HAN.test(a))) return "";
      return " "; // Latin and Latin, or Chinese and Latin, which the style rules space
    };
  }
  md.use(attrs, { leftDelimiter: "{", rightDelimiter: "}", allowedAttributes: ["id", "class", "title", "width", "style"] });
  // A ```{figure} fence would otherwise be read by markdown-it-attrs as an
  // attribute block; rename its info string before that plugin runs.
  md.core.ruler.before("curly_attributes", "figure_fence", (state: Any) => {
    for (const t of state.tokens) if (t.type === "fence" && t.info.trim() === "{figure}") t.info = FIGURE_INFO;
  });

  // ---------------------------------------------------------------- math
  const tex = (src: string, display: boolean, env: Any) => {
    try {
      return katex.renderToString(src, { displayMode: display, throwOnError: true, macros: { ...MACROS }, strict: "ignore", trust: false, output: "htmlAndMathml" });
    } catch (e) {
      env.errors.push(`${env.ch.file}: KaTeX: ${(e as Error).message.split("\n")[0]} in "${src.slice(0, 80)}"`);
      return `<code class="math-error">${escHtml(src)}</code>`;
    }
  };

  md.inline.ruler.after("escape", "math_inline", (state: Any, silent: boolean) => {
    const src: string = state.src;
    const pos: number = state.pos;
    if (src.charCodeAt(pos) !== 0x24 /* $ */ || src.charCodeAt(pos + 1) === 0x24) return false;
    const next = src[pos + 1];
    if (!next || /\s/.test(next)) return false;
    let end = pos + 1;
    for (;;) {
      end = src.indexOf("$", end);
      if (end < 0) return false;
      if (src[end - 1] === "\\" || /\s/.test(src[end - 1]) || /\d/.test(src[end + 1] ?? "")) { end++; continue; }
      break;
    }
    if (!silent) {
      const t = state.push("math_inline", "math", 0);
      t.content = src.slice(pos + 1, end);
    }
    state.pos = end + 1;
    return true;
  });

  const DISPLAY_END = /^(.*?)\$\$\s*(?:\{#([\w:.-]+)\})?\s*$/;
  md.block.ruler.before("fence", "math_block", (state: Any, startLine: number, endLine: number, silent: boolean) => {
    const start = state.bMarks[startLine] + state.tShift[startLine];
    const max = state.eMarks[startLine];
    if (state.sCount[startLine] - state.blkIndent >= 4) return false;
    if (state.src.slice(start, start + 2) !== "$$") return false;
    const first = state.src.slice(start + 2, max);
    let content = "", label: string | undefined, next = startLine, found = false;
    const single = first.trim() ? DISPLAY_END.exec(first) : null;
    if (single) { content = single[1]; label = single[2]; found = true; }
    else {
      const lines = [first];
      for (next = startLine + 1; next < endLine; next++) {
        const s = state.bMarks[next] + state.tShift[next];
        const line = state.src.slice(s, state.eMarks[next]);
        const m = DISPLAY_END.exec(line);
        if (m) { lines.push(m[1]); label = m[2]; found = true; break; }
        lines.push(line);
      }
      content = lines.join("\n");
    }
    if (!found) return false;
    if (silent) return true;
    state.line = next + 1;
    const t = state.push("math_block", "math", 0);
    t.block = true;
    t.content = content.trim();
    t.map = [startLine, state.line];
    t.meta = { label };
    return true;
  }, { alt: ["paragraph", "reference", "blockquote", "list"] });

  md.renderer.rules.math_inline = (tokens: Any, idx: number, _o: Any, env: Any) => tex(tokens[idx].content, false, env);
  md.renderer.rules.math_block = (tokens: Any, idx: number, _o: Any, env: Any) => {
    const t = tokens[idx];
    const body = tex(t.content, true, env);
    if (t.meta?.label) {
      return `<div class="math-display" id="${escHtml(t.meta.label)}"><div class="math-body">${body}</div><span class="eq-num">(${escHtml(t.meta.num ?? "?")})</span></div>\n`;
    }
    return `<div class="math-display"><div class="math-body">${body}</div></div>\n`;
  };

  // ---------------------------------------------------------- callouts
  md.block.ruler.before("fence", "callout", (state: Any, startLine: number, endLine: number, silent: boolean) => {
    let start = state.bMarks[startLine] + state.tShift[startLine];
    let max = state.eMarks[startLine];
    if (state.src.charCodeAt(start) !== 0x3a) return false;
    let pos = start;
    while (pos < max && state.src.charCodeAt(pos) === 0x3a) pos++;
    if (pos - start < 3) return false;
    const params = state.src.slice(pos, max).trim();
    if (!params) return false;
    const info = parseInfo(params);
    if (!info) return false;
    if (silent) return true;
    let next = startLine;
    let depth = 1;
    let closed = false;
    let inFence = false;
    for (;;) {
      next++;
      if (next >= endLine) break;
      start = state.bMarks[next] + state.tShift[next];
      max = state.eMarks[next];
      if (start < max && state.sCount[next] < state.blkIndent) break;
      const line = state.src.slice(start, max);
      if (/^(```|~~~)/.test(line)) { inFence = !inFence; continue; }
      if (inFence) continue;
      const m = /^(:{3,})\s*(.*)$/.exec(line);
      if (!m) continue;
      if (m[2].trim()) { depth++; continue; }
      depth--;
      if (depth === 0) { closed = true; break; }
    }
    const oldParent = state.parentType;
    const oldLineMax = state.lineMax;
    state.parentType = "callout";
    state.lineMax = next;
    const open = state.push("callout_open", "div", 1);
    open.block = true;
    open.meta = info;
    open.map = [startLine, next];
    state.md.block.tokenize(state, startLine + 1, next);
    const close = state.push("callout_close", "div", -1);
    close.block = true;
    close.meta = info;
    state.parentType = oldParent;
    state.lineMax = oldLineMax;
    state.line = next + (closed ? 1 : 0);
    return true;
  }, { alt: ["paragraph", "reference", "blockquote", "list"] });

  md.renderer.rules.callout_open = (tokens: Any, idx: number, _o: Any, env: Any) => {
    const m = tokens[idx].meta;
    const def = CALLOUTS[m.kind];
    const idAttr = m.id ? ` id="${escHtml(m.id)}"` : "";
    const title = m.titleTokens ? md.renderer.renderInline(m.titleTokens, md.options, env) : "";
    const extra = m.classes.filter((c: string) => c !== m.kind).map((c: string) => ` ${escHtml(c)}`).join("");
    if (!def) return `<div class="block block-${escHtml(m.kind)}${extra}"${idAttr}>\n`;
    const word = ui.callout[m.kind] ?? def.label;
    const label = def.numbered && m.num ? (ui.kind[def.prefix ?? ""]?.(m.num) ?? `${word} ${m.num}`) : word;
    const head = `<span class="callout-label">${escHtml(m.kv.label ?? label)}</span>${title ? ` <span class="callout-title">${title}</span>` : ""}`;
    const collapsed = m.kind === "solution" || m.kv.collapsed === "true";
    if (collapsed) return `<details class="callout callout-${m.kind}${extra}"${idAttr}><summary class="callout-head">${head}</summary><div class="callout-body">\n`;
    const tag = m.kind === "aside" ? "aside" : "div";
    return `<${tag} class="callout callout-${m.kind}${extra}"${idAttr}><div class="callout-head">${head}</div><div class="callout-body">\n`;
  };
  md.renderer.rules.callout_close = (tokens: Any, idx: number) => {
    const m = tokens[idx].meta;
    const def = CALLOUTS[m.kind];
    if (!def) return `</div>\n`;
    const collapsed = m.kind === "solution" || m.kv.collapsed === "true";
    if (collapsed) return `</div></details>\n`;
    return `</div></${m.kind === "aside" ? "aside" : "div"}>\n`;
  };

  // ---------------------------------------------------- citations, xrefs
  const KEY = "[A-Za-z0-9_][A-Za-z0-9_:.\\-+]*[A-Za-z0-9_]|[A-Za-z0-9_]";
  const ITEM = new RegExp(`^\\s*([^@]*?)([-+]?)@(${KEY})(?:,\\s*([^;]+?))?\\s*$`);

  md.inline.ruler.before("link", "cite_bracket", (state: Any, silent: boolean) => {
    const src: string = state.src;
    const pos: number = state.pos;
    if (src.charCodeAt(pos) !== 0x5b /* [ */) return false;
    const close = src.indexOf("]", pos);
    if (close < 0) return false;
    const body = src.slice(pos + 1, close);
    if (!body.includes("@") || body.includes("[")) return false;
    if (src[close + 1] === "(" || src[close + 1] === "[") return false;
    const items = body.split(";").map((p) => ITEM.exec(p));
    if (items.some((m) => !m)) return false;
    const canon = (k: string) => o.bib.get(k)?.key ?? k;
    const parsed = items.map((m) => ({ prefix: m![1].trim(), suppress: m![2] === "-", withTitle: m![2] === "+", key: XREF_PREFIX.test(m![3]) ? m![3] : canon(m![3]), locator: m![4]?.trim() }));
    if (!silent) {
      const env: ChapterEnv = state.env;
      const allXref = parsed.every((p) => XREF_PREFIX.test(p.key));
      if (allXref) {
        const t = state.push("xref_list", "", 0);
        t.meta = { items: parsed };
      } else {
        for (const p of parsed) if (!o.bib.has(p.key)) env.errors.push(`${env.ch.file}: unknown citation key "${p.key}"`);
        for (const p of parsed) env.cites.push(p.key);
        const t = state.push("cite", "", 0);
        t.meta = { items: parsed, mode: "paren" };
      }
    }
    state.pos = close + 1;
    return true;
  });

  md.inline.ruler.after("cite_bracket", "cite_narrative", (state: Any, silent: boolean) => {
    const src: string = state.src;
    const pos: number = state.pos;
    if (src.charCodeAt(pos) !== 0x40 /* @ */) return false;
    if (pos > 0 && /[\p{L}\p{N}_.\-]/u.test(src[pos - 1])) return false;
    const m = new RegExp(`^@(${KEY})`).exec(src.slice(pos));
    if (!m) return false;
    const key = m[1];
    if (XREF_PREFIX.test(key)) {
      if (!silent) { const t = state.push("xref", "", 0); t.meta = { id: key }; }
      state.pos = pos + 1 + key.length;
      return true;
    }
    if (!o.bib.has(key)) return false;
    if (!silent) {
      const ck = o.bib.get(key)!.key;
      state.env.cites.push(ck);
      const t = state.push("cite", "", 0);
      t.meta = { items: [{ key: ck, prefix: "", suppress: false }], mode: "narrative" };
    }
    state.pos = pos + 1 + key.length;
    return true;
  });

  const citeLink = (key: string, text: string) => `<a class="cite-link" href="#ref-${escHtml(key)}" data-cite="${escHtml(key)}">${escHtml(text)}</a>`;

  md.renderer.rules.cite = (tokens: Any, idx: number, _o: Any, env: Any) => {
    const { items, mode } = tokens[idx].meta;
    const sec = env.section ?? "_intro";
    for (const it of items) {
      const list = env.sectionCites.get(sec) ?? [];
      if (!list.includes(it.key)) list.push(it.key);
      env.sectionCites.set(sec, list);
    }
    const lab = (key: string) => env.citeLabels?.get(key) ?? { authors: "??", year: "??" };
    if (mode === "narrative") return `<span class="cite cite-narrative">${citeLink(items[0].key, inlineLabel(lab(items[0].key), "narrative", o.lang))}</span>`;
    const zh = o.lang === "zh";
    const parts = items.map((it: Any) => {
      const text = inlineLabel(lab(it.key), it.suppress ? "year" : "paren", o.lang);
      return `${it.prefix ? `${escHtml(it.prefix)} ` : ""}${citeLink(it.key, text)}${it.locator ? `${zh ? "，" : ", "}${escHtml(it.locator)}` : ""}`;
    });
    return zh ? `<span class="cite">（${parts.join("；")}）</span>` : `<span class="cite">(${parts.join("; ")})</span>`;
  };

  const xrefText = (l: Label, suppress: boolean) => {
    if (l.kind === "sec") {
      const name = l.level === 1 ? (l.chapterKind === "appendix" ? ui.appendix : l.chapterKind === "part" ? (n: string) => partName(o.lang, n) : ui.chapter) : ui.section;
      if (!l.num) return l.title ?? "";
      return suppress ? l.num : name(l.num);
    }
    if (l.kind === "eq") return suppress ? ui.eqSuppressed(l.num) : ui.equation(l.num);
    return suppress ? l.num : (ui.kind[l.kind]?.(l.num) ?? `${l.kind} ${l.num}`);
  };
  const xrefHtml = (id: string, suppress: boolean, env: ChapterEnv, withTitle = false) => {
    const l = env.labels.get(id);
    if (!l) { env.errors.push(`${env.ch.file}: unresolved cross-reference @${id}`); return `<span class="xref-missing">@${escHtml(id)}</span>`; }
    if (withTitle && l.title) {
      const t = md.renderInline(l.title, env as Any);
      return `<a class="xref xref-titled" href="@@ROOT@@${escHtml(l.href)}"><span class="xref-num">${escHtml(l.num || "")}</span> <span class="xref-title">${t}</span></a>`;
    }
    return `<a class="xref" href="@@ROOT@@${escHtml(l.href)}">${escHtml(xrefText(l, suppress))}</a>`;
  };
  md.renderer.rules.xref = (tokens: Any, idx: number, _o: Any, env: Any) => xrefHtml(tokens[idx].meta.id, false, env);
  md.renderer.rules.xref_list = (tokens: Any, idx: number, _o: Any, env: Any) =>
    tokens[idx].meta.items.map((it: Any) => `${it.prefix ? `${escHtml(it.prefix)} ` : ""}${xrefHtml(it.key, it.suppress, env, it.withTitle)}`).join(o.lang === "zh" ? "、" : ", ");

  // ------------------------------------------------------------ figures
  const defaultFence = md.renderer.rules.fence!;
  md.renderer.rules.fence = (tokens: Any, idx: number, opts: Any, env: Any, self: Any) => {
    const t = tokens[idx];
    if (t.info !== FIGURE_INFO) return defaultFence(tokens, idx, opts, env, self);
    const m = t.meta;
    if (!m || m.error) return `<pre class="fig-error">${escHtml(m?.error ?? "bad figure block")}</pre>`;
    const cap = m.capTokens ? md.renderer.renderInline(m.capTokens, md.options, env) : "";
 const num = m.num && !/\b(hero|map)\b/.test(m.cls ?? "") ? `<span class="fig-num">${escHtml(ui.figure(m.num))}</span> ` : "";
    const idAttr = m.label ? ` id="${escHtml(m.label)}"` : "";
    let body = "";
    if (m.module) {
      try {
        body = renderStatic(m.module, m.params, o.lang, m.label ?? `fig-${m.module.name}-${idx}`, m.t);
        const flags = `${m.autoplay ? ' data-autoplay="true"' : ""}${m.bare ? ' data-bare="true"' : ""}`;
        if (flags) body = body.replace('<div class="fig"', `<div class="fig${m.bare ? " fig-bare" : ""}"${flags}`);
      } catch (e) {
        env.errors.push(`${env.ch.file}: figure ${m.module.name}: ${(e as Error).message}`);
        body = `<pre class="fig-error">${escHtml((e as Error).message)}</pre>`;
      }
    } else if (m.src) {
      body = `<img class="fig-img" src="@@ROOT@@${escHtml(o.lang)}/${escHtml(m.src)}" alt="${escHtml(m.alt ?? "")}" loading="lazy">`;
    }
    return `<figure class="figure${m.cls ? ` ${escHtml(m.cls)}` : ""}"${idAttr}>${body}${cap || num ? `<figcaption>${num}${cap}</figcaption>` : ""}</figure>\n`;
  };

  // ------------------------------------------------------------ headings
  md.renderer.rules.heading_open = (tokens: Any, idx: number, _o: Any, env: Any) => {
    const t = tokens[idx];
    const level = Number(t.tag.slice(1));
    const id = t.attrGet("id");
    let pre = "";
    if (level <= 2) {
      // Close the previous section's reference list before a new section starts.
      if (level === 2 && env.section !== undefined) pre = `<!--SECREFS:${env.section}-->\n`;
      if (level === 2) {
        env.section = id;
        env.sectionOrder.push({ id, num: t.meta?.num ?? "", text: t.meta?.text ?? "" });
      } else env.section = "_intro";
    }
    const cls = t.attrGet("class");
    const num = t.meta?.num ? `<span class="sec-num">${escHtml(t.meta.num)}</span> ` : "";
    return `${pre}<${t.tag} id="${escHtml(id)}"${cls ? ` class="${escHtml(cls)}"` : ""}>${num}`;
  };
  md.renderer.rules.heading_close = (tokens: Any, idx: number) => {
    const t = tokens[idx];
    const open = tokens[idx - 2];
    const id = open?.attrGet?.("id");
    const anchor = id && t.tag !== "h1" ? ` <a class="anchor" href="#${escHtml(id)}" aria-label="Link to this section">#</a>` : "";
    return `${anchor}</${t.tag}>\n`;
  };

  md.renderer.rules.table_open = () => `<div class="table-wrap"><table>\n`;
  md.renderer.rules.table_close = () => `</table></div>\n`;

  // -------------------------------------------- numbering (parse phase)
  md.core.ruler.push("book_numbering", (state: Any) => {
    const env: ChapterEnv = state.env;
    const tokens = state.tokens;
    const ch = env.ch;
    const numbered = ch.kind === "chapter" || ch.kind === "appendix";
    const c = env.counters;
    const next = (k: string) => (c[k] = (c[k] ?? 0) + 1);
    const reg = (id: string, l: Label) => {
      if (env.labels.has(id)) env.errors.push(`${ch.file}: duplicate label "${id}" (also at ${env.labels.get(id)!.href})`);
      env.labels.set(id, l);
    };
    const seen = new Set<string>();
    let h2 = 0, h3 = 0;
    for (let i = 0; i < tokens.length; i++) {
      const t = tokens[i];
      if (t.type === "heading_open") {
        const level = Number(t.tag.slice(1));
        const inline = tokens[i + 1];
        const text = inline.content.replace(/\s*\{[^{}]*\}\s*$/, "");
        const classes = (t.attrGet("class") ?? "").split(/\s+/);
        const unnumbered = classes.includes("unnumbered") || !numbered;
        let id = t.attrGet("id");
        if (!id) {
          id = level === 1 ? "top" : slugify(text);
          let k = id, n = 1;
          while (seen.has(k)) k = `${id}-${++n}`;
          id = k;
          t.attrSet("id", id);
        }
        seen.add(id);
        let num = "";
        if (level === 1) { num = ch.num; env.title = inline; env.titleText = text; }
        if (level === 1 && ch.kind === "part") num = ch.num;
        else if (level === 2 && !unnumbered) { h2++; h3 = 0; num = `${ch.num}.${h2}`; }
        else if (level === 3 && !unnumbered && h2) { h3++; num = `${ch.num}.${h2}.${h3}`; }
        t.meta = { num: level === 1 ? "" : num, text };
        if (level === 1 && ch.num) t.meta.chapterNum = ch.num;
        if (level >= 2 && level <= 3) env.toc.push({ level, id, num, inline });
        if (/^sec-/.test(id)) reg(id, { kind: "sec", num, href: `${ch.href}${level === 1 ? "" : `#${id}`}`, level, chapterKind: ch.kind, title: text });
      } else if (t.type === "math_block" && t.meta?.label) {
        if (!/^eq-/.test(t.meta.label)) env.errors.push(`${ch.file}: equation label "${t.meta.label}" must start with eq-`);
        const num = numbered ? `${ch.num}.${next("eq")}` : String(next("eq"));
        t.meta.num = num;
        reg(t.meta.label, { kind: "eq", num, href: `${ch.href}#${t.meta.label}` });
      } else if (t.type === "callout_open") {
        const m = t.meta;
        const def = CALLOUTS[m.kind];
        if (def?.numbered) {
          const n = next(m.kind);
          m.num = numbered ? `${ch.num}.${n}` : String(n);
          if (m.id) reg(m.id, { kind: def.prefix!, num: m.num, href: `${ch.href}#${m.id}` });
        } else if (m.id) reg(m.id, { kind: "sec", num: "", href: `${ch.href}#${m.id}`, title: m.kv.title });
        if (m.kv.title) m.titleTokens = state.md.parseInline(m.kv.title, env)[0].children;
      } else if (t.type === "fence" && t.info === FIGURE_INFO) {
        t.meta = parseFigureBlock(t.content, o, env);
        if (!t.meta.error) {
          const n = next("fig");
          t.meta.num = numbered ? `${ch.num}.${n}` : String(n);
          if (t.meta.label) reg(t.meta.label, { kind: "fig", num: t.meta.num, href: `${ch.href}#${t.meta.label}` });
          if (t.meta.caption) t.meta.capTokens = state.md.parseInline(t.meta.caption, env)[0].children;
        } else env.errors.push(`${ch.file}: ${t.meta.error}`);
      }
    }
  });

  return md;
}

function parseFigureBlock(content: string, o: MdOptions, env: ChapterEnv): Any {
  const opts: Record<string, string> = {};
  const params: Record<string, string> = {};
  for (const raw of content.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    const om = /^\/\/\|\s*([\w-]+):\s*(.*)$/.exec(line);
    if (om) { opts[om[1]] = om[2].trim().replace(/^"(.*)"$/s, "$1").replace(/\\"/g, '"'); continue; }
    const pm = /^([\w-]+):\s*(.*)$/.exec(line);
    if (pm) { params[pm[1]] = pm[2].trim(); continue; }
    return { error: `figure block: cannot read line "${line}"` };
  }
  const caption = opts["fig-cap"];
  const label = opts.label;
  if (label && !/^fig-/.test(label)) return { error: `figure label "${label}" must start with fig-` };
  if (opts.figure) {
    const module = o.figures.get(opts.figure);
    if (!module) return { error: `unknown figure module "${opts.figure}"` };
    try {
      const { p, t } = resolve(module, params);
      return { module, params: p, t, caption, label, autoplay: opts.autoplay === "true", bare: opts.bare === "true", cls: opts.class };
    } catch (e) {
      return { error: (e as Error).message };
    }
  }
  if (opts.src) return { src: opts.src, alt: opts.alt ?? caption?.replace(/[*_`$\[\]@]/g, ""), caption, label };
  void env;
  return { error: "figure block needs //| figure: <module> or //| src: <image>" };
}
