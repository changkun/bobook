// Tests for the Markdown dialect: citations, cross-references, equations,
// callouts, figure blocks, and the BibTeX reader.

import { test } from "node:test";
import assert from "node:assert/strict";
import { createMarkdown, newEnv, type Label } from "../src/pipeline/markdown.ts";
import { parseBib, people } from "../src/pipeline/bib.ts";
import { citeLabels, type Bib } from "../src/pipeline/cite.ts";
import { tightenCjkSpaces } from "../src/pipeline/cjk.ts";
import { loadFigures } from "../src/figures/registry.ts";

const bibSrc = `
@article{chu2005preference,
  author = {Chu, Wei and Ghahramani, Zoubin},
  title = {Preference Learning with Gaussian Processes},
  year = {2005},
  journal = {ICML},
  url = {https://doi.org/10.1145/1102351.1102369},
  evidence = {peer-reviewed}
}
@misc{lin2022a,
  author = {Lin, Zhiyuan and others},
  title = {A},
  year = {2022},
  url = {https://example.org/a},
  evidence = {preprint}
}
@misc{lin2022b,
  author = {Lin, Zhiyuan and others},
  title = {B},
  year = {2022},
  url = {https://example.org/b},
  evidence = {preprint}
}`;

const bib: Bib = new Map(parseBib(bibSrc).map((e) => [e.key, e]));
const figures = await loadFigures();

function render(src: string, labels = new Map<string, Label>()) {
  const md = createMarkdown({ bib, figures, lang: "en" });
  const env = newEnv({ num: "3", kind: "chapter", href: "en/x/03.html", file: "test.md" }, labels);
  const tokens = md.parse(src, env);
  env.citeLabels = citeLabels(bib, env.cites);
  const html = md.renderer.render(tokens, md.options, env);
  return { html, env };
}

test("bib reader parses authors and et al.", () => {
  const e = bib.get("lin2022a")!;
  const { list, etal } = people(e.fields.author);
  assert.equal(list[0].last, "Lin");
  assert.equal(etal, true);
});

test("labels with the same author text and year get a and b", () => {
  const l = citeLabels(bib, ["lin2022a", "lin2022b", "chu2005preference"]);
  assert.equal(l.get("lin2022a")!.year, "2022a");
  assert.equal(l.get("lin2022b")!.year, "2022b");
  assert.equal(l.get("chu2005preference")!.authors, "Chu and Ghahramani");
});

test("parenthetical, narrative, and year-only citations", () => {
  const { html, env } = render("A [@chu2005preference] and @chu2005preference and [-@chu2005preference].");
  assert.match(html, /\(<a class="cite-link" href="#ref-chu2005preference"[^>]*>Chu and Ghahramani, 2005<\/a>\)/);
  assert.match(html, />Chu and Ghahramani \(2005\)</);
  assert.match(html, />2005</);
  assert.deepEqual(env.errors, []);
});

test("an email address is not a citation", () => {
  const { html, env } = render("Write to hi@chu2005preference.org.");
  assert.doesNotMatch(html, /cite-link/);
  assert.deepEqual(env.errors, []);
});

test("unknown citation keys are errors", () => {
  const { env } = render("See [@nobody2020nothing].");
  assert.equal(env.errors.length, 1);
});

test("headings are numbered and labeled", () => {
  const labels = new Map<string, Label>();
  const { html } = render("# Title {#sec-title}\n\n## First {#sec-first}\n\n### Sub {#sec-sub}\n\n## Plain {.unnumbered}\n", labels);
  assert.match(html, /<h2 id="sec-first"><span class="sec-num">3\.1<\/span>/);
  assert.match(html, /<h3 id="sec-sub"><span class="sec-num">3\.1\.1<\/span>/);
  assert.equal(labels.get("sec-first")!.num, "3.1");
  assert.equal(labels.get("sec-title")!.level, 1);
});

test("equations are numbered and cross-referenced", () => {
  const { html, env } = render("$$\na = b\n$$ {#eq-test}\n\nSee @eq-test and [-@eq-test].");
  assert.match(html, /<div class="math-display" id="eq-test">/);
  assert.match(html, /\(3\.1\)/);
  assert.match(html, /Equation \(3\.1\)/);
  assert.deepEqual(env.errors, []);
});

test("inline math does not swallow prices", () => {
  const { html } = render("It costs $5 and $10.");
  assert.doesNotMatch(html, /katex/);
});

test("callouts are numbered and nest", () => {
  const { html } = render("::: {.exercise #exr-a}\nQ\n\n::: {.solution}\nA\n:::\n:::\n\n::: {.definition #def-a title=\"Thing $x$\"}\nD\n:::\n");
  assert.match(html, /Exercise 3\.1/);
  assert.match(html, /<details class="callout callout-solution"/);
  assert.match(html, /Definition 3\.1/);
  assert.match(html, /class="katex"/);
});

test("figure blocks render static SVG and are numbered", () => {
  const { html, env } = render("```{figure}\n//| figure: gp-posterior\n//| label: fig-a\n//| fig-cap: \"Cap [@chu2005preference]\"\nlengthscale: 0.2\n```\n\nSee @fig-a.");
  assert.deepEqual(env.errors, []);
  assert.match(html, /<figure class="figure" id="fig-a">/);
  assert.match(html, /data-figure="gp-posterior"/);
  assert.match(html, /Figure 3\.1/);
  assert.match(html, /cite-link/);
});

test("out-of-range figure parameters are errors", () => {
  const { env } = render("```{figure}\n//| figure: gp-posterior\nlengthscale: 99\n```\n");
  assert.equal(env.errors.length, 1);
});

test("unresolved cross-references are errors", () => {
  const { env } = render("See @sec-nowhere.");
  assert.equal(env.errors.length, 1);
});

test("el() merges fill, stroke, and a caller's style into one style attribute", async () => {
  const { el } = await import("../src/figures/lib/svg.ts");
  const out = el("rect", { x: 1, fill: "red", style: "cursor:pointer" });
  assert.equal(out.match(/style=/g)!.length, 1);
  assert.match(out, /style="fill:red;cursor:pointer"/);
});

function renderZh(src: string, labels = new Map<string, Label>()) {
  const md = createMarkdown({ bib, figures, lang: "zh" });
  const env = newEnv({ num: "3", kind: "chapter", href: "zh/x/03.html", file: "test.md" }, labels);
  const tokens = md.parse(src, env);
  env.citeLabels = citeLabels(bib, env.cites);
  return md.renderer.render(tokens, md.options, env);
}

test("Chinese: a line break between Chinese characters is dropped; next to Latin it is a space", () => {
  const html = renderZh("后验均值\n随观测增加。The kernel\nmatters，而且\nBoTorch 默认如此。");
  assert.match(html, /后验均值随观测增加。The kernel matters，而且 BoTorch 默认如此。/);
});

test("Chinese: citations use 等人 in running text, 等 in parentheses, 与 between two authors", () => {
  const html = renderZh("见 @lin2022a。另见 [@lin2022b; @chu2005preference]。@chu2005preference 提出。");
  assert.match(html, /Lin 等人（2022a）/);
  assert.match(html, /（<a[^>]*>Lin 等，2022b<\/a>；<a[^>]*>Chu 与 Ghahramani，2005<\/a>）/);
  assert.match(html, /Chu 与 Ghahramani（2005）/);
});

test("Chinese: spaces between Chinese and a reference are removed; next to Latin, math, and code they stay", () => {
  const html = '<p>在 <a class="xref" href="#">第 10 章</a> 中，见 <span class="cite">（<a>Lin 等，2022</a>）</span> 。使用 EUBO 选择 <span class="katex"><span>x</span></span> 时 <code>a b</code> 与 第 3 步</p>';
  assert.equal(tightenCjkSpaces(html), '<p>在<a class="xref" href="#">第 10 章</a>中，见<span class="cite">（<a>Lin 等，2022</a>）</span>。使用 EUBO 选择 <span class="katex"><span>x</span></span> 时 <code>a b</code> 与第 3 步</p>');
});

test("Chinese: no space next to full-width punctuation, even before Latin or digits", () => {
  assert.equal(tightenCjkSpaces("<p><strong>结论。</strong> 2020 年，见 “好” EUBO。</p>"), "<p><strong>结论。</strong>2020 年，见“好”EUBO。</p>");
});
