# Authoring Conventions

This book teaches Bayesian optimization and preferential Bayesian optimization
from first principles to the research frontier. Every chapter should leave the
reader able to answer two questions: *why does it work this way?* and *how do we
know?* The first is answered by derivations and pictures, the second by
citations attached to the claims they support.

## The reader

The book assumes one reader and holds that level from the first chapter to the
last: **a strong software engineer who is new to machine learning.** They read
code fluently, know what a matrix and a derivative are, and have not used
probability since school. They do not know what a posterior, a kernel, a link
function, or regret is.

Four rules follow, and they are the most common defects to fix in review.

1. **Introduce before use. A link is not an introduction.** The first time a
   chapter leans on a term this reader would not know, give a one-clause plain
   gloss in the prose. A cross-reference to where the term is defined is
   welcome, but it does not discharge the debt. Test: read the chapter top to
   bottom with no outside knowledge and no clicking.
2. **Openers orient; they do not roll-call.** The lead paragraphs name the
   problem, connect it to the previous chapter, and say where the chapter is
   going. They do not list every mechanism the chapter will cover.
3. **Keep the gradient even.** Roughly one new idea per paragraph. Put an
   intuition, a picture, or a one-dimensional special case *before* a dense
   formula, never after. Mark optional depth as an `aside` so the floor does
   not drop out without warning.
4. **No paragraph out of nowhere.** Each paragraph connects to the one before.
   When the subject shifts, a sentence says why.

And one against the opposite failure: **do not over-repeat.** An idea gets one
good explanation and, where it helps, one figure. Do not restate the thesis in
every section.

## The section arc

**Teaching chapters (Parts I to IV)** follow this arc within each major section:

1. *Problem.* What question does this tool answer? Why does the previous
   chapter's tool not already answer it?
2. *Idea.* The intuition, ideally with a picture or a one-dimensional case.
3. *Formalism.* The definitions and the result, with a derivation in a
   `derivation` box when it is more than two lines. Every step states its
   justification ("by the product rule", "substituting @eq-x").
4. *Exploration.* An interactive figure that lets the reader vary what the
   formula depends on, with a short guided list of things to try.
5. *Practice.* What goes wrong in implementations; what libraries do.

Each teaching chapter ends with two to four exercises with collapsed solutions
and a **Further reading** section: first-hand sources first, one line each on
what the reader will find there.

**Case studies (Part V)** work one real problem end to end: the problem and
what an evaluation costs, the choices made before the first query, the method
running on measured data or on a clearly labeled calibrated simulation, and what
the published studies found.

**Frontier, human, neighbor, perspective, and synthesis chapters (Parts VI to
X)** report research. Each major section moves through:

1. *The question or common claim* in one or two sentences.
2. *The evidence*, study by study, with venue, year, design, and the numbers
   that matter (sample sizes, effect sizes, rates), each sentence carrying its
   citation.
3. *What it means for preferential optimization*, with the book's own
   inferences marked (see below).

Close each such chapter with a `frontier` callout titled "Settled, contested,
missing" that sorts the chapter's conclusions into those three bins.

These chapters still serve the same reader: a term from Parts I to IV gets a
cross-reference and a short reminder the first time it reappears; a term from
another discipline (effect size, preregistration, Weber's law, revealed
preference) gets a plain gloss.

## Tone

- Warm and scholarly, walking alongside the reader. Preempt confusion: "a
  reader may wonder why...".
- **No em dashes.** Use commas, colons, parentheses, or two sentences.
- No filler openings ("it is worth noting that", "it should be emphasized"),
  no intensifiers ("truly", "precisely", "incredibly"), no false agency ("the
  algorithm wants to").
- Complete, declarative sentences. Plain accurate words, not coined terms.
- Numbers keep their precision and their units. Say "about" only when the
  source does.
- American spelling (centered, neighboring, gray).
- Names: "preferential Bayesian optimization (PBO)" at first use in a chapter,
  "PBO" after that. "Preferential optimization" only for the broader class
  that includes non-Bayesian methods. qEUBO is the *multi-option* form of EUBO
  (one query showing q options), never "the batch form". A *duel* is one
  pairwise comparison, defined at its first use in `preferences/01` and
  interchangeable with "comparison" after that; inputs the reader fills in a
  figure are labeled "Answers".
- Exercises sit in a numbered final section,
  `## Exercises {#sec-<prefix>-exercises}`, before Further reading.
- Explain a thing fully once; later chapters recall it in a sentence with a
  cross-reference. A cross-reference promises something: the target must
  deliver it.

## The Markdown dialect

Source files are Markdown under `en/`, ordered by `en/book.yml`. The syntax
follows Pandoc and Quarto where they have one.

### Front matter

```yaml
---
status: draft        # outline | draft | review | done
synopsis: "One or two sentences, shown on the part page and as the description."
sources: ["Rasmussen and Williams 2006"]   # for editors, not rendered
---
```

### Headings and labels

```markdown
# Gaussian Process Regression {#sec-gp-regression}
## Conditioning on observations {#sec-gp-conditioning}
### The mean is a sum of bumps {#sec-gp-bumps}
## Further reading {#further-reading .unnumbered}
```

One `#` heading per file: the chapter title. `##` and `###` are numbered
(8.2, 8.2.1). Give every heading you might reference an id starting with
`sec-`; ids are global across the book, so make them specific. Never rename a
published id.

### Math

KaTeX, inline `$...$` and display `$$...$$`. A display equation is numbered
when it has a label after the closing `$$`:

```markdown
$$
\mu(\vx) = \vk(\vx)^\T \mK^{-1} \vy
$$ {#eq-gp-mean}
```

Label only the equations the text refers to. Shared macros live in
`src/pipeline/macros.ts` (`\R \E \Var \Cov \N \GP \X \D \KL \argmax \T \vx \vy
\vf \vk \vmu \mK \mI \mL \mSigma \EI \UCB \EUBO`, and others); use them so
notation is identical everywhere, and add to the notation appendix when you add
one. Vectors are bold lowercase (`\vx`), matrices bold uppercase (`\mK`),
transpose is `^\T`.

Book-wide notation, settled during review:

- The true maximum is `f^\star` at `\vx^\star`; the best value observed so far
  (the incumbent) is `f^*_n`. Never `f^*`, `y^*`, or `f^+` for either.
- `d` is the number of inputs, in prose, formulas, tables, and captions. An
  index over inputs is `j` (`\ell_j`, `j = 1, \dots, d`). Cohen's d in the
  psychology chapters is a different thing.
- A finite candidate set is `\X` with size `|\X|`, never `D` or `N_\X`; where a
  paper writes otherwise, say so once in a clause. Inside a table cell write
  `\lvert\X\rvert`: a bare `|` ends the cell even inside math.
- `\tau` is the scale of the logistic link; a lapse rate is
  `\lambda_{\text{lapse}}`.
- `\sigma_n^2` is the noise variance; the posterior standard deviation always
  carries its argument, `\sigma_n(\vx)`.
- Acquisition functions are `a_n(\vx)`; UCB is
  `\mu_n(\vx) + \beta^{1/2}\sigma_n(\vx)`; cumulative regret is `R_T`; maximum
  information gain is `\gamma_T`.
- The logistic function is `\operatorname{sigmoid}(z)`; `\sigma` is never the
  logistic function.

### Citations

```markdown
... a probit likelihood [@chu2005preference].
... as @gonzalez2017preferentialb showed ...         (narrative: "González et al. (2017)")
[@lin2022preference; @astudillo2023qeubo]            (several)
[see @garnett2023bayesian, ch. 4]                    (prefix and locator)
[-@thurstone1927law]                                 (year only)
```

Keys live in `refs/*.bib`. Find a key with `node tools/refs/lookup.ts <url |
arXiv id | DOI | title words>`. Add a missing work to `refs/foundations.bib`
(textbooks and classics) or `refs/additions.bib` (anything else), with a DOI or
stable URL and an `evidence` field. An unknown key fails the build.

The build attaches references at three levels: a hover card on every inline
citation, a collapsed list at the end of each section ("Sources cited in
Section 8.3"), a full list at the end of each chapter with links back to the
citing sections, and an aggregated list on each part page. Do not write
reference lists by hand.

### Evidence and inference

Every empirical claim carries a citation in the same sentence. The `evidence`
field of each entry is shown as a badge (*preprint*, *working paper*,
*workshop paper*, *software*, *non-peer-reviewed*). When a load-bearing claim
rests on a preprint, say so in the prose too ("a 2026 preprint reports...").

When a sentence is the book's own inference, drawn from the evidence but stated
by no single source, end it with **(inference)**. Use it sparingly; most
interpretation should be phrased so that its basis is clear.

"Not found" claims are bounded: "we found no study that...", "as of September
2026, no study has...".

Quotation marks are only for wording checked against the source itself. A
phrase known only in translation or from a secondary account is paraphrased,
without quotation marks, unless the original wording has been found. When a paper cannot be
read, report only what its title and abstract support, and say so.

A bibliography entry is added only when its metadata is verified on a primary
page (Crossref, the publisher, the proceedings, OpenReview). Venue fields do
not hedge: an entry has a confirmed venue, or it is `howpublished = {arXiv}`
with `evidence = {preprint}`.

### Cross-references

`@sec-gp-noise` renders "Section 8.3" (or "Chapter 8" for a chapter id, and
"Part II" for a part id such as `@sec-part-gp`),
`@fig-gp-posterior` "Figure 8.1", `@eq-gp-mean` "Equation (8.4)",
`@def-gp` "Definition 7.1", and similarly `@thm-`, `@ex-`, `@exr-`, `@alg-`,
`@tbl-`. `[-@eq-gp-mean]` drops the word: "(8.4)". `[+@sec-pbo]` renders the
number and the title together. **Never write a chapter or part number by hand**
("Chapter 18", "Part VIII"): numbers change when the outline changes, and a
cross-reference renumbers itself. Cross-references resolve
across chapters, and an unresolved one fails the build. Refer to a stub
chapter's planned section ids freely; they exist in the outline.

### Callouts

```markdown
::: {.definition #def-gp title="Gaussian process"}
...
:::
```

| Class | Use | Numbered |
|---|---|---|
| `definition` | a term defined formally | yes, `@def-` |
| `theorem`, `lemma`, `proposition`, `corollary` | a stated result | yes |
| `example` | a worked example | yes, `@ex-` |
| `exercise` | an exercise; nest a `solution` inside | yes, `@exr-` |
| `solution` | collapsed solution | no |
| `algorithm` | pseudocode as a numbered list | yes, `@alg-` |
| `table` | a captioned table (`title` is the caption) | yes, `@tbl-` |
| `derivation` | a multi-step derivation; add `collapsed=true` to fold it | no |
| `proof` | a proof, ends with ∎ | no |
| `keyidea` | the one sentence to remember from a section | no |
| `note` | a side remark | no |
| `aside` | optional depth, marked "optional" | no |
| `pitfall` | a common mistake | no |
| `code` | a short code sketch (Python/NumPy) | no |
| `frontier` | research status: settled, contested, missing | no |

Nest a callout inside another by using the same `:::` fence; the parser counts
depth.

### Figures

Interactive figures are TypeScript modules in `src/figures/`, embedded with a
`{figure}` fence:

````markdown
```{figure}
//| figure: gp-posterior
//| label: fig-gp-bumps
//| fig-cap: "The posterior mean as a sum of weighted kernels ... [@kanagawa2018gaussian]"
points: "0.15:0.2,0.24:0.9,0.6:-0.5,0.85:0.6"
bumps: true
```
````

Every other line sets a declared parameter. A static image uses `//| src:
images/name.svg` instead of `//| figure:`, with the file in `en/images/`.

The same module can be embedded several times with different parameters to
make different points; prefer that over a new module that differs only in its
defaults.

## Figure modules

A figure module declares parameters and renders a state to SVG with a pure
function; the build renders it statically (so it works without script) and the
browser hydrates it with controls. The contract is `src/figures/types.ts`;
read `src/figures/gp-posterior.ts` as the reference implementation. There is no
registry to edit: every `src/figures/<name>.ts` is a figure named `<name>`.

Rules:

- **Pure.** `render(state)` reads nothing but its arguments: no DOM, no clock,
  no `Math.random` (use `lib/random.ts` with a `seed` parameter).
- **Width.** One viewBox unit is one CSS pixel. The build renders at 660 and
  330 px; below 480 px, stack and simplify.
- **Colors by role**, from `lib/theme.ts`: `C.truth` for the unknown objective
  (dashed), `C.model` and `C.band` for the surrogate, `C.acq` and `C.acqFill`
  for the acquisition function and the next query, `C.ink` for observations,
  `C.c4` and onward for other series. Never hex.
- **Shared frame.** One-dimensional plots use `lib/plot.ts` (`frame`,
  `frameAxes`, `curve`, `band`, `dots`, `vline`, `legend`, `hitArea`), so every
  chapter's plots have the same margins and roles.
- **Shared math.** Kernels, GP regression, the Laplace preference model, and
  acquisition functions are in `lib/gp.ts`, `lib/acq.ts`, `lib/stats.ts`,
  `lib/linalg.ts`. The running objective is `lib/objectives.ts` (`running`).
  Reuse them; if you need something new, add it to the library.
- **Interaction.** Sliders, choices, and toggles come from the parameter
  declarations. A `data` parameter holds a serialized list (observations,
  duels). `pointer(p, e)` turns clicks into new parameters; `e.data` holds data
  coordinates when the click lands on a `hitArea`. `actions` add buttons; mark
  the main one `primary`. Every pointer interaction needs a keyboard path
  (an action or a control).
- **Animation.** A `timeline` with `duration`, `rate`, `keyframes`, and a
  `poster` frame. The state at position `t` must be a pure function of `t`.
- **Describe.** `describe(state)` returns one or two sentences of what the
  figure shows; it is the accessible name and the live-region announcement.
- **Labels** live in `labels.en`; `labels.zh` comes with the Chinese edition.
- **Captions** say what to look at and which values are illustrative.

Verify every figure: `npm run build && node tools/shots.ts --fig <name>
en/<chapter>.html` writes screenshots at 1280 px (light) and 390 px (dark) plus
a static render to `.cache/shots/`, and reports hydration failures, console
errors, and horizontal overflow. Look at the images.

## Working from the literature

- **Cite primary sources**: the paper, the proceedings, the software release,
  never a secondary summary of them. `node tools/refs/lookup.ts <URL>` finds
  the bibliography key of a paper by its URL.
- **Keep every number and qualifier exactly** as the source states it
  (preprint, only the abstract read); settle a venue on a primary page before
  writing it (see Evidence and inference).
- **Restate, do not attribute.** Where a widely repeated claim deserves
  correction, state it as a common claim in the literature and give the
  evidence ("A common claim is that PBO fails beyond 10 to 20 dimensions. The
  evidence points instead to...").

## Status

`outline` (headings and plan only) → `draft` (complete prose, figures,
citations; not reviewed) → `review` → `done`. Pages show a banner for outline
and draft. Set the status in the front matter when you finish a pass.

## Commands

```bash
npm install
npm run build                    # build _site/, fail on any error
npm run dev                      # rebuild on change, serve at http://localhost:4400
node tools/shots.ts <page.html>  # screenshots and checks for built pages
node tools/shots.ts --all        # check every page
node tools/refs/lookup.ts <q>    # find a citation key
make deploy                      # build and publish to changkun.de/bobook
```
