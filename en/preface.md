---
status: done
synopsis: "Where the book comes from, what it argues, whom it is for, and how to read it."
---

# Preface {#sec-preface}

This book grew out of my doctoral research at LMU Munich on human-in-the-loop
systems: optimizers that tune what only a person can judge, by asking that
person. Bayesian optimization, and its preferential variant that learns from
comparisons, is the best-developed method for this. Using it in studies with
people left me with a doubt. The method assumes that a person has a
preference, fixed and waiting to be found, and the people in our studies did
not always behave as if they had one.

Writing the book served two purposes: to give a systematic account of the
field, from its mathematical foundations to the question of what a preference
is, and to catch up with research that has moved quickly since I graduated.
The book accordingly has two halves. The first develops the methods from the
beginning, in enough detail to implement them. The second surveys the research
up to September 2026, including work that does not fit the methods'
assumptions: studies with real people, and what other disciplines know about
preference.

## What the book argues {#sec-preface-argument}

Between 2017 and 2026 the methods of preferential Bayesian optimization
matured: acquisition functions gained a decision-theoretic foundation, regret
theory caught up with scalar feedback, and software settled on a default
pipeline (@sec-part-frontier). The assumptions around the algorithm received
far less scrutiny, and the most consequential of them concern the person: that
a comparison measures a stable preference plus noise of constant size, that
the order of the questions does not matter, and that the preference found at
the end of a session is the one the person brought to it. The bottleneck has
moved from algorithms to measurement: what a single comparison measures, how
answers should be modeled, and what asking does to the person who answers
(@sec-syn-bottleneck).

It follows that a session of preferential optimization is both an estimate and
an intervention: it learns a preference and may also change it, and a study
that does not check for the second cannot be sure of the first
(@sec-syn-intervention). Much of what is known about measurement lies outside
machine learning, which is why @sec-part-humans covers studies with people and
@sec-part-perspectives covers what psychology, economics, neuroscience, and
philosophy know about preference. @sec-part-synthesis draws recommendations,
open problems, and an outlook from both.

## Who this book is for {#sec-preface-audience}

The book is intended for software engineers and students who want to
understand Bayesian optimization and learning from human comparisons, and for
researchers in machine learning and human-computer interaction who use these
methods. It assumes programming experience and basic calculus. Probability,
linear algebra, the Gaussian distribution, and information theory are
developed in @sec-part-foundations, so no prior study of machine learning is
needed. Readers who already know Bayesian optimization can begin with
@sec-part-preferences, or go directly to the research in @sec-part-frontier
through @sec-part-synthesis.

## How to read it {#sec-preface-reading}

@sec-how-to-read gives an overview of the ten parts. @sec-part-foundations
through @sec-part-preferences build on each other and are best read in order;
@sec-part-cases applies them to four case studies; the remaining parts can be
read in any order once @sec-part-preferences is familiar. The landing page
suggests three reading paths.

Most figures are interactive: sliders set parameters, buttons step through an
algorithm, and clicking a plot adds an observation. In several figures the
reader is the person being optimized.

## Sources and conventions {#sec-preface-ack}

The research parts are based on the literature from 2017 to September 2026 on
preferential Bayesian optimization and on preference, read from primary
sources: the theorems and tables in the papers, arXiv version histories,
proceedings, code repositories, and software changelogs. Every empirical claim
is cited in the sentence that makes it, and works that are not peer reviewed
are marked as such. Sentences that state the book's own inference rather than
a source's end with *(inference)*. A statement that no study was found means
none was found in this search, as of September 2026.

Many of the results discussed are recent, and some are preprints that may
change in review; the book will be revised accordingly. Corrections and
suggestions are welcome.

## Open source {#sec-preface-open}

Readers are encouraged to check, reproduce, and extend what this book shows.
The text in English and Chinese, the interactive figures and the code that computes
them, the scripts that regenerate the recorded data behind several figures,
and the bibliography are open source at
[github.com/changkun/bobook](https://github.com/changkun/bobook). Corrections
are welcome there as issues or pull requests.

## Use of language models {#sec-preface-llm}

This book was written entirely by large language models, prompted and steered
throughout by the author. That includes the text in English and Chinese, the
translation between them, the interactive figures and the code behind them,
the simulations, and much of the checking; no sentence was written or edited
by hand. The author set the book's scope, structure, and argument, directed
every revision, and decided what to keep and what to change. The checking
relied on primary sources and on computation: every cited work was matched
to its record on a publisher's, proceedings', or preprint page; the numbers
the text quotes from figures are pinned by tests that fail when a figure
changes; and derivations and worked numbers were recomputed in review passes
separate from their drafting. The author takes full responsibility for the
accuracy, integrity, and conclusions of the book.
