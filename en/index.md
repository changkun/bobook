---
synopsis: "An interactive book on Bayesian optimization and preferential Bayesian optimization, from the probability it rests on to the research frontier and what a human preference is."
---

<div class="cover-top">
<div class="book-cover" role="img" aria-label="Cover of Bayesian Optimization: From First Principles to Human Preferences, by Changkun Ou. Threads of a Gaussian process posterior pinch at six observations; below them, arcs join the pairs a person compared.">
<img src="@@ROOT@@assets/cover-art.svg" alt="" width="400" height="600">
<div class="bc-text" aria-hidden="true"><span class="bc-author">Changkun Ou</span><span class="bc-title">Bayesian Optimization</span><span class="bc-sub">From First Principles to Human Preferences</span></div>
</div>
<div class="cover-text">

# Bayesian Optimization {#sec-home}

<p class="book-subtitle">From First Principles to Human Preferences</p>

Some functions are expensive to evaluate and impossible to write down: the
accuracy of a model after a day of training, the comfort of an exoskeleton
after a minute of walking, the look of a design that only a person can judge.
Bayesian optimization finds good inputs for such functions with few
evaluations. It keeps a probabilistic model of what every untried input might
return, and spends each evaluation where the model expects to learn the most.
When the only measurement is a person saying "this one, not that one", the same
idea becomes preferential Bayesian optimization, and the person becomes part of
the system being studied.

This book teaches both from the beginning: the probability and linear algebra
a software engineer may not have used since school, Gaussian processes and the
analysis behind their kernels, the optimization loop, and learning from
comparisons. It works four real problems
end to end, then follows the research through 2026: what has been proved, what
happened with real people, and what psychology, economics, neuroscience, and
philosophy say about whether a preference is there to be found. Its argument
is that the algorithms are now mature and the hard part has moved to
measurement: what a single comparison measures, and what asking does to the
person who answers.

</div>
</div>

<div class="cover-live">
<div class="cover-fig">

```{figure}
//| figure: bo-loop
//| class: hero
//| bare: true
//| autoplay: true
//| fig-cap: "An optimizer at work on a function it cannot see (dashed). Blue is what it believes, the band is how unsure it is, and orange is where it looks next."
steps: 10
```

</div>
<div class="cover-live-text">

Every chapter has figures you can change, derivations you can step through,
and references attached to the section that uses them. In some figures you are
the person being optimized.

Start with @sec-optimizing-the-unknown, read the [preface](preface.html)
first, or pick a part from the [contents](#contents) below.

</div>
</div>

## Contents {#contents .unnumbered}

<!--CONTENTS-->

## Three ways in {#sec-reading-paths .unnumbered}

<div class="paths">
<div class="path">
<p class="path-name">From the beginning</p>
<p class="path-who">New to probability, or rusty. Every tool is built before it is used.</p>

- [+@sec-optimizing-the-unknown]
- [+@sec-gaussian]
- [+@sec-gp-regression]
- [+@sec-acquisition]
- [+@sec-pbo]

</div>
<div class="path">
<p class="path-name">Build a system</p>
<p class="path-who">You know some ML and want to run preferential optimization with people.</p>

- [+@sec-bo-loop]
- [+@sec-gp-preference]
- [+@sec-pbo]
- [+@sec-cs-photo]
- [+@sec-recommendations]

</div>
<div class="path">
<p class="path-name">The research</p>
<p class="path-who">You know BO and want the state of the field and its open questions.</p>

- [+@sec-pbo-history]
- [+@sec-observation-models]
- [+@sec-hci]
- [+@sec-what-comparisons-measure]
- [+@sec-open-problems]

</div>
</div>

```{figure}
//| figure: book-map
//| class: map
//| fig-cap: "How the chapters build on each other: one row per part. Click a chapter to see what it needs and what needs it, and to open it."
```
