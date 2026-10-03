---
status: done
synopsis: "What makes an objective a black box, why each evaluation is precious, and why the answer is to model the objective and spend every evaluation where it teaches the most. A map of the book."
sources: ["Garnett 2023", "Frazier 2018", "Bergstra and Bengio 2012"]
---

# Optimizing What You Cannot Write Down {#sec-optimizing-the-unknown}

Most optimization a software engineer meets has a formula. A loss function can
be differentiated, a query plan can be costed, a schedule can be checked
against its constraints. This book is about the other kind: problems where the
only way to learn how good an input is, is to try it, and every try is
expensive.

The first part of this chapter describes that kind of problem and why the usual
tools waste evaluations on it. The second part sketches the idea the rest of the
book develops, Bayesian optimization: keep a model of the unknown function that
knows how unsure it is, and let that uncertainty decide where to look next. The
chapter then turns to the case that gives the book its subtitle, where the
function lives in a person's head and the only measurement is a choice between
two options. It ends with a map.

## The cost of a single evaluation {#sec-cost-of-evaluation}

Four problems recur throughout the book.

**Training a model.** A machine learning model, such as a neural network, is
fitted to data by a training procedure, and the procedure has settings of its
own that someone must choose before it starts: how large a step each update
takes (the learning rate), how many examples each update uses (the batch
size), and a dozen more. These settings are called **hyperparameters**,
because training does not learn them. A poor choice can ruin a run: with too
large a learning rate, the updates overshoot and training never settles (it
*diverges*). Each combination has to be tried by training the model and
measuring its accuracy on data it was not trained on (the validation
accuracy), which can take hours on expensive hardware. A team might afford
fifty such runs.

**Tuning an exoskeleton.** A powered ankle exoskeleton applies torque during
each stride, and its timing and magnitude can be set by a few parameters. The
right setting differs from person to person. Measuring how much it helps means
having the person walk for minutes while their metabolic cost is measured, and
people tire. A session might afford a few dozen settings (@sec-health).

**Brewing coffee.** Grind size, water temperature, ratio, and steep time
determine the taste. Each cup takes minutes to make and is judged by tasting
it. There is no number to read off; there is a person deciding whether this cup
is better than the last one.

**Choosing the look of a design.** A typeface, a color palette, a material
shader, or the shape of a 3-D character has parameters, sometimes dozens. Whether
a setting is good is a judgment only a person can make, by looking.

These problems share four properties, and the book uses them as its working
definition of a **black-box objective**:

1. There is a function $f$ from inputs (settings) to a score, and we want an
   input with a high score. The input space $\X$ is usually a box of a few to a
   few dozen continuous parameters.
2. We have no formula for $f$, so no gradient and no way to reason about its
   shape except by evaluating it.
3. Each evaluation is expensive: in time, money, compute, or a person's patience.
   The budget is tens to hundreds of evaluations, not millions.
4. Evaluations are noisy: the same input can return different scores, because
   training runs differ by random seed and people differ from minute to minute.

The fourth property sometimes takes a stronger form, which the coffee and design
problems show: there may be no score at all, only a person's choice between
options.

## Why the usual tools struggle {#sec-usual-tools}

The first tools a software engineer reaches for spend evaluations freely,
because in their usual settings evaluations are cheap.

**Grid search** tries every combination of a few values per parameter. With
$k$ values for each of $d$ parameters it needs $k^d$ evaluations: 10 values for
each of 6 parameters is a million runs. It also wastes evaluations when some
parameters do not matter, since the grid repeats each value of an important
parameter once for every value of the unimportant ones. @bergstra2012random
made this argument for hyperparameter tuning and showed that **random search**,
drawing each input uniformly at random, often does better than a grid with the
same budget: it tries a different value of every parameter on every run.

Random search is a strong baseline, and this book will return to it repeatedly,
because in several settings nothing beats it by much (@sec-rec-simpler). But it
learns nothing from its own results. The hundredth random input is drawn as if
the first ninety-nine had never been evaluated.

**Gradient descent** learns from every step but needs a gradient, which a black
box does not provide. Estimating one by finite differences costs $d + 1$
evaluations per step and is ruined by noise. **Evolutionary methods** and
other population-based searches use their history, but typically need hundreds
to thousands of evaluations to do it, which is what a black-box budget cannot
afford.

```{figure}
//| figure: intro-search-race
//| label: fig-intro-race
//| fig-cap: "Three ways to spend the same budget. Left: where a grid (yellow), uniform random sampling (violet), and Bayesian optimization (orange) evaluate the running objective (dashed); the large dots mark the best point each has found. Right: the best value found so far against the number of evaluations. Press play or drag the timeline to watch the race, switch to 2-D to see a grid coarsen, and start a new random run to see how much luck random search needs. The functions are standard test functions, not real tuning problems."
```

The figure shows the trade-off on the book's running example, a
one-dimensional function with a wide hill on the left and a narrow, taller peak
on the right. In one dimension, sixteen evaluations are enough for every method
to find the peak eventually; the difference is how soon. Bayesian optimization
usually reaches the peak in a handful of evaluations, because after each one it
decides where the next will be most useful. In two dimensions the grid has only
four values per axis and may straddle the peak entirely, while random search
sometimes gets lucky and sometimes does not. The gap widens with every
additional dimension and with every evaluation that costs an hour.

## Model, then decide {#sec-model-then-decide}

Bayesian optimization replaces "try inputs" with a loop of two steps.

The first step keeps a **surrogate**: a statistical model of $f$ built from the
evaluations so far. The surrogate gives, for every input that has not been
tried, a prediction of its score *and an honest statement of how uncertain that
prediction is*. Near inputs already evaluated it is confident; far from them it
is not. The book's surrogate is the Gaussian process of @sec-part-gp, which turns
both statements into a probability distribution over $f(\vx)$ for every input
$\vx$ (bold, because an input is usually a list of several numbers).

The second step is a decision. An **acquisition function** scores every
untried input by how useful evaluating it would be, given the surrogate, and the
loop evaluates the input with the highest score. A good acquisition function
balances two reasons to evaluate an input: its predicted score is high
(*exploitation*), or its score is so uncertain that it might be high
(*exploration*). @sec-part-bo derives the classic acquisition functions from this
balance.

```{figure}
//| figure: bo-loop
//| label: fig-intro-loop
//| fig-cap: "The loop on the running objective. The blue curve and band are the surrogate's prediction and uncertainty; the orange curve below is the acquisition function, whose peak is the next evaluation. Step through it: early on, the acquisition function favors regions where the band is wide; later it concentrates where the prediction is high. @sec-bo-loop explains every part of this picture."
steps: 10
```

Spending effort on a model can look like overhead: the surrogate must be refit
and the acquisition function maximized after every evaluation. But that
computation takes milliseconds to seconds, while each evaluation of $f$ takes
minutes to days. When evaluations are expensive, thinking before every one of
them is the cheapest thing to do. That is the whole case for Bayesian
optimization, and it is also its boundary: when $f$ is cheap, the bookkeeping
stops paying for itself.

The idea is older than machine learning. @kushner1964new proposed choosing
where to evaluate a noisy one-dimensional function using a random-process model
of it, @mockus1975bayesian developed the Bayesian decision view, and
@jones1998efficient made it practical for engineering design as *efficient
global optimization*. @snoek2012practical brought it to machine learning
hyperparameters, and it is now a standard tool in that field, in chemistry and
materials science, and in robotics (@sec-bo-applications).

## When the objective is a person {#sec-person-objective}

The coffee and design problems break an assumption the loop above makes: that
evaluating an input returns a number. A person can rate a cup of coffee from 1
to 10, but ratings drift over a session, depend on what came before, and
compress toward the middle of the scale (@sec-people-objective). What people do
reliably is compare: this cup or that one.

**Preferential Bayesian optimization** (PBO) keeps the loop and changes the
measurement. Each query shows a person two options (or a few), and the answer is
which one they prefer. The surrogate becomes a model of a hidden *utility*, how
much the person likes each option, that would explain their choices; the
acquisition function chooses which pair to show next. You can try it now, before
any of the theory: the figure below shows two colors and learns which you
prefer.

```{figure}
//| figure: pbo-oracle
//| label: fig-intro-oracle
//| fig-cap: "A first taste of PBO. Answer a dozen comparisons; the curve is the model's estimate of how much you like each hue, and the star is its guess at your favorite. @sec-pbo explains how the next pair is chosen."
```

When the person is part of the loop, the system is no longer only measuring.
The pairs it chooses to show may shape what the person comes to like. People
tire, learn, and change their minds, and some decline to choose at all. @sec-part-humans
collects what happened when these methods were used with real people, and
@sec-part-perspectives asks what psychology, neuroscience, economics, and philosophy say about
whether there is a fixed preference to be found. The book treats these questions
as part of the subject, not as caveats.

## How to read this book {#sec-how-to-read}

The book has ten parts. The first four teach, the fifth puts the methods to
work on real problems, and the last five report research.

- @sec-part-foundations, **Foundations**, builds the probability, linear
  algebra, Gaussian distributions, Bayesian inference, and information theory
  the rest needs, for a reader who has not used them since school.
- @sec-part-gp, **Gaussian Processes**, builds the surrogate: distributions
  over functions, regression, and kernels, with the analysis of kernels that
  the regret bounds rely on.
- @sec-part-bo, **Bayesian Optimization**, builds the loop, the acquisition
  functions, the theory of regret, and the practice.
- @sec-part-preferences, **Learning from Comparisons**, extends everything to
  comparisons: choice models, approximate inference, preference learning,
  PBO, query design, and dueling bandits.
- @sec-part-cases, **Case Studies**, works through four real problems end to
  end: a classifier, a chemical reaction, an exoskeleton, and a photograph you
  enhance yourself.
- @sec-part-frontier, **The Research Frontier**, reports what nine years of
  research since @gonzalez2017preferential established about PBO's
  models, acquisition functions, theory, scaling, software, and evaluation.
- @sec-part-humans, **People in the Loop**, collects the evidence from
  interactive design, wearable robots, health, buildings, science, and
  industry.
- @sec-part-neighbors, **Neighbors in Computing**, relates PBO to
  the preference models behind large language models and to reward learning,
  recommendation, ranking, and automated science.
- @sec-part-perspectives, **What Is a Preference?**, reports what other
  disciplines know about the thing being optimized.
- @sec-part-synthesis, **Synthesis**, draws the threads together into
  recommendations, open problems, and an outlook.

The research parts share one argument. By 2026 the algorithms of PBO are
mature, and the hard part has moved to measurement: what a single comparison
measures, how answers should be modeled, and what asking does to the person
who answers. That is why the book spends three parts on people and on what
other disciplines know about preference, and it is the thread to follow
through them (@sec-preface-argument).

Readers who know probability can skim @sec-part-foundations; readers who know
Gaussian processes can start at @sec-part-bo; readers who know Bayesian
optimization can start at @sec-part-preferences.

A few conventions run through every chapter.

- **Figures are live.** Sliders, buttons, and clicks change what they show;
  each caption says what to try and which values are illustrative. Every figure
  also renders without JavaScript.
- **Derivations are written out.** Multi-step derivations appear in boxes with
  one justified step per line, and a *Step through* button reveals them one
  step at a time, so you can try the next step yourself.
- **References sit where they are used.** Hovering over a citation shows the
  full reference. Each section ends with a collapsed list of its sources, each
  chapter with a full list, and each part page with every work its chapters cite.
  Works that are not peer reviewed carry a badge: *preprint*, *working paper*,
  *workshop paper*, *software*, or *non-peer-reviewed*.
- **Inferences are marked.** In the research parts, a sentence that is the
  book's own inference from the evidence, rather than a finding any source
  reports, ends with *(inference)*.
- **Exercises have solutions**, collapsed under each exercise.

## Exercises {#sec-intro-exercises}

::: {.exercise #exr-intro-grid}
A tuning problem has 8 parameters, of which only 2 affect the score. You have a
budget of 256 evaluations. How many distinct values of each important parameter
does a full grid try, and how many does random search try?

::: {.solution}
A full grid with 256 points in 8 dimensions has $256^{1/8} = 2$ values per
parameter, so each important parameter is tried at only 2 distinct values, and
each combination of those values is repeated 64 times across the unimportant
parameters. Random search draws every parameter afresh on every evaluation, so
it tries 256 distinct values of each important parameter (with probability
one). This is the argument of @bergstra2012random.
:::
:::

::: {.exercise #exr-intro-budget}
An evaluation takes 2 hours and fitting the surrogate and maximizing the
acquisition function takes 20 seconds. What fraction of a 50-evaluation run is
spent on Bayesian optimization's bookkeeping? At what evaluation cost would the
bookkeeping take as long as the evaluations?

::: {.solution}
Each iteration spends 20 seconds deciding and 7,200 seconds evaluating, so the
bookkeeping is $20 / 7220 \approx 0.3\%$ of the run. It would equal the
evaluation time when an evaluation takes 20 seconds, at which point the same
budget could buy roughly twice as many random evaluations. In practice the
bookkeeping also grows with the number of observations (@sec-gp-computation),
so for cheap functions with large budgets, simpler methods are often the better
choice.
:::
:::

## Further reading {#further-reading .unnumbered}

- @garnett2023bayesian is a thorough modern textbook on Bayesian optimization,
  with the decision-theoretic view this book also takes.
- @frazier2018tutorial is a concise tutorial covering the Gaussian process
  surrogate and the main acquisition functions.
- @shahriari2016taking is a widely cited review of the field up to 2016.
- @bergstra2012random makes the case for random search as a baseline.
- @gonzalez2017preferential introduced PBO in its current form.
