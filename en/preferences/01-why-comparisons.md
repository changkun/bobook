---
status: done
synopsis: "Why a comparison is often a better measurement of a person than a rating, and the models that turn one into evidence about a hidden utility: psychophysics, Thurstone's comparative judgment, Bradley-Terry-Luce, random utility, how much one answer can carry, and the assumptions to watch."
sources: ["Thurstone 1927", "Bradley and Terry 1952", "Luce 1959", "McFadden 1974", "Miller 1956"]
---

# Why Ask for Comparisons {#sec-comparisons}

Every method in @sec-part-bo assumed that evaluating the objective returns a
number: a validation accuracy, a yield, a walking speed. When the objective is
a person's judgment, the obvious number is a rating, "how good is this, from 1
to 10?", and the obvious plan is to feed ratings to the Gaussian process of
@sec-gp-regression as if they were measurements. @sec-person-objective warned
that this plan runs into trouble, because people are poor at producing such
numbers and much better at saying which of two things they prefer.

This chapter makes that warning precise. It starts with an experiment you run
on yourself, then turns what you observe into a model: a comparison is a noisy
measurement of the difference between two hidden utilities, and a **choice
model**, the rule that maps that difference to the probability of each answer,
is the likelihood that every later chapter in this part uses.

@sec-approx-inference then deals with the mathematical cost of using these
likelihoods, @sec-gp-preference attaches them to a Gaussian process, and
@sec-pbo uses the result to optimize.

## People as the objective {#sec-people-objective}

Start with the figure below. It asks you to judge gray patches in two ways. In
the *rating* task one patch appears on its own, on a dark or a light surround,
and you give it a number from 1 (darkest) to 9 (lightest). In the *comparison*
task two patches appear side by side and you say which is lighter. Do a dozen
or more trials of each, honestly and without going back, then read the panel at
the bottom.

```{figure}
//| figure: cmp-rate-or-compare
//| label: fig-cmp-rate-or-compare
//| fig-cap: "Rate or compare. The patches take nine levels of lightness. Both tasks are scored on the same question: for two patches one or two levels apart, how often is the lighter one ranked higher? For ratings, every pair of rated patches is checked (a tie counts as half); for comparisons, every answered pair. The scatter shows each rating against the patch's true level, with filled dots for patches seen on the dark surround. Switch to the simulated observer to see the pattern an observer with the illustrative noise described in the text produces."
```

The bar chart asks both tasks one question, so they can be compared directly.
A rating does not answer it on its own: two ratings, given at different moments
to different patches, have to be set side by side after the fact. A comparison
answers it in a single trial. The simulated observer shows the pattern the rest
of this section explains. Its ratings put patches one level apart in the right
order only about seven times in ten, while its comparisons almost never err.
Its numbers are illustrative, not measured: each rating adds to the true level
a shift of 0.6 levels (up on the dark surround, down on the light one), a slow
drift in how the scale is used, and noise with a standard deviation of 0.8
levels, and is then rounded to the scale; each comparison sees only noise of
0.5 levels on the difference, because the surround and the drift are shared by
both patches. Your own numbers will be noisier with a dozen trials, but the gap
is usually easy to see, because the two tasks ask different things of you.

### Why ratings are hard {#sec-cmp-ratings-hard}

The first reason is that people can tell apart far fewer levels in isolation
than side by side. @miller1956magical collected experiments in which listeners
and viewers had to identify a single stimulus by giving it a number, as in the
rating task above. For the pitch of a tone, the information that got through
leveled off at about 2.5 bits, the equivalent of about six levels that a
listener never confuses; across all the one-dimensional attributes he reviewed
the mean was 2.6 bits, with a standard deviation of only 0.6 bit. The nine
levels in @fig-cmp-rate-or-compare are more than the six or so that Miller
found typical, yet any two adjacent levels are easy to tell apart when shown
together. The first remedy Miller listed for this limit was to make
relative rather than absolute judgments.

The second reason is that a number depends on its context. The patch is the
same gray whatever surrounds it, yet a gray on a dark surround looks lighter
than the same gray on a light one, the simultaneous contrast that Thurstone
already noted for gray values [@thurstone1927law]; the scatter in the figure shows whether
your ratings shifted with the surround. Ratings of real things show the same
dependence on what came just before. Preference ratings of photographs and
faces are pulled toward the rating of the previous item, and the pull survives
controls for response bias [@chang2017sequential]. In 2.2 million Yelp ratings
and 4.2 million Amazon ratings, a reviewer's rating was pushed *away* from
their previous ratings, a contrast effect [@vinson2019decision]. Attractiveness
ratings are pulled toward the previous face on average, but how strongly a
given person is pulled is not stable from one sequence to the next
[@kramer2026sequential]. The direction depends on the setting; the dependence
itself is reliable.

The third reason is that people use the scale unevenly. More than a century
ago @hollingworth1910central described the *central tendency of judgment*:
estimates of magnitude drift toward the middle of the range of stimuli a person
has been shown, so small values are overestimated and large ones
underestimated. A rating scale is likely to compress toward its middle in the
same way (inference).

These effects appear in exactly the settings this book is about. In a
three-month deployment of a preference-guided optimizer for processing 3D
meshes, in which two professional artists rated candidate meshes, the
optimization lacked mechanisms to deal with "inconsistent and contradictory
human judgments", and what the system showed influenced later answers through
heuristic biases and loss aversion; interviews pointed to anchoring on meshes
seen earlier and to judgments losing precision after a run of increasingly
good results [@ou2022human]. @koyama2020computational argue that an absolute rating
requires familiarity with the whole design space, which a person meeting the
space for the first time does not have, while a comparison between two
options can be answered at once. @sec-cs-photo works through one such
problem end to end, adjusting a photograph by choosing between versions.

### Why comparisons help {#sec-cmp-why-compare}

A comparison does not make these effects disappear. It arranges for many of
them to cancel. If your mood, the surround, or your sense of where "5" sits on
the scale shifts both options by the same amount, the difference between them
is untouched, and a comparison reports only the sign of that difference.
Thurstone made this argument in 1927: looking at two handwriting specimens "in
a mood slightly more generous and tolerant than ordinarily" raises the
impression of both, and to that extent the two impressions vary together
[@thurstone1927law]. @sec-thurstone shows how this shared variation drops out
of the comparison.

::: {.keyidea title="A comparison measures a difference"}
A rating mixes the person's preference with everything that sets their scale
at that moment. A comparison reports the sign of a difference, and a
difference cancels whatever shifts both options equally.
:::

Direct evidence that people compare more reliably than they rate comes from
several fields. In information retrieval, where assessors judge which of two
documents better answers a query, preference judgments are made faster and
more consistently than graded judgments [@clarke2021assessing]. A 2014
preprint reports experiments on Amazon Mechanical Turk across a variety of
tasks in which pairwise comparisons had lower noise per answer and were
typically faster to collect than numerical scores, though each answer carried
less information [@shah2014it]. In a study of how people report preferences in
markets, participants found it harder to report cardinal information (how
much they prefer something) than ordinal information (which they prefer)
[@budish2022market].

The advantage is not universal, and it is worth knowing where it fails. When
162 people repeated a health valuation task two weeks apart, 76.4% of their
discrete choices were the same (kappa 0.528, a measure of agreement corrected
for chance, where 1 is perfect), while the numerical time trade-off method
reached an intraclass correlation of 0.958 (the share of the variance that
comes from differences between people, not between occasions) even though
only 59.3% of its values were identical [@xie2022discrete]. For risk
preference, a meta-analysis of test-retest correlations found that
self-reported propensity to take risks was more stable over time than
behavioral measures such as lottery choices, with estimated reliabilities of
0.61 and 0.25 [@bagaini2025systematic]. These measures
are not comparisons and ratings of the same stimuli, so they do not settle the
question; they show that "compare instead of rate" is a hypothesis about a
task, not a law. Before 2026 we found no study that compared pairwise
comparisons, galleries, sliders, rankings, and ratings on the same design task
with real users; the first head-to-head comparisons of feedback formats
appeared in 2026 (@sec-hci-feedback-forms). The case for
comparisons is strongest where it matters most for this book: when there is no
external unit to anchor a rating, and when the context drifts during a session
(inference).

## Psychophysics {#sec-psychophysics}

If judgments are noisy, can the noise itself be used to measure something?
That was the program of nineteenth-century psychophysics, the study of how
physical stimuli map to sensations, and its answer shaped every model in this
chapter.

The starting observation is that a small enough difference cannot be told
apart reliably. Ask someone to lift two weights and say which is heavier, and
the smallest difference they notice, the **just-noticeable difference**, grows
with the weights themselves: roughly in proportion, so that the ratio of the
just-noticeable difference to the weight stays about constant. Fechner named
this regularity **Weber's law**, after Ernst Heinrich Weber's experiments on
lifted weights, and built on it [@fechner1860elemente]. If every
just-noticeable difference is one equal step of sensation, then counting steps up
from the threshold gives a sensation that grows with the logarithm of the
intensity, which is **Fechner's law**. Fechner also systematized the methods
for measuring discrimination, among them the one now called the *method of
constant stimuli*: present a fixed standard with a comparison stimulus many
times and record how often the comparison is judged larger.

The method of constant stimuli produces the curve that this chapter is about.
Plot the proportion of "comparison is heavier" answers against the true
difference, and it rises smoothly from near 0, through one half where the two
are equal, to near 1. This S-shaped curve is the **psychometric function**. Its
steepness measures the observer's noise: a precise observer has a steep curve,
a noisy one a shallow curve. The just-noticeable difference is usually defined
from it, as the difference that is judged correctly on some fixed fraction of
trials, often 75%.

A different method gives a different law. Instead of asking people to compare,
@stevens1957psychophysical asked them to assign numbers directly, "if this
sound is 10, how loud is that one?", a method called magnitude estimation. The
numbers grew as a power of the intensity rather than as its logarithm, with an
exponent that depends on the sensory continuum. Both laws describe data well
within their own method. The lesson for this book is that a number a person
gives directly is not a neutral readout of their sensation: it passes through
their own mapping from sensation to numbers, while discrimination data measure
something else, how often two things are confused. Comparisons measure
utilities in units of the person's noise, as Fechner counted sensation in
just-noticeable differences. @sec-pref-identifiability returns to what this means
for a learned utility.

Modern psychophysics has refined the picture without overturning it. Part of
the noise and bias in judgments of value is now explained as efficient
coding: the brain adapts its scale to the range of values it has recently
encountered, so the same option can be valued differently in a different
context [@bavard2018reference]. @sec-psychophysics-attention and
@sec-neuro-coding report this work in detail; here it is one more reason to
expect a person's scale to move during a session.

## Thurstone's comparative judgment {#sec-thurstone}

Fechner's psychophysics needed a physical scale, grams or decibels, on which to
measure the stimulus. @thurstone1927law removed that requirement. His law of
comparative judgment applies, in his words, "not only to the comparison of
physical stimulus intensities but also to qualitative comparative judgments
such as those of excellence of specimens", such as handwriting samples,
children's drawings, or opinions on public issues. This is the step that makes
comparisons useful for preferences: there is no physical scale of how much
someone likes a color, but there can still be a psychological one.

Thurstone's model has three ingredients. Each time an observer looks at a
stimulus, it evokes a *discriminal process*, a value on a psychological scale.
The process fluctuates from occasion to occasion, which is why the observer
gives different answers to the same pair on different occasions. Its most
frequent value is the stimulus's **scale value** $S$, and the standard
deviation of its fluctuation is the stimulus's **discriminal dispersion**
$\sigma$. On each occasion the observer reports as better the stimulus whose
process is higher at that moment.

Thurstone defined the scale so that the fluctuations are normally distributed,
which makes the probability of each answer computable. Write $u_A = S_A +
\varepsilon_A$ and $u_B = S_B + \varepsilon_B$ for the processes evoked by
stimuli $A$ and $B$ on one occasion.

::: {.derivation title="The law of comparative judgment"}
Let $\varepsilon_A$ and $\varepsilon_B$ be jointly Gaussian with mean zero,
standard deviations $\sigma_A$ and $\sigma_B$, and correlation $r$.

1. The observer answers "$A$" when $u_A > u_B$, that is, when the
   *discriminal difference* $D = u_A - u_B$ is positive.
2. $D$ is a linear function of a Gaussian vector, so it is Gaussian
   (@sec-gaussian-linear). Its mean is $S_A - S_B$.
3. Its variance is $\Var[\varepsilon_A - \varepsilon_B] = \sigma_A^2 + \sigma_B^2 - 2r\sigma_A\sigma_B$,
   by the rule for the variance of a difference (@sec-expectation).
4. Standardize: $\Prob(D > 0) = \Prob\!\left(\frac{D - (S_A - S_B)}{\sqrt{\Var D}} > -\frac{S_A - S_B}{\sqrt{\Var D}}\right)$,
   and the standardized variable is $\N(0, 1)$.
5. By the symmetry of the standard normal, $\Prob(Z > -c) = \Phi(c)$, where
   $\Phi$ is the standard normal cumulative distribution function. Hence,
   writing $A \succ B$ for "$A$ is preferred to $B$",
   $\Prob(A \succ B) = \Phi\big((S_A - S_B)/\sqrt{\sigma_A^2 + \sigma_B^2 - 2r\sigma_A\sigma_B}\big)$.
:::

Writing $p_{AB}$ for the observed proportion of "$A$" answers and $x_{AB} =
\Phi^{-1}(p_{AB})$ for the corresponding standard normal deviate, the result
is Thurstone's law of comparative judgment:

$$
S_A - S_B = x_{AB}\,\sqrt{\sigma_A^2 + \sigma_B^2 - 2r\,\sigma_A\sigma_B}.
$$ {#eq-cmp-thurstone}

The correlation $r$ is where the shared shifts of @sec-people-objective live.
A mood that raises the impression of both specimens makes their fluctuations
move together, $r > 0$, and the variance of the difference shrinks: the shared
part has cancelled. Thurstone also noted the opposite case. In simultaneous
contrast, seeing one gray next to a darker one makes it look lighter and the
darker one look darker, so the fluctuations move apart, $r < 0$, and the
difference is exaggerated [@thurstone1927law].

Equation @eq-cmp-thurstone has too many unknowns to fit in general, so
Thurstone listed five cases with progressively stronger assumptions. The last
and simplest, **Case V**, assumes that all discriminal dispersions are equal
and the correlation is zero (Thurstone suggested it was "legitimate for rough
measurement"). With a common dispersion $\sigma$, the probability becomes

$$
\Prob(A \succ B) = \Phi\!\left(\frac{S_A - S_B}{\sqrt{2}\,\sigma}\right).
$$ {#eq-cmp-case-v}

This is the **probit** choice model, named after the probability unit, an old
name for a standard normal deviate. Read it as a recipe: the probability of
choosing $A$ depends only on the difference in scale values, measured in units
of the noise, and passes through one half when the difference is zero.
Replace the scale value $S$ by a utility function $g(\vx)$ of a design $\vx$,
and @eq-cmp-case-v is the likelihood of Chu and Ghahramani's preference model
[@chu2005preference], used in @sec-gp-preference and in every chapter after it.

Thurstone used the model in the other direction, to measure. Taking $\sigma$ as
the unit, Case V gives $S_A - S_B = \sqrt{2}\,x_{AB} \approx 1.4142\,x_{AB}$,
his equation (4). If 75% of judgments prefer $A$ to $B$, then $x_{AB} =
\Phi^{-1}(0.75) \approx 0.674$ and $A$ sits about $0.95\sigma$ above $B$. If
99% prefer $A$, the gap is about $3.29\sigma$. Collecting such gaps for many
pairs places every stimulus on one scale, with no physical measurement
anywhere.

::: {.example #ex-cmp-scaling title="Scaling three options"}
Suppose $A$ is preferred to $B$ in 69% of judgments and $B$ to $C$ in 84%.
Case V gives $S_A - S_B = \sqrt{2}\,\Phi^{-1}(0.69) \approx 1.414 \times 0.496 \approx 0.70\sigma$
and $S_B - S_C = \sqrt{2}\,\Phi^{-1}(0.84) \approx 1.414 \times 0.994 \approx 1.41\sigma$.
If the model holds, $S_A - S_C \approx 2.11\sigma$, so it predicts that $A$
is preferred to $C$ in $\Phi(2.11/1.414) = \Phi(1.49) \approx 93\%$ of
judgments. Observing the third proportion is therefore a test: the gaps must
add up. A large mismatch means the options do not lie on one scale with equal
noise, which is the subject of @sec-comparison-assumptions.
:::

Thurstone also observed that proportions are not equally informative. A
proportion of 0.99 and one of 0.55 do not pin down their scale differences
equally well [@thurstone1927law]: near 0 or 1, a small change in the
proportion corresponds to a large change in the difference, so a pair whose
outcome is nearly certain says little about how far apart the options are.
@sec-comparison-information makes this precise.

## Bradley, Terry, and Luce {#sec-bradley-terry}

A quarter century later, statisticians working on paired-comparison
experiments arrived at a second model by a different route.
@bradley1952rank gave each option $i$ a positive *worth* $\pi_i$ and set

$$
\Prob(i \succ j) = \frac{\pi_i}{\pi_i + \pi_j}.
$$ {#eq-cmp-bt}

Writing the worth as an exponential of a utility, $\pi_i = \exp(g_i / \tau)$
with a positive scale $\tau$, and dividing through by $\pi_i$ turns
@eq-cmp-bt into

$$
\Prob(i \succ j) = \frac{1}{1 + \exp\!\big(-(g_i - g_j)/\tau\big)},
$$ {#eq-cmp-logit}

the **logistic** function, $\operatorname{sigmoid}(z) = 1/(1 + e^{-z})$, of
the scaled utility difference, also called the *logit* model. Like Case V, it depends only on the difference of utilities, it
is one half at a tie, and it approaches 0 and 1 for large differences. The
scale $\tau$ plays the role of the noise: a small $\tau$ makes the choice
nearly deterministic.

@luce1959individual extended the model from pairs to sets. His **choice axiom**
implies that the probability of choosing $i$ from any set $T$ of options is

$$
\Prob(i \mid T) = \frac{w_i}{\sum_{j \in T} w_j},
$$ {#eq-cmp-luce}

for positive weights $w$, the rule now usually called the *softmax* when $w_i =
\exp(g_i/\tau)$. For a set of two it is the Bradley-Terry model. The same idea
gives a model for rankings: choose the first-ranked option from the full set by
@eq-cmp-luce, the second from the remaining options, and so on.
@plackett1975analysis developed this model for permutations, and it is known
as the **Plackett-Luce** model. @sec-query-rankings uses it when a person
orders several options instead of answering a *duel*, the field's word for
one pairwise comparison, which the rest of the book uses too.

The axiom has a strong consequence. Divide @eq-cmp-luce for two options $i$
and $j$ in the same set: $\Prob(i \mid T) / \Prob(j \mid T) = w_i / w_j$,
whatever else is in $T$. This property is the **independence of irrelevant
alternatives**: the odds of choosing tea over coffee do not depend on whether
juice is also on offer. It is convenient and often false. A thought experiment
shows why: if a café offers coffee and tea and a person picks each half the
time, adding a second, identical pot of coffee should not change much, yet
under @eq-cmp-luce the two coffees together take two thirds of the choices.

Real choices violate the property in subtler ways. Adding a third option can
make one of the original two more attractive (the attraction or decoy effect),
make the middle option more attractive (the compromise effect), or draw share
mostly from the option most similar to it (the similarity effect). These
context effects hold on average in groups, but individuals rarely show all
three, and averaging can produce a pattern that no individual shows
[@liew2016appropriacy]. A dominated option can even make the option that
dominates it look worse, a repulsion effect that depends on how the options
are displayed [@spektor2018good]. A review attributes the appearance,
disappearance, and reversal of context effects to the spatial layout of the
options, how concrete their attributes are, and how long people deliberate
[@spektor2021elusiveness]. A pure duel has no third option, so the classic
effects need a set to act on; galleries, and options still in memory from
earlier queries, can bring them back (inference). @sec-jdm reports this
evidence, and @sec-interface-model what it means for interface design.

## Random utility models {#sec-random-utility}

The probit and the logit look like two unrelated formulas. They are two
instances of one idea, which economists developed into the main framework for
analyzing choices: the **random utility model**. Each option $i$ has a
systematic utility $g_i$, and on each occasion the person perceives
$U_i = g_i + \varepsilon_i$, with random noise $\varepsilon_i$; they choose the option
whose perceived utility is largest. Thurstone's discriminal process is a
random utility with Gaussian noise. @mcfadden1974conditional showed that when
the noise terms are independent with a **Gumbel** distribution (also called
the double exponential or type I extreme value distribution), the choice
probabilities are exactly Luce's rule @eq-cmp-luce with $w_i = \exp(g_i /
\beta)$, the *conditional logit* model, which became the starting point of
discrete choice analysis in economics (@sec-econ-conjoint).

So the choice model is a claim about the noise: Gaussian noise gives the
probit, Gumbel noise gives the logit. The curve that turns a utility
difference into a choice probability is also called the **link function**, or
link for short, and from here on we speak of the probit link and the logit
link. @yellott1977relationship studied how Luce's axiom, Thurstone's theory,
and the double exponential distribution are connected. For two options the
connection is short enough to derive.

::: {.derivation title="Gumbel noise gives the logistic link"}
Let $\varepsilon_A$ and $\varepsilon_B$ be independent Gumbel variables with
scale $\beta$. A location shift common to both cancels in the difference, so
take the standard form with cumulative distribution function
$F(u) = \exp(-e^{-u/\beta})$ and density $f(u) = \tfrac{1}{\beta} e^{-u/\beta} F(u)$.
Let $\Delta = g_A - g_B$.

1. $A$ is chosen when $\varepsilon_B < \Delta + \varepsilon_A$. Conditioning on
   $\varepsilon_A = a$ and averaging over $a$ (the law of total probability),
   $\Prob(A \succ B) = \int_{-\infty}^{\infty} f(a)\, F(a + \Delta)\, \dd a$.
2. Substitute $t = e^{-a/\beta}$. Then $F(a + \Delta) = \exp(-t\, e^{-\Delta/\beta})$,
   and $f(a)\, \dd a = \tfrac{1}{\beta} t\, e^{-t}\, \dd a = -e^{-t}\, \dd t$,
   since $\dd t = -\tfrac{1}{\beta} t\, \dd a$. As $a$ runs from $-\infty$ to
   $\infty$, $t$ runs from $\infty$ to $0$.
3. The integral becomes
   $\int_0^\infty e^{-t}\, e^{-t e^{-\Delta/\beta}}\, \dd t = \int_0^\infty e^{-t(1 + e^{-\Delta/\beta})}\, \dd t$.
4. The integral of $e^{-ct}$ over $[0, \infty)$ is $1/c$, so
   $\Prob(A \succ B) = \dfrac{1}{1 + e^{-\Delta/\beta}}$, the logistic
   @eq-cmp-logit with $\tau = \beta$.
:::

The figure below makes the random utility story concrete. The top panel shows
the perceived utilities of two options with a given difference and noise. Each
press of play simulates one occasion: the person perceives one value for each
option, marked by triangles, and chooses the higher. The middle panel shows the
probability of choosing $A$ against the utility difference, with the share of
$A$ choices among the occasions so far.

```{figure}
//| figure: cmp-choice-curve
//| label: fig-cmp-choice-curve
//| fig-cap: "A comparison as a random utility. Top: the perceived utility of each option on one occasion, with Gaussian or Gumbel noise of the same standard deviation σ; the triangles mark the latest simulated occasion, and the filled one was chosen. Middle: the probability of choosing A against the utility difference, for the chosen noise (solid) and the other (dashed), with the share of A among the simulated occasions and its 95% interval. Turn on the cost panel to see −log P(B chosen), the penalty each link charges for an answer that goes against the difference."
```

Some things to try:

- **Shrink the noise.** As $\sigma$ falls the curve steepens toward a step: a
  noise-free person always picks the better option, and the probability carries
  no information about *how much* better it is. As $\sigma$ grows the curve
  flattens toward one half everywhere.
- **Switch the noise.** The figure matches the two noise distributions in
  standard deviation, which for Gumbel noise means a scale $\beta = \sigma\sqrt{6}/\pi$.
  The two curves then nearly coincide. Their largest gap is about 0.023, at a
  difference of about 0.68 standard deviations of the perceived difference
  $u_A - u_B$.
- **Look at the tails.** The difference shows up far from zero. When the
  utility difference is 3 standard deviations of $u_A - u_B$, the logit gives
  the worse option a probability of about 0.0043 and the probit about 0.0013,
  3.2 times less. At 4 standard deviations the ratio is about 22.
- **Turn on the cost.** The penalty a model pays, in log likelihood, for an
  answer against the difference grows linearly in the difference for the logit
  and quadratically for the probit. One surprising answer pulls a probit model
  much harder than a logit model.
- **Press play.** Each occasion is a fresh draw. With 60 occasions the share
  of $A$ choices usually lands within its interval of the curve, but any
  single answer can go either way.

The last two points matter in practice. A person who is careless once, or
misreads one pair, gives an answer far out in the tail. Under the probit that
answer can dominate the fit; under the logit its influence is bounded. BoTorch's
probit implementation clips the argument of $\Phi$ to $[-3, 3]$, which caps the
penalty for any single comparison, a safeguard that @sec-obs-pairwisegp
examines [@botorch2026likelihood].

::: {.pitfall title="Per-option noise and noise on the difference"}
Papers write the probit link in at least three ways: $\Phi(\Delta/(\sqrt{2}\sigma))$
with noise $\sigma$ on each option, as in @eq-cmp-case-v; $\Phi(\Delta/\sigma)$
with noise $\sigma$ on the difference; and $\Phi(\Delta)$ with the noise
absorbed into the scale of $g$. BoTorch's `PairwiseGP` uses
$\Phi((g(v) - g(u))/\sqrt{2})$, which is per-option noise fixed at 1
[@botorch2026likelihood]. The three describe the same model with different
units, but a noise value copied from one paper into another's formula is off by
a factor of $\sqrt{2}$. The same care applies when comparing the probit with
the logit: match standard deviations, $\tau = \sigma\sqrt{6}/\pi$, not the raw
parameters.
:::

What is the noise? A random utility model is agnostic. The randomness can be a
person's moment-to-moment fluctuation, as in Thurstone's single observer; it
can be attributes of the options that the analyst does not observe; or it can
be variation across the people in a sample. For one person answering a
sequence of comparisons, the first reading is the natural one, and it can be
tested. @mccausland2020testing asked 141 participants to choose among five
lotteries, six times from every subset of at least two, and applied a set of
inequalities that choice probabilities must satisfy if *any* random utility
model generated them. Most participants were consistent with random utility;
only 4 showed strong evidence of violating it.

One more property is shared by every model in this section, and it shapes the
rest of @sec-part-preferences. The choice probability depends on utilities only through
$\Delta/\sigma$ (or $\Delta/\tau$). Adding a constant to every utility changes
nothing, and doubling every utility while doubling the noise changes nothing.
Comparisons can therefore determine utilities only up to a shift, and only in
units of the noise. @sec-pref-identifiability works out what this means for a
Gaussian process utility.

::: {.keyidea title="The link function is a claim about noise"}
Probit and logit are the same model, a random utility with the better-looking
option chosen, under Gaussian and Gumbel noise. They agree near a tie and
differ in the tails, where they disagree about how surprising a surprising
answer is.
:::

## What a comparison carries {#sec-comparison-information}

A rating on a scale of 1 to 9 can, in principle, carry $\log_2 9 \approx 3.2$
bits. A comparison has two possible answers, so it can carry at most one bit:
the information an answer gives about anything is bounded by the entropy of the
answer, and the entropy of a binary answer is at most $\log_2 2 = 1$ bit
(@sec-mutual-information). Most comparisons carry much less. To see how much,
and which comparisons carry the most, combine the choice model with what the
model already believes.

Suppose the model's belief about the difference $\Delta = g(A) - g(B)$ is
Gaussian, $\Delta \sim \N(m, v^2)$: its best guess is $m$ and its uncertainty
is $v$. The answer follows the probit @eq-cmp-case-v with per-option noise
$\sigma$; write $s = \sqrt{2}\,\sigma$ for the noise on the difference. Before
asking, the model predicts the answer by averaging the choice probability over
its belief.

::: {.derivation title="Predicting a comparison under uncertainty"}
1. Write the choice probability as an event: $\Phi(\Delta/s) = \Prob(\eta < \Delta \mid \Delta)$
   for an independent $\eta \sim \N(0, s^2)$, by the definition of $\Phi$.
2. Average over the belief: $\E[\Phi(\Delta/s)] = \Prob(\eta < \Delta) = \Prob(\Delta - \eta > 0)$,
   by the law of total probability.
3. $\Delta - \eta$ is a sum of independent Gaussians, so it is Gaussian with
   mean $m$ and variance $v^2 + s^2$ (@sec-gaussian-sums).
4. Standardizing as in the derivation of @eq-cmp-case-v,
   $\Prob(\Delta - \eta > 0) = \Phi\big(m / \sqrt{v^2 + s^2}\big)$.
:::

$$
\Prob(A \succ B) = \Phi\!\left(\frac{m}{\sqrt{v^2 + 2\sigma^2}}\right).
$$ {#eq-cmp-predictive}

The model's own uncertainty adds to the person's noise. A pair the model is
unsure about is predicted closer to one half than the same pair would be if the
model knew the difference. @sec-pref-predict uses exactly this formula to
predict a new comparison from a Gaussian process posterior.

How much will the answer teach? The answer is uncertain for two reasons: the
model does not know $\Delta$, and even if it did, the person is noisy. Only the
first kind of uncertainty can be reduced by asking, so the information the
answer carries about $\Delta$ is the total uncertainty minus the noise part:

$$
I = h\!\left(\Phi\!\left(\frac{m}{\sqrt{v^2 + s^2}}\right)\right) - \E_{\Delta}\!\left[h\!\left(\Phi\!\left(\frac{\Delta}{s}\right)\right)\right],
$$ {#eq-cmp-info}

where $h(p) = -p\log_2 p - (1 - p)\log_2(1 - p)$ is the entropy, in bits, of a
yes-or-no answer with probability $p$, and the expectation is over the belief
$\Delta \sim \N(m, v^2)$. The first term is the entropy of the predicted
answer; the second is the entropy the answer would keep if $\Delta$ were known,
averaged over the values the model considers plausible. This decomposition is
the basis of the *Bayesian active learning by disagreement* criterion of
@houlsby2011bayesian, which they also applied to preference learning: the most
informative question is one whose answer the model cannot predict, but whose
answer would be predictable if the model knew the truth.

```{figure}
//| figure: cmp-information
//| label: fig-cmp-information
//| fig-cap: "What one comparison can teach about a utility difference Δ, from @eq-cmp-info. Top: the model's belief about Δ (blue, scaled to fit) and the probit choice curve (orange). Bottom: as the belief's mean m varies, the entropy of the predicted answer (dashed), the entropy left if Δ were known (dotted), and their gap, the expected information in bits (orange, shaded). The numbers are in bits; the noise and belief values are illustrative."
```

The figure shows why two kinds of pairs teach little. Some things to try:

- **A known near-tie.** Set the mean to zero and shrink the belief's spread
  $v$. The predicted answer is a coin flip, one full bit of uncertainty, but
  almost all of it is the person's noise. The answer carries almost nothing,
  0.006 bits at $v = 0.05$ and $\sigma = 0.3$, because the model already knows
  the options are nearly equal.
- **A known gap.** Move the mean far from zero. The answer is foregone, both
  entropies fall toward zero, and so does the information: about 0.02 bits at
  $m = 3$ with $v = 1$ and $\sigma = 0.3$.
- **An open question.** With the mean near zero and the spread large compared
  with the noise, the model cannot predict the answer and would be able to if
  it knew $\Delta$. This is where a comparison is worth asking: about 0.6 bits
  at $m = 0$, $v = 1$, $\sigma = 0.3$, rising toward a full bit as the noise
  vanishes.
- **A noisy person.** Raise $\sigma$. The dotted curve rises toward the dashed
  one and every comparison teaches less. Noise cannot be designed away; it sets
  the price of each answer.

The most useful comparison, then, pairs options whose order the model is
unsure of, relative to how noisy the person is. That is the intuition behind
the acquisition rules of @sec-choosing-pairs and @sec-eubo, and behind the
query designs of @sec-query-design.

Fewer bits per answer do not necessarily mean slower learning. For estimating
the utilities of a fixed set of options under the Thurstone and Bradley-Terry
models, @shah2016estimation proved minimax bounds, the best error any method
can guarantee in the worst case, and found that the error depends on the
topology of the comparison graph (which pairs were compared) through the
eigenvalues of its graph Laplacian, a matrix that records which pairs were
compared, and that the ordinal and cardinal settings have error
rates with the same scaling, up to constant factors. Each comparison carries
less than a numerical measurement would, but the rate at which errors shrink
with more data is the same (inference). @sec-comparison-graph explains what the
comparison graph is and why its structure matters for a Gaussian process
utility too.

A related result tempers the role of the link function. For actively ranking a
set of items from noisy comparisons, @heckel2019active showed that a simple
counting method that assumes no parametric model is optimal up to logarithmic
factors, so parametric assumptions such as Bradley-Terry or Thurstone buy at
most a logarithmic gain. For preferential Bayesian optimization (PBO) this suggests
that sample efficiency comes mainly from the kernel sharing information between
nearby inputs, not from the exact form of the link (inference).

The point sharpens as the number of design parameters grows. A comparison
between two exoskeleton gaits that differ in four parameters [@li2021roial],
or between two simplifications of a 3D mesh controlled by nine parameters
[@ou2022human], still returns one bit at most, while the number of designs
that would have to be told apart grows exponentially with the number of
parameters. Sessions with people rarely run beyond a few dozen comparisons
(@sec-part-humans), so no choice model can extract more than a few dozen bits
from one. The rest has to come from assumptions about how utility varies
across the design space, the kernel of @sec-gp-preference, and from the choice
of which pairs to ask, the subject of @sec-pbo (inference).

## Assumptions to watch {#sec-comparison-assumptions}

Every model in this chapter, and the Gaussian process preference model built
on them in @sec-gp-preference, makes assumptions about the person answering.
They are reasonable starting points, and each has been tested.
@tbl-cmp-assumptions lists them with the evidence; the paragraphs after it add
what the table cannot hold.

::: {.table #tbl-cmp-assumptions title="Assumptions of the standard comparison model, and where the evidence stands."}
| Assumption | What the evidence says | Where the book returns |
|---|---|---|
| One stable utility behind every answer | Choosing changes preferences: a meta-analysis of 43 artifact-free studies finds a shift of $d = 0.40$ standard deviations | @sec-jdm, @sec-rec-nonstationary |
| Answers are independent given the utility | Choices and ratings depend on the preceding trials | @sec-psychophysics-attention |
| Preferences are transitive | Most people satisfy random utility; true cycles exist in specific designs | @sec-math-psych, @sec-dueling-bandits |
| The noise is the same for every pair | How noise is specified changes the inferred preferences | @sec-math-psych, @sec-obs-extensions |
| Every forced choice reflects a preference | People report preferences even between identical samples | @sec-ties |
| Choices do not depend on other options | Context effects exist but are conditional | @sec-jdm, @sec-query-forms |
| The utility scale is fixed across sessions | Values adapt to the recent range | @sec-psychophysics-attention |
:::

**Stability.** The models treat the person as a fixed utility plus noise. But
choosing can change what people like. A meta-analysis of 43 studies using the
free-choice paradigm with the known artifact removed (N = 2,191) found that
after people choose between two similar options, they rate the chosen one
higher and the rejected one lower, with an effect size of $d = 0.40$, a shift
of 0.40 standard deviations (95% confidence interval 0.32 to 0.49), and no
evidence of publication bias
[@enisman2021choice]. A sequential-sampling account explains part of the
mechanism: each choice raises the value of the chosen option and lowers that of
the rejected one, and the consistency of repeated choices between the same pair
declines as more trials intervene [@zylberberg2024value]. An optimizer that
keeps showing its current favorite may therefore be reinforcing it (inference).

**Independence.** The likelihood multiplies the probabilities of the answers as
if each were a fresh draw. The serial dependence just described, and the
sequential effects on ratings in @sec-people-objective, say that an answer
partly depends on what came before. The effects are reliable on average but
unstable within individuals [@kramer2026sequential], which makes them hard to
correct one person at a time.

**Transitivity.** Random utility models with independent noise are transitive
in a probabilistic sense: if $A$ usually beats $B$ and $B$ usually beats $C$,
then $A$ usually beats $C$. Direct tests mostly support this
[@mccausland2020testing], but not all observed cycles are noise. Using
response times to separate noise from preference, a 2023 working paper found
that transitivity violations shrink but do not disappear: on average across
participants, 19.24% and 13.83% of the cycles with revealed preferences in two
reanalyzed data sets were violations, most often arising from chains of small
trade-offs between attributes [@alosferrer2023identifying]. Lotteries designed
after the Steinhaus-Trybula paradox likewise produced cycles as the most
common pattern even after allowing for transitive preferences with noise
[@butler2018predictably]. @sec-dueling-bandits discusses what can be optimized
when no utility exists.

**Homogeneous noise.** Case V and the logit give every pair the same noise.
In risky choice, how the noise is specified changes what is inferred: combining
noise in preferences with noise in responding can make an expected-value
maximizer look risk averse or risk seeking [@bhatia2017noisy], and with
homogeneous noise the inferred risk aversion can behave non-monotonically, which
led @apesteguia2018monotone to recommend random-parameter models instead.
@sec-obs-extensions lists the preference models that let noise vary.

**Forced choice.** The duel offers no "neither" and no "I can't tell". In
sensory science, consumers asked to choose between two identical samples still
report a preference, which is why that field moved to "no preference" options
and placebo pairs [@omahony2017evolution]. @sec-ties covers likelihoods that
allow ties.

**Context and scale.** The models assume the choice between two options does
not depend on what else has been seen, and that the utility scale is the same
in every session. The context effects of @sec-bradley-terry and the range
adaptation of @sec-psychophysics both cut against this. If a person's values
are normalized to the range seen in a session, a utility learned in one session
is on a session-specific scale, and reusing it in another session needs
recalibration (inference).

None of this means the models are wrong to use. It means their output is an
estimate under assumptions that can be checked: by repeating a few early pairs
late in a session, by allowing ties, by randomizing which side an option
appears on, and by comparing the fitted noise level with the observed rate of
reversed answers. @sec-part-perspectives asks the deeper question these checks circle around,
whether repeated comparisons find a preference or partly make one
(@sec-what-comparisons-measure).

## Exercises {#sec-cmp-exercises}

::: {.exercise #exr-cmp-scaling}
In a taste test, $A$ beats $B$ in 75% of trials, $B$ beats $C$ in 75%, and $A$
beats $C$ in 84%. Under Case V, place the three options on one scale with
$\sigma = 1$, and check whether the three proportions are consistent with it.
What would a logistic model with matched standard deviation predict for $A$
against $C$?

::: {.solution}
From @eq-cmp-case-v, $S_A - S_B = S_B - S_C = \sqrt{2}\,\Phi^{-1}(0.75) \approx 1.414 \times 0.674 \approx 0.954$.
If the scale is consistent, $S_A - S_C \approx 1.908$, and Case V predicts
$\Prob(A \succ C) = \Phi(1.908/1.414) = \Phi(1.349) \approx 0.911$. The
observed 84% is lower, so the gaps do not add up: either the noise differs
between pairs (Cases I to IV allow this), or the proportions are noisy
estimates. With matched standard deviation the logistic scale is $\tau = \sqrt{6}/\pi \approx 0.780$,
so $A$ against $C$ has probability $1/(1 + e^{-1.908/0.780}) \approx 0.920$,
almost the same as the probit. Near the middle of the curve the two links are
nearly indistinguishable, and a test like this one cannot tell them apart.
:::
:::

::: {.exercise #exr-cmp-softmax}
Extend the derivation of the logistic link to $k$ options. Let
$U_i = g_i + \varepsilon_i$ with independent standard Gumbel noise of scale
$\beta$. Show that the probability that option $1$ has the largest perceived
utility is $e^{g_1/\beta} / \sum_{i=1}^k e^{g_i/\beta}$, Luce's rule
@eq-cmp-luce.

::: {.solution}
Condition on $\varepsilon_1 = a$. Option 1 wins when $\varepsilon_i < g_1 - g_i + a$
for every $i \neq 1$, which by independence has probability
$\prod_{i \neq 1} F(a + g_1 - g_i) = \exp\!\big(-e^{-a/\beta} \sum_{i \neq 1} e^{-(g_1 - g_i)/\beta}\big)$.
Substitute $t = e^{-a/\beta}$ as before, so that $f(a)\,\dd a = -e^{-t}\,\dd t$,
and write $c = \sum_{i \neq 1} e^{(g_i - g_1)/\beta}$. The probability is
$\int_0^\infty e^{-t} e^{-ct}\,\dd t = 1/(1 + c)$. Multiplying numerator and
denominator by $e^{g_1/\beta}$ gives $e^{g_1/\beta} / \sum_i e^{g_i/\beta}$.
:::
:::

::: {.exercise #exr-cmp-info}
Using @eq-cmp-info, show that the information carried by one comparison tends
to zero (a) as $v \to 0$ for any fixed $m$ and $\sigma > 0$, and (b) as
$|m| \to \infty$ for fixed $v$ and $\sigma$. Then show that it tends to one bit
when $m = 0$ and $\sigma \to 0$ with $v > 0$ fixed.

::: {.solution}
(a) As $v \to 0$ the belief concentrates at $m$, so the expectation in the
second term tends to $h(\Phi(m/s))$, and the first term tends to the same value,
since $\sqrt{v^2 + s^2} \to s$. The difference tends to zero. (b) As $|m|$ grows,
$\Phi(m/\sqrt{v^2 + s^2})$ tends to 0 or 1, so the first term tends to zero;
the second term is nonnegative and no larger than the first, because
information is never negative, so it tends to zero too. (c) With $m = 0$ the
first term is $h(\Phi(0)) = h(1/2) = 1$ bit for every $\sigma$. As $\sigma \to 0$,
$\Phi(\Delta/s)$ tends to 0 or 1 for every $\Delta \neq 0$, so the second term
tends to zero, and the information tends to one bit: a noise-free answer about
a difference whose sign is a fair coin is worth exactly one bit.
:::
:::

## Further reading {#further-reading .unnumbered}

- @thurstone1927law is short and readable; it defines the discriminal process,
  states the law, and lists the five cases.
- @bradley1952rank introduce the paired-comparison model now named after them;
  @luce1959individual develops the choice axiom and its consequences;
  @plackett1975analysis extends it to rankings.
- @mcfadden1974conditional derives the conditional logit from random utility
  with Gumbel noise; @yellott1977relationship connects Luce, Thurstone, and the
  double exponential distribution.
- @miller1956magical is the classic account of how little information an
  absolute judgment carries, and why relative judgments help.
- @fechner1860elemente and @stevens1957psychophysical are the two sides of the
  debate over measuring sensation by discrimination or by direct numbers.
- @shah2016estimation and @shah2014it compare ordinal and cardinal
  measurement, in theory and in crowdsourcing experiments.
- @houlsby2011bayesian derive the information criterion of
  @sec-comparison-information and apply it to preference learning.
