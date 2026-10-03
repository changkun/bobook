---
status: done
synopsis: "What biology, physics, network science, the mathematics of preference, information theory, game theory, and differential privacy tell someone building a PBO system: comparison graphs whose spectrum governs estimation error, a decomposition that measures how much of the data one utility can explain, models of incomplete preference, what the one-bit bound limits, and what privacy would cost."
---

# Natural and Formal Sciences {#sec-natural-sciences}

The previous chapter asked where preferences come from and what happens when
a system measures them. This chapter turns to disciplines that model choice
with mathematics. What they offer preferential Bayesian optimization (PBO) is
an account of what a set of comparisons can and cannot tell a model. The most
useful results are tools a PBO system can apply to its own logs today: the
shape of the comparison graph decides how well the utilities are pinned down
(@sec-nat-graph-spectra), and a decomposition of the comparison data measures
how much of it any single utility can explain (@sec-nat-hodge). Around them,
the chapter checks formal claims that circulate in writing about preference
learning, several of them wrong in their sources or conditions
(@sec-nat-claims), and ends with what privacy would cost a preference session.

## Biology and behavioral ecology {#sec-nat-biology}

Behavioral ecology studies animal choice as an adaptation, which makes it a
natural test of whether preferences are stable, scalar, and transitive. Its
answer is mostly a transitive core with context and state effects around it.
Of 15 honeybees given binary choices between artificial flowers, 3 violated
*weak stochastic transitivity* (if $a$ beats $b$ and $b$ beats $c$ at least
half the time, $a$ beats $c$ at least half the time), under an assumption
about how the flowers ranked on a utility scale [@shafir1994intransitivity].
Slime molds, often offered as an example of intransitive choice, had a linear,
transitive ranking of food options in the original study
[@latty2011irrational]; they violated only the *independence of irrelevant
alternatives*, the principle that adding a third option should not change the
relative preference between two others. Decoy effects (@sec-jdm) depend on
state and design. In starlings they appear or disappear with the animal's
state [@schuckpaim2004state]. In bumblebees, decoys differing in reward rate
shifted preferences as predicted, while decoys differing in sugar
concentration did not [@hemingway2024economic], and a 2025 preprint, later
published in *Ecological Entomology* in 2026, found that adding rewardless
flowers, a different decoy design, did not raise preference for neighboring
flowers, and very little support for decoy effects in bees in earlier
research [@armand2025no].

Two further results bear on a session's dynamics. @harhen2023overharvesting
showed that "overharvesting", staying in a food patch longer than Charnov's
*marginal value theorem* prescribes [@charnov1976optimal], can follow from
rational inference about the environment combined with discounting adjusted
for uncertainty, and that human participants behaved in line with their model.
And evolutionary theory predicts that the utility scale itself adapts, rising
most steeply where choices are frequent and mistakes are costly
[@netzer2009evolution].

**What it means for PBO.** A user who "stays too long" near the incumbent may
be acting rationally under uncertainty, so a stopping rule should compare the
marginal expected gain with the posterior distribution of the attainable
level, not with a point estimate (inference; @sec-rec-stopping). If utility
scales adapt within a session, PBO's concentration of queries near the optimum
would sharpen discrimination there, against the assumption of a fixed floor
set by Weber's law, under which the smallest noticeable difference grows with
the magnitude (@sec-psychophysics-attention) (inference); @sec-nat-information
describes how to test it. And because the core is transitive, a model should add
covariates for the choice set, recent history, and state to the likelihood
before switching to a surrogate that can represent intransitive preferences
(inference).

## Physics and the statistical mechanics of decisions {#sec-nat-physics}

**The claim and the check.** In Luce's *choice axiom*, the probability of
choosing option $x$ from a set $S$ is $w(x) / \sum_{y \in S} w(y)$ for
positive weights $w$ [@luce1959individual]. Writing $w(x) = e^{\beta f(x)}$
makes this the *Boltzmann distribution* of statistical physics, with the
negative utility $-f$ as the energy and the *inverse temperature* $\beta$ as
the precision of choice: at high temperature choices are nearly random, at low
temperature nearly deterministic. The identity is correct. The claim built on
it, that the PBO likelihood is a softmax and so a Boltzmann distribution,
holds only with a logistic link, such as that of @gonzalez2017preferentialb.
It does not hold for the probit likelihood of Chu and Ghahramani or for
BoTorch's default `PairwiseProbitLikelihood` [@botorch2026likelihood], which
use the Gaussian cumulative distribution function (@sec-obs-baseline).
Likewise, Luce's axiom and Thurstone's Case V are not equivalent under a
logistic distribution, contrary to a claim often made citing
@yellott1977relationship: Luce's form follows when the errors of a random utility model are Gumbel
(double-exponential), so that their differences are logistic, while Case V
assumes normal errors (@sec-random-utility).

**The precision is not fixed.** @lindigleon2022bayes found in a
two-alternative forced-choice task that participants relied on several
stimulus features when they had time and decided mainly on a single feature
under high time pressure. Rational inattention, the economic theory in which
attention is a costly resource (@sec-econ-inattention), arrives at a
generalization: @matejka2015rational proved that optimal acquisition of
information yields a generalized multinomial logit model in which choice
probabilities depend on both the options' true payoffs and the decision
maker's prior beliefs.

**What it means for PBO.** Rational inattention gives choice probabilities
proportional to a prior weight times $e^{\beta \cdot \text{utility}}$. For a
pair of options this reads

$$
\Prob(\vx \text{ chosen over } \vx') =
\frac{w(\vx)\, e^{\beta f(\vx)}}{w(\vx)\, e^{\beta f(\vx)} + w(\vx')\, e^{\beta f(\vx')}}
= \operatorname{sigmoid}\!\Big(\beta\big(f(\vx) - f(\vx')\big) + \log\frac{w(\vx)}{w(\vx')}\Big),
$$ {#eq-nat-prior-logit}

and the standard logistic link is the special case of equal weights. A PBO
likelihood could therefore add a default or familiarity term $\log w$ for the
incumbent or for the option shown first (inference; the abstract of Matějka
and McKay confirms only that the choice probabilities depend on the prior, and
the specific form above is a summary). Because under time pressure the
effective utility may collapse onto a single feature, changing an interface's
response deadline changes the structure of the revealed utility, not only the
noise (inference). And the temperature describes the user, so it should be
estimated, not annealed over the iterations as is sometimes proposed;
annealing belongs to the acquisition function, for example the temperature of
Thompson sampling (@sec-thompson) (inference).

## Network science: the comparison graph {#sec-nat-networks}

Network science's most useful contribution to PBO since 2017 is the theory of
the *comparison graph*, whose nodes are the compared options and whose edges
are the answered comparisons. @sec-obs-graphs showed that the likelihood
Hessian of the Laplace approximation is this graph's Laplacian matrix; this
section asks how the graph's shape governs how well the utilities are
estimated, and how much of the data one utility can explain.

### Comparison graphs and estimation error {#sec-nat-graph-spectra}

*Spectral graph theory* studies a graph through the eigenvalues of matrices
built from it, most often the Laplacian.

::: {.definition #def-nat-graph title="Graph Laplacian, algebraic connectivity, effective resistance"}
For a comparison graph on $n$ options, the *graph Laplacian* $\mL$ is the sum
of $(\mathbf{e}_i - \mathbf{e}_j)(\mathbf{e}_i - \mathbf{e}_j)^\T$ over the
answered pairs $(i, j)$, where $\mathbf{e}_i$ is the $i$-th unit vector. Its
eigenvalues are $0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_n$, and
the number of zero eigenvalues equals the number of connected components.

- The *algebraic connectivity* is $\lambda_2$. It is positive exactly when the
  graph is connected, and the larger it is, the harder the graph is to cut
  into two weakly linked halves [@fiedler1973algebraic].
- The *effective resistance* $R_{ij}$ between options $i$ and $j$ is the
  electrical resistance between them when every comparison is a resistor of
  one ohm:
  $R_{ij} = (\mathbf{e}_i - \mathbf{e}_j)^\T \mL^{+} (\mathbf{e}_i - \mathbf{e}_j)$,
  where $\mL^{+}$ is the pseudo-inverse of $\mL$.
:::

The effective resistance has a direct statistical reading.

::: {.derivation title="Why effective resistance is the variance of a utility difference"}
1. Suppose each answered pair $(i, j)$ yields a noisy measurement of the
   utility difference $s_i - s_j$ with unit precision: the Gaussian picture
   behind the Laplace approximation of @sec-pref-laplace, with unit curvature
   per comparison and no prior. The precision (inverse covariance) of the
   least-squares estimate is then the sum of all the pairs' contributions,
   which is $\mL$ by @def-nat-graph.
2. $\mL$ is singular, because adding a constant to every utility changes no
   difference, but differences within a connected component have covariance
   given by $\mL^{+}$. So the variance of the estimated difference between
   $i$ and $j$ is $R_{ij}$, and it is infinite if no path of comparisons links
   them.
3. The rules for resistors become rules for query design. Comparisons in
   series add: the ends of a chain of $k$ comparisons have $R = k$.
   Comparisons in parallel combine like parallel resistors: asking a pair
   twice halves its resistance.
:::

Three results make this picture quantitative. Shah et al. proved minimax
bounds for estimating the utilities under the Bradley-Terry and Thurstone
models, showing that the error depends on the topology of the comparison
graph through its Laplacian spectrum, and that the error rates for ordinal
data (comparisons) and cardinal data (numerical ratings) are identical up to
constant factors [@shah2016estimation]. @hendrickx2019graph proved that, when
every compared pair is asked many times, the relative error of the estimates
scales with the square root of the effective resistance, with a matching lower
bound up to logarithmic factors. And @heckel2019active proved that, for
actively ranking options from noisy comparisons, a simple counting algorithm
that assumes no model is optimal up to logarithmic factors: parametric
assumptions such as Bradley-Terry or Thurstone buy at most a logarithmic
improvement.

@fig-nat-anchoring makes the derivation concrete. It arranges twelve compared
designs on a circle and lets you choose the order in which a session adds
comparisons.

```{figure}
//| figure: nat-anchoring
//| label: fig-nat-anchoring
//| fig-cap: "How comparisons tie utilities together. Left: twelve compared designs, with the incumbent at the top; lines are answered comparisons, and the most recent one is orange. Each design is shaded by the standard deviation of its utility difference to the incumbent: with no prior this is the square root of the effective resistance, and a dashed circle means no chain of comparisons links the design to the incumbent at all. Right: the algebraic connectivity λ₂ of the comparison graph as comparisons are added, for all four query designs. *Disjoint pairs* mimics queries that never share an input, as EUBO tends to choose. With a Gaussian process prior, the designs are random points in d dimensions with an RBF kernel of lengthscale 0.5, close to the mode of BoTorch's default prior. Each comparison has unit curvature; the values are illustrative."
```

Things to try:

- With *Disjoint pairs* at 11 comparisons, the graph has 6 components and
  $\lambda_2 = 0$: ten designs are not tied to the incumbent at all. Drag to 24
  comparisons. The same pairs are asked again, each pair becomes tighter, and
  $\lambda_2$ stays at zero.
- Switch to *Incumbent vs challenger*. The graph connects at the 11th
  comparison, $\lambda_2$ jumps to 1, and every design's difference to the
  incumbent has standard deviation 1.00; at 22 comparisons $\lambda_2 = 2$
  and the standard deviation falls to 0.71.
- Switch to *Chain*. It also connects at 11 comparisons, but $\lambda_2$ is
  only 0.07 and the design at the far end has standard deviation 3.32, the
  square root of 11 resistors in series. Connected is not the same as well
  tied.
- Set the prior to *GP, d = 1* with *Disjoint pairs*. The kernel ties nearby
  designs together and the worst standard deviation is 0.68. Now raise the
  dimension: with $d = 10$ it is 1.09, and with $d = 30$ it is 1.13. In high
  dimension, random designs sit far apart relative to the lengthscale, the
  kernel ties almost nothing, and the shape of the comparison graph again
  decides what is known (compare @sec-hd-diagnosis).

**What it means for PBO.** The following are inferences. A PBO acquisition
function is, in effect, designing a comparison graph on the evaluated points,
and the kernel's correlations act as extra edges. The algebraic connectivity,
or the effective resistance, among the evaluated points is a practical
diagnostic: regions joined to the rest by few comparisons have poorly
anchored utilities, and if the kernel is misspecified, the posterior variance
will not show it. Heckel et al.'s result implies that a surrogate gains mainly
from the kernel's smoothness and little from the choice of link
(@sec-obs-baseline).

### Cycles and the Hodge decomposition {#sec-nat-hodge}

A scalar utility can only produce comparisons that are consistent around
every loop: if $A$ beats $B$ by a margin and $B$ beats $C$ by another, the
utility fixes the margin of $A$ over $C$ as their sum. Real comparison data
are rarely that consistent, and the *Hodge decomposition* of comparison data
[@jiang2011statistical], a foundational result from before the period this
part covers, measures by how much. It treats the comparison margins (for
example the log-odds that $i$ beats $j$) as a flow on the edges of the
comparison graph and splits the flow into three orthogonal parts:

- a *gradient* part, the differences $s_i - s_j$ of one score per option,
  found by least squares (this estimate is called HodgeRank); it is the part
  that one scalar utility explains;
- a *curl* part, made of local cycles around triangles of options;
- a *harmonic* part, made of global cycles around larger loops in the graph
  that triangles do not fill in.

Because the parts are orthogonal, the squared size of each part divided by
the squared size of the whole flow is the share of the data it accounts for.
Since 2017, @strang2022network used the decomposition to quantify cyclic
competition in tournaments, and the skew-symmetric "generalized preference
kernel" of @chau2022inconsistent lets a Gaussian process represent cycles
(@sec-obs-extensions).

@fig-nat-hodge applies the decomposition to five options with a
rock-paper-scissors component of adjustable strength added among three of
them, and shows the catch in using it on real data.

```{figure}
//| figure: nat-hodge
//| label: fig-nat-hodge
//| fig-cap: "How much of a set of comparisons one utility can explain. Five options have utilities 1, 0.5, 0, −0.5, −1, plus a rock-paper-scissors flow of strength κ among A, B, and C. Every pair is answered n times under a Bradley-Terry model, and each edge carries the empirical log-odds. The bar splits the flow into its gradient part (blue, explained by one utility) and its cyclic part (magenta); the dashed mark shows the cyclic share that a purely transitive person would produce by chance with the same n, averaged over 200 simulated sessions. Below, the HodgeRank scores against the transitive part of the truth. The setup is illustrative."
```

Things to try:

- The figure opens with exact win probabilities. At κ = 0.6 the cyclic share
  is 8%, the magenta residual sits only on the triangle A, B, C, and the
  majority preferences are still transitive; the cycle in the majorities (A
  beats B, B beats C, C beats A) appears only above κ = 1. The HodgeRank
  scores recover the transitive part exactly, because a pure cycle is
  orthogonal to every gradient.
- Choose 100 answers per pair. At κ = 0.6 the first draw shows a cyclic share
  of 11% against a chance level of 2%: the cycle is detectable.
- Choose 5 answers per pair. The chance level rises to 27%, and the first draw
  shows 34%, too close to tell apart; press *Draw new answers* a few times to
  see how much the share varies. With 1 answer per pair, the typical PBO
  situation, a perfectly transitive person already shows a cyclic share of
  47% on average.

**What it means for PBO.** The Hodge decomposition is a cheap intransitivity
diagnostic for PBO logs: the share of the curl and harmonic parts in the
comparison flow measures the fraction of the data that a scalar Gaussian
process utility cannot explain, and only a large share justifies a
skew-symmetric surrogate (inference). As the figure shows, the share must be
compared with what sampling noise alone produces, so a session that wants to
test for intransitivity needs repeated pairs or nearby designs pooled into
nodes (inference).

## The mathematics of preference {#sec-nat-math}

Two questions from the mathematics of order matter for PBO: when a
preference can be summarized by a utility function, and what to do when it is
not complete.

### Representation theorems, checked {#sec-nat-representation}

A *utility representation* of a preference is a function $u$ with $x$ weakly
preferred to $y$ exactly when $u(x) \ge u(y)$. Popular statements of when one
exists are often too strong. Debreu's theorem [@debreu1964continuity] is commonly stated for any
topological space: a continuous representation exists if and only if the
preference is complete (any two options can be compared), transitive, and
continuous (the sets of options better and worse than any given option are
closed). In fact, as the survey of @hervesbeloso2019continuous explains,
separability (a countable dense subset) is necessary, a connected and
separable space, or a second-countable one, suffices, and every
non-separable metric space carries a continuous preference order with no
utility representation at all. The *lexicographic order* on $\R^2$, which
ranks by the first coordinate and uses the second only to break ties, is said
to lack a *continuous* representation; it cannot be represented by *any*
real-valued function, continuous or not [@banerjee2018wold].

**What it means for PBO.** PBO evaluates only finitely many points, and every
complete and transitive relation on a countable set has a utility
representation, so these failures concern extending a utility to the
continuum. A stationary, smooth Gaussian process cannot represent the
lexicographic order, but on any finite set of points it can approximate it
with a very short lengthscale in the secondary dimension, with input warping,
or with threshold features: the practical issue is kernel design, not whether
a utility exists (inference; @sec-kernels).

### Incomplete preferences and contextuality {#sec-nat-incomplete}

Some pairs may simply not be comparable for a person. A *multi-utility
representation* handles this with a set of utility functions, declaring $x$
better than $y$ only when every function in the set agrees
[@evren2011multi, a foundational paper]. In choice data, a person who picks
each of two options about half the time may be *indifferent* (the options are
equally good), *indecisive* (they cannot rank them), or willing to
*experiment*. @ok2022indifference gave methods to identify which, and showed
that each identification yields a way to make deterministic welfare
comparisons from random choice. The distinction matters in practice. In the
experiments of @cettolin2019revealed, about half of the participants chose in
a way inconsistent with complete preferences plus certainty independence (an
axiom that says mixing two options with the same sure outcome should not
change which is preferred). Of those participants, about half behaved
consistently with incomplete preferences and about a third with a preference
for randomization, and further experiments showed that probability weighting,
errors, regret aversion, or intransitive indifference could not explain the
pattern.

A stronger claim is that preferences are *contextual* in the sense of quantum
physics: no single set of underlying values explains the measurements, even
after allowing each measurement to depend directly on its context. Reviewing
behavioral data, @dzhafarov2016there found no contextuality once direct
context effects were separated out; @cervantes2018snow then gave the first
clear demonstration of contextuality in human choice, and @basieva2019true
showed that contextual systems can be found in tasks designed for the
purpose, while earlier claims in judgments are explained by direct
influences.

**What it means for PBO.** Incomplete preferences represented by a set of
utilities correspond in PBO to a vector-valued Gaussian process with Pareto
dominance (@sec-multi-objective), close to Bayesian optimization with
preference exploration, in which a person compares outcome vectors
[@lin2022preferenceb] (inference). A 50:50 answer is ambiguous between
indifference, indecisiveness, and experimentation; an interface that offers
"cannot decide" as a separate answer from "about the same", modeled separately
from a tie, can tell them apart (inference; @sec-ties). Because most context
effects are direct influences, a likelihood with covariates for order, choice
set, and history can absorb them within one global utility; true
contextuality appears only in specially designed tasks (inference).

## Information theory {#sec-nat-information}

A common argument says that a pairwise comparison carries at most one bit, far
less than a rating could, which makes comparisons inefficient. The bound is
correct: the mutual information between a binary answer and the utility
(@sec-mutual-information) cannot exceed the entropy of the answer, at most 1
bit. It is not the binding constraint. Error rates for ordinal and cardinal
estimation are identical up to constant factors [@shah2016estimation]. Per
query, ordinal measurements have lower noise per sample and are usually faster
to collect but usually carry less information, and a preprint by the same
group [@shah2014it] quantifies at which noise levels ordinal measurement is
better. The one-bit argument concerns constants; the opposite argument, that
regret bounds for PBO comparable to those of scalar BO show comparisons are not
inefficient, concerns rates.

How much a comparison carries depends on the system and the person.
@ghosal2023effect showed that the "rationality coefficient" (the noise scale)
should be fitted separately for each type of feedback, that overestimating
human rationality severely harms the accuracy and regret of reward learning,
and that when people are very suboptimal, comparisons are more informative
than demonstrations. And efficient coding of value [@polania2019efficient],
introduced in @sec-neuro-coding, allocates precision to the values a person
expects to see.

**What it means for PBO.** The noise scale should be fitted per person and per
type of feedback, not fixed at a low default (inference from Ghosal et al.).
Under a probit or logistic link, a comparison's expected information
approaches one bit only when the predicted win probability is near 0.5 and the
noise is small (@exr-nat-bits); adding ties or graded confidence raises the
ceiling to $\log_2 3$ bits for three answers but adds noise parameters
(inference). As PBO narrows its candidates toward the optimum, efficient
coding predicts that discrimination there may improve rather than hit a fixed
floor, testable by fitting the slope of the psychometric function separately
early and late in a session (inference). We found no direct measurement of how
many bits a comparison carries for human design preferences.

## Game theory and symmetry breaking {#sec-nat-games}

When two options are identical under the description the chooser uses, any
choice between them must draw on information from outside that description,
such as position, history, or salience (Schelling's *focal points*). A
standard preference likelihood predicts 0.5 at equal utility and so treats
every such choice as noise, which is misspecified if the symmetry breaking is
systematic. In the network coordination experiments of @mas2016behavioral,
96% of decisions were myopic best responses, deviations
were rarer when they cost more, and individuals differed. And choosing between
options a person rated similarly, close to the symmetric case, itself changes
preference: across 43 free-choice studies that excluded a known
methodological artifact, Cohen's $d = 0.40$ (95% confidence interval 0.32 to
0.49) [@enisman2021choice], as @sec-jdm-choice describes.

Answers can also be strategic: every reasonable voting rule over three or more
options can be manipulated by misreporting (the Gibbard-Satterthwaite
theorem), and for rules on three alternatives that are far from a
dictatorship and from having a range of only two alternatives, manipulation
succeeds with non-negligible probability [@friedgut2011quantitative]. A single
binary query is strategy-proof, but misreports across a sequence remain
possible.

**What it means for PBO.** Near indifference, real choices are pushed one way
by position, salience, and defaults. Balancing the order of presentation and
adding a position or default term to the likelihood, the prior weight $w$ of
@eq-nat-prior-logit, turns this from noise into a signal the model can
estimate (inference). Choice-induced change means that a close comparison
biases later answers toward the chosen option, a drift correlated with the
optimizer's own queries (@sec-soc-measuring); asking some early close pairs
again later, or adding a choice-history term to the likelihood, can detect it
(inference). The payoff-sensitive deviations of Mäs and Nax support probit or
logistic errors, whose rate falls as the utility difference grows, as the
main mechanism of deviation, with a small lapse component (inference). We
found no study of strategic misreporting in PBO.

## Influence among users {#sec-nat-complexity}

Complexity and network science also study how choices spread through a group,
which matters once PBO has several users, a shared gallery, or a prior learned
from earlier users. In the two "multiple worlds" experiments of
@macy2019opinion ($n = 4{,}581$), social influence made partisan divisions
larger and less predictable: in parallel worlds, the same position became
attached to opposite parties. @frey2021social found that when people choose
one after another and see the running counts, majorities are more often
wrong, and on hard tasks a wrong majority perpetuates itself instead of being
corrected. And *homophily* (similar people connect) and *contagion*
(connected people become similar) are generally confounded in observational
social network studies [@shalizi2011homophily]; the frequently heard version,
that the two cannot be told apart even in principle, holds only for
observational data.

**What it means for PBO.** For PBO with many users, gallery-style
crowdsourcing, or transfer of earlier users' preference models to a new user
(as in Meta-PO [@li2025efficient]), independent first judgments should be
collected before any aggregate is shown, and a population prior trained on
socially influenced data carries an arbitrary, path-dependent component
(inference). Because homophily and influence cannot be separated in the logs
of a shared gallery, the remedy is in the design: randomize which users see
which earlier choices (inference from Shalizi and Thomas). For single-user
PBO, the link is weak.

## Differential privacy {#sec-nat-privacy}

Preference data are sensitive: the designs a person prefers can reveal their
body, their health, or their tastes. A randomized algorithm is
$\varepsilon$-*differentially private* if changing one person's data changes
the probability of any output by at most a factor of $e^{\varepsilon}$
[@dwork2006calibrating]; smaller $\varepsilon$ means stronger privacy. In
*local* differential privacy, each answer is randomized before it leaves the
user; in *central* differential privacy, a trusted curator holds the raw data
and randomizes only what it releases. The classic local mechanism for a
binary answer is *randomized response* [@warner1965randomized]: report the
true answer with probability $e^{\varepsilon}/(1 + e^{\varepsilon})$ and the
opposite answer otherwise.

**Since 2023, private learning from pairwise preferences has tight
theoretical bounds.** For a $d$-dimensional Bradley-Terry reward estimated
from $n$ comparisons under label privacy, @chowdhury2023differentially
(AISTATS 2024) showed that the additional error is
$\Theta\big((1/(e^{\varepsilon} - 1)) \sqrt{d/n}\big)$ under local privacy and
$\Theta\big(\mathrm{poly}(d)/(\varepsilon n)\big)$ under central privacy,
tight under local privacy and tight in $n$ and $\varepsilon$ under central
privacy. Matching bounds also exist, in preprints, for dueling bandits
(@sec-dueling-bandits) [@saha2024dp] and for offline reinforcement learning
from human feedback [@wu2025offline]. On the attack side, PREMIA
[@feng2024exposing] found that models aligned with direct preference
optimization (DPO) are more vulnerable to membership inference attacks on
their preference data than models aligned with proximal policy optimization
(@sec-llm-methods). Private Bayesian optimization with scalar feedback exists
[@kusner2015differentially], but a search of arXiv abstracts for
"preferential Bayesian optimization" together with "privacy" or "private"
returned nothing.

**What it means for PBO.** The following are inferences. Randomized response
on each comparison satisfies $\varepsilon$-local privacy, and the PBO
likelihood can absorb it as label noise with a known flip probability
$q = 1/(1 + e^{\varepsilon})$. The reported answer favors $\vx$ with
probability $q + (1 - 2q)\,p$, where $p$ is the unprivatized probability, so
the slope of the answer probability with respect to the utility difference
shrinks by $1 - 2q = \tanh(\varepsilon/2)$. For a weak signal, with both
probabilities near one half, the Fisher information of one answer shrinks by
$\tanh^2(\varepsilon/2)$, and keeping the same information needs
$1/\tanh^2(\varepsilon/2)$ times as many queries: about 1.7 times at
$\varepsilon = 2$, about 4.7 times at $\varepsilon = 1$, and about 17 times at
$\varepsilon = 0.5$, consistent with the $1/(e^{\varepsilon} - 1)$ rate
above. With only tens to hundreds of queries, that is expensive. In
single-user PBO, what leaks a preference is the optimizer's output, the
candidates it shows and the final design, so privacy of the outputs fits the
threat better than privacy of the labels. Transferring preference models
between users, as in Meta-PO, exposes earlier users' comparisons to membership
inference of the kind PREMIA demonstrated, and there central privacy on the
aggregated model is more practical than local privacy. The link is strong,
but as of September 2026 we found no paper on private PBO.

## Common claims, checked {#sec-nat-claims}

@tbl-nat-claims collects the formal claims checked in this chapter that bear
on how a PBO system is built.

::: {.table #tbl-nat-claims title="Claims from the natural and formal sciences, and what the sources say."}
| Claim | What the sources say | Section |
|---|---|---|
| The PBO likelihood is a softmax, a Boltzmann distribution | only with a logistic link; Chu and Ghahramani and BoTorch's default use the probit | @sec-nat-physics |
| Luce's axiom and Thurstone's Case V are equivalent under a logistic distribution (Yellott 1977) | Luce's form follows from Gumbel errors, whose differences are logistic; Case V assumes normal errors | @sec-nat-physics |
| Slime molds choose intransitively | their ranking was linear and transitive | @sec-nat-biology |
| The lexicographic order on $\R^2$ has no continuous representation | it has no real-valued representation at all | @sec-nat-representation |
| One bit per comparison makes comparisons inefficient | the bound holds, but ordinal and cardinal error rates differ only by constants | @sec-nat-information |
| Homophily and influence are indistinguishable in principle | confounded in observational data | @sec-nat-complexity |
:::

## Settled, contested, missing {#sec-nat-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** Estimation error from comparisons depends on the comparison
graph's Laplacian spectrum, and the relative error scales with the square root
of the effective resistance, with matching lower bounds up to logarithmic
factors. Ordinal and cardinal error rates differ only by constants, although a
binary answer carries at most one bit. Comparison data
split orthogonally into a part one utility explains and cyclic parts. The PBO
likelihood is a Boltzmann distribution under the logistic link, not under the
probit. Choosing between similar options changes preferences (Cohen's
$d = 0.40$ across 43 studies). Private learning from pairwise comparisons has
tight rates (some of them still in preprints).

**Contested.** How intransitive real preferences are: animal evidence mostly
shows context or state dependence on a transitive core, and decoy effects in
bees are found in one design and not another. Whether contextuality, beyond
direct context effects, exists in ordinary choices. Whether utility scales
adapt fast enough to change within a single session.

**Missing.** Spectral bounds of comparison graphs applied to Gaussian process
PBO. A Hodge analysis of real PBO logs against its chance level. A fit of
prior-dependent logit models to PBO's pairwise data. A
measurement of the bits per comparison for human design preferences. A test
of whether discrimination sharpens late in a session. A study of strategic
misreporting in PBO. A private PBO method.
:::

## Exercises {#sec-nat-exercises}

::: {.exercise #exr-nat-resistance}
In the *Incumbent vs challenger* design of @fig-nat-anchoring, every
challenger is compared once with the incumbent. Use the rules for resistors to
find the standard deviation of the utility difference between two challengers,
and between a challenger and the incumbent. What happens to both when every
comparison is asked twice?

::: {.solution}
A challenger and the incumbent are joined by one one-ohm resistor: resistance
1, standard deviation 1. Two challengers are joined through the incumbent by
two resistors in series: resistance 2, standard deviation
$\sqrt{2} \approx 1.41$. Asking every comparison twice halves every
resistance, giving $\sqrt{1/2} \approx 0.71$ (the value the figure shows at
22 comparisons) and 1.
:::
:::

::: {.exercise #exr-nat-bits}
Under a probit model, the person prefers $\vx$ with probability
$p = \Phi(d/\sigma)$, where $d$ is the utility difference and $\sigma$ the
noise. The model is uncertain about $d$: suppose $d = +\delta$ or $-\delta$
with equal probability. Show that the mutual information between the answer
and $d$ is $1 - H_2(\Phi(\delta/\sigma))$ bits, where $H_2$ is the binary
entropy. When does it approach one bit, and what is it when $\delta = \sigma$?

::: {.solution}
The mutual information is the entropy of the answer minus its expected
conditional entropy, $I = H(\text{answer}) - \E_d[H(\text{answer} \mid d)]$
(@sec-mutual-information). By symmetry the answer has entropy 1 bit overall,
and given either sign of $d$ it has entropy $H_2(\Phi(\delta/\sigma))$, so
$I = 1 - H_2(\Phi(\delta/\sigma))$. It approaches one bit when $\delta/\sigma$
is large: the model is unsure of the sign but the person answers almost
deterministically. At $\delta = \sigma$, $\Phi(1) \approx 0.841$ and
$H_2(0.841) \approx 0.63$, so the answer carries about 0.37 bits.
:::
:::

::: {.exercise #exr-nat-privacy}
A study plans 60 comparisons per participant and wants each answer protected
by randomized response with $\varepsilon = 1$. Using the calculation in
@sec-nat-privacy, roughly how many comparisons would carry the same
information as the 60 unprotected answers? What does this suggest about
where privacy should be applied in a single-user session?

::: {.solution}
The information per weak-signal answer shrinks by
$\tanh^2(1/2) \approx 0.462^2 \approx 0.21$, so matching 60 unprotected
answers needs about $60 / 0.21 \approx 280$ protected ones, more than four
times the budget. Because a single user is both the data subject and the
person answering, the more economical protection is on what the system
releases (the candidates shown to others, the stored model, the final
design), not on each answer.
:::
:::

## Further reading {#further-reading .unnumbered}

- @shah2016estimation and @hendrickx2019graph are the clearest statements of
  how the comparison graph's spectrum and resistance govern estimation error.
- @jiang2011statistical introduces the Hodge decomposition of comparison data;
  @strang2022network is a readable survey of its use on tournaments.
- @heckel2019active shows how little a parametric link buys in active ranking.
- @ok2022indifference separates indifference, indecision, and experimentation
  in random choice, the distinction behind a "cannot decide" answer.
- @hervesbeloso2019continuous surveys when continuous utility representations
  exist, with the conditions that popular summaries leave out.
- @chowdhury2023differentially (AISTATS 2024) sets out the cost of local and
  central privacy for learning from comparisons.
