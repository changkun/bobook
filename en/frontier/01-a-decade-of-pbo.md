---
status: done
synopsis: "From the 2005 baselines to September 2026: how the field got its name, how its tools and inference settled, the decision-theoretic turn, and the years in which theory caught up and the default pipeline came under scrutiny. An interactive timeline places every milestone in its lane and phase."
---

# A Decade of Preferential Bayesian Optimization {#sec-pbo-history}

@sec-part-preferences taught preferential Bayesian optimization the way it is
usually presented: a Gaussian process utility learned from comparisons
(@sec-gp-preference), a rule for choosing the next pair (@sec-pbo), the forms a
query can take (@sec-query-forms), and the bandit view of learning from duels
(@sec-dueling-bandits). That presentation is a snapshot. Each of its pieces
arrived at a particular time, from a particular community, in answer to a
particular question, and several of them have since been questioned.

This part of the book reports what research since 2017 has established,
contested, or left open. This first chapter is its map. It tells the story in
order, from the models that existed before the field had a name to the
preprints of 2026, so that the chapters after it can each follow one thread in
depth: the observation model (@sec-observation-models), the acquisition
function (@sec-acquisition-frontier), the theory (@sec-pbo-theory), high
dimensions (@sec-high-dimensions), and software and evaluation
(@sec-software-evaluation).

The story has a surprising shape. The pipeline most people run in 2026, a
Gaussian process prior with a probit link and the Laplace approximation, is a
model from 2005. What changed over the decade is less the machinery than the
question researchers asked of it.

```{figure}
//| figure: hist-timeline
//| label: fig-hist-timeline
//| fig-cap: "Preferential Bayesian optimization from 2005 to September 2026, in six lanes. Filled circles are peer-reviewed papers (and one workshop), hollow circles are preprints, squares are software releases. The shaded band is the phase of the selected event. The years 2005 to 2016 are compressed, and within a year the positions are spread for legibility, not dated. The central question shown for each period is this chapter's inference, not the claim of any source. Release dates come from the BoTorch and Ax changelogs and package pages [@botorch2026changelog; @ax2026changelog; @ax2022platform; @optuna2026dashboard]; every paper is cited where the text discusses it."
```

The figure shows the milestones this chapter discusses and two further BoTorch
releases (0.10.0 and 0.18), not every paper. A few things to look for:

- **Where the theory comes from.** Before 2021, every guarantee in the theory
  lane comes from the dueling-bandit community; kernelized results for
  continuous domains begin in 2021 and cluster from 2024 on.
- **When people enter.** Before 2022 the people lane holds only the two
  exoskeleton papers; afterwards it fills, with nothing in 2024.
- **What is not yet peer reviewed.** The hollow circles are all in 2026: the
  newest critiques of the default pipeline are preprints.
- **Replay the decade.** Turn on *Hide later years*, select the 2005 event, and
  press *Next* repeatedly to watch the field accumulate.

## Before 2017 {#sec-history-before}

Three foundations existed before the field had a name, each in a different
community.

The first is a model. @chu2005preference placed a Gaussian process prior on a
latent utility, a function that scores how much a person likes each option, and
linked it to comparisons through a **probit** likelihood: the probability that
$\vx$ is preferred to $\vx'$ is the standard normal distribution function
$\Phi$ applied to the scaled utility difference (@sec-thurstone). Because that
likelihood is not Gaussian, the posterior has no closed form, and they replaced
it with the **Laplace approximation**, a Gaussian centered at the posterior's
peak (@sec-laplace). This is the model of @sec-gp-preference, and it is still
the default two decades later.

The second is an interactive system. @brochu2007active used active preference
learning, in which the system itself decides which options to show next, to
help people design materials for computer graphics by choosing from a gallery
of candidates, with the loop of @sec-pbo-loop: show options, record a choice,
update the model, choose what to show next.

The third is a problem formulation from information retrieval.
@yue2009interactively cast the interactive optimization of a retrieval system,
such as a search engine learning from its users, as a **dueling bandit**
problem: a learner repeatedly picks two options and observes only which one
wins (@sec-dueling-bandits-intro). The framing brought the analytic tools of
online learning, regret bounds above all, to learning from comparisons.

The three lived apart: Gaussian process preference learning in machine
learning, galleries in graphics and interaction design, dueling bandits in
online learning. Much of the decade that followed can be read as these
communities meeting, slowly and incompletely.

## 2017 to 2019: how to ask {#sec-history-2017}

**The name.** @gonzalez2017preferentialb defined the problem on the *dueling
space*, the set of all pairs of inputs, gave it the name *preferential Bayesian
optimization*, and proposed three acquisition functions: pure exploration,
Copeland expected improvement, and dueling Thompson sampling
(@sec-dueling-formulation). It came without convergence theory, that is,
without a statement of how fast the recommended option approaches the best one
as queries accumulate.

**Guarantees on the bandit side.** The dueling-bandit community already had
formal results for closely related problems. The quantity those results control
is **regret**: the utility lost, summed over all queries, by showing the options
the algorithm chose instead of the best one (@sec-regret). SelfSparring, a
method for duels among several options at once, was proved to converge
asymptotically [@sui2017multi]; @kumagai2017regret gave a regret bound for
dueling bandits over a continuous space, in the same year the name appeared;
and StageOpt handled unknown safety constraints, with theorems stated for
numerical observations and a preference variant, without a convergence theorem
of its own, applied to spinal cord stimulation [@sui2018stagewise]
(@sec-acqf-extensions).

::: {.note title="A common claim, checked"}
One sometimes reads that learning from comparisons had no formal guarantees
before the kernelized regret bounds of 2024. The claim is right about one thing:
the Gaussian process formulation of González et al. came without a rate, which
a 2018 survey of dueling bandits pointed out, calling it "a pure Bayesian
optimization approach without theoretical guarantees on convergence rate"
[@sui2018advancements]. It is wrong as a statement about the field.
SelfSparring's asymptotic convergence [@sui2017multi] and Kumagai's
continuous-domain regret bound [@kumagai2017regret] both predate 2019, as does a
clinical application of preference feedback, StageOpt's [@sui2018stagewise].
The accurate statement is narrower: guarantees existed for bandit formulations
of the problem, while the Gaussian process pipeline that practitioners run had
none, and as @sec-pbo-theory reports, the analyzed algorithms and the practiced
pipeline are still not the same thing.
:::

**Changing the question's shape.** In graphics, @koyama2017sequentialb took a
different route. Their *sequential line search* turns every query into a single
slider: a crowd worker drags along a line through the design space and stops at
the point they like best (@sec-line-search). This paper began a line of
human-computer interaction research in which the form of the query, rather
than the rule for choosing it, is the main variable.

**The question of the period.** The central question of these years was how to
ask a person. Machine learning improved the rules for choosing comparisons,
while human-computer interaction mainly changed the shape of the query and
rarely the acquisition function (inference). The figure shows no milestone in
2019; work continued, for example on letting people answer "about the same"
[@byk2019asking] (@sec-obs-extensions).

## 2020 to 2021: tools and inference settle {#sec-history-2020}

**A default implementation.** In April 2020, version 0.2.3 of BoTorch, Meta's
open-source Bayesian optimization library built on PyTorch (@sec-bo-software),
added `PairwiseGP`, a model for pairwise comparison data
[@botorch2026changelog]. It implements the probit link with the Laplace
approximation, and from then on that combination has been the most widely used
implementation of preferential Bayesian optimization: from 2020 the field had a
de facto standard, and that standard was the 2005 model (@sec-obs-pairwisegp).

**Queries in a subspace.** The same year brought methods that handle more
dimensions by restricting each query to a low-dimensional subspace: in
*projective* preferential Bayesian optimization the person chooses the best
point along a line through the space [@mikkola2020projective], and the
*Sequential Gallery* shows a two-dimensional plane of designs as a grid
[@koyama2020sequential] (@sec-gallery-projective). In robotics, CoSpar learned
exoskeleton walking gaits from a wearer's preferences [@tucker2020preference],
and LineCoSpar extended it to six gait parameters, tested with six able-bodied
participants [@tucker2020human]; these two papers began the line of exoskeleton
applications that @sec-health-exo follows and @sec-cs-exoskeleton works through.

**The exact posterior.** In 2021, @benavoli2021preferentialb proved that the
exact posterior over the utility under a probit preference likelihood is a
**skew Gaussian process**: a distribution over functions like a Gaussian
process, except that its marginals are skewed rather than symmetric
(@sec-skew-gp). A Gaussian approximation such as Laplace's cannot represent
that skew, and from this point the quality of posterior inference became a
point of dispute (@sec-obs-inference).

**The first kernelized bound, and an alternative.** Also in 2021,
@kirschner2021bias gave the first cumulative regret bound for *kernelized*
dueling feedback, in which the unknown utility is assumed smooth in the sense
defined by a kernel, so the result covers continuous domains
(@sec-kernelized-dueling); their feedback model is the utility difference plus
noise, not the probit or logistic link of the preference models. Control
engineering attacked the same problem independently: GLISp fits a radial basis
function surrogate, a weighted sum of bumps, to the observed preferences, with
no probability model at all [@bemporad2021global] (@sec-obs-surrogates).

By the end of 2021, then, the field had a standard implementation, a known
weakness in that implementation's inference, and the beginning of a theory for
continuous domains. The acquisition functions in use were still largely
heuristics.

## 2022 to 2023: the decision-theoretic turn {#sec-history-2022}

**EUBO.** The heuristics gave way to a principled rule. In *Bayesian
optimization with preference exploration* (BOPE), a person's preferences over
the outcomes of an experiment are learned from comparisons while the experiments
run. For that setting, @lin2022preferenceb proposed the expected utility of the
best option, EUBO (@sec-eubo), and proved it **one-step Bayes optimal**: if the
session ended after one more answer, no other query would yield a better
expected final recommendation (@sec-eubo-theory). @astudillo2023qeubob
generalized it to qEUBO, for queries that show several options at once and for
answers with logistic noise, and proved that an adapted batch version of
expected improvement is not asymptotically consistent. This was the
decision-theoretic turn, and qEUBO became the basis of BoTorch's preference
acquisition functions; BoTorch, Ax, and optuna-dashboard shipped the new rules
within months (@sec-sw-software).

**Inference, measured.** @takeno2023practicalc measured how far the Laplace
approximation and expectation propagation are from the exact skew posterior,
and proposed the *hallucination believer*: draw one sample of the latent
utilities from the posterior, treat it as measured data, and apply any standard
acquisition function (@sec-choosing-pairs). The method became the basis of the
preferential sampler in optuna-dashboard, a web interface for the Optuna
optimization library [@optuna2026dashboard].

**People become the question.** At the same time, human-computer interaction
began to measure what happens to the person in the loop. Novice designers
whose design process was led by a multi-objective optimizer reported lower
agency and ownership than those who led it themselves, in a study that used
performance objectives, not comparisons [@chan2022investigating]
(@sec-hci-agency); in a three-month field deployment,
most rating loops never reached the optimization stage or did not converge
[@ou2022human] (@sec-sw-simulated-users); and experts iterated more than
novices and ended less satisfied [@ou2023impact] (@sec-hci-expertise).

**A neighbor grows large.** In 2023, direct preference optimization (DPO) put
the **Bradley-Terry** likelihood, under which the probability that one option
beats another is the logistic function of their utility difference
(@sec-bradley-terry), at the center of aligning large language models with
human preferences, fitting the language model to human comparisons directly
rather than through a reward model as reinforcement learning from human
feedback (RLHF) does [@rafailov2023direct]. The same year, the ICML workshop
*The Many Facets of Preference-Based Learning* brought dueling bandits, RLHF,
social choice, and optimization together [@icml2023many]. @sec-llms follows the
traffic between the two fields.

## 2024 to 2026: noise, theory, and scrutiny {#sec-history-2024}

The last three years divide into two stages. In 2024 and 2025, theory,
high-dimensional diagnosis, language models, and questions of legitimacy all
advanced at once. In 2026, the work concentrated on flaws in the default
pipeline and on simpler alternatives to it.

### 2024 and 2025: theory, dimension, language models, legitimacy {#sec-hist-2024}

**Bounds under the Bradley-Terry link.** Kernelized regret upper bounds,
guarantees that cumulative regret grows no faster than a stated rate, arrived
for the Bradley-Terry link with POP-BO [@xu2024principledb], MaxMinLCB
[@pasztor2024bandits], and MR-LPF, whose upper bound is of the same order as
for scalar feedback [@kayal2025bayesian]. Matching upper bounds do not show
that a comparison carries as much information as a number; they show only that
the guarantees are of the same order. @sec-theory-rates states each rate with
its assumptions.

**An amortized optimizer.** PABBO trains a neural network in advance, on many
synthetic tasks, to propose the next pair directly, so that a query costs one
forward pass instead of fitting a model and optimizing an acquisition function
[@zhang2025pabbob]. It is the only amortized optimizer for pairwise
preferences (@sec-obs-surrogates).

**High dimensions, re-diagnosed.** In ordinary Bayesian optimization with
numerical observations, a run of papers attributed the familiar failure in high
dimensions to the prior and the initialization rather than to the method itself
[@hvarfner2024vanilla; @xu2025standard; @papenmeier2025understanding]. The
remedy is a **dimension-scaled prior**, a prior on the kernel lengthscale whose
typical value grows with the number of inputs (@sec-lengthscale-priors). In
September 2024, BoTorch 0.12.0 switched most of its models to such priors but
explicitly excluded `PairwiseGP` [@botorch2026changelog], a gap that
@sec-hd-diagnosis examines.

**Language models and legitimacy.** Large language models entered
conversational preference elicitation [@austin2024bayesian] and the active
collection of alignment data [@dwaracherla2024efficient]; human-computer
interaction turned to population priors, which transfer what earlier users
preferred to a new user [@li2025efficient], and to collaboration in natural
language between designer and optimizer [@niwa2025cooperative]. Alignment
research also formalized a worry that preferential optimization shares: a
system that learns from preferences may change the preferences it measures.
@carroll2024aib compared eight notions of alignment for preferences that can
change and found that each either rewards the system for undue influence on
the person or is overly risk-averse, and @williams2025targeted found that
learners optimized on user feedback learn to target the users most open to
influence (@sec-econ-endogenous).

### 2026: the default pipeline under scrutiny {#sec-hist-2026}

**Theory.** Thompson sampling with preference feedback, PF-TS, has an upper
bound of order $\gamma_T \sqrt{T}$ for a fully sequential algorithm
[@lazzaro2026finiteb], against MR-LPF's $\sqrt{\gamma_T T}$ for a batched one
with a finite candidate set and a warm-up period [@kayal2025bayesian]; here
$\gamma_T$ is the **maximum information gain** of the kernel, which grows
slowly for smooth kernels (@sec-gp-information-gain).

**Flaws in the default.** Two 2026 preprints examined the pipeline of
`PairwiseGP`, the Laplace approximation, and EUBO: EUBO's queries collapse
toward the estimated best option [@wu2026knowledge], and its pairs, sharing no
input with earlier queries, make the Laplace likelihood Hessian rank deficient
[@shao2026adaptive]. @sec-pbo-failure-modes introduced both. The observation
model was extended to experiments that fail [@menn2026preferential], and local
preferential Bayesian optimization, also a preprint, took the method to about
100 dimensions [@menn2026local] (@sec-hd-local).

**Simpler alternatives and people.** In ordinary high-dimensional Bayesian
optimization, a spherical mapping of the inputs combined with Bayesian linear
regression (@sec-blr) reached the state of the art on tasks with 60 to 6000
dimensions [@doumont2026we]. Ax added preference optimization and trials in
which a language model turns free-text feedback into comparisons
[@ax2026changelog; @kobalczyk2026lilo] (@sec-llm-in-loop). And the human
studies of 2026 that used strong comparison conditions mostly found null
results, or found that people tuning by hand did about as well: cost-aware
Bayesian optimization for prototyping interactive devices reached the same
performance at about 67% of the cost, with no difference in final quality
[@langerak2026cost]; priors learned from user models helped only at the
second and third iterations, in a study with 12 participants and a performance
objective [@liao2026efficient]; and 11 healthy
adults who tuned their own exoskeleton assistance with a thumbstick remote
control reached a 16.6% metabolic reduction in about 10.9 minutes, comparable
to what algorithmic tuning reports [@schafer2026user]. Simpler methods,
including people tuning by hand, have thereby become controls that a new method
must be compared against (@sec-hci-baselines). The tutorial of Benavoli and
Azzimonti on learning from preferences and choices with Gaussian processes was
also formally published [@benavoli2026tutorial], though there is still no
dedicated survey of the field (@sec-sw-infrastructure).

## How the central question moved {#sec-hist-questions}

Read as a sequence of questions, the decade turned three times (inference).

From 2017 to 2021 the question was *how to ask a person, and how to infer from
the answers*. The dueling formulation, sequential line search, galleries and
projections, `PairwiseGP`, and the skew Gaussian process all answer some part of
it.

From 2022 to 2025 it was *whether the acquisition function is principled, and
whether it comes with guarantees*. EUBO and qEUBO answered the first part with
one-step Bayes optimality; POP-BO, MaxMinLCB, and MR-LPF answered the second
with regret bounds under the Bradley-Terry link.

From 2025 to 2026 it became four questions at once: *is the observation model
right, do people answer the way the model assumes, are simpler methods already
enough, and does the system change the preferences it measures?* The collapse
of EUBO, the rank-deficient Hessian, the null results under strong controls,
the linear model that wins in high dimension, and the alignment results on
influence all belong here. This last turn is the book's thesis seen from the
literature: the bottleneck moved from algorithms to measurement
(@sec-syn-bottleneck).

The periods overlap in 2025, when the second question was still being answered
and the third was already being asked, which is why @fig-hist-timeline shows
both for that year.

::: {.keyidea title="The machinery stayed; the question moved"}
The default pipeline of 2026, a Gaussian process prior with a probit link and
the Laplace approximation, is the model of 2005. What changed over the decade
is the question asked of it: first how to ask, then whether the choice of query
is principled and guaranteed, and finally whether the model of the person is
right at all.
:::

Each of the following chapters takes up one of these threads.
@sec-observation-models asks whether the model of a human answer is right;
@sec-acquisition-frontier, whether the query rules hold up; @sec-pbo-theory,
what is actually proved; @sec-high-dimensions, where the method stops working
and why; and @sec-software-evaluation, what the software does by default and
how methods are compared. @sec-part-humans asks the human questions, and
@sec-part-perspectives asks what a preference is in the first place.

## Publication and community {#sec-history-community}

The volume of work grew after 2023, but from a small base: on arXiv, entries
whose abstracts contain *preferential*, *Bayesian*, and *optimization* (or
*optimisation*) numbered 4 in 2023, 5 in 2024, 13 in 2025, and 12 in the first
nine months of 2026 [@arxiv2026preferential], a count that misses papers using
other words, such as "dueling" or "human feedback" (inference). The research is
spread across the venues of machine learning, control, robotics, and
human-computer interaction, and these communities cite one another little: the
2021 survey of dueling bandits in the *Journal of Machine Learning Research*
does not cite González et al. 2017 [@bengs2021preference]. @tbl-sw-groups maps
the main groups by discipline; the groups that prove regret bounds do not run
human studies, and the groups that run human studies rarely change the
acquisition function (inference). @sec-sw-community gives the fuller account.

## Settled, contested, missing {#sec-hist-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** The default pipeline did not change over the decade: the Gaussian
process, probit, and Laplace model of @chu2005preference, implemented in
BoTorch's `PairwiseGP` since April 2020 [@botorch2026changelog]. EUBO and qEUBO
are one-step Bayes optimal with noise-free answers
[@lin2022preferenceb; @astudillo2023qeubob]. Formal guarantees for learning from
comparisons existed on the dueling-bandit side well before 2024
[@sui2017multi; @kumagai2017regret], while kernelized bounds
under the Bradley-Terry link date from 2024 [@xu2024principledb]. BoTorch's
switch to dimension-scaled priors in 2024 explicitly excluded the preference
model [@botorch2026changelog].

**Contested.** Whether the flaws reported in 2026, EUBO's collapse toward the
current best and the rank-deficient Hessian, cost anything on human tasks; both
rest on preprints [@wu2026knowledge; @shao2026adaptive]. Whether preferential
optimization beats simpler alternatives once the comparison is strong: the
controlled human studies of 2026 found modest or no advantages
[@langerak2026cost; @liao2026efficient; @schafer2026user], and in scalar high
dimension a linear model reached the state of the art [@doumont2026we].

**Missing.** A dedicated survey of preferential Bayesian optimization.
Exchange between the communities: the 2021 JMLR survey of dueling bandits does
not cite the paper that named the field [@bengs2021preference]. A study that
randomizes people to different acquisition functions with the same interface
and budget, and a comparison with expert manual tuning on a preregistered
endpoint (@sec-open-decisive).
:::

## Further reading {#further-reading .unnumbered}

- @gonzalez2017preferentialb is the paper that named the field; read it next to
  @chu2005preference, whose model the field went on to use instead of the
  dueling formulation.
- @sui2018advancements and @bengs2021preference are two surveys of dueling
  bandits; together they show what the bandit community knew and what it did not
  cite.
- @lin2022preferenceb and @astudillo2023qeubob mark the decision-theoretic turn.
- @benavoli2026tutorial is the most complete single reference on Gaussian
  process models for preferences and choices.
- The BoTorch changelog [@botorch2026changelog] is a compact history of what
  practitioners could actually run, release by release.
