---
status: done
synopsis: "Eighteen open problems, each with an experiment that would settle it, the one experiment nobody has run (randomize the query order and retest a week later) described concretely enough to run, and the conclusion of the book."
---

# Open Problems and the Decisive Experiment {#sec-open-problems}

Preferential Bayesian optimization (PBO) finds the design a person likes best
by asking them to compare options and fitting a model of their utility to the
answers (@sec-pbo). The chapters before this one reported what is known about
it: the methods, the theory, the studies with people, and what other
disciplines say a preference is. The two previous chapters drew two
conclusions from that record. A single comparison mixes a stable preference,
structured noise, change caused by the questions, and answers with no
preference behind them (@sec-what-comparisons-measure), and a system built on
comparisons should be instrumented and judged against strong simple baselines
(@sec-recommendations).

This chapter lists what is not known. It collects eighteen open problems, and
for each one names an experiment and the result that would decide it. The
first group concerns preference itself, the second theory and dimension, and
the third the choice of methods in practice. One experiment, randomizing the
order of queries and retesting a week later, would change our understanding
more than any other, and the chapter describes it concretely enough to run. The
chapter, and the book, end with a conclusion.

The experiments are proposals. Where we add detail to them (sample sizes,
numbers of retest pairs, how an arm is balanced), the detail is our own
inference, offered as a starting point for a design rather than as a finding.

## Problems about preference itself {#sec-open-preference}

The first problems decide how a comparison should be understood: as a noisy
readout of a fixed utility, or as something the session partly makes.
@tbl-open-preference lists them.

::: {.table #tbl-open-preference title="Open problems about preference itself, each with an experiment and the result that would decide it."}
| Problem | Experiment | Result that would decide it |
|---|---|---|
| Does the order of queries change the final preference? (Noisy discovery against preference scaffolding) | randomize people to query orders and retest a week later; see @sec-open-decisive | see @sec-open-decisive |
| Does the current best gain value from being chosen again and again? | at several delays during the session, insert comparisons of the current best against options rejected earlier; compare likelihoods with and without a choice-history term by their predictive log-likelihood on held-out pairs | the model with a choice-history term is significantly better, and the rate at which rejected options win back rises with the delay |
| How much of the inconsistency within a session is noise, drift, or incompleteness? | repeat identical pairs at different lags, insert placebo pairs, and offer a "can't compare" answer | agreement that does not depend on the lag means noise dominates, agreement that falls with the lag means drift; placebo pairs give the false-preference rate; "can't compare" answers concentrated on complex pairs support the fourth kind of answer |
| Does allowing incomplete answers improve the results? | within subjects, compare forced choice with an interface that offers "about the same" and "can't compare" | with incomplete answers allowed, violations of transitivity fall and the delayed retest of the final design agrees more often; then the extra answers should be the default |
| Does the hierarchical view of preference apply to design? | the same people judge, repeatedly over several sessions, both abstract goals (such as "comfortable") and concrete parameter configurations | judgments of abstract goals have significantly higher test-retest reliability than judgments of configurations: the layered model holds; equal reliability: the hierarchical view has no empirical support in design tasks |
| How does the value of a population prior depend on the domain? | in faces or landscapes, and in architecture or artworks, compare a strong population prior, a prior per group, and a weak prior | a strong prior wins for the natural domains and a weak or group prior for cultural artifacts: set the strength of the prior by domain |
:::

**Choice history.** Choosing an option raises its value for the chooser
[@enisman2021choice; @zylberberg2024value], and in a PBO session the current
best is the option chosen most often. The test is a model comparison on held-out
answers: if a likelihood with a term for recent choices predicts better than
one without, and options the person rejected become more likely to win again as
time passes, the effect is present in design comparisons, where it has never
been measured (@sec-syn-components).

**Noise, drift, and incompleteness.** Repeated pairs at different lags separate
noise, which does not depend on the lag, from drift, which does. One caution
from @fig-rec-lagged-repeats: induced change that fades also makes agreement
fall with the lag, so a falling curve alone does not identify drift
(inference); comparing query orders does (@sec-open-decisive). Placebo pairs,
the same design shown twice, estimate how often people express a preference
where none can exist [@omahony2017evolution]. A "can't compare" answer that
concentrates on complex pairs would show that the fourth component of
@sec-syn-components is real in design tasks; in experiments with lotteries and
money, 40% to 50% of participants used such an answer when allowed
[@nielsen2026revealed].

**Incomplete answers as a default.** Forced choice turns indifference,
indecisiveness, and experimentation into the same coin flip
[@ok2022indifference]. If allowing the extra answers reduces transitivity
violations (choosing $a$ over $b$ and $b$ over $c$ but $c$ over $a$) and makes the
final design hold up better at a delayed retest, the cost of the extra buttons
is repaid (inference).

**The hierarchical view.** One position in the literature holds that
preferences over ultimate goals are stable while preferences over intermediate
outcomes are elicited (@sec-syn-hierarchical). The evidence for stability comes
from self-reported values [@vecchione2020stability] and the evidence for
construction from choices, so the two levels have never been tested on the same
task. Asking the same people, across sessions, to judge both an abstract goal
and concrete configurations would test the view where PBO operates.

**Population priors by domain.** Shared taste is high for faces and landscapes
and low for architecture and artworks [@vessel2018stronger]. If a strong
population prior helps in the first domains and a weak or group-specific prior
in the second, the strength of the prior is a parameter to set per domain
(@sec-rec-surrogate).

## Problems about theory and dimension {#sec-open-theory}

The next three problems decide what the theory guarantees and where the
dimension limit of PBO really comes from (@tbl-open-theory). Two quantities
from regret theory appear in them (@sec-syn-theory): $\gamma_T$, the maximum
information gain of the kernel after $T$ queries, and $\kappa$, an upper bound
on the inverse slope of the link function. A third is $B$, a bound on the norm
of the utility function in the kernel's own function space, which limits how
rough the utility may be.

::: {.table #tbl-open-theory title="Open problems about theory and dimension."}
| Problem | Experiment or analysis | Result that would decide it |
|---|---|---|
| A lower bound for kernelized preference feedback under a logistic or probit link | derive minimax lower bounds under Gumbel or normal noise with explicit dependence on $\kappa$; check whether the warm-up constant of MR-LPF hides $\kappa$ or $\exp(B)$ | a lower bound that matches $\tilde O(\sqrt{\gamma_T T})$ establishes order optimality; a mismatch means pairwise feedback really is more expensive, or the upper bound can be improved |
| Order-optimal guarantees for fully sequential algorithms and for the pipeline used in practice | analyze the frequentist regret of a Laplace or expectation-propagation posterior with EUBO, or construct a counterexample in which it is inconsistent; try to remove the extra $\sqrt{\gamma_T}$ factor of sequential algorithms | a sequential algorithm with $\tilde O(\sqrt{\gamma_T T})$, a guarantee for the practical pipeline, or a counterexample |
| Does the dimension ceiling come mainly from the default prior, and does that hold with human comparisons? | at budgets of 50, 100, and 200 comparisons, compare qEUBO with the default prior, qEUBO with a dimension-scaled prior, a local method that does not restrict the lengthscale, a linear utility model, and a reduced representation; first in simulation, then with people in 20- to 50-dimensional latent spaces of generative models; record the gradients of the Laplace marginal likelihood with respect to the hyperparameters | if the dimension-scaled prior closes most of the gap, the ceiling comes mainly from the defaults; if only the reduced representation works, the limit comes from the budget and the information per query |
:::

**Lower bounds.** All regret results for kernelized preference feedback are
upper bounds. The batched algorithm MR-LPF reaches $\tilde O(\sqrt{\gamma_T
T})$, matching the scalar rate, but needs a warm-up phase whose constant does
not depend on $T$ [@kayal2025bayesian], and the question is whether that
constant hides a dependence on $\kappa$ or on $\exp(B)$, which could be very
large. No lower bound exists under a logistic or probit link
(@sec-theory-lower).

**The pipeline in practice.** Theory analyzes elimination, optimism, and
Thompson sampling on frequentist kernel estimators; practice runs a Laplace
posterior with EUBO (@sec-syn-theory). A regret analysis of that pipeline, or a
counterexample showing that it can fail to converge, would close the gap
between what is proved and what is used.

**The dimension ceiling.** A common claim is that PBO fails beyond 10 to 20
dimensions. The evidence points instead to a property of the default
configuration: `PairwiseGP`'s lengthscale prior does not scale with dimension,
so the default kernel treats points in high dimension as nearly unrelated
(@sec-rec-surrogate gives the arithmetic; see also @sec-obs-pairwisegp and
@sec-hd-restated). For scalar
feedback, dimension-scaled priors repaired this [@hvarfner2024vanilla], but
with pairwise feedback and human budgets the repair is untested. The proposed
experiment separates two explanations: if the dimension-scaled prior closes most
of the gap to a reduced representation, the ceiling was the defaults; if only
the reduced representation works, the limit is how little each answer carries.
Recording the gradients of the marginal likelihood shows whether the
lengthscales are being learned at all. Simulation comes first, so that the
experiment with people, in a 20- to 50-dimensional latent space of a generative
model, is not confounded by the default prior.

@fig-hd-race runs the scalar version of this experiment on a toy problem: a
six-dimensional function hidden among up to 50 inputs, with fixed-scale and
dimension-scaled priors and two ways of searching the acquisition function. Its
lengthscale panel shows whether the lengthscales are learned, and in those runs
the answer is a matter of degree: under the fixed-scale prior the inputs that matter get lengthscales
only about three times shorter than the rest, while the dimension-scaled prior
switches most of the irrelevant inputs off. The figure's regret curves show why
the result must be read together with the acquisition search.

## Problems about practice {#sec-open-practice}

The remaining problems decide which methods to use (@tbl-open-practice). Most
of them could be settled by an experiment with people that compares a handful
of variants on the same task.

::: {.table #tbl-open-practice title="Open problems about the choice of methods in practice."}
| Problem | Experiment | Result that would decide it |
|---|---|---|
| Does algorithmic tuning beat manual self-tuning? | the same participants and the same time budget, an exoskeleton or a prosthesis, manual self-tuning against pairwise PBO, with the order of conditions balanced and an objective endpoint preregistered | metabolic or biomechanical endpoints, delayed consistency of preference, and time; if there is no difference, manual self-tuning should be the default |
| How large are the differences between acquisition functions with people? | the same interface and budget, people randomized to qEUBO, dueling Thompson sampling, the maximally uncertain challenge, and random queries | if the differences are smaller than the differences between people and the test-retest noise, the choice of acquisition function is a secondary question in practice |
| What are response times worth in the closed loop? | record response times; in the closed loop, compare a joint likelihood of choices and response times (with the overall value of the pair as a covariate) with the probit likelihood | the joint likelihood predicts held-out pairs better, and the stopping times it implies correlate with the agreement of the delayed retest |
| Can labels from language models replace labels from people? | the same task and the same language-in-the-loop pipeline, answered by people and by decision makers simulated with a language model; compare individual consistency, transitivity, position bias, and final designs | if individual-level agreement is near the 0.11 to 0.22 found for simulated users in PRISM-X, against 0.57 for the agreement between people's own ratings and rankings, simulated evaluations cannot be extrapolated to people [@kirk2026prism] (preprint) |
| There is no public data set of individual pairwise judgments | build a public data set with timestamps, presentation order, response times, repeated pairs, and delayed retests, covering perceptual parameters, aesthetic design, and device tuning | once it exists, link functions, noise and drift models, and the gap between simulated and real users can be compared offline |
| Do EUBO's collapse and the rank-deficient Hessian cost anything with real people? | on the same task with people, compare EUBO, EUBO with a connectivity constraint, a diagonal correction scaled by the prior uncertainty, and the exact knowledge gradient | if the corrections give significantly better final designs or delayed consistency, the default pipeline needs to change |
| Does the inference method matter at human noise levels? | a closed-loop experiment with people comparing the Laplace approximation, expectation propagation, exact skew Gaussian process sampling, and the hallucination believer | no difference in final designs: the debate about inference is secondary in practice; a difference: the default implementation should be replaced |
| When does active selection beat random selection in collecting preference data for language models? | the same model, labels, and compute budget; compare a reward model with explicit epistemic uncertainty and information-directed selection, the implicit reward of direct preference optimization with uncertainty-based selection, and random selection; report win rates and general capability | if only the variant with an explicit posterior beats random selection reliably, the gain comes from where the uncertainty is represented |
| A stopping rule for a pairwise likelihood | carry cost-aware stopping over to the pairwise likelihood, and judge it by the agreement of the delayed retest and by the person's satisfaction | if the carried-over rule saves comparisons without lowering delayed agreement, it can be the default stopping criterion |
:::

**Manual against algorithmic tuning.** In exoskeleton tuning, people adjusting
four hip-timing parameters with a thumbstick reached a 16.6% reduction in
metabolic cost in about 11 minutes [@schafer2026user], of the same order as
algorithmic tuning reported in other studies with different devices and
controls. No study has compared the two on the same participants against a
preregistered endpoint, an outcome fixed before data collection
(@sec-health-manual, @sec-cs-exo-self-tuning). Because people adapt to a device
over about 109 minutes of assisted walking [@poggensee2021adaptation], the
order of the two conditions has to be balanced across participants.

**Acquisition functions with people.** The candidates in the table are the
decision-theoretic qEUBO (@sec-eubo), two older heuristics, dueling Thompson
sampling, which takes the best option under one random sample of the
posterior and pairs it with the option whose duel against it is most uncertain,
and the maximally uncertain challenge, which pits the option with the largest
posterior mean against the option whose comparison with it is most uncertain in
the model's knowledge (@sec-acqf-early), and random queries as a control. We found no study that randomized people to
different acquisition functions with the same interface and budget
(@sec-acqf-empirical). If the differences between acquisition functions turn
out smaller than the differences between people and the noise of a retest,
the work on acquisition functions matters less, in practice, than the
observation model (inference).

**Response times, simulated users, and data.** Response times improved models
on real human data [@shvartsman2024response], but they have not been tested in
closed-loop optimization. Language-model simulations of users reproduce population-level
rankings almost perfectly but agree with individuals at a Kendall $\tau$ (a
rank correlation from $-1$ to $1$) of only 0.11 to 0.22 [@kirk2026prism], so
whether they can stand in for people in a PBO loop is open
(@sec-llm-judges). A public data set of individual comparisons, with the
timing and order information listed in the table, is the prerequisite for most
offline comparisons in this list (@sec-sw-datasets).

**The default pipeline.** EUBO's tendency to collapse toward the estimated
maximum, and the rank-deficient Hessian that its isolated pairs cause, are
documented in 2026 preprints [@wu2026knowledge; @shao2026adaptive], and the error of the Laplace
approximation was measured at very low noise [@takeno2023practicalc]. The
alternatives in the table are expectation propagation and exact sampling of the
skew Gaussian process posterior (@sec-approx-inference), and the hallucination
believer, which plugs a single posterior sample in as if it were data
(@sec-choosing-pairs). Whether
these failures cost anything with real people, at real noise levels, is the question
(@sec-obs-graphs, @sec-obs-inference).

**Language models and stopping.** In online direct preference optimization, a
2026 workshop paper found active selection only negligibly better than random
[@oh2026random]; whether an explicit posterior over rewards changes that is
open (@sec-llm-random). Stopping rules from scalar Bayesian optimization
[@wilson2024stopping; @xie2026cost] have not been carried over to a pairwise
likelihood, and @sec-rec-stopping explains why a person's cost per query is
not constant.

## The decisive experiment {#sec-open-decisive}

Of all the problems above, one would change our understanding most: whether
the order of queries changes the final preference. It decides between two
readings of every PBO session, **noisy discovery**, in which repeated
comparisons recover a preference that was there before, and **preference
scaffolding** or construction, in which the questions help make the preference
they measure (@sec-syn-found-or-made). It also supplies the controls that
judging the legitimacy of a system-induced change requires
(@sec-syn-intervention), and its retests separate noise from drift. As of
September 2026, no study has run it.

### The design {#sec-open-design}

In outline, the design is simple. Everyone gets the same candidate space and
the same budget of comparisons. Participants are randomly assigned to a
sequence of queries chosen by the acquisition function, a random sequence, or a
sequence in balanced order. Before the session, each person's attitude toward
having their taste changed is recorded. At the end of the session and again a
week later, without the system's framing, each person chooses between the final
design and options they rejected early. @alg-open-decisive fills in the details
we would choose; every number in it is our suggestion (inference).

::: {.algorithm #alg-open-decisive title="The decisive experiment: randomized query order with a delayed retest (details are our inference)"}
1. **Task.** Choose a perceptual design task with a small, fixed set of
   rendered candidates, for example 4 to 6 parameters of a visual design or a
   photo adjustment (@sec-cs-photo), so that final designs from different
   people can be compared and the task can run online.
2. **Arms.** Randomize each participant to one of three arms with the same
   interface and the same budget, for example 40 comparisons: *acquisition*
   (qEUBO or EUBO on `PairwiseGP`), *random* (uniformly random pairs), and
   *balanced* (every candidate shown equally often, in an order counterbalanced
   across participants). In every arm, compute the final design and the early
   favorite with the same model, so that the arms differ only in which
   questions were asked.
3. **Before the session.** Record the participant's attitude toward having
   their taste changed, and their familiarity with the domain.
4. **During the session.** At fixed positions shared by all arms, insert two
   placebo pairs and a few repeated pairs at different lags. Randomize left and
   right. Log every answer with its time and response time.
5. **Early favorite.** After the 10th answer, record the design with the
   highest posterior mean.
6. **End-of-session retest.** Immediately after the session, in a neutral
   interface with no labels and no history, show the final design against each
   of about 8 designs the participant rejected in the first 10 answers, in
   random order and position, mixed with filler pairs. Ask whether they endorse
   the final design.
7. **Delayed retest.** One week later, repeat step 6 with the same pairs.
8. **Analysis.** Preregister the hypotheses, outcomes, and analysis below
   before collecting data.
:::

### What decides it {#sec-open-criterion}

The decision criterion has two halves. If the final designs of the acquisition
arm lie closer to their own early posterior mean, and their delayed retest
agrees less often than in the random arm, then change induced by the queries is
a substantive component of the answers. If the distribution of final designs
and the delayed agreement do not differ between the arms, noisy discovery is
enough to explain the data.

Our analysis of the model in @fig-syn-four-components suggests a refinement
(inference). In that model, the second half of the criterion, comparing the
delayed agreement of the two arms directly, can fail even when induced change is
present and fades. At the figure's default settings, the acquisition arm's
final designs are still preferred more often a week later than the random
arm's (78% against 73%), because the acquisition arm reaches a higher agreement
at the end of the session (88% against 76%) and part of that lead survives. The
signature that holds up is the **drop** in agreement from the end of the session
to the delayed retest, compared across arms: 10 percentage points in the
acquisition arm against 3 in the random arm, while with induced change set to
zero both drops vanish. A reader can reproduce this by setting the figure's
sliders and reading its retest panel.

The model also shows the limit of the drop. When the induced change lasts
(the *Lasting share* slider at 1), neither arm drops, and only the first half
of the criterion, final designs close to the early favorite and a different
distribution of final designs across arms, still detects it
(@exr-syn-lasting). We would therefore preregister three contrasts (inference):

- **Primary:** the mean drop from the end-of-session retest to the delayed
  retest, acquisition arm against random arm (a difference in differences).
- **Secondary:** the distance between the final design and the early favorite,
  and the distribution of final designs, compared across arms; and the
  direct comparison of delayed agreement between arms.
- **Diagnostic:** agreement on repeated pairs by lag and the false-preference
  rate on placebo pairs, pooled over arms, which separate noise and drift as in
  @fig-rec-lagged-repeats.

::: {.table #tbl-open-outcomes title="Outcomes of the decisive experiment and how to read them (inference)."}
| Pattern | Reading |
|---|---|
| No drop in either arm, same distribution of final designs | noisy discovery explains the data |
| A larger drop in the acquisition arm than in the random arm | change induced by the queries, at least partly transient |
| No drop, but the acquisition arm's final designs sit closer to the early favorite and differ in distribution | lasting change induced by the queries; whether it is acceptable is a question of legitimacy, not measurement |
| Equal drops in all arms | drift or fatigue that does not depend on the queries |
| Low agreement at both retests in all arms, many "can't compare" answers on complex pairs | noise and incompleteness dominate |
:::

### How many people {#sec-open-size}

The experiment is cheap in equipment and expensive in participants, because the
primary outcome is a small difference between two differences of proportions.
@fig-open-sample-size gives a planning approximation: a person's measured drop
varies because of the binomial noise of a few retest choices at each time and
because people differ, and a standard two-sample test then needs the number of
participants per arm that the figure marks.

```{figure}
//| figure: open-sample-size
//| label: fig-open-sample-size
//| fig-cap: "A planning approximation for the decisive experiment, not a result. The curve is the power of a two-sided test at the 5% level to detect the chosen difference in the drop of retest agreement between the acquisition and random arms, against the number of participants per arm. A person's measured drop has variance equal to the spread of true drops between people plus the binomial noise of their retest choices at the two times; the approximation ignores clustering and any correlation between a person's two retests. The default difference of 7 percentage points is the one in the simulation of @fig-syn-four-components at its default settings; the other defaults are illustrative."
```

At the defaults, a difference of 7 percentage points in the drop, 8 retest
pairs per person, agreement around 80%, and a spread of 0.1 between people,
about 161 participants per arm reach 80% power, 483 in three arms. Doubling the
retest pairs to 16 cuts that to 97 per arm; a difference of 10 points with 16
pairs needs 48. The number of retest pairs is the cheapest lever, since each
pair costs seconds while each participant costs a session and a return visit a
week later (inference). Numbers of this size are within reach of an online
study with a visual task, and out of reach of a laboratory study with a device,
which is one reason to run the experiment first where it is cheap.

### Practical notes {#sec-open-notes}

**Keep the task low-stakes.** The experiment deliberately includes an arm that
may shape preferences more than others. A design task without consequences
beyond the session, informed consent that explains that the system may
influence choices, and a debriefing after the delayed retest keep it within
ordinary research ethics (inference).

**Extensions.** The same design answers other problems in this chapter with one
more arm each: an arm with manual self-tuning answers whether the algorithm is
needed at all, arms with different acquisition functions answer how much they
differ with people, and an arm with the extra answers "about the same" and
"can't compare" answers whether incomplete answers should be the default.

**Why it has not been run.** It needs no new method, only a commitment to
measuring the system's effect on the people it optimizes for. The two
retests, the random arm, and the attitude question are the same controls that
@alg-rec-session recommends for every session, so a study that already follows
those recommendations has most of the decisive experiment built in.

## Settled, contested, missing {#sec-open-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** None of the eighteen problems. What is settled is the ground they
stand on: choosing changes preferences by a considerable amount; a forced
choice hides incomplete preferences; regret bounds for preference feedback are
upper bounds only; the default `PairwiseGP` prior does not scale with
dimension; and a disconnected comparison graph leaves relative utilities
undetermined.

**Contested.** Whether EUBO's collapse and the rank-deficient Hessian matter
with people (2026 preprints); whether inference errors matter at human noise
levels (two groups disagree about realistic tests); whether random selection
is hard to beat beyond the online preference optimization of language models
(a workshop paper).

**Missing.** Every experiment with people in this chapter: randomized query
order with a delayed retest; manual against algorithmic tuning on the same
participants with a preregistered endpoint; acquisition functions compared on
people; dimension-scaled priors tested with human comparisons; and a public
data set of individual pairwise judgments. On the theoretical side, a lower
bound under a logistic or probit link.
:::

## Outlook: where the problem is growing {#sec-outlook}

The open problems above are about what is not yet known. This section is about
where the research is worth doing, and it is the book's own view rather than a
summary of evidence: every paragraph in it could carry the mark
*(inference)*, so it is marked once here.

It helps to separate two questions. Bayesian optimization and its preferential
variant are answers to one problem: finding what is good when each evaluation
is expensive, and, in the preferential case, when the only instrument is a
person's judgment. The first question is whether the algorithm family has much
further to go. The second is whether the problem it answers is becoming more
or less important.

**The algorithm family.** Its future is modest. The core is mature:
decision-theoretic acquisition functions with a multi-option form
(@sec-eubo), a default pipeline in widely used software (@sec-sw-botorch),
and regret theory that has caught up with scalar feedback at the level of
upper bounds (@sec-theory-rates). The evidence with people is small: the median user
study in the evidence map of @fig-hci-evidence has 12 participants, and
simpler methods often do as well (@sec-syn-simpler). Another acquisition
function is unlikely to change that picture. Better measurement could.

**The problem.** It is growing. Systems increasingly generate, adapt, and
personalize, and each of those needs to learn what one particular person
wants from a handful of judgments. The directions below are where that
problem meets an opening, ordered roughly from the measurement questions at
the center of this book outward. Each names what exists, what is missing, and
the combination of skills it rewards.

### Measurement science for preferences {#sec-outlook-measurement}

The niche closest to this book's argument. The observation model, probit or
logistic noise of constant size, has never been compared with an alternative
on human data; noise that depends on how hard a comparison is, response times,
"can't tell" answers, and drift over a session are all open
(@sec-open-preference, @sec-obs-extensions). The decisive experiment of
@sec-open-decisive, randomized query order with a retest a week later, would
separate a preference that was found from one that was shaped, and nobody has
run it. And no public data set of individual pairwise judgments exists
(@sec-sw-datasets), so almost every offline comparison of methods runs on
simulated people. The work is cheap, mostly participant time with existing
interfaces, and has high leverage, because every method in this book sits on
top of the observation model. It rewards someone who can design a careful
experiment with people and also write the model: human-computer interaction
or psychophysics together with probabilistic machine learning.

### Personalizing generative models {#sec-outlook-generative}

A foundation model is a strong prior over what people in general like and a
weak one about what a specific person likes. A handful of well-chosen
comparisons on top of such a prior is exactly the preferential setup, with a
far better prior than a Gaussian process on raw parameters. The first systems
exist: GimmBO searches the merging weights of 20 to 30 diffusion adapters
[@liu2026gimmbo], MultiBO shows several images per round
[@rajagopalan2026personalized], and APPO hides the text prompt and asks only
for binary preferences between images while a language model rewrites the
prompt [@li2026preference] (@sec-hci-generative). Missing are a surrogate that
uses the generative model itself rather than a Gaussian process over its
latent space (@sec-hd-pretrained), tests with people in the 20 to 50
dimensions such spaces have (@sec-hd-restated), and an evaluation that
separates what the person specifically likes from what the population likes
(@sec-hci-population). It rewards generative modeling together with
Bayesian optimization and user studies.

### Choosing which comparisons to collect for language models {#sec-outlook-alignment}

Reward models and direct preference optimization learn from pairwise
comparisons at scale, and they face the questions this book studies: which
pair to ask about next, and when to stop (@sec-llm-active, @sec-llm-online).
The setting differs in ways that matter: millions of comparisons instead of
dozens, many raters instead of one person, and a population preference as the
target (@sec-llm-population). Whether choosing pairs actively beats random
pairs at that scale is itself unsettled (@sec-llm-random), and disagreement
between raters may be signal rather than noise, the problem social choice
studies (@sec-econ-social-choice). This is where most of the funding and
attention are. It rewards large-scale machine-learning engineering together
with the theory of active learning, and it gains from the measurement work
above, since raters are people too.

### Devices on the body {#sec-outlook-body}

Exoskeletons, prostheses, hearing aids, and neurostimulation share the
conditions under which preferential optimization makes most sense: comparisons
are natural, objective measures are slow or expensive, and people differ more
than any population default allows. The strongest real-world evidence in this
book comes from here. Bayesian optimization tuned two timing parameters of a
soft exosuit in 21.4 ± 1.0 minutes [@ding2018human]; wearers tuning four
parameters themselves with a thumbstick remote control took 10.9 minutes
[@schafer2026user]; preset selection for over-the-counter hearing aids has
been posed as a dueling bandit [@vyas2022personalizing]; a variant of safe
optimization that takes preference feedback was applied to spinal cord
stimulation [@sui2018stagewise]; and a thesis tuned the encoders of retinal
prostheses with people in the loop [@fauvel2021human] (@sec-health). Missing
are a head-to-head comparison of manual and algorithmic tuning on the same
participants (@sec-health-manual), exploration that stays safe for the body,
and studies that follow a person for weeks while their body and preference
adapt. It rewards biomechanics or clinical research together with safe and
non-stationary Bayesian optimization.

### Simulated people and model judges {#sec-outlook-simulators}

Because studies with people are expensive, nearly every comparison of methods
uses simulated people (@sec-sw-simulated-users), and language models now act
as judges and stand-in users. One measurement shows the risk: simulated users
reproduced population-level rankings almost perfectly but agreed with
individuals at a Kendall $\tau$ of only 0.11 to 0.22 [@kirk2026prism]
(@sec-llm-judges). A simulator validated against real comparison data, with
order effects and drift, would make offline benchmarks informative; an
unvalidated one makes them circular. It rewards user modeling together with
evaluation methodology, and it depends on the public data set the first
direction would produce.

### Science and engineering with experts in the loop {#sec-outlook-labs}

Self-driving laboratories now report how much faster an optimizer reaches a
target than a reference strategy [@adesiji2026benchmarking]. Expert judgment
helped where the goal had no sensor, as in printing objects with subjective
qualities [@deneault2025preferential], and lost to the optimizer where yield
could be measured [@shields2021bayesian] (@sec-sci-experts). Missing are
models that combine measured objectives with an expert's pairwise
preferences, queries that treat the expert's time as a cost, and studies with
more than one expert in the loop. It rewards domain science together with
multi-objective and cost-aware Bayesian optimization.

### Many people, one setting {#sec-outlook-groups}

A thermostat in a shared office, a recommender, or a design meant for a
population is tuned by the preferences of many people at once
(@sec-sci-buildings). Aggregation, fairness, and strategic answers are
questions social choice has studied for decades (@sec-econ-social-choice),
and a prior learned from earlier people in the same domain could cut the
number of questions for each new one (@sec-open-preference). It rewards
mechanism design or social choice together with preference learning.

### Theory for the pipeline people deploy {#sec-outlook-theory}

All regret results for kernelized preference feedback are upper bounds; there
is no lower bound under a logistic or probit link, and the algorithms that
theory analyzes, elimination and optimism, are not the decision-theoretic
acquisition with a Laplace approximation that software runs
(@sec-open-theory). Preferences that drift during a session have almost no
theory. It rewards learning theory, with enough contact with practice to
analyze what is actually deployed.

### Signals beyond the click {#sec-outlook-signals}

A comparison records more than which option won: how long the person took,
and how much they hesitated. A drift-diffusion model links choice and time
(@sec-neuro-ddm), and response times improved models on real human data
[@shvartsman2024response], but they have not been used to choose the next
query. Eye movements and physiological signals are further candidates. The
open question is which of these signals are robust across people and
interfaces, and whether they improve the choice of queries and not only the
fit. It rewards cognitive modeling together with Bayesian optimization.

### The intervention as a design problem {#sec-outlook-legitimacy}

If a session can change a preference, a system should be able to detect when
it does, and a study should state the conditions under which that change is
acceptable (@sec-syn-legitimacy, @sec-phil-autonomy). Measuring preference
change within a session, stopping rules that account for it, and interfaces
that show a person how their answers moved are technical problems, not only
ethical ones. It rewards human-computer interaction together with statistics
and moral philosophy.

::: {.table #tbl-outlook title="Research directions where the problem of learning what is good from few judgments is growing: what is missing, and what each rewards."}
| Direction | Most important missing piece | Rewards | Start with |
|---|---|---|---|
| Measurement science | observation models tested on human data; the randomized retest | experiments with people + probabilistic modeling | @sec-open-decisive |
| Generative models | the generative model as the surrogate; tests in 20 to 50 dimensions | generative modeling + BO + user studies | @sec-hci-generative |
| Language-model comparisons | whether active pair selection helps at scale | large-scale ML + active learning | @sec-llm-active |
| Devices on the body | manual against algorithmic tuning, same people | biomechanics or clinical research + safe BO | @sec-health-manual |
| Simulated people | simulators validated on individual data | user modeling + evaluation | @sec-sw-simulated-users |
| Experts in the loop | measured objectives combined with expert preferences | domain science + multi-objective BO | @sec-sci-experts |
| Many people | aggregation and fairness for shared settings | social choice + preference learning | @sec-econ-social-choice |
| Theory | a lower bound under a logistic or probit link | learning theory | @sec-open-theory |
| Signals beyond the click | response times used to choose queries | cognitive modeling + BO | @sec-neuro-ddm |
| The intervention | measuring preference change within a session | HCI + statistics + ethics | @sec-syn-legitimacy |
:::

What the directions share is the book's argument. The bottleneck has moved
from algorithms to measurement, and most of the valuable work sits where a
careful experiment with people meets a well-specified model. People who can do
both are scarce, which is the opportunity.

## Conclusion {#sec-conclusion}

Between 2017 and 2026 the largest change in PBO
was not a new methodological paradigm but a shift in where the attention went.
Acquisition functions gained a decision-theoretic foundation, regret theory
caught up with scalar feedback at the level of upper bounds, and software fixed
a default pipeline. At the same time, every layer of that default was found to
rest on an untested assumption: the lengthscale prior does not scale with
dimension, the Laplace approximation is ill-conditioned when the comparison
graph is disconnected, and the probit link with constant noise has never been
compared with an alternative on human data. The evidence gives an order of
work: first get the observation model and the representation shown to the
person right, then choose the acquisition function.

About preference itself, the evidence since 2017 supports neither pure
discovery nor pure construction. A single pairwise judgment contains a stable
component, structured evaluation noise, change caused by the query, and
incomplete or deliberately randomized answers, and which of them dominates
depends on the domain, the similarity of the options, and the length of the
session. The practical consequence is that the output of a PBO session is both
an estimate and an intervention. A study that does not randomize the order of
queries or retest after a delay cannot tell whether what converged was the
preference or the system's effect on the person, and because the controls
needed to tell them apart are the controls needed to judge whether that effect
was legitimate, the methodological question and the normative one can be
answered in the same experiment (inference).

Three positions in the literature come out of this with qualified support. The
hierarchical view of preference is supported where it says that stability lies
at the broad, abstract, self-reported level, and not supported where it
predicts that iterated querying converges on an underlying preference
(@sec-syn-hierarchical). Preference scaffolding is partly supported as a
description of what PBO does, but as a normative position it needs conditions
of legitimacy, and endorsement after the fact cannot supply them on its own
(@sec-syn-scaffolding). And the common claim that PBO stops working beyond 10
to 20 dimensions describes the default configuration, not a limit of preference
feedback (@sec-hd-restated).

Four pieces of work are the most likely to change these conclusions: randomized
query order with a delayed retest; manual self-tuning against algorithmic
tuning on the same participants; dimension-scaled priors tested with human
comparisons on 20- to 50-dimensional representations; and a public data set of
individual pairwise judgments. The first two can be done with existing devices
and interfaces, and their main cost is participants' time; the third needs
simulation first, to rule out the default prior as a confound; the fourth is
the precondition for most of the offline comparisons in this chapter. On the
theoretical side, the most important gap is a lower bound under a logistic or
probit link.

Until these results exist, the evidence supports a modest practice: treat
PBO as one of several baselines, instrument
every session so that it measures the person as well as the design, and, in
every study, measure what the system does to the people it learns from.

## Exercises {#sec-open-exercises}

::: {.exercise #exr-open-size}
Using @fig-open-sample-size, a team can afford 300 participants in total across
three arms and expects a difference of 7 percentage points in the drop. How many
retest pairs per person do they need for 80% power, assuming agreement around
80% and a spread of 0.1 between people? What if the spread between people is
0.15?

::: {.solution}
With 100 participants per arm, the variance $s^2$ of a person's drop must
satisfy $2 (1.96 + 0.84)^2 s^2 / 0.07^2 \le 100$, so $s^2 \le 0.0312$. With a
spread of 0.1 between people, the binomial part, $2 \cdot 0.8 \cdot 0.2 / k =
0.32 / k$, must be at most $0.0312 - 0.01 = 0.0212$, so $k \ge 15.1$: 16 retest
pairs per person (the figure shows 97 per arm at 16 pairs). With a spread of
0.15, the binomial part must be at most $0.0312 - 0.0225 = 0.0087$, so
$k \ge 0.32 / 0.0087 \approx 37$, beyond the figure's range. Once differences
between people dominate, more retest pairs barely help, and the team needs more
participants or a larger expected effect.
:::
:::

::: {.exercise #exr-open-manual}
Sketch the experiment for "Does algorithmic tuning beat manual self-tuning?" on
an ankle exoskeleton with two parameters, peak torque and its timing. What are
the arms, the order, the primary endpoint, and the main threat to validity?

::: {.solution}
A within-subject design: each participant does both conditions, manual
self-tuning (adjusting the two parameters directly, as in self-tuning studies of
ankle torque) and pairwise PBO, with the same time budget, in an order
counterbalanced across participants. The primary endpoint, preregistered,
could be the metabolic cost of walking with each final setting, measured in a
blinded validation block at the end of each condition; secondary endpoints are
the time to converge and the delayed consistency of each person's preference.
The main threat is adaptation: people take on the order of 100 minutes of
assisted walking to become expert users [@poggensee2021adaptation], so a
familiarization phase before either condition, and counterbalancing, are
needed to keep learning from favoring whichever condition comes second.
:::
:::

## Further reading {#further-reading .unnumbered}

- @carroll2024aib and @dean2022preference explain why the order of queries is
  not a detail when the system can move the preference it measures.
- @keswani2026moral is the closest existing measurement of instability in
  repeated pairwise answers across sessions, and a model for the retests of the
  decisive experiment.
- @schafer2026user is the strongest current evidence that a simple baseline
  can match algorithmic tuning, and the starting point for the manual-tuning
  experiment.
- @kayal2025bayesian and @lazzaro2026finiteb state the best current upper bounds
  and their conditions, which a lower bound would have to match.
