---
status: done
synopsis: "Concrete recommendations for deciding whether to use preferential optimization at all, and then for the observation model, the surrogate, query design, non-stationarity, stopping, evaluation, and ethics, with the situations in which a simpler method should be the default."
---

# Building and Evaluating a Preferential Optimization System {#sec-recommendations}

Preferential Bayesian optimization (PBO) searches for the design a person likes
best by asking them to compare options and fitting a model of their utility to
the answers (@sec-pbo). The previous chapter concluded that a single answer
mixes a stable preference with structured noise, change caused by the
questions, and answers with no preference behind them, so that a session is
both an estimate of a preference and an intervention on it, and that the
largest errors come from what the model assumes about the person
(@sec-what-comparisons-measure).

This chapter turns that analysis into advice for someone who has to build or
evaluate a preferential optimization system: three questions that decide
whether PBO is the right tool, the parts of the system one at a time, and the
situations in which a simpler method should be the default. Every
recommendation comes from evidence reported earlier in the book or from our
inference from it, and **none has been tested as a whole in PBO**. Each table
also says whether a recommendation depends on the Gaussian process and PBO
framework. Those that do not apply to any preference model, including the
simple baselines of @sec-rec-simpler; those that do are repairs inside the
framework, and say nothing about whether the framework was the right choice.

## Three questions before choosing PBO {#sec-rec-three-questions}

Faced with a new problem of personalization or design with a person in the loop,
we suggest answering three questions before deciding to use PBO (inference).

::: {.algorithm #alg-rec-three-questions title="Deciding whether to use preferential Bayesian optimization (inference)"}
1. **Is there a numerical objective that can be measured?** If there is,
   optimize it with ordinary Bayesian optimization (@sec-bo-loop) or another
   method suited to it, and use preferences only as a constraint or an auxiliary
   signal. An expert's pairwise input pays off only when their judgment carries
   information the surrogate lacks (@sec-adj-sdl). Tuning a classifier's
   validation error (@sec-cs-classifier) and maximizing a reaction's yield
   (@sec-cs-chemistry) are problems of this kind.
2. **Can the dimension exposed to the person be brought to about 10 or
   fewer?** If not, design the representation first. With budgets of tens to two
   hundred comparisons, the limit comes from how little each answer can carry,
   at most one bit, and from the budget, not from the surrogate (inference).
3. **With few parameters and a flat optimum, would manual self-tuning or a
   coarse search already do?** If it might, it is the first baseline, and PBO has
   to beat it (@sec-cs-exo-self-tuning).
:::

When all three questions point toward PBO, qEUBO with BoTorch's `PairwiseGP` is
a reasonable default (@sec-eubo, @sec-pairwisegp), a choice that comes from the
software ecosystem rather than from a comparison with simpler methods. The
session should be instrumented as in @alg-rec-session: the errors caused by the
observation model and by the person's behavior exceed the differences between
acquisition functions (@sec-syn-bottleneck), and these additions both estimate
those errors and supply the controls needed to tell discovery of a preference
from shaping it (@sec-syn-intervention).

::: {.algorithm #alg-rec-session title="An instrumented preferential optimization session (inference)"}
1. *Before the session*, record the person's attitude toward having their
   taste changed by the system.
2. Decide in advance a fixed share of queries that the acquisition function
   does not choose: random pairs, or pairs in a balanced order. Randomize which
   option appears left or right, or where it sits in a gallery.
3. Offer two answers besides "A" and "B": **"about the same"** and **"can't
   compare"**.
4. Insert a few **placebo pairs**, the same design shown twice, and a few
   **repeated pairs** at different lags.
5. Record the response time of every answer.
6. Label the system's proposals as such, show the comparison history
   accurately, and let the person reset or reject the current best.
7. *At the end*, retest the final design against designs the person rejected
   earlier, without the system's framing, and ask whether they endorse it.
8. *At the next session*, a week or so later, repeat that retest.
:::

Steps 3 to 5 are justified in @sec-rec-observation, step 2 in @sec-rec-query,
step 4 again in @sec-rec-nonstationary, steps 7 and 8 in @sec-rec-stopping, and
steps 1, 6, 7, and 8 in @sec-rec-ethics.

**A note on dimension.** A comparison carries at most one bit
(@sec-comparison-information), so 100 comparisons over 20 parameters carry at
most five bits per parameter, and usually far fewer, because answers are noisy
and many queries are redundant (inference). Where PBO has worked in more than a
handful of dimensions, the dimension the person faced had been reduced by
design, to a 10-dimensional body-shape space of a generative human model or the
merging weights of 20 to 30 diffusion adapters [@koyama2020sequential;
@liu2026gimmbo], or to a 5-dimensional feasibility-aware latent space
(@tbl-rec-simpler).

## Observation model {#sec-rec-observation}

The observation model carries the largest error (@sec-syn-observation).
@tbl-rec-observation lists five recommendations for it; none depends on the
Gaussian process framework.

::: {.table #tbl-rec-observation title="Recommendations for the observation model."}
| Recommendation | Basis | Depends on the GP and PBO framework? |
|---|---|---|
| Offer "about the same" and "can't compare" besides the two choices; model them as an indifference threshold and as incompleteness (or a mixture parameter), and never read them as a 50:50 answer | @nielsen2026revealed (working paper); @ok2022indifference; @erarslan2025consecutive (preprint) | no |
| Insert a few placebo pairs (the same design twice) in every session to estimate the false-preference rate and calibrate the noise or lapse term | @omahony2017evolution | no |
| Let the noise scale vary with difficulty, similarity, and type of feedback (pairwise answers, sliders, ratings); do not fix a low default or one scale for everyone | @shen2025early; @ghosal2023effect; @keffert2024stochastic (preprint) | no |
| Record response times; in a joint likelihood, include the pair's overall value as a covariate | @shvartsman2024response; @sawarni2025preference; @shevlin2022high | recording no; the joint likelihood has a Gaussian process version |
| Add a choice-history term that raises the value of recently chosen options; estimate behavioral parameters such as loss aversion rather than fixing them | @zylberberg2024value; @brown2024meta | no |
:::

**Two extra answers.** A forced choice turns indifference, indecisiveness, and
experimentation [@ok2022indifference] into a coin flip that the model reads as
a weak preference, and when allowed, many people say they cannot compare
(@tbl-syn-strengthened). An "about the same" answer has a standard model, a
just-noticeable-difference threshold, and a 2025 preprint reports that a method
which models it clearly outperforms standard preferential baselines once 10%
to 20% of comparisons are indifferent, on synthetic benchmarks
[@erarslan2025consecutive] (@sec-ties).

**Placebo pairs.** Sensory science found that consumers report preferences
between identical samples, and made pairs of identical samples a standard
control condition [@omahony2017evolution]. In PBO, the rate at which a person
prefers one of two identical renders, identical down to the pixel, estimates
how many answers carry no preference at all and can calibrate a lapse term
(inference).

**Structured noise.** In estimates of risk aversion from a representative
sample, the standard expected-utility random utility model gave distorted
estimates when everyone shared one noise scale and not when each person had
their own, in a 2024 preprint [@keffert2024stochastic]. `PairwiseGP` fixes the
noise at 1 and lets the kernel's amplitude (BoTorch's output scale) stand in
for it (@sec-obs-pairwisegp).

**Response times** are free information about how far apart the options felt.
A joint likelihood of choices and drift-diffusion response times exists for
Gaussian process models [@shvartsman2024response], and a loss built on the
EZ-diffusion model reduces the error of preference learning with linear
rewards from exponential to polynomial growth in the magnitude of the reward
[@sawarni2025preference]. But decisions between two high-value options are
mostly faster and more accurate, not slower [@shevlin2022high], so late in a
session a quick answer does not mean a large difference unless the pair's
overall value is in the model (inference).

**Choice history.** The current best, the most-compared option, gains most
from choice-induced revaluation (@sec-syn-components), and the loss-aversion
coefficient has a meta-analytic mean of 1.955 but is about 1.07 when gains and
losses are symmetric and unsorted (@tbl-syn-weakened), so both belong in the
likelihood as estimated terms, not fixed values.

## Surrogate {#sec-rec-surrogate}

The surrogate is the prior over the utility function. @tbl-rec-surrogate lists
three recommendations, of which only the last is purely a matter of the Gaussian
process framework.

::: {.table #tbl-rec-surrogate title="Recommendations for the surrogate."}
| Recommendation | Basis | Depends on the GP and PBO framework? |
|---|---|---|
| Set the weight of a population prior by domain; expect a population prior to help only in the first iterations, and monitor how far the individual departs from it | @vessel2018stronger; a prior from user models was significantly better only at iterations 2 and 3 [@liao2026efficient] | no |
| In high dimension, first reduce the dimension the person sees, then consider a linear utility model; if you keep `PairwiseGP`, switch to a dimension-scaled lengthscale prior and validate it yourself | @owaki2026feasibility (preprint); @doumont2026we; @hvarfner2024vanilla | the first two steps no; the third is a repair inside the framework |
| At low noise, do not use the Laplace approximation; use expectation propagation or exact skew Gaussian process sampling | @takeno2023practicalc; @benavoli2021preferentialb | yes |
:::

**Population priors.** Shared taste varies by domain (@tbl-syn-components),
and in interactive design a prior pretrained on models of earlier users was
significantly better only at the second and third iterations; from the sixth
on, all methods were comparable [@liao2026efficient]. A population prior buys
a better start, not a better result, and its weight should fall as the
person's own answers accumulate (inference; @sec-many-users).

**High dimension.** The default `PairwiseGP` lengthscale prior is far shorter
than the typical distance between designs in high dimension (about 0.52
against 2.9 at $d = 50$), so the default model treats every comparison as
nearly uninformative about every other point (@sec-obs-pairwisegp,
@sec-hd-defaults). For scalar feedback, priors that scale with dimension
repaired this [@hvarfner2024vanilla], and a spherical input mapping with
Bayesian linear regression reached the state of the art [@doumont2026we].
Neither has been tested with pairwise feedback from people, so reduce the
dimension the person sees first, which helps under any model. @fig-hd-race
adds a second reason: on a toy problem with scalar feedback, how the
acquisition function is searched decided more of the outcome within 80
evaluations than the lengthscale prior did (@sec-hd-seeing).

**Inference.** At very low noise, the Laplace approximation misplaces the
posterior and expectation propagation distorts duel probabilities near 0 or 1
[@takeno2023practicalc]; exact sampling of the skew Gaussian process posterior
avoids both [@benavoli2021preferentialb]. How large the errors are at human
noise levels is unmeasured (@sec-obs-inference), so the switch matters most for
very consistent answerers (inference).

## Acquisition and query design {#sec-rec-query}

The observation model settles what a query records and what happens when the
person cannot answer (@sec-query-design); @tbl-rec-query collects what the
evidence says about choosing and presenting the queries.

::: {.table #tbl-rec-query title="Recommendations for acquisition and query design."}
| Recommendation | Basis | Depends on the GP and PBO framework? |
|---|---|---|
| Keep a fixed share of random or balanced queries; randomize left and right and positions in a gallery; rotate which option is the system's proposal | active learning is no better than random when preferences are unstable or the model class is wrong [@keswani2024pros]; gaze has a causal effect on choice [@bhatnagar2022meta]; @glickman2025human | no |
| Monitor the connectivity and algebraic connectivity of the comparison graph over evaluated designs, and avoid isolated pairs | @shah2016estimation; @hendrickx2019graph; @shao2026adaptive (preprint) | partly: rank deficiency concerns the Laplace approximation, connectivity matters for any Bradley-Terry estimate |
| When users may differ, use rankings of three or more options instead of binary comparisons; penalize pairs that differ in many attributes at once | @chidambaram2026direct; @johnston2017contemporary | no |
| Let people state goals and constraints in natural language, and use comparisons or rankings for fine judgments of appearance; make the state of the search visible and editable | @niwa2025cooperative; @peng2026efficient (preprint) | no |
:::

**Random and balanced queries.** A simulation study of moral preference
elicitation found that active learning rests on three premises (preferences
that are stable and unaffected by the order of queries, a correct model class,
and limited noise), and that when they fail, actively chosen queries do no
better than random ones, or worse [@keswani2024pros]. A fixed share of queries
that the acquisition function does not choose costs some efficiency and buys a
control group inside every session, free of the system's steering, against
which the model's predictions can be checked (@sec-rec-ethics). Position
matters too: steering how long people looked at an option raised its share of
choices to 0.541 (95% confidence interval 0.523 to 0.560), a causal but small
effect [@bhatnagar2022meta], so randomizing positions and rotating which option
the system proposes keeps such effects from adding up in one direction
(inference).

**The comparison graph.** If evaluated designs and answered pairs fall apart
into disconnected pieces, the answers say nothing about how the pieces compare,
and the Laplace approximation becomes ill-conditioned (@sec-comparison-graph,
@fig-comparison-graph). The graph's **algebraic connectivity**, the
second-smallest eigenvalue of its Laplacian matrix, is zero exactly then. Since
EUBO's pairs tend to start new components [@shao2026adaptive]
(@sec-pbo-failure-modes), the check is cheap insurance: when the next pair
would start a new component, swap one of its options for an already-compared
design (inference).

**Rankings and simple pairs.** When users differ in ways the model does not
know, binary comparisons cannot identify the user types, and rankings of at
least three options can [@chidambaram2026direct]. Guidance for
stated-preference studies notes that respondents' consistency falls as the
statistical efficiency and the complexity of the choice tasks rise
[@johnston2017contemporary]; in PBO, the complex tasks are pairs that differ in
many attributes at once (inference).

**Language for goals, comparisons for appearance.** In the two studies of
optimizer-led design ($N = 18$ and $N = 12$; @sec-syn-scaffolding), steering
through natural language gave lower workload than explicit constraints, and
90.9% of 187 requests in natural language described a desired outcome rather
than a parameter [@niwa2025cooperative]. For fine judgments of appearance,
comparisons work better than words: in a 2026 preprint, personalization from
just 8 pairwise judgments beat every baseline for 12 new users, including their
own written preferences, with an aggregate win rate of 60.35%
[@peng2026efficient]. The photo case study lets you try comparisons and line
search on an image (@sec-cs-photo).

## Non-stationarity {#sec-rec-nonstationary}

A preference that changes during a session breaks the model's central
assumption. Two kinds of change need to be separated: **drift**, where the
utility itself moves (through learning, adaptation, fatigue, or change caused by
the queries), and a change in the **choice rule**, where the utility stays put
but the way the person turns it into an answer changes, for instance from
weighing all attributes to deciding on the single one that has come to matter
(@sec-ecological). Drift calls for a time-varying utility, a changed choice
rule for a time-varying noise or link (inference).

::: {.table #tbl-rec-nonstationary title="Recommendations for non-stationarity."}
| Recommendation | Basis | Depends on the GP and PBO framework? |
|---|---|---|
| Show the same pairs again at different lags, to separate test-retest noise from drift; when tuning a device, schedule a familiarization phase and down-weight early comparisons | @keswani2026moral; becoming an expert user took about 109 minutes of assisted walking [@poggensee2021adaptation] | no |
| Where needed, use a time kernel with a fast and a slow scale, or a state-space drift per parameter | inference, from adaptation at several time scales | the time kernel belongs to the GP framework; state-space drift does not |
:::

**Repeated pairs.** When the same moral pairwise questions were asked in three
to five sessions, 6% to 20% of answers flipped, and for some participants the
fitted decision model itself changed [@keswani2026moral]. One repeated pair
cannot tell noise from change; pairs repeated at several lags can, because
noise does not depend on the lag and drift does (@fig-rec-lagged-repeats).

```{figure}
//| figure: rec-lagged-repeats
//| label: fig-rec-lagged-repeats
//| fig-cap: "A simulation of a model, not data. The vertical axis is the probability that a pair shown twice, a few to forty answers apart, gets the same answer both times: the magenta curve under the settings, the dashed curve without drift and induced change. Points are what an experiment with the chosen number of people and repeated pairs per lag would measure, with 95% intervals that treat every repeated pair as independent; real data, which cluster by person, would give wider intervals. Each pair's utility difference is drawn from a standard normal distribution, drift moves it by a random walk, and induced change favors the option chosen the first time and fades over about ten answers. All values are illustrative."
```

Things to try:

- **At the defaults** the curve falls from 72% at a lag of 2 answers to 61% at
  a lag of 40, because the utility differences drift; with 30 people and 4
  repeated pairs per lag, the measured points show the decline. **Set drift to
  0**, and the curve is flat at 73%: noise makes people disagree with
  themselves equally at every lag.
- **Raise *Incomplete* (the share of incomplete answers) to 0.4** with drift at
  0. The curve stays flat and drops to 58%: answers with no preference behind
  them look like extra noise unless "can't compare" is offered.
- **Return *Incomplete* to 0, keep drift at 0, and set *Induced* (induced
  change) to 1.** The curve falls again, from 92% to 74%, because at short
  lags the person repeats a choice that the first choice made more attractive.
  A falling curve does not distinguish drift from induced change; only
  comparing query orders does (@sec-syn-simulation).
- **Reset the figure, then set *People* to 5 and *Repeats* (repeated pairs per
  lag and person) to 1.** The intervals span about ±33 percentage points. A
  diagnostic built on repeated pairs needs on the order of a hundred repeated
  pairs per lag, pooled over people (inference).

**Familiarization.** People adapt to a device over longer than a typical
tuning session (in exoskeleton walking, training contributed about half of a
39% reduction in metabolic cost [@poggensee2021adaptation]), and comparisons
made before they have adapted measure a different person
(@sec-cs-exo-adaptation).

**Models of change.** Where drift is real, a time kernel with a fast and a slow
scale, or a state-space model in which each parameter drifts, can represent it
(inference); over months, a stable component plus deviations that follow
events and then decay fits the evidence better than a random walk
(@sec-social-sciences). Drifting utilities are not yet modeled in PBO
(@sec-obs-extensions), and the theory that exists is for finite arms
(@sec-theory-drift).

## Stopping {#sec-rec-stopping}

When should a session end? The tempting answer is when the posterior has
concentrated: the model is confident about the best design, and EUBO, asked to
choose a pair, would show the incumbent against itself (@exr-eubo-identical).
That is not enough, because the posterior concentrates whenever the answers are
consistent, and choice-induced revaluation makes answers consistent without any
stable preference behind them (@fig-syn-four-components,
@exr-syn-concentration).

::: {.table #tbl-rec-stopping title="Recommendations for stopping."}
| Recommendation | Basis | Depends on the GP and PBO framework? |
|---|---|---|
| Do not treat posterior concentration as evidence that the preference has stabilized, since revaluation after choosing also concentrates it; stop on a plateau in repeated-pair agreement, a delayed retest that reproduces the result, and an endorsement check | inference, from @zylberberg2024value | no |
| Carry over stopping rules based on the probability of regret, or on cost, only after testing them with a pairwise likelihood and a person's changing cost | @wilson2024stopping; @xie2026cost | yes |
:::

**Rules from scalar Bayesian optimization.** Scalar rules stop when a design is
$\varepsilon$-optimal with probability at least $1 - \delta$
[@wilson2024stopping], or when no point's expected improvement exceeds the
cost of evaluating it [@xie2026cost]; both assume a correctly specified
Gaussian process prior (@sec-hd-stopping, @sec-cost-stopping). Preference-based
reward learning has an optimal stopping rule for parametric reward models
[@byk2019asking]. None has been carried over to PBO's pairwise Gaussian
process likelihood, and a person's cost per query is not constant: fatigue
raises it, and learning the task lowers it. Until a rule is carried over,
@alg-rec-stopping is the practical one.

**Response times as a stopping signal.** Answers usually get faster late in a
session. That can mean the preference has settled, but people also stop
deliberating when the expected gain in confidence is no longer worth the
effort [@benon2024online], so faster answers need the same caution as
posterior concentration (inference).

**People stop on their own.** In a three-month field deployment, 415 of 549
evaluation sequences stopped at the first iteration, and only 16 of the
remaining 134 reached a satisfying result [@ou2022human]. A person's "good
enough" and a model's converged posterior are different signals, and a
stopping rule designed for the model must also survive the person's own
decision to quit.

::: {.algorithm #alg-rec-stopping title="A stopping rule for a session with a person (inference)"}
1. Continue while the model's own criterion says more answers are worth
   having, for example while a challenger still has a real chance of beating
   the incumbent.
2. Once it is met, check that agreement on repeated pairs has stopped rising
   over the last few repeats.
3. Retest the incumbent, without the system's framing, against two or three
   designs the person rejected earlier. If it loses, continue.
4. Ask the person whether they endorse the result. Economic experiments find
   that people revise their choices when asked to reflect on principles they
   themselves endorse [@nielsen2022choices], so the question is informative,
   though not by itself a justification (@sec-syn-scaffolding).
5. Stop, and schedule the delayed retest of @alg-rec-session.
:::

## Evaluation {#sec-rec-evaluation}

How should a new PBO method or system be evaluated? Current practice is weak,
with scalar test functions, incompatible noise models and definitions of
regret, and no shared benchmark (@sec-syn-evidence, @sec-sw-evaluation). Three
recommendations, all independent of the framework, address it, and
@tbl-rec-report collects them with the others into a reporting checklist.

**Strong baselines.** Most positive results for interactive Bayesian
optimization in design come from comparisons with sliders, parameter panels, or
random queries; against skilled manual work or a similar optimizer, final
quality usually does not differ, and these systems are best summarized as
reaching the same result at lower cost (@sec-hci-baselines). A method that has
not been compared with manual tuning, random search, and a linear model cannot
claim to be needed.

**Acceleration factors.** Self-driving laboratories report the acceleration
factor, how many experiments a strategy needs to reach a target relative to a
reference strategy: across 42 studies and 63 benchmarks, the median reported
factor was 6, with a range of 1.3 to 100 [@adesiji2026benchmarking]. PBO
studies with people can report the same number relative to manual or random
search, which makes results comparable across tasks (inference).

**Simulated users.** Test the assumptions of a simulated user on at least a few
real people. Simulated agents agreed with the people they stood in for on only
about 50% of choices [@schoinas2025evaluating] (@fig-syn-agreement), and
language-model simulations of users recover population-level rankings almost
perfectly while agreeing with individuals at a Kendall $\tau$ of only 0.11 to
0.22, against 0.57 for the agreement between people's own ratings and their own
rankings, in a 2026 preprint [@kirk2026prism]. A result obtained with a
simulated user is a result about that simulation until it has been checked on
people (@sec-sw-simulated-users).

::: {.table #tbl-rec-report title="What an evaluation of a preferential optimization system should report (inference)."}
| Item | What to report | Why |
|---|---|---|
| Baselines | manual tuning, random or coarse grid search, a linear utility model | positive results mostly come from weak comparisons (@sec-hci-baselines) |
| Setting | dimension, noise model, definition of regret, software and version, number of repetitions | rankings of methods flip with these (@sec-syn-acquisition) |
| Decision maker | a person or a simulated user; what preference information was collected; how stopping was decided | the norms of interactive multi-objective optimization (@sec-econ-moo) |
| Cost | an acceleration factor relative to manual or random search, in comparisons and in minutes | the typical benefit is lower cost, not a better result |
| Noise audit | agreement on repeated pairs at several lags; the false-preference rate on placebo pairs | separates noise from drift (@fig-rec-lagged-repeats) |
| Influence | the shift in preference during the session, against a group with random or balanced query order | the output is an estimate and an intervention (@sec-syn-intervention) |
| Delayed outcome | an unframed retest a week later, and, where possible, the result in actual use | a choice in the session is not the same as satisfaction in use (@sec-jdm) |
| Data | individual-level comparisons with timestamps, presentation order, response times, and repeated pairs | no such public data set exists (@sec-sw-datasets) |
:::

## Ethics and legitimacy {#sec-rec-ethics}

A system that learns a preference while shaping it needs rules about how much
shaping is acceptable. @sec-syn-legitimacy set out the candidate conditions;
@tbl-rec-ethics turns them into practice.

::: {.table #tbl-rec-ethics title="Recommendations for ethics and legitimacy."}
| Recommendation | Basis | Depends on the GP and PBO framework? |
|---|---|---|
| Report the shift in preference within the session as well as regret, with a control group that receives a random or balanced query order | @dean2022preference; @carroll2022estimating | no |
| Audit whether the objective and the stopping rule reward the user for becoming predictable; label proposals, show the history accurately, and allow a reset | @carroll2023characterizing; @williams2025targeted | no |
| Before the session, record the person's attitude to being changed; after it, retest after a delay and without framing; for morally weighty decisions, map the options and their trade-offs rather than output a decision | @pettigrew2023nudging; @kanwal2026constructive (workshop paper); @sec-philosophy | no |
:::

**Measure the shift, not only the regret.** Because low regret can be achieved
trivially by moving the user's preferences (@sec-syn-disciplines), regret alone
cannot certify a system that can influence its user. Measuring how far
preferences moved during the session, against a control group whose queries
were random or balanced, makes the influence visible, and the shift can then
be bounded [@carroll2022estimating].

**Audit the incentives.** Since learners optimized on user feedback learn to
target the users easiest to influence (@sec-syn-legitimacy-need), a team can
ask whether its acquisition function and stopping rule reward answers becoming
more predictable; labeling proposals, showing the history accurately, and
allowing a reset keep the influence in view. A system that chooses which
candidates to show and also benefits from particular outcomes is in the
position of a sender in the economics of persuasion, whose preferences differ
from the user's (@sec-econ-information).

**Before and after.** Recording the person's attitude before the session and
retesting after a delay without the system gives both the earlier and the
changed self a say (inference; @sec-syn-legitimacy). For decisions with moral
weight, the evidence on autonomy argues for a system that lays out the options
and trade-offs and leaves the decision to the person (@sec-phil-autonomy).

For systems deployed to consumers, these steps also serve compliance. Article
25 of the EU Digital Services Act prohibits providers of online platforms from
designing interfaces that materially distort or impair users' ability to make
free and informed decisions [@union2022regulation]. An acquisition function
that systematically reinforces the current favorite might fall within that
wording, and logging a balanced query order and ending with an explicit
endorsement check serve scientific validity and that requirement at once
(inference; this is our reading of the text, not legal advice; @sec-soc-law).

## When a simpler method should win {#sec-rec-simpler}

Random search learns nothing from its own results, and in several settings
nothing beats it by much (@sec-optimizing-the-unknown). @tbl-rec-simpler lists
the situations in which a simpler method should be the default or a required
baseline, and whether the advice comes from the problem itself, from direct
evidence, or from inside the Gaussian process and PBO framework.

::: {.table #tbl-rec-simpler title="When a simpler method should be the default or a required baseline."}
| Situation | Default or baseline | Why | Source of the advice |
|---|---|---|---|
| Few parameters (about 4 to 6), each evaluation must be felt with the body, the optimum may be flat | manual self-tuning | self-tuning with a thumbstick reduced metabolic cost by 16.6%, relative to walking with the exoskeleton in a zero-torque condition, in about 11 minutes [@schafer2026user]; self-tuning of ankle torque converged in 105 seconds within a trial [@ingraham2022role]; for 2 of 3 amputees, the self-selected setting differed between trials on the same day [@diaz2026user] | fit to the problem, and evidence |
| A flat optimum, candidates hard to tell apart | a coarse grid or discrete candidates | three users of a discrete version of PBO for a prosthesis recognized their final setting in 93% of validation trials, the one user of a continuous version in 67% [@taddei2026bayesian] (preprint) | fit to the problem |
| Attributes can be stated explicitly, utility roughly linear and additive | a linear utility model or adaptive conjoint analysis | adaptive Bayesian question selection with response error is a mature method [@saure2019ellipsoidal], with a documented estimation bias for utility-balanced questions [@hauser2005impact]; a linear model reached the state of the art on high-dimensional benchmarks [@doumont2026we] | fit to the problem; whether a linear model suffices for human pairwise data is untested |
| Nominal dimension 20 or more | a low-dimensional representation (a generative latent space, a latent space that respects feasibility) | a 5-dimensional feasibility-aware latent space gave higher shape similarity and more feasible suggestions than 9 raw parameters [@owaki2026feasibility] (preprint) | fit to the problem |
| Several measurable objectives, with anchoring and loss framing to avoid | an interactive multi-objective method, for example NAUTILUS, which starts from a dominated solution and improves every objective at each step | @miettinen2010nautilus; anchoring is stronger when deciding on someone else's behalf [@halstead2026multiobjective] | fit to the problem |
| Collecting preference data with a strong pretrained prior, or with very noisy data | random on-policy sampling | in online direct preference optimization of language models, active selection improved the proxy win rate only negligibly over random selection [@oh2026random] (workshop paper); with noisy data, random search with large batches can be a good choice [@siivola2021preferential] | evidence |
| Ideation, when goals have not yet formed | tools for divergence, or archives of diverse solutions | an AI image generator during ideation led to more fixation and fewer, less diverse, and less original ideas ($N = 60$) [@wadinambiarachchi2024effects]; quality-diversity search with human feedback returns diverse, high-quality archives [@ding2024quality] | fit to the problem |
| The goal is a conclusion about a population | a design that mixes in random queries | for population parameters, adaptive Bayesian designs were consistently less precise than random designs [@gibbard2025optimal] | evidence |
| Only perceptual comparison possible, no numerical objective, low dimension, tens to about 200 comparisons, manual search infeasible (for example, the parameters cannot be manipulated directly) | PBO with qEUBO and `PairwiseGP` | the best-maintained implementation; positive human studies exist in this range, but the controls were mostly sliders or random queries | the choice of implementation comes from the framework and the software ecosystem |
:::

The case studies work several of these situations end to end: the exoskeleton
case re-enacts the first row (@sec-cs-exo-self-tuning), the classifier case
shows random search holding its own against Bayesian optimization
(@sec-cs-classifier-race), and the photo case (@sec-cs-photo) is an instance of
the last row.

The overall verdict, which comes from the evidence and not from the framework,
is that in studies with people, an advantage of PBO over strong simple
baselines has not been shown directly. New studies should therefore make
manual tuning, random or coarse grid search, and a linear utility model
required controls; comparing only with sliders or with another PBO variant
cannot answer whether PBO is needed at all.

## Settled, contested, missing {#sec-rec-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** In exoskeleton tuning with few parameters, self-tuning reaches
metabolic savings of the same order as algorithmic tuning. For population-level
parameters, adaptive Bayesian designs are less precise than random ones. A
forced choice cannot distinguish indifference, incompleteness, and noise. A
disconnected comparison graph leaves relative utilities between its components
undetermined.

**Contested.** Whether random selection is hard to beat outside the online
preference optimization of language models (the main evidence is a workshop
paper). Whether the Laplace approximation's errors matter at human noise
levels. Whether dimension-scaled lengthscale priors repair the default model
under pairwise feedback.

**Missing.** A test of any of these recommendations as a whole, in particular
of the instrumented session and the stopping rule. A study in which PBO beats
manual tuning, random search, and a linear utility model on a preregistered
endpoint. A measured share of "can't compare" answers in design tasks. A
public data set with which these recommendations could be checked offline.
:::

## Exercises {#sec-rec-exercises}

::: {.exercise #exr-rec-hearing}
A team wants to personalize 12 parameters of a hearing aid's sound processing
for each user, in sessions of about 40 comparisons spread over two clinic
visits. Work through @alg-rec-three-questions. What would you build, what
baselines would you require, and what would you add to the session?

::: {.solution}
*Question 1.* Measurable quantities such as speech intelligibility do not
capture what the user prefers to hear; they can serve as constraints, and the
comparisons carry the preference. *Question 2.* Twelve exposed parameters
exceed the guideline of about 10, and 40 comparisons carry at most 40 bits, so
reduce what the user faces first, for example to a few perceptual directions
built from earlier users' data, or to a short list of discrete presets.
*Question 3.* With a few directions, a person adjusting sliders may do as
well; self-adjustment is the first baseline, alongside random presets and a
linear utility model. If PBO is still the choice, instrument it as in
@alg-rec-session; the two visits make repeated pairs and an unframed retest
across visits cheap, and they also capture adaptation to the device
(@sec-rec-nonstationary).
:::
:::

::: {.exercise #exr-rec-components}
Six designs have been compared in the pairs (1, 2), (3, 4), (5, 6), and (1, 3).
How many connected components does the comparison graph have, what is its
algebraic connectivity, and which kind of query would you ask next? Why does
this matter less for a model with a strong prior?

::: {.solution}
The edges join {1, 2, 3, 4} into one component, and {5, 6} form a second, so
there are two components and the algebraic connectivity, the second-smallest
eigenvalue of the graph Laplacian, is 0. No answer so far says how 5 and 6
compare with the other four; any query that pairs 5 or 6 with one of 1 to 4
connects the graph. A Gaussian process prior with a long lengthscale links
utilities of nearby designs even without comparisons, so the posterior is still
well defined, but the offset between the two groups is then set by the prior
rather than by the person (@sec-obs-graphs).
:::
:::

::: {.exercise #exr-rec-stop}
After 25 comparisons the posterior has concentrated on one design. Repeated
pairs agree 90% of the time at a lag of 2 answers and 60% at a lag of 20. What
do you conclude, and what do you do next?

::: {.solution}
Agreement that falls with the lag means the answers are changing: either the
utility is drifting, or earlier choices made some options temporarily more
attractive (@fig-rec-lagged-repeats). Either way the concentrated posterior may
reflect a moving target, or the system's own influence, rather than a stable
preference, so do not stop on it (@sec-rec-stopping). Check the answers to the
random share of queries against the model's predictions, retest the incumbent
without framing against designs rejected early, and schedule a delayed
retest; in a study, only a comparison with a group that received a random
query order tells drift from influence.
:::
:::

## Further reading {#further-reading .unnumbered}

- @schafer2026user is the clearest demonstration that a strong simple baseline
  can match algorithmic tuning with a person in the loop.
- @keswani2024pros states the premises on which active preference elicitation
  depends and shows what happens when they fail.
- @omahony2017evolution reviews how sensory science learned to handle forced
  choice, "no preference" answers, and placebo pairs, decades before PBO.
- @dean2022preference and @carroll2022estimating show why regret is not enough
  when the system can move the preference it is learning.
- @adesiji2026benchmarking sets out the acceleration and enhancement factors
  that make results comparable across tasks.
