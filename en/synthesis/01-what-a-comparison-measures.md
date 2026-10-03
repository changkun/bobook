---
status: done
synopsis: "Where the field's difficulty now lies, which results from other disciplines survived scrutiny, and what one comparison measures: a stable preference, structured evaluation noise, change caused by the query, and answers with no preference behind them. A preferential optimization session is both an estimate and an intervention."
---

# What a Comparison Measures {#sec-what-comparisons-measure}

Preferential Bayesian optimization (PBO) looks for the design a person likes
best by asking them to compare options. A statistical model reads each answer
as a noisy comparison of an unobserved number attached to every design, the
person's **utility**: by default a Gaussian process prior (a distribution over
smooth functions) on the utility function, and a probit link that turns the
difference between two utilities into a choice probability (@sec-comparisons,
@sec-gp-preference). An **acquisition function** picks the next pair
(@sec-pbo), and after a few dozen to a couple of hundred answers the design
with the highest estimated utility is the result.

That model assumes one fixed utility function and noise of constant size.
This chapter gathers what the earlier parts found about that assumption to
answer a plain question: what does one answer to "which do you prefer?"
measure? Our answer is four things at once, so a PBO session is both an
estimate of a preference and an intervention on it. A sentence that is our own
inference ends with "(inference)".

## The bottleneck moved {#sec-syn-bottleneck}

Between 2017 and 2026, most of the field's effort went into acquisition
functions and regret theory, and both advanced. The bottleneck has since moved
from algorithms to measurement: what a single comparison measures, how answers
should be modeled, and what asking does to the person who answers. The largest
errors now sit in the **observation model**, the part of the model that says
how a person's answer arises from their utility, and in how methods are
evaluated against real people.

### Acquisition: from heuristics to decision theory {#sec-syn-acquisition}

The acquisition rules common from 2017 to 2021 adapted scalar Bayesian
optimization (@sec-acqf-early); four independent groups reported that one of
them, modified expected improvement, stalls (@sec-acqf-failures), and on a
constructed instance batch expected improvement is not asymptotically
consistent: even unlimited queries need not find the best option
[@astudillo2023qeubob]. The rule that replaced these heuristics is **EUBO**,
the expected utility of the best option, which scores a pair by the expected
utility of whichever member the person would pick (@sec-eubo); qEUBO is its
multi-option form. EUBO is one-step Bayes optimal, the best possible choice if
only one question remained, a property with an older root in decision analysis:
the optimal set of options to recommend is also the myopically optimal set to
ask about [@viappiani2020equivalence]. Its flaws were documented in 2026, in
two preprints: its queries collapse toward the estimated maximum, and its pairs
tend to form isolated pieces of the comparison graph, which leaves the Hessian
of the Laplace approximation rank deficient [@wu2026knowledge;
@shao2026adaptive]. @sec-pbo-failure-modes tells both stories, and
@sec-pbo-dims shows a similar collapse in a recorded simulation.

Rankings of acquisition functions flip with the metric, the noise model, and
the inference method: in the paper that introduced the optimistic algorithm
POP-BO, qEUBO reported a slightly better final solution while its cumulative
regret, the summed shortfall of the options shown along the way (@sec-regret),
was more than 2.5 times higher [@xu2024principledb]. qEUBO with BoTorch's
`PairwiseGP` model is the best-maintained default implementation
(@sec-obs-pairwisegp), a standing that comes from the software ecosystem, not
from comparisons with simpler methods.

### Theory: mature, conditional, and only upper bounds {#sec-syn-theory}

Regret theory for preference feedback matured, but every result carries
conditions, and all are upper bounds: guarantees that regret grows no faster
than some rate in the number of queries $T$. Two quantities recur. $\gamma_T$,
the maximum information gain of the kernel, measures how much $T$ noisy
observations can reveal about a function drawn from the Gaussian process prior
(@sec-gp-information-gain). $\kappa$ is an upper bound on the inverse slope of
the link function, which becomes large when choice probabilities saturate near
0 or 1 (@sec-theory-link-slope). The sequential algorithms MaxMinLCB and PF-TS
have preference-probability regret $\tilde O(\gamma_T \sqrt{T})$, where
$\tilde O$ ignores logarithmic factors, with constants that grow with $\kappa$
[@pasztor2024bandits; @lazzaro2026finiteb], and a batched algorithm does
better only under extra conditions [@kayal2025bayesian] (@sec-theory-rates).
There is no kernelized lower bound under a logistic or
probit link (@sec-theory-lower). And the theory covers elimination, optimistic,
and Thompson algorithms on frequentist kernel estimators, while practice runs
a Laplace posterior with EUBO, for which there are only one-step optimality and
finite-domain consistency results; nothing connects the two (inference).

Whether a comparison costs more than a number depends on the goal. For the
regret of a single fixed utility, the upper bounds in the finite-arm, linear,
and kernel settings show that pairwise comparisons are not more expensive in
order. For identifying a heterogeneous population, pairwise data are the
weakest feedback: a model fitted to binary comparisons implicitly aggregates by
Borda count, which scores each option by its average chance of beating the
others [@siththaranjan2024distributional], and rankings of at least three
options are needed to identify latent user types [@chidambaram2026direct].

### The observation model is the main bottleneck {#sec-syn-observation}

The probit link with constant noise remains the default, and its extensions
mostly come from one research group with one evaluation each
(@sec-observation-models). Human answers depart a long way from "a fixed
utility plus noise of constant size". When 20 people with design training
judged the same 600 pairs of generated interfaces, their agreement was 0.25 on
Krippendorff's $\alpha$, a chance-corrected agreement score on which 0 is
chance and 1 is perfect agreement, in a 2026 preprint [@peng2026efficient].
Among 35 chemists, Fleiss' kappa, a score of the same kind, was 0.40 and 0.32
in two rounds [@choung2023extracting]. The same moral pairwise questions,
repeated within and across sessions, had on average 6% to 20% of answers flip
[@keswani2026moral]. In retinal-implant optimization, sighted participants and
the simulated agents meant to stand in for them agreed on only about 50% of
choices [@schoinas2025evaluating]. And in a meta-analysis of the stability of
risk preference, the estimated reliability (the correlation between two
measurements of the same people taken close together) was 0.61 for
self-reported propensity to take risks and only 0.25 for behavioral measures,
standardized tasks such as choices between lotteries [@bagaini2025systematic].
A pairwise choice is a behavioral measure, so a few stated-preference
questions alongside the comparisons might carry the stable component
(inference). @fig-syn-agreement puts these numbers side by side.

```{figure}
//| figure: syn-agreement
//| label: fig-syn-agreement
//| fig-cap: "How consistent human answers are, in the studies cited in this section. Each row is a different statistic on its own scale, stretched so that chance (or no stable signal) sits at the left end and perfect consistency at the right; the rows can be read for where they fall between those ends, not compared with each other in size. The flip rate is drawn reversed, since fewer flips mean more consistency. Choose a row to read its study and statistic."
```

Some things to try:

- **Choose the flips row.** A flip rate of 6% to 20% means the same person
  repeats 80% to 94% of their answers; a person answering by coin flip would
  repeat 50%.
- **Compare the two reliability rows.** They come from one meta-analysis and
  use one statistic, so they can be compared: self-reports (0.61) are more
  than twice as reliable as behavioral measures (0.25).

Simulation studies of acquisition functions assume an observation model, so
they cannot measure the error that comes from that model being wrong;
improving the likelihood is more likely to change research conclusions than
another acquisition function (inference). Of the signals that cost the person
no extra effort, response times [@shvartsman2024response; @li2024enhancing]
and stated confidence [@zhang2026vibrotactile] have improved models on real
human data, so far in offline fits rather than in closed-loop optimization.

### Small samples, weak evaluation {#sec-syn-evidence}

Studies of pure preference feedback in interactive design mostly enroll 6 to 60
people, applied studies elsewhere 1 to 35 evaluators with a median below 10,
and validation is almost always internal: the same person later chooses the
result (@sec-hci, @sec-sci-evidence). Benchmarks are weak in the same
direction: scalar test functions in 1 to 8 dimensions, at least six noise
models and five definitions of regret, no shared benchmark, and no public data
set of individual-level pairwise judgments (@sec-software-evaluation). Even an
expert's own cost function can miss their choices: in a robot-commissioning
study with a single expert operator, reported in a 2025 preprint, a cost
function the expert had designed did not fully capture the expert's own
choices, even after its weights were refitted to the expert's highest-rated
experiments [@witte2025capture]. Synthetic decision makers defined by a hidden
cost function may therefore overstate how well methods work on real people
(inference).

### Simpler methods often do as well {#sec-syn-simpler}

In several settings, simpler methods perform comparably: people tuning an
exoskeleton themselves reached, in about 11 minutes, metabolic savings of the
same order as algorithmic tuning [@schafer2026user]; a spherical input mapping
with Bayesian linear regression reached the state of the art on
high-dimensional benchmarks [@doumont2026we]; and random selection was hard to
beat in online direct preference optimization of language models, in a 2026
workshop paper [@oh2026random], while the authors of the amortized optimizer
PABBO write that a random strategy often beats some Gaussian process baselines
[@zhang2025pabbob]. Theory points to where PBO's advantage
must come from: when ranking items actively from noisy comparisons, parametric
assumptions such as Bradley-Terry or Thurstone buy at most a logarithmic gain
[@heckel2019active], so PBO's sample efficiency should come mainly from the
kernel sharing information between neighboring designs, not from the link
function (inference).

The setting in which PBO remains supported by evidence is therefore narrow:
options that can be judged only by perceptual comparison, no numerical
objective, a dimension already low after the representation has been designed,
and a budget of tens to about two hundred comparisons. @sec-rec-simpler turns
this into advice, including the baselines a study needs before it can claim an
advantage for PBO.

::: {.keyidea title="Where the error is"}
Acquisition functions now have a decision-theoretic footing and regret bounds
have caught up with scalar feedback, but the error that matters most comes from
what the model assumes about the person, which simulations with a known
observation model cannot see (inference).
:::

## What the disciplines changed {#sec-syn-disciplines}

@sec-part-perspectives collected results about preference from psychology,
economics, neuroscience, and other fields. Since 2017, some failed in
preregistered replications (studies whose hypotheses and analyses are fixed
before data collection) or shrank in meta-analyses corrected for publication
bias; others were strengthened and gained mechanisms that can be written into a
likelihood. @tbl-syn-weakened and @tbl-syn-strengthened list only the results
with direct consequences for preferential optimization; the last column of
each is our inference. The effect sizes $d$ and $g$ express a difference
between conditions in standard deviations, so $d = 0.06$ is a twentieth of a
standard deviation and $d = 0.40$ is a moderate effect.

::: {.table #tbl-syn-weakened title="Results about preference that weakened after 2017, and what that means for PBO (last column: inference)."}
| Result | Status | Key evidence | What it means for PBO (inference) |
|---|---|---|---|
| Ego depletion and domain-general decision fatigue: deciding draws on a limited resource | weakened to near zero | 36 laboratories, $d = 0.06$ [@vohs2021multisite]; 231,076 triage calls show no decision fatigue [@andersson2025no] | Do not set session length on this basis; measure fatigue within the task |
| Induced-compliance dissonance: people change attitudes to match what they were induced to do | not replicated | 39 laboratories, $N = 4{,}898$ [@vaidis2024multilab] | Explain post-choice preference change by revaluation, not by classic dissonance |
| Moral licensing within a person: a good deed licenses a later lapse | near zero after bias correction; only a reputational effect when observed remains | $g$ between $-0.08$ and $-0.02$ [@rotella2026observation] | Do not model compensation between successive answers in private sessions |
| Mood as information: current mood is read as evidence about the options | much weakened | most of nine replications not significant [@yap2017effect] | Do not schedule exploration by mood |
| Value discrimination worsens near the optimum | fails for value judgments; holds for perceptual parameters | decisions between high-value options are faster and more accurate [@shevlin2022high] | Value comparisons may become more accurate near convergence; perceptual parameters keep a just-noticeable-difference floor |
| A loss-aversion coefficient of about 2.25 | value unstable | meta-analytic mean 1.955 [@brown2024meta]; about 1.07 when symmetric and unsorted [@yechiam2025loss] | Do not fix an asymmetric likelihood around the current best |
| The attraction (decoy) effect: adding a dominated option boosts its neighbor | limited to attributes shown as numbers | @frederick2014limits | The mechanism is weak in perceptual design |
:::

::: {.table #tbl-syn-strengthened title="Results about preference that strengthened or appeared after 2017, and what that means for PBO (last column: inference)."}
| Result | Status | Key evidence | What it means for PBO (inference) |
|---|---|---|---|
| Choice-induced preference change: choosing an option raises its value for the chooser | strengthened | $d = 0.40$ [@enisman2021choice]; each choice adds about \$0.18 to the chosen item [@zylberberg2024value] | Add a choice-history term to the likelihood; the most-compared current best is affected most |
| Range normalization: values are coded relative to the range on offer | strengthened | @bavard2018reference; @bavard2023functional | The utility's scale is session-relative; recalibrate before reusing it across sessions |
| Efficient coding of value noise: precision goes to the values a person expects to see | new and strengthened | @polania2019efficient; @pratcarrabin2022efficient | Noise scales with the values presented, so fixed noise misreads late-session precision |
| Random utility consistency: repeated choices behave as if drawn from a distribution over utilities | strengthened for most people; part of the observed intransitivity is genuine | most of 141 participants satisfy random utility [@mccausland2020testing]; with response times separating noise from preference, 19.24% and 39.58% of the transitivity violations in two data sets are genuine, and the rest could be either [@alosferrer2023identifying]; the noise specification changes the inference [@bhatia2017noisy] | A scalar utility plus noise works for most people, but the link and noise form are substantive modeling choices |
| Incomplete preferences and deliberate randomization | strengthened, with limits | 40% to 50% of participants report incompleteness when allowed, on average 3.3 of 50 comparisons [@nielsen2026revealed]; about half choose inconsistently with complete preferences plus certainty independence, an axiom about mixing options with a sure outcome [@cettolin2019revealed]; most choose to randomize [@agranov2025ranges] | Offer a "can't compare" answer and model it separately from ties and noise |
| Information per comparison | at most one bit, but the error rate is of the same order as with numerical ratings | @shah2016estimation | The structure of the comparison graph, not the bits per query, governs the error |
:::

Two of these studies are working papers [@alosferrer2023identifying;
@nielsen2026revealed]. The rows are developed in @sec-part-perspectives:
fatigue in @sec-jdm-fatigue, moral licensing in @sec-sp-moral, mood in
@sec-sp-emotion, loss aversion in @sec-econ-behavioral, context effects in
@sec-jdm-context, choice-induced change in @sec-jdm-choice, range normalization
in @sec-jdm-range, efficient coding in @sec-neuro-coding, random utility in
@sec-jdm-rum, and incomplete preferences in @sec-econ-stochastic and
@sec-nat-incomplete.

The changes follow a pattern (inference). What weakened were mostly arguments
for session-design rules (set session length by decision fatigue, schedule
queries by mood) and for fixed numbers for directional biases (a loss-aversion
coefficient, compensation in moral licensing). What strengthened were
mechanisms with a magnitude that map onto parts of a likelihood. The evidence
therefore supports a few mechanisms whose sign and size are estimated in each
task, not one likelihood term for every documented bias.

**Formal tools PBO has not yet used.** Five families of results can be applied
to PBO, and as of September 2026 we found no PBO paper that uses them:

1. *comparison-graph theory*, in which the comparison graph's Laplacian
   spectrum and effective resistances govern the error of pairwise estimates
   [@shah2016estimation; @hendrickx2019graph], and the **Hodge
   decomposition** splits comparison data into a part a scalar utility can
   explain and a cyclic part none can [@jiang2011statistical;
   @strang2022network] (@sec-nat-graph-spectra, @sec-nat-hodge);
2. *performative prediction*, which separates learning a target from steering
   it [@perdomo2020performative; @hardt2025performative] (@sec-soc-measuring);
3. *trust regions on induced preference shift* [@carroll2022estimating],
   needed because when preferences move toward what a system shows, low
   regret can be achieved trivially by moving them [@dean2022preference];
4. *behavioral welfare economics*, which calls one option better for a
   person's welfare only if the other is never chosen over it in any frame
   [@bernheim2009revealed] and measures welfare losses from misunderstood
   consequences [@ambuehl2022evaluating];
5. *multi-utility representations of incomplete preferences*, in which an
   option counts as better only if every utility in a set agrees
   [@evren2011multi], with identification results that distinguish
   indifference, indecisiveness, and experimentation [@ok2022indifference].

The first and last apply to PBO logs and posteriors without new theory, so
they are the place to start (inference). Relatedly, fitting one Bradley-Terry
utility to several people's pooled comparisons is itself an aggregation rule,
and it violates Pareto optimality and pairwise majority consistency
[@ge2024axioms].

## Found or made? {#sec-syn-found-or-made}

Behind these results lies an old question: do repeated comparisons *find* a
preference that was there before, or do they partly *make* it? The
**constructive view** holds that preferences are built during elicitation, so
the measurement changes what it measures; the **stable view** holds that there
is an underlying preference that careful measurement can recover. Preferential
optimization takes a side whether its users notice or not. The evidence since
2017 leaves part of each view standing and supports neither extreme.

### What each view keeps {#sec-syn-construction}

The constructive view keeps its central claim: elicitation changes the object
it measures, by a considerable amount and through mechanisms that can be
modeled. Choosing an option raises its value (@tbl-syn-strengthened); for 59
people rating 55 morphed dog images, individual models with a learning
component explained on average 17% more variance for the real presentation
order than for a simulated random order [@brielmann2024modelling]; and forced
choice manufactures some inconsistency that would otherwise show up as
hesitation [@costagomes2022choice]. It loses some supporting arguments: the
advantage of unconscious thought was not found in a meta-analysis and large
replication [@nieuwenstein2015making], the induced-compliance and mood effects
failed or shrank (@tbl-syn-weakened), and Ruth Chang's work on hard choices,
sometimes read as calling a forced choice between options "on a par" a
category error, treats such a choice as an occasion for commitment
[@chang2024s].

The stable view keeps the claim that there is a component worth estimating.
Most people's repeated choices satisfy random utility
[@mccausland2020testing]; behavioral biases are nearly unchanged at the
population level over three years, in a working paper
[@stango2024behavioral]; and some anomalies in risky choice look like errors in
valuing complex options rather than preferences [@oprea2024decisions], though
that conclusion is disputed. It loses its strongest inference, that iterated
querying converges to the true underlying preference. Plott's "discovered
preference" hypothesis has been observed only after forced trading or after
people reflected on axioms they themselves endorse [@nielsen2022choices;
@engelmann2010reconsidering], not as a natural outcome of repeated comparison,
and in the field PBO loops often end before they converge, most evaluation
sequences of a three-month deployment stopping at the first iteration
[@ou2022human] (@sec-rec-stopping).

### The hierarchical view {#sec-syn-hierarchical}

One position in this debate, which we call the **hierarchical view**, holds
that preferences over ultimate goals (comfort, safety, beauty) always exist,
while preferences over intermediate outcomes, such as a particular parameter
setting, may have to be elicited. As a description of *where* stability lives,
the evidence largely supports it: stability concentrates at the broad,
abstract, self-reported level, and in a longitudinal study in early
adolescence, 75% of respondents had value hierarchies that correlated at least
0.85 across two years [@vecchione2020stability].

As a prediction about PBO, the view fares worse. The value computed during a
choice is relative to the current goal: goal congruence explains choices and
their neural correlates better than reward value [@fromer2019goal], so a
stable ultimate goal need not become a stable utility over pairwise
comparisons (inference). Iterated querying has not been seen to converge
(@sec-syn-construction), and because stability is measured with self-reports
and construction with choices, @sec-open-preference describes the study that
would test both levels on one design task. Nor does rational inattention, the
theory that people spend costly attention optimally, single out the logit link
as the view's defenders sometimes claim: it yields the logit only under a
Shannon-entropy cost of information, and it can generate any additive random
utility model [@fosgerau2020discrete].

### Preference scaffolding {#sec-syn-scaffolding}

A second position, **preference scaffolding**, holds that PBO is not a readout
of a fixed utility function but a structure that helps a person form a
preference, so that evaluation should cover the exploration process and the
person's reflective endorsement of the result, not only the final design. The
core holds up, with three qualifications. Not all inconsistency is preference
formation: in repeated discrete choice tasks, instability concentrates on
options close in utility and on hard tasks, and the underlying preferences of
unstable respondents do not differ from those of stable ones
[@fraser2021preference]. Scaffolding is not neutral: when preferences can be
influenced, each of the eight notions of alignment compared in one analysis
either errs toward undesirable influence or is overly risk-averse
[@carroll2024aib], so helping and steering have to be told apart by conditions
the objective does not supply (@sec-syn-intervention). And endorsement after
the fact cannot by itself justify a change the system caused, because the
changed person may endorse the result only because their values were changed
[@pettigrew2023nudging].

One argument offered for scaffolding does not hold. Active inference, a theory
in which action minimizes surprise relative to preferred observations, is
sometimes said to dissolve the dispute between construction and revelation,
but it only re-encodes it: in Bayesian optimization it reduces to weighted
information-gain acquisition functions whose weights still need tuning
[@millidge2021whence; @li2026curiosity] (@sec-neuro-active-inference). The
most direct support is a 2025 measurement: when Bayesian optimization led the
search, designers reached better results but reported significantly less sense
of agency, and collaboration through explicit constraints matched
collaboration through natural language in performance while giving more agency
[@niwa2025cooperative]. That supports a scaffold led by the designer, not
preference shaping led by the system.

## Four components of a comparison {#sec-syn-components}

Putting the pieces together, we propose that a single pairwise answer mixes
four components, whose proportions are moderated by two further factors
(@tbl-syn-components; inference). No single paper has tested this
decomposition, and the proportions change with familiarity with the domain,
the similarity of the options, and the length of the session, so they have to
be estimated in each application rather than assumed.

::: {.table #tbl-syn-components title="Four components of a pairwise answer and two moderating factors. The decomposition is our inference; the evidence column says where each piece of support comes from."}
| Component | What it is | Evidence, and where it comes from | Representation in the model |
|---|---|---|---|
| Stable preference | an estimable latent utility | tests of random utility, stability of biases, value hierarchies; from lotteries, money choices, and self-reports, not from design comparisons | the latent utility of a Gaussian process or other surrogate |
| Structured evaluation noise | noise that varies with difficulty, similarity, protocol, ambiguity of the representation, and the distribution of values presented | early versus late noise [@shen2025early]; efficient coding; range normalization; mechanisms supported by experiments, only offline fits in PBO | heteroscedastic noise; a joint likelihood with response times; separate noise scales per feedback type [@ghosal2023effect] |
| Query-induced change | revaluation after choosing, anchoring on the system's proposals, familiarity and fatigue | choice-induced preference change; biases grow after interacting with a biased AI [@glickman2025human]; magnitudes from free-choice paradigms, never measured in design comparisons | a choice-history term; a drift term; balanced query order as a control |
| Incomplete or deliberately randomized answers | the person cannot or will not compare, or deliberately randomizes | experiments on incomplete and randomized preferences; from lotteries and money choices; the share in design comparisons is unknown | a "can't compare" answer; a mixture parameter; a multi-utility representation |
| Moderator 1: how much taste is shared | the informativeness of a population prior varies by domain | shared taste is high for faces and landscapes and low for architecture and artworks; where landscapes and exterior architecture were compared within the same people, the judgments were equally reliable [@vessel2018stronger]; individual models against the group average, median correlation 0.65 against 0.01 [@brielmann2024modelling] | the weight of a population prior, as a domain or personal parameter |
| Moderator 2: legitimacy conditions | normative conditions that separate helping to form a preference from steering it | many candidate conditions, no consensus, none tested in PBO | not part of the likelihood; part of the experimental design and the report |
:::

The first component is the one the default model assumes. The second says that
noise is not one number: it grows with the ambiguity of how the options are
represented (early noise) and with time pressure (late noise) [@shen2025early],
depends on the values a person has recently seen [@polania2019efficient;
@bavard2023functional], and near the optimum can make candidates perceptually
indistinguishable, as in a 2026 preprint on prosthesis tuning
[@taddei2026bayesian] (@tbl-rec-simpler). The third component has no place in
the default model at all. The fourth is invisible under forced choice, where a
person with no preference must still answer and the answer looks like noise.

Which component dominates depends on the domain (inference). Familiar domains
and broad tendencies come closer to noisy discovery of a stable preference;
novel, similar, multi-attribute options and long sessions come closer to
construction and drift. PBO's candidates are usually close variants of one
design, and in repeated aesthetic ratings the assimilation and contrast effects
between successive items grow with the similarity of the stimuli
[@pombo2023intrinsic], so the mix has to be measured in each application.

### A simulated session {#sec-syn-simulation}

Within a single session the four components are hard to tell apart, because
each can produce mostly consistent answers and a posterior that sharpens.
@fig-syn-four-components shows this in a simulation whose components, forms,
and numbers are illustrative choices of ours. A simulated person compares
designs on a line. Their **stable preference** has two peaks
of different heights. **Structured noise** has a standard deviation that rises
from 0.15 for very different designs toward 0.15 plus the slider value for
nearly identical ones. **Query-induced change** raises the chosen design by the
slider value after every choice and lowers the rejected one by half that;
most of it fades over the next ten answers (15% of what is left with each
answer), but the **lasting share** remains. **Incomplete answers** are a coin
flip under forced choice, or "can't compare" when it is offered. The model is
the default of @sec-pbo, with pairs chosen by EUBO or at random.

```{figure}
//| figure: syn-four-components
//| label: fig-syn-four-components
//| fig-cap: "A simulation of a model, not data. The dashed curve is the simulated person's stable preference, the magenta curve their utility at the moment (stable part plus induced change), and the blue curve the fitted utility of a Gaussian process preference model (probit, Laplace), with the next pair chosen by EUBO (never repeating a pair) or at random. Curves are centered, because comparisons fix a utility only up to an added constant (@sec-pref-identifiability). The middle panel lists the answers in order. The bottom panel retests, without the system, whether the final design (star) is still preferred to designs rejected in the first ten answers (crosses), at the end of the session and a week later, averaged over 24 simulated people; each runs under both query orders with the same random draws, which a real experiment cannot do. All parameter values are illustrative."
```

Things to try:

- **Play the default session.** The blue curve follows the magenta bump that
  grows where the person keeps choosing, not the dashed curve, and the
  acquisition order keeps the current favorite in almost every query. The
  final design wins 88% of retests at the end of the session and 78% a week
  later under the acquisition order, against 76% and 73% under the randomized
  order.
- **Set *Induced change* to 0.** Each order's two bars become equal (73% and
  73% under the acquisition order, 71% and 71% under the randomized one). Now
  raise the noise to 1: agreement falls to 69% and 65%, the same at both
  times, since noise leaves no trace of time.
- **Return the noise to 0.3 and set incomplete answers to 0.4.** Agreement
  falls toward chance (58% and 56%), again equally at both times. Tick *Offer
  "can't compare"*: the model fits only real answers, and agreement recovers
  (74% and 67%).
- **Set the stable preference to 0** (other settings at their defaults).
  There is nothing to discover, yet the acquisition order produces a confident
  fit and an end-of-session agreement of 74%, falling to 59% a week later;
  under the randomized order, 55% and 52%.
- **Return the stable preference to 0.4 and set *Lasting share* to 1.** Each
  order's bars are equal again (94% under the acquisition order, 85% under the
  randomized one): a retest cannot see change that lasts.

Two lessons follow, both statements about this model (inference). First, the
posterior concentrates whenever the answers are consistent, whatever made them
consistent, so concentration is not evidence that a preference has
stabilized, and @sec-rec-stopping does not use it as a stopping rule. Second,
no single measurement separates the components. The clearest signature of
induced change is how much agreement falls between the end of the session and
a week later, and whether it falls more under the acquisition order; delayed
agreement alone would mislead, since at the defaults the acquisition order's
final designs still win more often a week later (78% against 73%), although by
the stable preference they are no better.

### What separates the components {#sec-syn-signatures}

@tbl-syn-signatures states what each component does to the data and which
measurement isolates it. The measurements are cheap additions to an ordinary
session, and @sec-open-decisive turns them into one experiment.

::: {.table #tbl-syn-signatures title="How each component shows up in the data, and the measurement that isolates it (inference)."}
| Component | Signature in the data | Measurement that isolates it |
|---|---|---|
| Stable preference | the same answers at any delay and under any query order | delayed retest agreeing with the end-of-session retest |
| Structured evaluation noise | lower consistency for hard or similar pairs, unrelated to the delay | the same pair repeated at different lags; response times |
| Query-induced change | answers that favor recently chosen designs; agreement that falls with delay, more under acquisition order; final designs close to the early favorite | randomized query order compared with acquisition order, with a retest at the end of the session and a week later |
| Incomplete or randomized answers | a false-preference rate on identical pairs; "can't compare" answers concentrated on complex pairs | placebo pairs (the same design shown twice); a "can't compare" option |
:::

## Estimate and intervention {#sec-syn-intervention}

If a comparison mixes these components, the output of a PBO session is two
things at once: an estimate of the person's preference and an intervention on
it. A study that neither randomizes the order of queries nor retests after a
delay cannot tell whether what converged was the preference or the system's
effect on the person. Choosing is known to change preferences by a
considerable amount, but **no study has directly measured query-induced change
in a PBO session**; @sec-open-decisive describes the experiment that would. Its
controls are the ones needed to judge whether a change the system caused was
legitimate, so the methodological and the normative question can be answered
in the same experiment (inference).

### Why legitimacy conditions are needed {#sec-syn-legitimacy-need}

Two results make legitimacy a technical question rather than an afterthought.
When preferences can change, standards of legitimacy cannot be read off the
objective function: each of the eight notions of alignment that
@carroll2024aib compare fails in one of two ways (@sec-syn-scaffolding). And
learners optimized on user feedback learn to target the users who are easiest
to influence [@williams2025targeted]. An acquisition function that rewards
fast convergence or a clean signal could in principle favor queries that make
the person more predictable over queries that serve their interests
(inference), so a team should audit its own acquisition function and stopping
rule for that incentive (@sec-rec-ethics).

### Candidate conditions {#sec-syn-legitimacy}

There is a set of operational candidates, though no consensus: respect the
person's meta-preferences, their preferences about how their preferences may
change [@ashton2022solutions], a workshop paper; keep influence overt, and give
the system no incentive to make the user more predictable
[@carroll2023characterizing]; let the user choose the mode of autonomy
[@fischli2026agents]; require reflective endorsement with bounded influence
that does not degrade the person's factual beliefs and keeps future options
open while the preference is uncertain [@kanwal2026constructive], a workshop
paper; and judge a change from the standpoint of the selves both before and
after it [@pettigrew2023nudging].

Combining these, the defensible target of single-user PBO is not "the latent
utility" but "the utility the user would reflectively endorse, within a
protocol of bounded influence" (inference). @alg-rec-session makes that target
operational: record the person's attitude toward having their taste changed,
measure and bound the shift against a random or balanced query order, and
retest the final design without the system's framing at the end of the
session and again at the next one.

## Settled, contested, missing {#sec-syn-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** The acquisition rules of 2017 to 2021 have documented failures,
and EUBO has a decision-theoretic justification. Regret theory for preference
feedback offers only conditional upper bounds. Domain-general decision
fatigue, ego depletion, induced-compliance dissonance, and moral licensing
within a person did not survive large replications or bias correction.
Choice-induced preference change, range normalization, efficient coding of
value noise, and random-utility consistency for most people are well
supported, and a sizable share of people report incomplete preferences when
allowed.

**Contested.** Whether EUBO's collapse and the rank-deficient Hessian cost
anything on human tasks (the evidence is from 2026 preprints). Whether complex
valuation errors explain risky-choice anomalies. How much of the inconsistency
in PBO sessions is preference formation rather than structured error around a
stable target. Whether the hierarchical view, true of self-reported values,
says anything about pairwise design judgments.

**Missing.** A direct measurement of query-induced change in a PBO session,
with randomized query order and a delayed retest (@sec-open-decisive). A test
of the four-component decomposition in design tasks. A comparison of link
functions on human data, and a model of drifting utility. A public data set of
individual-level pairwise judgments with timestamps, presentation order, and
repeated pairs. A demonstration, for or against, that acquisition functions
favor queries that make people predictable. Agreed, tested conditions for when
a change a system causes is legitimate.
:::

## Exercises {#sec-syn-exercises}

Both exercises use @fig-syn-four-components.

::: {.exercise #exr-syn-lasting}
Set the *Lasting share* slider to 1 and leave everything else at its
default. The retest bars of each query order are now equal. Has
query-induced change disappeared? What evidence, available in a real experiment,
would still reveal it?

::: {.solution}
No. Induced change no longer fades, so the person's preference a week later is
the changed one, and the retest has nothing to detect. The orders still
differ: the acquisition order's final designs are preferred more strongly (94%
against 85%) because its queries kept reinforcing one favorite, though by the
stable preference they are no better. In a real experiment the remaining
evidence is that people randomized to different query orders end with
different distributions of final designs, those of the acquisition order lying
close to each person's early favorite. Because an efficient acquisition rule
also settles early, this evidence is weaker than a drop after a delay
(inference). Whether such a lasting change is acceptable is a question of
legitimacy (@sec-syn-intervention).
:::
:::

::: {.exercise #exr-syn-concentration}
Set the stable preference to 0 and induced change to 0.12. The fitted utility
under the acquisition order has a clear peak, and the end-of-session retest
agreement is high. Explain why the model is confident although the person has
no stable preference, and what this implies for stopping rules.

::: {.solution}
The likelihood sees only which option won each comparison. Under the
acquisition order the current favorite appears in most queries, and every win
raises it further through induced change, so the answers are highly
consistent, and consistency is all the posterior measures. A stopping rule
that waits for the posterior to concentrate would stop here with confidence; a
rule that also requires repeated pairs to agree across a delay, or the final
design to survive a retest without the system, would not (@sec-rec-stopping).
:::
:::

## Further reading {#further-reading .unnumbered}

- @enisman2021choice and @zylberberg2024value establish, by meta-analysis and
  by a computational model, that choosing changes value; they are the empirical
  core of query-induced change.
- @carroll2024aib argues, by comparing eight notions of alignment, that
  alignment with influenceable preferences has no straightforward solution, and
  @pettigrew2023nudging explains why endorsement after the fact is not enough;
  read them together.
- @nielsen2026revealed (a working paper) and @ok2022indifference show how to
  elicit and model incomplete preferences rather than forcing them into ties.
- @shah2016estimation and @strang2022network are the clearest entry points to
  comparison-graph theory and the Hodge decomposition.
- @niwa2025cooperative is the most direct measurement of the trade-off between
  performance and agency when an optimizer leads a design session.
