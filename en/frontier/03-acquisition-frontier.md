---
status: done
synopsis: "How the rules for choosing queries evolved from heuristics to decision theory, the failure modes several groups found independently, the forms a query can take and the problem variants built on preferential Bayesian optimization, and why the published comparisons, each run at its own dimension and noise level, cannot simply be pooled."
---

# Acquisition, Query Forms, and Problem Extensions {#sec-acquisition-frontier}

An acquisition function is the rule that decides what to ask next
(@sec-acq-definition). In preferential Bayesian optimization (PBO) it chooses a
pair, or a set, of options to show a person, and @sec-choosing-pairs and @sec-eubo
introduced the main candidates. This chapter follows those rules through the
research record from 2017 to September 2026: where each came from, what is
proved about it, under which conditions it did well or badly, and what forms a
query can take besides "A or B?".

The design of acquisition functions went through three phases: heuristic and
information-theoretic rules from 2017 to 2021; a turn to decision theory in
2022 and 2023, when the expected utility of the best option (EUBO) and its
multi-option form qEUBO were proved one-step Bayes optimal for noise-free
answers; and, from 2024 to 2026, kernelized dueling algorithms driven by
frequentist regret (@sec-pbo-theory) alongside criticism and repair of the EUBO
default. Two findings frame everything below. The failure modes reported by
different groups agree with one another. And the empirical comparisons were run
in settings that cannot be compared with one another; at small budgets with
realistic noise, random queries sometimes keep up.

## 2017 to 2021: heuristics and information {#sec-acqf-early}

**Two rules from before 2017.** The interactive Bayesian optimization of
@brochu2007active chose as the first option the queried point with the largest
posterior mean, and as the second the point with the largest expected
improvement over it (@sec-ei); @astudillo2023qeubob later noted that qEUBO with
two options, if forced to include the current best point, reduces to exactly
this rule. *Bayesian active learning by disagreement* (BALD), proposed for
Gaussian process classifiers by @houlsby2011bayesian (a preprint), chooses the
query whose answer has the largest mutual information with the model's
parameters (@sec-mutual-information): the query on which plausible models
disagree most. The dueling information gain of @benavoli2021preferentialb
extends BALD to preferences.

**The three rules of González et al.** @gonzalez2017preferentialb proposed three
acquisition functions on the dueling space (@sec-dueling-formulation).
*Pure exploration* (PE) picks the pair whose duel outcome has the largest
variance. *Copeland expected improvement* (CEI) computes the one-step lookahead
improvement in the soft-Copeland value of the Condorcet winner, the option that
beats every other with probability above one half. *Dueling Thompson sampling*
(DTS) picks the first point to maximize the soft-Copeland score computed from
one continuous Thompson sample of the preference function (@sec-thompson), and
the second purely to explore, as the point whose duel against the first is most
uncertain. In their experiments on one- and two-dimensional functions
(@tbl-acqf-studies), DTS was consistently the best strategy, and the
dueling-bandit baseline Sparring needed about 4000 iterations to approach what
DTS reached in 200 duels.

**Duels among many, and easy questions.** SelfSparring reduced
*multi-dueling*, in which several options are compared in each round, to an
ordinary bandit problem solved by Thompson sampling; its kernel version,
KernelSelfSparring, adds a Gaussian process prior so that information is shared
between options [@sui2017multi]. @byk2019asking showed (their Theorem 1)
that the global optimum of the commonly used *volume removal* objective, which
scores a query by how much of the space of reward parameters an answer is
expected to rule out, is a trivial query of $K$ identical options. Replacing it
with information gain favors queries a person can answer with confidence, and
learned faster in simulation and in a user study. This work belongs to
preference-based reward learning with a parametric reward, not to Gaussian
process PBO (@sec-adj-pbrl).

**A burst of new rules, 2020 and 2021.** Projective PBO
[@mikkola2020projective] came with five rules for its projective queries,
among them projective expected improvement and preferential coordinate
descent. @benavoli2021preferentialb defined three rules relative to the current
winner $\vx_r$: a *dueling upper confidence bound*, the upper end of the 95%
credible interval of $f(\vx) - f(\vx_r)$; dueling Thompson sampling; and EIIG,
the logarithm of the probability of improvement plus $k$ times the dueling
information gain, with $k = 0.1$ or $0.5$. @nguyen2021top proposed *multinomial
predictive entropy search* (MPES), which they describe as the first
information-theoretic acquisition function for Bayesian optimization with
preference observations (@sec-entropy-search); it optimizes all inputs of a
query jointly, but it must enumerate the possible answers, so it suits only
small query sets. @siivola2021preferential adapted batch expected improvement
(qEI) and batch Thompson sampling to batch-winner feedback, in which the person
names the best option of a set.

**Separating two kinds of uncertainty.** @fauvel2021efficient (a preprint)
split the uncertainty about a duel's outcome into an *epistemic* part, which
more data would remove, and an *aleatoric* part, the person's own randomness,
which it would not. Their *maximally uncertain challenge* (MUC) takes as
champion the point of largest posterior mean and as challenger the point that
maximizes the epistemic variance of $\Phi(f(\vx_1) - f(\vx))$, in closed form
and with a batch version. In the same year came the first kernelized dueling
algorithm with a cumulative regret guarantee [@kirschner2021bias], whose
feedback model @sec-theory-kernelized examines.

## 2022 to 2023: EUBO and the decision-theoretic turn {#sec-acqf-eubo}

**EUBO in preference exploration.** EUBO first appeared in *Bayesian
optimization with preference exploration* (BOPE), where experiments produce
several outcomes and a decision maker's utility over those outcomes is learned
from comparisons. @lin2022preferenceb used EUBO to choose the two outcome
vectors to show the decision maker, and proved that it is the one-step Bayes
optimal preference-exploration policy (@sec-eubo-theory). Two variants address
the fact that some outcomes are not achievable: EUBO-ζ generates outcomes from
one posterior sample of the outcome model, and EUBO-f̃ compares only outcomes
that are likely achievable. On an unrestricted outcome set, EUBO tended to
over-explore outcomes that could not be achieved. The paper also concluded that
a Monte Carlo version of BALD, BALD-f̃, is a strong and fast baseline.

**qEUBO.** @astudillo2023qeubob generalized EUBO to

$$
\operatorname{qEUBO}_n(\vx_1, \dots, \vx_q) = \E_n\!\left[\max\{f(\vx_1), \dots, f(\vx_q)\}\right],
$$ {#eq-acqf-qeubo}

where $q$ is the number of options shown to the person in one query, from which
the person picks the best, and $f$ is the latent utility (written $g$ in
@sec-eubo). It is not a batch of separate queries. @sec-eubo-theory
summarized their four theorems; their conditions matter for how far they
reach:

1. **Theorem 1.** With noise-free answers, qEUBO is one-step Bayes optimal and
   equivalent to the knowledge gradient (@sec-knowledge-gradient).
2. **Theorem 2.** Under logistic (multinomial logit) noise of scale $\tau$
   (@eq-cmp-logit; the paper writes $\lambda$), the one-step value of qEUBO's
   maximizer is at least the noise-free one-step optimal value minus
   $\tau\, W\!\left((q - 1)/e\right)$, where $W$ is the Lambert W function,
   the inverse of $w \mapsto w e^w$ (@exr-acqf-lambert computes how fast the
   guarantee loosens with $q$).
3. **Theorem 3.** On a finite domain, with $q = 2$ and further technical
   conditions, the Bayesian simple regret of qEUBO, the expected gap between
   the best utility and that of the recommended option averaged over the prior
   (@sec-regret-definitions), is $o(1/n)$. Examples of sufficient conditions
   are a logistic likelihood with a prior under which, almost surely,
   $\delta \le \lvert f(\vx) - f(\vy)\rvert \le \Delta$ for all
   $\vx \neq \vy$; or a nondegenerate Gaussian process prior with a likelihood
   equal to a constant $a > 1/2$ whenever $f(\vx_1) \neq f(\vx_2)$.
4. **Theorem 4.** On some instances satisfying the same assumptions, the
   Bayesian simple regret of qEI stays above a constant $R > 0$ for every $n$:
   qEI is not asymptotically consistent.

Two readings follow (inference). EUBO's one-step optimality was first proved
in BOPE; qEUBO's contribution is the extension to noise and to $q > 2$, and
consistency on finite domains. And the conditions behind the $o(1/n)$ rate, a
finite domain with utility gaps bounded away from zero, turn the problem into
one of identifying the best of finitely many options, so the rate cannot be
compared with the rates for continuous domains in @sec-theory-rates. In
software, BoTorch 0.10.0 (February 2024) added qEUBO [@botorch2026changelog],
and its preference tutorial runs the loop with `PairwiseGP` and the analytic
EUBO [@botorch2026pairwise].

**The hallucination believer.** In the same year, @takeno2023practicalc
proposed the hallucination believer (HB) of @sec-choosing-pairs: take the
current winner as the first point of the duel, and apply expected improvement
or an upper confidence bound to a Gaussian process fitted to one sample of the
latent comparison values from the truncated posterior (a *hallucination*).
They had observed that
standard acquisition functions applied to a preference Gaussian process keep
choosing similar duels, because a duel carries little information and the
variance of the preference model hardly decreases. @ignatenko2025preference
later proposed *remaining system uncertainty*, from a minimax view of data
collection, as a performance measure that needs no ground truth.

## 2024 to 2026: noise, exact knowledge gradient, amortization {#sec-acqf-recent}

**2024 and 2025.** @ozaki2024multi used a mutual-information active-learning
acquisition function to choose preference queries in a multi-objective setting
(see @sec-acqf-query-forms). @sinaga2024anchor (first posted in 2024; its 2026
version appeared in the proceedings track of ProbML 2026) proposed a
risk-averse acquisition function that trades utility against how hard a
comparison is to answer, and showed that their risk-adjusted EUBO stays
one-step Bayes optimal up to an additive constant. POP-BO [@xu2024principledb]
and MaxMinLCB [@pasztor2024bandits] are optimistic algorithms with regret
bounds (@sec-theory-bt), and in 2025 PABBO [@zhang2025pabbob] used a pretrained transformer
policy that outputs query pairs directly.

**2026: the default under examination.** Most of the 2026 method work examined
the default pipeline, in two preprints reported with the failure modes below
[@wu2026knowledge; @shao2026adaptive]. Other work adapted classical ideas.
@erarslan2025consecutive (a preprint)
adapted max-value entropy search to a production-cost constraint under which
every comparison must involve a candidate that has already been produced.
@haltia2026elicitation (a preprint) used a cost-aware value of information to
choose between a direct evaluation and a pairwise query; they report
performance close to the convex hull of the two single-source trajectories, and
the method falls back to standard Bayesian optimization when queries are
expensive or noisy. Local PBO [@menn2026local] (a preprint) ported trust-region
search (TuRPBO) and derivative-guided local search (GIPBO, PrefSQP) to pairwise
feedback (@sec-hd-local). And PF-TS [@lazzaro2026finiteb] chooses a pair by
drawing two independent posterior samples and maximizing each against a common
anchor point.

## Documented failure modes {#sec-acqf-failures}

@sec-pbo-failure-modes previewed these failures. Here they are with the
mechanism each paper gives and the conditions under which each was seen.

**Adapted expected improvement stalls, again and again.** Four groups reported
the same phenomenon independently. González et al. found that CEI over-exploits
and that the interactive Bayesian optimization of Brochu et al. performs poorly
[@gonzalez2017preferentialb]. Fauvel and Chalk found Brochu et al.'s expected
improvement only slightly better than random, and attributed it to a frequent
pathology in which the function samples the same duel members
[@fauvel2021efficient]. Takeno et al. found that expectation propagation with
expected improvement often stalls through over-exploitation
[@takeno2023practicalc]. And Astudillo et al. found that qEI tended to stall
late in a run, on the 7-dimensional Alpine1 function with initial data that
included comparisons against a known good point, and proved the mechanism in
their Theorem 4: when the value of the incumbent, the point of largest
posterior mean, is already known fairly precisely, qEI is reluctant to include
it in a query, so it learns only how the other options compare with one
another [@astudillo2023qeubob]. Rules of this family stop testing a well-known
incumbent (inference; @sec-choosing-pairs).

**EUBO over-exploits, and its pipeline becomes ill-conditioned.** On BOPE's
unrestricted outcome set, EUBO over-explored unachievable outcomes
[@lin2022preferenceb]. In single-objective PBO, EUBO's queries collapse toward
the estimated maximum, as @sec-pbo-failure-modes reported and @sec-pbo-dims
reproduces in simulation. The source, @wu2026knowledge (a preprint), adds the
mechanism: under a Gaussian process prior and a probit likelihood, the one-step
lookahead posterior is an *extended skew-normal* distribution, a skewed
relative of the Gaussian whose mean has a closed form, and EUBO is only a lower
bound on the exact knowledge gradient, approximately equal to it only when the
probit noise goes to zero. Their test case is the two-dimensional Levy
function, and their abstract also acknowledges a case showing that the
knowledge gradient has limits in some situations. KappaSharp, also a preprint,
reports that EUBO's queries create isolated comparison pairs and a
rank-deficient Hessian [@shao2026adaptive] (@sec-obs-graphs). And POP-BO's
authors report that, on instances sampled from a Gaussian process, qEUBO's
reported solution was slightly better than theirs, but its cumulative regret
was more than 2.5 times higher [@xu2024principledb].

**Thompson sampling over-explores as the dimension grows.** DTS was the best
rule in one and two dimensions [@gonzalez2017preferentialb], but rules based on
Thompson sampling performed only modestly over 34 functions, where
KernelSelfSparring's weaker batch performance was attributed to its choosing
the members of a batch independently [@fauvel2021efficient], and Thompson
sampling over-explored on the 4- and 6-dimensional Hartmann functions
[@takeno2023practicalc]. Every study that tested a Thompson-sampling rule at
four or more dimensions found it behind another rule
(@fig-acqf-evidence-map). A plausible mechanism, which none of these
papers isolates, is that the maximum of one posterior sample tends to fall
where the posterior is most uncertain, and the share of the domain that is far
from every observation grows quickly with dimension (inference;
@exr-acqf-coverage).

**The hallucination believer depends on the noise.** Takeno et al.'s results
were obtained at noise variance $10^{-4}$, exactly the regime in which they
showed the Laplace approximation fails worst. Their Appendix G.4 shows that the
hallucination believer did relatively poorly when combined with the maximally
uncertain challenge or with binary expected improvement, which they suspect is
due to over-exploration [@takeno2023practicalc]. Under logistic noise, the
POP-BO authors report that the hallucination believer got stuck in local optima
because it trusts random preference feedback too much, treating it as a hard
constraint when it draws its Thompson sample [@xu2024principledb]. Its
advantage, in other words, depends on the noise level (inference).

**Other costs.** MPES must enumerate the possible answers: in qEUBO's 4- to
7-dimensional experiments it took 12.7 to 24.8 seconds per iteration against
about 7 to 12 seconds for qEUBO, and it lost to qEUBO [@astudillo2023qeubob].

## The acquisition functions compared {#sec-acqf-table}

@tbl-acqf-compare collects the main rules. Every entry under "Favorable
evidence" and "Documented failures" carries the dimensions and noise under
which it was observed, taken from the study that reported it
(@tbl-acqf-studies gives each study's full setting). The frequentist regret
bounds are detailed in @sec-pbo-theory.

::: {.table #tbl-acqf-compare title="The main acquisition rules for preferential Bayesian optimization, with the conditions under which each finding was observed."}
| Rule (proposers, venue) | Theory | Favorable evidence | Documented failures |
|---|---|---|---|
| PE [@gonzalez2017preferentialb] | none | 1-D and 2-D, 33 grid points per dimension; noise not stated | worse as dimension grows |
| CEI [@gonzalez2017preferentialb] | none | run only on the 1-D Forrester function | over-exploits; too costly to compute |
| DTS [@gonzalez2017preferentialb] | no regret bound for one objective; a multi-objective version is asymptotically consistent [@astudillo2025preferential] | best on 1-D and 2-D grids; noise not stated | over-explores from 4-D up (4-D and 6-D Hartmann, noise variance $10^{-4}$) [@takeno2023practicalc]; fourth of nine rules over 34 functions (probit, unit variance) [@fauvel2021efficient] |
| KernelSelfSparring [@sui2017multi] | the independent-arm version is asymptotically no-regret; the kernel version only conjectured | no dedicated evidence | falls behind in batches because members are chosen independently [@fauvel2021efficient] |
| Adapted EI and qEI [@brochu2007active; @siivola2021preferential] | qEI not asymptotically consistent (qEUBO Theorem 4) | no clear difference from batch Thompson sampling in batch experiments (at most 4-D, utility noise sd 0.05) | stalls, over-exploits, picks identical duel members (four groups; 1-D to 7-D, noise from $10^{-4}$ to logistic) |
| MUC [@fauvel2021efficient] | none | tied first by Borda rank over 34 functions (probit, unit variance) | relatively poor when combined with HB; stalled on Bukin and Ackley under expectation propagation |
| Dueling UCB, dueling Thompson sampling, EIIG [@benavoli2021preferentialb] | none | dueling UCB tied first over 34 functions (probit, unit variance) [@fauvel2021efficient] | EIIG seventh in the same comparison |
| MPES [@nguyen2021top] | none | consistently best in 1-D to 3-D and on SUSHI; noise not stated | must enumerate answers; lost to qEUBO and slowest in 4-D to 7-D (logistic noise) [@astudillo2023qeubob] |
| BALD [@houlsby2011bayesian; @lin2022preferenceb; @botorch2026changelog] | none | competitive but slightly worse in BOPE (10% wrong choices) | none recorded specifically |
| EUBO, qEUBO [@lin2022preferenceb; @astudillo2023qeubob] | one-step Bayes optimal without noise, equivalent to the knowledge gradient; additive-constant guarantee under logistic noise; Bayesian simple regret $o(1/n)$ on finite domains with $q = 2$ | best on all but one problem in 4-D to 7-D, moderate logistic noise, 150 queries | queries collapse toward the estimated maximum (2-D Levy, probit; preprint) [@wu2026knowledge]; rank-deficient Hessian (preprint) [@shao2026adaptive]; higher cumulative regret (6-D Ackley and GP samples, logistic) [@xu2024principledb] |
| HB [@takeno2023practicalc] | none (listed as future work) | best overall over 12 functions up to 6-D at noise variance $10^{-4}$ | stuck in local optima under logistic noise [@xu2024principledb]; over-explores when combined with MUC |
| Exact knowledge gradient [@wu2026knowledge] | EUBO is a lower bound on it | a 2-D Levy case study (probit) | its abstract acknowledges limits in some situations |
| POP-BO, MaxMinLCB, MR-LPF, PF-TS [@xu2024principledb; @pasztor2024bandits; @kayal2025bayesian; @lazzaro2026finiteb] | see @sec-theory-rates | all low-dimensional experiments (1-D to 6-D, logistic) | PF-TS reports higher cumulative regret for MR-LPF at $T = 300$ (1-D Ackley, 3-D catalyst) [@lazzaro2026finiteb] |
| PABBO [@zhang2025pabbob] | none | first or second on most tasks (1-D, 2-D, 6-D, HPO-B, Candy, Sushi; noise-free) | fixed dimension; noise-free default evaluation; weaker on 6-D Hartmann |
:::

The conditions in the table are easier to compare as a picture.
@fig-acqf-evidence-map places every comparison study by its noise model and
the input dimensions it tested, and lets you choose a rule to see which studies
report on it and with what result.

```{figure}
//| figure: acqf-evidence-map
//| label: fig-acqf-evidence-map
//| fig-cap: "Where each finding about an acquisition rule was observed. Each row is a study, grouped by its noise model and drawn across the input dimensions it tested (log scale); a dashed segment means the study also tested lower dimensions without listing them, and a square in the right-hand column marks tasks whose dimension the study does not give, or real-data tasks. Choosing a rule colors the studies that report on it by whether the finding was favorable, mixed, or unfavorable for that rule, and lists the findings with their conditions. Findings and conditions are from the studies cited in @tbl-acqf-compare and @tbl-acqf-studies; the verdict colors are this chapter's summary."
```

Some things to look for:

- **Thompson sampling** (the default view): both favorable findings sit at
  one to three dimensions, and even there it trailed MPES.
- **The hallucination believer**: favorable in the top row, near noise-free
  answers; unfavorable in the logistic-noise row.
- **EUBO and qEUBO**: favorable at 4 to 7 dimensions with logistic noise; mixed
  or unfavorable in the 2024 to 2026 studies that measured cumulative regret or
  looked for collapse.
- **The empty regions**: above 20 dimensions, only the local-PBO preprint
  compares rules.

## Query forms {#sec-acqf-query-forms}

A query need not be a pair. @sec-query-design teaches the forms and reports
the evidence for the main ones. For larger sets (@sec-query-batch), qEUBO found
$q = 4$ clearly better than $q = 2$, while Siivola et al. found only marginal
gains from larger batches, with different acquisition functions and noise
[@astudillo2023qeubob; @siivola2021preferential]. For galleries and projections
(@sec-gallery-projective), every projective variant beat every pairwise
variant in 2 to 20 dimensions [@mikkola2020projective], and the Sequential
Gallery's plane search beat line search in 5 to 20 dimensions, with 6
participants satisfied after 5.36 iterations on average
[@koyama2020sequential]. With pairs alone, @fig-pbo-dims shows how quickly the
share of the gap that forty comparisons close shrinks as parameters are added.
@benavoli2023choice let a person pick several mutually incomparable options
from a set. This section adds the remaining forms and what the evidence on
forms, taken together, says.

**Coactive feedback, ordinal labels, and robot-specific forms.** CoSpar
[@tucker2020preference] adds coactive feedback to the posterior sampling of
SelfSparring: the user both compares trials and suggests improvements.
LineCoSpar [@tucker2020human] restricts the computation to a random line
(@sec-query-linecospar) and tuned 6 gait parameters with 6 able-bodied
participants. ROIAL [@li2021roial] combines
ordinal labels with preferences inside a region of interest meant to guarantee
safety and comfort. @sec-cs-exoskeleton follows this line through a case study.

**Preferences over hypothetical outcomes, and requests for improvement.**
@astudillo2020multib let a decision maker compare attribute vectors, and BOPE
alternates a stage in which the person compares outcome vectors that may be
hypothetical, sampled from the outcome model, with an experimentation stage
[@botorch2026bope]. @ozaki2024multi add an *improvement request*: the decision
maker indicates which objective of a shown result they want improved, and the
utility is a Chebyshev scalarization (a weighted worst-case combination of the
objectives) with uncertain weights.

**Weak preferences, validity labels, numbers, and language.** The "About Equal"
answer of @byk2019asking, C-GLISp's better, worse, or similar, validity labels,
and crash reports are extensions of the observation model
(@sec-obs-extensions). @xu2020zeroth allowed both direct queries and duels
(COMP-GP-UCB). For finitely many options, @wang2025fusing proved that an
efficient algorithm pays, for each option, only the smaller of the two regrets
it would incur from reward feedback or from dueling feedback. Social BO
[@adachi2025bayesian], a preprint, proved that under mild rationality axioms
noisy group feedback alone cannot reach a consensus free of social influence,
and mixed cheap public votes with expensive private ones. Natural language
enters as a source of labels: PEBOL [@austin2024bayesian] uses
natural-language inference as its likelihood over independent items, and LILO
[@kobalczyk2026lilo] has a language model translate free-text feedback into
pairwise labels for `PairwiseGP` with qEUBO, with a simulated decision maker and
no study with people (@sec-llm-in-loop). Labels generated by a language model
are correlated and biased rather than independent probit noise (inference).

**What the evidence on forms says.** It points two ways. Forms that let each
human action carry more information (projections, planes, $q = 4$) beat pairs
in the papers that introduced them, while the only direct comparison of batch
winners and full rankings found little difference. Forms that ask for a
continuous answer (sliders, projections, coactive suggestions) bring other kinds
of noise, motor and perceptual precision and cognitive load, which every cited
paper handles with a single generic noise term (inference). As of September
2026 we found no controlled study with people that compares pairs, batch
winners, full rankings, and sliders under the same acquisition function and
budget, and no acquisition function derived for forms other than pairs and
multi-option sets: slider, plane, and projective queries use adapted expected
improvement or random subspaces. Until such a study exists, the choice of form
is best made on the human side, by what people can answer reliably and quickly
(@sec-query-form-evidence, @sec-hci-feedback-forms), and a richer form should be
tested against pairs within the same system (inference).

## Problem extensions {#sec-acqf-extensions}

PBO has been extended in the same directions as ordinary Bayesian
optimization: several objectives, constraints, context, many dimensions, mixed
inputs, several fidelities, and stopping. @tbl-acqf-extensions
summarizes each direction; the paragraphs after it give the points that matter.

::: {.table #tbl-acqf-extensions title="Extensions of preferential Bayesian optimization, the evidence behind each, and the main gap."}
| Extension | Representative work (in time order) | Type of evidence | Main gap |
|---|---|---|---|
| Several objectives and preferences over outcomes | Astudillo and Frazier 2020 [@astudillo2020multib]; BOPE 2022 [@lin2022preferenceb]; Ozaki et al. 2024 [@ozaki2024multi]; PUB-MOBO 2025 [@ip2025user]; Astudillo et al. 2025 [@astudillo2025preferential]; Wang et al. 2025 [@wang2025bayesian]; Huber et al. 2025 [@huber2025bayesian]; Active-MoSH 2026 [@chen2026interactive] | simulated decision makers; engineering benchmarks | few studies with people |
| Constraints | StageOpt 2018 [@sui2018stagewise]; Benavoli et al. 2021 [@benavoli2021preferentialb]; C-GLISp 2022 [@zhu2022c]; Kwon et al. 2022 [@kwon2022physically]; Iwai et al. 2025 [@iwai2025constrainedb] | controller calibration; 11 designers | no new constraint paper after 2025 |
| Safety | StageOpt 2018 [@sui2018stagewise]; ROIAL 2021 [@li2021roial]; Cosner et al. 2022 [@cosner2022safety]; CrashPBO 2026 [@menn2026preferential] | spinal cord stimulation; a quadruped robot; three robot platforms | none named |
| Context | Khan et al. 2025 [@khan2025efficient]; Wang et al. 2025 [@wang2025personalized]; Coutinho et al. 2025, 2026 [@coutinho2025accelerated; @coutinho2026efficient] | simulated buildings; a report of negative transfer | multi-task Gaussian processes untested with a preference likelihood |
| Transfer across users | Granley et al. 2023 [@granley2023human]; Meta-PO 2025 [@li2025efficient]; PABBO 2025 [@zhang2025pabbob] | 36 people for Meta-PO | few hierarchical utility priors |
| High dimension | sequential line search 2017 [@koyama2017sequentialb]; projective PBO 2020 [@mikkola2020projective]; LineCoSpar 2020 [@tucker2020human]; Sequential Gallery 2020 [@koyama2020sequential]; qEUBO 2023 [@astudillo2023qeubob]; local methods 2026 [@menn2026local]; GimmBO 2026 [@liu2026gimmbo] | simulated comparisons up to 102 dimensions | dimension-scaled and sparse priors untested |
| Mixed and categorical inputs | piecewise affine surrogates 2025 [@zhu2025global] | benchmarks | no Gaussian process preference method with categorical kernels |
| Several fidelities | Theiner et al. 2026 [@theiner2026efficient]; elicitation-augmented BO 2026 [@haltia2026elicitation] | preprint or conference paper | none before mid-2025 |
| Stopping | Bıyık et al. 2019, Theorem 3 [@byk2019asking]; the Sequential Gallery's satisfaction button [@koyama2020sequential]; Ignatenko et al. 2025 [@ignatenko2025preference] | parametric models | no stopping rule for pairwise Gaussian processes |
:::

**Several objectives.** @astudillo2025preferential let every objective be
observed only through preferences and proposed *dueling scalarized Thompson
sampling* (DSTS): sample from the posterior, apply a random Chebyshev
scalarization, then run dueling Thompson sampling. They proved it
asymptotically consistent, which they call the first convergence guarantee for
dueling Thompson sampling in PBO, while noting that even for single-objective
PBO the regret bound of DTS remains unknown; DSTS did best on four synthetic
functions and on simulated exoskeleton and autonomous-driving tasks. The
"preferences" of @abdolshah2019multi (NeurIPS 2019) are an importance order
over objectives, not pairwise feedback on designs; the other works in the
table continue the line of @sec-multi-objective.

**Constraints and safety.** StageOpt [@sui2018stagewise] optimizes a utility
under unknown safety constraints and separates expanding the safe region from
maximizing utility. Its guarantees of constraint satisfaction and convergence
are stated for numerical observations; a variant in an appendix learns the
utility from preference feedback while the safety functions still receive
numerical measurements, comes without a convergence theorem of its own, and
was used clinically for spinal cord stimulation. Constrained PBO
[@iwai2025constrainedb] proposed EUBOC, which weights EUBO by the probability,
modeled with a Gaussian process, that the constraints are satisfied, in the
manner of constrained expected improvement [@gardner2014bayesian], and
evaluated it in a banner-ad study with 11 professional designers, with
predicted click-through rate as the constraint. Constraints have been handled
in two ways: by learning feasibility from labels a person provides (C-GLISp, Benavoli et
al.'s validity labels, crash feedback), or by measuring the constraint
separately (the click-through rate in constrained PBO, StageOpt's safety
signal). Which route fits depends on whether the constraint can be observed
without the person (inference).

**Context and transfer.** Context enters through offline utilities learned
from expert knowledge [@khan2025efficient] or through contextual variables such
as outdoor temperature in controller tuning [@wang2025personalized], a
preprint. Transfer has been implemented
through amortization (PABBO), stored models of earlier users (Meta-PO), and
encoders trained on a population (Granley et al.), not through a multi-task
Gaussian process across users (inference); @sec-hci-population reports what
population priors buy.

**High dimension, mixed inputs, fidelities, and stopping.** The
high-dimensional methods of 2017 to 2020 all restrict each query to a
low-dimensional subspace through the current best point, which turns a
$d$-dimensional acquisition optimization into a one- or two-dimensional one and
lets each human action carry more than one bit (inference); @sec-hd-pbo-dims
lists how far each reached, and @sec-hd-local examines the local methods of
2026 and the lengthscale bound that confounds their comparison with qEUBO. The
piecewise affine surrogate of @zhu2025global uses mixed-integer linear
programming to handle known linear constraints and mixed numerical and
categorical variables; it is the only preference method for mixed inputs we
found, and the Sequential Gallery lists not handling discrete parameters, such
as layouts, fonts, or filter types, among its limitations. The only
multi-fidelity preference methods are those of @theiner2026efficient and
elicitation-augmented Bayesian optimization [@haltia2026elicitation], both from
2026. For stopping, the only explicit optimal rule assumes a parametric reward
model [@byk2019asking], and the Sequential Gallery stops when the user presses
a "satisfied" button; @sec-hd-stopping reports what scalar Bayesian
optimization has learned about stopping and what a preferential rule would
need.

## Competing 'first' claims {#sec-acqf-firsts}

PBO sits between dueling bandits, preference-based reinforcement learning,
control (the GLISp line), and human-computer interaction, and claims of being
"first" in one community often overlook earlier work in another. Each claim
below holds only within its exact setting (inference).

- **Constrained PBO** [@iwai2025constrainedb] claims to be the first to
  introduce inequality constraints, after C-GLISp [@zhu2022c] handled unknown
  constraints in 2022 and a variant of StageOpt [@sui2018stagewise] learned a
  utility from preference feedback under numerically measured safety
  constraints in 2018; it does not cite the validity labels of Benavoli et al.
  2021 [@benavoli2021preferentialb]. The claim holds for inequality constraints
  on a Gaussian process surrogate.
- **LineSpar** [@cheng2020preference], a workshop paper, claims to be the first
  high-dimensional preference-based Bayesian optimization, after sequential
  line search (2017) and alongside projective PBO (ICML 2020).
- **DSTS** [@astudillo2025preferential] claims the first multi-objective PBO
  framework, which overlaps with a preprint on choice functions
  [@benavoli2021choice]; the claim holds for latent objectives observed only
  through preferences.
- **POP-BO** [@xu2024principledb] says existing methods lack cumulative regret
  or global convergence guarantees, with the qualifier "continuous input
  space", and does not cite @kirschner2021bias.
- **MPES** [@nguyen2021top] calls itself the first information-theoretic
  acquisition function with preference observations (first arXiv version
  December 2020), while the EIIG rule [@benavoli2021preferentialb] (August 2020)
  also uses the dueling information gain; both descend from BALD
  [@houlsby2011bayesian] (inference).
- **EUBO's one-step Bayes optimality** was first shown in BOPE
  [@lin2022preferenceb]; the qEUBO paper's claim holds for its extension to
  logistic noise and $q > 2$.

## The state of empirical comparison {#sec-acqf-empirical}

@tbl-acqf-studies lists the main comparison studies and their settings. In
it, qTS is batch Thompson sampling, qNEI is noisy batch expected improvement,
HPO-B is a benchmark built from hyperparameter optimization tasks, and a Sobol
sequence is a quasi-random space-filling design.

::: {.table #tbl-acqf-studies title="The main empirical comparisons of preferential acquisition rules and the settings in which they were run."}
| Study | Compared | Dimensions | Budget and repetitions | Noise model | Main conclusion |
|---|---|---|---|---|---|
| González et al. 2017 [@gonzalez2017preferentialb] | PE, CEI, DTS, random, interactive BO, Sparring | 1, 2 | 5 + 200 duels; 20 repetitions | not stated | DTS best |
| Mikkola et al. 2020 [@mikkola2020projective] | 5 projective rules against pairwise random and DTS variants | 2, 6, 10, 20 | 100 queries; 25 initializations | small Gaussian noise | every projective variant beat every pairwise variant |
| Siivola et al. 2021 [@siivola2021preferential] | batch EI, batch Thompson sampling, random; three inference methods | 6 functions; real data with $d \le 4$ | batch sizes 2 to 6; 10 repetitions | utility noise sd 0.05 | differences between acquisition functions larger than between feedback types; only slightly better than baseline on real data |
| Nguyen et al. 2021 [@nguyen2021top] | MPES, EI, DTS | 1 to 3; CIFAR-10 embedding; SUSHI | not stated | not stated | MPES consistently best |
| Fauvel and Chalk 2021 [@fauvel2021efficient] | 9 rules | 34 functions | 80 iterations; 40 repetitions | probit, unit variance after normalization | MUC and dueling UCB tied first; Brochu EI eighth; random ninth |
| Lin et al. 2022 [@lin2022preferenceb] | EUBO-ζ, EUBO-f̃, BALD-f̃, random, others | multi-outcome problems | 75 comparisons in 3 stages; 30 repetitions | 10% wrong choices | EUBO variants best |
| Astudillo et al. 2023 [@astudillo2023qeubob] | qEUBO, MPES, qTS, qEI, qNEI, random | 4 to 7 | $4d$ initial + 150 queries; 50 or 100 repetitions | logistic, calibrated to 10%, 20%, 30% errors on the top 1% of point pairs | with $q = 2$, qEUBO best on all problems except Car cab |
| Takeno et al. 2023 [@takeno2023practicalc] | HB, Laplace and EP with EI, MUC, skew-GP sampling rules | up to 6 (12 functions) | $3d$ initial; 10 repetitions | noise variance $10^{-4}$ | HB best overall; qEUBO not included |
| Xu et al. 2024, POP-BO [@xu2024principledb] | DTS, HB, qEUBO | GP samples; 6-D Ackley | not stated | logistic | qEUBO's reported solution slightly better, cumulative regret more than 2.5 times higher |
| Zhang et al. 2025, PABBO [@zhang2025pabbob] | qEUBO, qEI, qNEI, qTS, MPES, random | 1, 2, 6; HPO-B; Candy; Sushi | 30 repetitions | noise-free | PABBO first or second; random often beat some GP baselines |
| Lazzaro et al. 2026, PF-TS [@lazzaro2026finiteb] | MR-LPF, POP-BO, MaxMinLCB | 1-D Ackley; 3-D catalyst | $T = 300$; 30 repetitions | logistic | PF-TS cumulative regret significantly lower than MR-LPF and POP-BO |
| Menn et al. 2026, local (preprint) [@menn2026local] | local methods, qEUBO, HB with EI, GLISp, Sobol | up to 102 | about $10d$ comparisons, after $5d$ random evaluations for policy search | Gaussian, 10% of the value range | local methods better on steep optima |
:::

The noise calibration of the qEUBO experiments comes from their Appendix C.2
and code [@astudillo2023noise].

**The comparisons do not add up.** The studies differ in metric (@sec-sw-regret
lists five definitions of regret), in noise model, from a nearly noise-free
probit to 10% flipped answers and moderate Gumbel noise (@sec-sw-noise-models),
and in posterior inference: Laplace, expectation propagation, variational
inference, or skew Gaussian process sampling. Rankings flip with these choices,
as POP-BO's comparison with qEUBO shows (inference). Fauvel and Chalk's Borda analysis over 34
functions is the broadest single comparison, but it predates qEUBO and the
hallucination believer, uses expectation propagation throughout, and runs only
80 iterations. Takeno et al.'s comparison comes next, but its duels are almost
noise-free and it does not include qEUBO. No comparison has breadth, realistic
noise, and the acquisition functions of 2023 and later at the same time.

**Random queries sometimes keep up.** The PABBO authors write that the random
strategy often beat some of the Gaussian process baselines
[@zhang2025pabbob]. On real data with a low signal-to-noise ratio, Siivola et
al. found that all methods only barely beat the baseline, and wrote that
preference observations are inherently less informative than direct ones, that
larger batches do not alleviate this, and that for very noisy data, random
search with large batches may be a good choice [@siivola2021preferential]. In a
single replication, BoTorch's BOPE tutorial prints candidate utilities of
$-0.473$ for EUBO-ζ, $-0.216$ for random preference exploration, $-0.101$ for
the true utility, and $-1.380$ for random experimentation, and notes that
EUBO-ζ's win is not guaranteed in a single replication [@botorch2026bope]. The
three pieces of evidence point the same way: at small budgets and realistic
noise, the gap between an elaborate acquisition function and a random design may
be small, and a single run says little (inference).

**Evidence from people.** The human studies in this chapter are all small, and
none was designed to compare acquisition functions: projective PBO's
materials-science user, the Sequential Gallery's 6 participants, LineCoSpar's
6, constrained PBO's 11 designers, and Meta-PO's 36. **As of September 2026, we
found no study that randomizes people to different acquisition functions, such
as qEUBO, DTS, and MUC, under the same interface and budget.** @sec-open-decisive
describes the experiment that would settle it. Until then, a practitioner
should treat the ranking of acquisition functions as unknown for people and
keep a share of random queries in every session as a control, which costs
little given the evidence above (inference). Nor is there a benchmark for PBO
with agreed metrics, noise models, and budgets: each paper reuses Forrester,
six-hump camel, Hartmann, Ackley, Sushi, and Candy as it sees fit
(@sec-sw-datasets).

## Settled, contested, missing {#sec-acqf-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** qEI is not asymptotically consistent on the instances the qEUBO
paper constructs [@astudillo2023qeubob]. Adapted expected improvement stalled in
the experiments of four independent groups. EUBO's one-step Bayes optimality
holds without noise, has an additive-constant guarantee under logistic noise,
and was first shown in BOPE [@lin2022preferenceb]. Query forms in which each
action carries more information, projections and planes, beat pairs in the
simulations of the papers that proposed them [@mikkola2020projective;
@koyama2020sequential].

**Contested.** The value of more options per query: qEUBO found $q = 4$ clearly
better than $q = 2$, Siivola et al. found only marginal gains, and their
acquisition functions and noise differ. The advantage of the hallucination
believer: it is best near noise-free answers, and stuck in local optima under
logistic noise according to POP-BO. EUBO's collapse and ill-conditioning: two
2026 preprints, not yet peer reviewed or tested on human data. The ranking of
Thompson-sampling rules, which flips with dimension and setting.

**Missing.** A broad comparison of acquisition functions on a shared benchmark
with matched noise and budget. An experiment that randomizes people to
acquisition functions. Acquisition functions derived for slider, plane, and
projective queries. A stopping rule with guarantees for pairwise Gaussian
process models. Regret bounds for DTS and HB, and continuous-domain guarantees
for qEUBO under noise (@sec-pbo-theory).

**For a choice that must be made now.** The evidence supports including random
queries and simple baselines among the comparisons, and reporting dimension,
noise, and metric. qEUBO with `PairwiseGP` is the best-maintained default, but
that recommendation comes from the software ecosystem and from within the PBO
framework, not from a direct comparison with simpler methods (inference).
:::

## Exercises {#sec-acqf-exercises}

::: {.exercise #exr-acqf-lambert}
Theorem 2 of the qEUBO paper bounds the loss from logistic noise of scale
$\tau$ by $\tau\, W((q - 1)/e)$, where $W$ is the inverse of $w \mapsto w e^w$.
Compute $W((q - 1)/e)$ for $q = 2$ and $q = 4$ by solving $w e^w = (q - 1)/e$,
and say what the bound implies about showing more options per query when
answers are noisy.

::: {.solution}
For $q = 2$, solve $w e^w = 1/e \approx 0.368$: $w = 0.28$ gives
$0.28 \cdot e^{0.28} \approx 0.37$, so $W(1/e) \approx 0.28$. For $q = 4$,
solve $w e^w = 3/e \approx 1.10$: $w = 0.60$ gives $0.60 \cdot 1.82 \approx 1.09$,
so $W(3/e) \approx 0.60$. The guaranteed loss relative to the noise-free
one-step optimum roughly doubles from $q = 2$ to $q = 4$, in units of the noise
scale $\tau$, while the value of a noise-free answer from four options
is at least that from two. The bound alone does not say whether more options
help; it says the guarantee loosens slowly, which is consistent with qEUBO's
experiments finding $q = 4$ better than $q = 2$ at moderate noise.
:::
:::

::: {.exercise #exr-acqf-coverage}
A rough way to see why Thompson sampling over-explores as the dimension grows.
Suppose the posterior is confident within distance $r = 0.1$ of each of $n = 20$
observed points in the unit cube $[0, 1]^d$, and uncertain elsewhere. Estimate
the fraction of the cube that is confident for $d = 1$ and $d = 6$, using the
volume of a $d$-dimensional ball, $\pi^{d/2} r^d / \Gamma(d/2 + 1)$, and
ignoring overlaps and edges. Where will the maximum of one posterior sample
tend to fall?

::: {.solution}
For $d = 1$ the "ball" is an interval of length $2r = 0.2$, so 20 points could
cover the whole interval (the estimate $20 \times 0.2 = 4$ exceeds 1; overlaps
make the true coverage at most 1). For $d = 6$, the ball volume is
$\pi^3 r^6 / 3! \approx 31.0 \times 10^{-6} / 6 \approx 5.2 \times 10^{-6}$, so 20
balls cover about $10^{-4}$ of the cube. Almost all of the six-dimensional cube
is uncertain, so a posterior sample has many chances to be high somewhere
nobody has looked, and its maximum tends to land there. That is exploration by
construction, and in six dimensions it rarely returns to refine the best region.
This picture is our illustration of the mechanism, not a result of the cited
papers, which report the over-exploration without isolating its cause.
:::
:::

::: {.exercise #exr-acqf-pool}
Using @tbl-acqf-studies, find a study that compares qEUBO with the
hallucination believer. Under which noise model, at which dimensions, and with
which metric? What would a comparison need in order to settle whether either
rule is better at human noise levels?

::: {.solution}
Only the POP-BO paper [@xu2024principledb] includes both (with DTS), under
logistic noise, on Gaussian process samples and the 6-dimensional Ackley
function, with an unstated budget. It reports that the hallucination believer
got stuck in local optima, and that qEUBO's reported solution was slightly
better than POP-BO's but its cumulative regret more than 2.5 times higher.
Takeno et al., where the hallucination believer did best, did not include qEUBO
and used noise variance $10^{-4}$. A settling comparison would fix a noise model
calibrated to human comparisons, run both rules with the same surrogate,
inference, budget, and dimensions, report both simple and cumulative regret,
include random queries, and ideally be repeated with people.
:::
:::

## Further reading {#further-reading .unnumbered}

- @astudillo2023qeubob is the reference for qEUBO; read its four theorems with
  their conditions, and its Appendix C.2 on how the noise was calibrated.
- @fauvel2021efficient is the broadest comparison of rules (34 functions,
  nine rules) and the clearest treatment of epistemic against aleatoric
  uncertainty in acquisition.
- @takeno2023practicalc and @xu2024principledb disagree about the hallucination
  believer; together they show how much the noise level decides.
- @wu2026knowledge derives the exact knowledge gradient under a probit
  likelihood and is the best place to see why EUBO's equivalence with it breaks
  under noise.
- @mikkola2020projective and @koyama2020sequential are the two reference
  papers on queries richer than a pair.
