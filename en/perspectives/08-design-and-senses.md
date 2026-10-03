---
status: done
synopsis: "What design research, architecture, urban planning, engineering design, food and sensory science, empirical aesthetics and music, and the health sciences have learned about eliciting preferences, from placebo pairs and fixation to test-retest benchmarks and validation against real behavior, ending with the methods PBO can adopt directly."
---

# Design, Sensory Science, Art, and Health {#sec-design-senses}

The last two chapters asked where preferences come from and what a set of
comparisons can tell a model. This chapter turns to the disciplines that ask
people about designed things for a living: designers and engineers, architects
and planners, food scientists who run consumer panels, researchers of music
and visual aesthetics, and health researchers who elicit patients' priorities
before a treatment decision. Many of them have faced, at scale and for
decades, the measurement problems that preferential Bayesian optimization
(PBO) meets in every user study. Their most transferable answer is the placebo
pair of sensory science (@sec-ds-placebo); the chapter ends by gathering what
PBO can adopt from this part of the book without new theory (@sec-ds-adopt).

## Design research {#sec-ds-design}

Design research has long described creativity as the *co-evolution* of problem
and solution, with designers who discover what they want by designing
[@dorst2001creativity]. But "co-evolution" names a family of different
concepts, some meaning that problem and solution change each other, others
only that attention alternates between them, and parallel processes appear
across creative work, so it looks like a general feature of creative work
rather than something specific to design [@crilly2021evolution;
@crilly2021evolutionb]. Product-innovation methods fare worse. In a working
paper presented at a practitioner conference, not peer reviewed,
@chapman2022kano had 1,501 people answer the same item of the *Kano model*,
which sorts product attributes into "must-be" attributes and "delighters",
twice within seconds: only 39% gave the same "expect" answer again.

**Fixation is the clearest new evidence.** *Design fixation* is the tendency
to reuse features of examples one has seen, even when they are flawed or the
task asks for something new. In a large experiment with novice engineers,
@leahy2020design showed half of them a given example and let the other half
generate their own first concept. The students who saw the example fixated on
it less than the control students fixated on their own first concept, which
revises the classic experiment of @jansson1991design, whose control condition
had been assumed to be free of fixation. In a between-subjects experiment
($N = 60$), @wadinambiarachchi2024effects found that participants who used an
AI image generator during ideation fixated more on the initial example and
produced fewer, less varied, and less original ideas, with effects that
depended on how participants prompted and ideated. And in a study of
designers working with an optimizer [@mo2024cooperative], 12 of 18
participants preferred a mode in which designer and optimizer collaborate, and
the mode led by the optimizer gave the lowest sense of control (compare
@sec-hci-agency).

**What it means for PBO.** A PBO gallery is a sequence of examples supplied by
the system; used for ideation rather than for tuning parameters, it should be
expected to produce the fixation and loss of diversity that
Wadinambiarachchi et al. observed (inference). The incumbent, the design the
user chose last, plays the role of a designer's own first concept and may
anchor later comparisons (inference), the more so because choosing changes
value: in a fitted revaluation model of repeated food choices, each choice
raised the value of the chosen snack by about 0.18 US dollars and lowered that
of the rejected one by as much [@zylberberg2024value]. Comparing periodically
against diverse designs not derived from the incumbent is a practical check
(inference). A Kano "must-be" attribute is a utility with a threshold, better
represented as a constraint or a hinge-shaped component (@sec-constrained-bo),
but Kano classifications are too unreliable to serve as a hard prior
(inference). We found no study that measures fixation within a PBO session,
for example against a control group shown a random gallery.

## Architecture {#sec-ds-architecture}

Architecture is often credited with universal spatial preferences, which has
prompted the suggestion to encode a universal good as a prior separate from
personal preference. Jay Appleton's *prospect-refuge theory*, which holds that
people prefer places where they can see without being seen, has inconsistent
quantitative support. A meta-analysis of 34 quantitative studies, published
slightly before the period this part covers [@dosen2016evidence], found
prospect supported in 19 of its 31 tests (61%) and refuge in only 8 of its 27
tests (30%); the 53% and 22% in the paper's own summary are these factors'
shares of the 36 supporting findings, not rates of support. Refuge found
support mainly in studies of natural environments (5 of 8 tests in
landscapes, against 2 of 17 in interiors), and of the 29 studies with survey
results, 16 had 20 or fewer participants. Perceptual dimensions fare better:
@coburn2020psychological had 798 participants rate 200 images of interiors on
16 scales, and three components, coherence, fascination, and hominess,
explained 90% of the variance in the ratings; the structure replicated in an
independent sample ($n = 614$).

**Shared and individual taste.** The key result since 2017 is
@vessel2018stronger. Agreement between individuals was high for faces and
landscapes and low for building exteriors, building interiors, and artworks.
Within participants, agreement was significantly higher for landscapes than
for building exteriors, while the reliability of each person's own ratings did
not differ between the two. The authors proposed that taste is shared for
natural kinds and individual for cultural artifacts. Expertise splits taste
further: young architects and matched laypeople rating Czech detached houses
had the same overall means (5.08 against 5.09), but the architects rated
modern and wooden houses higher, and catalog houses and "McMansions" lower
[@safarova2019differences].

**What it means for PBO.** Low agreement combined with equal reliability means
that a population-average prior gives little guidance about one person's
architectural taste: architecture calls mainly for learning per person, while
landscapes or faces can borrow more from population data, so the choice
between informative population priors and weak priors depends on the domain
(inference; @sec-sw-priors). Defining the surrogate on a few perceptual
dimensions, rather than on dozens of raw geometric parameters, can reduce the
effective dimension (inference; @sec-high-dimensions). An architect who runs
PBO on a client's behalf optimizes the architect's utility unless the client
is in the loop (inference). Given the low agreement on buildings, a universal
architectural good should not be encoded as a prior (inference).

## Urban planning {#sec-ds-urban}

Surveys of urban perception are the largest practical application of pairwise
visual comparison outside machine learning, built on Place Pulse 2.0
[@dubey2016deep], which collected comparisons of street-view images along
dimensions such as "safer" or "livelier". @quintana2025global (with
methodological details in the arXiv version) had 1,000 demographically
balanced participants from five countries each make 50 pairwise comparisons,
with the option "Both are the same to me". They scored images with the
*Strength of Schedule* method, which credits a win by the strength of the
images it was won against, because it needs fewer comparisons per image than
the TrueSkill rating system (4 against 22 to 29). Gender produced the most
differences between groups, especially for "safe", and a vision transformer
trained on Place Pulse overestimated positive indicators such as "lively"
relative to the human ratings. In *participatory budgeting*, where residents
decide how to spend part of a public budget, @benade2017preference [with the
journal version @benade2021preference] found that *threshold approval*
voting, in which each voter approves the projects whose value to them exceeds
a threshold, was qualitatively better than knapsack voting and value rankings
on *distortion* (how far the welfare of the chosen outcome falls short of the
best possible when only ordinal information is available) and on regret.

**What it means for PBO.** Aggregating pairwise judgments from a heterogeneous
population biases the learned scores and hides differences between subgroups,
so group or civic PBO should model respondents' covariates or keep a surrogate
per group (inference; @sec-many-users). Urban surveys treat an explicit
indifference option as standard, and a forced-choice interface discards that
signal (inference; @sec-ties). With several stakeholders, the query format is
a design variable with social-choice consequences (inference;
@sec-econ-social-choice).

## Materials and engineering design {#sec-ds-engineering}

Engineering design produced a direct precursor of PBO: @ren2011design cast
design preference elicitation as an optimization problem, using a support
vector machine with efficient global optimization in user tests of car
exterior styling. Since 2017, @kanarik2023human compared human engineers with Bayesian optimization
algorithms on the design of a semiconductor plasma etch process, in a
controlled virtual process game. Humans did well early, algorithms were far
more cost-efficient close to tight tolerances, and a "human first, computer
last" strategy halved the cost of reaching the target compared with humans
alone. @nandy2025exploring optimized the perceived comfort of parametric mugs
with interactive Bayesian optimization per participant ($N = 31$), started
from scratch or from the data of an earlier group ($N = 25$), and interactive
models aligned intent and generated designs better than non-interactive ones.
A 2025 paper [@huber2025bayesian] built a Bayesian model of a decision maker's
utility over a Pareto set from pairwise comparisons, tested it on problems
with up to nine objectives, and released its code.

**What it means for PBO.** A workflow should plan a handoff from the person to
the algorithm instead of using one mode throughout; but Kanarik et al.'s
evidence concerns expert search with an objective metric, and whether it
transfers to design driven by taste is uncertain (inference; compare
@sec-cs-chem-experts). Pairwise acquisition on a precomputed Pareto front is a
practical way to bring a designer's preference in after multi-objective
optimization (@sec-multi-objective, @sec-econ-moo).

## Food and sensory science {#sec-ds-food}

Food scientists have long run paired preference tests with consumers
[@omahony2017evolution], and they have learned uncomfortable things about what
people say when asked "which do you prefer?". One famous result needs care
first. In @morrot2001color, white wine colored red with an odorless dye was
described in olfactory terms as red wine by 54 tasters, oenology students in
Bordeaux. One reading says that color completely overrides chemosensory
information, with devastating consequences for preference data. The effect is
real and has been replicated conceptually, but the task was choosing
descriptors, not stating a preference or a liking. In @wang2019drinking,
participants with wine experience judged a white wine dyed to look like rosé
far more similar to the rosé than to the identical white wine, yet the dyed
wine was liked less than either real wine, and participants felt it was
somehow different. @spence2020wine reviewed how contextual factors, from the
color of ambient light to background music, have profound and sometimes
predictable effects on tasting.

### Paired preference tests and placebo pairs {#sec-ds-placebo}

The most useful lesson for PBO is the methodology of paired preference tests.
@omahony2017evolution reviewed its history. Forced choice was adopted first
because its binomial statistics are simple, but it gave consumers no way to
say "no preference". Consumers then turned out to report preferences between
stimuli that should have been identical, so *placebo pairs*, pairs made of
two identical samples, became the control condition. The foundational study
carries the finding in its title, "Consumers report preferences when they
should not" [@marchisano2003consumers]. Frequency measures (how many consumers
chose each sample) were later supplemented by the discrimination index $d'$ of
signal detection theory, the perceived distance between two stimuli in units
of the standard deviation of perceptual noise, which measures the strength of
a preference more accurately.

Sensory science reads such data with *Thurstonian models*, the same model PBO
uses (@sec-thurstone). A paired preference test with a "no preference" option
is technically identical to the *2-AC* protocol ("two alternatives, with a
no-difference option"), whose Thurstonian model has two parameters: the
distance $\delta$ between the two stimuli and a decision threshold $\tau$
below which the person reports no preference; it is closely related to a
cumulative probit model, the ordinal likelihood of @sec-obs-extensions
[@christensen2012estimation].

::: {.derivation title="What the placebo pairs tell a Thurstonian model"}
1. In the Thurstonian model of @sec-thurstone, each of the two options is
   perceived with independent Gaussian noise of standard deviation $\sigma$
   around its utility, so the perceived difference is
   $D \sim \N(\delta, 2\sigma^2)$, where $\delta$ is the true difference.
2. In the 2-AC model, the person reports "no preference" when
   $\lvert D \rvert \le \tau$ and otherwise prefers the option that $D$
   favors.
3. A placebo pair has $\delta = 0$. The probability that the person
   nevertheless states a preference is
   $r = \Prob(\lvert D \rvert > \tau) = 2\,\Phi\!\big(-\tau / (\sqrt{2}\,\sigma)\big)$,
   by the symmetry of the Gaussian.
4. Solving for the threshold gives $\tau / \sigma = -\sqrt{2}\,\Phi^{-1}(r / 2)$.
   A false-preference rate of $r = 0.5$ means $\tau \approx 0.95\,\sigma$; a
   forced-choice interface has $\tau = 0$ and therefore $r = 1$ by
   construction.
5. The placebo pairs thus identify the person's tie threshold relative to their
   noise, which is exactly the quantity a PBO likelihood with ties needs and
   cannot learn from forced choices.
:::

@fig-ds-placebo lets you take a short paired preference test with placebo
pairs. Answer as you would in a study, and only then read the results.

```{figure}
//| figure: ds-placebo
//| label: fig-ds-placebo
//| fig-cap: "A paired preference test with placebo pairs, in which you are the consumer. Twelve pairs of small posters appear one at a time; choose the one you prefer, or *No preference*. After the last answer the figure reveals which pairs were identical and which differed slightly (each different pair appears twice with the sides swapped), and reports your false-preference rate on identical pairs, your side bias, your consistency, and the 'no preference' band that a Thurstonian 2-AC model would infer from your answers. With so few pairs, the numbers describe this session, not you; the stimuli and their differences are illustrative."
```

Things to notice:

- Consumers in paired tests often state preferences between identical
  samples [@marchisano2003consumers]; check whether you did. Every preference
  you stated on an identical pair is a "false preference": in a PBO session it
  would enter the likelihood as evidence that one design beats the other.
- If your false preferences went mostly to one side, your answers carried a
  position bias, which a likelihood can absorb with a side term
  (@eq-nat-prior-logit) if the sides are balanced.
- A forced-choice interface, the default in most PBO studies, would have
  recorded every identical pair as a preference.

Several studies refine the method. @halim2020paired found that placing the
placebo pair after the target pair raised the share of "no preference" answers,
though not always significantly. @xia2020paired found that for some products
the effect is weaker because samples of the same product vary a lot, so the
"identical" pairs were not really identical. On reliability,
@nijman2022stability had 62 participants rate their liking of one beer, and
ten emotions it evoked, in two sessions each in a bar and in a
central-location test. The *intraclass correlation coefficient* (ICC), here
the share of rating variance due to stable differences between participants
rather than to noise between sessions, averaged 0.66 (±0.1) in the bar and
0.60 (±0.15) in the central-location test across the eleven ratings, and for
liking alone lay between 0.58 and 0.76.

**What it means for PBO.** Occasionally presenting an identical pair, the same
design twice, estimates the user's false-preference rate; allowing a tie
answer and fitting the noise or lapse term of the probit likelihood with the
tie and placebo data calibrates the model. Because the Thurstonian model of
sensory science is PBO's probit likelihood, the transfer is mathematically
direct, and $d'$ is simply the utility difference in units of the noise
standard deviation (inference). An "identical" pair must be truly identical,
with the same random seed and the same rendering, or the placebo is
contaminated (inference). Hedonic test-retest
correlations of about 0.6 to 0.7 give a reasonable expectation of consistency
across sessions for sensory stimuli, and a PBO posterior far more confident
than that has probably overfitted a single session (inference).

## Empirical aesthetics and music {#sec-ds-aesthetics}

Empirical aesthetics has well-known principles: Berlyne's *Wundt curve* says
liking peaks at intermediate complexity or novelty, and *processing fluency*
says that the easier a stimulus is to process, the more positively it is
judged. The inverted U is well supported at the level of groups: of 57 music
studies reviewed by @chmiel2017back, 50 (87.7%) were consistent with a
(segmented) inverted U between complexity or arousal and liking, though the
authors note that the model may not match Berlyne's arousal mechanism. But
@gucluturk2016liking showed, with 30 participants rating grayscale images,
that the group's inverted U was composed of two clusters of people: in one,
liking fell with complexity; in the other, it rose. @fig-ds-mixture-u shows
how two kinds of monotone individual produce an inverted U that nobody has.

```{figure}
//| figure: ds-mixture-u
//| label: fig-ds-mixture-u
//| fig-cap: "An inverted U that nobody has. Forty simulated viewers rate images of increasing complexity; for magenta viewers liking falls past a personal turning point, for green viewers it rises. Every individual is monotone, yet the population average (blue) peaks at intermediate complexity. Below the plot, the liking that a design at the population peak gives each group, as a share of what a design aimed at that group would give. The curves are illustrative, in the spirit of the two clusters found by @gucluturk2016liking."
```

At an even split, the population peaks near complexity 0.5, and a design
placed there gives each group only about 70% of what a design aimed at it would
give. Moving the slider to a 25% minority shows the minority served at only
about a third of its best.

Fluency is not uniformly positive either. @graf2017aesthetic found that for
chairs and lamps, the effect of fluency on attractiveness ran through
pleasure, especially under automatic processing, while under controlled
processing disfluent designs could gain attractiveness through interest. And
pleasure depends on what came before. @cheung2019uncertainty used a machine
learning model to quantify the uncertainty and surprise of 80,000 chords in US
Billboard pop songs: chords were most pleasurable when they were surprising
in a context of low uncertainty, or unsurprising in a context of high
uncertainty. On repetition the evidence conflicts: @gold2019predictability
found that seven repetitions of a stimulus lowered liking but did not destroy
the preference for intermediate complexity, while in @madison2017repeated,
liking for 40 unfamiliar pieces heard 28 times each over about four weeks rose
monotonically at every level of complexity. The timescale and whether
listening is voluntary may explain the difference, but no study has resolved
it.

**Since 2017, a computational theory of aesthetic value.**
@brielmann2022computational proposed a model in which the observer's
sensory-cognitive state is a generative model of stimuli, and aesthetic value
has two parts: an immediate sensory reward, the fluency of the stimulus
measured as its likelihood under the current state, and the change in expected
future reward as that state moves toward or away from the distribution of
stimuli expected in the long run. @brielmann2024modelling had 59 participants
rate 55 morphed images of dogs. Individual models using features from the
deep convolutional network VGG-16 captured trial-by-trial liking with a median
correlation of 0.65, against 0.01 for the population average, and the
learning component explained on average 17% more variance for the actual order
of presentation than for simulated random orders.

**What it means for PBO.** In the Brielmann-Dayan model every stimulus seen
updates the observer's state, so the latent utility is a function of the
stimulus and of a state driven by the sequence of queries: in preference
elicitation every query is also a write (inference; @sec-soc-measuring). The
order of a gallery is part of the stimulus, and showing the incumbent many
times may lower or raise its value, depending on the timescale (inference).
For individualized content, individual models beat the population average by a
wide margin (0.65 against 0.01), so per-user surrogates are necessary, and
deep network features are a usable input space (inference). A population
inverted U can be a mixture of opposite monotone preferences, so a proposal
that the acquisition function should target a zone of optimal novelty assumes
that each person has an inverted U, which is not established (inference).
Fast gallery-style judgments may favor fluent, typical designs and slower
comparisons interesting ones (inference). An observer-state model such as
Brielmann and Dayan's could serve as the non-stationarity of a PBO surrogate
(@sec-rec-nonstationary) (inference).

## Health sciences and pharmacology {#sec-ds-health}

Health research elicits patients' preferences before treatment decisions, for
regulators approving medical products, and for health technology assessment.
Its main stated-preference method, the *discrete choice experiment* (DCE), in
which respondents choose between hypothetical alternatives described by their
attributes (@sec-econ-conjoint), is close to PBO but developed independently;
from 2018 to 2023, 1,279 health DCEs were published [@nouwens2025evolving].
Pharmacology supplies a favorite argument against stable preferences, that
drugs can change choices; the substance survives, though its citations
circulate with errors (@tbl-ds-claims). Manipulating serotonin changed moral
judgments and responses to unfair offers [@crockett2008serotonin;
@crockett2010serotonin], in the 2010 study more so in highly empathic
individuals, and impulse control disorders in Parkinson's disease
are linked mainly to dopamine agonists [@voon2006prevalence;
@weintraub2010impulse].

**Do stated choices predict real ones?** On *external validity*,
@quaife2018well pooled 8 studies (6 in the meta-analysis): sensitivity 88%
(the share of actual choices to take up an option that the DCE predicted),
specificity 34% (the share of actual refusals it predicted), and an area under
the ROC curve of 0.60, where 0.5 is chance. @zhang2025prediction updated this
to 14 studies (10 in the meta-analysis): sensitivity 89%, specificity 52%,
area under the curve 0.81, with very high heterogeneity between studies
($I^2$ of 95% to 97%, the share of variation not due to chance). Prediction
was better for preventive, opt-in decisions. @fig-des-validity turns the two
pooled estimates into counts of people.

```{figure}
//| figure: des-validity
//| label: fig-des-validity
//| fig-cap: "What a stated choice predicts about real behavior. One hundred people either take up a health option in real life (blue) or refuse it (magenta); a filled dot means a discrete choice experiment predicted take-up for that person, a hollow dot that it predicted refusal. Sensitivity and specificity are the pooled estimates of two meta-analyses that compared stated with actual choices: 88% and 34% over 6 studies [@quaife2018well], 89% and 52% over 10 studies [@zhang2025prediction]. The share of people who actually take the option up is illustrative, not from either source, and counts are rounded to whole people."
```

Things to try:

- At the defaults, with Quaife et al.'s estimates and half the people taking
  the option up, the experiment predicts take-up for 77 people, and 44 of them
  (57%) do. The 33 filled magenta dots are refusals it called take-ups.
- Switch to Zhang et al. The higher specificity cuts the false take-ups to 24:
  69 predicted, 45 of them real (65%).
- Lower the share who take it up to 20%. With Quaife et al.'s estimates, 71
  people are predicted to take it up and only 18 of them (25%) do; with
  Zhang et al.'s, 56 and 18 (32%). When most people would refuse, a stated
  "yes" says little.

**Reliability and dependence on method.** Retesting 162 people after two
weeks, @xie2022discrete found that 76.4% of DCE choices were identical
(Cohen's kappa 0.528, agreement corrected for chance), while the time
trade-off method, which asks how many years in full health a person would
trade for a longer life in a worse state, had an intraclass correlation of
0.958 with 59.3% of values identical. In @whichello2023discrete, 459 Dutch
adults with diabetes completed a DCE and *swing weighting* (a direct method
that asks how much moving each attribute from its worst to its best level
matters; @sec-econ-mcda) in balanced order. Both ranked cost and precision
highest, but the DCE's weights for the most and least important attributes
differed 14.9-fold against 1.4-fold for swing weighting; among 307 lung cancer
patients, @veldwijk2024comparing likewise found DCE weights more spread out.

**Adaptive elicitation and decision aids.** The "patient preference
diagnostic" of @sepulveda2023patient uses latent preference classes from an
earlier survey as a prior and asks adaptive choice questions to place a
patient in one of the known preference phenotypes. For first-time anterior
shoulder dislocation with four classes, the posterior class probabilities
reached 87% to 89% after a sequence of two questions; these are simulation
results only. A Cochrane review of 209 randomized trials with 107,698
participants [@stacey2024decision] found that *decision aids* probably improve
the congruence between informed values and the choice made (relative risk
1.75, 95% confidence interval 1.44 to 2.13; 21 studies; moderate certainty),
with no difference in decision regret. And measured preferences can drift for
reasons other than a change in what is measured: *response shift* is observed
change that is not fully explained by change in the target construct
[@vanier2021response].

**What it means for PBO.** This is one of the chapter's strongest links; the
following are inferences. Population latent classes as a prior and two
adaptive questions translate into a mixture prior over utility functions
learned from earlier users: early queries identify the class, later ones
refine within it (@sec-many-users), though the accuracy with real patients is
unknown. Stated choices predict actual uptake well but refusal poorly, so the
"winner" of a PBO session should be validated in real use before it is
treated as the user's real choice. Pairwise PBO is a choice-based method, so
the trade-offs it learns may be more extreme than direct ratings would give
(14.9 against 1.4). About 76% identical binary health choices after two weeks
gives a realistic noise floor for repeated pairwise judgments. Values
congruence and decision regret can evaluate PBO as a decision aid, beside
simulated regret (@sec-rec-evaluation). In long-term PBO for assistive
devices (@sec-health), drift may be response shift rather than a
change of preference, which a time-varying kernel alone cannot tell apart.
And clinical PBO should record medication status and timing as covariates. We
found no health DCE or shared decision-making study that uses Gaussian process
PBO or an acquisition function.

## Common claims, checked {#sec-ds-claims}

@tbl-ds-claims collects the claims from design, the senses, art, and health
that were checked in this chapter.

::: {.table #tbl-ds-claims title="Claims from design, sensory science, art, and health, and what the sources say."}
| Claim | What the sources say | Section |
|---|---|---|
| Color completely overrides taste and smell in wine, with devastating consequences for preference data | color biases verbal description and replicates; the task was choosing descriptors, and tasters still noticed a difference | @sec-ds-food |
| Prospect-refuge theory is established | prospect supported in 19 of 31 tests (61%), refuge in 8 of 27 (30%); refuge supported mainly in landscapes | @sec-ds-architecture |
| The Kano model shows that deep needs are stable | only 39% of "expect" answers repeated within seconds | @sec-ds-design |
| The more fluent, the more liked | true for fast judgments; under deliberate processing, disfluent designs can win through interest | @sec-ds-aesthetics |
| Liking follows an inverted U in complexity | at group level; in one data set it is a mixture of rising and falling individuals | @sec-ds-aesthetics |
| SSRIs change moral preferences (Crockett et al., *Science* 2010) | *PNAS* 2010 (citalopram); *Science* 2008 is tryptophan depletion; the substance holds | @sec-ds-health |
| 13.6% of levodopa patients develop impulse control disorders (Voon et al., *Annals of Neurology* 2006) | Voon et al. is *Neurology*: 13.7% on dopamine agonists; 13.6% is Weintraub et al. 2010; mainly agonists, with levodopa use also independently associated | @sec-ds-health |
:::

## Methods PBO can adopt directly {#sec-ds-adopt}

@tbl-ds-adopt gathers what in this chapter and the previous one is solid
enough to use now, without new theory, in the order of building a PBO system;
@sec-social-sciences adds controls for evaluation and deployment, and
@sec-recommendations turns all of it into advice. The third column is the
book's inference.

::: {.table #tbl-ds-adopt title="Methods PBO can adopt directly, from measuring the person to evaluating the result."}
| Method | Evidence | What changes in PBO (inference) |
|---|---|---|
| Placebo pairs: the same option twice (@sec-ds-placebo) | @omahony2017evolution; @halim2020paired; @xia2020paired | calibrate the noise or lapse term with the answers to identical pairs |
| A "no preference" option | @omahony2017evolution; @quintana2025global | add a tie likelihood (@sec-ties) instead of reading indifference as a random choice |
| "Cannot decide" modeled apart from "about the same" (@sec-nat-incomplete) | @ok2022indifference; @cettolin2019revealed | distinguish indifference, indecision, and experimentation |
| Noise scale fitted per feedback type (@sec-nat-information) | @ghosal2023effect | estimate noise separately for pairs, sliders, and ratings |
| Position, default, and choice-history terms (@eq-nat-prior-logit) | @matejka2015rational; @enisman2021choice | turn tilts near indifference into an estimable signal; balance sides and order |
| Diverse comparisons not derived from the incumbent (@sec-ds-design) | @wadinambiarachchi2024effects; @leahy2020design | add diverse candidates to galleries, and measure fixation |
| Latent-class prior with a short adaptive diagnosis (@sec-ds-health) | @sepulveda2023patient (simulation) | spend early queries on identifying the user's preference class |
| Per-person models where taste is individual (@sec-ds-architecture) | @vessel2018stronger; @brielmann2024modelling | per-person surrogates for buildings and art; population data for faces and landscapes |
| Comparison-graph and Hodge diagnostics (@sec-nat-graph-spectra, @sec-nat-hodge) | @hendrickx2019graph; @jiang2011statistical | check how well utilities are anchored, and the cyclic share against chance |
| Test-retest benchmarks | @nijman2022stability; @xie2022discrete | flag posteriors far more confident than an ICC of 0.6 to 0.7 or 76% repeated choices |
| Values congruence, decision regret, and validation in use | @stacey2024decision; @quaife2018well; @zhang2025prediction | report human outcomes beside simulated regret; validate the "winner" in real use |
:::

If a study can afford only a few changes, five have the best ratio of
evidence to cost, and none needs new software beyond a likelihood with ties
(inference):

1. Mix identical pairs into the session, a few percent of the queries, and
   report the false-preference rate.
2. Offer "no preference" (and, where it matters, "cannot decide") and model it.
3. Fit the noise scale to this person and this interface, rather than fixing it.
4. Randomize a control group to a random or balanced query order, and retest
   some early pairs at the end or a few days later.
5. Report a human outcome, such as endorsement at retest, values congruence,
   or real use, next to regret.

## Settled, contested, missing {#sec-ds-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** Consumers state preferences between identical samples, which is
why sensory science uses placebo pairs and a "no preference" option. System-supplied examples, including the outputs of AI image
generators, increase fixation and reduce the diversity of ideas. Taste for
faces and landscapes is shared, taste for buildings and artworks individual,
with equally reliable ratings where the two were compared. Individual models
of liking far outperform population averages for individualized content (0.65
against 0.01). Choice-based and direct weighting give different trade-offs for
the same people (14.9 against 1.4). Decision aids probably improve the
congruence of choices with values. Color biases the verbal description of
wine, replicably.

**Contested.** Whether the inverted U describes individuals or only groups.
Whether repetition raises or lowers liking. Whether a handoff from humans to
algorithms helps in design driven by taste, as it does in expert search with
an objective metric. How well stated choices predict real refusals
(specificity 34% to 52%). Whether co-evolution is specific to design.

**Missing.** A measurement of fixation within a PBO session, against a random
gallery. Test-retest consistency of designers' pairwise preferences across
sessions. PBO with human sensory panels for food or flavor. A measurement of
how much presentation changes preference rather than description. A pairwise
loop with residents driven by an acquisition function for public space. A
health DCE or shared decision-making study that uses Gaussian process PBO. An
observer-state model, such as Brielmann and Dayan's, used as the
non-stationarity of a PBO surrogate.
:::

## Exercises {#sec-ds-exercises}

::: {.exercise #exr-ds-tau}
In a pilot study, participants state a preference on 30% of placebo pairs.
Using the derivation in @sec-ds-placebo, what tie threshold $\tau$ does a
Thurstonian model infer, in units of the noise $\sigma$? A second interface
lowers the false-preference rate to 10%. What changed in the model's terms, and
what might have changed in the interface?

::: {.solution}
With $r = 0.3$, $\tau/\sigma = -\sqrt{2}\,\Phi^{-1}(0.15) \approx \sqrt{2} \times 1.036 \approx 1.47$.
With $r = 0.1$, $\tau/\sigma = -\sqrt{2}\,\Phi^{-1}(0.05) \approx \sqrt{2} \times 1.645 \approx 2.33$.
In the model, either the threshold for reporting no preference rose or the
perceptual noise fell, relative to each other; placebo pairs alone identify
only the ratio. In the interface, the "no preference" option may have become
more prominent, or the order changed (Halim et al. found that placing the
placebo pair after the target pair raises "no preference" answers). To
separate the two explanations, a study also needs pairs that differ by known
amounts.
:::
:::

::: {.exercise #exr-ds-render}
A PBO system renders each design with a random texture seed, so the same
parameters never look exactly the same twice. What happens to placebo pairs
in this system, and what does @xia2020paired suggest about the consequence for
the estimated false-preference rate?

::: {.solution}
Two renderings of the same parameters are no longer identical stimuli, so a
"placebo" pair contains a real, if small, perceptual difference. Answers to it
mix false preferences with genuine discriminations of the texture, and the
estimated false-preference rate is biased; Xia et al. found exactly this when
samples of the same food product varied. The fix is to render placebo pairs
with the same seed, or, if variation between renderings is part of the
product, to measure it separately with pairs of renderings of the same design.
:::
:::

## Further reading {#further-reading .unnumbered}

- @omahony2017evolution tells the history of paired preference testing, from
  forced choice to placebo pairs and $d'$, the most directly transferable
  method in this part of the book.
- @christensen2012estimation shows how the Thurstonian model of a preference
  test with a "no preference" option becomes a cumulative probit model.
- @vessel2018stronger separates shared from individual taste across domains.
- @brielmann2022computational is the computational theory of aesthetic value
  that ties liking to the observer's changing state.
- @stacey2024decision is the Cochrane review of decision aids, whose outcome
  measures PBO evaluations can borrow.
