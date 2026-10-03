---
status: done
synopsis: "What social, motivational, consumer, emotional, moral, personality, and other branches of psychology found between 2017 and 2026 about stable preferences: which famous effects failed to replicate, why broad dispositions are stable while single choices are not, and what this implies for modeling a person's comparisons."
---

# Social, Affective, and Developmental Psychology {#sec-social-psych}

@sec-judgment examined the seconds in which a person decides between two
options. This chapter turns to the branches of psychology that study people
over longer spans and in richer settings: under the eyes of others, in pursuit
of goals, as consumers, in moods, facing moral trade-offs, across the
lifespan, and in rooms and cities. Each has a view on whether a person carries
a stable preference that a sequence of comparisons could find.

Three findings run through them. Several famous effects once used to argue
that answers depend on one another did not replicate in multi-lab tests, or
shrank to near zero after correction for publication bias. Broad dispositions
that people report about themselves are quite stable, while incentivized
choices and single judgments are the least stable measures. And the evidence
closest to preferential Bayesian optimization (PBO) comes from moral
preference elicitation: when people answer the same pairwise question in
different sessions, 6% to 20% of the answers flip, and in simulations in which
preferences are unstable or the model class is wrong, actively chosen queries
can do no better than random ones. @sec-sp-pattern assembles the observation model these
results point to; the yardsticks used throughout are glossed in
@sec-jdm-assumptions.

## Social psychology {#sec-sp-social}

Social psychology contributes claims about choice, about attitudes people
cannot report, and about the influence of others. **Cognitive dissonance
theory** holds that people feel discomfort when their actions and attitudes
conflict and reduce it by changing the attitude, and it has long been used to
explain why the chosen option gains after a choice (@sec-jdm-choice). The
*dual attitudes* model holds that people can have an implicit and an explicit
attitude toward the same object at once [@wilson2000model], so a user might
have "two true preferences". And Asch's conformity experiments, usually
summarized as showing that about 75% of participants went along with a wrong
majority at least once [@asch1956studies], and *information cascades*, in
which people copy earlier choices instead of using their own information
[@bikhchandani1992theory], suggest that showing a user other people's ratings
may create a false consensus.

### Choice changes preference, from infancy {#sec-sp-choice}

Beyond the artifact-free meta-analysis with $d = 0.40$, 95% confidence
interval $[0.32, 0.49]$ (@sec-jdm-choice),
@silver2020not found in 7 experiments ($N = 189$) that preverbal infants avoid
the option they did not choose, ruling out novelty and prior attitudes: choice
shapes preference before a mature self-concept exists to be defended. The
other classic route to dissonance did not replicate. In the
*induced-compliance paradigm*, people write an essay against their own view,
freely or under instruction, and the theory predicts more attitude change when
they feel they chose to. Across 39 laboratories in 19 countries
($N = 4{,}898$), @vaidis2024multilab found no difference in attitudes between
the high-choice and low-choice conditions, although writing a
counter-attitudinal essay did change attitudes relative to writing a neutral
one. This favors sequential sampling and revaluation (@sec-jdm-choice) over
classic dissonance as the mechanism.

### Implicit attitudes and conformity {#sec-sp-implicit}

Implicit measures, such as the Implicit Association Test, which infers
attitudes from how quickly people pair concepts, predict poorly for
individuals. Across 217 reports ($N = 36{,}071$), implicit and explicit
measures each made a unique contribution to predicting intergroup behavior,
$\beta = .14$ and $.11$, with highly heterogeneous correlations
[@kurdi2019relationship]. A network meta-analysis of 492 studies found that
implicit measures can be changed, but only weakly ($|d| < .30$), and implicit
change did not mediate explicit or behavioral change [@forscher2019meta]. The
temporal stability of implicit measures (weighted mean $r = .54$) was lower
than that of explicit ones ($r = .75$) [@gawronski2017temporal]. Conformity
replicated: in an Asch replication with five confederates ($N = 210$), the
error rate on the standard line-judgment task was 33%, and monetary incentives
lowered it to 25% [@franzen2023power].

### Social influence as belief updating {#sec-sp-influence}

Social influence now reads as belief updating weighted by uncertainty. In a
large sample of people aged 14 to 24, the tendency to adopt others'
preferences in *delay discounting* (choosing between smaller rewards sooner
and larger ones later) came from greater uncertainty about one's own
preferences, and this uncertainty, and with it susceptibility to influence,
declined over 1.5 years [@reiter2021preference]. Artificial intelligence is
now one of the sources: while people interacted with a biased AI system, the
share of arrays of faces they classified as more sad than happy rose from
49.9% at baseline to 56.3% ($d = 0.84$) [@glickman2025human].

### What this means for the observation model {#sec-sp-social-implications}

- **Keep a choice-history term, with a revaluation mechanism.** It can predict
  the size of the drift from the difficulty of the choice and the response
  time, both of which an interface can record (inference).
- **Do not show others' choices early.** Other people's choices, popularity,
  or a system-endorsed default should not appear early in a session, when the
  user's own uncertainty, and with it susceptibility, is highest (inference).
  @sec-many-users discusses population models that pool people without
  showing one person's answers to another.
- **Preference uncertainty is measurable.** Inconsistency on repeated pairs
  estimates it, as a noise scale and as a warning that the person is open to
  influence (inference).
- **Implicit measures are not a second preference channel.** Response time is
  the better-supported auxiliary signal (inference), as @sec-neuro-ddm
  reports.

## Motivation and goals {#sec-sp-motivation}

A hierarchical view of preference holds that ultimate preferences are stable
while instrumental ones are constructed in context. In control theory,
behavior is a hierarchy of goals, with higher goals more abstract and slower
to change [@carver1982control]; in **self-determination theory**, needs for
autonomy, competence, and relatedness drive motivation [@deci2000what]; and
terminal values are said to be stable [@schwartz1992universals]. If ultimate
goals are stable, there is something stable for PBO to find.

### Habits, needs, and values {#sec-sp-goals-evidence}

How habits relate to goals remains unresolved: @wood2022habits argue that
habits operate independently of goals, yet in human laboratories
@wit2018shifting report five failed attempts to induce habits by overtraining.
Value hierarchies are stable for most people: among early adolescents over two
years, for 75% of people the value hierarchy at the two times correlated at
least .85, and for 5% at most .12 [@vecchione2020stability]. But value is
relative to a goal. When participants had to choose the worst option instead
of the best, behavior and the neural activity usually read as value were
dominated by *goal congruency*, how well an option served the current goal,
rather than expected reward [@fromer2019goal].

### What this means for the observation model {#sec-sp-motivation-implications}

- **The utility is relative to the task framing.** Choosing what you like,
  what suits a client, or which to rule out are different goals, so
  instructions should be held constant and recorded, and "which is worse?"
  should not share a likelihood with "which is better?" without a test
  (inference).
- **Response habits are nuisance, not preference.** Always choosing the left
  option or keeping the incumbent belongs in person-specific nuisance terms
  (inference).
- **Autonomy gives a reason to bound influence**, a concern
  @sec-phil-autonomy develops (inference).

## Consumer psychology {#sec-sp-consumer}

Consumer research holds that making choices, and choosing repeatedly, make
preferences more stable [@hoeffler1999constructing], and that early
experiences strongly predict final preferences [@hoeffler2006path], so the
initial candidates of a session would decide what *crystallizes*. **Choice
overload** is the claim that too many options harm choice: in a famous field
study, 3% of shoppers bought jam when 24 varieties were on display against
30% when there were 6 [@iyengar2000choice], but a meta-analysis of 63
conditions found a mean effect near zero with large variance
[@scheibehenne2010can].

### Choice overload and crystallization {#sec-sp-consumer-evidence}

Choice overload is real but highly conditional: a multilevel multivariate
meta-analysis showed that the effect varies greatly across six dependent
variables and four moderators, which interact [@mcshane2018multilevel]. When
people chose from 6, 12, or 24 items, activity in the striatum and anterior
cingulate cortex followed an inverted U with its peak at 12, the number
participants also judged about right, and the pattern vanished when people
only browsed without choosing [@reutskaja2018choice]. We found no direct
replication of preference crystallization. In a repeated discrete choice
experiment with eye tracking, the latent preferences of unstable respondents
did not differ from those of stable ones, and instability reflected the
difficulty of choosing when utilities were close [@fraser2021preference].

### What this means for the observation model {#sec-sp-consumer-implications}

- **Inconsistency is a noise scale, not a user type.** It should rise near
  indifference and for complex options (inference).
- **Galleries may have a best size, but not a universal one.** A multi-option
  interface (@sec-gallery-projective) may have an interior optimum that
  depends on whether the user must choose or may browse; 12 is a result from
  one domain (inference).
- **Randomize and record the initial designs.** Early experience and choice
  both shape where a session ends, so randomizing the starting designs across
  users, as the starting image settings in the photo-enhancement case study
  (@sec-cs-photo) could be, separates what the optimizer did from what the
  user preferred (inference).

## Emotion and hedonic psychology {#sec-sp-emotion}

Research on emotion supplies three claims. **Hedonic adaptation**: lottery
winners were famously found to be no happier than others
[@brickman1978lottery], so early preference pairs might expire. **Feelings as
information**: people consult their current mood when judging, as when
ratings of life satisfaction depended on the weather [@schwarz1983mood], so a
session might need an "emotional cool-down". And dual-process theories hold
that fast forced choices capture intuitive preferences.

### Three revisions {#sec-sp-emotion-evidence}

All three need revision. A preregistered analysis of Swedish lottery players
5 to 22 years after winning found that life satisfaction rose and stayed
higher for more than a decade with no sign of fading, while the effects on
happiness and mental health were significantly smaller [@lindqvist2020long];
across 18 life events, affect adapted to every positive event within two
years, but *evaluative* well-being kept lasting benefits of financial gains
and retirement [@kettlewell2020differential]. In nine direct and conceptual
replications with larger samples ($N$ from 118 to 401), the effect of mood on
judgments of life satisfaction was mostly not significant, and much smaller
than earlier results when it was [@yap2017effect]. And in 21 preregistered
replications pooled, the difference in contributions to a common project
between people who had to decide quickly and people forced to wait was
$-0.37$ percentage points in an intention-to-treat analysis (which compares groups as assigned), against 8.6
percentage points in the original data, and 65.9% of the time-pressure group
did not meet the time limit [@bouwmeester2017registered].

### What this means for the observation model {#sec-sp-emotion-implications}

- **Incidental mood is a weak noise source.** Emotional cool-downs and
  mood-timed exploration lack support (inference).
- **Two forces act on a repeated option, in opposite directions.**
  Choice-induced revaluation raises the incumbent that keeps winning, while
  the decline in enjoyment with repeated consumption [@galak2018properties]
  lowers it; the sign of a drift term should be estimated, not fixed
  (inference).
- **Fast forced choices do not reveal an intuitive self** (inference).
- **Satisfaction at the end of a session measures early affect.** Liking now
  and fitness for purpose diverge over time, so a delayed follow-up is
  needed (inference), as @sec-rec-evaluation recommends.

## Moral psychology {#sec-sp-moral}

Moral psychology offers claims about the structure of preference and the most
direct evidence on repeated pairwise elicitation. **Sacred values** resist
trade-offs: a proposal to trade a sacred value for money provokes anger
[@tetlock2000psychology], from which comes the argument that sacred values
should be hard constraints in PBO. A second argument holds that successive
moral answers compensate for one another. In **moral licensing**, a good deed
licenses a later lapse [@monin2001moral]; in *self-concept maintenance*,
honest people cheat only as much as lets them keep seeing themselves as
honest, and reminders of morality reduce cheating [@mazar2008dishonesty].

### Honesty, licensing, and retractions {#sec-sp-honesty}

In 25 direct replications of self-concept maintenance ($N = 5{,}786$), the
primary meta-analysis (19 replications, $n = 4{,}674$) found $d = -0.04$
against an original $d = 0.48$ [@verschuere2018registered]. The *Journal of
Marketing Research* issued an expression of concern about the 2008 paper in
2024 [@research2024expression], and a 2012 paper from the same group of
authors, on signing an honesty pledge at the top of a form, has been
retracted [@shu2012retracted].

Moral licensing does not hold in the within-person form that the compensation
argument needs. @rotella2026observation pooled 115 experiments
($N = 21{,}770$). The uncorrected multilevel estimate was Hedges' $g = 0.21$
($g$ is Cohen's $d$ with a small-sample correction), but a bias-corrected
robust Bayesian meta-analysis gave an estimate near zero ($g$ between $-0.08$
and $-0.02$), with very strong evidence of publication bias. The effect
depended on being watched: $g = 0.65$ when participants were explicitly
observed, against $g = 0.13$ when they were not, which the authors read as
an interpersonal, reputational effect. A registered replication of the original
moral credentials study ($N = 932$) did not support a consistent effect
[@xiao2024licensing]. @fig-sp-effect-sizes sets these results beside the
choice-induced change that held.

```{figure}
//| figure: jdm-effect-sizes
//| label: fig-sp-effect-sizes
//| fig-cap: "Effects once used to argue that answers depend on each other, on one scale. Open dots are original or uncorrected estimates, filled dots later, bias-corrected, or subgroup estimates, bars a reported interval or range. Click a row or move the slider: the lower panel draws two normal distributions whose means differ by d, a property of the normal model rather than data."
set: social
d: 0.48
```

Things to try:

- At the original moral-reminder effect, $d = 0.48$, the distributions overlap
  81% and a person from the higher group outscores one from the lower group
  63% of the time. At the replication's $-0.04$ they overlap 98%.
- Licensing under observation, $g = 0.65$, overlaps 75%; without it,
  $g = 0.13$, 95%.

### Sacred values and dilemmas {#sec-sp-moral-structure}

Sacred values are neither universal nor fixed. Seeing a sacred value used for
profit lowered its sacredness and resistance to trade-offs (7 studies,
$N = 2{,}785$) [@ruttan2021instrumental]. When a penalty for taboo trade-offs
was added to a random utility model in discrete choice experiments, a latent
class analysis showed some groups treating the trade-offs as taboo and others
not, and ignoring the aversion overestimated the willingness to pay for saving
lives by a factor of about 3.5 [@smeele2025taboo]. And hypothetical moral
judgments do not predict real behavior: answers to a trolley-style dilemma
did not predict whether people would actually deliver an electric shock to
one mouse to spare five [@bostyn2018mice].

### How stable are moral answers? {#sec-sp-moral-stability}

Aggregated utilitarian judgments are highly consistent over time and across
eight contexts [@helzer2017once]. Single judgments are not: between two waves
6 to 8 days apart, an average of 49% of participants changed their rating of a
given sacrificial dilemma, and only about 17% of the changes came from people
who said they had changed their mind [@rehren2022stable]. Two studies asked
the same pairwise question again and again. Asked ten times over two weeks
which of two patients should receive one available kidney, people changed
their answers on controversial scenarios about 10% to 18% of the time, more
often for slower and harder choices [@boerstler2024stability]. In a larger
study, more than 400 participants answered pairwise kidney-allocation
comparisons in 3 to 5 sessions; on average 6% to 20% of their answers to the
same scenario changed, and the predictive performance of simple models fell
with instability and over time [@keswani2026moral]. Take a short version of
the test in @fig-sp-repeat.

```{figure}
//| figure: sp-repeat
//| label: fig-sp-repeat
//| fig-cap: "The same allocation question, asked twice. Answer eight questions about which of two hypothetical patients should receive one available kidney; round 2 asks them again in a new order, sides swapped. The summary shows how many answers changed, beside the rates found when repeats were days apart. Your answers stay on this page; eight questions minutes apart are a demonstration, not a measurement."
```

The result most relevant to PBO concerns active learning, choosing each next
question to be as informative as possible, as acquisition functions do
(@sec-choosing-pairs). @keswani2024pros identify its three premises:
preferences are stable and unaffected by question order, the hypothesis class
is right, and noise is limited. In simulations that violate them, active
learning did as well as or worse than random questions in some settings, and
stayed worthwhile only when instability and noise were small and the
preferences could be approximately represented by the hypothesis class.

### Noise or change? {#sec-sp-noise-or-change}

A flip rate of 6% to 20% can come from a fixed preference answered with noise,
or from a preference that moves between sessions, a possible "moral change"
in the words of @keswani2026moral. Noise should be averaged away; change
should be tracked. Let the long-run utility difference between two options be
$\Delta$, shifted in each session by $\delta \sim \N(0, \tau^2)$, with probit
response noise of scale $\sigma$ per option (@sec-thurstone). Two answers in
one session share $\delta$; answers in two sessions do not. With
$p_\delta = \Phi\big((\Delta + \delta)/(\sqrt{2}\,\sigma)\big)$, the
probabilities that the two answers differ are

$$
P_{\text{within}}(\Delta) = \E_\delta\!\left[2\,p_\delta\,(1 - p_\delta)\right],
\qquad
P_{\text{across}}(\Delta) = 2\,\bar p\,(1 - \bar p),\quad
\bar p = \Phi\!\left(\frac{\Delta}{\sqrt{2\sigma^2 + \tau^2}}\right),
$$ {#eq-sp-flip}

since two independent sessions each say "yes" with average probability $\bar
p$. When $\tau > 0$, answers differ more often across sessions than within
one, as @fig-sp-flip shows.

```{figure}
//| figure: sp-flip
//| label: fig-sp-flip
//| fig-cap: "Noise or change? The probability that two answers to the same pair differ, against the long-run utility gap Δ, asked twice in one session (blue) or in two sessions (orange, dashed); σ is the response noise per option and τ the change between sessions. The side panel averages over pairs across the shaded range, beside the 6% to 20% reported in kidney-allocation studies. Within-session repeats ignore memory; the model is illustrative."
```

With $\tau = 0$ and $\sigma = 0.3$ the curves coincide, and the average flip
rate, about 16%, is inside the reported range: pure noise is enough. With
$\sigma = 0.05$ and $\tau = 0.4$ the average across sessions is again about
15%, but within a session it falls to about 3%. Only comparing a retest within
the session with one in a later session tells the two apart, which is why
@sec-jdm-mathpsych-implications recommends retesting either immediately or
much later.

### What this means for the observation model {#sec-sp-moral-implications}

- **Test acquisition functions against random queries.** In value-sensitive
  domains such as allocation, policy, and safety trade-offs, compare them with
  random queries on held-out repeated pairs, and keep a fixed share of random
  or repeated queries in every session (inference); the comparisons in
  @sec-acqf-empirical rarely include such a control.
- **Expect a floor on consistency.** A flip rate of 6% to 20% across sessions
  calls for session-level random effects, and convergence within one session
  should not be extrapolated across sessions (inference).
- **Replace hard constraints with a person-specific penalty** on trading a
  protected attribute, placed in a mixture over classes of users (inference).
- **Drop licensing, and check the evidence before citing it.** Compensation
  based on moral licensing does not belong in private sessions, since
  licensing appears only under observation; who can see the answers is then
  an experimental factor. Papers that justify a design choice with a
  behavioral effect should check its replication and retraction status
  (inference).
- **Hypothetical validation is not real validation** (inference). A PBO study
  in a moral domain, one of the gaps in @sec-sp-status, should validate against
  real decisions, not only against further hypothetical answers.

## Personality and lifespan development {#sec-sp-personality}

A widely cited meta-analysis found that the rank-order stability of
personality traits rises from .31 in childhood to .64 at age 30 and reaches a
plateau around .74 between ages 50 and 70 [@roberts2000rank], from which comes
the suggestion to treat data from middle-aged users as more reliable.

### What is stable, and from when {#sec-sp-personality-evidence}

A meta-analysis of longitudinal studies published since 2005, covering
rank-order stability (189 studies, $N = 178{,}503$) and mean-level change
(276 studies, $N = 242{,}542$), found that rank-order stability reaches a
plateau in early adulthood, with little evidence of further increase after
age 25 [@bleidorn2022personality]. Stated, trait-like preferences are quite
stable: in 1,507 adults who completed 39 measures of risk preference, stated
and behavioral measures correlated only weakly, but the stated measures formed
a general factor that was highly reliable over six months [@frey2017risk].
The Global Preferences Survey of 80,000 people in 76 countries found more
heterogeneity within countries than between them [@falk2018global].

### What this means for the observation model {#sec-sp-personality-implications}

- **Population priors help, but each user still has to be learned,** since
  heterogeneity within countries is large (inference); @sec-hci-population
  reports how population priors have fared in interactive systems.
- **Pair a few stated items with the comparisons.** Pairwise choices belong to
  the less reliable, behavioral class of measures (@sec-jdm-assumptions).
  Stated items can carry the stable component, for example through a Gaussian
  process prior mean that depends on them (inference; the implementation
  comes from the Gaussian process framework).
- **Report retest agreement across sessions,** since convergence within one
  session is weak evidence of a stable preference (inference).

## Evolutionary psychology {#sec-sp-evolutionary}

Sex differences in mate preferences, first reported across 37 cultures
[@buss1989sex], replicated in 45 countries ($N = 14{,}399$)
[@walter2020sex], but they say little about individuals. A registered report
($N = 10{,}358$, 43 countries) found that stated ideal-partner preferences
matched evaluations of partners in the overall pattern ($\beta = .19$), but
the trait-specific effects averaged only $\beta = .04$
[@eastwick2025worldwide]. A prior initialized from the importance a user
states for each attribute should therefore be discounted, or corrected from
revealed comparisons (inference).

## Developmental psychology and behavioral genetics {#sec-sp-developmental}

**Heritability** is the share of a trait's variation across a population that
is associated with genetic differences. In 9,169 Swedish twins, genetic
effects explained up to 54% of the variance in musical reward sensitivity
[@bignardi2025twin], while individual preferences for faces come mostly from
each person's unique environment [@germine2015individual]. Heritability does
not by itself set the strength of a prior; what a model needs is the split
between a component shared across people and a person-specific one, and the
weight on the shared part can itself be a person-level parameter
(inference). Since choice-induced change exists from infancy (@sec-sp-choice),
it cannot be avoided by recruiting more reflective users (inference).

## Sleep and circadian rhythms {#sec-sp-sleep}

A *morning morality effect*, in which people are more honest in the morning
[@kouchaki2014morning], is cited to suggest eliciting preferences at a
person's circadian peak. A conceptual replication with $N = 1{,}006$ found an
odds ratio of 1.04 (95% confidence interval $[0.93, 1.17]$), and a
meta-analysis found $d = 0.04$ [@zickfeld2024investigating], a 2024 preprint.
Self-reported sleep is better used as a covariate for the noise scale and the
lapse rate than as a signed bias on utility (inference).

## Neurodiversity {#sec-sp-neurodiversity}

In a preregistered adversarial collaboration, people who reported an autism
diagnosis learned much less about features irrelevant to the outcome, and the
reduction scaled with autistic traits across the whole sample, a dimensional
rather than categorical pattern [@benartzi2026autism]. Nuisance biases such as
position, order, and irrelevant visual features therefore call for
person-level nuisance parameters estimated from the interaction itself, which
adapts to each user without collecting any diagnostic information
(inference).

## Environmental psychology {#sec-sp-environmental}

The quantitative evidence for *prospect-refuge theory*, which says people
prefer places that offer a view while affording shelter, is inconsistent
[@dosen2016evidence]. When 798 participants rated 200 interior spaces, three
components, *coherence*, *fascination*, and *hominess*, explained 90% of the
variance, and the structure replicated in an independent sample ($n = 614$)
[@coburn2020psychological]; they could serve as interpretable objectives in a
multi-objective formulation (@sec-multi-objective) (inference). 81,630
volunteers made 1.17 million pairwise comparisons of street-view images from
56 cities [@dubey2016deep], data that could supply population priors for urban
design if kept specific to a culture or city (inference). @sec-ds-architecture
and @sec-ds-urban take up design for buildings and cities.

## A common pattern {#sec-sp-pattern}

Across the branches the same pattern recurs (@tbl-sp-stability): broad
self-reports are the most stable measures of preference, single choices and
incentivized behavior the least, and population regularities predict little
of one person's specific shape.

::: {.table #tbl-sp-stability title="Stability of preference by kind of measure, across the branches of this chapter and the last."}
| Domain | Broad, self-reported | Single choices, behavior, or specific matches |
|---|---|---|
| Risk | stated propensity: mean reliability over time 0.61 | behavioral measures such as lottery choices: 0.25 |
| Morality | aggregated utilitarian judgments consistent over time | 49% change a dilemma rating within days; 6% to 20% of pairwise answers flip across sessions |
| Partners | stated ideals match evaluations in overall pattern, $\beta = .19$ | trait-specific matching $\beta = .04$ |
| Attitudes | explicit measures: stability $r = .75$ | implicit measures: stability $r = .54$ |
:::

Taken together with @sec-judgment, these results suggest an observation model
of the following shape (inference). Let a person's utility in session $s$ be

$$
u_s(\vx) = m(\vx) + f(\vx) + d_s(\vx),
$$ {#eq-sp-hierarchical}

where $m$ is a population mean, informed by population data or by a few stated
preferences; $f$ is the person's stable component, the Gaussian process of
@sec-gp-preference; and $d_s$ is a session-level deviation with a small
variance, which lets the utility move between sessions without moving within
one. A comparison at step $t$ then follows

$$
\Prob(\vx \succ \vx') = \frac{\lambda_{\text{lapse}}}{2} + (1 - \lambda_{\text{lapse}})\,
\Phi\!\left(\frac{u_s(\vx) - u_s(\vx') + b\, c_t}{\sqrt{2}\,\sigma}\right),
$$ {#eq-sp-observation}

where $\lambda_{\text{lapse}}$ is a lapse rate (the probability of an answer
unrelated to the options), $c_t = \pm 1$ codes which option was on the left,
$b$ is a person-specific position bias, and $\sigma$ is a noise scale that may
depend on the person's uncertainty and on how close the options are. Repeated
pairs inform $\sigma$ and $\lambda_{\text{lapse}}$, counterbalanced positions
inform $b$, and repeats in later sessions inform the variance of $d_s$. No
preferential optimization method we found implements this model, and its
pieces have been tested separately, mostly outside optimization (inference).
Two practices complete it: a share of random queries as a control for the
acquisition function, and a delayed evaluation of the result, separate from
the session that produced it (inference).

## Settled, contested, missing {#sec-sp-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** Choice changes preference, from infancy on [@silver2020not]. The
induced-compliance route to dissonance did not replicate across 39 labs
[@vaidis2024multilab]. Moral reminders do not reduce cheating
($d = -0.04$ in the primary analysis of 25 direct replications)
[@verschuere2018registered], and moral licensing appears only under
observation [@rotella2026observation]. Conformity on simple judgments is still
about one in three [@franzen2023power]. Broad stated dispositions are stable
while single choices are not, and personality stability plateaus at about 25
[@bleidorn2022personality]. The same moral pairwise question flips 6% to 20%
of the time across sessions [@keswani2026moral]. Mood effects on judgments of
well-being are much smaller than once reported [@yap2017effect], and time
pressure does not reveal an intuitive preference
[@bouwmeester2017registered].

**Contested.** Whether habits run independently of goals. Whether choice
overload exists outside particular moderator combinations. Whether flips
across sessions are noise or genuine change in preference, which the flip rate
alone cannot decide (@fig-sp-flip). Whether stated and behavioral preferences
measure one construct.

**Missing.** A direct replication of preference crystallization. A test of
ultimate-goal stability and instrumental construction in the same design
task. Any use of PBO in a moral domain. A PBO study that compares its
acquisition function with random queries on held-out repeated pairs, models
session-level deviations, or estimates person-specific nuisance parameters.
Delayed evaluations of designs chosen by PBO, separate from the session.
:::

## Further reading {#further-reading .unnumbered}

- @keswani2024pros is a short, clear statement of the assumptions behind active
  preference elicitation and what happens when they fail; read it with the
  repeated-elicitation data of @boerstler2024stability and
  @keswani2026moral.
- @rotella2026observation is a model of how a bias-corrected meta-analysis can
  turn a famous effect into a conditional one.
- @frey2017risk explains why stated and behavioral measures of the same
  preference disagree, and which one is stable.
- @eastwick2025worldwide tests whether stated preferences predict evaluations
  of real people, a direct analogue of initializing a prior from stated
  attribute importance.
