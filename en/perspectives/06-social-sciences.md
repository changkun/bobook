---
status: done
synopsis: "Where tastes come from and who gets sampled, what happens when a system that measures preferences also shapes them (performative prediction, recommender feedback, EU rules on manipulative interfaces), what language, institutions, and expertise add, and which widely repeated claims from these fields do not survive a check of the sources."
---

# Social Sciences and the Humanities {#sec-social-sciences}

The previous chapters of this part looked inside one person. This chapter
steps outside. Sociology, anthropology, law, linguistics, education, and their
neighbors ask where a taste comes from, who is in the sample, what happens
when the system that measures a preference also shows people what to choose,
and what institutions that have collected human judgments for decades have
learned. Each question presses on an assumption of preferential Bayesian
optimization (PBO): in the formulation of @sec-pbo-problem, a person carries a
latent utility function that stays fixed during the session, and every answer
to "which is better?" is a noisy reading of it.

Three threads bear most directly on PBO. Machine learning has formalized the
old observation that measuring something can change it, under the names
*performative prediction* and *induced preference shift*
(@sec-soc-measuring). European Union law has begun to prohibit interface
designs that, in effect, distort users' autonomous decisions (@sec-soc-law).
And information retrieval has accumulated evidence that judges make preference
judgments faster and more consistently than graded ones
(@sec-soc-information). Several widely repeated claims from these fields turn
out to be wrong or overstated; @sec-soc-claims collects them. Disciplines with
weaker links share one section (@sec-soc-briefer).

## Where taste comes from {#sec-soc-origins}

If the latent utility is formed by class, culture, and life events, then a
model of one person's answers is also a model of their social position, and a
prior learned from other people carries those people's positions with it.

### The sociology of taste {#sec-soc-taste}

The classic claim is that taste is socially produced and stratified by class.
@bourdieu1984distinction explained it with *habitus*, durable dispositions,
acquired through upbringing and social position, that generate likings
without deliberate calculation, and *cultural capital*, familiarity with
legitimate culture that works like an asset. Hennion emphasized taste as an
active practice that amateurs work at [@hennion2001music; @hennion2007those];
reading this as saying that every preference query trains the user's taste is
an extrapolation, not Hennion's argument. Refining the class picture,
@childress2021genres found that higher-status people are more inclusive at the
level of genres and more exclusive at the level of specific objects such as
particular artists and works. Airoldi's *Machine Habitus* treats machine
learning systems as socialized agents whose training data steer how they
classify and whose outputs feed back into culture [as summarized by
@rodl2022airoldi], and @webster2021promise argued that personalization took
over the labor of curating music, which weakens one traditional way of gaining
distinction, by choosing for oneself.

**What it means for PBO.** In cultural domains such as music, fonts, or visual
style, a surrogate defined on concrete parameters cannot see openness at the
genre level, and one defined on genres cannot see exclusiveness at the object
level, so the level of the features is a substantive modeling decision
(inference). Priors built from population data or from a large language model
encode the taste of whoever supplied the training data, so an informative
prior (@sec-sw-priors, @sec-hci-population) should be audited for whose taste
it encodes (inference). And where part of a result's value comes from having
curated it oneself, interaction in which the designer leads is favored
(inference; compare @sec-hci-agency).

### Cultural psychology and WEIRD samples {#sec-soc-weird}

Citing an analysis of top psychology journals from 2003 to 2007,
@henrich2010weirdest reported that 96% of participants came from Western
industrialized countries, which house 12% of the world's population, and
called such societies Western, Educated, Industrialized, Rich, and Democratic
(WEIRD). Little has changed: in six top psychology journals from 2014 to 2018,
a little over 60% of authors and samples were American, and authors and
samples from outside the English-speaking world and Western Europe remained at
4% to 5% [@thalmayer2021neglected].

For aesthetic preferences, a 2025 paper at the annual meeting of the Cognitive
Science Society gives unusually broad data. @lee2025visual collected 401,403
preference judgments from 4,835 participants in 10 countries across shape,
curvature, color, musical harmony, and melody, sampling a two-dimensional
parameter space for each. Liking for symmetrical forms was consistent across
cultures; color preferences were consistent at the level of categories while
"ratio-like" preferences varied; and melody showed the largest cross-cultural
variation. The authors caution that online samples are exposed to global
media, so the results need replication in laboratories and small-scale
societies. Smell shows a similar split: when 225 people in 9 non-Western
communities ranked ten odors by pleasantness, the identity of the odorant
explained 41% of the variance in the rankings, individual variability 54%, and
culture only 6% [@arshamian2022perception] (the institution's press release
gives 235 participants [@sciencedaily2022people]).

**What it means for PBO.** The pattern Lee et al. report suggests a
hierarchical or multi-task Gaussian process, a shared component plus a group
deviation (@sec-many-users), with the expected size of the deviation depending
on the domain: small for symmetry-like visual features, larger for melody and
style; for smells, a model of the individual matters more than a cultural
prior (inference). Most PBO user studies recruit Western university
populations, so a finding such as "users prefer option A" is bound to that
sample (inference). Rating scales are known to be vulnerable to *response
styles*, culturally varying habits such as favoring the extremes or the middle
of a scale; whether pairwise comparisons are less affected is open, as we
found no study that tests it.

### Demography and the life course {#sec-soc-demography}

Life-course research [@elder1998life] and socioemotional selectivity theory,
which holds that people shift toward emotionally meaningful goals as they
perceive their remaining time as shorter [@carstensen1999taking], both predict
that preferences change systematically over a life. @schildberghorisch2018risk
summarized that risk preferences are persistent but only moderately stable,
and suggested treating a preference parameter as a distribution with a fairly
stable mean and systematic variance around it. Using Australian panel data,
@kettlewell2019risk found that financial changes, the birth of a child, and
the death of someone close shift risk preferences, with effects largest near
the event and fading afterward. A *test-retest correlation* measures how well
a measurement agrees with itself when repeated on the same people later; for
risk preference, the meta-analysis of @bagaini2025systematic estimates a
reliability of 0.61 for self-reported propensity to take risks and 0.25 for
behavioral measures such as lottery choices.

**What it means for PBO.** For personalization that runs over months or
years, such as tuning a hearing aid or an exoskeleton (@sec-health-audio,
@sec-health-exo; @sec-cs-exoskeleton works the exoskeleton case end to end), a
model with a stable component plus deviations triggered by events that then
decay fits this evidence better than a random-walk time kernel that lets
preferences drift anywhere (inference; @sec-theory-drift and
@sec-rec-nonstationary discuss time-varying models). For a single session, the
connection is weak.

## When measuring changes what is measured {#sec-soc-measuring}

This group contains the chapter's strongest links. Its disciplines study the
same loop from different sides: a system observes people's choices, acts on
what it learned, and in acting changes the choices it will observe next. A PBO
session is a small instance of that loop.

### Opinion dynamics and recommender feedback {#sec-soc-opinion}

In the *bounded confidence* model of opinion dynamics, agents move toward
others' opinions only when those opinions are within a confidence bound of
their own; with a large bound the population reaches consensus, with a medium
bound it polarizes, and with a small bound it fragments
[@deffuant2000mixing]. A PBO system serving many users should expect clusters
rather than one shared optimum.

Two famous empirical claims need care. The finding of @christakis2007spread
that obesity spreads through social networks, a person's chance of becoming
obese rising by 57% if a friend became obese, faces an identification
problem: *homophily* (similar people become connected) and *contagion*
(connected people become similar) leave the same trace in observational
network data, and @shalizi2011homophily proved that the two are generally
confounded there. And a summary in circulation says that on Facebook,
individual choice limited exposure to ideologically diverse content by 70% and
the ranking algorithm by 15%. Those numbers are not in the source:
@bakshy2015exposure, analyzing 10.1 million American users, found that
algorithmic ranking reduced cross-cutting content by 5% for conservatives and
8% for liberals, and individual choice by 17% and 6%: choice outweighed the
algorithm for conservatives, not for liberals.

**The theory since 2017 writes preference dynamics into the recommender's
objective.** @dean2022preference assumed that a user's preferences move toward
content they consume and enjoy and away from content they consume and dislike,
and proved that under these dynamics standard reward maximization is "an
almost trivial goal": a large class of simple algorithms achieves constant
regret. (*Regret*, from @sec-regret, is the utility lost by recommending what
the system recommended instead of the best option.) The more meaningful goal,
they argue, is to keep the user's preferences approximately stationary.
@carroll2022estimating pointed out that a recommender optimized over a long
horizon has an incentive to push users' preferences toward ones that are
easier to satisfy, and proposed constraining the system to a trust region of
"safe shifts", for example the shifts that would have happened without the
system. @kleinberg2023challenge proved that when users' preferences are
inconsistent, a platform can raise engagement while lowering user utility, and
@carroll2024aib and @williams2025targeted showed that a static preference
objective implicitly rewards influencing the user and that optimizing on user
feedback can learn targeted manipulation.

The mechanism behind the "almost trivial goal" is simple enough to watch.
@fig-soc-moving-target puts a simulated person with plastic preferences in
front of a deliberately simple optimizer and scores the result twice.

```{figure}
//| figure: soc-moving-target
//| label: fig-soc-moving-target
//| fig-cap: "Who moved, the person or the system? A simulated person whose ideal design is x = 0.75 answers 40 incumbent-versus-challenger comparisons with probit noise. After each answer, their ideal moves a fraction η toward the design they chose, a one-dimensional version of the preference dynamics assumed by @dean2022preference. Top: one session, with the person's utility before and after, the system's starting design (hollow circle), and its final pick (star). Bottom: regret of the system's current pick, averaged over 200 sessions and judged two ways: by the person's preferences at that moment (blue, what a benchmark would report) and by the preferences they started with (magenta). Dashed gray is the same optimizer facing a person whose preferences do not move. The optimizer is a simple local search, not a Gaussian process, and every value is illustrative; @sec-soc-status names the measurement that would set η."
```

Things to try:

- Press *Fixed preferences*. The blue and magenta curves coincide, which is
  the world PBO assumes; the simple local search is still far from the ideal
  after 40 comparisons, with mean regret 0.43.
- Return η to 0.1. The blue curve falls to about 0.014: by the benchmark, the
  same weak optimizer now looks excellent, much better than against a fixed
  person, while judged by the preferences the person arrived with its regret is
  0.76. The person moved about 0.36 toward the system, more than the system
  moved toward the person.
- Set the challenger to *Anywhere*. The system reaches the person's region
  before the person has moved far, and the gap between the two scoreboards
  shrinks (about 0.006 against 0.14 at η = 0.1). How far the person moves
  depends on which queries the system asks.
- Set the starting design to 0.75. When the system starts where the person
  already is, there is little to steer.

Does this happen outside a simulation? The evidence splits by design and
timescale (@tbl-soc-field). Simulations, observational studies, and short
experiments find homogenization, including stories that became more similar
to each other when writers used ideas from generative AI
[@doshi2024generative]. Large randomized field experiments over weeks to
months mostly find small or null effects on attitudes [@guess2023social],
including eight attitude measures that were *preregistered*, declared before
the data were collected, which guards against searching for a positive result
[@nyhan2023like].

::: {.table #tbl-soc-field title="Does a recommender change what people want? The evidence by design and timescale."}
| Study | Design | Scale | Finding |
|---|---|---|---|
| @chaney2018algorithmicb | simulation | simulated users | training on confounded feedback homogenizes behavior without raising utility |
| @anderson2020algorithmic | observational | Spotify listeners | algorithm-driven listening goes with lower consumption diversity |
| @doshi2024generative | experiment | story writers | AI ideas raise individual creativity, make stories more alike |
| @glickman2025human | experiments | human-AI interaction | biased AI amplifies human bias more than biased humans |
| @huszar2022algorithmic | platform audit | seven countries | mainstream right amplified more than left in six of seven; attitudes not measured |
| @guess2023social | randomized field experiment | Facebook and Instagram users, three months | reverse-chronological feeds changed time spent and content seen; key attitudes such as polarization did not change significantly |
| @nyhan2023like | randomized field experiment | 23,377 users | like-minded exposure cut by about a third; no effect on eight preregistered attitudes |
| @hosseinmardi2024causally | counterfactual bots | YouTube | recommendations moderate on average; users' own preferences dominate |
| @aridor2026recommendation (preprint; authors affiliated with Netflix) | randomized experiment | 8.5 million Netflix subscribers | better recommendations lower concentration by 5.7% (recommendations) and 1.2% (plays) |
:::

**What it means for PBO.** Dean and Morgenstern's result means that low regret
can be achieved by moving the user's preferences, so regret or speed of
convergence alone is not a sufficient criterion of success. A study should
also report how much the preference moved during the session, against a
control group with a random or balanced query order and a delayed retest of
options the person rejected early, the PBO counterpart of Carroll et al.'s
"safe shift" baseline (inference). The timescales differ, though. Platform
studies concern weeks to months, while a single PBO session is closer to
revaluation within a session, such as the drift in
subjective values during a single experimental session that
@zylberberg2024value traced to how value is constructed during deliberation.
The link is therefore strong for personalization over months and weaker for a
one-off design session (inference). With many users, the bounded confidence
results favor mixture or clustered utility models, and argue for not showing
users what others chose while eliciting their preferences (inference).
@sec-adj-recsys follows the same loop in the recommender systems literature.

### Science and technology studies {#sec-soc-sts}

Science and technology studies (STS) examines how technical artifacts and
social order shape each other. One story often used to show that artifacts
carry politics, that Robert Moses built low overpasses on the parkways to
Jones Beach so that buses could not reach it [@winner1980artifacts], is
disputed [@joerges1999politics] and should not be retold as fact without that
caveat. The general point survives: instruments of measurement take part in
producing what they measure, as traders' use of option-pricing theory helped
make market prices conform to the theory [@mackenzie2003constructing].

**The most important development since 2017 is that machine learning gave
this idea a formal theory.** @perdomo2020performative called a prediction
*performative* when it influences the outcome it aims to predict, and defined
a stability notion for it.

::: {.definition #def-soc-performative title="Performative stability"}
Let a model with parameters $\theta$ be deployed, and let $\mathcal{D}(\theta)$
be the distribution of data $Z$ that the world produces in response. With a
loss $\ell(Z; \theta)$, the model $\theta_{\text{PS}}$ is *performatively
stable* if

$$
\theta_{\text{PS}} = \argmin_\theta \; \E_{Z \sim \mathcal{D}(\theta_{\text{PS}})} \, \ell(Z; \theta).
$$ {#eq-soc-performative}

In words: retraining on the data that the deployed model itself induces
returns the same model.
:::

The natural algorithm is *repeated risk minimization*: deploy, collect the
data the deployment induces, retrain, and repeat. Perdomo et al. give
conditions that are both necessary and sufficient for it to converge to a
stable point whose loss is nearly minimal. Informally, the distribution map is
$\varepsilon$-*sensitive* if moving the parameters by a distance $d$ moves the
induced data distribution by at most $\varepsilon d$ in Wasserstein
distance; if the loss has
curvature at least $\gamma$ ($\gamma$-strongly convex) and a gradient that
changes at rate at most $\beta$ ($\beta$-smooth), repeated risk minimization
converges at a linear rate to a unique performatively stable point when
$\varepsilon < \gamma / \beta$, and it can fail to converge when
$\varepsilon \ge \gamma / \beta$. The world's sensitivity to the model must be
small relative to how sharply the loss pins the model down.
@hardt2025performative connected the concept to performativity in economics
and the social sciences and distinguished two mechanisms, *learning*
(predicting the world better) and *steering* (moving the world toward the
prediction). Asking about behavior also changes it: a meta-analysis of 116
tests of the *question-behavior effect* found that asking people about their
intentions or to predict their own behavior changes their later behavior
slightly, with Cohen's $d = 0.24$, a difference between groups in units of the
standard deviation [@wood2016impact].

**What it means for PBO.** This is the chapter's strongest formal link. A PBO
loop whose queries change the person's preferences is a performative
prediction problem: the surrogate's fixed point is the preference that is
stable under the queries the surrogate itself induces, which need not be the
preference that a procedure without influence would find. In
@fig-soc-moving-target, the local optimizer ends at such a point. Perdomo et
al.'s condition is a candidate criterion for whether the loop still converges
when preferences are co-constructed by the system and the person, and the
distinction between learning and steering can be used to evaluate PBO
directly (inference). The choice of success metric is itself performative:
regret against the final inferred utility, and the concentration of the
posterior, reward steering the person toward a predictable state, while
endorsement at a delayed retest and comparison with a random-order control
group are much harder to steer (inference; @sec-rec-evaluation).

### Law {#sec-soc-law}

Legal templates such as Alexy's proportionality test with its weight formula
[@alexy2024abwagung] invite analogies with preference aggregation, but the
substantive development is European Union law on manipulation and
*dark patterns*, interface designs that steer users into choices they would not
otherwise make, such as pre-ticked boxes or cancellation flows that are harder
than sign-up.

In the Planet49 case (C-673/17, 1 October 2019), the Court of Justice of the
European Union held that a pre-ticked checkbox does not constitute valid
consent, which requires active behavior [@observatory2019court]. Article 25(1)
of the Digital Services Act (Regulation (EU) 2022/2065) provides that
"providers of online platforms shall not design, organise or operate their
online interfaces in a way that deceives or manipulates the recipients of
their service or in a way that otherwise materially distorts or impairs the
ability of the recipients of their service to make free and informed
decisions" [@union2022digital]; paragraph 2 excludes practices already covered
by the Unfair Commercial Practices Directive or the General Data Protection
Regulation. Recital 67 defines dark patterns as practices that materially
distort or impair the user's ability to make autonomous and informed choices
"either on purpose or in effect" [@union2022regulation]. Article 5(1)(a) of the
AI Act (Regulation (EU) 2024/1689) prohibits AI systems that deploy subliminal
techniques, or purposefully manipulative or deceptive techniques, with the
objective or the effect of materially distorting behavior by appreciably
impairing a person's ability to make an informed decision, in a manner that
causes or is reasonably likely to cause significant harm; Article 5(1)(b)
covers the exploitation of vulnerabilities due to age, disability, or a
specific social or economic situation. The prohibitions apply from 2 February
2025 [@union2024regulation]; the Commission's guidelines on them, published on
4 February 2025, are non-binding [@commission2025commission]. A Digital
Fairness Act covering dark patterns, addictive design, and unfair
personalization has not been adopted; the European Parliament's legislative
tracker, updated 20 June 2026, lists the proposal as expected in the fourth
quarter of 2026 [@parliament2026legislative].

The effects these rules target can be large. In the experiments of
@luguri2021shining, mild dark patterns more than doubled the rate at which
people signed up for a dubious service and aggressive ones nearly quadrupled
it. @mathur2019dark found 1,818 instances of dark patterns, in 15 types, on
about 11,000 shopping websites. @esposito2026back, by contrast, argue that
Article 25 is vague, narrow in scope, and weakened by the exclusion in its
paragraph 2.

**What it means for PBO.** The following are inferences from the text of the
law, not legal advice. A PBO system deployed on an online platform decides
which options a user sees and in what order, so it is part of the design of an
online interface. If its acquisition function systematically steers users, for
example by repeatedly reinforcing the incumbent or by favoring options that
make the user more predictable, the words "in effect" mean that intent need
not be shown under Article 25. The threshold of Article 5 of the AI Act,
purposeful manipulation or subliminal technique combined with significant
harm, is much higher, and ordinary design tuning is unlikely to reach it; a
Digital Fairness Act may reach personalization by optimization directly
(inference). By the logic of Planet49, an option the user "accepts" at the end
of a long, system-led sequence of queries is only weak evidence of preference
if the sequence was steering; logging a balanced query order and ending with
an explicit endorsement check serve scientific validity and compliance at once
(inference; @sec-rec-ethics). We have not reviewed the full text of the
Commission's 2025 guidelines, and we found no court or regulatory decision that
applies Article 25 to a recommendation or preference elicitation algorithm
rather than to static interface elements.

### Rhetoric and argumentation {#sec-soc-rhetoric}

Rhetoric holds that preferences change through reasons exchanged in dialogue,
a mechanism a model of fixed utility ignores. (The argumentative theory of
reasoning of @mercier2011humans is often compressed to the claim that
reasoning evolved to persuade others; the theory concerns both producing
arguments and evaluating them.) Classical PBO has no channel for reasons, but
BO systems that add one through large language models have appeared
(@sec-llm-in-loop), and conversational systems move views.
@costello2024durably had 2,190 people who believed in a conspiracy theory hold
three rounds of personalized dialogue with GPT-4 Turbo; belief fell by about
20%, and the effect lasted at least 2 months. @tessler2024ai trained a
language-model mediator, the "Habermas Machine", with a reward model to
maximize the group's endorsement of statements. Participants
($N = 5{,}734$, in the United Kingdom) preferred its group statements to those
written by human mediators, discussants' views converged after mediation, and
dissenting views were incorporated into the successful statements.

**What it means for PBO.** An interface that explains ("this option is
brighter because ...") or presents options through a language model is itself
a channel of persuasion, so the presence of an explanation should be
randomized, or at least recorded, so that its effect can be separated from the
comparison data (inference; @sec-hci-trust). The pipeline of Tessler et al.
(generate candidates, collect endorsements, refine) is close to PBO for a
group, and its convergence can be read as common ground or as homogenization
caused by the optimizer, so group PBO needs a criterion such as "are minority
positions retained", which Tessler et al. measured (inference).

## Linguistics {#sec-soc-linguistics}

When feedback is given in words, language shapes what is measured. *Verbal
overshadowing* is the finding that describing something can impair later
recognition of it: in @schooler1990verbal, participants who described a
robber's face were about 25% worse at picking him out of a lineup. That figure
is correct for the original study but overstates the effect. A
multi-laboratory registered replication [@alogna2014registered] found that
participants who described the robber were 4% less likely to identify him when
the description immediately followed the event, and 16% less likely when it
came after a 20-minute delay: robust, and strongly dependent on timing.
@wilson1991thinking found that introspecting about one's reasons can lower the
quality of preferences and choices. *Gradable adjectives*, words such as
"tall" or "warm" whose meaning is a threshold on a scale, have been modeled
with an uncertain threshold [@lassiter2017adjectival] that varies with context
and with the adjective [@xiang2022pragmatic].

**What it means for PBO.** When a system accepts graded verbal feedback such as
"slightly warmer" or "much better", the likelihood should treat the magnitude a
word expresses as uncertain and relative to the current context of comparison,
not as a fixed increment of utility (inference). Asking users to write down
reasons before they choose may change the judgments that follow, so reasons
are better collected after the choice (inference).

## Institutions that already collect judgments {#sec-soc-institutions}

Search engines, schools, and armies have gathered human judgments at scale
for a long time. Their experience speaks to which question to ask, how to read
the answer, and when a person's preference is the wrong target.

### Information science {#sec-soc-information}

Information retrieval is the discipline closest to the lineage of PBO: it
framed interactive optimization as a *dueling bandit* problem
[@yue2009interactively], in which the learner sees only which of two options
wins (@sec-dueling-bandits). **Its evaluation research since 2017 gives direct
evidence for preference judgments.** @sakai2020good found that the best
preference-based evaluation measures agree with users' preferences between
search result pages at least as well as an average assessor does.
@clarke2021assessing found that assessors make preference judgments faster and
more consistently than graded judgments, and that preferences can distinguish
items that graded judgments treat as equivalent; but fully ordering a pool by
preferences takes more than linear effort. Partial preference judgments that
identify and order only the top items detected improvements in neural rankers
that the standard graded measure (normalized discounted cumulative gain)
missed.

Language models as judges have produced results in both directions.
@thomas2024large report that at Bing, relevance labels from a large language
model were as accurate as those of human labelers, but that simple rewordings
of the prompt changed the accuracy. A 2024 preprint [@upadhyay2024large]
found that in the retrieval-augmented generation track of the Text REtrieval
Conference (TREC) 2024, the system ranking produced by an automatic judge
correlated highly with the ranking from fully manual judgments. Another 2024
preprint [@clarke2024llm] built a system designed to exploit automatic
evaluation, obtained inflated scores, and pointed out that if every system
uses the same language-model judge as a reranker, the resulting rankings are
distorted by circularity.

**What it means for PBO.** The retrieval evidence supports PBO's core design:
pairwise queries aimed at locating the best region rather than learning the
whole utility well. When the goal is selection, the acquisition function
should target identifying the top few options rather than global accuracy
(inference; @sec-eubo). When a language model serves as a proxy oracle of
preference, the final evaluation must use independent human judgments, or it
inherits the circularity Clarke and Dietz describe; and sensitivity to the
prompt means a language model's "preferences" are partly products of the
prompt (inference; @sec-hd-llm, @sec-sw-simulated-users). @sec-adj-ltr follows
learning to rank, the machine learning side of the same field.

### Education {#sec-soc-education}

Education research supplies the clearest evidence that what people prefer and
what works for them can come apart. In a randomized comparison in a large
introductory physics course, @deslauriers2019measuring found that students in
active-learning classes learned more but felt they had learned less, partly
because of the greater cognitive effort; the authors warn that evaluating
teaching by students' perceptions may favor inferior passive teaching. On
*learning styles*, the claim that instruction matched to a student's preferred
modality works better, the meta-analysis of @clintonlisell2024it (21 studies,
1,712 participants) found an overall benefit of matching of $g = 0.31$ (95%
confidence interval 0.05 to 0.57), where Hedges' $g$ is an effect size like
Cohen's $d$; but only 26% of the outcome measures showed the crossover
interaction the hypothesis requires, in which each group does better with its
own matched instruction, and the authors concluded that the benefits are too
small and too infrequent to justify widespread adoption.

**What it means for PBO.** Tuning instructional parameters with the learner's
preference as the objective risks optimizing the feeling of learning. The
objective should be the learning outcome, with preference as a constraint or a
secondary objective. Bayesian optimization with preference exploration (BOPE)
has this structure: a model maps parameters to measured outcomes, and the
person compares outcome vectors rather than parameters [@lin2022preferenceb]
(inference). Where better options take more effort to appreciate, early
pairwise judgments may penalize them, and a delayed retest is a safeguard
(inference).

### Military decision-making {#sec-soc-military}

A strong claim from research on expert decisions is that experts under time
pressure never compare options. Klein's original study of fireground
commanders, a 1985 report reprinted in 2010 [@klein2010rapid], covered 156
decision points: evidence of concurrent comparison of options appeared at
fewer than 12% of them, and at more than 80% the commanders recognized the
situation as typical and identified the typical action directly. "Never
compare" is an overgeneralization. According to a secondary summary with page
references [@ambur2004recognition], Klein's 1998 book reports that the share of
recognitional decisions across studies ranged from 46% to 96%, that people
compare options more when they must justify a choice, seek the best option, or
meet an unfamiliar situation, and that they use a "face-off" procedure in
which one option is compared with a second and the winner meets the next.

**What it means for PBO.** Searching for the best option in an unfamiliar
continuous design space is exactly when Klein expects comparison, and the
incumbent-versus-challenger query of many PBO systems matches the face-off
procedure (inference). For experts in familiar domains, feedback on whether a
single option is acceptable fits better (inference; @sec-hci-feedback-forms),
and expert status does not justify lowering the noise parameter in a new
design space (inference; @sec-hci-expertise).

## Briefer links {#sec-soc-briefer}

Six more disciplines offer one lesson each for PBO.

**Museums.** Context shifts liking more than it shifts rankings. The same
exhibition was liked more in a museum than in a laboratory simulation
[@brieber2015white], while a rehang of the Belvedere in Vienna increased
viewing time but did not change 259 visitors' interest in specific artworks or
their preferences among art forms [@reitstatter2020display]. Pairwise
comparisons made within one context should transfer better than absolute
ratings made across contexts (inference).

**Geography.** People choose from the options they actually consider. Treating
that consideration set as latent, @tsoleridis2023probabilistic found that
ignoring constraints on it misestimated preferences. Comparisons involving
options outside the range a user considers may be close to random, which a
likelihood with a lapse component handles (inference; @sec-obs-extensions).

**Archaeology.** Many different processes of cultural transmission produce
the same population-level frequency patterns, so matching a pattern is weak
inference [@kandler2019analysing]. Pooled comparisons from many users likewise
cannot pin down both the individual utilities and the rule that aggregated
them, in line with the analysis of hidden context by
@siththaranjan2024distributional (inference).

**Semiotics.** Consumer culture theory studies how people use goods to make
identities and social meanings [@arnould2005consumer; @mccracken1986culture].
If the target of optimization lives in a space of such meanings, a Gaussian
process on semantic embeddings should predict held-out comparisons better
than one on physical design parameters, a testable comparison (inference).

**Narratology.** Under *radical uncertainty*, when outcomes cannot be
enumerated and probabilities cannot be assigned, people choose by constructing
narratives [@johnson2023conviction]. A typical PBO problem is not radically
uncertain, but a short elicitation of purpose and audience before the
pairwise queries helps when the user is not yet clear what the design is for
(inference).

**Organization theory.** Adaptive processes refine exploitation faster than
exploration, which is effective in the short run and self-destructive in the
long run [@march1991exploration] (@sec-explore-exploit). An exploitative
acquisition function combined with revaluation after choice will lock in the
incumbent prematurely, as the local optimizer does in @fig-soc-moving-target,
and an exploration budget that does not depend on the posterior counters it
(inference).

## Common claims, checked {#sec-soc-claims}

The claims in @tbl-soc-claims are found in writing that applies these
disciplines to preference learning and recommendation. Each was checked
against its source; the sections above give the details.

::: {.table #tbl-soc-claims title="Claims from the social sciences and humanities, and what the sources say."}
| Claim | What the source says | Section |
|---|---|---|
| On Facebook, individual choice limited ideological diversity by 70% and the algorithm by 15% | not in the paper: the algorithm cut cross-cutting content by 5% (conservatives) and 8% (liberals), choice by 17% and 6%; choice outweighed the algorithm only for conservatives | @sec-soc-opinion |
| Obesity spreads through friendship networks, raising risk by 57% | the estimate is real, but homophily and contagion are generally confounded in observational network data | @sec-soc-opinion |
| Moses built low bridges to keep buses from Jones Beach | the factual basis of the story is disputed | @sec-soc-sts |
| Verbal description cuts recognition by about 25% | the original figure; a registered replication found 4% or 16% depending on timing | @sec-soc-linguistics |
| Expert commanders never compare options | concurrent comparison at fewer than 12% of decision points; recognition rates range from 46% to 96% across studies | @sec-soc-military |
| Reasoning evolved to persuade others | the argumentative theory covers producing and evaluating arguments | @sec-soc-rhetoric |
| Every preference query trains the user's taste (after Hennion) | an extrapolation, not Hennion's argument | @sec-soc-taste |
:::

None of these corrections removes a general point: algorithms still embed
choices of value without the Moses story, and verbal reasoning still affects
perceptual judgment at 4% to 16%. What changes is how strongly each point can
be asserted, and that is the currency of a model's prior (inference).

## Settled, contested, missing {#sec-soc-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** Behavioral science samples remain overwhelmingly Western: a
little over 60% American in top psychology journals from 2014 to 2018, with
the majority world at 4% to 5%. Under the preference dynamics assumed by Dean
and Morgenstern, low regret is "an almost trivial goal" reachable by simple
algorithms, so regret alone cannot show that a system found what the person
wanted. Performative prediction gives necessary and sufficient conditions for
retraining to converge to a performatively stable point. Assessors make
preference judgments faster and more consistently than graded judgments.
Article 25 of the EU Digital Services Act prohibits providers of online
platforms from designing interfaces that materially distort or impair users'
ability to make free and informed decisions, and the Act's recital 67 counts
distortion "on purpose or in effect"; Article 5 of the AI Act has applied
since 2 February 2025.

**Contested.** How strongly recommender systems shape preferences:
simulations and observational studies point to homogenization, while large
randomized field experiments over weeks to months mostly find small or null
attitude effects. Whether language-model relevance judges can replace human
assessors. Whether matching instruction to learning styles helps: a small
pooled effect, but rarely the required crossover. How far the Digital Services
Act reaches into algorithmic, rather than static, interface design.

**Missing.** An application of performative prediction, or of the
induced-shift frameworks of Dean and Morgenstern and Carroll et al., to
Gaussian process preference learning. A measurement of how much preferences
move within a PBO session, against a random-order control and a delayed
retest. A test of whether pairwise comparisons resist cultural response styles
better than ratings. A comparison of Gaussian processes on physical parameters
against Gaussian processes on semantic embeddings for the same comparisons.
:::

## Further reading {#further-reading .unnumbered}

- @perdomo2020performative and the review by @hardt2025performative are the
  formal core of this chapter: what it means for a model to be stable when its
  deployment changes the data.
- @dean2022preference and @carroll2022estimating show, in recommender
  settings, why regret stops measuring success when preferences move, and how
  to bound the shift a system induces.
- @guess2023social and @nyhan2023like are the large randomized field
  experiments to read before claiming that systems reshape what people want.
- @clarke2021assessing is the clearest evidence from information retrieval
  for preference judgments, including their cost.
- @lee2025visual (CogSci 2025) is the broadest cross-cultural data on aesthetic
  preferences over parameterized stimuli, close in form to PBO.
