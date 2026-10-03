---
status: done
synopsis: "What judgment and decision research, psychophysics, mathematical psychology, and the study of heuristics found between 2017 and 2026 about the assumptions behind the comparison likelihood: which famous effects faded in large replications, which held up, and what each implies for modeling a person's answers."
---

# Judgment, Decision, and Psychophysics {#sec-judgment}

@sec-part-preferences built preferential Bayesian optimization (PBO) on one
modeling decision: a person's answer to "which do you prefer?" is a noisy
reading of a fixed latent utility. In @sec-obs-baseline, the probability that
$\vx$ beats $\vx'$ is $\Phi\big((f(\vx) - f(\vx'))/(\sqrt{2}\,\sigma)\big)$, no
matter when the question is asked, what came before it, or what else is on the
screen. @sec-part-perspectives asks whether a preference of that kind is there
to be found. This chapter begins with the disciplines that study the act of
comparing itself: judgment and decision-making research, psychophysics,
mathematical psychology, and the study of heuristics.

Their evidence from 2017 to September 2026 points two ways. Findings once used
to argue that people tire quickly shrank to near zero in large preregistered
tests, while effects that break the independence of successive comparisons
were confirmed and now come with mechanisms a likelihood could include. The
picture is a mixed observation model: a stable component, evaluation noise
with structure, and drift caused by the queries themselves (inference).

## What the comparison model assumes {#sec-jdm-assumptions}

The likelihood of @sec-obs-baseline makes more commitments than its formula
shows. Multiplying the likelihoods of separate comparisons, as every PBO paper
does, treats answers as conditionally independent given the utility. Four
assumptions follow, and this chapter tests each: *stability*, one utility for
the whole session (@sec-jdm-choice, @sec-jdm-range); *order independence*, no
carryover from earlier questions (@sec-jdm-sequential, @sec-jdm-order);
*constant noise*, one noise scale for every pair (@sec-jdm-range,
@sec-jdm-rum); and *context independence*, no influence from other options on
screen or in memory (@sec-jdm-context).

The evidence is weighed with psychology's own yardsticks. The most common
**effect size**, **Cohen's $d$**, is the difference between two group means
divided by the standard deviation within the groups; Cohen's conventions call
0.2 small, 0.5 medium, and 0.8 large [@cohen1988statistical], and at $d = 0.1$
the two distributions overlap almost completely. A 95% **confidence interval**
is the range of effect sizes compatible with the data. A **meta-analysis**
pools many studies, and **publication bias**, the tendency of significant
results to be published more often, inflates its estimate.
**Preregistration** fixes the hypotheses and analysis before data collection;
a **multi-lab replication** runs one protocol in many laboratories, and a
*registered replication report* is one accepted for publication before its
results are known. A **Bayes factor** of 4 in favor of the null means the data
are four times as probable if there is no effect.

**Test-retest reliability**, the correlation between two measurements of the
same quantity on the same people, decides whether a per-user parameter can be
estimated at all. A meta-analysis of the stability of risk preference, built
on test-retest correlations, estimated a reliability of 0.61 for
self-reported risk propensity and of 0.25 for behavioral measures, tasks such
as lottery choices and balloon-pumping games played for real or hypothetical
money [@bagaini2025systematic]: the "behavioral" measure of a stable trait is
the less stable one, and at 0.25 most of the variation between people in a
single measurement is noise. For calibration, the Many Labs 2 project ran 28
published findings across many samples and settings: 15 of the 28 (54%)
replicated, 75% of the 28 replication effect sizes were smaller than the
originals, and the median Cohen's $d$ fell from 0.60 to 0.15
[@klein2018many].

## Judgment and decision-making {#sec-jdm}

The view that preferences are constructed during elicitation, rather than
retrieved from memory, has a long history in this field
[@slovic1995construction]. On it, **preference reversals**, in which two ways
of asking the same question produce opposite answers, show that a single
latent utility is a simplification; on the opposing view they are errors
around a stable preference, which newer evidence partly supports.
@oprea2024decisions found the patterns associated with prospect theory in
riskless "mirror" tasks and interprets them as an effect of complexity rather
than of attitudes toward risk, a dispute conducted outside peer review that is
unresolved [@oprea2025initial], and @gigerenzer2018bias
argues that many reported biases come from treating random error as
systematic, from small-sample statistics, or from counting reasonable
inferences as mistakes. The effects most often cited against independent
comparisons are fatigue, choice-induced change, context effects, and
anchoring, in which an arbitrary number shifts later estimates
[@tversky1974judgment].

### Self-control findings that faded {#sec-jdm-fatigue}

The case for short sessions with breaks usually cites parole boards whose
favorable rulings fell from about 65% to nearly zero between food breaks
[@danziger2011extraneous], and **ego depletion**, the idea that self-control
draws on a limited resource that an effortful first task uses up
[@baumeister1998ego]. Ego depletion faded in three large tests: $d = 0.04$,
with a confidence interval that included zero, in a registered replication
across 23 labs ($N = 2{,}141$) [@hagger2016multilab]; a nonsignificant
$d = 0.06$ across 36 labs ($N = 3{,}531$), preregistered, with the data four
times more probable under the null in a Bayesian analysis with an informed
prior [@vohs2021multisite]; and a significant but small $d = 0.10$, or 0.16
after excluding participants who may have answered at random, across 12 labs
($N = 1{,}775$) [@dang2021multilab]. Even if it exists, its size is around 0.1
and depends on the paradigm.

The parole-board effect did not survive scrutiny of its size: prisoners
without legal representation tended to be heard last in each session
[@weinshallmargel2011overlooked], and rational scheduling of cases alone can
inflate the apparent effect [@glockner2016irrational]. **Decision fatigue**, a
decline in the quality of decisions over a run of them, met a large field
test: in 231,076 triage calls handled by 174 nurses at Sweden's national
healthcare advice line, four preregistered confirmatory tests all supported
the null hypothesis, with one-sided Bayes factors above 22 [@andersson2025no].
A preregistered systematic review of 82 studies of healthcare professionals
found a significant effect in 45% of quantitative tests, but notes that the
construct is defined inconsistently and operationalized poorly
[@maier2025systematic]. That people under high cognitive load chose chocolate
cake over fruit salad more often (63% against 41%) [@shiv1999heart] is often
described as replicated many times, but we found no registered or large-sample
replication, and cognitive load did not change a large decoy effect in grocery
choices ($N = 96$, within-subjects) [@wedell2022context]. The **unconscious
thought advantage**, choosing better after distraction than after
deliberation [@dijksterhuis2006making], remains unsupported by a meta-analysis
and large-scale replication [@nieuwenstein2015making]. Long sessions are not
thereby harmless, but depletion and fatigue are no longer a sound basis for
designing them.

### Choice changes preference {#sec-jdm-choice}

In the **free-choice paradigm** [@brehm1956postdecision], people rate items,
choose between two they rated about equally, and rate everything again. The
chosen item tends to rise and the rejected one to fall, a *spreading of
alternatives* long read as the mind justifying its choice. Try it in
@fig-jdm-free-choice before reading on.

```{figure}
//| figure: jdm-free-choice
//| label: fig-jdm-free-choice
//| fig-cap: "The free-choice paradigm on your own answers: rate six posters, choose within two pairs you rated alike, then rate all six again. Open dots are first ratings, filled dots second ratings; the spread of alternatives is how much more the chosen posters gained than the rejected ones. Your answers stay on this page. With two choices, this is a demonstration, not a measurement."
```

Only part of your spread can be a change in preference. If a rating measures
liking imperfectly and a choice follows true liking, then of two items rated
alike, the one you choose is more likely to be the one your first rating
*under*estimated, and a fresh rating will favor it even if your liking never
changed; @chen2010how proved this as a theorem and demonstrated it
experimentally. Pooling four studies that address the critique,
@izuma2013choice found an effect that was statistically significant but small,
$d = 0.26$. @fig-jdm-free-choice-sim simulates the artifact.

```{figure}
//| figure: jdm-free-choice-sim
//| label: fig-jdm-free-choice-sim
//| fig-cap: "The free-choice artifact, simulated with 3,000 participants and two choices each. The classic design (rate, choose, rate) measures a spread even when the true change δ is zero, because the choice reveals which item the noisy first rating underrated. The control design (rate, rate, choose) measures the artifact alone; their difference recovers δ (dashed line). Spreads are in standard deviations of true liking; the noise levels are illustrative."
```

At $\delta = 0$ the classic design still shows a clear positive spread, as
large as the control design's; set $\delta = 0.4$ and only the classic spread
rises, by about 0.4. Designs that separate the effect of choosing from what
the choice reveals are called *artifact-free*. A meta-analysis of 43 studies
with artifact-free free-choice designs ($N = 2{,}191$) found $d = 0.40$, 95%
confidence interval $[0.32, 0.49]$, with no evidence of publication bias, and
concluded that choice causes real preference change, not merely reflects it
[@enisman2021choice]. Each choice raises the value of the chosen option and
lowers that of the unchosen one, and choices between the same pair become
less consistent as more trials intervene [@zylberberg2024value]; a
**sequential sampling model**, in which evidence for each option accumulates
noisily until it reaches a threshold (@sec-neuro-ddm), reproduced the
post-choice spreading of 457 participants [@lee2026choice]. Exposure matters
too: options seen more often come to be liked more
[@zajonc1968attitudinal]; across 268 curve estimates from 81 articles, liking
follows an inverted U in the number of exposures, for visual but not auditory
stimuli [@montoya2017re]; and what matters is *relative* exposure, how often an option is seen compared with the
others [@mrkva2020salience]. @fig-jdm-effect-sizes sets the effects of this section
side by side.

```{figure}
//| figure: jdm-effect-sizes
//| label: fig-jdm-effect-sizes
//| fig-cap: "What faded and what held, on one scale. Each row is an effect size reported in this chapter: open dots are original estimates, filled dots later tests, bars reported 95% confidence intervals, and faint lines Cohen's small, medium, and large. Click a row or move the slider: the lower panel draws two normal distributions of equal spread whose means differ by d, a property of the normal model rather than data."
set: judgment
d: 0.06
```

Things to try:

- At the starting value, $d = 0.06$ (36 labs of ego depletion), the two
  distributions overlap 98%, and a person drawn from the higher group outscores
  one from the lower group 52% of the time, barely better than a coin flip.
- Click the 43 artifact-free studies, $d = 0.40$: the overlap falls to 84%,
  and the higher group wins 61% of the time. The Many Labs 2 medians, 0.60 and
  0.15, overlap 76% and 94%.

### Context effects {#sec-jdm-context}

Luce's choice axiom (@sec-bradley-terry) says that the odds of choosing $A$
over $B$ do not depend on what else is offered. Three classic **context
effects** violate it. Adding a *decoy*, an option worse than $A$ on every
attribute but not worse than $B$, raises the share choosing $A$ (the
*attraction effect*) [@huber1982adding]; an option gains when it becomes the
middle one of three (*compromise*) [@simonson1989choice]; and a new option
takes its share mostly from the option it resembles (*similarity*)
[@tversky1972elimination]. Try the first in @fig-jdm-decoy.

```{figure}
//| figure: jdm-decoy
//| label: fig-jdm-decoy
//| fig-cap: "A decoy added to your own choices. Round 1 asks six everyday choices between two options; round 2 asks them again, shuffled, each with a third option. Answer both rounds before reading the text. The summary counts your switches and maps each situation by its two attributes, better to the right and up. Your answers stay on this page. Three answers per condition make this a demonstration, not a measurement."
```

Every third option in round 2 was a decoy. In three situations it favored the
option you passed over in round 1, so the attraction effect predicts a switch;
in the other three it favored your choice, so a switch there measures the
plain inconsistency of repeated choice (@sec-math-psych). With three answers
each, chance alone produces either ordering often.

Since 2017 the evidence says that context effects hold at the population level
under strong boundary conditions. Individuals usually do not show all three at
once, and averaging across people can produce a pattern that no individual
shows [@liew2016appropriacy]. A dominated option can also make the option
that dominates it look *worse*, a **repulsion effect**, depending on how the
stimuli are presented [@spektor2018good]. A review traced the appearance, disappearance, and
reversal of context effects to the spatial arrangement of the options, how
concrete the attributes are, and how long people deliberate
[@spektor2021elusiveness]; the attraction effect appears mostly when every
attribute is a number, and usually disappeared when people experienced the
products or when even one attribute was presented perceptually
[@frederick2014limits]. A *distractor effect* attributed to **divisive
normalization**, which divides each option's value by the total value on offer
(@sec-neuro-coding), failed to replicate in two preregistered experiments
[@gluth2020value].

The most useful result for system design is about cues. In an incentivized,
preregistered experiment with 909 participants and 40 trials each, every 10
percentage points of historical predictive power that a decoy cue had for the
better option raised the probability of choosing the option the decoy favored
by 1.29 percentage points; the corresponding figures were 1.58 for default
options and 2.21 for an arbitrary rule [@forsgren2025probabilistic]. People
learn which cues point to good options and follow them. Anchoring is robust in
groups, but a meta-analysis of more than 50,000 anchoring estimates found that
the reliability of individual susceptibility is low in most tasks
[@roseler2024measurements], so a person's "anchoring score" cannot be measured
well.

### What this means for the observation model {#sec-jdm-implications}

None of the following was tested in PBO; each is the book's inference.

- **Put choice-induced updating in the likelihood.** After $\vx_w$ beats
  $\vx_l$, a model in the style of @zylberberg2024value updates
  $f(\vx_w) \mathrel{+}= \eta$ and $f(\vx_l) \mathrel{-}= \eta$. Whether the
  free-choice magnitude transfers to design comparisons is untested; a simple
  check is to repeat a few early pairs late in the session and read a
  systematic flip toward the recently chosen option as drift (inference).
- **The optimizer causes the drift it should guard against.** The incumbent is
  compared most often, so it is the most exposed to revaluation and relative
  exposure; apparent convergence may be lock-in on an early winner
  (inference), one more entry for @sec-pbo-failure-modes and a candidate
  explanation for the unstable feedback of @sec-hci-unstable.
- **Binary queries avoid the classic decoy, galleries do not.** A gallery
  (@sec-gallery-projective), and options still in memory, reintroduce a
  context set in which, by the repulsion effect, the incumbent can
  distort how new candidates are perceived (inference).
- **"New" is a learnable cue.** An optimizer that keeps proposing better
  candidates teaches the person that the new option is usually better.
  Randomize where and in what order proposals appear, and occasionally compare
  two old options (inference).
- **Set session length by measurement, not by depletion.** Measure trends in
  lapse rates and response times rather than assume fatigue (inference).
- **Do not fit per-user bias parameters.** Individual susceptibility to
  anchoring and decoys is measured unreliably; a population-level bias term,
  or none, is safer (inference).
- **Evaluate in use, not only in the session.** We found no direct
  replication or new large-sample test since 2017 of the distinction between
  decision utility and experienced utility [@kahneman1997back], so the
  outcome still needs a test in actual use, separate from the comparisons
  that produced it (inference), as @sec-rec-evaluation recommends.

## Psychophysics and attention {#sec-psychophysics-attention}

Psychophysics supplied the first models of comparison (@sec-psychophysics),
and two common claims about PBO come from it. By Weber's law, the smallest
noticeable difference grows with the magnitude being judged
[@fechner1860elemente], and judged magnitude follows a power law of physical
magnitude [@stevens1957psychophysical], so as the optimizer converges the
person may no longer tell the candidates apart. And by **decision by
sampling**, the subjective value of an amount is its rank among amounts
sampled from memory and from the current context [@stewart2006decision], so
an optimizer that concentrates its queries on good options would lead the
person to undervalue them. A further claim cites the limit on absolute
judgment, 2.6 bits on average along a single dimension [@miller1956magical],
as strong evidence that comparisons beat ratings.

### Relative value: sampling, range, and normalization {#sec-jdm-range}

For described attributes such as amounts and probabilities, decision by
sampling weakened. A multi-lab, quasi-adversarial replication (proponents and
skeptics agree on the design in advance) reproduced the
effect of the distribution of values presented on the shapes of the utility
and probability-weighting functions, but also observed it where decision by
sampling predicts no effect, and showed by simulation that a misspecified
choice model can produce it [@alempaki2019reexamining]; a preregistered
falsification test then found strong evidence against the prediction that
manipulating the ranks of the amounts changes choices
[@forsgren2025preregistered].

For values learned from experienced outcomes, range adaptation is well
supported. Human reinforcement learning is best fitted by models that center
values on a reference point and adapt them to the range of outcomes
[@bavard2018reference]; the adaptation causes systematic errors when values
are extrapolated to new contexts, larger when the task is easier
[@bavard2021two]; and a task built to tell the two candidate rules apart
falsified divisive normalization and supported **range normalization**, in
which a value is rescaled by the difference between the best and worst values
in its context [@bavard2023functional]. @fig-jdm-range computes what that
would do to a session. A range-normalized person experiences, in session $S$,

$$
v_S(x) = (1 - \lambda)\, u(x) + \lambda\, \frac{u(x) - \min_S u}{\max_S u - \min_S u},
$$ {#eq-jdm-range}

where $\min_S u$ and $\max_S u$ are the worst and best utilities shown and
$\lambda$ between 0 and 1 sets how strongly they adapt; ratings are
$100\,v_S$, and comparisons follow the probit model on the experienced scale.

```{figure}
//| figure: jdm-range
//| label: fig-jdm-range
//| fig-cap: "Range normalization makes the scale session-relative. Dashed: a fixed utility u(x); blue: the value experienced in session A, which explores everything; orange: in session B, which explores only the shaded region, near the best design as a converging optimizer does, or a poor one (dotted where it extrapolates). The table compares two designs. An idealized model, not fitted to data; the utility, noise, and ranges are illustrative."
```

Things to try:

- With $\lambda = 1$ and session B near the best, the probability of
  preferring $x_2$ is much closer to 1 in session B, so a model fitted there
  would learn a larger amplitude. At $\lambda = 0$ the sessions agree.
- Move a design outside the shaded region: session B extrapolates, the regime
  in which @bavard2021two found systematic errors.

### Sequential effects {#sec-jdm-sequential}

A judgment can be pulled toward the previous one (**assimilation**) or pushed
away from it (**contrast**); the direction depends on the setting, and neither
is a stable personal trait. In visual perception, the positive pull decays
over time, requires the features to be similar, depends on spatial location,
and is modulated by attention [@manassi2023serial]. Preference ratings of
photographs and faces assimilate toward the previous stimulus even after
response bias is controlled [@chang2017sequential], while in 2.2 million Yelp
ratings and 4.2 million Amazon ratings the same reviewer's ratings show
*contrast* relative to their previous ratings [@vinson2019decision].
Assimilation in ratings of facial attractiveness is not stable in the same
person across two sequences [@kramer2026sequential], and in repeated aesthetic
ratings both assimilation and contrast grow with the similarity of the
stimuli [@pombo2023intrinsic].

### Attention {#sec-jdm-attention}

That attention affects choice holds in direction, but whether it raises value
is disputed. Across six eye-tracking data sets, the sum of the options' values
affected response times and, in most data sets, the relation between gaze and
choice, consistent with gaze multiplying value [@smith2019gaze]. A review
concluded that there is not enough evidence that attention itself raises the
perceived value of an option [@mormann2021attention]. When only one option is
visible at a time, the attentional choice bias roughly doubles ($N = 50$)
[@eum2023peripheral]. @sec-neuro-ddm reports a meta-analysis of the causal
effect of attention.

### What this means for the observation model {#sec-jdm-psychophysics-implications}

- **The learned utility is on a session-relative scale** (@fig-jdm-range).
  Comparing posteriors across sessions or users, or warm-starting from an old
  posterior, needs explicit recalibration, and extrapolation beyond the
  explored range is where errors are most likely (inference).
- **For perceptual options, expect range and recency, not curvature.** Range
  adaptation and sequential dependence change the scale and recent-trial bias
  rather than the shape of the utility (inference).
- **Present simultaneously and balance positions.** Show both candidates side
  by side, balance left and right, and do not make the system's proposal more
  salient; where options can only be experienced in sequence (audio,
  animation), counterbalance order across repeated pairs (inference).
  @sec-interface-model makes the general case, and the photo-enhancement case
  study (@sec-cs-photo) is the kind of perceptual task where these effects
  should be expected.
- **Model recency at the population level, and estimate its direction.** A
  per-user sequential-bias parameter is unstable within a person, and the
  sign, assimilation or contrast, should be estimated rather than assumed
  (inference).
- **Discrimination near the optimum depends on what is judged.** For value
  judgments the Weber-based claim fails: decisions between two high-value
  options tend to be faster and more accurate [@shevlin2022high], and
  discrimination tracks how densely values occur rather than how large they
  are [@polania2019efficient] (@sec-neuro-coding). For the physical parameters
  of a device, such as the stiffness of a prosthetic ankle, a discrimination
  floor does exist [@shepherd2018amputee; @maberry2026just] (@sec-neuro-motor,
  @sec-cs-exoskeleton).

Two common claims need correcting. Luce's axiom and Thurstone's Case V are
not equivalent "under a logistic distribution": Case V assumes normal errors
(the probit link, @sec-thurstone), and @yellott1977relationship shows that the
choice axiom follows from independent double exponential (Gumbel) errors (the
logistic link, @sec-random-utility); for pairs other distributions do the
same, while for choices among three options the double exponential is the only
one. And we found no test of Miller's channel-capacity argument with design
stimuli.

## Mathematical psychology {#sec-math-psych}

Mathematical psychology tests the formal models from which the probit and
logistic links come. Random utility theory (@sec-random-utility) has long
allowed the noise to be read as variation within a person, variation between
people, or genuine instability, readings that are formally interchangeable
[@mcfadden1981econometric]. A preference is **transitive** if preferring $A$
to $B$ and $B$ to $C$ implies preferring $A$ to $C$; since
@tversky1969intransitivity reported violations, many have been claimed, and
@regenwetter2011transitivity concluded that most of them can be explained by
people mixing among transitive preferences. With noisy choices, every model in
which the probability of a choice depends on a utility difference, the probit
model of @eq-obs-probit included, satisfies *strong stochastic
transitivity*: if $a$ beats $b$ and $b$ beats $c$ at least half the time, $a$
beats $c$ at least as often as in either pair, because $u_a - u_c$ is at least
as large as either difference. A violation of even the weak form, $a$ beating
$c$ less than half the time, is evidence against the whole family.

**Quantum cognition** models judgments with the probability rules of quantum
mechanics, where asking question 1 and then question 2 can give different
answers than the reverse order. A parameter-free prediction about such order
effects, the *QQ equality*, was supported in 70 national surveys and two
laboratory experiments [@wang2014context]. From this comes the argument that
each query changes the state of a person's preference, that repeated queries
"may never converge" to a true preference, and that PBO should use a
surrogate built on quantum probability.

### Random utility, tested directly {#sec-jdm-rum}

Direct tests support the random utility hypothesis. Using Falmagne's
inequalities, which characterize exactly which choice probabilities a random
utility model can produce, @mccausland2020testing tested 141 participants,
each of whom chose six times from every subset of at least two of five
lotteries. Most participants satisfied random utility; only 4 showed strong
evidence of violating it.

The way the noise is specified, however, changes the inference. Combining
noise in the preference parameters with noise in the response process can make
an expected-value maximizer look risk averse or risk seeking, so modal choices
cannot simply be used to infer latent preferences [@bhatia2017noisy]. When
noise of one scale is added to the utilities of every pair, the probability of
choosing the riskier option need not fall as risk aversion grows, which makes
risk aversion hard to identify; random parameter (random preference) models do
not have this problem [@apesteguia2018monotone]. For example (computed for
this book, not from their data), with a lottery paying 80 or 10 against a sure
40, power utility, and probit noise of scale 1, a person with risk-aversion
parameter $r = 0.5$ picks the lottery with probability 0.29 and a more
risk-averse one with $r = 2$ with probability 0.49: a large $r$ compresses all
utility differences, and a fixed noise scale turns them into coin flips. A
2024 preprint qualifies the criticism: in a representative Danish sample (253
people), the standard expected-utility specification squeezed estimates of
risk aversion into a narrow range when everyone shared one noise scale, and
gave estimates in line with other models when each person had their own
[@keffert2024stochastic].

### Transitivity and cycles {#sec-jdm-transitivity}

Response times can separate noisy cycles from genuine ones: choices between
options of similar value are both slower and more error-prone, so a cycle made
only of fast choices is hard to explain as noise. In two reanalyzed data sets,
a working paper found that 54.38% and 90.25% of the violations of weak
stochastic transitivity involved only fast choices, and that 19.24% and 39.58%
of them qualified under a stricter criterion, in which choice frequencies and
response times reveal the preference on each pair of the cycle; for the rest,
neither noise nor intransitive preference can be ruled out
[@alosferrer2023identifying]. Violations shrink but do not disappear once
noise is separated from preference, and the most frequent arise from chains of
small trade-offs. Lotteries designed after the Steinhaus-Trybula paradox still
produced cycles as the most common pattern after random-but-transitive
explanations were accounted for [@butler2018predictably].

### Order effects and quantum models {#sec-jdm-order}

Quantum cognition holds up in part, but its claim to uniqueness is contested.
Classical repeat-choice models also yield the QQ equality as a parameter-free
prediction and account for the survey data comparably [@kellen2018classic].
In an experiment with 325 participants and 12 sets of issues, inserting an
incompatible question between two presentations of the same question made the
answer to the repeated question uncertain, as quantum probability predicts
[@busemeyer2017there], though a 2023 preprint argues that this experiment is
logically inconsistent with results on the replicability of answers
[@ozawa2023logical]. @kvam2021temporal found that the strength of a preference
oscillates over time and that eliciting a choice earlier changes the later
oscillation.

### What this means for the observation model {#sec-jdm-mathpsych-implications}

- **The standard likelihood is a choice, not a neutral default.** A fixed-scale
  probit or logistic link is a homoscedastic random utility model and can
  distort the inferred strength of a preference. Alternatives are a *random
  preference* likelihood, which draws each comparison from one whole sample of
  the Gaussian process posterior, and a *heteroscedastic* link whose noise
  scale varies with the difficulty of the pair (inference; shaped by the
  Gaussian process framework). At the least, report the reversal rate on a few
  repeated pairs next to any fitted noise scale (inference).
- **Local cycles may need modeling, general intransitivity rarely does.**
  Chains of small trade-offs describe exactly the local queries around the
  incumbent late in a session; a mixture of rankings or a random preference
  model can absorb them without giving up a latent utility (inference).
- **Retest immediately or much later, not in between.** By the
  interleaved-question and oscillation results, a repeat separated by other
  queries is contaminated. Retest right away to measure response noise, or
  much later to measure drift (inference).
- **Quantum surrogates are not a default.** They are untested in any
  optimization setting, and the order effects behind them have classical
  explanations; a covariate for query order is simpler, and "converging to a
  point that depends on the session" is closer to the evidence than "may never
  converge" (inference).
- **Better simulated users.** Differentiable decision theories fitted to the
  largest experiment on risky choice to date, more accurate than the classic
  ones [@peterson2021large], and a language model fine-tuned on 10 million
  choices from 160 experiments, which predicted held-out participants better
  than existing cognitive models [@binz2025foundation], could serve as
  simulated users (@sec-sw-simulated-users), although how well they fit design
  comparisons is unknown (inference).

## Ecological rationality and heuristics {#sec-ecological}

The research program on **ecological rationality** studies simple decision
rules, *heuristics*, and the environments in which they work. *Take-the-best*,
which decides on the most valid cue that discriminates between the options,
predicted out of sample no worse than multiple regression across 20 data sets,
and the bias-variance trade-off explains why less information can lead to
better predictions [@gigerenzer2009homo]. From this comes the argument that
simpler surrogates may do better when data are sparse, and that a smooth
*compensatory* utility, in which a loss on one attribute can be made up by a
gain on another, misdescribes a person who uses a lexicographic rule.

### When people use which strategy {#sec-jdm-strategies}

Since 2017 the question has moved from whether heuristics work to when each is
used. Under time pressure take-the-best is slower and less accurate than
tallying cues, and the result reverses when the stimulus format makes its
search cheaper [@bobadillasuarez2018fast]. *Bounded meta-learned inference*
predicts that people decide by a single cue when they know the ranking of the
attributes' importance, weigh attributes equally when they know only the
direction of each, and combine weighted attributes when they know neither;
three experiments with pairwise comparisons of options described by
continuous features supported these predictions [@binz2022heuristics]. On very
large risky-choice data, more expressive but still interpretable models beat
the classic theories [@peterson2021large], so "less is more" depends on how
much data there is, as the bias-variance argument implies.

### What this means for the observation model {#sec-jdm-ecological-implications}

**The choice rule can change while the utility does not.** At the start of a
session, a person usually does not know which design attribute matters, a
state that the meta-learning theory maps to weighted multi-attribute
comparison. As the session teaches them which attribute decides, the same
theory predicts a shift toward single-reason, lexicographic choice, so a smooth
compensatory utility would fit the early session better than the late one.
This is non-stationarity of the choice rule, distinct from drift in the
utility itself, and the two should be modeled separately (inference);
@sec-rec-nonstationary takes up both. The interface matters too: listing
parameter values side by side makes single-reason strategies cheaper, while a
holistic rendering favors integrated judgment (inference).

**A lexicographic person within a smooth model.** A continuous latent utility
can only approximate a lexicographic ordering. A repair inside the framework is
to allow a much shorter lengthscale (@sec-ard) along the attribute the person
has locked onto, or a function close to a step (inference; this repair comes
from the Gaussian process framework, not from the problem itself). In a
six-parameter design problem, a person who has decided that one parameter
matters most would show a short lengthscale along it and long ones along the
other five, a shape that one lengthscale per input can represent and a single
shared lengthscale cannot (inference).

**Simpler surrogates and bias terms need evidence.** Whether a low-dimensional
or additive utility beats a full Gaussian process on human pairwise data from
real design sessions is untested. The "bias bias" critique and the complexity
results counsel caution before adding directional bias terms: each needs
evidence that it is systematic in the task at hand rather than random error
(inference).

## Settled, contested, missing {#sec-jdm-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** Ego depletion, if it exists, is around $d = 0.1$
[@vohs2021multisite; @dang2021multilab]; a preregistered field test of
231,076 calls found no decision fatigue [@andersson2025no]. Choice changes
preference: $d = 0.40$ across 43 artifact-free studies [@enisman2021choice].
Context effects exist at the population level but appear, vanish, and reverse
with presentation, and individual susceptibility to anchoring cannot be
measured reliably. Most people's repeated choices satisfy random utility
[@mccausland2020testing]. For values learned from experience, range
normalization fits better than divisive normalization [@bavard2023functional].

**Contested.** Whether the remaining reports of decision fatigue in healthcare
reflect a real effect or a poorly defined construct. Whether patterns
attributed to risk preference are effects of complexity
[@oprea2024decisions]. How much observed intransitivity is noise: a working
paper finds that part of it survives once noise is separated from preference
[@alosferrer2023identifying]. Whether quantum models are needed for order
effects. Which heuristic people use in design tasks.

**Missing.** Any test of whether the size of choice-induced change carries
over to comparisons of designs. A PBO study that models choice-induced
updating, relative exposure, or the "new is better" cue, or that retests early
pairs late in a session. A comparison of homoscedastic, heteroscedastic, and
random preference likelihoods on human design comparisons. Taken together, the
evidence favors an observation model with a stable component, structured
evaluation noise, and query-induced drift, which no preferential optimization
method yet implements (inference).
:::

## Further reading {#further-reading .unnumbered}

- @chen2010how is a short lesson in how a measurement design can manufacture
  an effect; read it with the artifact-free meta-analysis of
  @enisman2021choice.
- @spektor2021elusiveness review when context effects appear, vanish, and
  reverse, the best single source for anyone designing a multi-option
  interface.
- @klein2018many shows what happened to 28 classic findings run across many
  labs, a calibration for every effect size in this part.
- @mccausland2020testing and @apesteguia2018monotone together explain how to
  test a random utility model and why its noise specification matters.
