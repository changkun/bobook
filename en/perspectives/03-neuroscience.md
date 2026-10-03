---
status: done
synopsis: "What neuroscience and computational cognitive science found between 2017 and 2026 about value in the brain, neural forecasts of preference, response times and attention, efficient coding, active inference, pharmacology, and motor adaptation, and what each implies for modeling a person's comparisons."
---

# Neuroscience and Computational Cognitive Science {#sec-neuroscience}

The two previous chapters reported behavior: what people choose, how
consistently, and under which conditions (@sec-judgment, @sec-social-psych).
This chapter asks what the brain and computational models of the mind add for
someone who builds or evaluates a preferential Bayesian optimization (PBO)
system: whether a single scale of value exists inside the head, as the latent
utility of PBO presumes; what a response time reveals beyond the answer; and
where the noise in a comparison comes from, and whether it stays fixed.

The main change between 2017 and 2026 was to treat value as constructed during
a decision and revised by the choice itself. A scalar value has causal support
within one task, but a cardinal "common currency" across tasks remains
disputed. The noise in value judgments depends on the values seen before, so
discrimination does not decline near the optimum for value judgments, only for
perceptual parameters. Attention moves choice causally, but a little. Response
times entered preference learning with guarantees, though not yet in live
experiments with people. Neural signals add only about 4 percentage points
over self-report for an individual, and active inference is not yet a usable
alternative to PBO. The centerpiece is the drift-diffusion model of
@sec-neuro-ddm, which you can run in @fig-neuro-ddm.

## Neuroeconomics and value-based decisions {#sec-neuro-value}

Neuroeconomics looks for the brain's representation of value. A common claim
holds that the ventromedial prefrontal cortex (vmPFC) and orbitofrontal cortex
(OFC), regions behind the forehead and above the eyes, integrate the values of
different kinds of reward into a **common currency**, a single scale on which
food, money, and music can be compared [@levy2012root; @bartra2013valuation].
A critique, published in *Behavioral Neuroscience*, argues that value-related
signals may reflect salience, arousal, or attention, that OFC coding adapts to
the range of values on offer, and that choice can bypass value altogether
through direct learning of action policies; its central claim is that a
cardinal common-currency value is not, by default, represented in the brain
and used for choice [@hayden2021case].

Within a task, the causal evidence grew stronger. Electrically stimulating the
OFC of macaque monkeys shifted their choices by raising the value of an
individual offer [@ballesta2020values], and in intracranial recordings from 36
patients with epilepsy, subjective value could be decoded from vmPFC and
lateral OFC activity, with both a linear signal (value) and a quadratic one
(confidence) [@lopezpersem2020four]. OFC neurons adapt to the range of values
on offer, but only partially, and in a linear decision model the resulting
raised floor of activity increases choice variability [@conen2019partial].
Value is also constructed during the decision. A *metacognitive control* model,
in which a fast, uncertain initial estimate of value is refined under a
trade-off between mental effort and confidence, predicts at once the response
time, confidence, changes of mind, and choice-induced preference change, and
its authors argue that a preference reversal can be a re-evaluation rather than
noise [@lee2021trading]. The size of choice-induced preference change is in
@sec-jdm-choice.

**What this means for PBO.** The evidence supports a scalar latent utility
within one task and one session, identifiable only up to a monotone
transformation (@sec-pref-identifiability), in line with the argument that
preference learning needs only order consistency [@sun2025rethinking]; it does
not support reusing a learned scale in another task without recalibration
(inference). Because coding partly adapts to the range
presented, overly wide exploratory queries early on may leave higher
variability later (inference). Confidence is encoded together with value, so
scaling the noise of the probit link by the confidence of each comparison is a
direct way to implement heteroscedastic noise (inference). All of this
evidence comes from food, consumer goods, juice offers, and money, so for
continuous design parameters these are working hypotheses (@sec-neuro-status).

## Neuromarketing and neuroforecasting {#sec-neuro-forecasting}

A common claim holds that neural signals reveal preferences people cannot
report. Its best-known support is a study in which activity in the nucleus
accumbens (part of the ventral striatum) of adolescents listening to songs
predicted the songs' later sales while their own ratings did not
[@berns2012neural]. The evidence base is small: after exclusions there were 27
adolescents; 87 of the 120 songs had sales data; accumbens activity correlated
with log sales at $R = 0.32$ ($p = 0.004$), the mean likability rating at $R =
0.110$ (not significant); a logistic regression classified only 30% of the hits
correctly; and the data came from an experiment originally designed to study
social influence [@berns2012neural]. We found no independent direct
replication. A later paper reporting 97% accuracy in classifying hit songs
[@merritt2023accurately] was audited: the data had been oversampled before they
were split into training and test sets, covered only 24 songs and 33 listeners,
and the corrected accuracy was hardly better than chance [@kapoor2023no], a
non-peer-reviewed audit.

The sturdier progress is at the level of populations. In 30 participants
scanned with functional magnetic resonance imaging (fMRI), nucleus accumbens
activity predicted the outcomes of internet crowdfunding campaigns weeks later
while the participants' own choices did not, and the result replicated in a
second study [@genevsky2017brain]. With internet samples of 2,956 and 992
people, forecasts based on behavior varied with the demographic
representativeness of the laboratory sample, while forecasts based on brain
activity stayed significant [@genevsky2025neuroforecasting]. For individuals
the gain is limited: combining several electroencephalography (EEG) measures
with machine learning predicted each person's most and least liked of six food
products with 68.5% accuracy, 4.07 percentage points above self-report alone
[@hakim2021machines]. We found no meta-analysis of the accuracy of neural
prediction, and the out-of-sample evidence comes mainly from one laboratory.

**What this means for PBO.** PBO learns one person's utility, while the
advantage of neural forecasting is documented for aggregates, so its most
defensible use is a population prior or a warm start, not a personal
likelihood (inference). A session of a few dozen comparisons cannot train a
personal decoder, and any PBO system that uses a neural proxy needs strict
held-out evaluation (inference).

## Sequential sampling and response times {#sec-neuro-ddm}

A common claim, drawing on decision field theory [@busemeyer1993decision] and
the attentional drift-diffusion model [@krajbich2010visual], treats response
time as a natural measure of preference uncertainty: a fast answer signals a
large difference in value, a slow one a small difference. Two theoretical
results back it: in preference-based bandits, response times are most
informative for *easy* queries [@li2024enhancing], and choices and response
times together can identify the distribution of latent preferences at several
points, according to a 2026 working paper [@benkert2026time].

### The drift-diffusion model {#sec-neuro-ddm-model}

The **drift-diffusion model** (DDM) describes a two-option decision as noisy
evidence accumulating over time [@ratcliff1978theory]. Think of a counter that
starts at zero. Every few milliseconds it moves up a little if the sample of
evidence favors option A and down if it favors B; on average it drifts at rate
$v$, the *drift rate*, but each step is noisy. The decision is made when the
counter first reaches $+a$ (choose A) or $-a$ (choose B); the *boundary* $a$
expresses caution. The response time is the time to reach a boundary plus a
fixed *non-decision time* $t_0$ for perceiving and pressing a key. In
value-based choice the drift is proportional to the difference in value,
$v = k\,\Delta u$ with $\Delta u = u(A) - u(B)$.

Two closed forms describe the model with unit noise [@bogacz2006physics]. The
probability of choosing A and the mean decision time are

$$
\Prob(\text{choose A}) = \frac{1}{1 + e^{-2 a v}}, \qquad
\E[T] = \frac{a}{v} \tanh(a v).
$$ {#eq-neuro-ddm}

The first is the logistic function of $2 a k \Delta u$. A model of the decision
*process*, in other words, produces the logistic (Bradley-Terry) link of
@sec-bradley-terry as its choice probability, with a slope set jointly by the
person's caution $a$ and the drift scale $k$.

::: {.derivation title="Why the drift-diffusion model gives the logistic link" collapsed=true}
1. Let $h(x)$ be the probability of reaching $+a$ before $-a$ when the counter
   is at $x$. Over a short time $dt$ the counter moves by $v\,dt$ plus noise
   of variance $dt$, so $h(x) = \E[h(x + v\,dt + \sqrt{dt}\,Z)]$ with
   $Z \sim \N(0, 1)$.
2. Expand $h$ to second order and take the expectation, using $\E[Z] = 0$ and
   $\E[Z^2] = 1$: $h(x) = h(x) + v\,h'(x)\,dt + \tfrac12 h''(x)\,dt$, so
   $\tfrac12 h'' + v\,h' = 0$.
3. The general solution of this linear equation is
   $h(x) = A + B\,e^{-2 v x}$, as substituting it confirms.
4. The boundaries fix the constants: $h(-a) = 0$ and $h(a) = 1$ give
   $h(x) = \dfrac{1 - e^{-2v(x + a)}}{1 - e^{-4 v a}}$.
5. At the start, $x = 0$. Factor the denominator as
   $1 - e^{-4va} = (1 - e^{-2va})(1 + e^{-2va})$ and cancel:
   $h(0) = \dfrac{1}{1 + e^{-2va}}$, the logistic function of $2av$.
:::

The model makes the claim about response times precise. When $\Delta u = 0$
the drift vanishes, choices are coin flips, and the mean decision time takes
its largest value, $a^2$. As $|\Delta u|$ grows, the choice probability
saturates near 0 or 1 while the response time keeps falling. Beyond a moderate
utility difference the choice says almost nothing new, but the response time
still does, which is why response times help most for easy queries
[@li2024enhancing].

```{figure}
//| figure: neuro-ddm
//| label: fig-neuro-ddm
//| fig-cap: "The drift-diffusion model. Top: sample paths of accumulated evidence for option A over option B, ending at the upper boundary (choose A, blue) or the lower one (choose B, orange); the strips show the decision times of 600 simulated decisions for each answer, leaving out decisions still undecided after 3 seconds. Bottom: the closed-form probability of choosing A, a logistic function of the utility difference, and the mean response time, which peaks at Δu = 0; the dot marks the current pair. The overall-value effect multiplies the drift by 1 + 1.5V, an illustrative stand-in for the finding that pairs of good options are decided faster and more accurately; the dashed curves show the same pair at V = 0. Drift scale, non-decision time (0.3 s), and the size of the value effect are illustrative."
```

Things to try:

- Set $\Delta u = 0$. About half of the simulated decisions end at each
  boundary, and the decision-time distributions are widest.
- Raise $\Delta u$ to 1 and beyond. The choice probability is already near 1,
  while the mean response time keeps dropping: in this range only the response
  time distinguishes a large utility difference from a very large one.
- Raise the caution $a$. Choices become more accurate and slower, and the
  logistic curve steepens: the slope of the link, which a comparison model
  absorbs into its noise scale $\sigma$, reflects the person's caution as well
  as their utility.
- Switch on the overall-value effect and raise $V$. The same utility
  difference now yields a faster and more accurate choice, and a model that
  ignores overall value would read the response as a larger utility gap.

### Attention and response times, measured {#sec-neuro-ddm-evidence}

The causal effect of attention now has a number, which @fig-neuro-attention
shows. Pooling experiments that manipulated attention in two-option
preferential choice, increasing the total time an option was looked at raised
the probability of choosing it to $P = .541$ (95% confidence interval
$[.523, .560]$), controlling which option was looked at last gave $P = .532$
($[.518, .547]$), and manipulating the first fixation had no effect
($P = .507$, $[.497, .516]$, $p = .18$); slight publication bias lowers these
values a little [@bhatnagar2022meta]. The discount on the unattended option is
not a constant. In the study that introduced the attentional model, 0.3 was
the best fit to the pooled data of 39 participants, while fits to each
participant gave a mean of 0.52 with a standard deviation of 0.3
[@krajbich2010visual]; in a preregistered study ($N = 61$), gaze effects
weakened when the overall value of the options was high [@ting2025high].

```{figure}
//| figure: neuro-attention
//| label: fig-neuro-attention
//| fig-cap: "How much attention moves choice. Top: the pooled probability of choosing an option when an experiment steered attention toward it, with 95% confidence intervals, from a meta-analysis of two-option preferential choice [@bhatnagar2022meta]; the dashed line at 0.5 is no effect, and slight publication bias would lower each value a little. The readout converts the selected row into answers out of a session in which every comparison carried that manipulation, which is arithmetic on the pooled values, not a further finding. Bottom: the discount on the option not being looked at in the attentional drift-diffusion model, fitted to the pooled data of 39 participants and to each participant separately (mean and one standard deviation) [@krajbich2010visual]."
```

Things to try:

- Compare the three intervals with the 0.5 line. Those for total looking time
  (.523 to .560) and for the last fixation (.518 to .547) lie above it; the one
  for the first fixation (.497 to .516) contains it.
- Select total looking time and set the session to 40 comparisons. If every
  comparison carried that manipulation, the option looked at longer would win
  about 1.6 more answers (0.9 to 2.4) out of 40 than with no effect.
- In the lower panel, the pooled discount, 0.3, lies below the mean of the
  individual fits, 0.52, whose standard deviation, 0.3, is as large as the
  pooled value itself.

Response time measures preference strength, with confounds. In three domains
of choice, decisions between two high-value options were mostly faster and
more accurate, which in the DDM means a higher drift rate, contrary to
diminishing sensitivity to value [@shevlin2022high]; the negative relation
between overall value and response time replicated in @ting2025high.
Response-time proxies for preference strength are often noisy and confounded
[@kaufmann2025responserank].

### Response times enter preference learning {#sec-neuro-rt-learning}

The main development of 2017 to 2026 is that response times entered preference
learning with guarantees, Gaussian processes included.
@shvartsman2024response approximated the DDM likelihood differentiably with a
family of skewed three-parameter distributions, so that response times can
enter a Gaussian process model of binary choice, and report better estimates of
latent value and predictions of held-out choices on three real psychophysics
and preference data sets (UAI 2024; @sec-obs-extensions gives the details).
@sawarni2025preference built a loss that uses response times under the
**EZ-diffusion model**, a simplified DDM whose parameters can be computed in
closed form from the mean and variance of response times and the accuracy
[@wagenmakers2007ez]; for linear rewards it turns the error of standard
preference learning, which grows exponentially with the magnitude of the
reward, into polynomial growth, and it extends to nonparametric reward spaces.
A July 2026 preprint analyzes each pairwise label under rational inattention,
the idea that people pay for information and attend only as much as it is
worth (@sec-econ-inattention). It argues that heterogeneous attention can lead
a Bradley-Terry model to recover a misleading ranking, and that in perceptual
comparisons response times and gaze carry information about the size of the
gap that labels lack [@xing2026attention].

**What this means for PBO.** A joint likelihood of choice and response time
can already be used with a Gaussian process surrogate. It should take overall
value (or a proxy, such as the current posterior means of the two options) as
a covariate of the drift rate or the boundary; otherwise fast answers late in
a session, when both options are good, will be read as large utility
differences (inference), the misreading @fig-neuro-ddm shows. Simultaneous,
side-by-side presentation should be preferred, and options that must be
experienced in sequence (audio, touch, gait) should have their order
counterbalanced, with an order bias term in the likelihood; the expected
presentation effect is a few percentage points of choice probability
(inference). The spread of individual discounts supports per-user bias or
discount parameters rather than a population constant such as 0.3 (inference).
The existing response-time Gaussian process work used offline data; the
experiments that would test these recommendations with people in the loop are
listed in @sec-neuro-status.

## Efficient coding and normalization {#sec-neuro-coding}

The Weber argument holds that as the optimizer converges on the high-utility
region, the person's ability to tell options apart declines, so queries should
target regions where the expected utility difference exceeds the
just-noticeable difference (the smallest difference a person reliably
detects). A related argument treats comparison noise as fixed expression noise
around a stable utility, with rational inattention [@sims2003implications;
@matejka2015rational] as its microfoundation.

For value judgments, the Weber argument fails: decisions between high-value
options are faster and more accurate (@sec-neuro-ddm-evidence). **Efficient
coding**, the principle that a system with limited capacity should spend its
resolution where inputs are most frequent, predicts that discrimination tracks
the *density* of values in the environment rather than their magnitude.
@polania2019efficient had 38 participants rate and choose among 64 familiar
foods, the first of four experiments with 127 participants in all: a valuation
model that maximizes information under limited coding resources explained
rating variability, choice consistency, and confidence together, with choices
predicted to be more consistent at values where the prior density is high.
Related work found that rarer numbers are coded more noisily
[@pratcarrabin2022efficient], that in two preregistered experiments risk-taking
was more sensitive to payoffs that occurred more often
[@frydman2022efficient], and that monkeys chose more sensitively in
environments with lower reward variance [@zimmermann2018multiple]. The noise
itself changes with context and history, then, rather than being fixed
expression noise around a stable utility.

**Divisive normalization**, in which each option's value is divided by a
measure of the total value on offer, remains contested. A discrete choice model
with divisive normalization fitted effects of choice-set size and composition
better than alternatives including range normalization
[@webb2021normalization], but a distractor effect attributed to normalization
failed to replicate (@sec-jdm-context), and in tasks with learned values range
normalization won (@sec-jdm-range). A distinction between early and late noise
offers a reconciliation: when the representation of the options is blurry
(early noise), a third option improves discrimination, and when time pressure
dominates (late noise), the context harms it; a divisive normalization model
with both kinds of noise fitted best [@shen2025early].

**What this means for PBO.** An acquisition function that concentrates queries
near the incumbent narrows the range of utilities the person experiences.
Efficient coding predicts that sensitivity in that region rises during the
session, partially and with a lag, and a probit likelihood with a fixed noise
scale would misread that late precision as a steeper utility surface
(inference). A noise scale that depends on how typical the options' utilities
are, relative to those presented so far, can represent this; a simpler practice
is to insert a few fixed reference pairs from rarely visited regions and
measure discrimination there directly (inference). Whether Weber-style
compression or efficient coding dominates in a domain can be checked with a few
repeated pairs at different stages of a session: for value judgments the
evidence favors efficient coding, while perceptual parameters (stiffness,
torque, color) have floors (@sec-neuro-motor) (inference).

## Predictive processing and active inference {#sec-neuro-active-inference}

The **free-energy principle** proposes that brains, and perhaps all
self-organizing systems, act to minimize variational free energy, a bound on
how surprising their sensations are under their internal model
[@friston2010free]. **Active inference** applies it to action: an agent chooses
actions that minimize *expected free energy*, and preferences are prior
beliefs about the outcomes the agent expects to observe. Expected free energy
decomposes into a pragmatic value (reaching preferred outcomes) and an
epistemic value (gaining information) [@friston2015active]. The decomposition
has been called "exactly" the balance of exploration and exploitation in PBO,
with an exploration term "derived from first principles rather than added by
hand" like the weight of an upper confidence bound (@sec-ucb).

That claim needs qualification. A functional argued to be the natural extension
of variational free energy into the future actively *suppresses* exploration,
so exploration does not follow directly from minimizing free energy
[@millidge2021whence]. The first theoretical guarantee for agents that minimize
expected free energy, a 2026 preprint, shows that "sufficient curiosity"
ensures both posterior consistency and bounded cumulative regret, while too
little curiosity leads to myopic exploitation and too much to unnecessary
exploration and regret [@li2026curiosity]. The weight, in other words, still
has to be set, just like the exploration weight of an upper confidence bound.
The foundations are contested too: Markov blankets, the boundaries that
separate a system from its environment in the theory, are persistently
conflated between a tool of inference and physical boundaries
[@bruineberg2021emperor], and the "dark room problem", that an agent minimizing
surprise should seek a dark, quiet room, is a real difficulty for the idea that
preference is prediction [@klein2018predictive]. Three 2026 preprints combine
active inference with Bayesian optimization: the regret guarantee above;
"pragmatic curiosity", which scores queries by the information gained about
task-relevant latent variables plus a potential of expected regret
[@li2026pragmatic]; and BOBA, an acquisition function that adds lookahead
uncertainty about how a time-varying objective will change, which improved
regret on synthetic dynamic benchmarks [@kelly2026boba].

**What this means for PBO.** As of September 2026, active inference is not a
usable alternative to PBO (inference, from the studies above). Every concrete
implementation in Bayesian optimization reduces to an acquisition function, a
pragmatic term plus a weighted information-gain term, close to existing
information-theoretic acquisition functions, and it replaces neither the
Gaussian process preference model nor the pairwise likelihood. The abstracts of
the three preprints report no experiments with human pairwise feedback (we did
not read the full texts), and no study compares an expected-free-energy
acquisition function with EUBO-type acquisition functions (@sec-eubo) on human
preference data. And the two frameworks mean different things by preference:
in active inference it is the agent's prior over its own observations, in PBO
the person's utility is a hidden state to be inferred, so rewriting PBO as
active inference re-encodes the objective without adding any constraint on the
structure of human preference (inference). One role could be useful: modeling
the person as an agent whose prior preferences are learned and updated
[@sajid2021active] would give a generative model of preference drift, though
we found no human preference data fitted that way (inference).

## Psychopharmacology and addiction {#sec-neuro-pharma}

Pharmacology offers what look like the strongest counterexamples to stable
preference. A common claim distinguishes "wanting", attributed to mesolimbic
dopamine, from "liking", attributed to opioid hedonic hotspots in the brain
[@berridge1998role], and proposes modeling the two with a multi-output Gaussian
process. Evidence that the two separate in human self-report is weak. In a
double-blind within-subject study ($n = 27$), levodopa raised and risperidone
lowered both musical pleasure and music-related motivation together
[@ferreri2019dopamine]; in 131 volunteers, the dopamine antagonist amisulpride
and the opioid antagonist naltrexone both reduced physical effort, but neither
changed subjective ratings of wanting or liking [@korb2020dopaminergic].
Dopaminergic drugs do change behavior. In a cross-sectional study of 3,090
treated patients with Parkinson's disease, impulse control disorders were more
common among patients taking a dopamine agonist than among those not taking one
(17.1% against 6.9%, odds ratio 2.72) [@weintraub2010impulse], and in 31
healthy men levodopa weakened directed exploration [@chakroun2020dopaminergic].

**What this means for PBO.** Pairwise choices and ratings may not separate
wanting from liking, so a two-output model would need different observation
channels, such as effort for wanting and ratings during consumption for liking
(inference). Medication state is a slow context variable: studies of PBO in
health and rehabilitation should record it and allow for drift over weeks that
need not be monotone, and since exploration and noise vary with dopamine
state, noise parameters should be fixed per session rather than per person
(inference). When a drug changes what a person prefers, which preference counts
is a question of authority; in clinical settings a reflective endorsement
check, made away from the peak effect of a drug or at follow-up, is a
reasonable safeguard (inference), one @sec-phil-autonomy discusses. No study
has elicited pairwise preferences under a pharmacological manipulation.

## Motor learning and adaptation {#sec-neuro-motor}

A common claim holds that PBO works well where preferences are experiential
and users have real expertise. The claim about expertise needs refining: in an
ankle exoskeleton, users the study calls knowledgeable (the two groups
differed in technical background) preferred higher torque than naive users,
and naive users' preferred torque rose over the course of the experiment, so
knowledge and exposure change the preferred setting itself, not just the noise
around it [@ingraham2022role].

People adapt on several time scales while a device is tuned for them. In
@poggensee2021adaptation, after training with moderate variability,
personalized assistance reduced metabolic cost by 39% relative to the
exoskeleton switched off, training contributed about half of the benefit and
personalization about a quarter, and becoming an expert user took about 109
minutes of assisted walking. When people met a new exoskeleton context, the
variability of step frequency, ankle angle range, and muscle activity first
rose and then fell, on time scales that differed between variables
[@abram2022general]. Preferences are repeatable but change: when 12 naive and
12 knowledgeable participants tuned the torque magnitude and timing of an ankle
exoskeleton themselves, their preferences ranged from 7.9 to 19.4 N·m and from
54.1% to 59.2% of the gait cycle, with trial-to-trial standard deviations of
1.7 N·m and 1.5%, and convergence within a trial took 105 seconds
[@ingraham2022role]. The preferred stiffness of a prosthetic
ankle maximized the kinematic symmetry between the prosthetic and the intact
joint and was not significantly related to body weight or metabolic rate
[@clites2021understanding]. Perceptual resolution differs by a factor of about
five between judging the stiffness of a prosthetic ankle and that of an ankle
exoskeleton (inference, from two studies with different devices and people):
eight people with below-knee amputations detected a 7.7% change in stiffness
with 75% accuracy [@shepherd2018amputee], while walking in an ankle exoskeleton
the just-noticeable difference was about 42% for stiffness [@maberry2026just].
The optimizer and the person also adapt to each other: in three online
experiments on co-adaptation, when the machine used a policy gradient, human
behavior moved toward the machine's global optimum at the person's expense,
although the machine never estimated the person's utility [@chasnov2025human].

**What this means for PBO.** Adaptation drift has structure: variability rises
and then falls, parameters adapt at different rates, and novices' preferred
magnitudes rise with exposure. A temporal kernel with a fast and a slow
lengthscale, in the spirit of the two-state model of motor adaptation
[@smith2006interacting], suits this better than a single exponential decay,
and a familiarization phase with lower weights on early comparisons is a
simpler alternative (inference). Because the optimizer's update rule changes
where the user ends up, the final setting should be compared with alternatives
again after a washout period, with the optimizer frozen (inference). Once the
remaining differences fall below a parameter's just-noticeable difference,
further comparisons carry almost no information, which gives a grounded
stopping rule for that parameter; this is where a Weber-style floor does apply
(inference). The preferred optimum differs from the physiological one, which
supports a multi-objective formulation (@sec-multi-objective) that keeps
preference and physiology separate (inference). And simpler methods may
suffice: in a study of user-driven manual tuning, people tuned exoskeleton
assistance in about 11 minutes, testing 30.5 settings, and reduced metabolic
cost by 16.6% [@schafer2026user]. The exoskeleton case study
(@sec-cs-exoskeleton) and @sec-health-exo work through these trade-offs.

## What the evidence changes {#sec-neuro-summary}

@tbl-neuro-changes collects the claims examined in this chapter and where the
evidence leaves them; the status column describes only the direction of the
evidence. The claims of all chapters in this part, with what each means for
PBO, are gathered in @tbl-syn-weakened and @tbl-syn-strengthened.

::: {.table #tbl-neuro-changes title="Claims about preference examined in this chapter, and where the evidence from 2017 to September 2026 leaves them."}
| Claim | Status | Key evidence | Where |
|---|---|---|---|
| Discrimination declines near the optimum (Weber) | fails for value judgments, holds for perceptual parameters | high-value decisions faster and more accurate; just-noticeable stiffness difference while walking about 42% | @sec-neuro-coding, @sec-neuro-motor |
| Attention raises value; unattended discount a constant 0.3 | direction holds, effect small; discount varies with overall value and person | $P$ about .53 to .54 | @sec-neuro-ddm-evidence |
| Response times carry preference strength | strengthened; entered preference learning with guarantees | a loss with guarantees; Gaussian process likelihood; overall-value confound | @sec-neuro-rt-learning |
| Efficient coding explains value noise | new and strengthened | food values; number coding; risky choice | @sec-neuro-coding |
| Ventral striatum predicts song sales | small sample, unreplicated; a similar 97% result failed an audit | 27 adolescents; 24 songs | @sec-neuro-forecasting |
| Wanting and liking separate in human reports | weak support | drugs move both or neither | @sec-neuro-pharma |
| Active inference derives exploration and is a usable alternative to PBO | not as of September 2026 | the weight is still tuned; no human pairwise experiments | @sec-neuro-active-inference |
| Users adapt during device tuning | strengthened, quantified | about 109 minutes to expertise; co-adaptation | @sec-neuro-motor |
:::

## Settled, contested, missing {#sec-neuro-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** Value signals in OFC and vmPFC are causally related to choice
within a task [@ballesta2020values; @lopezpersem2020four]. Attention causally
affects choice, but only by a few percentage points [@bhatnagar2022meta].
Decisions between high-value options are faster and more accurate
[@shevlin2022high], so the Weber argument fails for value judgments, while
perceptual parameters of devices have measurable discrimination floors
[@maberry2026just]. Response times can enter preference learning with
guarantees [@sawarni2025preference]. Neural forecasts work better for
aggregates than for individuals [@hakim2021machines]. People adapt to a device
on several time scales while it is tuned [@poggensee2021adaptation].

**Contested.** Whether a cardinal common currency exists across tasks
[@hayden2021case]. Divisive against range normalization, and which applies to
which task. Whether active inference adds anything to PBO beyond a weighted
information-gain acquisition function, and whether its foundations hold.

**Missing.** A PBO study that uses a response-time-augmented Gaussian process
with people in the loop, or that models overall value in a response-time
likelihood. An attentional DDM fitted to pairwise judgments of continuous
design parameters. Any test of common currency or value construction on design
parameters. A test of efficient-coding predictions in an interactive
optimization session. An expected-free-energy acquisition function compared
with EUBO on human pairwise data. Pairwise preference elicitation under
pharmacological manipulation. A validation of a tuned device setting after
washout with the optimizer frozen.
:::

## Further reading {#further-reading .unnumbered}

- @bogacz2006physics derives the drift-diffusion model's closed forms and
  relates them to optimal decision making; it is the clearest route from the
  model to @eq-neuro-ddm.
- @shvartsman2024response and @sawarni2025preference are the two most direct
  bridges from response times to preference learning, one with Gaussian
  processes and one with guarantees.
- @shevlin2022high is a short paper whose result, that high-value decisions are
  fast and accurate, changes how late-session comparisons should be read.
- @bhatnagar2022meta puts a number on how much attention moves choice.
- @polania2019efficient shows efficient coding of subjective value with
  ordinary foods, and is a good entry to noise that depends on context.
- @millidge2021whence and @li2026curiosity together explain why the
  exploration term of active inference still needs a weight.
- @poggensee2021adaptation and @ingraham2022role are the best quantitative
  accounts of how people change while a device is tuned for them.
