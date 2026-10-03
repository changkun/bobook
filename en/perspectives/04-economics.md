---
status: done
synopsis: "What economics, decision theory, and operations research know about preference that preferential optimization uses or ignores: how large behavioral effects really are, what choices reveal, three sources of random choice, rational inattention, aggregation across people, incomplete preferences, adaptive conjoint designs, stated-preference practice, multi-criteria decision analysis, and interactive multi-objective optimization."
---

# Economics, Decision Theory, and Operations Research {#sec-economics}

Preferential Bayesian optimization (PBO) models a person in one sentence: there
is a stable utility function, and each answer to "which do you prefer?" reads
it through a little Gaussian noise. Economics has spent a century testing every
part of that sentence. The probit and logit links of @sec-random-utility came
from its econometrics, adaptive questioning under response error is the daily
business of its marketing science, and its decision theorists have asked
whether a person must be able to rank every pair of options at all. No other
discipline in this part is as close to the method.

The chapter moves from what a single choice reveals to how preferences are
aggregated and represented, then to how practitioners elicit them, and ends
with how well the underlying experiments replicate. Two findings run through
it. First, several problems that PBO papers treat as new already have
worked-out treatments elsewhere, collected in @sec-econ-solved: what PBO adds
is a nonparametric surrogate over a continuous design space, not adaptive
questioning itself. Second, the behavioral parameters most often proposed for
a preference likelihood, such as a loss aversion coefficient or a discount
factor, are exactly the quantities whose magnitudes replicate worst.

## Behavioral economics {#sec-econ-behavioral}

Behavioral economics documents systematic departures from the textbook model
of a rational chooser. The one most often borrowed for PBO is **loss
aversion**: a loss looms larger than a gain of the same size, and the ratio
between the two weights is the loss aversion coefficient. Tversky and Kahneman
estimated it at about 2.25 [@tversky1992advances]. Losses and gains are counted
from a **reference point**, which in the model of Kőszegi and Rabin is the
outcome a person rationally expected [@koszegi2006model]. Applied to PBO, the
argument runs: the current best design becomes the reference point, so a
challenger that is worse in some respect is coded as a loss, and the
likelihood near the incumbent should be asymmetric by a factor of about 2.
Further proposals add present bias (overweighting what comes now against what
comes later, often modeled as quasi-hyperbolic discounting, which shrinks every
delayed outcome by one extra fixed factor on top of a steady discount per
period) across iterations, ambiguity aversion (preferring known odds to unknown
odds, as in Ellsberg's urns [@ellsberg1961risk]) to describe how users trust
the posterior, and a sizable share of purely selfish people in multi-user
settings.

These claims hold in direction and mostly fail in magnitude. Loss aversion and
reference points are two ingredients of **prospect theory**, Kahneman and
Tversky's account of choice under risk, in which outcomes are valued as gains
and losses and probabilities are weighted unevenly, small ones counting for
more than they should and large ones for less (probability weighting). Its
qualitative pattern replicated across 19 countries, 4,098 participants, and 13
languages, though with smaller effects and clear differences between countries
[@ruggeri2020replicating]. The size of loss aversion is contested, as
@fig-econ-loss-aversion shows. A meta-analysis (a statistical pooling of
published estimates) of 607 estimates from 150 articles found a mean of 1.955
(95% interval 1.820 to 2.102) [@brown2024meta]. Fitting cumulative prospect
theory, the 1992 version of the theory, to individual choices between mixed
gambles (gambles that can end in a gain or a loss) gave 1.31 (95% confidence
interval 1.10 to 1.53), with heterogeneity between studies accounting for 91.6%
of the total variation [@walasek2024meta]. A reanalysis of 163 estimates from
84 papers (n = 149,218) found that the coefficient depends on the design of the
task: it is about 2.33 when losses are smaller than gains and the items are
presented in order of size, and about 1.07, not significantly different from 1,
when gains and losses are symmetric and unordered [@yechiam2025loss].

```{figure}
//| figure: econ-loss-aversion
//| label: fig-econ-loss-aversion
//| fig-cap: "How large is loss aversion? Each row is an estimate of the loss aversion coefficient reported in this section, with its 95% interval where the text gives one: the original estimate [@tversky1992advances], a meta-analytic mean [@brown2024meta], a meta-analysis of cumulative prospect theory fitted to individual choices [@walasek2024meta], and the two task designs of a reanalysis [@yechiam2025loss]. The vertical line is a coefficient that a preference likelihood might fix; the readout says which intervals contain it. At 1, losses and gains weigh the same."
```

Things to try:

- At the default, 2.25 (Tversky and Kahneman's estimate), the line lies outside
  both pooled intervals: above 1.820 to 2.102 and far above 1.10 to 1.53.
- Move the line anywhere between 1.53 and 1.82. It lies in neither interval,
  and no setting lies in both: the two pooled estimates do not overlap.
- Set the line to 1. It lies below both intervals but next to the estimate of
  about 1.07 for symmetric, unordered presentations, the condition closest to
  showing two designs side by side.

The supporting mechanisms fare no better. In a real-effort experiment (one in
which subjects perform an actual task for pay), effort responded to changes in
expectations in a clearly non-monotonic way, against the monotonic response
that models of the Kőszegi-Rabin type predict [@gneezy2017limits]. The
**endowment effect**, owners asking more to give up a good than buyers offer
for it [@kahneman1990experimental], showed little relation to loss aversion for
risky prospects in four incentivized surveys (answers had real monetary
consequences) of 4,000 American adults, according to a working paper
[@chapman2023willingness]. List's field experiments with traders found that
market experience reduced the endowment effect [@list2003does;
@list2004neoclassical], which has been read as showing that repeated
questioning makes preferences converge; but when subjects were made to trade
repeatedly, the asymmetry disappeared in later tests, and the authors attribute
it mainly to uncertainty about trading [@engelmann2010reconsidering].
Experience brings learning, but the mechanism is not that preferences converge.

The other parameters are shakier still. A meta-analysis of 220 estimates from
28 articles using convex time budgets (subjects split a budget between an
earlier and a later date) found present bias on average, with large
heterogeneity and modest selective reporting [@imai2021meta], and hyperbolic
patterns (much more impatience over a short delay now than over the same delay
later) appear almost as strongly in tasks with no payment delay at all, so
complexity-driven errors produce hyperbolic-looking choices even when the
underlying discount function is exponential, discounting every period at the
same rate [@enke2025complexity]. This reading of anomalies as errors in valuing
complex options [@oprea2024decisions] is itself disputed
(@sec-econ-replication). Ambiguity aversion replicated for gains at moderate
probabilities, but with losses or low probabilities most subjects were
ambiguity neutral or seeking [@kocher2018ambiguity]. A structural mixture model
of social preferences, which sorts people into a few types with their own
preference parameters, found three other-regarding types (types that also weigh
what others receive) that were stable over time and no purely selfish type
[@bruhin2019many], although a later study did identify a "predominantly
selfish" type [@fehr2026social]. Defaults are robust on average: across 58
studies (combined n = 73,675) the effect was Cohen's d = 0.68 (95% confidence
interval 0.53 to 0.83), where d is a difference in means divided by the
standard deviation, but with large variation, several null results, and two
negative effects [@jachimowicz2019defaults].

**What this means for PBO.** A likelihood should not fix an asymmetry of about
2 near the incumbent. PBO usually shows two designs side by side and
symmetrically, which is the condition under which Yechiam and Zeif found a
coefficient close to 1; if an asymmetry is modeled at all, it belongs in a
per-user parameter whose prior includes 1 (inference). Quasi-hyperbolic
discounting across iterations does not fit the setting, since iterations are
minutes apart and nothing is consumed later; what carries over is that measured
impatience partly reflects complexity, which points to noise that grows with
the difficulty of the comparison (@sec-obs-extensions) (inference). For several
users, stable types favor a finite mixture of user types or a hierarchical
prior (each user's parameters drawn from a shared population distribution)
over fully independent per-user models (inference). And the initial design of
a session is a default, so it should be randomized or balanced across sessions
and users (inference).

## Revealed and stable preferences {#sec-econ-revealed}

**Revealed preference** is the economist's method of inferring preferences from
choices instead of asking about them: if a person chose $x$ when $y$ was also
available, $x$ is revealed preferred to $y$. With choices from budgets (the
sets of bundles a person can afford at given prices and income), some utility
function explains all the choices exactly when they satisfy the Generalized
Axiom of Revealed Preference (GARP): the revealed-preference relation may
contain no cycle that includes a strict preference. Afriat's critical cost
efficiency index [@afriat1972efficiency] measures how far data are from
passing: it is the largest fraction $e \le 1$ such that shrinking every budget
to $e$ of its size removes all violations, so an index of 0.95 means the
choices become consistent once the person is allowed to waste 5% of their
money. In Andreoni and Miller's experiment on altruism, 18 of 176 subjects
violated at least one revealed-preference axiom and only 3 had an index below
0.95 [@andreoni2002giving]. Combined with Plott's discovered preference
hypothesis, that preferences are not available at first but are discovered
through experience with a task [@plott2001rational], such results support a
strong claim about PBO: that iterative questioning converges to a person's true
underlying preference.

That reading needs three qualifications. Power indices for these tests are
"better thought of as ordinal rather than cardinal measures of power"
[@andreoni2013power] (a working paper), that is, they rank tests by power but
do not say how far apart two tests are, so a high pass rate need not mean the
test was demanding. Consistency may not carry across domains: a 2025 preprint
found that consistency scores from risk choices in an experiment were
uncorrelated with the consistency of the same people's supermarket purchases
[@chen2025general]. And forced choice itself manufactures inconsistency: when
subjects could pay a small cost to defer a choice, their active choices were
"generally more consistent", and deferral and choice together fit a model of
dominant choice with incomplete preferences [@costagomes2022choice], the topic
of @sec-econ-decision-theory. The discovered preference hypothesis did receive
direct support. Nielsen and Rehbeck asked subjects, under incentives, which
axioms they wanted their choices to satisfy; subjects often revised conflicting
lottery choices to agree with those axioms [@nielsen2022choices].

**What this means for PBO.** A session's comparison log is revealed-preference
data on two-element menus, and for such data GARP reduces to a simple
condition: the directed graph with an edge from each winner to its loser must
have no cycle. A system can check this throughout a session with cycle
detection on the comparison graph of @sec-comparison-graph, without a Gaussian
process (inference), although acquisition functions concentrate queries near
the incumbent, so these tests have little power late in a session and a clean
log is weak evidence (inference).

::: {.code title="Checking a session log for revealed-preference cycles"}
```python
import networkx as nx

def revealed_cycles(duels):
    """duels: (winner, loser) pairs from one session.
    Returns every cycle of the revealed-preference graph; an empty
    list means some utility function explains all answers."""
    g = nx.DiGraph()
    g.add_edges_from(duels)
    return list(nx.simple_cycles(g))

# The fewest answers whose reversal removes every cycle form a minimum
# feedback arc set: NP-hard in general, small for session-sized logs.
```
:::

When the log contains a cycle, showing it to the user and asking which answer
to revise is the Nielsen and Rehbeck procedure: preference discovery through
reflection, not convergence through repetition (inference). Offering a "no
clear preference" answer should reduce manufactured inconsistency, with a third
outcome in the likelihood (@sec-ties) (inference).

## Endogenous and changing preferences {#sec-econ-endogenous}

PBO assumes the utility it learns does not change because it is being learned.
Economists call preferences **endogenous** when they are shaped by the very
institutions, choices, or experiences being studied. Machine learning's drift
models and non-stationary bandits (methods for repeated choices whose payoffs
shift over time, @sec-bandits) treat change as exogenous, coming from outside,
while constructed preferences change endogenously. If preferences are
endogenous, conventional welfare economics, which judges outcomes by the
preferences people hold, loses its footing; Dold asks whether "welfare analysis
and normative economics are still possible" in such a world
[@dold2023endogenous]. Infante, Lecouteux, and Sugden criticized the
"preference purification" of behavioral welfare economics, the search for the
preferences of an inner rational agent behind observed errors
[@infante2016preference], which suggests that the optimum a PBO session finds
might reflect a self-reinforcing construction rather than a discovery. The
empirical answer is again Nielsen and Rehbeck's result, which supports a
limited purification endorsed by the person, not purification by a third party.

Several welfare criteria do without stable preferences. Bernheim and Rangel
proposed that one option counts as better for a person than another only if
the second is never chosen over the first in any frame (way of presenting the
choice) in which both appear [@bernheim2009revealed]; Sugden's opportunity
criterion judges institutions by the opportunities they offer rather than by
stable preferences, set out in a book whose existence we verified through a
review [@engelen2019community]; and with a welfare measure for people who
misunderstand consequences, conventional metrics turned out to rate two
financial-education interventions as equally beneficial although only one
reduced the average severity of mistakes [@ambuehl2022evaluating].

**What this means for PBO.** When preferences change during a session, the
system needs an explicit welfare criterion. The Bernheim-Rangel rule applied to
the session log recommends design $A$ over design $B$ only if $B$ was never
chosen over $A$ in any observed frame or at any time, and otherwise reports the
set of designs that no other design unambiguously beats (inference). A check in
the spirit of Ambuehl and colleagues compares the final design once more with
its main alternatives in a simplified frame with labeled attributes; the
disagreement between the two frames estimates how much of the result rests on
misunderstanding rather than preference (inference). The Nielsen and Rehbeck
endorsement step is the economic counterpart of a reflective-endorsement check
at stopping time (@sec-rec-stopping) (inference), with the limit that
@sec-phil-autonomy explains: when the system is what changed the preference,
endorsement after the fact cannot by itself show the change was legitimate.

::: {.keyidea title="A gap no field has filled"}
We found no economics paper that defines welfare for a person whose preferences
are changed by the elicitation process itself, as opposed to changes caused by
consumption, institutions, or worldviews. That is the case PBO creates.
:::

## Stochastic choice {#sec-econ-stochastic}

Asked the same question twice, people often answer differently. **Random
utility models** (@sec-random-utility) explain this by giving each option's
utility a random component; the probit and logit links of @sec-obs-baseline are
random utility models with Gaussian and Gumbel noise respectively (the Gumbel
is a skewed bell curve, and the difference of two independent Gumbel draws has
the logistic distribution, which gives the logit). A long-standing view,
usually traced to McFadden [@mcfadden1981econometric], is that the randomness
can stand for variation within one person across occasions, for differences
between people, or for genuine instability of preference, and that the three
readings are formally equivalent. New experiments separate them. Agranov and
Ortoleva repeated the same question either far apart or back to back, telling
subjects that questions would repeat, and "a large majority of subjects exhibit
stochastic choice in both cases" [@agranov2017stochastic]. When a later study
let subjects randomize on every row of a choice list (a table of binary choices
in which one option stays fixed and the other changes from row to row), "the
majority of subjects chose to randomize in the majority of questions"
[@agranov2025ranges]; an overview is in @agranov2022revealed. Deliberate
randomization is one outward sign of a person who cannot rank two options;
@sec-econ-decision-theory reports how often people say so directly when allowed
to.

Two lines of work since 2017 bear directly on PBO. In **learning-based
stochastic choice**, a Bayesian probit model identifies stable preferences from
context-dependent data and accommodates the attraction and compromise effects
(an option gains when a similar but worse option, or a more extreme one, is
added beside it) that random utility models cannot [@natenzon2019random].
**Response-time models with uncertain utility** cast choice as optimal
sequential sampling, taking noisy looks one at a time and deciding after each
whether to look again, and show that "choices are more likely to be correct
when the agent chooses to decide quickly, provided the agent's prior beliefs
are correct", which matches the data better than the classic drift-diffusion
model of @sec-neuro-ddm [@fudenberg2018speed].

**What this means for PBO.** Evaluation noise, random preference, and
deliberate randomization now each have an empirical signature, and each could
have its own likelihood component: a probit or logit link for evaluation noise,
a utility function drawn afresh from the posterior for each judgment for random
preference, and a third outcome or an explicit mixture parameter for deliberate
randomization (inference). Repeating the same pair immediately and treating
every disagreement as noise confounds evaluation noise with deliberate
randomization; retest pairs should be spaced out, and a 50/50 split on a
repeated pair should not automatically be read as indifference (inference).
Letting comparison noise depend on similarity, as Natenzon's Bayesian probit
model does, would let the likelihood absorb part of the context effects of
@sec-jdm (inference). And the prediction of Fudenberg and colleagues that fast
choices are accurate choices, when the prior is right, differs from the
drift-diffusion intuition that a slow choice signals a small difference in
value; which pattern holds in design comparisons should be tested before
response times enter a likelihood (@sec-obs-extensions) (inference).

## Rational inattention {#sec-econ-inattention}

**Rational inattention** models a chooser who cannot take in everything about
the options and decides how much information to acquire, paying a cost for
attention. Choices come out noisy even when the person's utilities are fixed,
because sometimes it is not worth looking carefully. The theory is due to Sims
[@sims2003implications], and its application to discrete choice to Matějka and
McKay [@matejka2015rational]: with an attention cost proportional to Shannon's
mutual information (@sec-mutual-information), the optimal behavior is a
generalized multinomial logit (the logit rule extended to several options and
shifted by prior beliefs),

$$
\Prob(\text{choose } i) = \frac{P^0_i \, e^{v_i/\lambda}}{\sum_j P^0_j \, e^{v_j/\lambda}},
$$ {#eq-econ-ri-logit}

where $v_i$ is option $i$'s true value, $\lambda$ is the cost of a unit of
information, and $P^0_i$ is the probability of choosing $i$ before looking,
which summarizes the chooser's prior beliefs. The result has been used to argue
that the noise of a preference likelihood has a normative foundation: even
with a fully stable utility, an optimally attentive person answers with noise.

The foundation is narrower than that. With Bregman information costs, a class
defined in that paper that contains the Shannon cost as a special case, and for
a given prior, rationally inattentive choice probabilities "may take the form of
any additive random utility discrete choice model", that is, any model in which
each option's utility is a fixed value plus a random term
[@fosgerau2020discrete]. In experiments, subjects adjusted their attention to
incentives as rational inattention predicts, but the results were
qualitatively inconsistent with a cost linear in Shannon entropy (the standard
measure of uncertainty, so that the cost is proportional to how much the
person's uncertainty shrinks on average) [@dean2023experimental]. The logit
foundation holds as a theorem under Shannon costs, but the empirical case for
that cost is weak; the broader claim, that optimal attention produces random
choice even with stable utilities, still stands.

**What this means for PBO.** The likelihood should be heteroscedastic: the
closer two designs are in the user's perceptual metric, the noisier the
comparison, and the more the user cares, the less noisy (inference). The prior
acts as a utility shift, the $P^0_i$ term in @eq-econ-ri-logit: a user who
believes the system's incumbent is probably good will favor it even with stable
preferences. That explains the anchoring on the system's proposal reported in
interactive studies (@sec-hci-unstable) without any change of preference, and
only designs that change what the user believes about where the options came
from, such as hiding which option is the system's proposal, can tell the two
apart (inference). And because rational inattention can produce any additive
random utility model, it cannot decide between the probit and logit links; only
fit to data can (inference).

## Information economics {#sec-econ-information}

In **Bayesian persuasion**, or information design, a sender commits in advance
to how the information a receiver sees will depend on the state of the world,
choosing that rule to steer the receiver's action [@kamenica2011bayesian]. In
Kamenica and Gentzkow's example, a prosecutor who designs the investigation,
facing a judge who convicts only when the probability of guilt is at least one
half, with a prior probability of guilt of 0.3, can obtain a conviction rate of
60%. From this it has been argued that, in PBO, choosing which candidates to
show is itself information design, which raises ethical questions. Closer to
elicitation, a payment rule is incentive compatible if answering truthfully is
in the respondent's interest, and behaviorally incentive compatible if real
people, not ideal ones, then answer truthfully. The binarized scoring rule, a
standard way to pay people for reporting beliefs, violates two weak conditions
of behavioral incentive compatibility: explaining the incentives to subjects
increased their deviations from truthful reports [@danz2022belief].

**What this means for PBO.** The information structure of PBO is the reverse of
persuasion: the user holds, or is forming, the preference, and the system is
trying to learn it, so choosing queries is an elicitation or screening
problem, in which the side that lacks information designs the questions that
draw it out. Persuasion applies to a different question, how the candidates
shown change the user's beliefs about what is achievable and what is good, and
it matters most when users are unsure how to evaluate the options themselves
(inference). The ethical argument holds in a narrower form: a system that
decides which candidates to show and also benefits from particular outcomes is
a sender whose preferences differ from the user's, and the persuasion framework
bounds how far it can move the user's choices (inference); @sec-rec-ethics
returns to this. If a user study pays for "accurate" answers, the payment rule
should be simple and not explained in mechanical detail (inference).

## Social choice and mechanism design {#sec-econ-social-choice}

**Social choice theory** studies how to combine many people's preferences into
one collective ranking or decision, and judges each rule by axioms it satisfies
or violates; **mechanism design** designs the rules so that participants who
act in their own interest still reveal what the designer needs to know. The
classic results are impossibility theorems. Arrow's theorem says no ranking
rule over three or more options satisfies a short list of reasonable axioms at
once [@arrow1950difficulty]; the Gibbard-Satterthwaite theorem says every
voting rule over three or more outcomes that is not a dictatorship (a rule that
always follows one fixed voter) can be manipulated by some voter
[@gibbard1973manipulation; @satterthwaite1975strategy]. Black's single-peaked
preferences (each person's preference falls away on both sides of a favorite
point along one shared axis) [@black1948rationale], May's theorem on majority
rule [@may1952set], Harsanyi's weighted sums of utilities
[@harsanyi1955cardinal], and the Vickrey-Clarke-Groves mechanisms that use
money to make truth-telling optimal, by charging each participant the cost
their report imposes on everyone else [@vickrey1961counterspeculation;
@clarke1971multipart; @groves1973incentives], complete the toolkit usually
cited when preference learning meets social choice. Chichilnisky's topological
theorem, often stated as saying that on continuous spaces of preferences no
social choice function satisfies unanimity (if everyone agrees, the rule
follows) and anonymity (the rule treats all people alike) together, concerns
*continuous* aggregation maps [@chichilnisky1980social]; the common statement
omits that continuity requirement on the rule.

The main development is that social choice theory is now applied directly to
learning a utility from pooled comparisons, which is exactly the aggregation
problem of PBO with several users (@sec-many-users). @ge2024axioms show that the
Bradley-Terry-Luce model and its broad generalizations "fail to meet basic
axioms": any aggregation rule defined by a nondecreasing convex loss violates
**Pareto optimality** (if everyone prefers $a$ to $b$, the output should too)
and **pairwise majority consistency** (if majorities over every pair agree with
one ranking, the output should be that ranking).
@siththaranjan2024distributional show that under hidden heterogeneity,
standard preference learning implicitly aggregates by **Borda count**, which
scores each option by how many others it beats on average across people. A 2025
preprint shows that standard regularized maximum likelihood is not robust to
approximate clones (adding near-duplicates of an option should not
substantially change the learned reward), and that weighting by similarity
restores the property [@procaccia2025clone].

**What this means for PBO.** A multi-user system that puts everyone's
comparisons into one Gaussian process with a probit or Bradley-Terry likelihood
inherits these axiomatic failures. The alternative is to fit a posterior per
user and aggregate with an explicit rule, or to report the distribution of
utilities when disagreement is large (inference). Acquisition functions
generate many near-identical candidates around the incumbent, which is the
clone situation: even with a single user, a cluster of near-identical
comparisons can dominate the fit, and similarity weighting is a remedy
(inference). Averaging cardinal utility functions is continuous, anonymous, and
unanimous, so Chichilnisky's obstacle does not apply to averaging posterior
means; the real difficulty is that each user's utility scale is fixed only
relative to that user's own noise variance (@sec-pref-identifiability), so
averaging first needs an interpersonal normalization, a decision about how one
user's unit of utility compares with another's, which is a normative choice in
Harsanyi's sense (inference). Strategic misreporting in single-user PBO hurts
only the user; with several users, three or more designs, and non-dictatorial
aggregation, the Gibbard-Satterthwaite theorem applies (inference).

## Decision theory {#sec-econ-decision-theory}

Decision theory states axioms for rational choice and derives what follows from
them, most famously that a person who satisfies Savage's axioms acts as if
maximizing expected utility under some probability belief. Savage himself
restricted his theory to "small worlds" whose possibilities can be listed in
advance [@savage1954foundations]; Binmore has argued that Bayesian decision
theory does not carry over to "large worlds" [@binmore2009rational], and
creative design spaces look like large worlds, in which expecting a complete
preference ordering is unreasonable. Debreu's representation theorems imply
that a continuous utility representation requires continuous preferences,
while lexicographic preferences (ranking by one attribute and using the next
only to break exact ties) cannot be represented by any real-valued utility at
all [@debreu1983representation], so a Gaussian process utility presupposes
continuity. The classical alternative to completeness is Bewley's Knightian
decision theory, which drops the completeness axiom and adds an inertia
assumption: an alternative is accepted only if it is preferred to the status
quo [@bewley2002knightian].

::: {.definition #def-econ-incomplete title="Incomplete preferences"}
A preference relation is **complete** if, for every pair of options $a$ and
$b$, the person prefers $a$, prefers $b$, or is indifferent. It is
**incomplete** if for some pairs none of the three holds: the person cannot
compare them. In a **multi-utility** representation, the person has a set
$\mathcal{U}$ of utility functions and prefers $a$ to $b$ exactly when
$u(a) > u(b)$ for every $u \in \mathcal{U}$; when members of $\mathcal{U}$
disagree, $a$ and $b$ are incomparable. Indifference ($u(a) = u(b)$ for all
$u$) and incomparability are different states.
:::

The most important development of 2017 to 2026 is that incompleteness has been
measured directly. Cettolin and Riedl found that about half of their
participants made choices inconsistent with a model of complete preferences
plus certainty independence (mixing both options, in the same proportion,
with the same sure outcome does not change which one is preferred); of these,
about half behaved in line with incomplete preferences, and probability
weighting, choice errors, regret aversion, and intransitive indifference were
all ruled out as explanations [@cettolin2019revealed]. Nielsen and Rigotti
elicited incompleteness directly, in a working paper [@nielsen2026revealed].
About 40% to 50% of subjects expressed incomplete preferences when allowed to,
and forced choice produced more inconsistency: transitivity violations ran at
4.0% in forced choice against 1.7% among strict preferences expressed when
choice was not forced. Subjects who ever reported incompleteness did so in 3.3
of 50 comparisons on average, and when the objects were made deliberately
complex, 76% reported incompleteness. Decision theory under growing awareness
adds a rule for options that appear later: when new possibilities are
discovered, beliefs update by "reverse Bayesianism", which keeps the relative
probabilities of the old possibilities and moves mass to the new ones
[@karni2013reverse].

What does a forced binary likelihood do with an answer that is not a
preference? In @fig-econ-incomplete you choose between flats that trade living
space against commute time, and may also answer "can't compare" or ask for a
coin flip. The forced-choice model has one weight and the probit likelihood of
@sec-obs-baseline, and receives every "can't compare" as the outcome of the
coin you would otherwise have had to toss. The second model follows
@def-econ-incomplete: you hold a set of admissible weights, and "can't compare"
means the set straddles the pair's tie point.

```{figure}
//| figure: econ-incomplete
//| label: fig-econ-incomplete
//| fig-cap: "Choosing between flats that trade living space against commute time. Each answer sits on the weight axis at the weight where its two flats tie; magenta marks show what a forced binary interface would have recorded for a 'can't compare' or a coin request. Top: a forced-choice model with one weight and a probit likelihood; the curve is its posterior density, scaled so that the flat prior sits at one eighth of the panel. Middle: a model of incomplete preferences in which the person holds a set of admissible weights; the curve is the posterior probability that each weight belongs to that set. Bottom: a shortlist of seven flats, each the best flat on one interval of weights, with the forced model's probability that each is best and the set model's probability that no other flat beats it under every admissible weight (Bewley's criterion). Both models share probit noise of 0.05 on the weight scale and a 5% lapse rate (the chance that an answer is a random slip), and both posteriors are computed exactly on a grid. The linear utility, the attribute ranges, the flats, and the simulated person (admissible weights 0.38 to 0.62) are illustrative. This sketch treats 'can't compare' and a coin request alike, although experiments distinguish them [@cettolin2019revealed; @agranov2025ranges]."
```

Things to try:

1. Answer every pair with Flat A or Flat B only. The two models then roughly
   agree on where your weight lies and on the shortlist. Unless your answers
   push the weight to one end of the axis, the set model's curve stays below
   even odds everywhere, because answers that never say "can't compare" point
   to a set so narrow that it could sit anywhere in that region.
2. Reset and answer again, choosing "Can't compare" or "Flip a coin for me"
   whenever the trade-off feels impossible. Each such answer narrows the forced
   model around a weight chosen by a coin, while the set model widens its range
   of admissible weights and keeps more flats on the shortlist.
3. Switch to the simulated person, press "Simulate all", and reveal the
   simulated weights. In the default run the forced model puts 90% of its
   belief on weights 0.43 to 0.57 and calls flat 4 best with probability 0.71;
   the set model keeps flats 3, 4, and 5, which are the flats that are best
   somewhere inside the person's true range. The forced model's confidence
   comes from the coin.

**What this means for PBO.** Queries should include a "can't compare" answer,
and the likelihood should model it. Under Bewley's multi-utility reading, one
design is better than another only when a whole set of utilities agrees; a
system can return the designs that no other design beats under every posterior
sample instead of a single optimum, although the spread of posterior samples
measures the system's uncertainty about the person, not the person's own
incompleteness, unless "can't compare" answers enter the likelihood and widen
the set, as in the middle panel of @fig-econ-incomplete (inference). The
evidence also bounds the worry that forced choice is a category error: people
who report incompleteness do so in a minority of comparisons, 3.3 of 50 on
average, though design comparisons often are complex (inference). Reverse
Bayesianism gives a rule for expanding the design space in the middle of a
session without restarting, a formal answer to the "large world" objection that
no PBO system implements yet (inference). Lexicographic rules and hard limits,
like the veto thresholds of @sec-econ-mcda, are better represented as
constraints (@sec-constrained-bo) than forced into a smooth utility
(inference).

## Discrete choice and conjoint analysis {#sec-econ-conjoint}

**Conjoint analysis** is a survey method from marketing. Respondents choose
among, or rate, product profiles that vary in their attributes (price, size,
brand), and a model, usually a logit, estimates the **part-worth** utility of
each attribute level. Choice-based conjoint shows a few profiles at a time and
asks for the preferred one, which makes it a close relative of PBO, with a
parametric, usually linear, utility in place of a Gaussian process.
**Adaptive** designs choose each question from the answers so far; one
principle, **utility balance**, picks questions whose options have nearly equal
predicted utility, which is what an acquisition function does when it favors
pairs whose outcome is most uncertain. Before PBO existed, marketing science
had polyhedral adaptive designs [@toubia2003fast; @toubia2004polyhedral], a
probabilistic version that adds response error and informative priors
[@toubia2007probabilistic], Bayesian designs that exploit managers' prior
beliefs about the parameters [@sandor2001designing], designs for mixed logit
models [@sandor2002profile], and sequential Bayesian designs adapted to each
respondent's earlier answers, which in simulations held up under low response
accuracy where the polyhedral method did not [@yu2011individually].

The field also documented the known trap of adaptive questioning. Hauser and
Toubia showed that adaptive, utility-balanced metric questions (paired
comparisons in which respondents state how much they prefer one profile, not
just which) bias part-worth estimates relative to one another, through an
endogeneity like the winner's curse (the questions are chosen because of the
current estimates, so their answers are not independent of those estimates).
Utility-balanced questions can also be inefficient, and shrinkage estimation
(pulling each respondent's estimates toward the sample average) did not remove
the biases, although in their data the biases and inefficiencies were of the
order of the response errors [@hauser2005impact]. Their analysis concerns these
metric questions only. Since 2017, Sauré and Vielma approximated the posterior
with a normal distribution, bringing response error into an approximate
Bayesian method and allowing question selection by optimization
[@saure2019ellipsoidal], and Gibbard and Sadlier found that for
population-level parameters the estimation accuracy of an optimal adaptive
Bayesian design is "uniformly worse than the random design", so such designs
suit studies focused on the specific individuals in the sample
[@gibbard2025optimal].

Environmental economics values goods that are not sold in markets, such as
clean rivers, with **stated preference** surveys, in which respondents state
what they would pay for a described change or choose among described
alternatives with different costs; these methods and PBO have been described
as isomorphic but developed independently. The 2017 guidelines of Johnston and
colleagues [@johnston2017contemporary], which go further than those of the 1993
panel chaired by Kenneth Arrow [@noaa1993report], say that a single binary
choice is the most directly incentive-compatible format, that question order
should be randomized across respondents, and that as statistical efficiency
and choice complexity rise, the consistency of respondents' choices falls; a
large choice experiment found robust order effects in repeated-response stated
preference studies [@day2012ordering].

**What this means for PBO.** Whether the winner's-curse bias of Hauser and
Toubia appears in the utility magnitudes or lengthscales a Gaussian process
estimates has not been tested; by its mechanism, small effects would be
inflated relative to large ones, and mixing in random queries is an
off-the-shelf remedy, one that studies aiming at population conclusions need
anyway (inference). An acquisition function that picks the most informative
query also tends to raise choice complexity and lower answer consistency, so
the trade-off belongs in the acquisition design, for example as a penalty on
pairs that differ on many attributes at once (inference). PBO differs from this field in
its surrogate, a Gaussian process or other nonparametric model over a
continuous design space; an adaptive conjoint study with such a surrogate is
one of the gaps in @sec-econ-status.

## Multi-criteria decision analysis {#sec-econ-mcda}

**Multi-criteria decision analysis** (MCDA) is a family of methods from
operations research for helping a decision maker rank or choose options judged
on several conflicting criteria, such as cost, safety, and comfort.
**Compensatory** methods let a gain on one criterion offset a loss on another,
as a weighted sum does; **non-compensatory** methods do not. The outranking
methods ELECTRE [@roy1968classement] and PROMETHEE [@brans1985note] compare
options criterion by criterion with thresholds for indifference, preference,
and **veto** (a difference on one criterion so large that no advantage
elsewhere can make up for it); the analytic hierarchy process
[@saaty1977scaling] derives weights from pairwise ratio judgments; TOPSIS ranks
options by their distance to an ideal and an anti-ideal point
[@hwang1981multiple].

Closest to PBO, **robust ordinal regression** ranks options using every
additive value function (a sum of one value function per criterion) consistent
with the decision maker's pairwise statements [@greco2008ordinal]. It reports
two relations: $a$ is **necessarily** preferred to $b$ if every compatible
value function ranks $a$ at least as high as $b$, and **possibly** preferred if
at least one does. Corrente and colleagues compared the method with preference
learning in machine learning [@corrente2013robust]. Since 2017, Ciomek,
Kadziński, and Tervonen compared question-selection heuristics for pairwise
elicitation and validated the best of them in an experiment in which 101
participants answered pairwise questions to rank 10 mobile phone packages on
four criteria [@ciomek2017heuristics]; Grillo, Kotłowski, and Kadziński proved
regret bounds for Bayesian methods and for regularized maximum likelihood
(follow the regularized leader, which after each answer refits to all answers
so far with a penalty that keeps the estimate from jumping) under linear and
Bradley-Terry models [@grillo2025ordinal]; and Huber, Rojas Gonzalez, and
Astudillo built a Bayesian model of the decision maker's utility from pairwise
comparisons, chose queries by balancing exploration and exploitation, and
tested on problems with up to nine objectives [@huber2025bayesian].

**What this means for PBO.** The necessary and possible relations correspond to
preferences with posterior probability 1 and greater than 0; reporting both
would give users an interpretable summary of what their comparisons have
settled (inference), and the "P(not beaten)" row of @fig-econ-incomplete is a
summary of this kind. Veto thresholds express hard limits that a smooth
Gaussian process utility cannot, which supports treating some preferences as
constraints (inference). We found no study that compares the question-selection
heuristics of robust ordinal regression with preferential acquisition functions
on the same human task; that experiment could be run today.

## Interactive multi-objective optimization {#sec-econ-moo}

In multi-objective optimization (@sec-multi-objective) several objectives
conflict, and the answer is a **Pareto front**: designs that cannot be improved
on one objective without worsening another. **Interactive** methods let a human
decision maker steer the search toward the part of the front they like by
stating preferences during the optimization. A related line in artificial
intelligence chose queries by value of information [@chajewska2000making] and
by minimax regret (recommending the option whose worst-case loss, over all
utilities still consistent with the answers, is smallest). A journal paper in
that line proves that under both the Bayesian and the minimax-regret
frameworks, an optimal recommendation set of size $k$ is also an optimal choice
query of size $k$, so "there is no tradeoff to be made between good
recommendations and good queries"; the analysis covers noise-free responses and
constant and logistic noise, and the abstract calls the results robust to such
noise [@viappiani2020equivalence]. The abstract of the 2010 conference version
says that for noisy responses it provides worst-case guarantees
[@viappiani2010optimal], which points to a bounded loss rather than an exact
equivalence; we have not checked the theorem itself. qEUBO, which scores a set
of $q$ shown candidates by the expected maximum utility among them (@sec-eubo),
was later proposed and analyzed as a decision-theoretic acquisition function
[@astudillo2023qeubob]; the equivalence is the earlier foundation of this
family.

Interactive multi-objective optimization also designed interactions that
resist anchoring well before PBO. The NAUTILUS method starts from the worst
point and lets the decision maker improve every objective at each step without
trading any off, which avoids both anchoring and loss framing
[@miettinen2010nautilus]. Anchoring has also been measured: with 128
participants using interactive goal programming on a two-objective problem,
anchoring occurred, made decision makers miss their most preferred solution,
and was stronger when they used someone else's preferences
[@halstead2026multiobjective]. And the field has evaluation norms: a survey of
how interactive methods are assessed records the type of decision maker (a
utility or value function, an artificial decision maker, or a human) and the
type of preference information [@afsar2021assessing], and a reusable
questionnaire and experimental design support comparisons with human decision
makers [@afsar2022designing].

**What this means for PBO.** The standard interaction, incumbent against
challenger, trades one attribute against another; a NAUTILUS-style path that
only ever improves answers the concern about reference dependence without
estimating a loss aversion parameter, and varying the initial design between
sessions counteracts anchoring (inference). User studies of PBO should adopt
the evaluation norms of this field: report whether the decision maker is a
person or a simulated user, what preference information was collected, and how
stopping was decided (@sec-sw-simulated-users, @sec-rec-evaluation)
(inference).

## Replication in experimental economics {#sec-econ-replication}

Since several sections above lean on behavioral findings, it matters how well
those findings replicate. In general, moderately well. Of 18 laboratory
experiments published in two leading economics journals, 11 (61%) replicated
with a significant effect in the same direction, and the replicated effects
averaged 66% of the originals [@camerer2016evaluating]. Of 21 social-science
experiments published in *Nature* and *Science* between 2010 and 2015, 13 (62%)
replicated significantly in the same direction, and replication effect sizes
averaged about 50% of the originals [@camerer2018evaluating]. *Economic
Inquiry* ran a two-part symposium on reproducibility and replicability,
concerned mainly with non-experimental studies [@bokhari2025introduction;
@bokhari2025introductionb]. @tbl-econ-replication lists the status of the
specific findings that have been proposed as ingredients of preference models;
works already cited in this chapter appear by author and year.

::: {.table #tbl-econ-replication title="Behavioral findings proposed as ingredients of preference models, and their replication status as of September 2026."}
| Finding | Proposed use | Status | Basis |
|---|---|---|---|
| Loss aversion coefficient of about 2.25 | fixed asymmetry near the incumbent | contested; pooled estimates range from about 1.07 to 1.955 | Brown et al. 2024; Walasek et al. 2024; Yechiam and Zeif 2025 |
| Risk and time anomalies arise from complexity | noise that grows with difficulty | contested | Banki et al. 2025 and Oprea's reply |
| Endowment effect explained by loss aversion | asymmetric value of the incumbent | weakened | Chapman et al. 2023 (working paper) |
| Attraction (decoy) effect | candidates reshape utilities through decoys | limited to stylized numerical displays | Frederick, Lee, and Baskin 2014 |
| Default effect | initial design acts as a default | robust on average; heterogeneous, with null and negative effects | Jachimowicz et al. 2019 |
| Status quo bias | advantage of the incumbent | replicated in 3 of 4 decision scenarios | Xiao et al. 2021 |
:::

The table relies on four sources not yet introduced. On the complexity account
of risk and time anomalies, Oprea answered a 2025 working paper by Banki,
Simonsohn, Walatka, and Wu [@banki2025decisions] by arguing that their reversal
of his conclusion came mainly from using medians, and that with means the
deviations remain highly significant [@oprea2025initial]; the dispute is open.
Frederick, Lee, and Baskin found that the attraction effect may be confined to
stylized presentations in which every product dimension is a number; it
usually disappears when consumers experience the products, or when even one
attribute is presented perceptually [@frederick2014limits]. In two
preregistered replications (the analysis plan was filed before the data were
collected; n = 311 and n = 316), status quo bias had strong support in three of
four decision scenarios and no substantial support in the fourth
[@xiao2021revisiting].

**What this means for PBO.** The behavioral parameters proposed for preference
likelihoods rest on exactly the findings whose magnitudes are least stable; the
direction of these effects is more reliable than their size, so a likelihood
should not fix their magnitudes (inference). The limit on the attraction effect
is especially relevant: PBO is often used for perceptual designs such as color,
shape, and motion, which is where Frederick and colleagues found decoy effects
rarely appear, so the worry that adding candidates reshapes the utility
landscape through decoys is weakened, at least for this mechanism, although
divisive normalization (@sec-neuro-coding) may still apply to unfamiliar
options (inference).

## Already solved elsewhere {#sec-econ-solved}

The sections above found, one by one, that problems PBO papers treat as new
have established treatments in neighboring fields. @tbl-econ-solved gathers
them; its last column is this chapter's reading of what each treatment
implies, and every entry in it is an inference.

::: {.table #tbl-econ-solved title="Problems PBO treats as new that other fields have already addressed (implications are inferences)."}
| Problem | Earlier treatment elsewhere | Practice in PBO | Implication |
|---|---|---|---|
| Adaptive Bayesian questioning with response error | conjoint designs of Sándor and Wedel (2001, 2002), Yu, Goos, and Vandebroek (2011), Toubia et al. (2003 to 2007), Sauré and Vielma (2019) | a separate line, built on a Gaussian process posterior | the novelty is the nonparametric surrogate; use adaptive conjoint as a baseline |
| Bias from adaptive questions | winner's-curse bias of utility-balanced questions (Hauser and Toubia 2005) | not checked | test for distorted magnitudes and lengthscales; mix in random queries |
| Anchoring and loss framing | NAUTILUS improves every objective at each step (Miettinen et al. 2010); anchoring measured (Halstead et al. 2026) | incumbent against challenger; loss aversion in the likelihood proposed | use improvement-only paths; randomize the initial design |
| Recommendation sets and query sets | the optimal recommendation set is the optimal choice query (Viappiani and Boutilier 2010, 2020) | qEUBO (Astudillo et al. 2023) | EUBO-type acquisition has an earlier decision-theoretic foundation |
| Uncertainty and question selection with people | necessary and possible relations (Greco, Mousseau, and Słowiński 2008); heuristics tested with 101 people (Ciomek, Kadziński, and Tervonen 2017) | the posterior-mean maximizer; tests on simulated users | report relations with posterior probability 1 or above 0; compare heuristics on the same human task |
| Order, incentives, complexity | randomize order; single binary choices; consistency falls with complexity (Johnston et al. 2017; Day et al. 2012) | seldom addressed | penalize pairs that differ on many attributes; randomize order |
| Evaluation with people | norms and designs for human decision makers (Afsar, Miettinen, and Ruiz 2021; Afsar et al. 2022) | mostly simulated users; stopping rules unreported | report the decision maker, the preference information, and the stopping rule |
| One utility from many people | Bradley-Terry likelihood violates Pareto optimality (Ge et al. 2024) | all comparisons pooled in one Gaussian process | fit per user, then aggregate with an explicit rule |
:::

## Settled, contested, missing {#sec-econ-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** The qualitative patterns of prospect theory replicate, with
smaller effects than originally reported [@ruggeri2020replicating]. The
direction of loss aversion is reliable, but its size depends on the design of
the task, from about 1.07 for symmetric, unordered presentations to about 2.33
for ordered ones [@yechiam2025loss]. Random choice can be separated
experimentally into evaluation noise, random preference, and deliberate
randomization [@agranov2017stochastic; @agranov2025ranges]. Forced choice
increases inconsistency [@costagomes2022choice], and when allowed, about 40% to
50% of people express incomplete preferences, according to a working paper
[@nielsen2026revealed]. Fitting one Bradley-Terry utility to pooled comparisons
by maximum likelihood is an aggregation rule that violates Pareto optimality
and pairwise majority consistency [@ge2024axioms] and approximates the Borda
count under hidden heterogeneity [@siththaranjan2024distributional]. Optimal
recommendation sets are optimal choice queries [@viappiani2020equivalence].
Adaptive Bayesian questioning under response error predates PBO by more than a
decade in marketing science. Rational inattention with a Shannon cost yields
the logit model, and with Bregman costs it can yield any additive random
utility model [@fosgerau2020discrete].

**Contested.** Whether complexity explains risk and time anomalies (Oprea
against Banki and colleagues). The share of
selfish types. Whether expectations set the reference point. Whether the
Shannon cost describes human attention [@dean2023experimental]. Whether the
endowment effect stems from loss aversion. Whether high revealed-preference
consistency indicates stable underlying preferences, given doubts about test
power and domain generality.

**Missing.** A test of whether "worse than the incumbent" is coded as a loss
in a design space. A definition of welfare for a person whose preferences are
changed by the elicitation itself. An adaptive conjoint study with a Gaussian
process or other nonparametric surrogate over a continuous design space, and a
head-to-head comparison of MCDA question-selection heuristics with preferential
acquisition functions on the same human task. A test of whether the
winner's-curse bias of adaptive questions distorts Gaussian process utility
estimates. A PBO system that implements reverse Bayesianism or models a "can't
compare" answer (compare @sec-obs-status). A test, in design comparisons, of
whether fast answers are the accurate ones or the ones between distant options.
:::

## Further reading {#further-reading .unnumbered}

- @strzalecki2025stochastic is the reference text on stochastic choice: random
  utility, its alternatives, and what choice frequencies can identify.
- @agranov2022revealed give a short overview of the experiments showing that
  people sometimes randomize on purpose.
- @nielsen2026revealed (working paper) and @cettolin2019revealed are the two
  most direct measurements of incomplete preferences; read them before adding
  a "can't compare" button.
- @johnston2017contemporary distill decades of stated preference practice into
  guidance that applies almost unchanged to preferential query design.
- @hauser2005impact is the clearest account of how adaptive questions can bias
  what they estimate.
- @viappiani2020equivalence prove the equivalence of recommendation sets and
  query sets that underlies EUBO-type acquisition functions.
- @ge2024axioms show, axiom by axiom, what pooling everyone's comparisons into
  one Bradley-Terry model does.
- @miettinen2010nautilus and @afsar2021assessing describe an interaction design
  that avoids anchoring and the norms for evaluating such designs with people.
