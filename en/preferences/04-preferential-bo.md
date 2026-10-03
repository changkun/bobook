---
status: done
synopsis: "Finding the best option from duels alone: the dueling formulation, how to pick the next pair, the decision-theoretic acquisition EUBO and qEUBO, its form for queries of several options, a complete loop with a person or a simulated one, and the failure modes reported in 2026."
sources: ["Gonzalez et al. 2017", "Lin et al. 2022", "Astudillo et al. 2023"]
---

# Preferential Bayesian Optimization {#sec-pbo}

Start with the figure below. It shows two colors and asks which you prefer.
Answer a dozen times, honestly, and watch the curve underneath: it is the
model's estimate of how much you like each hue, with a band for how unsure it
is, and the star is its current guess at your favorite. You never typed a
number. Every answer was a choice between two options, and the system decided
which two to show you next.

```{figure}
//| figure: pbo-oracle
//| label: fig-pbo-oracle
//| fig-cap: "Preferential Bayesian optimization with you as the oracle. Each answer is a comparison between two hues; the curve is the posterior over a latent utility, learned with the model of @sec-gp-preference, and the next pair is the one with the highest expected utility of the better option (@sec-eubo). The first pair is fixed; every later pair is chosen by the model. Switch to *Random* to compare with pairs chosen at random."
```

That loop is preferential Bayesian optimization (PBO). @sec-gp-preference built
its first half: a Gaussian process utility that learns from comparisons. This
chapter builds the second half, the rule for choosing the next comparison, and
then looks at what happens when the loop runs. The ideas carry over from
@sec-bo-loop with one change that turns out to matter a great deal: a query is
now a *pair* of inputs, and the answer is one bit.

## The problem {#sec-pbo-problem}

There is a latent utility $g$ on a domain $\X$: how much a person likes each
option. We want an input with high utility,

$$
\vx^\star \in \argmax_{\vx \in \X} g(\vx),
$$

but $g$ cannot be evaluated. What can be done is to show the person two
options, $\vx$ and $\vx'$, and record which one they choose. Following
@sec-comparisons, the answer is random, with a probability that grows with the
utility difference:

$$
\Prob(\vx \succ \vx') = \Phi\!\left(\frac{g(\vx) - g(\vx')}{\sqrt{2}\,\sigma}\right),
$$ {#eq-pbo-probit}

where $\vx \succ \vx'$ reads "$\vx$ is preferred to $\vx'$", $\Phi$ is the
standard normal CDF, and $\sigma$ is the noise in the person's evaluation of
each option. The logistic link $1/(1 + e^{-(g(\vx) - g(\vx'))/\tau})$ of the
Bradley-Terry model is used just as often; the choice between them matters
for theory (@sec-dueling-bandits) more than for the loop in this chapter.

The budget is small. A person can answer perhaps a few dozen comparisons in a
session before fatigue sets in, and @sec-hci shows that many real sessions end
much sooner. At the end the system must recommend one option, usually the one
with the highest posterior mean utility.

Three things make this harder than ordinary Bayesian optimization. Each
answer carries at most one bit, far less than a measured value. The answer
says nothing about the absolute level of $g$, only about differences, so
utilities are identified up to a shift (@sec-pref-identifiability). And the
query has twice as many inputs, so choosing it means searching over pairs.

## The dueling formulation {#sec-dueling-formulation}

The name *preferential Bayesian optimization* comes from
@gonzalez2017preferentialb, who posed the problem in a way that avoids the
latent utility altogether. They modeled the preference function
$\pi(\vx, \vx') = \Prob(\vx \succ \vx')$ directly, as a Gaussian process
classifier on the product space $\X \times \X$ of pairs, which they called the
*dueling space*, with a logistic link.

Without a utility, "the best option" needs a definition that uses only
pairwise probabilities. They used the **Condorcet winner**: an option that
beats every other option with probability above one half. It may not exist
when preferences are not transitive, so they scored each option by its
*soft-Copeland* value, the average probability that it wins against a uniformly
random opponent,

$$
C(\vx) = \frac{1}{\operatorname{Vol}(\X)} \int_{\X} \pi(\vx, \vx')\, \dd\vx',
$$

and sought its maximizer. If preferences do come from a utility as in
@eq-pbo-probit, the soft-Copeland maximizer is the utility maximizer, because
$\pi(\vx, \vx')$ increases with $g(\vx)$ for every opponent.

They proposed three acquisition functions. *Pure exploration* picks the duel
whose outcome is most uncertain. *Copeland expected improvement* looks one
step ahead at the soft-Copeland value. *Dueling Thompson sampling* draws one
sample of the preference function, takes the option with the best soft-Copeland
score under that sample as the first member of the duel, and pairs it with the
option whose duel against it is most uncertain. On one- and two-dimensional
test functions, discretized to 33 points per dimension, with 5 initial and 200
further duels over 20 repetitions, dueling Thompson sampling was consistently
the best strategy; Copeland expected improvement over-exploited and was
expensive enough that they ran it only on one function, and the dueling-bandit
baseline needed about 4000 iterations to approach what Thompson sampling
reached in 200 [@gonzalez2017preferentialb].

The dueling space doubles the input dimension and puts an integral inside
every evaluation of the objective, which is part of why most later work went
back to the latent-utility model of @chu2005preference that @sec-gp-preference
develops, and why the default implementation in BoTorch is built on it
[@balandat2020botorch]. The rest of this chapter uses that model. The two views
agree when preferences come from a utility; @sec-dueling-bandits returns to
what the dueling view buys when they do not.

## Choosing a pair {#sec-choosing-pairs}

With a posterior over the utility in hand, the question is which pair to show.
A good pair usually has two jobs. One option should be a strong candidate, so
that the answer refines what we know near the top. The other should be a
challenger whose comparison with the first is genuinely uncertain, so that the
answer teaches something. A pair of two obviously bad options, or of a strong
option against an obviously worse one, wastes a question.

The first rules followed this recipe literally. @brochu2007active took the
best option shown so far, by posterior mean, as the first member, and as the
second the option with the highest expected improvement over it, the
acquisition function of @sec-ei applied to the latent utility. The *maximally
uncertain challenge* of @fauvel2021efficient keeps the same champion and picks
the challenger whose duel outcome has the largest epistemic variance, the part
of the outcome's uncertainty that more data would remove. The *hallucination
believer* of @takeno2023practicalc draws one sample of the latent comparison
values from the posterior, treats it as data, and then applies any standard
acquisition function to the resulting Gaussian process.

These rules work, but four independent groups reported the same weakness of
the expected-improvement family: it stalls. Expected improvement of a challenger
over a well-known champion is small, so the rule stops testing the champion,
learns only how the challengers compare with one another, and never learns
whether any of them beats the incumbent [@gonzalez2017preferentialb;
@fauvel2021efficient; @takeno2023practicalc; @astudillo2023qeubob]. That is an
inference from the four reports rather than a result any one of them proves
(inference); @astudillo2023qeubob do prove the stall for the batch version,
below.

## Expected utility of the best option {#sec-eubo}

A cleaner rule comes from asking what the comparison is *for*. Suppose the
session ended right after this query and we recommended whichever of the two
options the person picked. If their answer is reliable, they pick the one with
the higher utility, and the value of the query is the utility of the better of
the two. We do not know $g$, so we take its expectation under the posterior:

$$
\EUBO(\vx_1, \vx_2) = \E_n\!\left[\max\{g(\vx_1),\, g(\vx_2)\}\right],
$$ {#eq-eubo}

the **expected utility of the best option**, where $\E_n$ is the expectation
under the posterior after $n$ comparisons. EUBO was introduced for preference
exploration in multi-objective problems [@lin2022preferenceb] and generalized
to queries of $q$ options, $\E_n[\max_i g(\vx_i)]$, under the name qEUBO
[@astudillo2023qeubob].

Under the Laplace approximation of @sec-pref-laplace, the posterior values
$A = g(\vx_1)$ and $B = g(\vx_2)$ are jointly Gaussian, so @eq-eubo has a
closed form.

::: {.derivation title="EUBO in closed form"}
Let $A$ and $B$ be jointly Gaussian with means $\mu_A, \mu_B$, variances
$v_A, v_B$, and covariance $c$.

1. Write the maximum as one variable plus a positive part:
   $\max\{A, B\} = B + \max\{A - B,\, 0\}$.
2. The difference $\Delta = A - B$ is Gaussian (a linear map of a Gaussian,
   @sec-gaussian-linear), with mean $\delta = \mu_A - \mu_B$ and variance
   $s^2 = v_A + v_B - 2c$.
3. $\E[\max\{\Delta, 0\}]$ is the expected improvement of $\Delta$ over zero, which
   @sec-ei computes as $\delta\,\Phi(\delta/s) + s\,\phi(\delta/s)$, with
   $\phi$ the standard normal density.
4. By linearity of expectation,
   $\E[\max\{A, B\}] = \mu_B + \delta\,\Phi(\delta/s) + s\,\phi(\delta/s)$.
5. Using $\mu_B = \mu_B\Phi(\delta/s) + \mu_B\Phi(-\delta/s)$, this is
   $\mu_A\,\Phi(\delta/s) + \mu_B\,\Phi(-\delta/s) + s\,\phi(\delta/s)$, the
   formula of @clark1961greatest.
:::

$$
\EUBO(\vx_1, \vx_2) = \mu_A\,\Phi\!\left(\frac{\delta}{s}\right) + \mu_B\,\Phi\!\left(-\frac{\delta}{s}\right) + s\,\phi\!\left(\frac{\delta}{s}\right).
$$ {#eq-eubo-closed}

The formula shows how one expression does both jobs from @sec-choosing-pairs.
The first two terms are a weighted average of the two means, large when either
option is good. The last term grows with $s$, the uncertainty about which option
is better. Exploitation and exploration appear in one formula with no tuning
constant. Three properties are worth checking against the figure below:

- A pair of identical options is worth one option: $s = 0$ and
  $\EUBO(\vx, \vx) = \mu(\vx)$.
- A pair is never worth less than its better mean:
  $\EUBO \ge \max\{\mu_A, \mu_B\}$, by Jensen's inequality.
- At fixed means, EUBO increases with $s$ (@exr-eubo-s).

```{figure}
//| figure: eubo-map
//| label: fig-eubo-map
//| fig-cap: "EUBO over all pairs after five duels on the running objective. Left: the latent utility posterior (blue), the objective it is learning (dashed, rescaled, since utilities are identified only up to shift and scale), and the five duels as segments from loser to winner at the top. Right: EUBO for every pair on a 31 by 31 grid; darker is higher. The matrix is symmetric, its diagonal is the posterior mean, and its maximum (orange) is the next pair, also marked on the left. Press *Ask the next pair* to answer it as the objective would and see the map change."
```

The heatmap makes the trade-off visible. After five duels the model cannot
tell whether the wide bump near $x = 0.25$ or the region near $x = 0.7$ is
better, and EUBO asks exactly that question: its maximum pairs a point from
each. Pressing *Ask the next pair* a few times shows the map sharpening as the
answers come in.

### What is known about EUBO {#sec-eubo-theory}

EUBO is more than a plausible heuristic. A query's *one-step Bayes optimal*
value is the best expected utility of the final recommendation achievable
after one more answer; an acquisition function that maximizes it is the
*knowledge gradient* of @sec-knowledge-gradient. @lin2022preferenceb proved that
EUBO is one-step Bayes optimal for preference exploration. @astudillo2023qeubob
extended the analysis to $q$ options:

- With noise-free answers, a maximizer of qEUBO is one-step Bayes optimal, so
  qEUBO coincides with the knowledge gradient.
- With logistic noise of scale $\tau$ (@eq-cmp-logit), the one-step value of
  qEUBO's choice is at most $\tau\, W\!\left((q-1)/e\right)$ below the optimum,
  where $W$ is the Lambert W function, the inverse of $w \mapsto we^w$.
- On a finite domain with $q = 2$ and further technical conditions, the
  Bayesian simple regret of qEUBO decays faster than $1/n$.
- Under the same assumptions, a batch version of expected improvement, qEI,
  can have simple regret bounded away from zero for every $n$: it is not
  asymptotically consistent. This is the stall from @sec-choosing-pairs, now
  proved.

The third result assumes a finite set of options, which makes it an
identification problem; it does not compare directly with the rates for
continuous domains in @sec-dueling-bandits (inference). In BoTorch, the
analytic EUBO and qEUBO are available with the `PairwiseGP` model; qEUBO arrived
in version 0.10.0 in February 2024 [@botorch2026changelog].

## A complete loop {#sec-pbo-loop}

Putting the pieces together gives the algorithm that runs inside
@fig-pbo-oracle.

::: {.algorithm #alg-pbo title="PBO with EUBO"}
Input: domain $\X$, kernel $k$, noise scale $\sigma$, budget $N$ comparisons.

1. Ask a few comparisons between random or space-filling pairs.
2. Fit the preference model of @sec-gp-preference to all answers so far: find
   the posterior mode of the utility at the compared inputs, and the Laplace
   approximation around it.
3. Choose the next pair $(\vx_1, \vx_2)$ by maximizing @eq-eubo-closed over
   pairs: on a grid of candidates for small problems, by gradient ascent from
   several starting pairs otherwise.
4. Show the pair, record the answer, and return to step 2 until the budget is
   spent.
5. Recommend the input with the highest posterior mean utility.
:::

The figure below runs the same loop against a simulated person with a hidden
favorite hue, so you can check the model against a known answer.

```{figure}
//| figure: pbo-oracle
//| label: fig-pbo-simulated
//| fig-cap: "The same loop with a simulated person who answers according to @eq-pbo-probit with a hidden utility. Press *Simulate ten* a few times, then reveal the favorite. Raise the noise to see how a less consistent person slows the model down, and switch to random pairs to see what the acquisition function buys. The model always assumes noise σ = 0.15, whatever the simulated person's true noise is."
mode: sim
```

Some things to try. With the default noise, EUBO usually puts the star within
a few degrees of the hidden favorite after ten to fifteen answers. Random pairs
get there more slowly and less reliably, because many random pairs compare two
mediocre hues. Raise the noise to 0.5 and the curve flattens: the model reads
inconsistent answers as small utility differences, exactly as @eq-pbo-probit
says it should. For some seeds, EUBO stops a little short of the favorite and
keeps asking about nearly the same pair; that is the first failure mode below.

### More than one parameter {#sec-pbo-dims}

A hue is one number. Real designs have many: a typeface has weight, width,
contrast, and slant; an exoskeleton controller has timing and torque for each
phase of the stride. Nothing in @alg-pbo depends on the dimension, but the
amount the model must learn does. The figure below runs the same loop over
generated designs with 3, 6, or 10 parameters: background hue, corner
roundness, shape size, then saturation, stripes, rotation, and so on.

```{figure}
//| figure: dim-oracle-nd
//| label: fig-pbo-dims
//| fig-cap: "PBO over designs with 3, 6, or 10 parameters. Choose between designs A and B; the small multiples show what the model has learned about each parameter, as the posterior mean utility along that parameter with the others fixed at the current best guess (orange line). The lower panel is recorded, not live: for 24 simulated people with hidden favorites and answer noise 0.1, the median remaining gap between the model's best guess and the favorite after each comparison (1 is no better than a random design, 0 is the favorite), with pairs chosen by EUBO (solid, interquartile range shaded) or at random (dashed). In simulated mode your session's own curve is drawn over it. The model is the one in this chapter with a lengthscale that grows with the number of parameters, and EUBO searches a few hundred candidate designs per query; both are simplifications."
```

Two things stand out in the recorded curves. The first is the cost of
dimension. When EUBO chooses the pairs, forty comparisons close about two
thirds of the gap to the favorite with three parameters, about 40 percent with
six, and about 30 percent with ten. Each comparison still carries at most one
bit, while the space it must locate the favorite in grows with every
parameter. Learning a person's taste over ten parameters from a few dozen choices is a different problem from
learning it over one, and @sec-high-dimensions follows that problem into the
research literature.

The second is that, in this simple implementation, choosing pairs by EUBO is
not better than choosing them at random once the first twenty or so answers are
in. With three and six parameters the EUBO curves flatten after about twenty
answers while the random curves keep falling: after forty answers, random
pairs have closed about 87 percent of the gap with three parameters and about
half of it with six. With ten parameters the two medians are within about
0.05 of each other from twenty answers on, and both rules close about 30
percent. The mechanism is visible if you watch the
pairs: EUBO keeps proposing two designs near the current best guess, so the
answers refine a small region and stop testing the rest. That is the collapse
reported for EUBO in the failure modes below, here amplified by searching a
small candidate set (inference). The simulated photo sessions of
@sec-cs-photo-race show the opposite ordering, with EUBO pairs well ahead of
random ones, under a different setup: six adjustments of a real photograph,
only pairs at least 0.2 apart, and a recommendation restricted to compared
settings. Both simulations search a candidate pool that includes perturbations
of the best guesses, so the pool is not the difference; we have not separated
which of the others accounts for the reversal (inference). It is a reminder that an acquisition function's guarantees
concern one step under the model's assumptions, and that whether it beats
random pairs over a whole session is an empirical question, one that has rarely been asked with people
(@sec-acqf-empirical).

## Known failure modes {#sec-pbo-failure-modes}

The loop above is close to what practitioners run: a Gaussian process
preference model with the probit link and the Laplace approximation, and EUBO
or qEUBO to choose queries. Between 2024 and 2026 several groups examined it
closely and found problems. Most of these reports are preprints, so they should
be read as findings to be confirmed rather than settled results.

**EUBO collapses toward the current best.** @wu2026knowledge derived the exact
knowledge gradient under the probit likelihood in closed form, showed that EUBO
is a lower bound on it, and on a two-dimensional test function showed EUBO's
queries gathering around the estimated maximum, while the exact knowledge
gradient kept exploring. Under noise, the equivalence of EUBO and the knowledge
gradient no longer holds, and the gap is where the collapse comes from.

**The comparison graph falls apart.** Think of every compared input as a node
and every answered pair as an edge. @shao2026adaptive observe that EUBO tends to
choose new pairs that share no input with earlier queries, so each pair is an
isolated edge, and the likelihood Hessian in the Laplace approximation becomes
rank deficient. @sec-comparison-graph explains why connectivity matters for any
comparison model, and reports the correction the authors propose and its size.

**Good final answers can hide a costly path.** On samples from a Gaussian
process, @xu2024principledb reported that qEUBO's recommended solution was
slightly better than their optimistic algorithm's, but its cumulative regret,
which counts the utility of everything shown along the way, was more than 2.5
times higher. Which number matters depends on whether the person has to live
with the options they are shown.

**Older rules have their own failures.** Thompson sampling over-explores as
the dimension grows, and the hallucination believer, which does best at very
low noise, can get stuck when answers are noisy [@takeno2023practicalc;
@xu2024principledb].

@sec-acquisition-frontier collects these results with the conditions under
which each was observed, and @sec-observation-models looks at the inference
side of the same pipeline.

::: {.frontier title="Settled, contested, missing"}
**Settled.** EUBO and qEUBO are one-step Bayes optimal with noise-free answers,
and near-optimal under logistic noise. Expected-improvement-type rules can stall,
and qEI is provably not consistent.

**Contested.** Whether EUBO's collapse and the rank-deficient Hessian cost
anything on human tasks: the evidence is from simulations and preprints.

**Missing.** No study has randomized people to different acquisition functions
with the same interface and budget; @sec-open-problems lists this experiment.
:::

## When the oracle is a person {#sec-pbo-people}

The simulated person in @fig-pbo-simulated has a fixed utility, constant noise,
and unlimited patience. You do not, and neither does anyone in the studies of
@sec-hci and @sec-health. Go back to @fig-pbo-oracle and answer another twenty
questions. Did you ever choose against your own earlier answers? Did the colors
you liked change as you saw more of them? Did you start to like the hue the
star was pointing at partly because the system kept showing it to you?

These are not hypothetical worries. In a three-month field deployment of a
human-in-the-loop optimizer, 415 of 549 evaluation sequences stopped at the first
iteration [@ou2022human]. When an optimizer rather than the person leads the
search, people tend to reach better designs while reporting less agency over
them [@chan2022investigating; @niwa2025cooperative]. Whether repeated
comparisons *find* a preference or partly *make* one is the question
@sec-part-perspectives
takes up, and the experiment that would separate the two, randomizing the
order of queries and retesting a week later, is described in
@sec-open-decisive. The algorithm of this chapter is the right starting point;
the people it runs on are the reason the book continues past it.

## Exercises {#sec-pbo-exercises}

::: {.exercise #exr-eubo-s}
Show that at fixed $\mu_A$ and $\mu_B$, the derivative of @eq-eubo-closed with
respect to $s$ is $\phi(\delta/s)$. Why does this mean that EUBO never prefers a
pair whose comparison is less uncertain, when the means are the same?

::: {.solution}
Write $z = \delta/s$, so $\partial z/\partial s = -\delta/s^2$. Differentiate
term by term, using $\Phi' = \phi$ and $\phi'(z) = -z\phi(z)$:

$$
\frac{\partial}{\partial s}\Big[\mu_A\Phi(z) + \mu_B\Phi(-z) + s\phi(z)\Big]
= (\mu_A - \mu_B)\,\phi(z)\,\frac{\partial z}{\partial s} + \phi(z) + s\,(-z\phi(z))\,\frac{\partial z}{\partial s}.
$$

The first term is $\delta\phi(z)(-\delta/s^2) = -z^2\phi(z)$ and the last is
$s(-z\phi(z))(-\delta/s^2) = z^2\phi(z)$. They cancel, leaving $\phi(z) > 0$.
EUBO strictly increases with the uncertainty of the comparison, so between two
pairs with the same means it always prefers the one whose outcome is less
predictable.
:::
:::

::: {.exercise #exr-eubo-identical}
Why is it not a problem for EUBO that $\EUBO(\vx, \vx) = \mu(\vx)$ can be large
when $\vx$ is a good option? Under what posterior would the maximizer of EUBO
be a pair of identical options, and what would that say about the search?

::: {.solution}
Because $\EUBO \ge \max\{\mu_A, \mu_B\}$, any pair $(\vx, \vx')$ with
$\vx' \neq \vx$ is worth at least $\mu(\vx)$, and strictly more as soon as the
comparison has positive uncertainty (@exr-eubo-s). The diagonal can only win
when every comparison involving the best option is already certain, that is,
when the model is sure no other option beats it. At that point asking more
questions has no expected value, which is a natural stopping signal, though
@sec-rec-stopping explains why a converged posterior is not by itself proof
that the person's preference has stabilized.
:::
:::

::: {.exercise #exr-pbo-noise}
In @fig-pbo-simulated, the model assumes $\sigma = 0.15$ while the simulated
person may be much noisier. Predict how the posterior changes if the person's
true noise is 0.5, then check. What does this suggest about fixing the noise
scale instead of fitting it?

::: {.solution}
The model reads every answer as if it came from a person with noise 0.15. A
person with noise 0.5 contradicts themselves far more often than such a model
expects, and the only way the model can explain answers that go both ways on
similar pairs is a small utility difference between them. So the posterior
mean flattens where the answers conflict, the star moves more from one answer
to the next, and it ends farther from the hidden favorite. The band does not
widen to match, because the model's noise is fixed, so the model is more
confident than the answers justify (inference). Fixing the noise scale is safe
only when it is roughly right. When it may not be, fit it (or the kernel
amplitude, which is the same degree of freedom by @sec-pref-identifiability),
or check it with a few repeated pairs, as @sec-cs-photo-consistency does.
:::
:::

## Further reading {#further-reading .unnumbered}

- @gonzalez2017preferentialb define the problem, the dueling space, and the
  soft-Copeland score; @brochu2007active is the earlier gallery-based
  formulation for material design.
- @chu2005preference is the preference model this chapter builds on.
- @lin2022preferenceb introduce EUBO and prove its one-step optimality;
  @astudillo2023qeubob give qEUBO, its noisy-answer bound, and the inconsistency
  of qEI.
- @takeno2023practicalc study the inference behind the loop and propose the
  hallucination believer; @wu2026knowledge and @shao2026adaptive are the 2026
  preprints on EUBO's failure modes.
- BoTorch's preference tutorial runs this chapter's loop with `PairwiseGP` and
  EUBO [@botorch2026pairwise].
