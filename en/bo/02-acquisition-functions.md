---
status: done
synopsis: "Probability of improvement, expected improvement, upper confidence bounds, Thompson sampling, the knowledge gradient, and entropy search: each a different answer to 'what is the next evaluation worth?', derived and compared on the same posterior in one and two dimensions, followed by how the acquisition function itself is maximized when the domain has many dimensions."
sources: ["Kushner 1964","Jones et al. 1998","Srinivas et al. 2010","Frazier et al. 2009","Hennig and Schuler 2012","Wang and Jegelka 2017","Garnett 2023, ch. 7, 8, 9","Frazier 2018"]
---

# Acquisition Functions {#sec-acquisition}

@sec-bo-loop built the Bayesian optimization loop and ran it with one rule for
choosing the next evaluation: the posterior mean plus a multiple of the
posterior standard deviation. The rule worked, but its weight was a knob, and
@fig-loop-tradeoff showed that turning the knob too far either way costs a
great deal. A reader may reasonably ask whether there is a principled way to
decide how much uncertainty is worth.

This chapter gives several answers. Each acquisition function starts from a
statement about what an evaluation is *for*: beating the best value seen so
far, improving the final recommendation, or learning where the maximum is.
From that statement and the Gaussian posterior, the formula follows, and the
balance between exploring and exploiting comes out of the derivation instead
of being set by hand. We derive the classic acquisition functions one at a
time, look at all of them on the same posterior in one and then two
dimensions, and end with the problem every one of them leaves behind: finding
the maximum of the acquisition function itself, which in many dimensions is a
hard optimization problem of its own.

## What an acquisition function is {#sec-acq-definition}

Recall the setting. After $n$ evaluations the data are
$\D_n = \{(\vx_i, y_i)\}_{i=1}^n$, and the Gaussian process posterior gives
every input $\vx$ a Gaussian belief about $f(\vx)$, with mean $\mu_n(\vx)$ and
standard deviation $\sigma_n(\vx)$ (@sec-gp-regression). An acquisition
function $a_n(\vx)$ scores each input, and the loop evaluates the
objective where the score is highest (@def-loop-acquisition).

The cleanest way to build such a score is to say what we would be happy to
have at the end and then ask how much one more evaluation is expected to add.
Write $u(\D)$ for the *utility* of a data set: a number that says how good
our situation is if we stop with data $\D$. If we evaluate at $\vx$ and observe
$y$, the utility changes from $u(\D_n)$ to $u(\D_n \cup \{(\vx, y)\})$. We do
not know $y$ before evaluating, but the posterior says what it is likely to
be, so we can average over it.

::: {.definition #def-acq-one-step title="One-step lookahead acquisition function"}
Given a utility $u$, the *one-step lookahead* acquisition function is the
expected gain in utility from one more evaluation at $\vx$:

$$
a_n(\vx) = \E_n\!\left[\,u(\D_n \cup \{(\vx, y)\}) - u(\D_n)\,\right],
$$ {#eq-acq-one-step}

where $\E_n$ averages over the outcome $y$ under its posterior predictive
distribution given $\D_n$.
:::

Different utilities give different acquisition functions.
@tbl-acq-family previews the ones in this chapter. Two of them, the upper
confidence bound and Thompson sampling, do not come from a utility at all;
they come from the bandit problems of @sec-bandits, where they have
guarantees of a different kind (@sec-regret).

::: {.table #tbl-acq-family title="Acquisition functions and the question each one answers."}
| Acquisition function | What an evaluation is worth | Section |
|---|---|---|
| Probability of improvement (PI) | the chance of beating the best value seen so far | @sec-pi |
| Expected improvement (EI) | the expected amount by which the best value seen so far rises | @sec-ei |
| Upper confidence bound (UCB) | an optimistic estimate of the value at the input | @sec-ucb |
| Thompson sampling (TS) | the chance that the input is the maximizer | @sec-thompson |
| Knowledge gradient (KG) | the expected rise in the value of the final recommendation | @sec-knowledge-gradient |
| Entropy search (ES, PES, MES) | the expected information about the maximizer or the maximum | @sec-entropy-search |
:::

### Why look only one step ahead {#sec-acq-myopia}

@eq-acq-one-step looks one evaluation ahead, as if the next evaluation were
the last. That is a simplification. With $N - n$ evaluations left, the
optimal choice now depends on what we will be able to do afterward, and the
next evaluation is worth more if it sets up good later ones. Writing that out
gives a dynamic program in which every future outcome branches into every
future choice. It is the optimal policy, and it is also, in Kushner's words,
"virtually impossible to compute" [@kushner1964new; quoted in
@garnett2023bayesian, sec. 12.3].

So almost all acquisition functions in use are *myopic*: they are optimal if
the next evaluation is the last one, and only approximately optimal
otherwise [@frazier2018tutorial]. The approximation may be better than it
sounds. In a few special problems where the optimal multi-step policy can be
computed, the myopic rules come close; in one such study, cited by
@frazier2018tutorial, the knowledge gradient came within 98% of optimal.
Myopia also has a cost that is easy to see, though: a rule that ignores the
future undervalues exploration, because the payoff of exploring arrives
later. Several of the rules below add exploration back in one way or another.

## Probability of improvement {#sec-pi}

The oldest rule asks the simplest question: how likely is it that $\vx$ beats
the best value seen so far [@kushner1964new]? Write
$f^*_n = \max_{i \le n} y_i$ for that value, the *incumbent*. With exact
evaluations, it is the value of the best input we could recommend right now.

The posterior says that $f(\vx)$ is Gaussian with mean $\mu_n(\vx)$ and
standard deviation $\sigma_n(\vx)$. The probability that it exceeds the
incumbent by at least a margin $\xi \ge 0$ follows in three steps.

::: {.derivation title="Probability of improvement"}
1. Standardize: $Z = (f(\vx) - \mu_n(\vx)) / \sigma_n(\vx)$ is a standard
   normal variable (@sec-gaussian-linear).
2. Rewrite the event: $f(\vx) > f^*_n + \xi$ is the same as
   $Z > (f^*_n + \xi - \mu_n(\vx)) / \sigma_n(\vx)$.
3. Use $\Prob(Z > a) = 1 - \Phi(a) = \Phi(-a)$, by the symmetry of the
   standard normal density.
:::

$$
\PI_n(\vx) = \Phi\!\left(\frac{\mu_n(\vx) - f^*_n - \xi}{\sigma_n(\vx)}\right).
$$ {#eq-acq-pi}

In terms of @def-acq-one-step, PI is the expected gain of a utility that is 1
if the best value improves by at least $\xi$ and 0 otherwise.

PI has a well-known flaw: it counts improvements but ignores their size. With
$\xi = 0$, an input whose mean sits just above the incumbent and whose
standard deviation is tiny has PI close to 1, and the rule prefers it to an
uncertain input that might improve on the incumbent by a lot. The result is a
search that creeps along in tiny steps near the best point. The margin $\xi$
is the fix. Raising it asks for improvements that are worth having, which
pushes the mean term in @eq-acq-pi below zero for most inputs; then the only
way to have a sizable probability is a large $\sigma_n(\vx)$, and the rule
explores.

How large should $\xi$ be? Kushner suggested starting high and lowering it as
the search proceeds [@brochu2010tutorial], and his papers discussed adjusting
it by hand during the search [@garnett2023bayesian, sec. 12.3]. Jones found
the method "extremely sensitive to the choice of the target": too small and
the search stays local, too large and it never refines a promising solution
[@jones2001taxonomy; quoted in @brochu2010tutorial]. The next section's rule
makes the margin far less important, because it accounts for the size of the
improvement directly.

## Expected improvement {#sec-ei}

Instead of asking *whether* the incumbent improves, ask *by how much*, on
average. Evaluating at $\vx$ raises the best value seen from $f^*_n$ to
$\max(f^*_n, f(\vx))$. The gain is the *improvement*
$\max(f(\vx) - f^*_n, 0)$, which is zero when $f(\vx)$ falls short, and its
expectation under the posterior is the expected improvement:

$$
\EI_n(\vx) = \E_n\!\left[\max\!\left(f(\vx) - f^*_n,\; 0\right)\right].
$$ {#eq-acq-ei-def}

This is @eq-acq-one-step with the utility $u(\D) = \max_i y_i$, the best value
observed. EI is usually credited to Močkus and his colleagues
[@mockus1975bayesian; @frazier2018tutorial], with an earlier explicit formula
traced to Šaltenis in 1971 [@garnett2023bayesian, sec. 12.3], and it became
standard through the EGO algorithm of @jones1998efficient.

The difference $f(\vx) - f^*_n$ is a Gaussian variable, since $f^*_n$ is a known
number. So everything reduces to one fact about Gaussians, which we state for
any Gaussian $D$ because @sec-eubo uses it for a different one.

::: {.derivation title="The expected positive part of a Gaussian"}
Let $D$ be Gaussian with mean $\delta$ and standard deviation $s > 0$. We want
$\E[\max(D, 0)]$.

1. Write $D = \delta + s Z$ with $Z$ standard normal (@sec-gaussian-linear).
2. $\max(D, 0)$ is zero unless $D > 0$, that is, unless $Z > -\delta/s$.
   Therefore
   $\E[\max(D, 0)] = \int_{-\delta/s}^{\infty} (\delta + s z)\, \phi(z)\, \dd z$,
   where $\phi$ is the standard normal density.
3. Split the integral into two:
   $\delta \int_{-\delta/s}^{\infty} \phi(z)\, \dd z + s \int_{-\delta/s}^{\infty} z\, \phi(z)\, \dd z$.
4. The first integral is $1 - \Phi(-\delta/s) = \Phi(\delta/s)$, by the
   symmetry of $\phi$.
5. For the second, the density satisfies $\phi'(z) = -z\,\phi(z)$, so
   $z\,\phi(z)$ has antiderivative $-\phi(z)$, and
   $\int_{a}^{\infty} z\,\phi(z)\, \dd z = \phi(a)$. With $a = -\delta/s$ and
   $\phi(-a) = \phi(a)$, the second integral is $\phi(\delta/s)$.
6. Combine the two terms.
:::

$$
\E[\max(D, 0)] = \delta\, \Phi\!\left(\frac{\delta}{s}\right) + s\, \phi\!\left(\frac{\delta}{s}\right).
$$ {#eq-acq-positive-part}

When $s = 0$, $D$ is the constant $\delta$ and the expectation is
$\max(\delta, 0)$, which is also the limit of @eq-acq-positive-part as
$s \to 0$.

For expected improvement, take $D = f(\vx) - f^*_n - \xi$, with the same
optional margin $\xi$ as in PI. Its mean is $\delta = \mu_n(\vx) - f^*_n - \xi$
and its standard deviation is $s = \sigma_n(\vx)$:

$$
\EI_n(\vx) = \left(\mu_n(\vx) - f^*_n - \xi\right) \Phi(z) + \sigma_n(\vx)\, \phi(z),
\qquad
z = \frac{\mu_n(\vx) - f^*_n - \xi}{\sigma_n(\vx)}.
$$ {#eq-acq-ei}

This is the closed form that @jones1998efficient made standard, with the
margin $\xi$ as a later addition. @brochu2010tutorial report experiments by
Lizotte suggesting that $\xi = 0.01$, scaled by the signal variance if
necessary, works well in almost all cases.

### Reading the formula {#sec-acq-ei-reading}

The two terms of @eq-acq-ei are exploitation and exploration in one
expression. The first is large where the mean is above the incumbent. The
second is large where the standard deviation is large. Unlike the upper
confidence bound, nobody chose the exchange rate between them: it comes out
of the integral.

Three properties make this precise. Write $\EI(\delta, s)$ for the right-hand
side of @eq-acq-positive-part.

- **EI is never below the improvement of the mean.** Since $\max(\cdot, 0)$ is
  convex, Jensen's inequality gives
  $\E[\max(D, 0)] \ge \max(\E[D], 0) = \max(\delta, 0)$. Uncertainty can only
  add value.
- **At a mean equal to the incumbent, EI is about $0.4\,s$.** With
  $\delta = 0$, @eq-acq-positive-part gives
  $\EI = s\,\phi(0) = s / \sqrt{2\pi} \approx 0.399\,s$. This is the example of
  @sec-expectation, where the improvement of the mean is zero but the expected
  improvement is not.
- **EI increases with both the mean and the uncertainty.** The partial
  derivatives are
  $\partial \EI / \partial \delta = \Phi(\delta/s)$ and
  $\partial \EI / \partial s = \phi(\delta/s)$ (@exr-acq-ei-derivatives),
  and both are positive. More uncertainty always makes EI larger, wherever
  the mean is.

The last property is where EI and PI part ways. PI's derivative with respect
to $s$ is $-(\delta/s^2)\,\phi(\delta/s)$, which is negative whenever
$\delta > 0$: once the mean is above the incumbent, PI prefers certainty.
@fig-acq-improvement shows both quantities for a single input.

```{figure}
//| figure: acq-improvement
//| label: fig-acq-improvement
//| fig-cap: "Improvement at a single input. Top: the density of $D = f(x) - f^*_n$, Gaussian with mean $\delta$ and standard deviation $s$ set by the sliders. The shaded area right of zero is the probability of improvement; the dashed curve is the improvement $\max(D, 0)$ times the density, and its area is the expected improvement, @eq-acq-positive-part. Bottom: PI and EI as $s$ varies with $\delta$ held fixed; the dot is the current $s$. When $\delta > 0$, the dashed line in the EI panel marks $\delta$, the value EI approaches as $s$ shrinks."
```

Three settings show the difference.

**A mean below the incumbent.** The figure opens with $\delta = -0.3$. Both PI
and EI grow as $s$ grows: the only way to beat the incumbent is for the
function to be higher than the model expects, and more uncertainty makes
that more likely.

**A mean above the incumbent.** Set $\delta = 0.5$. Now PI *falls* as $s$
grows, because a wider belief puts more weight on falling short. EI still
rises: the extra weight on large improvements outweighs the extra weight on
falling short, which costs nothing, since improvement is never negative.

**A tiny, certain improvement.** Set $\delta = 0.05$ and $s = 0.05$. PI is
about 0.84, while EI is only about 0.05. An input with $\delta = -0.3$ and
$s = 1$ has a PI of only 0.38, but its EI is about 0.27, about five times
larger. EI ranks the second input higher; PI ranks the first.

::: {.pitfall title="Expected improvement underflows"}
Far from the data in a large domain, or late in a run when the incumbent is
high, $\delta / s$ is very negative and @eq-acq-ei is the difference of
numbers so small that floating point rounds them to zero. The acquisition
function then has value and gradient exactly zero on most of the domain, and
a gradient-based optimizer started there cannot move. @ament2023unexpected
argued that this numerical problem, rather than the idea of expected
improvement, is behind EI's inconsistent and often weak performance in the
literature, and proposed LogEI, which computes $\log \EI$ with formulas that
stay accurate in the tail. Its maximizers are the same as EI's or nearly so,
and in their experiments it matched or beat more recent acquisition
functions. BoTorch provides it as `LogExpectedImprovement`.
:::

Two practical notes complete the picture. First, @eq-acq-ei assumes exact
evaluations, so that $f^*_n$ is a known value. With noise the best observed
value is itself uncertain and biased upward, and several noisy variants
replace it; @sec-practice-noise compares them. Second, a real application
shows the formula at work. @snoek2012practical tuned machine learning
algorithms with expected improvement, averaging @eq-acq-ei over samples of the
Gaussian process hyperparameters instead of fixing them. They also modeled
the training time with a second Gaussian process and maximized *expected
improvement per second*, which prefers inputs that are both promising and
quick to evaluate. @sec-cs-classifier replays a
problem of this kind.

## Upper confidence bounds {#sec-ucb}

The upper confidence bound is the rule @sec-bo-loop already used:

$$
\UCB_n(\vx) = \mu_n(\vx) + \beta^{1/2}\, \sigma_n(\vx).
$$ {#eq-acq-ucb}

Its principle comes from the bandit literature (@sec-bandits) and is called
*optimism in the face of uncertainty*: act as if the world were as good as
it plausibly could be, then let the evaluation correct you. If the optimism
was justified, the evaluation finds a good input; if not, the evaluation
shrinks $\sigma_n(\vx)$ there and the inflated estimate deflates, so the rule
moves on.

The weight has a probabilistic reading. Under the posterior,
$\Prob(f(\vx) \le \mu_n(\vx) + \beta^{1/2} \sigma_n(\vx)) = \Phi(\beta^{1/2})$,
so @eq-acq-ucb is a quantile of the belief about $f(\vx)$. With
$\beta^{1/2} = 2$ it is the 97.7% quantile, and maximizing it picks the input
whose plausible best case is highest.

What should $\beta$ be? @srinivas2010gaussian answered with a schedule that
grows slowly with the number of evaluations $t$. For a finite domain $\X$ of
candidate inputs and a confidence parameter $\delta \in (0, 1)$, their
*GP-UCB* rule uses

$$
\beta_t = 2 \log\!\left(\frac{|\X|\, t^2 \pi^2}{6 \delta}\right),
$$ {#eq-acq-gpucb-beta}

and they proved that with this schedule the regret grows sublinearly, so the
average gap to the maximum goes to zero (@sec-gp-bandits gives the theorem
and its proof). The schedule is just large enough that, with probability at
least $1 - \delta$, the bands $\mu \pm \beta_t^{1/2} \sigma$ contain the true
function at every candidate and every step at once. For continuous domains,
the schedule gains a term proportional to $d \log t$ [@srinivas2010gaussian].

The numbers show how cautious the theory is. With $|\X| = 1000$ candidates and
$\delta = 0.1$, @eq-acq-gpucb-beta gives $\beta_t^{1/2} \approx 5.4$ at
$t = 10$ and $\approx 6.2$ at $t = 100$: five to six standard deviations of
optimism. @fig-loop-tradeoff found weights of 1 to 2 best on the running
example, and weights near 6 noticeably worse. The authors themselves found
that their algorithm improved when $\beta_t$ was scaled down by a factor of 5
[@srinivas2010gaussian]. In practice, libraries take a constant $\beta$ from
the user; BoTorch's `UpperConfidenceBound` computes the mean plus
$\sqrt{\beta}$ times the standard deviation. The schedule matters for the
proof, which needs exploration that never switches off. A constant that works
well over a fixed budget is a different, practical question.

## Thompson sampling {#sec-thompson}

The oldest idea in this chapter is from 1933. @thompson1933likelihood
considered two medical treatments of unknown effectiveness and proposed
assigning each new patient a treatment with the probability that it is the
better one, given the evidence so far. For a Gaussian process the rule is:

1. Draw one function $g$ from the posterior given $\D_n$ (@sec-gp-posterior-samples).
2. Evaluate the objective where that sample is largest:
   $\vx_{n+1} \in \argmax_{\vx} g(\vx)$.

The first step is the only random one, and it is what makes the rule work.
Given the data, the sample $g$ and the unknown $f$ have the same distribution,
so the maximizer of $g$ has the same distribution as the maximizer of $f$:

$$
\Prob\!\left(\vx_{n+1} \in A \mid \D_n\right) = \Prob\!\left(\vx^\star \in A \mid \D_n\right)
\quad \text{for every region } A.
$$ {#eq-acq-ts-matching}

This is called *probability matching*: Thompson sampling evaluates each
region exactly as often as the model believes the maximum lies there. A
region the model is sure is poor is almost never chosen. A region that could
hide the maximum is chosen in proportion to how likely that is, however
uncertain the rest of the model is.

Thompson sampling has no weight or margin to tune, and it parallelizes
naturally: to choose ten evaluations at once, draw ten samples and take each
one's maximizer. Its guarantees are close to those of UCB.
@russo2014learning established a connection between posterior sampling and
upper confidence bound algorithms that converts regret bounds proved for UCB
algorithms into Bayesian regret bounds for posterior sampling, including one
for Gaussian process models, and @chowdhury2017kernelized
proved a regret bound for a Gaussian process version when the unknown
function is fixed rather than drawn from the prior.

The cost lies in step 1. A sample on a grid of $m$ inputs needs a Cholesky
factorization of the $m \times m$ posterior covariance, $O(m^3)$, which limits
exact sampling to a few thousand inputs. In more dimensions, implementations
draw the sample on a candidate set, a few thousand points chosen to cover the
promising regions, or draw approximate sample functions that can be evaluated
anywhere, built from a finite set of random basis functions (random features)
or from a prior sample corrected by the data (pathwise updates), and maximize
them with gradients [@rahimi2007random; @wilson2020efficiently]. @sec-acq-optimization explains why the choice of
candidate set becomes the hard part in many dimensions.

## Knowledge gradient {#sec-knowledge-gradient}

Expected improvement makes a quiet assumption: at the end, we recommend one of
the inputs we evaluated, the one with the best observed value
[@frazier2018tutorial]. Often we would happily recommend an input we never
evaluated, if the model is confident it is good. And with noisy evaluations,
no observed value can be trusted at face value anyway, so the natural
recommendation is the input with the highest posterior mean
(@sec-loop-recommend).

The knowledge gradient takes that recommendation seriously. Its utility is
the value of the final recommendation: if we stopped now, we would recommend
the maximizer of the posterior mean, and its expected value under the
posterior is

$$
u(\D_n) = \max_{\vx'} \mu_n(\vx') =: \mu^*_n.
$$

One more evaluation at $\vx$ changes the posterior mean everywhere, not only at
$\vx$, so it can raise $\mu^*$ even if $f(\vx)$ itself turns out to be
mediocre. Plugging this utility into @eq-acq-one-step gives the *knowledge
gradient*:

$$
\mathrm{KG}_n(\vx) = \E_n\!\left[\,\max_{\vx'} \mu_{n+1}(\vx') \;\middle|\; \vx_{n+1} = \vx\right] - \max_{\vx'} \mu_n(\vx').
$$ {#eq-acq-kg}

The value of a query in this sense is the expected value of the final
recommendation after one more answer. A query that maximizes it is *one-step
Bayes optimal*: if the session ended after this one evaluation, no other
choice would leave a better recommendation in expectation, and the knowledge
gradient is the acquisition function that picks it. @sec-eubo-theory relies
on exactly this property when queries are pairs of options.

To compute @eq-acq-kg we need to know how the posterior mean moves when one
observation arrives.

::: {.derivation title="The posterior mean after one more observation"}
Let the next observation be $y = f(\vx) + \varepsilon$ with noise variance
$\sigma_\varepsilon^2$. (This is the noise variance that the rest of the book
writes $\sigma_n^2$; it is renamed in this section only, because the posterior
variance $\sigma_n^2(\vx)$ stands next to it in every formula.) Write
$k_n(\vx', \vx)$ for the posterior covariance between $f(\vx')$ and $f(\vx)$
given $\D_n$.

1. Given $\D_n$, the pair $(f(\vx'), y)$ is jointly Gaussian with means
   $\mu_n(\vx')$ and $\mu_n(\vx)$, variance of $y$ equal to
   $\sigma_n^2(\vx) + \sigma_\varepsilon^2$, and covariance $k_n(\vx', \vx)$,
   since the noise is independent of $f$.
2. Conditioning on $y$ (@sec-gaussian-conditioning) gives
   $\mu_{n+1}(\vx') = \mu_n(\vx') + \dfrac{k_n(\vx', \vx)}{\sigma_n^2(\vx) + \sigma_\varepsilon^2}\,(y - \mu_n(\vx))$.
3. Before we observe it, $y - \mu_n(\vx)$ is Gaussian with mean zero and
   standard deviation $\sqrt{\sigma_n^2(\vx) + \sigma_\varepsilon^2}$, so it
   equals that standard deviation times a standard normal $Z$.
4. Substitute into step 2.
:::

$$
\mu_{n+1}(\vx') = \mu_n(\vx') + \tilde\sigma_n(\vx', \vx)\, Z,
\qquad
\tilde\sigma_n(\vx', \vx) = \frac{k_n(\vx', \vx)}{\sqrt{\sigma_n^2(\vx) + \sigma_\varepsilon^2}}.
$$ {#eq-acq-kg-update}

Every point of the new posterior mean moves by a multiple of the *same*
standard normal $Z$, because a single number, the outcome $y$, moves them all.
Inputs strongly correlated with $\vx$ move a lot; inputs far from $\vx$ hardly
move. So the knowledge gradient is

$$
\mathrm{KG}_n(\vx) = \E\!\left[\max_{\vx'} \left(\mu_n(\vx') + \tilde\sigma_n(\vx', \vx)\, Z\right)\right] - \max_{\vx'} \mu_n(\vx').
$$ {#eq-acq-kg-lines}

Three consequences follow from this form.

- **KG is never negative.** The maximum of functions is convex, and $Z$ has
  mean zero, so by Jensen's inequality the expected maximum is at least the
  maximum at $Z = 0$, which is $\mu^*_n$. Information never hurts in
  expectation.
- **An exact repeat is worth nothing.** If $\vx$ was already evaluated
  without noise, $\sigma_n(\vx) = 0$ and $k_n(\vx', \vx) = 0$ for every $\vx'$,
  so nothing moves and $\mathrm{KG}_n(\vx) = 0$. With noise
  ($\sigma_\varepsilon > 0$), repeating an evaluation can still be worth
  something, which is why KG handles noisy problems gracefully
  [@frazier2018tutorial].
- **With two inputs in play, KG is Clark's formula.** If the maximum in
  @eq-acq-kg-lines runs over only two inputs, it is the maximum of two jointly
  Gaussian values, $a_1 + b_1 Z$ and $a_2 + b_2 Z$, and its expectation is
  the formula of @clark1961greatest (@sec-id-clark). The same formula
  returns in preferential Bayesian optimization (PBO), where a query is a pair
  of options and the acquisition function is the expected utility of the
  better one (EUBO, @eq-eubo-closed); with noise-free answers it has the
  knowledge gradient's one-step optimality (@sec-eubo-theory).

The knowledge gradient also clarifies expected improvement. If the
recommendation must be an evaluated input and evaluations are exact, the
utility in @eq-acq-one-step becomes the best observed value, and the
one-step gain is exactly EI [@frazier2018tutorial]. EI is the knowledge
gradient of a decision maker who will only recommend what has been measured.

@fig-acq-kg makes the lookahead visible.

```{figure}
//| figure: acq-kg
//| label: fig-acq-kg
//| fig-cap: "The knowledge gradient as a one-step lookahead on the running objective. Choose a candidate $x$ (orange) with the slider or by clicking. The violet curves are the posterior means the model would have after evaluating there, @eq-acq-kg-update, for seven equally likely outcomes $Z$ (the quantiles of the predictive distribution); dots mark where each is highest, and the dashed blue line is the highest mean now, $\mu^*_n$. The knowledge gradient is the expected height of the new maximum minus $\mu^*_n$, averaged over all outcomes, not only the seven drawn. The strip shows it for every candidate, computed exactly; the black triangle marks its maximum. *Evaluate at x* adds the candidate to the data."
```

**Look where the violet curves fan out.** At the default candidate, in the
unexplored gap on the right, the fantasies spread widely: a high outcome
would put a new maximum of the mean there, a low one would leave the old
maximum in place. Since a low outcome cannot lower $\mu^*$ (the old maximum is
still available) while a high one raises it, the average rises.

**Move the candidate onto an evaluated point.** The violet curves collapse
onto the blue one and the readout shows KG equal to zero, the second
consequence above. Now move it a little to either side: KG jumps back up.
Two exact evaluations very close together reveal the slope of $f$ between
them, and near the top of a bump a slope can move the maximum of the mean.
The narrow notches in the strip are this effect.

**Compare with the strip.** The knowledge gradient is largest around the
current best region, not in the middle of the widest gap. Near the best
point, an evaluation directly moves the maximum of the mean; in the far gap,
an outcome must be large to matter at all. KG explores less than UCB with
$\beta^{1/2} = 2$, a pattern @fig-acq-compare shows again.

**Add noise.** Set the noise to 0.2. The fantasies spread less, because a
noisy outcome moves the mean less, and the notches widen into valleys. At the
evaluated inputs on the broad bump, KG is now slightly positive: a second
noisy measurement there could still move the maximum of the mean. At the
evaluated input in the dip near $x = 0.45$ it stays at zero, since no outcome
there could make that region the best.

The knowledge gradient was introduced for choosing among a finite set of
alternatives with independent beliefs [@frazier2008knowledge] and extended to
correlated beliefs [@frazier2009knowledge], the setting of a Gaussian process
on a grid. On a finite set, @eq-acq-kg-lines can be computed exactly: the
maximum of the lines $a_j + b_j z$ is a piecewise linear function of $z$, and
integrating each piece against the normal density gives a closed form
[@frazier2009knowledge]; @fig-acq-kg does exactly this. On a continuous
domain the inner maximum has no closed form, and implementations estimate KG
by simulating outcomes, re-solving the inner maximization for each
[@frazier2018tutorial], or, as in BoTorch, by optimizing the candidate
together with one maximizer per simulated outcome in a single "one-shot"
problem [@balandat2020botorch]. (@sec-bo-history tells how the knowledge
gradient and expected improvement were entangled at their origin.)

## Entropy search {#sec-entropy-search}

The rules so far value an evaluation by what it does for a final answer. A
different family values it by what it teaches about the maximum. The
posterior over $f$ induces a distribution over the location of the maximum,
$p(\vx^\star \mid \D_n)$: draw a function from the posterior, find where it
is largest, and repeat. Where this distribution is spread out, we do not
know where the maximum is. An evaluation is valuable if it is expected to
concentrate the distribution.

*Entropy*, from @sec-entropy, measures the spread of a distribution in a way
that depends only on probabilities, not on distances. *Entropy search* picks
the evaluation that most reduces the entropy of $p(\vx^\star \mid \D)$ in
expectation:

$$
\mathrm{ES}_n(\vx) = H\!\left[p(\vx^\star \mid \D_n)\right] - \E_n\!\left[H\!\left[p(\vx^\star \mid \D_n \cup \{(\vx, y)\})\right]\right].
$$ {#eq-acq-es}

The expected reduction in entropy is the *mutual information* between the
outcome $y$ and $\vx^\star$ (@sec-mutual-information), and @eq-acq-es is
@eq-acq-one-step with the negative entropy of $\vx^\star$ as the utility. The
idea was proposed by @villemonteix2009informational and independently by
@hennig2012entropy, who coined the name [@garnett2023bayesian, sec. 12.4].

Computing @eq-acq-es is hard: the distribution of $\vx^\star$ has no closed
form, its entropy must be approximated, and that approximation must be
repeated for every hypothetical outcome $y$. Two reformulations made the idea
practical.

**Predictive entropy search** uses the symmetry of mutual information. The
information that $y$ carries about $\vx^\star$ equals the information that
$\vx^\star$ carries about $y$, so

$$
\mathrm{PES}_n(\vx) = H\!\left[p(y \mid \D_n, \vx)\right] - \E_{\vx^\star}\!\left[H\!\left[p(y \mid \D_n, \vx, \vx^\star)\right]\right].
$$

The first term is the entropy of a Gaussian, available in closed form; the
second averages over sampled maximizers, each requiring an approximation of
how knowing $\vx^\star$ would change the prediction at $\vx$
[@hernandezlobato2014predictive]. Exact ES and PES are the same function;
their approximations differ [@frazier2018tutorial].

**Max-value entropy search** (MES) changes the target: it asks for
information about the maximum *value* $f^\star = f(\vx^\star)$, a single
number, instead of its location [@wang2017maxvalue]. Knowing $f^\star$ tells us
one simple thing about $f(\vx)$: it cannot exceed $f^\star$. So the belief
about $f(\vx)$ given $f^\star$ is the Gaussian posterior cut off above
$f^\star$, a truncated Gaussian whose entropy has a closed form. Averaging
over $K$ sampled maximum values $f^\star_1, \dots, f^\star_K$ gives

$$
\mathrm{MES}_n(\vx) \approx \frac{1}{K} \sum_{k=1}^{K} \left[\frac{\gamma_k(\vx)\, \phi(\gamma_k(\vx))}{2\, \Phi(\gamma_k(\vx))} - \log \Phi(\gamma_k(\vx))\right],
\qquad
\gamma_k(\vx) = \frac{f^\star_k - \mu_n(\vx)}{\sigma_n(\vx)},
$$ {#eq-acq-mes}

which is equation (6) of @wang2017maxvalue. The samples $f^\star_k$ can be
taken as the maxima of posterior sample functions, or drawn from a cheaper
approximation of their distribution. Each term is large when $\gamma_k$ is
small, that is, when the sampled maximum is not far above the mean at $\vx$,
measured in standard deviations: an evaluation there could reveal whether
the maximum is about that high. The authors report that MES matches or improves
on ES and PES at a fraction of the cost, and that it is much less sensitive to
the number of samples [@wang2017maxvalue].

::: {.aside title="Where the bracket in the MES formula comes from"}
For a Gaussian with standard deviation $\sigma$ truncated above at a point
$\gamma$ standard deviations above its mean, the entropy is
$\log\!\left(\sqrt{2\pi e}\,\sigma\,\Phi(\gamma)\right) - \gamma\,\phi(\gamma)/(2\,\Phi(\gamma))$.
The untruncated Gaussian has entropy $\log(\sqrt{2\pi e}\,\sigma)$. Their
difference, the entropy removed by learning that $f(\vx) \le f^\star$, is the
bracket in @eq-acq-mes. It is zero when $\gamma \to \infty$ (the cut is far
above anything plausible, so it teaches nothing) and grows as the cut moves
into the bulk of the distribution.
:::

## Comparing them {#sec-acq-compare}

Each rule has now been derived on its own. To see how differently they
behave, @fig-acq-compare puts all six on one posterior of the running
objective. You play the optimizer: click the plot to evaluate the objective
anywhere, and watch where each rule would go next.

```{figure}
//| figure: acq-compare
//| label: fig-acq-compare
//| fig-cap: "Six acquisition functions on one posterior. Top: the running objective (dashed), the posterior after the evaluations so far (dots), and, in violet bars, how often each input was the maximizer among 64 posterior samples, an estimate of $p(x^\star \mid \mathcal{D}_n)$. Click the plot to evaluate the objective there; click a dot to remove it. Below: PI and EI (@eq-acq-pi, @eq-acq-ei, margin $\xi$), UCB (@eq-acq-ucb, weight $\beta^{1/2}$), one Thompson sample, the knowledge gradient (@eq-acq-kg-lines, exact), and MES (@eq-acq-mes, from the maxima of the 64 samples). Each strip is rescaled to fill its height, so only its shape and the location of its maximum (dot) matter. The selected rule is orange; its next query is the orange line on the posterior. *Evaluate its maximum* runs one step of the loop with it; *New samples* redraws the random samples behind Thompson sampling and MES."
```

The default posterior has four evaluations: two on the broad bump, one in the
dip after it, and one at the far right. The tall bump near $x = 0.73$ sits
in an unexplored gap. Some guided experiments:

**The rules disagree.** PI, EI, MES, and KG choose near the broad bump, at
$x$ between about 0.19 and 0.27, where the mean is already close to the
incumbent and a modest improvement is likely. UCB with $\beta^{1/2} = 2$ goes
into the gap, to $x \approx 0.68$, where the upper edge of the band is
highest. Thompson sampling depends on its sample: press *New samples* a few
times and its choice jumps among the broad bump, the gap, and the left edge,
roughly in proportion to the violet bars.

**Raise the margin.** Move $\xi$ up from 0.01. At $\xi = 0.1$, EI already
switches to the gap: asking for an improvement of at least 0.1 makes the
small, likely gains near the broad bump worthless. PI holds on to the broad
bump until $\xi$ reaches 0.39. Both rules change their choice abruptly at
some margin, which is the sensitivity that Jones warned about.

**Run the loop.** Pick a rule and press *Evaluate its maximum* repeatedly,
then press Reset and try another. Every rule reaches the tall bump
eventually, by different routes: UCB goes there first, EI, KG, and MES after
one more evaluation on the broad bump, PI after two, and Thompson sampling
later still. On this posterior
PI happens to come within 0.05 of the maximum fastest, and KG, which is
trying to improve the maximum of the mean rather than the best observed
value, is slowest by the best-observed yardstick. One run on one problem
ranks nothing.

**Watch the violet bars.** Once the tall bump has been evaluated, the
estimated distribution of $x^\star$ collapses onto it. Thompson sampling and
MES then concentrate their evaluations there; UCB keeps visiting other
regions while their bands are wide.

No rule is best on every problem. Comparisons on benchmark functions favor
different rules on different problems, and the regret bounds of
@sec-regret-bounds-not-say are proved in different settings for different
rules, so they do not rank them either (inference). @tbl-acq-summary lists
the practical differences that do hold.

::: {.table #tbl-acq-summary title="Practical properties of the acquisition functions in this chapter."}
| Rule | Closed form for a GP? | Parameter to set | Values evaluations by | Cost per candidate |
|---|---|---|---|---|
| PI | yes, @eq-acq-pi | margin $\xi$, sensitive | chance of beating the incumbent | one prediction |
| EI | yes, @eq-acq-ei | margin $\xi$, often 0 or small | expected gain over the incumbent | one prediction |
| UCB | yes, @eq-acq-ucb | weight $\beta$, matters | optimistic value | one prediction |
| Thompson | no; a random sample | none | chance of being the maximizer | one joint sample over all candidates |
| KG | on a finite set, @eq-acq-kg-lines | none | gain in the recommended value | a maximization per outcome |
| ES, PES | no | none | information about $\vx^\star$ | expensive approximations |
| MES | given sampled maxima, @eq-acq-mes | number of samples | information about $f^\star = f(\vx^\star)$ | one prediction per sample |
:::

### In two dimensions {#sec-acq-2d}

One-dimensional pictures hide an important fact: the number of places to
look grows exponentially with dimension, while the posterior is informative
only near the data. @fig-acq-2d repeats the comparison on a two-dimensional
problem, the Branin function, a standard test function with three global
maxima of equal height (rescaled here to the unit square and negated so that
larger is better).

```{figure}
//| figure: acq-2d
//| label: fig-acq-2d
//| fig-cap: "Acquisition functions on the two-dimensional Branin function, after six initial evaluations (dots). Top left: the hidden objective, with its three global maxima marked +; values below −1.5 (one corner falls to about −4.7) are drawn as the faintest shade. Top right: the posterior mean on the same color scale. Bottom left: the posterior standard deviation. Bottom right: the selected acquisition function (for Thompson sampling, one posterior sample). In every panel, a stronger color means a higher value. The orange ring on every panel is the selected rule's next query. Click any panel to evaluate the objective there; *Evaluate the maximum* lets the rule choose. The grid has 31 by 31 points, which also serve as the candidates."
```

**Compare the bottom two panels.** The standard deviation is low only in
small disks around the six evaluations; almost the whole square is
uncertain. EI is large where a fairly high mean meets a large standard
deviation, here a broad region between the evaluations in the lower half,
and small both on top of the evaluations and where the mean is low, in the
upper right.

**Switch between rules.** PI and KG choose close to the best evaluation,
since an evaluation there is likely to raise the incumbent or the maximum of
the mean. UCB and MES reach farther out. Thompson sampling's sample is a
whole surface, with its own peaks in unexplored corners; its maximizer can
land anywhere the model allows a maximum.

**Run the loop.** Select PI and press *Evaluate the maximum* ten times. The
evaluations creep in small steps from the best initial point down the slope
to the maximum near $(0.54, 0.15)$: the cautious behavior of @sec-pi, which
here happens to work. Reset and do the same with EI or UCB. Several of their
evaluations go to the edges and corners of the square, where the standard
deviation stays large because no evaluation lies beyond them, and the rest
land near two of the three maxima. With three maxima of equal height, which
one a run finds first depends on its first few evaluations.

**Imagine six dimensions.** In the figure the uncertain region is most of the
square, and a 31 by 31 grid of candidates covers it finely. The same grid in
six dimensions would have $31^6 \approx 9 \times 10^8$ points. The acquisition
function would still be informative only near the data, now in a tiny
fraction of the volume. Finding its maximum is the subject of the next
section.

## Optimizing the acquisition function {#sec-acq-optimization}

Every rule in this chapter ends with "evaluate where the acquisition function
is largest." The figures did that by checking every point of a grid. That is
fine in one or two dimensions and impossible in ten. Maximizing the
acquisition function is a global optimization problem of its own, and the
loop solves one at every step.

What makes it workable is cost. One evaluation of EI or UCB needs one
posterior prediction, $O(n)$ for the mean and $O(n^2)$ for the variance after
the Cholesky factorization is done once per step (@sec-gp-computation). With
a few hundred observations that is microseconds, so an optimizer can afford
tens of thousands of acquisition evaluations per step, while the objective,
which takes hours, gets one. The acquisition function is also smooth and,
for a Gaussian process, differentiable in closed form, so gradient methods
apply [@frazier2018tutorial].

What makes it hard is shape. Acquisition functions are nonconvex and have
many local maxima, one or more near each region of interest. Worse, they are
nearly flat away from the data: with a stationary kernel, far from all
observations the posterior returns to the prior, so its mean and standard
deviation, and with them the acquisition function and its gradient, stop
changing [@garnett2023bayesian, sec. 9.2]. In high dimensions almost the
whole domain is far from the data, so a gradient method started at a random
point usually finds a gradient of nearly zero and goes nowhere.

The standard answer is multi-start local optimization, the approach
recommended in Garnett's textbook [@garnett2023bayesian, sec. 9.2] and used
by BoTorch [@balandat2020botorch].

::: {.algorithm #alg-acq-multistart title="Maximizing an acquisition function by multi-start gradient ascent"}
Input: acquisition function $a_n$ with gradient, domain $[0, 1]^d$,
numbers $R \ll M$.

1. **Screen.** Evaluate $a_n$ at $M$ quasi-random points, for example
   the first $M$ points of a scrambled Sobol sequence (@sec-initial-design).
   Add points near the best observations if the acquisition function is
   likely to be flat elsewhere.
2. **Select.** Choose $R$ starting points among them, favoring high values
   while keeping some variety.
3. **Climb.** From each start, run a gradient-based local optimizer that
   respects the box, such as L-BFGS-B. (It is a quasi-Newton method: it
   estimates the curvature of $a_n$ from successive gradients instead of
   computing second derivatives.)
4. Return the best local maximum found.
:::

BoTorch's defaults follow this outline: the screening points come from a
scrambled Sobol sequence, the starting points are drawn at random with
probabilities proportional to $\exp(\eta Z)$, where $Z$ is the standardized
acquisition value and $\eta$ a temperature, and the local climbs use
L-BFGS-B. The climbs are independent of one another, so they parallelize
well [@garnett2023bayesian, sec. 9.2].

Monte Carlo versions of acquisition functions, which estimate an expectation
by averaging over samples instead of using a closed form, need one more
idea. @wilson2018maximizing showed that when the samples are written as a
fixed transformation of fixed random numbers, the Monte Carlo estimate is a
smooth function of the input, so gradient ascent works on it too. They also
identified a family of acquisition functions, including EI and UCB, whose
properties justify building a batch of evaluations greedily, one point at a
time, each maximized given the points already chosen. Batch selection is the
subject of @sec-batch-bo.

### From two dimensions to twenty {#sec-acq-high-dim}

The grid in @fig-acq-2d had 961 points and missed nothing. Three things
change as the dimension grows to the 6 to 20 of a typical tuning problem.

**Grids are out, and so is uniform screening.** A grid with ten values per
input has $10^6$ points in six dimensions and $10^{20}$ in twenty. Uniform
random or Sobol screening points do not need a grid, but they share its
weakness in a different form: they land mostly far from the data, where the
acquisition function is flat. A useful picture: if an evaluation informs the
model within a radius of about $0.2$, one evaluation influences 0.033% of the
six-dimensional unit cube (the computation behind @sec-explore-exploit). With
60 evaluations, at most about 2% of uniformly placed screening points land
within that radius of any evaluation, and the acquisition function is
essentially constant at the rest.

@fig-acq-highdim measures this directly on Hartmann-6, a standard
six-dimensional test function, hidden among irrelevant inputs when $d > 6$.

```{figure}
//| figure: acq-highdim
//| label: fig-acq-highdim
//| fig-cap: "Two ways to look for the maximum of expected improvement, in 2 to 20 dimensions. After a Latin hypercube of $10 + 2d$ evaluations of Hartmann-6 (for $d = 2$, a slice through its maximum; for $d > 6$, with $d - 6$ irrelevant inputs added), a Gaussian process with lengthscale $0.2\sqrt{d}$ is fitted, and EI is computed at 1,000 points drawn uniformly from the cube (gray) and at 1,000 random perturbations, with standard deviation 0.05 per coordinate, of the five best evaluations (orange). Left: each candidate's distance to the nearest evaluation, in lengthscales. Right: the EI values on a log scale, with each set's best marked; values below $10^{-12}$ are counted in the leftmost bin. EI is in units of the standardized observations. *New draw* changes the evaluations and candidates; the numbers are illustrative."
```

**Switch from $d = 2$ to $d = 20$.** In two dimensions the best uniform
candidate is as good as the best perturbation of the data: both sets find
the same peak of EI. In twenty dimensions the best of 1,000 uniform
candidates has, in the default draw, an expected improvement more than a
hundred times smaller than the best perturbation. The lengthscale grows with
$\sqrt{d}$, the rate at which @hvarfner2024vanilla scale their lengthscale
prior for high dimensions, yet the
typical uniform candidate is still about 1.4 lengthscales from the nearest
evaluation in twenty dimensions, against about 0.4 in two (left panel). There the posterior mean is ordinary and the
incumbent is several standard deviations away. EI there is tiny and
nearly constant, so the screening step of @alg-acq-multistart would start its
climbs from uninformative points.

**Candidates come from near the data.** So practical implementations put
their candidates where the acquisition function has structure. One recipe
mixes a quasi-random sample of the whole cube with random perturbations of
the best inputs found so far, perturbing only some coordinates at a time.
TuRBO, a method for high-dimensional problems that searches inside a *trust
region*, a box around the best point found so far (@sec-hd-practice-local),
draws its Thompson samples on candidate sets of $\min(100d, 5000)$ points built this
way: each coordinate of a candidate takes a quasi-random value within the
trust region with probability $\min(1, 20/d)$ and otherwise keeps the value
of the region's center [@eriksson2019scalable]. The perturbation is what keeps
the candidates in the informative region as $d$ grows. The same idea, in the
form of BoTorch's option to add points sampled around the best inputs, feeds
step 1 of @alg-acq-multistart.

**Numerical flatness becomes the default.** With many dimensions and many
observations, the region where expected improvement is distinguishable from
zero in floating point shrinks, and plain EI hands the optimizer a function
that is exactly zero with zero gradient almost everywhere. This is the regime
where computing $\log \EI$ instead of $\EI$ pays off most
[@ament2023unexpected].

None of this changes the statistics of the acquisition function; it changes
whether we find its maximum. That matters in practice. The guarantees of
@sec-regret assume the maximization is exact [@srinivas2010gaussian], and
@ament2023unexpected found that better maximization alone changed how EI
compared with newer acquisition functions. Whether a published comparison
used a good inner optimizer is worth checking before trusting it (inference).
@sec-high-dim-practice discusses the surrogate side of high dimensions,
where the default lengthscale priors matter as much as the acquisition
function.

## Exercises {#sec-acq-exercises}

::: {.exercise #exr-acq-ei-derivatives}
Let $\EI(\delta, s) = \delta\,\Phi(\delta/s) + s\,\phi(\delta/s)$ from
@eq-acq-positive-part. Show that $\partial \EI / \partial \delta = \Phi(\delta/s)$
and $\partial \EI / \partial s = \phi(\delta/s)$. Conclude that EI increases
with both the mean and the standard deviation. Then compute the derivative
of PI, $\Phi(\delta/s)$, with respect to $s$, and say when it is negative.

::: {.solution}
Use $\phi'(z) = -z\,\phi(z)$ and write $z = \delta/s$. For $\delta$:
$\partial_\delta[\delta\,\Phi(z)] = \Phi(z) + \delta\,\phi(z)/s = \Phi(z) + z\,\phi(z)$
and $\partial_\delta[s\,\phi(z)] = s\,\phi'(z)/s = -z\,\phi(z)$. The sum is
$\Phi(z)$. For $s$, with $\partial z / \partial s = -\delta/s^2 = -z/s$:
$\partial_s[\delta\,\Phi(z)] = -\delta\,\phi(z)\,z/s = -z^2\phi(z)$ and
$\partial_s[s\,\phi(z)] = \phi(z) + s\,\phi'(z)(-z/s) = \phi(z) + z^2\phi(z)$.
The sum is $\phi(z)$. Both $\Phi$ and $\phi$ are positive, so EI increases in
both arguments. For PI, $\partial_s \Phi(z) = \phi(z)\,(-z/s) = -(\delta/s^2)\,\phi(\delta/s)$,
which is negative exactly when $\delta > 0$: once the mean is above the
target, PI prefers less uncertainty.
:::
:::

::: {.exercise #exr-acq-kg-two}
Two inputs have independent posterior beliefs $f(a) \sim \N(0.5, 0.1^2)$ and
$f(b) \sim \N(0.3, 0.4^2)$, and no other input can be recommended. One exact
evaluation is allowed. Use @eq-acq-kg-lines to compute the knowledge gradient
of evaluating $b$, and show that it equals the expected improvement of $b$
over the value $0.5$. Why is the knowledge gradient of evaluating $a$ so
small, and would that still be true if $a$ and $b$ were correlated?

::: {.solution}
Evaluating $b$ exactly reveals $f(b)$, so $\tilde\sigma(b, b) = \sigma(b) = 0.4$
and, by independence, $\tilde\sigma(a, b) = 0$. The new means are $0.5$ for
$a$ and $0.3 + 0.4Z$ for $b$, so
$\mathrm{KG}(b) = \E[\max(0.5, 0.3 + 0.4Z)] - 0.5 = \E[\max(0.3 + 0.4Z - 0.5, 0)]$,
the expected positive part of a Gaussian with $\delta = -0.2$ and $s = 0.4$.
By @eq-acq-positive-part this is
$-0.2\,\Phi(-0.5) + 0.4\,\phi(-0.5) \approx -0.2 \times 0.3085 + 0.4 \times 0.3521 \approx 0.079$,
which is EI of $b$ against the incumbent value $0.5$. Evaluating $a$ moves
only $a$'s mean: $\max(0.5 + 0.1Z, 0.3)$ has expectation
$0.5 + \E[\max(0.3 - 0.5 - 0.1Z, 0)] = 0.5 + \E[\max(-0.2 + 0.1Z', 0)]$
with $Z' = -Z$, which is $0.5 + (-0.2\,\Phi(-2) + 0.1\,\phi(-2))$, less than $0.5 + 0.001$.
So $\mathrm{KG}(a) < 0.001$: evaluating $a$ almost never changes the
recommendation, because $a$ would have to fall two standard deviations
below its mean to lose to $b$. With a positive correlation, evaluating $a$ would also move
$b$'s mean, and KG would count that; this is how KG values evaluations by
their effect on the whole posterior.
:::
:::

::: {.exercise #exr-acq-ts-two}
Two inputs have independent beliefs $f(a) \sim \N(\mu_a, s_a^2)$ and
$f(b) \sim \N(\mu_b, s_b^2)$. Show that Thompson sampling chooses $a$ with
probability $\Phi\!\left((\mu_a - \mu_b) / \sqrt{s_a^2 + s_b^2}\right)$, and
check that this is the posterior probability that $a$ is the better input.
What happens to this probability as both standard deviations shrink while
$\mu_a > \mu_b$ stays fixed?

::: {.solution}
Thompson sampling draws $g(a) \sim \N(\mu_a, s_a^2)$ and
$g(b) \sim \N(\mu_b, s_b^2)$ independently and chooses $a$ if $g(a) > g(b)$. The
difference $g(a) - g(b)$ is Gaussian with mean $\mu_a - \mu_b$ and variance
$s_a^2 + s_b^2$, so $\Prob(g(a) > g(b)) = \Phi((\mu_a - \mu_b)/\sqrt{s_a^2 + s_b^2})$.
The same computation with $f$ in place of $g$ gives the posterior probability
that $f(a) > f(b)$, which is @eq-acq-ts-matching for this case. As the
standard deviations shrink, the argument of $\Phi$ grows without bound and
the probability tends to 1: Thompson sampling stops exploring $b$ exactly as
fast as the evidence rules it out.
:::
:::

::: {.exercise #exr-acq-ucb-quantile}
Show that maximizing $\mu_n(\vx) + \beta^{1/2}\sigma_n(\vx)$ is the same as
maximizing the $q$-quantile of the posterior belief about $f(\vx)$, and find
$q$ for $\beta^{1/2} = 1$, $2$, and the value $\beta_t^{1/2} \approx 6.2$ from
@sec-ucb. What does the last value say about how often the theory expects the
true function to exceed the band?

::: {.solution}
For a Gaussian belief with mean $\mu$ and standard deviation $\sigma$, the
$q$-quantile is $\mu + \Phi^{-1}(q)\,\sigma$. So
$\mu + \beta^{1/2}\sigma$ is the quantile with $q = \Phi(\beta^{1/2})$, the same
$q$ at every input, and maximizing one maximizes the other. For
$\beta^{1/2} = 1$, $q \approx 0.841$; for $2$, $q \approx 0.977$; for $6.2$,
$q$ differs from 1 by about $3 \times 10^{-10}$. The theory sets the band so
wide that, with high probability, the function stays inside it at every one
of the candidates and at every step simultaneously, which requires a tiny
failure probability per candidate and per step. This is the union bound behind
@eq-acq-gpucb-beta (the probability that any one of many events happens is at
most the sum of their probabilities; @sec-regret-gpucb-bound uses it), and it
is why the schedule grows with $\log |\X|$ and $\log t$.
:::
:::

## Further reading {#further-reading .unnumbered}

- @garnett2023bayesian, chapters 7 and 8, derives every acquisition function
  in this chapter from one decision-theoretic framework and computes each for
  Gaussian processes, with gradients; its chapter 9 covers their optimization.
- @frazier2018tutorial explains expected improvement, the knowledge gradient,
  and entropy search side by side, including when the knowledge gradient is
  worth its cost.
- @jones1998efficient derives expected improvement in closed form and shows
  it at work, and @jones2001taxonomy compares the improvement-based rules
  with care.
- @srinivas2010gaussian introduces GP-UCB with its regret analysis; read it
  with @sec-gp-bandits.
- @hennig2012entropy, @hernandezlobato2014predictive, and @wang2017maxvalue
  trace the entropy search family from its first full statement to the
  max-value variant.
- @ament2023unexpected is a lesson in how numerics can masquerade as
  statistics, and @wilson2018maximizing is the reference for maximizing
  Monte Carlo acquisition functions.
