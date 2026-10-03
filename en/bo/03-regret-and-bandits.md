---
status: done
synopsis: "How optimizers are scored: simple and cumulative regret, the multi-armed bandit and its classic algorithms, the Lai-Robbins lower bound, GP-UCB's regret bound through the maximum information gain, and what such guarantees do and do not promise in practice."
sources: ["Lai and Robbins 1985","Auer et al. 2002","Srinivas et al. 2010","Lattimore and Szepesvari 2020","Abbasi-Yadkori et al. 2011","Chowdhury and Gopalan 2017"]
---

# Regret, Bandits, and Guarantees {#sec-regret}

@sec-acquisition left us with a shelf of acquisition functions, each a
reasonable answer to the question of where to evaluate next. Choosing among them, or
proposing a new one, needs a way to say that one optimizer is better than
another. The everyday answer is a benchmark plot: the best value found against
the number of evaluations, on a handful of test functions. Later chapters use
such plots. This chapter asks for something sturdier: a score that is defined
for every problem, and statements about that score that hold for every problem
in a stated class.

The score is called *regret*, and its theory grew up around a problem simpler
than Bayesian optimization: the multi-armed bandit, in which a gambler chooses
again and again among a few slot machines with unknown payout rates. The bandit
strips the exploration and exploitation trade-off of @sec-explore-exploit down
to its bones. In that setting we can count how much exploring an algorithm
does, show that the best algorithms explore only logarithmically often, and
prove that no algorithm can explore less. We then carry the analysis back to
functions, where a Gaussian process turns infinitely many correlated arms into
a problem with a measurable complexity, and the GP-UCB algorithm comes with a
bound in terms of it.

The last section is the one to read even if you skip the proofs. A regret bound
is a precise statement about a stylized problem. It says some useful things
about practice and is silent about others, and the figures in this chapter let
you see how far apart a bound and an algorithm's behavior can be.

## Scoring an optimizer {#sec-regret-definitions}

Picture two optimizers working on the running objective of @sec-bo-loop. The
first spends most of its evaluations near the taller peak. The second wanders
over the whole domain for most of its budget and, at the end, recommends the
same peak. Which one did better? It depends on whether the evaluations along
the way cost something beyond their price: whether we care about the journey
or only the destination. Two kinds of regret make the two answers precise.

We maximize an unknown function $f$ over a domain $\X$. Write
$f^\star = \max_{\vx \in \X} f(\vx)$ for the best value and $\vx^\star$ for an
input that attains it, as in @eq-loop-problem. An optimizer evaluates $f$ at $\vx_1, \vx_2, \dots$, possibly
with noise, and after $T$ evaluations it recommends an input $\hat\vx_T$,
usually the best one observed or the maximizer of the posterior mean.

::: {.definition #def-regret-kinds title="Regret"}
The *instantaneous regret* of the $t$-th evaluation is the shortfall of the
input chosen,
$$
r_t = f^\star - f(\vx_t) \;\ge\; 0.
$$
The *cumulative regret* after $T$ evaluations adds up the shortfalls along
the way, and the *simple regret* scores only the recommendation:
$$
R_T = \sum_{t=1}^{T} r_t,
\qquad
s_T = f^\star - f(\hat\vx_T).
$$ {#eq-regret-defs}
:::

Regret is measured with the true $f$ at the chosen inputs, not with the noisy
observations, and the optimizer can never compute it, because it does not
know $f^\star$. Regret is the analyst's score: computable on a benchmark whose
maximum is known, and the quantity that theorems bound.

::: {.example #ex-regret-three title="Same destination, different journeys"}
Let $f^\star = 1$. Optimizer A evaluates three inputs with values $0.2$, $0.7$,
and $0.9$. Its instantaneous regrets are $0.8$, $0.3$, and $0.1$, so
$R_3 = 1.2$, and if it recommends its best input, $s_3 = 0.1$. Optimizer B
evaluates three inputs worth $0.9$ each. It has the same simple regret,
$s_3 = 0.1$, and a quarter of the cumulative regret, $R_3 = 0.3$.
:::

Since every $r_t$ is at most the range of $f$, cumulative regret can grow at
most linearly in $T$. An optimizer that never learns, such as one that queries
uniformly at random forever, does grow linearly: each evaluation costs the same
amount on average. An optimizer is called *no-regret* when its cumulative regret
grows *sublinearly*, more slowly than any straight line, so that the average
regret $R_T / T$ tends to zero. A bound of the form $R_T \le C\sqrt{T}$ says
the average regret falls like $1/\sqrt{T}$; a bound of the form
$R_T \le C \log T$ says almost all late evaluations are spent at nearly optimal
inputs.

A bound on cumulative regret also bounds the simple regret of the best input
visited. The smallest of $T$ numbers is at most their average, so

$$
f^\star - \max_{t \le T} f(\vx_t) \;=\; \min_{t \le T} r_t \;\le\; \frac{R_T}{T}.
$$ {#eq-regret-simple-from-cumulative}

This is how cumulative regret bounds become convergence rates for optimization
[@srinivas2010gaussian]. Two caveats come with it. With noisy observations, the
optimizer does not know which of its inputs had the largest $f$, so the best
input visited is not the same as the best input it can identify; choosing what
to report is a practical problem of its own (@sec-practice-noise). And the
converse fails: an optimizer that explores evenly can have small simple regret
and linear cumulative regret. In the bandit problems of the next section the
gap is sharp. On a fixed problem, the simple regret of uniform exploration
falls exponentially with the budget, while algorithms that keep cumulative
regret low have simple regret that falls only polynomially, because they stop
sampling the near-best competitors as soon as they can
[@bubeck2009pure; @lattimore2020bandit, ch. 33].

Which score to use depends on who pays for the evaluations. When tuning the
hyperparameters of a model, or searching for a material, only the final
recommendation is used; the evaluations along the way are a cost measured in
compute or lab time, and simple regret is the right score. In an online
experiment, every evaluation is a product variant shown to real users, and in a
preference study, every option is something a person has to look at, wear, or
listen to; there the path matters and cumulative regret is the right score.
@sec-pbo-failure-modes describes a preference method with the better final
answer and more than 2.5 times the cumulative regret of its competitor
[@xu2024principledb].

A last distinction concerns what a bound quantifies over. A *frequentist*
bound holds for every function in a stated class, such as all functions of a
given smoothness. A *Bayesian* bound holds on average, or with high
probability, over functions drawn from a prior. Both kinds appear in
@sec-gp-bandits.

::: {.keyidea title="The destination and the journey"}
Simple regret scores the final recommendation; cumulative regret scores every
evaluation along the way. Low cumulative regret implies low simple regret for
the best input visited, but not the other way around.
:::

## Multi-armed bandits {#sec-bandits}

Counting exploration requires a setting simple enough to count in. The
*multi-armed bandit* is that setting. There are $K$ actions, called *arms*
after the lever of a slot machine. Pulling arm $i$ returns a random reward
whose distribution is fixed but unknown, with mean $\theta_i$. A player pulls
one arm per round for $T$ rounds and wants the largest total reward. The
earliest rule for such a problem is Thompson's, from 1933
[@thompson1933likelihood]; Robbins stated the problem formally in 1952, as a
question in the sequential design of experiments, and introduced the notion of
regret [@robbins1952some; @lattimore2020bandit, ch. 4].

A bandit is Bayesian optimization with two simplifications. The domain is a
finite set of $K$ inputs, and the inputs are unrelated: pulling arm 3 says
nothing about arm 4. Everything else carries over, including the noise, the
budget, and the tension between trying the arm that looks best and checking
the ones that might be better.

In this section each reward is a coin flip: a pull of arm $i$ pays 1 with
probability $\theta_i$ and 0 otherwise. Write $\theta^* = \max_i \theta_i$
for the best arm's mean and $\Delta_i = \theta^* - \theta_i$ for the *gap* of
arm $i$, how much is lost on average each time it is pulled instead of the best
arm. (In this section $\theta_i$ is an arm's mean reward; the symbol $\mu$ stays
reserved for the posterior mean of a Gaussian process.)

The regret of a bandit algorithm is cumulative regret with the arm means in the
role of $f$: $R_T = \sum_{t} (\theta^* - \theta_{a_t})$, where $a_t$ is the arm
pulled in round $t$. Because it uses the means rather than the coin flips that
happened, it is sometimes called the *pseudo-regret*. Let $N_i(T)$ count the
pulls of arm $i$ in the first $T$ rounds. Grouping the sum by arm gives

$$
\E[R_T] = \sum_{i=1}^{K} \Delta_i \, \E[N_i(T)].
$$ {#eq-regret-decomposition}

The decomposition turns the problem into bookkeeping. Regret is the number of
times each worse arm is pulled, weighted by how much worse it is. An algorithm
keeps regret low by pulling bad arms rarely, but it can only learn that an arm
is bad by pulling it. The whole subject is the question of how many pulls are
enough.

Before reading about algorithms, try it yourself. The figure below hides the
payout rates of five arms; you have 50 pulls.

```{figure}
//| figure: regret-bandits
//| label: fig-regret-you
//| fig-cap: "Pull the arms yourself. Each pull pays 1 (a filled square) or 0 (an empty square) with the arm's hidden probability. The regret curve stays hidden until you reveal the arms or spend all 50 pulls, because its slope would give the best arm away. Your single run is then compared with the average of 20 runs of the algorithms of the next sections, which faced the same sequence of payouts as you did in their first run. *New arms* draws a new set of arms."
mode: you
```

You probably formed a favorite after two or three pulls per arm. Did you go
back to an arm that started with two losses? With payout rates between 0.15
and 0.6, two losses in a row happen for the best arm 16% of the time, so
abandoning an arm that early is a gamble. Each algorithm below is a rule for
that decision.

### Greedy and epsilon-greedy {#sec-regret-eps-greedy}

The simplest rule is *greedy*: pull every arm once, then always pull the arm
with the highest average reward so far. It fails in an instructive way. If the
best arm happens to pay 0 on its first pull, its average is 0, some worse arm's
average is positive, and greedy may never pull the best arm again. That
happens with a fixed positive probability, and when it does the regret grows
linearly forever.

*Epsilon-greedy* repairs this by forcing exploration: in each round, with
probability $\varepsilon$ pull an arm chosen uniformly at random, and otherwise
pull the greedy arm. Every arm is now pulled infinitely often, so every average
converges to its mean. But the repair has a price that never stops being paid.
With constant $\varepsilon$, every round spends probability $\varepsilon$ on a
uniformly random arm, which costs $\frac{\varepsilon}{K} \sum_i \Delta_i$ in
expected regret per round, so regret grows linearly with slope at least that
(@exr-regret-eps-linear). Auer, Cesa-Bianchi, and Fischer showed that letting
$\varepsilon$ decay as $\varepsilon_n = \min\{1, cK/(\Delta_0^2 n)\}$ in round
$n$ gives logarithmic regret, but only if $\Delta_0$ (written $d$ in their
paper) is a lower bound on the gap between the best and the second-best arm,
which a player does not know. In their
experiments no value of $c$ worked well for all the reward distributions they
tried [@auer2002finite].

### Where confidence bounds come from {#sec-regret-concentration}

Forcing exploration at a fixed rate keeps pulling arms that are already known
to be bad. A better rule explores an arm only while the evidence about it is
still weak, and for that it needs a number: after $n$ pulls, how far from the
arm's mean can its average plausibly be? The answer comes from *concentration
inequalities*, bounds on the probability that an average strays from its mean
by more than a given distance. Every upper bound on regret in this chapter
rests on one, and the choice of inequality decides the constants and the
logarithms in the algorithms built on it.

Fix one arm with mean $\theta$, and let $Y_1, \dots, Y_n$ be $n$ of its
rewards, independent and each in $[0, 1]$, with average
$\hat\theta_n = \frac1n \sum_{s=1}^{n} Y_s$. The question is how large the
*tail probability* $\Prob(\hat\theta_n \ge \theta + a)$ can be for a deviation
$a > 0$. The lower tail, $\Prob(\hat\theta_n \le \theta - a)$, works the same
way, so we treat only the upper one.

**Markov's inequality.** A quantity that is never negative and has a small
mean cannot often be large: if it were at least $c$ more than a fraction
$\E[Z]/c$ of the time, those occasions alone would push its mean above
$\E[Z]$. For $Z \ge 0$ and $c > 0$,

$$
\Prob(Z \ge c) \le \frac{\E[Z]}{c}.
$$ {#eq-regret-markov}

To see it, note that $Z$ is at least $c$ times the indicator of the event
$Z \ge c$ (1 when the event happens, 0 when not), and take expectations of
both sides. The average $\hat\theta_n$ is never negative and has mean
$\theta$, so $\Prob(\hat\theta_n \ge \theta + a) \le \theta/(\theta + a)$. For
a fair coin and $a = 0.1$ that is $0.83$, and it stays $0.83$ however many
rewards are averaged. Markov's inequality knows only the mean, and the mean of
an average does not change with $n$.

**Chebyshev's inequality.** The cure is to apply Markov's inequality to
something that does shrink with $n$. The squared deviation
$(\hat\theta_n - \theta)^2$ has mean $\Var[\hat\theta_n] = \Var[Y]/n$, because
the variance of a sum of independent terms is the sum of their variances
(@sec-independence) and dividing a sum by $n$ divides its variance by $n^2$.
The event $\hat\theta_n \ge \theta + a$ implies
$(\hat\theta_n - \theta)^2 \ge a^2$, so

$$
\Prob(\hat\theta_n \ge \theta + a) \le \frac{\Var[Y]}{n a^2}.
$$ {#eq-regret-chebyshev}

(Chebyshev's inequality bounds both tails together, and so each one.) For a
coin, $\Var[Y] = \theta(1 - \theta)$, at most $\tfrac14$ (@sec-prob-variance).
The bound now falls like $1/n$, but turned around it is expensive. Setting the
right side to a target failure probability $\delta$ gives the width
$a = \sqrt{\Var[Y]/(n\delta)}$, which grows like $1/\sqrt{\delta}$. Bandit
algorithms need very small failure probabilities, as small as $t^{-4}$ in round
$t$ for the algorithm of the next subsection, and a width that grows like
$t^2$ would keep every arm in play forever. The truth is much better. By the
central limit theorem an average of many independent terms is close to
Gaussian (@sec-gauss-why), and a Gaussian's tail falls like $e^{-c^2/2}$ in the
number $c$ of standard deviations, not like $1/c^2$.

**The Chernoff method.** Markov's inequality applied to an exponential
captures that behavior [@lattimore2020bandit, ch. 5]. For any $\lambda > 0$,
the event $\hat\theta_n - \theta \ge a$ is the same as the event
$e^{\lambda n(\hat\theta_n - \theta)} \ge e^{\lambda n a}$, and the exponential
turns the sum inside the average into a product,
$e^{\lambda n(\hat\theta_n - \theta)} = \prod_{s=1}^{n} e^{\lambda(Y_s - \theta)}$.
The expectation of a product of independent factors is the product of their
expectations, because their joint distribution factorizes
(@def-prob-independence). What remains is a bound on one factor, and the
condition that supplies it has a name.

::: {.definition #def-regret-subgaussian title="Sub-Gaussian"}
A random variable $Z$ with mean zero is *$R$-sub-Gaussian* if, for every real
$\lambda$,
$$
\E\big[e^{\lambda Z}\big] \le e^{\lambda^2 R^2 / 2}.
$$
:::

A Gaussian with mean zero and standard deviation $R$ satisfies this with
equality, so the condition says that the tails of $Z$ are no heavier than that
Gaussian's. Bounded variables qualify too. By *Hoeffding's lemma*, a variable
with mean zero that always lies in an interval $[l, u]$ is
$\tfrac12(u - l)$-sub-Gaussian [@lattimore2020bandit, ch. 5]. A reward in
$[0, 1]$ minus its mean is therefore $\tfrac12$-sub-Gaussian, and noise with
mean zero that is never larger than $\sigma$ in absolute value is
$\sigma$-sub-Gaussian. (The
letter $R$ follows the papers cited below; it is not the regret $R_T$.)

With this condition the Chernoff method gives *Hoeffding's inequality*, which
Hoeffding proved for sums of bounded random variables
[@hoeffding1963probability]: for rewards in $[0, 1]$,

$$
\Prob(\hat\theta_n \ge \theta + a) \le e^{-2 n a^2}.
$$ {#eq-regret-hoeffding}

::: {.derivation title="Hoeffding's inequality by the Chernoff method"}
1. By @eq-regret-markov applied to $e^{\lambda n(\hat\theta_n - \theta)}$ with
   $c = e^{\lambda n a}$, and the product above,
   $\Prob(\hat\theta_n - \theta \ge a) \le e^{-\lambda n a} \prod_{s=1}^{n} \E\big[e^{\lambda(Y_s - \theta)}\big]$
   for every $\lambda > 0$.
2. Each $Y_s - \theta$ is $R$-sub-Gaussian, so each factor is at most
   $e^{\lambda^2 R^2/2}$, and the bound becomes
   $\exp(-\lambda n a + n \lambda^2 R^2/2)$.
3. The exponent is a parabola in $\lambda$, smallest at $\lambda = a/R^2$,
   where it equals $-n a^2/(2R^2)$. Step 1 holds for every $\lambda$, so it
   holds for this one: $\Prob(\hat\theta_n - \theta \ge a) \le e^{-n a^2/(2R^2)}$.
4. Rewards in $[0, 1]$ have $R = \tfrac12$, which gives @eq-regret-hoeffding.
:::

Turned around, Hoeffding's inequality says that with probability at least
$1 - \delta$, the average is below $\theta + \sqrt{\ln(1/\delta)/(2n)}$. The
failure probability now enters through its logarithm. Shrinking $\delta$ from
0.05 to $10^{-6}$ widens a Hoeffding interval by a factor of 2.1 and a
Chebyshev interval by a factor of 224. Applied to a Gaussian $Z$ with standard
deviation 1, the same method gives $\Prob(Z \ge c) \le e^{-c^2/2}$, and a
direct calculation halves this [@srinivas2010gaussian, Lemma 5.1], so both
tails together have probability at most $e^{-c^2/2}$. That is the Gaussian
bound in step 1 of the GP-UCB proof in @sec-regret-gpucb-bound.

::: {.aside title="Sharper exponents for coins"}
Hoeffding's inequality uses only the range of the rewards. For coin flips the
Chernoff method can use the whole distribution, and gives
$\Prob(\hat\theta_n \ge \theta + a) \le e^{-n\,\mathrm{kl}(\theta + a,\, \theta)}$,
with $\mathrm{kl}$ the Kullback-Leibler divergence between two coins of
@eq-info-kl-bernoulli [@lattimore2020bandit, Lemma 10.3]. Pinsker's
inequality, $\mathrm{kl}(p, q) \ge 2(p - q)^2$, which @sec-regret-lower-bounds
uses again, shows that this is never weaker than $e^{-2na^2}$. The two
exponents nearly agree for a fair coin and differ most for coins near 0 or 1,
whose variance is small. An upper confidence bound built on the
$\mathrm{kl}$ version, KL-UCB, attains the Lai-Robbins constant of
@sec-regret-lower-bounds for Bernoulli arms [@lattimore2020bandit, Thm. 10.6].
:::

**The union bound.** An algorithm does not use one interval. It uses one per
arm in every round, and its analysis needs them all to hold, or at least needs
to count how often one fails. The tool is elementary: the probability that at
least one of several events happens is at most the sum of their probabilities,

$$
\Prob(A_1 \text{ or } A_2 \text{ or } \cdots \text{ or } A_m) \le \sum_{j=1}^{m} \Prob(A_j),
$$ {#eq-regret-union}

because an outcome in which any of the events happens is counted at least once
on the right. This *union bound* asks nothing about how the events depend on
one another. To make $m$ intervals hold together with probability at least
$1 - \delta$, give each the failure probability $\delta/m$. With Hoeffding's
inequality the width becomes

$$
a = \sqrt{\frac{\ln(m/\delta)}{2n}} = \sqrt{\frac{\ln m + \ln(1/\delta)}{2n}},
$$ {#eq-regret-union-width}

so the number of intervals enters as an additive $\ln m$ under the square root.
With Chebyshev's inequality the width would grow like $\sqrt{m}$. An
exponential tail is what makes many simultaneous intervals affordable.

This is where the logarithms in bandit algorithms come from. An interval that
must hold in every round up to a horizon $T$ needs $m = T$ and pays $\ln T$;
one for each of $K$ arms in every round pays $\ln(KT)$. When the horizon is not
known in advance, the budget $\delta$ can be spread unevenly instead, giving
round $t$ the share $6\delta/(\pi^2 t^2)$. The shares add up to $\delta$
because $\sum_{t \ge 1} 1/t^2 = \pi^2/6$, and the width in round $t$ has
$\ln(\pi^2 t^2/(6\delta))$ in place of $\ln(m/\delta)$, so it grows like
$\sqrt{\ln t}$. GP-UCB's $\beta_t$ is this construction with one more union,
over the $|\X|$ inputs (@sec-regret-gpucb-bound). UCB1, in the next
subsection, sets the failure probability of each interval to $t^{-4}$ in round
$t$. Solving $e^{-2na^2} = t^{-4}$ for $a$ gives $a = \sqrt{2\ln t / n}$, the
bonus of @eq-regret-ucb1. The exponent 4 pays for a union of its own: in round
$t$ two arms being compared can have been pulled any numbers of times up to
$t$, about $t^2$ combinations of counts, and $t^2 \cdot t^{-4} = t^{-2}$ still
adds up to a finite total over all rounds.

**When the data choose the sample size.** Hoeffding's inequality is about an
average of $n$ rewards with $n$ fixed before the rewards are seen. A bandit
algorithm decides how often to pull an arm from the rewards it has seen, and an
arm that starts badly is pulled less. A reader may wonder whether that
matters, since each reward is still an honest draw. It does. Flip a fair coin
and stop as soon as heads lead tails. The average at the moment of stopping is
always above one half, and stopping is likely: it happens within 100 flips with
probability 0.92, and within 1000 flips with probability 0.97. For every fixed
$n$, Hoeffding's inequality still holds for the average of the first $n$ flips;
it says nothing about the average at an $n$ chosen by looking at the flips.

The bandit analyses repair this with the union bound. Picture each arm's
rewards as a list drawn before play begins, with the algorithm choosing only
how far down each list to read; this model gives the same probabilities to
everything the algorithm sees [@lattimore2020bandit, sec. 4.6]. For each fixed
$n$, the first $n$ entries of a list are $n$ independent rewards, so
Hoeffding's inequality applies to every $n$ separately, and a union over
$n = 1, \dots, t$ covers whichever count the algorithm reaches. That union is
the $t^2$ above. It is not a formality. If the single-round width
$\sqrt{\ln(1/\delta)/(2n)}$ is used at every count, the interval of a fair coin
with $\delta = 0.05$ fails at least once within 1000 pulls with probability
0.11, more than twice $\delta$, and within 10,000 pulls with probability 0.15,
because the largest swings of a running average shrink slightly more slowly
than $1/\sqrt{n}$ [see @lattimore2020bandit, ex. 20.9]. With the width of the
uneven split above, the same probability within 1000 pulls is $8.5 \times
10^{-6}$: the union bound is safe, and here very cautious. (These
probabilities, like those for the fair coin above, are computed exactly, by
tracking the distribution of the number of heads flip by flip.) A Gaussian process is harder still, because each
evaluation changes the estimate everywhere, with weights that depend on where
the algorithm chose to look; @sec-regret-self-normalized gives the tool for
that case.

The figure puts the three inequalities side by side for a coin.

```{figure}
//| figure: regret-concentration
//| label: fig-regret-concentration
//| fig-cap: "The probability that the average of $n$ flips of a coin with mean $\theta$ is at least $\theta + a$ (solid, computed exactly from the binomial distribution), against the bounds of Markov's inequality, $\theta/(\theta + a)$, which ignores $n$; Chebyshev's inequality with the coin's variance, $\theta(1 - \theta)/(na^2)$; and Hoeffding's inequality, @eq-regret-hoeffding. Both axes are logarithmic, and the dotted line marks 0.05. The readout gives each value at the marked $n$ and the $n$ from which each curve stays below 0.05. With the union set above 1, every curve is for $T$ averages that must all stay below $\theta + a$, such as one per round or one per arm: each bound is multiplied by $T$ (@eq-regret-union), and the exact curve becomes the probability that at least one of $T$ independent averages exceeds the line. The exact probability zigzags with $n$ because the number of heads is a whole number; where several values of $n$ share a pixel, the curve shows the largest."
```

Some things to try.

**Read the default.** For a fair coin and $a = 0.1$, the chance that 100 flips
average 0.6 or more is 0.028. Hoeffding's inequality bounds it by 0.14,
Chebyshev's by 0.25, and Markov's by 0.83. The readout gives the number of
flips from which each curve stays below 0.05: 76 for the exact probability,
150 for Hoeffding's bound, 500 for Chebyshev's, and never for Markov's.

**Move the marker to $n = 1000$.** The exact probability is
$1.4 \times 10^{-10}$ and Hoeffding's bound $2.1 \times 10^{-9}$, while
Chebyshev's is still 0.025. The exact curve and Hoeffding's fall at nearly the
same exponential rate, and the gap between them grows only slowly, from a
factor of about 5 at $n = 100$ to 15 at $n = 1000$; Chebyshev's is a straight
line on these axes, falling only like $1/n$.

**Set the union to 1000.** Every bound is multiplied by 1000. Hoeffding's now
stays below 0.05 from 496 flips instead of 150, an extra
$\ln 1000/(2a^2) \approx 345$; Chebyshev's from 500,000 instead of 500, a
thousand times as many. The exact probability for 1000 independent averages
needs 381 flips. This is the $\ln m$ of @eq-regret-union-width, seen from the
side of the sample size.

**Move the mean to 0.1**, with the union back at 1. A coin that rarely pays
has variance 0.09 instead of 0.25, and its exact probability stays below 0.05
from 36 flips. Chebyshev's bound, which uses the variance, improves to 180
flips; Hoeffding's, which knows only that rewards lie in $[0, 1]$, stays at
150, and at $n = 100$ it gives 0.14 against an exact 0.002. Inequalities that
use the variance as well, such as Bernstein's, recover much of this gap
[@lattimore2020bandit, ex. 5.14].

::: {.keyidea title="Two prices in every confidence width"}
A confidence width pays for two things: how fast the tail of an average falls,
which for bounded rewards Hoeffding's inequality bounds by $e^{-2na^2}$, and
how many intervals must hold at once, which the union bound charges as an
additive logarithm. UCB1's bonus, $\sqrt{2 \ln t / N_i}$, is both prices
together.
:::

### Optimism: UCB1 {#sec-regret-ucb1}

A better rule follows from the principle that @sec-ucb called optimism in the
face of uncertainty. For each arm, compute the largest mean that is still plausible
given its rewards so far, and pull the arm whose plausible best is largest. An
arm pulled often has a tight interval, so its plausible best is close to its
average. An arm pulled rarely has a wide interval and a generous plausible
best, so it gets another look. A clearly bad arm stops being pulled once its
interval has shrunk below the best arm's mean.

How wide should the interval be? For rewards in $[0, 1]$, Hoeffding's
inequality (@eq-regret-hoeffding) says that the average $\hat\theta_{i}$ of $n$
independent rewards overestimates the mean by more than $a$ with probability at
most $e^{-2na^2}$, and likewise underestimates it. Choosing the width so that
this probability is $t^{-4}$ in round $t$, for the reason given in
@sec-regret-concentration, gives the *UCB1* rule of @auer2002finite: after
pulling each arm once, pull

$$
a_t = \argmax_{i} \left( \hat\theta_i + \sqrt{\frac{2 \ln t}{N_i}} \right),
$$ {#eq-regret-ucb1}

where $N_i$ is the number of times arm $i$ has been pulled so far and
$\hat\theta_i$ is its average reward. The bonus shrinks like $1/\sqrt{N_i}$ as
an arm is pulled and grows like $\sqrt{\ln t}$ for arms left alone, so no arm is
abandoned for good, yet a bad arm is revisited only rarely.

::: {.theorem #thm-regret-ucb1 title="UCB1 (Auer, Cesa-Bianchi, and Fischer, 2002)"}
For any $K > 1$ arms with reward distributions supported in $[0, 1]$, the
expected regret of UCB1 after any number $T$ of rounds is at most
$$
8 \sum_{i:\,\Delta_i > 0} \frac{\ln T}{\Delta_i}
\;+\; \left(1 + \frac{\pi^2}{3}\right) \sum_{j=1}^{K} \Delta_j .
$$
:::

The proof is worth seeing in outline, because the same three moves reappear in
the Gaussian process bound of @sec-gp-bandits: a confidence interval that
holds with high probability, an argument that optimism costs at most the width
of the interval, and a count of how often intervals can be wide.

::: {.derivation title="Why a bad arm is pulled about ln T / Δ² times"}
Fix a suboptimal arm $i$ and write $a(n, t) = \sqrt{2 \ln t / n}$ for the bonus
of an arm pulled $n$ times by round $t$. This is a sketch of the proof of
Theorem 1 in @auer2002finite.

1. By Hoeffding's inequality, an average of $n$ rewards misses its mean by more
   than $a(n, t)$ in a given direction with probability at most
   $e^{-2n \cdot 2\ln t / n} = t^{-4}$.
2. Suppose arm $i$ has already been pulled $n \ge 8 \ln T / \Delta_i^2$ times.
   Then $a(n, t) \le \Delta_i / 2$ for every $t \le T$, by solving the
   inequality for $n$.
3. If neither interval fails, the best arm's index is at least $\theta^*$, and
   arm $i$'s index is at most $\theta_i + 2a(n, t) \le \theta_i + \Delta_i =
   \theta^*$. So arm $i$ cannot win the comparison, and it is pulled only when
   one of the two intervals has failed.
4. In round $t$ each arm can have been pulled any number of times up to $t$.
   By the union bound (@eq-regret-union), adding the failure
   probability $t^{-4}$ of step 1 over both arms' possible counts gives at
   most $2t^2 \cdot t^{-4} = 2t^{-2}$, and summing over all rounds gives at
   most $2\sum_t t^{-2} = \pi^2/3$ extra pulls in expectation.
5. Together, $\E[N_i(T)] \le 8 \ln T/\Delta_i^2 + 1 + \pi^2/3$. Multiplying by
   $\Delta_i$ and summing over arms, as @eq-regret-decomposition says, gives the
   theorem.
:::

The bound depends on the problem through the gaps. Arms that are much worse
than the best are discarded quickly and contribute little; arms that are
nearly as good contribute $\ln T / \Delta_i$ each, which is large for small
$\Delta_i$. The upper confidence bound rule of @sec-ucb is the same idea with the
posterior standard deviation of a Gaussian process in place of the Hoeffding
width.

### Thompson sampling {#sec-regret-thompson}

The oldest rule is Bayesian. Treat each unknown mean $\theta_i$ as a random
quantity with a prior, keep its posterior up to date, and in each round pull
each arm with the probability that it is the best one. Thompson's trick is that
this probability never has to be computed: draw one plausible mean from each
arm's posterior, and pull the arm whose draw is largest
[@thompson1933likelihood].

For coin-flip rewards the posterior is a Beta distribution. Starting from a
uniform prior, an arm with $S_i$ wins and $F_i$ losses has posterior
$\mathrm{Beta}(1 + S_i, 1 + F_i)$, the conjugate update of @sec-beta-binomial.
Its mean is close to the arm's average, and its spread shrinks as the arm is
pulled.

::: {.algorithm #alg-regret-thompson title="Thompson sampling for Bernoulli arms"}
Input: $K$ arms, horizon $T$.

1. Set $S_i \leftarrow 0$ and $F_i \leftarrow 0$ for every arm.
2. In each round $t = 1, \dots, T$, draw $\tilde\theta_i \sim
   \mathrm{Beta}(1 + S_i, 1 + F_i)$ independently for each arm.
3. Pull $a_t = \argmax_i \tilde\theta_i$ and observe the reward $y_t \in \{0, 1\}$.
4. If $y_t = 1$, increase $S_{a_t}$ by one; otherwise increase $F_{a_t}$ by one.
:::

Exploration comes from the randomness of the draws. An arm with few pulls has a
wide posterior and sometimes produces a high draw; an arm with many pulls and a
low average almost never does. Thompson's rule was not widely circulated, and
it became popular only after several groups rediscovered it around 2010 and
found it strong in experiments, before any proof existed
[@lattimore2020bandit, ch. 36]. @agrawal2012analysis then gave the first proof
that its expected regret grows logarithmically, and @kaufmann2012thompson and
@agrawal2013further showed that for Bernoulli rewards its leading constant is
the best possible, the constant of the lower bound in the next section. The
Gaussian process version of the rule, which draws a whole function from the
posterior and evaluates where the draw is largest, is the Thompson sampling of
@sec-thompson.

### Watching them play {#sec-regret-bandit-race}

The figure below runs the three algorithms on the same arms many times and
plots the average cumulative regret of each, with a band covering the middle
80% of runs. All three start from the same payout sequences in each run, so the
differences come from the rules and not from luck.

```{figure}
//| figure: regret-bandits
//| label: fig-regret-bandits
//| fig-cap: "Cumulative regret of ε-greedy (constant ε), UCB1, and Thompson sampling on Bernoulli arms whose means are listed in the table, averaged over 20 runs, with bands from the 10th to the 90th percentile of runs. The table shows where each algorithm spent its pulls, as a share of all rounds. The dashed curve is the Lai-Robbins rate $c^* \ln t$ of @sec-regret-lower-bounds, an asymptotic statement about growth, not a floor at every $t$. The arm means are illustrative."
```

Some things to try.

**Switch on the log time axis.** On a logarithmic time axis, regret that grows
like $\ln t$ is a straight line, and regret that grows linearly curves sharply
upward. Thompson sampling settles onto a straight line after a few hundred
rounds, and ε-greedy turns upward. UCB1 is still in between at this horizon:
with a gap of 0.1, its cautious bonus keeps the second-best arm in play for
thousands of rounds, and its curve straightens only later.

**Set ε to 0.** This is the greedy rule. The band widens: most runs settle on
the best arm, and a few lock onto a worse arm and stay there. Set the runs to 1
and press *New arms* a few times to see individual outcomes.

**Compare UCB1 and ε-greedy at the default horizon.** With these arms,
ε-greedy with $\varepsilon = 0.1$ has lower regret than UCB1 after 1000 rounds:
UCB1's bonus is conservative and buys its guarantee with extra exploration.
Raise the gap to 0.3 and the horizon to 5000. The straight line of ε-greedy
eventually overtakes the logarithm of UCB1.

**Show the UCB1 guarantee.** The scale stretches to hold it. At the default
settings, the bound of @thm-regret-ucb1 is about 1100 at $T = 1000$, more than
twice the 450 that the worst possible play, always pulling the worst arm,
would lose. The guarantee is true and, at this horizon, uninformative.

**Watch Thompson sampling.** It has the lowest regret here, and for most
settings its curve sits below the dashed Lai-Robbins line. The next section
explains why this is not a contradiction.

## Lower bounds {#sec-regret-lower-bounds}

UCB1 and Thompson sampling both have regret that grows like $\ln T$. Could a
cleverer algorithm do better, with regret bounded by a constant?
@lai1985asymptotically answered no.

The intuition is about evidence. To stop pulling a worse arm, an algorithm must
be confident that the arm is not secretly the best. Abandoning the best arm by
mistake costs on the order of $\Delta T$ over the remaining rounds, so the
probability of that mistake must be of order $1/T$. The amount of evidence
needed to rule out an alternative at that level grows like $\ln T$, and every
pull of arm $i$ adds a fixed expected amount of evidence: the
*Kullback-Leibler divergence* $\mathrm{kl}(\theta_i, \theta^*)$, the average
log-likelihood ratio per pull between the hypothesis that the arm pays at rate
$\theta_i$ and the hypothesis that it pays at rate $\theta^*$. For coins,

$$
\mathrm{kl}(p, q) = p \ln\frac{p}{q} + (1 - p) \ln\frac{1 - p}{1 - q},
$$

which is zero when $p = q$ and grows as the two coins become easier to tell
apart (@sec-kl treats the divergence in general). Dividing the evidence needed
by the evidence per pull gives about $\ln T / \mathrm{kl}(\theta_i, \theta^*)$
pulls of arm $i$.

To make this a theorem, one has to exclude algorithms that are lucky on one
problem by being terrible on others, such as the rule that always pulls arm 1,
which has zero regret whenever arm 1 is best. An algorithm is *consistent* if
on every bandit in the class its regret grows more slowly than every power of
$T$: $\E[R_T] / T^a \to 0$ for every $a > 0$.

::: {.theorem #thm-regret-lai-robbins title="Lai and Robbins (1985), for Bernoulli arms"}
For every consistent algorithm and every Bernoulli bandit with $\theta^* < 1$,
$$
\liminf_{T \to \infty} \frac{\E[R_T]}{\ln T} \;\ge\; c^*
\;=\; \sum_{i:\,\Delta_i > 0} \frac{\Delta_i}{\mathrm{kl}(\theta_i, \theta^*)}.
$$ {#eq-regret-lai-robbins}
:::

This is the special case of a general result for reward distributions in a
parametric family; Lattimore and Szepesvári state and prove the modern form
[@lai1985asymptotically; @lattimore2020bandit, Thm. 16.2].

Three remarks connect the theorem to the algorithms above. First, UCB1 is
logarithmic but not optimal. Pinsker's inequality gives
$\mathrm{kl}(p, q) \ge 2(p - q)^2$, so each term of $c^*$ is at most
$1/(2\Delta_i)$, while the corresponding term of @thm-regret-ucb1 is
$8/\Delta_i$, at least 16 times larger. Thompson sampling with Beta posteriors
attains $c^*$ exactly, in the limit [@kaufmann2012thompson; @agrawal2013further].

Second, the theorem is about the limit, and the $\liminf$ is doing real work.
It says that the ratio of regret to $\ln T$ cannot stay below $c^*$ forever. It
does not say that $\E[R_T] \ge c^* \ln T$ at every finite $T$, and an algorithm
whose lower-order terms are negative can sit below that curve for a long time.
That is what Thompson sampling does in @fig-regret-bandits. The comparison the
theorem licenses is between slopes on a logarithmic time axis, as $t$ grows
large.

Third, the bound depends on the instance through its gaps, and it blows up as
a gap shrinks. That does not mean regret becomes large when arms are nearly
equal, since each pull of a nearly equal arm costs little. The *worst case*
over all instances is a different quantity: for any algorithm and any $T \ge K
- 1$ there is a $K$-armed bandit with Gaussian rewards on which its regret is
at least $\frac{1}{27}\sqrt{(K - 1)T}$ [@lattimore2020bandit, Thm. 15.2]. The
gap that does the damage shrinks with $T$, like $\sqrt{K/T}$: large enough to
matter, small enough to be hard to detect. Instance-dependent bounds grow like
$\ln T$ with a problem-dependent constant; worst-case bounds grow like
$\sqrt{T}$. Both views return for functions.

## From arms to functions {#sec-gp-bandits}

Bayesian optimization has an arm for every input, infinitely many on a
continuous domain. The bandit bounds above grow with the number of arms, either
through the sum over gaps or through the $\sqrt{K}$ of the worst case, and they
say nothing when $K$ is infinite. What rescues the analysis is that the arms
are no longer independent. A Gaussian process prior says that nearby inputs
have similar values, so evaluating one input teaches about its neighbors. The
number of arms has to be replaced by a measure of how many *effectively
different* arms there are, and the maximum information gain is that measure.

### The setting {#sec-regret-gp-setting}

The analysis of @srinivas2010gaussian, which this section follows, takes the
model of @sec-gp-noise at face value. The function is a draw from a Gaussian
process, $f \sim \GP(0, k)$, with $k(\vx, \vx) \le 1$ so that the prior
standard deviation is at most 1 everywhere. Each evaluation returns
$y_t = f(\vx_t) + \varepsilon_t$ with independent noise
$\varepsilon_t \sim \N(0, \sigma_n^2)$ of known variance. For now the domain
$\X$ is finite, for instance a fine grid (the paper writes $D$ for this set);
@sec-regret-other-settings relaxes this. A $K$-armed bandit with Gaussian
rewards is the special case of a kernel that is 1 on the diagonal and 0
elsewhere.

After $t - 1$ evaluations, the posterior has mean $\mu_{t-1}(\vx)$ and
standard deviation $\sigma_{t-1}(\vx)$, computed as in @eq-gp-noisy. The
*GP-UCB* rule is the optimism of UCB1 with the posterior in place of the
Hoeffding interval:

$$
\vx_t = \argmax_{\vx \in \X} \; \mu_{t-1}(\vx) + \beta_t^{1/2}\, \sigma_{t-1}(\vx).
$$ {#eq-regret-gpucb}

The posterior mean plays the part of the empirical average, the posterior
standard deviation plays the part of the bonus, and $\beta_t$ sets how many
standard deviations of optimism to allow. This is the rule of @eq-loop-ucb with
a weight that may change from round to round. As there and in
@srinivas2010gaussian, $\beta_t$ multiplies the variance, so its square root
multiplies the standard deviation; some texts and libraries call the
multiplier of the standard deviation itself $\beta$.

### Maximum information gain {#sec-regret-info-gain}

How much can $T$ noisy evaluations teach about $f$? @sec-gp-information-gain
answered this question, and the bound needs three facts from there. First, the
*mutual information* between the observations $\vy_A$ at a set of inputs $A$
and the function, the part of the observations' uncertainty that reflects $f$
and not the noise, is

$$
I(\vy_A; f) = \tfrac12 \log\det\!\left(\mI + \sigma_n^{-2} \mK_A\right),
$$ {#eq-regret-info}

where $\mK_A$ is the kernel matrix of $A$ (@eq-info-gp-gain). It depends on
where we evaluate, not on what we observe, because the posterior variance of a
Gaussian process does not depend on the observed values
(@sec-gp-conditioning). Second, the most that any $T$ evaluations could teach
is the largest value this quantity can take.

::: {.definition #def-regret-gamma title="Maximum information gain"}
The *maximum information gain* after $T$ evaluations is
$$
\gamma_T = \max_{A \subset \X,\; |A| = T} \; \tfrac12 \log\det\!\left(\mI + \sigma_n^{-2} \mK_A\right).
$$ {#eq-regret-gamma}
:::

$\gamma_T$ is a property of the kernel, the domain, and the noise level, fixed
before any data arrive.

Two extreme cases show its range. If all $T$ evaluations are made at the same
input, they gain $\tfrac12 \log(1 + T/\sigma_n^2)$, which grows only
logarithmically: repeating an evaluation teaches less and less, as
@exr-noise-floor showed. If the kernel is diagonal, as in a $K$-armed bandit,
spreading the evaluations evenly over the arms is best, and $\gamma_T$ grows
like $\tfrac{K}{2}\log(1 + T/(K\sigma_n^2))$ (@exr-regret-independent). A
smooth kernel lies between these: observations at nearby inputs are largely
redundant, so the information grows much more slowly than for $T$ independent
arms. A short lengthscale, a rough kernel, a high dimension, or low noise each
make more of the domain distinguishable and increase $\gamma_T$.

Third, although computing the maximum exactly is intractable, it is easy to
approximate. A new evaluation at $\vx$ adds exactly
$\tfrac12 \log(1 + \sigma_n^{-2}\sigma_{t-1}^2(\vx))$ to the information
(@eq-info-gain-chain), so the greedy rule of always evaluating where the
posterior variance is largest (uncertainty sampling, @sec-info-sequential)
reaches at least a fraction $1 - 1/e \approx 0.63$ of $\gamma_T$
[@srinivas2010gaussian]. That is how the figure in @sec-bounds-and-practice
estimates $\gamma_T$.

### The GP-UCB bound {#sec-regret-gpucb-bound}

With $\gamma_T$ in hand, the bound reads like a bandit bound with $K$ replaced.

::: {.theorem #thm-regret-gpucb title="GP-UCB on a finite domain (Srinivas, Krause, Kakade, and Seeger, 2010)"}
Let $\X$ be finite, $\delta \in (0, 1)$, and
$$
\beta_t = 2 \log\!\left(\frac{|\X|\, t^2 \pi^2}{6\delta}\right).
$$
If $f$ is a draw from $\GP(0, k)$ with $k(\vx, \vx) \le 1$ and the noise is
$\N(0, \sigma_n^2)$, GP-UCB with this $\beta_t$ satisfies, with probability at
least $1 - \delta$,
$$
R_T \le \sqrt{C_1\, T\, \beta_T\, \gamma_T}
\quad \text{for all } T \ge 1,
\qquad C_1 = \frac{8}{\log(1 + \sigma_n^{-2})}.
$$ {#eq-regret-gpucb-bound}
:::

The proof follows the same three moves as the UCB1 sketch, and every step is
elementary.

::: {.derivation title="Where the square root comes from"}
These are Lemmas 5.1 to 5.4 of the extended version of @srinivas2010gaussian.

1. *Confidence.* Given the data, $f(\vx)$ is Gaussian with mean $\mu_{t-1}(\vx)$
   and standard deviation $\sigma_{t-1}(\vx)$, and a Gaussian lands more than
   $\beta^{1/2}$ standard deviations from its mean with probability at most
   $e^{-\beta/2}$ (@sec-regret-concentration). A union bound over the $|\X|$ inputs and over all rounds,
   with $\beta_t$ as in the theorem, makes
   $|f(\vx) - \mu_{t-1}(\vx)| \le \beta_t^{1/2}\sigma_{t-1}(\vx)$ hold for every
   $\vx$ and $t$ at once, with probability at least $1 - \delta$. The $\pi^2/6$
   is $\sum_t 1/t^2$, which spreads $\delta$ over the rounds.
2. *Optimism costs at most twice the width.* On that event, since $\vx_t$
   maximizes the upper bound,
   $\mu_{t-1}(\vx_t) + \beta_t^{1/2}\sigma_{t-1}(\vx_t) \ge
   \mu_{t-1}(\vx^\star) + \beta_t^{1/2}\sigma_{t-1}(\vx^\star) \ge f(\vx^\star)$. Subtracting
   $f(\vx_t) \ge \mu_{t-1}(\vx_t) - \beta_t^{1/2}\sigma_{t-1}(\vx_t)$ gives
   $r_t \le 2\beta_t^{1/2}\sigma_{t-1}(\vx_t)$.
3. *The run's information.* By @eq-info-gain-chain, the information gained by
   the inputs GP-UCB actually chose is a sum over rounds,
   $I(\vy_T; f) = \tfrac12 \sum_{t} \log\!\left(1 + \sigma_n^{-2}\sigma_{t-1}^2(\vx_t)\right)$.
   This is at most $\gamma_T$, the best any $T$ inputs could do.
4. *Variance into information.* Write $s^2 = \sigma_n^{-2}\sigma_{t-1}^2(\vx_t)$,
   which lies in $[0, \sigma_n^{-2}]$ because
   $\sigma_{t-1}^2(\vx_t) \le k(\vx_t, \vx_t) \le 1$.
   On that interval the concave function $\log(1 + s^2)$ lies above its chord,
   so $s^2 \le C_2 \log(1 + s^2)$ with $C_2 = \sigma_n^{-2}/\log(1 + \sigma_n^{-2})$.
   Squaring step 2 and using $\beta_t \le \beta_T$,
   $r_t^2 \le 4\beta_T \sigma_n^2 s^2 \le C_1 \beta_T \cdot \tfrac12\log(1 + s^2)$,
   with $C_1 = 8\sigma_n^2 C_2 = 8/\log(1 + \sigma_n^{-2})$. Summing over rounds
   and using step 3, $\sum_t r_t^2 \le C_1 \beta_T \gamma_T$.
5. *Cauchy-Schwarz.* $R_T^2 = \left(\sum_t r_t\right)^2 \le T \sum_t r_t^2 \le
   C_1 T \beta_T \gamma_T$. Taking square roots gives @eq-regret-gpucb-bound.
:::

Reading the result: $\beta_T$ grows like $\log T$ (and $\log |\X|$), and
$\gamma_T$ grows sublinearly for the kernels used in practice, so $R_T$ grows
like $\sqrt{T}$ times slowly growing factors. GP-UCB is therefore no-regret,
and by @eq-regret-simple-from-cumulative, with probability at least
$1 - \delta$, the best input it has evaluated is within
$\sqrt{C_1 \beta_T \gamma_T / T}$ of the maximum.

A finite domain is not only a mathematical convenience. In the experiments of
@srinivas2010gaussian, the inputs were the 46 temperature sensors of a sensor
network at Intel Research Berkeley, and in a second test the 357 traffic
sensors along a stretch of the I-880 highway in California, where the goal was
to find the most congested point. The kernel matrix was not a formula: it was
the empirical covariance of the sensors' readings over the first two thirds of
the recorded data, and the functions to optimize were snapshots from the
remaining third. On the temperature data GP-UCB and expected improvement
clearly outperformed the other heuristics, with no significant difference
between the two; the authors summarize that GP-UCB performed at least on par
with existing approaches that had no regret bounds. The bound also
connects two traditions. Step 4 says that a GP-UCB step can be expensive only
when it is informative, so an optimizer's regret is controlled by how much
there is to learn about $f$, which is the currency of experimental design
(@sec-expected-information-gain).

How fast $\gamma_T$ grows decides how good the bound is. @tbl-regret-gamma
collects the known rates for a domain in $d$ dimensions, which follow from how
fast the eigenvalues of the kernel decay (@sec-ka-infogain); $\nu$ is the
smoothness parameter of the Matérn kernel (@sec-kernel-family).

::: {.table #tbl-regret-gamma title="How the maximum information gain grows with the number of evaluations T, on a closed and bounded (compact) domain in d dimensions"}
| Kernel | $\gamma_T$ | Source |
|---|---|---|
| Linear | $O(d \log T)$ | @srinivas2010gaussian |
| RBF (squared exponential) | $O\big((\log T)^{d+1}\big)$ | @srinivas2010gaussian |
| Matérn, $\nu > 1$ | $O\big(T^{d(d+1)/(2\nu + d(d+1))} \log T\big)$ | @srinivas2010gaussian |
| Matérn, $\nu > 1/2$ | $O\big(T^{d/(2\nu + d)} (\log T)^{2\nu/(2\nu + d)}\big)$ | @vakili2021information |
:::

For the RBF kernel, the dimension appears only as the exponent of $\log T$, so
the bound grows like $\sqrt{T}(\log T)^{(d+2)/2}$ (a factor
$(\log T)^{(d+1)/2}$ from $\gamma_T$ and one more $(\log T)^{1/2}$ from
$\beta_T$): very smooth functions are learned quickly even in several
dimensions [@srinivas2010gaussian]. For Matérn kernels the original rate was loose; the
2021 rate of @vakili2021information matches the known lower bounds up to
logarithmic factors.

### Other settings {#sec-regret-other-settings}

The finite-domain theorem extends in three directions, each with its own
assumptions. This subsection is a map of results for readers who will meet
them in papers. It can be skipped on a first reading; the one idea used later,
the reproducing kernel Hilbert space, is explained again in
@sec-kernelized-dueling, and @sec-kernel-analysis treats it in depth.

**Continuous domains.** For a compact, convex domain in $d$ dimensions, such
as a box, @srinivas2010gaussian prove a bound of the same form, with $\beta_t$ gaining a
term of order $d \log t$ and the bound gaining an additive constant. The proof
discretizes the domain more finely as $t$ grows, which requires sample paths
smooth enough that values at nearby grid points are close. The RBF kernel and
Matérn kernels with $\nu > 2$ qualify. The rough Matérn
1/2 kernel violates the assumption, and the authors conjecture that no result
of this form holds for it.

**Fixed functions.** The theorems above are Bayesian: they hold with high
probability for functions drawn from the prior. A frequentist version asks
for a guarantee for one fixed function from a class. The natural class is the
*reproducing kernel Hilbert space* (RKHS) of the kernel, a space of functions
built from sums of kernel bumps like @eq-gp-representer, whose norm
$\lVert f \rVert_k$ measures how rough $f$ is relative to the kernel
(@sec-ka-rkhs constructs the space and its norm). (Sample
paths of the Gaussian process itself are rougher than this, with infinite
norm, so neither setting contains the other.) If $\lVert f \rVert_k^2 \le B$
and the noise is bounded, GP-UCB with
$\beta_t = 2B + 300\gamma_t \log^3(t/\delta)$ has regret of order
$\sqrt{T}(\sqrt{B\gamma_T} + \gamma_T)$ up to logarithmic factors
[@srinivas2010gaussian]. @chowdhury2017kernelized sharpened this analysis and
proved a regret bound for a Gaussian process version of Thompson sampling.
@sec-regret-self-normalized shows where widths of this kind come from.

**Lower bounds.** In the RKHS setting, @scarlett2017lower proved that every
algorithm has cumulative regret of at least order $T^{(\nu + d)/(2\nu + d)}$ on
some function in the Matérn class. For the RBF kernel they showed that
cumulative regret is at least of order $\sqrt{T(\log T)^{d/2}}$, which matches
the upper bounds up to replacing $d/2$ by $2d + O(1)$ in the exponent of
$\log T$ under the square root. The lower bounds play the role of Lai and Robbins for
functions: they say how much exploring the class forces on any algorithm.

The same machinery, with comparisons in place of evaluations, gives the
kernelized dueling bandit bounds of @sec-kernelized-dueling and the theory of
@sec-pbo-theory, where some of the basic lower bounds are still missing
(@sec-theory-lower).

### Confidence for a fixed function {#sec-regret-self-normalized}

The confidence step of @thm-regret-gpucb used two tools from
@sec-regret-concentration. A Gaussian lands more than $\beta^{1/2}$ standard
deviations from its mean with probability at most $e^{-\beta/2}$, and a union
bound over the $|\X|$ inputs and over rounds, giving round $t$ the share
$6\delta/(\pi^2 t^2)$ of $\delta$, asks for
$|\X|\, e^{-\beta_t/2} = 6\delta/(\pi^2 t^2)$. Solving for $\beta_t$ gives the
theorem's $\beta_t = 2 \log\!\big(|\X|\, t^2 \pi^2/(6\delta)\big)$. The
adaptive choice of inputs did no harm there: given the observations so far,
the inputs chosen from them are fixed, and $f(\vx)$ is Gaussian with mean
$\mu_{t-1}(\vx)$ and standard deviation $\sigma_{t-1}(\vx)$ whatever rule chose
them [@srinivas2010gaussian, Lemma 5.1].

The frequentist results of @sec-regret-other-settings remove that support.
There $f$ is one fixed function with $\lVert f \rVert_k^2 \le B$, and the only
randomness is the noise. The error $\mu_t(\vx) - f(\vx)$ is a bias, from the
prior pulling the estimate toward zero, plus a weighted sum of the noise terms
$\varepsilon_1, \dots, \varepsilon_t$, and the weights depend on where the
algorithm chose to look, which depended on earlier noise. That is not a sum of
independent terms with weights fixed in advance, so Hoeffding's inequality
does not apply, and on a continuous domain there is no finite list of inputs
to take a union over. This subsection shows the tool that replaces both and
how it produces a width that grows with $\gamma_t$ and $\log(1/\delta)$. Like
the map above, it can be skipped on a first reading.

**Martingales.** Consider a running sum $M_t = \sum_{s \le t} g_s
\varepsilon_s$ in which each weight $g_s$ may depend on everything observed
before round $s$ but is fixed before $\varepsilon_s$ is drawn, and each
$\varepsilon_s$ has mean zero given everything before it. Such a sum is a
*martingale*: the fortune of a gambler in a fair game who chooses each stake by
looking at the history. The stakes are adaptive, and the game is still fair.
The Chernoff method survives the adaptivity. Given the past, $g_t$ is a fixed
number and $\varepsilon_t$ is $R$-sub-Gaussian (@def-regret-subgaussian), so
$\E\big[e^{\lambda g_t \varepsilon_t} \given \text{past}\big] \le e^{\lambda^2 g_t^2 R^2/2}$.
Peeling off one round at a time, starting from the last, shows that

$$
Z_t = \exp\!\Big(\lambda M_t - \tfrac12 \lambda^2 R^2 V_t\Big),
\qquad
V_t = \sum_{s \le t} g_s^2,
$$

has expectation at most 1 for every $t$. Where the derivation of
@eq-regret-hoeffding factored an expectation over independent terms, this one
factors it over rounds, each conditioned on the rounds before. More is true:
$Z_t$ is never negative and does not drift upward on average from one round to
the next, and for such a process Markov's inequality holds in a stronger form,
the *maximal inequality*: the probability that $Z_t$ *ever* reaches $1/\delta$
is at most $\delta$ [@lattimore2020bandit, Thm. 3.9]. A bound for all rounds at
once then needs no union over rounds.

Two problems remain. The best $\lambda$ depends on $V_t$, which is random. And
a Gaussian process estimate is not of the form $M_t$: its weight on the
observation $y_s$ depends on inputs chosen after round $s$. The first problem
is solved by averaging $Z_t$ over $\lambda$ instead of choosing it, the *method
of mixtures*.

::: {.derivation title="The method of mixtures in one dimension"}
1. For each fixed $\lambda$, $Z_t$ is never negative, starts at 1, and does not
   drift upward. An average of such processes over $\lambda$ is another one
   [@lattimore2020bandit, Lemma 20.3].
2. Average over $\lambda \sim \N\big(0, 1/(cR^2)\big)$ for a constant $c > 0$.
   The density of $\lambda$ is $\sqrt{cR^2/(2\pi)}\,e^{-cR^2\lambda^2/2}$, so
   $\bar Z_t = \sqrt{cR^2/(2\pi)}\int \exp\!\big(\lambda M_t - \tfrac12\lambda^2R^2(V_t + c)\big)\,\dd\lambda$.
   Completing the square in $\lambda$, as in @sec-gaussian-1d, leaves
   $\exp\!\big(M_t^2 / (2R^2(V_t + c))\big)$ times a Gaussian integral equal to
   $\sqrt{2\pi/(R^2(V_t + c))}$, so
   $\bar Z_t = \sqrt{c/(V_t + c)}\, \exp\!\big(M_t^2 / (2R^2(V_t + c))\big)$.
3. By the maximal inequality, with probability at least $1 - \delta$,
   $\bar Z_t < 1/\delta$ for every $t$. Taking logarithms and rearranging,
   $M_t^2 < R^2 (V_t + c) \big(2 \log(1/\delta) + \log(1 + V_t/c)\big)$ for
   every $t$.
:::

The result reads as a statement in standard deviations. $R^2 V_t$ plays the
role of the variance of $M_t$, so the sum stays within about one standard deviation,
$R\sqrt{V_t + c}$, times $\sqrt{2\log(1/\delta) + \log(1 + V_t/c)}$. The
$2\log(1/\delta)$ is the price of confidence that every Chernoff bound pays.
The $\log(1 + V_t/c)$ is the price of not knowing in advance how large the
variance would be, and it grows only like the logarithm of the variance. A
bound of this kind is called *self-normalized*: the sum is measured against
its own accumulated variance.

The second problem is solved by working with vectors. Write the kernel through
features, $k(\vx, \vx') = \boldsymbol{\phi}(\vx)^\T\boldsymbol{\phi}(\vx')$, the
weight-space view of @sec-bayes-blr-posterior with the prior covariance of the
weights set to $\mI$. Two quantities summarize the first $t$ rounds: the
posterior precision of the weights, and the noise pushed along the features of
the inputs that received it,

$$
\mA_t = \mI + \sigma_n^{-2} \sum_{s \le t} \boldsymbol{\phi}(\vx_s)\boldsymbol{\phi}(\vx_s)^\T,
\qquad
\mathbf{s}_t = \sum_{s \le t} \varepsilon_s\, \boldsymbol{\phi}(\vx_s).
$$

Each term of $\mathbf{s}_t$ has its weight vector $\boldsymbol{\phi}(\vx_s)$
fixed before its noise is drawn, so $\mathbf{s}_t$ is a martingale with vector
values. Averaging over a Gaussian distribution of directions, in place of the
Gaussian distribution of $\lambda$ in the box above, gives the following bound.

::: {.theorem #thm-regret-self-normalized title="Self-normalized bound (Abbasi-Yadkori, Pál, and Szepesvári, 2011)"}
Let the features have finitely many entries. Suppose each input $\vx_s$ is
chosen from the observations before round $s$, and each noise term
$\varepsilon_s$, given everything before it, is $R$-sub-Gaussian. Then for any
$\delta \in (0, 1)$, with probability at least $1 - \delta$,
$$
\mathbf{s}_t^\T \mA_t^{-1} \mathbf{s}_t \le \sigma_n^2 R^2 \big(\log\det\mA_t + 2\log(1/\delta)\big)
\quad \text{for all } t \ge 0 \text{ at once.}
$$ {#eq-regret-self-normalized}
:::

This is Theorem 1 of @abbasiyadkori2011improved with their regularizer set to
$\sigma_n^2$, so that their matrix is $\sigma_n^2 \mA_t$. Lattimore and
Szepesvári prove the case $R = 1$, to which any $R$ reduces by rescaling the
noise, with the method of mixtures [@lattimore2020bandit, Thm. 20.4]. With a
single feature the theorem is the box above with $c = \sigma_n^2$.

The log-determinant is the information gain. By the matrix determinant lemma
(@eq-id-det-lemma), $\det \mA_t = \det(\mI + \sigma_n^{-2}\mK_t)$, where
$\mK_t$ is the kernel matrix of the first $t$ inputs, so
$\tfrac12 \log\det\mA_t$ is the information $I(\vy_t; f)$ of @eq-regret-info
gathered by the inputs the algorithm chose, and at most $\gamma_t$. The union
bound charged a logarithm per input; this bound charges per direction the data
have measured.

::: {.aside title="What the log-determinant counts"}
Suppose the inputs were fixed in advance and the noise were Gaussian with
standard deviation $\sigma_n$, so that $R = \sigma_n$. Write the eigenvalues of
$\mA_t$ as $1 + \eta_1, 1 + \eta_2, \dots$. Since
$\E[\mathbf{s}_t\mathbf{s}_t^\T] = \sigma_n^4(\mA_t - \mI)$, the left side of
@eq-regret-self-normalized divided by $\sigma_n^4$ has mean
$\sum_j \eta_j/(1 + \eta_j)$. Each term lies between 0 and 1, near 1 for a
direction the data have measured well and near 0 for one they have barely
touched, so the mean counts the measured directions. The right side divided by
$\sigma_n^4$ is $\sum_j \log(1 + \eta_j) + 2\log(1/\delta)$, and
$\log(1 + \eta) \ge \eta/(1 + \eta)$. The bound charges each measured
direction a little more than its average share, and that margin buys a
statement that holds for adaptive inputs and for all $t$ at once.
:::

On a finite domain, as in @thm-regret-gpucb, features with finitely many
entries always exist, and the theorem turns into a confidence bound for a
fixed function.

::: {.derivation title="A confidence bound for a fixed function"}
Take $\boldsymbol{\phi}(\vx)$ to be the column of $\mK_\X^{1/2}$ that belongs
to $\vx$, where $\mK_\X$ is the kernel matrix of the whole domain. Then
$k(\vx, \vx') = \boldsymbol{\phi}(\vx)^\T\boldsymbol{\phi}(\vx')$, and every
function in the RKHS is $f(\vx) = \boldsymbol{\phi}(\vx)^\T\vw$ with
$\lVert \vw \rVert = \lVert f \rVert_k \le \sqrt{B}$. With
$\boldsymbol{\Phi}_t$ the matrix whose rows are
$\boldsymbol{\phi}(\vx_1)^\T, \dots, \boldsymbol{\phi}(\vx_t)^\T$, the
posterior of @eq-bayes-blr-posterior and @eq-bayes-blr-predictive is
$\mu_t(\vx) = \boldsymbol{\phi}(\vx)^\T\bar\vw_t$ with
$\bar\vw_t = \sigma_n^{-2}\mA_t^{-1}\boldsymbol{\Phi}_t^\T\vy_t$, and
$\sigma_t^2(\vx) = \boldsymbol{\phi}(\vx)^\T\mA_t^{-1}\boldsymbol{\phi}(\vx)$.

1. *Split the error.* Substituting $\vy_t = \boldsymbol{\Phi}_t\vw + (\varepsilon_1, \dots, \varepsilon_t)^\T$
   and $\boldsymbol{\Phi}_t^\T\boldsymbol{\Phi}_t = \sigma_n^2(\mA_t - \mI)$
   gives $\bar\vw_t = \vw - \mA_t^{-1}\vw + \sigma_n^{-2}\mA_t^{-1}\mathbf{s}_t$,
   so $\mu_t(\vx) - f(\vx) = -\boldsymbol{\phi}(\vx)^\T\mA_t^{-1}\vw + \sigma_n^{-2}\boldsymbol{\phi}(\vx)^\T\mA_t^{-1}\mathbf{s}_t$.
2. *Separate the input from the rest.* By the Cauchy-Schwarz inequality in the
   inner product $\mathbf{u}^\T\mA_t^{-1}\mathbf{v}$, for any vector $\mathbf{v}$,
   $|\boldsymbol{\phi}(\vx)^\T\mA_t^{-1}\mathbf{v}| \le \sigma_t(\vx)\sqrt{\mathbf{v}^\T\mA_t^{-1}\mathbf{v}}$.
3. *Bias.* $\mA_t$ is $\mI$ plus positive semidefinite terms, so
   $\vw^\T\mA_t^{-1}\vw \le \lVert\vw\rVert^2 \le B$, and the first term of
   step 1 is at most $\sqrt{B}\,\sigma_t(\vx)$.
4. *Noise.* By @eq-regret-self-normalized, the second term is at most
   $\sigma_t(\vx)\,(R/\sigma_n)\sqrt{\log\det\mA_t + 2\log(1/\delta)}$.
5. *Information.* $\log\det\mA_t = 2I(\vy_t; f) \le 2\gamma_t$.

Together, with probability at least $1 - \delta$, for every input and every
$t \ge 0$ at once,
$$
|f(\vx) - \mu_t(\vx)| \le \Big(\sqrt{B} + \frac{R}{\sigma_n}\sqrt{2\big(\gamma_t + \log(1/\delta)\big)}\Big)\, \sigma_t(\vx).
$$ {#eq-regret-selfnorm-width}
:::

GP-UCB in round $t$ uses the posterior after $t - 1$ observations, so
@eq-regret-selfnorm-width gives it a valid confidence bound with

$$
\beta_t^{1/2} = \sqrt{B} + \frac{R}{\sigma_n}\sqrt{2\big(\gamma_{t-1} + \log(1/\delta)\big)}.
$$ {#eq-regret-beta-frequentist}

The first term is the bias: a function of large norm can sit far from what the
prior expects, but by at most $\sqrt{B}$ posterior standard deviations. The
second is the noise. It grows with $\gamma_{t-1}$ because every direction the
data have measured is a direction in which the noise could have pushed the
estimate, and with $\log(1/\delta)$ as every Chernoff bound does. The number
of inputs $|\X|$ does not appear: step 2 covers every input at once, where the
Bayesian width took a union over them. That is why bounds of this kind carry
over to continuous domains.

The width agrees with the frequentist statement of @sec-regret-other-settings.
There the noise is bounded by $\sigma_n$ in absolute value and the model's
noise variance is $\sigma_n^2$ [@srinivas2010gaussian, Thm. 3], so
$R = \sigma_n$, and squaring @eq-regret-beta-frequentist with
$(u + v)^2 \le 2u^2 + 2v^2$ gives
$\beta_t \le 2B + 4\big(\gamma_{t-1} + \log(1/\delta)\big)$. The schedule
$\beta_t = 2B + 300\gamma_t \log^3(t/\delta)$ of @srinivas2010gaussian has the
same term $2B$ from the norm of $f$, and the same information gain for the
noise, multiplied by $300\log^3(t/\delta)$. Their proof applies Freedman's
inequality, a martingale version of Bernstein's inequality that uses the
conditional variances, and a union bound over rounds
[@srinivas2010gaussian, app. B]; the factor 300 and the cube of the logarithm
come from that route (inference).

For a general domain the features can have infinitely many entries, and the
argument of @abbasiyadkori2011improved breaks down. @chowdhury2017kernelized
proved a self-normalized bound that holds in that case. With
$\lVert f \rVert_k^2 \le B$ and noise that is $R$-sub-Gaussian given the past,
their Theorem 2 states that with probability at least $1 - \delta$,
$$
|\mu_{t-1}(\vx) - f(\vx)| \le \Big(\sqrt{B} + R\sqrt{2\big(\gamma_{t-1} + 1 + \log(1/\delta)\big)}\Big)\, \sigma_{t-1}(\vx)
$$
for every input and every round up to the horizon $T$, where the posterior and
$\gamma_{t-1}$ are computed with noise variance $1 + 2/T$ in place of
$\sigma_n^2$. (They write $B$ for the norm itself, and their $\beta_t$ is the
multiplier of $\sigma_{t-1}(\vx)$, our $\beta_t^{1/2}$.) The extra 1 pays for
that slightly inflated noise variance, which adds $t\log(1 + 2/T)$, at most 2,
to the log-determinant. Their algorithm, IGP-UCB, uses this width, which is
narrower than that of GP-UCB by a factor that grows like
$\log^{3/2}(t/\delta)$.

Steps 2 to 5 of the GP-UCB proof in @sec-regret-gpucb-bound use nothing about
$f$ except the confidence statement and $k(\vx, \vx) \le 1$. With the width of
@eq-regret-beta-frequentist on a finite domain, they give
$R_T \le \sqrt{C_1 T \beta_T \gamma_T}$ again, now with probability at least
$1 - \delta$ for every fixed $f$ with $\lVert f \rVert_k^2 \le B$. Since
$\beta_T \le 2B + 4(R/\sigma_n)^2\big(\gamma_T + \log(1/\delta)\big)$, the
regret for a fixed $\delta$ is of order $\sqrt{T}\big(\sqrt{B\gamma_T} + \gamma_T\big)$,
the order stated in @sec-regret-other-settings, now without hidden logarithmic
factors. @chowdhury2017kernelized state their bound, in this notation, as
$R_T = O\big(\sqrt{BT\gamma_T} + \sqrt{T\gamma_T(\gamma_T + \log(1/\delta))}\big)$.

::: {.keyidea title="The information gain is the price of adaptivity"}
When the inputs are chosen from the noise that came before, a confidence bound
for a fixed function must hold in every direction the data could have
measured. The self-normalized bound pays for that with the log-determinant of
the posterior precision, twice the information gain, so the width grows like
$\sqrt{\gamma_t + \log(1/\delta)}$ instead of with the number of inputs.
:::

## What bounds say about practice {#sec-bounds-and-practice}

@thm-regret-gpucb gives a number. It is worth computing it on a problem where
every assumption holds: a function drawn from the Gaussian process that GP-UCB
uses, on a finite grid, with the noise level the algorithm is told. The figure
below does that, and runs GP-UCB twice on the same function and noise, once
with the theorem's $\beta_t$ and once with a constant multiplier of the kind
practitioners use.

```{figure}
//| figure: regret-gpucb
//| label: fig-regret-gpucb
//| fig-cap: "GP-UCB on a function drawn from its own prior (top), on a finite domain of 160 inputs: a grid in one dimension, or a fixed Latin hypercube sample of the cube in three or six dimensions, where the top panel plots every input against its first coordinate. The ticks under the function mark where each run evaluated; taller ticks mean repeated evaluations. The bottom panel shows cumulative regret on a logarithmic scale: GP-UCB with the theorem's $\beta_t$ ($\delta = 0.1$), GP-UCB with a constant multiplier $\sqrt\beta$, the expected regret of uniformly random queries, and the bound of @thm-regret-gpucb. The bound uses a greedy estimate of $\gamma_T$, which can only understate it, so the true bound is at least as high as drawn. Values are for one random function; press *New function* for another."
```

Some things to try.

**Read the default.** At the default settings and $T = 200$, the bound is
several hundred, larger than the regret that uniformly random queries would
incur, while GP-UCB with the theorem's $\beta_t$ has regret below 20 and GP-UCB
with $\sqrt\beta = 2$ about half of that. On this problem the theorem is true and, at this horizon,
weaker than the trivial bound of doing nothing clever.

**Drag the noise.** As $\sigma_n$ goes from 0.1 to 1, $C_1$ grows from 1.73 to
11.5, while $\gamma_T$ falls, because noisy evaluations teach less. The bound
moves much less than either constant. Both runs suffer more from noise than the
bound does.

**Switch the kernel to Matérn 1/2.** Rough functions have many more
distinguishable regions, $\gamma_T$ grows several times faster, and both runs
take longer to find the peak.

**Set the practical multiplier to 0.** This is pure exploitation: evaluate
wherever the posterior mean is highest. Starting from a mean of zero
everywhere, the rule keeps evaluating the first input whose value comes out
above zero, because nothing else ever looks better. Unless that input happens
to be the peak, its regret grows in a straight line, as it does for the
default function. Press *New function* to see both cases. Notice also that the
bound does not move when the function changes: it depends only on the kernel,
the noise, and $T$.

**Switch to 3-D, then 6-D.** The domain still has 160 inputs, but they are now
scattered through a cube, and at lengthscale 0.1 almost no two of them are
close enough to be correlated. Each input is effectively its own arm:
$\gamma_T$ jumps from about 42 to about 350 in three dimensions and 380 in six,
close to the value for 160 independent arms, and both runs lose much more.
Raise the lengthscale to 0.5 and the information gain falls again, because a
longer lengthscale lets each evaluation speak for more of the cube. How to
choose the lengthscale as the dimension grows is a practical question of its
own (@sec-high-dim-practice).

**Compare the multipliers.** The readout shows $\sqrt{\beta_T}$, about 6 at
$T = 200$ (@exr-regret-beta). Six standard deviations of optimism is far more
than the error of a well-specified posterior requires, which is why the
theorem's run keeps exploring long after the practical run has settled.

### What the bounds do say {#sec-regret-bounds-say}

The bounds of this chapter establish that sublinear regret is possible for
black-box optimization at all, under stated assumptions, and they identify
what governs its rate. For bandits, it is the gaps and the number of arms; for
Gaussian processes, it is the maximum information gain, a property of the
prior and the noise rather than of any algorithm. The rates in @tbl-regret-gamma
rank problems sensibly: smoother kernels are easier than rough ones, and
dimension hurts most where smoothness is low.

The proofs also explain why particular design choices matter. Exploration must
never switch off entirely: greedy and constant-ε rules fail for structural
reasons, and the confidence parameter $\beta_t$ grows, slowly, for the same
reason that the UCB1 bonus contains $\ln t$. Optimism, posterior sampling, and
information-seeking are the mechanisms with guarantees, and several of the
acquisition functions of @sec-acquisition are built from them.

### What the bounds do not say {#sec-regret-bounds-not-say}

**The constants matter at practical horizons.** Bounds are proved with
whatever constants make the proof go through, and they are rarely tight. The
authors of GP-UCB found, by cross-validation (trying each scaling on data held
out from the fit), that their algorithm improved when $\beta_t$ was scaled
down by a factor of 5 from the theorem's value, and
noted that they did not optimize the constants of their bounds
[@srinivas2010gaussian]. Auer and colleagues' variant UCB1-TUNED performed
substantially better than UCB1 in essentially all their experiments, and they
could not prove a regret bound for it [@auer2002finite]. @fig-regret-bandits
and @fig-regret-gpucb both show bounds that are true and, at the horizons a
practitioner has, larger than the regret of naive play.

**The model is assumed correct.** @thm-regret-gpucb assumes that the kernel,
its hyperparameters, and the noise level are known, and that $f$ was drawn
from exactly that prior; the RKHS version assumes a known bound $B$ on the
norm. In practice hyperparameters are fitted to the data as they arrive
(@sec-fitting-hyperparameters), and that changes the algorithm. @bull2011convergence
proved convergence rates for expected improvement with a fixed prior, and
showed that with standard sequential estimates of the prior's parameters the
procedure may never find the optimum; alternative estimators restore the
rates. @berkenkamp2019noregret gave the first algorithm that is provably
no-regret without knowing the hyperparameters, by slowly enlarging the
function class it considers.

**The score may not be yours.** The GP-UCB bound is about cumulative regret.
When only the final recommendation matters, @eq-regret-simple-from-cumulative
turns it into a guarantee, but an algorithm tuned for cumulative regret can be
needlessly cautious about exploring [@bubeck2009pure]. Expected improvement,
which @bull2011convergence calls perhaps the most popular method for this
problem, was analyzed there for simple regret with noise-free evaluations. Bounds proved in
different settings and for different scores do not rank the acquisition
functions of @sec-acq-compare against each other (inference).

**The acquisition function is assumed maximized exactly.** The theorems take
$\vx_t$ to be the exact maximizer of @eq-regret-gpucb. On a continuous domain
that maximization is itself a hard, multimodal problem, solved approximately
by the methods of @sec-acq-optimization [@srinivas2010gaussian]. No bound in
this chapter accounts for the error.

**Dimension enters through exponents.** The three- and six-dimensional
settings of @fig-regret-gpucb show the effect at small scale. The RBF rate
$(\log T)^{d+1}$ is mild in $T$ but not in $d$: for $d = 10$ and $T = 1000$,
$(\ln T)^{11}$ is about $1.7 \times 10^9$, so unless the constant hidden in the
$O(\cdot)$ is minute, a bound of order $\sqrt{T\gamma_T}$ exceeds the trivial
bound, linear in $T$, by orders of magnitude (inference). For Matérn kernels in the RKHS
setting, combining the regret of order $\gamma_T\sqrt{T}$ for GP-UCB with the
rate of @vakili2021information gives an exponent of $\tfrac12 + d/(2\nu + d)$,
which reaches 1, and so says nothing at all, once $d \ge 2\nu$ (inference). How
Bayesian optimization copes with many dimensions in practice is the subject of
@sec-high-dim-practice and @sec-high-dimensions.

**A bound is about a class, not about your function.** A worst-case bound
holds for every function in the class, and a Bayesian bound for most functions
drawn from the prior. The function you face is one particular function, and
neither kind of bound predicts how a method will rank on it. Benchmarks answer
that question, with their own limits (@sec-sw-evaluation).

The useful stance is to treat regret bounds as design principles and as
sanity checks rather than as forecasts. An algorithm with a guarantee is built
from mechanisms that cannot get permanently stuck; its constants are then
tuned empirically, as the authors of the guarantees did themselves. The next
chapter turns to those empirical decisions.

## Exercises {#sec-regret-exercises}

::: {.exercise #exr-regret-eps-linear}
Show that ε-greedy with constant $\varepsilon > 0$ on $K$ arms has expected
regret at least $\frac{\varepsilon T}{K} \sum_i \Delta_i$ after $T$ rounds
(ignoring the initial round in which each arm is pulled once). Evaluate the
slope for the default arms of @fig-regret-bandits, whose means are $0.6$,
$0.5$, $0.383$, $0.267$, and $0.15$, with $\varepsilon = 0.1$, and compare it
with the figure.

::: {.solution}
In each round, with probability $\varepsilon$ the algorithm pulls an arm
chosen uniformly, which has expected gap $\frac1K \sum_i \Delta_i$; with
probability $1 - \varepsilon$ it pulls the greedy arm, whose gap is at least 0.
So the expected regret of every round is at least
$\frac{\varepsilon}{K}\sum_i \Delta_i$, and summing over $T$ rounds gives the
bound. For the default arms the gaps are $0, 0.1, 0.217, 0.333, 0.45$, with sum
$1.1$, so the slope is at least $0.1 \times 1.1 / 5 = 0.022$ per round, or 22
over 1000 rounds. The figure shows ε-greedy at about 60 after 1000 rounds: the
rest comes from rounds in which the greedy arm is not the best one, because
the averages of rarely explored arms are still noisy.
:::
:::

::: {.exercise #exr-regret-pinsker}
Two Bernoulli arms have means $0.6$ and $0.5$. Compute the Lai-Robbins
constant $c^*$ of @eq-regret-lai-robbins and the coefficient of $\ln T$ in
@thm-regret-ucb1. What do the two numbers predict for the number of pulls of
the worse arm after $T = 10^4$ rounds?

::: {.solution}
$\mathrm{kl}(0.5, 0.6) = 0.5\ln(0.5/0.6) + 0.5\ln(0.5/0.4) = 0.5(-0.1823 +
0.2231) = 0.0204$. With $\Delta = 0.1$, $c^* = 0.1/0.0204 = 4.90$, and the
asymptotic number of pulls of the worse arm is $\ln T/\mathrm{kl} = 9.21/0.0204
\approx 450$. UCB1's coefficient is $8/\Delta = 80$, about 16 times $c^*$, and
its bound on the pulls of the worse arm is $8\ln T/\Delta^2 + 1 + \pi^2/3
\approx 7{,}370 + 4$, which says little when the budget is $10^4$ pulls in
total. Pinsker's inequality, $\mathrm{kl} \ge 2\Delta^2 = 0.02$, is nearly tight
here, so the factor of 16 is close to the worst case.
:::
:::

::: {.exercise #exr-regret-independent}
For a diagonal kernel on $K$ arms (each $k(i, i) = 1$, all other entries 0)
and $T$ a multiple of $K$, show that
$\gamma_T = \frac{K}{2}\log\!\left(1 + \frac{T}{K\sigma_n^2}\right)$. What does
@thm-regret-gpucb then say about the growth of $R_T$ in $K$ and $T$, and how
does it compare with the worst-case lower bound of @sec-regret-lower-bounds?

::: {.solution}
With independent arms, $\mK_A$ is block diagonal: if arm $i$ is evaluated
$m_i$ times, its block is an $m_i \times m_i$ matrix of ones,
$\mathbf{1}\mathbf{1}^\T$. The corresponding block of
$\mI + \sigma_n^{-2}\mK_A$ is $\mI + \sigma_n^{-2}\mathbf{1}\mathbf{1}^\T$,
which has the eigenvalue $1 + m_i/\sigma_n^2$ (eigenvector $\mathbf{1}$) and
all others equal to 1, so its determinant is $1 + m_i/\sigma_n^2$ (a case of
the matrix determinant lemma, @eq-id-det-lemma), and the information is
$\frac12\sum_i \log(1 + m_i/\sigma_n^2)$ subject to $\sum_i m_i = T$. The
logarithm is concave, so the sum is largest when the $m_i$ are equal, $m_i =
T/K$, which gives the formula. Then $\sqrt{C_1 T \beta_T \gamma_T}$ grows like
$\sqrt{KT}$ times logarithmic factors in $T$ and $K$. The worst-case lower
bound is $\frac{1}{27}\sqrt{(K-1)T}$, so in this special case the GP-UCB bound
is tight up to logarithmic factors, as @srinivas2010gaussian note.
:::
:::

::: {.exercise #exr-regret-beta}
Compute the theorem's $\beta_T$ and $\sqrt{\beta_T}$ for the setting of
@fig-regret-gpucb: $|\X| = 160$, $\delta = 0.1$, $T = 200$. How does $\sqrt{\beta_T}$
change if the grid is refined to $|\X| = 16{,}000$ points, and what does that
say about the role of $|\X|$?

::: {.solution}
$|\X| T^2 \pi^2/(6\delta) = 160 \times 40{,}000 \times 9.8696/0.6 \approx 1.053
\times 10^8$, whose natural logarithm is $18.47$, so $\beta_T \approx 36.9$ and
$\sqrt{\beta_T} \approx 6.08$. A grid 100 times finer adds $2\ln 100 \approx
9.2$ to $\beta_T$, giving $\beta_T \approx 46.1$ and $\sqrt{\beta_T} \approx
6.79$. The dependence on $|\X|$ is logarithmic, but it never disappears on a
finite grid, and it is why the continuous-domain version needs a separate
argument: refining the grid without limit would make $\beta_t$ infinite.
:::
:::

::: {.exercise #exr-regret-sample-sizes}
Rewards lie in $[0, 1]$. How many pulls does an arm need before the
probability that its average exceeds its mean by 0.1 or more is at most 0.05,
according to Chebyshev's inequality (with the largest possible variance,
$\tfrac14$) and according to Hoeffding's? Repeat for the probability $10^{-4}$,
and for an interval that must hold in each of 1000 rounds with total failure
probability 0.05. Compare the last answers with @fig-regret-concentration with
the union set to 1000.

::: {.solution}
Chebyshev's inequality, @eq-regret-chebyshev, needs
$n \ge \Var[Y]/(\delta a^2) = 0.25/(0.05 \times 0.01) = 500$. Hoeffding's,
@eq-regret-hoeffding, needs $n \ge \ln(1/\delta)/(2a^2) = \ln 20/0.02 = 149.8$,
so 150. For $\delta = 10^{-4}$, Chebyshev needs 250,000 pulls and Hoeffding
$\ln(10^4)/0.02 = 460.5$, so 461. Shrinking $\delta$ by a factor of 500
multiplies Chebyshev's answer by 500 and Hoeffding's by about 3.1. For 1000
rounds the union bound gives each round $\delta = 5 \times 10^{-5}$: Chebyshev
needs 500,000 pulls and Hoeffding $\ln(20{,}000)/0.02 = 495.2$, so 496, the
numbers in the figure's readout. The extra pulls Hoeffding needs,
$\ln 1000/0.02 \approx 345$ (346 after rounding both answers up), come from the
$\ln 1000$ of @eq-regret-union-width.
:::
:::

::: {.exercise #exr-regret-linear-width}
For the linear kernel $k(\vx, \vx') = \vx^\T\vx'$ in $d$ dimensions, with
features $\boldsymbol{\phi}(\vx) = \vx$ and inputs of length at most 1, show
that $\log\det\mA_T \le d\log\!\big(1 + T/(d\sigma_n^2)\big)$ for any $T$
inputs, so that their information gain is at most
$\tfrac{d}{2}\log\!\big(1 + T/(d\sigma_n^2)\big)$. What does
@eq-regret-beta-frequentist then say about how $\beta_T$ grows, and what plays
the role that $\log|\X|$ plays in @thm-regret-gpucb? Evaluate the bound on the
information gain for $d = 3$, $T = 1000$, and $\sigma_n = 1$.

::: {.solution}
$\mA_T = \mI + \sigma_n^{-2}\sum_s \vx_s\vx_s^\T$ is a $d \times d$ matrix with
positive eigenvalues $\kappa_1, \dots, \kappa_d$. Its determinant is their product, which
by the inequality between the geometric and the arithmetic mean is at most
$\big(\tfrac1d\sum_j \kappa_j\big)^d = (\tr\mA_T/d)^d$. The trace is
$d + \sigma_n^{-2}\sum_s \lVert\vx_s\rVert^2 \le d + T/\sigma_n^2$, so
$\log\det\mA_T \le d\log\!\big(1 + T/(d\sigma_n^2)\big)$, and half of it bounds
the information gain, as in step 5 of the derivation of
@eq-regret-selfnorm-width. This is the $O(d\log T)$ of @tbl-regret-gamma. The
features have $d$ entries however many inputs the domain holds, so the
self-normalized bound applies directly, and since $2\gamma_{T-1}$ obeys the same
bound, @eq-regret-beta-frequentist gives
$\beta_T^{1/2} \le \sqrt{B} + (R/\sigma_n)\sqrt{d\log\!\big(1 + T/(d\sigma_n^2)\big) + 2\log(1/\delta)}$,
which grows like $\sqrt{d\log T}$: the dimension takes the place of $\log|\X|$,
and the domain may be infinite. For $d = 3$, $T = 1000$, and $\sigma_n = 1$,
the bound is $1.5\log(334.3) = 8.72$ nats, against the
$T \cdot \tfrac12\log 2 = 346.6$ nats that 1000 evaluations at completely
unrelated inputs would gather.
:::
:::

## Further reading {#further-reading .unnumbered}

- @lattimore2020bandit is the reference on bandit theory: concentration
  inequalities in chapter 5, UCB in chapters 7 and 8, self-normalized bounds
  and the method of mixtures in chapter 20, the lower bounds in chapters 15
  and 16, simple regret and pure exploration in chapter 33, and Thompson
  sampling in chapter 36. The book is free online.
- @auer2002finite is short and readable, with the UCB1 proof sketched above,
  the decaying ε-greedy rule, and experiments that compare them.
- @russo2018tutorial is a practical tutorial on Thompson sampling, with many
  worked examples beyond Bernoulli arms.
- @srinivas2010gaussian introduced GP-UCB, the maximum information gain, and
  the regret bounds of @sec-gp-bandits; the extended arXiv version has the
  proofs.
- @abbasiyadkori2011improved prove the self-normalized bound of
  @thm-regret-self-normalized for linear bandits, and @chowdhury2017kernelized
  extend it to kernels on general domains, with the confidence width and
  regret bound of @sec-regret-self-normalized.
- @garnett2023bayesian, chapter 10, surveys the theoretical analysis of
  Bayesian optimization, including results for expected improvement and
  information-based policies.
- @lai1985asymptotically is the original lower bound; most readers will find
  the modern statement in @lattimore2020bandit easier to read first.
