---
status: done
synopsis: "The bandit view of learning from duels: what 'the best option' means when preferences are not transitive, the classic algorithms and their guarantees, the kernelized bounds of 2021 to 2026 with their assumptions and units, and the lower bound nobody has proved."
sources: ["Yue et al. 2012", "Sui et al. 2018", "Zoghi et al. 2014", "Wu and Liu 2016", "Kirschner and Krause 2021", "Pásztor et al. 2024", "Kayal et al. 2025"]
---

# Dueling Bandits and the Theory of Comparisons {#sec-dueling-bandits}

@sec-pbo built a loop that learns from duels and asked how to choose the next
pair. It did not ask how good any rule for choosing pairs can be. That question
belongs to bandit theory, which @sec-regret introduced for ordinary
evaluations: an algorithm is scored by its *regret*, the total shortfall of
what it chose compared with the best choice, and a good algorithm is one whose
regret grows slowly. This chapter asks the same question for comparisons.

The question turns out to have a twist that ordinary bandits do not have.
With numbers, "the best option" is the one with the largest value. With
duels, it is the one that wins, and wins against whom is not always a
consistent notion: preferences can go in a circle, as in rock, paper,
scissors. So the chapter begins with what "best" can mean, then follows the
algorithms that find it among finitely many options, then moves to the
continuous, kernelized setting that preferential Bayesian optimization (PBO) lives
in, where the bounds of 2021 to 2026 are, and ends with what is still
unproved.

## Dueling bandits {#sec-dueling-bandits-intro}

The *dueling bandit* problem was posed for search engines. Suppose an
intranet search system ships with $K$ built-in ranking functions and must
find the best one for a new customer. Asking users to rate result lists is
unreliable, but a trick called *interleaving* merges the results of two
rankers into one list and infers from the user's clicks which ranker they
preferred. Each query to the system is then a duel between two rankers, and
the system pays every time it shows results from a worse ranker
[@yue2012karmed; @yue2009interactively].

Formally, there are $K$ options, called *arms* as in @sec-bandits. In each
round $t$ the algorithm picks two arms $a_t$ and $b_t$, possibly the same, and
observes which one wins. Arm $i$ beats arm $j$ with an unknown probability
$P_{ij}$, with $P_{ji} = 1 - P_{ij}$ and $P_{ii} = 1/2$. The $K \times K$
matrix $\mathbf{P}$ of these probabilities is everything there is to know,
and the algorithm never sees it, only the outcomes of the duels it chooses.
It is convenient to measure each probability from one half:
$\Delta_{ij} = P_{ij} - 1/2$, positive when $i$ tends to beat $j$.

When the preferences come from a utility, as in @eq-pbo-probit or the
Bradley-Terry model of @sec-bradley-terry, $P_{ij}$ is a link function of the
utility difference, and everything in this section is simple: the arm with
the highest utility beats every other arm. The dueling bandit formulation does
not assume a utility. It starts from the matrix, which is more general, and
that generality is why the next question has more than one answer.

### What "best" can mean {#sec-duel-winners}

The most natural definition asks for an arm that beats everyone.

::: {.definition #def-duel-condorcet title="Condorcet winner"}
An arm $i$ is a *Condorcet winner* if $P_{ij} > 1/2$ for every $j \ne i$.
:::

The name comes from voting theory, where a Condorcet winner is a candidate
who beats every other candidate in a head-to-head vote. There is at most one,
but there may be none. If A beats B, B beats C, and C beats A, no arm beats
everyone. Three weaker definitions always produce an answer
[@sui2018advancements].

::: {.definition #def-duel-copeland title="Copeland winner"}
The *Copeland score* of arm $i$ is the number of other arms it beats,
$\#\{j \ne i : P_{ij} > 1/2\}$. A *Copeland winner* is an arm with the highest
score.
:::

::: {.definition #def-duel-borda title="Borda winner"}
The *Borda score* of arm $i$ is its average probability of beating another arm,
$\frac{1}{K - 1}\sum_{j \ne i} P_{ij}$: the probability that it wins a duel
against an opponent drawn uniformly at random. A *Borda winner* is an arm with
the highest score.
:::

::: {.definition #def-duel-vonneumann title="von Neumann winner"}
A *von Neumann winner* is a probability distribution $\boldsymbol{\pi}$ over the
arms such that an arm drawn from $\boldsymbol{\pi}$ beats every fixed arm with
probability at least one half on average:
$\sum_i \pi_i P_{ij} \ge 1/2$ for every $j$.
:::

Each definition answers a slightly different question. A Copeland winner
counts victories and ignores their margins, so it exists and coincides with
the Condorcet winner when there is one, but it may still lose to some arms. A
Borda winner weighs margins, so it can differ from the Condorcet winner even
when one exists: an arm that beats everyone narrowly can have a lower average
than an arm that loses to it narrowly and crushes everyone else
[@sui2018advancements; @urvoy2013generic; @jamieson2015sparse]. A von Neumann
winner is not a single arm but a mixture, the optimal strategy of the
zero-sum game in which each player picks an arm and the payoff is
$P_{ij} - 1/2$. Von Neumann's minimax theorem guarantees that one exists, and
when a Condorcet winner exists, the von Neumann winner puts all its weight on
it [@dudik2015contextual; @sui2018advancements].

Two of these have already appeared in this book under other names. The
*soft-Copeland* score that @gonzalez2017preferentialb maximize
(@sec-dueling-formulation) is the average probability of winning against a
uniformly random opponent, so despite its name it is the continuous version
of the Borda score. And the Borda count appeared in @sec-query-aggregation:
when people with different hidden preferences are pooled into one
Bradley-Terry model, the fitted utility ranks options by their Borda counts
[@siththaranjan2024distributional].

The figure below lets you build a non-transitive tournament and watch the
definitions part ways.

```{figure}
//| figure: duel-winners
//| label: fig-duel-winners
//| fig-cap: "Four ways to name the best of five arms. The matrix starts from a Bradley-Terry model with utilities 1, 0.7, 0.45, 0.2, and 0 (scale 3), and the cycle strength c adds a rock-paper-scissors component among A, B, and C on the logit scale, so that B gains on A, C on B, and A on C. Left: an arrow from the winner of each pair to the loser, thicker for more lopsided pairs; arrows in a three-cycle are magenta. Right: the probability that the row arm beats the column arm. Below: each concept's scores, with its winner highlighted (for the von Neumann winner, the arms in the mixture). Click a cell to reverse who wins that pair. The matrices are illustrative."
```

Some things to try:

- **Set the cycle strength c to 0.** The matrix comes from a utility, so A beats
  everyone: it is the Condorcet winner, the Copeland winner, the Borda winner,
  and the von Neumann winner, all at once (@exr-duel-utility).
- **Raise c past about 1.** B now beats A, C beats B, and A still
  beats C. The Condorcet winner disappears, A, B, and C tie on Copeland score,
  A keeps the highest Borda score because it crushes D and E, and the von
  Neumann winner becomes a mixture of the three. At the default of 1.5, it puts
  the most weight on B (@exr-duel-rps explains the pattern).
- **Set c back to 0 and switch to "A narrow champion".** A beats every arm with probability 0.55,
  and B beats C, D, and E with probability 0.9. A is the Condorcet and Copeland
  winner, but B is the Borda winner: the average rewards B's large margins.
- **Flip a pair at the bottom.** Make E beat D. Nothing changes at the top:
  the definitions disagree only when the strong arms do.

### Transitivity, and regret {#sec-duel-transitivity}

How much structure a preference matrix has is described by conditions of
*stochastic transitivity*, which extend "if A beats B and B beats C, then A
beats C" to probabilities. For arms with $\Delta_{ij} \ge 0$ and
$\Delta_{jk} \ge 0$, *strong* stochastic transitivity requires
$\Delta_{ik} \ge \max\{\Delta_{ij}, \Delta_{jk}\}$, *moderate* requires
$\Delta_{ik} \ge \min\{\Delta_{ij}, \Delta_{jk}\}$, and *weak* only
$\Delta_{ik} \ge 0$. A separate condition, the *stochastic triangle
inequality*, requires $\Delta_{ik} \le \Delta_{ij} + \Delta_{jk}$ for arms in
the order $i \succ j \succ k$; it is not implied by strong transitivity
[@bengs2021preference]. Any model of the form "utility plus a monotone link",
Bradley-Terry and Thurstone included, satisfies strong stochastic transitivity,
because the winning probability grows with the utility difference; and since
these links are concave for positive differences, they also satisfy the
triangle inequality (inference, from the definitions; @exr-duel-utility).

Whether real preferences violate transitivity is contested.
@chau2022inconsistent placed a Gaussian process on a skew-symmetric preference
function, one that can represent cycles, and found it more accurate than the
utility model of @chu2005preference on chameleon contests, NFL games, and a
citation graph, concluding that violations are common; these are data sets in
which intransitivity is expected, and none is a single person's design
preference (@sec-obs-extensions). A 2026 result, about many annotators judging the
responses of language models, shows that preferences can be represented by a
single reward function if and only if they contain no Condorcet cycle, and
that under a Luce model of the population such cycles exist with probability
converging to one exponentially fast [@liu2026statistical]. For one person judging designs, a
utility remains the working assumption of this book.

With a definition of the best arm comes a definition of regret. When a
Condorcet winner, call it arm 1, exists, the regret of a duel between $a_t$
and $b_t$ is how much more often the winner would have beaten them,

$$
r_t = \Delta_{1 a_t} + \Delta_{1 b_t}, \qquad R_T = \sum_{t=1}^{T} r_t,
$$ {#eq-duel-regret}

the formulation of @yue2012karmed. It can be read as the fraction of users who
would have preferred the best ranker to the two that were shown. A duel of the
winner against itself costs nothing, so a learner that has found the winner
can stop paying. Without a Condorcet winner, regret is measured against a
Copeland winner: with normalized Copeland scores $\zeta_i$ (the fraction of
other arms that $i$ beats), a duel costs
$\max_i \zeta_i - (\zeta_{a_t} + \zeta_{b_t})/2$ [@zoghi2015copeland;
@wu2016double].

Dueling bandits are harder than ordinary bandits in one specific way. The
algorithm pays for the arms it plays, but it observes only how those two arms
compare with each other, never how either compares with the unknown best arm.
Learning that both are bad requires duels between other pairs
[@sui2018advancements].

## Algorithms {#sec-dueling-algorithms}

Dueling bandit algorithms come in two styles [@sui2018advancements]. Most are
*asymmetric*: they pick a reference arm, the current champion or a plausible
winner, and then a challenger to test against it. Some are *symmetric*: two
copies of the same learner each pick one arm, as two players of a game would.

### Interleaved Filter {#sec-duel-if}

The first algorithm, Interleaved Filter [@yue2012karmed], works like a
tournament with a reigning champion. It picks a candidate at random and duels
it against every remaining arm in turn. Any arm that loses to the candidate
with high confidence is eliminated; as soon as an arm beats the candidate with
high confidence, that arm becomes the new candidate. When one arm is left, the
algorithm plays it against itself for the rest of the run. It assumes that
the arms have a total order, that strong stochastic transitivity holds, and
that the triangle inequality holds. Under these assumptions its version IF2
has expected regret of order $(K/\varepsilon_{1,2}) \log T$, where
$\varepsilon_{1,2} = \Delta_{12}$ is the margin between the two best arms, and
@yue2012karmed prove that every algorithm suffers regret of order
$(K/\varepsilon)\log T$ on some problems, with $\varepsilon$ the smallest
margin of the best arm. IF2 is optimal up to a constant factor.

### Relative upper confidence bounds {#sec-duel-rucb}

RUCB [@zoghi2014relative] carries the optimism of UCB1 (@sec-regret-ucb1) to
duels and needs only one assumption: that a Condorcet winner exists. It keeps
counts $W_{ij}$ of how often $i$ has beaten $j$ and forms an optimistic
estimate of every winning probability.

::: {.algorithm #alg-duel-rucb title="RUCB (relative upper confidence bound)"}
Input: $K$ arms, exploration parameter $\alpha > 1/2$.

1. For each pair with $n_{ij} = W_{ij} + W_{ji} > 0$ duels so far, set
   $U_{ij} = W_{ij}/n_{ij} + \sqrt{\alpha \ln t / n_{ij}}$; set $U_{ij} = 1$
   if the pair has never dueled and $U_{ii} = 1/2$.
2. *Champion.* Among the arms that beat every other arm optimistically
   ($U_{cj} \ge 1/2$ for all $j$), pick one, $c$. If there is none, pick any arm.
3. *Challenger.* Pick $d = \argmax_j U_{jc}$, the arm with the best optimistic
   chance of beating the champion. This may be $c$ itself, when no arm can
   plausibly beat it.
4. Duel $c$ against $d$, update the counts, and repeat.
:::

The champion is an arm that could still be the Condorcet winner; the
challenger is the arm most likely to prove that it is not. RUCB has a
finite-time regret bound of order $K \log T$ plus a constant that grows like
$K^2$ [@zoghi2014relative; @sui2018advancements].

The best possible rate is known exactly. @komiyama2015regret proved an
asymptotic lower bound for every algorithm, a sum over suboptimal arms in which
each contributes $\log T$, weighted by the regret of the cheapest duel that
exposes it and divided by a Kullback-Leibler divergence between Bernoulli
distributions (@sec-kl), and gave an algorithm, RMED, that matches it
asymptotically [@sui2018advancements]. @saha2022versatile were the first to reach the optimal finite
order $\sum_i \log T / \Delta_i$ against a Condorcet winner, which they
describe as resolving a long-standing problem.

### Thompson sampling, twice {#sec-duel-dts}

Thompson sampling (@sec-regret-thompson) carries over just as naturally.
Double Thompson Sampling (D-TS) [@wu2016double] keeps a Beta posterior on every
winning probability $P_{ij}$, starting from Beta(1, 1), and samples twice. The
first sample of the whole matrix picks the arm with the highest sampled
Copeland score, among the arms whose optimistic Copeland score is highest. A
second sample picks the challenger: the arm most likely to beat the first,
among arms not already known to lose to it. Because it targets Copeland
winners, it works with or without a Condorcet winner. Its regret is of order
$K^2 \log T$ for general Copeland problems, and a simplified version reaches
$K \log T + K^2 \log\log T$ when a Condorcet winner exists [@wu2016double].
Copeland winners also have their own optimistic algorithms with bounds of
order $K \log T$ under mild assumptions [@zoghi2015copeland] and an
asymptotically optimal algorithm [@komiyama2016copeland].

The figure runs simplified versions of RUCB and Double Thompson Sampling on
the matrices of @fig-duel-winners, together with random pairs, and plots the
cumulative Copeland regret.

```{figure}
//| figure: duel-race
//| label: fig-duel-race
//| fig-cap: "RUCB (@alg-duel-rucb, in a simplified form that draws the champion uniformly from the optimistic candidates) and Double Thompson Sampling (Copeland version) on the five-armed matrices of @fig-duel-winners, against uniformly random pairs. Cumulative Copeland regret, averaged over 10 runs. With the cycle strength c at 0, A is a Condorcet winner and both learning rules flatten out; past about 1 there is no Condorcet winner and RUCB, which assumes one, accumulates regret at a constant rate. The random line is clipped when it leaves the plot; its final value is printed. Illustrative, with exploration parameter α = 0.51."
```

Some things to try. At the default, with A as the Condorcet winner, both
curves bend over like a logarithm, and Double Thompson Sampling ends lower:
about 30 against about 70 for RUCB after 2000 duels, while random pairs pay
about 1000. Move the cycle strength to 1.5. Now A, B, and C share the Copeland title,
no arm beats every other arm even optimistically once enough data arrive, and
RUCB's champion step falls back to a random arm in most rounds: its regret
grows in a straight line, past 200 at 2000 duels. Double Thompson Sampling keeps
bending. Then choose "A narrow champion" with the cycle strength at 0: the winner
beats everyone with probability 0.55, a margin of 0.05, and both algorithms
are still paying hundreds after 2000 duels. The $1/\Delta$ in every bound
above is this effect: a narrow margin takes many duels to resolve.

### Sparring, and the first uses with people {#sec-duel-sparring}

The symmetric style treats the two arms as two players. *SelfSparring*
[@sui2017multi] draws each arm of the duel (or each of several arms, for
multi-dueling) from the same Thompson sampling posterior, so the algorithm
duels against itself; with a Gaussian process prior over the arms it can share
information between similar arms. Its theory is thinner than its use. It
assumes "approximate linearity", a winning probability that is approximately a
linear function of the utility difference, which the authors call more
restrictive than strong stochastic transitivity; it proves convergence to the
best arm for independent arms and an asymptotically optimal rate
$O(K\ln T/\Delta)$; and the authors write that a finite-time guarantee would
require a more refined analysis and that an analysis of the kernelized
version is lacking.

These algorithms have run with people in the loop. CorrDuel, a dueling bandit
for large sets of correlated options, chose spinal cord stimulation settings
in a live clinical trial, which its authors describe as the first application
of an online learning algorithm to spinal cord injury treatment
[@sui2017correlational]. CoSpar tuned exoskeleton gaits with the posterior
sampling of SelfSparring, adding coactive feedback in which the user also
suggests improvements [@tucker2020preference]; @sec-cs-exoskeleton follows that
line of work.

### Beyond a list of arms {#sec-duel-beyond-arms}

Two extensions lead toward the continuous problem. For a continuous, convex
problem, the original dueling bandit paper of @yue2009interactively proposed
gradient descent from duels, and @kumagai2017regret proved regret of order
$\sqrt{T\log T}$ for strongly convex, smooth costs, optimal up to logarithmic
factors. For linear utilities over $d$ features, @saha2021optimal gave an
algorithm with regret of order $\sqrt{dT}$ (up to logarithmic factors) for
choosing subsets and observing their winner, and a matching lower bound that
does not depend on the subset size: the winner of a larger subset does not
help, the finite-dimensional counterpart of @sec-query-batch.

## Kernelized dueling {#sec-kernelized-dueling}

PBO is a dueling bandit with infinitely many
arms, one for every point of a continuous domain, and with a utility $f$ that
is smooth. The analysis of @sec-gp-bandits handled the same jump for ordinary
evaluations by replacing the number of arms with the maximum information gain
$\gamma_T$ (@def-regret-gamma) and by assuming that $f$ belongs to the
*reproducing kernel Hilbert space* (RKHS) of a kernel, a space of functions
built from kernel bumps, with a norm bound $\lVert f\rVert_k \le B$ that limits
how rough $f$ can be (@sec-regret-other-settings; @sec-ka-norm-bound says
what the bound assumes). Kernelized dueling bandits
make the same two moves.

What is learned from a duel is a difference, $h(\vx, \vx') = f(\vx) - f(\vx')$.
A function on pairs needs a kernel on pairs, and @sec-pref-model already
derived it: when $f \sim \GP(0, k)$, the difference $h$ is a Gaussian process
on pairs whose covariance is the preference kernel, there written with $g$
for the utility. The bandit literature calls it the *dueling kernel*:

$$
k^D\big((\vx, \vx'), (\vy, \vy')\big) = k(\vx, \vy) - k(\vx, \vy') - k(\vx', \vy) + k(\vx', \vy').
$$ {#eq-duel-kernel}

Adding a constant to $f$ leaves $h$ unchanged, and the dueling kernel is blind
to it, the shift invariance of @sec-pref-identifiability. The analyses below
state their rates in terms of the information gain of
$k^D$ or of $k$, and the two grow at the same rate
[@pasztor2024bandits; @kayal2025bayesian]. When the eigenfunctions of $k$
average to zero under the input distribution, as for a stationary kernel on a
circle, each eigenvalue of the dueling kernel is exactly twice one of $k$; on
an interval they do not, and the correspondence is not exact (@sec-ka-mercer).

### The first kernelized bound {#sec-duel-kk}

@kirschner2021bias gave what they describe as the first efficient kernelized
dueling bandit algorithm with a cumulative regret guarantee, an
information-directed sampling rule. Their feedback model is quantitative: a
duel returns $f(\vx_1) - f(\vx_2) + \xi$, the utility difference plus
sub-Gaussian noise (noise whose tails are no heavier than a Gaussian's), and
covers binary answers only in the sense that a binary answer is a bounded
noisy observation. With $f$ in an RKHS of norm at most
$B$ and $k(\vx, \vx) \le 1$, the regret, summed over both points of each duel,
is of order $\sqrt{T\beta_T(\gamma_T + \log 1/\delta)}$, roughly
$\gamma_T\sqrt{T}$ after $T$ rounds. Their motivation was robustness: in
Bayesian optimization with a bias common to both evaluations, such as drift
in the system being tuned, the difference cancels the bias, and the bound
stays sublinear even when the bias is unbounded. The model is not the
Bradley-Terry or probit one, so the bound does not pay the cost that the
nonlinear link adds below (inference, from the stated feedback model). An
earlier result combined duels with direct evaluations [@xu2020zeroth].

### Bradley-Terry bounds, 2024 to 2026 {#sec-duel-bt-bounds}

Four results analyze the model this book uses for comparisons, a Bernoulli
answer whose probability is the logistic function of the utility difference,
$\Prob(\vx \succ \vx') = \operatorname{sigmoid}\big(f(\vx) - f(\vx')\big)$
with $\operatorname{sigmoid}(a) = 1/(1 + e^{-a})$, the link of @eq-cmp-logit
with $\tau = 1$. Each builds a confidence set for $f$ from a kernelized logistic regression and
chooses pairs by optimism, elimination, or sampling. They differ in what they
assume and in the unit in which they count regret. The next four paragraphs
are a reference for readers of those papers. On a first reading, go to
@tbl-duel-rates, which holds what the rest of the chapter uses.

**POP-BO** [@xu2024principledb] chooses the next point optimistically against
the previous one. Its regret, counted in utility, is
$O(\sqrt{\beta_T\gamma_T T})$, where its confidence width $\beta_T$ itself
grows like $\sqrt{T}$ times the square root of the logarithm of a covering
number, a count of how many functions are needed to approximate every
function in the class. For linear and squared exponential kernels this gives $T^{3/4}$ times
polylogarithmic factors; for Matérn kernels, the result holds only when the
smoothness $\nu$ exceeds $(d/4)\big(3 + d + \sqrt{d^2 + 14d + 17}\big)$, which
is of order $d^2$. The authors read the extra factor as the price of
preference feedback, roughly $T^{1/4}$, on the grounds that scalar evaluations
imply preferences but not the reverse.

**MaxMinLCB** [@pasztor2024bandits] treats choosing a pair as a game between
a leader and a follower: the leader picks a point that does well even
against the follower's best response, both judged by lower confidence bounds.
It counts regret in preference probability: a duel costs
$\big(\Prob(\vx^\star \succ \vx_t) + \Prob(\vx^\star \succ \vx'_t) - 1\big)/2$,
zero when both points are optimal. Its Theorem 6 gives, with probability at
least $1 - \delta$ and for all $T$ at once,
$R_T \le C_3\,\beta_T\sqrt{T\gamma_T} = O(\gamma_T\sqrt{T})$, where $\beta_T$
grows like $\sqrt{\gamma_T}$ and the constant $C_3 = (8 + 2\kappa)/\sqrt{\log(1 + 4/(\lambda\kappa))}$
contains the link-slope constant $\kappa$ of @sec-duel-link and the
regularization weight $\lambda$ of the kernelized logistic regression. The
abstract
calls the guarantee rate-optimal; it is $O(\gamma_T\sqrt{T})$, not
$O(\sqrt{\gamma_T T})$, a factor $\sqrt{\gamma_T}$ above the best known rates,
so "rate-optimal" holds at most relative to GP-UCB-type analyses (inference).
The analysis also restricts the choice to a set of plausible maximizers.

**MR-LPF** [@kayal2025bayesian] takes the batched route of scalar Bayesian
optimization. It runs in at most $\lceil\log_2\log_2 T\rceil + 1$ rounds of
growing length; within a round it duels pairs of maximal uncertainty among the
surviving candidates, without looking at the answers, and at the end of a
round it eliminates every candidate whose optimistic chance of beating some
other candidate is below one half. It assumes $f$ in an RKHS of norm at most
$B$, the logistic link, and a *finite* candidate set $\X$ of size $|\X|$. Its
Theorem 4.1 holds for $T \ge T_0$, a warm-up length that does not depend on
$T$ and is specified in the paper's appendix, and simplifies to

$$
R_T = \tilde O\!\left(\sqrt{\gamma_T\, T \log(|\X|/\delta)}\right),
$$ {#eq-duel-mrlpf}

in preference-probability regret, where $\tilde O$ hides logarithmic factors.
The link-slope constant enters only the first round, so it drops out of the
leading term. This is the same order as the best scalar results, which
contradicts POP-BO's reading at the level of upper bounds: the $T^{1/4}$ came
from POP-BO's covering-number confidence width, not from preference feedback
itself (inference). The authors note that the matching scalar lower bound
assumes Gaussian noise while the Bradley-Terry model corresponds to Gumbel
noise, so they offer the comparison as an informal argument for tightness,
not a proof.

**PF-TS** [@lazzaro2026finiteb] is Thompson sampling for preferences: two
independent posterior samples are each maximized against a common anchor
point. With probability at least $1 - 2\delta$ its regret, in preference
probability, is $\tilde O(\beta_T\sqrt{T\gamma_T})$ with
$\beta_T = O(\sqrt{\gamma_T + \log(1/\delta)})$, that is, $\tilde O(\gamma_T\sqrt{T})$,
which the authors note matches the bound of @chowdhury2017kernelized for
scalar Thompson sampling. A continuous domain is handled by a discretization,
and the kernel is assumed known. On a one-dimensional Ackley function (300
rounds, 30 runs) its cumulative regret was lower than MR-LPF's and POP-BO's and
comparable to MaxMinLCB's; the PF-TS and MR-LPF papers share two authors.

A fifth line replaces the kernel with a neural network. Neural dueling bandits
[@verma2025neural] hold for Bradley-Terry, Thurstone, and other links as long
as stochastic transitivity holds, with a bound in average utility regret that
depends on an effective dimension and on the minimum slope of the link, and
that the authors expect to be weaker than its scalar neural counterpart.

All of these analyze frequentist estimators, kernelized logistic regression
with confidence sets, and algorithms built for the analysis. None analyzes the
pipeline practitioners run, a Laplace-approximated Gaussian process posterior
with EUBO (@sec-pbo-loop), whose guarantees are the one-step Bayes optimality
and finite-domain consistency of @sec-eubo-theory (inference).

## The rates, side by side {#sec-dueling-rates}

@tbl-duel-rates puts the kernelized results next to each other with the
scalar results they echo. Two scalar references matter: GP-UCB and GP-TS
analyses give $O^*(\gamma_T\sqrt{T})$ [@chowdhury2017kernelized], and a batched
pure exploration algorithm reaches $O^*(\sqrt{\gamma_T T})$ within
$O(\log\log T)$ batches, near-optimal for several kernels [@li2022gaussian],
where $O^*$ hides logarithmic factors. The full table, with the finite-arm and
neural results, is @tbl-theory-rates.

::: {.table #tbl-duel-rates title="Kernelized dueling bounds: feedback, regret unit, main assumptions, rate, and whether the link-slope constant κ multiplies the leading term."}
| Result | Feedback | Regret unit | Main assumptions | Rate | κ in leading term | Scalar counterpart |
|---|---|---|---|---|---|---|
| @kirschner2021bias | utility difference plus sub-Gaussian noise | utility, both points | RKHS norm $\le B$ | $\approx \gamma_T\sqrt{T}$ | no (no link) | GP-UCB |
| POP-BO [@xu2024principledb] | logistic | utility | compact domain; Matérn needs $\nu$ of order $d^2$ | $O(\sqrt{\beta_T\gamma_T T})$, about $T^{3/4}$ | through the confidence set | weaker than GP-UCB |
| MaxMinLCB [@pasztor2024bandits] | logistic | preference probability | RKHS norm $\le B$ | $O(\gamma_T\sqrt{T})$ | yes | GP-UCB |
| MR-LPF [@kayal2025bayesian] | logistic | preference probability | finite $\X$; $T \ge T_0$; batched | $\tilde O(\sqrt{\gamma_T T\log \lvert\X\rvert})$ | first round only | batched pure exploration |
| PF-TS [@lazzaro2026finiteb] | logistic | preference probability | discretized domain; known kernel | $\tilde O(\gamma_T\sqrt{T})$ | through $\beta_T$, $\gamma_T$ | GP-TS |
:::

The pattern is that preferential theory has reproduced scalar theory almost
item by item: optimism and Thompson sampling reach $\gamma_T\sqrt{T}$, as
GP-UCB and GP-TS do, and batched elimination reaches $\sqrt{\gamma_T T}$, as
its scalar model does (inference). Whether a fully sequential preference
algorithm can reach $\sqrt{\gamma_T T}$ is open. For scalar feedback,
sequential algorithms that reach it exist [@salgia2021domain]; what was posed
as an open problem at COLT 2021 is whether GP-UCB itself can
[@vakili2021open], and that has been partly resolved
[@whitehouse2023sublinear].

The factor $\sqrt{\gamma_T}$ between the two families matters most in higher
dimensions. For a Matérn kernel with smoothness $\nu > 1/2$ in $d$
dimensions, $\gamma_T$ grows like $T^{d/(2\nu + d)}$ up to logarithmic factors
[@vakili2021information], the last row of @tbl-regret-gamma, so
$\gamma_T\sqrt{T}$ grows like $T^{1/2 + d/(2\nu + d)}$.
For the common Matérn 5/2 kernel the exponent is $1/2 + d/(5 + d)$, which
reaches 1 at $d = 5$: from five dimensions on, the sequential bounds no longer
say that regret grows more slowly than $T$, and so say nothing. The exponent of
$\sqrt{\gamma_T T}$ is $(\nu + d)/(2\nu + d)$, below 1 in every dimension and
equal to the exponent of the scalar lower bound of @scarlett2017lower
(inference, our arithmetic on the stated rates; @exr-duel-exponents). The
figure draws both.

```{figure}
//| figure: theory-rates
//| label: fig-duel-rates
//| fig-cap: "The kernelized rates side by side. Top: the shapes of the bounds on a one-dimensional domain with every constant and link factor set to 1, with γ_T estimated greedily; the levels are illustrative and the regret units differ between results, so only the growth is comparable. Bottom: the exponent a in T^a as the dimension d grows, from the published orders of γ_T for Matérn kernels, ignoring logarithmic factors (the order is stated for ν > 1/2; the Matérn 1/2 choice applies the same formula at the boundary). Above 1, a bound grows faster than T and guarantees nothing. The same figure appears in @sec-theory-rates."
```

### The slope of the link {#sec-duel-link}

The constant $\kappa$ that appears in several bounds measures how flat the
link function can get. A duel between a much better and a much worse option is
almost always won by the better one, so its answer is almost certain and
carries little information about how much better it is. Where the logistic
curve is flat, large changes in the utility difference produce small changes
in the answer, and learning there is slow. The constant is the reciprocal of
the smallest slope of the link over the range $[-D, D]$ of utility
differences the analysis must allow:

$$
\kappa = \sup_{\lvert a\rvert \le D} \frac{1}{\operatorname{sigmoid}'(a)}, \qquad \frac{1}{\operatorname{sigmoid}'(a)} = 2 + e^{a} + e^{-a}.
$$ {#eq-duel-kappa}

It grows exponentially with the range. Near zero the logistic slope is
$1/4$, so $\kappa$ is at least 4; if utilities may lie anywhere in
$[-5, 5]$, differences reach 10 and $\kappa$ exceeds 22,000
[@kayal2025bayesian]. A bound with $\kappa$ in its leading term can therefore
be vacuous for realistic ranges even when its dependence on $T$ looks good.

Scalar logistic bandits met the same problem first. @faury2020improved showed
that earlier guarantees of order $\kappa\sqrt{T}$ could be improved to order
$\sqrt{T}$ with $\kappa$ only in a second-order term, and @abeille2021instance
proved a problem-dependent lower bound of order $d\sqrt{T/\kappa}$ with a
matching upper bound: where the link is flat, the problem can even be easier,
because answers there are predictable. For duels, @di2025nearly removed
$\kappa$ from the leading term for linear utilities with the sigmoid link, and
MR-LPF did so for kernels; MaxMinLCB, PF-TS, and neural dueling bandits keep it
there.

### The unit of regret {#sec-duel-units}

The table mixes two units, and they are not interchangeable. *Utility regret*
counts $f(\vx^\star) - f(\vx_t)$, how much utility was lost.
*Preference-probability regret* counts $\Prob(\vx^\star \succ \vx_t) - 1/2$, how
much more often the best option would have won. The second saturates: a
terrible option and a bad one both lose almost surely, so both cost almost
$1/2$, while their utility losses can differ by any amount.

::: {.derivation title="How the two units compare"}
Write $\operatorname{sigmoid}$ for the logistic link and let
$a = f(\vx^\star) - f(\vx) \in [0, D]$ be a utility gap.

1. $\operatorname{sigmoid}(0) = 1/2$, so the preference-probability regret is
   $\operatorname{sigmoid}(a) - 1/2 = \int_0^a \operatorname{sigmoid}'(u)\,\dd u$.
2. The slope $\operatorname{sigmoid}'$ is largest at 0, where it equals $1/4$,
   and decreases on $[0, \infty)$. Bounding the integrand above by $1/4$ gives
   $\operatorname{sigmoid}(a) - 1/2 \le a/4$.
3. Because its slope decreases, $\operatorname{sigmoid}$ is concave on
   $[0, \infty)$ and lies above the straight line from $(0, 1/2)$ to
   $(D, \operatorname{sigmoid}(D))$:
   $\operatorname{sigmoid}(a) - 1/2 \ge a\,c_D$ with
   $c_D = \big(\operatorname{sigmoid}(D) - 1/2\big)/D$. The factor $c_D$ is
   always below $1/(2D)$ and close to it once $D \ge 4$.
4. So $a\,c_D \le \operatorname{sigmoid}(a) - 1/2 \le a/4$; the upper bound is
   approached as $a \to 0$ and the lower one is reached at $a = D$. One unit
   of preference-probability regret stands for about 4 units of utility regret
   when the gap is small, and for $1/c_D$ units, about $2D$, when the gap is
   as large as it can be. (Bounding the integrand below by its smallest
   value, $\operatorname{sigmoid}'(D) = 1/\kappa$ with $\kappa$ as in
   @eq-duel-kappa, gives the cruder
   $\operatorname{sigmoid}(a) - 1/2 \ge a/\kappa$, which is valid but far from
   tight: at $D = 10$ it allows a factor of 22,028 where the worst case is
   20.)
:::

Statements that a preferential algorithm is of the same order as scalar
Bayesian optimization usually compare preference-probability regret with
scalar utility regret. Near the optimum, where gaps are small, the comparison
is fair up to the factor 4; far from it, the factor grows with the gap, to
about $2D$ at the largest one (inference).

::: {.pitfall title="Three misreadings of the rates"}
**Misreading: MaxMinLCB is rate-optimal at $O(\sqrt{\gamma_T T})$.** Its Theorem 6 gives
$O(\gamma_T\sqrt{T})$, a factor $\sqrt{\gamma_T}$ above $\sqrt{\gamma_T T}$, with
$\kappa$ in the constant [@pasztor2024bandits].

**Misreading: MR-LPF proves that comparisons are as sample-efficient as evaluations.**
It proves an upper bound of the same order as the best scalar upper bound, in
preference-probability units, for a finite candidate set, after a warm-up
$T_0$, with a batched algorithm whose rounds ignore the answers until they end;
its tightness is argued informally [@kayal2025bayesian]. That is not a
statement about information per query, and no lower bound settles it.

**Misreading: with a Gaussian process prior, SelfSparring needs $O(d)$ instead
of $O(K)$ samples.** A 2018 survey states that a Gaussian process prior reduces
the sample complexity from $O(K)$ to $O(d)$ [@sui2018advancements], but the
SelfSparring
paper proves no such theorem and states that the analysis of its kernelized
version is lacking [@sui2017multi]. It is a conjecture.
:::

## What is missing {#sec-dueling-lower-bounds}

A lower bound says how much regret every algorithm must pay on some problem,
and so whether an upper bound can be improved (@sec-regret-lower-bounds). For
duels among finitely many arms they are known: the bounds of @yue2012karmed and
@komiyama2015regret above, matched by algorithms. For linear utilities they are
known too: order $\sqrt{dT}$ for choosing subsets with winner feedback, whatever
the subset size [@saha2021optimal]. For the kernelized problem with a
Bradley-Terry or probit link, the one PBO
poses, we found no algorithm-independent lower bound as of September 2026
(@sec-theory-lower).

The scalar problem has them. For a Matérn kernel, every algorithm suffers
cumulative regret of order at least $T^{(\nu + d)/(2\nu + d)}$ on some
function of bounded RKHS norm, and needs at least $(1/\varepsilon)^{2 + d/\nu}$
evaluations to find an $\varepsilon$-optimal point [@scarlett2017lower]. The
only bridge to comparisons is the informal argument of @kayal2025bayesian: if
two noisy evaluations are turned into one comparison by keeping only which is
larger, the comparison cannot be more informative than the two evaluations, so
a preference lower bound should be at least half the scalar one under the
matching noise. The scalar bound assumes Gaussian noise, and the Bradley-Terry
model corresponds to Gumbel noise, so the argument is not a proof.

Three lower bounds are missing, in order of how much their absence limits the
theory (inference): a lower bound for kernelized duels under a nonlinear link;
a lower bound that says how the slope constant $\kappa$ must enter kernelized
preference regret; and a kernelized lower bound for rankings and choices from
sets, which would say whether the finite-arm finding of @sec-query-batch,
that only richer answers than the winner help, carries over.

So is a comparison more expensive than a number? The honest answer depends on
the setting (inference). For finitely many arms and for linear utilities,
dueling rates match the rates for numerical rewards up to constants and the
link factor. For kernels, the best upper bounds have the same order, under
the assumptions of MR-LPF, and no lower bound says whether that is tight. And
one binary answer carries at most one bit while a noisy number carries only a
limited amount too, so equal rates do not mean equal information per query.

::: {.frontier title="Settled, contested, missing"}
**Settled.** For finitely many arms, optimal logarithmic regret and matching
lower bounds are known against a Condorcet winner, and Copeland winners have
algorithms of order $K\log T$. For linear utilities, order $\sqrt{dT}$ is
optimal. Kirschner and Krause (2021) gave the first kernelized cumulative bound
for duels, with a difference-plus-noise feedback model; under the logistic
link, POP-BO, MaxMinLCB and PF-TS, and MR-LPF give rates of about $T^{3/4}$,
$\gamma_T\sqrt{T}$, and $\sqrt{\gamma_T T}$, the last for a finite candidate
set after a warm-up.

**Contested.** Whether MR-LPF's rate is tight, which its authors argue only
informally. Whether comparisons are as efficient as evaluations, which holds
only for upper bounds under specific assumptions. Batched and order-optimal
against sequential and a factor $\sqrt{\gamma_T}$ worse: the one direct
comparison comes from overlapping authors, in low dimension.

**Missing.** Kernelized lower bounds under a nonlinear link, with the role of
$\kappa$; a fully sequential preference algorithm with $\sqrt{\gamma_T T}$
regret; bounds
for rankings and choices from sets with kernels; and any frequentist analysis
of the Laplace-plus-EUBO pipeline that practitioners use.
:::

@sec-pbo-theory reports this theory in full, including drift, contamination,
response times, and the identifiability results behind @sec-query-aggregation.

## Exercises {#sec-duel-exercises}

::: {.exercise #exr-duel-utility}
Suppose $P_{ij} = s(u_i - u_j)$ for utilities $u_1 > u_2 > \dots > u_K$ and a
continuous, strictly increasing link $s$ with $s(-a) = 1 - s(a)$. (a) Show that
arm 1 is the Condorcet, Copeland, and Borda winner, and that the von Neumann
winner puts all its weight on it. (b) Show that strong stochastic transitivity
holds.

::: {.solution}
(a) $s(0) = 1/2$ by symmetry, and $s$ is strictly increasing, so
$P_{1j} = s(u_1 - u_j) > 1/2$ for every $j \ne 1$: arm 1 is the Condorcet
winner, and with $K - 1$ wins it has the highest possible Copeland score. For
Borda, compare arm 1 with any arm $k$. For every third arm $j$,
$s(u_1 - u_j) > s(u_k - u_j)$; and in their direct duel,
$s(u_1 - u_k) > 1/2 > s(u_k - u_1)$. Each term of arm 1's average exceeds the
matching term of arm $k$'s, so arm 1 has the higher Borda score. For von
Neumann, the point mass on arm 1 gives $\sum_i \pi_i P_{ij} = P_{1j} \ge 1/2$ for
every $j$, so it qualifies; it is also the only one, because any weight on
another arm $i$ loses to arm 1 on average.

(b) If $\Delta_{ij} \ge 0$ and $\Delta_{jk} \ge 0$, then $u_i \ge u_j \ge u_k$.
So $u_i - u_k \ge u_i - u_j$ and $u_i - u_k \ge u_j - u_k$, and since $s$ is
increasing, $\Delta_{ik} \ge \max\{\Delta_{ij}, \Delta_{jk}\}$.
:::
:::

::: {.exercise #exr-duel-rps}
Three arms form a cycle, oriented as in @fig-duel-winners: B beats A with
probability $1/2 + a$, C beats B with probability $1/2 + b$, and A beats C with
probability $1/2 + c$, with $a, b, c > 0$. (a) Show that there is no Condorcet
winner and that all three arms tie on Copeland score. (b) Show that the von
Neumann winner is $\boldsymbol{\pi} = (b, c, a)/(a + b + c)$ on (A, B, C).
(c) With $a = 0.15$, $b = 0.18$, $c = 0.46$, which arm gets the most weight,
and what pattern do the weights follow?

::: {.solution}
(a) Each arm beats exactly one other arm and loses to the other, so no arm
beats both, and each has Copeland score 1.

(b) With payoffs $M_{ij} = P_{ij} - 1/2$, the conditions are
$\sum_i \pi_i M_{ij} \ge 0$ for each column $j$. The nonzero payoffs are
$M_{BA} = a$, $M_{CB} = b$, $M_{AC} = c$ and their negatives. Column A:
$\pi_B M_{BA} + \pi_C M_{CA} = a\pi_B - c\pi_C$. Column B:
$\pi_A M_{AB} + \pi_C M_{CB} = -a\pi_A + b\pi_C$. Column C:
$\pi_A M_{AC} + \pi_B M_{BC} = c\pi_A - b\pi_B$. With
$\boldsymbol{\pi} \propto (b, c, a)$ these are $ac - ca = 0$, $-ab + ba = 0$,
and $cb - bc = 0$: every column ties, and the weights are positive and sum to
one after normalization.

(c) The weights are $(0.18, 0.46, 0.15)/0.79 \approx (0.23, 0.58, 0.19)$, so B
gets the most. Each arm's weight is the margin of the duel it is not part of:
B's weight is the margin by which A beats C. These are, up to rounding, the
margins and the weights of @fig-duel-winners at its default setting.
:::
:::

::: {.exercise #exr-duel-exponents}
For a Matérn kernel with smoothness $\nu$ in $d$ dimensions, take
$\gamma_T \propto T^{d/(2\nu + d)}$ and ignore logarithmic factors. (a) Show that
the exponent of $T$ in $\sqrt{\gamma_T T}$ equals that of the scalar lower
bound, $(\nu + d)/(2\nu + d)$. (b) Show that $\gamma_T\sqrt{T}$ grows at least as
fast as $T$ exactly when $d \ge 2\nu$. (c) For Matérn 5/2, at which dimension
do the MaxMinLCB and PF-TS bounds stop being sublinear?

::: {.solution}
(a) $\sqrt{\gamma_T T} \propto T^{\frac12(1 + d/(2\nu + d))} = T^{(2\nu + 2d)/(2(2\nu + d))} = T^{(\nu + d)/(2\nu + d)}$.

(b) The exponent of $\gamma_T\sqrt{T}$ is $1/2 + d/(2\nu + d)$, which is at least
1 when $d/(2\nu + d) \ge 1/2$, that is, $2d \ge 2\nu + d$, or $d \ge 2\nu$.

(c) With $\nu = 5/2$, at $d = 5$: from five dimensions on, the bounds grow at
least linearly and guarantee nothing, while $\sqrt{\gamma_T T}$ remains
sublinear in every dimension.
:::
:::

::: {.exercise #exr-duel-kappa}
For the logistic link $\operatorname{sigmoid}(a) = 1/(1 + e^{-a})$, verify that
$1/\operatorname{sigmoid}'(a) = 2 + e^{a} + e^{-a}$, and compute $\kappa$ when utility differences may
reach 2, 6, and 10. Using the derivation in @sec-duel-units, by how much can
preference-probability regret understate utility regret for a gap of 10?

::: {.solution}
$\operatorname{sigmoid}'(a) = e^{-a}/(1 + e^{-a})^2 = 1/\big((1 + e^{-a})(1 + e^{a})\big) = 1/(2 + e^{a} + e^{-a})$.
So $\kappa$ is about $2 + 7.39 + 0.14 = 9.5$ at 2, $2 + 403.4 = 405$ at 6, and
$2 + 22026.5 = 22{,}028$ at 10. For a gap of 10, $\operatorname{sigmoid}(10) - 1/2 \approx 0.49995$,
while the utility gap is 10: the ratio is about 20, the upper end
$1/c_D = D/(\operatorname{sigmoid}(D) - 1/2) \approx 2D$ of the derivation and far below
$\kappa = 22{,}028$. A bad option costs almost the same $1/2$
in preference-probability regret however bad it is.
:::
:::

## Further reading {#further-reading .unnumbered}

- @yue2012karmed define the K-armed dueling bandit problem, its regret, and
  Interleaved Filter, with the matching lower bound.
- @sui2018advancements survey the algorithms (IF, RUCB, MergeRUCB, RMED, D-TS,
  Sparring, SelfSparring) and the alternative winners; @bengs2021preference is
  the longer survey, organized by assumptions on the preference matrix.
- @zoghi2014relative and @wu2016double are the original RUCB and Double
  Thompson Sampling papers; @zoghi2015copeland treats Copeland winners.
- @kirschner2021bias, @pasztor2024bandits, @kayal2025bayesian, and
  @lazzaro2026finiteb are the kernelized bounds, each stating its feedback
  model and regret unit precisely.
- @scarlett2017lower gives the scalar lower bounds that any preference lower
  bound will be compared with; @faury2020improved explains the link-slope
  constant in logistic bandits.
