---
status: done
synopsis: "What is proved about learning from comparisons: the finite-arm and linear dueling-bandit results, the kernelized regret bounds of 2021 to 2026 with their links, assumptions, and regret units, the decision-theoretic results for EUBO, the missing lower bounds, the theory of the observation model, identifiability, and drift, contamination, response times, and stopping."
---

# Theory: From Dueling Bandits to Kernelized Preference Optimization {#sec-pbo-theory}

How many comparisons does it take to find the best option, and does a
comparison teach as much as a number? @sec-dueling-bandits introduced the
theory of duels and its kernelized bounds; this chapter reports it in full,
with each result's assumptions, and says where the proofs stop. Of the four
layers of theory, finite-arm dueling bandits, linear and contextual dueling
bandits, continuous convex dueling optimization, and kernelized preference
bandits, only the last corresponds to preferential Bayesian optimization (PBO)
itself, and its results came last.

The state of things as of September 2026, in three sentences. Kernelized upper
bounds for a batched algorithm have reached the same order as order-optimal
scalar Bayesian optimization, but no lower bound exists for any kernelized
preference problem. The pipeline used in practice, the Laplace approximation
with EUBO, has only one-step Bayes optimality and finite-domain consistency.
And identification and aggregation theory shows that pairwise data are the
weakest form of feedback when people differ or answers depend on hidden
context.

::: {.note title="Notation used in this chapter"}
- $T$: the number of rounds (queries). $K$: the number of arms (options) in a
  finite problem. $d$: the input or feature dimension.
- $\gamma_T$: the maximum information gain of the kernel after $T$
  observations (@def-regret-gamma, @tbl-regret-gamma).
- $B$: a bound on the utility's norm in the **reproducing kernel Hilbert
  space** (RKHS) of the kernel, a measure of how rough the function is relative
  to the kernel (@sec-regret-other-settings, @sec-ka-rkhs).
- $\kappa$: the reciprocal of the smallest slope of the link function over the
  relevant range of utilities (@sec-theory-link-slope).
- $\tilde O$: order of growth ignoring logarithmic factors. $\Omega$: a lower
  bound on the order.
- **Two units of regret.** *Utility regret* sums $f(\vx^\star) - f(\vx_t)$.
  *Preference-probability regret* sums $\Prob(\vx^\star \succ \vx_t) - 1/2$,
  averaged over the two points of a query; @sec-duel-units explains why the
  two are not interchangeable.
:::

## Finite-arm and linear dueling bandits {#sec-theory-finite}

**Before 2017.** The finite-arm results of @sec-dueling-algorithms anchor the
theory: the Interleaved Filter's expected regret $O(K \log T / \Delta_{\min})$,
matching its lower bound under strong stochastic transitivity and the
stochastic triangle inequality [@yue2012karmed], and RMED, which matches an
asymptotic lower bound expressed through the Kullback-Leibler divergence
between Bernoulli distributions (@sec-kl) [@komiyama2015regret]. Dueling bandit
gradient descent had expected regret of order $T^{3/4}$ on a continuous convex
space [@yue2009interactively], and contextual dueling bandits introduced the
*von Neumann winner*, a randomized choice over options that beats any single
option with probability at least one half [@dudik2015contextual].

**2017 and after.** The paper of González et al. [@gonzalez2017preferentialb] has no regret or convergence
theorem (@sec-history-2017). SelfSparring [@sui2017multi] assumes "approximate
linearity", that the winning probability is a function of the utility
difference that is approximately linear, which the authors call a stricter
requirement than strong stochastic transitivity. Their Theorem 1 proves that the
independent-arm version converges to the best arm, and their Theorem 2 gives an
asymptotically optimal no-regret rate of $O(K \ln(T)/\Delta)$; an analysis of
kernelized multi-dueling is missing, and a 2018 survey's statement that a
Gaussian process prior reduces the sample complexity from $O(K)$ to $O(d)$
[@sui2018advancements] is a conjecture (@sec-duel-units). Winner Stays [@chen2017dueling] has *weak regret*, which
counts a round as free if either option shown is the best, of $O(N^2)$
independent of $T$, for $N$ arms. The survey of @bengs2021preference organizes
regret and *PAC* (probably approximately correct) sample-complexity results by
the assumptions they make about the matrix of pairwise winning probabilities.
Later, @saha2022versatile were the first to reach the optimal
$O(\sum_i \log T / \Delta_i)$ against a Condorcet-winner benchmark.

**Continuous convex dueling.** @kumagai2017regret obtained regret
$O(\sqrt{T \log T})$ with stochastic mirror descent when the cost is strongly
convex and smooth, and used lower bounds from convex optimization to argue that
this is optimal up to logarithmic factors; the abstract does not state the link
assumption. @saha2021duelingb gave the query complexity when each pair yields
only a single noisy comparison bit, and proved the non-stationary online convex
case impossible; their 2025 paper handles general transfer functions
[@saha2025dueling]. @blum2024dueling proved an $\Omega(d)$ lower bound under a
monotone adversary.

**Linear and contextual dueling.** In the setting of @saha2021optimal, each
round offers $K$ items with context features, the learner picks a subset of
$q$ of them, and a noisy winner is observed. They gave an optimal
$\tilde O(\sqrt{dT})$ algorithm with a matching $\Omega(\sqrt{dT})$ lower bound,
and the lower bound is independent of the subset size $q$: winner feedback from
a larger subset does not help. Efficient algorithms and general links followed
[@saha2022efficient; @bengs2022stochastic; @di2024variance], and Feel-Good
Thompson sampling reached a near-minimax $\tilde O(d\sqrt{T})$ [@li2024feel]. @wu2024borda proved an
$\Omega(d^{2/3} T^{2/3})$ lower bound and a matching upper bound for *Borda
regret*, which measures an option by its average winning probability against
all others. @di2025nearly obtained $\tilde O(\kappa d\sqrt{T} + \kappa dC)$
with $C$ adversarially flipped labels (their paper writes $\kappa$ for the
smallest slope of the link, the reciprocal of ours, and so divides by it), with
a nearly matching lower bound, and for the sigmoid link removed $\kappa$ from
the main term. And @sekhari2023contextual proved that comparison queries alone
can achieve regret comparable to a standard contextual bandit that observes
rewards. The difference between Saha's $\sqrt{dT}$ and Feel-Good Thompson
sampling's $d\sqrt{T}$ comes from the arm sets, $K$ items per round against
large or continuous sets, and is not a contradiction (inference).

**Reinforcement learning from preferences.** For learning policies from
preferences over trajectories (@sec-adj-pbrl), results run from asymptotic
Bayesian no-regret [@novoseller2020dueling] and a first finite-time analysis
[@xu2020preference] to general function approximation [@chen2022human] and a
first Bayesian simple-regret guarantee [@agnihotri2026best].
@zhu2023principled proved that under the Plackett-Luce model both the full
$K$-wise maximum likelihood estimator and the one that splits rankings into
pairs converge, the former asymptotically more efficiently.

## The first kernelized result {#sec-theory-kernelized}

One kernel-based result came before 2021: @xu2020zeroth allowed both direct
queries and duels, and their COMP-GP-UCB has simple regret $O(\Phi/\sqrt{T})$
after $T$ direct queries, where $\Phi$ is an information gain computed on the
part of the domain that the comparisons leave as candidates for the optimum.
The first kernelized dueling bound on cumulative regret, by @kirschner2021bias
(@sec-duel-kk), rests on assumptions that decide what it covers:

1. **Feedback.** Their Equation 2 is *quantitative* dueling feedback,
   $d_t = f(\vx_t^1) - f(\vx_t^2) + \xi_t$, with $\xi_t$ sub-Gaussian noise of
   variance proxy $\rho^2$. The model covers binary feedback only in the sense
   that bounded noise is sub-Gaussian.
2. **Function class.** $f$ lies in a known RKHS with norm at most $B$, and
   $k(\vx, \vx) \le 1$.
3. **Regret.** The sum of the utility gaps of both points of each duel.
4. **Theorem 1.** Regret $O\big(\sqrt{T\, \beta_{T,\delta}\,(\gamma_T + \log 1/\delta)}\big)$,
   roughly $\gamma_T\sqrt{T}$ for $T$ rounds (the paper writes $n$).
5. **Theorem 2.** On a finite domain with a unique optimum,
   $O\big(\Delta_{\min}^{-1}\beta(\gamma_T + \log(T/\delta))\big)$; for the linear
   kernel $O(\Delta_{\min}^{-1} d^2 \log(T)^2)$, and for the RBF (squared
   exponential) kernel $O(\Delta_{\min}^{-1} \log(T)^{2d+2})$.

The result does not cover Bernoulli outcomes under a Bradley-Terry or probit
link, nor the cost $\kappa$ that such links bring; kernelized regret under the
Bradley-Terry link begins with POP-BO in 2024 (inference, from the feedback
model the theorems state).

## Bradley-Terry bounds, 2024 to 2026 {#sec-theory-bt}

Four algorithms analyze the setting that PBO actually faces: Bernoulli answers
whose probability is the logistic function of a utility difference
(@sec-bradley-terry), with the utility in an RKHS. @sec-duel-bt-bounds
introduced them; here are the theorems with their constants and conditions.

**POP-BO.** @xu2024principledb assume a compact domain, $f$ in an RKHS, and
Bernoulli feedback with $\Prob = \operatorname{sigmoid}(f(\vx) - f(\vx'))$, the
logistic function of @eq-cmp-logit. The algorithm chooses optimistically within a
likelihood-ratio confidence set and uses the previous round's point as the
reference point.

- **Theorem 5.2.** Utility regret $R_T = O\big(\sqrt{\beta_T \gamma_T T}\big)$,
  where $\beta_T = O\big(\sqrt{T \log(T\, \mathcal{N}(\mathcal{B}_f, 1/T, \lVert\cdot\rVert_\infty)/\delta)}\big)$
  and $\mathcal{N}$ is the covering number of the function class, the number of
  small balls needed to cover it. For the linear and RBF kernels
  this gives $T^{3/4}$ times polylogarithmic factors (their Theorem 5.5). For a
  Matérn kernel the exponent is larger than $3/4$, and the bound is stated only
  for $\nu > (d/4)\big(3 + d + \sqrt{d^2 + 14d + 17}\big)$, that is, when the
  smoothness parameter $\nu$ is of order $d^2$.
- **Theorem 5.4.** The gap of the reported solution is
  $O\big(\sqrt{\beta_T \gamma_T}/\sqrt{T}\big)$.
- **Remark 5.6.** The authors suggest that preference feedback costs about an
  extra factor of $T^{1/4}$, on the intuition that a numerical evaluation
  implies a preference but not the reverse.

Later papers often abbreviate POP-BO's rate as $\tilde O((\gamma_T T)^{3/4})$;
when quoting it, say that its unit is utility regret.

**MaxMinLCB.** @pasztor2024bandits frame the choice of a pair as a
*Stackelberg game*, a game in which one player commits first and the other
responds, and build preference confidence sequences for a kernelized logistic
estimator. **Theorem 6**: with probability at least $1 - \delta$, for all $T$,

$$
R_T \le C_3\, \beta_T \sqrt{T \gamma_T} = O(\gamma_T \sqrt{T}),
$$ {#eq-theory-maxmin}

with $\beta_t = 4LB + 2L\sqrt{(2\kappa/\lambda)(\gamma_t + \log 1/\delta)}$,
$\kappa = \sup_{\lvert a\rvert \le B} 1/\operatorname{sigmoid}'(a)$, and
$C_3 = (8 + 2\kappa)/\sqrt{\log(1 + 4/(\lambda\kappa))}$, where $s$ is the link
function, $L$ its Lipschitz constant (a bound on its slope), and $\lambda$ the
regularization parameter. The regret is preference-probability regret, and
"rate-optimal" in the abstract holds at most relative to GP-UCB-type analyses
(inference; @sec-duel-bt-bounds). @kayal2025bayesian summarize it as
$\tilde O(\gamma_T \kappa^2 \sqrt{T})$.

**MR-LPF.** The multi-round learning from preference-based feedback algorithm
of @kayal2025bayesian assumes $f$ in the RKHS of a known kernel with norm at
most $B$, a kernel bounded by 1, the Bradley-Terry (logistic) link only, and a
**finite** candidate set $\X$. It runs in $R \le \lceil \log_2 \log_2 T\rceil + 1$
rounds of lengths $N_1 = \sqrt{T}$ and $N_r = \sqrt{N_{r-1} T}$, picking pairs by
largest kernel variance within a round and eliminating, at its end, every point
whose upper confidence bound on the probability of beating some opponent is
below one half.

- **Theorem 4.1.** There is a constant $T_0$, independent of $T$ (given in
  their Appendix B), such that for all $T \ge T_0$, with probability at least
  $1 - \delta$,
  $R_T \le 2CR\, \beta_{(R)}(\delta) \sqrt{\gamma_{4\lambda}(T)}\,(\sqrt{T} + 1)$,
  with $\beta_{(r)}(\delta) = L\big(B + \sqrt{(\kappa_r/\lambda)\log(2R|\X|/\delta)}\big)$,
  where $|\X|$ is the size of the candidate set $\X$ (the paper writes $N_\X$), $\kappa_1 = \kappa$, and
  $\kappa_r = 6$ for $r > 1$. Simplified:
  $\tilde O\big(\sqrt{\gamma_T T \log(|\X|/\delta)}\big)$.
- **Where $\kappa$ goes.** It appears only in the first round and so drops out
  of the main term. The paper notes that when the utility takes values in
  $[-5, 5]$, $\kappa$ can exceed 22,000.
- **Corollary 4.5.** The number of comparisons needed to find a solution with
  $\Prob(\vx^\star \succ \hat\vx) - 1/2 \le \varepsilon$ is
  $\tilde O(d \log(1/\delta)/\varepsilon^2)$ for the linear kernel,
  $\tilde O(\log(1/\delta)/\varepsilon^2)$ for the RBF kernel,
  and $\tilde O\big(\log(1/\delta)/\varepsilon^{2 + d/\nu}\big)$ for the Matérn
  kernel, the same order as the order-optimal sample complexity with scalar
  feedback.
- **Tightness.** The authors note that the lower bound of Scarlett et al.
  assumes Gaussian noise while Bradley-Terry corresponds to Gumbel noise, so a
  formal comparison does not strictly hold; they offer it only as an informal
  argument for tightness, and argue that a lower bound for preference feedback
  should be at least half the scalar one.

Six differences from the scalar setting should travel with any statement that
MR-LPF "matches scalar Bayesian optimization" (@sec-duel-units lists the common
misreadings): the regret unit is preference probability; whether $T_0$ hides a
dependence on $\kappa$ or $e^B$ has not been checked; the $\log |\X|$ comes
from a union bound, so a continuous domain needs a discretization argument; the
algorithm is batched and non-adaptive within a round; the optimality is an
informal comparison; and every query involves two points, both counted in the
regret. A machine-generated review site claims that
a Loewner-order inequality used in their Theorem 4.7 may fail for small
$\lambda$ [@pith2026arxiv]; that claim is not peer reviewed and has not been
confirmed by a human source, and the PF-TS paper cites MR-LPF's rate as
correct.

**PF-TS.** @lazzaro2026finiteb analyze Thompson sampling with preference
feedback: the two points of a query are obtained by maximizing two independent
posterior samples against a common anchor point. **Theorem 1**: with
probability at least $1 - 2\delta$,
$R_T = \tilde O\big(\beta_T\sqrt{T\gamma_T}\big)$ with
$\beta_T = O\big(\sqrt{\gamma_T + \log(1/\delta)}\big)$, that is,
$\tilde O(\gamma_T\sqrt{T})$, in preference-probability regret; $\kappa$ enters
$\beta_T$ and $\gamma_T$ through a ridge term $\lambda\kappa$. The paper says
the bound matches the one Chowdhury and Gopalan established for standard
Thompson sampling in 2017, which is itself not order-optimal in scalar Bayesian
optimization. A continuous domain must be discretized into $(B G w d T^2)^d$
points, the kernel is assumed known, and the experiments are a one-dimensional
Ackley function and a catalyst data set of 63 compositions of three metals. On
the Ackley function, where PF-TS had lower cumulative regret than MR-LPF and
POP-BO (@sec-duel-bt-bounds), MR-LPF's instantaneous regret was competitive at
the horizon of 300 rounds; the two papers share authors (Vakili, Shiu).

**Neural dueling bandits.** @verma2025neural assume
$\kappa_\mu = \inf \mu'(f(\vx) - f(\vx')) > 0$ for the link $\mu$, and their
result applies to Bradley-Terry, Thurstone, and exponential noise as long as
stochastic transitivity holds. Their average utility regret is
$\tilde O\big((\sqrt{d_{\text{eff}}}/\kappa_\mu + B\sqrt{\lambda/\kappa_\mu})\sqrt{T d_{\text{eff}}}\big)$,
where $d_{\text{eff}}$ is an effective dimension built from all pairwise context
differences, and the network width must be polynomial in quantities such as
$T$; the authors expect the bound to be weaker than for scalar neural bandits.
@oh2026neural gave $\tilde O\big(d\sqrt{\sum_t \sigma_t^2} + \sqrt{dT}\big)$ and
reduced the width requirement to $\tilde\Omega(T^6)$.

## The rates compared {#sec-theory-rates}

@tbl-theory-rates collects the main results for continuous or kernelized
preference optimization next to the scalar Bayesian optimization references:
$O^*(\sqrt{T}\gamma_T)$ for GP-UCB- and GP-TS-type analyses
[@chowdhury2017kernelized]; $O^*(\sqrt{T\gamma_T})$ within $O(\log\log T)$
batches for batched pure exploration (BPE), near-optimal for several kernels
[@li2022gaussian]; and, for the Matérn kernel, lower bounds of
$\Omega(T^{(\nu + d)/(2\nu + d)})$ on cumulative regret and
$\Omega((1/\varepsilon)^{2 + d/\nu})$ on the simple-regret sample complexity
[@scarlett2017lower]. Here $O^*$, like $\tilde O$, hides logarithmic factors.

::: {.table #tbl-theory-rates title="Rates for continuous and kernelized preference optimization, with their feedback, regret unit, assumptions, and the scalar result each corresponds to."}
| Result | Feedback and link | Regret | Main assumptions | Rate | $\kappa$ in main term | Scalar counterpart |
|---|---|---|---|---|---|---|
| SelfSparring [@sui2017multi] | multi-duel; approximately linear link | finite-arm strong regret | independent arms | asymptotic $O(K\ln T/\Delta)$; no bound for the kernel version | not applicable | same asymptotic order as finite-arm bandits |
| Kumagai [@kumagai2017regret] | noisy comparisons; strongly convex smooth cost | dueling regret | strongly convex, smooth | $O(\sqrt{T\log T})$ | not stated in the abstract | optimal up to log factors in the sense of convex lower bounds |
| Xu et al. [@xu2020zeroth] | duels plus direct queries | simple regret | RKHS | $O(\Phi/\sqrt{T})$ | not applicable | GP-UCB type, with information gain on a comparison-based constraint set |
| Kirschner and Krause [@kirschner2021bias] | utility difference plus sub-Gaussian noise (linear link) | sum of both points' utility gaps | norm $\le B$ | $O(\sqrt{T\beta_T(\gamma_T + \log 1/\delta)})$, about $\gamma_T\sqrt{T}$ | no | same form as GP-UCB |
| POP-BO [@xu2024principledb] | logistic | utility; reference is the previous point | compact domain; Matérn needs $\nu$ of order $d^2$ | $O(\sqrt{\beta_T\gamma_T T})$, about $T^{3/4}$ times polylog | through the confidence set | weaker than GP-UCB |
| MaxMinLCB [@pasztor2024bandits] | logistic; a footnote says the analysis may extend to other symmetric increasing links | preference probability | norm $\le B$ | $O(\gamma_T\sqrt{T})$ | yes (about $\kappa^2$) | same order as GP-UCB |
| Neural dueling bandits [@verma2025neural] | general link | average utility | polynomial network width | $\tilde O((\sqrt{d_{\text{eff}}}/\kappa_\mu)\sqrt{Td_{\text{eff}}})$ and further terms | yes | expected by the authors to be weaker than NeuralUCB |
| MR-LPF [@kayal2025bayesian] | logistic | preference probability | finite $\X$; $T \ge T_0$; batched | $\tilde O(\sqrt{\gamma_T T\log |\X|})$ | first round only | same order as BPE; tightness argued informally |
| PF-TS [@lazzaro2026finiteb] | logistic | preference probability | continuous domain discretized; kernel known | $\tilde O(\gamma_T\sqrt{T})$ | through $\beta_T$ and $\gamma_T$ | same order as GP-TS |
| qEUBO [@astudillo2023qeubob] | logistic or constant likelihood | Bayesian simple regret | finite $\X$; $q = 2$; gap or constant-likelihood conditions | $o(1/n)$ | not applicable | a Bayesian finite-domain result, not comparable with frequentist rates |
:::

Preferential theory has almost reproduced scalar Bayesian optimization item by
item: optimistic algorithms and Thompson sampling reach $\gamma_T\sqrt{T}$, and
batched elimination reaches $\sqrt{\gamma_T T}$ (inference;
@sec-dueling-rates). Whether a fully sequential preference algorithm can reach
$\sqrt{\gamma_T T}$ is open. The scalar setting had a related problem, posed at
COLT 2021 [@vakili2021open]: whether GP-UCB itself can reach that rate. More
elaborate scalar algorithms already attain it [@salgia2021domain], and
@whitehouse2023sublinear partly resolved the question with a sublinear bound
for GP-UCB under Matérn kernels, of order $T^{(\nu + 2d)/(2\nu + 2d)}$ up to
logarithmic factors, still above the lower bound; the improved information-gain
rates for Matérn kernels are those of @vakili2021information, stated for
$\nu > 1/2$. How much the factor $\sqrt{\gamma_T}$ between the two families
matters depends on how fast $\gamma_T$ grows, which depends on the kernel
(@tbl-regret-gamma), and @fig-theory-rates makes this concrete.

```{figure}
//| figure: theory-rates
//| label: fig-theory-rates
//| fig-cap: "The kernelized rates side by side, the figure of @sec-dueling-rates. Top: the shapes of the bounds on a one-dimensional domain, with $\gamma_T$ computed by the greedy rule of @sec-regret-info-gain on a grid of 300 inputs with regularization 0.25, and every constant and link factor $\kappa$ set to 1; the levels are illustrative and the units differ, so only the growth is comparable. The dashed line $T$ is the growth of a learner that never improves. Bottom: the exponent $a$ in $T^a$ as the dimension grows, our arithmetic on the published orders of $\gamma_T$ (@tbl-regret-gamma), ignoring logarithmic factors; the order is stated for $\nu > 1/2$ [@vakili2021information], and the Matérn 1/2 choice applies the same formula at the boundary."
```

Some things to try:

- **The default (Matérn 5/2, $T = 1000$).** The bracket at the right shows the
  factor $\sqrt{\gamma_T}$ between $\gamma_T\sqrt{T}$ and $\sqrt{\gamma_T T}$.
  With every constant set to 1, $\gamma_T\sqrt{T}$ is still above the line $T$
  at this horizon, while $\sqrt{\gamma_T T}$ is far below it: the extra
  $\sqrt{\gamma_T}$ is the difference between a bound that says something and
  one that does not, even before constants.
- **Switch to the squared exponential kernel.** $\gamma_T$ grows like a power of
  $\log T$, so asymptotically the two families differ only by polylogarithmic
  factors, and POP-BO's $T^{3/4}$ appears. Asymptotically it is the
  fastest-growing of the three, yet at these horizons it lies below
  $\gamma_T\sqrt{T}$ and close to $\sqrt{\gamma_T T}$: order of growth and size
  at a practical horizon can disagree.
- **Switch to Matérn 1/2 and lengthen the lengthscale.** A rough kernel makes
  $\gamma_T$ grow almost like $\sqrt{T}$ in one dimension, so the exponent of
  $\gamma_T\sqrt{T}$ is already 1 at $d = 1$. A longer lengthscale lowers
  $\gamma_T$ at a fixed horizon but does not change the exponents.
- **Read the bottom panel for Matérn 5/2.** The exponent of $\gamma_T\sqrt{T}$
  is $1/2 + d/(5 + d)$, which reaches 1 at $d = 5$; the exponent of
  $\sqrt{\gamma_T T}$ is $(2.5 + d)/(5 + d)$, below 1 in every dimension
  (@exr-duel-exponents). POP-BO's Matérn result needs $\nu > 2.41$ even at
  $d = 1$, so for Matérn 5/2 it applies only in one dimension.

## Link slope and regret units {#sec-theory-link-slope}

The constant $\kappa$ is the reciprocal of the smallest slope of the link over
the range of utility differences the analysis allows. For the logistic link it
grows exponentially with that range, because the logistic curve is nearly flat
far from zero: MR-LPF's authors note that it can exceed 22,000 for utilities in
$[-5, 5]$ (@exr-duel-kappa). @sec-duel-link tells how scalar logistic bandits
moved $\kappa$ out of the main term [@faury2020improved; @abeille2021instance],
and @sec-duel-units why utility regret and preference-probability regret are
proportional only for small gaps, with the factor $1/4$ (@exr-theory-units).
Two points belong here. For dueling, @di2025nearly (for the sigmoid link) and
MR-LPF (for kernels) removed $\kappa$ from the main term, while MaxMinLCB,
PF-TS, and neural dueling bandits keep it in the main term's constant; apart
from the neural $\kappa_\mu$, no kernelized result gives an explicit slope
dependence for a general, non-logistic link, and MaxMinLCB only remarks in a
footnote that its analysis may extend to other symmetric increasing links. And
POP-BO's 2024 intuition, that preferences cost an extra $T^{1/4}$, was refuted
at the level of upper bounds by MR-LPF: the gap came from POP-BO's
covering-number confidence width, not from preference feedback itself
(inference).

## Decision-theoretic results {#sec-theory-decision}

qEUBO [@astudillo2023qeubob] is the main Bayesian decision-theoretic result in
PBO. The **one-step Bayes optimal value** of a query
$X = (\vx_1, \dots, \vx_q)$ is
$V_n(X) = \E_n\big[\max_{\vx} \E_{n+1}[f(\vx)] - \max_{\vx}\E_n[f(\vx)] \mid X_{n+1} = X\big]$,
how much one more answer to $X$ is expected to raise the best posterior mean;
stopping at $N$, the recommendation is $\argmax_{\vx} \E_N[f(\vx)]$; and the
noisy likelihood is
$L_i(f(X); \lambda) = \exp(f(\vx_i)/\lambda) / \sum_j \exp(f(\vx_j)/\lambda)$,
with $\lambda = 0$ meaning noise-free answers. The four theorems and their
conditions are listed in @sec-acqf-eubo. Three readings of them (inference):

- **Why $o(1/n)$ is fast.** The conditions turn the problem into finite
  identification with a positive utility gap almost surely. The result says
  nothing about continuous domains, nor about how the rate depends on the
  number of options or the dimension, and it cannot be compared with
  frequentist simple regret of order $T^{-1/2}$.
- **What the sufficient conditions exclude.** An almost-sure gap bound rules out
  ordinary Gaussian process priors with continuous marginals, and a winning
  probability equal to a constant $a > 1/2$ whenever utilities differ is not the
  probit or logistic likelihood used in practice. The result is best read as
  evidence of qEUBO's consistency and a proof of qEI's inconsistency, not as a
  rate for practical PBO.
- **No contradiction with the 2026 critiques.** One-step optimality does not
  include multi-step or asymptotic optimality, so it does not conflict with the
  over-exploitation and ill-conditioning reported in two preprints
  (@sec-acqf-failures); on continuous domains qEUBO has neither such a result
  nor a frequentist regret bound.

::: {.keyidea title="The analyzed algorithms are not the practiced pipeline"}
Theory papers analyze elimination, optimistic, or Thompson-sampling algorithms
built on frequentist kernel estimators. Practice uses a Gaussian process
posterior under the Laplace approximation with EUBO-type acquisition, which has
only one-step Bayes optimality and finite-domain consistency. No paper analyzes
the frequentist regret of the practiced pipeline, and we found no Bayesian
regret bound (for example one based on the information ratio) for Gaussian
process preference acquisition on continuous domains (inference). A
practitioner who wants a guarantee must run an analyzed algorithm; one who runs
the default should read the theory as a statement about the problem, not about
their method.
:::

## Lower bounds {#sec-theory-lower}

A lower bound says how much regret *every* algorithm must incur on some problem
in a class (@sec-regret-lower-bounds). Without one, an upper bound cannot be
called optimal.

**Where order optimality is established.** For finite-arm dueling bandits, the
instance-dependent $\sum_i \log T/\Delta_i$ is matched by
@saha2022versatile, and RMED matches asymptotically [@komiyama2015regret].
Adversarial, batched, and weak-regret variants and the identification of
Condorcet and Copeland winners also have lower bounds [@saha2021adversarial;
@saha2021dueling; @agarwal2022batched; @saad2024weak;
@haddenhorst2021identification; @bengs2024identifying], and the linear and
contextual cases have the $\Omega(\sqrt{dT})$, Borda, and corruption lower
bounds of @sec-theory-finite.

**How much information a query carries.** @sec-query-batch reported the
finite-arm answer under the Plackett-Luce model: winner feedback from
$k$-subsets has optimal sample complexity $O((n/\varepsilon^2)\ln(1/\delta))$
for $n$ arms, the same as pairs, while top-$m$ ranking feedback lowers it by a
factor of $m$ [@saha2019pac]. The same authors gave matching instance-dependent
bounds [@saha2020pac] and order-optimal regret $O((n/m)\ln T)$ for top-$m$ feedback and $O((n/k)\ln T)$
for full rankings [@saha2019combinatorial]. In sign-feedback convex
optimization, the gain from $m$-way argmin feedback is of order
$\min\{\log m, d\}$ [@saha2024faster]; for the linear Plackett-Luce model,
@lee2025preference obtained
$\tilde O\big((d/T)\sqrt{\sum_t 1/\lvert S_t\rvert}\big)$, with $S_t$ the subset
shown in round $t$, so larger subsets provably help; and with feedback that only
ranks arms by past empirical performance, no logarithmic instance-dependent
regret is possible [@maran2024bandits]. The apparent conflict between "winner
feedback from larger subsets does not help" and "larger subsets help" is
resolved by the feedback type: more information per round needs more than the
winner. For PBO, this predicts that an interface asking a person to "pick one
of $K$" does no better than pairwise duels in worst-case order, while a ranking
interface can; this prediction is untested for kernelized PBO (inference).

**The kernelized case: no lower bound.** Within our search (PMLR 2017 to 2025,
NeurIPS 2017 to 2024, and a scan of 2025 and 2026), we found **no
algorithm-independent lower bound for kernelized preference feedback under a
Bradley-Terry or probit link**; targeted searches returned only scalar kernel
lower bounds and finite-arm dueling lower bounds. MR-LPF's near-optimality rests
on an informal comparison with the Gaussian-noise lower bound of Scarlett et
al. Any preference lower bound would have to be compared with the scalar kernel
lower bounds [@scarlett2017lower; @cai2021lower] and the time-varying kernel
lower bound of @iwazaki2025near. @sec-dueling-lower-bounds lists the three
kinds that are missing and what the absence means for the question whether a
comparison is more expensive than a number; the only firm statement in the
kernel case is that, in a batched, finite-domain setting with a warm-up period,
comparisons cost at most constant and logarithmic factors more (inference).

## Theory of observation models {#sec-theory-obs}

**The skew Gaussian process theorem, and where it comes from.** Three papers by
Benavoli, Azzimonti, and Piga need to be told apart: the classification paper
in *Machine Learning* volume 109 (2020) that introduced the skew Gaussian
process [@benavoli2020skew]; the GECCO 2021 Companion paper (arXiv 2008.06677)
that holds the posterior theorem for PBO [@benavoli2021preferentialb]; and the
*Machine Learning* volume 110 paper (2021) that proved conjugacy to normal and
affine probit likelihoods and their products [@benavoli2021unified].
Attributing the preference posterior theorem to "Machine Learning 2020" is
therefore wrong, and "Machine Learning 2021" applies only to the general
conjugacy statement.

A *unified skew-normal* (SUN) distribution generalizes the multivariate
Gaussian by multiplying its density with a normal distribution function, which
tilts it; @sec-skew-gp introduces it. In the theorem, $\Phi_m$ is the
distribution function of $m$ independent standard normal variables, and
$\Omega = D_\Omega \bar\Omega D_\Omega$ splits a covariance matrix into the
diagonal matrix $D_\Omega$ of standard deviations and the correlation matrix
$\bar\Omega$.

::: {.theorem #thm-theory-skewgp title="The preference posterior is a skew Gaussian process (Benavoli, Azzimonti, and Piga, GECCO 2021 Companion)"}
Let $f \sim \GP(\xi, \Omega)$, and let the $m$ observations about the values
$f(X)$ at $n$ inputs have the affine probit likelihood
$p(W \mid f(X)) = \Phi_m(W f(X))$, where $W$ is an $m \times n$ data matrix.

1. (Theorem 1.) The posterior of $f(X)$ is the unified skew-normal distribution
   $\mathrm{SUN}_{n,m}$ with skewness parameters
   $\Delta = \bar\Omega D_\Omega W^\T$, $\gamma = W\xi$, and
   $\Gamma = W\Omega W^\T + I_m$.
2. (Theorem 2.) The posterior of $f$ is a skew Gaussian process with mean
   function $\xi$, covariance function $\Omega$, and skewness function
   $\Delta(\vx, X) = \Omega(\vx, X) W^\T$.
3. (Corollary 1.) For the likelihood of Chu and Ghahramani,
   $\prod_k \Phi\big((f(\mathbf{v}_k) - f(\mathbf{u}_k))/(\sqrt2\,\sigma)\big)$, with
   $\sigma^2 = 1/2$ for identifiability, the posterior follows by taking
   $W_{ij} = V_{ij} - U_{ij}$, where $V$ and $U$ mark the preferred and the
   rejected input of each comparison.
:::

For parametric probit regression, @durante2019conjugate had already shown
conjugacy with the unified skew-normal distribution. The theorem is exact
Bayesian inference for the probit model, not a consistency or rate result. It
explains why the Laplace approximation and expectation propagation misreport
duel probabilities, which matters for acquisition functions that rely on
predicted winning probabilities (@sec-obs-inference), and it does not apply to
the logistic link (inference). The extended skew-normal lookahead posterior of
@wu2026knowledge builds on it.

**Posterior consistency.** *Posterior consistency* means that the posterior
concentrates on the true function as data accumulate; a *contraction rate*
says how fast. We found no such theorem for Gaussian process preference models
of the Chu-Ghahramani type. The nearest results are the frequentist confidence
sets for kernelized logistic estimators in POP-BO, MaxMinLCB, and MR-LPF;
qEUBO's Bayesian consistency on finite domains; and the asymptotic convergence
of SelfSparring's independent-arm version.

**Stochastic transitivity.** The regret theorems above assume different
regularity conditions on the winning probabilities, which @sec-duel-transitivity
defines: strong, moderate, and weak stochastic transitivity, and the stochastic
triangle inequality, which strong transitivity does not imply
[@bengs2021preference]. @yue2012karmed need strong stochastic transitivity and
the triangle inequality; SelfSparring's approximate linearity is stricter than
strong stochastic transitivity; the neural dueling result holds whenever
stochastic transitivity holds; and the tracking results of @suk2023we need the
intersection of strong stochastic transitivity and the triangle inequality. Any
model of a utility plus a monotone link, Bradley-Terry or probit, satisfies
both at every moment (inference, derived from the definitions;
@exr-theory-sst). @chau2022inconsistent question the assumption of
rankability; @sec-obs-extensions reports how far their conjecture reaches.

## Identifiability and aggregation {#sec-theory-identifiability}

A quantity is *identifiable* if different values of it produce different
distributions of data, so that enough data could in principle tell them apart.
A group of results shows that pairwise data are the weakest feedback when
answers depend on something the model does not see.

**Hidden context.** @siththaranjan2024distributional studied a finite set of
options, infinite data, uniformly sampled pairs, and an L2-regularized
Bradley-Terry loss, where each answer may depend on a *hidden context* (who is
answering, in what state) that the model does not observe:

- **Theorem 3.1.** Bradley-Terry preference learning implicitly aggregates
  hidden contexts by the *Borda count*: the learned utility satisfies
  $\hat u(a) > \hat u(b)$ if and only if $\mathrm{BC}(a) > \mathrm{BC}(b)$,
  where $\mathrm{BC}(a)$ is the average probability that $a$ beats a random
  opponent.
- **Theorem 3.2.** If the hidden-context noise is independent and identically
  distributed across options and the support of its differences contains a
  neighborhood of zero, the learned order equals the order of expected utility.
- **Proposition 3.3.** The majority preference can agree with expected utility
  while Bradley-Terry does not.
- **Theorem 3.4.** No deterministic method using infinite comparison data can
  always recover expected utility, even up to a monotone transformation.

The authors point out that annotators therefore have an incentive to misreport.
@an2026differential (a preprint) also note that the Bradley-Terry-Luce loss
corresponds to the Borda count. For single-user PBO, this means that if a
person's answers depend on unmodeled context, such as fatigue, framing, or
order, the Gaussian process utility recovers a Borda-type aggregate rather than
the mean utility (inference; @exr-theory-hidden works an example).

**Heterogeneous people.** @chidambaram2026direct (AISTATS 2026; arXiv
2405.15065 and 2510.15716 are versions with the same title) proved three
results about a *random-coefficient logit* model, in which each user has their
own preference weights $\beta$:

- **Lemma 4.1.** If each user makes a single binary comparison, even with
  infinitely many users the distribution of types is not identifiable: a
  half-and-half mixture of $\beta$ and $-\beta$ gives probability 0.5
  everywhere.
- **Theorem 4.2** (restating Fox et al. 2012). If the moments satisfy the
  Carleman condition, the support of the features contains an open set around
  zero, $\beta$ is independent of the features, and there are at least 3
  options, the type distribution is nonparametrically identifiable, even from
  incomplete rankings of three.
- **Lemma 4.3.** Many diverse binary comparisons by one user identify that
  user's $\beta$ when the matrix of feature differences has full rank.

**Cycles, partial orders, and context effects.** @liu2026statistical proved
that preferences can be represented by a reward model if and only if there are
no Condorcet cycles (cases where $a$ beats $b$, $b$ beats $c$, and $c$ beats $a$
by majority), that under the Luce model Condorcet cycles appear with
probability tending to 1 exponentially fast, and that Nash learning from human
feedback yields a mixed strategy if and only if no response is preferred by a
majority to every other. @drago2025theoretical prove that constructing a
multi-objective utility of minimal dimension compatible with a partial order of
preferences is NP-hard. @depeuter2024preference used a tractable surrogate of a
cognitive model of preferential choice with context effects, and inferred
better than Bradley-Terry variants on large-scale human data. And
@cao2026provably showed for a Plackett-Luce subset-choice model that learning
from queries alone faces a shift-invariance barrier and needs bandit feedback as
an anchor.

Taken together (inference): for regret on a single fixed utility, pairwise
comparisons are not inherently more expensive than numbers in order; but for
identifying a heterogeneous population, recovering expected utility under hidden
context, and handling cyclic preferences, pairwise data are the weakest
feedback, and ranking feedback beats winner-only feedback. The single-utility
assumption behind every PBO regret bound is questioned twice over, by Chau et
al.'s empirical conjecture and by Liu et al.'s asymptotic result.
@sec-query-aggregation and @sec-many-users take up what this means for
designing queries.

## Drift, contamination, response times, stopping {#sec-theory-drift}

Four features of real people, preferences that change, answers that are wrong,
answers that take time, and sessions that must end, each have some theory,
mostly outside the kernelized setting.

**Drift.** For finitely many arms the theory is mature. @saha2022optimal gave
$O(\sqrt{KT})$ static regret against adversarial preference sequences, and
dynamic regret $\tilde O(\sqrt{SKT})$ for $S$ effective switches and
$\tilde O(V_T^{1/3}K^{1/3}T^{2/3})$ for continuous variation $V_T$, all with
matching lower bounds, and ANACONDA [@kleinebuening2023anaconda] adapts to an
unknown number of switches; stationary segments [@kolpaczki2022non], a
preprint, and change points in a high-dimensional Bradley-Terry model
[@li2022detecting] have their own results. @suk2023we proved that adapting to "significant shifts"
at the rate $O(\sqrt{KLT})$, with $L$ the number of significant shifts, is
impossible under the Condorcet class or the strong-stochastic-transitivity
class, while the intersection of strong stochastic transitivity and the
triangle inequality is the largest of the common classes in which it is
feasible, and @liu2026online proved that with feedback that ranks by
instantaneous utility, sublinear external regret is in general impossible,
becoming possible when the total variation of the utility sequence is
sublinear; @son2025right give bounds for direct preference optimization under
unknown drift. For scalar kernels, @iwazaki2025near gave the
first algorithm-independent lower bound for non-stationary kernelized bandits,
and Theorem 4.1 of @bogunovic2016time shows that when the per-step change
$\varepsilon$ of their Markov model is fixed, every algorithm has cumulative
regret $\Omega(T\varepsilon)$. Since utility-plus-link models satisfy strong
stochastic transitivity and the triangle inequality at every moment, the
obstacle to tracking drift in kernelized PBO is technical, a missing dynamic
regret analysis for kernels with a link, not a known impossibility (inference);
with no PBO model of a drifting utility (@sec-obs-extensions), there is no
dynamic regret guarantee either.

**Contamination and bias.** @agarwal2021stochastic gave regret that depends
linearly on the number of corrupted comparisons involving the Condorcet winner,
and proved the linear dependence necessary; @saha2022versatile pay only an
additive $2C$ in the corrupted Condorcet setting. In the linear case,
@di2025nearly gave $\tilde O(\kappa d\sqrt{T} + \kappa dC)$, with $\kappa$
multiplying the corruption term, and @oh2026robust gave
$\tilde O(d(\sqrt{T} + C + D))$, where $D$ measures delays; known or unknown
evaluator bias [@tang2025tackling] and corrupted pairs in reinforcement learning
from human feedback [@bukharin2024robust; @mandal2025corruption] have been
handled too. In the kernel case, the
only robustness result remains the linear-link bias model of Kirschner and
Krause 2021; the scalar reference is @bogunovic2020corruption. Since the
results of Siththaranjan et al. give annotators a reason to misreport, a
mechanism for eliciting truthful feedback matters too; an AISTATS 2026 paper
does this with the Vickrey-Clarke-Groves mechanism [@landolt2026eliciting], of
which only the title and venue were verified.

**Response times.** Response time is the only human channel with theory, and
only for linear utilities. @li2024enhancing used the EZ-diffusion model, a
simplified drift-diffusion model of how evidence accumulates during a choice
(@sec-neuro-ddm), and showed in theory and experiment that for queries with
strong preferences response times complement choices.
@benkert2026time (a working paper, 2026 version) proved that binary choice
frequencies identify only one point of the latent preference distribution, and
that adding a monotone response-time function identifies it at several points.
The Gaussian process response-time model of @shvartsman2024response has no
theoretical guarantee.

**Stopping.** The nearest results to a stopping rule with guarantees for PBO
are these. @haddenhorst2021testification combine identifying a Condorcet
winner with testing whether one exists, so that the learner can stop and
decline to answer, with a lower bound on the expected sample complexity and an
algorithm optimal up to logarithmic factors. @shukla2024preference gave a lower
bound and a matching preference-aware Track-and-Stop algorithm for vector
rewards ordered by a cone. Fixed-confidence identification has its own stopping
rules [@bengs2024identifying; @saha2019pac; @saha2020pac]. @sec-hd-stopping
reports the parametric rule of Bıyık et al. and the scalar rules that have not
been carried over to a pairwise likelihood.

## Settled, contested, missing {#sec-theory-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** Instance-optimal logarithmic regret with matching lower bounds for
finite-arm dueling bandits [@komiyama2015regret; @saha2022versatile]. Minimax
rates for linear and contextual dueling, within link constants and logarithmic
factors [@saha2021optimal; @li2024feel]. Formal guarantees before 2024:
SelfSparring (2017), Kumagai (2017), Xu et al. (2020), Kirschner and Krause
(2021), qEUBO (2023). Kirschner and Krause 2021 as the first kernelized dueling
bound on cumulative regret, under the difference-plus-noise model. The
kernelized upper bounds under the Bradley-Terry link: POP-BO about $T^{3/4}$
(utility regret); MaxMinLCB and PF-TS $\gamma_T\sqrt{T}$; MR-LPF
$\sqrt{\gamma_T T}$ (batched, finite domain, warm-up period), the last three in
preference-probability regret. The probit preference posterior is a skew
Gaussian process [@benavoli2021preferentialb]. Winner feedback from larger
subsets brings no gain in order, and top-$m$ rankings a factor of $m$ (finite
arms and linear models) [@saha2019pac; @saha2021optimal].

**Contested.** MR-LPF's optimality: its authors call the tightness argument
informal, and an unconfirmed machine review questions one inequality in its
Theorem 4.7. "Comparisons are as sample-efficient as numbers": this is equality
of upper-bound order in a batched, finite-domain, logistic-link setting, not
equality of information, and not settled. Sequential against batched: the
order-optimal but batched MR-LPF coexists with the sequential MaxMinLCB and
PF-TS, which lose a factor of $\sqrt{\gamma_T}$; the only direct comparison
comes from the PF-TS paper, which shares authors with MR-LPF, and is
low-dimensional.

**Missing,** in order of how much each limits the theory of PBO (inference):
kernelized lower bounds under a Bradley-Terry or probit link with explicit
$\kappa$ dependence; an order-optimal fully sequential algorithm; ranking or
multi-option likelihoods in kernelized bounds; analysis of the approximate
posteriors used in practice (Laplace, expectation propagation) rather than exact
or frequentist estimators; theory of drift, contamination, and response times in
the kernel case; Bayesian regret for qEUBO-type rules on continuous domains; a
stopping rule linking the Bayesian recommendation $\argmax_{\vx}\E_N f(\vx)$ to a
guarantee; posterior consistency for Gaussian process preference models; and
regret bounds for DTS and the hallucination believer, and a proof that
KernelSelfSparring is no-regret.
:::

## Exercises {#sec-theory-exercises}

::: {.exercise #exr-theory-hidden}
Half of the people answering a pairwise question are of type 1, with utilities
$(10, 1, 0)$ for options $(a, b, c)$; the other half are of type 2, with
utilities $(0, 2, 1)$. Each person answers deterministically by their own
utility, and the model does not know who is answering. (a) Compute the expected
utility of each option and the probability that each option beats each other
one. (b) Compute the Borda count, the average probability that an option beats
a uniformly chosen other option. (c) Which order does Bradley-Terry learning
recover, by Theorem 3.1 of Siththaranjan et al. as stated in
@sec-theory-identifiability, and does it agree with expected utility?

::: {.solution}
(a) Expected utilities are $5$ for $a$, $1.5$ for $b$, and $0.5$ for $c$, so
$a \succ b \succ c$. Type 1 prefers $a$ to $b$ and type 2 prefers $b$ to $a$,
so $\Prob(a \succ b) = 1/2$; likewise $\Prob(a \succ c) = 1/2$; both types
prefer $b$ to $c$, so $\Prob(b \succ c) = 1$.
(b) $\mathrm{BC}(a) = (1/2 + 1/2)/2 = 0.5$, $\mathrm{BC}(b) = (1/2 + 1)/2 = 0.75$,
and $\mathrm{BC}(c) = (1/2 + 0)/2 = 0.25$.
(c) The learned utility orders the options by Borda count, $b \succ a \succ c$,
while expected utility puts $a$ first: the large gain type 1 gets from $a$
never shows in a binary answer, which records only the direction of a
preference. These probabilities are already the infinite-data limit, so more
of the same comparisons do not change the learned order; this is the situation
Theorem 3.4 describes. In one person's session, the "types" can be moods,
framings, or states of fatigue (inference).
:::
:::

::: {.exercise #exr-theory-units}
Under the logistic link, compare the two units of regret for a single query
whose utility gap to the optimum is $g$: utility regret $g$ against
preference-probability regret $\operatorname{sigmoid}(g) - 1/2$. Compute both for $g = 0.1$ and
$g = 4$. When is it safe to compare a rate stated in one unit with a rate
stated in the other?

::: {.solution}
For $g = 0.1$: $\operatorname{sigmoid}(0.1) - 1/2 \approx 0.0250$, close to $g/4 = 0.025$,
because the slope of $\operatorname{sigmoid}$ at zero is $1/4$. For $g = 4$:
$\operatorname{sigmoid}(4) - 1/2 \approx 0.482$, while $g/4 = 1$; preference-probability regret
saturates at $1/2$ and understates large gaps by a factor of about two here,
and by more as $g$ grows. The units are proportional only when the gaps that
dominate the sum are small, so a rate in one unit transfers to the other only
under that approximation, with the factor $1/4$, and comparisons across papers
should say so.
:::
:::

::: {.exercise #exr-theory-sst}
Let $\Prob(i \succ j) = F(u_i - u_j)$ for a utility $u$ and a strictly
increasing link $F$ with $F(0) = 1/2$ and $F(-a) = 1 - F(a)$. Show that strong
stochastic transitivity holds. Then show that if $F$ is concave on
$[0, \infty)$, the stochastic triangle inequality holds as well.

::: {.solution}
Write $a = u_i - u_j$ and $b = u_j - u_k$. If $\Delta_{ij} \ge 0$ and
$\Delta_{jk} \ge 0$, then $a, b \ge 0$, since $F$ is increasing with
$F(0) = 1/2$. Then $u_i - u_k = a + b \ge \max\{a, b\}$, and because $F$ is
increasing, $\Delta_{ik} = F(a + b) - 1/2 \ge \max\{F(a), F(b)\} - 1/2$, which
is strong stochastic transitivity. For the triangle inequality, let
$G(x) = F(x) - 1/2$, so $G(0) = 0$ and $G$ is concave on $[0, \infty)$. A
concave function with $G(0) = 0$ is subadditive there:
$G(a) \ge \tfrac{a}{a + b}G(a + b)$ and $G(b) \ge \tfrac{b}{a + b}G(a + b)$ by
concavity between $0$ and $a + b$, and adding gives
$G(a) + G(b) \ge G(a + b)$, that is,
$\Delta_{ik} \le \Delta_{ij} + \Delta_{jk}$. The logistic and probit links are
increasing, symmetric, and concave on $[0, \infty)$, so both conditions hold for
the models of this book at every moment, which is the basis of the inference in
@sec-theory-drift.
:::
:::

## Further reading {#further-reading .unnumbered}

- @bengs2021preference is the survey of dueling bandits, organized by the
  assumptions on pairwise winning probabilities; @sui2018advancements is the
  shorter earlier survey.
- @kirschner2021bias, @xu2024principledb, @pasztor2024bandits,
  @kayal2025bayesian, and @lazzaro2026finiteb are the kernelized results; read
  each theorem with its feedback model and regret unit.
- @scarlett2017lower and @vakili2021information give the scalar lower bounds
  and information-gain rates that every preference result must be compared with.
- @astudillo2023qeubob is the Bayesian decision-theoretic side; its conditions
  repay careful reading.
- @siththaranjan2024distributional is the clearest account of what
  Bradley-Terry learning recovers when answers depend on hidden context.
