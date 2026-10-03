---
status: done
synopsis: "The decisions a working Bayesian optimization system makes beyond the textbook loop: how to set up the surrogate, what to report under noise, how to choose batches, constraints and safety, several objectives, many dimensions, costs and stopping, and which software implements what."
sources: ["Snoek et al. 2012", "Balandat et al. 2020"]
---

# Bayesian Optimization in Practice {#sec-bo-practice}

@sec-regret ended by treating guarantees as design principles whose constants
are settled by experiment. This chapter is about the decisions those
experiments have to make. The loop of @sec-bo-loop and the acquisition
functions of @sec-acquisition were built for a tidy problem. One evaluation
runs at a time and returns a value with little noise. There is one objective, no constraint other than the
box of allowed inputs, a handful of inputs, and every evaluation costs the
same. The budget is fixed in advance. Real problems break every one of these
assumptions, often several at once.

Take the problem that made Bayesian optimization popular in machine learning:
tuning the hyperparameters of a model, the settings such as a learning rate or
a regularization strength that are chosen before training rather than learned
from the data. @snoek2012practical, who showed that Bayesian optimization could
match or beat expert tuning of convolutional networks and other models, had to
decide how to model an error rate that spans orders of magnitude, what to do
when a training run is noisy, how to use several machines at once, and how to
account for runs that take minutes or hours. Each of those decisions is a
section of this chapter. @sec-cs-classifier works through the same decisions
on a real classifier, with measured landscapes; this chapter is the general
toolkit.

The chapter moves from the model outward: first the surrogate, then noise and
what to report, then batches, constraints, several objectives, many
dimensions, and cost. It closes with the software that implements these
choices and the versions current as of September 2026.

## Surrogate choices {#sec-practice-surrogate}

A Gaussian process prior is a set of assumptions: that the function's values
are of a certain size around a certain mean, that it varies on a certain
lengthscale, and that it is equally smooth everywhere. @sec-gp-pitfalls listed
the habits that make these assumptions hold well enough. In Bayesian
optimization they matter more than in regression, because the posterior is
not just reported but acted on: a wrong lengthscale sends the next evaluation
to the wrong place, and the error compounds over the run.

### Scaling outputs and inputs {#sec-practice-scaling}

**Standardize the outputs.** The default prior has mean zero and unit
amplitude, so the observed values should be centered and scaled to unit
standard deviation before fitting, and the predictions transformed back. Since
version 0.12.0 (September 2024), BoTorch applies this transformation by
default in most of its models, partly because its new default priors,
discussed in @sec-high-dim-practice, "will work less well when data is not
standardized" [@botorch2026changelog].

**Transform skewed outputs.** An error rate, a runtime, or a reaction yield
often spans orders of magnitude, with a few very bad evaluations. A Gaussian
process with constant amplitude then spends its flexibility on the outliers.
Modeling the logarithm, or another monotone transformation, makes the values
closer to what the prior expects. Monotone transformations do not change
where the maximum is, only how the model sees the landscape.
@cowenrivers2022hebo studied 108 hyperparameter tuning tasks and found that
most exhibit heteroscedasticity, noise whose size varies across the domain,
and non-stationarity, smoothness that varies across the domain; their method
HEBO, which learns transformations of both inputs and outputs, placed first in
the NeurIPS 2020 Black-Box Optimization Challenge.

**Scale the inputs.** Map each input to $[0, 1]$, and put inputs that act by
multiplication on a logarithmic scale first: a learning rate between $10^{-5}$
and $10^{-1}$ is naturally searched in $\log_{10}$ units, where each decade
gets the same share of the box. When the right transformation is not known,
it can be learned. @snoek2014input warp each input through the cumulative
distribution function of a Beta distribution, whose two parameters are fitted
along with the kernel's, so that the model can stretch regions where the
function changes quickly and compress regions where it is flat.

### Kernel and hyperparameters {#sec-practice-kernel}

**Kernel.** The RBF (squared exponential) kernel produces functions that are
infinitely differentiable, and @snoek2012practical argued that this is
"unrealistically smooth for practical optimization problems"; they
recommended a Matérn 5/2 kernel with one lengthscale per input (@sec-ard). This
is the default of several libraries; Optuna's Gaussian process sampler, for
instance, uses exactly that kernel [@optuna2026gpsampler]. In a comparison on
12 high-dimensional benchmarks, standard Bayesian optimization with a Matérn
kernel was consistently among the best methods, while the RBF kernel often led
to poor performance, for a numerical reason that @sec-hd-practice-prior
explains [@xu2025standard].

**Hyperparameters.** The lengthscales, the amplitude, and the noise variance
are fitted to the data at every step, by maximizing the marginal likelihood
plus the log of a prior on each hyperparameter (@sec-fitting-hyperparameters).
With few observations the fitted values jump from one step to the next, and a
single fit can be badly wrong. One remedy is to average over hyperparameters
instead of choosing one setting: @snoek2012practical drew hyperparameter
samples by Markov chain Monte Carlo (here, slice sampling, a way of drawing
from a distribution known only up to a constant) and averaged the
acquisition function over the samples. The other remedy is a better prior,
the subject of @sec-high-dim-practice. Either way, plot the fitted model now
and then (@sec-model-checking); a lengthscale that has run to the edge of its
allowed range is a warning.

**Inputs that are not continuous.** Integer settings (a number of layers) and
categorical ones (a choice of optimizer) do not fit a kernel on a box without
help. Common treatments round a continuous relaxation, encode categories as
one indicator per value, or use a surrogate that handles categories natively.
SMAC, for example, was designed for configuring algorithms with many
categorical and conditional parameters and uses a random forest, an average
of many decision trees, as its model [@hutter2011sequential]. @sec-cs-chemistry returns to categorical choices,
where the categories are chemical reagents.

::: {.pitfall title="Defaults are decisions"}
Every library makes the choices above for you, and they differ between
libraries and between versions of the same library. BoTorch changed its
default lengthscale priors and output standardization in a single release
[@botorch2026changelog]. When a run behaves strangely, or when results must be
reproduced, record the library version and read what its defaults were.
:::

## Noise and the incumbent {#sec-practice-noise}

Expected improvement measures improvement over an *incumbent*, the best value
so far (@sec-ei). With exact evaluations the incumbent is unambiguous. With
noise it is not, and two questions arise that the textbook loop does not ask:
what value should expected improvement improve on, and what should the run
report at the end?

### The winner's curse {#sec-practice-winners-curse}

Suppose many evaluations land near the top of the objective, all with true
values close to the maximum, each perturbed by independent noise with
standard deviation $\sigma_n$ (@sec-gp-noise). The largest observed value is the one with the
luckiest noise. It overstates the truth at its input by about $\sigma_n$ times
the expected maximum of that many standard normal draws, which is about 1.97
for 25 draws (@exr-practice-winners-curse). Auction theorists call this the
winner's curse: whoever wins the comparison is, on average, the one who was
most overestimated.

The figure below runs Bayesian optimization (the upper confidence bound with
$\beta^{1/2} = 2$, on the running objective) with noisy evaluations, then
compares two ways of choosing what to report.

```{figure}
//| figure: practice-incumbent
//| label: fig-practice-incumbent
//| fig-cap: "What a noisy run should report. Top: one run of 25 evaluations of the running objective with noise sd $\sigma_n = 0.25$ (dots), the final posterior, and the two recommendations: the evaluation with the best observed value (ring) and the evaluated input with the best posterior mean (square). Each is drawn at the value it claims, with a bar down (or up) to the true value at its input. Bottom: averages over 24 runs at six noise levels: the simple regret of each rule, and how much the best observed value overstates the truth (dashed). The model knows the noise level; the objective and the runs are illustrative."
```

Some things to try.

**Read the default run.** The best observed value claims 1.24, at an input
whose true value is 0.79. The best posterior mean claims 0.83, at an input
whose true value is 0.82, the maximum. Both rules pick inputs on the tall
peak; only one of them tells the truth about it.

**Look at the dashed curve.** Averaged over runs, the best observed value
overstates the truth by about twice the noise level, at every noise level:
0.19 at $\sigma_n = 0.1$, 0.99 at $\sigma_n = 0.5$. The posterior mean shrinks
each observation toward its neighbors and does not suffer this inflation in
the same way.

**Compare the regret curves.** The rules choose similar inputs at low noise.
As noise grows, choosing by posterior mean gives lower regret at most noise
levels, and the gap widens with more evaluations: with 40 evaluations and
$\sigma_n = 0.5$, the average regret is 0.10 for the posterior mean against
0.18 for the best observation.

### Improving on an uncertain incumbent {#sec-practice-noisy-ei}

The same problem affects the acquisition function itself. Plain expected
improvement over the best *observed* value aims to beat a lucky draw.
@picheny2013benchmark compared ten criteria for noisy
problems and found that the two most natural choices, plugging the best noisy
observation into expected improvement and repeatedly evaluating the minimum
of the posterior mean, were poor in all their test cases; which of the other
criteria worked best depended on the problem.

*Noisy expected improvement* treats the incumbent as unknown too. Since the
true values at the evaluated inputs are uncertain, so is the best of them, and
the right quantity to average is the improvement of the candidate over that
uncertain best. @letham2019constrained derived this criterion and computed it
by quasi-Monte Carlo integration, averaging over draws of the function at the
candidate and at the evaluated points jointly. They developed it for online
experiments at Facebook, where each evaluation is a randomized test with high
variance, and demonstrated it by optimizing a ranking system and the flags of a
server compiler. In BoTorch the criterion has a Monte Carlo form: draw joint
posterior samples of the function at the candidates and at the previously
evaluated points, compute for each sample how far the best candidate value
exceeds the best previously evaluated value (zero if it does not), and average
over the samples [@balandat2020botorch]; its logarithmic variant, which is easier to
maximize numerically, is the current recommendation
[@ament2023unexpected; @botorch2026changelog].

::: {.keyidea title="Report the model, not the luckiest measurement"}
Under noise, recommend the input with the best posterior mean, and report the
posterior mean as its value. If the decision matters, evaluate the few best
candidates again before choosing. @sec-loop-recommend made the same
recommendation for the loop; noise is what makes it necessary.
:::

## Batches and parallel evaluation {#sec-batch-bo}

Many evaluations can run at the same time. A cluster trains eight models at
once; a laboratory robot fills a plate of 96 wells; a clinic sees several
participants in a day. The loop of @sec-bo-loop, which waits for each result
before choosing the next input, would leave most of that capacity idle. A
*batch* method chooses $q$ inputs at once.

### Why the top q points are a bad batch {#sec-batch-naive}

The obvious batch is the $q$ inputs with the largest acquisition values. It
is a bad one. An acquisition function is smooth, so the inputs ranked second,
third, and fourth are usually neighbors of the first. They ask almost the same
question $q$ times. What a batch should maximize is the value of the whole set:
for expected improvement, the expected improvement of the *best* point in the
batch,

$$
\text{q-EI}(\vx_1, \dots, \vx_q) = \E_n\!\left[\max\Big\{\max_{j \le q} f(\vx_j) - f^*_n, \; 0\Big\}\right],
$$ {#eq-practice-qei}

where $f^*_n$ is the incumbent of @sec-pi and the expectation is over the joint posterior
of $f$ at the $q$ inputs. Because neighbors are strongly correlated, adding a
neighbor of $\vx_1$ adds little to the maximum: when $f(\vx_1)$ is low, so is
$f(\vx_2)$. @ginsbourger2010kriging introduced q-EI, gave a closed form for
$q = 2$, and estimated it by Monte Carlo for larger batches.

Maximizing q-EI jointly means searching over $q$ inputs at once, a problem in
$q \times d$ dimensions. The practical methods build the batch one point at a
time instead, and each needs a way to account for the points already chosen
whose outcomes are not yet known.

### Pretending to know the outcome {#sec-batch-fantasies}

@ginsbourger2010kriging proposed two heuristics. The *kriging believer* picks
the maximizer of expected improvement, pretends that its outcome equals the
posterior mean there, adds this fantasized observation to the data, refits,
and picks again. The *constant liar* does the same with a fixed fantasized
value, the lie: a pessimistic lie, such as the worst value observed so far,
pushes later points away from earlier ones; an optimistic lie draws them
together. Either way, a fantasized observation shrinks the posterior variance
around the chosen point, so the next maximizer moves elsewhere.

Pretending to know one outcome is a crude summary of an uncertain one.
@snoek2012practical integrated instead: they averaged the acquisition function
over several fantasized outcomes for the pending evaluations, drawn from the
posterior. This is the idea behind the Monte Carlo acquisition functions of
BoTorch. The joint posterior at the batch and at pending points is sampled with
the device of @sec-acq-optimization, fixed random numbers passed through a
fixed transformation, which makes the Monte Carlo estimate a deterministic,
differentiable function of the inputs [@wilson2018maximizing]. BoTorch then
fixes those random numbers for the
whole optimization, an approach called sample average approximation, and
maximizes with ordinary quasi-Newton methods [@balandat2020botorch]. It can
optimize all points of a batch jointly, which is the default of its
`optimize_acqf` function, or build the batch by sequential greedy
optimization: choose one candidate, add it to the pending points, choose the
next. For expected improvement, upper confidence
bounds, and similar criteria, the value of a set has diminishing returns, which
is why greedy construction works well [@wilson2018maximizing;
@balandat2020botorch].

### Batches without fantasies {#sec-batch-other}

Thompson sampling parallelizes with no extra machinery: each worker draws its
own sample of the function from the posterior and evaluates where that sample
is largest. The randomness of the draws spreads the batch.
@kandasamy2018parallelised showed that making $n$ evaluations distributed
among $M$ workers this way is essentially equivalent to making $n$
evaluations in sequence, and that with evaluation times modeled, the
asynchronous version, in which a worker starts its next evaluation as soon as
it finishes, achieves lower regret under a time limit than the synchronous and
sequential versions.

Two other routes avoid refitting the model inside the batch.
@gonzalez2016batch penalize the acquisition function around each chosen point,
by an amount derived from an estimate of how fast the function can change.
@desautels2014parallelizing extended GP-UCB to batches (GP-BUCB) by updating
only the posterior variance with the pending inputs, which needs no outcomes,
and proved a regret bound; for some common kernels, the asymptotic average
regret can be made independent of the batch size.

The figure compares four rules on the running objective, after four
observations.

```{figure}
//| figure: practice-batch
//| label: fig-practice-batch
//| fig-cap: "Choosing a batch of q evaluations. Top: the posterior after the observations so far (dots), the hidden objective (dashed), and the batch (orange lines, numbered in the order chosen); for the kriging believer and the constant liar, the orange dots are the fantasized outcomes; for Thompson sampling, the violet curves are the posterior draws. Bottom: expected improvement before the first pick, and for the two fantasy rules after each fantasy (fainter curves). The readout gives a Monte Carlo estimate of q-EI, @eq-practice-qei, for the batch. *Evaluate the batch* runs the q evaluations (without noise) and chooses the next batch."
```

Some things to try.

**Compare top q with the kriging believer.** With $q = 4$, the four largest
values of expected improvement sit on neighboring inputs around $x = 0.17$,
and the batch's q-EI is 0.195, barely more than the 0.174 of the best single
point alone. The kriging believer places its four points in four regions,
including the tall peak near $x = 0.7$, and its q-EI is 0.432.

**Watch the fantasies.** In the bottom panel, each fantasy flattens expected
improvement where it was added, and the next maximum appears elsewhere. The
constant liar's lie here is the worst value observed so far, which lowers the
posterior mean around each chosen point as well as its variance; its batch
differs from the believer's, with a q-EI of 0.402.

**Try Thompson sampling.** Each draw has its own maximum, so the batch is
spread by chance rather than by design. Its q-EI is lower than the kriging
believer's here, but it needs no refitting and no fantasies, which is why it
scales to large batches.

**Evaluate a few batches.** After two or three rounds, the batches crowd into
the tall peak and q-EI falls toward zero: there is little left to gain. At
that point a batch has nothing useful to explore, and a larger budget would be
better spent elsewhere.

## Constraints and safety {#sec-constrained-bo}

Some inputs are not allowed, and the reason is not always known in advance.
A box of allowed ranges or a linear constraint between inputs, such as
proportions that must sum to one, is known before any evaluation and is
handled by the acquisition optimizer. The harder case is a constraint that
must itself be measured: a training run that exceeds its memory limit, a
reaction that produces too much of an unwanted byproduct, a design that fails
a stress test. These *black-box constraints* are as expensive to evaluate as
the objective, and often come from the same experiment.

### Expected improvement that respects constraints {#sec-constrained-ei}

The standard approach models each constraint $c_k(\vx) \le 0$ with its own
Gaussian process, alongside the objective. If the constraints are independent
of the objective under the model, the expected improvement of a candidate,
counting only improvement that is feasible, factors into two terms:

$$
\EI_c(\vx) = \EI(\vx) \cdot \Prob\!\big(c_1(\vx) \le 0, \dots, c_K(\vx) \le 0\big),
$$ {#eq-practice-cei}

where $\EI$ is computed against the best *feasible* value observed so far and
the probability of feasibility is a product of Gaussian cumulative
distribution functions when the constraints are independent of one another.
@gardner2014bayesian describe this as "precisely the standard expected
improvement of x̂ over the best feasible point so far weighted by the
probability that x̂ is feasible", and showed that it finds feasible optima
"even when small feasible regions cause standard methods to fail".
@gelbart2014bayesian use the same constraint-weighted expected improvement
for constraints that are unknown in advance, such as a simulation that crashes
for some inputs, and @letham2019constrained extended noisy expected
improvement to noisy constraints. @exr-practice-cei derives the product.

Two cases need care. If no feasible point has been found yet, there is no
incumbent, and the sensible first goal is to find one, for instance by
maximizing the probability of feasibility alone (inference). And a large product can come
from a modest improvement with near-certain feasibility or from a large
improvement with uncertain feasibility; @eq-practice-cei treats them as
equivalent, which is right only if a failed evaluation costs nothing beyond
the evaluation itself.

### When an unsafe evaluation is unacceptable {#sec-safe-bo}

Sometimes a failed evaluation costs a great deal more. A robot controller that
makes a quadrotor crash, a stimulation setting that hurts a patient, or a
recommendation that drives a user away cannot be treated as a lost evaluation.
Then the requirement is that *every* evaluation be safe with high
probability, not just the final answer.

SafeOpt, introduced by @sui2015safe, enforces this for the case where the
objective itself must stay above a safety threshold at every evaluation. It
keeps a *safe set*: the inputs that the model's confidence intervals, together
with an assumption on how fast the function can change, certify as above the
threshold. It evaluates only inside the safe set, and chooses between inputs
that might be the best safe point and inputs that might enlarge the safe set
if evaluated. Starting from a known safe
input, the safe set grows outward as evidence accumulates. The authors proved
that SafeOpt converges to the best point reachable without leaving the safe
region, and demonstrated it on movie recommendation and on therapeutic spinal
cord stimulation, where the stimulation settings had to remain safe for the
patient [@sui2015safe]. A later variant separates the expansion of the safe
region from the optimization within it [@sui2018stagewise].

The price of safety is reach. A safe method cannot jump to a distant region
that is separated from the start by unsafe inputs, so it may never find the
global optimum; it finds the best optimum connected to where it began.
@sec-app-robotics returns to safe optimization in robotics.

## Multiple objectives {#sec-multi-objective}

A model should be accurate and fast; a reaction should give high yield with
little waste; a prosthesis should be comfortable and efficient. When
objectives conflict, there is usually no single best input, only trade-offs.

### Pareto fronts and hypervolume {#sec-mo-pareto}

An input *dominates* another if it is at least as good on every objective and
strictly better on at least one. The inputs that no other input dominates
form the *Pareto set*, and their objective values form the *Pareto front*:
the menu of best possible trade-offs, from which a decision maker eventually
chooses. Multi-objective Bayesian optimization tries to find a good
approximation of this front with few evaluations.

To compare approximations of a front with a single number, the standard
measure is the *hypervolume*: the volume of objective space that the points
found so far dominate, measured up to a reference point that marks the worst
acceptable value of each objective. A larger hypervolume means a front that is
both further out and more spread. Two families of acquisition functions
follow.

**Scalarization.** Turn the objectives into one by a weighted combination,
and use any single-objective acquisition function on that. ParEGO draws a new
weight vector at random at each iteration, so that over many iterations
different parts of the front get attention. It scores a point by an augmented
Chebyshev function of its normalized costs, the largest weighted cost plus
0.05 times their weighted sum, which, unlike a weighted sum alone, can be
minimized at points on nonconvex parts of the front [@knowles2006parego].

**Hypervolume improvement.** Score a candidate by the expected increase in
hypervolume if it were evaluated [@emmerich2006single]. This criterion targets
the front directly but is expensive to compute in several objectives.
@daulton2020differentiable derived a batch version, qEHVI, with exact
gradients through automatic differentiation, which made it practical to
optimize with gradient methods, and @daulton2021parallel extended it to noisy
observations (qNEHVI), integrating over the uncertainty in the current front;
they show it is one-step Bayes optimal for hypervolume and reduces the cost of
batch selection from exponential to polynomial in the batch size.

### When the trade-off belongs to a person {#sec-mo-preference}

The front is a menu, and someone still has to order from it. Often that person
has a preference among trade-offs that could have guided the search from the
start, saving evaluations on parts of the front nobody wants.
@lin2022preferenceb proposed alternating between experiments and *preference
exploration*: showing the decision maker pairs of predicted outcomes, learning
a utility function over outcomes from their choices, and then optimizing the
expected utility. Learning a utility from comparisons is the subject of
@sec-part-preferences; this is one of the places where it meets scalar
Bayesian optimization directly.

## Many dimensions {#sec-high-dim-practice}

The examples so far had one or two inputs. Real problems have more: the
classifier of @sec-cs-classifier-many has seven hyperparameters, and
controllers, chemical processes, and engineering designs often have a dozen
or more. As the number of inputs $d$ grows, three
things get harder at once. The volume to search grows exponentially
(@sec-explore-exploit computed how little of a six-dimensional cube one
evaluation informs). Points in the cube grow
further apart, like $\sqrt{d/6}$ on average, so a kernel with a fixed
lengthscale sees every pair of points as unrelated. And the acquisition
function becomes flat almost everywhere, which makes it hard to maximize
(@sec-acq-high-dim).

The rule of thumb used to be that Bayesian optimization works up to 10 or 20
inputs [@frazier2018tutorial]. Research since 2024 has changed what that rule
should say, and @sec-high-dimensions reports the debate in detail. This
section describes the main families of methods and what a practitioner can do
today.

### Assume structure {#sec-hd-practice-structure}

The older approach assumes that the function, though defined on many inputs,
is simpler than it looks.

**Low effective dimension.** If only a few directions in the input space
matter, the search can happen in a random low-dimensional subspace. REMBO
optimizes in a random linear embedding of small dimension and, its authors
report, can solve problems with billions of dimensions provided the intrinsic
dimensionality is low; it also tuned the 47 discrete parameters of a
mixed-integer programming solver [@wang2016bayesian]. @letham2020re identified
design choices in such embeddings that hurt performance and showed that
correcting them improves the method substantially, including on learning a gait
policy for a robot.

**Additive structure.** If the function is a sum of terms that each depend on
a few inputs, a kernel built as a sum of low-dimensional kernels can be learned
from far fewer evaluations. For such additive functions, @kandasamy2015high
proved that the regret depends only linearly on the dimension.

**Sparse relevance.** If only some inputs matter but nobody knows which, the
lengthscales can be given a prior that favors ignoring most inputs, so that
the data must make the case for each relevant one. SAASBO does this with a
sparsity-inducing prior and Hamiltonian Monte Carlo, a sampling method that
uses gradients, to average over the lengthscales [@eriksson2021high].

### Search locally {#sec-hd-practice-local}

A different approach accepts that a global model of a high-dimensional
function is hopeless and searches locally. TuRBO keeps a *trust region*, a
box centered on the best point found so far, and runs Bayesian optimization
only inside it [@eriksson2019scalable]. The box starts with side length 0.8 of
the unit cube, stretched along inputs with long lengthscales and shrunk along
short ones. After three consecutive improvements it doubles; after $d$
consecutive failures (for batches of one) it halves; when it shrinks below
$2^{-7}$, the region is abandoned and a new one starts elsewhere. Several
trust regions can run at once, competing for evaluations through Thompson
sampling. On a 14-dimensional robot pushing task and a 60-dimensional rover
trajectory problem, with budgets of 10,000 and 20,000 evaluations, TuRBO
outperformed the other Bayesian optimization methods and the local optimizers
it was compared with [@eriksson2019scalable].

### Fix the prior {#sec-hd-practice-prior}

The third approach, and the one that changed practice, says that standard
Bayesian optimization was failing in high dimensions for a mundane reason: its
default lengthscale prior. A prior of fixed scale that favors lengthscales of
a few tenths is reasonable in two dimensions, where typical distances between
points are about 0.6. In 50 dimensions, typical distances are about 2.9, so every pair of
points is several lengthscales apart, the kernel treats them as unrelated, and
the model cannot learn. @hvarfner2024vanilla scaled the lengthscale prior with
the square root of the dimension and found that, with this change, standard
Bayesian optimization performed best on three of five real tasks, ahead of
methods designed for high dimensions. They also noticed what the old default
had been doing: on the six-dimensional Hartmann function, the run was
"effectively performing an initial random search followed by 140 iterations of
local search". BoTorch adopted the scaled prior as its default in version 0.12.0: a
log-normal distribution with location $\sqrt{2} + \tfrac12 \log d$ and scale
$\sqrt{3}$, whose mode grows like $\sqrt{d}$ [@botorch2026gpytorch;
@botorch2026changelog]; @exr-practice-prior-mode works out the numbers.

Two other groups reached a compatible diagnosis from the optimization side:
with short initial lengthscales, the gradients used to fit the hyperparameters
vanish numerically, so the fit never moves; a sensible initialization, without
any special prior, largely fixes the problem [@xu2025standard;
@papenmeier2025understanding].

The same analysis is the numerical reason, promised in @sec-practice-kernel,
that the RBF kernel fails where the Matérn kernel does not. Let $\rho$ be the
distance between two inputs measured in units of the initial lengthscale; in
high dimensions it is large for almost every pair. For the RBF kernel, written
$\exp(-\rho^2)$ in that paper, the derivative of a kernel value with respect
to a lengthscale is bounded by a multiple of $\rho^2 e^{-\rho^2}$, which falls
below the precision of double-precision arithmetic once $\rho$ exceeds 6.58.
For the Matérn 5/2 kernel the corresponding factor is
$\rho^2 e^{-\sqrt{5}\rho}$, which shrinks much more slowly and passes the same
level only at $\rho = 21.98$. So with the same initial lengthscales, the
Matérn kernel's fit keeps moving in much higher dimensions
[@xu2025standard].

Why the fix works is contested.
@papenmeier2025understanding argue that much of the success in very high
dimensions comes from local search behavior rather than from a well-fitted
global model, and @doumont2026we reached state-of-the-art results on tasks
with 60 to 6,000 dimensions with a linear model, concluding that a surrogate's
expressiveness and predictive accuracy do not necessarily translate into
optimization performance. @sec-hd-fixes lays out the evidence.

The figure below lets you see both effects on a problem where the answer is
known: the six-dimensional Hartmann function, a standard test function,
hidden among irrelevant inputs so that the total dimension is 6, 10, 20, or 50.

```{figure}
//| figure: dim-bo-race
//| label: fig-practice-race
//| fig-cap: "Recorded runs on the six-dimensional Hartmann function with irrelevant inputs added, opening at 50 dimensions with candidates for the acquisition search drawn uniformly from the cube. Top: regret (the true maximum minus the best value found) of random search and of Bayesian optimization with a fixed-scale or a dimension-scaled lengthscale prior; medians over six seeds, with the range shaded, after a shared initial design of 10 points. Middle: the points one run evaluated, in parallel coordinates, with the six relevant inputs shaded. Bottom: the lengthscale the model learned for each input. Switch the dimension and the acquisition search. Six seeds and 80 evaluations make this a demonstration, not a benchmark."
dims: 50
search: global
```

Some things to try.

**Start at 50 dimensions with global search.** With candidates drawn
uniformly from the cube, both Bayesian optimization runs make little progress:
after 80 evaluations the median regret is 0.89 with the fixed prior and 1.21
with the scaled one, against 1.75 for random search. Uniform candidates in 50
dimensions almost never land where the acquisition function is informative
(@sec-acq-high-dim), so the evaluations stay mediocre, and the bottom panel
shows the consequence for the model: in the run shown, it has given a short
lengthscale to only one of the six inputs that matter.

**Switch to global plus local search.** Adding perturbations of the best
points to the candidates brings the median regret of both priors to about
0.18. In these runs, how the acquisition function is searched matters more
than which prior the model uses, the position of @papenmeier2025understanding.

**Go to 20 dimensions and compare the two priors' lengthscales.** Under the
scaled prior, the six relevant inputs get lengthscales below 1 and most of the
others lengthscales near 20: the model has found out which inputs matter.
Under the fixed prior every lengthscale stays between about 0.2 and 2: the
relevant inputs still get the shorter ones, but by a factor of about 3, where
the scaled prior separates them by a factor of about 40. The regret curves differ much less than the
models do.

**Go down to 6 dimensions.** With no irrelevant inputs, every method except
random search reaches a small regret, and the choices matter much less.

### What to do in practice {#sec-hd-practice-advice}

In order of how much evidence supports each step, a practitioner facing more
than about ten inputs can do the following (inference, drawing on the studies
above).

1. Reduce the dimension the optimizer sees. Fix inputs that experience says do
   not matter, and search in a lower-dimensional parameterization when one
   exists.
2. Use a current library with dimension-scaled lengthscale priors or a robust
   initialization, which since 2024 is the default in BoTorch's standard
   models [@botorch2026changelog].
3. Search the acquisition function locally as well as globally: start from
   perturbations of the best points, perturbing only some coordinates at a
   time, as TuRBO's candidate sets and BoTorch's options do
   (@sec-acq-high-dim). Or use a trust region method outright.
4. Check the fitted lengthscales after a few dozen evaluations. If every input
   has the same lengthscale and that value equals the prior's starting point,
   the hyperparameter fit has not moved.

## Costs and stopping {#sec-cost-stopping}

The loop counts evaluations, but the budget is usually time or money, and
evaluations differ in what they cost. Training a network with 1,000 hidden
units takes longer than with 10; a reaction at high temperature uses more
energy; a long walking trial tires a participant. A method that ignores cost
will happily spend a day on an evaluation that a slightly worse but much
cheaper one would have made unnecessary.

### Paying attention to cost {#sec-cost-aware}

The simplest correction divides by cost. @snoek2012practical modeled the
duration of each training run with a second Gaussian process and maximized
*expected improvement per second*. A more principled rule comes from
economics. In the *Pandora's box* problem, a searcher opens boxes of known
cost and uncertain reward and wants the best reward net of the costs paid; its
optimal solution is an index computed for each box, a Gittins index.
@xie2024cost showed that this index can be used as an acquisition function
for cost-aware Bayesian optimization, the Pandora's Box Gittins index (PBGI),
and found that it performs well, particularly in medium to high dimensions,
and also without explicit costs.

### Cheaper versions of the same evaluation {#sec-multi-fidelity}

Often an evaluation can be made cheaper at the price of accuracy: train for
fewer epochs, on a subset of the data, on a coarser simulation mesh. These are
*fidelities*. A multi-fidelity method models the objective jointly across
fidelities and spends most of its budget on cheap, rough evaluations, saving
the expensive ones for promising inputs.

- @swersky2013multi modeled related tasks, such as the same model on a small
  and a large data set, with a multi-task Gaussian process, and used the cheap
  task to speed up optimization of the expensive one.
- FABOLAS models the validation error as a function of both the
  hyperparameters and the size of the training subset, and in its authors'
  experiments often found good configurations 10 to 100 times faster than
  other Bayesian optimization methods or Hyperband [@klein2017fast].
- BOCA handles a continuous range of approximations, such as the amount of data
  and the number of training iterations together [@kandasamy2017multi], and
  the trace-aware knowledge gradient of @wu2019practical also uses the whole
  learning curve that one training run produces.
- Hyperband, which is not Bayesian, trains many random configurations briefly
  and continues only the promising ones [@li2018hyperband]; BOHB replaces its
  random sampling with a model and consistently outperformed both Bayesian
  optimization and Hyperband in its authors' experiments
  [@falkner2018bohb].

@sec-cs-classifier shows how much cost varies across one real tuning problem,
by a factor of 350 there, and what that does to the choice of method.

### When to stop {#sec-stopping}

Most runs stop when the budget is spent. A better rule stops when further
evaluations are not worth their cost, and since 2023 several such rules have
come with guarantees.

- @ishibashi2023stopping use the gap between the expected minimum simple
  regrets of consecutive steps as the stopping criterion.
- @wilson2024stopping stop when, under the model, some evaluated point is
  within $\varepsilon$ of the optimum with probability at least $1 - \delta$,
  and show that Bayesian optimization with Gaussian process priors reaches
  this condition under mild technical assumptions.
- @xie2026cost, accepted at ICML 2026 according to its arXiv listing, continue
  only while some point's expected improvement exceeds its evaluation cost.
  Paired with PBGI or with log expected improvement per unit cost, the
  resulting policy has bounded expected *cost-adjusted* simple regret, the
  simple regret of @sec-regret-definitions plus the total cost paid.

Each guarantee holds under the model, so it is only as good as the model's
hyperparameters, and the costs are assumed known or learnable. When the cost
is a person's time and attention, both assumptions are doubtful;
@sec-hd-stopping discusses stopping when the evaluations are human judgments.

## Software {#sec-bo-software}

Most practitioners use a library, and the library's defaults make many of the
decisions in this chapter. @tbl-practice-software lists the main open-source
libraries for Bayesian optimization with scalar feedback. Versions, dates,
and maintenance status are as of September 2026.

::: {.table #tbl-practice-software title="Open-source Bayesian optimization libraries, as of September 2026"}
| Library | Built on | Notes | License | Latest release |
|---|---|---|---|---|
| BoTorch [@balandat2020botorch] | PyTorch, GPyTorch | Monte Carlo acquisition functions, batches, constraints, multiple objectives, multi-fidelity; dimension-scaled priors and standardization by default since 0.12.0 | MIT | 0.18.1, June 8, 2026 [@botorch2026pypi] |
| Ax [@olson2025ax] | BoTorch | Experiment management on top of BoTorch: search spaces, trials, storage, automatic choice of method | MIT | 1.3.1, June 9, 2026 [@ax2026platform] |
| Optuna [@akiba2019optuna] | own; PyTorch for its Gaussian process sampler | Default sampler is TPE, a density-based method [@bergstra2011algorithms]; Gaussian process sampler since 3.6.0 [@optuna2026gpsampler] | MIT | 5.0.0, September 7, 2026 [@optuna2026pypi] |
| SMAC3 [@lindauer2022smac3] | own | Random forest or Gaussian process surrogates; algorithm configuration across problem instances | BSD-3-Clause | 2.4.1, September 10, 2026 [@automlorg2026smac] |
| HEBO [@cowenrivers2022hebo] | PyTorch | Input and output warping, ensembles of acquisition functions | MIT | 0.3.6, November 4, 2024 [@lab2024hebo] |
| Trieste | GPflow, TensorFlow | Gaussian process Bayesian optimization in the TensorFlow ecosystem | Apache-2.0 | 4.6.0, June 17, 2026 [@labs2026trieste] |
| scikit-optimize | scikit-learn | Simple interface; repository archived | BSD-3-Clause | 0.10.2, June 4, 2024 [@scikitoptimize2024pypi] |
| GPyOpt | GPy | Repository archived January 17, 2023 | BSD-3-Clause | 1.2.6, March 19, 2020 [@sheffieldml2023gpyopt] |
| Dragonfly | own | Effectively no longer maintained | MIT | 0.1.7, October 1, 2022 [@dragonfly2022opt] |
:::

A few remarks help in choosing. BoTorch is a research library: flexible, with
the newest acquisition functions, and assuming some knowledge of PyTorch. Ax
wraps it for people who want to run experiments rather than write acquisition
functions. Optuna is widely used for hyperparameter tuning in machine
learning, and its default sampler is not a Gaussian process; its
Gaussian process sampler uses a Matérn 5/2 kernel with one lengthscale per
input and the logarithmic forms of expected improvement and its constrained,
batch, and multi-objective variants [@optuna2026gpsampler]. SMAC suits problems
with many categorical and conditional parameters. Of these libraries, BoTorch
and Ax support learning from comparisons in the core package;
@sec-sw-software surveys the software for preferential Bayesian optimization
(PBO).

::: {.code title="BoTorch, one step of the loop (a sketch)"}
```python
import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

# toy data: 12 noisy evaluations of a function of 3 inputs, to be maximized
X = torch.rand(12, 3, dtype=torch.double)                          # n x d inputs
Y = -(X - 0.5).square().sum(dim=-1, keepdim=True)                  # n x 1 outputs
Y = Y + 0.05 * torch.randn_like(Y)
bounds = torch.tensor([[0.0] * 3, [1.0] * 3], dtype=torch.double)  # 2 x d: lower, upper

model = SingleTaskGP(X, Y, input_transform=Normalize(d=X.shape[-1]))
# outputs are standardized by default (BoTorch >= 0.12)
fit_gpytorch_mll(ExactMarginalLogLikelihood(model.likelihood, model))

acqf = qLogNoisyExpectedImprovement(model, X_baseline=X)
batch, _ = optimize_acqf(acqf, bounds=bounds, q=4, sequential=True,
                         num_restarts=10, raw_samples=512)
```
`optimize_acqf` runs the multi-start gradient ascent of @alg-acq-multistart:
it screens `raw_samples` quasi-random starting points, starts `num_restarts`
L-BFGS-B runs from promising ones, and with `sequential=True` builds the batch
of 4 one point at a time; the default, `sequential=False`, optimizes the 4
points jointly [@balandat2020botorch]. The listing runs as printed with
BoTorch 0.18.1.
:::

## Exercises {#sec-practice-exercises}

::: {.exercise #exr-practice-winners-curse}
Twenty-five evaluations all sit at the maximum, with true value $f^\star$, and
have independent noise $\varepsilon_i \sim \N(0, \sigma_n^2)$. Show that the
expected largest observation is $f^\star + \sigma_n m_{25}$, where $m_n$ is
the expected maximum of $n$ independent standard normal variables, and write
$m_n$ as an integral. The value is $m_{25} \approx 1.97$; compare it with the
dashed curve in @fig-practice-incumbent and with the rough approximation
$\sqrt{2 \ln n}$.

::: {.solution}
The largest observation is
$f^\star + \max_i \varepsilon_i = f^\star + \sigma_n \max_i z_i$ with
$z_i = \varepsilon_i/\sigma_n$ standard normal, so for $n$ evaluations its
expectation is $f^\star + \sigma_n m_n$. The maximum of $n$ independent
standard normals has distribution function $\Phi(z)^n$ and density
$n\,\phi(z)\Phi(z)^{n-1}$, so
$m_n = \int_{-\infty}^{\infty} z\, n\,\phi(z)\,\Phi(z)^{n-1}\,\dd z$. Numerically
$m_{25} \approx 1.97$. The figure's dashed curve, the average overstatement
after 25 evaluations, is 0.19 at $\sigma_n = 0.1$ and 0.99 at $\sigma_n = 0.5$,
close to $1.97\sigma_n$, although not all 25 evaluations sit at the top. The
approximation $\sqrt{2\ln 25} \approx 2.54$ overestimates; it is the leading
term of an expansion that converges slowly.
:::
:::

::: {.exercise #exr-practice-cei}
Let the utility of evaluating $\vx$ be the improvement
$\max\{f(\vx) - f^*_{n,\text{feas}}, 0\}$ if all constraints are satisfied at
$\vx$, and 0 otherwise. Show that if $f(\vx)$ is independent of the constraint
values $c_1(\vx), \dots, c_K(\vx)$ under the posterior, the expected utility
is @eq-practice-cei. What changes if a constraint is correlated with the
objective, for example when the fastest model also uses the most memory?

::: {.solution}
The utility is $I(\vx)\,\mathbf{1}[\text{feasible}]$, with
$I(\vx) = \max\{f(\vx) - f^*_{n,\text{feas}}, 0\}$. If $I$ and the indicator are
independent, the expectation of their product is the product of their
expectations: $\E[I(\vx)] \cdot \Prob(\text{feasible})$, which is
$\EI(\vx)$ against the best feasible value times the probability of
feasibility. With correlation, the factorization fails. If high objective
values tend to come with constraint violations, the product overstates the
expected utility, because the improvement is largest exactly where
feasibility is least likely; a joint model of objective and constraints, and
a Monte Carlo estimate of the expectation, are then needed.
:::
:::

::: {.exercise #exr-practice-prior-mode}
A log-normal distribution with location $\mu$ and scale $s$ has its mode at
$e^{\mu - s^2}$. Compute the mode of BoTorch's dimension-scaled lengthscale
prior, with $\mu = \sqrt{2} + \tfrac12 \ln d$ and $s = \sqrt{3}$, and compare
it with the root-mean-square distance $\sqrt{d/6}$ between two uniform random
points of the unit cube, for $d = 2, 10, 50$.

::: {.solution}
The mode is $e^{\sqrt{2} + \frac12 \ln d - 3} = \sqrt{d}\, e^{\sqrt2 - 3}
\approx 0.205\sqrt{d}$: 0.29, 0.65, and 1.45 for $d = 2, 10, 50$. The typical
distances are 0.58, 1.29, and 2.89. The ratio is constant,
$0.205\sqrt{d}/\sqrt{d/6} = 0.205\sqrt6 \approx 0.50$, so typical pairs of
points stay about two lengthscales apart in every dimension. A fixed prior
with mode near 0.5 would put them about 5.8 lengthscales apart at $d = 50$,
where the RBF kernel value is $e^{-5.8^2/2}$, about
$5 \times 10^{-8}$.
:::
:::

::: {.exercise #exr-practice-believer}
After a noise-free fantasized observation at $\vx_1$ with value equal to the
posterior mean $\mu(\vx_1)$, show that the posterior mean is unchanged
everywhere and that the posterior variance at $\vx_1$ becomes zero. What does
this imply for the kriging believer's next choice, and how would a pessimistic
constant lie change it?

::: {.solution}
By @eq-gp-posterior applied sequentially, adding an observation $y$ at
$\vx_1$ changes the mean at $\vx$ by $k_{n}(\vx, \vx_1)\,(y -
\mu(\vx_1))/\sigma_n^2(\vx_1)$, where $k_n$ is the current posterior
covariance. With $y = \mu(\vx_1)$ the correction is zero, so the mean is
unchanged everywhere. The variance update does not depend on $y$, and at
$\vx_1$ it becomes $\sigma_n^2(\vx_1) - \sigma_n^2(\vx_1)^2/\sigma_n^2(\vx_1) = 0$.
Expected improvement at $\vx_1$ therefore drops to
$\max\{\mu(\vx_1) - f^*_n, 0\}$, and nearby, where the variance has shrunk, it falls too, so the next
maximizer moves away, but only as far as the variance reduction reaches. A
pessimistic lie $y < \mu(\vx_1)$ also lowers the posterior mean around
$\vx_1$, which reduces expected improvement there further and pushes the next
point further away.
:::
:::

## Further reading {#further-reading .unnumbered}

- @snoek2012practical is the paper that brought Bayesian optimization to
  machine learning practice: the Matérn 5/2 kernel, averaging over
  hyperparameters, expected improvement per second, and parallel evaluations,
  in a few pages.
- @balandat2020botorch describes BoTorch's design: Monte Carlo acquisition
  functions, sample average approximation, and the one-shot knowledge
  gradient. Its appendix explains how the acquisition function is maximized.
- @garnett2023bayesian covers implementation in chapter 9 and extensions such
  as constraints, batches, costs, and multiple objectives in chapter 11.
- @eriksson2019scalable and @hvarfner2024vanilla are two short, readable
  entries into high-dimensional Bayesian optimization, from opposite
  directions; @sec-high-dimensions reports what followed.
- @letham2019constrained shows Bayesian optimization run on live online
  experiments, with noise and constraints, and the reasoning behind each
  choice.
