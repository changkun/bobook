---
status: done
synopsis: "Why Bayesian optimization was said to fail beyond 10 to 20 dimensions, what scalar BO learned about lengthscale priors and why, how far local preferential methods reach and what confounds them, and where pretrained surrogates, language models, and cost-aware stopping stand for comparisons."
---

# High Dimensions and the Changing Landscape of Bayesian Optimization {#sec-high-dimensions}

A widely repeated rule of thumb says that Bayesian optimization works on a
handful of inputs and stops working somewhere past 10 or 20. Frazier's
tutorial, a preprint that many papers cite, describes the method as
"best-suited for optimization over continuous domains of less than 20
dimensions" [@frazier2018tutorial], and a 2025 paper summarizes the common view
as "the number of optimization variables should not exceed 15 or 20"
[@xu2025standard]. For preferential Bayesian optimization (PBO) the rule
looked even safer, since a comparison carries at most one bit and the methods
of @sec-pbo were almost never tested above 20 dimensions.

Between 2024 and September 2026, research on Bayesian optimization with scalar
feedback (the ordinary setting of @sec-part-bo, where each evaluation returns a
number) changed in four ways that matter for preferences. The failure in high
dimension was traced to the lengthscale prior and its initialization rather
than to the dimension itself. Pretrained surrogates entered the main software
library. Language models entered optimization loops. And cost-aware stopping
rules acquired guarantees. This chapter takes them in that order and asks, for
each, what carries over to a pairwise comparison likelihood.

## The diagnosis: lengthscale priors {#sec-hd-diagnosis}

The quantity at the center of the story is the **lengthscale** of the kernel
(@sec-kernel-family, @sec-ard), roughly how far one must move along an input
before the function's value is essentially unrelated to where it started.
Software places a prior on it and fits it by maximizing the marginal likelihood
plus the log prior (@sec-fitting-hyperparameters), starting from the prior's
**mode**, its most probable value, and following gradients.

### Four papers, one diagnosis {#sec-hd-four-papers}

@hvarfner2024vanilla (ICML 2024) attributed the failure of standard Gaussian
process Bayesian optimization in high dimension to the complexity that the
usual priors impose on the objective: with lengthscales of a fixed size, a
function of many inputs is assumed to be far too complex to learn from the
available data. Their fix is to scale the lengthscale prior with the dimension.
With that change, standard Bayesian optimization performed best on three of
the five real tasks (all but Mopta08 and Ant), ahead of methods designed
specifically for high dimension. They also observed what the usual prior does
to the search: on the six-dimensional Hartmann function, Bayesian optimization
with the conventional prior, fitted by maximum a posteriori estimation, was
"effectively performing an initial random search followed by 140 iterations of
local search". BoTorch implements the fix as a log-normal prior with location
$\sqrt{2} + \tfrac{1}{2}\log d$, where $d$ is the input dimension, and scale
$\sqrt{3}$; the lengthscale is initialized at the prior's mode and constrained
to stay above 0.025 [@botorch2026gpytorch].

@xu2025standard (ICLR 2025) compared methods on 12 benchmarks and found that
standard Bayesian optimization with a Matérn kernel was consistently among the
best, while the RBF (squared exponential) kernel often failed. They traced the
failure to the initial lengthscale: when it is too short, the gradients used to
fit it **vanish**, meaning they become too small for the optimizer to move the
lengthscale at all. With the kernel written $\exp(-\lVert\vx - \vx'\rVert^2/\ell^2)$,
the factor that every lengthscale gradient carries falls below the rounding
unit of double-precision arithmetic once the scaled distance
$\rho = \lVert\vx - \vx'\rVert/\ell$ exceeds 6.58. In a numerical check with
uniformly sampled inputs, the RBF kernel's gradients began to vanish at 200
dimensions when initialized at 0.5 or 0.693, at 400 when initialized at 1, and
never (up to 600 dimensions) when initialized at $\sqrt{d}$; a robust
initialization fixes this with no extra prior.

@papenmeier2025understanding (ICML 2025) likewise found that "vanishing
gradients caused by Gaussian process (GP) initialization schemes play a major
role in the failures of high-dimensional Bayesian optimization (HDBO)", and that
maximum likelihood estimation of the lengthscales, without any prior, already
suffices for state-of-the-art performance. With a constant initial lengthscale
of $\ln 2 \approx 0.693$, their runs were satisfactory on the 124-dimensional
Mopta08 and 180-dimensional Lasso-DNA benchmarks but failed on the
888-dimensional Ant and 6392-dimensional Humanoid tasks. @hvarfner2025informed
(NeurIPS 2025) add that a space-filling initial design, the usual way to start
the loop (@sec-initial-design), works against learning hyperparameters, and
choose the first points to inform the hyperparameters as well.

The four papers agree on the diagnosis. A lengthscale prior or initialization
of fixed size makes the kernel values between points collapse toward zero as
the dimension grows, and with them the gradients of the hyperparameters.
Scaling the lengthscale with the square root of the dimension, through the
prior or through the initial value of a maximum likelihood fit, removes most of
the gap to specialized high-dimensional methods (inference, combining the
papers above).

### Seeing the diagnosis {#sec-hd-seeing}

Why should the square root of the dimension appear? Because that is how fast
typical distances grow. The derivation below needs nothing but expectations of
uniform random numbers.

::: {.derivation title="How far apart random points are, and what a kernel makes of it"}
Let $\vx$ and $\vx'$ be independent and uniform on the unit cube $[0, 1]^d$, and
let $r = \lVert \vx - \vx' \rVert$ be their distance.

1. The squared distance is a sum over coordinates,
   $r^2 = \sum_{i=1}^{d} V_i$ with $V_i = (x_i - x'_i)^2$, and the $V_i$ are
   independent because the coordinates are.
2. For one coordinate, with $u, v$ independent and uniform on $[0, 1]$:
   $\E[u^2] = 1/3$ and $\E[u] = 1/2$. Expanding the square,
   $\E[V] = \E[u^2] - 2\E[u]\E[v] + \E[v^2]$, which is
   $2/3 - 1/2 = 1/6$.
3. By linearity of expectation, $\E[r^2] = d/6$. The root-mean-square distance
   is $\sqrt{d/6}$: about 1.3 at $d = 10$, 2.9 at $d = 50$, 4.1 at $d = 100$.
4. A similar computation gives $\E[V^2] = \E[(u - v)^4] = 1/15$, so
   $\Var[V] = 1/15 - 1/36 = 7/180$, and by independence
   $\Var[r^2] = 7d/180$.
5. The spread of $r^2$ relative to its mean is
   $\sqrt{7d/180}\,/\,(d/6) \approx 1.18/\sqrt{d}$, which shrinks as $d$
   grows: in high dimension almost every pair of points is about equally far
   apart.
6. For $r$ itself, a first-order expansion of the square root around $d/6$
   gives a standard deviation of about
   $\sqrt{7d/180}\,/\,(2\sqrt{d/6})$, which simplifies to
   $\sqrt{7/120} \approx 0.24$, the same in every dimension. The distances form a bump of fixed width that slides to the
   right like $\sqrt{d/6}$.
7. The RBF kernel in the convention BoTorch uses is
   $k(r) = \exp(-r^2 / (2\ell^2))$. At a typical distance,
   $\log k \approx -d / (12\ell^2)$. For a fixed $\ell$ this falls linearly in
   $d$: with $\ell = 0.52$, about one factor of ten every 7.4 dimensions.
8. If instead $\ell = c\sqrt{d}$, the exponent is
   $(d/6) / (2c^2 d) = 1/(12c^2)$, the same in every dimension. The mode of
   BoTorch's dimension-scaled prior is
   $\exp(\sqrt{2} + \tfrac12\log d - 3)$, that is,
   $\sqrt{d}\,e^{\sqrt{2} - 3}$, so
   $c \approx 0.205$, the exponent is about 1.99, and a typical pair has
   $k \approx e^{-2} \approx 0.14$ whatever the dimension.
9. Every gradient of the marginal likelihood with respect to a lengthscale
   passes through the kernel values: for the RBF kernel with one lengthscale
   $\ell_i$ per input, $\partial k / \partial \ell_i = k \cdot (x_i - x'_i)^2 / \ell_i^3$.
   When $k$ collapses, so does every term of the gradient.
:::

@fig-hd-lengthscale makes this concrete. The reader picks the dimension; the
top panel shows the exact distribution of distances between two random points,
and the bottom panel shows the kernel value each default assigns at each
distance.

```{figure}
//| figure: hd-lengthscale
//| label: fig-hd-lengthscale
//| fig-cap: "Distances in the unit cube against default lengthscales. Top: the distribution of the distance between two independent uniform points in the $d$-dimensional unit cube, computed exactly (by convolving the one-coordinate law $d$ times, not by sampling). Dashed lines mark the lengthscale each default starts from: the mode of PairwiseGP's Gamma(2.4, 2.7) prior (0.52, also its initial value) and the mode of BoTorch's dimension-scaled log-normal prior (also its initial value). Bottom: the kernel value at each distance, on a logarithmic scale, with the middle 90% of pairs shaded; the dotted line is the rounding unit of double-precision numbers. The menu swaps the fixed default for optuna-dashboard's (Matérn 3/2, prior mode 0.4) or for the upper bound that local PBO places on the lengthscale (0.5). Illustrative simplifications: the points are uniform and independent, whereas Bayesian optimization queries are not; and every input has the same lengthscale, fixed at the prior's mode, whereas a fitted model learns one per input from data."
dim: 50
fixed: pairwisegp
```

Some things to try:

- Start at $d = 2$. Both lengthscales sit inside the cloud of distances, and
  both kernels give typical pairs values between 0.1 and 1.
- Move to $d = 20$, the edge of the rule of thumb. A typical pair is now about
  3.5 PairwiseGP lengthscales apart, with a kernel value near $2 \times
  10^{-3}$, while the dimension-scaled lengthscale has moved right with the
  distances and its typical kernel value is still about 0.14.
- At $d = 50$ the typical pair under the PairwiseGP default is 5.6 lengthscales
  apart and has kernel value $2.0 \times 10^{-7}$: a comparison involving one
  point says almost nothing about the utility at a typical other point.
- Keep going. The typical PairwiseGP kernel value crosses the
  double-precision line at about $d = 120$, and by $d = 200$ it is near
  $10^{-27}$, while the blue dot slides right with the median distance at the
  same height.
- Switch to optuna-dashboard. Its Matérn 3/2 kernel decays like
  $e^{-\sqrt{3}\,r/\ell}$ rather than $e^{-r^2/(2\ell^2)}$, so at $d = 50$ a
  typical pair still has $k \approx 5 \times 10^{-5}$ despite the shorter
  lengthscale. This is the mechanism behind Xu et al.'s finding that Matérn
  kernels are far less susceptible to vanishing gradients.

The figure also calibrates two claims (our computation). Kernel values between
typical points are *near zero* under the PairwiseGP default from about
$d = 30$, where $k$ is about $10^{-4}$, but fall *below double-precision
rounding* only beyond about 120 dimensions; Xu et al.'s threshold of 6.58, which
bounds the gradient factor $\rho^2 e^{-\rho^2}$ for a kernel without the factor
$\tfrac12$, corresponds in BoTorch's convention to $r/\ell \approx 9.3$, which
the typical pair reaches at about $d = 140$ (inference). At the 20 to 100
dimensions where preferential studies operate, the default preference model is
in the near-zero regime, with small hyperparameter gradients even where they
are not lost to rounding.

What that belief does to a whole optimization run is the next question.
@fig-hd-race replays recorded runs of scalar Bayesian optimization on the
six-dimensional Hartmann function padded with irrelevant inputs, under either
prior, with expected improvement maximized over uniform random candidates alone
or over those plus perturbations of the best points so far, the local move that
Papenmeier et al. identify. Because only six inputs matter, this is the
favorable case of low effective dimension.

```{figure}
//| figure: dim-bo-race
//| label: fig-hd-race
//| fig-cap: "Recorded runs on the six-dimensional Hartmann function hidden among irrelevant inputs, the figure of @sec-kern-race. Top: regret for random search and for Bayesian optimization with a fixed-scale or a dimension-scaled lengthscale prior, medians over six seeds with the range shaded. Middle: every point one run evaluated, in parallel coordinates. Bottom: the lengthscale that run's model learned for each input. Six seeds and 80 evaluations make this a demonstration, not a benchmark."
dims: 20
```

@sec-kern-race reads these runs in detail: the dimension-scaled prior gives
the six inputs that matter short lengthscales and switches most of the others
off, as designed, yet that difference barely shows in the regret. With the
local perturbations in the acquisition search, both priors reach a median
regret between 0.08 and 0.19 at every dimension from 6 to 50, against 1.26 to
1.75 for random search; with global random candidates only, both make little
progress at 50 dimensions (median regret 0.89 under the fixed-scale prior and
1.21 under the scaled one). In these runs, how the acquisition function is
searched matters more than which prior the model uses, the position of
Papenmeier et al. in @sec-hd-fixes. Six seeds, one test function, and 80
evaluations do not settle the debate, and a dimension-scaled prior that has
correctly identified the relevant inputs is the better model to have when the
budget grows (inference).

## Why the fixes work, contested {#sec-hd-fixes}

The papers agree on what goes wrong and disagree on why the fix helps. Three
explanations compete: that the dimension-scaled model is simply a good global
surrogate; that the success comes from local search; and that the benchmarks
are easier than their dimension suggests.

**The local-search explanation.** @papenmeier2025understanding give empirical
evidence that "good BO performance on extremely high-dimensional problems (on
the order of 1000 dimensions) is due to local search behavior and not to a
well-fit surrogate model". Their method, MSR, is maximum likelihood estimation
with a dimension-scaled initial lengthscale, plus starting candidates for the
acquisition search that perturb the best 5% of observed points, half of them in
every coordinate and half in a random subset of about 20 coordinates on
average. MSR beat the dimension-scaled prior on the 124-dimensional Mopta08 and
the 888-dimensional Ant tasks and was slightly worse on the others.

**The easy-benchmark explanation.** The same paper, in its main text and
Appendix D, argues that two popular benchmarks, the 180-dimensional Lasso-DNA
and the 124-dimensional Mopta08, "are not truly as high-dimensional as their
nominal number of input variables might suggest": in the best solutions found,
many variables sit on the boundary of the search space, and every method
converges to similarly long lengthscales. The authors warn of the risk of
algorithms being "'overfitted' to these benchmarks".

**The strongest challenge.** @doumont2026we, which received the AISTATS 2026
best student paper award, replaced the Gaussian process with Bayesian linear
regression, which is a Gaussian process with a linear kernel, after mapping
inputs onto a sphere. The mapping matters: they prove that a linear model with
an acquisition function that increases in the posterior mean and variance always
proposes points with at least one coordinate on the boundary of the search
space, and the spherical mapping removes that pull. The result matches the
state of the art on tasks with **60 to 6,000 dimensions**, and because a linear
model's cost grows linearly with the data, they scaled it to more than 20,000
observations on molecular tasks. Their analysis shows that "model
expressiveness and predictive accuracy on random test points do not necessarily
translate to optimization performance with adaptively chosen data". Two
related results concern the search itself: measures of how much an
acquisition function explores, related to performance
[@papenmeier2025exploring] (UAI 2025), and the finding that Thompson-sampling
candidate sets become exponentially sparse as dimension grows
[@fan2026adaptive] (AISTATS 2026).

Doumont et al. argue against the view that an expressive nonparametric model is
the key to success, not against dimension scaling as a practice (inference).
Whatever the mechanism, the practical direction is clear: on the common
high-dimensional benchmarks, simple approaches, such as a linear model with an
input mapping, or maximum likelihood with local perturbations, perform as well
as elaborate ones.

## Restating the dimension ceiling {#sec-hd-ceiling}

A common claim in the preferential literature is that Gaussian process PBO
fails beyond 10 to 20 dimensions. This section checks the claim against the
evidence and replaces it with a statement that names its conditions.

### Where preferential studies have been {#sec-hd-pbo-dims}

Almost all empirical work on PBO stays within 20 dimensions. The early
subspace methods restricted each query to a one- or two-dimensional slice
through the current best point (@sec-line-search, @sec-gallery-projective):
sequential line search [@koyama2017sequentialb] gives the person one slider;
the plane search of the Sequential Gallery [@koyama2020sequential] reached 20
dimensions on the synthetic Rosenbrock function and handled 12-dimensional
photo enhancement and a 10-dimensional human body-shape space, with going
beyond 20 dimensions listed as an open problem; and projective preferential
Bayesian optimization (PPBO) [@mikkola2020projective] was tested up to the
20-dimensional Ackley function, an experiment that took 24 hours in total.
@astudillo2023qeubob described their 4- to 7-dimensional qEUBO experiments as
more challenging problems of moderate dimension (more than 3), noting that
earlier work was mostly limited to low dimensions. The highest-dimensional
study with human participants is GimmBO [@liu2026gimmbo] (SIGGRAPH North
America 2026, published in ACM Transactions on Graphics), which optimizes the
merging weights of 20 to 30 diffusion-model adapters from preferences; in its
user study, on 20-dimensional inputs, 12 participants did 20 iterations with
each of three interfaces, and its two-stage backend exploits sparsity and a
restricted range of weights.

### What the software defaults do {#sec-hd-defaults}

The preference software did not follow the scalar fix. BoTorch 0.12.0
(September 17, 2024) switched most models to the dimension-scaled log-normal
prior, and its changelog says plainly that "the only models that are _not_
changed are those for fully Bayesian models and `PairwiseGP`"
[@botorch2026changelog]; in 0.18.1 (June 8, 2026), `PairwiseGP` still defaults
to an RBF kernel with a Gamma(2.4, 2.7) lengthscale prior
[@botorch2026pairwisegp], and the preferential sampler of optuna-dashboard
0.21.0 (September 10, 2026) to a Matérn 3/2 kernel with a Gamma(5, 10) prior
[@optuna2026sampler; @optuna2026dashboard], both independent of dimension
(@tbl-sw-priors). Computed from the BoTorch source, the mode of the
dimension-scaled prior on the unit cube, which is also its initial lengthscale,
is about 0.29, 0.65, 0.92, 1.45, and 2.05 at $d = 2$, 10, 20, 50, and 100,
against about 0.52 in every dimension for `PairwiseGP` and 0.4 for
optuna-dashboard. With random points about $\sqrt{d/6}$ apart, 2.9 at $d = 50$
and 4.1 at $d = 100$, the default preference models are in the regime that Xu
et al. and Papenmeier et al. identified, in the graded sense of @sec-hd-seeing
(inference, computed; not verified directly on `PairwiseGP`).

### Learning lengthscales from comparisons {#sec-hd-learning-ls}

How lengthscales are learned from pairwise data at all has received little
study. @sec-obs-graphs lists the approaches in use, which range from a weak
Gamma prior with the Laplace evidence (BoTorch) to a tight log-normal prior
that nearly fixes the lengthscale [@koyama2020sequential]. In BoTorch's probit
parameterization the noise scale is fixed at 1, so the output scale acts as the
inverse of the noise level, and both it and the lengthscales must be learned
jointly from comparisons alone [@botorch2026pairwisegp]. **We found no study of
whether lengthscales or output scales can be identified from comparisons at
human-scale budgets, tens to one or two hundred comparisons, and no paper that
applies Hvarfner et al.'s dimension-scaled prior, Xu et al.'s robust
initialization, or the MSR initialization to a pairwise or ranking
likelihood.** Both questions can be answered in simulation before any person is
involved: fit the model to comparisons simulated from known lengthscales at
those budgets, with and without a dimension-scaled prior, and check whether the
fit recovers them and what the hyperparameter gradients look like under the
Laplace evidence (inference).

On the theoretical side, @ziomek2024bayesian (NeurIPS 2024) point out that
maximum likelihood estimates of the lengthscale can be misspecified if the
objective is rougher in regions not yet explored; their method keeps several
candidate lengthscales alive at once and achieves regret (@sec-regret) within a
factor of $\log g(T)$ of the regret with known hyperparameters, against a
factor of $g(T)$ for the earlier A-GP-UCB, but has not been combined with a
pairwise likelihood.

### The restated claim {#sec-hd-restated}

The evidence therefore supports the 10 to 20 dimension rule as an empirical
description of the default configuration, and not as a limit intrinsic to
preference feedback: the scalar evidence shows that the same empirical
threshold comes mainly from the default prior, and the preference software
still uses such priors (inference). A more accurate statement is the following.

::: {.keyidea title="The dimension ceiling, restated"}
With lengthscale priors of fixed scale (not scaled with dimension) and global
acquisition functions, Gaussian process PBO degrades beyond about 10 to 20
dimensions. Scalar evidence shows that dimension-scaled priors combined with
local or trust-region search can work in tens to hundreds of dimensions when
the number of evaluations is about ten times the dimension and the effective
dimension is low. Whether this holds with human pairwise feedback at budgets of
50 to 200 comparisons is the open question of @sec-hd-learning-ls.
:::

The statement depends on five conditions, listed in @tbl-hd-conditions.

::: {.table #tbl-hd-conditions title="Five conditions that limit how far scalar high-dimensional results carry over to preferences."}
| Condition | What it says | Basis |
|---|---|---|
| Budget | Scalar successes use about as many observations as dimensions, or more (about 1,000 observations at $d$ up to 6,000); local PBO uses about ten comparisons per dimension (about 640 at 64 dimensions, about 1,500 at 102); a human session typically allows tens to one or two hundred comparisons | Doumont et al. 2026; Menn et al. 2026; (inference) |
| Information per query | At most one bit per query; the Hessian of the Laplace posterior mean is more sensitive than its gradient to the lengthscale, the conditioning of the kernel matrix, and the local geometry of the sampled points | Menn et al. 2026, Lemma 1 and the discussion after it; Kayal et al. 2025 [@kayal2025bayesian] give rates only |
| Human noise | Existing high-dimensional preference studies simulate homoscedastic noise (10% of the function's range); real comparisons show intransitivity, drift, and fatigue, which none of these papers models | Menn et al. 2026; @sec-sw-simulated-users |
| Effective dimension | Successful benchmarks often have few influential variables, or optima on the boundary; GimmBO relies on sparse structure at 20 to 30 dimensions | Papenmeier et al. 2025, Appendix D; Liu et al. 2026 |
| Starting point | Local methods need a reasonable initial region, or they settle in poor local optima | Menn et al. 2026 |
:::

One more caution belongs with the restated claim. A dimension-scaled prior does
not make the lengthscale identifiable from the data; it replaces data with a
stronger prior belief in smoothness, which helps only when the true utility is
smooth or its effective dimension is low (inference).

### What to do above 20 dimensions {#sec-hd-practice}

In decreasing order of evidential support, the options for a preferential
problem with more than 20 inputs are these (inference). First, reduce the number
of dimensions the person faces, for instance by optimizing in the
low-dimensional latent space of a generative model, as the Sequential Gallery
does with its body-shape space, or over a small number of patient-specific
parameters, as in @granley2023human (NeurIPS 2023). Second, consider a linear
utility model, on the strength of Doumont et al.'s results. Both steps come from
the structure of the problem and do not depend on the Gaussian process
framework. Third, if `PairwiseGP` is kept, pass it a dimension-scaled kernel or
adopt the initialization of Xu et al. or Papenmeier et al., and verify the
result yourself on simulated comparisons first: this is a repair inside the
Gaussian process framework with no test under pairwise feedback.

::: {.code title="A dimension-scaled kernel for PairwiseGP"}
```python
from botorch.models import PairwiseGP
from botorch.models.utils.gpytorch_modules import get_covar_module_with_dim_scaled_prior
from gpytorch.kernels import ScaleKernel

D = X.shape[-1]
# PairwiseGP raises an error unless covar_module is a ScaleKernel. A custom
# ScaleKernel does not carry over the default's output-scale prior
# (SmoothedBox on [0.01, 100]) or its constraint ([0.005, 200]); add them if
# you want them. The base kernel is RBF with LogNormal(sqrt(2) + log(D)/2,
# sqrt(3)) lengthscale priors, initialized at the mode.
covar = ScaleKernel(get_covar_module_with_dim_scaled_prior(ard_num_dims=D))
model = PairwiseGP(X, comparisons, covar_module=covar)
```
:::

The comments restate what the two source files do
[@botorch2026pairwisegp; @botorch2026gpytorch].

## Local PBO {#sec-hd-local}

The only work that pushes PBO to about 100 dimensions is the local preferential
Bayesian optimization of @menn2026local, a preprint (version 1 on June 1, 2026,
version 2 on June 8). The authors argue that existing methods "struggle to
efficiently optimize beyond low- and medium-dimensional problems due to their
global search approaches". They transfer two ideas from scalar
high-dimensional optimization. TuRPBO is a preference version of TuRBO, which
confines each query to a **trust region**, a box around the current best point
that grows after successes and shrinks after failures. GIPBO and PrefSQP follow
derivatives of the posterior: PrefSQP, named after sequential quadratic
programming, takes steps based on the gradient and the Hessian (the matrix of
second derivatives) of the Laplace posterior mean.

The experiments cover Gaussian process sample paths in 8, 16, 32, 64, and 96
dimensions; the Hartmann (6 dimensions), Rosenbrock (16), Levy (32), and
Styblinski-Tang (64) test functions; and linear control policies for the MuJoCo
Hopper (33 dimensions) and Walker2D (102 dimensions) simulated robots. The
budget is $2 + 10d$ evaluations for the synthetic tasks and $5d$ random warm-up
evaluations plus $10d$ iterations for policy search, where $d$ is the
dimension; for most methods $N$ evaluations give $N - 1$ comparisons. Comparisons are simulated
with Gaussian noise whose standard deviation is 10% of the function's range.
The model is BoTorch's `PairwiseGP` with an RBF kernel, and the
lengthscales are constrained to **[0.05, 0.5]** on the unit cube. The baselines
are qEUBO, the hallucination believer with expected improvement
(@sec-choosing-pairs), GLISp (@sec-obs-surrogates), and a Sobol sequence (a
quasi-random, evenly spread set of points).

The local methods "excel in high-dimensional, short-lengthscale problems". The
authors also state the limits: for very smooth functions locality helps little;
"with only two initial points, local methods can get trapped in poor local
optima and underperform global baselines"; local PBO is most useful "when
prior knowledge or an existing controller provides a reasonable starting
region"; and the synthetic and policy-search benchmarks "do not fully capture
the structure of human preferences".

**The comparison is confounded.** The lengthscale constraint of [0.05, 0.5]
applies to all the `PairwiseGP`-based methods, the qEUBO baseline included. It
rules out the long lengthscales that a dimension-scaled prior would choose
(about 1.45 at 50 dimensions and 2.05 at 100), so the global baseline runs in
the degenerate regime that Xu et al. and Papenmeier et al. identified. The
authors motivate the constraint by the local methods' need to avoid excessively
large steps, and the paper cites none of Hvarfner et al., Xu et al., or
Papenmeier et al. The confound arises wherever the hyperparameters are learned;
on the Gaussian process sample paths the paper also reports a setting with
known hyperparameters (true lengthscales 0.1 and 0.5), where the bound does not
bind. A fair test would give qEUBO a dimension-scaled prior and
then compare it with the local methods (inference). @fig-hd-local shows the
size of the effect at 64 dimensions.

```{figure}
//| figure: hd-lengthscale
//| label: fig-hd-local
//| fig-cap: "The lengthscale bound of local PBO at 64 dimensions, the size of the Styblinski-Tang benchmark. Even at the bound's upper end, 0.5, a typical pair of points is about 6.5 lengthscales apart and has kernel value about 6×10⁻¹⁰, while the dimension-scaled prior would start at 1.64 with typical kernel values near 0.14. Uniform random pairs and a single shared lengthscale are illustrative simplifications; the global baseline in the paper fits per-input lengthscales within the bound."
dim: 64
fixed: local
```

Seen from the side of query design, the local methods continue the subspace
methods of @sec-hd-pbo-dims and LineCoSpar [@tucker2020human]: shrink the
region each query must search (inference).

## Pretrained and in-context surrogates {#sec-hd-pretrained}

A different answer to scarce data is to learn the prior itself from many
synthetic problems. A **prior-data fitted network** (PFN) is a neural network
trained on a very large number of small data sets drawn from a chosen prior over
functions; given a new small data set as input, it outputs a posterior
predictive distribution in a single forward pass, without any fitting. This is
**in-context learning**: the data set is the input, not the training set.
TabPFN is a model of this kind for tabular data. In a position paper,
@muller2025position (ICML 2025) argue that prior-fitted networks are the future
of Bayesian prediction in data-scarce problems, and BoTorch integrated a PFN
surrogate in 0.14.0 (May 6, 2025), with Ax compatibility, Monte Carlo
acquisition functions, and batched noisy expected improvement following through
0.17.0 (February 2026) [@botorch2026changelog]. These models live in the
`botorch_community` package, which has no pairwise or preference models.

**The calibration evidence comes from hyperparameter tuning.**
@rogers2026zero (AutoML 2026, per the Amazon Science page; we found no arXiv
version) used TabPFN v2 as a zero-shot surrogate, with no training on the task
at hand. On the 16 search spaces of the hyperparameter-optimization benchmark
HPO-B, at a budget of 50 trials, it beat 5 of 7 baselines and tied with 2, and
the **empirical coverage of its 95% intervals was 0.94 on HPO-B and 0.95 on a
second benchmark, YAHPO-Gym**, meaning that 94% to 95% of true values fell
inside intervals meant to hold 95% of them. Among the limits the authors name:
five seeds per search space, the need for a GPU, and that all evaluations are
hyperparameter-optimization benchmarks. In high dimension there is one paper:
GIT-BO [@yu2026git] (ICLR 2026) uses gradients of TabPFN v2's predictive mean
to find an **active subspace**, the few directions along which the function
changes most, and on 20 benchmarks with 60 problem variants up to 500
dimensions reports a better trade-off between performance and run time than
four Gaussian process methods for high dimension, among them Hvarfner et al.'s
dimension-scaled prior; its authors acknowledge the memory footprint and the
dependence on the foundation model's capacity.

**Fully amortized optimizers are multiplying.** An amortized optimizer is
trained once, in advance, on many tasks, so that a new task needs only forward
passes, as in ZeroShotOpt [@meindl2025zeroshotopt], a preprint pretrained on
the trajectories of 12 Bayesian optimization variants, and TAMO
[@zhang2026context] (ICLR 2026), a multi-objective in-context optimizer whose
authors expect performance to be sensitive to the synthetic Gaussian process
corpus.

**Three limits for human comparisons.** Carrying these methods over to pairwise
preferences runs into three obstacles.

1. *Nothing is trained on comparisons except PABBO*, the only amortized
   optimizer for pairwise preferences, whose claims and limits
   @sec-obs-surrogates reports; TAMO cites it as amortized optimization from
   binary feedback and leaves preferential feedback to future work. We found no
   PFN or TabPFN surrogate trained on comparison data and paired with a
   preference acquisition function such as EUBO or Bayesian active learning by
   disagreement (BALD, which picks the query whose answer the model's
   plausible hypotheses disagree about most).
2. *A PFN cannot tell noise from ignorance.* @bergna2026decoupled, a 2026
   preprint, show that a standard PFN outputs a predictive distribution over
   noisy observations and that "this epistemic–aleatoric split is not
   identifiable in general from the posterior predictive distribution alone,
   even when that distribution is known exactly". **Epistemic** uncertainty is
   what the model does not yet know and could learn; **aleatoric** uncertainty
   is noise that no amount of data removes. Training with separate heads for
   the latent signal and the noise mitigates the problem. The noise in human
   comparisons is aleatoric, while BALD-type pairwise acquisition functions
   need the epistemic part, which a standard PFN cannot separate out
   (inference).
3. *The prior must look like human utilities.* Pairwise feedback carries little
   information per query, which is exactly where a strong prior helps most; but
   the prior must match the shape of human utilities. PABBO's synthetic prior,
   the same in its training and evaluation configurations, draws lengthscales
   from a distribution centered near 1/3 and truncated to [0.05, 2], with RBF
   and Matérn 5/2, 3/2, and 1/2 kernels [@zhang2025pabbob; @zhang2025pabboc], a
   general prior over functions rather than a prior over human utilities
   (inference).

## Language models as optimizers and as components {#sec-hd-llm}

Language models entered Bayesian optimization in two roles, with opposite
results so far. This section covers the scalar side; @sec-llms treats the
relation in full, and @sec-llm-in-loop the uses inside a preference loop.

**As autonomous optimizers, the controlled evidence is negative.**
@gupta2025llms (EMNLP 2025 Findings) found, on gene perturbation and molecular
property tasks, that replacing the true outcomes with randomly permuted labels
had no effect on the performance of language-model agents, and classical
methods such as linear bandits and Gaussian process optimization consistently
won. An optimizer that does equally well with shuffled feedback is not using the
feedback.

**As components of a Gaussian process, the evidence is positive.** GOLLuM
[@rankovic2026large] (Nature Machine Intelligence 2026) trains language-model
embeddings and a Gaussian process jointly through the marginal likelihood. On
23 chemistry and materials tasks, at 50 experiments it covered 36.3% of the top
5% of candidates, against 26.5% for a Gaussian process on fixed embeddings, and
it reached the level of conventional Bayesian optimization with a median of
41% fewer iterations. LGBO [@yuan2026unleashing] (ICLR 2026) uses a language model's
judgment about promising regions as a shift of the Gaussian process mean and
proves that, in the worst case, it is not significantly worse than standard
Bayesian optimization. Human priors need not come from a language model:
@xu2024principled (NeurIPS 2024) prove that with expert advice, even
adversarial advice, convergence is no slower than without it.

These results point to one design (inference): an external prior enters through
the mean or the acquisition function, with a safeguard, and the Gaussian process
makes the final decision. For PBO, where LILO [@kobalczyk2026lilo] already
turns free-text feedback into pairwise labels for `PairwiseGP`, there is a
further caution (inference): comparisons generated by a language model are
correlated labels with systematic biases, not independent probit noise, and the
likelihood of `PairwiseGP` does not model a shared bias.

## Cost-aware stopping {#sec-hd-stopping}

Every optimization loop has to stop, and a person in the loop makes stopping
costly in a new way. @sec-stopping described three scalar rules with
guarantees [@ishibashi2023stopping; @wilson2024stopping; @xie2026cost]. The
cost-aware one, from @xie2026cost (ICML 2026), continues as long as some
point's expected improvement exceeds its cost of evaluation; combined with the
Pandora's Box Gittins index of @sec-cost-aware [@xie2024cost] or with log
expected improvement per unit cost, it bounds the expected cost-adjusted simple
regret, which the authors describe as the first guarantee in correlated
Bayesian optimization of doing no worse than stopping immediately. Three
limitations can be read off the paper: the guarantee is proved for the
unsmoothed rule, while the 8-dimensional experiments smooth the stopping signal
with a moving average over 20 iterations; the guarantee is Bayesian, that is,
it holds under the assumed Gaussian process prior, and results are slightly
worse on two NATS-Bench data sets (CIFAR-100 and ImageNet16-120), which the
authors attribute to likely model misspecification; and the cost in the main
experiments is a fitted proxy for run time [@xie2026cost]. BoTorch 0.18.1 has
no PBGI acquisition function [@botorch2026changelog].

**Preference results.** The only explicit optimal stopping rule on the
preference side is Theorem 3 of @byk2019asking (CoRL 2019): subtract a cost per
query from the information gain, and stop asking when the best achievable value
becomes negative. It assumes a parametric reward model, not a Gaussian process
over comparisons. In practice, systems let the person stop: in the user study
of the Sequential Gallery, participants pressed a "satisfied" button after 5.36
iterations on average, although they were asked to continue to 15 iterations
[@koyama2020sequential]. When people stop depends on who
they are. In a study of 60 participants by @ou2023impact (IUI 2023), on text,
photo, and 3D mesh tasks, experts iterated more, expressed clearer preferences,
and were less satisfied, while novices were more easily satisfied and finished
sooner (@sec-hci-expertise).

**The gap is specific** (inference). The natural preferential version of
cost-aware stopping compares the EUBO gain of the next comparison with the
human cost of making it, but no paper derives such a rule or proves a similar
guarantee under a probit or Bradley-Terry likelihood with a Laplace posterior.
Because the guarantees depend on a correctly specified prior, and lengthscales
are hard to identify from small pairwise budgets, the guarantee is weakest
exactly where it is most needed. Wilson's rule needs only posterior samples of
the latent function, which `PairwiseGP` can provide, so it is usable
mechanically, but its reliability under the Laplace approximation and human
noise is untested. The cost in these papers is run time; a person's cost
changes with fatigue and learning, which violates the assumption that the cost
is known or learnable. Transfer across users is untested too:
@hvarfner2026pitfalls, a 2026 preprint, find that the multi-task Gaussian
process "misestimates the cross-task correlation even in the simplest
non-trivial case", that of affinely related source and target tasks, and
transferring utility models between people adds the further problem that a
utility is identified only up to a monotone transformation. Until a rule exists,
the practical stopping rule remains the person's own "satisfied", recorded
together with how far the posterior mean still moves between iterations
(inference).

## Settled, contested, missing {#sec-hd-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** With scalar feedback, a lengthscale prior or initialization of
fixed scale drives kernel values toward zero and hyperparameter gradients toward
vanishing in high dimension, and a dimension-scaled prior or initialization
removes most of the gap to specialized high-dimensional methods
[@hvarfner2024vanilla; @xu2025standard; @papenmeier2025understanding]. The
default lengthscale priors of BoTorch's `PairwiseGP` and of optuna-dashboard do
not scale with dimension. PABBO is the only amortized optimizer for pairwise
preferences. Language models acting as autonomous optimizers were insensitive to
their feedback in a controlled test.

**Contested.** What makes high-dimensional optimization succeed: a good global
surrogate, local search, or easy benchmarks. Whether local PBO's advantage over
global methods comes from the lengthscale constraint placed on the global
baseline. How sample-efficient pretrained surrogates are in high dimension, with
GIT-BO the only paper. How cost-aware stopping guarantees behave when the
prior is misspecified.

**Missing.** A test of dimension-scaled priors or robust initialization under a
pairwise likelihood. A measurement of hyperparameter gradients under the Laplace
marginal likelihood. A study with human pairwise feedback above 30 dimensions at
a budget of 50 to 200 comparisons. A prior-fitted network trained and
calibrated on comparison data. A stopping rule designed for pairwise
likelihoods and human fatigue.
:::

## Exercises {#sec-hd-exercises}

::: {.exercise #exr-hd-scaling title="Choosing the constant in the square-root rule"}
A lengthscale of the form $\ell = c\sqrt{d}$ keeps the typical RBF kernel value
between uniform random points in $[0, 1]^d$ the same in every dimension. Which
$c$ makes a typical pair exactly one lengthscale apart, so that the kernel value
at the root-mean-square distance is $e^{-1/2}$? How does BoTorch's choice
compare?

::: {.solution}
At the root-mean-square distance, $r^2 = d/6$, so the exponent is
$r^2/(2\ell^2) = (d/6)/(2c^2 d) = 1/(12c^2)$. Setting it to $\tfrac12$ gives
$c^2 = 1/6$, so $c = 1/\sqrt{6} \approx 0.41$: the lengthscale equals the
root-mean-square distance $\sqrt{d/6}$. The mode of BoTorch's prior has
$c = e^{\sqrt{2} - 3} \approx 0.205$, half of that, so a typical pair starts
about two lengthscales apart, with kernel value near $e^{-2} \approx 0.14$, as
in @fig-hd-lengthscale.
:::
:::

::: {.exercise #exr-hd-threshold title="When the default crosses a threshold"}
Using step 7 of the derivation in @sec-hd-seeing, $\log k \approx -d/(12\ell^2)$
at the root-mean-square distance, estimate the dimension at which the PairwiseGP
default ($\ell = 1.4/2.7$) gives typical pairs a kernel value below $10^{-3}$,
and the dimension at which it falls below the double-precision rounding unit
$2^{-53}$. Compare with the figure.

::: {.solution}
Solving for $d$ gives $d \approx 12\ell^2 \ln(1/k)$, with
$12\ell^2 = 12 \times (1.4/2.7)^2 \approx 3.23$. For $k = 10^{-3}$,
$\ln 1000 \approx 6.91$ and $d \approx 22$. For $k = 2^{-53}$,
$\ln 2^{53} \approx 36.7$ and $d \approx 119$. The figure, which evaluates the
kernel at the median rather than the root-mean-square distance, puts the second
crossing between $d = 118$ and $d = 120$. The rule of 10 to 20 dimensions sits
just below the first threshold.
:::
:::

::: {.exercise #exr-hd-matern title="Why the Matérn kernel lasts longer"}
optuna-dashboard's Matérn 3/2 kernel is $k(\rho) = (1 + \sqrt{3}\,\rho)\,e^{-\sqrt{3}\,\rho}$
with $\rho = r/\ell$ for a distance $r$, and prior mode $\ell = 0.4$. Show that at a typical distance
$\log k$ falls roughly like $-\sqrt{d}$ rather than like $-d$, and estimate the
dimension at which a typical pair's kernel value reaches $2^{-53}$.

::: {.solution}
For large $\rho$ the exponential dominates, $\log k \approx -\sqrt{3}\,\rho$, and
$\rho \approx \sqrt{d/6}/0.4$, so $\log k \approx -\sqrt{3}\sqrt{d/6}/0.4 \approx
-1.77\sqrt{d}$: a square root, not a linear function, of $d$. Solving
$\ln(1 + \sqrt{3}\,\rho) - \sqrt{3}\,\rho = -36.7$ gives $\rho \approx 23.4$, so
$r \approx 9.34$ and $d = 6r^2 \approx 524$, beyond the range of the figure.
The RBF default reaches the same value near $d = 119$. This slower decay is
the mechanism behind Xu et al.'s finding that Matérn kernels are far less
susceptible to vanishing gradients [@xu2025standard]. (Their own threshold for
the Matérn 5/2 kernel, $\rho = 21.98$, concerns the factor that the gradients
carry; the 524 here is where the kernel value of optuna-dashboard's Matérn 3/2
reaches $2^{-53}$.)
:::
:::

## Further reading {#further-reading .unnumbered}

- @hvarfner2024vanilla is the paper that changed the defaults; its appendix on
  what the conventional prior does to exploration is worth reading on its own.
- @xu2025standard give the gradient argument in two pages and the numerical
  check that shows when gradients actually vanish.
- @papenmeier2025understanding and @doumont2026we are the two sides of the
  mechanism debate; Doumont et al.'s Theorem 1 on linear models and the
  boundary is short and surprising.
- @menn2026local is the reference for local and trust-region PBO; read its
  limitations section and its hyperparameter table together.
- @xie2026cost is the clearest statement of a stopping rule with a guarantee,
  and the natural starting point for a preferential version.
