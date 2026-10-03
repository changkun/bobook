---
status: done
synopsis: "Conditioning a Gaussian process on observations: the predictive equations, how the posterior mean and uncertainty respond to data and noise, and how to compute them stably."
sources: ["Rasmussen and Williams 2006, ch. 2", "Garnett 2023, ch. 2"]
---

# Gaussian Process Regression {#sec-gp-regression}

@sec-function-space ended with a prior: a Gaussian process that assigns a
probability to every function before any data arrives. This chapter adds the
data. We evaluate the unknown function at a few inputs, and ask what the prior,
together with those values, says about the function everywhere else.

The answer needs no new machinery. A Gaussian process says that any finite set
of function values is jointly Gaussian, and @sec-gaussian-conditioning showed
how to condition a joint Gaussian on some of its coordinates. Gaussian process
regression is that one formula, applied with the observed inputs on one side
and the inputs we want to predict on the other. Everything else in the chapter
is about reading the result: what the posterior mean does between and beyond
the data, why the uncertainty depends on where we looked but not on what we
saw, what changes when observations are noisy, and how to compute all of it
without numerical trouble.

The posterior uncertainty is the quantity the rest of the book spends. Every
acquisition function in @sec-acquisition is a rule for turning it into the next
query, so it is worth knowing its shape well.

## Conditioning on observations {#sec-gp-conditioning}

Write $f$ for the unknown function on an input domain $\X$, and suppose it has
a Gaussian process prior with mean zero and kernel $k$:

$$
f \sim \GP(0, k).
$$

We observe $f$ at $n$ inputs $\vx_1, \dots, \vx_n$, collected in a set $X$, and
for now the observations are exact: $y_i = f(\vx_i)$. Stack the values into a
vector $\vy = (y_1, \dots, y_n)^\T$. We want the distribution of $f$ at $m$ new
inputs $X_*$, whose unknown values we stack into $\vf_*$.

The smallest case was worked in @ex-gauss-two-inputs: two function values
with correlation 0.8, one of them observed at 1.2, and a belief about the
other that moved to mean 0.96 and standard deviation 0.6. What follows is
that computation with $n$ observed values, any number of unobserved ones, and
a kernel supplying the correlations.

By the definition of a Gaussian process, $\vy$ and $\vf_*$ are jointly
Gaussian. Their covariance is built entry by entry from the kernel, so it has
four blocks: covariances among the observed inputs, between observed and new
inputs, and among the new inputs.

$$
\begin{bmatrix} \vy \\ \vf_* \end{bmatrix}
\sim \N\!\left( \mathbf{0},\;
\begin{bmatrix} \mK & \mK_* \\ \mK_*^\T & \mK_{**} \end{bmatrix} \right).
$$ {#eq-gp-joint}

Here $[\mK]_{ij} = k(\vx_i, \vx_j)$ holds the covariances among observed
inputs, $[\mK_*]_{ij} = k(\vx_i, \vx_{*j})$ those between observed and new
inputs, and $[\mK_{**}]_{ij} = k(\vx_{*i}, \vx_{*j})$ those among new inputs.

Conditioning on $\vy$ is now the operation from @sec-gaussian-conditioning.

::: {.derivation title="The predictive distribution"}
By @eq-gauss-conditional, for a joint Gaussian with blocks $\vmu_a, \vmu_b$
and $\mSigma_{aa}, \mSigma_{ab}, \mSigma_{bb}$, the conditional of
$\mathbf{b}$ given $\mathbf{a}$ is Gaussian with mean
$\vmu_b + \mSigma_{ab}^\T \mSigma_{aa}^{-1}(\mathbf{a} - \vmu_a)$
and covariance $\mSigma_{bb} - \mSigma_{ab}^\T \mSigma_{aa}^{-1} \mSigma_{ab}$.

1. Take $\mathbf{a} = \vy$ and $\mathbf{b} = \vf_*$. Both prior means are zero, so
   $\vmu_a = \vmu_b = \mathbf{0}$.
2. Read the blocks from @eq-gp-joint: $\mSigma_{aa} = \mK$,
   $\mSigma_{ab} = \mK_*$, $\mSigma_{bb} = \mK_{**}$.
3. Substitute. The conditional mean is $\mK_*^\T \mK^{-1} \vy$ and the
   conditional covariance is $\mK_{**} - \mK_*^\T \mK^{-1} \mK_*$.
:::

So the posterior over the new values is

$$
\vf_* \mid X, \vy, X_* \sim \N\!\left(\mK_*^\T \mK^{-1} \vy,\;
\mK_{**} - \mK_*^\T \mK^{-1} \mK_*\right).
$$ {#eq-gp-posterior}

Nothing in this derivation depended on how many new inputs we chose, or which.
The posterior is again a Gaussian process: for any finite set of new inputs,
the predictions are jointly Gaussian with the mean and covariance above. That
closure is what makes the method practical. The data are absorbed once, and the
result can be queried anywhere.

For a single new input $\vx$, the blocks become a vector and two numbers. Write
$\vk(\vx) = (k(\vx, \vx_1), \dots, k(\vx, \vx_n))^\T$ for the covariances
between $\vx$ and the observed inputs. Then the posterior mean and variance are

$$
\mu(\vx) = \vk(\vx)^\T \mK^{-1} \vy,
\qquad
\sigma^2(\vx) = k(\vx, \vx) - \vk(\vx)^\T \mK^{-1} \vk(\vx).
$$ {#eq-gp-pointwise}

These two lines are the working form of the whole chapter. The rest of the
book writes $\mu_n(\vx)$ and $\sigma_n(\vx)$, always with their argument,
when the number $n$ of observations matters. They are not to be confused with
the noise standard deviation $\sigma_n$ of @sec-gp-noise, which has no
argument and whose subscript stands for noise.

::: {.keyidea title="Uncertainty depends on where, not on what"}
The variance in @eq-gp-pointwise contains the inputs and the kernel but not the
observed values $\vy$. With the kernel fixed, how uncertain the model is at
$\vx$ depends only on where we have looked, never on what we found there. The
values enter only through the mean. When the kernel's hyperparameters are
fitted to the data (@sec-kernels), the values reach the variance indirectly,
through the fitted lengthscale and amplitude.
:::

## Reading the posterior {#sec-reading-posterior}

The figure below computes @eq-gp-pointwise on a grid of 160 inputs. The blue
line is the posterior mean; the shaded band covers the mean plus and minus 1.96
posterior standard deviations, which holds 95% of the posterior probability at
each input. The strip underneath plots the standard deviation by itself.

```{figure}
//| figure: gp-posterior
//| label: fig-gp-posterior
//| fig-cap: "Gaussian process regression on four observations with an RBF kernel. Click the plot to add an observation, click a point to remove it, and move the lengthscale to see how far each observation's influence reaches. The strip below shows the posterior standard deviation: zero-width at the data when noise is small, back to the prior value of 1 about two lengthscales away. The points are illustrative."
```

A few experiments make the equations concrete.

**Add a point far from the others.** The band pinches to nearly nothing at the
new input and opens again on either side. How quickly it opens is set by the
lengthscale $\ell$ of the kernel: the RBF kernel
$k(x, x') = \exp\!\left(-(x - x')^2 / 2\ell^2\right)$ has fallen to about
$0.14$ two lengthscales away, so an observation says little about inputs more
than two lengthscales from it.

**Watch the mean between and beyond the data.** Between nearby observations the
mean interpolates smoothly. Far from all observations it returns to zero, the
prior mean, and the band returns to the prior width. The model does not
extrapolate trends; it reverts to what it believed before seeing data.

**Shrink the lengthscale.** The mean starts to wiggle back to zero between
points, and the band balloons in every gap. Lengthen it and the mean becomes a
stiff curve that may miss the points entirely if they disagree. Choosing the
lengthscale is the subject of @sec-kernels.

The case of a single observation shows the structure without any matrices.

::: {.example #ex-one-observation title="One observation"}
Observe $y_1 = f(x_1)$ under an RBF prior with unit amplitude, so
$k(x_1, x_1) = 1$. Then $\mK = [1]$, $\vk(x) = k(x, x_1)$, and
@eq-gp-pointwise becomes

$$
\mu(x) = k(x, x_1)\, y_1,
\qquad
\sigma^2(x) = 1 - k(x, x_1)^2.
$$

The mean is a copy of the kernel, centered at $x_1$ and scaled to pass through
$y_1$. The variance is zero at $x_1$ and rises to 1 as $k(x, x_1)$ falls to
zero. Every feature of the posterior in @fig-gp-posterior is a superposition of
this picture, corrected for how the observations overlap.
:::

### The mean is a sum of bumps {#sec-gp-bumps}

The single-observation case generalizes. Define the weights
$\bm{\alpha} = \mK^{-1}\vy$. Then the mean in @eq-gp-pointwise is

$$
\mu(\vx) = \sum_{i=1}^n \alpha_i\, k(\vx, \vx_i),
$$ {#eq-gp-representer}

a weighted sum of $n$ kernels, one centered on each observation. Turning on the
kernel bumps in the figure draws each term. When two observations are close,
their kernels overlap and the weights must compensate for each other, which is
why a weight can be much larger than the value it helps to fit, or of the
opposite sign.

```{figure}
//| figure: gp-posterior
//| label: fig-gp-bumps
//| fig-cap: "The posterior mean (blue) as a sum of weighted kernels (dashed), one per observation, as in @eq-gp-representer. The two close observations at the left receive weights of opposite sign that nearly cancel outside the gap between them."
points: "0.15:0.2,0.24:0.9,0.6:-0.5,0.85:0.6"
bumps: true
lengthscale: 0.1
```

@eq-gp-representer is also the prediction of kernel ridge regression, a
method with no probabilistic reading, when its regularization strength equals
the noise variance introduced in the next section. The Gaussian process adds
the variance, which ridge regression does not have, and which Bayesian
optimization needs [@kanagawa2018gaussian].

## Noisy observations {#sec-gp-noise}

Real evaluations are rarely exact. A training run with a different random seed
gives a different accuracy; a person walking in an exoskeleton has good and bad
strides. The standard model adds independent Gaussian noise to each
observation:

$$
y_i = f(\vx_i) + \varepsilon_i,
\qquad
\varepsilon_i \sim \N(0, \sigma_n^2).
$$ {#eq-gp-noise-model}

Because the noise is independent of $f$ and across observations, it adds
$\sigma_n^2$ to the variance of each observation and nothing to any covariance.
The joint distribution @eq-gp-joint keeps its shape with $\mK$ replaced by
$\mK + \sigma_n^2 \mI$, and so does the derivation:

$$
\mu(\vx) = \vk(\vx)^\T (\mK + \sigma_n^2 \mI)^{-1} \vy,
\qquad
\sigma^2(\vx) = k(\vx, \vx) - \vk(\vx)^\T (\mK + \sigma_n^2 \mI)^{-1} \vk(\vx).
$$ {#eq-gp-noisy}

Two things change in the picture. Move the noise slider in @fig-gp-posterior
and the mean stops passing through the points: it now trades fit against
smoothness, the way ridge regression does. The band also stops pinching to
zero, because a noisy observation cannot pin down $f$ exactly.

::: {.pitfall title="Latent variance and predictive variance"}
@eq-gp-noisy gives the posterior variance of the latent value $f(\vx)$. A new
noisy observation $y$ at $\vx$ would vary more, by the noise:
$\Var[y \mid \text{data}] = \sigma^2(\vx) + \sigma_n^2$. Which one a method needs
depends on the question. Expected improvement in @sec-ei asks about $f$, so it
uses $\sigma^2(\vx)$; a prediction interval for the next measurement uses the
sum. Libraries differ in which one they return by default.
:::

The noise variance is usually not known. It is a hyperparameter, fitted along
with the lengthscale in @sec-kernels. Too little noise makes the model chase
every fluctuation; too much makes it ignore real structure. The two can also
trade off against each other: a short lengthscale with little noise and a long
lengthscale with much noise can explain the same wiggly data, which is one
reason fitted hyperparameters deserve a skeptical look.

## Computing it {#sec-gp-computation}

The formulas contain a matrix inverse, but a careful implementation never forms
one. The matrix $\mK + \sigma_n^2 \mI$ is symmetric and positive definite, so it
has a Cholesky factorization $\mL\mL^\T$ with $\mL$ lower triangular
(@sec-cholesky). Solving a triangular system costs $O(n^2)$ and is numerically
stable; inverting a nearly singular matrix is neither.

::: {.algorithm #alg-gp-regression title="Gaussian process regression"}
Input: inputs $X$, observations $\vy$, kernel $k$, noise variance $\sigma_n^2$,
test input $\vx$.

1. $\mL \leftarrow \operatorname{cholesky}(\mK + \sigma_n^2 \mI)$, so that $\mL \mL^\T = \mK + \sigma_n^2 \mI$.
2. $\bm{\alpha} \leftarrow \mL^\T \backslash (\mL \backslash \vy)$, two triangular solves.
3. $\mu(\vx) \leftarrow \vk(\vx)^\T \bm{\alpha}$.
4. $\mathbf{v} \leftarrow \mL \backslash \vk(\vx)$.
5. $\sigma^2(\vx) \leftarrow k(\vx, \vx) - \mathbf{v}^\T \mathbf{v}$.

Here $\mA \backslash \mathbf{b}$ denotes the solution $\mathbf{z}$ of
$\mA\mathbf{z} = \mathbf{b}$. This is Algorithm 2.1 of
@rasmussen2006gaussian, which also returns the log marginal likelihood used in
@sec-marginal-likelihood.
:::

Step 4 is the variance formula in disguise: $\mathbf{v}^\T\mathbf{v} =
\vk^\T \mL^{-\T}\mL^{-1}\vk = \vk^\T(\mK + \sigma_n^2\mI)^{-1}\vk$.

The cost splits into a part paid once per data set and a part paid per
prediction. The factorization in step 1 takes $O(n^3)$ time and $O(n^2)$
memory. After that, each mean costs $O(n)$ and each variance $O(n^2)$. A
laptop factorizes a matrix with a few thousand rows in well under a second, and
a Bayesian optimization run rarely has more than a few hundred observations, so
the cubic cost is seldom the bottleneck in this book. Larger data sets need
approximations that summarize the data with a smaller set of inducing points
[@quinonero2005unifying; @titsias2009variational].

::: {.code title="NumPy"}
```python
import numpy as np

def rbf(a, b, ell=0.12):
    return np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell**2)

def gp_posterior(x, y, xs, noise=1e-4, ell=0.12):
    L = np.linalg.cholesky(rbf(x, x, ell) + noise * np.eye(len(x)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    Ks = rbf(x, xs, ell)                 # n x m
    mean = Ks.T @ alpha
    v = np.linalg.solve(L, Ks)           # n x m
    var = 1.0 - np.sum(v**2, axis=0)     # k(x, x) = 1 for this kernel
    return mean, var
```
A production implementation would use a triangular solver
(`scipy.linalg.solve_triangular`) for the solves, which is faster and states
the structure. @sec-minimal-implementation builds on this function.
:::

Even a well-posed kernel matrix can fail to factorize in floating point when
two inputs are nearly identical, because two rows become nearly equal and the
smallest eigenvalue rounds to zero or below. Implementations add a small
"jitter", a constant such as $10^{-6}\sigma_f^2$, to the
diagonal and retry with a larger one if the factorization still fails. With
observation noise the problem rarely arises, since $\sigma_n^2$ already plays
that role.

## Posterior samples {#sec-gp-posterior-samples}

The mean and the band summarize the posterior one input at a time. They do not
show what a single plausible function looks like, because neighboring values
are strongly correlated. To see whole functions, draw samples from the joint
posterior @eq-gp-posterior on a grid of inputs. With $\mSigma_*$ the posterior
covariance and $\mL_*$ its Cholesky factor, each sample is

$$
\vf_* = \vmu_* + \mL_* \vz, \qquad \vz \sim \N(\mathbf{0}, \mI),
$$

the sampling recipe of @sec-gaussian-linear. Turn on the samples in
@fig-gp-posterior. Each violet curve passes through (or near) every
observation, stays inside the band most of the time, and is as smooth as the
kernel allows. Each is a function the model considers possible.

Samples are not only a visualization. Thompson sampling, one of the
acquisition rules in @sec-thompson, draws one posterior sample and evaluates
the objective where that sample is largest. On a grid of $m$ points, sampling
costs $O(m^3)$ for the factorization, which limits grids to a few thousand
points; for larger or continuous domains, samples can be drawn as functions
using random features or pathwise updates [@rahimi2007random;
@wilson2020efficiently].

## Pitfalls {#sec-gp-pitfalls}

Three habits prevent most surprises in practice.

**Standardize the outputs.** The zero prior mean and unit amplitude assume the
function's values are centered near zero with spread near one. A function whose
values sit around 1000 would be pulled toward zero away from the data. Subtract
the mean of the observations and divide by their standard deviation before
fitting, and undo the transformation on the predictions. Libraries such as
BoTorch do this with an outcome transform [@balandat2020botorch].

**Scale the inputs.** A single lengthscale assumes all input directions vary on
comparable scales. Map each input to $[0, 1]$ first, and give each dimension its
own lengthscale when they matter differently (@sec-ard).

**Distrust extrapolation.** Beyond the data, the posterior returns to the prior
by construction. If the objective has a trend that continues past the observed
range, a stationary kernel will not predict it. That is usually acceptable in
optimization over a bounded domain, but it is a reason to make the domain no
larger than it needs to be.

## Exercises {#sec-gp-exercises}

::: {.exercise #exr-two-points}
Two observations $y_1 = f(0)$ and $y_2 = f(\delta)$ are made under a noise-free
RBF prior with unit amplitude and lengthscale $\ell$. Let
$\rho = \exp(-\delta^2 / 2\ell^2)$. Compute the posterior variance at the
midpoint $x = \delta/2$ in terms of $\rho$, and check that it tends to the
single-observation value as $\delta \to 0$.

::: {.solution}
Here $\mK = \begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}$ and, with
$r = \exp(-\delta^2 / 8\ell^2)$ the kernel value between the midpoint and each
observation, $\vk = (r, r)^\T$. The inverse is
$\mK^{-1} = \frac{1}{1 - \rho^2}\begin{bmatrix} 1 & -\rho \\ -\rho & 1 \end{bmatrix}$,
so $\vk^\T \mK^{-1}\vk = \frac{2r^2(1 - \rho)}{1 - \rho^2} = \frac{2r^2}{1 + \rho}$
and

$$
\sigma^2(\delta/2) = 1 - \frac{2r^2}{1 + \rho}.
$$

Since $r^2 = \rho^{1/2}$, as $\delta \to 0$ both $\rho$ and $r$ tend to 1 and
the variance tends to $1 - 2/2 = 0$, the same as observing the midpoint itself.
For small $\delta$ the second observation adds almost nothing: two nearly equal
inputs carry almost the same information as one.
:::
:::

::: {.exercise #exr-noise-floor}
With noise variance $\sigma_n^2$ and $n$ observations all at the same input
$x_0$, show that the posterior variance at $x_0$ is
$\sigma_n^2 / (n + \sigma_n^2)$ for a unit-amplitude kernel. What does this say
about repeating an evaluation instead of trying a new input?

::: {.solution}
All entries of $\mK$ are 1, so $\mK = \mathbf{1}\mathbf{1}^\T$ and
$\vk(x_0) = \mathbf{1}$. Since $\mathbf{1}^\T\mathbf{1} = n$,
$(\mathbf{1}\mathbf{1}^\T + \sigma_n^2\mI)\mathbf{1} = (n + \sigma_n^2)\mathbf{1}$,
and so
$(\mathbf{1}\mathbf{1}^\T + \sigma_n^2\mI)^{-1}\mathbf{1} = \mathbf{1}/(n + \sigma_n^2)$
(@exr-id-sherman reaches the same result with the Sherman-Morrison formula).
So $\vk^\T(\mK + \sigma_n^2\mI)^{-1}\vk = n/(n + \sigma_n^2)$ and the variance is
$1 - n/(n + \sigma_n^2) = \sigma_n^2/(n + \sigma_n^2)$. Repeating an evaluation
shrinks uncertainty at that input like $1/n$, the rate of averaging $n$ noisy
measurements, and teaches little about anywhere else. Bayesian optimization
repeats an input only when the noise is large relative to the differences it
is trying to resolve.
:::
:::

## Further reading {#further-reading .unnumbered}

- @rasmussen2006gaussian, chapter 2, is the standard derivation, in both the
  weight-space and function-space views, with the algorithm used here.
- @garnett2023bayesian, chapters 2 to 4, develops Gaussian processes with
  Bayesian optimization in mind, including the inference choices this chapter
  treats as fixed.
- @gortler2019visual is an interactive visual introduction that complements
  the figures in this chapter.
- @williams1996gaussian introduced Gaussian process regression to machine
  learning. The same predictor had long been used in geostatistics as kriging
  [@krige1951statistical; @matheron1963principles].
- @kanagawa2018gaussian sets out the exact correspondences between Gaussian
  process regression and kernel methods such as kernel ridge regression.
