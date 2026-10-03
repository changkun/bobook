---
status: done
synopsis: "Which kernel, which lengthscales, how much noise: the kernel family and how kernels combine, one lengthscale per input, the marginal likelihood as the data's verdict on a model, how hyperparameters are fitted and how the fit fails, why the lengthscale prior must grow with the number of inputs, and how to check the result."
sources: ["Rasmussen and Williams 2006, ch. 4 and 5", "Garnett 2023, ch. 3 and 4", "Snoek et al. 2012", "Neal 1996"]
---

# Kernels and Hyperparameters {#sec-kernels}

@sec-function-space showed that a kernel is a statement about the unknown
function, and @sec-gp-regression showed what that statement becomes once data
arrive. Both chapters fixed the kernel and its hyperparameters by hand. Each
figure had a lengthscale slider, and moving it changed the predictions and
the uncertainty everywhere. A Bayesian optimizer has nobody to move the
slider. It must choose the kernel's settings from the same handful of
evaluations it is trying to learn the objective from.

This chapter is about that choice. Its center is the marginal likelihood: a
single number that says how probable the observed data are under a given
kernel and hyperparameters, and whose maximum is the usual answer to "which
lengthscale?". Before it come the kernels there are to choose from; after it,
the ways the answer misleads, and why it misleads more as the number of
inputs grows.

## The kernel family {#sec-kernel-family}

A stationary kernel depends on its inputs only through the distance
$r = \lVert \vx - \vx' \rVert$ between them (@sec-fs-stationarity). The ones
used in Bayesian optimization differ in a single respect: how smooth the
functions they draw are.

### The Matérn ladder {#sec-kern-matern}

The Matérn family has a smoothness parameter $\nu > 0$ (@sec-fs-smoothness).
When $\nu$ is a half-integer, the kernel is an exponential times a
polynomial, and three such values cover practice. @tbl-kern-family lists them
with the RBF kernel, which is the limit $\nu \to \infty$.

::: {.table #tbl-kern-family title="The stationary kernels in common use, written for unit amplitude as functions of the distance r. Multiply by the squared amplitude for the general form. A draw is q times differentiable exactly when ν exceeds q."}
| Kernel | $\nu$ | $k(r)$ | Draws are |
|---|---|---|---|
| Matérn 1/2 (exponential) | $1/2$ | $\exp\!\left(-\dfrac{r}{\ell}\right)$ | continuous, nowhere differentiable |
| Matérn 3/2 | $3/2$ | $\left(1 + \dfrac{\sqrt{3}\,r}{\ell}\right)\exp\!\left(-\dfrac{\sqrt{3}\,r}{\ell}\right)$ | once differentiable |
| Matérn 5/2 | $5/2$ | $\left(1 + \dfrac{\sqrt{5}\,r}{\ell} + \dfrac{5r^2}{3\ell^2}\right)\exp\!\left(-\dfrac{\sqrt{5}\,r}{\ell}\right)$ | twice differentiable |
| RBF (squared exponential) | $\infty$ | $\exp\!\left(-\dfrac{r^2}{2\ell^2}\right)$ | infinitely differentiable |
:::

The formulas and the differentiability rule are standard
[@rasmussen2006gaussian, sec. 4.2.1]; differentiability is meant in the
mean-square sense of @sec-fs-smoothness. The parameter $\nu$ returns in
@sec-regret-gpucb-bound, in the bounds on an optimizer's regret (its total
shortfall from the best value available): the theory of Bayesian optimization
is stated kernel by kernel.

::: {.aside title="The general Matérn kernel"}
For any $\nu > 0$ the kernel is

$$
k_\nu(r) = \sigma_f^2\, \frac{2^{1-\nu}}{\Gamma(\nu)}\, z^{\nu} K_\nu(z),
\qquad z = \frac{\sqrt{2\nu}\, r}{\ell},
$$ {#eq-kern-matern}

where $\Gamma$ is the gamma function and $K_\nu$ a modified Bessel function
[@rasmussen2006gaussian, sec. 4.2.1]. Nobody computes with this form: for a
half-integer $\nu$ the Bessel function collapses to the expressions in
@tbl-kern-family.
:::

Two kernels from @sec-function-space complete the set. The **periodic**
kernel @eq-fs-periodic draws functions that repeat exactly. The **linear**
kernel is the kernel of Bayesian linear regression (@ex-fs-line),

$$
k(\vx, \vx') = \sigma_0^2 + \sigma_1^2\, (\vx - \mathbf{c})^\T (\vx' - \mathbf{c}),
$$ {#eq-kern-linear}

whose draws are lines (planes, for several inputs) with a random offset and
slope, pivoting around the point $\mathbf{c}$. It is not stationary: its
variance grows with the distance from $\mathbf{c}$.

### Sums and products {#sec-kern-combine}

Kernels can be combined, and the two basic rules have short proofs.

**The sum of two kernels is a kernel.** If $f_1 \sim \GP(0, k_1)$ and
$f_2 \sim \GP(0, k_2)$ are independent, their sum is a Gaussian process with
kernel $k_1 + k_2$ (@exr-fs-sum). A sum models a function made of independent
parts: a slow trend plus a fast wiggle, or a signal plus correlated noise.

**The product of two kernels is a kernel.** The product $f_1 f_2$ of the same
two processes has covariance $k_1 k_2$ (@exr-kern-product). The product
process is not Gaussian, but its covariance function is positive semidefinite,
which is all a kernel needs [@rasmussen2006gaussian, sec. 4.2.4]. A product is
large only when both factors are, so it models structure that must hold in
both senses at once: a periodic kernel times an RBF kernel says "repeats, and
nearby repetitions resemble each other more than distant ones".

A product is also how one kernel covers several inputs. If $k_1$ acts on the
first input and $k_2$ on the second, $k_1(x_1, x_1')\,k_2(x_2, x_2')$ is a
kernel on pairs. The RBF kernel on several inputs is exactly such a product
of one-dimensional RBF kernels, since the exponential of a sum is a product
of exponentials. A sum across inputs, $k_1(x_1, x_1') + k_2(x_2, x_2')$, says
instead that the function is a sum of one function per input, with no
interaction between them, which is a much stronger assumption.

The figure puts the family side by side. Each panel draws two functions, and
every panel uses the same random numbers, so what differs between panels is
the kernel alone.

```{figure}
//| figure: kern-family
//| label: fig-kern-family
//| fig-cap: "Six kernels, the same random numbers. Each panel shows two functions drawn from a Gaussian process prior, and under it the kernel $k(0.7, x)$. *Building blocks*: the Matérn ladder from 1/2 to the RBF, the periodic kernel with period 0.25, and the linear kernel. *Sums and products*: kernels built from those. Press *Draw again* for new random numbers; the lengthscale slider acts on the RBF and Matérn kernels. Weights and periods of the combinations are illustrative."
```

**Read the first four panels in order.** The Matérn draws have the same
large-scale shape, because the random numbers are shared, and lose their
roughness one step at a time. The Matérn 5/2 draws are close to the RBF
draws at first glance. The difference is in the tails of the kernel curve
under each panel and in the fine detail of the draws.

**Switch to *Sums and products*.** A long RBF plus a short one gives a slow
drift with a fast wiggle on top. An RBF plus a periodic kernel gives a
repeating pattern riding on a trend. Their product gives a pattern that
repeats but slowly changes shape. A linear kernel times itself draws
parabolas, and times a periodic kernel it draws oscillations whose size grows
away from the center.

A real example shows what composition buys. The monthly concentration of
carbon dioxide measured at Mauna Loa, Hawaii, 545 observations from 1958 to
2003, is a standard demonstration [@rasmussen2006gaussian, sec. 5.4.3]. The
kernel used there is a sum of
four parts, each built for one feature of the record: an RBF kernel for the
long-term rise, a periodic kernel multiplied by an RBF for a seasonal cycle
that may slowly change, a third term for irregularities over a few years,
and a noise term. It has 11 hyperparameters, all fitted by the method of
@sec-marginal-likelihood. The fitted values read like a report on the
data: the trend has a lengthscale of 67 years, and the seasonal pattern
decays over 90 years, so it is close to exactly periodic. The search for such
structure can itself be automated by adding and multiplying base kernels
greedily, scoring each candidate with its marginal likelihood
[@duvenaud2013structure].

Bayesian optimization rarely goes that far. With a few dozen evaluations
there is too little data to choose among structures, so the usual model is
one Matérn 5/2 or RBF kernel with a separate lengthscale for each input. That
last ingredient deserves its own section.

## One lengthscale per input {#sec-ard}

An objective with several inputs seldom depends on all of them equally. A
neural network's validation error may swing with one setting of its training
procedure, the learning rate, and barely move with another. A single
lengthscale cannot say that. It claims the function varies at the same rate in
every direction.

The remedy is to measure distance with a ruler per input. Replace the squared
distance $\lVert \vx - \vx' \rVert^2/\ell^2$ by a sum of per-input terms:

$$
k(\vx, \vx') = \sigma_f^2 \exp\!\left(-\frac12 \sum_{j=1}^{d} \frac{(x_j - x_j')^2}{\ell_j^2}\right),
$$ {#eq-kern-ard}

where $d$ is the number of inputs, and likewise inside any Matérn kernel, with
$r^2 = \sum_j (x_j - x_j')^2/\ell_j^2$ and $\ell = 1$ in @tbl-kern-family.
Each $\ell_j$ says how far one must move along input $j$ before the function
changes appreciably. A short $\ell_j$ means the function is sensitive to
input $j$. As $\ell_j$ grows, the term for input $j$ vanishes from the sum,
the kernel stops noticing differences in that input, and every function the
prior draws becomes constant along it. The input has been switched off
without being removed.

Because the lengthscales are fitted to data, the model can discover which
inputs matter. This is called **automatic relevance determination** (ARD), a
term due to @neal1996bayesian, and the inverse lengthscale $1/\ell_j$ is read
as the relevance of input $j$ [@rasmussen2006gaussian, sec. 5.1].

```{figure}
//| figure: kern-ard
//| label: fig-kern-ard
//| fig-cap: "One function of two inputs, drawn from a Gaussian process prior with the RBF kernel of @eq-kern-ard. The map shows its value over the unit square; the two panels cut through the center along each input. Lengthen $\ell_2$ and the map turns into vertical stripes while the cut along $x_2$ flattens: the function no longer depends on that input. The random numbers stay fixed while you move the sliders. The draw uses the RBF kernel only, whose product form makes a two-dimensional draw cheap."
```

**Set both lengthscales to 0.2.** The map is a landscape of round hills and
hollows about 0.2 across, and the two cuts wiggle equally.

**Lengthen $\ell_2$ to 5.** The hills stretch into stripes that run along
$x_2$. The cut along $x_2$ is nearly a horizontal line. The function is
still random, but it is a function of $x_1$ alone.

**Press *Swap the lengthscales*.** The stripes turn by a quarter turn. Which
input matters is a property of the kernel, not of the random numbers.

Real tuning problems rely on this. @snoek2012practical proposed the ARD
Matérn 5/2 kernel for tuning machine learning models. With the Gaussian
process model of their paper they tuned nine hyperparameters of a
convolutional network on the CIFAR-10 image benchmark, and reached a test
error of 14.98%, more than three percentage points better than the settings
an expert had found. In the
seven-hyperparameter problem of @sec-cs-classifier-many, two of the seven
account for most of the variation in the error (@sec-cs-classifier-importance),
which is the situation ARD is meant for. @fig-kern-race in
@sec-lengthscale-priors shows fitted lengthscales separating six inputs that
matter from fourteen that do not.

ARD has a price: one more hyperparameter per input, each to be learned from
the same few evaluations. With real numbers as observations that is usually
affordable. When each observation is a single comparison between two
options, as in @sec-part-preferences, it may not be: @sec-cs-photo fixes its
six lengthscales instead of learning them, and @sec-hd-learning-ls reports
that no study has checked whether they can be learned at such budgets. In
the other direction, a model that needs a much shorter lengthscale along one
attribute than along the others is how a smooth utility approximates a person
who attends to that attribute first (@sec-jdm-ecological-implications).

## The marginal likelihood {#sec-marginal-likelihood}

Write $\bm{\theta}$ for all the hyperparameters together: the lengthscales,
the amplitude $\sigma_f$, and the noise standard deviation $\sigma_n$. The
question is how to choose $\bm{\theta}$ from the data.

The obvious criterion fails. If we score a setting by how closely the
posterior mean passes through the observations, the winner is always the
shortest lengthscale and the smallest noise: that model bends through every
point exactly. It also predicts nothing between the points
(@sec-reading-posterior). A model that can fit anything has learned nothing.

@sec-evidence met this problem for the straight line and answered it with
the model evidence: score a model by the probability it assigned to the
observed data *before* seeing them. The same idea works here, and for a
Gaussian process the probability has a closed form.

::: {.derivation title="The log marginal likelihood"}
1. The model of @sec-gp-noise is $y_i = f(\vx_i) + \varepsilon_i$. By
   @def-fs-gp, the function values at the $n$ observed inputs are Gaussian,
   $\vf \sim \N(\mathbf{0}, \mK)$, with $[\mK]_{ij} = k(\vx_i, \vx_j)$.
2. The noise is $\bm{\varepsilon} \sim \N(\mathbf{0}, \sigma_n^2\mI)$,
   independent of $\vf$.
3. The sum of independent Gaussian vectors is Gaussian, and their covariances
   add (@sec-gauss-sum, applied through @eq-gauss-affine). So
   $\vy = \vf + \bm{\varepsilon} \sim \N(\mathbf{0}, \mK_y)$ with
   $\mK_y = \mK + \sigma_n^2\mI$.
4. The probability density of the data is this Gaussian's density
   @eq-gauss-density evaluated at the observed $\vy$:
   $p(\vy \given X, \bm{\theta}) = (2\pi)^{-n/2}\,\lvert\mK_y\rvert^{-1/2} \exp\!\left(-\tfrac12 \vy^\T\mK_y^{-1}\vy\right)$.
5. Take the logarithm, and call the result
   $\mathcal{L}(\bm{\theta}) = \log p(\vy \given X, \bm{\theta})$.
:::

$$
\mathcal{L}(\bm{\theta}) =
\underbrace{-\tfrac12 \vy^\T \mK_y^{-1} \vy}_{\text{data fit}}
\underbrace{-\tfrac12 \log \lvert \mK_y \rvert}_{\text{complexity}}
-\tfrac{n}{2}\log 2\pi.
$$ {#eq-kern-lml}

This is the **log marginal likelihood**
[@rasmussen2006gaussian, sec. 5.4.1]. "Marginal" because the unknown function
values have been summed out: step 3 is the integral
$\int p(\vy \given \vf)\, p(\vf \given X, \bm{\theta})\, \dd\vf$ done in one
line. It is the formula @eq-bayes-log-evidence of @sec-evidence with a kernel
matrix in place of $\boldsymbol{\Phi}\mSigma_p\boldsymbol{\Phi}^\T$, and it
depends on $\bm{\theta}$ only through $\mK_y$.

### Two terms that pull apart {#sec-kern-two-terms}

A single observation shows the mechanism.

::: {.example #ex-kern-one-point title="One observation"}
With one observation $y$, $\mK_y$ is the number
$s^2 = \sigma_f^2 + \sigma_n^2$, the total variance the model expects, and

$$
\log p(y) = -\frac{y^2}{2s^2} - \frac12\log s^2 - \frac12\log 2\pi.
$$

The first term rewards a large $s^2$: an observation far from zero is
unsurprising to a model that expects large values. The second term punishes
it: a model that spreads its probability over a wide range gives each
particular value less. Setting the derivative with respect to $s^2$ to zero
gives $s^2 = y^2$. The best model expects values of the size it saw, no
smaller and no larger. The data say nothing about how $s^2$ splits into
signal and noise, and no single observation could.
:::

With $n$ observations the two terms keep these roles. The **data fit**
$-\tfrac12\vy^\T\mK_y^{-1}\vy$ is the only term that contains the observed
values. It is a squared distance of $\vy$ from zero, measured in the units
the model expects (the Mahalanobis distance of @sec-gauss-shape), and it is
least negative when the data vary the way the kernel says they should.

The **complexity** term $-\tfrac12\log\lvert\mK_y\rvert$ contains no
observed values at all. The determinant is the volume of the region where
the model expects data to fall (@sec-determinants), so the term charges the
model for every data set it could have explained, whether or not it
occurred. A short lengthscale makes the entries of $\vy$ nearly independent,
$\mK_y$ nearly diagonal, and the volume as large as the variances allow. A
long lengthscale ties the observations together, which flattens the region
and shrinks the volume. For a fixed noise level, lengthening $\ell$
therefore relaxes the complexity charge while it tightens the data fit,
because a stiffer function can match fewer data sets
[@rasmussen2006gaussian, sec. 5.4.1]. The maximum of their sum is the
lengthscale at which the model is as simple as the data permit. This is the
automatic Occam's razor of @sec-bayes-occam, and it needs no held-out data.

### The surface {#sec-kern-surface}

The figure computes @eq-kern-lml over a grid of lengthscales and noise
levels for seven observations, with the amplitude fixed at $\sigma_f = 1$.

```{figure}
//| figure: kern-evidence
//| label: fig-kern-evidence
//| fig-cap: "The log marginal likelihood of an RBF kernel with unit amplitude, for the seven observations on the right. Left: its value for every lengthscale and noise level; each step in shade is a band of log probability (within 0.25 of the maximum, then 0.5, 1, 2, 4, 8, 16), and crosses mark local maxima. Right: the regression fit the selected pair implies. Bottom: a cut through the surface along the lengthscale at the selected noise, split into the data-fit and complexity terms of @eq-kern-lml. Drag the ring on the surface, or use the sliders. Click the fit panel to add an observation and a point to remove it. The data are illustrative."
```

**Start where the figure opens.** The ring sits on a local maximum: a
lengthscale of 0.44 with noise 0.74. The fit on the right is a gentle
downward slope, and the model attributes everything else to noise.

**Press *Go to the best pair*.** The ring jumps to a lengthscale of 0.075 and
noise 0.23. Now the fit wiggles through the points and the band balloons
between them. The log marginal likelihood rises from $-9.65$ to $-9.18$, so
this explanation is $e^{0.46} \approx 1.6$ times more probable than the
other. Seven points cannot decide between "a wiggly function measured
precisely" and "a smooth function measured badly".

**Drag the ring straight down from the best pair.** The value barely
changes: at noise 0.01 it is $-9.23$. Once the function passes through the
points, the model cannot tell small noise from none, and the surface is a
flat ridge.

**Read the cut at the bottom while moving the lengthscale.** At the opening
noise level, the data-fit curve falls from $-2.1$ to $-5.5$ as the
lengthscale grows from 0.05 to 2, and the complexity curve rises from $-1.4$
to $0.7$. Their sum is nearly flat between 0.1 and 0.44. Lower the noise to
0.23 and the data-fit curve becomes a cliff: at a lengthscale of 0.5 it is
$-24$, because a stiff curve with little noise cannot be near these points.

**Press *Data: three points*, then *Data: ten points*.** With three points
the maximum sits in the corner of the surface, and a wide region of short
lengthscales and small noise is within a few hundredths of it. Three points
say almost nothing about the lengthscale. With ten the surface has one compact peak, at
a lengthscale of 0.13 and noise 0.27.

The same picture, for another seven observations, appears in the standard
reference, with the lesson stated plainly: every local maximum is a
particular interpretation of the data, and with so few points the model
cannot confidently reject either [@rasmussen2006gaussian, sec. 5.4.1].

::: {.keyidea title="The marginal likelihood scores a model by what it predicted"}
The marginal likelihood is the probability the model gave the observed data
before seeing them. Fitting the data well is not enough; the model must also
not have predicted many data sets that did not occur.
:::

## Fitting hyperparameters {#sec-fitting-hyperparameters}

The standard practice is to choose the hyperparameters that maximize
@eq-kern-lml. It is called **type II maximum likelihood**, or empirical
Bayes: maximum likelihood applied to the settings of the prior instead of to
the function itself (@sec-evidence).

### Climbing the surface {#sec-kern-gradient}

A grid like the one in @fig-kern-evidence works for two hyperparameters and
not for ten, so the maximum is found by following the gradient.

::: {.derivation title="The gradient of the log marginal likelihood"}
Let $\theta_j$ be one hyperparameter and write $\partial\mK_y$ for
$\partial\mK_y/\partial\theta_j$, the matrix of entrywise derivatives.

1. Two matrix derivative rules are needed
   [@petersen2012matrix; @rasmussen2006gaussian, app. A.3]:
   $\partial(\mK_y^{-1}) = -\mK_y^{-1}(\partial\mK_y)\mK_y^{-1}$ and
   $\partial \log\lvert\mK_y\rvert = \tr\!\left(\mK_y^{-1}\,\partial\mK_y\right)$,
   where $\tr$, the trace, is the sum of a matrix's diagonal entries.
2. Apply the first rule to the data-fit term of @eq-kern-lml:
   $\partial\!\left(-\tfrac12\vy^\T\mK_y^{-1}\vy\right) = \tfrac12\,\vy^\T\mK_y^{-1}(\partial\mK_y)\mK_y^{-1}\vy = \tfrac12\,\bm{\alpha}^\T(\partial\mK_y)\,\bm{\alpha}$,
   with $\bm{\alpha} = \mK_y^{-1}\vy$, the weights of @sec-gp-bumps.
3. Apply the second rule to the complexity term:
   $\partial\!\left(-\tfrac12\log\lvert\mK_y\rvert\right) = -\tfrac12\tr\!\left(\mK_y^{-1}\,\partial\mK_y\right)$.
4. A number equals its own trace, and $\tr(\mA\mathbf{B}) = \tr(\mathbf{B}\mA)$
   for any two matrices whose products exist. Moving the last factor to the
   front, $\bm{\alpha}^\T(\partial\mK_y)\bm{\alpha}$ equals
   $\tr\!\left(\bm{\alpha}\bm{\alpha}^\T\,\partial\mK_y\right)$.
5. Add steps 2 and 3.
:::

$$
\frac{\partial\mathcal{L}}{\partial\theta_j}
= \frac12 \tr\!\left( \left(\bm{\alpha}\bm{\alpha}^\T - \mK_y^{-1}\right) \frac{\partial\mK_y}{\partial\theta_j} \right).
$$ {#eq-kern-gradient}

The Cholesky factorization of @alg-gp-regression, which costs $O(n^3)$, gives
$\bm{\alpha}$, the log determinant (@sec-linalg-logdet), and $\mK_y^{-1}$.
After that each hyperparameter's derivative costs $O(n^2)$
[@rasmussen2006gaussian, sec. 5.4.1]. In practice an automatic
differentiation library produces the gradient from the code that computes
@eq-kern-lml.

::: {.algorithm #alg-kern-fit title="Fitting hyperparameters by type II maximum likelihood"}
Input: inputs $X$ scaled to the unit cube, standardized observations $\vy$, a
kernel family.

1. Parametrize every positive hyperparameter by its logarithm, so the
   optimizer works on an unconstrained scale.
2. Choose a starting point: unit amplitude, a small noise level, and
   lengthscales suited to the number of inputs (@sec-lengthscale-priors).
3. Maximize @eq-kern-lml with a quasi-Newton method such as L-BFGS, an
   optimizer that follows the gradient @eq-kern-gradient and estimates the
   curvature from how the gradient changes between steps.
4. Repeat steps 2 and 3 from several other starting points, and keep the
   result with the highest marginal likelihood.
:::

::: {.code title="NumPy and SciPy"}
```python
import numpy as np
from scipy.optimize import minimize

# x: observed inputs, y: standardized observations (one-dimensional arrays)
def neg_log_marginal(log_theta, x, y):
    ell, sf, sn = np.exp(log_theta)
    K = sf**2 * np.exp(-0.5 * (x[:, None] - x[None, :]) ** 2 / ell**2)
    L = np.linalg.cholesky(K + (sn**2 + 1e-8) * np.eye(len(x)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return 0.5 * y @ alpha + np.log(np.diag(L)).sum() + 0.5 * len(x) * np.log(2 * np.pi)

starts = [np.log([0.1, 1.0, 0.1]), np.log([0.5, 1.0, 0.5]), np.log([0.03, 1.0, 0.3])]
fits = [minimize(neg_log_marginal, s, args=(x, y), method="L-BFGS-B") for s in starts]
ell, sf, sn = np.exp(min(fits, key=lambda f: f.fun).x)
```
The sum of the logarithms of the Cholesky diagonal is half the log
determinant. Without a gradient function, SciPy differentiates numerically,
which is adequate for three hyperparameters.
:::

### How the fit fails {#sec-kern-failures}

@fig-kern-evidence already contains the three ways this procedure misleads.

**Several maxima.** The restarts in step 4 exist because the surface can
have more than one peak, and a gradient method finds the one nearest its
starting point. With little data the peaks can be nearly level, and which
one the optimizer reports is then an accident of where it started.

**Flat directions.** The ridge toward zero noise, and the plateau with three
observations, are regions where the data do not determine a hyperparameter.
An optimizer still returns a single number there. This is why a Bayesian
optimization loop starts with a handful of evaluations chosen without the
model (@sec-initial-design): the first fits are otherwise taken on a
plateau.

**Too many hyperparameters.** Maximizing over $\bm{\theta}$ is itself a form
of fitting, and it can overfit. With one lengthscale per input and few
evaluations, a setting that switches off most inputs can explain the data by
chance, and the model then ignores inputs that matter.
@sec-lengthscale-priors shows this happening in fifty dimensions.

A fourth consequence is specific to optimization. The hyperparameters are
refitted as evaluations arrive, so the model the acquisition function
consults keeps changing, and the convergence guarantees for a fixed kernel no
longer apply directly (@sec-regret-bounds-not-say).

### Priors, and averaging instead of choosing {#sec-kern-map}

Two refinements address these failures. The first is to place a prior
$p(\bm{\theta})$ on the hyperparameters and maximize the posterior,

$$
\log p(\bm{\theta} \given \vy, X) = \mathcal{L}(\bm{\theta}) + \log p(\bm{\theta}) + \text{const},
$$

which is **maximum a posteriori** (MAP) estimation. The prior adds a gentle
slope to plateaus and ridges, so the optimizer has somewhere to go where the
data are silent. BoTorch's default models carry such priors on their
lengthscales [@botorch2026gpytorch].

A prior on a lengthscale needs care, because what counts as vague depends on
the scale it is stated on. A density that is flat over $\ell$ between 0.01
and 10 puts 90% of its probability above $\ell = 1$. A density that is flat
over $\log \ell$ on the same range puts a third of its probability on each
factor of ten. Neither is neutral (@sec-prob-continuous-examples). Lengthscale priors are
therefore usually given on the logarithmic scale, as log-normal
distributions, or as gamma distributions with a stated mode.

The second refinement is not to choose at all. A fully Bayesian treatment
averages predictions over the posterior of $\bm{\theta}$, usually by drawing
samples of $\bm{\theta}$ with a Markov chain method, which produces draws from
a distribution known only up to a constant (@sec-mcmc), and averaging the
acquisition function over them. @snoek2012practical did this and found,
on their tuning problems, that it beat a single fitted value. A plausible
reason is that with few evaluations, uncertainty about the hyperparameters
is a large part of the uncertainty about the objective (inference). The cost
is one Cholesky factorization per sample at every step of the loop.

## Priors on lengthscales and dimension {#sec-lengthscale-priors}

For years, Bayesian optimization was said to stop working beyond ten or
twenty inputs. Between 2024 and 2026 the failure was traced to a mundane
cause: the default prior on the lengthscale [@hvarfner2024vanilla;
@xu2025standard; @papenmeier2025understanding].

The arithmetic was done in @sec-linalg-distances and @sec-fs-lengthscale. Two
random points in the unit cube $[0, 1]^d$ are about $\sqrt{d/6}$ apart, so
distances grow with $\sqrt{d}$ while a prior whose typical lengthscale is, say,
0.5 stays where it is. In high dimension every pair of evaluations is then many
lengthscales apart, every kernel value is close to zero, and the model sees a
collection of unrelated points. @sec-hd-seeing derives this and plots it.

The damage is done before any data can object. @eq-kern-gradient multiplies
by $\partial\mK_y/\partial\ell$, and for the RBF kernel the derivative of an
entry is $k \cdot r^2/\ell^3$. When the kernel values are near zero, so is
the gradient, and an optimizer started at a short lengthscale has no slope
to climb (@exr-kern-vanish). @xu2025standard and
@papenmeier2025understanding identify these vanishing gradients, caused by
the initial lengthscale, as a main reason for failure in high dimension.

The fix is to let the lengthscale's prior grow with the dimension.
@hvarfner2024vanilla scaled it with $\sqrt{d}$ and found that standard Bayesian
optimization then performed best on three of the five real tasks they tried,
ahead of methods built specially for high dimension. BoTorch has used this
prior for most of its models since version 0.12.0 [@botorch2026changelog]: a
log-normal distribution with location $\sqrt{2} + \tfrac12\log d$ and scale
$\sqrt{3}$, with the fit started at the prior's mode [@botorch2026gpytorch].
That mode is about $0.2\sqrt{d}$: 0.65 for ten inputs and 0.92 for twenty.

### ARD at work in twenty dimensions {#sec-kern-race}

The figure replays Bayesian optimization on a test problem with a known
structure. The objective is the six-dimensional Hartmann function, a
standard test function with several local maxima, hidden among inputs that
do nothing: in twenty dimensions, six inputs matter and fourteen are
decoys. The model is a Gaussian process with the ARD kernel @eq-kern-ard,
refitted by MAP every five evaluations, under two priors: a fixed
Gamma(2.4, 2.7) prior with mode 0.52 in every dimension, which is the default
of BoTorch's preference model [@botorch2026pairwisegp], and the
dimension-scaled prior above.

```{figure}
//| figure: dim-bo-race
//| label: fig-kern-race
//| fig-cap: "Bayesian optimization of the six-dimensional Hartmann function embedded in 6 to 50 dimensions, replayed from recorded runs with 80 evaluations each. Top: regret (the gap to the true maximum) for random search and for Bayesian optimization with a fixed and a dimension-scaled lengthscale prior, median of six seeds with the range shaded. Middle: every point one run evaluated, one vertical axis per input, the six inputs that matter shaded. Bottom: the lengthscale that run's model learned for each input, on a log scale; short bars are inputs the model treats as relevant. Switch the run shown, the dimension, and how the acquisition function was maximized. These are this book's own runs, not published results."
dims: 20
show: scaled
```

**Read the bars at the bottom first.** In the run shown, the model under
the scaled prior ends with lengthscales between 0.25 and 0.56 for the six
inputs that matter, and between 2.6 and 30 for the fourteen decoys, most of
them near 20. The model was not given which inputs were which. The data are
explained as well without the decoys, and the complexity term of @eq-kern-lml
is larger for the simpler explanation. That is automatic relevance
determination doing what its name says.

**Switch to the fixed prior.** The six relevant lengthscales are similar,
0.39 to 0.70, but the decoys stop between 1.5 and 2.1. The prior is the
reason: Gamma(2.4, 2.7) puts less than 5% of its probability above a
lengthscale of 2, so under it a lengthscale long enough to switch an input
off is improbable, where the scaled prior at twenty dimensions puts about 90%
there (inference).

**Now compare the regret curves, and change the acquisition search.** Here
the picture is less tidy, and worth reading honestly. In twenty dimensions,
with the acquisition function maximized over uniform random candidates
only, the two priors end level: a median regret of 0.29 for the fixed prior
and 0.31 for the scaled one, against 1.26 for random search. Adding
candidates near the best points found so far lowers both, to 0.16 and 0.08.
In fifty dimensions the same change takes the fixed prior from 0.89 to 0.18
and the scaled prior from 1.21 to 0.18. In ten dimensions the fixed prior
finishes ahead, 0.10 against 0.18. In these runs, how the
acquisition function is maximized matters more than which prior is used
(inference).

**Go to fifty dimensions and look at the bars again.** With the scaled prior,
two of the six relevant inputs now have lengthscales near 15: the model has
switched off inputs that matter. Eighty evaluations are too few to determine
fifty lengthscales, and this is the overfitting of @sec-kern-failures.

These runs are six seeds on one test function with a small budget, and they
do not settle anything. They do agree with the state of the research. The
diagnosis, that fixed lengthscale priors fail as the dimension grows, is
shared by the papers above. Why the remedy works is disputed:
@papenmeier2025understanding argue that good results in very high dimension
come from local search behavior more than from a well-fitted model.
@sec-hd-diagnosis and @sec-hd-fixes follow the argument. As of September
2026, the defaults of the software for learning from comparisons have not
changed: every preference package examined in @sec-sw-priors still uses a
prior that ignores dimension.

## Checking the model {#sec-model-checking}

The marginal likelihood ranks models against each other. It does not say
whether the best of them is any good. A kernel that cannot express the
objective still has a maximum somewhere. Before an optimizer acts on a
model's uncertainty, it is worth asking whether that uncertainty is honest,
and the data already collected can answer.

The check is **leave-one-out prediction**. Remove observation $i$, predict
it from the other $n - 1$, and compare the prediction with what was
observed. Repeating this for every $i$ sounds like $n$ separate fits, but
for a Gaussian process with fixed hyperparameters all $n$ predictions come
from one matrix.

::: {.derivation title="Leave-one-out predictions from one inverse"}
1. Under the model, $\vy \sim \N(\mathbf{0}, \mK_y)$. Predicting $y_i$ from
   the rest is conditioning this Gaussian on all coordinates but one.
2. @sec-gauss-conditioning-general showed that the conditional distribution
   can be read from the precision matrix $\bm{\Lambda} = \mK_y^{-1}$: the
   conditional variance of coordinate $i$ is $1/\Lambda_{ii}$, and its
   conditional mean is $-\Lambda_{ii}^{-1}\sum_{j \ne i}\Lambda_{ij}\,y_j$.
3. The sum over $j \ne i$ is the full sum minus its own term:
   $\sum_{j \ne i}\Lambda_{ij}y_j = [\bm{\Lambda}\vy]_i - \Lambda_{ii}\,y_i$.
4. Substitute: the mean is $y_i - [\bm{\Lambda}\vy]_i/\Lambda_{ii}$.
:::

$$
\mu_{-i} = y_i - \frac{[\mK_y^{-1}\vy]_i}{[\mK_y^{-1}]_{ii}},
\qquad
\sigma_{-i}^2 = \frac{1}{[\mK_y^{-1}]_{ii}}.
$$ {#eq-kern-loo}

These are the leave-one-out mean and variance for a noisy observation at
$\vx_i$ [@rasmussen2006gaussian, sec. 5.4.2]. Despite appearances $\mu_{-i}$
does not depend on $y_i$ (@exr-kern-loo). Three uses follow.

**Calibration.** The standardized residuals
$z_i = (y_i - \mu_{-i})/\sigma_{-i}$ should look like draws from a standard
normal distribution: about 95% of them within $\pm 1.96$. Many large
residuals mean the model is overconfident, typically a lengthscale too long
or a noise level too small. Residuals all near zero mean it is
underconfident, and the optimizer will explore more than it needs to.

**A second opinion on the hyperparameters.** The sum of the leave-one-out
log densities, $\sum_i \log \N(y_i;\, \mu_{-i}, \sigma_{-i}^2)$, can replace
the marginal likelihood as the quantity to maximize. The marginal likelihood
is the probability of the data assuming the model is right; the
leave-one-out score estimates predictive performance whether or not it is,
which has been argued to make it more robust when the kernel is wrong
[@rasmussen2006gaussian, sec. 5.4.2].

**Finding the point that does not fit.** One residual far outside the rest
marks an observation the model cannot reconcile with its neighbors: a failed
run, a mistyped value, or a region where the function changes character.

Turn on *Leave-one-out check* in @fig-kern-evidence to see @eq-kern-loo. At
the best pair, each observation's interval contains it or nearly does. Drag
to a long lengthscale with small noise, and the intervals shrink to dashes
that miss their points, the picture of overconfidence. Drag to a very short
lengthscale and every interval spans the whole prior: the model predicts
nothing about a point from its neighbors, which is honest and useless.

In optimization there is one more check, and it is the simplest. Plot the
posterior mean against the observations along one input at a time, through
the best point so far. A model whose band is narrow where no evaluation has
been made, or wide between evaluations that agree, has hyperparameters worth
a second look before its next suggestion is trusted.

The kernels of this chapter were chosen and fitted as recipes for covariance
matrices; @sec-kernel-analysis looks at them as objects in their own right,
the view in which the guarantees of @sec-regret are stated.

## Exercises {#sec-kern-exercises}

::: {.exercise #exr-kern-product}
Let $f_1 \sim \GP(0, k_1)$ and $f_2 \sim \GP(0, k_2)$ be independent, and
define $g(\vx) = f_1(\vx)\,f_2(\vx)$. (a) Show that
$\Cov[g(\vx), g(\vx')] = k_1(\vx, \vx')\,k_2(\vx, \vx')$. (b) Why does this
prove that the product of two kernels is positive semidefinite, even though
$g$ is not a Gaussian process?

::: {.solution}
(a) $g$ has mean zero, because by independence
$\E[f_1(\vx)f_2(\vx)] = \E[f_1(\vx)]\,\E[f_2(\vx)] = 0$. So the covariance is
$\E[g(\vx)g(\vx')] = \E[f_1(\vx)f_1(\vx')\,f_2(\vx)f_2(\vx')]$, and
independence splits the expectation into
$\E[f_1(\vx)f_1(\vx')]\;\E[f_2(\vx)f_2(\vx')] = k_1(\vx, \vx')\,k_2(\vx, \vx')$.
(b) The argument of @sec-fs-valid-kernels used nothing about Gaussians: for
any random function with finite variances, $\sum_{ij} a_i a_j \Cov[g(\vx_i), g(\vx_j)]$
is the variance of $\sum_i a_i g(\vx_i)$ and so cannot be negative. The
covariance function of any random function is a valid kernel, and a
Gaussian process with that kernel then exists by @def-fs-gp.
:::
:::

::: {.exercise #exr-kern-identical}
Two noise-free observations under a unit-amplitude kernel have correlation
$\rho = k(x_1, x_2)$. (a) Write the log marginal likelihood as a function of
$\rho$. (b) Suppose the two observed values are identical, $y_1 = y_2 = y$.
Show that the log marginal likelihood grows without bound as $\rho \to 1$.
What does the fitted model believe, and what does this say about fitting
without a noise term or a prior?

::: {.solution}
(a) $\mK = \begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}$ has
determinant $1 - \rho^2$ and inverse
$\frac{1}{1 - \rho^2}\begin{bmatrix} 1 & -\rho \\ -\rho & 1 \end{bmatrix}$, so
by @eq-kern-lml

$$
\mathcal{L} = -\frac{y_1^2 - 2\rho\,y_1 y_2 + y_2^2}{2(1 - \rho^2)} - \frac12\log(1 - \rho^2) - \log 2\pi.
$$

(b) With $y_1 = y_2 = y$ the numerator is $2y^2(1 - \rho)$, and since
$1 - \rho^2 = (1 - \rho)(1 + \rho)$ the data fit is $-y^2/(1 + \rho)$, which
stays between $-y^2$ and $-y^2/2$. The complexity term
$-\tfrac12\log(1 - \rho^2)$ tends to $+\infty$ as $\rho \to 1$. So the
marginal likelihood is maximized by an infinite lengthscale: the model
concludes that the function is constant, with certainty, from two equal
values. Two equal values are weak evidence for that. A noise term, which
keeps $\mK_y$ away from singular, or a prior on the lengthscale, keeps the
optimizer from running off to this degenerate answer.
:::
:::

::: {.exercise #exr-kern-loo}
(a) Show that the leave-one-out mean $\mu_{-i}$ of @eq-kern-loo does not
depend on $y_i$. (b) For two observations with
$\mK_y = \begin{bmatrix} s^2 & c \\ c & s^2 \end{bmatrix}$, compute
$\mu_{-1}$ and $\sigma_{-1}^2$ from @eq-kern-loo and check them against the
conditioning formula @eq-gauss-cond-2d, with $\sigma_1 = \sigma_2 = s$,
$\rho = c/s^2$, and zero means.

::: {.solution}
(a) Write $\bm{\Lambda} = \mK_y^{-1}$. Then
$[\bm{\Lambda}\vy]_i = \Lambda_{ii}y_i + \sum_{j \ne i}\Lambda_{ij}y_j$, so
$\mu_{-i} = y_i - y_i - \Lambda_{ii}^{-1}\sum_{j \ne i}\Lambda_{ij}y_j$, and
the two $y_i$ cancel. (b) The inverse is
$\bm{\Lambda} = \frac{1}{s^4 - c^2}\begin{bmatrix} s^2 & -c \\ -c & s^2 \end{bmatrix}$.
So $\sigma_{-1}^2 = 1/\Lambda_{11} = (s^4 - c^2)/s^2 = s^2 - c^2/s^2$, and
$\mu_{-1} = -\Lambda_{12}\,y_2/\Lambda_{11} = (c/s^2)\,y_2$.
@eq-gauss-cond-2d, with the roles of the two coordinates exchanged, gives
mean $\rho\,y_2 = (c/s^2)\,y_2$ and variance
$s^2(1 - \rho^2) = s^2 - c^2/s^2$, the same.
:::
:::

::: {.exercise #exr-kern-vanish}
For the RBF kernel $k = \exp(-r^2/2\ell^2)$, the derivative with respect to
the lengthscale is $\partial k/\partial\ell = k\,r^2/\ell^3$. (a) Write it as
a function of the scaled distance $\rho = r/\ell$, and find the $\rho$ at
which it is largest. (b) Evaluate it, relative to that largest value, at the
typical distance between two random points in $[0, 1]^{50}$ when
$\ell = 0.5$, and when $\ell = 0.2\sqrt{50}$.

::: {.solution}
(a) $\partial k/\partial\ell = \rho^2 e^{-\rho^2/2}/\ell$. Differentiating
$\rho^2 e^{-\rho^2/2}$ gives $(2\rho - \rho^3)e^{-\rho^2/2}$, which vanishes
at $\rho = \sqrt{2}$, where the factor equals $2/e \approx 0.74$. A pair of
points informs the lengthscale most when it is about 1.4 lengthscales apart.
(b) The typical distance is $\sqrt{50/6} \approx 2.89$ (@sec-linalg-distances).
With $\ell = 0.5$, $\rho = 5.77$ and
$\rho^2 e^{-\rho^2/2} = 33.3 \times e^{-16.7} \approx 2 \times 10^{-6}$,
about three millionths of the largest value. With
$\ell = 0.2\sqrt{50} \approx 1.41$, $\rho = 2.04$ and the factor is
$4.17 \times e^{-2.08} \approx 0.52$, about 70% of the largest value. At the
fixed lengthscale almost no pair of points carries a usable gradient; at the
scaled one, typical pairs do.
:::
:::

## Further reading {#further-reading .unnumbered}

- @rasmussen2006gaussian, chapter 4, catalogs kernels and the rules for
  combining them; chapter 5 derives the marginal likelihood, its gradient,
  and the leave-one-out formulas, and works the Mauna Loa example in full.
- @garnett2023bayesian, chapters 3 and 4, treats kernel choice, model
  assessment, and averaging over models with Bayesian optimization in mind.
- @snoek2012practical is the paper that made the ARD Matérn 5/2 kernel and
  the fully Bayesian treatment of hyperparameters standard in Bayesian
  optimization.
- @duvenaud2013structure searches over sums and products of kernels
  automatically, with the marginal likelihood as the score.
- @hvarfner2024vanilla is the paper that traced the failure of standard
  Bayesian optimization in high dimension to the lengthscale prior.
- @neal1996bayesian introduced automatic relevance determination, for
  neural networks.
- @stein1999interpolation gives the theory behind preferring Matérn kernels
  to the RBF kernel.
