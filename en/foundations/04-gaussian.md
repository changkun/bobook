---
status: done
synopsis: "The one distribution the whole book runs on: its shape in one and many dimensions, why a linear map of a Gaussian is Gaussian and how that gives a sampler, and the conditioning formula, derived through the Schur complement, that Gaussian process regression applies unchanged."
sources: ["textbooks", "Bishop 2006, sec. 2.3", "Rasmussen and Williams 2006, app. A"]
---

# The Gaussian Distribution {#sec-gaussian}

The previous two chapters built a language and a toolkit. @sec-probability
gave the rules for reasoning about uncertain quantities: distributions, the
sum rule that marginalizes, and the product rule that conditions.
@sec-linear-algebra gave the matrices those rules turn into when there are many
quantities at once: covariance matrices, their Cholesky factors, and block
matrices. This chapter puts the two together in a single distribution, the
Gaussian, which carries the rest of the book.

One distribution deserves a chapter because of what a Bayesian optimizer does
with its beliefs. It keeps a belief about an unknown function, updates the
belief after each evaluation, and asks it where to look next. Each step is an
operation on a probability distribution, and for most distributions these
operations are integrals with no closed form. For the Gaussian, each has an
exact answer in a few lines of linear algebra: a linear map of a Gaussian is
Gaussian, ignoring some of its coordinates leaves a Gaussian, and conditioning
on some of its coordinates leaves a Gaussian. A Gaussian process
(@sec-gp-definition) is a Gaussian over function values, so these three
closure properties are also what make Gaussian process regression and Bayesian
optimization computable.

The chapter starts with one variable, where the formulas can be read at a
glance, moves to many variables, where a covariance matrix gives the
distribution its shape, and then derives the three operations in turn. The
last of them, conditioning, is the chapter's main result; @sec-gp-regression
applies it without change. A later section separates two ways of combining
Gaussians that are easy to confuse, and prepares the Bayesian updating of
@sec-bayesian-inference.

## One dimension {#sec-gaussian-1d}

Suppose a training run with a new learning rate has not happened yet. From
experience with similar runs you expect a validation accuracy of about 0.82,
rarely off by more than 0.06 in either direction. That belief has a center and
a spread, treats deviations up and down alike, and makes large deviations
rarer than small ones. The Gaussian distribution, also called the normal
distribution, is the standard way to write such a belief with two numbers.

A random variable $X$ is Gaussian with mean $\mu$ and variance
$\sigma^2 > 0$, written $X \sim \N(\mu, \sigma^2)$, when its density is

$$
\N(x;\, \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}
\exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right).
$$ {#eq-gauss-density-1d}

The semicolon separates the point $x$ where the density is evaluated from the
parameters $\mu$ and $\sigma^2$. As @sec-continuous explained, a density is not
a probability; probabilities are areas under it.

Read the formula from the inside out. The exponent $-(x - \mu)^2 / 2\sigma^2$
is a downward parabola with its peak at $x = \mu$. The exponential turns the
parabola into a bell: it equals 1 at the peak and falls quickly as the
parabola goes negative. The factor in front scales the bell so that its total
area is one. Taking logarithms undoes the exponential:

$$
\log \N(x;\, \mu, \sigma^2) = -\frac{(x - \mu)^2}{2\sigma^2} - \tfrac12 \log(2\pi\sigma^2).
$$

A Gaussian is the exponential of a quadratic. The converse is the fact that
does most of the work in this chapter: any density whose logarithm is a
quadratic function of $x$, opening downward, is Gaussian, and its parameters
can be read off the coefficients. If

$$
\log p(x) = -\tfrac12 \alpha x^2 + \beta x + \text{const}, \quad \alpha > 0,
\qquad\text{then}\qquad
p(x) = \N\!\left(x;\, \beta/\alpha,\; 1/\alpha\right).
$$ {#eq-gauss-recipe}

To see it, complete the square: $-\tfrac12\alpha x^2 + \beta x =
-\tfrac12\alpha(x - \beta/\alpha)^2 + \beta^2/2\alpha$, and the last term is a
constant that the normalization absorbs. The coefficient $\alpha$, the
reciprocal of the variance, is called the **precision**. To show that some
distribution is Gaussian, we will repeatedly show that its log density is
quadratic and then apply this recipe.

The two parameters mean what their names say. The mean is the expected value,
$\E[X] = \mu$, and the variance is the expected squared deviation,
$\Var[X] = \E[(X - \mu)^2] = \sigma^2$ (@sec-expectation). The standard
deviation $\sigma$, the square root of the variance, is in the units of $x$,
so it is the number to think with. The accuracy belief above is about
$\N(0.82, 0.03^2)$: a standard deviation of 0.03, with deviations of twice that
size rare.

### The standard normal {#sec-gauss-standard}

Every Gaussian is a shifted and stretched copy of one reference distribution,
the standard normal $Z \sim \N(0, 1)$. If $X \sim \N(\mu, \sigma^2)$, then
$Z = (X - \mu)/\sigma$ is standard normal, and conversely
$X = \mu + \sigma Z$. The value $z = (x - \mu)/\sigma$, the number of standard
deviations by which $x$ lies above the mean, is called its z-score. The
standard normal has its own symbols, used throughout the book:

$$
\phi(z) = \frac{1}{\sqrt{2\pi}}\, e^{-z^2/2},
\qquad
\Phi(z) = \int_{-\infty}^{z} \phi(t)\,\dd t.
$$ {#eq-gauss-phi}

The function $\phi$ is the density and $\Phi$ the cumulative distribution
function: $\Phi(z)$ is the probability that $Z \le z$. There is no formula for
$\Phi$ in elementary functions. Libraries compute it through the error
function, $\Phi(z) = \tfrac12\left(1 + \operatorname{erf}(z/\sqrt{2})\right)$,
to full floating-point precision.

Standardizing turns every probability question about $X$ into one about
$\Phi$:

$$
\Prob(a \le X \le b) = \Phi\!\left(\frac{b - \mu}{\sigma}\right) - \Phi\!\left(\frac{a - \mu}{\sigma}\right).
$$ {#eq-gauss-interval}

A few values are worth remembering. The interval $\mu \pm \sigma$ holds 68.3%
of the probability, $\mu \pm 1.96\sigma$ holds 95%, $\mu \pm 2\sigma$ holds
95.4%, and $\mu \pm 3\sigma$ holds 99.7%. The tails fall off fast: a value more
than five standard deviations from the mean, in either direction, has
probability about $5.7 \times 10^{-7}$.

::: {.example #ex-gauss-beat-incumbent title="Will the new run beat the best so far?"}
Keep the belief $f \sim \N(0.82, 0.03^2)$ about the accuracy $f$ of the new
run, and suppose the best run so far reached 0.85. By @eq-gauss-interval, the
probability that the new run does better is

$$
\Prob(f > 0.85) = 1 - \Phi\!\left(\frac{0.85 - 0.82}{0.03}\right) = 1 - \Phi(1) \approx 0.159.
$$

A belief centered 0.03 below the best result still gives the new run about
one chance in six. Computed at every candidate input from a Gaussian process
posterior, this number is the acquisition function called probability of
improvement (@sec-pi). Expected improvement (@sec-ei) is assembled from $\phi$
and $\Phi$ in the same way.
:::

### Why this distribution {#sec-gauss-why}

A reader may wonder why this particular bell curve, and not some other, is the
default. There are three reasons, of different kinds.

The first comes from a theorem. The central limit theorem says that a sum of
many independent random quantities with finite variance, none of which
dominates, is approximately Gaussian once standardized, whatever the
distribution of the individual terms [@blitzstein2019introduction]. Measurement
noise is often the sum of many small disturbances, such as thermal
fluctuations, timing jitter, and rounding, and is then close to Gaussian. A
classic demonstration: the sum of twelve independent uniform numbers on
$[0, 1]$, minus 6, has mean 0 and variance 1, and its histogram is already
hard to tell from $\phi$.

The second is a principle. Among all distributions on the real line with a
given mean and variance, the Gaussian has the largest entropy, a measure of
how spread out a distribution is that @sec-entropy makes precise
[@cover2006elements, ch. 12]. If all we are willing to commit to is a center
and a spread, the Gaussian is the choice that assumes nothing more.

The third is convenience, and it is why this book uses Gaussians even where
the first two reasons do not apply: every operation in the rest of this
chapter has a closed form. Convenience is a reason to choose a model, not
evidence that the world is Gaussian. Real quantities can be skewed, bounded,
or heavy-tailed, with extreme values far more common than the
$5.7 \times 10^{-7}$ above suggests. A comparison between two options, the
observation at the heart of @sec-part-preferences, is not Gaussian at all;
@sec-approx-inference shows how to approximate the resulting posterior by a
Gaussian so that the machinery of this chapter still applies.

## Many dimensions {#sec-gaussian-nd}

A Bayesian optimizer never holds a belief about a single number. It holds
beliefs about the objective at many inputs at once, and those beliefs are
linked: if the accuracy at a learning rate of 0.010 turns out high, the
accuracy at 0.011 is probably high too. A list of separate one-dimensional
Gaussians cannot express that link. We need a joint distribution over a vector
of values that records how each pair of values moves together.

Start with two independent coordinates. If $x_1 \sim \N(0, \sigma_1^2)$ and
$x_2 \sim \N(0, \sigma_2^2)$ are independent, their joint density is the product
of the two densities (@sec-independence), and multiplying exponentials adds
the exponents:

$$
p(x_1, x_2) \propto \exp\!\left(-\frac{x_1^2}{2\sigma_1^2} - \frac{x_2^2}{2\sigma_2^2}\right).
$$

The density is constant wherever the exponent is, on the curves
$x_1^2/\sigma_1^2 + x_2^2/\sigma_2^2 = r^2$. These are ellipses with axes along
the coordinate directions, and circles when $\sigma_1 = \sigma_2$. To link the
two coordinates, we allow the quadratic in the exponent a cross term
$x_1 x_2$. The ellipses then tilt, so that a large $x_1$ makes a large $x_2$
more likely, or less likely, depending on the direction of the tilt. All of
this is recorded in one matrix.

The matrix is the covariance matrix $\mSigma$ of @sec-prob-covariance. For a
random vector $\vx = (x_1, \dots, x_d)^\T$ with mean vector $\vmu$, its entries
are $\Sigma_{ij} = \Cov[x_i, x_j] = \E[(x_i - \mu_i)(x_j - \mu_j)]$, with the
variances on the diagonal. It is symmetric and positive semidefinite
(@sec-positive-definite), and dividing an entry by the two standard deviations
gives the correlation
$\rho_{ij} = \Sigma_{ij} / \sqrt{\Sigma_{ii}\Sigma_{jj}}$.

::: {.definition #def-gauss-mvn title="Multivariate Gaussian"}
A random vector $\vx \in \R^d$ has a Gaussian distribution with mean
$\vmu \in \R^d$ and positive definite covariance matrix
$\mSigma \in \R^{d \times d}$, written $\vx \sim \N(\vmu, \mSigma)$, when its
density is

$$
\N(\vx;\, \vmu, \mSigma) = \frac{1}{(2\pi)^{d/2}\, \lvert\mSigma\rvert^{1/2}}
\exp\!\left(-\tfrac12 (\vx - \vmu)^\T \mSigma^{-1} (\vx - \vmu)\right),
$$ {#eq-gauss-density}

where $\lvert\mSigma\rvert$ is the determinant of $\mSigma$.
:::

Every piece has a one-dimensional counterpart. With $d = 1$ and
$\mSigma = [\sigma^2]$ the formula is @eq-gauss-density-1d. The exponent is
again a quadratic, now a quadratic form in the vector $\vx - \vmu$, with the
inverse covariance $\mSigma^{-1}$ in the role of $1/\sigma^2$. The inverse
covariance is called the **precision matrix**, and it is the natural object
in several derivations below. The determinant in the normalizer measures the
volume over which the distribution spreads (@sec-determinants) and plays the
role of $\sigma$: a more spread-out distribution has a lower peak, so that the
total probability stays one.

The recipe @eq-gauss-recipe carries over unchanged. If a density satisfies

$$
\log p(\vx) = -\tfrac12 \vx^\T \bm{\Lambda} \vx + \mathbf{h}^\T \vx + \text{const}
$$

for a positive definite matrix $\bm{\Lambda}$ and a vector $\mathbf{h}$, then
$p$ is Gaussian with precision $\bm{\Lambda}$, that is,

$$
\vx \sim \N\!\left(\bm{\Lambda}^{-1}\mathbf{h},\; \bm{\Lambda}^{-1}\right).
$$ {#eq-gauss-recipe-nd}

Completing the square works as before, and expanding the right side checks
it:

$$
-\tfrac12\vx^\T\bm{\Lambda}\vx + \mathbf{h}^\T\vx
= -\tfrac12(\vx - \bm{\Lambda}^{-1}\mathbf{h})^\T\bm{\Lambda}(\vx - \bm{\Lambda}^{-1}\mathbf{h})
+ \tfrac12\mathbf{h}^\T\bm{\Lambda}^{-1}\mathbf{h}.
$$

The last term does not depend on $\vx$.

### Shape {#sec-gauss-shape}

The quantity in the exponent has a name. The **Mahalanobis distance** of
$\vx$ from $\vmu$ is

$$
r(\vx) = \sqrt{(\vx - \vmu)^\T \mSigma^{-1} (\vx - \vmu)},
$$ {#eq-gauss-mahalanobis}

the multivariate z-score. In one dimension it is $\lvert x - \mu\rvert / \sigma$,
the number of standard deviations. In general it measures distance in units of
the distribution's own spread, direction by direction: a point can be far from
the mean in ordinary distance and still close in Mahalanobis distance, if it
lies along a direction in which the distribution is wide. The density depends
on $\vx$ only through $r$, so its contours are the sets where $r$ is constant:
ellipses in two dimensions, ellipsoids in more.

Where do the ellipses point? Write the covariance in its eigendecomposition
$\mSigma = \mathbf{U}\bm{\Lambda}_{\mathrm{e}}\mathbf{U}^\T$, with orthonormal
eigenvectors $\mathbf{u}_i$ in the columns of $\mathbf{U}$ and eigenvalues
$\lambda_i$ on the diagonal of $\bm{\Lambda}_{\mathrm{e}}$ (@sec-eigen). In the
rotated coordinates $\mathbf{y} = \mathbf{U}^\T(\vx - \vmu)$ the quadratic form
becomes $\sum_i y_i^2/\lambda_i$, with no cross terms. The axes of every
contour therefore point along the eigenvectors, and the ellipse at Mahalanobis
distance $r$ has semi-axes of length $r\sqrt{\lambda_i}$. The eigenvalues are
the variances along the principal directions, and their product is
$\lvert\mSigma\rvert$. A multivariate Gaussian is an axis-aligned bell in some
rotated coordinate system, and the eigenvectors say which one.

For two coordinates with standard deviations $\sigma_1, \sigma_2$ and
correlation $\rho$, the covariance matrix and its determinant are

$$
\mSigma = \begin{bmatrix} \sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2 \end{bmatrix},
\qquad
\lvert\mSigma\rvert = \sigma_1^2\sigma_2^2(1 - \rho^2).
$$ {#eq-gauss-2d-cov}

The figure below draws this case with mean zero.

```{figure}
//| figure: gauss-2d
//| label: fig-gauss-2d
//| fig-cap: "A two-dimensional Gaussian with mean zero and the covariance @eq-gauss-2d-cov. The shaded ellipses are the contours at Mahalanobis distance 1, 2, and 3; the dashed lines are the eigenvector axes; the strips above and to the right show the marginal densities of $x_1$ and $x_2$. The sliders set the two standard deviations and the correlation. The Samples and Condition views belong to @sec-gaussian-linear and @sec-gaussian-conditioning."
```

**Set $\rho$ to zero and the two standard deviations equal.** The ellipses
become circles. The distribution looks the same in every direction, and the
eigenvectors could point anywhere.

**Move $\rho$ toward 0.95.** The ellipses narrow into a needle along the
diagonal. With both standard deviations at 1 the eigenvalues are $1 + \rho$
and $1 - \rho$, so the first approaches 2 while the second and the
determinant approach zero: the distribution concentrates near a line, and
once $x_1$ is known, $x_2$ is nearly determined. At $\rho = \pm 1$ the
covariance would be singular, the density of @eq-gauss-density would not exist,
and a Cholesky factorization would fail. This is the floating-point trouble
that a small diagonal jitter repairs (@sec-gp-computation).

**Make $\rho$ negative.** The ellipses tilt the other way: a large $x_1$ now
goes with a small $x_2$.

**Watch the two strips while you move $\rho$.** They do not change.
@sec-gaussian-marginal explains why.

How much probability does each ellipse hold? Less than the one-dimensional
numbers suggest. In two dimensions the probability inside the ellipse at
Mahalanobis distance $r$ is $1 - e^{-r^2/2}$: 39% inside $r = 1$, 86% inside
$r = 2$, and 99% inside $r = 3$. Enclosing 95% takes $r \approx 2.45$, not
1.96. The gap widens with the dimension.

::: {.aside title="Where the probability is in high dimensions"}
For a standard Gaussian in $d$ dimensions, the squared Mahalanobis distance
$r^2 = \sum_i z_i^2$ is a sum of $d$ independent squared standard normals. Each
term has mean 1 and variance 2, so $r^2$ has mean $d$ and standard deviation
$\sqrt{2d}$, and $r$ itself stays within about one unit of $\sqrt{d}$. In 100
dimensions, nearly all the probability lies in a thin shell about 10 units
from the mean, and the neighborhood of the peak, where the density is highest,
holds almost none of it. The one-dimensional intuition that typical values lie
near the mean does not survive in high dimensions, and this geometry is part of
why Bayesian optimization becomes hard in many dimensions
(@sec-high-dimensions).
:::

One more property is special to Gaussians. When $\mSigma$ is diagonal, the
quadratic form has no cross terms and the density factorizes into a product
of one-dimensional densities, so the coordinates are independent. For a
Gaussian vector, uncorrelated therefore means independent. For other
distributions it does not (@sec-independence), and @exr-gauss-not-joint shows
two variables that are each Gaussian and uncorrelated, yet dependent, because
they are not jointly Gaussian.

## Linear maps and sampling {#sec-gaussian-linear}

Two practical questions lead to the same result. First, if $f(\vx_1)$ and
$f(\vx_2)$ are jointly Gaussian, what is the distribution of their difference,
or of their average? @sec-part-preferences needs the difference whenever a person compares
two options. Second, a random number generator produces independent standard
normal numbers. How do we turn them into a draw from $\N(\vmu, \mSigma)$ with an
arbitrary covariance? Every picture of functions drawn from a model needs such
draws, and so does Thompson sampling (@sec-thompson), a rule that draws one
plausible objective from the model and evaluates where that draw is highest.

Both answers follow from one closure property. Let $\vx \sim \N(\vmu, \mSigma)$
in $d$ dimensions, let $\mA$ be an $m \times d$ matrix, and let $\mathbf{c}$ be
a vector in $\R^m$. Then

$$
\vy = \mA\vx + \mathbf{c} \;\sim\; \N\!\left(\mA\vmu + \mathbf{c},\; \mA\mSigma\mA^\T\right).
$$ {#eq-gauss-affine}

A linear map of a Gaussian is Gaussian. The mean is mapped like a point, and
the covariance is sandwiched between the matrix and its transpose.

One caveat. The density of @def-gauss-mvn needs a positive definite
covariance, and $\mA\mSigma\mA^\T$ is positive definite only when no row of
$\mA$ is a combination of the others. When it is not, as when $\mA$ has more
rows than columns, $\vy$ is still Gaussian in the sense used in step 6 below,
but it is confined to a flat of lower dimension and has no density on $\R^m$.
The formulas for its mean and covariance hold in both cases.

::: {.derivation title="A linear map of a Gaussian"}
1. By linearity of expectation (@sec-expectation),
   $\E[\vy] = \mA\,\E[\vx] + \mathbf{c} = \mA\vmu + \mathbf{c}$.
2. Subtracting the mean, $\vy - \E[\vy] = \mA(\vx - \vmu)$.
3. By the definition of the covariance matrix,
   $\Cov[\vy] = \E\big[(\vy - \E[\vy])(\vy - \E[\vy])^\T\big]
   = \E\big[\mA(\vx - \vmu)(\vx - \vmu)^\T\mA^\T\big]$.
4. $\mA$ is constant, so it moves outside the expectation:
   $\Cov[\vy] = \mA\,\E\big[(\vx - \vmu)(\vx - \vmu)^\T\big]\mA^\T = \mA\mSigma\mA^\T$.
5. That $\vy$ is Gaussian, and not merely some distribution with this mean
   and covariance, needs one more argument. When $\mA$ is square and
   invertible, substituting $\vx = \mA^{-1}(\vy - \mathbf{c})$ into
   @eq-gauss-density leaves an exponent that is a quadratic function of
   $\vy$, so $\vy$ is Gaussian by @eq-gauss-recipe-nd.
6. For a general $\mA$, such as the single row that forms a difference, use
   the equivalent definition that a vector is Gaussian exactly when every
   linear combination of its coordinates is a one-dimensional Gaussian
   [@blitzstein2019introduction]. A linear combination of the coordinates of
   $\vy$ is a linear combination of the coordinates of $\vx$, so it is
   Gaussian, and so is $\vy$.
:::

### Sampling with the Cholesky factor {#sec-gauss-sampling}

Sampling runs the map in the useful direction. Let $\vz \sim \N(\mathbf{0}, \mI)$
be a vector of $d$ independent standard normal numbers, which every numerical
library provides (the classic construction from uniform random numbers is the
transform of @box1958note). Choose any matrix $\mL$ with
$\mL\mL^\T = \mSigma$ and set

$$
\vx = \vmu + \mL\vz.
$$ {#eq-gauss-sample}

By @eq-gauss-affine, $\vx$ is Gaussian with mean $\vmu$ and covariance
$\mL\mI\mL^\T = \mSigma$. The Cholesky factor of @sec-cholesky, lower triangular
with a positive diagonal, is the usual choice of $\mL$: it exists for every
positive definite $\mSigma$, it costs $O(d^3)$ operations to compute once, and
after that each draw costs one triangular matrix-vector product
[@rasmussen2006gaussian, app. A.2]. Any other square root would do, such as
$\mathbf{U}\bm{\Lambda}_{\mathrm{e}}^{1/2}$ from the eigendecomposition. It
would map a given $\vz$ to a different point, but the distribution of the
points would be the same.

In two dimensions the Cholesky factor of @eq-gauss-2d-cov can be written down,
and multiplying it by $\vz$ shows how each coordinate is built:

$$
\mL = \begin{bmatrix} \sigma_1 & 0 \\ \rho\sigma_2 & \sigma_2\sqrt{1 - \rho^2} \end{bmatrix},
\qquad
\begin{aligned}
x_1 &= \sigma_1 z_1, \\
x_2 &= \rho\sigma_2 z_1 + \sigma_2\sqrt{1 - \rho^2}\, z_2.
\end{aligned}
$$ {#eq-gauss-chol-2d}

Multiplying out confirms $\mL\mL^\T = \mSigma$. The formula also says what
correlation is, mechanically. The coordinate $x_2$ is built partly from the
same random number $z_1$ that drives $x_1$ and partly from fresh randomness
$z_2$; a fraction $\rho^2$ of its variance comes from the shared part. At
$\rho = 0$ the two coordinates share nothing, and at $\rho = \pm 1$ they share
everything.

```{figure}
//| figure: gauss-2d
//| label: fig-gauss-samples
//| fig-cap: "Sampling with the Cholesky factor, @eq-gauss-sample. Gray points are draws of a standard normal $\vz$; blue points are the same draws after the map $\mL\vz$, which has the covariance set by the sliders. Orange arrows follow six draws from $\vz$ to $\mL\vz$. The dashed ellipse is the contour at Mahalanobis distance 2. New samples draws a fresh set."
view: samples
```

**Follow the arrows.** Every gray point is moved by the same matrix. Because
$\mL$ is lower triangular, $x_1 = \sigma_1 z_1$ depends on $z_1$ alone; with
$\sigma_1 = 1$ the map leaves the first coordinate unchanged and every arrow
is vertical. With a positive $\rho$, the map adds $\rho\sigma_2 z_1$ to the
second coordinate, which pushes points on the left down and points on the
right up, and it shrinks $z_2$ by the factor $\sqrt{1 - \rho^2}$. The round
cloud becomes a tilted ellipse.

**Compare the sample correlation with $\rho$.** The readout computes the
correlation of the blue points. It differs from $\rho$ by a few
hundredths, and a fresh set of samples gives a different error: sampling noise,
with a standard deviation of roughly $(1 - \rho^2)/\sqrt{n}$ for $n$ points.

**Set $\rho$ to zero with unequal standard deviations.** $\mL$ is then
diagonal and only stretches the cloud along the axes.

For a Gaussian process, $\vx$ holds the function values on a grid of inputs,
one coordinate per grid point, and the cubic cost of the factorization is what limits posterior
samples to grids of a few thousand points (@sec-gp-posterior-samples). Grids
run out quickly as the input dimension grows: 20 points per axis is 20 values
on a line, 400 on a square, and 8,000 in a three-dimensional cube, whose
covariance matrix has 64 million entries. Beyond two or three input
dimensions, samples are drawn at a few thousand scattered candidate points
instead of a grid.

The difference of two function values is the other question this section
began with.

::: {.example #ex-gauss-difference title="The difference of two function values"}
Let $A = f(\vx_1)$ and $B = f(\vx_2)$ be jointly Gaussian with means
$\mu_A, \mu_B$, variances $v_A, v_B$, and covariance $c$. The difference
$D = A - B$ is the map with the single row $(1, -1)$, so by @eq-gauss-affine
it is Gaussian with mean $\mu_A - \mu_B$ and variance

$$
\begin{bmatrix} 1 & -1 \end{bmatrix}
\begin{bmatrix} v_A & c \\ c & v_B \end{bmatrix}
\begin{bmatrix} 1 \\ -1 \end{bmatrix}
= v_A + v_B - 2c.
$$

The covariance enters with a minus sign. When the two values are strongly
positively correlated, as for two nearby inputs under a smooth Gaussian
process, their difference is much less uncertain than either value alone. A
model can be confident that one of two similar options is better while being
unsure how good either one is. @sec-thurstone builds a model of human
comparisons on this difference, and @sec-eubo uses it to score pairs of
queries.
:::

## Marginalizing {#sec-gaussian-marginal}

A Gaussian process describes infinitely many function values, but a computer
holds finitely many. For that to make sense, the belief about a few values
must not depend on which other values we chose to keep track of. In the terms
of @sec-joint-marginal-conditional, we need the marginal distribution of a
sub-vector, which the sum rule obtains by integrating out everything else. For
most joint densities that integral is the hard part of a calculation. For a
Gaussian it costs nothing.

Split the vector into two blocks: $\mathbf{a}$, the coordinates we keep, and
$\mathbf{b}$, the coordinates we drop. Partition the mean and the covariance to
match:

$$
\begin{bmatrix} \mathbf{a} \\ \mathbf{b} \end{bmatrix}
\sim \N\!\left(
\begin{bmatrix} \vmu_a \\ \vmu_b \end{bmatrix},\;
\begin{bmatrix} \mSigma_{aa} & \mSigma_{ab} \\ \mSigma_{ab}^\T & \mSigma_{bb} \end{bmatrix}
\right).
$$ {#eq-gauss-partition}

The diagonal blocks $\mSigma_{aa}$ and $\mSigma_{bb}$ hold the covariances
within each block, and $\mSigma_{ab}$ holds the covariances between a
coordinate of $\mathbf{a}$ and a coordinate of $\mathbf{b}$. The marginal
distribution of $\mathbf{a}$ is

$$
\mathbf{a} \sim \N(\vmu_a, \mSigma_{aa}).
$$ {#eq-gauss-marginal}

Marginalizing a Gaussian is reading off a sub-block. The proof is one line
from the previous section: keeping $\mathbf{a}$ and dropping $\mathbf{b}$ is
the linear map with matrix $[\mI \;\; \mathbf{0}]$, an identity block beside a
block of zeros, and @eq-gauss-affine gives mean
$[\mI \;\; \mathbf{0}]\vmu = \vmu_a$ and covariance
$[\mI \;\; \mathbf{0}]\mSigma[\mI \;\; \mathbf{0}]^\T = \mSigma_{aa}$. The
integral the sum rule calls for has been done once and for all; it can also
be carried out directly by completing the square [@bishop2006pattern, sec. 2.3.2].

Two consequences matter later. First, marginals are consistent: the
distribution of $\mathbf{a}$ does not depend on how many other coordinates
$\mathbf{b}$ contains, or which. A Gaussian process relies on this to be well
defined, since its definition specifies only the joint distributions of
finite sets of function values (@sec-gp-definition). Second, the marginal
discards the cross-covariances $\mSigma_{ab}$, and with them everything the two
blocks say about each other. In @fig-gauss-2d the strips above and to the
right of the plot are the two marginals, and moving the correlation slider
rotates and squeezes the joint density without changing either strip. Many
joint distributions share the same marginals.

The marginal of $x_2$ answers the question "what do I believe about $x_2$ if I
ignore $x_1$?" The next section answers a different question: "what do I
believe about $x_2$ once I know $x_1$?"

## Conditioning {#sec-gaussian-conditioning}

This is the operation the rest of the book runs on. A Bayesian optimizer has
evaluated the objective at a few inputs and wants its belief about the
objective everywhere else. If the prior belief about the values at all these
inputs is a joint Gaussian, the question becomes: once some coordinates of a
Gaussian vector have been observed, what is the distribution of the others?
Unlike the marginal, the answer must use what was observed.

### A slice through the bell {#sec-gauss-slice}

Look at two dimensions first. By the product rule, the conditional density of
$x_2$ given $x_1 = a$ is

$$
p(x_2 \given x_1 = a) = \frac{p(a, x_2)}{p(a)}.
$$

The numerator is the joint density along the vertical line $x_1 = a$: a slice
through the bell. The denominator does not depend on $x_2$; it only rescales
the slice so that its area is one. So the conditional density has the shape of
the slice. Along the slice, the exponent of the joint density is a quadratic
function of $x_2$, because fixing one variable of a quadratic in two variables
leaves a quadratic in the other. The slice is therefore itself a bell, a
Gaussian by @eq-gauss-recipe.

Working out that quadratic gives the slice's mean and variance. With the
covariance @eq-gauss-2d-cov and means $\mu_1, \mu_2$,

$$
x_2 \given x_1 = a \;\sim\; \N\!\left(\mu_2 + \rho\,\frac{\sigma_2}{\sigma_1}(a - \mu_1),\;\; \sigma_2^2(1 - \rho^2)\right).
$$ {#eq-gauss-cond-2d}

@exr-gauss-cond-2d obtains this from the general formula below. Each part has
a reading. The observation enters through its z-score $(a - \mu_1)/\sigma_1$.
The mean of $x_2$ moves away from $\mu_2$ by $\rho$ times that many of $x_2$'s
own standard deviations: a value of $x_1$ one standard deviation above its
mean predicts $x_2$ to lie $\rho$ standard deviations above its mean. The
variance shrinks by the factor $1 - \rho^2$, the fraction of $x_2$'s variance
that $x_1$ does not explain, and it does not depend on $a$ at all.

```{figure}
//| figure: gauss-2d
//| label: fig-gauss-condition
//| fig-cap: "Conditioning as slicing, @eq-gauss-cond-2d. The orange line marks the observed value $x_1$; drag across the plot to move it, or use the buttons. The strip on the right compares the marginal density of $x_2$ (blue) with its conditional density given the observation (orange), and the orange bar on the slice marks the conditional mean with its 95% interval. The dashed magenta line traces the conditional mean for every observed value."
view: condition
```

**Drag the slice from left to right.** The conditional density slides along
the dashed line but keeps its width. The width depends on the correlation,
never on the value observed.

**Move $\rho$ toward $\pm 0.95$.** The conditional density collapses, and the
readout shows the fraction of variance removed, $\rho^2$, passing 90%. At
$\rho = 0$ nothing is removed and the conditional equals the marginal: an
observation uncorrelated with $x_2$ teaches nothing about it.

**Compare the dashed line with the ellipses.** The line of conditional means
is not the long axis of the ellipses; it is flatter. It passes through the
leftmost and rightmost points of every ellipse, because along a vertical slice
the density is highest where the slice just touches an ellipse.

That flattening has a long history. Francis Galton noticed that the children
of unusually tall parents were, on average, less unusual than their parents,
and called the effect regression toward mediocrity [@galton1886regression];
the statistical term "regression" descends from it. In @eq-gauss-cond-2d with
equal standard deviations, the predicted deviation of $x_2$ is $\rho$ times the
observed deviation of $x_1$, closer to the mean whenever $\lvert\rho\rvert < 1$.
Nothing pulls the children back. The shrinkage is what conditioning a
correlated Gaussian does.

### The general formula {#sec-gauss-conditioning-general}

The same reasoning works in any number of dimensions, with blocks in place of
numbers. Partition the vector into an observed block $\mathbf{a}$ and an
unobserved block $\mathbf{b}$ as in @eq-gauss-partition. Then

$$
\mathbf{b} \given \mathbf{a} \;\sim\; \N\!\left(
\vmu_b + \mSigma_{ab}^\T \mSigma_{aa}^{-1} (\mathbf{a} - \vmu_a),\;\;
\mSigma_{bb} - \mSigma_{ab}^\T \mSigma_{aa}^{-1} \mSigma_{ab}
\right).
$$ {#eq-gauss-conditional}

Compare the two-dimensional case. The matrix $\mSigma_{ab}^\T\mSigma_{aa}^{-1}$
plays the role of $\rho\sigma_2/\sigma_1 = \Sigma_{12}/\Sigma_{11}$, and the
subtracted term $\mSigma_{ab}^\T\mSigma_{aa}^{-1}\mSigma_{ab}$ that of
$\rho^2\sigma_2^2 = \Sigma_{12}^2/\Sigma_{11}$.

The dimensions in this formula are easy to misread, so consider a real-sized
case. A machine learning model has six hyperparameters to tune, and 40
training runs have finished. In a Bayesian optimizer, $\mathbf{a}$ holds the
40 observed validation errors and $\mathbf{b}$ the unknown errors at, say,
1,000 untried configurations, so $\mSigma_{aa}$ is $40 \times 40$ and
$\mSigma_{ab}$ is $40 \times 1000$. The number six appears nowhere: the
Gaussian lives over function values, one coordinate per configuration, and
the dimension of the input space enters only through the covariances that a
kernel assigns to pairs of configurations (@sec-kernels). The cost of
conditioning grows with the number of evaluations, not with the number of
hyperparameters, though in more dimensions more evaluations are needed to pin
the function down (@sec-high-dimensions). This is the computation that
@snoek2012practical used to tune latent Dirichlet allocation, structured
support vector machines, and convolutional networks, reaching or surpassing
the settings chosen by human experts; @sec-cs-classifier works through such a
tuning problem on real data.

The derivation needs one fact from @sec-block-matrices, restated in the
notation of @eq-gauss-partition. The Schur complement of the block
$\mSigma_{aa}$ is

$$
\mathbf{S} = \mSigma_{bb} - \mSigma_{ab}^\T \mSigma_{aa}^{-1} \mSigma_{ab},
$$ {#eq-gauss-schur}

and the precision matrix $\bm{\Lambda} = \mSigma^{-1}$ has the blocks

$$
\begin{bmatrix} \bm{\Lambda}_{aa} & \bm{\Lambda}_{ab} \\ \bm{\Lambda}_{ab}^\T & \bm{\Lambda}_{bb} \end{bmatrix}
=
\begin{bmatrix}
\mSigma_{aa}^{-1} + \mSigma_{aa}^{-1}\mSigma_{ab}\mathbf{S}^{-1}\mSigma_{ab}^\T\mSigma_{aa}^{-1} & -\mSigma_{aa}^{-1}\mSigma_{ab}\mathbf{S}^{-1} \\
-\mathbf{S}^{-1}\mSigma_{ab}^\T\mSigma_{aa}^{-1} & \mathbf{S}^{-1}
\end{bmatrix}.
$$ {#eq-gauss-block-inverse}

Multiplying $\mSigma$ by this matrix gives the identity, block by block, which
is how the formula is checked [@petersen2012matrix]. Only the bottom row is
needed: $\bm{\Lambda}_{bb} = \mathbf{S}^{-1}$ and
$\bm{\Lambda}_{ab}^\T = -\mathbf{S}^{-1}\mSigma_{ab}^\T\mSigma_{aa}^{-1}$.

::: {.derivation title="Conditioning a Gaussian through the Schur complement"}
1. By the product rule, $p(\mathbf{b} \given \mathbf{a}) = p(\mathbf{a}, \mathbf{b}) / p(\mathbf{a})$.
   With $\mathbf{a}$ held fixed, $p(\mathbf{a})$ is a constant, so as a
   function of $\mathbf{b}$ the conditional is proportional to the joint
   density @eq-gauss-density.
2. Write $\mathbf{u} = \mathbf{a} - \vmu_a$ and $\mathbf{v} = \mathbf{b} - \vmu_b$.
   The log of the joint density is $-\tfrac12 Q$ plus a constant, where
   $Q = (\vx - \vmu)^\T\bm{\Lambda}(\vx - \vmu)$ with the blocks of
   @eq-gauss-block-inverse.
3. Expand $Q$ block by block:
   $Q = \mathbf{u}^\T\bm{\Lambda}_{aa}\mathbf{u} + 2\,\mathbf{v}^\T\bm{\Lambda}_{ab}^\T\mathbf{u} + \mathbf{v}^\T\bm{\Lambda}_{bb}\mathbf{v}$.
   The two cross terms $\mathbf{u}^\T\bm{\Lambda}_{ab}\mathbf{v}$ and
   $\mathbf{v}^\T\bm{\Lambda}_{ab}^\T\mathbf{u}$ are equal, because each is a
   number and the transpose of the other.
4. Keep only the terms that involve $\mathbf{v}$; the rest are constant given
   $\mathbf{a}$. Then
   $\log p(\mathbf{b} \given \mathbf{a}) = -\tfrac12\mathbf{v}^\T\bm{\Lambda}_{bb}\mathbf{v} - \mathbf{v}^\T\bm{\Lambda}_{ab}^\T\mathbf{u} + \text{const}$.
5. This is the form of @eq-gauss-recipe-nd in the variable $\mathbf{v}$, with
   precision $\bm{\Lambda}_{bb}$ and $\mathbf{h} = -\bm{\Lambda}_{ab}^\T\mathbf{u}$.
   ($\bm{\Lambda}_{bb}$ is positive definite, as is every diagonal block of a
   positive definite matrix.) So $\mathbf{v}$ given $\mathbf{a}$ is Gaussian
   with covariance $\bm{\Lambda}_{bb}^{-1}$ and mean
   $-\bm{\Lambda}_{bb}^{-1}\bm{\Lambda}_{ab}^\T\mathbf{u}$, and
   $\mathbf{b} = \vmu_b + \mathbf{v}$ has the same covariance and its mean
   shifted by $\vmu_b$.
6. Substitute the bottom row of @eq-gauss-block-inverse. The covariance is
   $\bm{\Lambda}_{bb}^{-1} = \mathbf{S}$, the Schur complement.
7. The mean offset is
   $-\mathbf{S}\,(-\mathbf{S}^{-1}\mSigma_{ab}^\T\mSigma_{aa}^{-1})\,\mathbf{u}
   = \mSigma_{ab}^\T\mSigma_{aa}^{-1}(\mathbf{a} - \vmu_a)$.
8. Together: $\mathbf{b} \given \mathbf{a}$ is Gaussian with mean
   $\vmu_b + \mSigma_{ab}^\T\mSigma_{aa}^{-1}(\mathbf{a} - \vmu_a)$ and covariance
   $\mSigma_{bb} - \mSigma_{ab}^\T\mSigma_{aa}^{-1}\mSigma_{ab}$, which is
   @eq-gauss-conditional.
:::

The derivation follows section 2.3.1 of @bishop2006pattern, and
@sec-id-conditioning collects the result with related identities. Two remarks
follow from it. The conditional covariance is the Schur complement
@eq-gauss-schur itself. And step 5 shows that the conditional precision is
the block $\bm{\Lambda}_{bb}$ of the joint precision: conditioning reads off a
block of the precision matrix, just as marginalizing reads off a block of the
covariance matrix.

::: {.keyidea title="Conditioning a Gaussian"}
Observing part of a Gaussian vector leaves a Gaussian. The mean of the rest
shifts by a linear function of how far the observation fell from its
expectation, and the covariance shrinks by an amount that depends on which
coordinates were observed but not on the values observed.
:::

The second half of the key idea is the reason a Gaussian process's
uncertainty depends on where we evaluated and not on what we found
(@sec-gp-conditioning). A second derivation makes the Schur complement less
mysterious.

::: {.aside title="A second route: subtract what the observation predicts"}
Define the residual $\mathbf{r} = \mathbf{b} - \mSigma_{ab}^\T\mSigma_{aa}^{-1}\mathbf{a}$,
what is left of $\mathbf{b}$ after subtracting the best linear prediction from
$\mathbf{a}$.

1. The pair $(\mathbf{a}, \mathbf{r})$ is a linear map of $(\mathbf{a}, \mathbf{b})$,
   so it is jointly Gaussian by @eq-gauss-affine.
2. The cross-covariance is
   $\Cov[\mathbf{r}, \mathbf{a}] = \Cov[\mathbf{b}, \mathbf{a}] - \mSigma_{ab}^\T\mSigma_{aa}^{-1}\Cov[\mathbf{a}, \mathbf{a}]
   = \mSigma_{ab}^\T - \mSigma_{ab}^\T = \mathbf{0}$.
3. For jointly Gaussian vectors, uncorrelated means independent
   (@sec-gaussian-nd), so observing $\mathbf{a}$ says nothing about $\mathbf{r}$.
4. Given $\mathbf{a}$, then, $\mathbf{b} = \mathbf{r} + \mSigma_{ab}^\T\mSigma_{aa}^{-1}\mathbf{a}$
   is a known vector plus a Gaussian whose distribution has not changed. That
   distribution has mean $\vmu_b - \mSigma_{ab}^\T\mSigma_{aa}^{-1}\vmu_a$ and,
   expanding $\Cov[\mathbf{r}]$, covariance
   $\mSigma_{bb} - \mSigma_{ab}^\T\mSigma_{aa}^{-1}\mSigma_{ab} = \mathbf{S}$.
5. Adding the known vector to the mean gives @eq-gauss-conditional.

The conditional covariance is the covariance of what $\mathbf{a}$ cannot
predict about $\mathbf{b}$.
:::

A small numerical case shows the whole of Gaussian process regression in
miniature.

::: {.example #ex-gauss-two-inputs title="A Gaussian process in miniature"}
Two inputs $\vx$ and $\vx'$ lie close together. A prior says the objective
values $f(\vx)$ and $f(\vx')$ are each $\N(0, 1)$ with correlation 0.8, the
kind of value a smooth kernel assigns to nearby inputs (@sec-kernel-trick). We
evaluate $f(\vx) = 1.2$. By @eq-gauss-cond-2d with
$\sigma_1 = \sigma_2 = 1$ and $\rho = 0.8$,

$$
f(\vx') \given f(\vx) = 1.2 \;\sim\; \N(0.8 \times 1.2,\; 1 - 0.8^2) = \N(0.96,\; 0.36).
$$

The belief about the unevaluated input moved 80% of the way toward the
observation, and its standard deviation fell from 1 to 0.6. Had we observed
$-1.2$, the mean would have moved to $-0.96$, and the standard deviation would
again have been 0.6. @sec-gp-conditioning does the same with a kernel
supplying the correlations, for every unevaluated input at once.
:::

### Computing it {#sec-gauss-conditioning-code}

In code, @eq-gauss-conditional is evaluated with the Cholesky factor of
$\mSigma_{aa}$ and triangular solves, never with an explicit inverse
(@sec-cholesky). With $\mL\mL^\T = \mSigma_{aa}$, set
$\mathbf{V} = \mL^{-1}\mSigma_{ab}$ and $\vw = \mL^{-1}(\mathbf{a} - \vmu_a)$.
Since $\mSigma_{aa}^{-1} = \mL^{-\T}\mL^{-1}$, the mean offset is
$\mathbf{V}^\T\vw$ and the subtracted covariance is $\mathbf{V}^\T\mathbf{V}$.

::: {.code title="NumPy"}
```python
import numpy as np

def condition(mu, Sigma, ia, ib, a):
    """Mean and covariance of x[ib] given x[ia] = a, for x ~ N(mu, Sigma)."""
    Saa = Sigma[np.ix_(ia, ia)]
    Sab = Sigma[np.ix_(ia, ib)]
    Sbb = Sigma[np.ix_(ib, ib)]
    L = np.linalg.cholesky(Saa)
    V = np.linalg.solve(L, Sab)          # L^{-1} Sigma_ab
    w = np.linalg.solve(L, a - mu[ia])   # L^{-1} (a - mu_a)
    return mu[ib] + V.T @ w, Sbb - V.T @ V
```
With a zero mean and a kernel matrix as `Sigma`, this function is the
Gaussian process predictor of @alg-gp-regression under different names.
:::

When the observed coordinates nearly determine the others, the subtraction
$\mSigma_{bb} - \mathbf{V}^\T\mathbf{V}$ cancels most of its digits, and the
result can come out slightly asymmetric or with tiny negative eigenvalues.
Symmetrizing it and adding a small jitter before factorizing it again, for
example to draw samples, repairs the damage.

## Sums and products {#sec-gaussian-sums}

Two more operations combine Gaussians, and they are easy to confuse because
both take two Gaussians and return one. Adding two independent Gaussian random
variables describes a quantity built from two uncertain parts, such as a
function value plus measurement noise. Multiplying two Gaussian densities
describes two independent pieces of evidence about one quantity, which is what
Bayes' rule does with a Gaussian prior and a Gaussian likelihood. The first
operation makes the uncertainty larger; the second makes it smaller.

### Sums of independent Gaussians {#sec-gauss-sum}

If $X \sim \N(\mu_1, \sigma_1^2)$ and $Y \sim \N(\mu_2, \sigma_2^2)$ are
independent, the pair $(X, Y)$ is jointly Gaussian with a diagonal covariance,
and $X + Y$ is the linear map with the single row $(1, 1)$. By
@eq-gauss-affine,

$$
X + Y \sim \N\!\left(\mu_1 + \mu_2,\; \sigma_1^2 + \sigma_2^2\right).
$$ {#eq-gauss-sum}

Variances add; standard deviations do not. Two independent errors with
standard deviations 3 and 4 add up to an error with standard deviation 5,
not 7. If $X$ and $Y$ are correlated, the same map gives the variance
$\sigma_1^2 + \sigma_2^2 + 2\Cov[X, Y]$. Two uses recur in the book. An
observation $y = f(\vx) + \varepsilon$ with independent noise
$\varepsilon \sim \N(0, \sigma_n^2)$ has variance $\Var[f(\vx)] + \sigma_n^2$,
the predictive variance of @sec-gp-noise. And the average of $n$ independent
measurements, each with variance $\sigma^2$, has variance $\sigma^2/n$, so its
standard deviation falls like $1/\sqrt{n}$.

### Products of Gaussian densities {#sec-gauss-product}

Now take two Gaussian densities over the same variable $x$ and multiply them
pointwise. The result is not normalized, but its shape is Gaussian: the sum of
two quadratic exponents is quadratic.

::: {.derivation title="The product of two Gaussian densities"}
Take the densities $\N(x;\, \mu_1, \sigma_1^2)$ and $\N(x;\, \mu_2, \sigma_2^2)$.

1. Multiplying exponentials adds the exponents:
   $-\tfrac12\left[(x - \mu_1)^2/\sigma_1^2 + (x - \mu_2)^2/\sigma_2^2\right]$.
2. Collect powers of $x$: the coefficient of $-\tfrac12 x^2$ is
   $1/\sigma_1^2 + 1/\sigma_2^2$, and the coefficient of $x$ is
   $\mu_1/\sigma_1^2 + \mu_2/\sigma_2^2$.
3. By @eq-gauss-recipe, the product is proportional to a Gaussian density with
   precision $1/\sigma^2 = 1/\sigma_1^2 + 1/\sigma_2^2$ and mean
   $\mu = \sigma^2\left(\mu_1/\sigma_1^2 + \mu_2/\sigma_2^2\right)$.
4. The terms that do not involve $x$ are
   $-\tfrac12\left[\mu_1^2/\sigma_1^2 + \mu_2^2/\sigma_2^2 - \mu^2/\sigma^2\right]$.
   Over a common denominator the bracket simplifies to
   $(\mu_1 - \mu_2)^2/(\sigma_1^2 + \sigma_2^2)$.
5. The normalizing factors multiply to $1/(2\pi\sigma_1\sigma_2)$. Since
   $\sigma^2(\sigma_1^2 + \sigma_2^2) = \sigma_1^2\sigma_2^2$, this equals
   $\big[1/\sqrt{2\pi\sigma^2}\big]\big[1/\sqrt{2\pi(\sigma_1^2 + \sigma_2^2)}\big]$.
6. Collecting the factors:
   $\N(x;\, \mu_1, \sigma_1^2)\,\N(x;\, \mu_2, \sigma_2^2) = \N(\mu_1;\, \mu_2, \sigma_1^2 + \sigma_2^2)\;\N(x;\, \mu, \sigma^2)$.
:::

In $d$ dimensions the same steps, with @eq-gauss-recipe-nd in place of
@eq-gauss-recipe, give

$$
\N(\vx;\, \vmu_1, \mSigma_1)\,\N(\vx;\, \vmu_2, \mSigma_2)
= Z\;\N(\vx;\, \vmu, \mSigma),
\qquad
\mSigma = \left(\mSigma_1^{-1} + \mSigma_2^{-1}\right)^{-1},\;\;
\vmu = \mSigma\left(\mSigma_1^{-1}\vmu_1 + \mSigma_2^{-1}\vmu_2\right),
$$ {#eq-gauss-product}

with the constant $Z = \N(\vmu_1;\, \vmu_2,\, \mSigma_1 + \mSigma_2)$
[@rasmussen2006gaussian, app. A.2].

Read the result in terms of precision. The precisions add, so the product is
narrower than either factor. The mean is a weighted average of the two means,
with weights proportional to the precisions, so the sharper density pulls
harder. The constant $Z$ is the area under the raw product. It is large when
the two densities agree and tiny when they put their mass in different places.

This is Bayes' rule for a Gaussian prior and a Gaussian measurement. Suppose a
prior belief about a quantity $\theta$ is $\N(\mu_0, \sigma_0^2)$, and we
observe $y = \theta + \varepsilon$ with noise $\varepsilon \sim \N(0, \sigma_n^2)$.
As a function of $\theta$, the likelihood $\N(y;\, \theta, \sigma_n^2)$ is the
density $\N(\theta;\, y, \sigma_n^2)$, since the formula is symmetric in $y$
and $\theta$. The posterior is proportional to prior times likelihood, so it
is Gaussian with precision $1/\sigma_0^2 + 1/\sigma_n^2$ and a mean between
the prior mean and the observation. The constant
$Z = \N(y;\, \mu_0, \sigma_0^2 + \sigma_n^2)$ is the density of the observation
under the prior, which @sec-evidence calls the model evidence. Notice that it
is the distribution of the sum $\theta + \varepsilon$ from @eq-gauss-sum: the
two operations of this section meet. @sec-prior-likelihood-posterior develops
this view, and @sec-blr extends it from one number to a vector of weights.

```{figure}
//| figure: gauss-combine
//| label: fig-gauss-combine
//| fig-cap: "Two ways to combine two Gaussians. *Sum of variables* shows the density of $X + Y$ for independent $X$ and $Y$, @eq-gauss-sum: wider than both inputs. *Product of densities* shows the normalized pointwise product of the two density curves, @eq-gauss-product: narrower than both, with its mean pulled toward the sharper input. The dashed curve is the raw product, whose area is $Z$. Set the means and standard deviations with the sliders, or drag near a peak to move it."
```

**Switch between the two operations with the same inputs.** The sum is
centered at $\mu_1 + \mu_2$ and is wider than either input. The product lies
between $\mu_1$ and $\mu_2$ and is narrower than either.

**In the product, make one input very wide.** With $\sigma_1$ at 2.5 the
product nearly coincides with the second input. A vague prior hardly changes
what a precise measurement says.

**Pull the two means apart.** The normalized product keeps its width, because
the precisions do not depend on the means, but the raw product sinks toward
zero, and $Z$ with it. Two confident densities that disagree produce a
confident compromise in a region where neither puts much mass. A small $Z$ is
the warning sign that the prior and the measurement are in conflict.

::: {.pitfall title="The product of two Gaussian variables is not Gaussian"}
@eq-gauss-product multiplies density functions. Multiplying two Gaussian
random variables is a different operation with a different answer: if $X$ and
$Y$ are independent standard normals, their product $XY$ has a density that
grows without bound near zero and has heavier tails than any Gaussian.
Gaussian random variables are closed under addition and linear maps; Gaussian
densities are closed under multiplication. Keep the two apart when reading a
derivation.
:::

## Working with Gaussians in code {#sec-gauss-code}

Three habits avoid most numerical trouble.

**Work with log densities.** In many dimensions the density @eq-gauss-density
is a product of many small factors and underflows. For a standard Gaussian in
$d = 1000$ dimensions, even the density at the mean,
$(2\pi)^{-500}$, is about $10^{-399}$, below the smallest positive
double-precision number. The log density is a sum of moderate numbers. Compute
it from the Cholesky factor $\mL$ of $\mSigma$: with
$\vw = \mL^{-1}(\vx - \vmu)$ the quadratic form is $\vw^\T\vw$, and
$\log\lvert\mSigma\rvert = 2\sum_i \log L_{ii}$ (@sec-determinants).

::: {.code title="NumPy"}
```python
import numpy as np

def gaussian_logpdf(x, mu, Sigma):
    L = np.linalg.cholesky(Sigma)
    w = np.linalg.solve(L, x - mu)       # L^{-1} (x - mu)
    d = len(mu)
    return -0.5 * w @ w - np.log(np.diag(L)).sum() - 0.5 * d * np.log(2 * np.pi)
```
:::

**Never form $\mSigma^{-1}$.** Every formula in this chapter that contains an
inverse is evaluated with a Cholesky factorization and triangular solves, as
in the conditioning code above, for the reasons of @sec-linalg-why-not-invert.

**Check the parameterization.** The notation $\N(\mu, \sigma^2)$ puts the
variance second, but most libraries take the standard deviation: NumPy's
`random.normal(loc, scale)`, SciPy's `stats.norm(loc, scale)`, and PyTorch's
`Normal(loc, scale)` all expect $\sigma$. Passing a variance where a standard
deviation is expected is a silent and common bug. Multivariate versions take
the covariance matrix, or sometimes its Cholesky factor, as in PyTorch's
`MultivariateNormal(loc, scale_tril=L)`.

## Exercises {#sec-gauss-exercises}

::: {.exercise #exr-gauss-compare}
The values $f(\vx_1)$ and $f(\vx_2)$ are jointly Gaussian with means 0.3 and
0.1, standard deviations 0.2 each, and correlation 0.75. Compute the
probability that $f(\vx_1) > f(\vx_2)$. Repeat with correlation 0 and explain
the difference.

::: {.solution}
By @ex-gauss-difference, $D = f(\vx_1) - f(\vx_2)$ has mean $0.2$ and variance
$0.04 + 0.04 - 2 \times 0.75 \times 0.04 = 0.02$, so its standard deviation is
$0.141$ and $\Prob(D > 0) = \Phi(0.2/0.141) = \Phi(1.41) \approx 0.92$. With
correlation 0 the variance is $0.08$, the standard deviation $0.283$, and the
probability $\Phi(0.71) \approx 0.76$. Positive correlation means the two
values tend to err in the same direction, so the errors partly cancel in the
difference, and the ordering is more certain than either value.
:::
:::

::: {.exercise #exr-gauss-cond-2d}
Derive @eq-gauss-cond-2d from @eq-gauss-conditional. Then show that, in
general, observing $\mathbf{a}$ never increases the variance of any linear
combination $\vw^\T\mathbf{b}$.

::: {.solution}
Take $\mathbf{a} = x_1$ and $\mathbf{b} = x_2$, so $\mSigma_{aa} = \sigma_1^2$,
$\mSigma_{ab} = \rho\sigma_1\sigma_2$, and $\mSigma_{bb} = \sigma_2^2$. The mean
is $\mu_2 + (\rho\sigma_1\sigma_2/\sigma_1^2)(a - \mu_1) = \mu_2 + \rho(\sigma_2/\sigma_1)(a - \mu_1)$,
and the variance is $\sigma_2^2 - \rho^2\sigma_1^2\sigma_2^2/\sigma_1^2 = \sigma_2^2(1 - \rho^2)$.
In general, the variance of $\vw^\T\mathbf{b}$ falls from $\vw^\T\mSigma_{bb}\vw$
to $\vw^\T\mSigma_{bb}\vw - \vw^\T\mSigma_{ab}^\T\mSigma_{aa}^{-1}\mSigma_{ab}\vw$.
With $\mathbf{q} = \mSigma_{ab}\vw$, the subtracted amount is
$\mathbf{q}^\T\mSigma_{aa}^{-1}\mathbf{q} \ge 0$, because the inverse of a
positive definite matrix is positive definite. The decrease is zero exactly
when $\mathbf{q} = \mathbf{0}$, that is, when $\vw^\T\mathbf{b}$ is uncorrelated
with every observed coordinate.
:::
:::

::: {.exercise #exr-gauss-not-joint}
Let $X \sim \N(0, 1)$, and let $S$ be independent of $X$ and equal to $+1$ or
$-1$ with probability one half each. Set $Y = SX$. Show that $Y$ is standard
normal and that $X$ and $Y$ are uncorrelated, but that they are not
independent and not jointly Gaussian.

::: {.solution}
Because $\N(0, 1)$ is symmetric, $-X$ has the same distribution as $X$, so
$\Prob(Y \le y) = \tfrac12\Prob(X \le y) + \tfrac12\Prob(-X \le y) = \Phi(y)$.
The covariance is $\E[XY] = \E[S]\,\E[X^2] = 0 \times 1 = 0$. They are
dependent, because $\lvert Y\rvert = \lvert X\rvert$: knowing $X$ leaves only
two possible values for $Y$. They are not jointly Gaussian, because the linear
combination $X + Y = (1 + S)X$ equals zero with probability one half and is
otherwise $\N(0, 4)$; a linear combination of jointly Gaussian variables would
be Gaussian (@sec-gaussian-linear). Uncorrelated implies independent only for
jointly Gaussian variables.
:::
:::

::: {.exercise #exr-gauss-repeat}
A quantity $\theta$ has prior $\N(0, \sigma_0^2)$ and is measured $n$ times,
$y_i = \theta + \varepsilon_i$, with independent noise
$\varepsilon_i \sim \N(0, \sigma_n^2)$. Use @eq-gauss-product to find the
posterior of $\theta$. What happens as $\sigma_0 \to \infty$? What is the
posterior variance for $\sigma_0 = 1$?

::: {.solution}
Each measurement contributes a likelihood factor $\N(\theta;\, y_i, \sigma_n^2)$.
Multiplying the prior by the $n$ factors one at a time, the precisions add:
the posterior precision is $1/\sigma_0^2 + n/\sigma_n^2$, and the posterior mean
is $\left(\sum_i y_i/\sigma_n^2\right)\big/\left(1/\sigma_0^2 + n/\sigma_n^2\right)$.
As $\sigma_0 \to \infty$ the prior's precision vanishes, the mean tends to the
sample average $\bar{y}$, and the variance to $\sigma_n^2/n$, the familiar
standard error of an average. For $\sigma_0 = 1$ the variance is
$1/(1 + n/\sigma_n^2) = \sigma_n^2/(n + \sigma_n^2)$. The same number returns
in @exr-noise-floor, where a Gaussian process is evaluated $n$ times at one
input.
:::
:::

## Further reading {#further-reading .unnumbered}

- @bishop2006pattern, section 2.3, derives the conditional and marginal
  distributions of a partitioned Gaussian by completing the square, the route
  taken here, and continues to the linear Gaussian models of
  @sec-bayesian-inference.
- @rasmussen2006gaussian, appendix A, collects the Gaussian and matrix
  identities that Gaussian processes need, including products of Gaussian
  densities and sampling with the Cholesky factor.
- @murphy2022probabilistic, chapter 3, treats the multivariate Gaussian and
  linear Gaussian systems with many worked examples.
- @blitzstein2019introduction is a gentle probability text whose treatment of
  the multivariate normal defines it through linear combinations, the
  definition used in @sec-gaussian-linear.
- @petersen2012matrix lists the block-inverse and Gaussian identities in a
  compact reference form.
- @galton1886regression is the paper that named regression toward the mean,
  the effect the conditioning figure shows.
