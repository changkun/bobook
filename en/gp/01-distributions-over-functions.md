---
status: done
synopsis: "From a prior over a model's weights to a prior over whole functions: features, the kernel they induce, the definition of a Gaussian process, and what functions drawn from it look like as the kernel and its hyperparameters change."
sources: ["Rasmussen and Williams 2006, ch. 2 and 4", "Garnett 2023, ch. 2 and 3", "Neal 1996", "Stein 1999", "Snoek et al. 2012"]
---

# Distributions over Functions {#sec-function-space}

The last sections of @sec-information borrowed a model this book had not yet
built, the Gaussian process. This part builds it, starting from the last model
that was built in full. @sec-blr fitted a straight line the Bayesian way. It
put a Gaussian prior on the line's two weights, observed a few noisy points,
and found that the posterior over the weights was again Gaussian, available in
closed form. Every quantity it produced, a prediction, its uncertainty, a
plausible line drawn at random, came out of a few lines of algebra on
Gaussians.

The trouble is the line. The book's running example has a wide hill on the
left and a narrow, taller peak on the right (@sec-usual-tools), and nothing
guarantees that a real objective, a validation accuracy or a person's comfort,
is straight either. We want to keep the algebra and drop the line.

This chapter does that in three moves. First, it gives the linear model many
features, so that it can bend, and asks what the model's prior says about the
function's values. Second, it notices that this prior depends on the features
only through one function of two inputs, the kernel, and lets the number of
features grow without bound until only the kernel is left. Third, it takes
what remains as a definition: a Gaussian process, a probability distribution
whose samples are whole functions. The chapter ends by drawing functions from
that distribution and reading what each choice of kernel says about the
function we are looking for. @sec-gp-regression then conditions this prior on
data.

## From weights to functions {#sec-weights-to-functions}

Write the straight line as $f(x) = w_1 + w_2 x$: two weights, each multiplying
a fixed function of the input, the constant $1$ and $x$ itself. Bayesian
linear regression places a Gaussian prior on the weight vector $\vw$ and needs
nothing else from the model, because the function is linear in $\vw$. That
linearity is what makes the algebra work, and nothing in it requires the two
fixed functions to be $1$ and $x$.

So replace them with any $M$ fixed functions $\phi_1, \dots, \phi_M$ of the
input, called **features** or basis functions, and keep the weights:

$$
f(\vx) = \sum_{j=1}^{M} w_j\, \phi_j(\vx) = \boldsymbol{\phi}(\vx)^\T \vw,
\qquad
\vw \sim \N(\mathbf{0}, \mSigma_p).
$$ {#eq-fs-linear-model}

Here $\boldsymbol{\phi}(\vx) = (\phi_1(\vx), \dots, \phi_M(\vx))^\T$ stacks the
features of one input, and $\mSigma_p$ is the prior covariance of the weights
(the subscript is for *prior*). The model is still linear in $\vw$, so every
Gaussian computation of @sec-blr still applies. Only the features changed.

Which features? Polynomials and sines are common choices. This chapter uses
bumps. For a one-dimensional input $x$, the $j$th bump is a Gaussian-shaped
function centered at its own point $c_j$, with a width $s$ shared by all bumps:

$$
\phi_j(x) = \exp\!\left(-\frac{(x - c_j)^2}{2 s^2}\right).
$$

A weighted sum of bumps rises where the weights are positive and dips where
they are negative. With enough bumps, close enough together, it can take
almost any smooth shape.

### A prior over weights is a prior over functions {#sec-fs-weights-prior}

Draw a weight vector from its prior and the model becomes one particular
function. Draw again and it becomes another. The prior over weights is
therefore, with no further assumption, a prior over functions, and the most
direct way to see what it believes is to draw from it.

```{figure}
//| figure: fs-features
//| label: fig-fs-features
//| fig-cap: "Functions drawn from a linear model with six Gaussian-bump features. Each solid curve is one draw of the weights. The lightly filled dashed bumps are the weighted features that add up to the violet curve; where bumps barely overlap, the curve runs along the top of each one. Triangles under the plot mark the bump centers. The wide band covers 95% of the prior probability at each input. The strip below shows the covariance between $f(x_0)$ and $f(x)$ that the features induce (@eq-fs-induced-kernel), next to the kernel it approaches as bumps are added (@sec-kernel-trick). Press *Draw new weights* for new functions, and click or drag on either panel to move $x_0$. The number, width, and placement of the bumps are illustrative."
```

Look at the band first. With six bumps it bulges over each bump and narrows
between them: the model is confident that $f$ is close to zero midway between
two centers, for no reason except that no feature lives there. The functions
it draws inherit the pattern. They swing over the centers and sag toward zero
in the gaps. Nobody believes that about an objective function. It is an
artifact of where the features happen to sit, and @sec-kernel-trick removes
it.

### What the prior says about function values {#sec-fs-prior-values}

To make "what the prior believes" precise, ask about the function's values at
a finite set of inputs $\vx_1, \dots, \vx_n$. Stack the values into a vector
$\vf = (f(\vx_1), \dots, f(\vx_n))^\T$, and stack the features of those inputs
into the $n \times M$ matrix $\boldsymbol{\Phi}$ whose $i$th row is
$\boldsymbol{\phi}(\vx_i)^\T$. Then @eq-fs-linear-model, applied to each input
in turn, says $\vf = \boldsymbol{\Phi}\vw$: the function values are a linear
map of the weights.

::: {.derivation title="The prior over function values"}
1. $\vf = \boldsymbol{\Phi}\vw$, and $\boldsymbol{\Phi}$ is fixed: it depends
   on the inputs, not on $\vw$.
2. A linear map of a Gaussian vector is Gaussian (@sec-gaussian-linear), so
   $\vf$ is Gaussian. It remains to find its mean and covariance.
3. Mean: by linearity of expectation,
   $\E[\vf] = \boldsymbol{\Phi}\,\E[\vw] = \mathbf{0}$.
4. Covariance: because the mean is zero,
   $\Cov[\vf] = \E[\vf\vf^\T] = \E[\boldsymbol{\Phi}\vw\vw^\T\boldsymbol{\Phi}^\T]
   = \boldsymbol{\Phi}\,\E[\vw\vw^\T]\,\boldsymbol{\Phi}^\T
   = \boldsymbol{\Phi}\mSigma_p\boldsymbol{\Phi}^\T$,
   again by linearity, with $\E[\vw\vw^\T] = \mSigma_p$ since $\vw$ has mean
   zero.
5. Entry $(i, j)$ of that matrix is
   $\Cov[f(\vx_i), f(\vx_j)] = \boldsymbol{\phi}(\vx_i)^\T \mSigma_p\, \boldsymbol{\phi}(\vx_j)$.
:::

So the values at any $n$ inputs are jointly Gaussian,

$$
\vf \sim \N\!\left(\mathbf{0},\; \boldsymbol{\Phi}\mSigma_p\boldsymbol{\Phi}^\T\right),
$$ {#eq-fs-prior-values}

and every entry of the covariance has the same form:

$$
k(\vx, \vx') = \boldsymbol{\phi}(\vx)^\T \mSigma_p\, \boldsymbol{\phi}(\vx').
$$ {#eq-fs-induced-kernel}

Two things about this result carry the rest of the chapter. First, the weights
have disappeared. The distribution of the function values is stated entirely
through $k$, a function that takes two inputs and returns the covariance of
the function's values there. Second, nothing restricted the number of inputs or
which ones we chose. Any finite list of inputs gets a joint Gaussian, and every
one of those Gaussians is built from the same $k$.

The strip under @fig-fs-features plots $k(x_0, x)$ against $x$ for one fixed
$x_0$: the covariance of $f(x_0)$ with the value at every other input. Drag
$x_0$ onto the center of a bump and the curve is tall and narrow. Drag it
between two centers and the curve shrinks. In this model, how strongly two
values are related depends on where they are, not only on how far apart they
are.

::: {.example #ex-fs-line title="The straight line as a kernel"}
Take the line again: $\boldsymbol{\phi}(x) = (1, x)^\T$, with independent
weights of variances $\sigma_0^2$ and $\sigma_1^2$, so
$\mSigma_p = \diag(\sigma_0^2, \sigma_1^2)$. Then @eq-fs-induced-kernel gives

$$
k(x, x') = \sigma_0^2 + \sigma_1^2\, x x'.
$$

The prior variance at $x$ is $k(x, x) = \sigma_0^2 + \sigma_1^2 x^2$, which grows
quadratically away from $x = 0$: the fan of prior lines in @sec-blr, now in one
formula. With $\sigma_0 = \sigma_1 = 1$, the values at $x = 1$ and $x = 2$ have
covariance $3$ and variances $2$ and $5$, so their correlation is
$3/\sqrt{10} \approx 0.95$. A line that is high at $x = 1$ is almost certainly
high at $x = 2$, because two points determine a line.

The same fact shows up as a defect of the covariance matrix. With two
features, $\boldsymbol{\Phi}$ has two columns, so for three or more inputs the
$n \times n$ matrix $\boldsymbol{\Phi}\mSigma_p\boldsymbol{\Phi}^\T$ has rank
at most 2 and is singular. The prior gives zero variance to some combinations
of values; @exr-fs-line finds one.
:::

## The kernel {#sec-kernel-trick}

The bump model has two defects, and the example has just named the second. Its
beliefs depend on where the bumps were put, as the pinched band in
@fig-fs-features shows. And with $M$ features it has only $M$ degrees of
freedom, so with more than $M$ inputs its covariance matrix is singular: the
prior is certain about combinations of values it has no reason to be certain
about. Both defects have the same cure. Use more bumps, packed more densely,
all the way to a continuum.

That cure sounds expensive. Computing $\boldsymbol{\phi}(x)$ for a million
features costs a million evaluations per input. But @eq-fs-prior-values never
needs the features themselves, only the inner products in
@eq-fs-induced-kernel. If those can be computed directly, the number of
features stops mattering. For bumps they can, in closed form.

### Infinitely many bumps {#sec-fs-rbf-limit}

Space the centers evenly, $\Delta$ apart, along the whole real line, and give
the weights the same variance $\sigma_w^2$, independently, so that
$\mSigma_p = \sigma_w^2 \mI$. Each new bump adds variance, so as the spacing
shrinks the weight variance must shrink in proportion, or the functions would
grow without bound. Choose $\sigma_w^2 = \Delta / (s\sqrt{\pi})$; the constant
is the one that gives the result below unit variance.

::: {.derivation title="From bumps to the RBF kernel"}
1. With $\mSigma_p = \sigma_w^2\mI$ and centers $c_j = j\Delta$, the inner
   product @eq-fs-induced-kernel is a sum over bumps:
   $$
   k(x, x') = \frac{\Delta}{s\sqrt{\pi}} \sum_{j} \phi_j(x)\,\phi_j(x').
   $$
2. A sum of function values at points $\Delta$ apart, multiplied by $\Delta$,
   is a Riemann sum. As $\Delta \to 0$ it becomes an integral over the center
   $c$. Writing $\phi_c$ for the bump centered at $c$,
   $$
   k(x, x') = \frac{1}{s\sqrt{\pi}} \int_{-\infty}^{\infty} \phi_c(x)\,\phi_c(x')\, \dd c.
   $$
3. The product of two bumps is one exponential, with exponent
   $-\left[(x - c)^2 + (x' - c)^2\right]/2s^2$. Complete the square in $c$:
   with $m = (x + x')/2$, expanding both sides confirms
   $(x - c)^2 + (x' - c)^2 = 2(c - m)^2 + (x - x')^2/2$. So
   $\phi_c(x)\,\phi_c(x') = e^{-(x - x')^2/4s^2}\, e^{-(c - m)^2/s^2}$.
4. The first factor does not involve $c$, so it comes out of the integral:
   $k(x, x') = e^{-(x - x')^2/4s^2}\, I / (s\sqrt{\pi})$, with
   $I = \int e^{-(c - m)^2/s^2}\, \dd c$.
5. The integrand of $I$ is an unnormalized Gaussian density in $c$ with
   variance $s^2/2$, so $I$ is that density's normalizing constant,
   $\sqrt{2\pi \cdot s^2/2} = s\sqrt{\pi}$ (@sec-gaussian-1d).
6. The factors $s\sqrt{\pi}$ cancel, leaving
   $k(x, x') = \exp\!\left(-(x - x')^2 / 4s^2\right)$.
:::

Step 3 used the fact that the product of two Gaussian bumps is again a
Gaussian bump; @sec-id-products states the general rule. Write $\ell = \sqrt{2}\,s$,
and multiply every weight variance by a constant $\sigma_f^2$, which multiplies
$k$ by $\sigma_f^2$. The result is

$$
k(x, x') = \sigma_f^2 \exp\!\left(-\frac{(x - x')^2}{2\ell^2}\right).
$$ {#eq-fs-rbf}

This is the **RBF kernel** (for radial basis function), also called the
squared exponential kernel, and it is the kernel @sec-gp-regression works
with. The same formula is the similarity measure of the RBF support vector
machine, a classifier whose kernel width @sec-cs-classifier tunes with
Bayesian optimization. The construction is a standard one [@rasmussen2006gaussian, sec. 4.2.1]. For an
input with several coordinates, bumps in every direction give the same formula
with $(x - x')^2$ replaced by the squared distance $\lVert \vx - \vx' \rVert^2$.

Of the whole feature model, two numbers remain. The **lengthscale** $\ell$ says
how far apart two inputs must be before their values become nearly unrelated:
the kernel falls to $0.61$ of its peak at distance $\ell$, to $0.14$ at $2\ell$,
and to $0.011$ at $3\ell$. The **amplitude** $\sigma_f$ says how large the
function is: $k(x, x) = \sigma_f^2$ is the prior variance of $f$ at any single
input. A kernel with $\sigma_f = 1$, the setting of most of
@sec-gp-regression, is said to have unit amplitude.

The figure below is the model of @fig-fs-features with the number of features
as a control. The readout gives the largest gap, over all inputs, between the
covariance the bumps induce and the limit @eq-fs-rbf.

```{figure}
//| figure: fs-features
//| label: fig-fs-limit
//| fig-cap: "Taking the limit yourself. The strip compares the covariance that $M$ bumps induce (solid) with the RBF kernel of @eq-fs-rbf (dashed). Step from 4 features to 16, then to ∞, which drops the features and draws from the kernel directly. Move the lengthscale to see that narrower bumps need more of them."
features: 8
showFeatures: false
```

A few experiments show how the limit is approached.

**Start at 4 features.** The band swells and pinches, and the solid curve in
the strip changes shape as you drag $x_0$ across a bump. The gap to the RBF
kernel is as large as the kernel itself, or larger.

**Step up to 16.** By 12 features the band is nearly flat and the gap is
about 0.1; at 16 it is below 0.01, and the functions look like those drawn
at ∞, where no features exist at all.

**Shorten the lengthscale to 0.05.** The bumps narrow, 16 of them are no
longer enough, and the band pinches again. The number of bumps needed is
roughly the width of the domain divided by the width of one bump.

That last experiment is the practical case for working with the kernel. On
$[0, 1]$ with $\ell = 0.1$ a few dozen bumps would do. In $d$ input dimensions
the bumps must tile a $d$-dimensional grid, so the count is raised to the power
$d$: 16 bumps per axis in 10 dimensions is $16^{10} \approx 10^{12}$ features.
The kernel @eq-fs-rbf costs $d$ subtractions and one exponential per pair of
inputs, whatever the dimension.

Exchanging an explicit list of features for a function that computes their
inner product directly is called the **kernel trick**
[@rasmussen2006gaussian, sec. 2.1.2]. It runs through the rest of the book.
@sec-gp-conditioning will show that predictions from data, too, need the
kernel and nothing else.

### Which functions are kernels {#sec-fs-valid-kernels}

Once we stop writing features down, we need to know which functions $k$ may
serve as covariances. Two conditions are necessary. The function must be
symmetric, $k(\vx, \vx') = k(\vx', \vx)$, because covariance is. And for every
finite set of inputs and every vector of coefficients $\mathbf{a}$,

$$
\sum_{i=1}^n \sum_{j=1}^n a_i a_j\, k(\vx_i, \vx_j) \ge 0,
$$

because the left side is the variance of the weighted sum
$\sum_i a_i f(\vx_i)$, and no variance is negative. A matrix with this property
is positive semidefinite (@sec-positive-definite), and a function whose
matrices all have it is a **positive semidefinite kernel**.

Every inner product of features passes the test, since
$\mathbf{a}^\T\boldsymbol{\Phi}\mSigma_p\boldsymbol{\Phi}^\T\mathbf{a}
= (\boldsymbol{\Phi}^\T\mathbf{a})^\T\mSigma_p(\boldsymbol{\Phi}^\T\mathbf{a}) \ge 0$
for a covariance $\mSigma_p$. The converse is what licenses designing kernels
directly: under mild technical conditions, every symmetric positive
semidefinite kernel is the inner product of some list of features, possibly
infinitely many, a result known as Mercer's theorem
[@rasmussen2006gaussian, secs. 2.2 and 4.3], which @sec-ka-mercer states with
its conditions and uses to build the features from the kernel's eigenfunctions. We may therefore propose any function that passes the test and never
write its features down.

The test is not a formality. Similarities that look reasonable can fail it.
The "sigmoid" kernel $\tanh(a + b\,\vx^\T\vx')$, sometimes proposed by analogy
with neural networks, is never positive definite, so it is not a valid
covariance [@rasmussen2006gaussian, sec. 4.2.3]. @exr-fs-box shows a simpler
failure and what it would mean: a combination of function values with negative
variance.

::: {.note title="Neural networks in the limit"}
Bumps are not special. @neal1996bayesian showed that a neural network with one
hidden layer and random weights also becomes a Gaussian process as the number
of hidden units grows, provided the prior variance of the output weights
shrinks in proportion. There the hidden units are themselves random, and the
Gaussian limit comes from the central limit theorem rather than from Gaussian
weights [@rasmussen2006gaussian, sec. 4.2.3]. Several routes lead to the same
object, which is one reason to define it on its own terms.
:::

## Definition of a Gaussian process {#sec-gp-definition}

The previous sections arrived at a function $k$, and at the claim that it
alone fixes a joint Gaussian for the function's values at any finite set of
inputs. Taken as the starting point instead of a consequence, that claim is
the definition.

::: {.definition #def-fs-gp title="Gaussian process"}
Let $\X$ be a set of inputs. A **Gaussian process** on $\X$ is a collection of
random variables $f(\vx)$, one for each $\vx \in \X$, any finite number of
which have a joint Gaussian distribution [@rasmussen2006gaussian, def. 2.1]. It
is specified by a **mean function** $m(\vx) = \E[f(\vx)]$ and a **covariance
function**, or kernel, $k(\vx, \vx') = \Cov[f(\vx), f(\vx')]$, and we write
$f \sim \GP(m, k)$. For any inputs $\vx_1, \dots, \vx_n$,

$$
\begin{bmatrix} f(\vx_1) \\ \vdots \\ f(\vx_n) \end{bmatrix}
\sim \N(\mathbf{m}, \mK),
$$ {#eq-fs-gp-finite}

where $m_i = m(\vx_i)$ and $[\mK]_{ij} = k(\vx_i, \vx_j)$.
:::

A software engineer can read the definition as an interface. A Gaussian
process is an object that accepts any finite list of inputs and answers with a
mean vector and a covariance matrix, computed entry by entry from $m$ and $k$.
It never has to produce the function at all inputs at once, and no computation
in this book asks it to. In that sense it is a lazily evaluated random
function: defined everywhere, materialized only where we look.

The answers to different queries must agree with each other. Query $\vx_1$ and
$\vx_2$ together, ignore the second value, and the distribution of $f(\vx_1)$
must be what querying $\vx_1$ alone would have returned. For a Gaussian,
ignoring coordinates means reading off a sub-block of the mean vector and the
covariance matrix (@sec-gaussian-marginal). Each entry of $\mK$ depends only on
its own two inputs, so that sub-block is exactly what the smaller query builds.
Consistency is automatic for any covariance specified entry by entry through a
kernel [@rasmussen2006gaussian, sec. 2.2].

::: {.aside title="Does the infinite collection exist?"}
A reader may wonder whether a collection of random variables, one for each of
infinitely many inputs, is a well-defined object. It is, and the condition
that makes it so is the consistency just described, also called the
marginalization property: looking at a larger set of values must not change
the distribution of a smaller set [@rasmussen2006gaussian, sec. 2.2].
A classical theorem of probability theory, the Kolmogorov extension theorem,
turns any family of finite-dimensional distributions with this property (and
with no dependence on the order in which inputs are listed) into a single
random process on the whole input set. Consistency is not automatic for every
recipe. Specifying a function for the entries of the *inverse* covariance
matrix, for example, would make the distribution of $f(\vx_1)$ depend on
which other inputs were queried with it, and the result would not be a
Gaussian process [@rasmussen2006gaussian, sec. 2.2]. Whether the functions a
Gaussian process draws are continuous, or differentiable, is a separate
question, which the kernel answers (@sec-fs-smoothness).
:::

The **mean function** is the guess before any data: where $f$ should be, input
by input. In Bayesian optimization it is almost always zero or a constant,
after the observed values have been standardized to mean zero and unit spread
(@sec-gp-pitfalls). That is a weaker assumption than it sounds. Together with
the kernel's amplitude, it says only that before any evaluation $f(\vx)$ lies
within about $\pm 2\sigma_f$ of zero at each input, and the data move the
posterior wherever they disagree. A mean function that is not constant is
useful when there is real knowledge of a trend, such as a physical model whose
errors the Gaussian process should learn. Unless stated otherwise, the rest of
the book takes $m = 0$.

The **kernel** carries everything else: how strongly values at different
inputs move together, and through that, how smooth, how wiggly, and how large
the functions are. It must be symmetric and positive semidefinite
(@sec-fs-valid-kernels), and nothing more is required. The line kernel of
@ex-fs-line and the RBF kernel of @eq-fs-rbf both qualify.

Two words are easy to confuse. A Gaussian *distribution* describes a vector of
fixed length. A Gaussian *process* describes a function, a collection of random
variables indexed by the inputs, and any finite slice of it is a Gaussian
distribution. The general name for a random collection indexed by a set is a
stochastic process, and the set is its index set; in this book the index set
is always the domain being optimized over. One draw of the whole function is
called a sample path, or simply a sample.

::: {.keyidea title="Two functions specify the prior"}
A Gaussian process prior is specified by a mean function, where we expect $f$
to be, and a kernel, how values at different inputs move together. Every
computation queries them at finitely many inputs and works with an ordinary
Gaussian vector.
:::

## Drawing functions {#sec-gp-prior-samples}

A computer cannot draw a whole function. It can draw the function's values on
a fine grid of inputs and connect them, which on a screen amounts to the same
thing. By @eq-fs-gp-finite those values form a Gaussian vector with mean
$\mathbf{m}$ and covariance $\mK$, so drawing them uses the recipe of
@sec-gaussian-linear: factor $\mK = \mL\mL^\T$ by Cholesky (@sec-cholesky),
draw a vector $\vz$ of independent standard normal numbers, and return
$\mathbf{m} + \mL\vz$. The result has covariance
$\mL\,\E[\vz\vz^\T]\,\mL^\T = \mL\mL^\T = \mK$, as required.

::: {.algorithm #alg-fs-prior-sample title="Drawing functions from a Gaussian process prior"}
Input: mean function $m$, kernel $k$, grid $x_1, \dots, x_N$, number of
samples $S$.

1. Build $m_i = m(x_i)$ and $[\mK]_{ij} = k(x_i, x_j)$.
2. $\mL \leftarrow \operatorname{cholesky}(\mK + \varepsilon\mI)$, with a small
   jitter $\varepsilon$ such as $10^{-8}\sigma_f^2$.
3. For $s = 1, \dots, S$: draw $\vz^{(s)} \sim \N(\mathbf{0}, \mI)$ and set
   $\vf^{(s)} \leftarrow \mathbf{m} + \mL\vz^{(s)}$.
4. Plot each $\vf^{(s)}$ against the grid.
:::

The jitter in step 2 is the one of @sec-linalg-kernel-spectrum: a kernel matrix
on a fine grid is nearly singular. The perturbation it adds to each sample is
of order $\sqrt{\varepsilon}$, far below what a plot can show. The
factorization costs $O(N^3)$ time and is paid once; each sample then costs
$O(N^2)$.

::: {.code title="NumPy"}
```python
import numpy as np

def rbf(a, b, ell=0.1, sf=1.0):
    return sf**2 * np.exp(-0.5 * (a[:, None] - b[None, :]) ** 2 / ell**2)

xs = np.linspace(0, 1, 200)
K = rbf(xs, xs)
L = np.linalg.cholesky(K + 1e-8 * np.eye(len(xs)))
rng = np.random.default_rng(0)
F = L @ rng.standard_normal((len(xs), 3))   # three samples, one per column
```
Plotting the columns of `F` against `xs` gives pictures like the one below.
:::

The figure runs @alg-fs-prior-sample on 201 grid points. It keeps the three
standard normal vectors $\vz$ fixed while you change the kernel, so each
change deforms the same three functions instead of drawing new ones.

```{figure}
//| figure: fs-prior
//| label: fig-fs-prior
//| fig-cap: "Three functions drawn from a zero-mean Gaussian process prior. Change the kernel, the lengthscale, the amplitude, and (for the periodic kernel) the period; the same standard normal draws are reused, so each change deforms the same three functions. The shaded band covers 95% of the prior probability at each input. The strip shows the correlation of $f(x)$ with $f(x_0)$; click or drag on the plot to move $x_0$. The readout compares the expected number of upward zero crossings on $[0, 1]$ (@eq-fs-upcrossings) with the count in the three draws."
```

Each control is worth a minute.

**Drag the lengthscale from 0.3 down to 0.03.** The three functions compress
horizontally, like an accordion. The expected number of zero crossings in the
readout grows in proportion to $1/\ell$, and the counts in the draws scatter
around it.

**Drag the amplitude.** The functions stretch vertically and nothing else
changes: not their shapes, not their crossings, not the correlation strip.
Multiplying the kernel by $\sigma_f^2$ multiplies $\mL$ by $\sigma_f$, and
nothing more.

**Switch from RBF to Matérn 5/2 to Matérn 1/2** at a fixed lengthscale. These
are kernels whose functions are rougher than the RBF's; @sec-fs-smoothness
defines them. The Matérn 5/2 draws are hard to tell from the RBF ones except for a slightly
rougher texture. The Matérn 1/2 draws are jagged at every scale, and the
readout no longer gives an expected number of crossings.

**Choose the periodic kernel.** Every function repeats exactly, once per
period, and the correlation strip returns to 1 at $x_0 \pm p$. If the
lengthscale was below 0.4, it jumps to 0.8 when you switch, because for this
kernel it is measured against the period (@sec-fs-periodicity).

**Move $x_0$.** Where the strip is close to 1, each function's value near
$x_0$ tracks its value at $x_0$, marked with a dot. Where the strip falls to
zero, the two values are unrelated.

## What the kernel encodes {#sec-kernel-beliefs}

Every control in @fig-fs-prior is a statement about the unknown function, made
before any evaluation. The kernel's own parameters, the lengthscale, the
amplitude, and the period, are its hyperparameters in the sense of
@sec-evidence: settings of the prior, kept apart from the function values the
model is about. @sec-kernels fits them to data. This section reads what each
one says.

### Stationarity {#sec-fs-stationarity}

The RBF kernel depends on its two inputs only through their difference
$x - x'$. A kernel with this property is **stationary**: shifting every input
by the same amount leaves the prior unchanged, so the prior looks the same in
every part of the domain. A stationary kernel can be written as a function of
one argument, the difference $r = x - x'$, and we will write $k(r)$ when that
is convenient. All four kernels in @fig-fs-prior are stationary, which is why
their bands have constant width. The six-bump model of @fig-fs-features was
not stationary, because its centers made some inputs special. Nor is the line
kernel $\sigma_0^2 + \sigma_1^2 x x'$ of @ex-fs-line, whose variance grows away
from zero.

Stationary kernels are the default in Bayesian optimization. One consequence
appears in @sec-reading-posterior: far from all data, the posterior returns to
the same prior everywhere.

### Lengthscale and amplitude {#sec-fs-lengthscale}

For a stationary kernel the lengthscale is a frequency in disguise. A classical
result in the theory of random processes gives the expected number of times a
zero-mean stationary Gaussian process crosses zero upward on an interval of
unit length [@rasmussen2006gaussian, sec. 4.1]:

$$
\E[N_0] = \frac{1}{2\pi}\sqrt{\frac{-k''(0)}{k(0)}},
$$ {#eq-fs-upcrossings}

where $k''(0)$ is the second derivative of $k(r)$ at $r = 0$. The formula
needs only the curvature of the kernel at zero distance, that is, how fast the
correlation of two nearby values drops as they move apart. For the RBF kernel,
$k(r) = \sigma_f^2 \exp(-r^2/2\ell^2)$ has $k''(0) = -\sigma_f^2/\ell^2$, so

$$
\E[N_0] = \frac{1}{2\pi\ell}.
$$

With $\ell = 0.1$ a draw crosses zero upward 1.6 times on $[0, 1]$ on average;
with $\ell = 0.03$, 5.3 times. The amplitude cancels, as the amplitude
experiment in @fig-fs-prior showed.

The lengthscale is measured in the units of the input. Measure time in
milliseconds instead of seconds and the right lengthscale becomes a thousand
times larger. That is why @sec-gp-pitfalls recommends scaling every input to
$[0, 1]$ before fitting: it makes a lengthscale of 0.1 mean the same thing in
every problem. When the input has several coordinates that matter differently,
each can have its own lengthscale (@sec-ard).

Even with scaled inputs, the same lengthscale does not mean the same thing in
every dimension. @sec-linalg-distances showed that two inputs drawn at random
from the unit cube $[0, 1]^d$ are typically $\sqrt{d/6}$ apart: about $0.41$ in
one dimension, $1.0$ in six, and $2.9$ in fifty. With $\ell = 0.2$, the RBF
kernel at that distance, $\exp(-(d/6)/2\ell^2)$, is $0.12$ in one dimension, $4
\times 10^{-6}$ in six, and below $10^{-45}$ in fifty. In six dimensions, then,
this prior already treats the values at two random inputs as unrelated, and
each observation informs only a small neighborhood around itself, which is what
@fig-info-gain-dim showed from the side of information. Scaling the lengthscale
with $\sqrt{d}$, here $\ell = 0.2\sqrt{d}$, keeps that correlation at $0.12$ in
every dimension. In 2024, standard Bayesian optimization was shown to perform
well in high dimensions once its lengthscale prior was scaled with the
dimension [@hvarfner2024vanilla], and this arithmetic is one way to see why
such scaling helps (inference). @sec-lengthscale-priors and @sec-hd-diagnosis
tell that story.

The amplitude is simpler. $k(x, x) = \sigma_f^2$ is the prior variance of $f$
at each input, so before any data the model puts $f(x)$ within
$\pm 1.96\,\sigma_f$ of the mean with 95% probability. Scaling a Gaussian
process by a constant $c$ scales its kernel by $c^2$ (@exr-fs-sum), which is
why the amplitude only stretched the draws. With standardized outputs, an
amplitude near 1 is the natural starting point. Libraries often store it as a
separate "output scale" hyperparameter that multiplies a kernel of unit
amplitude.

::: {.pitfall title="A lengthscale in the wrong units"}
Suppose the inputs range over $[0, 1000]$ and the lengthscale is left at a
default near 1. Under the prior, values a few units apart are then nearly
independent: at a distance of 3, the RBF kernel is $0.011$. The posterior
fits each observation with a narrow spike and reverts to the prior mean a few
units away, and an optimizer built on it explores almost at random. Scale the
inputs, or set the lengthscale relative to each input's range.
:::

### Smoothness {#sec-fs-smoothness}

The lengthscale says how fast a function varies. Smoothness says how rough it
is at the smallest scales, and the two are independent: a function can wander
slowly and still be jagged up close, like a coastline seen from a plane.
Smoothness is set by the kernel's behavior very near zero distance. If $k(r)$
is smooth at $r = 0$, values at nearby inputs are almost perfectly correlated
and the draws are smooth. If $k(r)$ has a corner at $r = 0$, nearby values
decorrelate quickly and the draws are rough.

The RBF kernel is infinitely differentiable at zero, and its draws have
derivatives of every order (in the mean-square sense, the one the theory of
random processes uses) [@rasmussen2006gaussian, sec. 4.2.1]. That is a
strong assumption. The **Matérn** family, named by @stein1999interpolation
after earlier work of Matérn, relaxes it with a smoothness parameter $\nu$:
its draws are $q$ times differentiable exactly when $\nu > q$
[@rasmussen2006gaussian, sec. 4.2.1]. The values used in practice are half-integers, for which the kernel is
an exponential times a polynomial. The roughest, $\nu = 1/2$, is

$$
k(r) = \sigma_f^2 \exp\!\left(-\frac{|r|}{\ell}\right).
$$

Its corner at $r = 0$ produces draws that are continuous but nowhere
differentiable. They are the paths of the Ornstein-Uhlenbeck process, first
introduced as a model of the velocity of a particle buffeted by random
collisions [@rasmussen2006gaussian, sec. 4.2.1]. Then $\nu = 3/2$ gives once-differentiable
draws, $\nu = 5/2$ twice-differentiable ones, and as $\nu \to \infty$ the
Matérn kernel becomes the RBF. @sec-kernel-family writes out the family.

@eq-fs-upcrossings needs $k''(0)$, which the Matérn 1/2 kernel does not have.
Its draws do not cross zero a finite expected number of times: a draw that
crosses zero once crosses it infinitely often nearby
[@rasmussen2006gaussian, sec. 4.1]. That is why the readout in @fig-fs-prior
gives no expectation for this kernel, and why the count it reports depends on
how fine the grid is.

Which smoothness to assume is a real modeling choice. @stein1999interpolation
argued that the smoothness the RBF kernel assumes is unrealistic for many
physical processes and recommended the Matérn class instead
[@rasmussen2006gaussian, sec. 4.2.1]. For Bayesian optimization,
@snoek2012practical made the same argument: draws from the RBF kernel "are
unrealistically smooth for practical optimization problems", and they proposed
Matérn 5/2, whose twice-differentiable draws match the assumption made by
quasi-Newton methods, optimizers that estimate second derivatives from how
the gradient changes between steps. They also tested the choice. Tuning three
hyperparameters of a structured support vector machine that finds motifs in
protein DNA sequences, about 40,000 of them, they repeated the optimization
100 times with each of several kernels and found that the choice of kernel
"significantly affects performance": the RBF kernel's assumption of infinite
differentiability was "too restrictive for this problem"
[@snoek2012practical]. In the other direction, values of $\nu$ above
5/2 are hard to tell apart from each other and from the RBF using finite,
noisy data, which is one reason 3/2 and 5/2 are the values seen most often
[@rasmussen2006gaussian, sec. 4.2.1].

### Periodicity {#sec-fs-periodicity}

Some objectives repeat: an angle, a time of day, a hue on the color wheel. The
**periodic kernel** builds the repetition into the prior,

$$
k(x, x') = \sigma_f^2 \exp\!\left(-\frac{2\sin^2\!\left(\pi |x - x'| / p\right)}{\ell^2}\right),
$$ {#eq-fs-periodic}

with period $p$. A construction explains the formula. Map each input onto a
circle, $\mathbf{u}(x) = \left(\cos(2\pi x/p),\, \sin(2\pi x/p)\right)$, and
apply the RBF kernel to the mapped points. The squared distance between two
mapped points is $4\sin^2(\pi(x - x')/p)$, which turns the RBF formula into
@eq-fs-periodic [@rasmussen2006gaussian, sec. 4.2.3]. Inputs exactly one
period apart land on the same point of the circle, so their values have
correlation 1, and every draw repeats exactly.

Because the RBF kernel is applied on the circle, the lengthscale $\ell$ of the
periodic kernel is measured relative to the circle, not in the units of $x$.
Near zero distance the kernel behaves like an RBF kernel with lengthscale
$p\ell/(2\pi)$, so @eq-fs-upcrossings gives $1/(p\ell)$ upward crossings per
unit length. That is why @fig-fs-prior moves the lengthscale when you switch
to this kernel. The color-preference figure of @sec-pbo uses a periodic kernel
because hue wraps around.

### What a stationary kernel cannot say {#sec-fs-limits}

A kernel is a strong statement, and some beliefs cannot be expressed with the
kernels above. A stationary kernel cannot express a trend that continues
beyond the data, because far from the data its posterior returns to the
prior mean (@sec-gp-pitfalls). It cannot express an abrupt change, such as a setting
beyond which a training run diverges, because its draws are continuous; even
the Matérn 1/2 draws, rough as they are, have no jumps. And a single
lengthscale cannot express a function that is flat in one region and wiggly
in another. When these matter, practitioners add or multiply kernels
(@sec-kernel-family), transform the inputs before applying a kernel as the
periodic kernel does, or turn to models that are not stationary.

The prior also matters more in Bayesian optimization than in most regression.
A model fitted to thousands of points is shaped mostly by its data. An
optimizer that can afford twenty evaluations of a ten-dimensional function has
data almost nowhere, so the prior decides what it believes almost everywhere.
That is why the next two chapters spend so long on the kernel:
@sec-gp-regression shows how the prior and the data combine, and @sec-kernels
shows how to let the data choose the hyperparameters.

::: {.keyidea title="The kernel is the prior"}
Choosing a kernel and its hyperparameters is a statement about the objective
made before any evaluation. The lengthscale says how far an observation's
influence reaches, the amplitude how large the function is, and the kernel's
form how smooth or periodic each draw is.
:::

## Exercises {#sec-fs-exercises}

::: {.exercise #exr-fs-line}
Take the line features $\boldsymbol{\phi}(x) = (1, x)^\T$ with
$\mSigma_p = \diag(\sigma_0^2, \sigma_1^2)$, as in @ex-fs-line. (a) Write the
covariance matrix $\mK$ of $(f(0), f(1), f(2))$. (b) Find a nonzero vector
$\mathbf{a}$ with $\mK\mathbf{a} = \mathbf{0}$, and say what
$\Var[\mathbf{a}^\T\vf] = 0$ means for every line the prior can draw.

::: {.solution}
(a) With $k(x, x') = \sigma_0^2 + \sigma_1^2 x x'$,

$$
\mK = \begin{bmatrix}
\sigma_0^2 & \sigma_0^2 & \sigma_0^2 \\
\sigma_0^2 & \sigma_0^2 + \sigma_1^2 & \sigma_0^2 + 2\sigma_1^2 \\
\sigma_0^2 & \sigma_0^2 + 2\sigma_1^2 & \sigma_0^2 + 4\sigma_1^2
\end{bmatrix}.
$$

(b) Take $\mathbf{a} = (1, -2, 1)^\T$. Each row of $\mK\mathbf{a}$ vanishes:
the $\sigma_0^2$ terms sum to $\sigma_0^2(1 - 2 + 1) = 0$, and the
$\sigma_1^2$ terms in rows two and three are $\sigma_1^2(-2 + 2) = 0$ and
$\sigma_1^2(-4 + 4) = 0$. So
$\Var[f(0) - 2f(1) + f(2)] = \mathbf{a}^\T\mK\mathbf{a} = 0$: every line drawn
from this prior satisfies $f(0) - 2f(1) + f(2) = 0$ exactly, because the
second difference of a line is zero. The prior is certain that $f$ does not
bend. With two features, any three values satisfy one such exact constraint,
which is the rank deficiency noted in @ex-fs-line.
:::
:::

::: {.exercise #exr-fs-box}
@sec-linalg-impossible claimed that the "box" similarity, $k(x, x') = 1$ if
$|x - x'| < 1$ and $0$ otherwise, is not a valid kernel. Under it, values
closer than 1 apart are perfectly correlated, and values farther apart are
independent. Prove the claim with the inputs $0, 0.9, 1.8$ and the
coefficients $\mathbf{a} = (1, -1, 1)^\T$, and explain the contradiction in
words.

::: {.solution}
The matrix is
$\mK = \begin{bmatrix} 1 & 1 & 0 \\ 1 & 1 & 1 \\ 0 & 1 & 1 \end{bmatrix}$,
so $\mK\mathbf{a} = (0, 1, 0)^\T$ and $\mathbf{a}^\T\mK\mathbf{a} = -1$. The
combination $f(0) - f(0.9) + f(1.8)$ would have variance $-1$, which no random
quantity can have. In words: correlation 1 between two values of equal
variance forces them to be equal, so $f(0) = f(0.9)$ and $f(0.9) = f(1.8)$,
hence $f(0) = f(1.8)$, which contradicts their independence. Covariances
cannot be chosen pair by pair; they must be jointly consistent, and positive
semidefiniteness is the check.
:::
:::

::: {.exercise #exr-fs-sum}
(a) Let $f \sim \GP(0, k)$ and let $c$ be a constant. Show that
$c f \sim \GP(0, c^2 k)$. (b) Let $f_1 \sim \GP(0, k_1)$ and
$f_2 \sim \GP(0, k_2)$ be independent. Show that
$f_1 + f_2 \sim \GP(0, k_1 + k_2)$. (c) If $k_1$ and $k_2$ come from features
as in @eq-fs-induced-kernel, what model does $k_1 + k_2$ come from?

::: {.solution}
By @def-fs-gp it is enough to check every finite set of inputs.
(a) At inputs $\vx_1, \dots, \vx_n$, the values of $cf$ are $c$ times a vector
distributed as $\N(\mathbf{0}, \mK)$. A linear map of a Gaussian is Gaussian
(@sec-gaussian-linear), here $\N(\mathbf{0}, c^2\mK)$, whose entries are
$c^2 k(\vx_i, \vx_j)$. (b) The values of $f_1 + f_2$ are the sum of
independent vectors distributed as $\N(\mathbf{0}, \mK_1)$ and
$\N(\mathbf{0}, \mK_2)$, which is $\N(\mathbf{0}, \mK_1 + \mK_2)$
(@sec-gaussian-sums). (c) The linear model whose feature list is the two lists
concatenated, with independent weights for the two parts. Its weight
covariance is block-diagonal, and @eq-fs-induced-kernel splits into the two
inner products. Adding kernels pools features, which @sec-kernel-family uses to
build kernels for functions with several kinds of structure at once.
:::
:::

::: {.exercise #exr-fs-matern}
The Matérn 5/2 kernel with unit amplitude is
$k(r) = \left(1 + \sqrt{5}|r|/\ell + 5r^2/(3\ell^2)\right)\exp\!\left(-\sqrt{5}|r|/\ell\right)$.
Expand it to second order in $r$ around $0$ and use @eq-fs-upcrossings to find
the expected number of upward zero crossings on $[0, 1]$. Compare with the RBF
kernel at the same $\ell$, and check the result against the readout in
@fig-fs-prior.

::: {.solution}
Write $a = \sqrt{5}/\ell$ and take $r \ge 0$. Then
$e^{-ar} = 1 - ar + a^2r^2/2 + O(r^3)$, and multiplying by
$1 + ar + a^2r^2/3$ gives a constant term of 1, linear terms $ar - ar = 0$, and
quadratic terms $a^2r^2(1/3 - 1 + 1/2) = -a^2r^2/6$. So
$k(r) = 1 - 5r^2/(6\ell^2) + O(r^3)$ and $k''(0) = -5/(3\ell^2)$. By
@eq-fs-upcrossings,
$\E[N_0] = \sqrt{5/3}/(2\pi\ell) \approx 1.29/(2\pi\ell)$, about 29% more
crossings than the RBF kernel at the same lengthscale. At $\ell = 0.1$ that is
2.05 against 1.59, as the readout shows. The same $\ell$ therefore does not
mean the same wiggliness in different kernel families; comparing them fairly
takes a conversion like this one, or hyperparameters fitted to the data
(@sec-marginal-likelihood).
:::
:::

## Further reading {#further-reading .unnumbered}

- @rasmussen2006gaussian, chapter 2, sets the weight-space and function-space
  views side by side, with the kernel trick and the definition used here;
  chapter 4 catalogs kernels, including the bump construction of the RBF
  kernel, the zero-crossing formula, and the Matérn class.
- @garnett2023bayesian, chapters 2 and 3, develops Gaussian processes and the
  choice of kernel with Bayesian optimization in mind.
- @neal1996bayesian, chapter 2, shows that neural networks with random weights
  become Gaussian processes as they grow infinitely wide.
- @stein1999interpolation makes the case, from the theory of spatial
  prediction, that the smoothness a kernel assumes matters, and argues for the
  Matérn class.
- @gortler2019visual is an interactive essay in which the reader changes
  kernels and watches prior samples change, a companion to @fig-fs-prior.
