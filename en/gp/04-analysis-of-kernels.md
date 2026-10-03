---
status: done
synopsis: "The mathematics under the regret bounds: functions as vectors, the reproducing kernel Hilbert space a kernel defines and what a bound on its norm means, Mercer's eigen-expansion, Bochner's spectral view and random features, how eigenvalue decay sets the information gain, and what a Gaussian process is as a random object."
sources: ["Rasmussen and Williams 2006, ch. 4 and 6", "Kanagawa et al. 2018", "Vakili et al. 2021", "Srinivas et al. 2010", "Rahimi and Recht 2007", "Da Costa et al. 2026"]
---

# The Analysis Behind Kernels {#sec-kernel-analysis}

The three chapters before this one built a working model: a kernel says how
the values of an unknown function move together (@sec-function-space),
conditioning turns it into predictions (@sec-gp-regression), and the marginal
likelihood picks its settings (@sec-kernels). That is enough to run Bayesian
optimization. It is not enough to say why it works. The guarantees of
@sec-regret are of two kinds. The Bayesian ones hold for a function drawn
from the prior. The frequentist ones, and their counterparts for comparisons
in @sec-dueling-bandits and @sec-pbo-theory, hold for one fixed function
whose norm in a reproducing kernel Hilbert space is at most a number $B$. The
rates of both depend on how fast the eigenvalues of the kernel decay. These
statements treat a kernel not
as a recipe for covariance matrices but as an object in its own right: a
linear map on functions, with eigenvalues, a spectrum of frequencies, and a
space of functions it calls simple.

This chapter builds that view from what the reader already has: inner
products and eigenvectors (@sec-linear-algebra) and the information a
Gaussian model gains from noisy data (@sec-gp-information-gain). The plan is
to treat a function as a very long vector and to carry each finite fact over.
It ends where the results are used, with the growth of the maximum
information gain and with a precise account of what a Gaussian process is as
a random object. None of it is needed to use the methods of @sec-part-bo; it
is needed to read their guarantees.

## Functions as vectors {#sec-ka-functions}

A vector $\vx \in \R^n$ is a list of $n$ numbers, which is the same thing as a
function from the indices $\{1, \dots, n\}$ to $\R$: give it an index $i$,
and it returns $x_i$. A function $f$ on $[0, 1]$ is the same kind of object
with a continuum of indices. @sec-gp-prior-samples used this reading when it
drew a function as the vector of its values on a grid.

### Inner products of functions {#sec-ka-inner}

Take the grid of midpoints $x_i = (i - \tfrac12)/n$ and the vectors $\vf$ and
$\vg$ of two functions' values there. Their inner product $\vf^\T\vg$
(@eq-linalg-inner) grows with $n$, because it adds more terms. Divided by $n$
it settles, a Riemann sum turning into an integral as in @sec-fs-rbf-limit:

$$
\frac{1}{n}\sum_{i=1}^n f(x_i)\, g(x_i) \;\longrightarrow\; \int_0^1 f(x)\, g(x)\, \dd x .
$$

The limit is the **inner product of two functions**, and length,
$\lVert f \rVert = (\int_0^1 f^2\, \dd x)^{1/2}$, and orthogonality follow as in
@sec-linalg-inner-products. A weighting density $q(x) > 0$ can make some
inputs count more: $\langle f, g \rangle_q = \int f(x)\, g(x)\, q(x)\, \dd x$.
This chapter uses the uniform weighting $q = 1$ on $[0, 1]$ unless it says
otherwise.

### Orthonormal bases: Fourier as the example {#sec-ka-fourier}

An orthonormal basis gives vectors coordinates, and functions too. The
best-known basis on $[0, 1]$ is Fourier's: the constant $1$ and the functions
$\sqrt{2}\cos(2\pi jx)$ and $\sqrt{2}\sin(2\pi jx)$ for $j = 1, 2, \dots$,
each of length 1 and orthogonal to the others. The coordinates of $f$ are its
Fourier coefficients $a_j$ (cosine) and $b_j$ (sine), and its squared length
is the sum of their squares, Parseval's identity.

Coefficients carry smoothness. For a differentiable $f$ with $f(0) = f(1)$,
integrating by parts moves the derivative onto the basis function and brings
out a factor $1/(2\pi j)$: the cosine coefficient of $f$ is $-1/(2\pi j)$
times the sine coefficient of $f'$, and the sine coefficient of $f$ is
$1/(2\pi j)$ times the cosine coefficient of $f'$. Parseval's identity for
$f'$ then gives

$$
\int_0^1 f'(x)^2\, \dd x = \sum_{j \ge 1} (2\pi j)^2 \left(a_j^2 + b_j^2\right).
$$ {#eq-ka-parseval-derivative}

The energy of the slope, a natural measure of roughness, is a sum of squared
coefficients with weights that grow with frequency. High frequencies are
expensive and low ones cheap. The norm of @sec-ka-rkhs has this shape, with
weights chosen by the kernel.

### The kernel matrix as a view of an operator {#sec-ka-operator}

A matrix maps vectors to vectors. A kernel maps functions to functions:

$$
(\mathcal{K} g)(x) = \int k(x, x')\, g(x')\, q(x')\, \dd x' ,
$$ {#eq-ka-operator}

a combination of the values of $g$, weighted by how strongly $x$ is
correlated with each $x'$. On the grid the integral becomes
$\frac1n\sum_j k(x_i, x_j)\, g(x_j)$, the matrix-vector product
$\frac1n\mK\vg$. So $\mK/n$ is the operator $\mathcal{K}$ seen through $n$
points, and its eigenvalues approximate the operator's.

@sec-linalg-kernel-spectrum met one such matrix: the RBF kernel with
lengthscale 0.1 on 100 evenly spaced inputs, with largest eigenvalues 23.9,
21.2, and 17.5. Divided by 100 they are 0.239, 0.212, and 0.175. Those inputs
include both ends of $[0, 1]$. Placed instead at the 100 midpoints of
@sec-ka-inner, as in @fig-ka-mercer below, they give 0.241, 0.214, and 0.176,
and finer grids of midpoints leave these digits unchanged. The rapid decay
that broke the Cholesky factorization there belongs to the operator, not to
the grid.

## The space a kernel defines {#sec-ka-rkhs}

The frequentist regret theorems of @sec-regret, which bound how much an
optimizer loses against the best value it could have found for one fixed
function, need a class of functions large enough to contain realistic
objectives and small enough that finitely many evaluations can pin a member
down. A kernel defines one, and the construction starts from the weight-space
view of @sec-function-space.

### Functions built from bumps {#sec-ka-bumps}

Take a kernel that comes from features,
$k(\vx, \vx') = \boldsymbol{\phi}(\vx)^\T\boldsymbol{\phi}(\vx')$, as in
@eq-fs-induced-kernel with prior weight covariance $\mI$. Each weight vector
$\vw$ gives a function $\vw^\T\boldsymbol{\phi}(\vx)$, and the kernel bump
$k(\cdot, \vx')$ is the one with weights $\boldsymbol{\phi}(\vx')$. A sum of
bumps $f = \sum_i \alpha_i k(\cdot, \vx_i)$ therefore has weights
$\sum_i \alpha_i \boldsymbol{\phi}(\vx_i)$, and for a second sum
$g = \sum_j \beta_j k(\cdot, \vx'_j)$ the inner product of the two weight
vectors is

$$
\langle f, g \rangle_k = \sum_{i}\sum_{j} \alpha_i \beta_j\, k(\vx_i, \vx'_j).
$$ {#eq-ka-inner}

The right side mentions no features, so it serves as a definition for any
positive semidefinite kernel. It does not depend on how $f$ and $g$ are
written as sums: grouping by $j$ gives $\sum_j \beta_j f(\vx'_j)$ and grouping
by $i$ gives $\sum_i \alpha_i g(\vx_i)$, which depend only on the functions'
values. The squared norm of $f$ is
$\lVert f \rVert_k^2 = \bm{\alpha}^\T\mK\bm{\alpha} \ge 0$.

### The reproducing property {#sec-ka-reproducing}

With $g = k(\cdot, \vx)$, a single bump with coefficient 1, the same
grouping gives

$$
\langle f, k(\cdot, \vx) \rangle_k = f(\vx).
$$ {#eq-ka-reproducing}

Taking the inner product with a bump evaluates the function. This is the
**reproducing property**. With $f = k(\cdot, \vx')$ it gives
$\langle k(\cdot, \vx'), k(\cdot, \vx) \rangle_k = k(\vx, \vx')$: the bumps
are features for the kernel. The Cauchy-Schwarz inequality,
$\lvert\langle f, g\rangle_k\rvert \le \lVert f\rVert_k\lVert g\rVert_k$,
applied to @eq-ka-reproducing and to the difference of two bumps, gives two
bounds [@chowdhury2017kernelized]:

$$
\lvert f(\vx) \rvert \le \lVert f \rVert_k \sqrt{k(\vx, \vx)},
\qquad
\lvert f(\vx) - f(\vx') \rvert \le \lVert f \rVert_k \sqrt{k(\vx, \vx) - 2k(\vx, \vx') + k(\vx', \vx')}.
$$ {#eq-ka-bounds}

They say what the norm controls. Under a kernel with $k(\vx, \vx) \le 1$, a
function of norm at most $B$ never exceeds $B$ in absolute value. And it
cannot change quickly: for the RBF kernel the second square root is
$\sqrt{2 - 2e^{-r^2/2\ell^2}}$ at distance $r$, which is 0.0999 at $r = 0.01$
when $\ell = 0.1$, close to $r/\ell$. Over a distance short compared with
the lengthscale, such a function changes by at most about $B\,r/\ell$. Large
values and fast changes cost norm.

The first bound also shows that only the zero function has norm zero, so
$\lVert\cdot\rVert_k$ is a genuine length, and that sums of bumps converging
in this norm converge at every input, so their limits are functions too.
Adding those limits completes the construction.

::: {.definition #def-ka-rkhs title="Reproducing kernel Hilbert space"}
Let $k$ be a symmetric positive semidefinite kernel on $\X$. Its
**reproducing kernel Hilbert space** (RKHS) $\mathcal{H}_k$ is the space of
functions obtained by completing the sums of bumps under the norm of
@eq-ka-inner. It is the unique Hilbert space of functions (an inner-product
space in which every sequence whose terms come arbitrarily close to one
another has a limit in the space) that contains every bump $k(\cdot, \vx)$
and in which @eq-ka-reproducing holds for every member and every input
[@aronszajn1950theory; @rasmussen2006gaussian, thm. 6.1].
:::

For familiar kernels the space is recognizable. The line kernel $k(x, x') =
xx'$ (@ex-fs-line with $\sigma_0 = 0$, $\sigma_1 = 1$) has the lines
$f(x) = wx$ as its RKHS, with norm $\lvert w \rvert$, the slope. For a kernel
from finitely many features with weight covariance $\mI$, the norm of $f$ is
the length of the shortest weight vector that produces it
[@steinwart2008support, ch. 4]. The RKHS of a Matérn kernel with smoothness
$\nu$ on a bounded domain in $d$ dimensions holds the same functions as a
Sobolev space, those whose derivatives up to order $\nu + d/2$ are
square-integrable, with an equivalent norm, when $\nu + d/2$ is a whole number
and the boundary is regular [@kanagawa2018gaussian, ex. 2.6]. The RBF
kernel's RKHS holds only functions whose Fourier transforms decay
exponentially fast, so they are extremely smooth
[@kanagawa2018gaussian, ex. 2.7].

### The posterior mean lives in the space {#sec-ka-representer}

The posterior mean of Gaussian process regression is a weighted sum of
bumps, one per observation (@eq-gp-representer), so it lies in
$\mathcal{H}_k$. It also solves a problem stated without probability.

::: {.derivation title="The representer theorem for squared error"}
Find the $f \in \mathcal{H}_k$ that minimizes
$L(f) = \sum_{i=1}^n \big(y_i - f(\vx_i)\big)^2 + \sigma_n^2 \lVert f \rVert_k^2$.

1. Let $S$ be the span of $k(\cdot, \vx_1), \dots, k(\cdot, \vx_n)$, and write
   $f = f_S + f_\perp$ with $f_\perp$ orthogonal to every bump in $S$.
2. By @eq-ka-reproducing, $f(\vx_i) = \langle f_S + f_\perp, k(\cdot, \vx_i)\rangle_k = f_S(\vx_i)$.
   The data term sees only $f_S$.
3. By Pythagoras, $\lVert f\rVert_k^2 = \lVert f_S\rVert_k^2 + \lVert f_\perp\rVert_k^2$,
   so dropping $f_\perp$ lowers the penalty and the minimizer lies in $S$:
   $f = \sum_j \alpha_j k(\cdot, \vx_j)$.
4. Then the fitted values are $\mK\bm{\alpha}$ and
   $L = \lVert\vy - \mK\bm{\alpha}\rVert^2 + \sigma_n^2\bm{\alpha}^\T\mK\bm{\alpha}$.
5. The gradient, $-2\mK\big(\vy - (\mK + \sigma_n^2\mI)\bm{\alpha}\big)$,
   vanishes at $\bm{\alpha} = (\mK + \sigma_n^2\mI)^{-1}\vy$, so
   $f(\vx) = \vk(\vx)^\T(\mK + \sigma_n^2\mI)^{-1}\vy$: the posterior mean of
   @eq-gp-noisy.
:::

That a penalized fit over an infinite-dimensional space has a solution with
$n$ terms is the **representer theorem**, first stated for squared error by
@kimeldorf1971some; the match with the Gaussian process posterior mean goes
back to @kimeldorf1970correspondence [see also @kanagawa2018gaussian,
prop. 3.6]. With the data term a plain sum, as here, the penalty weight is the
noise variance, the kernel ridge regression of @sec-gp-bumps. In weight space
the match is expected: with $\vw \sim \N(\mathbf{0}, \mI)$, minus twice the
log posterior is $\sigma_n^{-2}\sum_i(y_i - f(\vx_i))^2 + \lVert\vw\rVert^2$
up to a constant, and the shortest $\vw$ for $f$ has length
$\lVert f\rVert_k$. The squared norm plays the part of minus twice the log
prior.

The posterior mean of @fig-gp-posterior at its defaults (lengthscale 0.12,
four observations, the largest 0.85 in absolute value) has
$\lVert\mu\rVert_k^2 = \bm{\alpha}^\T\mK\bm{\alpha} = 1.39$, so
@eq-ka-bounds caps it at 1.18, above its actual maximum: the norm is a
guarantee, not a description.

### What a norm bound assumes {#sec-ka-norm-bound}

The frequentist regret theorems (@sec-regret-other-settings,
@sec-kernelized-dueling, @sec-pbo-theory), in which $f$ is one fixed function
and the only randomness is the evaluation noise, assume that $f$ lies in
$\mathcal{H}_k$ with a known bound on its norm. By the sections above, if
$\lVert f\rVert_k \le B$ and $k(\vx, \vx) \le 1$, then $f$ is no larger than
$B$ anywhere, changes by at most about $B$ per lengthscale, and puts little
weight where the kernel says weight is expensive. The assumption depends on
the kernel and its lengthscale, not only on $f$: a longer lengthscale makes
fast changes more expensive, and under the RBF kernel a Gaussian bump of
width $\ell/\sqrt{2} \approx 0.71\,\ell$ or less has infinite norm
(@exr-ka-narrow-bump).

Two conventions are in use, and they differ by a square.
@srinivas2010gaussian assume $\lVert f \rVert_k^2 \le B$, as
@sec-regret-other-settings does; @chowdhury2017kernelized and the preference
papers of @sec-kernelized-dueling and @sec-pbo-theory assume
$\lVert f\rVert_k \le B$. In the second convention the bound enters the
confidence width directly. For noise that is $R$-sub-Gaussian (tails no
heavier than those of a Gaussian with standard deviation $R$),
Chowdhury and Gopalan show that with probability at least $1 - \delta$,
$\lvert\mu_{t-1}(\vx) - f(\vx)\rvert \le \beta_t^{1/2}\sigma_{t-1}(\vx)$
for all $\vx$ and all rounds $t \le T$, with
$\beta_t^{1/2} = B + R\sqrt{2(\gamma_{t-1} + 1 + \log(1/\delta))}$
[@chowdhury2017kernelized, thm. 2]; their posterior and their
$\gamma_{t-1}$ use the noise variance $1 + 2/T$ in place of $\sigma_n^2$, and
their $\beta_t$ is the square root of the book's. @sec-regret explains where
such widths come from. In practice $B$ is unknown, and the kernel that sets its
units is fitted from the same data (@sec-regret-bounds-not-say).

### Samples are rougher than the space {#sec-ka-samples}

The Bayesian theorem of @sec-regret-gp-setting assumes instead that $f$ is a
draw from $\GP(0, k)$. It is natural to guess that a typical draw has a
moderate norm in $\mathcal{H}_k$. It has none at all.

::: {.derivation title="A draw from the prior is almost never in its RKHS"}
Let $\vx_1, \vx_2, \dots$ be distinct inputs whose kernel matrices $\mK_n$ are
all invertible; for the RBF and Matérn kernels any distinct inputs qualify
(@sec-ka-bochner-theorem).

1. *Interpolation bound.* For $g \in \mathcal{H}_k$ with values $\vg_n$ at
   $\vx_1, \dots, \vx_n$, steps 1 to 3 above (with no data term) show that its
   part $g_S$ in the span of the first $n$ bumps has the same values and no
   larger norm. With $g_S = \sum_j \alpha_j k(\cdot, \vx_j)$ and
   $\mK_n\bm{\alpha} = \vg_n$, this gives
   $\lVert g\rVert_k^2 \ge \bm{\alpha}^\T\mK_n\bm{\alpha} = \vg_n^\T\mK_n^{-1}\vg_n$.
2. *The same quantity for a draw.* Write the draw's values as
   $\vf_n = \mL_n\vz$, with $\mL_n$ the Cholesky factor of $\mK_n$ and $\vz$
   standard normal (@sec-gauss-sampling). Then
   $Q_n = \vf_n^\T\mK_n^{-1}\vf_n = \vz^\T\vz = z_1^2 + \dots + z_n^2$.
3. *Nesting.* Adding $\vx_{n+1}$ appends a row to $\mL_n$ without changing it,
   so the first $n$ entries of $\vz$ stay put and $Q_{n+1} = Q_n + z_{n+1}^2$.
4. *No bound.* $Q_n$ has mean $n$ and standard deviation $\sqrt{2n}$, so for
   any fixed $c$ the probability that $Q_n \le c$ tends to zero. Since $Q_n$
   only grows, the probability that it stays below $c$ for every $n$ is zero.
5. A draw in $\mathcal{H}_k$ would have $Q_n \le \lVert f\rVert_k^2$ for every
   $n$ by step 1, so for some whole number $c$ it would have $Q_n \le c$ for
   every $n$. The norm differs from draw to draw, so step 4, which is about a
   fixed $c$, does not apply to it directly. But for each of the countably
   many $c = 1, 2, 3, \dots$ the event has probability zero by step 4, and so
   does their union, since the probability of a union is at most the sum of
   the probabilities (@eq-regret-union, with countably many events).
:::

The general statement is a zero-one law: a Gaussian process lies in a given
RKHS with probability 0 or 1, and in the RKHS of its own kernel with
probability 0 whenever that space is infinite-dimensional
[@driscoll1973reproducing; @lukic2001stochastic; @kanagawa2018gaussian,
thm. 4.9 and cor. 4.10]. Read through $n$ inputs, a draw looks like a
function of squared norm about $n$, and each new input adds about 1. The posterior mean is a
finite sum of bumps, and averaging removes the roughness
[@rasmussen2006gaussian, sec. 6.1].

Draws do lie in slightly larger spaces of rougher functions. For the RBF
kernel the difference rarely matters [@kanagawa2018gaussian, cor. 4.13 and
remark 4.13]. For a Matérn kernel it does: the RKHS asks for Sobolev
smoothness $\nu + d/2$, while draws have every order below $\nu$ and no more,
rougher by $d/2$ [@kanagawa2018gaussian, cor. 4.15 and remarks 4.14 and 4.15].

So the two settings of @sec-regret assume different things. The Bayesian
theorem is about draws, which no bound $B$ covers; the frequentist theorems
are about members of $\mathcal{H}_k$, a set to which the prior gives
probability zero. Neither contains the other, as @srinivas2010gaussian note.
Choosing a Matérn 5/2 kernel in $d$ dimensions and then quoting a frequentist
bound assumes smoothness $5/2 + d/2$, more than the draws of the same prior
have (inference).

## Mercer's theorem {#sec-ka-mercer}

Bumps overlap and are not orthogonal, so they make awkward coordinates. For a
symmetric matrix, @sec-linalg-spectral found better ones, the eigenvectors.
The operator $\mathcal{K}$ is the continuous version of a symmetric matrix,
and the same move works.

### Eigenfunctions and the expansion of the kernel {#sec-ka-mercer-theorem}

An **eigenfunction** of $\mathcal{K}$ is a function $\varphi$ that the
operator only rescales, $\mathcal{K}\varphi = \lambda\varphi$, with
eigenvalue $\lambda$. (This chapter writes $\varphi_i$ for eigenfunctions, to
keep them apart from the features $\boldsymbol{\phi}$ of
@sec-function-space and the normal density $\phi$.)

::: {.theorem #thm-ka-mercer title="Mercer's theorem"}
Let $\X$ be a closed and bounded subset of $\R^d$, $k$ a continuous, symmetric,
positive semidefinite kernel on $\X$, and $q$ a weighting that gives positive
weight to every open region of $\X$ (a density with $q > 0$, or more generally
a finite measure whose support is $\X$). Then $\mathcal{K}$ has eigenvalues
$\lambda_1 \ge \lambda_2 \ge \dots > 0$, finitely or countably many, with
eigenfunctions $\varphi_1, \varphi_2, \dots$ orthonormal under
$\langle\cdot,\cdot\rangle_q$, and

$$
k(\vx, \vx') = \sum_{i} \lambda_i\, \varphi_i(\vx)\, \varphi_i(\vx')
$$ {#eq-ka-mercer}

for all $\vx, \vx' \in \X$, the series converging absolutely and uniformly
[@mercer1909functions; @steinwart2008support, thm. 4.49;
@kanagawa2018gaussian, thm. 4.1].
:::

@eq-ka-mercer is the spectral theorem @thm-linalg-spectral,
$\mA = \sum_i \lambda_i\mathbf{u}_i\mathbf{u}_i^\T$, with functions in place of
vectors. The conditions matter: where the weighting gives no weight, the
expansion can fail [@kanagawa2018gaussian, remark 4.2]. The eigenvalues and
eigenfunctions depend on the weighting, while the kernel and its RKHS do not
[@kanagawa2018gaussian, remarks 4.1 and 4.3]. Three consequences follow.

**The eigenvalues are a variance budget.** Setting $\vx' = \vx$ in
@eq-ka-mercer and integrating against $q$ gives, by orthonormality,
$\sum_i \lambda_i = \int k(\vx, \vx)\, q(\vx)\, \dd\vx$. For a stationary
kernel and a probability density $q$, the eigenvalues sum to $\sigma_f^2$,
the prior variance, and say how it is shared among directions.

**Every kernel is an inner product of features.** With
$\boldsymbol{\phi}(\vx) = (\sqrt{\lambda_1}\varphi_1(\vx), \sqrt{\lambda_2}\varphi_2(\vx), \dots)$,
@eq-ka-mercer reads $k(\vx, \vx') = \boldsymbol{\phi}(\vx)^\T\boldsymbol{\phi}(\vx')$.
This is the converse quoted in @sec-fs-valid-kernels.

**The RKHS norm in eigen-coordinates.** Expand $f = \sum_i c_i\varphi_i$ with
$c_i = \langle f, \varphi_i\rangle_q$. Then $f \in \mathcal{H}_k$ exactly
when the following sum is finite, and

$$
\lVert f \rVert_k^2 = \sum_i \frac{c_i^2}{\lambda_i}
$$ {#eq-ka-norm-eigen}

[@rasmussen2006gaussian, sec. 6.1; @kanagawa2018gaussian, thm. 4.2]. The
reproducing property checks it: by @eq-ka-mercer the bump $k(\cdot, \vx)$ has
coefficients $\lambda_i\varphi_i(\vx)$, so
$\langle f, k(\cdot,\vx)\rangle_k = \sum_i c_i\lambda_i\varphi_i(\vx)/\lambda_i = f(\vx)$.
This is @eq-ka-parseval-derivative with weights $1/\lambda_i$ chosen by the
kernel, and the continuous form of $\vf^\T\mK^{-1}\vf$ written in the
eigenvectors of $\mK$ [@rasmussen2006gaussian, sec. 6.1]. A direction with a
small eigenvalue is expensive: a coefficient $c$ along it costs
$c^2/\lambda_i$.

### The eigen-expansion of a sample {#sec-ka-karhunen-loeve}

The same coordinates describe the prior. With independent standard normal
numbers $z_1, z_2, \dots$, set

$$
f(\vx) = \sum_i \sqrt{\lambda_i}\, z_i\, \varphi_i(\vx).
$$ {#eq-ka-kl}

Then $\E[f(\vx)f(\vx')] = \sum_{i,j}\sqrt{\lambda_i\lambda_j}\,\E[z_iz_j]\,\varphi_i(\vx)\varphi_j(\vx') = \sum_i\lambda_i\varphi_i(\vx)\varphi_i(\vx') = k(\vx, \vx')$,
since $\E[z_iz_j]$ is 1 for $i = j$ and 0 otherwise. So @eq-ka-kl is a draw
from $\GP(0, k)$ written in the eigenbasis, with independent coefficients of
variance $\lambda_i$. This is the **Karhunen-Loève expansion**; the series
converges in mean square, uniformly over the domain
[@kanagawa2018gaussian, thm. 4.3; @berlinet2004reproducing, sec. 2.3]. By
orthonormality, keeping the first $m$ terms leaves an average squared error
of

$$
\E\!\left[\int \big(f(\vx) - f_m(\vx)\big)^2 q(\vx)\, \dd\vx\right] = \sum_{i > m} \lambda_i ,
$$ {#eq-ka-kl-error}

the eigenvalue mass left out.

The two readings of a small eigenvalue agree: the prior gives the direction
little variance, $\lambda_i$, and the norm charges it heavily, $1/\lambda_i$.
By @eq-ka-norm-eigen the truncated draw has
$\lVert f_m\rVert_k^2 = \sum_{i \le m} z_i^2$, about $m$, the picture of
@sec-ka-samples in eigen-coordinates. This last step is intuition rather than
proof, because the full series converges in mean square and not in the norm
[@kanagawa2018gaussian, remark 4.9]; @sec-ka-samples gave the proof.

### Computing them {#sec-ka-mercer-figure}

Eigenfunctions rarely have a closed form. They are computed as
@sec-ka-operator suggested: on a grid of $n$ points with uniform weighting,
the eigenvalues of $\mK/n$ estimate the $\lambda_i$, and the eigenvectors
times $\sqrt{n}$ estimate the $\varphi_i$ at the grid points. This is the
Nyström method; it estimates the larger eigenvalues better than the smaller
ones [@rasmussen2006gaussian, sec. 4.3.2].

```{figure}
//| figure: ka-mercer
//| label: fig-ka-mercer
//| fig-cap: "Mercer's theorem computed. Top left: the eigenvalues $\lambda_i$ of four kernels with unit amplitude on $[0, 1]$ under the uniform weighting, on log axes, from the Nyström method on 100 grid midpoints [@rasmussen2006gaussian, sec. 4.3.2]; values below $10^{-14}$ are rounding error and are not drawn. Top right: the first four eigenfunctions of the chosen kernel. Bottom: a draw built from the first $m$ terms of the Karhunen-Loève expansion @eq-ka-kl (solid), against the draw from all 100 terms with the same random numbers, an exact draw of the process on the grid (dashed). The readout gives the share of the prior variance in the first $m$ terms and the squared RKHS norm $\sum_{i \le m} z_i^2$ of the truncated draw. Press *Draw again* for new random numbers."
```

**Read the default.** The RBF kernel with lengthscale 0.1 has largest
eigenvalues 0.241, 0.214, and 0.176, the values of @sec-ka-operator; all of
its eigenvalues together sum to 1.00, the prior variance, and they reach the
rounding floor by the 32nd. The
first four eigenfunctions look like cosines with 0, 1, 2, and 3 sign changes.

**Move the terms slider.** One term holds 24% of the prior variance and ten
hold 99.5%; the ten-term draw is hard to tell from the exact one. The squared
norm of the truncated draw keeps growing, 8.7 with ten terms and 96.5 with
all 100, though the draw barely changes.

**Switch to Matérn 1/2.** The eigenvalues fall on a straight line on log
axes, a power law. Ten terms hold 80% of the variance and twenty about 90%.
The truncated draw is smooth and the exact one jagged: the roughness lives in
the many small eigenvalues.

**Compare the slopes.** The Matérn lines have slopes near $-2$, $-4$, and
$-6$ for $\nu = 1/2$, $3/2$, and $5/2$; on a grid of 3,000 points, between the
20th and 60th eigenvalues, they are $-2.03$, $-4.00$, and $-5.89$. The
eigenvalues decay like $i^{-(2\nu + 1)}$, in line with the result of Ritter and
colleagues that a process with $r$ mean-square derivatives on $[0, 1]$ has
eigenvalues decaying like $i^{-(2r + 2)}$ [@rasmussen2006gaussian, sec. 4.3].

**Lengthen the lengthscale to 0.3.** The first RBF eigenvalue rises to 0.590
and four terms hold 99.6% of the variance: a longer lengthscale concentrates
the prior on fewer directions.

## Bochner's theorem {#sec-ka-bochner}

Mercer's eigenfunctions depend on the domain and the weighting, and must be
computed. For a stationary kernel, which depends only on
$\mathbf{r} = \vx - \vx'$ (@sec-fs-stationarity), there is a description that
needs neither: a list of frequencies and the variance each carries.

### Kernels as spectra {#sec-ka-bochner-theorem}

Start from one cosine with a random amplitude. If
$f(x) = a\cos(\omega x) + b\sin(\omega x)$ with $a, b$ independent standard
normals, then
$\Cov[f(x), f(x')] = \cos\omega x\cos\omega x' + \sin\omega x\sin\omega x' = \cos\big(\omega(x - x')\big)$,
a stationary kernel. Mixtures over frequencies give more, and Bochner's
theorem says they give all.

::: {.theorem #thm-ka-bochner title="Bochner's theorem"}
A continuous function $k$ on $\R^d$ is a stationary kernel, meaning that
$k(\vx - \vx')$ is positive semidefinite, exactly when
$k(\mathbf{r}) = \int e^{i\boldsymbol{\omega}^\T\mathbf{r}}\, \Lambda(\dd\boldsymbol{\omega})$
for a finite nonnegative measure $\Lambda$ over frequencies
$\boldsymbol{\omega}$ [@bochner1933monotone; @rasmussen2006gaussian, thm. 4.1].
:::

For the kernels of this book $\Lambda$ has a density $s(\boldsymbol{\omega})$,
the **spectral density**, symmetric in $\boldsymbol{\omega}$, and
$k(\mathbf{r}) = \int s(\boldsymbol{\omega})\cos(\boldsymbol{\omega}^\T\mathbf{r})\,\dd\boldsymbol{\omega}$.
Frequencies here are in radians per unit of input; Rasmussen and Williams use
cycles, $\boldsymbol{\omega} = 2\pi\mathbf{s}$, which rescales the density but
not its shape [@rasmussen2006gaussian, eq. 4.6]. Since $\int s = k(\mathbf{0}) = \sigma_f^2$,
$p = s/\sigma_f^2$ is a probability density, and

$$
k(\mathbf{r}) = \sigma_f^2\, \E_{\boldsymbol{\omega} \sim p}\!\left[\cos(\boldsymbol{\omega}^\T\mathbf{r})\right].
$$ {#eq-ka-bochner}

One direction of the theorem is short. For inputs $\vx_1, \dots, \vx_n$ and
coefficients $a_i$, the identity
$\cos(\theta - \theta') = \cos\theta\cos\theta' + \sin\theta\sin\theta'$ gives

$$
\sum_{i,j} a_ia_j\, k(\vx_i - \vx_j) = \int s(\boldsymbol{\omega})\left[\Big(\sum_i a_i\cos\boldsymbol{\omega}^\T\vx_i\Big)^2 + \Big(\sum_i a_i\sin\boldsymbol{\omega}^\T\vx_i\Big)^2\right]\dd\boldsymbol{\omega} \;\ge\; 0 .
$$

If $s$ is positive at every frequency, as for the RBF and Matérn kernels, the
integral is positive unless all $a_i$ are zero, because for distinct inputs
the two sums cannot vanish together at every frequency. Every kernel matrix of
distinct inputs is then invertible, which @sec-ka-samples used
[@wendland2004scattered, ch. 6]. The converse, that every stationary kernel
has such a spectrum, is the deep part.

### The spectra of the RBF and Matérn kernels {#sec-ka-spectra}

For the RBF kernel the frequencies are Gaussian, $p = \N(\mathbf{0}, \ell^{-2}\mI)$,
so a typical frequency is about $1/\ell$ (@exr-ka-gauss-spectrum). For the
Matérn kernel they follow a Student-t distribution with $2\nu$ degrees of
freedom and scale $1/\ell$,

$$
p(\boldsymbol{\omega}) \propto \left(1 + \frac{\ell^2\lVert\boldsymbol{\omega}\rVert^2}{2\nu}\right)^{-(\nu + d/2)},
$$ {#eq-ka-matern-spectrum}

which is eq. 4.15 of @rasmussen2006gaussian in radians. For $\nu = 1/2$ in one
dimension it is the Cauchy distribution, $p(\omega) = (\ell/\pi)/(1 + \ell^2\omega^2)$.

The tails differ. The Gaussian falls faster than any power, while
@eq-ka-matern-spectrum falls like $\lVert\boldsymbol{\omega}\rVert^{-(2\nu + d)}$,
keeping a little variance at every high frequency, more for smaller $\nu$.
The tails set the smoothness. Differentiating @eq-ka-bochner twice at
$\mathbf{r} = 0$ in one dimension gives
$-k''(0) = \sigma_f^2\int\omega^2 p(\omega)\,\dd\omega$, the variance of the
mean-square derivative $f'(x)$ (@sec-fs-smoothness). It is finite only if the
tail falls faster than $\lvert\omega\rvert^{-3}$, which for the Matérn kernel
means $2\nu + 1 > 3$, or $\nu > 1$. With higher even powers of $\omega$ in
place of $\omega^2$ the same argument gives the rule of @tbl-kern-family: a
mean-square derivative of every whole order below $\nu$, and none of order
$\nu$ or above. And $\sqrt{-k''(0)/k(0)}$, the root-mean-square
frequency, is what @eq-fs-upcrossings counts: for Matérn 5/2 the Student-t
with 5 degrees of freedom has variance $5/3$ in units of $1/\ell^2$, the
$k''(0) = -5/(3\ell^2)$ of @exr-fs-matern.

Mercer and Bochner describe the same kernel, and on a domain without ends
they coincide. Join the ends of $[0, 1]$ into a circle and wrap the kernel
around it, $k_\circ(r) = \sum_m k(r + m)$ over all integers $m$. Integrating
$k_\circ(x - y)\cos(2\pi jy)$ over one turn equals integrating
$k(x - y)\cos(2\pi jy)$ over the whole line: substituting $u = y - m$ turns
the integral of the term $k(x - y + m)\cos(2\pi jy)$ over $[0, 1]$ into the
integral of $k(x - u)\cos(2\pi ju)$ over $[-m, 1 - m]$, since the cosine
repeats with period 1, and these intervals cover the line. With $r = x - y$,
the cosine splits as $\cos(2\pi jx)\cos(2\pi jr) + \sin(2\pi jx)\sin(2\pi jr)$.
The sine part integrates to zero because $k$ is even, and the cosine part
gives $2\pi s(2\pi j)\cos(2\pi jx)$, because the spectral density is recovered
from the kernel by $s(\omega) = \frac{1}{2\pi}\int k(r)\cos(\omega r)\,\dd r$
[@rasmussen2006gaussian, eq. 4.6]. Sines work the same way. So the eigenfunctions
on the circle are Fourier's, and the eigenvalues are
$\lambda_j = 2\pi s(2\pi j)$: the spectral density sampled at the frequencies
that fit around the circle. Eigenvalue decay is spectral decay. Since the
$i$th eigenvalue sits near frequency $\pi i$, a Matérn spectrum falling like
$\lvert\omega\rvert^{-(2\nu+1)}$ gives eigenvalues falling like
$i^{-(2\nu + 1)}$, the slopes of @fig-ka-mercer. The interval's ends change
the values but not the tail: at lengthscale 0.1 the 80th Matérn 1/2
eigenvalue is $3.24 \times 10^{-4}$ on the interval and $3.16 \times 10^{-4}$
on the circle.

### Random Fourier features {#sec-ka-rff}

@eq-ka-bochner turns a kernel into an expectation, which an average can
estimate. Draw frequencies $\boldsymbol{\omega}_1, \dots, \boldsymbol{\omega}_M$
from $p$ and use the $2M$ features

$$
\mathbf{z}(\vx) = \frac{\sigma_f}{\sqrt{M}}\big(\cos\boldsymbol{\omega}_1^\T\vx, \dots, \cos\boldsymbol{\omega}_M^\T\vx, \sin\boldsymbol{\omega}_1^\T\vx, \dots, \sin\boldsymbol{\omega}_M^\T\vx\big).
$$ {#eq-ka-rff}

By the cosine identity,
$\mathbf{z}(\vx)^\T\mathbf{z}(\vx') = \frac{\sigma_f^2}{M}\sum_{j}\cos\boldsymbol{\omega}_j^\T(\vx - \vx')$,
an average whose expectation is $k(\vx - \vx')$. These are the **random
Fourier features** of @rahimi2007random. With $\sigma_f = 1$ each term lies in
$[-1, 1]$ and has variance $\tfrac12(1 + k(2r)) - k(r)^2$, which tends to
$\tfrac12$ at large distances (@exr-ka-rff), so the error at one distance is
typically about $1/\sqrt{2M}$. Accuracy $\varepsilon$ at all pairs of inputs
in a bounded domain at once needs $M$ of order
$(d/\varepsilon^2)\log(1/\varepsilon)$, with the domain's diameter and the
spread of $p$ entering through the logarithm [@rahimi2007random, claim 1].

A weighted sum $\mathbf{w}^\T\mathbf{z}(\vx)$ with
$\mathbf{w} \sim \N(\mathbf{0}, \mI)$ is the linear model of
@eq-fs-linear-model with $2M$ features: an approximate draw from
$\GP(0, k)$ that is a formula, costs $O(Md)$ per input, and can be maximized
like any function. This is one way for Thompson sampling, which evaluates where a random draw
from the posterior is largest (@sec-thompson), to draw whole functions on a
continuous domain, as @sec-gp-posterior-samples
mentioned [@rahimi2007random; @wilson2020efficiently].

```{figure}
//| figure: ka-bochner
//| label: fig-ka-bochner
//| fig-cap: "Bochner's theorem and random Fourier features, for kernels with unit amplitude. Top left: the spectral density $p(\omega)$ of the chosen kernel relative to its peak, on a log scale, against the scaled frequency $\omega\ell$, with the RBF density dashed for comparison; ticks along the bottom mark the frequencies drawn (up to 200, those beyond the axis piled at its end). Top right: the kernel $k(r)$ (dashed) and its estimate from $M$ random features, the average of $\cos(\omega_j r)$ (solid). Bottom: a draw built from the $M$ features with standard normal weights, against an exact draw from the same kernel by the Cholesky method of @sec-gauss-sampling. The readout gives the root-mean-square gap between estimate and kernel over distances in $[0, 1]$. Raising $M$ adds frequencies without replacing earlier ones; *Draw new frequencies* gives another set."
```

**Read the default.** The Matérn 3/2 density falls like a power and is
still near $10^{-3}$ of its peak at $\omega\ell = 10$, where the RBF density
has vanished. With $M = 20$ the estimated kernel wanders around the true one,
with a root-mean-square gap of 0.152 against $1/\sqrt{2M} = 0.158$.

**Raise M to 200, then 2000.** The gap falls to 0.058, then 0.023, against
0.050 and 0.016: roughly the $1/\sqrt{M}$ rate of an average, a factor of
about three per tenfold increase in $M$.

**Switch to Matérn 1/2 at M = 20.** The heavy Cauchy tail puts a few of the
twenty frequencies far out, and they show as a regular ripple, in the kernel
estimate and in the feature draw, which is otherwise smooth. The exact draw is
rough at every scale. A heavy-tailed spectrum spreads its roughness over many
high frequencies, each with little variance, and twenty features sample only
a few of them.

**Switch to RBF.** The frequencies all lie within a few multiples of
$1/\ell$, and the feature draw has the character of the exact one at small
$M$, because RBF draws are made of such frequencies. The kernel estimate is no
more accurate than before: its gap at $M = 20$ is 0.161.

## From eigenvalues to information gain {#sec-ka-infogain}

The regret bounds of @sec-regret (bounds on an optimizer's total shortfall
from the best value) depend on the maximum information gain $\gamma_T$
(@def-regret-gamma), the most that $T$ noisy evaluations could reveal about
$f$. @sec-gp-information-gain quoted its growth for the RBF and Matérn
kernels. The rates come from the eigenvalues.

### Counting resolved directions {#sec-ka-resolved}

Write the kernel matrix of $T$ inputs through @eq-ka-mercer as
$\mK_A = \boldsymbol{\Phi}\boldsymbol{\Lambda}\boldsymbol{\Phi}^\T$, where
row $t$ of $\boldsymbol{\Phi}$ holds the eigenfunctions at $\vx_t$ and
$\boldsymbol{\Lambda}$ holds the eigenvalues on its diagonal. The determinant
lemma (@eq-id-det-lemma) moves the information gain @eq-info-gp-gain from the
evaluations to the eigen-directions:
$\det(\mI + \sigma_n^{-2}\boldsymbol{\Phi}\boldsymbol{\Lambda}\boldsymbol{\Phi}^\T) = \det(\mI + \sigma_n^{-2}\boldsymbol{\Lambda}^{1/2}\boldsymbol{\Phi}^\T\boldsymbol{\Phi}\boldsymbol{\Lambda}^{1/2})$.
If the inputs are spread in proportion to $q$, then
$\frac1T\sum_t\varphi_i(\vx_t)\varphi_j(\vx_t)$ approximates
$\int\varphi_i\varphi_j\, q$, which is 1 or 0, as the Riemann sum of
@sec-ka-inner did, so $\boldsymbol{\Phi}^\T\boldsymbol{\Phi} \approx T\mI$ and

$$
I(\vy_A; f) \approx \frac12\sum_i \log\!\left(1 + \frac{T\lambda_i}{\sigma_n^2}\right).
$$ {#eq-ka-info-eigen}

Each eigen-direction is a separate experiment: its coefficient has prior
variance $\lambda_i$, and $T$ spread-out evaluations measure it with noise
variance about $\sigma_n^2/T$, which @eq-info-mi-gp turns into the term
above. A direction with $T\lambda_i \gg \sigma_n^2$ is **resolved** and adds
about $\tfrac12\log(T\lambda_i/\sigma_n^2)$; one with
$T\lambda_i \ll \sigma_n^2$ adds about $T\lambda_i/(2\sigma_n^2)$. The
information is roughly the number of resolved directions times half a
logarithm, plus $T/(2\sigma_n^2)$ times the unresolved eigenvalue mass.

@eq-ka-info-eigen is an approximation. Against the exact information of $T$
evenly spaced evaluations on $[0, 1]$ (lengthscale 0.1, $\sigma_n = 0.1$), it
is within 0.25% at $T = 100$ for the RBF and the Matérn 5/2 and 3/2 kernels.
For Matérn 1/2 it is 46% too high at $T = 100$, because it counts 143
resolved directions, more than 100 evaluations can resolve, and 7.4% too high
at $T = 1000$.

### A bound from the eigenvalues {#sec-ka-gamma-bound}

The count can be made rigorous for any design by cutting at $m$ directions
and paying for the eigenvalue mass beyond them.

::: {.theorem #thm-ka-gamma title="Information gain from eigenvalue decay (Vakili, Khezeli, and Picheny, 2021)"}
Let $k$ satisfy the conditions of @thm-ka-mercer, with
$\lvert k(\vx, \vx')\rvert \le \bar k$ and $\lvert\varphi_i(\vx)\rvert \le \psi$
for all $i$ and $\vx$. For every $m = 1, 2, \dots$,

$$
\gamma_T \le \frac{m}{2}\log\!\left(1 + \frac{\bar k\, T}{\sigma_n^2\, m}\right) + \frac{T\,\delta_m}{2\sigma_n^2},
\qquad
\delta_m = \psi^2\sum_{i > m}\lambda_i .
$$ {#eq-ka-gamma-bound}

This is Theorem 3 of @vakili2021information, whose $\tau$ is $\sigma_n^2$ and
whose $D$ is $m$.
:::

The tail $\delta_m$ is eigenvalue mass, not a failure probability like the
$\delta$ of @sec-ka-norm-bound; the letter follows the paper. The bound on the
eigenfunctions is an assumption, which the authors state holds for the kernels
used in practice; we found no proof that it holds for the RBF and Matérn
kernels on every domain and weighting. The proof needs only the tools of
@sec-mutual-information.

::: {.derivation title="Proof of the bound"}
Fix $T$ inputs $A$ and split the kernel at $m$: $k = k_P + k_O$ with
$k_P = \sum_{i \le m}\lambda_i\varphi_i\varphi_i$ and $k_O$ the rest, both
positive semidefinite.

1. *Split the function.* Independent $f_P \sim \GP(0, k_P)$ and
   $f_O \sim \GP(0, k_O)$ sum to a process with kernel $k$ (@exr-fs-sum), so
   with $\vy = \vf_P + \vf_O + \bm{\varepsilon}$ at $A$,
   $\tfrac12\log\det(\mI + \sigma_n^{-2}\mK_A)$ is the information in $\vy$
   about the sum (@eq-info-mi-gp). The sum is computed from the pair
   $(f_P, f_O)$, so by the data processing inequality
   (@eq-info-data-processing) this is at most the information about the pair.
2. *Chain rule.* That is $I(\vy; f_P) + I(\vy; f_O \given f_P)$
   (@sec-info-sequential) [@cover2006elements, ch. 2]. Given $f_P$, the data
   are $f_O$ plus noise, so the second term is
   $\tfrac12\log\det(\mI + \sigma_n^{-2}\mK_{O})$. The first is at most the
   information of $\vf_P + \bm{\varepsilon}$ about $f_P$, since adding the
   independent $\vf_O$ is further processing:
   $\tfrac12\log\det(\mI + \sigma_n^{-2}\mK_{P})$.
3. *The first $m$ directions.* $\mK_P = \boldsymbol{\Phi}_m\boldsymbol{\Lambda}_m\boldsymbol{\Phi}_m^\T$,
   so by @eq-id-det-lemma its determinant is that of the $m \times m$ matrix
   $\mI + \mathbf{G}$ with
   $\mathbf{G} = \sigma_n^{-2}\boldsymbol{\Lambda}_m^{1/2}\boldsymbol{\Phi}_m^\T\boldsymbol{\Phi}_m\boldsymbol{\Lambda}_m^{1/2}$.
   For its eigenvalues $g_j \ge 0$, concavity of the logarithm gives
   $\sum_j\log(1 + g_j) \le m\log(1 + \tfrac1m\sum_j g_j)$. A trace is
   unchanged when the factors of a product are rotated, so
   $\sum_j g_j = \tr\mathbf{G} = \sigma_n^{-2}\tr\mK_P = \sigma_n^{-2}\sum_t k_P(\vx_t, \vx_t)$.
   Each term satisfies
   $k_P(\vx, \vx) = k(\vx, \vx) - k_O(\vx, \vx) \le k(\vx, \vx) \le \bar k$,
   since $k_O(\vx, \vx) = \sum_{i > m}\lambda_i\varphi_i(\vx)^2 \ge 0$, so
   $\sum_j g_j \le \sigma_n^{-2}T\bar k$.
4. *The rest.* Since $\log(1 + g) \le g$,
   $\log\det(\mI + \sigma_n^{-2}\mK_O) \le \sigma_n^{-2}\tr\mK_O = \sigma_n^{-2}\sum_t k_O(\vx_t, \vx_t)$,
   and $k_O(\vx, \vx) = \sum_{i > m}\lambda_i\varphi_i(\vx)^2 \le \delta_m$.
5. Add steps 3 and 4, halve, and maximize over $A$.
:::

### Polynomial and exponential decay {#sec-ka-rates}

The best $m$ balances the two terms of @eq-ka-gamma-bound, directions kept
times $\log T$ against $T$ times the mass beyond them, and the balance
depends only on how fast the eigenvalues fall.

::: {.derivation title="Rates from the two kinds of decay"}
*Polynomial decay*, $\lambda_i \le C i^{-\beta}$ with $\beta > 1$.

1. $\sum_{i > m} i^{-\beta} \le \int_m^\infty u^{-\beta}\,\dd u = m^{1-\beta}/(\beta - 1)$,
   so the second term is of order $T m^{1-\beta}$; the first is of order
   $m\log T$.
2. They match for $m \approx (T/\log T)^{1/\beta}$, where both are of order
   $T^{1/\beta}(\log T)^{1 - 1/\beta}$.

For a Matérn kernel with $\nu > 1/2$ in $d$ dimensions,
$\lambda_i = O(i^{-(2\nu + d)/d})$ [@santin2016approximation;
@vakili2021information, remark 2], so $1/\beta = d/(2\nu + d)$ and
$\gamma_T = O\big(T^{d/(2\nu + d)}(\log T)^{2\nu/(2\nu + d)}\big)$.

*Exponential decay*, in one dimension $\lambda_i \le Ce^{-ci}$.

3. $\sum_{i > m}e^{-ci} \le e^{-cm}/(1 - e^{-c})$.
4. With $m = \lceil(\log T)/c\rceil$, $Te^{-cm} \le 1$, so the second term is
   bounded by a constant and the first is of order $(\log T)^2/c$.

For the RBF kernel in $d$ dimensions, $\lambda_i = O(e^{-c\,i^{1/d}})$
[@belkin2018approximation; @vakili2021information, remark 2], and $m$ of
order $(\log T)^d$ gives $\gamma_T = O\big((\log T)^{d+1}\big)$
[@vakili2021information, cor. 1].
:::

These are the rates of @sec-info-gamma and @tbl-regret-gamma: the Matérn rate
is that of @vakili2021information, stated for $\nu > 1/2$, and the RBF rate,
first proved by @srinivas2010gaussian, is recovered by the same theorem. A
kernel built from $d$ features, such as the linear kernel in $d$ dimensions,
has at most $d$ nonzero eigenvalues, the tail vanishes at $m = d$, and the
bound gives $O(d\log T)$ (@exr-ka-linear). In
words: a polynomially decaying spectrum keeps about $T^{d/(2\nu + d)}$
directions within reach as $T$ grows, an exponentially decaying one only
about $(\log T)^d$, and each resolved direction costs a logarithm.
Smoothness lowers the exponent; dimension raises it.

The rates are asymptotic. @fig-ka-infogain computes exactly the information
that $T$ evaluations spread evenly over $[0, 1]$ gather, a lower bound on
$\gamma_T$, since $\gamma_T$ takes the best design.

```{figure}
//| figure: ka-infogain
//| label: fig-ka-infogain
//| fig-cap: "The information $\tfrac12\log\det(\mI + \sigma_n^{-2}\mK_T)$ that $T$ evaluations at evenly spaced inputs in $[0, 1]$ gather about $f$, in nats, for four kernels with unit amplitude, on log axes. It is computed exactly and is a lower bound on $\gamma_T$. The dashed line, $T$ times the information of one evaluation, is what $T$ completely new evaluations would gather. The dotted segments between $T = 100$ and $1000$ have the slopes $1/(2\nu + 1)$ that the eigenvalue decay of each Matérn kernel predicts for large $T$ in one dimension. The readout gives each kernel's information at the chosen $T$ and its local slope, the slope on log axes between $T/2$ and $2T$ (between $T/2$ and $T$ at the right edge)."
```

**Read the default.** At $T = 100$, with lengthscale 0.1 and noise standard
deviation 0.1, the RBF kernel gathers 36.5 nats, Matérn 5/2 53.4, Matérn 3/2
70.2, and Matérn 1/2 150.4, against 230.8 for a hundred completely new
evaluations. The curves leave the dashed line, falling below 90% of it,
between $T = 10$ and $T = 22$, when evaluations start to overlap.

**Move T to 1000.** The local slopes are 0.16, 0.22, 0.28, and 0.58, still
above the large-$T$ values of 0 for the RBF kernel and $1/6$, $1/4$, and $1/2$
for the Matérn kernels. A thousand evaluations in one dimension is not yet
the long run.

**Lengthen the lengthscale to 0.3.** At $T = 1000$ the four values fall to
25.2, 41.5, 63.9, and 398.0 nats. The rates do not depend on the lengthscale;
the constants in front of them do, through how many eigenvalues are large.

### From information gain to regret {#sec-ka-regret}

@tbl-ka-rates carries the rates through the regret bounds of GP-UCB, the rule
that evaluates where the posterior mean plus a multiple of the posterior
standard deviation is largest (@eq-regret-gpucb), with the setting each result
assumes.

::: {.table #tbl-ka-rates title="From eigenvalue decay to rates on a compact domain in d dimensions. The regret upper bound is for GP-UCB on draws from the prior, with a confidence parameter that grows like log T; on a continuous domain it needs the RBF kernel or a Matérn kernel with ν > 2. The lower bound is for any algorithm on functions of bounded RKHS norm."}
| Kernel | Eigenvalues $\lambda_i$ | $\gamma_T$ | Regret upper bound, Bayesian | Lower bound, fixed function |
|---|---|---|---|---|
| RBF | $O(e^{-c\,i^{1/d}})$ [@belkin2018approximation] | $O\big((\log T)^{d+1}\big)$ | $O\big(\sqrt{T}(\log T)^{d/2 + 1}\big)$ | $\Omega\big(\sqrt{T(\log T)^{d/2}}\big)$ |
| Matérn, $\nu > 1/2$ | $O(i^{-(2\nu + d)/d})$ [@santin2016approximation] | $O\big(T^{\frac{d}{2\nu + d}}(\log T)^{\frac{2\nu}{2\nu + d}}\big)$ | $O\big(T^{\frac{\nu + d}{2\nu + d}}(\log T)^{\frac{4\nu + d}{4\nu + 2d}}\big)$ | $\Omega\big(T^{\frac{\nu + d}{2\nu + d}}\big)$ |
:::

The $\gamma_T$ and regret columns are those of @vakili2021information; the
regret is $\sqrt{T\log T\,\gamma_T}$ with $\gamma_T$ substituted, and for the
RBF kernel it is the $\sqrt{T}(\log T)^{(d+2)/2}$ of @sec-regret-gpucb-bound.
The lower bounds are those of @scarlett2017lower. On a continuous domain the
Bayesian bound needs sample paths smooth enough to discretize, which the RBF
kernel and Matérn kernels with $\nu > 2$ provide (@sec-regret-other-settings,
@sec-ka-regularity).

For a fixed function in the RKHS, the frequentist analysis of GP-UCB gives
regret of order $\gamma_T\sqrt{T}$ up to logarithmic factors
[@chowdhury2017kernelized], whose exponent with the Matérn rate,
$\tfrac12 + d/(2\nu + d)$, reaches 1 once $d \ge 2\nu$ (inference, as in
@sec-regret-bounds-not-say). The factor $\sqrt{\gamma_T}$ between that and the
lower bound is the subject of @sec-dueling-rates and @sec-theory-rates.

## A Gaussian process made precise {#sec-ka-process}

@def-fs-gp defined a Gaussian process as a collection of random variables,
any finite number of them jointly Gaussian, and @sec-gp-definition read it as
an interface that answers queries at finitely many inputs. The regret bounds
ask more: they take the maximum of $f$ over a continuous domain, infinitely
many inputs at once. This section says what kind of object a Gaussian process
is, and when such questions have answers.

### Random variables indexed by inputs {#sec-ka-stochastic-process}

A random variable is a function of the outcome $\omega$ of a random
experiment (@sec-prob-random-variables; here $\omega$ is an outcome, not a
frequency). A **stochastic process** on a domain $\X$ is a function
$f(\vx, \omega)$ such that $f(\vx, \cdot)$ is a random variable for every
fixed $\vx$. Fixing the outcome instead gives the **sample path**
$\vx \mapsto f(\vx, \omega)$. A Gaussian process is a stochastic process whose
values at any finite list of inputs are jointly Gaussian
[@dacosta2026sample, def. 2.1].

The joint distributions at finite lists of inputs, the
**finite-dimensional distributions**, must agree in two ways: listing the
inputs in another order permutes the distribution accordingly, and dropping an
input gives the marginal of the rest. @sec-gp-definition checked the second
for every mean function and kernel.

::: {.theorem #thm-ka-kolmogorov title="Kolmogorov extension theorem"}
Every family of finite-dimensional distributions consistent in these two ways
belongs to some stochastic process on $\X$, and it determines that process's
probabilities for every event that involves countably many inputs
[@kolmogorov1933grundbegriffe, ch. III, sec. 4].
:::

This is the theorem the aside of @sec-gp-definition named. For a Gaussian
process it says that a mean function and a positive semidefinite kernel
define a random function on any domain [@dacosta2026sample, sec. 2]. Its
proof needs measure theory, and the book does not give it.

### What the finite-dimensional distributions do not decide {#sec-ka-versions}

Whether the path is continuous, or what its maximum on $[0, 1]$ is, are
questions about uncountably many inputs, and the finite-dimensional
distributions do not decide them. Let $U$ be uniform on $[0, 1]$, and define
$f(x) = 0$ for every $x$ and $g(x) = 1$ if $x = U$, 0 otherwise. At any fixed
$x$, $g(x) = 0$ unless $U$ lands exactly on $x$, which has probability zero,
so $f$ and $g$ have the same finite-dimensional distributions. Yet every path
of $f$ is continuous with maximum 0, and every path of $g$ jumps and has
maximum 1.

Processes that agree with probability 1 at each fixed input are called
**versions** of each other [@kanagawa2018gaussian, def. 4.8], and a statement
about sample paths says that some version has them. When a Gaussian process
on a closed and bounded domain has a continuous version, that version is the
one meant, and its maximum $f^\star$ and maximizer $\vx^\star$ exist, because a
continuous function on such a domain attains its maximum. Entropy search, which chooses evaluations by what they reveal about where
$\vx^\star$ is (@sec-entropy-search), presupposes this.

### Sample-path regularity {#sec-ka-regularity}

Whether a continuous version exists depends on the kernel near zero
distance. For a stationary kernel and mean zero, the expected squared
difference of two nearby values is

$$
\E\big[(f(\vx + \mathbf{h}) - f(\vx))^2\big] = 2\big(k(\mathbf{0}) - k(\mathbf{h})\big).
$$

If this shrinks like $\lVert\mathbf{h}\rVert^{2\eta}$ for some $\eta$
between 0 and 1, the process has a version whose paths are continuous, and
Hölder continuous of every order $\eta' < \eta$, meaning that
$\lvert f(\vx) - f(\vx')\rvert \le C\lVert\vx - \vx'\rVert^{\eta'}$ near each
point [@dacosta2026sample, thm. 3.1]. For Gaussian processes this is a sharp
form of the continuity theorem of Kolmogorov, on which the proof rests.

For the Matérn 1/2 kernel, $k(0) - k(h) = 1 - e^{-\lvert h\rvert/\ell} \le \lvert h\rvert/\ell$,
so $2\eta = 1$: the paths are continuous, Hölder of every order below 1/2
like Brownian motion (a random walk taken to continuous time), and no more, so not differentiable
[@dacosta2026sample, remark 3.4]. Applying the criterion to the derivative
process, whose kernel is $-k''$, climbs the ladder: for $\nu$ that is not a
whole number, a Matérn path is (for a suitable version) $\lfloor\nu\rfloor$
times continuously differentiable and no more, so $\nu = 1/2, 3/2, 5/2$ give
0, 1, and 2 continuous derivatives [@dacosta2026sample, cor. 1.2 and
prop. 3.1]. For such $\nu$ these sample-path statements agree with the
mean-square ladder of @tbl-kern-family, which also gives derivatives of every
whole order below $\nu$ and no more. RBF paths have derivatives of every order
[@dacosta2026sample, remark 3.5].

The agreement is a theorem, not a definition. Mean-square differentiability
(@sec-fs-smoothness) concerns second moments of difference quotients and
follows from $k''(0)$ alone [@rasmussen2006gaussian, sec. 4.1.1];
sample-path differentiability concerns each drawn function. The
continuous-domain regret bound of @sec-regret-other-settings needs the
second, with some to spare: its proof discretizes the domain and asks that
nearby values be close with high probability, which the RBF kernel and Matérn
kernels with $\nu > 2$ provide [@srinivas2010gaussian].

For a Matérn kernel, then, draws have $\lfloor\nu\rfloor$ continuous
derivatives and Sobolev smoothness just below $\nu$, while RKHS functions,
the posterior mean among them, have Sobolev smoothness $\nu + d/2$
(@sec-ka-samples). Every regret bound assumes one of these levels.

## Exercises {#sec-ka-exercises}

::: {.exercise #exr-ka-narrow-bump}
Let $k$ be the RBF kernel with unit amplitude and lengthscale $\ell$ on $\R$.
(a) Show that $\lVert k(\cdot, a) - k(\cdot, b)\rVert_k^2 = 2 - 2k(a, b)$ and
evaluate it at $\lvert a - b\rvert = \ell$ and $3\ell$. (b) For a stationary
kernel on $\R$, $f \in \mathcal{H}_k$ exactly when
$\int \lvert\hat f(\omega)\rvert^2/s(\omega)\,\dd\omega$ is finite, where
$\hat f$ is the Fourier transform of $f$ and $s$ the spectral density
[@wendland2004scattered, thm. 10.12; @kanagawa2018gaussian, thm. 2.4]. For the
bump $f(x) = e^{-x^2/2w^2}$, with $\lvert\hat f(\omega)\rvert^2 \propto e^{-w^2\omega^2}$,
find the widths $w$ for which $f \in \mathcal{H}_k$.

::: {.solution}
(a) By @eq-ka-inner with coefficients $1$ and $-1$, the squared norm is
$k(a,a) - 2k(a,b) + k(b,b) = 2 - 2k(a,b)$. At distance $\ell$,
$k = e^{-1/2} = 0.607$ and it is $0.787$; at $3\ell$, $k = e^{-9/2} = 0.011$
and it is $1.978$, close to 2, the value for two bumps that do not overlap.
(b) The RBF spectral density is proportional to $e^{-\ell^2\omega^2/2}$, so
the integrand is proportional to $e^{-(w^2 - \ell^2/2)\omega^2}$, and the
integral is finite exactly when $w > \ell/\sqrt{2} \approx 0.71\,\ell$. A
narrower bump has infinite norm: the kernel gives its high frequencies too
little prior variance to pay for them. The kernel's own bump, of width
$\ell$, qualifies, as it must.
:::
:::

::: {.exercise #exr-ka-gauss-spectrum}
Let $\omega \sim \N(0, 1/\ell^2)$ and $g(r) = \E[\cos(\omega r)]$. (a) Show,
by integrating by parts, that $\E[\omega h(\omega)] = \ell^{-2}\E[h'(\omega)]$
for a differentiable $h$ that does not grow too fast. (b) Use it to show
$g'(r) = -(r/\ell^2)g(r)$, and conclude that $g(r) = e^{-r^2/2\ell^2}$.

::: {.solution}
(a) The density $p(\omega) = c\,e^{-\ell^2\omega^2/2}$ has
$p'(\omega) = -\ell^2\omega\,p(\omega)$, so
$\E[\omega h(\omega)] = -\ell^{-2}\int h\,p'\,\dd\omega = \ell^{-2}\int h'\,p\,\dd\omega$
by parts, the boundary terms vanishing because $p$ decays faster than $h$
grows. (b) $g'(r) = -\E[\omega\sin(\omega r)]$, and with $h(\omega) = \sin(\omega r)$,
$h' = r\cos(\omega r)$, so $g'(r) = -(r/\ell^2)g(r)$. With $g(0) = 1$ the
unique solution is $e^{-r^2/2\ell^2}$, the RBF kernel: Gaussian frequencies
with standard deviation $1/\ell$ give lengthscale $\ell$, and a short
lengthscale needs high frequencies.
:::
:::

::: {.exercise #exr-ka-rff}
One random feature estimates a unit-amplitude kernel at distance $r$ by
$\cos(\omega r)$, $\omega \sim p$. (a) Show that its variance is
$\tfrac12\big(1 + k(2r)\big) - k(r)^2$. (b) Evaluate it at $r = 0$ and as
$r \to \infty$. (c) For the RBF kernel, show that it never exceeds
$\tfrac12$, so the average of $M$ features has standard deviation at most
$1/\sqrt{2M}$.

::: {.solution}
(a) $\cos^2\theta = \tfrac12(1 + \cos 2\theta)$, so
$\E[\cos^2(\omega r)] = \tfrac12(1 + k(2r))$ by @eq-ka-bochner; subtract the
squared mean $k(r)^2$. (b) At $r = 0$ it is $\tfrac12 \cdot 2 - 1 = 0$, since
every feature gives $k(0) = 1$ exactly; as $r \to \infty$ it tends to
$\tfrac12$. (c) For the RBF kernel $k(2r) = k(r)^4$, so the variance is
$\tfrac12 - k^2(1 - \tfrac12k^2) \le \tfrac12$ for $0 \le k \le 1$. The $M$
terms are independent, so the average has variance at most $1/(2M)$, the
comparison value of @fig-ka-bochner.
:::
:::

::: {.exercise #exr-ka-linear}
(a) The linear kernel $k(\vx, \vx') = \vx^\T\vx'$ on the unit ball of $\R^d$
has at most $d$ nonzero eigenvalues. Use @thm-ka-gamma with $m = d$ to show
$\gamma_T \le \frac d2\log\big(1 + T/(\sigma_n^2 d)\big)$. (b) Compare with the
$K$ independent arms of @exr-regret-independent, for which
$\gamma_T = \frac K2\log\big(1 + T/(K\sigma_n^2)\big)$, and explain the shared
form.

::: {.solution}
(a) With $m = d$ the tail $\delta_d$ is zero, and on the unit ball
$\lvert k\rvert \le 1$, so $\bar k = 1$ and only the first term of
@eq-ka-gamma-bound remains: $\gamma_T \le \frac d2\log(1 + T/(\sigma_n^2 d))$,
which is $O(d\log T)$, the linear rate of @sec-info-gamma. The bound $\psi$
does not enter, because step 4 of the proof is not needed. (b) Both kernels
have a fixed number of directions, $d$ or $K$, holding all the variance. Each
can be measured repeatedly but contributes only a logarithm, and the best
design spreads the $T$ evaluations evenly among them. A kernel with infinitely
many eigenvalues behaves like one with $m$ directions, where $m$ grows with
$T$ as fast as the decay allows.
:::
:::

## Further reading {#further-reading .unnumbered}

- @kanagawa2018gaussian, a long review available as a preprint, sets the
  Gaussian process and RKHS views side by side, with Mercer's theorem, the
  Karhunen-Loève expansion, the zero-one law for sample paths, and the
  Sobolev spaces of Matérn kernels.
- @rasmussen2006gaussian, chapter 4, treats stationary kernels, Bochner's
  theorem, and eigenfunction analysis; chapter 6 introduces the RKHS and its
  link to regularization.
- @aronszajn1950theory founded the theory of reproducing kernels;
  @steinwart2008support, chapter 4, and @berlinet2004reproducing give modern
  treatments, the second with probability in view.
- @wendland2004scattered develops positive definite functions (chapter 6) and
  the native spaces of kernels (chapter 10) from approximation theory.
- @rahimi2007random introduced random Fourier features, with the uniform
  convergence bound quoted here.
- @vakili2021information derives the information-gain rates from eigenvalue
  decay; @thm-ka-gamma is their Theorem 3.
- @dacosta2026sample gives necessary and sufficient conditions on the kernel
  for the sample paths to have a given smoothness, with the Matérn and RBF
  kernels as examples.
