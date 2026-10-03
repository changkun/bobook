---
status: done
synopsis: "The identities the derivations rely on, each stated, proved, and tied to where the book uses it: block inversion and the Schur complement, the Woodbury identity, Gaussian conditioning and the linear-Gaussian pair, products of Gaussian densities, and Clark's formula for the expected maximum of two Gaussians."
sources: ["Rasmussen and Williams 2006, app. A", "Petersen and Pedersen 2012", "Bishop 2006, sec. 2.3", "Clark 1961"]
---

# Matrix and Gaussian Identities {#sec-gaussian-identities}

The chapters of this book reuse a small number of identities. Each was
introduced where it was first needed, usually with a derivation shaped by
that chapter's example. This appendix collects them in one place, in their
general form, each with a proof short enough to check and a note on where
the book relies on it.

The five sections build on one another. Block inversion is the root: the
Woodbury identity is block inversion read in two ways, Gaussian conditioning
is block inversion applied to a covariance matrix, and the product of two
Gaussian densities is conditioning in disguise. The last section, on the
expected maximum of two Gaussian values, is independent of the others and
serves the acquisition functions of @sec-acquisition and @sec-pbo.

Throughout, vectors are columns, $\mI$ is an identity matrix of whatever size
fits, and every matrix that is inverted is assumed to be invertible.

## Block inversion and the Schur complement {#sec-id-block}

A matrix whose rows and columns fall into two groups can be inverted group
by group. Write it in blocks,

$$
\mathbf{M} = \begin{bmatrix} \mA & \mathbf{B} \\ \mathbf{C} & \mathbf{D} \end{bmatrix},
$$ {#eq-id-blocks}

with $\mA$ of size $p \times p$ and $\mathbf{D}$ of size $q \times q$. The
matrix need not be symmetric. The **Schur complement** of $\mA$ in
$\mathbf{M}$ is

$$
\mathbf{S} = \mathbf{D} - \mathbf{C}\mA^{-1}\mathbf{B},
$$ {#eq-id-schur}

what remains of $\mathbf{D}$ after the first group has been eliminated, in
the same way that elimination on two equations leaves $d - cb/a$ in the
corner of a $2 \times 2$ matrix.

::: {.theorem #thm-id-block title="Block inversion"}
If $\mA$ and $\mathbf{S}$ are invertible, then

$$
\begin{aligned}
\mathbf{M}^{-1} &=
\begin{bmatrix} \mathbf{P} & \mathbf{Q} \\ \mathbf{R} & \mathbf{S}^{-1} \end{bmatrix}, \\
\mathbf{P} &= \mA^{-1} + \mA^{-1}\mathbf{B}\,\mathbf{S}^{-1}\mathbf{C}\mA^{-1}, \\
\mathbf{Q} &= -\mA^{-1}\mathbf{B}\,\mathbf{S}^{-1}, \\
\mathbf{R} &= -\mathbf{S}^{-1}\mathbf{C}\mA^{-1},
\end{aligned}
$$ {#eq-id-block-inverse}

and $\det\mathbf{M} = \det\mA \cdot \det\mathbf{S}$.
:::

::: {.proof}
Eliminate the off-diagonal blocks with two triangular matrices,

$$
\mathbf{E} = \begin{bmatrix} \mI & \mathbf{0} \\ -\mathbf{C}\mA^{-1} & \mI \end{bmatrix},
\qquad
\mathbf{F} = \begin{bmatrix} \mI & -\mA^{-1}\mathbf{B} \\ \mathbf{0} & \mI \end{bmatrix}.
$$

1. Multiplying by $\mathbf{E}$ on the left subtracts $\mathbf{C}\mA^{-1}$
   times the first block row from the second. The bottom-left block becomes
   $\mathbf{C} - \mathbf{C}\mA^{-1}\mA = \mathbf{0}$ and the bottom-right
   block becomes $\mathbf{D} - \mathbf{C}\mA^{-1}\mathbf{B} = \mathbf{S}$.
2. Multiplying the result by $\mathbf{F}$ on the right subtracts the first
   block column times $\mA^{-1}\mathbf{B}$ from the second, which clears the
   top-right block and leaves the rest unchanged. So
   $$
   \mathbf{E}\,\mathbf{M}\,\mathbf{F} = \begin{bmatrix} \mA & \mathbf{0} \\ \mathbf{0} & \mathbf{S} \end{bmatrix}.
   $$
3. $\mathbf{E}$ and $\mathbf{F}$ are invertible: each is undone by the same
   matrix with the sign of its off-diagonal block flipped. Inverting both
   sides of step 2 gives
   $\mathbf{F}^{-1}\mathbf{M}^{-1}\mathbf{E}^{-1} = \operatorname{diag}(\mA^{-1}, \mathbf{S}^{-1})$,
   the block-diagonal matrix with those two blocks, and so
   $\mathbf{M}^{-1} = \mathbf{F}\operatorname{diag}(\mA^{-1}, \mathbf{S}^{-1})\,\mathbf{E}$.
4. Multiply out. $\mathbf{F}\operatorname{diag}(\mA^{-1}, \mathbf{S}^{-1})$ has
   top row $(\mA^{-1},\, -\mA^{-1}\mathbf{B}\mathbf{S}^{-1})$ and bottom row
   $(\mathbf{0},\, \mathbf{S}^{-1})$, and multiplying by $\mathbf{E}$ on the
   right adds $-\mathbf{C}\mA^{-1}$ times the second column to the first,
   which gives the four blocks of @eq-id-block-inverse.
5. $\mathbf{E}$ and $\mathbf{F}$ are triangular with ones on the diagonal, so
   their determinants are 1, and the determinant of a product is the product
   of the determinants (@sec-determinants). Step 2 then gives
   $\det\mathbf{M} = \det\mA\cdot\det\mathbf{S}$.
:::

Nothing singled out the first group. Eliminating the second group instead,
with the Schur complement of $\mathbf{D}$,
$\mathbf{T} = \mA - \mathbf{B}\mathbf{D}^{-1}\mathbf{C}$, the same argument
gives a second expression for the same inverse:

$$
\begin{aligned}
\mathbf{M}^{-1} &=
\begin{bmatrix} \mathbf{T}^{-1} & \mathbf{Q}' \\ \mathbf{R}' & \mathbf{P}' \end{bmatrix}, \\
\mathbf{P}' &= \mathbf{D}^{-1} + \mathbf{D}^{-1}\mathbf{C}\,\mathbf{T}^{-1}\mathbf{B}\mathbf{D}^{-1}, \\
\mathbf{Q}' &= -\mathbf{T}^{-1}\mathbf{B}\mathbf{D}^{-1}, \\
\mathbf{R}' &= -\mathbf{D}^{-1}\mathbf{C}\,\mathbf{T}^{-1},
\end{aligned}
$$ {#eq-id-block-inverse-2}

with $\det\mathbf{M} = \det\mathbf{D}\cdot\det\mathbf{T}$.

**Where the book uses it.** @sec-linalg-schur derives the symmetric case,
$\mathbf{C} = \mathbf{B}^\T$, step by step, and shows that a symmetric
$\mathbf{M}$ is positive definite exactly when $\mA$ and $\mathbf{S}$ are.
The bottom-right block of @eq-id-block-inverse, $\mathbf{S}^{-1}$, is why the
covariance of a conditioned Gaussian is a Schur complement
(@sec-id-conditioning). Taking the second group to be a single coordinate
gives the leave-one-out formulas @eq-kern-loo, and adding one row and column
to a factored matrix gives the Cholesky update of
@sec-linalg-cholesky-update.

## The Woodbury identity {#sec-id-woodbury}

The two expressions for $\mathbf{M}^{-1}$ must agree block by block.
Comparing their top-left blocks produces the most useful identity in this
appendix. It says how the inverse of a matrix changes when a low-rank matrix
is added to it.

::: {.theorem #thm-id-woodbury title="Woodbury identity"}
Let $\mathbf{Z}$ be $n \times n$, $\mW$ be $m \times m$, and $\mathbf{U}$ and
$\mathbf{V}$ be $n \times m$. Then

$$
\begin{aligned}
&\left(\mathbf{Z} + \mathbf{U}\mW\mathbf{V}^\T\right)^{-1}
= \mathbf{Z}^{-1} - \mathbf{Z}^{-1}\mathbf{U}\,\mathbf{N}^{-1}\mathbf{V}^\T\mathbf{Z}^{-1}, \\
&\text{where } \mathbf{N} = \mW^{-1} + \mathbf{V}^\T\mathbf{Z}^{-1}\mathbf{U}.
\end{aligned}
$$ {#eq-id-woodbury}
:::

::: {.proof}
Apply @sec-id-block to the block matrix with $\mA = \mathbf{Z}$,
$\mathbf{B} = -\mathbf{U}$, $\mathbf{C} = \mathbf{V}^\T$, and
$\mathbf{D} = \mW^{-1}$.

1. The Schur complement of $\mA$ is
   $\mathbf{S} = \mW^{-1} + \mathbf{V}^\T\mathbf{Z}^{-1}\mathbf{U} = \mathbf{N}$,
   and by @eq-id-block-inverse the top-left block of the inverse is
   $\mathbf{P} = \mathbf{Z}^{-1} - \mathbf{Z}^{-1}\mathbf{U}\,\mathbf{N}^{-1}\mathbf{V}^\T\mathbf{Z}^{-1}$.
2. The Schur complement of $\mathbf{D}$ is
   $\mathbf{T} = \mathbf{Z} + \mathbf{U}\mW\mathbf{V}^\T$, and by
   @eq-id-block-inverse-2 the top-left block of the inverse is
   $\mathbf{T}^{-1}$.
3. A matrix has one inverse, so the two blocks are equal.
:::

The identity is also called the matrix inversion lemma
[@rasmussen2006gaussian, app. A.3]. Its value is in the sizes. The left side
inverts an $n \times n$ matrix. The right side, given $\mathbf{Z}^{-1}$,
inverts only the $m \times m$ matrix $\mathbf{N}$. When $\mathbf{Z}$ is easy
to invert, a diagonal matrix for example, and $m$ is much smaller than $n$,
the cost falls from $O(n^3)$ to $O(nm^2)$.

Three companions follow from the same block matrix.

**Rank one.** With $m = 1$, $\mW = 1$, and vectors $\mathbf{u}, \mathbf{v}$
in place of $\mathbf{U}, \mathbf{V}$, the middle inverse is a number. This is
the Sherman-Morrison formula:

$$
\left(\mathbf{Z} + \mathbf{u}\mathbf{v}^\T\right)^{-1}
= \mathbf{Z}^{-1} - \frac{\mathbf{Z}^{-1}\mathbf{u}\,\mathbf{v}^\T\mathbf{Z}^{-1}}{1 + \mathbf{v}^\T\mathbf{Z}^{-1}\mathbf{u}}.
$$ {#eq-id-sherman-morrison}

**Determinants.** Equating the two determinant formulas of @sec-id-block
for the same block matrix gives the matrix determinant lemma
[@rasmussen2006gaussian, app. A.3]:

$$
\det(\mathbf{Z} + \mathbf{U}\mW\mathbf{V}^\T)
= \det\mathbf{Z}\,\det\mW\,\det\mathbf{N}.
$$ {#eq-id-det-lemma}

**Pushing through.** For any $\mathbf{X}$ of size $n \times m$ and
$\mathbf{Y}$ of size $m \times n$,

$$
\left(\mI + \mathbf{X}\mathbf{Y}\right)^{-1}\mathbf{X} = \mathbf{X}\left(\mI + \mathbf{Y}\mathbf{X}\right)^{-1},
$$ {#eq-id-push-through}

because $\mathbf{X}(\mI + \mathbf{Y}\mathbf{X}) = (\mI + \mathbf{X}\mathbf{Y})\mathbf{X}$,
and multiplying by the two inverses, one on each side, moves them across.

::: {.example #ex-id-blr title="Weight space and function space agree"}
@sec-bayes-blr-posterior found the posterior of a linear model with $M$
features in weight space: covariance $\mA_w^{-1}$ with
$\mA_w = \mSigma_p^{-1} + \sigma_n^{-2}\boldsymbol{\Phi}^\T\boldsymbol{\Phi}$, an
$M \times M$ matrix, and mean
$\bar{\vw} = \sigma_n^{-2}\mA_w^{-1}\boldsymbol{\Phi}^\T\vy$.
@sec-gp-noise predicts in function space with the $n \times n$ matrix
$\mK_y = \boldsymbol{\Phi}\mSigma_p\boldsymbol{\Phi}^\T + \sigma_n^2\mI$. The
two must give the same predictions, and the identities show that they do.

*Covariance.* Apply @eq-id-woodbury with $\mathbf{Z} = \mSigma_p^{-1}$,
$\mathbf{U} = \mathbf{V} = \boldsymbol{\Phi}^\T$, and $\mW = \sigma_n^{-2}\mI$:

$$
\mA_w^{-1} = \mSigma_p - \mSigma_p\boldsymbol{\Phi}^\T\mK_y^{-1}\boldsymbol{\Phi}\mSigma_p.
$$

Multiply by the features $\boldsymbol{\phi}_*$ of a new input on both sides.
The left side is the weight-space predictive variance
$\boldsymbol{\phi}_*^\T\mA_w^{-1}\boldsymbol{\phi}_*$. The right side is
$k(\vx_*, \vx_*) - \vk_*^\T\mK_y^{-1}\vk_*$, with the kernel
$k(\vx, \vx') = \boldsymbol{\phi}(\vx)^\T\mSigma_p\boldsymbol{\phi}(\vx')$ of
@eq-fs-induced-kernel and $\vk_* = \boldsymbol{\Phi}\mSigma_p\boldsymbol{\phi}_*$.
That is @eq-gp-noisy.

*Mean.* From the definitions,
$\mA_w\mSigma_p\boldsymbol{\Phi}^\T = \boldsymbol{\Phi}^\T + \sigma_n^{-2}\boldsymbol{\Phi}^\T\boldsymbol{\Phi}\mSigma_p\boldsymbol{\Phi}^\T = \sigma_n^{-2}\boldsymbol{\Phi}^\T\mK_y$.
Multiplying by $\mA_w^{-1}$ on the left and $\mK_y^{-1}$ on the right gives
$\sigma_n^{-2}\mA_w^{-1}\boldsymbol{\Phi}^\T = \mSigma_p\boldsymbol{\Phi}^\T\mK_y^{-1}$,
a push-through identity. So
$\boldsymbol{\phi}_*^\T\bar{\vw} = \vk_*^\T\mK_y^{-1}\vy$, again @eq-gp-noisy.

Weight space inverts an $M \times M$ matrix and function space an
$n \times n$ one [@rasmussen2006gaussian, sec. 2.1.2]. With a few features
and many observations, weight space is cheaper. With infinitely many
features, only function space is possible (@sec-kernel-trick).
:::

**Where the book uses it.** Besides the example, @eq-id-sherman-morrison
gives a second solution of @exr-noise-floor (@exr-id-sherman), and the
low-rank approximations that make Gaussian
processes affordable for large data sets (@sec-gp-computation) apply
@eq-id-woodbury and @eq-id-det-lemma with $m$ inducing points in place of
$n$ observations [@quinonero2005unifying].

## Conditioning a Gaussian {#sec-id-conditioning}

Let a Gaussian vector be split into two blocks, with the notation of
@eq-gauss-partition:

$$
\begin{bmatrix} \mathbf{a} \\ \mathbf{b} \end{bmatrix}
\sim \N\!\left(
\begin{bmatrix} \vmu_a \\ \vmu_b \end{bmatrix},\;
\begin{bmatrix} \mSigma_{aa} & \mSigma_{ab} \\ \mSigma_{ab}^\T & \mSigma_{bb} \end{bmatrix}
\right).
$$

Two operations reduce it to a statement about one block, and both return a
Gaussian.

::: {.theorem #thm-id-conditioning title="Marginal and conditional of a Gaussian"}
The marginal distribution of $\mathbf{a}$ is $\N(\vmu_a, \mSigma_{aa})$. The
conditional distribution of $\mathbf{b}$ given $\mathbf{a}$ is Gaussian,

$$
\begin{aligned}
\mathbf{b} \given \mathbf{a} &\sim \N\!\left(\vmu_{b \mid a},\, \mSigma_{b \mid a}\right), \\
\vmu_{b \mid a} &= \vmu_b + \mSigma_{ab}^\T\mSigma_{aa}^{-1}(\mathbf{a} - \vmu_a), \\
\mSigma_{b \mid a} &= \mSigma_{bb} - \mSigma_{ab}^\T\mSigma_{aa}^{-1}\mSigma_{ab}.
\end{aligned}
$$ {#eq-id-conditional}

In terms of the precision matrix $\bm{\Lambda} = \mSigma^{-1}$, partitioned
the same way, the conditional has precision $\bm{\Lambda}_{bb}$ and mean
$\vmu_b - \bm{\Lambda}_{bb}^{-1}\bm{\Lambda}_{ab}^\T(\mathbf{a} - \vmu_a)$.
:::

::: {.proof}
The marginal is the linear map that keeps $\mathbf{a}$ and drops
$\mathbf{b}$, and linear maps of Gaussians are Gaussian with the mapped mean
and covariance (@eq-gauss-affine).

For the conditional, the density of $\mathbf{b}$ given $\mathbf{a}$ is
proportional, as a function of $\mathbf{b}$, to the joint density, whose
logarithm is $-\tfrac12(\vx - \vmu)^\T\bm{\Lambda}(\vx - \vmu)$ plus a
constant. Collecting the terms in $\mathbf{b}$ leaves a quadratic with
precision $\bm{\Lambda}_{bb}$ and the mean stated in precision form. By
@thm-id-block applied to $\mSigma$, with the Schur complement
$\mathbf{S} = \mSigma_{bb} - \mSigma_{ab}^\T\mSigma_{aa}^{-1}\mSigma_{ab}$, the
blocks of the precision are $\bm{\Lambda}_{bb} = \mathbf{S}^{-1}$ and
$\bm{\Lambda}_{ab}^\T = -\mathbf{S}^{-1}\mSigma_{ab}^\T\mSigma_{aa}^{-1}$.
Substituting gives covariance $\mathbf{S}$ and mean offset
$\mSigma_{ab}^\T\mSigma_{aa}^{-1}(\mathbf{a} - \vmu_a)$.
@sec-gauss-conditioning-general carries out each step, and gives a second
proof that avoids the precision matrix.
:::

Three features of @eq-id-conditional recur throughout the book. The
conditional mean is linear in the observed block. The conditional covariance
does not depend on the observed values at all. And the conditional
covariance is the prior covariance minus a positive semidefinite matrix, so
observing can only reduce uncertainty.

Most models in this book are not given as a joint Gaussian. They are given
as a Gaussian prior and an observation that is a linear function of the
unknown plus Gaussian noise. The next result converts one form into the
other.

::: {.theorem #thm-id-linear-gaussian title="The linear-Gaussian pair"}
Let $\mathbf{a} \sim \N(\vmu, \mSigma)$ and
$\mathbf{b} \given \mathbf{a} \sim \N(\mathbf{H}\mathbf{a} + \mathbf{c},\, \mathbf{R})$.
Then $\mathbf{a}$ and $\mathbf{b}$ are jointly Gaussian with

$$
\begin{aligned}
\E[\mathbf{b}] &= \mathbf{H}\vmu + \mathbf{c}, \\
\Cov[\mathbf{b}] &= \mathbf{H}\mSigma\mathbf{H}^\T + \mathbf{R}, \\
\Cov[\mathbf{a}, \mathbf{b}] &= \mSigma\mathbf{H}^\T,
\end{aligned}
$$ {#eq-id-lg-joint}

and the posterior of $\mathbf{a}$ given $\mathbf{b}$ is Gaussian with

$$
\begin{aligned}
\E[\mathbf{a} \given \mathbf{b}] &= \vmu + \mathbf{G}\,(\mathbf{b} - \mathbf{H}\vmu - \mathbf{c}), \\
\Cov[\mathbf{a} \given \mathbf{b}] &= \mSigma - \mathbf{G}\,\mathbf{H}\mSigma, \\
\mathbf{G} &= \mSigma\mathbf{H}^\T\left(\mathbf{H}\mSigma\mathbf{H}^\T + \mathbf{R}\right)^{-1}.
\end{aligned}
$$ {#eq-id-lg-posterior}
:::

::: {.proof}
Write $\mathbf{b} = \mathbf{H}\mathbf{a} + \mathbf{c} + \mathbf{e}$ with
$\mathbf{e} \sim \N(\mathbf{0}, \mathbf{R})$ independent of $\mathbf{a}$. The
pair $(\mathbf{a}, \mathbf{e})$ is jointly Gaussian because its parts are
independent Gaussians, and $(\mathbf{a}, \mathbf{b})$ is a linear map of it,
so it is jointly Gaussian too. Its moments follow by linearity:
$\Cov[\mathbf{a}, \mathbf{b}] = \Cov[\mathbf{a}, \mathbf{H}\mathbf{a}] = \mSigma\mathbf{H}^\T$,
and $\Cov[\mathbf{b}] = \mathbf{H}\mSigma\mathbf{H}^\T + \mathbf{R}$ because
the two independent parts of $\mathbf{b}$ add their covariances. The
posterior is @eq-id-conditional with the roles of the two blocks exchanged.
:::

**Where the book uses it.** With $\mathbf{H} = \mI$ on the observed inputs
and $\mathbf{R} = \sigma_n^2\mI$, @eq-id-lg-joint is the joint distribution
behind Gaussian process regression with noise (@sec-gp-noise), and its
middle formula is the covariance $\mK_y$ of the marginal likelihood
(@sec-marginal-likelihood). With $\mathbf{H} = \boldsymbol{\Phi}$ it is
Bayesian linear regression (@sec-blr), in the function-space form of
@ex-id-blr. The matrix $\mathbf{G}$ is called the gain: it converts the
surprise in the observation into a correction of the prior mean.

## Products of Gaussian densities {#sec-id-products}

Multiplying two Gaussian densities over the same variable gives a Gaussian
shape again, scaled by a constant. Write $\N(\vx;\, \vmu, \mSigma)$ for the
Gaussian density with mean $\vmu$ and covariance $\mSigma$, evaluated at
$\vx$.

::: {.theorem #thm-id-product title="Product of two Gaussian densities"}
$$
\N(\vx;\, \vmu_1, \mSigma_1)\;\N(\vx;\, \vmu_2, \mSigma_2)
= Z\;\N(\vx;\, \vmu, \mSigma),
$$ {#eq-id-product}

with
$\mSigma = \left(\mSigma_1^{-1} + \mSigma_2^{-1}\right)^{-1}$,
$\vmu = \mSigma\left(\mSigma_1^{-1}\vmu_1 + \mSigma_2^{-1}\vmu_2\right)$, and
$Z = \N(\vmu_1;\, \vmu_2,\, \mSigma_1 + \mSigma_2)$.
:::

::: {.proof}
Read the product as a model. Let $\vx \sim \N(\vmu_1, \mSigma_1)$ and
$\vy \given \vx \sim \N(\vx, \mSigma_2)$.

1. The joint density is
   $p(\vx)\,p(\vy \given \vx) = \N(\vx;\, \vmu_1, \mSigma_1)\,\N(\vy;\, \vx, \mSigma_2)$.
   A Gaussian density depends on its argument and its mean only through
   their difference, so $\N(\vy;\, \vx, \mSigma_2) = \N(\vx;\, \vy, \mSigma_2)$.
   At $\vy = \vmu_2$ the joint density is the left side of @eq-id-product.
2. The same joint density factors the other way, as
   $p(\vy)\,p(\vx \given \vy)$. By @thm-id-linear-gaussian with
   $\mathbf{H} = \mI$, $\mathbf{c} = \mathbf{0}$, and $\mathbf{R} = \mSigma_2$:
   $p(\vy) = \N(\vy;\, \vmu_1, \mSigma_1 + \mSigma_2)$, and $p(\vx \given \vy)$
   is Gaussian with covariance
   $\mSigma_1 - \mSigma_1(\mSigma_1 + \mSigma_2)^{-1}\mSigma_1$ and mean
   $\vmu_1 + \mSigma_1(\mSigma_1 + \mSigma_2)^{-1}(\vy - \vmu_1)$.
3. At $\vy = \vmu_2$, the first factor is the constant $Z$.
4. By @eq-id-woodbury with $\mathbf{Z} = \mSigma_1^{-1}$,
   $\mathbf{U} = \mathbf{V} = \mI$, and $\mW = \mSigma_2^{-1}$, the covariance
   in step 2 equals $(\mSigma_1^{-1} + \mSigma_2^{-1})^{-1} = \mSigma$.
5. From step 4,
   $\mSigma\mSigma_1^{-1} = \mI - \mSigma_1(\mSigma_1 + \mSigma_2)^{-1}$, and
   since $\mSigma(\mSigma_1^{-1} + \mSigma_2^{-1}) = \mI$,
   $\mSigma\mSigma_2^{-1} = \mSigma_1(\mSigma_1 + \mSigma_2)^{-1}$. So the mean in
   step 2 at $\vy = \vmu_2$ is
   $\mSigma\mSigma_1^{-1}\vmu_1 + \mSigma\mSigma_2^{-1}\vmu_2 = \vmu$.
:::

The proof explains the three parts of the result. Precisions add because two
independent pieces of evidence about the same quantity are being combined.
The mean is the precision-weighted average of the two means. And the
constant $Z$ is the probability density that the first Gaussian, blurred by
the second, assigns to the second's center: it is large when the two
densities overlap and tiny when they do not. @sec-gauss-product derives the
one-dimensional case by completing the square and shows it in a figure.

::: {.example #ex-id-bumps title="Two bumps in one variable"}
@sec-fs-rbf-limit needed the integral over $c$ of a product of two
unnormalized bumps, $e^{-(x - c)^2/2s^2}\,e^{-(x' - c)^2/2s^2}$. As functions
of $c$, each bump is $\sqrt{2\pi s^2}$ times a Gaussian density with variance
$s^2$, centered at $x$ and at $x'$. By @eq-id-product the product is

$$
2\pi s^2 \cdot \N(x;\, x',\, 2s^2) \cdot \N\!\left(c;\, \tfrac{x + x'}{2},\, \tfrac{s^2}{2}\right).
$$

The last factor integrates to 1 over $c$, so the integral is
$2\pi s^2\,\N(x;\, x', 2s^2) = s\sqrt{\pi}\; e^{-(x - x')^2/4s^2}$, the value
found there by completing the square. The RBF kernel is the constant $Z$ of
a product of two bumps.
:::

**Where the book uses it.** Bayes' rule with a Gaussian prior and a Gaussian
likelihood is @eq-id-product, with $Z$ the model evidence (@sec-evidence).
Expectation propagation (@sec-ep) multiplies and divides Gaussian factors
with the same rule, one factor at a time.

## The expected maximum of two Gaussians {#sec-id-clark}

Several acquisition functions ask for the expected value of the larger of
two uncertain quantities. Expected improvement compares an uncertain value
with a known one (@sec-ei). The expected utility of the best option compares
two uncertain values of a utility (@sec-eubo). When the two quantities are
jointly Gaussian, the expectation has a closed form.

::: {.theorem #thm-id-clark title="Expected maximum of two Gaussians"}
Let $A$ and $B$ be jointly Gaussian with means $\mu_A, \mu_B$, variances
$v_A, v_B$, and covariance $c$. Write $\delta = \mu_A - \mu_B$ and
$s^2 = v_A + v_B - 2c$, the variance of $A - B$. If $s > 0$,

$$
\begin{aligned}
\E[\max\{A, B\}] ={}& \mu_A\,\Phi(\alpha) + \mu_B\,\Phi(-\alpha) \\
&+ s\,\phi(\alpha), \qquad \alpha = \delta / s,
\end{aligned}
$$ {#eq-id-clark}

where $\phi$ and $\Phi$ are the standard normal density and distribution
function. If $s = 0$, the expectation is $\max\{\mu_A, \mu_B\}$.
:::

::: {.proof}
1. For any two numbers, $\max\{A, B\} = B + \max\{A - B, 0\}$.
2. $D = A - B$ is a linear map of a Gaussian vector, so it is Gaussian
   (@eq-gauss-affine), with mean $\delta$ and variance
   $\Var[A] + \Var[B] - 2\Cov[A, B] = s^2$.
3. If $s > 0$, write $D = \delta + sZ$ with $Z$ standard normal. Then
   $\max\{D, 0\}$ vanishes unless $Z > -\delta/s$, and
   $\E[\max\{D, 0\}] = \int_{-\delta/s}^{\infty}(\delta + sz)\,\phi(z)\,\dd z$.
   The first part is $\delta\,(1 - \Phi(-\delta/s)) = \delta\,\Phi(\delta/s)$.
   For the second, $\phi'(z) = -z\,\phi(z)$, so $z\,\phi(z)$ has antiderivative
   $-\phi(z)$ and the integral of $z\,\phi(z)$ from $-\delta/s$ to infinity is
   $\phi(-\delta/s) = \phi(\delta/s)$. So
   $\E[\max\{D, 0\}] = \delta\,\Phi(\delta/s) + s\,\phi(\delta/s)$.
4. By linearity of expectation and step 1,
   $\E[\max\{A, B\}] = \mu_B + \delta\,\Phi(\delta/s) + s\,\phi(\delta/s)$.
   Since $\mu_B + \delta\,\Phi(\delta/s) = \mu_A\Phi(\delta/s) + \mu_B(1 - \Phi(\delta/s))$
   and $1 - \Phi(t) = \Phi(-t)$, this is @eq-id-clark.
5. If $s = 0$, $D$ equals the constant $\delta$, and
   $\max\{A, B\} = B + \max\{\delta, 0\}$ has expectation
   $\max\{\mu_A, \mu_B\}$.
:::

The formula is due to @clark1961greatest, whose paper gives exact results
for two jointly normal variables with any correlation and approximations,
by repeated application, for more than two. Step 3 is the computation behind
expected improvement, which @sec-ei carries out step by step
(@eq-acq-positive-part).

The formula can be read term by term. $\Phi(\delta/s)$ is the probability
that $A$ exceeds $B$, so the first two terms average the means, each
weighted by the probability that its variable is the larger. The third term
is a bonus for not knowing which is larger, and it depends on the
uncertainty only through $s$. Four consequences are used in the book.

- **Never below the better mean.** $\max\{D, 0\} \ge D$ and
  $\max\{D, 0\} \ge 0$, so $\E[\max\{D, 0\}] \ge \max\{\delta, 0\}$ and
  $\E[\max\{A, B\}] \ge \max\{\mu_A, \mu_B\}$.
- **Increasing in $s$.** Differentiating step 3 with respect to $s$, the
  terms from $\Phi$ and from $\phi'$ cancel and leave exactly
  $\phi(\delta/s) > 0$. More uncertainty about the difference is always worth
  more.
- **Correlation matters only through $s$.** Positive correlation between $A$
  and $B$ lowers $s$ and with it the expected maximum; negative correlation
  raises it. Two options that rise and fall together offer little choice.
- **Equal means.** With $\delta = 0$ the formula reduces to
  $\mu + s/\sqrt{2\pi}$: the bonus is about $0.4\,s$.

The figure shows the whole distribution of the maximum, of which
@eq-id-clark is the mean.

```{figure}
//| figure: id-clark
//| label: fig-id-clark
//| fig-cap: "The maximum of two jointly Gaussian values. Thin curves: the densities of $A$ and $B$. Shaded: the exact density of $\max\{A, B\}$. The solid vertical line is its mean, @eq-id-clark; the dashed line is the better of the two means. The readout checks the formula against a numerical integral of the shaded density and reports the bonus over the better mean."
```

**Press *Equal means*.** The bonus is $s/\sqrt{2\pi}$: with standard
deviations 1 and 0.5 and no correlation, $s = 1.12$ and the bonus is 0.446.

**Raise the correlation toward 0.95.** The bonus shrinks, because $s$ does.
With both standard deviations equal and the correlation near 1, the two
values move together, their difference is almost constant, and the maximum
is worth almost exactly the better mean.

**Lower the correlation toward $-0.95$.** Now one value is high when the
other is low, the maximum is almost always well above both means, and the
bonus is at its largest.

**Press *One certain option*.** With $B$ nearly constant, the shaded density
is the density of $A$ with everything below $B$ swept up to $B$. Its mean
is $\mu_B$ plus the expected improvement of $A$ over $\mu_B$. Expected
improvement is the special case of @eq-id-clark in which one of the two
values is known.

**Where the book uses it.** @eq-id-clark is the closed form of EUBO for a
pair of options, @eq-eubo-closed, with $A$ and $B$ the posterior utilities of
the two options. It is also the knowledge gradient when only two inputs are
in play (@sec-knowledge-gradient). There the two values are linear functions
of one standard normal variable, so they are perfectly correlated, and the
theorem applies with $s$ equal to the difference of their slopes in absolute
value.

## Exercises {#sec-id-exercises}

::: {.exercise #exr-id-sherman}
Let $\mathbf{1}$ be the vector of $n$ ones. Use @eq-id-sherman-morrison to
compute $(\sigma^2\mI + \mathbf{1}\mathbf{1}^\T)^{-1}\mathbf{1}$, and from it
the posterior variance at $x_0$ after $n$ noisy observations at $x_0$ under a
unit-amplitude kernel, as in @exr-noise-floor.

::: {.solution}
With $\mathbf{Z} = \sigma^2\mI$ and $\mathbf{u} = \mathbf{v} = \mathbf{1}$,
$\mathbf{Z}^{-1}\mathbf{1} = \mathbf{1}/\sigma^2$ and
$\mathbf{1}^\T\mathbf{Z}^{-1}\mathbf{1} = n/\sigma^2$. So

$$
\begin{aligned}
(\sigma^2\mI + \mathbf{1}\mathbf{1}^\T)^{-1}\mathbf{1}
&= \frac{\mathbf{1}}{\sigma^2} - \frac{\mathbf{1}\,(n/\sigma^2)}{\sigma^2\,(1 + n/\sigma^2)} \\
&= \frac{\mathbf{1}}{\sigma^2}\cdot\frac{1}{1 + n/\sigma^2}
= \frac{\mathbf{1}}{\sigma^2 + n}.
\end{aligned}
$$

All kernel values are 1, so $\mK = \mathbf{1}\mathbf{1}^\T$ and
$\vk(x_0) = \mathbf{1}$, and the posterior variance of @eq-gp-noisy is
$1 - \mathbf{1}^\T\mathbf{1}/(\sigma^2 + n) = \sigma^2/(\sigma^2 + n)$.
:::
:::

::: {.exercise #exr-id-det}
Use @eq-id-det-lemma to compute the determinant of
$\sigma^2\mI + \mathbf{1}\mathbf{1}^\T$ for $n$ observations, and with it the
complexity term $-\tfrac12\log\lvert\mK_y\rvert$ of @eq-kern-lml for a
unit-amplitude kernel with an infinitely long lengthscale. Compare with a
very short lengthscale, where $\mK_y = (1 + \sigma^2)\mI$.

::: {.solution}
With $\mathbf{Z} = \sigma^2\mI$, $\mathbf{U} = \mathbf{V} = \mathbf{1}$, and
$\mW = 1$:
$\det(\sigma^2\mI + \mathbf{1}\mathbf{1}^\T) = \sigma^{2n}\,(1 + n/\sigma^2) = \sigma^{2(n-1)}(\sigma^2 + n)$.
The complexity term is
$-\tfrac12\left[(n - 1)\log\sigma^2 + \log(\sigma^2 + n)\right]$. For small
noise this is large and positive: with $n = 7$ and $\sigma = 0.1$ it is
$-\tfrac12\left[6 \times (-4.61) + 1.95\right] = 12.8$. At a very short
lengthscale the determinant is $(1 + \sigma^2)^n$ and the term is
$-\tfrac{n}{2}\log(1 + \sigma^2) = -0.03$. The long lengthscale is charged
far less for complexity, because it expects all observations to be nearly
equal, a thin sliver of the space of data sets. It pays in the data-fit
term unless the observations really are nearly equal.
:::
:::

::: {.exercise #exr-id-max-iid}
Let $A$ and $B$ be independent standard normal variables. (a) Use
@eq-id-clark to find $\E[\max\{A, B\}]$. (b) Show that
$\E[\max\{A, B\}^2] = 1$ without integrating, and find the variance of the
maximum. (c) Check both numbers in @fig-id-clark.

::: {.solution}
(a) $\delta = 0$ and $s^2 = 2$, so the expectation is
$s\,\phi(0) = \sqrt{2}/\sqrt{2\pi} = 1/\sqrt{\pi} \approx 0.564$.
(b) $\max\{A, B\}^2 + \min\{A, B\}^2 = A^2 + B^2$, whose expectation is 2.
The pair $(-A, -B)$ has the same distribution as $(A, B)$, and
$\min\{A, B\} = -\max\{-A, -B\}$, so $\min\{A, B\}^2$ and $\max\{A, B\}^2$
have the same expectation, which must be 1. The variance is
$1 - 1/\pi \approx 0.682$: the maximum of two draws is less variable than
either draw. (c) Set both means to 0, both standard deviations to 1, and the
correlation to 0. The readout gives 0.564, and the shaded density is
visibly narrower than the two thin curves.
:::
:::

## Further reading {#further-reading .unnumbered}

- @petersen2012matrix is a free reference of matrix identities, including
  every one in this appendix and the derivative rules used in
  @sec-kern-gradient.
- @rasmussen2006gaussian, appendix A, lists the Gaussian and matrix
  identities used in Gaussian process regression, in the same notation as
  the rest of that book.
- @bishop2006pattern, section 2.3, derives the marginal, the conditional,
  and the linear-Gaussian pair at textbook length.
- @golub2013matrix is the standard reference on computing with matrices,
  including why factorizations are preferred to explicit inverses.
- @clark1961greatest is the original paper on the maximum of jointly normal
  variables.
