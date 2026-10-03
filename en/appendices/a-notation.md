---
status: done
synopsis: "The symbols used in the book, grouped by topic, each with the section that introduces it."
---

# Notation {#sec-notation}

The book uses one notation throughout, and this appendix collects it. Each
entry points to the section that introduces the symbol, where it is explained
in words before it is used in a formula. Where the literature uses several
conventions, the entry says which one the book follows.

A few typographic rules hold everywhere. Scalars are italic ($x$, $\sigma$),
vectors are bold lowercase ($\vx$, $\vy$), and matrices are bold uppercase
($\mK$, $\mL$). A transpose is written $^\T$. Inputs live in a domain $\X$,
usually the unit cube $[0, 1]^d$ after rescaling, where $d$ is the number of
inputs; when an index runs over the inputs, as for one lengthscale per input,
it is $j = 1, \dots, d$. A finite set of candidate inputs is also written
$\X$, with $|\X|$ elements, even where the papers the book reports write $D$. "Larger
is better" throughout: the book maximizes, and a function that is naturally
minimized, such as an error rate, is negated.

## Sets, vectors, and matrices {#sec-notation-linear}

::: {.table #tbl-notation-linear title="Linear algebra."}
| Symbol | Meaning | Introduced |
|---|---|---|
| $\R$, $\R^d$ | the real numbers; vectors of $d$ real numbers | @sec-vectors |
| $d$ | the number of inputs (dimension of $\X$) | @sec-usual-tools, @sec-vectors |
| $\vx$, $x_i$ | a vector and its $i$-th entry | @sec-vectors |
| $\vx^\T$, $\mA^\T$ | transpose of a vector and of a matrix | @sec-vectors, @sec-linalg-transpose |
| $\mI$ | identity matrix | @sec-linalg-composition |
| $\mA^{-1}$ | inverse of $\mA$ (computed by solving, never formed) | @sec-linalg-composition |
| $\mL$ | lower-triangular Cholesky factor, $\mL\mL^\T = \mA$ | @sec-linalg-cholesky-def |
| $\mA \backslash \mathbf{b}$ | the solution $\mathbf{z}$ of $\mA\mathbf{z} = \mathbf{b}$ | @sec-gp-computation |
| $\det \mA$, $\log\det\mA$ | determinant, and its logarithm (from the Cholesky diagonal) | @sec-determinants, @sec-linalg-logdet |
| $\tr \mA$ | trace, the sum of the diagonal | @sec-info-kl-examples, @sec-kern-gradient |
:::

## Probability {#sec-notation-probability}

::: {.table #tbl-notation-probability title="Probability."}
| Symbol | Meaning | Introduced |
|---|---|---|
| $\Prob(A)$ | probability of an event | @sec-prob-axioms |
| $p(x)$, $p(x \given y)$ | density or mass function; conditional on $y$ | @sec-prob-random-variables |
| $X \sim p$ | $X$ is distributed according to $p$ | @sec-prob-random-variables |
| $\D$ | the observed data | @sec-bayes-rule |
| $\E[X]$, $\E_n[\cdot]$ | expectation; expectation under the posterior after $n$ observations | @sec-prob-expectation-def, @sec-acq-definition |
| $\Var[X]$, $\Cov[X, Y]$ | variance; covariance | @sec-prob-variance, @sec-prob-covariance |
| $\N(\mu, \sigma^2)$, $\N(\vmu, \mSigma)$ | Gaussian with mean and variance; with mean vector and covariance matrix | @sec-gaussian-1d, @sec-gaussian-nd |
| $\phi(z)$, $\Phi(z)$ | standard normal density and cumulative distribution function | @sec-gauss-standard |
| $\operatorname{sigmoid}(z)$ | logistic function $1/(1 + e^{-z})$, the Bradley-Terry link | @sec-bradley-terry |
| $H(p)$, $H(X)$ | entropy of a distribution or of a random variable | @sec-info-entropy-def |
| $\KL(p \,\Vert\, q)$ | Kullback-Leibler divergence | @sec-info-kl-def |
| $I(X; Y)$ | mutual information | @sec-mutual-information |
| $\delta$ | failure probability: a high-probability statement holds with probability at least $1 - \delta$ | @sec-regret-concentration |
| $R$-sub-Gaussian | a mean-zero $Z$ with $\E[e^{\lambda Z}] \le e^{\lambda^2 R^2/2}$ for all $\lambda$; this $R$ is not the regret $R_T$ | @sec-regret-concentration |
:::

The book writes $\N(\mu, \sigma^2)$ with the *variance* as the second argument,
as most statistics texts do, and states the standard deviation $\sigma$
separately where it matters.

## Gaussian processes {#sec-notation-gp}

::: {.table #tbl-notation-gp title="Gaussian processes."}
| Symbol | Meaning | Introduced |
|---|---|---|
| $f$ | the unknown objective (or latent utility) | @sec-cost-of-evaluation |
| $\X$ | the domain of inputs | @sec-cost-of-evaluation |
| $\lvert\X\rvert$ | the number of candidates, when the domain is a finite set | @sec-ucb |
| $\GP(m, k)$ | Gaussian process with mean function $m$ and kernel $k$ | @sec-gp-definition |
| $k(\vx, \vx')$ | kernel (covariance function) | @sec-linalg-inner-products, @sec-fs-prior-values |
| $\ell$, $\ell_j$ | lengthscale; one lengthscale per input $j$ (ARD) | @sec-linalg-inner-products, @sec-ard |
| $\sigma_f^2$ | squared amplitude, $k(\vx, \vx)$ for a stationary kernel | @sec-fs-rbf-limit |
| $\sigma_n^2$ | observation noise variance | @sec-prob-total-variance, @sec-gp-noise |
| $X$, $\vy$ | observed inputs and observed values | @sec-gp-conditioning |
| $\mK$ | kernel matrix of the observed inputs, $[\mK]_{ij} = k(\vx_i, \vx_j)$ | @sec-gp-definition, @sec-gp-conditioning |
| $\vk(\vx)$ | covariances between $\vx$ and the observed inputs | @sec-gp-conditioning |
| $\mu(\vx)$, $\mu_n(\vx)$ | posterior mean (after $n$ observations) | @sec-gp-conditioning, @sec-bo-algorithm |
| $\sigma^2(\vx)$, $\sigma_n^2(\vx)$ | posterior variance of the latent value (after $n$ observations) | @sec-gp-conditioning, @sec-bo-algorithm |
| $\bm{\alpha}$ | weights $(\mK + \sigma_n^2\mI)^{-1}\vy$ in the posterior mean | @sec-linalg-why-not-invert, @sec-gp-bumps |
| $M$ | number of features of a linear model; for random Fourier features, the number of frequencies drawn | @sec-weights-to-functions, @sec-ka-rff |
| $\mathcal{H}_k$, $\langle f, g\rangle_k$, $\lVert f\rVert_k$ | reproducing kernel Hilbert space (RKHS) of $k$, its inner product, and its norm | @sec-ka-rkhs |
| $B$ | bound on the RKHS norm; some papers bound $\lVert f\rVert_k$, others $\lVert f\rVert_k^2$ | @sec-ka-norm-bound |
| $\mathcal{K}$, $q(\vx)$ | integral operator of a kernel, and the weighting density it integrates against | @sec-ka-operator |
| $\lambda_i$, $\varphi_i$ | eigenvalues and eigenfunctions of $\mathcal{K}$ (Mercer's theorem) | @sec-ka-mercer |
| $s(\boldsymbol{\omega})$, $p(\boldsymbol{\omega})$ | spectral density of a stationary kernel; the same normalized to a probability density | @sec-ka-bochner |
:::

One collision is worth knowing about. $\sigma_n^2$, with no argument, is the
noise variance, following @rasmussen2006gaussian; its subscript stands for
noise. $\sigma_n(\vx)$, always written with its argument, is the posterior
standard deviation after $n$ observations, following @srinivas2010gaussian.
The argument $(\vx)$ tells them apart. Where both appear in one formula
(@sec-knowledge-gradient), the noise variance is written
$\sigma_\varepsilon^2$.
In @sec-kernel-analysis, $\lambda_i$ with an index is an eigenvalue and $q$ a
weighting density; elsewhere $\lambda$ is the inverse Mills ratio or a lapse
rate, and $q$ the number of options in a query.

## Optimization and preferences {#sec-notation-bo}

::: {.table #tbl-notation-bo title="Optimization and preferences."}
| Symbol | Meaning | Introduced |
|---|---|---|
| $\vx^\star$, $f^\star$ | a maximizer and the maximum | @sec-info-eig-limits, @sec-bo-problem |
| $f^*_n$ | the incumbent, the best value observed after $n$ evaluations | @sec-pi |
| $a_n(\vx)$ | an acquisition function, after $n$ observations | @sec-bo-algorithm |
| $\PI$, $\EI$, $\UCB$ | probability of improvement, expected improvement, upper confidence bound | @sec-pi, @sec-ei, @sec-ucb |
| $\beta$, $\beta_t$ | exploration weight of UCB, written $\mu_n(\vx) + \beta^{1/2}\sigma_n(\vx)$ | @sec-loop-first-acq, @sec-ucb |
| $\xi$ | improvement margin in PI and EI (a choice of experiment in @sec-expected-information-gain) | @sec-pi |
| $r_t$, $R_T$ | instantaneous regret; cumulative regret over $T$ rounds | @sec-regret-definitions |
| $\gamma_T$ | maximum information gain of a kernel after $T$ observations | @sec-info-gamma |
| $\vx \succ \vx'$ | $\vx$ is preferred to $\vx'$ | @sec-thurstone |
| $g$ or $f$ | latent utility of a person (the book uses $f$ when the GP machinery is shared) | @sec-pref-model |
| $\sigma$ (in a link) | noise scale of a person's evaluation of one option | @sec-thurstone |
| $\tau$ | scale of the logistic link, $\operatorname{sigmoid}\big((g_i - g_j)/\tau\big)$ | @sec-bradley-terry |
| $\lambda(z)$ | inverse Mills ratio $\phi(z)/\Phi(z)$, only in the chapters that define it | @sec-ep, @sec-pref-laplace |
| $\lambda_{\text{lapse}}$ | lapse rate, the probability that an answer is a random slip | @sec-query-confidence |
| $\mW$ | negative Hessian of the log-likelihood; for pairwise answers, a weighted graph Laplacian | @sec-laplace, @sec-pref-laplace |
| $\EUBO(\vx_1, \vx_2)$ | expected utility of the best option | @sec-eubo |
| $q$ | number of options shown in one query (qEUBO) | @sec-eubo |
:::

Regret bounds are stated with $\tilde O(\cdot)$, which hides logarithmic
factors, and every rate in the book is given with its assumptions and its
regret unit (@sec-theory-rates).
