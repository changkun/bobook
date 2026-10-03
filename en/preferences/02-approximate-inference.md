---
status: done
synopsis: "Comparisons make the posterior non-Gaussian. Using one utility difference whose exact posterior can be drawn, the chapter derives and compares the Laplace approximation, expectation propagation, variational inference, and sampling, shows that the exact answer is a skew-normal (a skew Gaussian process in general) whose skew lives only along compared directions, and reports how much the choice matters."
sources: ["Rasmussen and Williams 2006 ch. 3", "Kuss and Rasmussen 2005", "Nickisch and Rasmussen 2008", "Minka 2001", "Murray et al. 2010", "Benavoli et al. 2021", "Takeno et al. 2023"]
---

# When the Posterior Is Not Gaussian {#sec-approx-inference}

Gaussian process regression owes its tidy formulas to one fact: a Gaussian
prior combined with Gaussian observation noise gives a Gaussian posterior. The
comparison models of @sec-comparisons break that fact. Their likelihood is a
probit or logistic curve, not a Gaussian, and the posterior over the utility is
no longer Gaussian. Everything downstream, the predictive mean and band, the
probability of the next answer, and the acquisition functions of @sec-pbo,
needs that posterior.

This chapter works almost entirely with the smallest case that shows the
difficulty: one utility difference, a Gaussian prior, and a few comparisons.
Its exact posterior can be computed on a grid and drawn, so each approximation
can be checked against the truth. We then move to two and more latent values,
where the exact answer turns out to be skewed only in the directions that
comparisons touch, and end with the evidence on how much the choice of
approximation matters in practice.

## Non-Gaussian likelihoods {#sec-non-gaussian-likelihoods}

Recall why regression was easy. With a prior $\vf \sim \N(\mathbf{0}, \mK)$
and observations $\vy = \vf + \bm{\varepsilon}$ with Gaussian noise, Bayes'
rule multiplies two Gaussian densities in $\vf$. The product of Gaussian
densities is again a Gaussian density up to a constant (@sec-gaussian-sums),
so the posterior has a closed form and so does its normalizing constant, the
marginal likelihood. Prior and likelihood form a *conjugate* pair.

A comparison does not. Write $\vf$ for the utilities at the inputs that have
been compared, and suppose $m$ answers have been recorded, the $k$-th saying
that input $v_k$ was preferred to input $u_k$. With the probit model of
@eq-cmp-case-v, Bayes' rule gives

$$
p(\vf \mid \D) = \frac{1}{Z}\, \N(\vf;\, \mathbf{0}, \mK) \prod_{k=1}^{m} \Phi\!\left(\frac{f_{v_k} - f_{u_k}}{\sqrt{2}\,\sigma}\right),
\qquad
Z = \int \N(\vf;\, \mathbf{0}, \mK) \prod_{k=1}^{m} \Phi\!\left(\frac{f_{v_k} - f_{u_k}}{\sqrt{2}\,\sigma}\right) \dd\vf.
$$ {#eq-approx-posterior}

The prior is Gaussian, but each likelihood factor is an S-shaped curve in one
direction of $\vf$, the direction of the difference $f_{v_k} - f_{u_k}$. The
product is not Gaussian, and $Z$ is the probability that a Gaussian vector
lands in a region bounded by $m$ soft walls, an $m$-dimensional integral with
no closed form in general (@sec-no-closed-form). Gaussian process
classification, in which each input carries a yes-or-no label $y_i = \pm 1$
instead of a number, has the same structure, with one factor
$\Phi(y_i f(\vx_i))$ per labeled input [@rasmussen2006gaussian, ch. 3], so
the methods of this chapter were developed and tested there first.

### The smallest case {#sec-approx-smallest}

Strip the problem down to one number. Let $\Delta = g(A) - g(B)$ be the
utility difference between two options, as in @sec-comparison-information,
with a Gaussian prior $\Delta \sim \N(0, v_0)$. Here $v_0$ is a variance;
@eq-cmp-predictive described the same kind of belief by its standard
deviation $v$, so $v_0 = v^2$. Suppose a person says once that $A$ is better.
Write $s = \sqrt{2}\,\sigma$ for the noise on the difference. The posterior is

$$
p(\Delta \mid A \succ B) = 2\, \N(\Delta;\, 0, v_0)\, \Phi(\Delta/s).
$$ {#eq-approx-skew-normal}

The factor 2 is $1/Z$: by @eq-cmp-predictive with a belief centered at zero,
the prior probability of the answer is $\Phi(0) = 1/2$. A Gaussian density multiplied by
a normal distribution function is called a **skew-normal** density, a family
introduced by @azzalini1985class. It keeps the upper tail of the prior and
cuts away the lower one, softly when the noise is large and sharply when it is
small. Its mean has a closed form.

::: {.derivation title="The mean of the one-comparison posterior"}
1. For $\Delta \sim \N(0, v_0)$ and a differentiable $h$, integration by parts
   gives $\E[\Delta\, h(\Delta)] = v_0\, \E[h'(\Delta)]$, because the derivative of the density
   $\N(\Delta; 0, v_0)$ is $-(\Delta/v_0)\, \N(\Delta; 0, v_0)$. This is *Stein's lemma*.
2. Take $h(\Delta) = \Phi(\Delta/s)$, so $h'(\Delta) = \phi(\Delta/s)/s$. Then
   $\int \Delta\, \N(\Delta; 0, v_0)\, \Phi(\Delta/s)\, \dd\Delta = v_0\, \E[\phi(\Delta/s)/s]$.
3. $\phi(\Delta/s)/s$ is the density $\N(0; \Delta, s^2)$ seen as a function of $\Delta$, so
   its prior expectation is the density at 0 of $\Delta - \eta$ with
   $\eta \sim \N(0, s^2)$ independent (@sec-gaussian-sums), namely
   $\N(0;\, 0, v_0 + s^2) = 1/\sqrt{2\pi(v_0 + s^2)}$.
4. Divide by $Z = 1/2$: the posterior mean is
   $\E[\Delta \mid A \succ B] = 2 v_0 / \sqrt{2\pi(v_0 + s^2)} = \sqrt{2/\pi}\; v_0 / \sqrt{v_0 + s^2}$.
:::

With $v_0 = 1$, the mean grows from about 0.46 at $\sigma = 1$ to
$\sqrt{2/\pi} \approx 0.80$ as the noise vanishes, and the standard deviation
falls from about 0.89 to 0.60 (@exr-approx-variance). The *mode*, the peak of
the density, behaves differently: it solves $\Delta/v_0 = \phi(\Delta/s)/(s\,\Phi(\Delta/s))$,
and as $s$ shrinks it slides toward zero, to 0.05 at $\sigma = 0.01$. In the
noise-free limit the posterior is the prior's upper half, a half-normal, whose
peak sits at the cut while its mass lies well to the right of it.

```{figure}
//| figure: approx-posterior-1d
//| label: fig-approx-exact
//| fig-cap: "The posterior of a utility difference Δ = g(A) − g(B) after comparisons of A with B, with a standard normal prior. The shaded curve is the exact posterior, computed on a fine grid; the dashed and solid vertical lines mark its mode and its mean. The table gives the mean, the standard deviation, and the probability that B is in fact better, P(Δ < 0). Turn on the prior and the likelihood to see the product that forms the posterior."
method: exact
factors: true
```

Some things to try:

- **Shrink the noise.** Drag $\sigma$ toward 0.01. The likelihood becomes a
  step, the posterior becomes the half-normal, and the mode and the mean
  separate: the mode goes to the cut, the mean stays near 0.80.
- **Raise the noise.** At $\sigma = 2$ the likelihood is a gentle slope, the
  posterior is a slightly shifted bell, and mode and mean nearly coincide.
- **Let both win.** With three wins for $A$ and three for $B$ the posterior is
  pinned from both sides and is close to a Gaussian centered at zero. Skew
  comes from one-sided evidence.
- **Let $A$ keep winning.** Each further win pushes the mass up, but with small
  noise the left edge stays sharp: after a run of identical answers the
  evidence says "$A$ is better" more firmly, not "by how much".

This lopsided shape is what every method below has to summarize, usually with a
single Gaussian.

## The Laplace approximation {#sec-laplace}

The simplest summary puts a Gaussian at the posterior's peak and gives it the
peak's curvature. A bell shape is determined by where it is highest and how
fast it falls off from there; if the posterior is close to a bell, those two
facts recover it. This is the **Laplace approximation**, named after Laplace's
method for approximating integrals, and @tierney1986accurate showed how far it
goes for Bayesian computation: approximate posterior moments and marginal
densities need only a maximization and the curvature at the maximum.

Write $\Psi(\vf) = \log p(\D \mid \vf) + \log \N(\vf; \mathbf{0}, \mK)$ for the
log of the unnormalized posterior, and $\hat\vf$ for its maximizer, the
**mode** or *maximum a posteriori* estimate.

::: {.derivation title="The Laplace approximation"}
1. Write $\nabla\Psi$ for the gradient of $\Psi$, the vector of its first
   derivatives with respect to the entries of $\vf$, and $\nabla\nabla\Psi$
   for its *Hessian*, the matrix of its second derivatives. Expand $\Psi$ to
   second order around $\hat\vf$ (Taylor's theorem):
   $\Psi(\vf) \approx \Psi(\hat\vf) + \nabla\Psi(\hat\vf)^\T(\vf - \hat\vf) - \tfrac12 (\vf - \hat\vf)^\T \mA\, (\vf - \hat\vf)$,
   with $\mA = -\nabla\nabla\Psi(\hat\vf)$.
2. At a maximum the gradient vanishes, $\nabla\Psi(\hat\vf) = \mathbf{0}$, so the
   linear term drops.
3. The Hessian of the log prior is $-\mK^{-1}$. Write
   $\mW = -\nabla\nabla \log p(\D \mid \vf)$ at $\hat\vf$ for the curvature of
   the log likelihood; then $\mA = \mK^{-1} + \mW$.
4. Exponentiate: $p(\vf \mid \D) \propto e^{\Psi(\vf)} \approx e^{\Psi(\hat\vf)} \exp\!\big(-\tfrac12 (\vf - \hat\vf)^\T \mA\, (\vf - \hat\vf)\big)$,
   which is a Gaussian density up to a constant.
5. Hence $q(\vf) = \N\big(\hat\vf,\, (\mK^{-1} + \mW)^{-1}\big)$.
:::

$$
p(\vf \mid \D) \approx \N\!\left(\hat\vf,\; (\mK^{-1} + \mW)^{-1}\right).
$$ {#eq-approx-laplace}

Two things are needed: the mode and the curvature there. For the probit
likelihood the log likelihood is concave, because $\log\Phi$ is concave, and
the log prior is a concave quadratic, so $\Psi$ has a single maximum and
**Newton's method** finds it quickly. Newton's method replaces the function by
its quadratic approximation at the current point and jumps to that quadratic's
maximum.

::: {.derivation title="Newton's step for the mode"}
1. The gradient is $\nabla\Psi(\vf) = \nabla \log p(\D \mid \vf) - \mK^{-1}\vf$
   and the Hessian is $\nabla\nabla\Psi(\vf) = -(\mK^{-1} + \mW)$, with $\mW$
   now evaluated at the current $\vf$.
2. The maximizer of the local quadratic is
   $\vf_{\text{new}} = \vf - (\nabla\nabla\Psi)^{-1}\nabla\Psi = \vf + (\mK^{-1} + \mW)^{-1}\big(\nabla \log p(\D \mid \vf) - \mK^{-1}\vf\big)$.
3. Write $\vf = (\mK^{-1} + \mW)^{-1}(\mK^{-1} + \mW)\vf$ and collect terms:
   $\vf_{\text{new}} = (\mK^{-1} + \mW)^{-1}\big(\mW\vf + \nabla \log p(\D \mid \vf)\big)$.
4. Avoid inverting $\mK$: since $(\mK^{-1} + \mW)\,\mK = \mI + \mW\mK$, we have
   $(\mK^{-1} + \mW)^{-1} = \mK(\mI + \mW\mK)^{-1}$, so
   $\vf_{\text{new}} = \mK(\mI + \mW\mK)^{-1}\big(\mW\vf + \nabla \log p(\D \mid \vf)\big)$.
:::

$$
\vf_{\text{new}} = \mK\,(\mI + \mW\mK)^{-1}\big(\mW\vf + \nabla \log p(\D \mid \vf)\big).
$$ {#eq-approx-newton}

Starting from $\vf = \mathbf{0}$ and repeating @eq-approx-newton, halving the
step whenever $\Psi$ fails to increase, converges in a handful of iterations.
For classification, where $\mW$ is diagonal, @rasmussen2006gaussian give a
numerically stable version as their Algorithm 3.1. For comparisons $\mW$ is
not diagonal; @sec-pref-laplace works out its structure.

The same expansion approximates the normalizing constant $Z$, the marginal
likelihood used to fit hyperparameters (@sec-marginal-likelihood). Integrating
the Gaussian of step 4 gives

$$
\log Z \approx \log p(\D \mid \hat\vf) - \tfrac12\, \hat\vf^\T \mK^{-1} \hat\vf - \tfrac12 \log\left|\mI + \mK\mW\right|,
$$ {#eq-approx-laplace-evidence}

the Laplace evidence that @chu2005preference used for preferences (their
equation 12) and that BoTorch maximizes to fit its preference model
(@sec-pairwisegp).

### Where it goes wrong {#sec-approx-laplace-wrong}

The figure below adds the Laplace Gaussian to the exact posterior.

```{figure}
//| figure: approx-posterior-1d
//| label: fig-approx-laplace
//| fig-cap: "The Laplace approximation (blue) to the posterior of a utility difference: a Gaussian at the mode with the curvature there. Shrink the noise toward 0.01 and compare its mean, its standard deviation, and its probability that B is better with the exact values in the table."
method: laplace
sigma: 0.1
```

At moderate noise the blue curve sits close to the exact one. As the noise
shrinks it fails in a specific way. At $\sigma = 0.01$ the exact posterior has
mean 0.80, standard deviation 0.60, and gives $B$ a probability of 0.004 of
being better. The Laplace Gaussian is centered at the mode, 0.05, with
standard deviation 0.27, and gives $B$ a probability of 0.43: after a single
nearly noise-free answer it is almost as unsure of the order as before
(@exr-approx-laplace-limit). The mode sits at the edge of the mass, and a
Gaussian centered there must spill over the edge.

The general lesson was established for classification.
@kuss2005assessing compared Laplace's method and expectation propagation with
long sampling runs on binary Gaussian process classifiers and found that
Laplace's method "systematically underestimates the mean", so that the
approximate posterior over latent functions has "too small amplitude" and its
predictive probabilities are over-conservative, although the sign of the
latent function is mostly right; they concluded that it is "so inaccurate that
we advise against its use, especially when predictive probabilities are to be
taken seriously." The same pattern appears for comparisons: with nearly
noise-free duels, the mode can be very far from the mean
[@takeno2023practicalc].

Laplace's method survives because it is fast and simple. In robot reward
learning from pairwise preferences, @byk2020active described expectation
propagation as "more accurate than Laplace approximation" but "slower in
practice", and chose Laplace for its computational efficiency. It is also the
default in BoTorch (@sec-pairwisegp).

## Expectation propagation {#sec-ep}

The Laplace approximation looks at one point of the posterior. A better
Gaussian would match the posterior's *mean and variance*, its mass rather than
its peak. Computing those moments for the full posterior is as hard as the
original problem, but computing them for a posterior with only one non-Gaussian
factor is easy. **Expectation propagation** (EP) builds the approximation from
such one-factor problems [@minka2001expectation].

EP replaces each likelihood factor $t_k(\vf)$, called a *site*, by an
unnormalized Gaussian $\tilde t_k(\vf)$ in the same direction, so that the
approximate posterior $q(\vf) \propto \N(\vf; \mathbf{0}, \mK) \prod_k
\tilde t_k(\vf)$ is Gaussian. It then refines one site at a time.

::: {.algorithm #alg-approx-ep title="Expectation propagation"}
Input: prior $\N(\mathbf{0}, \mK)$, sites $t_1, \dots, t_m$.

1. Initialize every site approximation to a constant, so that $q$ is the
   prior.
2. Pick a site $k$ and remove its approximation, forming the *cavity*
   $q_{\setminus k}(\vf) \propto q(\vf) / \tilde t_k(\vf)$, a Gaussian.
3. Multiply in the exact factor, forming the *tilted* distribution
   $\hat p_k(\vf) \propto q_{\setminus k}(\vf)\, t_k(\vf)$.
4. Compute the mean and covariance of $\hat p_k$.
5. Choose the new $\tilde t_k$ so that $q_{\setminus k}\, \tilde t_k$ has
   exactly those moments, and update $q$.
6. Repeat steps 2 to 5 over all sites, in sweeps, until the site
   approximations stop changing.
:::

The tilted distribution in step 3 has one non-Gaussian factor, which acts
along one direction, so its moments reduce to a one-dimensional problem. For a
probit site the answer is in closed form. Let the cavity give the difference
$\Delta$ the Gaussian $\N(\mu, v)$, with $v$ a variance like $v_0$, and let
the site be $\Phi(y\, \Delta / s)$ with $y = +1$ if $A$ won and $-1$ if $B$
won.

::: {.derivation title="Moments of a probit site"}
1. The normalizer is $Z(\mu) = \int \N(\Delta; \mu, v)\, \Phi(y\Delta/s)\, \dd\Delta = \Phi(z)$
   with $z = y\mu / \sqrt{s^2 + v}$, by the argument of @eq-cmp-predictive.
2. Differentiating under the integral, $\partial_\mu \N(\Delta; \mu, v) = \tfrac{\Delta - \mu}{v}\, \N(\Delta; \mu, v)$,
   so $\partial_\mu \log Z = \E_{\hat p}[\Delta - \mu]/v$, and the tilted mean is
   $\mu + v\, \partial_\mu \log Z$.
3. Differentiating once more gives the tilted variance
   $v + v^2\, \partial^2_\mu \log Z$.
4. With $\lambda = \phi(z)/\Phi(z)$, the chain rule gives
   $\partial_\mu \log Z = y\lambda / \sqrt{s^2 + v}$ and
   $\partial^2_\mu \log Z = -\lambda(z + \lambda)/(s^2 + v)$, using
   $\tfrac{\dd}{\dd z}\lambda = -\lambda(z + \lambda)$.
5. Hence the tilted mean is $\mu + y v \lambda / \sqrt{s^2 + v}$ and the
   tilted variance is $v - v^2 \lambda (z + \lambda)/(s^2 + v)$.
:::

These are the same quantities, with $s = 1$, as in the expectation propagation
algorithm for probit classification [@rasmussen2006gaussian, ch. 3]. Each site
update is a rank-one change to the covariance, which costs $O(n^2)$ for $n$
latent values, so a sweep over $m$ sites costs $O(mn^2)$, comparable to a
Newton iteration when $m$ is of the order of $n$.

```{figure}
//| figure: approx-posterior-1d
//| label: fig-approx-ep
//| fig-cap: "Expectation propagation (green) against the exact posterior. With one comparison, EP matches the exact mean and standard deviation; with several comparisons on the same pair it is close but not exact. Compare its P(Δ < 0) with the exact value at small noise."
method: ep
sigma: 0.03
```

With a single comparison there is a single site, the tilted distribution is the
exact posterior, and EP returns its exact mean and variance: at $\sigma =
0.03$, mean 0.80 and standard deviation 0.60. That is far better than
Laplace's 0.12 and 0.32. Matching moments has a cost of its own, though. A
Gaussian with the right mean and variance still puts mass where the exact
posterior has none: EP gives $B$ a probability of 0.093 of being better, against
an exact value of 0.013. With five wins at $\sigma = 0.1$, several sites act on
the same direction and EP is no longer exact (mean 0.93 and standard deviation
0.45, against 0.90 and 0.58).

That small misplacement matters for comparisons specifically.
@takeno2023practicalc took a long run of a sampling method as ground truth
(10,000 samples from the Gibbs sampler of @sec-mcmc, after discarding the
first 1,000 and keeping every tenth), and found that EP estimates means and
credible intervals very accurately but, for a duel already observed with
$\vx_w$ beating $\vx_l$, overestimates the probability that
$f(\vx_w) \le f(\vx_l)$. On the Ackley function, a standard test function, it
underestimated duel probabilities near 0 or 1, and its
estimates did not keep the true ordering between pairs, which can change which
pair an acquisition function picks.

For classification the verdict on EP is favorable. Kuss and Rasmussen found its
predictive probabilities and marginal likelihood estimates very close to those
of long sampling runs [@kuss2005assessing], and a broader comparison of
approximations for binary Gaussian process classification concluded that "the
Expectation Propagation algorithm is almost always the method of choice unless
the computational budget is very tight" [@nickisch2008approximations]. Unlike
Newton's method on a concave function, EP is not guaranteed to improve an
objective at every step, so implementations damp the site updates when they
oscillate.

## Variational inference {#sec-vi}

A third route turns approximation into optimization. Pick a family of simple
distributions, Gaussians here, and find the member closest to the posterior,
measuring closeness by the Kullback-Leibler divergence $\KL(q \,\|\, p)$
(@sec-kl). The divergence to the posterior cannot be computed directly,
because it contains the unknown $\log Z$, but it can be minimized anyway.

::: {.derivation title="The evidence lower bound"}
1. By Bayes' rule, $\log p(\vf \mid \D) = \log p(\D \mid \vf) + \log p(\vf) - \log Z$.
2. Take the expectation under $q$ of $\log q(\vf) - \log p(\vf \mid \D)$:
   $\KL(q \,\|\, p(\cdot \mid \D)) = \E_q[\log q(\vf) - \log p(\vf)] - \E_q[\log p(\D \mid \vf)] + \log Z$.
3. The first expectation is $\KL(q \,\|\, p)$ to the prior. Rearranging,
   $\log Z = \underbrace{\E_q[\log p(\D \mid \vf)] - \KL(q \,\|\, p)}_{\mathcal{L}(q)} + \KL(q \,\|\, p(\cdot \mid \D))$.
4. The last term is nonnegative, so $\mathcal{L}(q) \le \log Z$, and since
   $\log Z$ does not depend on $q$, maximizing $\mathcal{L}$ minimizes the
   divergence to the posterior.
:::

$\mathcal{L}(q)$ is the **evidence lower bound** (ELBO). For a Gaussian $q$ and
a probit likelihood, every term is cheap: the divergence between two Gaussians
has a closed form, and the expected log likelihood is a sum of one-dimensional
integrals, one per comparison, each over the Gaussian that $q$ assigns to a
difference, so standard gradient-based optimizers can maximize it.

The direction of the divergence decides the character of the fit. $\KL(q \,\|\,
p)$ averages $\log(q/p)$ under $q$, so it is enormous wherever $q$ puts mass
and $p$ has almost none. The optimal $q$ therefore stays inside the posterior's
support and tends to be too narrow. Expectation propagation's local moment
update instead chooses the Gaussian closest to the tilted distribution in the
other direction, $\KL(\hat p_k \,\|\, q)$, which punishes $q$ for missing mass
and tends to make it too wide.

```{figure}
//| figure: approx-posterior-1d
//| label: fig-approx-vi
//| fig-cap: "Gaussian variational inference (magenta), the Gaussian with the highest evidence lower bound, against the exact posterior. At small noise it keeps almost no mass on the impossible side, Δ < 0, at the price of a standard deviation about half the exact one. Choose All to compare it with Laplace and EP."
method: vi
sigma: 0.01
```

At $\sigma = 0.01$ the variational Gaussian has mean 0.88 and standard
deviation 0.30, against the exact 0.80 and 0.60, and gives $B$ a probability of
0.002 of being better, close to the exact 0.004. It respects the hard edge that
EP crosses, but it halves the uncertainty. Each method errs where its criterion
is blind.

Variational inference is the approximation that scales. With *inducing
points*, a small set of pseudo-inputs that summarize the function
[@titsias2009variational], and stochastic optimization, it handles thousands of
comparisons and any likelihood whose expected log can be estimated. That is why
it underlies much of the recent preference modeling: crowd preference learning
with thousands of users and items [@simpson2020scalable], top-$k$ rankings
[@nguyen2021top], choice functions [@benavoli2023choice], response times
[@shvartsman2024response], mixed likelihoods that combine comparisons with
confidence ratings, in a 2025 preprint [@wu2025mixed], and the experiments of qEUBO
[@astudillo2023qeubob]. As of September 2026 we found no systematic assessment
of its error for preference likelihoods comparable to those for Laplace and
EP.

## Sampling {#sec-mcmc}

The three methods so far replace the posterior by a Gaussian. **Markov chain
Monte Carlo** (MCMC) methods instead draw a sequence of samples whose
distribution converges to the posterior itself. Any quantity of interest, the
mean, a credible interval, the probability that one option beats another, is
then estimated by averaging over the samples. The answer becomes exact as the
number of samples grows, at the price of computation and of having to judge
when a chain has run long enough.

For a Gaussian prior times a likelihood, **elliptical slice sampling** is the
natural choice. @murray2010elliptical designed it for "models with multivariate
Gaussian priors": it has "simple, generic code", "no free parameters" to tune,
and "works well for a variety of Gaussian process based models". Its idea is to
move along an ellipse that passes through the current state and a fresh draw
from the prior, so every proposal is already plausible under the prior and only
the likelihood needs checking.

::: {.algorithm #alg-approx-ess title="One step of elliptical slice sampling"}
Input: current state $\vf$, prior $\N(\mathbf{0}, \mK)$, log likelihood $\ell$.

1. Draw $\bm{\nu} \sim \N(\mathbf{0}, \mK)$, which defines the ellipse
   $\vf' (\theta) = \vf\cos\theta + \bm{\nu}\sin\theta$.
2. Draw $u$ uniform on $(0, 1)$ and set the threshold $\log y = \ell(\vf) + \log u$.
3. Draw $\theta$ uniform on $[0, 2\pi)$ and set the bracket $[\theta - 2\pi, \theta]$.
4. If $\ell(\vf'(\theta)) > \log y$, accept $\vf'(\theta)$ and stop.
5. Otherwise shrink the bracket toward zero, replacing the end on the same side
   of zero as $\theta$, draw a new $\theta$ uniformly in it, and return to step 4.
:::

The shrinking bracket always contains $\theta = 0$, the current state, so the
loop terminates. Comparisons add a refinement. Because the probit factor is
the probability that a Gaussian noise variable falls below the utility
difference, the exact posterior is the marginal of a Gaussian restricted to a
region cut out by linear constraints. @benavoli2021preferentialb sample it with
LinESS, a rejection-free elliptical slice sampler for linearly truncated
Gaussians, at a cost of $O(n^3)$ time and $O(n^2)$ memory in the number of
comparisons. @takeno2023practicalc used Gibbs sampling, which redraws one
variable at a time from its distribution given all the others, for the same
distribution and found it faster: about 0.54 against 1.61 seconds in their
Table 1, though LinESS should win when the number of truncations far exceeds
the dimension.

```{figure}
//| figure: approx-posterior-1d
//| label: fig-approx-mcmc
//| fig-cap: "Elliptical slice sampling for the posterior of a utility difference. The violet histogram shows the samples; the table compares their mean, standard deviation, and share below zero with the exact values. Press More samples to go from 200 to 1,000 and 5,000, and New chain to rerun with another seed."
method: mcmc
sigma: 0.03
samples: 200
```

With 200 samples the histogram is ragged and, with the default seed, the mean
is off by almost 0.2: successive samples of a Markov chain are correlated, so
200 of them carry less information than 200 independent draws. With 1,000 the
error is under 0.05, and with 5,000 the estimates agree with the exact column
to about two digits.
Sampling is the only method in this chapter whose error can be driven to zero
by spending more computation. That is why studies of the other methods use it
as the reference.

## An exact answer: the skew Gaussian process {#sec-skew-gp}

The one-comparison posterior @eq-approx-skew-normal has a name and a closed
form. Does the general posterior @eq-approx-posterior have one too? It does,
and the reason is the same random-utility trick used throughout
@sec-comparisons: a probit factor is the probability that a hidden Gaussian
variable is positive.

::: {.derivation title="The posterior as a selected Gaussian"}
1. For each comparison $k$, write $\mathbf{a}_k = \mathbf{e}_{v_k} - \mathbf{e}_{u_k}$,
   so that $\mathbf{a}_k^\T\vf = f_{v_k} - f_{u_k}$, and introduce independent
   $\eta_k \sim \N(0, s^2)$. Then $\Phi(\mathbf{a}_k^\T\vf/s) = \Prob(\mathbf{a}_k^\T\vf - \eta_k > 0 \mid \vf)$.
2. Stack the $\mathbf{a}_k^\T$ into an $m \times n$ matrix $\mathbf{D}$ and the
   $\eta_k$ into $\bm{\eta}$. By independence, the likelihood is
   $\Prob(\mathbf{D}\vf - \bm{\eta} > \mathbf{0} \mid \vf)$, all inequalities
   holding at once.
3. By Bayes' rule, $p(\vf \mid \D)$ is the distribution of $\vf$ given the
   event $\mathbf{D}\vf - \bm{\eta} > \mathbf{0}$, where $(\vf, \bm{\eta})$ is
   jointly Gaussian.
4. The normalizer $Z$ is the probability of that event. The vector
   $\mathbf{D}\vf - \bm{\eta}$ is Gaussian with mean zero and covariance
   $\mathbf{D}\mK\mathbf{D}^\T + s^2\mI$, so $Z$ is the probability that an
   $m$-dimensional Gaussian vector is positive in every coordinate, an
   *orthant probability*.
:::

A Gaussian vector conditioned on a linear transformation of itself, plus
independent Gaussian noise, being positive is a **unified skew-normal**
distribution. @durante2019conjugate showed that the posterior of parametric
probit regression belongs to this family for any Gaussian prior, which makes
the family conjugate to the probit likelihood. For functions,
@benavoli2021preferentialb proved that "the true posterior distribution of the
preference function is a Skew Gaussian Process (SkewGP), with highly skewed
pairwise marginals", and argued from it that Laplace's method "usually provides a very
poor approximation". @thm-theory-skewgp states the theorem with its
parameters.

The derivation also says *where* the posterior is skewed. Every factor depends
on $\vf$ only through the differences $\mathbf{D}\vf$. In any direction
orthogonal to all the $\mathbf{a}_k$, the likelihood is constant and the
posterior is the Gaussian prior conditioned on the differences. The skew lives
in at most $m$ directions, and never in the direction that shifts every utility
by the same amount, because no difference changes along it. The figure below
shows the smallest instance: two utilities and comparisons between them.

```{figure}
//| figure: approx-posterior-2d
//| label: fig-approx-2d
//| fig-cap: "The posterior of two utilities g(A) and g(B) after A wins, with a prior correlation ρ that a kernel would assign to nearby inputs. Shading and the thick contours show the exact posterior, with its 50% and 90% highest-density regions; the dashed ellipse is the prior's 90% region; the blue ellipses are the Laplace approximation at the same levels. The strip along the top compares the marginal of g(A) alone. The skewness readout is the standardized third moment, 0 for a Gaussian and about 1 for a half-normal."
```

Some things to try:

- **Look along the diagonal.** The exact contours are cut off by the line
  $g(A) = g(B)$ and extend freely along it. The skew is entirely across the
  diagonal, in the difference; along the diagonal, in the sum, the posterior is
  the Gaussian prior.
- **Look at the top strip.** The marginal of $g(A)$ mixes the skewed difference
  with the Gaussian sum. At $\sigma = 0.05$ and $\rho = 0.5$ the difference has
  skewness about 0.98, while $g(A)$ alone has about 0.04. A single utility can
  look Gaussian even when the posterior is far from it. @kuss2005assessing made
  the same observation for classification: marginals of a high-dimensional
  truncated Gaussian "can be relatively similar to a Gaussian".
- **Raise the correlation.** Inputs close together in a kernel's eyes have
  correlated prior utilities, so their difference has a small prior variance,
  $2 - 2\rho$. The same noise is then large relative to that spread, the cut is
  softer, and the skewness of the difference falls, from about 0.98 at
  $\rho = 0.5$ to 0.90 at $\rho = 0.9$. Two options the model already considers
  similar are also the ones a comparison moves least.
- **Raise the noise.** At $\sigma = 1$ the cut softens and the contours are
  nearly elliptical; the skewness of the difference drops to about 0.06.

The lesson carries to many dimensions. In a session of 30 comparisons among 60
distinct designs, each a point in a six-dimensional parameter space, the
posterior lives on 60 latent utilities. It can be skewed only within the span
of the 30 comparison directions, and the input dimension enters only through
the kernel, which sets how correlated the utilities are. Laplace and EP both
treat the remaining directions exactly. What they get wrong is the shape across
the comparisons, and that is exactly what the probability of the next answer
depends on (inference).

The exact posterior has a cost. Every prediction requires samples from a
truncated multivariate Gaussian, and the marginal likelihood requires
high-dimensional normal orthant probabilities [@benavoli2021preferentialb].
Those are the reasons the approximations remain in use.

## How much the approximation matters {#sec-approximation-matters}

The figures show what each approximation gets wrong in the smallest case. Does
it change what a preferential optimizer does with real answers?
@tbl-approx-methods summarizes the methods before the evidence.

::: {.table #tbl-approx-methods title="Approximate inference for preference posteriors: what each method matches, what it costs, and how it fails in the one-difference case of this chapter."}
| Method | What it matches | Cost per fit | Typical failure | Used by |
|---|---|---|---|---|
| Laplace | the mode and the curvature there | a few Newton steps, $O(n^3)$ each | centered at the edge of the mass when answers are nearly noise-free; mean and spread too small | Chu and Ghahramani; BoTorch `PairwiseGP`; Bıyık et al. 2020 |
| Expectation propagation | site by site, the moments of one factor at a time | sweeps of rank-one updates, $O(mn^2)$ per sweep | mass on the impossible side; wrong order of duel probabilities near 0 or 1 | Siivola et al. 2021; optuna-dashboard |
| Variational (Gaussian) | the Gaussian with the highest evidence lower bound | an optimization; scales with inducing points | too narrow | crowdGPPL; top-$k$ ranking; qEUBO experiments |
| Sampling | the posterior itself, in the limit | many samples; truncated Gaussians | correlated samples, so long runs; Monte Carlo error | Benavoli et al. 2021; Takeno et al. 2023 |
:::

The strongest evidence comes from @takeno2023practicalc, with Gibbs sampling as
ground truth, an RBF kernel, noise variance $10^{-4}$, and uniformly random
duels. Laplace was inaccurate "since the mode can be very far away from the
mean, particularly when" the noise variance is small, and they concluded that
"although LA is very fast, LA-based preferential BO will fail". EP was very
accurate for means and credible intervals but distorted duel probabilities, as
described in @sec-ep. They also disagreed with the earlier test of
@benavoli2021preferentialb, in which all inputs in one interval lose and all
inputs in another win; they called such biased training duels "unrealistic"
and used random duels instead. Even so, they fitted hyperparameters with the
Laplace evidence, which they considered accurate enough at small noise.

These results come from nearly noise-free comparisons, the regime in which
@fig-approx-laplace shows Laplace at its worst. Human comparisons are noisy,
and as the figures show, noise softens the cut and shrinks the skew. How large
the errors of Laplace and EP are at realistic human noise levels is a question
no paper we found answers, and we found no independent replication of Takeno
et al.'s comparison on real human comparisons (inference; both as of September
2026). @sec-obs-inference reports this evidence in full, with the libraries'
choices.

In practice, three questions decide the choice. If the posterior feeds an
acquisition function through probabilities of pairwise outcomes, as EUBO
(@sec-eubo) does, the misplaced mass of the Gaussian approximations matters most,
and sampling or at least EP deserves the extra cost (inference). If only the
posterior mean is needed, to recommend a final design, every method that gets
the order of the utilities right is adequate. And if the session is long or the
model has many users, variational inference is the method that scales.

## Exercises {#sec-approx-exercises}

::: {.exercise #exr-approx-variance}
Use Stein's lemma twice to show that the one-comparison posterior
@eq-approx-skew-normal has second moment $\E[\Delta^2 \mid A \succ B] = v_0$, the
same as the prior, and hence variance $v_0 - \tfrac{2}{\pi} v_0^2/(v_0 + s^2)$.
Check the values 0.89 at $\sigma = 1$ and 0.60 as $\sigma \to 0$ for $v_0 = 1$.

::: {.solution}
Apply Stein's lemma with $h(\Delta) = \Delta\,\Phi(\Delta/s)$:
$\E[\Delta^2\Phi(\Delta/s)] = v_0\,\E[\Phi(\Delta/s) + \Delta\,\phi(\Delta/s)/s]$ under the prior. The
first term is $v_0 \cdot \tfrac12$. In the second, $\N(\Delta; 0, v_0)\,\phi(\Delta/s)$ is
proportional to a Gaussian density in $\Delta$ with mean zero, so the expectation of
$\Delta$ times it vanishes. Dividing by $Z = 1/2$ gives $\E[\Delta^2 \mid A \succ B] = v_0$.
The variance is $v_0 - (\E[\Delta \mid A \succ B])^2 = v_0 - \tfrac{2}{\pi}\, v_0^2/(v_0 + s^2)$.
With $v_0 = 1$ and $\sigma = 1$, $s^2 = 2$ and the variance is
$1 - \tfrac{2}{3\pi} \approx 0.788$, a standard deviation of about 0.89. As
$s \to 0$ the variance tends to $1 - 2/\pi \approx 0.363$, a standard deviation
of about 0.60. The comparison moves the mean but leaves the second moment
alone: it reshapes the prior's mass without making it smaller.
:::
:::

::: {.exercise #exr-approx-laplace-limit}
For the one-comparison posterior with $v_0 = 1$, show that as $s \to 0$ the
Laplace approximation assigns probability tending to $1/2$ to the event $\Delta < 0$,
even though the exact probability tends to zero. You may use that the inverse
Mills ratio $\lambda(z) = \phi(z)/\Phi(z)$ satisfies $\lambda(z) \approx \phi(z)$
for large $z$.

::: {.solution}
The mode $\hat\Delta$ solves $\hat\Delta = \lambda(\hat\Delta/s)/s$. Write $\hat z = \hat\Delta/s$;
then $s^2 \hat z = \lambda(\hat z) \approx \phi(\hat z)$, so $\hat z$ grows
without bound as $s \to 0$, but only like $\sqrt{2\log(1/s^2)}$, and
$\hat\Delta = s\hat z \to 0$. The curvature is
$1 + \lambda(\hat z)(\hat z + \lambda(\hat z))/s^2 \approx 1 + \hat z^2$, using
$\lambda(\hat z) \approx s^2\hat z$, so the Laplace standard deviation is about
$1/\sqrt{1 + \hat z^2}$. The Laplace probability of $\Delta < 0$ is
$\Phi(-\hat\Delta\sqrt{1 + \hat z^2}) \approx \Phi(-s\hat z^2)$, and $s\hat z^2 \to 0$
because $\hat z^2$ grows only logarithmically. So the probability tends to
$\Phi(0) = 1/2$. The exact posterior is the half-normal, which has no mass
below zero. A noise-free answer that settles the order completely is reported
by Laplace as no information about the order at all.
:::
:::

::: {.exercise #exr-approx-kl}
Let the target be the half-normal $p(\Delta) = 2\,\N(\Delta; 0, 1)$ for $\Delta > 0$ and zero
otherwise. (a) Explain why $\KL(q \,\|\, p)$ is infinite for every Gaussian $q$.
(b) Explain why the Gaussian minimizing $\KL(p \,\|\, q)$ has the mean and variance
of $p$. (c) Relate both answers to what @fig-approx-vi and @fig-approx-ep show
at very small noise.

::: {.solution}
(a) $\KL(q \,\|\, p) = \E_q[\log q - \log p]$, and every Gaussian puts positive
mass on $\Delta < 0$, where $\log p = -\infty$, so the divergence is infinite. At
small but positive noise, $p$ is tiny but not zero there, and the best Gaussian
under this divergence squeezes almost all its mass into $\Delta > 0$, becoming
narrow. (b) $\KL(p \,\|\, q) = \E_p[\log p] - \E_p[\log q]$; only the second term
depends on $q$, and for a Gaussian
$\E_p[\log q] = -\tfrac12\log(2\pi v) - \E_p[(\Delta - \mu)^2]/(2v)$, which is maximized
by $\mu = \E_p[\Delta]$ and $v = \Var_p[\Delta]$. (c) The variational fit in
@fig-approx-vi keeps out of $\Delta < 0$ and is too narrow; EP's moment matching in
@fig-approx-ep has the right mean and variance but spills over the edge.
:::
:::

## Further reading {#further-reading .unnumbered}

- @rasmussen2006gaussian, chapter 3, derives the Laplace approximation and
  expectation propagation for Gaussian process classification, with stable
  algorithms that carry over to comparisons.
- @kuss2005assessing and @nickisch2008approximations are the careful
  comparisons of approximations for classification, explaining why Laplace
  fails and EP does well.
- @minka2001expectation introduces expectation propagation; @murray2010elliptical
  introduces elliptical slice sampling; @tierney1986accurate is the classic on
  Laplace's method for posterior moments.
- @durante2019conjugate shows conjugacy of the probit model with unified
  skew-normal distributions; @benavoli2021preferentialb extends it to the
  preference posterior and samples it exactly.
- @takeno2023practicalc measure how wrong Laplace and EP are on duels and
  propose a cheaper exact-sampling alternative.
