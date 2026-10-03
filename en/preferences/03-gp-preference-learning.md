---
status: done
synopsis: "Chu and Ghahramani's model: a Gaussian process utility observed only through noisy comparisons, fitted by Newton's method with the Laplace approximation. The chapter derives the fit step by step, predicts new comparisons, shows the model in one, two, and more dimensions, explains what comparisons cannot identify and what the comparison graph does to the posterior, and opens BoTorch's PairwiseGP."
sources: ["Chu and Ghahramani 2005", "Houlsby et al. 2011", "Shah et al. 2016", "Hendrickx et al. 2019"]
---

# Gaussian Process Preference Learning {#sec-gp-preference}

@sec-comparisons turned an answer to "which do you prefer?" into a likelihood:
the probit choice model, under which the probability of preferring one option
grows with the difference in their utilities. @sec-approx-inference showed how
to handle the non-Gaussian posterior that such a likelihood creates. This
chapter joins the two with the Gaussian process prior of @sec-gp-regression,
which is what lets a few dozen answers about a few dozen designs say
something about every design, including those never shown.

The result is the model of @chu2005preference. It is the default model behind
preferential Bayesian optimization (PBO) two decades later, and it is what runs
inside the figures of @sec-pbo. We set it up, derive how to fit it, use it to
predict, and then ask two questions that the regression model never raised:
what comparisons cannot tell us at all, and how the pattern of which pairs were
compared shapes what they can.

## The model {#sec-pref-model}

A person has a latent utility $g$ over a domain $\X$ of designs: the color of
a poster, the parameters of an exoskeleton controller, the settings of a photo
filter. We place a Gaussian process prior on it,

$$
g \sim \GP(0, k),
$$

so that before any answer, the utilities of any finite set of designs are
jointly Gaussian with covariances given by the kernel $k$ (@sec-function-space).
The kernel encodes the one assumption that makes learning from few answers
possible: designs that are close in $\X$ have similar utilities.

The data are $m$ answers among $n$ distinct designs $\vx_1, \dots, \vx_n$. The
$k$-th answer says that design $v_k$ was preferred to design $u_k$, where $v_k$
and $u_k$ are indices into the list of designs. A design can appear in many
answers, so $n$ is at most $2m$ and often much smaller. Collect the utilities
at the designs in a vector $\vf = (g(\vx_1), \dots, g(\vx_n))^\T$, whose prior
is $\N(\mathbf{0}, \mK)$ with $[\mK]_{ij} = k(\vx_i, \vx_j)$. Each answer
follows Thurstone's Case V (@eq-cmp-case-v), and answers are independent given
the utilities:

$$
p(\D \mid \vf) = \prod_{k=1}^{m} \Phi\!\left(\frac{f_{v_k} - f_{u_k}}{\sqrt{2}\,\sigma}\right).
$$ {#eq-pref-lik}

@chu2005preference derived this likelihood exactly as @sec-thurstone did: start
from an ideal judge who prefers $v_k$ whenever $g(\vx_{v_k}) \ge g(\vx_{u_k})$,
then contaminate each latent value with independent Gaussian noise of variance
$\sigma^2$. The noise scale $\sigma$ is a hyperparameter, alongside the kernel's
lengthscale and amplitude.

The posterior over $\vf$ is @eq-approx-posterior: the Gaussian prior times
@eq-pref-lik, normalized. It is not Gaussian, so @chu2005preference replaced
it by the Laplace approximation of @sec-laplace. Predictions at new designs then
follow from the Gaussian process exactly as in regression, because given $\vf$
the utility anywhere else is conditionally Gaussian.

The same model can be seen from a second angle. The likelihood depends only on
differences, so one can define a function of *pairs*, $h(\vx, \vx') = g(\vx) -
g(\vx')$, and treat each answer as a binary label on a pair. Because $h$ is a
linear transformation of $g$, it is itself a Gaussian process, with covariance

$$
\Cov\big(h(\vx, \vx'),\, h(\mathbf{y}, \mathbf{y}')\big) = k(\vx, \mathbf{y}) + k(\vx', \mathbf{y}') - k(\vx, \mathbf{y}') - k(\vx', \mathbf{y}),
$$

by expanding the covariance of two differences term by term.
@houlsby2011bayesian (a preprint) derived this *preference kernel* and
concluded that the model of Chu and Ghahramani "is equivalent to GPC with a
particular class of kernels", Gaussian process classification on pairs. Every
tool for classification, including the approximations of
@sec-approx-inference, therefore applies.

Preference learning of this kind reached people early. @brochu2007active used
the same Thurstone model to help people find material appearances for computer
graphics by choosing between rendered examples, and it has since been used,
with the Laplace approximation, to learn which exoskeleton gaits users prefer
from their comparisons and ordinal labels [@li2021roial]. @sec-cs-exoskeleton
and @sec-cs-photo work through two such problems end to end.

## Fitting it {#sec-pref-laplace}

Fitting the model means finding the posterior mode $\hat\vf$ and the curvature
there. @eq-approx-newton gave Newton's step for any likelihood; what it needs
from the likelihood is its gradient and its negative Hessian $\mW$. For
comparisons both have a structure that explains much of the rest of this
chapter, so we derive them in full. Write $s = \sqrt{2}\,\sigma$, and for the
$k$-th answer write $z_k = (f_{v_k} - f_{u_k})/s$ and
$\mathbf{a}_k = \mathbf{e}_{v_k} - \mathbf{e}_{u_k}$, the vector with $+1$ at
the winner, $-1$ at the loser, and 0 elsewhere, so that $z_k = \mathbf{a}_k^\T\vf/s$.

::: {.derivation title="Gradient and Hessian of the pairwise probit log-likelihood"}
1. Take logs of @eq-pref-lik: $\ell(\vf) = \sum_k \log\Phi(z_k)$.
2. Differentiate one term. Since $\tfrac{\dd}{\dd z}\log\Phi(z) = \phi(z)/\Phi(z)$,
   which we write $\lambda(z)$ (the *inverse Mills ratio*), and
   $\partial z_k / \partial\vf = \mathbf{a}_k/s$, the chain rule gives
   $\nabla \log\Phi(z_k) = \tfrac{\lambda(z_k)}{s}\,\mathbf{a}_k$.
3. Sum over answers:
   $\nabla\ell(\vf) = \sum_k \tfrac{\lambda(z_k)}{s}\,\mathbf{a}_k$. Each answer
   adds $\lambda(z_k)/s$ to its winner's entry and subtracts it from its loser's.
4. Differentiate again. Using $\phi'(z) = -z\phi(z)$ and the quotient rule,
   $\lambda'(z) = -\lambda(z)\big(z + \lambda(z)\big)$, so
   $\nabla\nabla \log\Phi(z_k) = -\tfrac{\lambda(z_k)(z_k + \lambda(z_k))}{s^2}\,\mathbf{a}_k\mathbf{a}_k^\T$.
5. Define the weight $w_k = \lambda(z_k)\big(z_k + \lambda(z_k)\big)/s^2$. Then
   $\mW = -\nabla\nabla\ell(\vf) = \sum_k w_k\, \mathbf{a}_k\mathbf{a}_k^\T$.
6. Each weight is positive: $1 - \lambda(z)(z + \lambda(z))$ is the variance of a
   standard normal variable conditioned to exceed $-z$, which lies between 0 and
   1, so $0 < \lambda(z)(z + \lambda(z)) < 1$ and $0 < w_k < 1/s^2$.
7. Hence $\mW$ is a sum of positive multiples of $\mathbf{a}_k\mathbf{a}_k^\T$, a
   positive semidefinite matrix, and $\ell$ is concave. With the concave log
   prior, the log posterior has a single maximum, as @chu2005preference proved
   (their Lemma 1).
:::

$$
\nabla\ell(\vf) = \sum_{k=1}^{m} \frac{\lambda(z_k)}{s}\,\big(\mathbf{e}_{v_k} - \mathbf{e}_{u_k}\big),
\qquad
\mW = \sum_{k=1}^{m} w_k\,\big(\mathbf{e}_{v_k} - \mathbf{e}_{u_k}\big)\big(\mathbf{e}_{v_k} - \mathbf{e}_{u_k}\big)^\T.
$$ {#eq-pref-grad-hess}

Look at what @eq-pref-grad-hess says entry by entry. The gradient pushes each
design up by $\lambda(z_k)/s$ for every answer it won and down for every answer
it lost. The push is large when the answer was surprising: $\lambda(z)$ grows
roughly like $-z$ when $z$ is very negative, that is, when the current utilities
say the loser should have won, and it vanishes when the answer was already
expected. The matrix $\mW$ has, on its diagonal, the total weight of the answers
each design took part in, and off the diagonal, minus the total weight of the
answers between two designs. That is the **weighted graph Laplacian** of the
*comparison graph*, whose nodes are designs and whose edges are answered pairs.
@sec-comparison-graph returns to what that means.

Two consequences follow immediately. Every $\mathbf{a}_k$ sums to zero, so the
gradient sums to zero, $\mathbf{1}^\T\nabla\ell = 0$, and $\mW\mathbf{1} =
\mathbf{0}$: raising every utility by the same amount changes no answer's
probability. And the weights depend on $\vf$, so $\mW$ is recomputed at every
Newton step.

With these two ingredients, the fit is the algorithm below. It is what
`fitPreference` in this book's figure code does.

::: {.algorithm #alg-pref-fit title="Fitting the preference model by Newton's method"}
Input: designs $\vx_1, \dots, \vx_n$, answers $(v_k, u_k)$ for $k = 1, \dots, m$,
kernel function $k(\cdot, \cdot)$, noise $\sigma$.

1. Form $\mK$, adding a small jitter to its diagonal, and set $\vf \leftarrow \mathbf{0}$.
2. Compute $\nabla\ell(\vf)$ and $\mW$ from @eq-pref-grad-hess.
3. Take the Newton step @eq-approx-newton:
   $\vf_{\text{new}} \leftarrow \mK(\mI + \mW\mK)^{-1}\big(\mW\vf + \nabla\ell(\vf)\big)$.
4. If the log posterior $\ell(\vf) - \tfrac12\vf^\T\mK^{-1}\vf$ decreased, halve
   the step, $\vf_{\text{new}} \leftarrow \vf + \tfrac12(\vf_{\text{new}} - \vf)$,
   until it increases.
5. Set $\vf \leftarrow \vf_{\text{new}}$ and return to step 2 until the largest
   change is below a tolerance. Call the result $\hat\vf$.
6. Return $\hat\vf$, $\hat{\bm\beta} = \nabla\ell(\hat\vf)$, and $\hat\mW$
   evaluated at $\hat\vf$.
:::

At the start, every $z_k = 0$, every answer is a coin flip under the model, and
every weight is $\lambda(0)^2/s^2 = (2/\pi)/s^2$. The first step therefore
treats all answers alike, and later steps reweight them by how surprising each
turns out to be. In the figures of this chapter the fit converges in five to
thirteen steps. Each step solves an $n \times n$ system, so the cost is $O(n^3)$
per step, the same as regression.

At the mode the gradient of the log posterior vanishes:
$\nabla\ell(\hat\vf) - \mK^{-1}\hat\vf = \mathbf{0}$, so

$$
\hat\vf = \mK\,\hat{\bm\beta}, \qquad \hat{\bm\beta} = \nabla\ell(\hat\vf),
$$ {#eq-pref-mode}

which is equation (11) of @chu2005preference. The vector $\hat{\bm\beta}$ plays
the role that $\bm\alpha = \mK^{-1}\vy$ played in regression
(@eq-gp-representer): one weight per design, positive for designs that won more
than the model expected and negative for those that lost.

::: {.example #ex-pref-two title="One answer between two designs"}
Take two designs with prior correlation $\rho$, so
$\mK = \begin{bmatrix} 1 & \rho \\ \rho & 1 \end{bmatrix}$, and one answer, $A$
preferred to $B$. Then $\mathbf{a} = (1, -1)^\T$ and
$\mW = w \begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}$ for the weight $w$ at
the mode. The Laplace covariance is $(\mK^{-1} + \mW)^{-1}$; by the
Sherman-Morrison formula it equals $\mK - \frac{\mK\mathbf{a}\mathbf{a}^\T\mK}{1/w + \mathbf{a}^\T\mK\mathbf{a}}$.
Here $\mathbf{a}^\T\mK\mathbf{a} = 2 - 2\rho$, the prior variance of the
difference $\Delta = f_A - f_B$, and $\mK\mathbf{a} = (1 - \rho)(1, -1)^\T$.

- The posterior variance of the difference is
  $\mathbf{a}^\T\Sigma\mathbf{a} = v_\Delta - \frac{v_\Delta^2}{1/w + v_\Delta} = \frac{v_\Delta}{1 + w v_\Delta}$,
  with $v_\Delta = 2 - 2\rho$. The answer shrinks it.
- The posterior variance of the sum $f_A + f_B$ is unchanged, because
  $(1, 1)\,\mK\mathbf{a} = 0$. The answer says nothing about the overall level.

This is the two-dimensional picture of @fig-approx-2d, now in formulas: a
comparison acts only along the difference.
:::

The same computations give the Laplace evidence @eq-approx-laplace-evidence,
which @chu2005preference maximized over the kernel hyperparameters and
$\sigma$ (their equation 12); BoTorch fits its preference model the same way
(@sec-pairwisegp).

::: {.code title="NumPy"}
```python
import numpy as np
from scipy.stats import norm

def terms(f, duels, s):
    """Gradient and negative Hessian of the pairwise probit log-likelihood."""
    n = len(f)
    grad, W = np.zeros(n), np.zeros((n, n))
    for i, j in duels:                        # design i was preferred to j
        z = (f[i] - f[j]) / s
        lam = np.exp(norm.logpdf(z) - norm.logcdf(z))   # phi(z) / Phi(z)
        grad[i] += lam / s
        grad[j] -= lam / s
        w = lam * (z + lam) / s**2
        W[i, i] += w; W[j, j] += w; W[i, j] -= w; W[j, i] -= w
    return grad, W

def fit_preference(K, duels, sigma=0.1, iters=50):
    n, s = len(K), np.sqrt(2) * sigma
    f = np.zeros(n)
    for _ in range(iters):
        grad, W = terms(f, duels, s)
        f_new = K @ np.linalg.solve(np.eye(n) + W @ K, W @ f + grad)
        done = np.max(np.abs(f_new - f)) < 1e-8
        f = f_new
        if done:
            break
    grad, W = terms(f, duels, s)              # beta and W at the mode
    return f, grad, W
```
The sketch omits the step halving of @alg-pref-fit, which matters only when
the first steps overshoot, and uses a general solver where a careful
implementation would exploit symmetry, as @rasmussen2006gaussian do for
classification. @sec-impl-pref builds the full model in the minimal
implementation.
:::

## Predicting preferences {#sec-pref-predict}

With the mode and the curvature in hand, the Laplace posterior over $\vf$ is
Gaussian, $\N(\hat\vf, (\mK^{-1} + \hat\mW)^{-1})$, and predicting the utility
at new designs is Gaussian conditioning, as in @sec-gp-conditioning. For a new
design $\vx$, write $\vk(\vx)$ for its prior covariances with the compared
designs.

::: {.derivation title="The latent posterior at a new design"}
1. Given $\vf$, the Gaussian process prior gives
   $g(\vx) \mid \vf \sim \N\big(\vk(\vx)^\T\mK^{-1}\vf,\; k(\vx, \vx) - \vk(\vx)^\T\mK^{-1}\vk(\vx)\big)$
   (@eq-gp-posterior with noise-free "observations" $\vf$).
2. Average over the Laplace posterior of $\vf$. The mean is
   $\vk(\vx)^\T\mK^{-1}\hat\vf = \vk(\vx)^\T\hat{\bm\beta}$, by @eq-pref-mode.
3. The variance adds the spread of the conditional mean to the conditional
   variance (the law of total variance):
   $k(\vx, \vx) - \vk(\vx)^\T\mK^{-1}\vk(\vx) + \vk(\vx)^\T\mK^{-1}(\mK^{-1} + \hat\mW)^{-1}\mK^{-1}\vk(\vx)$.
4. By the Woodbury identity (@sec-id-woodbury),
   $\mK^{-1} - \mK^{-1}(\mK^{-1} + \hat\mW)^{-1}\mK^{-1} = (\mK + \hat\mW^{-1})^{-1}$
   when $\hat\mW$ is invertible, and in general
   $(\mI + \hat\mW\mK)^{-1}\hat\mW$, which needs no inverse of $\hat\mW$.
5. Hence the variance is $k(\vx, \vx) - \vk(\vx)^\T(\mI + \hat\mW\mK)^{-1}\hat\mW\,\vk(\vx)$.
:::

$$
\mu(\vx) = \vk(\vx)^\T \hat{\bm\beta},
\qquad
\sigma^2(\vx) = k(\vx, \vx) - \vk(\vx)^\T (\mI + \hat\mW\mK)^{-1}\hat\mW\, \vk(\vx).
$$ {#eq-pref-predict}

These match equations (17) and (18) of @chu2005preference and Rasmussen and
Williams's predictive equations for classification [@rasmussen2006gaussian,
ch. 3], with the comparison $\hat\mW$ in place of the diagonal one. The second
form in step 4 matters: $\hat\mW$ is a graph Laplacian, which always has a zero
eigenvalue, so $\hat\mW^{-1}$ does not exist.

The quantity a preferential optimizer most often needs is the probability that
a person will prefer one new design $\vx$ to another $\vx'$. The two latent
values are jointly Gaussian under the approximation, with means $\mu, \mu'$,
variances $\sigma^2(\vx), \sigma^2(\vx')$, and covariance $c$ from the same
formula with $\vk(\vx')$ on the right. Their difference has mean $\mu - \mu'$
and variance $\sigma^2(\vx) + \sigma^2(\vx') - 2c$, and @eq-cmp-predictive gives

$$
\Prob(\vx \succ \vx' \mid \D) = \Phi\!\left(\frac{\mu - \mu'}{\sqrt{2\sigma^2 + \sigma^2(\vx) + \sigma^2(\vx') - 2c}}\right),
$$ {#eq-pref-pair}

equation (19) of @chu2005preference. The covariance $c$ matters. Two nearby
designs have strongly correlated latent values, so their difference is much
better known than either value, and the model can be confident about their
order even when it is unsure how good either one is.

The figure below runs the model on the running objective of this book, the
two-bump function of @sec-bo-loop. A simulated person whose utility is that
function answers each comparison with a little probit noise; you choose the
pairs. Above the plot, the comparison graph draws one arc per answer.

```{figure}
//| figure: pref-duels-1d
//| label: fig-pref-duels
//| fig-cap: "The Gaussian process preference model learning the running objective from comparisons. Click two inputs to ask the simulated person to compare them, or press Random pair. The blue curve and band are the Laplace posterior @eq-pref-predict; the dashed curve is the objective, shifted and scaled to fit, since comparisons cannot fix the level or the scale; the arcs above are the answered pairs, colored by connected component. The status line reports the Newton steps of @alg-pref-fit and the probability @eq-pref-pair that the model gave the last winner before it answered. Switch to You to answer yourself. The person's noise (σ = 0.05) and the model's settings are illustrative."
duels: "0.05>0.85,0.65>0.95,0.4>0.55,0.75>0.3,0.55>0.85,0.15>0.85"
```

Some things to try:

- **Ask ten random pairs, twice.** The figure starts with six answers, too few
  to find either bump. After 20 more the mean follows both bumps and places its
  maximum at $x = 0.72$, next to the true one at 0.73. Every comparison moved
  the curve only at its two ends and, through the kernel, their neighborhoods.
- **Watch the band.** It narrows near compared designs but stays wide
  everywhere, about $\pm 1$ even after 20 answers. That is not a failure;
  @sec-pref-identifiability explains it.
- **Read the last prediction.** Among the first answers the probabilities
  scatter, and some winners had been predicted to lose (a probability below one
  half); later most winners are predicted at 0.8 or above. A surprising answer
  moves the curve the most, because its $\lambda(z)$ is large.
- **Change the lengthscale.** At $\ell = 0.03$ each answer is a local bump,
  and the curve falls back toward zero wherever nothing nearby was compared. At
  $\ell = 0.5$ the curve is too stiff to hold two bumps, and its maximum lands
  in the wrong place.
- **Press Best against random repeatedly.** Every pair now includes the current
  best design, and the graph becomes a star. The region around the best is
  learned well and the rest poorly, a pattern that returns with the
  champion-and-challenger rules of @sec-choosing-pairs and with EUBO's
  collapse toward the current best (@sec-pbo-failure-modes).

### In two and more dimensions {#sec-pref-higher-dim}

Nothing in @eq-pref-lik to @eq-pref-pair depends on the input dimension.
Designs enter only through the kernel, as distances. The figure below runs the
same model on a two-dimensional design space, with a simulated person whose
utility is the Branin function, a standard test function with three equally
good maxima along a curved valley, rescaled to the unit square with larger
values better.

```{figure}
//| figure: pref-duels-2d
//| label: fig-pref-2d
//| fig-cap: "The preference model on a two-dimensional design space. Left: the hidden utility (Branin, rescaled), colored by rank. Middle: the posterior mean, colored by rank, with the compared designs (dots, colored by connected component), the answered pairs (lines), and the recommended design (star), the compared design with the highest posterior mean. Right: the posterior standard deviation of each design's utility relative to the average over the square. The four designs spend the same number of comparisons differently: random pairs among a pool, a chain, a star around one design, or isolated pairs. The readout gives the share of pairs of compared designs that the posterior mean orders correctly and its rank correlation with the hidden utility over the square. Noise σ = 0.1 for both the person and the model; all values are illustrative."
```

Some things to try:

- **Compare the left and middle maps.** With 20 random pairs among 17 designs
  the posterior mean recovers the valley's broad shape. In the default draw it
  orders 89% of all pairs of compared designs correctly, and its rank agreement
  with the hidden utility over the whole square is 0.87; press New draw a few
  times to see that agreement range from about 0.4 to 0.9.
- **Look at the right map.** Uncertainty is lowest where designs were compared
  and highest in the corners no comparison reached.
- **Switch to Star.** Every answer involves the same central design. Its
  comparisons with the others are learned, but the others are never compared
  with one another, and the model does worst here: averaged over 30 draws it
  orders 81% of the pairs correctly, with a rank agreement of 0.63, against 85%
  and 0.71 for random pairs.
- **Switch to Isolated pairs.** Twenty answers now touch 40 designs in 20
  separate components. Each answer orders two designs and says nothing directly
  about how they compare with the other pairs, so the offsets between pairs
  come only from the kernel. With a kernel this smooth, that works: over 30
  draws the model still orders 82% of the pairs correctly, and because 40
  designs cover the square better than 17, its rank agreement over the square
  is slightly higher (0.79).
- **Shorten the lengthscale to 0.08.** The kernel now links designs less. The
  isolated pairs drop to 76% of pairs ordered correctly, the random pairs to
  82%: the less the prior ties designs together, the more the answers must do
  it themselves.

How far does this go in more dimensions? The same code answers that. @tbl-pref-dims
reports an illustrative computation: 40 random designs in the unit cube,
$m$ random pairs among them answered by a simulated person with noise $\sigma =
0.1$, an RBF kernel with lengthscale 0.52 (the mode of BoTorch's default prior,
@sec-pairwisegp), and 20 repetitions. The utility is the Hartmann function in 3
and 6 dimensions, two standard benchmarks, and the 6-dimensional one hidden in
20 dimensions, where 14 inputs do not matter. The column "kernel value" is the
average prior correlation between two of the designs.

::: {.table #tbl-pref-dims title="An illustrative computation with the model of this chapter: how well 20, 40, or 80 comparisons among 40 random designs teach the utility as the dimension grows. ρ is the rank correlation (1 when two orderings agree completely, 0 when they are unrelated) between the posterior mean and the true utility on 300 new random designs. Rank is the true rank of the recommended design among the 40 compared (1 is best). Averages over 20 repetitions."}
| Utility | Kernel value | ρ, 20 | ρ, 40 | ρ, 80 | Rank, 20 | Rank, 40 | Rank, 80 |
|---|---|---|---|---|---|---|---|
| Hartmann, 3-D | 0.47 | 0.75 | 0.79 | 0.86 | 4.7 | 5.0 | 3.1 |
| Hartmann, 6-D | 0.23 | 0.49 | 0.55 | 0.61 | 7.7 | 5.5 | 3.1 |
| Hartmann 6-D in 20-D | 0.006 | 0.17 | 0.26 | 0.26 | 10.8 | 8.1 | 2.3 |
:::

Two patterns stand out. The model's grasp of *new* designs collapses with
dimension: in 20 dimensions, random designs are so far apart in the kernel's
eyes (an average kernel value of 0.006) that each answer informs almost
nothing beyond its own two designs. Meanwhile its ranking of the designs it has
*seen* keeps improving with more answers in every dimension, because those are
ordered by direct comparisons. Making the lengthscale grow with the number of
inputs $d$, to $0.3\sqrt{d}$, raised the average kernel value in 20 dimensions
to 0.41 but improved the rank correlation on new designs only to 0.19, 0.30,
and 0.28: the problem there is the 14 inputs that do not matter, which a
kernel with one lengthscale shared by all inputs (an isotropic kernel) cannot
ignore. These are one illustrative setup, not a benchmark (inference).
They suggest why PBO in many dimensions depends on the
kernel's structure, such as one lengthscale per input (@sec-ard), and on
choosing pairs well (@sec-pbo); @sec-high-dimensions reports the research.

## What can be identified {#sec-pref-identifiability}

The band in @fig-pref-duels stayed wide after 20 answers, while the band of a
regression model with 20 observations would have pinched at each one. The
difference is not the approximation. It is what comparisons can measure.

**Shift.** Every answer's probability depends on a difference of utilities, so
adding the same constant $c$ to every utility, $\vf \mapsto \vf + c\mathbf{1}$,
leaves the likelihood unchanged. In @eq-pref-grad-hess this appears as
$\mathbf{1}^\T\nabla\ell = 0$ and $\mW\mathbf{1} = \mathbf{0}$: no answer pushes
along the direction $\mathbf{1}$, and no answer adds curvature along it. Only
the prior says anything about the overall level, so the posterior keeps the
prior's uncertainty about it, and that shared uncertainty is what keeps the
band wide. The figure below shows the same model with the band redrawn for
$g(\vx)$ minus its average over the domain, a quantity comparisons can measure.

```{figure}
//| figure: pref-duels-1d
//| label: fig-pref-relative
//| fig-cap: "The posterior after 20 answers, with the 95% band for g(x) minus its average over [0, 1] (the dotted line). Removing the unmeasurable overall level roughly halves the band: the average standard deviation falls from 0.59 to 0.33, and the smallest from 0.51 to 0.17. Turn the relative band off to compare."
duels: "0.05>0.85,0.65>0.95,0.4>0.55,0.75>0.3,0.55>0.85,0.15>0.85,0.4>0.9,0.25>0.6,0.55>0.5,0.65>0.4,0.25>0.05,0.65>0.6,0.1>0.85,0.05>0.4,0.2>0.15,0.8>0.5,0.35>0.6,0.85>0.9,0.6>0.95,0.25>0.65"
relative: true
```

Shift invariance is harmless for optimization, which needs only the order of
utilities, but it trips up anyone who reads the posterior variance as a
measure of how much has been learned. Some models remove it by construction.
@byk2020active, learning robot reward functions from preferences, fixed the
utility to zero at an arbitrary reference design, building the constraint into
the kernel, because query responses are invariant to such shifts.
@chau2022inconsistent note likewise that utilities are determined only up to a
global shift.

**Scale versus noise.** The likelihood depends on $\vf/\sigma$. Doubling every
utility and doubling the noise gives exactly the same probabilities for every
answer. Comparisons therefore measure utility in units of the person's noise,
just as Thurstone measured scale values in units of the discriminal dispersion
(@sec-thurstone). In the model, only the prior separates the two: the kernel's
amplitude says how large utilities are a priori, and with it fixed, $\sigma$ is
identified relative to it. Fitting both from comparisons alone means fitting
one ratio. BoTorch makes this explicit by fixing the noise and learning the
amplitude (@sec-pairwisegp). In @fig-pref-relative the amplitude is fixed at 1,
so the model-noise slider changes the ratio. At $\sigma = 0.03$ the answers are
satisfied by small differences, a few multiples of $\sigma$, and the posterior
mean spans about 0.7 from its lowest to its highest point; at $\sigma = 0.1$ it
spans about 1.4, with the maximum unchanged at $x = 0.72$. The answers fix the
order and the rough shape; how large the differences are depends on what the
model is told about the noise. At $\sigma = 0.3$ the answers count for so little
that the maximum moves to the wide bump.

**What survives.** The order of utilities, the location of the maximum, and
probabilities of future answers are identified; the level is not, and the
scale is identified only relative to the noise. A practical consequence follows
for anyone who compares learned utilities across people or sessions: two
posteriors on different levels or scales may describe the same preferences,
and comparing them requires recalibration, for instance by including common
reference comparisons (inference). @sec-many-users returns to this for models
with several people.

## The comparison graph {#sec-comparison-graph}

@eq-pref-grad-hess showed that the curvature of the log-likelihood is a
weighted graph Laplacian: designs are nodes, answered pairs are edges with
weights $w_k$. Graph Laplacians have well-understood properties, and three of
them translate directly into statements about what the answers determine.

**Components are zero eigenvalues.** For any vector $\vf$,
$\vf^\T\mW\vf = \sum_k w_k (f_{v_k} - f_{u_k})^2$. This is zero exactly when
$\vf$ is constant on every connected component of the graph, so $\mW$ has one
zero eigenvalue per component. One of them is the global shift of
@sec-pref-identifiability. Each additional one is an offset between two groups
of designs that were never compared, directly or through a chain of other
designs. Along those directions the answers carry no information, and the
posterior relies on the prior alone. The figure below shows this for four
designs of the same number of comparisons; @fig-pref-2d showed the same thing
on a map.

```{figure}
//| figure: comparison-graph
//| label: fig-pref-spectrum
//| fig-cap: "The comparison graph and the eigenvalues of the likelihood curvature W for the same number of comparisons arranged in different designs, with unit weight per comparison. Each zero eigenvalue (bottom) is a direction no answer constrains: the first is the global shift, each further one (red) an offset between components. The designs are illustrative."
design: chain
comparisons: 8
```

**Connectivity is the second eigenvalue.** A connected graph has exactly one
zero eigenvalue, and the second-smallest eigenvalue, which @fiedler1973algebraic
called the *algebraic connectivity*, measures how well connected it is. A chain
has a tiny algebraic connectivity, a star or a densely connected graph a large
one. Switch @fig-pref-spectrum between Chain and Star with the same number of
comparisons: the chain's smallest nonzero eigenvalues crowd toward zero, while
the star's sit at 1. Directions with small eigenvalues are nearly unconstrained
by the answers.

**Differences are resistances.** Think of each answer as a resistor with
conductance $w_k$ between its two designs. Then, if the prior is weak, the
uncertainty about the difference between two designs is the *effective
resistance* between them.

::: {.derivation title="Uncertainty of a difference under a flat prior"}
1. With a prior so broad that $\mK^{-1} \approx \mathbf{0}$ along the relevant
   directions, the Laplace precision of $\vf$ is $\mW$, which is singular.
   Differences $\mathbf{a}^\T\vf$ with $\mathbf{1}^\T\mathbf{a} = 0$, inside one
   component, are still well defined.
2. Their variance is $\mathbf{a}^\T\mW^{+}\mathbf{a}$, where $\mW^{+}$ is the
   pseudo-inverse: the inverse on the space orthogonal to the zero
   eigenvectors.
3. For $\mathbf{a} = \mathbf{e}_i - \mathbf{e}_j$, the quantity
   $(\mathbf{e}_i - \mathbf{e}_j)^\T\mW^{+}(\mathbf{e}_i - \mathbf{e}_j)$ is, by
   definition, the effective resistance $R_{ij}$ between nodes $i$ and $j$ of an
   electrical network with conductances $w_k$.
4. Hence $\Var(f_i - f_j) \approx R_{ij}$: resistors in series add, so long
   chains of comparisons leave large uncertainty; resistors in parallel combine,
   so many independent paths between two designs pin their difference down.
:::

$$
\Var(f_i - f_j \mid \D) \approx R_{ij} = (\mathbf{e}_i - \mathbf{e}_j)^\T \mW^{+} (\mathbf{e}_i - \mathbf{e}_j).
$$ {#eq-pref-resistance}

In a chain of eight comparisons with equal weights, the two ends are eight
resistors apart and their difference has eight times the variance of a single
comparison; in a star, any two leaves are two apart. The prior then caps every
variance at its prior value, but the ordering of designs by how well their
differences are known follows the resistances.

This picture is not only an analogy for the Gaussian process model. For
estimating the utilities of a finite set of items under the Thurstone and
Bradley-Terry models, @shah2016estimation proved minimax error bounds that
depend on the topology of the comparison graph through its Laplacian spectrum,
and @hendrickx2019graph proved that the relevant quantity is "the square root of
the resistance of the comparison graph", with a matching lower bound up to
logarithmic factors. Which pairs are asked therefore matters for any comparison
model, not only for this one.

**Why it matters in practice.** With a Gaussian process prior, the Laplace
precision $\mK^{-1} + \mW$ is always invertible, because the prior adds
$\mK^{-1}$, and @fig-pref-2d showed that a smooth kernel can supply much of the
missing linkage. The trouble starts when the prior cannot: along a zero or
near-zero direction of $\mW$ the precision comes from the prior alone, and when
the prior's variance in that direction is large, the precision is small and the
matrix is badly conditioned. A 2026
preprint traced numerical trouble in the default pipeline to exactly this,
observing that EUBO tends to choose pairs that share no design with earlier
queries, which makes the likelihood Hessian rank deficient; a diagonal
correction scaled by the prior uncertainty improved results by up to 10.9%
across 11 benchmarks in 5 to 20 dimensions, with $p = 0.003$ (a *p-value*: the
probability of seeing a difference at least this large if the correction made
no difference) [@shao2026adaptive]. That margins and connectivity of the comparison graph
govern the sample efficiency of Bradley-Terry estimation was also argued at
ICML 2026 [@pukdee2026preference]. @sec-obs-graphs reports this work, and
@sec-pbo-failure-modes places it among the known failure modes of the loop.

::: {.keyidea title="The comparison graph is part of the data"}
Comparisons constrain differences along the edges of the comparison graph. A
disconnected graph leaves the offsets between components to the prior, and a
long chain leaves the ends loosely tied. Which pairs were asked shapes the
posterior as much as how they were answered.
:::

## In software {#sec-pairwisegp}

Most people who fit this model use BoTorch's `PairwiseGP`, which its docstring
describes as "a probit-likelihood GP that learns via pairwise comparison data,
using a Laplace approximation of the posterior of the estimated utility
values" [@botorch2026pairwisegp]. The model takes the designs as a tensor and the answers as a list of
index pairs, preferred design first, and fits its hyperparameters by
maximizing the Laplace evidence [@botorch2026pairwise].

::: {.code title="BoTorch"}
```python
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize

X = torch.rand(10, 3, dtype=torch.double)     # 10 designs with 3 parameters
comparisons = torch.tensor([[0, 1], [2, 0], [3, 4], [4, 2]])  # row (i, j): i preferred to j

model = PairwiseGP(X, comparisons, input_transform=Normalize(d=X.shape[-1]))
mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_mll(mll)

post = model.posterior(torch.rand(5, 3, dtype=torch.double))
post.mean, post.variance                      # the latent utility at 5 new designs
```
This follows the structure of BoTorch's preference tutorial; @sec-impl-botorch
runs a complete loop.
:::

The defaults matter, because they are, in effect, the field's defaults. Reading
the source of version 0.18.1 shows the following choices
[@botorch2026pairwisegp; @botorch2026likelihood].

::: {.table #tbl-pref-botorch title="Choices inside BoTorch's PairwiseGP, and where this chapter explains them."}
| Choice | Default | Where explained |
|---|---|---|
| Likelihood | probit, $\Phi\big((g(v) - g(u))/\sqrt{2}\big)$, noise implicitly fixed at 1; argument clipped to $[-3, 3]$ | @eq-pref-lik, @sec-random-utility |
| Alternative likelihood | logistic, with the logit clipped to $[-8, 8]$ | @sec-bradley-terry |
| Noise scale $\sigma$ | dropped; its role taken by the kernel's output scale, BoTorch's name for the amplitude $\sigma_f^2$ | @sec-pref-identifiability |
| Posterior mode | solved with `scipy.optimize.fsolve`, warm-started from the previous solution | @alg-pref-fit |
| Numerics | $10^{-6}$ jitter in the Cholesky factorization; duplicate designs within $10^{-4}$ merged | @sec-gp-computation |
| Hyperparameters | `PairwiseLaplaceMarginalLogLikelihood`, the Laplace evidence of Chu and Ghahramani's equation (12) | @eq-approx-laplace-evidence |
| Kernel | scaled RBF with one lengthscale per input; constant mean not optimized | @sec-ard |
| Lengthscale prior | Gamma(2.4, 2.7), initialized at its mode, about 0.52 in every dimension | @sec-pref-higher-dim |
| Output-scale prior | smoothed box on $[0.01, 100]$, constraint $[0.005, 200]$ | @sec-pref-identifiability |
:::

Three entries connect to this chapter. Fixing the noise at 1 and learning the
output scale is the scale-versus-noise trade of @sec-pref-identifiability made
explicit: a large amplitude means decisive answers. Clipping the probit
argument at $\pm 3$ caps the likelihood of any single comparison at about
0.9987 during fitting, which limits the pull of one surprising answer, the
probit tail problem of @fig-cmp-choice-curve; as of September 2026 we found
no paper that records this as a modeling choice (inference). And the lengthscale prior does not scale
with the dimension: version 0.12.0 (September 2024) switched most BoTorch
models to dimension-scaled lengthscale priors but explicitly excluded
`PairwiseGP`, which in 0.18.1 (June 2026) still uses Gamma(2.4, 2.7)
[@botorch2026changelog]. As @tbl-pref-dims illustrates, a fixed lengthscale of
about 0.5 puts typical designs in many dimensions where kernel values are near
zero; that this happens with `PairwiseGP` itself has not been verified directly
(inference).

Other libraries choose differently. The preferential sampler in
optuna-dashboard, based on @takeno2023practicalc, uses a Matérn 3/2 kernel with
one lengthscale per input and a Gamma(5, 10) prior, also independent of the
dimension [@optuna2026sampler], with the approximations of
@sec-approx-inference. @sec-obs-pairwisegp reports the version history and
these choices in more detail.

## Exercises {#sec-pref-exercises}

::: {.exercise #exr-pref-zero-sum}
Show that the weights $\hat{\bm\beta}$ of @eq-pref-mode always sum to zero.
What does this imply about the posterior mean @eq-pref-predict far from all
compared designs, and about $\mu(\vx)$ averaged over many designs when the
kernel is stationary and the domain is large?

::: {.solution}
$\hat{\bm\beta} = \nabla\ell(\hat\vf) = \sum_k (\lambda(z_k)/s)(\mathbf{e}_{v_k} - \mathbf{e}_{u_k})$,
and each $\mathbf{e}_{v_k} - \mathbf{e}_{u_k}$ sums to zero, so
$\mathbf{1}^\T\hat{\bm\beta} = 0$. Far from all compared designs, every
$k(\vx, \vx_i)$ is near zero and $\mu(\vx) \approx 0$, the prior mean, as in
regression. The posterior mean is a sum of kernel bumps whose weights cancel:
with a stationary kernel, each bump integrates to the same amount over a large
domain, so the integral of $\mu$ over the domain is $\sum_i \hat\beta_i$ times
that amount, which is zero. The data raise some regions and lower others by the
same total; they never raise the level, which they cannot measure.
:::
:::

::: {.exercise #exr-pref-two-weight}
In @ex-pref-two with $\rho = 0$ and $\sigma = 1/\sqrt{2}$ (so $s = 1$), find the
mode $\hat\Delta$ of the difference and the weight $w$ numerically, given that the
mode satisfies $\hat\Delta / 2 = \lambda(\hat\Delta)$, where 2 is the prior variance of
the difference. Then compute the Laplace standard deviation of the difference
and compare it with the prior's.

::: {.solution}
The log posterior of $\Delta$ is $-\Delta^2/4 + \log\Phi(\Delta)$, whose derivative vanishes at
$\hat\Delta/2 = \lambda(\hat\Delta)$. Trying values: at $\Delta = 0.8$, $\lambda(0.8) \approx 0.29/0.79 \approx 0.37$
while $\Delta/2 = 0.40$; at $\Delta = 0.75$, $\lambda \approx 0.301/0.773 \approx 0.39$
while $\Delta/2 = 0.375$. So $\hat\Delta \approx 0.77$, where $\lambda \approx 0.38$.
Then $w = \lambda(\hat\Delta + \lambda) \approx 0.38 \times 1.15 \approx 0.44$, and
the Laplace variance of the difference is $v_\Delta/(1 + w v_\Delta) = 2/(1 + 0.88) \approx 1.06$,
a standard deviation of about 1.03 against the prior's $\sqrt{2} \approx 1.41$.
One answer at moderate noise removes about half of the variance of the
difference and none of the variance of the sum.
:::
:::

::: {.exercise #exr-pref-resistance}
Three designs are compared with unit weights. In design (a) the answers form a
chain, 1 with 2 and 2 with 3. In design (b) they form a triangle, adding 1
with 3. Using @eq-pref-resistance, find $\Var(f_1 - f_3)$ under a flat prior in
both, and say how many more comparisons of the pair (1, 3) alone would match the
triangle.

::: {.solution}
(a) Two unit resistors in series: $R_{13} = 2$. (b) The direct edge (resistance
1) is in parallel with the path through 2 (resistance 2), so
$R_{13} = (1 \cdot 2)/(1 + 2) = 2/3$. Repeating the comparison of 1 with 3
directly $r$ times gives $r$ unit resistors in parallel, resistance $1/r$, so
$r = 1.5$ direct comparisons would match the triangle, and one direct comparison
(resistance 1) is already better than the chain. The indirect path through 2 is
worth half a direct comparison.
:::
:::

::: {.exercise #exr-pref-scale}
Show that the posterior of $\vf/\sigma$ under prior $\N(\mathbf{0}, a^2\mK)$
and noise $\sigma$ depends on $a$ and $\sigma$ only through the ratio $a/\sigma$.
What does this imply for fitting both the amplitude and the noise by
maximizing the evidence?

::: {.solution}
Let $\mathbf{u} = \vf/\sigma$. Its prior is $\N(\mathbf{0}, (a/\sigma)^2\mK)$,
and the likelihood @eq-pref-lik is $\prod_k \Phi\big((u_{v_k} - u_{u_k})/\sqrt{2}\big)$,
which does not involve $\sigma$. The posterior of $\mathbf{u}$ is prior times
likelihood, so it depends only on $a/\sigma$. The evidence is the normalizer of
the same product, so it too depends only on $a/\sigma$: any pair $(a, \sigma)$
with the same ratio fits the answers equally well. Only one of them can be
learned, which is why BoTorch fixes the noise and fits the amplitude.
:::
:::

## Further reading {#further-reading .unnumbered}

- @chu2005preference is short and complete: the likelihood, the convexity
  proof, the Laplace approximation, the evidence, and the predictive
  probability of a new preference.
- @rasmussen2006gaussian, chapter 3, gives the numerically stable algorithms for
  the classification case, which carry over with the Laplacian $\mW$.
- @houlsby2011bayesian derive the preference kernel and an information-based
  rule for choosing pairs.
- @shah2016estimation and @hendrickx2019graph explain, for ranking finite sets
  of items, why the comparison graph's spectrum and resistances govern the
  error; @fiedler1973algebraic introduced algebraic connectivity.
- BoTorch's preference tutorial [@botorch2026pairwise] fits `PairwiseGP` and runs
  the loop of @sec-pbo.
