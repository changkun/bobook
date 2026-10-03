---
status: done
synopsis: "Bayesian optimization and preferential Bayesian optimization in about 180 lines of NumPy, mirroring the book's figures and notation, with a three-dimensional example, and the same loops in BoTorch."
sources: ["Rasmussen and Williams 2006, ch. 2 and 3", "Chu and Ghahramani 2005", "Lin et al. 2022", "BoTorch 0.18.1 source"]
---

# A Minimal Implementation {#sec-minimal-implementation}

Every method in the first four parts of this book fits in a short program. This
appendix writes that program: Gaussian process regression, expected
improvement, the optimization loop in one and three dimensions, the Laplace
preference model, EUBO, and the loop of preferential Bayesian optimization
(PBO), about 180 lines of NumPy in all. The code uses the book's notation and its running objective, so each
function can be read next to the equation it implements and each loop next to
the figure that animates it.

The code is written to be read. It refits the model from scratch at every
step, searches a grid or a random candidate set instead of using gradients, and
fixes most hyperparameters. The last section shows the same two loops in
BoTorch, which does these things properly, and says what changes. The
objectives here are test functions; the case studies of @sec-part-cases run
the same loops on measured problems.

The five NumPy listings below are one script: concatenated in order, they
run as printed, with no dependency other than NumPy. The outputs quoted in the
text come from running that script with NumPy 2.5.3 on Python 3.14; other
versions may differ in the last digits, since the results depend on the
random number generator.

## Gaussian process regression {#sec-impl-gp}

The first listing is @alg-gp-regression. `rbf` builds the kernel matrix,
`gp_posterior` computes the posterior mean and variance of @eq-gp-noisy with a
Cholesky factorization, and `log_marginal` computes the log marginal
likelihood of @sec-marginal-likelihood, which a later listing uses to choose
the lengthscale. Inputs are arrays of shape $(n, d)$; the helper `as2d` lets a
plain array of $n$ numbers stand for $n$ points in one dimension, so the same
functions serve both examples.

::: {.code title="NumPy: Gaussian process regression"}
```python
import numpy as np
from math import erf, sqrt, pi

def as2d(x):
    """Inputs as an (n, d) array; a 1-D array is n points in one dimension."""
    x = np.asarray(x, dtype=float)
    return x[:, None] if x.ndim == 1 else x

def rbf(A, B, ell=0.12):
    """RBF kernel matrix k(a, b) = exp(-|a - b|^2 / (2 ell^2)), unit amplitude."""
    A, B = as2d(A) / ell, as2d(B) / ell
    d2 = (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-0.5 * np.maximum(d2, 0.0))

def gp_posterior(X, y, Xs, ell=0.12, noise=1e-4):
    """Posterior mean and variance at Xs (Algorithm 2.1 of Rasmussen and Williams)."""
    L = np.linalg.cholesky(rbf(X, X, ell) + noise * np.eye(len(y)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    Ks = rbf(X, Xs, ell)                       # n x m
    v = np.linalg.solve(L, Ks)
    return Ks.T @ alpha, np.maximum(1.0 - (v**2).sum(0), 1e-12)

def log_marginal(X, y, ell, noise=1e-4):
    """Log marginal likelihood, for choosing the lengthscale."""
    L = np.linalg.cholesky(rbf(X, X, ell) + noise * np.eye(len(y)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return -0.5 * y @ alpha - np.log(np.diag(L)).sum() - 0.5 * len(y) * np.log(2 * pi)
```
:::

Three details are worth a comment. The squared distances are computed from
$\lVert a - b\rVert^2 = \lVert a\rVert^2 + \lVert b\rVert^2 - 2a^\T b$, which is
fast but can come out slightly negative in floating point, hence the
`np.maximum`. The variance is clipped at a tiny positive value for the same
reason: a variance of $-10^{-17}$ would turn into `nan` under a square root.
And the kernel has unit amplitude, so `1.0` stands for $k(\vx, \vx)$ in the
variance; the code compensates by standardizing the outputs before fitting
(@sec-gp-pitfalls).

## Expected improvement and the loop {#sec-impl-bo}

The second listing adds the acquisition function and the loop. `ei` is the
closed form of @sec-ei,
$\EI_n(\vx) = (\mu_n(\vx) - f^*_n)\,\Phi(z) + \sigma_n(\vx)\,\phi(z)$ with
$z = (\mu_n(\vx) - f^*_n)/\sigma_n(\vx)$, where $f^*_n$ is the best value
observed so far (the incumbent of @sec-pi). `Phi`
and `phi` are the standard normal CDF and density; NumPy has no error function,
so the listing vectorizes the one in Python's `math` module. `bo_loop` is the
loop of @sec-bo-algorithm on the running objective: standardize, fit, maximize
expected improvement over a grid of 501 candidates, evaluate, repeat.

::: {.code title="NumPy: expected improvement and the loop in one dimension"}
```python
_erf = np.vectorize(erf)
def Phi(z): return 0.5 * (1.0 + _erf(np.asarray(z) / sqrt(2.0)))
def phi(z): return np.exp(-0.5 * np.asarray(z) ** 2) / sqrt(2.0 * pi)

def ei(mean, var, best):
    """Expected improvement over `best`, in closed form."""
    sd = np.sqrt(var)
    z = (mean - best) / sd
    return (mean - best) * Phi(z) + sd * phi(z)

def f(x):  # the running objective of the book
    return (0.62 * np.exp(-(x - 0.25) ** 2 / (2 * 0.1**2))
            + np.exp(-(x - 0.73) ** 2 / (2 * 0.055**2))
            + 0.1 * np.sin(11 * x + 0.6) - 0.35 * x)

def bo_loop(f, n_init=2, n_iter=10, seed=0):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0, 1, 501)
    X = rng.uniform(0, 1, n_init)
    Y = f(X)
    for _ in range(n_iter):
        y = (Y - Y.mean()) / (Y.std() + 1e-9)    # standardize the outputs
        mean, var = gp_posterior(X, y, grid, ell=0.08, noise=1e-6)
        x_next = grid[np.argmax(ei(mean, var, y.max()))]
        X, Y = np.append(X, x_next), np.append(Y, f(x_next))
    return X, Y

X, Y = bo_loop(f)
print(f"1-D: recommend x = {X[np.argmax(Y)]:.3f}, f = {Y.max():.3f} "
      f"(best on [0, 1]: f = {f(np.linspace(0, 1, 4001)).max():.3f})")
```
:::

It prints

```text
1-D: recommend x = 0.730, f = 0.816 (best on [0, 1]: f = 0.818)
```

Twelve evaluations find the tall, narrow bump. This is not luck with the seed:
over seeds 0 to 19, the best value found had a median of 0.817 and a minimum of
0.788, so every run reached the tall bump. @fig-loop-animation runs the same loop
interactively; choose expected improvement there to watch it leave the wide
bump once the model is sure of it.

### Three dimensions {#sec-impl-hartmann}

Nothing in `gp_posterior` or `ei` depends on the dimension. What does is the
inner search: a grid of 501 points per axis would have 125 million points in
three dimensions. @sec-acq-optimization describes what replaces the grid. The
listing uses the simplest version that works: score a few thousand random
candidates, plus a cloud of small perturbations around the best points
evaluated so far, and take the best. The random candidates explore; the local
cloud refines, since expected improvement often peaks in a small region next
to the incumbent that random candidates would miss.

The test problem is the three-dimensional Hartmann function, a standard
benchmark with several local maxima, negated here so that larger is better;
its maximum on the unit cube is 3.86278. The lengthscale is no longer fixed by
hand: at every step the loop picks, from five candidates, the one with the
highest marginal likelihood.

::: {.code title="NumPy: the loop in three dimensions"}
```python
H_A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
H_P = 1e-4 * np.array([[3689, 1170, 2673], [4699, 4387, 7470],
                       [1091, 8732, 5547], [381, 5743, 8828]])
H_C = np.array([1.0, 1.2, 3.0, 3.2])

def hartmann3(X):  # negated Hartmann-3 on [0, 1]^3; maximum 3.86278
    X = as2d(X)
    return (H_C * np.exp(-(H_A * (X[:, None, :] - H_P) ** 2).sum(-1))).sum(-1)

def next_point(X, y, ell, rng, n_global=2000, n_local=1000, step=0.05):
    """Maximize EI over random candidates plus perturbations of the best points."""
    d = X.shape[1]
    top = X[np.argsort(y)[-5:]]
    local = top[rng.integers(0, len(top), n_local)] + step * rng.normal(size=(n_local, d))
    cand = np.clip(np.vstack([rng.uniform(0, 1, (n_global, d)), local]), 0, 1)
    mean, var = gp_posterior(X, y, cand, ell, noise=1e-6)
    return cand[np.argmax(ei(mean, var, y.max()))]

def bo_nd(f, d, n_init=6, n_iter=30, seed=0, ells=(0.1, 0.15, 0.2, 0.3, 0.5)):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, (n_init, d))
    Y = f(X)
    for _ in range(n_iter):
        y = (Y - Y.mean()) / (Y.std() + 1e-9)
        ell = max(ells, key=lambda l: log_marginal(X, y, l, noise=1e-6))
        X = np.vstack([X, next_point(X, y, ell, rng)])
        Y = np.append(Y, f(X[-1:]))
    return X, Y

X3, Y3 = bo_nd(hartmann3, d=3)
R3 = hartmann3(np.random.default_rng(1).uniform(0, 1, (36, 3)))
print(f"3-D: best of 36 evaluations {Y3.max():.3f} at {np.round(X3[np.argmax(Y3)], 3)}; "
      f"random search {R3.max():.3f}; maximum 3.863")
```
:::

It prints

```text
3-D: best of 36 evaluations 3.850 at [0.172 0.539 0.848]; random search 3.614; maximum 3.863
```

Over seeds 0 to 9, the best of 36 evaluations ranged from 3.801 to 3.850, with
a median of 3.837; random search with the same budget (36 uniform points from
`np.random.default_rng(s)` for the same ten seeds) ranged from 2.771 to 3.782,
with a median of 3.495. The figure shows the same kind of run, where
you can rotate the cube of evaluated points and see the model along slices
through the best one.

```{figure}
//| figure: dim-bo-3d
//| label: fig-impl-hartmann
//| fig-cap: "Bayesian optimization on the three-dimensional Hartmann function, as in the listing above: 6 initial points, then expected improvement, 36 evaluations in all. The figure's initial design is a Latin hypercube and its lengthscale is fixed by the slider, where the listing uses uniform random points and chooses the lengthscale by marginal likelihood; the loop is otherwise the same. Left: the evaluated points in the unit cube, with the true maximizer as a star. Right: slices of the model through the best point. Below: the best value found."
init: 6
steps: 36
```

::: {.pitfall title="A single lengthscale"}
`rbf` uses one lengthscale for all inputs. Hartmann-3 varies much faster along
its third input than along its first, which a lengthscale per input
(@sec-ard) would capture and one shared lengthscale cannot. The loop works
here because three dimensions forgive a rough model; @sec-high-dimensions
shows where that stops.
:::

## The preference model {#sec-impl-pref}

The fourth listing is the model of @sec-gp-preference: a Gaussian process
utility observed through comparisons, with the probit likelihood
$\Prob(\vx_w \succ \vx_l) = \Phi\big((g(\vx_w) - g(\vx_l))/(\sqrt{2}\sigma)\big)$
and the Laplace approximation (@sec-pref-laplace). The data are a list of
compared inputs and a list of comparisons, each a pair of indices (winner,
loser) into that list. The unknowns are the utilities $\vf$ at the compared
inputs.

The Laplace approximation needs the mode of the posterior over $\vf$, found by
Newton's method. Each step needs the gradient $\nabla\ell$ and the negative
Hessian $\mW$ of the log-likelihood, @eq-pref-grad-hess, which
`laplace_terms` computes. A comparison between inputs $w$ and $l$ adds
$\lambda(z)/(\sqrt{2}\sigma)$ to the winner's gradient entry and subtracts it
from the loser's, where $\lambda(z) = \phi(z)/\Phi(z)$ is the inverse Mills
ratio at the comparison's standardized difference $z$ (`r` in the code). It
adds the weight $w_k = \lambda(z)\big(z + \lambda(z)\big)/(2\sigma^2)$ (`c` in
the code) to the $2 \times 2$ block of $\mW$ for those two inputs, with a plus
sign on the diagonal and a minus sign off it. Summed over comparisons, $\mW$
is the weighted Laplacian of the comparison graph of @sec-comparison-graph.

The Newton step is @eq-approx-newton with $\nabla\ell(\vf)$ as the gradient of
the log-likelihood,
$\vf_{\text{new}} = \mK(\mI + \mW\mK)^{-1}\big(\mW\vf + \nabla\ell(\vf)\big)$,
which `fit_preference` computes with one linear solve and no inverse of
$\mK$; the code's variable `g` holds $\nabla\ell(\vf)$. One more fact saves
work at the end. At the mode the gradient of the log posterior is zero, so
$\mK^{-1}\hat{\vf} = \nabla\ell(\hat\vf)$ (@eq-pref-mode), and the predictive
mean $\vk(\vx)^\T \mK^{-1}\hat{\vf}$ is $\vk(\vx)^\T \nabla\ell(\hat\vf)$,
with no solve at all.

The predictive covariance is that of a Gaussian process whose "observations"
have noise covariance $\mW^{-1}$:
$k(\vx, \vx') - \vk(\vx)^\T(\mK + \mW^{-1})^{-1}\vk(\vx')$, which the code
computes as $\vk(\vx)^\T(\mI + \mW\mK)^{-1}\mW\,\vk(\vx')$ subtracted from the
prior, because $\mW$ is singular and has no inverse (every row of a graph
Laplacian sums to zero, the shift invariance of @sec-pref-identifiability).

::: {.code title="NumPy: the Laplace preference model"}
```python
def mills(z):
    """phi(z) / Phi(z), stable far into the lower tail."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    lo = z < -5
    out[~lo] = phi(z[~lo]) / Phi(z[~lo])
    out[lo] = -z[lo] / (1 - 1 / z[lo] ** 2 + 3 / z[lo] ** 4)
    return out

def laplace_terms(f, w, l, sigma):
    """Gradient and negative Hessian W of the probit log-likelihood of the comparisons."""
    s = sqrt(2.0) * sigma
    z = (f[w] - f[l]) / s
    r = mills(z)                               # lambda(z), the inverse Mills ratio
    g = np.zeros(len(f)); np.add.at(g, w, r / s); np.add.at(g, l, -r / s)
    c = r * (z + r) / s**2                     # weight w_k of each comparison, positive
    W = np.zeros((len(f), len(f)))
    np.add.at(W, (w, w), c); np.add.at(W, (l, l), c)
    np.add.at(W, (w, l), -c); np.add.at(W, (l, w), -c)
    return g, W                                # W is a weighted graph Laplacian

def fit_preference(X, comps, ell=0.12, sigma=0.1, iters=50):
    """Laplace approximation for comparisons [(winner, loser), ...] indexing rows of X."""
    X = as2d(X)
    w, l = np.array(comps).T
    K = rbf(X, X, ell) + 1e-8 * np.eye(len(X))
    f = np.zeros(len(X))
    for _ in range(iters):                     # Newton's method on the log posterior
        g, W = laplace_terms(f, w, l, sigma)
        f_new = K @ np.linalg.solve(np.eye(len(X)) + W @ K, W @ f + g)
        done = np.max(np.abs(f_new - f)) < 1e-9
        f = f_new
        if done:
            break
    g, W = laplace_terms(f, w, l, sigma)
    return dict(X=X, K=K, g=g, W=W, ell=ell)

def predict_preference(m, Xs):
    """Latent posterior mean and covariance at Xs under the Laplace approximation."""
    Ks = rbf(m["X"], Xs, m["ell"])
    mean = Ks.T @ m["g"]                       # at the mode, K^-1 f equals the gradient
    B = np.eye(len(m["X"])) + m["W"] @ m["K"]
    cov = rbf(Xs, Xs, m["ell"]) - Ks.T @ np.linalg.solve(B, m["W"] @ Ks)
    return mean, cov
```
:::

The Newton iteration here takes full steps. The probit log-likelihood is
concave, so the log posterior has a single mode and full steps converge in the
runs of this appendix; a production implementation damps the step or uses a
trust region for safety, and BoTorch's `PairwiseGP` hands the problem to a
root finder from SciPy [@botorch2026pairwisegp]. `mills` switches to an
asymptotic expansion below $z = -5$, where $\Phi(z)$ underflows and the plain
ratio would be $0/0$.

## EUBO and the preferential loop {#sec-impl-pbo}

The fifth listing completes @alg-pbo. `eubo` evaluates @eq-eubo-closed for
every pair of candidates at once, from the posterior mean vector and
covariance matrix over a candidate grid: the entry in row $i$ and column $j$
is the expected utility of the better of candidates $i$ and $j$. `ask` is a
simulated person who answers by the probit model with a hidden utility. In
`pbo_loop`, each round asks one comparison, refits the model, and takes the next
pair from the largest off-diagonal entry of the EUBO matrix.

::: {.code title="NumPy: EUBO and the preferential loop"}
```python
def eubo(mean, cov):
    """EUBO for every pair of candidates (Clark's formula), as a matrix."""
    v = np.diag(cov)
    s = np.sqrt(np.maximum(v[:, None] + v[None, :] - 2 * cov, 1e-12))
    d = mean[:, None] - mean[None, :]
    return mean[:, None] * Phi(d / s) + mean[None, :] * Phi(-d / s) + s * phi(d / s)

def ask(u, a, b, sigma, rng):
    """A simulated person: prefers a with the probit probability."""
    return rng.uniform() < Phi((u(a) - u(b)) / (sqrt(2.0) * sigma))

def pbo_loop(u, n_iter=15, sigma=0.1, ell=0.08, rule="eubo", seed=0):
    rng = np.random.default_rng(seed)
    cand = np.linspace(0, 1, 51)
    pts, comps = [], []
    def idx(x):                                # one latent value per distinct input
        for i, p in enumerate(pts):
            if abs(p - x) < 1e-9:
                return i
        pts.append(x)
        return len(pts) - 1
    a, b = rng.choice(cand, 2, replace=False)  # the first pair is random
    for _ in range(n_iter):
        ia, ib = idx(a), idx(b)
        comps.append((ia, ib) if ask(u, a, b, sigma, rng) else (ib, ia))
        m = fit_preference(np.array(pts), comps, ell=ell, sigma=sigma)
        mean, cov = predict_preference(m, cand)
        if rule == "random":
            a, b = rng.choice(cand, 2, replace=False)
            continue
        E = eubo(mean, cov)
        np.fill_diagonal(E, -np.inf)           # a pair needs two different options
        i, j = np.unravel_index(np.argmax(E), E.shape)
        a, b = cand[i], cand[j]
    return cand[np.argmax(mean)], len(pts)

x_best, n_pts = pbo_loop(f)
print(f"PBO: after 15 comparisons among {n_pts} inputs, recommend x = {x_best:.2f}")
```
:::

With the running objective as the hidden utility, it prints

```text
PBO: after 15 comparisons among 14 inputs, recommend x = 0.70
```

The true favorite is at $x = 0.73$, so fifteen one-bit answers were enough here
to find the tall bump on a grid with spacing 0.02. The run is less reliable
than its counterpart with numbers. Over seeds 0 to 19, the recommendation was
within 0.05 of the favorite in 15 runs after 15 comparisons, and in 16 after
30. In
the others the model settled on the wide bump near $x = 0.2$ and kept asking
about nearly the same pairs there: the collapse of EUBO toward the current
best that @sec-pbo-failure-modes describes. Random pairs, which the listing
selects with `rule="random"`, did worse with few comparisons and caught up
with more: 5 of 20 after 15 comparisons and 12 of 20 after 30. The lengthscale
matters too: with 0.12 in place of 0.08, EUBO found the tall bump in 10 of 20
runs after 15 comparisons, because a model that smooth cannot represent a bump that
narrow.

The figure runs the same loop on the same objective, one comparison at a time, and
shows the EUBO matrix that `eubo` returns.

```{figure}
//| figure: eubo-map
//| label: fig-impl-eubo
//| fig-cap: "The matrix returned by `eubo`, after five comparisons on the running objective with lengthscale 0.08, the value the listing uses. Left: the utility posterior and the comparisons so far. Right: EUBO for every pair of candidates; the largest off-diagonal entry is the next pair. Press *Ask the next pair* to take one step of `pbo_loop`, with an answer given by the objective."
lengthscale: 0.08
```

## The same in BoTorch {#sec-impl-botorch}

BoTorch [@balandat2020botorch] provides both loops as library components.
The names below were checked against the source of BoTorch 0.18.1, and the
listings were run with that version, PyTorch 2.14.1, and GPyTorch 1.15.2.

The numerical loop first, on the same Hartmann problem:

::: {.code title="BoTorch: Bayesian optimization with expected improvement"}
```python
import warnings

warnings.filterwarnings("ignore")              # PairwiseGP is noisy; see the note below

import torch
from botorch.acquisition import LogExpectedImprovement
from botorch.acquisition.preference import AnalyticExpectedUtilityOfBestOption
from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import PairwiseGP, SingleTaskGP
from botorch.models.pairwise_gp import PairwiseLaplaceMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.test_functions import Hartmann
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.manual_seed(0)
torch.set_default_dtype(torch.double)
f3 = Hartmann(dim=3, negate=True)              # maximum 3.86278 on [0, 1]^3
bounds = torch.stack([torch.zeros(3), torch.ones(3)])

# --- Bayesian optimization with expected improvement --------------------------
X = torch.rand(6, 3)
Y = f3(X).unsqueeze(-1)                         # n x 1
for _ in range(30):
    gp = SingleTaskGP(X, Y)                     # standardizes Y, fits one lengthscale per input
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    acq = LogExpectedImprovement(gp, best_f=Y.max())
    x_next, _ = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=10, raw_samples=512)
    X = torch.cat([X, x_next])
    Y = torch.cat([Y, f3(x_next).unsqueeze(-1)])
print(f"BO: best of {len(Y)} evaluations {Y.max().item():.3f}")
```
:::

It prints `BO: best of 36 evaluations 3.862`; over seeds 0 to 4 the result
ranged from 3.855 to 3.862, against 3.801 to 3.850 for the NumPy loop. The
structure is the same as `bo_nd`, and each line hides a better version of
what the NumPy code does:

- `SingleTaskGP` standardizes the outputs with an outcome transform and uses
  an RBF kernel with one lengthscale per input (@sec-ard).
- `fit_gpytorch_mll` maximizes the marginal likelihood over the lengthscales
  and the noise by gradient-based optimization, where `bo_nd` compared five
  values of one lengthscale.
- `LogExpectedImprovement` is the logarithm of expected improvement, computed
  in a numerically stable way. It has the same maximizer, and it keeps useful
  gradients in the large regions where expected improvement itself is too close
  to zero to guide a search [@ament2023unexpected].
- `optimize_acqf` maximizes the acquisition function with a gradient-based
  optimizer started from several points, chosen from 512 random candidates.

The preferential loop uses `PairwiseGP`, the model of @sec-pairwisegp, with
the analytic EUBO:

::: {.code title="BoTorch: preferential Bayesian optimization with EUBO"}
```python
# continues the script above
def answer(pair, sigma=0.1):
    """A simulated person: index pair [winner, loser] within the two options."""
    u = f3(pair) + sigma * torch.randn(2)
    return torch.tensor([[0, 1]]) if u[0] > u[1] else torch.tensor([[1, 0]])

def fit_pairwise(X, comps):
    model = PairwiseGP(X, comps)
    try:
        fit_gpytorch_mll(PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model))
    except ModelFittingError:                   # keep the default hyperparameters
        model.eval()
    return model

X = torch.rand(2, 3)
comps = answer(X)                               # m x 2, rows [winner index, loser index]
for _ in range(30):
    model = fit_pairwise(X, comps)
    acq = AnalyticExpectedUtilityOfBestOption(pref_model=model)
    pair, _ = optimize_acqf(acq, bounds=bounds, q=2, num_restarts=8, raw_samples=256)
    comps = torch.cat([comps, answer(pair) + len(X)])
    X = torch.cat([X, pair])
best = X[fit_pairwise(X, comps).posterior(X).mean.argmax()]
print(f"PBO: after {len(comps)} comparisons, recommended point has utility {f3(best).item():.3f}")
```
:::

`PairwiseGP` takes the compared inputs and an integer tensor of comparisons
whose rows are (winner index, loser index), the same data layout as
`fit_preference`. It uses the probit likelihood and the Laplace approximation,
with one difference of parameterization: instead of a noise scale $\sigma$ it
fixes the scale of the likelihood and learns the amplitude of the kernel, which
is the same model, since only the ratio of the two can be identified
(@sec-pref-identifiability). `AnalyticExpectedUtilityOfBestOption` is
@eq-eubo-closed, and calling `optimize_acqf` with `q=2` optimizes it over both
members of the pair jointly, in the six-dimensional space of pairs. For
queries of more than two options, `qExpectedUtilityOfBestOption` estimates
$\E_n[\max_i g(\vx_i)]$ by sampling from the posterior, and is used the same
way with a larger `q`. Since `PairwiseGP` accepts only pairs, a choice among
$q$ options then has to be recorded as comparisons of the chosen option with
each of the others, an approximation of @eq-query-luce.

The numerical loop is reproducible: it printed the same 3.862 in every run.
The preferential loop was not, in our environment, even with the seed fixed.
Eleven runs of the script printed utilities for the recommended point between
2.18 and 3.84, with a median of 3.23, against a maximum of 3.863. Small
numerical differences in fitting the model change which pair is asked next,
and after 30 rounds the runs have diverged. Either way the comparison with the
numerical loop stands: it reached 3.86 with 36 evaluations, while 31
comparisons, each worth at most one bit, usually leave the three-dimensional search
unfinished. This is the gap that the query designs of @sec-query-design try to
close.

::: {.pitfall title="Warnings and failed fits from PairwiseGP"}
The listing silences warnings before importing BoTorch, as BoTorch's own
tutorial does [@botorch2026pairwise], because `PairwiseGP` prints many
numerical warnings ("added jitter of 1.0e-06 to the diagonal") when compared
points lie close together. In some of our runs with BoTorch 0.18.1 a
hyperparameter fit also failed outright with `ModelFittingError`; the helper
`fit_pairwise` catches that and keeps the model's default hyperparameters for
the round. A loop that talks to a person should never crash on a failed fit.
:::

## Exercises {#sec-impl-exercises}

::: {.exercise #exr-impl-ucb}
Replace expected improvement in `bo_loop` by the upper confidence bound of
@sec-ucb, $\mu_n(\vx) + \beta^{1/2}\sigma_n(\vx)$ with $\beta^{1/2} = 2$. What changes in
the code, and why does the standardization of the outputs matter more for this
rule than for expected improvement?

::: {.solution}
One line changes: `x_next = grid[np.argmax(mean + 2.0 * np.sqrt(var))]`. The
incumbent `y.max()` is no longer needed. Standardization matters because the
weight 2 multiplies a standard deviation in the units of the model: with unit
prior variance on standardized outputs, 2 means "two prior standard
deviations". Without standardization, the same number would be too large for
outputs that vary by 0.01 and too small for outputs that vary by 1000.
Expected improvement has no such constant; standardization affects it only
through the fit of the model.
:::
:::

::: {.exercise #exr-impl-choice}
Extend `laplace_terms` to choices from a set (@eq-query-luce): each
observation is a chosen index `c` and an array `S` of the indices shown,
including `c`, with temperature `tau`. Write the contribution of one
observation to the gradient and to $\mW$.

::: {.solution}
With $p_j = e^{f_j/\tau}/\sum_{k \in S} e^{f_k/\tau}$ for $j \in S$, the
gradient contribution is $(\mathbb{1}[j = c] - p_j)/\tau$ and the block of
$\mW$ on $S$ is $(\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\T)/\tau^2$
(@sec-query-choice):

```python
def choice_terms(f, c, S, tau, g, W):
    p = np.exp((f[S] - f[S].max()) / tau)
    p /= p.sum()
    g[S] += ((S == c) - p) / tau
    W[np.ix_(S, S)] += (np.diag(p) - np.outer(p, p)) / tau**2
```

Subtracting the maximum before exponentiating avoids overflow. Each block is
positive semidefinite and its rows sum to zero, so $\mW$ is again a weighted
graph Laplacian and the rest of `fit_preference` is unchanged.
:::
:::

::: {.exercise #exr-impl-graph}
After `pbo_loop` has run, the answered pairs form a comparison graph on the compared
inputs (@sec-comparison-graph). Write a function that counts its connected
components, and explain what a count above 1 means for the posterior.

::: {.solution}
```python
def components(n, comps):
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for w, l in comps:
        parent[find(w)] = find(l)
    return len({find(i) for i in range(n)})
```

With more than one component, no chain of answers connects the groups, so the
likelihood says nothing about how the utilities of one group compare with
those of another: $\mW$ has one zero eigenvalue per component. Only the prior,
through the kernel, relates the groups. EUBO tends to produce such graphs
because its pairs often share no input with earlier ones
(@sec-pbo-failure-modes).
:::
:::

## Further reading {#further-reading .unnumbered}

- @rasmussen2006gaussian, Algorithm 2.1 for regression and Section 3.4 for
  the Laplace approximation with Newton's method, are the templates for the
  first and fourth listings.
- @chu2005preference is the preference model; @lin2022preference and
  @astudillo2023qeubob are EUBO and qEUBO.
- @balandat2020botorch describes BoTorch; its preference tutorial
  [@botorch2026pairwise] runs the preferential loop on a four-dimensional
  problem.
- @ament2023unexpected explain why the logarithm of expected improvement is
  easier to optimize than expected improvement.
