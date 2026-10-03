---
status: done
synopsis: "The problem Bayesian optimization solves, the loop that solves it (fit a surrogate, maximize an acquisition function, evaluate, update), the trade-off between exploring and exploiting that every acquisition function must strike, how to choose the first points, and where the method came from."
sources: ["Garnett 2023, ch. 1, 9, 12","Frazier 2018","Shahriari et al. 2016","Brochu et al. 2010","Jones et al. 1998"]
---

# The Bayesian Optimization Loop {#sec-bo-loop}

@sec-model-then-decide named the two ingredients of Bayesian optimization: a
model of the objective that knows what it does not know, and a rule that turns
that knowledge into the next experiment. @sec-part-gp built the first ingredient.
A Gaussian process, conditioned on the evaluations made so far, gives a
posterior mean that says what the function probably looks like and a posterior
standard deviation that says how unsure the model is at every input
(@sec-gp-regression). This chapter puts that model inside a loop.

The loop itself is short enough to fit on an index card: fit the model, pick
the input where one more evaluation looks most worthwhile, evaluate it, add the
result to the data, and repeat. Most of the ideas in the rest of the book are
refinements of one of those four steps. What makes the loop interesting is the
second step, because "most worthwhile" has no single right answer. An input
can be worth evaluating because the model expects a high value there, or
because the model knows little about it and a high value might be hiding
there. Every rule for choosing the next input strikes some balance between
those two reasons, and this chapter lets you feel what happens when the
balance is wrong in either direction.

We start by stating the problem precisely, then write the loop as an
algorithm and watch it run on the book's running example. The middle of the
chapter is about the trade-off between exploring and exploiting. It ends with
two practical questions, how to choose the first few points before the model
has anything to go on, and where the method came from. @sec-acquisition then
derives the rules for choosing the next input one by one.

## The problem {#sec-bo-problem}

Let $f$ be a function from an input domain $\X$ to the real numbers. We want an
input where $f$ is largest:

$$
\vx^\star \in \argmax_{\vx \in \X} f(\vx).
$$ {#eq-loop-problem}

Written this way, the problem looks like any other optimization problem.
What sets Bayesian optimization apart are the conditions under which it is
meant to work, which @frazier2018tutorial lists explicitly.

- **Each evaluation is expensive.** Evaluating $f$ at one input takes minutes
  or hours, costs money, or asks something of a person. The number of
  evaluations is limited, typically to at most a few hundred. We call the
  allowed number the *budget* and write it $N$.
- **The function is a black box.** We can evaluate $f$ wherever we like, but we
  have no formula for it and no known structure such as convexity or
  linearity that a specialized method could exploit.
- **There are no derivatives.** An evaluation returns a value and nothing else,
  so gradient descent and its relatives are unavailable.
- **The domain is simple and not too large.** Usually $\X$ is a box, a range for
  each of $d$ inputs, which we rescale to $[0, 1]^d$. Most successful
  applications have $d \le 20$ [@frazier2018tutorial]; @sec-high-dim-practice
  discusses what changes beyond that.
- **The function is reasonably smooth.** Nearby inputs tend to give similar
  values. This is the assumption the Gaussian process prior encodes, and
  without it no finite number of evaluations would say anything about the
  inputs between them.

Two published problems show what these conditions look like in practice.
@snoek2012practical tuned nine hyperparameters of a convolutional neural
network for image classification, among them the learning rate, the number of
training epochs, and four weight penalties. Each evaluation was a full
training run, and the best setting found reached a test error of 14.98% on
the CIFAR-10 benchmark, against 18% for the setting a human expert had tuned.
@shields2021bayesian optimized chemical reactions, where each evaluation is an
experiment in the laboratory. In a benchmark run as an online game against
expert chemists and engineers, the optimizer needed fewer experiments on
average, and its results depended less on the data it started from.
@sec-cs-classifier and @sec-cs-chemistry work through problems of these two
kinds end to end.

Evaluations may also be noisy. Training a neural network twice with
different random seeds gives two different accuracies; a person rating the
same coffee twice gives two different scores. We then observe
$y = f(\vx) + \varepsilon$ with noise $\varepsilon$, the model of
@sec-gp-noise, and the task is still to maximize $f$, not the noisy $y$.

Because the budget is finite, the method does not have to find $\vx^\star$
exactly. After $N$ evaluations it must return a single *recommendation*
$\hat\vx_N$, and it is judged by how close $f(\hat\vx_N)$ comes to the best
achievable value $f^\star = f(\vx^\star)$. The gap $f^\star - f(\hat\vx_N)$ is called
the *simple regret*; @sec-regret-definitions defines it and its relatives
carefully. Here it is enough that the gap is what we want small, and that
evaluations which teach us a lot but score badly are not wasted if they lead
to a better recommendation.

The figures in this part of the book use one running objective on $[0, 1]$,
drawn as a dashed aqua curve. It has a broad bump near $x = 0.23$ that reaches
about $0.53$ and a narrower, taller bump near $x = 0.73$ that reaches $0.82$,
on a gently falling, slightly wavy baseline. The shape is chosen to be a trap:
a search that finds the broad bump first can easily settle there and never
discover the taller one. The figures can show the dashed curve because they
know the formula. The loop never sees it. It sees only the values at the
inputs it chooses to evaluate.

## The loop {#sec-bo-algorithm}

The central idea of Bayesian optimization is to trade one hard problem for a
sequence of easy ones. Maximizing $f$ directly is hard because every
evaluation is expensive. Instead, at each step, the method builds a cheap
function from the model and maximizes that. The cheap function scores every
candidate input by how useful it would be to evaluate $f$ there next. Finding
its maximum may take thousands of evaluations, but each costs microseconds,
not hours.

Write $\D_n = \{(\vx_1, y_1), \dots, (\vx_n, y_n)\}$ for the data after $n$
evaluations, and $\mu_n(\vx)$ and $\sigma_n(\vx)$ for the posterior mean and
standard deviation of $f(\vx)$ given $\D_n$, computed with @eq-gp-noisy. Here
the subscript $n$ counts evaluations, and $\sigma_n(\vx)$ is always written
with its argument. It is not the noise standard deviation $\sigma_n$ of
@sec-gp-noise, which has no argument and whose subscript stands for noise.

::: {.definition #def-loop-acquisition title="Acquisition function"}
An *acquisition function* is a function $a_n : \X \to \R$, computed from
the posterior given $\D_n$, that scores how useful it would be to evaluate $f$
at each input next. The loop evaluates $f$ where the acquisition function is
largest:

$$
\vx_{n+1} \in \argmax_{\vx \in \X} a_n(\vx).
$$ {#eq-loop-next}
:::

Most acquisition functions depend on $\vx$ only through $\mu_n(\vx)$ and
$\sigma_n(\vx)$, sometimes with the best value observed so far, so they can be
evaluated anywhere at the cost of one posterior prediction. With the two
ingredients in place, the whole method is one loop.

::: {.algorithm #alg-loop-bo title="Bayesian optimization"}
Input: domain $\X$, budget $N$, a Gaussian process prior (kernel and mean),
an acquisition function $a$, an initial design size $n_0 < N$.

1. **Initial design.** Choose $\vx_1, \dots, \vx_{n_0}$ without the model
   (@sec-initial-design), evaluate $f$ at each, and collect $\D_{n_0}$.
2. For $n = n_0, n_0 + 1, \dots, N - 1$:
   1. **Fit.** Condition the Gaussian process on $\D_n$ to get $\mu_n(\vx)$
      and $\sigma_n(\vx)$, refitting the kernel's hyperparameters if desired
      (@sec-fitting-hyperparameters).
   2. **Decide.** Find $\vx_{n+1} \in \argmax_{\vx} a_n(\vx)$ with an
      ordinary numerical optimizer (@sec-acq-optimization).
   3. **Query.** Evaluate the objective at $\vx_{n+1}$.
   4. **Answer.** Receive $y_{n+1} = f(\vx_{n+1}) + \varepsilon_{n+1}$.
   5. **Update.** Set $\D_{n+1} = \D_n \cup \{(\vx_{n+1}, y_{n+1})\}$.
3. Return a recommendation $\hat\vx_N$ (@sec-loop-recommend).

This is the structure of Algorithm 1 of @frazier2018tutorial and Algorithm
1.1 of @garnett2023bayesian.
:::

The algorithm separates *query*, *answer*, and *update* into three steps even
though here they are one function call. Keeping them apart pays off later.
In preferential Bayesian optimization (PBO, @sec-pbo) the query becomes a pair of
inputs shown to a person, the answer becomes a single bit saying which one
they preferred, and the update can no longer use the closed-form Gaussian
process formulas, because a comparison is not a noisy value
(@sec-gp-preference). The loop around those steps stays the same.

The three kinds of work inside the loop have very different costs, and the
design of the method follows from that. Evaluating $f$ is the expensive step,
which is the reason for everything else. Fitting the model costs $O(n^3)$ for
the Cholesky factorization (@sec-gp-computation), which for the few hundred
observations of a typical run takes well under a second. Maximizing the
acquisition function needs many evaluations of $\mu_n(\vx)$ and
$\sigma_n(\vx)$, each costing $O(n^2)$ after the factorization. All of this computation is
negligible next to an evaluation that takes an hour, so it is worth spending
a great deal of arithmetic to choose each evaluation well.

### A first acquisition function {#sec-loop-first-acq}

To run the loop we need one concrete acquisition function. @sec-acquisition
derives the standard ones; here a single, transparent rule is enough.
Score each input by an optimistic estimate of its value, the posterior mean
plus a multiple of the posterior standard deviation:

$$
a_n(\vx) = \mu_n(\vx) + \beta^{1/2}\, \sigma_n(\vx),
$$ {#eq-loop-ucb}

where $\beta \ge 0$ is a constant we choose. With $\beta^{1/2} = 2$, the score
is close to the top edge of the 95% credible band that the figures draw (the
band extends $1.96$ standard deviations above the mean), so the rule picks the
input whose plausible best case is highest. The rule is called the *upper
confidence bound*, UCB for short, and @sec-ucb returns to it, including the
theory that tells how $\beta$ should grow over time.

The two terms of @eq-loop-ucb pull in different directions, which is the
subject of @sec-explore-exploit. The mean term favors inputs the model already
believes are good. The standard deviation term favors inputs the model knows
little about. The weight $\beta^{1/2}$ sets the exchange rate between them: how
many units of expected value one unit of uncertainty is worth.

### The loop on the running example {#sec-loop-running}

@fig-loop-animation runs @alg-loop-bo on the running objective with
@eq-loop-ucb and $\beta^{1/2} = 2$. The run starts from two random inputs. One
of them happens to land on top of the broad bump, at $x \approx 0.23$, so the
loop starts in exactly the trap the objective was designed to set.

```{figure}
//| figure: bo-loop
//| label: fig-loop-animation
//| fig-cap: "Bayesian optimization on the running objective. Each step of the timeline is one iteration of @alg-loop-bo. Top: the hidden objective (dashed), the posterior mean and 95% band given the evaluations so far (dots, the newest ringed), and the next query (orange line). Bottom: the acquisition function and its maximum. The figure opens on the upper confidence bound of @eq-loop-ucb; its *UCB weight √β* slider sets the multiplier of the standard deviation, $\beta^{1/2}$. The other choices are derived in @sec-acquisition. The kernel and its lengthscale are fixed, not fitted, so the pictures stay comparable."
acq: ucb
seed: 9
```

Stepping through the timeline shows the pattern most runs follow.

**The first queries go where the band is widest.** With only two observations
the standard deviation term dominates @eq-loop-ucb almost everywhere, so the
loop spends its first steps on inputs far from both observations, including
the two ends of the domain. These evaluations are not wasted. Each one
collapses the band near it and rules out a region.

**Optimism finds the tall bump.** Around the fifth or sixth
step, the remaining wide part of the band sits over the tall bump, and an
evaluation there returns a value well above anything seen so far. The mean
jumps up, and from then on the mean term pulls the queries back to that
region.

**The end of the run refines.** Once the band is narrow everywhere except near
the best region, the queries cluster around $x \approx 0.72$, the best value
found approaches the true maximum, and the acquisition function becomes a
narrow spike.

**Try another rule and another start.** Switch the acquisition to EI
(expected improvement), PI (probability of improvement), or Thompson sampling,
three rules derived in @sec-acquisition, and press *New run* for different
initial points. The details change; the pattern of broad search followed by
refinement does not. Switch back to UCB, set the *UCB weight √β* slider to 0,
and the loop never leaves the broad bump.

**The same loop in three dimensions.** Nothing in @alg-loop-bo depends on the input
being a single number. The figure below runs it on the Hartmann function, a
standard three-dimensional test problem with one global maximum and a few local
ones. A curve can no longer show the posterior, so the figure shows the evaluated
points inside the unit cube (drag it to rotate) and three slices of the model
through the best point found so far, one along each input.

```{figure}
//| figure: dim-bo-3d
//| label: fig-loop-3d
//| fig-cap: "The loop on the three-dimensional Hartmann function: 5 initial points, then expected improvement, 25 evaluations in all. Left: evaluated points in the unit cube, larger and more opaque for higher values, with drop lines to the floor for depth; the newest point is ringed, the best is orange, and the star is the true maximizer. Right: slices of the model through the best point, one per input, with the posterior mean and 95% band (blue) and the true function along the same line (dashed). Below: the best value found. Step through the run, rotate the cube, and press *New run* for other initial designs."
```

Two things are worth looking for. Early points spread through the cube, and
later ones cluster, the same broad-then-narrow pattern as in one dimension.
And the default run stalls for most of its budget: from the ninth evaluation
to the twenty-third, the best point is $(0.83, 0.56, 0.87)$, with value 3.59
against a maximum of 3.86. Step back to evaluation 23, and the slice along
$x_1$ shows the true function climbing toward $x_1 \approx 0.1$, exactly where
the model's band is still wide. Evaluation 24 goes that way and reaches 3.76
at $(0.33, 0.51, 0.85)$, still short of the maximizer at $x_1 = 0.11$. A
longer budget, a different start, or a more exploratory rule finds the global
maximum; a slice is often the quickest way to see that a run has not.

### What the loop returns {#sec-loop-recommend}

When the budget runs out, the loop must name one input. Two choices are
common: the evaluated input with the best observed value, or the input with
the highest posterior mean [@frazier2018tutorial]. When evaluations are
exact, the first is safe: its value has been measured and is not in doubt.

With noisy evaluations the best observed value is a biased guide. Among many
noisy measurements, the largest tends to be one whose noise happened to be
positive, so the input that produced it is probably not as good as its
measurement suggests. Recommending the input with the highest posterior mean,
either among the evaluated inputs or over the whole domain, uses all the
evaluations near an input instead of a single lucky one. @sec-practice-noise
returns to this choice, and @sec-knowledge-gradient shows that it changes
which acquisition function is the principled one.

::: {.code title="NumPy"}
```python
import numpy as np
# rbf() and gp_posterior() from the Gaussian process chapter

def f(x):  # the running objective; pretend each call takes an hour
    return (0.62 * np.exp(-(x - 0.25) ** 2 / (2 * 0.1**2))
            + np.exp(-(x - 0.73) ** 2 / (2 * 0.055**2))
            + 0.1 * np.sin(11 * x + 0.6) - 0.35 * x)

rng = np.random.default_rng(0)
grid = np.linspace(0, 1, 501)      # candidates for the inner search
X = rng.uniform(0, 1, size=2)      # initial design
Y = f(X)
for n in range(10):
    m = Y.mean()                   # constant prior mean
    mean, var = gp_posterior(X, Y - m, grid, ell=0.08)
    sd = np.sqrt(np.maximum(var, 0.0))
    acq = mean + m + 2.0 * sd      # mean + sqrt(beta) * sd
    x_next = grid[np.argmax(acq)]  # decide
    X = np.append(X, x_next)       # query ...
    Y = np.append(Y, f(x_next))    # ... answer, and update

print("recommend x =", X[np.argmax(Y)], "with f =", Y.max())
```
The function `gp_posterior` is the one from @sec-gp-computation. Maximizing
over a grid works in one dimension; @sec-acq-optimization explains what
replaces it in more. With this seed the loop finds the tall bump on its ninth
query and recommends $x = 0.72$, where $f \approx 0.81$.
@sec-minimal-implementation extends this sketch to expected improvement.
:::

## Exploration and exploitation {#sec-explore-exploit}

Every acquisition function must answer the question from the opening of the
chapter: is an input worth evaluating because the model expects a high value
there, or because the model does not know? The two answers have names.
*Exploitation* means evaluating where the posterior mean is high, to refine
what already looks good. *Exploration* means evaluating where the posterior
standard deviation is high, to learn about regions the model knows little
about. The terms come from the study of bandit problems (@sec-bandits), where
a gambler must choose between the slot machine that has paid best so far and
one that has been tried too rarely to judge.

Neither pure strategy works, and the running objective shows why.

**Pure exploitation gets stuck.** Set $\beta = 0$ in @eq-loop-ucb, so the loop
always evaluates where the posterior mean is highest. Suppose the first
evaluations land on the broad bump. The mean is then highest near the best of
them, so the next evaluation lands nearby, confirms that the region is good,
and raises the mean there further. The loop climbs the broad bump, reaches
its top at about $0.53$, and stays. Nothing in its rule ever sends it to the
right half of the domain, where the mean is lower only because nothing has
been observed there. With exact observations it can even evaluate the same
input again and again, learning nothing each time (@exr-loop-greedy).

**Pure exploration never settles.** Make $\beta$ very large, so the standard
deviation term dominates. The loop then evaluates wherever the model is most
uncertain, which on an interval means filling the largest gap between
previous evaluations. The result is close to a grid built one point at a time.
It eventually lands near the tall bump, but it gives the region no more
attention than the poor region at the right end, so the best value it finds
is limited by how fine its grid has become when the budget runs out.

Pure exploration also scales badly with dimension. Suppose one evaluation
makes the model confident within a distance of $0.2$ of it, roughly one
lengthscale. A ball of radius $0.2$ covers 40% of the unit interval, 13% of
the unit square, and 3.4% of the unit cube, but only 0.033% of the
six-dimensional unit cube. Covering six dimensions this way would take at
least 3,000 evaluations, and covering the nine dimensions of the network
tuned by @snoek2012practical at least 590,000. The true numbers are larger,
since the balls overlap and stick out of the cube. No budget allows this. In
more than a few dimensions, exploration has to be selective: it can only
afford to reduce uncertainty where a high value is still plausible.

Between these extremes lies a range of weights that do well. How wide is that
range, and how badly do the extremes fail? @fig-loop-tradeoff answers by
running the loop many times.

```{figure}
//| figure: loop-tradeoff
//| label: fig-loop-tradeoff
//| fig-cap: "The exploration weight $\beta^{1/2}$ of @eq-loop-ucb, set by the slider. Top: one run of twelve evaluations (two random initial points, hollow, then ten chosen by the rule) and the posterior at the end; the best value found is ringed. Lower left: the best value found after each evaluation in that run. Lower right: the same experiment repeated from 32 random initial designs for each weight from 0 to 8, showing the average gap between the true maximum and the best value found; the orange marker is the current weight. The figure opens on the greedy rule, $\beta = 0$. The objective, kernel, and budget are illustrative; the shape of the curve, not its numbers, is the point."
```

Four experiments with the figure make the trade-off concrete.

**Start greedy.** At weight 0 the run in the top panel climbs the broad bump
and stops at $0.53$. Over the 32 starts, only 12 come within $0.05$ of the
maximum; the others never leave the bump they started on, and the average gap
is $0.18$.

**Add a little optimism.** At weights between 1 and 2, all 32 starts come
within $0.05$ of the maximum and the average gap is at most $0.003$. Even
weight $0.5$ helps a lot: 24 of 32 starts succeed.

**Overdo it.** At weight 8 the top panel shows evaluations spread almost
evenly across the domain, and the run's best value is $0.62$. Over the 32
starts the average gap rises to about $0.07$. The explorer does find the
tall bump's neighborhood, but with its budget spent everywhere, it rarely has
an evaluation close to the peak itself.

**Change the start.** Press *Another start* a few times. For some initial
designs even the greedy rule succeeds, because one of the two random points
happens to land on the tall bump's slope. Luck in the initial design can make
any rule look good on a single run, which is why comparisons of optimizers
average over many runs (@sec-sw-evaluation).

The lower right panel has the shape that recurs throughout the subject: a
valley between two failure modes. On this problem the valley is wide, so the
weight need not be tuned finely, but the extremes cost a great deal. The
location of the valley depends on the budget. Exploration pays only if there
are evaluations left to exploit what it finds, so a short budget favors a
smaller weight and a long one tolerates a larger weight (inference).

::: {.keyidea title="An acquisition function prices uncertainty"}
Every acquisition function trades expected value against uncertainty. UCB
does it with an explicit exchange rate $\beta^{1/2}$; the rules of
@sec-acquisition derive the rate from a model of what an evaluation is for.
:::

The weighted sum in @eq-loop-ucb already does something a fixed schedule
cannot. A schedule such as "explore at random for the first half of the
budget, then exploit" spends its exploration everywhere, including regions
the model already knows to be poor. The upper confidence bound explores only
where uncertainty and a plausible high value coincide: an input whose mean
plus two standard deviations is still below the best value seen is never
chosen, however uncertain it is. Not all uncertainty is worth reducing, only
the uncertainty that could change which input we end up recommending. That
observation is the starting point of the more principled acquisition
functions in @sec-acquisition, which ask directly how much an evaluation is
expected to improve the outcome.

## Starting the loop {#sec-initial-design}

The loop needs data before its model can say anything useful, and there are
two reasons not to let the acquisition function choose the very first points
[@garnett2023bayesian, sec. 9.3].

The first reason is that the acquisition function has nothing to go on. Before
any data, a Gaussian process with a constant prior mean and a stationary
kernel, one whose covariance depends only on the distance between inputs,
gives the same mean and the same standard deviation at every input. Every
acquisition function built from them is then constant, and its maximum is
anywhere (@exr-loop-flat).

The second reason is the model's hyperparameters. The lengthscale, the signal
amplitude, and the noise level are usually fitted to the data
(@sec-fitting-hyperparameters), and two or three points cannot pin them down.
A badly fitted lengthscale makes the model either overconfident between
points or uninformative everywhere, and the early decisions made with it can
send the search in the wrong direction. A handful of points chosen without the
model gives the fit something to work with.

So the loop begins with an *initial design* of $n_0$ points chosen by a rule
that ignores the objective. The goal is coverage: the points should spread over
the domain so that no large region is left unexamined. There are four common
ways to place them.

**A grid** takes $k$ values of each input and evaluates every combination, so
$k^d$ points in $d$ dimensions. Grids are easy to describe and wasteful in a
specific way: they use only $k$ distinct values of each input. When the
objective turns out to depend mainly on one input, which is common, the grid
has spent $k^d$ evaluations to learn about $k$ values of that input. The
count itself grows quickly: in six dimensions, a grid with only three values
per input needs $3^6 = 729$ evaluations, and in the nine dimensions of the
network tuned by @snoek2012practical it needs $3^9 = 19{,}683$.
@bergstra2012random found that for most of the data sets they studied, only a
few of a learning algorithm's hyperparameters really mattered, and that
different ones mattered on different data sets, which makes grids a poor
default.

**Uniform random sampling** gives every point a distinct value of every input,
and needs no planning. Its weakness is clumping. Some points land close
together and leave gaps elsewhere. Cut one input's range into $n$ equal bins,
and $n$ random points leave on average a fraction $(1 - 1/n)^n$ of the bins
empty, about 37% for large $n$.

**A Latin hypercube** fixes the clumping along every axis at once
[@mckay1979comparison]. Cut each input's range into $n$ equal bins. Place the
$n$ points so that every bin of every input contains exactly one point: for
each input independently, randomly permute the bins and assign the $i$-th
point to the $i$-th bin of the permutation, at a random position inside it. The
name comes from the Latin square, a grid in which each symbol appears once in
every row and column. A Latin hypercube guarantees coverage of each input
separately, but not of the space as a whole: the points could all sit on the
diagonal and still satisfy the definition. One remedy is to draw many Latin
hypercubes and keep the one whose two closest points are farthest apart.

**A Sobol sequence** is deterministic [@sobol1967distribution]. It is built so
that each new point falls into the largest gaps left by the earlier ones, a
property called *low discrepancy*: every box in the domain contains close to
its fair share of points. In the two-dimensional version in the figure below,
the first $2^k$ points of the sequence put exactly one point in each of the
$2^k$ bins of each input, like a Latin hypercube. Unlike a Latin hypercube, a
sequence can be extended. Adding points to a Latin hypercube breaks its
one-point-per-bin property, but adding the next points of a Sobol sequence
restores it at the next power of two.

```{figure}
//| figure: loop-designs
//| label: fig-loop-designs
//| fig-cap: "Initial designs in the unit square. The strips along the bottom and the left show each point projected onto one input, with the input's range cut into as many equal bins as there are points; shaded bins received no point. The orange segment joins the two closest points. Switch between designs, change the number of points, and press *New draw* for another random instance."
```

The projection strips in @fig-loop-designs carry the main lesson.

**Grid.** At 16 points, the grid is a $4 \times 4$ array, and 12 of the 16 bins
of each input are empty. If only the first input mattered, these 16
evaluations would amount to 4.

**Random.** The 16 random points typically leave five or six bins of each
input empty, close to the 37% predicted above, and the closest pair is often much closer than any pair
in the other designs. Press *New draw* to see how much the picture varies.

**Latin hypercube.** No bin is ever empty, by construction. The closest pair
is usually farther apart than for random points, but not always: press *New
draw* until two points nearly touch.

**Sobol.** With 16 or 32 points no bin is empty. Move the slider to 20 and
four bins of the second input empty out; of the counts between 16 and 32, only
24 fills every bin. This is why Sobol designs are usually drawn in powers of
two.

How many initial points to use is a trade-off of its own, since every point in
the design is a point not chosen by the model. A common rule in the design of
computer experiments, used by @jones1998efficient, is ten points per input
dimension; @loeppky2009choosing gave reasons and evidence for it. Their
criterion is how accurately the Gaussian process predicts the function
everywhere, which is more than optimization needs, since an optimizer only
has to be right near the top. With a budget of 30 evaluations in five
dimensions, the rule would ask for 50, so small budgets need smaller designs,
leaving the rest of the exploration to the acquisition function (inference).
The figures in this chapter, in one dimension, start from two random points.

## A short history {#sec-bo-history}

Bayesian optimization is older than its name suggests. Its ideas come from
statistics, operations research, and engineering design, and several of them
were invented more than once.

**A model and a decision rule (1960s).** Statisticians had studied how to
design experiments sequentially, each one chosen in light of the last, since
the 1940s [@garnett2023bayesian, sec. 12.2]. @kushner1964new applied the idea
to finding the maximum of a one-dimensional function observed with noise. He
modeled the function with a Wiener process, a Gaussian process whose sample
paths look like the trace of a random walk, continuous everywhere and smooth
nowhere. Early work favored such processes because their updates were cheap
enough for the computers of the time [@garnett2023bayesian, sec. 12.3].
Kushner set aside the optimal sequential policy as impractical to compute and
proposed simpler rules instead, including maximizing the probability of
improving on the best value so far (@sec-pi). His papers also discussed how a
human expert could adjust that rule's improvement threshold during the search
[@garnett2023bayesian, sec. 12.3], an early person in the loop, a theme that
returns in @sec-part-preferences.

**One-step lookahead (1970s).** A line of work in the Soviet Union developed
acquisition functions that look exactly one evaluation ahead. Expected
improvement (@sec-ei) is usually credited to Močkus and his colleagues
[@mockus1975bayesian; @jones1998efficient; @frazier2018tutorial; @brochu2010tutorial]. Garnett's history traces an
explicit formula for it to Šaltenis in 1971, and reads Močkus's one-step
criterion, which values an evaluation by how much it raises the best expected
value anywhere, as what is now called the knowledge gradient
(@sec-knowledge-gradient); for the Wiener process the two criteria coincide
[@garnett2023bayesian, sec. 12.3]. Either way, both one-step criteria were in
print by the early 1970s.

**Kriging and computer experiments (1950s to 1990s).** Independently, the
estimation of ore grades in mines had produced Gaussian process regression
under the name *kriging* [@krige1951statistical; @matheron1963principles].
@sacks1989design brought it to the design and analysis of computer
experiments, where an expensive simulation stands in for a physical
experiment. @jones1998efficient combined such a model with expected
improvement in its closed form, together with a careful treatment of model
validation and a branch-and-bound method for maximizing the acquisition
function, and called the result Efficient Global Optimization, or EGO. EGO
brought the method to wide attention, first in engineering design
[@frazier2018tutorial].

**Guarantees from bandits (2010).** The upper confidence bound of @eq-loop-ucb
was proposed by Kushner and rediscovered several times
[@garnett2023bayesian, sec. 12.5]. @srinivas2010gaussian connected it to the
multi-armed bandit literature and proved how fast it converges for Gaussian
process models, the analysis that @sec-gp-bandits explains.

**Machine learning (2012 onward).** @snoek2012practical showed that with
careful choices of the prior and of how its hyperparameters are handled,
Bayesian optimization could tune machine learning algorithms, including
convolutional neural networks, as well as or better than human experts. The
paper set off a surge of interest in machine learning: more than half of the
works cited in Garnett's 2023 textbook appeared after 2012
[@garnett2023bayesian, sec. 12.4], and software frameworks such as BoTorch
followed [@balandat2020botorch]. In the same years, a separate line of work
developed information-theoretic acquisition functions, first proposed by
Villemonteix and colleagues and named *entropy search* by
@hennig2012entropy [@garnett2023bayesian, sec. 12.4], then refined by
@hernandezlobato2014predictive and @wang2017maxvalue (@sec-entropy-search).

**Preferences (2007 onward).** @brochu2007active used the same machinery with
a person choosing between options instead of reporting numbers, for designing
the appearance of rendered materials. That line of work, PBO, is the subject
of @sec-part-preferences.

::: {.table #tbl-loop-history title="Milestones in the development of Bayesian optimization."}
| Year | Work | Contribution |
|---|---|---|
| 1933 | @thompson1933likelihood | Allocate treatments by the posterior probability that each is better; the origin of Thompson sampling (@sec-thompson) |
| 1964 | @kushner1964new | One-dimensional optimization with a Wiener process model; probability of improvement |
| 1975 | @mockus1975bayesian | One-step lookahead in the Bayesian approach; usually credited with expected improvement |
| 1989 | @sacks1989design | Gaussian process models for expensive computer simulations |
| 1998 | @jones1998efficient | EGO: expected improvement in closed form with a fitted Gaussian process |
| 2009 | @frazier2009knowledge | Knowledge gradient for correlated beliefs |
| 2010 | @srinivas2010gaussian | GP-UCB and its regret bounds |
| 2012 | @hennig2012entropy | Entropy search: choose evaluations by information about the maximizer |
| 2012 | @snoek2012practical | Hyperparameter tuning of machine learning algorithms |
| 2020 | @balandat2020botorch | BoTorch: Monte Carlo acquisition functions with automatic differentiation |
:::

## Exercises {#sec-loop-exercises}

::: {.exercise #exr-loop-flat}
Consider a Gaussian process prior with constant mean $m$ and a stationary
kernel, $k(\vx, \vx') = \kappa(\vx - \vx')$ for some function $\kappa$. Show
that before any data, $\mu_0(\vx)$ and $\sigma_0(\vx)$ do not depend on $\vx$.
What does this imply for any acquisition function that depends on $\vx$ only
through $\mu_0(\vx)$ and $\sigma_0(\vx)$?

::: {.solution}
With no data, the posterior is the prior, so $\mu_0(\vx) = m$ and
$\sigma_0^2(\vx) = k(\vx, \vx) = \kappa(\mathbf{0})$, the same at every input.
An acquisition function of the form $a_0(\vx) = h(\mu_0(\vx), \sigma_0(\vx))$
is then a constant, and every input maximizes it. The first query is
arbitrary, which is one reason to choose the first points with a design rule.
:::
:::

::: {.exercise #exr-loop-greedy}
With exact observations, suppose the greedy rule $a_n(\vx) = \mu_n(\vx)$
selects an input $\vx_i$ that has already been evaluated. Show that evaluating
it again leaves the posterior unchanged, so the loop will select the same
input forever.

::: {.solution}
With exact observations the posterior variance at an evaluated input is zero
and its posterior mean equals the observed value: $\sigma_n(\vx_i) = 0$ and
$\mu_n(\vx_i) = y_i$ (@sec-reading-posterior). A new evaluation at $\vx_i$
returns $y_i$ again, a value the model already predicted with certainty.
Conditioning on an event that had probability one does not change a
distribution, so $\mu_{n+1}(\vx) = \mu_n(\vx)$ and
$\sigma_{n+1}(\vx) = \sigma_n(\vx)$ at every $\vx$.
The greedy rule therefore chooses $\vx_i$ again, and so on until the budget is
spent. (In floating point the repeated input makes the kernel matrix
singular; the small jitter of @sec-gp-computation keeps the factorization
working and changes the posterior only negligibly.)
:::
:::

::: {.exercise #exr-loop-random}
Random search draws $n$ inputs uniformly from the domain. Let $p$ be the
fraction of the domain's volume where $f$ is within some tolerance of its
maximum. Show that the probability that at least one of the $n$ inputs lands
in that region is $1 - (1 - p)^n$, and find the smallest $n$ that makes this
probability at least $0.95$ when $p = 0.05$. Why does this number not depend
on the dimension $d$, and why is that less reassuring than it sounds?

::: {.solution}
Each input misses the region independently with probability $1 - p$, so all
$n$ miss with probability $(1 - p)^n$ and at least one hits with probability
$1 - (1 - p)^n$. Setting $1 - 0.95^n \ge 0.95$ gives
$n \ge \log 0.05 / \log 0.95 \approx 58.4$, so $n = 59$. The calculation uses
only the volume fraction $p$, not $d$. But in $d$ dimensions a region that
spans a fraction $q$ of each input's range has volume fraction $p = q^d$. With
$q = 0.5$ and $d = 10$, $p \approx 0.001$, and the required $n$ grows to about
3,000. Random search is a strong baseline when only a few inputs matter
[@bergstra2012random], because then $p$ is set by those few inputs alone.
:::
:::

## Further reading {#further-reading .unnumbered}

- @frazier2018tutorial is a short tutorial that covers the loop, expected
  improvement, the knowledge gradient, and entropy search, and surveys the
  problem variants of @sec-bo-practice.
- @garnett2023bayesian is the textbook of the field. Chapters 5 to 7 derive
  the loop from Bayesian decision theory, its chapter 9 covers initial designs
  and stopping, and its chapter 12 is the history summarized here.
- @shahriari2016taking is a broad review with a large bibliography of
  applications, written as Bayesian optimization was spreading through
  machine learning.
- @brochu2010tutorial is an accessible tutorial that also introduces
  preference learning, the bridge to @sec-part-preferences.
- @jones1998efficient remains readable, and shows the method as engineers met
  it: a fitted Gaussian process, expected improvement, and diagnostics for the
  model.
- @snoek2012practical is the paper that brought the method to machine
  learning, with practical advice on priors and hyperparameters that still
  applies.
