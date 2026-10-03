---
status: done
synopsis: "Entropy, KL divergence, and mutual information, built from the surprise of a single outcome; Lindley's expected information gain for choosing what to ask next; and the information gain of a Gaussian process, whose maximum sets every regret bound in the book and grows quickly with the input dimension."
sources: ["textbooks", "Cover and Thomas 2006, ch. 2, 8, 12", "MacKay 2003, ch. 2", "Lindley 1956", "Srinivas et al. 2010"]
---

# Measuring Information {#sec-information}

@sec-bayes-point-limits left a question open. A Bayesian model keeps its
uncertainty so that an optimizer can spend each evaluation where it teaches
the most. But how much does an evaluation teach? To compare two candidate
queries, or to say how much a person's answer to a comparison can possibly
reveal, we need to measure what is learned in a unit, the way bytes measure
storage.

Claude Shannon supplied that unit in 1948, for a different problem: how many
binary digits it takes to transmit a message [@shannon1948mathematical]. His
measure, entropy, turns out to quantify uncertainty in general, and two
quantities built from it, the Kullback-Leibler divergence and mutual
information, measure how far apart two beliefs are and how much one variable
tells about another. This chapter builds the three from a single idea, the
surprise of one outcome, and then uses them for the two jobs the rest of the
book needs.

The first job is choosing what to ask. @lindley1956measure proposed to choose
an experiment by the information its outcome is expected to provide, and
several acquisition functions in @sec-acquisition and most query rules for
comparisons in @sec-query-design are versions of his criterion. The second job
is analysis. An optimizer's *regret* is the shortfall of the values it
obtained from the best value available, added up over its evaluations, and the
theory in @sec-regret bounds how fast it can grow. Those bounds measure how
hard a problem is by the most information that $T$ evaluations could gather
about the objective, a number written $\gamma_T$. The chapter ends by
computing it and watching how fast it grows with the dimension of the input.

A note on units. Information is measured with logarithms, and the base of the
logarithm sets the unit. Base 2 gives **bits**, natural for yes/no questions;
base $e$ gives **nats**, natural for Gaussians. We use bits for discrete
examples and nats for continuous ones, as the literature does. One nat is
$1/\ln 2 \approx 1.443$ bits, so converting is a single multiplication.

## Surprise and entropy {#sec-entropy}

### Surprise {#sec-info-surprise}

Start with one outcome. A fair coin landing heads is mildly surprising; a
lottery ticket winning is very surprising; the sun rising is not surprising at
all. A measure of surprise should depend only on the probability $p$ of what
happened, should be zero when $p = 1$, and should grow as $p$ shrinks. One more
requirement fixes it. Two independent events, such as a coin landing heads
and a die showing six, have probability $\tfrac12 \times \tfrac16$, and we would
like the surprise of seeing both to be the sum of the two surprises. The
function that turns products into sums is the logarithm, so the **surprise**,
or information content, of an outcome with probability $p$ is

$$
-\log p.
$$ {#eq-info-surprise}

In bits, an outcome with probability $1/2$ carries 1 bit, one with
probability $1/8$ carries 3 bits, and one with probability $1/1024$ carries 10
bits. The surprise is the number of fair coin flips that would have to come out
a particular way to be as unlikely as the outcome.

### Entropy {#sec-info-entropy-def}

Surprise belongs to an outcome. Before the outcome is known, we can ask how
surprised we expect to be. That expectation is the **entropy** of the
distribution:

$$
H(X) = \E\left[-\log p(X)\right] = -\sum_x p(x) \log p(x),
$$ {#eq-info-entropy}

with the convention $0 \log 0 = 0$, since an outcome that never happens
contributes nothing. Entropy is a property of a distribution, not of a value,
and it measures how uncertain the distribution is.

A few cases make the scale concrete. A fair coin has $H = 1$ bit. A fair die
has $\log_2 6 \approx 2.585$ bits. A coin that lands heads with probability
$0.9$ has

$$
h(0.9) = -0.9 \log_2 0.9 - 0.1 \log_2 0.1 \approx 0.469 \text{ bits},
$$

where $h(p) = -p\log_2 p - (1 - p)\log_2(1 - p)$ is the **binary entropy**
function. It equals 1 bit at $p = 1/2$ and falls to 0 as $p$ approaches 0 or 1.
A yes/no question whose answer is nearly certain has almost no entropy, and,
as @sec-mutual-information will show, can teach almost nothing.

Entropy has an operational meaning that makes it more than a formula. Suppose
someone draws an outcome from $p$ and you must identify it by asking yes/no
questions, any questions you like. The smallest possible average number of
questions lies between $H(X)$ and $H(X) + 1$ when entropy is measured in bits,
because a strategy of questions is a binary code for the outcomes and the best
binary codes achieve this length [@cover2006elements, ch. 5]. To identify one
of 128 equally likely outcomes takes exactly 7 questions, each halving the
remaining set, and $\log_2 128 = 7$. A distribution that piles its probability
onto a few outcomes takes fewer questions on average, because the likely
outcomes can be asked about first.

Among distributions over $K$ outcomes, the uniform one has the largest
entropy, $\log K$, and a distribution that puts all its probability on one
outcome has the smallest, zero. @exr-info-uniform proves the first claim with
a tool from the next section.

### Entropy of a continuous variable {#sec-info-differential}

For a continuous variable with density $p$, the sum becomes an integral,

$$
H(X) = -\int p(x) \log p(x)\, \dd x,
$$ {#eq-info-differential}

called the **differential entropy**. It keeps the meaning "how spread out",
but it is not a limit of the discrete entropy, and two of its properties are
surprising at first. It can be negative: a density squeezed into an interval
of width $0.1$ has values around 10, so $-\log p(x)$ is negative there. And it
depends on the units: measuring the same quantity in millimeters instead of
meters adds $\log 1000$. Differences of differential entropies, which is all
the rest of this chapter uses, have neither problem.

::: {.derivation title="The entropy of a Gaussian"}
Let $X \sim \N(\mu, \sigma^2)$, with the density of @eq-gauss-density-1d.

1. Take logarithms: $-\log p(x) = \tfrac12\log(2\pi\sigma^2) + \frac{(x - \mu)^2}{2\sigma^2}$.
2. Take the expectation under $p$. The first term is a constant. The second
   has expectation $\E[(X - \mu)^2]/(2\sigma^2) = \sigma^2/(2\sigma^2) = \tfrac12$,
   by the definition of the variance.
3. So $H(X) = \tfrac12\log(2\pi\sigma^2) + \tfrac12 = \tfrac12\log(2\pi e\,\sigma^2)$,
   writing $\tfrac12 = \tfrac12\log e$.
4. For $\vx \sim \N(\vmu, \mSigma)$ in $d$ dimensions, the same steps with
   @eq-gauss-density give $\tfrac12\log\lvert 2\pi\mSigma\rvert + \tfrac12\E[(\vx - \vmu)^\T\mSigma^{-1}(\vx - \vmu)]$.
   The expected squared Mahalanobis distance is $d$: in the rotated and
   rescaled coordinates of @sec-gauss-shape it is a sum of $d$ squared
   standard normal numbers, each with mean 1. So
   $H(\vx) = \tfrac12\log\lvert 2\pi\mSigma\rvert + \tfrac{d}{2} = \tfrac12\log\lvert 2\pi e\,\mSigma\rvert$.
:::

$$
H\big(\N(\mu, \sigma^2)\big) = \tfrac12\log(2\pi e\,\sigma^2),
\qquad
H\big(\N(\vmu, \mSigma)\big) = \tfrac12\log\det(2\pi e\,\mSigma).
$$ {#eq-info-gaussian-entropy}

The entropy of a Gaussian does not depend on its mean, only on its spread: it
grows like the log of the standard deviation, and in many dimensions like the
log of the volume $\lvert\mSigma\rvert^{1/2}$ of its ellipsoids (@sec-determinants).
A standard normal has $\tfrac12\log(2\pi e) \approx 1.419$ nats; the entropy
drops below zero once $\sigma < 1/\sqrt{2\pi e} \approx 0.242$.

@sec-gauss-why claimed that the Gaussian assumes the least of any distribution
with a given mean and variance. In the language of this section: among all
densities on the real line with variance $\sigma^2$, the Gaussian has the
largest differential entropy, $\tfrac12\log(2\pi e\,\sigma^2)$
[@cover2006elements, ch. 12]. The proof takes two lines once the next
section's tool is in hand, and @sec-info-maxent gives it.

## KL divergence {#sec-kl}

Entropy measures one distribution. Much of inference compares two: the
posterior and an approximation to it, the true distribution of answers and a
model's prediction, a fair coin and a biased one. The comparison the rest of
the book uses is the Kullback-Leibler divergence.

### Definition {#sec-info-kl-def}

Suppose the data come from a distribution $p$, and we score them with a model
$q$. Each outcome $x$ costs us a surprise of $-\log q(x)$ under the model,
where the best possible model, $p$ itself, would have charged $-\log p(x)$.
The **Kullback-Leibler divergence** is the average excess:

$$
\KL(p \,\|\, q) = \E_{x \sim p}\left[\log\frac{p(x)}{q(x)}\right]
= \sum_x p(x)\log\frac{p(x)}{q(x)},
$$ {#eq-info-kl}

with an integral for densities. It is also called relative entropy.

The same quantity has a second reading that explains its role in statistics.
The ratio $\log(p(x)/q(x))$ is the log-likelihood ratio of one observation, the
evidence it provides for "the data come from $p$" against "the data come from
$q$". Averaged over observations that really do come from $p$, it is the
expected evidence per observation for the truth. This is how
@kullback1951information introduced it, as the mean information per
observation for discriminating between two hypotheses, and it is how the
lower bounds of @sec-regret-lower-bounds use it. In a bandit problem, where
one chooses repeatedly among a few options with unknown reward distributions,
the number of tries needed to tell a worse option from the best over $T$
rounds grows like $\ln T$ divided by the divergence between their reward
distributions.

The divergence behaves like a distance in one important way and fails to in
another.

::: {.derivation title="The KL divergence is never negative (Gibbs' inequality)"}
1. Write $-\KL(p \,\|\, q) = \sum_x p(x)\log\frac{q(x)}{p(x)}$, summing over
   the outcomes with $p(x) > 0$.
2. The logarithm is concave, so by Jensen's inequality the average of the log
   is at most the log of the average:
   $\sum_x p(x)\log\frac{q(x)}{p(x)} \le \log\sum_x p(x)\frac{q(x)}{p(x)}$.
3. The right side simplifies to $\log\sum_{x : p(x) > 0} q(x)$.
4. That sum of probabilities is at most 1, so its log is at most 0, and
   $\KL(p \,\|\, q) \ge 0$.
5. Equality in step 2 requires $q(x)/p(x)$ to be the same for every $x$, and
   equality in step 4 requires $q$ to put all its probability where $p$ does.
   Together, $\KL(p \,\|\, q) = 0$ exactly when $q = p$.
:::

So the divergence is zero for identical distributions and positive otherwise,
like a distance. It is not symmetric, though: $\KL(p \,\|\, q)$ and
$\KL(q \,\|\, p)$ are different numbers in general, and the difference can be
large. It is a measure of how badly $q$ stands in for $p$, and that is not the
same as how badly $p$ stands in for $q$.

Readers who have trained a classifier have minimized a KL divergence without
the name. The quantity such training minimizes, called **cross-entropy**,
$-\sum_x p(x)\log q(x)$, splits as

$$
-\sum_x p(x)\log q(x) = H(p) + \KL(p \,\|\, q).
$$ {#eq-info-cross-entropy}

The entropy of the data does not depend on the model, so minimizing the
cross-entropy over $q$ minimizes the divergence from the data to the model.

### Two coins and two Gaussians {#sec-info-kl-examples}

Two closed forms appear later in the book. For coins with heads
probabilities $p$ and $q$, the divergence, written $\mathrm{kl}(p, q)$ in the
bandit literature, is

$$
\mathrm{kl}(p, q) = p\ln\frac{p}{q} + (1 - p)\ln\frac{1 - p}{1 - q}.
$$ {#eq-info-kl-bernoulli}

A coin with $p = 0.6$ judged against a fair coin has
$\mathrm{kl}(0.6, 0.5) \approx 0.020$ nats: each flip carries a fiftieth of a nat
of evidence, which is why telling a 0.6 coin from a fair one takes on the order
of a hundred flips, as @sec-prob-coin found.

For two one-dimensional Gaussians,

$$
\KL\big(\N(\mu_1, \sigma_1^2) \,\|\, \N(\mu_2, \sigma_2^2)\big)
= \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac12,
$$ {#eq-info-kl-gauss}

which @exr-info-kl-gauss derives. The asymmetry is easy to see in it. With
equal means, a narrow $p = \N(0, 1)$ scored by a wide $q = \N(0, 2^2)$ costs
$\log 2 + \tfrac18 - \tfrac12 \approx 0.318$ nats, while the wide one scored by
the narrow one costs $-\log 2 + 2 - \tfrac12 \approx 0.807$ nats. A model that is
too confident is punished more than one that is too vague, because it assigns
tiny probability to outcomes that do happen.

::: {.aside title="The same divergence in many dimensions"}
For two Gaussians in $d$ dimensions the same computation gives

$$
\KL\big(\N(\vmu_1, \mSigma_1) \,\|\, \N(\vmu_2, \mSigma_2)\big)
= \tfrac12\left[\tr(\mSigma_2^{-1}\mSigma_1) + (\vmu_2 - \vmu_1)^\T\mSigma_2^{-1}(\vmu_2 - \vmu_1) - d + \log\frac{\lvert\mSigma_2\rvert}{\lvert\mSigma_1\rvert}\right].
$$

Here $\tr$, the trace, is the sum of a matrix's diagonal entries. With
$d = 1$ the formula is @eq-info-kl-gauss.
:::

### Which direction {#sec-info-kl-directions}

The asymmetry matters most when we approximate a complicated distribution $p$
by a simple one $q$, such as a Gaussian, by minimizing a divergence. The two
directions ask for different things.

- **$\KL(p \,\|\, q)$** averages over $p$. Wherever $p$ has probability and $q$
  has almost none, the ratio $p/q$ explodes, so the minimizer spreads $q$ to
  cover everything $p$ covers. For a Gaussian $q$, the minimizer matches the
  mean and the covariance of $p$ (@exr-info-moment-matching). This is called
  mass-covering.
- **$\KL(q \,\|\, p)$** averages over $q$. Wherever $q$ puts probability and
  $p$ has almost none, the ratio $q/p$ explodes, so the minimizer keeps $q$
  inside the regions where $p$ is large, even if that means ignoring some of
  them. This is called mode-seeking.

```{figure}
//| figure: info-kl
//| label: fig-info-kl
//| fig-cap: "The two directions of the KL divergence. The distribution $p$ (magenta) has two equal bumps; $q$ (blue) is a single Gaussian whose mean and standard deviation you set, or fit with the buttons. The lower panel draws the integrand of the chosen divergence, so its shaded area is the divergence. The readout gives both divergences and both entropies in nats. The shape of $p$ is illustrative."
```

**Fit by $\KL(p \,\|\, q)$.** The Gaussian centers between the bumps and
stretches to cover both, with the mean and variance of $p$. It puts its peak
where $p$ has almost no probability, but it never misses an outcome that $p$
produces.

**Fit by $\KL(q \,\|\, p)$.** The Gaussian locks onto one bump and ignores the
other. Which one depends on where $q$ starts: drag its mean to the other side
and fit again. The reverse divergence has one local minimum per bump, and an
optimizer finds whichever is nearest.

**Compare the numbers after each fit.** The mode-seeking fit has a reverse
divergence of about $\log 2 \approx 0.69$ nats, the price of ignoring half of
$p$, and a forward divergence above 10 nats, because $p$ produces many values
that this $q$ calls nearly impossible.

**Push the bumps together.** Below a separation of about 3.3, both directions
prefer a single wide Gaussian, and the difference between them almost
vanishes. The choice of direction matters most when the distribution being
approximated is lopsided or has several peaks [@bishop2006pattern, sec. 10.1.2].

These two behaviors return in @sec-approx-inference. Variational inference
(@sec-vi) fits an approximation by minimizing $\KL(q \,\|\, p)$ and inherits
its tendency to be overconfident; expectation propagation (@sec-ep) matches
moments, in the spirit of $\KL(p \,\|\, q)$, and tends to cover more. The
divergence also appears as a penalty: fine-tuning a language model from human
preferences adds the divergence between the tuned model's distribution and
the original one to the objective, so that the model cannot drift arbitrarily
far to please a learned reward [@ouyang2022training], as @sec-llm-rlhf
explains.

### The Gaussian has the most entropy {#sec-info-maxent}

The nonnegativity of the divergence settles the claim deferred from
@sec-info-differential.

::: {.derivation title="Among densities with a given variance, the Gaussian has the largest entropy"}
Let $p$ be any density with mean $\mu$ and variance $\sigma^2$, and let
$\varphi$ be the density of $\N(\mu, \sigma^2)$.

1. By Gibbs' inequality, $0 \le \KL(p \,\|\, \varphi) = -H(p) - \int p(x)\log\varphi(x)\,\dd x$.
2. The log of the Gaussian density is
   $\log\varphi(x) = -\tfrac12\log(2\pi\sigma^2) - (x - \mu)^2/(2\sigma^2)$.
3. Its expectation under $p$ uses only the variance of $p$, which is
   $\sigma^2$: $-\int p\log\varphi = \tfrac12\log(2\pi\sigma^2) + \tfrac12 = H(\varphi)$,
   by @eq-info-gaussian-entropy.
4. So $0 \le H(\varphi) - H(p)$, that is, $H(p) \le H(\varphi)$, with equality
   only when $p = \varphi$.
:::

The same argument with a uniform $\varphi$ over $K$ outcomes shows that no
distribution over $K$ outcomes has more entropy than $\log K$
(@exr-info-uniform).

## Mutual information {#sec-mutual-information}

The divergence compares two distributions over the same variable. The
question an experimenter asks is different: if I observe $Y$, how much will I
learn about $X$? Entropy answers it directly. Before the observation, the
uncertainty about $X$ is $H(X)$. After observing $Y = y$, it is the entropy of
the conditional distribution, $H(X \given Y = y)$. Averaging that over the
possible observations gives the **conditional entropy**

$$
H(X \given Y) = \sum_y p(y)\, H(X \given Y = y),
$$

and the expected reduction in uncertainty is the **mutual information**

$$
I(X; Y) = H(X) - H(X \given Y).
$$ {#eq-info-mi}

Three facts make mutual information easy to work with.

1. **It is symmetric.** By the product rule, $p(x, y) = p(y)\,p(x \given y)$,
   and taking $-\E\log$ of both sides gives the chain rule
   $H(X, Y) = H(Y) + H(X \given Y)$. Written the other way round,
   $H(X, Y) = H(X) + H(Y \given X)$. Subtracting the two shows
   $H(X) - H(X \given Y) = H(Y) - H(Y \given X)$: $Y$ tells as much about $X$
   as $X$ tells about $Y$.
2. **It is a divergence.** Substituting the definitions,
   $I(X; Y) = \KL\big(p(x, y) \,\|\, p(x)\,p(y)\big)$, how far the joint
   distribution is from the one in which the two variables are independent. By
   Gibbs' inequality, $I(X; Y) \ge 0$, with equality exactly when $X$ and $Y$
   are independent: on average, an observation never increases uncertainty.
3. **It is bounded by what the observation can hold.** Since conditional
   entropy of a discrete variable is never negative,
   $I(X; Y) = H(Y) - H(Y \given X) \le H(Y)$.

The third fact has a consequence the book returns to often. A yes/no answer
has at most one bit of entropy, so it can carry at most one bit of information
about anything: about a coin, a threshold, or a person's utility function. A
comparison between two options is such an answer, and @sec-comparison-information
shows that a typical comparison carries much less than the full bit.

A fourth fact concerns chains. If $Z$ is computed from $Y$ alone, possibly
with added randomness, so that $X \to Y \to Z$ form a chain in which $Z$
depends on $X$ only through $Y$, then

$$
I(X; Z) \le I(X; Y).
$$ {#eq-info-data-processing}

This is the **data processing inequality** [@cover2006elements, ch. 2]: no
processing of an observation, however clever, can create information about
$X$ that the observation did not contain. Recording a person's graded answer
as a forced binary choice, for example, can only lose information, which
@sec-ties uses to compare answer formats.

### Mutual information of Gaussians {#sec-info-mi-gaussian}

For jointly Gaussian variables the conditional entropies come from the
conditional variances of @sec-gaussian-conditioning, so mutual information has
a closed form. For two variables with correlation $\rho$, the conditional
variance of $X$ given $Y$ is $\sigma_X^2(1 - \rho^2)$ for every observed value
(@eq-gauss-cond-2d), so by @eq-info-gaussian-entropy

$$
I(X; Y) = \tfrac12\log(2\pi e\,\sigma_X^2) - \tfrac12\log\big(2\pi e\,\sigma_X^2(1 - \rho^2)\big)
= -\tfrac12\log(1 - \rho^2).
$$ {#eq-info-mi-rho}

Correlation 0.8 gives about 0.51 nats; correlation 0.99 gives about 1.96 nats;
perfect correlation gives infinite information, because a continuous value
would then be known exactly. The case that matters most for this book is noisy
observations of a Gaussian vector.

::: {.derivation title="What noisy observations reveal about a Gaussian vector"}
Let $\vf \sim \N(\mathbf{0}, \mK)$ be a vector of $n$ function values and
$\vy = \vf + \boldsymbol{\varepsilon}$ the observations, with independent noise
$\boldsymbol{\varepsilon} \sim \N(\mathbf{0}, \sigma_n^2\mI)$.

1. By symmetry of @eq-info-mi, $I(\vy; \vf) = H(\vy) - H(\vy \given \vf)$.
2. The sum of independent Gaussian vectors is Gaussian, and their covariances
   add (@sec-gauss-sum, applied through @eq-gauss-affine), so
   $\vy \sim \N(\mathbf{0}, \mK + \sigma_n^2\mI)$, and by
   @eq-info-gaussian-entropy, $H(\vy) = \tfrac12\log\det\big(2\pi e(\mK + \sigma_n^2\mI)\big)$.
3. Given $\vf$, only the noise is uncertain:
   $H(\vy \given \vf) = H(\boldsymbol{\varepsilon}) = \tfrac12\log\det(2\pi e\,\sigma_n^2\mI)$.
4. Subtracting, and using $\log\det\mA - \log\det\mathbf{B} = \log\det(\mathbf{B}^{-1}\mA)$,
   $I(\vy; \vf) = \tfrac12\log\det\big(\sigma_n^{-2}(\mK + \sigma_n^2\mI)\big)
   = \tfrac12\log\det\big(\mI + \sigma_n^{-2}\mK\big)$.
:::

$$
I(\vy; \vf) = \tfrac12\log\det\!\left(\mI + \sigma_n^{-2}\mK\right).
$$ {#eq-info-mi-gp}

With a single observation of a value with prior variance 1, this is
$\tfrac12\log(1 + \sigma_n^{-2})$: about 2.31 nats, or 3.3 bits, when the noise
standard deviation is 0.1. The formula contains the covariance and the noise
level but not the observed values, for the reason we met in
@sec-gaussian-conditioning: how much a Gaussian model expects to learn
depends on where it looks, not on what it finds.

## Expected information gain {#sec-expected-information-gain}

We can now say which experiment to run. Let $\theta$ be what we want to learn
and $\xi$ a choice of experiment: an input to evaluate, a question to ask, a
pair of options to show. Each choice leads to an outcome $y$ we cannot predict
exactly. @lindley1956measure proposed to measure the information an experiment
provides by the expected reduction in the entropy of $\theta$, averaged over
its possible outcomes, and to prefer the experiment for which this is
largest:

$$
\operatorname{EIG}(\xi) = H(\theta) - \E_{y \given \xi}\left[H(\theta \given y, \xi)\right] = I(\theta; y \given \xi).
$$ {#eq-info-eig}

The expected information gain is the mutual information between the unknown
and the outcome, for the experiment $\xi$. Choosing experiments this way is the
core of Bayesian experimental design, which @chaloner1995bayesian review, and
@mackay1992information brought it to the selection of training data for
neural networks.

The definition is stated in terms of $\theta$, which may have many dimensions,
and computing posterior entropies over it is expensive. Symmetry of mutual
information gives a second form in terms of the outcome, which is usually a
single number or a yes/no answer:

$$
\operatorname{EIG}(\xi) = H(y \given \xi) - \E_{\theta}\left[H(y \given \theta, \xi)\right].
$$ {#eq-info-eig-outcome}

The first term is how uncertain we are about the outcome. The second is how
uncertain we would still be if we knew $\theta$, which is the noise of the
experiment. An informative experiment is one whose outcome we cannot predict
because we do not know $\theta$, not because the measurement is noisy. Put
differently, it is the question on which the plausible values of $\theta$
disagree most. This form was popularized for classifiers and preference
learning under the name BALD, Bayesian active learning by disagreement, in a
2011 preprint [@houlsby2011bayesian].

### Twenty questions with noise {#sec-info-threshold}

The smallest problem that shows the criterion at work is locating a threshold.
An unknown $\theta$ lies somewhere in $[0, 1]$, and we may ask "is $\theta$
below $x$?" for any $x$ we like. Answers are noisy: the probability of "yes"
is $\Phi\big((x - \theta)/s\big)$, the standard normal distribution function
of @sec-gauss-standard used as an S-shaped response curve (a *probit* curve),
with a noise scale $s$, so questions far from $\theta$ are answered reliably
and questions close to it are a coin flip. The belief about $\theta$ is a grid
of 128 cells, so the uniform prior has an entropy of exactly 7 bits, and a
perfect yes/no answer could remove at most one of them.

```{figure}
//| figure: info-eig
//| label: fig-info-eig
//| fig-cap: "Locating a threshold with noisy yes/no questions. Top: the belief about $\theta$ on 128 cells, with the questions asked so far as dots (green for yes, red for no; the latest ringed). Bottom: the expected information gain of asking at each $x$, @eq-info-eig-outcome, in bits; it can never exceed the 1-bit line. Click either panel to ask at that $x$, or let the button ask at the maximum. A hidden threshold answers; reveal it to check the belief. The answer model and the noise levels are illustrative."
```

**Ask at the best $x$ a few times.** The first question goes to the middle,
where the answer is least predictable, and carries 0.90 bits in expectation
at the default noise; the shortfall from a full bit is the noise. Each later
question goes to the middle of the remaining belief. This is bisection, the
binary search a programmer would write, rediscovered by the criterion.

**Set the noise to its minimum and start over.** Now the first several
questions each carry a full bit, and eight to ten questions pin $\theta$ to a
single cell, close to the $\log_2 128 = 7$ that perfect answers would need.

**Ask at random $x$ instead.** Questions far from the remaining belief are
answered with near certainty, so they carry almost nothing; the entropy
plateaus for several questions at a time. In our runs, twenty random questions
at the default noise left about 4.1 bits of uncertainty, where twenty chosen
by the criterion left about 2.7.

**Raise the noise.** Every answer is now partly a coin flip, and the expected
gain of the best question drops well below a bit. Late in a session, the best
questions sit close to $\theta$, where answers are least reliable, and each
teaches less. Noisy answers can still be informative; there just have to be
more of them.

This is not a toy. Measuring a perceptual threshold, such as the faintest
contrast a person can detect, is the same problem, with a trial in place of a
question, and Bayesian adaptive methods have been widely used in
psychophysics since QUEST [@watson1983quest]. The method of
@kontsevich1999bayesian keeps a posterior over both the threshold and the
slope of the psychometric function (the curve that gives the probability of a
correct response at each stimulus strength) and sets each trial's stimulus to
maximize the expected information gained by that trial. In their simulations
and an experiment in which each trial is a choice between two alternatives,
the threshold was estimated to within 2 dB (23%) in fewer than 30
trials, while the slope took about 300 trials for the same precision.

### What the criterion does not say {#sec-info-eig-limits}

Two cautions apply to every use of the criterion in this book.

The criterion is **myopic**: it scores one experiment at a time, assuming no
more will follow. A sequence of individually best experiments is not always
the best sequence, although for many problems, including the threshold above,
it is close.

More importantly for optimization, **information about $\theta$ is not the
same as progress toward a goal.** An optimizer does not need to know the
objective everywhere, only where its maximum is. Spending evaluations to learn
the objective precisely in regions that are clearly poor is informative and
wasteful. Entropy search changes the unknown: instead of the whole function,
it asks for the information an evaluation provides about the location
$\vx^\star$ of the maximum [@hennig2012entropy; @hernandezlobato2014predictive],
or about the maximum value $f^\star$ [@wang2017maxvalue]. These are @eq-info-eig with a
different $\theta$, and @sec-entropy-search develops them. For comparisons,
the same move gives the information-based query rules of @sec-choosing-pairs.

## Information gain of a Gaussian process {#sec-gp-information-gain}

The last job is analysis: how much can $T$ evaluations of an unknown function
reveal about it? The answer needs the model that @sec-part-gp builds, so this
section borrows three of its objects ahead of time. A reader meeting them for
the first time can follow the figure now and return to the formulas after
@sec-gp-regression.

A *Gaussian process prior*, written $f \sim \GP(0, k)$, says that the values
of $f$ at any finite set of inputs are jointly Gaussian with mean zero, and
that the covariance between the values at two inputs $\vx$ and $\vx'$ is
$k(\vx, \vx')$, the kernel of @sec-linalg-inner-products. The *kernel matrix*
$\mK_A$ of a set $A$ of inputs is the covariance matrix of the values there.
And the *posterior variance* $\sigma_t^2(\vx)$ is the variance of $f(\vx)$
after conditioning on $t$ observations with @eq-gauss-conditional.

With these the answer is already in hand. If we evaluate $f$ with Gaussian
noise of variance $\sigma_n^2$ at a set $A$ of $T$ inputs, the observations
depend on $f$ only through its values $\vf_A$ there, and @eq-info-mi-gp gives

$$
I(\vy_A; f) = \tfrac12\log\det\!\left(\mI + \sigma_n^{-2}\mK_A\right),
$$ {#eq-info-gp-gain}

with $\mK_A$ the $T \times T$ kernel matrix of the chosen inputs. This is the
information gain that @sec-regret-info-gain uses, in the same notation.

### One evaluation at a time {#sec-info-sequential}

The determinant hides a simple sequential structure. Mutual information obeys
the same chain rule as entropy, so the information from $T$ evaluations is the
sum of what each one adds given the ones before it. The $t$th evaluation, at
$\vx_t$, has predictive variance $\sigma_{t-1}^2(\vx_t) + \sigma_n^2$ given the
first $t - 1$ observations, of which $\sigma_n^2$ would remain if $f$ were
known. As in @eq-info-mi-rho, only the ratio of the two variances matters,
and

$$
I(\vy_A; f) = \sum_{t=1}^{T} \tfrac12\log\!\left(1 + \sigma_n^{-2}\sigma_{t-1}^2(\vx_t)\right),
$$ {#eq-info-gain-chain}

where, as above, $\sigma_{t-1}^2(\vx_t)$ is the posterior variance at $\vx_t$
after the first $t - 1$ observations [@srinivas2010gaussian]. Each evaluation
contributes in proportion to the log of how uncertain the model was where it
looked. An evaluation at an input the model already knows well adds almost
nothing; one at an input where the model is as uncertain as its prior adds the
full $\tfrac12\log(1 + \sigma_n^{-2})$, the most any single evaluation can add
when the prior variance is 1.

The sum suggests a rule for gathering information quickly: always evaluate
where the posterior variance is largest. This is **uncertainty sampling**, and
@mackay1992information showed that, for interpolation models with Gaussian
noise of constant variance, maximizing the expected information about the
model's parameters means sampling where the model's error bars are largest. It
is not an optimizer, since it ignores the values it observes, but it is the
yardstick for how much there is to learn.

### The maximum information gain {#sec-info-gamma}

The most that any $T$ evaluations could teach about $f$ is

$$
\gamma_T = \max_{A \subset \X,\; |A| = T} \; \tfrac12\log\det\!\left(\mI + \sigma_n^{-2}\mK_A\right),
$$ {#eq-info-gamma}

the **maximum information gain**, which @def-regret-gamma states formally.
Like the information of any fixed design, it depends on the kernel, the
domain, and the noise, never on observed values, so it is a property of the
problem before any data are collected. Computing the maximum exactly means
searching over all sets of $T$ inputs, which is intractable. Uncertainty
sampling comes within a factor $1 - 1/e \approx 0.63$ of it, because the
information gain has diminishing returns, a property called submodularity
[@srinivas2010gaussian].

::: {.aside title="How fast the maximum information gain grows"}
How fast $\gamma_T$ grows with $T$ decides how fast the regret bounds of
@sec-regret shrink, and it depends on how flexible the kernel's functions are.
Three kernels from @sec-kernel-family serve as examples: the linear kernel,
whose functions are straight lines; the RBF (squared exponential) kernel,
whose functions are very smooth; and the Matérn kernel, whose functions are
rougher, with a parameter $\nu$ that sets how smooth they are (a larger $\nu$
means more derivatives). A linear kernel in $d$ dimensions has
$\gamma_T = O(d\log T)$, because a linear function is pinned down by a few
well-placed evaluations and later ones only average away noise. The RBF
kernel has $\gamma_T = O\big((\log T)^{d+1}\big)$ [@srinivas2010gaussian], and
a Matérn kernel with $\nu > 1/2$ has
$\gamma_T = O\big(T^{d/(2\nu + d)}(\log T)^{2\nu/(2\nu + d)}\big)$
[@vakili2021information]. @tbl-regret-gamma collects the rates, and
@sec-ka-infogain derives them from how fast the kernel's eigenvalues decay. All grow more
slowly than $T$, which is what lets the bounds of @sec-regret promise that the
average regret goes to zero. And all of them, in different ways, grow with the
dimension $d$.
:::

### Dimension and the number of evaluations {#sec-info-dimension}

How $\gamma_T$ grows as $T$ becomes large is a statement about the long run. In
practice the question is what happens in the first hundred evaluations, and
here the dimension of the input dominates. The figure computes
@eq-info-gain-chain under uncertainty sampling in five dimensions at once.

```{figure}
//| figure: info-gain-dim
//| label: fig-info-gain-dim
//| fig-cap: "Information gathered by $T$ evaluations of a Gaussian process in the unit cube in 1, 2, 3, 6, and 10 dimensions, each on a fixed set of 800 candidate points. Top: the information gain of uncertainty sampling, @eq-info-gain-chain, in nats; it is within a factor $1 - 1/e$ of $\gamma_T$ on the candidates. The dashed line is what $T$ evaluations would gather if each were completely new. Bottom: the largest posterior standard deviation left among the candidates, with a dotted line at 0.5. The readout counts the evaluations until no candidate is more than half as uncertain as under the prior. The candidate sets and the 0.5 threshold are illustrative choices."
```

**Read the default.** With lengthscale 0.2, one dimension is covered after 5
evaluations, two after 21, and three after 84. In six and ten dimensions, the
information curves lie on the dashed line for all 100 evaluations: each
evaluation is nearly uncorrelated with every other, so the model learns about
each evaluated point and generalizes to almost nothing else.

**Count the regions.** These numbers track $(1/\ell)^d$, the number of cells
of side $\ell$ in the unit cube: 5, 25, and 125 for $d = 1, 2, 3$, and about
15,600 for $d = 6$. A model with lengthscale $\ell$ treats such cells as
roughly independent, and it has to visit a fair fraction of them before it can
say something about all of them (inference). The asymptotic rates of
@tbl-regret-gamma apply only after that initial phase, which grows
exponentially with the dimension (inference).

**Scale the lengthscale by $\sqrt{d}$ and raise it to 0.3.** Now the curves
for different dimensions bunch together: 4, 9, 13, 34, and 68 evaluations
cover the candidates in 1, 2, 3, 6, and 10 dimensions. The typical distance
between two random points in the unit cube grows like $\sqrt{d/6}$, so a
lengthscale proportional to $\sqrt{d}$ keeps the correlation between typical
points roughly fixed. This is the idea behind the dimension-scaled
lengthscale priors with which standard Bayesian optimization became
competitive on real high-dimensional tasks [@hvarfner2024vanilla]. It is not
free: a longer lengthscale is a stronger assumption, that the objective varies
slowly along every input, and when the assumption is wrong the model
generalizes confidently and incorrectly. @fig-hd-lengthscale shows the
distances behind the scaling, and @sec-high-dimensions reports what the
literature has found.

**Switch to Matérn 5/2.** Its functions are rougher than the default RBF
kernel's at the same lengthscale (@sec-fs-smoothness), so each evaluation
speaks for less of the cube, and the information grows faster and more
evaluations are needed to cover it.

Two cautions keep the picture honest. The figure measures information about
the function everywhere, which overstates what an optimizer needs: it only
has to rule out regions that cannot contain the maximum, and good acquisition
functions skip most of the cube. And in ten dimensions, 800 candidates are
sparse, so covering them is much easier than covering the cube. The figure
shows the direction and rough size of the effect, not a budget for any
particular problem (inference). For a real tuning problem with seven
hyperparameters worked end to end, see @sec-cs-classifier-many.

### From information to regret {#sec-info-to-regret}

The reason $\gamma_T$ appears in every bound in @sec-regret is the sequential
sum @eq-info-gain-chain. An optimizer such as GP-UCB (@sec-ucb), which
evaluates the input where the posterior mean plus a multiple of the posterior
standard deviation is largest, can only lose much at a step where the model is
uncertain about the input it chooses, and large posterior variance at the
chosen input is what makes a term of the sum large. Summing over steps, the
total regret is controlled by the total information gathered, which is at most
$\gamma_T$. The result, @thm-regret-gpucb, bounds the cumulative regret after
$T$ steps by a constant times $\sqrt{T\beta_T\gamma_T}$, where $\beta_T$ is the
square of that multiple at step $T$, which sets the width of the confidence
bound. A problem whose maximum information gain grows slowly is one where each
mistake teaches a lot, so mistakes cannot go on for long. The figure above
shows the other side: in high dimensions with a short lengthscale, $\gamma_T$
stays close to its largest possible value, $T$ times the gain of one fresh
evaluation, for a long time, and the bound says nothing useful until it bends.

## Exercises {#sec-info-exercises}

::: {.exercise #exr-info-uniform}
Show that no distribution over $K$ outcomes has entropy greater than $\log K$,
and that the uniform distribution attains it. Then compute the entropy of the
distribution $(1/2, 1/4, 1/8, 1/8)$ in bits, and find a strategy of yes/no
questions that identifies the outcome in that many questions on average.

::: {.solution}
Let $u$ be uniform over the $K$ outcomes. For any $p$,
$0 \le \KL(p \,\|\, u) = \sum_x p(x)\log\big(p(x)K\big) = \log K - H(p)$, so
$H(p) \le \log K$, with equality exactly when $p = u$. For
$(1/2, 1/4, 1/8, 1/8)$, $H = \tfrac12 \cdot 1 + \tfrac14 \cdot 2 + 2 \cdot \tfrac18 \cdot 3 = 1.75$
bits. Ask "is it the first?"; if not, "is it the second?"; if not, "is it the
third?". The questions needed are 1, 2, 3, and 3, with average
$\tfrac12 \cdot 1 + \tfrac14 \cdot 2 + \tfrac18 \cdot 3 + \tfrac18 \cdot 3 = 1.75$:
when every probability is a power of one half, the entropy is achieved exactly.
:::
:::

::: {.exercise #exr-info-kl-gauss}
Derive @eq-info-kl-gauss. Then fix $p = \N(0, 1)$ and find the $q = \N(0, s^2)$
that minimizes $\KL(p \,\|\, q)$, and the one that minimizes $\KL(q \,\|\, p)$.

::: {.solution}
With $p = \N(\mu_1, \sigma_1^2)$ and $q = \N(\mu_2, \sigma_2^2)$,
$\log p(x) - \log q(x) = \log(\sigma_2/\sigma_1) - (x - \mu_1)^2/(2\sigma_1^2) + (x - \mu_2)^2/(2\sigma_2^2)$.
Under $p$, $\E[(x - \mu_1)^2] = \sigma_1^2$ and
$\E[(x - \mu_2)^2] = \sigma_1^2 + (\mu_1 - \mu_2)^2$, which gives
$\log(\sigma_2/\sigma_1) - \tfrac12 + \big(\sigma_1^2 + (\mu_1 - \mu_2)^2\big)/(2\sigma_2^2)$.
With $\mu_1 = \mu_2 = 0$ and $\sigma_1 = 1$, $\KL(p \,\|\, q) = \log s + 1/(2s^2) - \tfrac12$,
whose derivative $1/s - 1/s^3$ vanishes at $s = 1$. In the other direction,
$\KL(q \,\|\, p) = -\log s + s^2/2 - \tfrac12$, whose derivative $-1/s + s$ also
vanishes at $s = 1$. Both are minimized by $q = p$, with value 0: when $p$ is
itself in the family, the direction does not matter. It matters when $p$ is
not, as in @fig-info-kl.
:::
:::

::: {.exercise #exr-info-moment-matching}
Show that, over all Gaussians $q = \N(m, s^2)$, the divergence
$\KL(p \,\|\, q)$ is minimized by the $m$ and $s^2$ equal to the mean and
variance of $p$, whatever the shape of $p$.

::: {.solution}
$\KL(p \,\|\, q) = -H(p) - \E_p[\log q(x)]$, and only the second term depends
on $q$. It equals $\tfrac12\log(2\pi s^2) + \E_p[(x - m)^2]/(2s^2)$. Write
$\mu$ and $v$ for the mean and variance of $p$; then
$\E_p[(x - m)^2] = v + (\mu - m)^2$, which is smallest at $m = \mu$. With
$m = \mu$, the expression is $\tfrac12\log(2\pi s^2) + v/(2s^2)$, whose
derivative in $s^2$, $1/(2s^2) - v/(2s^4)$, vanishes at $s^2 = v$. This is the
mass-covering fit in @fig-info-kl, and the reason expectation propagation is
described as moment matching.
:::
:::

::: {.exercise #exr-info-noisy-answer}
A yes/no answer is correct with probability $1 - \varepsilon$ and flipped with
probability $\varepsilon$, independently of everything else. Show that the
most information such an answer can carry about the truth is
$1 - h(\varepsilon)$ bits, where $h$ is the binary entropy. How much is that
for $\varepsilon = 0.1$, and how many such answers are needed at the least to
learn 7 bits?

::: {.solution}
Let $X$ be the true answer and $Y$ the reported one. By the symmetry of
@eq-info-mi, $I(X; Y) = H(Y) - H(Y \given X)$. Given $X$, the report is
wrong with probability $\varepsilon$ whatever $X$ is, so
$H(Y \given X) = h(\varepsilon)$. And $H(Y) \le 1$ bit, with equality when
$X$ is equally likely to be yes or no, which makes $Y$ equally likely too. So
$I(X; Y) \le 1 - h(\varepsilon)$, attained by a question whose answer is a
priori a coin flip. For $\varepsilon = 0.1$, $h(0.1) \approx 0.469$, so each
answer carries at most about 0.531 bits, and learning 7 bits takes at least
$7/0.531 \approx 13.2$, that is, 14 answers. By the data processing
inequality, the information about anything upstream of $X$, such as the
threshold of @fig-info-eig, is no larger.
:::
:::

## Further reading {#further-reading .unnumbered}

- @cover2006elements is the standard textbook. Its chapter 2 develops entropy,
  relative entropy, and mutual information with their inequalities, chapter 8
  treats differential entropy, and chapter 12 maximum entropy.
- @mackay2003information, chapter 2 onward, introduces the same ideas
  through inference and coding, with many worked examples; the book is free
  online.
- @shannon1948mathematical defined entropy as the measure of information in a
  message, and @kullback1951information defined the divergence as the
  information for discriminating between two hypotheses.
- @lindley1956measure proposed the expected information gain as the criterion
  for choosing experiments; @chaloner1995bayesian review the Bayesian
  experimental design that grew from it.
- @srinivas2010gaussian connected the information gain of a Gaussian process
  to the regret of Bayesian optimization, the link @sec-regret builds on.
