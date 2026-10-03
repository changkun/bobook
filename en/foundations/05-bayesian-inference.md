---
status: done
synopsis: "Prior, likelihood, posterior, and predictive distribution, worked out exactly for a coin and then for a line and a plane. Why a point estimate cannot say where to look next, how the evidence weighs models, and why the posterior over a line's weights is the stepping stone to a posterior over whole functions."
sources: ["textbooks", "Bishop 2006, sec. 2.1 and ch. 3", "Rasmussen and Williams 2006, sec. 2.1", "Gelman et al. 2013, ch. 2", "MacKay 2003, ch. 28"]
---

# Bayesian Inference {#sec-bayesian-inference}

@sec-bayes-rule updated a belief about a coin flip by flip, on a grid of eleven
possible biases, and ended with a warning: grids do not scale. One unknown on
eleven values needs eleven numbers, but an unknown function needs a number for
every input. The way out it named was to choose distributions whose updates
have a closed form, so that a few numbers summarize the whole belief.
@sec-gaussian then supplied the main such distribution, the Gaussian, and
showed that conditioning and multiplying Gaussians stays inside the family.

This chapter puts the two together. It treats Bayesian inference as a
procedure: write down what you believe before the data, write down how the
data arise, and let Bayes' rule produce what you should believe afterward. We
work the procedure exactly for two models. The first is a coin with a
continuous bias, where a Beta distribution plays the role the grid played
before. The second is a straight line, and then a plane, with Gaussian
uncertainty about their coefficients. Along the way we meet the questions a
Bayesian optimizer asks of its model: what single value to report, what the
next observation will be, which of two models the data favor, and what to do
when no formula is available.

The line matters more than it seems. Its prior and posterior are Gaussians
over two weights, and @sec-function-space turns the prior into a prior over
whole functions by giving the line many more weights; @sec-gp-regression then
does the same for the posterior. The Gaussian processes of the next part are
this chapter's linear model with the number of features taken to infinity.

## Prior, likelihood, posterior {#sec-prior-likelihood-posterior}

@sec-bayes-rule named the four parts of Bayes' rule for an unknown that takes
finitely many values. Most unknowns in this book are continuous: a coin's
bias anywhere in $[0, 1]$, the slope of a line, the value of a function. The
rule keeps its form, with densities in place of probabilities and an integral
in place of the sum:

$$
p(\theta \given \D) = \frac{p(\D \given \theta)\, p(\theta)}{p(\D)},
\qquad
p(\D) = \int p(\D \given \theta)\, p(\theta)\, \dd\theta.
$$ {#eq-bayes-rule}

Here $\theta$ stands for everything unknown, possibly a vector, and $\D$ for
the data. The prior $p(\theta)$ and the likelihood $p(\D \given \theta)$
together make up the **model**. The posterior $p(\theta \given \D)$ is what
the model concludes, and the evidence $p(\D)$ is the probability the model
gave to the data before seeing them.

### A model is a program that generates data {#sec-bayes-generative}

A reader who writes software may find it easiest to think of a model as a
program that simulates data. The prior says how to draw the unknown, and the
likelihood says how to draw data given the unknown. For a coin:

1. Draw a bias $\theta$ from the prior.
2. For each flip $i = 1, \dots, n$, draw heads with probability $\theta$.

For a noisy straight line, with an input $x_i$ for each observation:

1. Draw an intercept and a slope from the prior.
2. For each input $x_i$, compute the line's value there and add independent
   Gaussian noise.

Bayesian inference runs such a program backwards. Given the output, it asks
which values of the hidden draws could have produced it, and how probable
each one is. Writing the model as a simulator also gives a practical check
that the prior says what we mean: run the program with draws from the prior
and look at the fake data it produces. If the fake data look absurd, the prior
is wrong, and no amount of later computation will fix it. Such simulations
are called prior predictive checks, and @gabry2019visualization show how to
make them a routine step of building a model.

The second step of each program assumes that the observations are
independent once $\theta$ is known (@sec-independence). The likelihood of the
whole data set then factorizes,

$$
p(\D \given \theta) = \prod_{i=1}^n p(y_i \given \theta),
\qquad
\log p(\D \given \theta) = \sum_{i=1}^n \log p(y_i \given \theta),
$$ {#eq-bayes-iid}

and implementations work with the sum of logs, which does not underflow. As
@sec-prob-coin showed, independence also lets the data arrive one at a time:
the posterior after the first $n$ observations serves as the prior for
observation $n + 1$, and the order of arrival does not matter.

### Closed forms and conjugacy {#sec-bayes-conjugacy}

The integral in the evidence is where continuous Bayesian inference gets hard.
For most pairs of prior and likelihood it has no formula, and in more than a
few dimensions it cannot be computed on a grid either. The proportional form
of Bayes' rule, @eq-prob-proportional, sidesteps the integral whenever we can
recognize the shape of its right-hand side, likelihood times prior, as a known
distribution, because then the normalizing constant is known too.
@sec-gaussian-sums did this: the product of a Gaussian prior and
a Gaussian likelihood had the shape of a Gaussian, so the posterior was a
Gaussian with no integral computed.

A prior with this property has a name. A family of priors is **conjugate** to
a likelihood when every posterior is again in the family. Updating a conjugate
model means updating a few parameters, which is why the next two sections can
show exact posteriors that change as you add data.

::: {.keyidea title="The posterior is the whole answer"}
Everything a Bayesian model can tell us is computed from the posterior: a best
guess, a range of plausible values, a prediction, a decision. Summaries throw
information away, and the information they most often discard, how uncertain
the model is, is the information an optimizer needs to choose its next query.
:::

## A conjugate example {#sec-beta-binomial}

Return to the coin, now with a bias $\theta$ that can be any number in
$[0, 1]$. The question is the one Thompson posed in 1933: given the successes
and failures seen so far, what should we believe about an unknown success
probability, and how likely is it that one such probability exceeds another
[@thompson1933likelihood]? The same question is asked today of a button's
click-through rate in an online experiment (@sec-app-online) and of each
option in a multi-armed bandit, the problem of choosing repeatedly among a few
options ("arms") with unknown success rates (@sec-bandits).

### The Beta distribution {#sec-bayes-beta-dist}

We need a family of densities on $[0, 1]$ flexible enough to express a prior
and closed under the coin's update. The likelihood of $h$ heads and $t$ tails
in a particular sequence of flips is $\theta^h(1 - \theta)^t$. A prior of the
same shape will multiply with it and keep its shape. That is the **Beta
distribution**:

$$
\mathrm{Beta}(\theta;\, \alpha, \beta) = \frac{\theta^{\alpha - 1}(1 - \theta)^{\beta - 1}}{B(\alpha, \beta)},
\qquad \alpha, \beta > 0,
$$ {#eq-bayes-beta}

where $B(\alpha, \beta) = \int_0^1 \theta^{\alpha - 1}(1 - \theta)^{\beta - 1}\,\dd\theta$
is the constant that makes the area one, the Beta function. With
$\alpha = \beta = 1$ the density is flat, the uniform prior. Larger and equal
$\alpha$ and $\beta$ give a bump centered at one half, and unequal ones tilt
it. Its mean and, for $\alpha, \beta > 1$, its mode are

$$
\E[\theta] = \frac{\alpha}{\alpha + \beta},
\qquad
\operatorname{mode}[\theta] = \frac{\alpha - 1}{\alpha + \beta - 2},
$$

and its variance is $\alpha\beta / \big((\alpha + \beta)^2(\alpha + \beta + 1)\big)$,
which shrinks as $\alpha + \beta$ grows.

::: {.derivation title="The Beta prior is conjugate to coin flips"}
1. By @eq-bayes-iid, $h$ heads and $t$ tails have likelihood
   $p(\D \given \theta) = \theta^h(1 - \theta)^t$.
2. By the proportional form of Bayes' rule and @eq-bayes-beta,
   $p(\theta \given \D) \propto \theta^h(1 - \theta)^t \cdot \theta^{\alpha - 1}(1 - \theta)^{\beta - 1}$.
   The constant $B(\alpha, \beta)$ does not involve $\theta$ and drops out.
3. Adding exponents, this is $\theta^{\alpha + h - 1}(1 - \theta)^{\beta + t - 1}$.
4. This has the shape of @eq-bayes-beta with parameters $\alpha + h$ and
   $\beta + t$. A density is determined by its shape, since the constant is
   whatever makes the area one, so
   $p(\theta \given \D) = \mathrm{Beta}(\theta;\, \alpha + h,\, \beta + t)$.
5. The evidence is the ratio of the constants:
   $p(\D) = B(\alpha + h, \beta + t) / B(\alpha, \beta)$.
:::

The update is counting. Add the heads to $\alpha$ and the tails to $\beta$.

### Prior strength in units of data {#sec-bayes-pseudo-counts}

The update suggests a reading of the prior's parameters. A
$\mathrm{Beta}(\alpha, \beta)$ prior behaves as if we had already seen
$\alpha$ heads and $\beta$ tails, so we can describe it by two numbers that
mean something: its mean $m_0 = \alpha / (\alpha + \beta)$, the bias we expect,
and its strength $n_0 = \alpha + \beta$, the number of flips it is worth. After
$n = h + t$ real flips, the posterior mean is

$$
\E[\theta \given \D] = \frac{\alpha + h}{n_0 + n}
= \frac{n_0}{n_0 + n}\, m_0 \;+\; \frac{n}{n_0 + n}\, \frac{h}{n}.
$$ {#eq-bayes-shrinkage}

The posterior mean is a weighted average of the prior mean and the observed
fraction of heads, with weights in proportion to the prior's strength and the
number of flips. With few flips the prior dominates; with many, the data do.
Pulling an estimate toward a prior value in this way is called **shrinkage**,
and @eq-bayes-shrinkage says exactly how much to shrink: by the ratio of
pretend data to real data.

The figure lets you set both halves and watch the posterior respond.

```{figure}
//| figure: bayes-beta
//| label: fig-bayes-beta
//| fig-cap: "The beta-binomial model, @eq-bayes-beta and @eq-bayes-shrinkage. The prior (dashed) is set by its mean and its strength $n_0$ in flips; the dotted curve is the likelihood of the observed flips, rescaled to unit area so it can share the axis; the posterior (solid) is $\mathrm{Beta}(\alpha + h, \beta + t)$ with its central 95% credible interval shaded. Add flips with the buttons or set the counts with the sliders."
```

**Start from the uniform prior.** With the default $n_0 = 2$ and mean 0.5, the
prior is $\mathrm{Beta}(1, 1)$, flat. The posterior then has exactly the shape
of the likelihood, and the dotted and solid curves coincide: with a flat prior,
Bayes' rule only normalizes the likelihood.

**Make the prior strong.** Raise $n_0$ to 50. Seven heads and three tails now
barely move the posterior from 0.5: the prior is worth five times as many flips
as the data. Then raise the heads to 70 and the tails to 30 and watch the data
take over.

**Make the prior wrong and confident.** Set the prior mean to 0.2 with
$n_0 = 50$, and the data to 70 heads and 30 tails. The posterior settles
between the two, closer to the data. A confident prior that conflicts with
the data costs many observations to overcome, which is a reason to keep priors
honest about what is not known.

**Watch the interval shrink.** Keep the ratio of heads to tails fixed and
multiply both counts by four. The credible interval narrows by about half. The
posterior standard deviation of a fraction falls like $1/\sqrt{n}$, so four
times the data buys twice the precision.

The 95% **credible interval** shaded in the figure is the range that holds 95%
of the posterior probability. It answers the question most people think a
confidence interval answers: given these data, where does $\theta$ probably
lie?

### What the coin predicts {#sec-bayes-coin-predicts}

The posterior also answers a question about the future: what is the
probability that the next flip lands heads? Average the probability of heads,
$\theta$, over the posterior:

$$
\Prob(\text{next flip is heads} \given \D) = \int_0^1 \theta\, p(\theta \given \D)\,\dd\theta
= \E[\theta \given \D] = \frac{\alpha + h}{\alpha + \beta + n}.
$$ {#eq-bayes-coin-predictive}

With the uniform prior this is $(h + 1)/(n + 2)$, Laplace's rule of
succession. After three heads in three flips it gives $4/5$, not the certainty
that the observed fraction $3/3$ would suggest. This averaging over the
posterior is the general recipe for prediction, and @sec-predictive returns to
it.

Two uses of this model recur later. Thompson sampling, which @sec-thompson
develops for functions, began as a rule for this setting: draw a bias
from each option's Beta posterior and try the option whose draw is largest
[@thompson1933likelihood]. And the model needs nothing but a yes or a no per
observation, so it reappears whenever the data are binary answers. A 2024
system that recommends items through a dialogue, for example, keeps a Beta
posterior for each item and updates it after each of the user's answers, by an
amount that a language model assigns for how strongly the answer supports that
item [@austin2024bayesian].

## Point estimates and their limits {#sec-point-estimates}

Often a single number is wanted: the bias of the coin, the slope of the line,
the best setting of a learning rate. There are three standard ways to reduce a
posterior to one value, and it is worth knowing what each one does before
seeing what all of them lose.

The **maximum likelihood estimate** (MLE) ignores the prior and picks the
value under which the data were most probable:
$\hat\theta_{\text{MLE}} = \argmax_\theta p(\D \given \theta)$. For the coin it
is the fraction of heads, $h/n$. For a line with Gaussian noise, maximizing the
likelihood means minimizing the sum of squared residuals, so the MLE is the
least-squares fit (@exr-bayes-ridge).

The **maximum a posteriori estimate** (MAP) picks the peak of the posterior:
$\hat\theta_{\text{MAP}} = \argmax_\theta p(\D \given \theta)\,p(\theta)$. For
the coin it is the posterior mode, $(\alpha + h - 1)/(\alpha + \beta + n - 2)$.
In log form, the MAP maximizes the log likelihood plus the log prior, and the
log prior acts as a penalty. With a Gaussian prior on a line's weights the
penalty is a multiple of the squared length of the weight vector, and the MAP
is ridge regression, the least-squares fit with an $L_2$ penalty that many
software engineers have met as regularization.

The **posterior mean** $\E[\theta \given \D]$ averages over the posterior. It
is the estimate with the smallest expected squared error, and for the coin it
is @eq-bayes-shrinkage. A reader who has used add-one smoothing to avoid zero
counts has used the posterior mean under a uniform prior.

```{figure}
//| figure: bayes-beta
//| label: fig-bayes-estimates
//| fig-cap: "The three point estimates of a coin's bias after three heads and no tails, with a uniform prior. The maximum likelihood estimate sits at 1, a coin that can never land tails; the MAP coincides with it under this flat prior; the posterior mean is 0.8. The shaded 95% interval reaches down to about 0.4. Move the prior and the counts to see the estimates separate and converge."
heads: 3
tails: 0
estimates: true
```

**Read the MLE after three heads.** It says the coin always lands heads, and a
model built on it would bet anything against a tail. The posterior says
something milder: the bias is probably high, but anywhere from about 0.4 to 1
is plausible.

**Give the prior a little weight.** Set the prior strength to 4 with mean 0.5,
that is, $\mathrm{Beta}(2, 2)$. The MAP moves off the edge to 0.8 and the
posterior mean to 0.71. The prior acts as two imaginary heads and two
imaginary tails, a guard against overconfidence from little data.

**Add data and watch the estimates converge.** At 60 heads and 20 tails all
three estimates lie within 0.02 of each other. With enough data the choice of
estimate stops mattering, and so does the prior.

### What a point estimate cannot do {#sec-bayes-point-limits}

The deeper problem with any point estimate is not that it can be wrong. It is
that it does not say how wrong it might be. A coin with 1 head in 2 flips and
a coin with 500 heads in 1000 flips both have an MLE of 0.5. Anyone would trust
the second estimate more, and the posterior records why: the first is
$\mathrm{Beta}(2, 2)$ under a uniform prior, nearly as wide as the prior
itself, while the second has a standard deviation of about 0.016.

For optimization, this missing information is the whole game. Suppose two
learning rates have been tried. The first was run three times, with a mean
validation accuracy of 0.80; the second was run once and reached 0.78. A
point estimate says the first is better and the second should be abandoned.
The posterior says the second might well be better, because one noisy run
leaves its accuracy uncertain, and that one more run there would teach more
than a fourth run of the first. Choosing between exploiting what looks best
and exploring what is uncertain is the central trade-off of Bayesian
optimization (@sec-explore-exploit), and it can only be made by a model that
keeps its uncertainty. Every acquisition function in @sec-acquisition is a
rule that reads both the posterior mean and the posterior spread.

## Bayesian linear regression {#sec-blr}

The coin had one unknown number. An objective function is an unknown
relationship between inputs and outputs, and the simplest such relationship
is a straight line. We observe noisy values $y_i$ at inputs $x_i$ and model
them as

$$
y_i = w_1 + w_2 x_i + \varepsilon_i,
\qquad \varepsilon_i \sim \N(0, \sigma_n^2),
$$

an intercept $w_1$, a slope $w_2$, and independent Gaussian noise with
standard deviation $\sigma_n$. The question is what we should believe about
the line, and therefore about its value at inputs we have not tried, after a
few observations.

### The model {#sec-bayes-blr-model}

Collect the weights into $\vw = (w_1, w_2)^\T$, and collect what each weight
multiplies, here the constant 1 and the input $x$, into a vector of *features*
$\boldsymbol{\phi}(x) = (1, x)^\T$, so that the line's value is
$f(x) = \boldsymbol{\phi}(x)^\T\vw$. For $n$ observations, stack the feature
vectors as the rows of the $n \times 2$ **design matrix** $\boldsymbol{\Phi}$
and the observations into $\vy$. The model is:

$$
\vw \sim \N(\mathbf{0}, \mSigma_p),
\qquad
\vy \given \vw \sim \N\!\left(\boldsymbol{\Phi}\vw,\; \sigma_n^2\mI\right).
$$ {#eq-bayes-blr-model}

The prior covariance $\mSigma_p$ says how large we expect the weights to be;
in the figures it is $\sigma_p^2\mI$, independent weights with standard
deviation $\sigma_p$. Nothing in this section depends on the features being
$1$ and $x$. Any fixed list of features works, which is the door that
@sec-function-space walks through.

Before any data, the prior over weights is already a prior over lines. Each
draw of $\vw$ is one line, and the value at $x$ has variance
$\boldsymbol{\phi}(x)^\T\mSigma_p\boldsymbol{\phi}(x) = \sigma_p^2(1 + x^2)$, by
@eq-gauss-affine. The prior lines therefore fan out from the region around
$x = 0$, where only the intercept varies, and spread quadratically away from
it.

### The posterior {#sec-bayes-blr-posterior}

Both the prior and the likelihood are Gaussian in $\vw$, so the posterior is
too, and the recipe of @eq-gauss-recipe-nd finds it.

::: {.derivation title="The posterior over the weights"}
1. By Bayes' rule, $\log p(\vw \given \vy) = \log p(\vy \given \vw) + \log p(\vw) + \text{const}$,
   where the constant is the log evidence, which does not involve $\vw$.
2. The log likelihood of @eq-bayes-blr-model is
   $-\frac{1}{2\sigma_n^2}\lVert \vy - \boldsymbol{\Phi}\vw \rVert^2 + \text{const}$,
   and the log prior is $-\tfrac12\vw^\T\mSigma_p^{-1}\vw + \text{const}$.
3. Expand the squared norm:
   $\lVert \vy - \boldsymbol{\Phi}\vw \rVert^2 = \vy^\T\vy - 2\vy^\T\boldsymbol{\Phi}\vw + \vw^\T\boldsymbol{\Phi}^\T\boldsymbol{\Phi}\vw$.
   The first term does not involve $\vw$.
4. Collect the terms quadratic and linear in $\vw$:
   $\log p(\vw \given \vy) = -\tfrac12\vw^\T\left(\sigma_n^{-2}\boldsymbol{\Phi}^\T\boldsymbol{\Phi} + \mSigma_p^{-1}\right)\vw + \sigma_n^{-2}\vy^\T\boldsymbol{\Phi}\vw + \text{const}$.
5. This is the form of @eq-gauss-recipe-nd with precision
   $\mA = \sigma_n^{-2}\boldsymbol{\Phi}^\T\boldsymbol{\Phi} + \mSigma_p^{-1}$
   and linear coefficient $\mathbf{h} = \sigma_n^{-2}\boldsymbol{\Phi}^\T\vy$.
   So the posterior is Gaussian with covariance $\mA^{-1}$ and mean
   $\mA^{-1}\mathbf{h}$.
:::

In one line:

$$
\vw \given \vy \;\sim\; \N\!\left(\bar{\vw},\; \mA^{-1}\right),
\qquad
\mA = \frac{1}{\sigma_n^2}\boldsymbol{\Phi}^\T\boldsymbol{\Phi} + \mSigma_p^{-1},
\qquad
\bar{\vw} = \frac{1}{\sigma_n^2}\mA^{-1}\boldsymbol{\Phi}^\T\vy.
$$ {#eq-bayes-blr-posterior}

This is the weight-space result of Gaussian process textbooks
[@rasmussen2006gaussian, sec. 2.1], and it has the shape we met in
@sec-gauss-product. The posterior precision $\mA$ is the prior
precision plus a term from the data, and the data term is a sum over
observations, $\sigma_n^{-2}\sum_i \boldsymbol{\phi}(x_i)\boldsymbol{\phi}(x_i)^\T$:
each observation adds precision along its own feature vector and nowhere else.
The posterior mean is the ridge regression fit with penalty
$\sigma_n^2/\sigma_p^2$ (@exr-bayes-ridge). As in @sec-gaussian-conditioning,
the posterior covariance does not depend on the observed values $\vy$, only on
where the observations were made.

The same posterior can be reached a second way. The pair $(\vw, \vy)$ is
jointly Gaussian, since $\vy$ is a linear map of $\vw$ plus independent noise,
so we could condition on $\vy$ with @eq-gauss-conditional. That route inverts
an $n \times n$ matrix, one row per observation, where
@eq-bayes-blr-posterior inverts a $2 \times 2$ matrix, one row per weight. The
two answers agree, by an identity that turns the inverse of an $n \times n$
matrix of this form into the inverse of a $2 \times 2$ one (the Woodbury
identity, @sec-id-woodbury), and @sec-function-space follows the $n \times n$ route because it survives when
the number of weights becomes infinite.

### Two views of the same posterior {#sec-bayes-two-views}

The figure shows the posterior in two spaces. On the left is data space, where
each observation is a point and each weight vector is a line. On the right is
weight space, where each weight vector is a point: the ellipses are the prior
and the posterior over $(w_1, w_2)$, drawn like the two-dimensional Gaussians
of @fig-gauss-2d. Each violet line on the left is one draw from the posterior,
and the violet dot of the same draw sits on the right.

```{figure}
//| figure: bayes-line
//| label: fig-bayes-line
//| fig-cap: "Bayesian linear regression, @eq-bayes-blr-posterior, in data space (left) and weight space (right). Click the left plot to add an observation; click a point to remove it. Each violet line is one draw from the posterior and appears as a violet dot in weight space. The shaded band is the 95% band for the line's value, the dashed curves the 95% band for a new noisy observation. Dashed ellipses: the prior; shaded: the posterior at Mahalanobis distance 1 and 2. The data are illustrative."
```

**Clear the data.** The posterior is the prior: a circle in weight space and,
in data space, a fan of lines narrowest at $x = 0$.

**Add a single point.** In weight space the circle collapses into a long thin
ellipse. One observation pins down the line's value at one input, a single
combination of intercept and slope, and leaves the other combination free: in
data space every drawn line passes near the point and pivots around it
(@exr-bayes-one-point).

**Add a second point far from the first.** The ellipse shrinks to a small blob
and the lines nearly agree. Two well-separated points determine a line.

**Now put the two points close together.** The slope stays uncertain, the
ellipse stays long, and the lines fan out on both sides of the pair. Where the
observations sit matters as much as how many there are.

**Raise the noise.** Each observation adds less precision, the posterior
relaxes toward the prior, and the lines miss the points more freely.

### A plane, and more inputs {#sec-bayes-plane}

Objectives in this book usually have more than one input, and a line becomes a
plane. With two inputs, the features are $\boldsymbol{\phi}(\vx) = (1, x_1, x_2)^\T$
and the weights are an intercept and two slopes. We write them
$\vw = (w_0, w_1, w_2)^\T$, numbering from zero so that the slope $w_j$
carries the index of its input $x_j$, and @eq-bayes-blr-posterior applies
unchanged with a $3 \times 3$ precision matrix.
What changes is the geometry of what the data can determine, and the figure
below makes it visible.

```{figure}
//| figure: bayes-plane
//| label: fig-bayes-plane
//| fig-cap: "Bayesian linear regression with two inputs. The reader evaluates a hidden plane, with noise of standard deviation 0.1, by clicking the map on the right; the shading there is the posterior standard deviation of $f$, stronger where the model is less sure. The 3-D view shows the observations on stems, the posterior mean plane (blue), and five planes drawn from the posterior (violet); drag it, or use the buttons, to turn it. The hidden plane and the starting inputs are illustrative."
```

**Look at the starting data.** The three inputs lie on the diagonal
$x_1 = x_2$. The drawn planes all pass near the three points, but they swing
about the diagonal like a door on a hinge, and the map is light along the
diagonal and dark in the two far corners. The data determine the slope along
the diagonal and say nothing about the tilt across it; the readout names that
direction.

**Click the map near a far corner.** One observation off the diagonal stops
the swinging. The planes snap together and the map lightens everywhere.

**Clear the map and evaluate random inputs.** Three inputs in general position,
not on one line, are enough to pin down a plane up to the noise.

The general rule follows from the structure of $\mA$. Each observation adds
precision only along its feature vector, so the data constrain only the
directions in weight space that their inputs span. With $d$ inputs a linear
model has $d + 1$ weights, and it needs at least $d + 1$ observations whose
inputs do not all lie in a lower-dimensional flat before every direction is
constrained. A six-parameter tuning problem fitted with a linear model needs
seven well-spread evaluations before the model stops being as uncertain as its
prior in some direction. A Gaussian process, which can bend, needs many more,
and how that number grows with the dimension is a theme of
@sec-high-dimensions.

Linear models in many dimensions are not only a teaching device. When the
features are themselves learned by a neural network, Bayesian linear
regression on those features has served as the surrogate for tuning
image-recognition and image-captioning models at large scale, with a cost
linear in the number of observations [@snoek2015scalable]. More strikingly, a
2026 study found that Bayesian linear regression, after a geometric
transformation of the inputs, matched the leading high-dimensional Bayesian
optimization methods on tasks with 60 to 6,000 dimensions [@doumont2026we].

### Computing it {#sec-bayes-blr-code}

The cost of @eq-bayes-blr-posterior is $O(nM^2)$ to form $\mA$ for $M$
features and $n$ observations, and $O(M^3)$ to factorize it. It grows only
linearly with the number of observations, which is the property the two
studies above exploit. A Gaussian process, by contrast, pays $O(n^3)$
(@sec-gp-computation).

::: {.code title="NumPy"}
```python
import numpy as np

def blr_posterior(Phi, y, prior_var, noise_var):
    """Posterior mean and covariance of w for y = Phi w + noise, w ~ N(0, prior_var I)."""
    M = Phi.shape[1]
    A = Phi.T @ Phi / noise_var + np.eye(M) / prior_var   # posterior precision
    L = np.linalg.cholesky(A)
    mean = np.linalg.solve(L.T, np.linalg.solve(L, Phi.T @ y / noise_var))
    Linv = np.linalg.solve(L, np.eye(M))
    cov = Linv.T @ Linv                                     # A^{-1}
    return mean, cov

x = np.array([-0.8, -0.3, 0.2, 0.7]); y = np.array([-0.5, 0.0, 0.35, 0.9])
Phi = np.column_stack([np.ones_like(x), x])              # features (1, x)
mean, cov = blr_posterior(Phi, y, prior_var=1.0, noise_var=0.09)
```
:::

## The predictive distribution {#sec-predictive}

A posterior over weights is a means to an end. What an optimizer needs is a
belief about the objective at an input it has not tried, and about the value it
would observe there. Both come from the same recipe as the coin's next flip:
average the prediction of each possible parameter value over the posterior,

$$
p(y_* \given x_*, \D) = \int p(y_* \given x_*, \vw)\, p(\vw \given \D)\, \dd\vw.
$$ {#eq-bayes-predictive}

This is the **posterior predictive distribution**. It accounts for two sources
of uncertainty at once: we do not know the parameters exactly, and even if we
did, a new observation would carry fresh noise.

For the linear model the integral needs no computing. The latent value
$f_* = \boldsymbol{\phi}(x_*)^\T\vw$ is a linear map of the Gaussian posterior
over $\vw$, so by @eq-gauss-affine it is Gaussian, and the observation adds
independent noise, which by @eq-gauss-sum adds its variance:

$$
f_* \given \D \sim \N\!\left(\boldsymbol{\phi}_*^\T\bar{\vw},\; \boldsymbol{\phi}_*^\T\mA^{-1}\boldsymbol{\phi}_*\right),
\qquad
y_* \given \D \sim \N\!\left(\boldsymbol{\phi}_*^\T\bar{\vw},\; \boldsymbol{\phi}_*^\T\mA^{-1}\boldsymbol{\phi}_* + \sigma_n^2\right),
$$ {#eq-bayes-blr-predictive}

with $\boldsymbol{\phi}_* = \boldsymbol{\phi}(x_*)$. The first variance is the
model's uncertainty about the line; the second adds the noise of one more
measurement, the distinction drawn in @sec-gp-noise.

```{figure}
//| figure: bayes-line
//| label: fig-bayes-predictive
//| fig-cap: "The predictive distribution of a line fitted to four observations clustered near the middle, @eq-bayes-blr-predictive. The shaded band covers 95% of the belief about the line's value; the dashed band adds the noise of a new observation. The band is narrowest at the center of the data and flares out away from it, because the variance of the line grows quadratically with the distance. The data are illustrative."
points: "-0.2:0.1,-0.1:0.3,0:0.2,0.1:0.4"
samples: false
```

**Read the band's shape.** It is narrowest at the center of the data and grows
on both sides, a bow tie. The line is pinned near the data and free to pivot,
so its uncertainty grows with the distance from the center.

**Compare the two bands.** Near the data the dashed band is much wider than
the shaded one: there the uncertainty about a new observation is mostly
measurement noise. Far away the two bands converge, because uncertainty about
the line dominates.

**Spread the points out.** Add observations near both ends. The bow tie
flattens, because well-separated inputs determine the slope.

A tempting shortcut is to predict with a single fitted line, plugging the
point estimate $\bar{\vw}$ into the likelihood. The resulting **plug-in
predictive** has variance $\sigma_n^2$ everywhere: it claims to know the line
exactly, even at $x = 10$, far from any data. It is overconfident exactly
where an optimizer most needs to know that it does not know.

The bow tie also shows what a linear model cannot represent. Between two
clusters of data, where nothing has been observed, a line is still pinned by
the clusters on either side, so its uncertainty there is small. For a straight
line that is correct. For an unknown objective it is not: the function could do
anything in the gap. A model for optimization needs uncertainty that grows
wherever data are absent, not only far from their center, and @sec-function-space
gets it by giving the linear model enough features to bend.

## Model evidence {#sec-evidence}

Every model so far came with choices: the prior's mean and strength for the
coin, the prior width $\sigma_p$ and the noise $\sigma_n$ for the line, the
features themselves. Numbers of this kind, which set up the model and are not
among the unknowns $\theta$ it reasons about, are the model's
**hyperparameters**. (Machine learning uses the same word for the settings of
a training procedure, as in @sec-cost-of-evaluation. In both uses it means a
setting one level above the quantities being learned.) How should the data
inform such choices? Picking the model whose best fit has the highest
likelihood does not work, because a more flexible model always fits at least
as well, including fitting the noise.

Bayes' rule answers one level up. Treat the model $\mathcal{M}$ itself as
unknown, and its probability after the data is

$$
p(\mathcal{M} \given \D) \propto p(\D \given \mathcal{M})\, p(\mathcal{M}),
\qquad
p(\D \given \mathcal{M}) = \int p(\D \given \theta, \mathcal{M})\, p(\theta \given \mathcal{M})\, \dd\theta.
$$ {#eq-bayes-evidence}

The quantity that decides is the evidence of @eq-bayes-rule, also called the
**marginal likelihood**: the probability the model assigned to the data
before seeing them, averaged over its prior. It does not reward the best fit
the model could achieve, but the fit it predicted on average.

### Occam's razor, automatically {#sec-bayes-occam}

That average builds in a preference for simple models. A probability
distribution over possible data sets must sum to one. A flexible model, able
to explain many different data sets, spreads that probability thin, and gives
each data set it could explain only a little. A rigid model concentrates its
probability on fewer data sets. When the data fall among the rigid model's
predictions, it wins; when they do not, the flexible model wins. This is
Occam's razor as a consequence of probability rather than a rule of thumb
[@mackay2003information, ch. 28].

::: {.example #ex-bayes-fair-coin title="Is the coin fair?"}
Compare two models of a coin. $\mathcal{M}_0$ says it is fair, $\theta = 1/2$
exactly. $\mathcal{M}_1$ says the bias is unknown, with a uniform prior. For a
particular sequence of $n$ flips with $h$ heads, $\mathcal{M}_0$ gives
probability $2^{-n}$, and by step 5 of the derivation in @sec-bayes-beta-dist,
$\mathcal{M}_1$ gives $B(h + 1, t + 1) = h!\,t!/(n + 1)!$.

With 5 heads in 10 flips, the evidence is $1/1024 \approx 9.8 \times 10^{-4}$
for the fair coin and $5!\,5!/11! \approx 3.6 \times 10^{-4}$ for the unknown
bias: the data favor the fair coin by a factor of about 2.7. The flexible model
spent probability on lopsided outcomes that did not happen. With 9 heads in 10
flips, the fair coin's evidence is unchanged while the flexible model's rises
to $9!\,1!/11! = 1/110 \approx 9.1 \times 10^{-3}$, and the data favor an
unknown bias by a factor of about 9.3.
:::

For the linear model the evidence has a closed form. The observations are a
linear map of the Gaussian weights plus independent Gaussian noise, so by
@eq-gauss-affine and @eq-gauss-sum they are Gaussian,
$\vy \sim \N(\mathbf{0},\, \mK_y)$ with
$\mK_y = \boldsymbol{\Phi}\mSigma_p\boldsymbol{\Phi}^\T + \sigma_n^2\mI$, and the
log evidence is the log density of that Gaussian at the observed $\vy$:

$$
\log p(\vy) = \underbrace{-\tfrac12\vy^\T\mK_y^{-1}\vy}_{\text{data fit}}
\;\underbrace{-\;\tfrac12\log\lvert\mK_y\rvert}_{\text{complexity}}
\;-\;\tfrac{n}{2}\log 2\pi.
$$ {#eq-bayes-log-evidence}

The two terms pull in opposite directions. A wider prior makes $\mK_y$ larger,
which improves the data fit term for data that need large weights but makes
the determinant larger, the cost of spreading probability over more data sets.
The same formula, with a kernel matrix in place of
$\boldsymbol{\Phi}\mSigma_p\boldsymbol{\Phi}^\T$, is the marginal likelihood
that @sec-marginal-likelihood maximizes to fit a Gaussian process's
hyperparameters.

```{figure}
//| figure: bayes-line
//| label: fig-bayes-evidence
//| fig-cap: "The log evidence of a straight-line model, @eq-bayes-log-evidence, for four observations. The readout under weight space reports it. With these data it peaks at a prior standard deviation of about 0.5 to 0.7 and falls on both sides: a narrow prior cannot reach the slope the data need, and a wide one wastes probability on lines the data rule out. The data are illustrative."
points: "-0.8:-0.5,-0.3:0,0.2:0.35,0.7:0.9"
```

**Sweep the prior width.** Move $\sigma_p$ from 0.1 to 4 and watch the log
evidence. It rises from about $-4.8$ to a maximum near $-2.2$ and falls again to
about $-4.9$. Choosing $\sigma_p$ at the peak is the practice called
**type II maximum likelihood**, or empirical Bayes: maximize the evidence over
the prior's settings rather than the likelihood over the weights.

**Vary the noise too.** A noise level far too small makes the data fit term
punishing; one far too large makes every data set look plausible and none
likely. The evidence is highest for the noise that explains the scatter of
the points around a line.

The evidence has known weaknesses. It depends on the prior's width even where
the posterior barely does: doubling an already vague prior hardly changes the
fitted line but lowers the evidence. And maximizing it over many
hyperparameters can overfit, that is, fit the accidents of a small data set, a
failure @sec-fitting-hyperparameters discusses for Gaussian processes. It remains the standard tool for setting the
hyperparameters of the models in this book.

## When there is no closed form {#sec-no-closed-form}

Conjugate pairs are the exception. The Beta prior fitted the coin because the
likelihood was a power of $\theta$ and $1 - \theta$; the Gaussian prior fitted
the line because the noise was Gaussian. Change either and the posterior
leaves the family.

The case this book cares most about is a comparison. When a person says that
option $A$ is better than option $B$, a common model, developed in
@sec-thurstone, sets the probability of that answer to
$\Phi\big((f_A - f_B)/\sigma\big)$, the standard normal distribution function
from @eq-gauss-phi applied to the difference in utility. (Here $\sigma$ is the
noise of the difference; @sec-thurstone writes it as $\sqrt{2}$ times the noise
of each option.) With a Gaussian prior on the difference
$\Delta = f_A - f_B$, the posterior after one answer is proportional to

$$
\N(\Delta;\, 0, s^2)\;\Phi(\Delta/\sigma),
$$

a Gaussian density multiplied by an S-shaped curve. The product is lopsided:
it keeps the prior's upper tail and cuts away its lower one. It is not a
Gaussian, and after many answers about many options, its normalizing integral
runs over as many dimensions as there are options.

Four families of methods handle such posteriors, and @sec-approx-inference
compares them. The **Laplace approximation** (@sec-laplace) replaces the
posterior by a Gaussian centered at its peak, with a covariance taken from the
curvature there. **Expectation propagation** (@sec-ep) and **variational
inference** (@sec-vi) choose a Gaussian by matching the posterior in other
senses. **Sampling** methods (@sec-mcmc) draw from the posterior itself and
replace every integral by an average over draws. The first three bring the
problem back to Gaussians, so that everything in @sec-gaussian applies again.
The Gaussian process preference model of @chu2005preference, on which much
of @sec-part-preferences builds, used the Laplace approximation.

One tool is still missing. This chapter argued that an optimizer should
evaluate where it will learn the most, without saying how learning is
measured. @sec-information supplies the unit.

## Exercises {#sec-bayes-exercises}

::: {.exercise #exr-bayes-pseudo-counts}
A coin has a $\mathrm{Beta}(2, 2)$ prior, and you observe 8 heads in 10 flips.
Find the posterior, its mean, its mode, and the maximum likelihood estimate.
How many flips is the prior worth, and what is its weight in the posterior
mean?

::: {.solution}
The posterior is $\mathrm{Beta}(2 + 8, 2 + 2) = \mathrm{Beta}(10, 4)$. Its
mean is $10/14 \approx 0.714$ and its mode is $9/12 = 0.75$, while the MLE is
$8/10 = 0.8$. The prior is worth $n_0 = 4$ flips with mean 0.5, so by
@eq-bayes-shrinkage the posterior mean is
$\frac{4}{14}(0.5) + \frac{10}{14}(0.8) = 0.143 + 0.571 = 0.714$: the prior
carries a weight of $4/14 \approx 0.29$.
:::
:::

::: {.exercise #exr-bayes-one-point}
Take the line model with $\mSigma_p = \mI$ and a single observation at input
$x_1$, so $\boldsymbol{\Phi}$ is the row $\boldsymbol{\phi}_1^\T = (1, x_1)$.
Show that the posterior precision $\mA$ leaves the prior variance unchanged
in one direction of weight space. Which direction is it, and what does it mean
for the lines in @fig-bayes-line?

::: {.solution}
Here $\mA = \mI + \sigma_n^{-2}\boldsymbol{\phi}_1\boldsymbol{\phi}_1^\T$. For any
$\mathbf{u}$ orthogonal to $\boldsymbol{\phi}_1$, $\mA\mathbf{u} = \mathbf{u}$,
so $\mathbf{u}$ is an eigenvector with eigenvalue 1, and the posterior
variance in that direction equals the prior variance 1. Along
$\boldsymbol{\phi}_1$ itself the eigenvalue is
$1 + \lVert\boldsymbol{\phi}_1\rVert^2/\sigma_n^2$, larger, so the variance
shrinks. The constrained combination is
$\boldsymbol{\phi}_1^\T\vw = w_1 + w_2 x_1$, the line's value at $x_1$. The
unconstrained direction $\mathbf{u} = (-x_1, 1)^\T$ changes the slope by one
unit and the intercept by $-x_1$, which rotates the line about the point
$x = x_1$. That is the pivoting of the drawn lines.
:::
:::

::: {.exercise #exr-bayes-ridge}
Show that with $\mSigma_p = \sigma_p^2\mI$ the posterior mean
@eq-bayes-blr-posterior minimizes the ridge regression objective
$\lVert\vy - \boldsymbol{\Phi}\vw\rVert^2 + \lambda\lVert\vw\rVert^2$ with
$\lambda = \sigma_n^2/\sigma_p^2$. What does the posterior mean become as
$\sigma_p \to \infty$?

::: {.solution}
Setting the gradient of the objective to zero gives
$-2\boldsymbol{\Phi}^\T(\vy - \boldsymbol{\Phi}\vw) + 2\lambda\vw = \mathbf{0}$,
so $\vw = (\boldsymbol{\Phi}^\T\boldsymbol{\Phi} + \lambda\mI)^{-1}\boldsymbol{\Phi}^\T\vy$.
The posterior mean is
$\bar{\vw} = \sigma_n^{-2}(\sigma_n^{-2}\boldsymbol{\Phi}^\T\boldsymbol{\Phi} + \sigma_p^{-2}\mI)^{-1}\boldsymbol{\Phi}^\T\vy$;
multiplying inside and outside the inverse by $\sigma_n^2$ turns it into
$(\boldsymbol{\Phi}^\T\boldsymbol{\Phi} + (\sigma_n^2/\sigma_p^2)\mI)^{-1}\boldsymbol{\Phi}^\T\vy$,
the same expression. Since the posterior is Gaussian, its mean is also its
mode, so this is the MAP estimate too. As $\sigma_p \to \infty$, $\lambda \to 0$
and the posterior mean tends to the least-squares solution
$(\boldsymbol{\Phi}^\T\boldsymbol{\Phi})^{-1}\boldsymbol{\Phi}^\T\vy$, the
maximum likelihood estimate, provided $\boldsymbol{\Phi}^\T\boldsymbol{\Phi}$ is
invertible.
:::
:::

::: {.exercise #exr-bayes-evidence}
Under a uniform prior, find the evidence for a particular sequence of 3 heads
in 3 flips, and compare it with the fair-coin model of @ex-bayes-fair-coin.
Which model do the data favor, and by how much? How many heads in a row are
needed before the unknown-bias model is favored by a factor of 10?

::: {.solution}
The unknown-bias model gives $B(4, 1) = 3!\,0!/4! = 1/4$, and the fair coin
gives $2^{-3} = 1/8$, so the data favor the unknown bias by a factor of 2.
For $n$ heads in a row the ratio is $\frac{n!/(n+1)!}{2^{-n}} = \frac{2^n}{n + 1}$,
which is $64/7 \approx 9.1$ for $n = 6$ and $128/8 = 16$ for $n = 7$. Seven
heads in a row are needed for a factor of 10. A run of heads is evidence
against fairness, but less overwhelming than intuition suggests, because the
flexible model had to spread its probability over every possible bias.
:::
:::

## Further reading {#further-reading .unnumbered}

- @gelman2013bayesian, chapter 2, treats single-parameter models, starting
  with the binomial and its Beta prior, and the book as a whole is the
  standard reference for applied Bayesian modeling, including prior
  predictive checks.
- @bishop2006pattern, section 2.1, covers the Beta distribution and the coin,
  and chapter 3 covers Bayesian linear regression, its predictive
  distribution, and the evidence for model comparison in the notation used
  here.
- @rasmussen2006gaussian, section 2.1, derives Bayesian linear regression in
  the weight-space view and then passes to Gaussian processes, the step
  @sec-function-space takes.
- @mackay2003information, chapter 28, explains Occam's razor through the
  evidence, with the pictures that made the argument famous.
- @thompson1933likelihood posed the question of comparing two unknown success
  probabilities from data, and with it the sampling rule that now carries his
  name.
