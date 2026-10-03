---
status: done
synopsis: "Probability as a budget of belief spread over possibilities: random variables, discrete and continuous distributions, and the two rules everything else follows from, the sum rule and the product rule. Bayes' rule falls out of them in one line, and expectation, variance, and independence complete the toolkit."
sources: ["textbooks", "Blitzstein and Hwang 2019, ch. 1-7", "MacKay 2003, ch. 2-3", "Bishop 2006, sec. 1.2"]
---

# Probability as Bookkeeping for Uncertainty {#sec-probability}

@sec-model-then-decide set a requirement. To decide where to
evaluate an expensive objective next, a Bayesian optimizer needs a model that
knows what it does not know: one that can say "the objective is probably high
here, and we have no idea over there". This chapter supplies the language in
which such statements become numbers that a program can store, combine, and
update. That language is probability.

We will treat probability as bookkeeping. There is a fixed budget of belief,
one unit in total, spread over everything that might be true. Two rules say how
the books must balance: the *sum rule*, used when we stop tracking a quantity,
and the *product rule*, used when we combine a quantity with something that
depends on it. Bayes' rule, the step that turns data into revised belief,
follows from these two in one line. No later chapter needs a third rule. The
surrogate model, the acquisition function, and the model of a person's
preferences are all these rules applied to larger objects.

The chapter starts with what a probability is and what it is about, builds the
two rules from a small table of numbers, and then uses them to update a belief
one observation at a time, with a coin you can flip. It ends with the summaries
(expectation and variance) and the assumption (independence) that later
chapters lean on constantly. Readers who have taken a probability course can
skim, but should not skip @sec-bayes-rule and @sec-independence, which set up
ideas the Gaussian process chapters reuse.

## Uncertainty as a quantity {#sec-uncertainty-quantity}

Recall the first scenario of @sec-cost-of-evaluation: tuning the learning rate
of a neural network, where each training run takes hours. Before running
anything, you already hold opinions. A learning rate of $10^{-3}$ will probably
train; a learning rate of $10$ takes steps so large that training will almost
certainly diverge; $10^{-2}$ could go either way. These opinions decide which
run you start first. An automatic method must hold opinions too, in a form it
can compute with: it has to say
"probably" with a number, and it has to change that number in a disciplined way
when a run finishes.

### Two readings of a probability {#sec-prob-two-readings}

There are two common answers to the question of what a probability measures.

The *frequency* reading says that the probability of heads is the fraction of
heads in a long run of flips. It is concrete and can be checked by experiment,
but it needs an experiment that can be repeated.

The *belief* reading says that a probability measures how strongly one should
believe a statement, given the information at hand. It applies to statements
that cannot be repeated. "The best learning rate for this network on this
data set lies between $10^{-3}$ and $10^{-2}$" is either true or false; there is
no long run of networks to count over. What is uncertain is our knowledge, not
the world.

Bayesian optimization needs the belief reading, because its unknown is a fixed
function. The objective does not change between evaluations (apart from
measurement noise, which @sec-independence treats separately). When the model
says "the value at $x$ is probably between 0.8 and 0.9", the spread describes
our ignorance about one particular function, and each evaluation reduces it.
This is why the methods in this book are called *Bayesian*, after Thomas Bayes,
whose essay, published posthumously in 1763, worked out how to revise such a
belief about an unknown chance from the number of times an event had happened
and failed [@bayes1763essay]. That is the coin problem of @sec-prob-coin.

A reader may wonder why degrees of belief should obey the same arithmetic as
frequencies rather than some other calculus. @cox1946probability argued that
any way of attaching numbers to plausibility that meets a few consistency
requirements must follow the same sum and product rules. Two examples of such
requirements: information that is equivalent must lead to equal plausibility,
and the plausibility of "A and B" must be determined by the plausibility of A
together with the plausibility of B once A is known. @jaynes2003probability
builds the whole of probability theory on this argument. For our purposes the
upshot is practical: both readings share one calculus, so we never have to
choose between them in order to compute.

### Outcomes, events, and the axioms {#sec-prob-axioms}

To compute with beliefs we need a small amount of structure. An *outcome* is
one complete way the world could turn out, and the *sample space* $\Omega$ is
the set of all outcomes. An *event* is a set of outcomes, such as "the training
run diverges". A probability assigns a number to each event.

::: {.definition #def-prob-probability title="Probability"}
A probability $\Prob$ assigns to every event $A$ a number $\Prob(A)$ such that

1. $\Prob(A) \ge 0$ for every event $A$;
2. $\Prob(\Omega) = 1$;
3. $\Prob(A \cup B) = \Prob(A) + \Prob(B)$ whenever $A$ and $B$ have no
   outcome in common.
:::

These three axioms are the bookkeeping rules in their most basic form
[@blitzstein2019introduction]. One unit of belief is spread over the
outcomes; the probability of an event is the belief sitting on its outcomes;
two events that share no outcome cannot share any belief, so their amounts add.
Everything else follows. For example, an event $A$ and its complement "not
$A$" share no outcome and together cover $\Omega$, so
$\Prob(\text{not } A) = 1 - \Prob(A)$.

### Random variables {#sec-prob-random-variables}

We rarely care about outcomes in full detail. We care about numbers computed
from them: an accuracy, a count, the value of an objective. A *random variable*
is such a number.

::: {.definition #def-prob-random-variable title="Random variable"}
A random variable $X$ is a function that assigns a number $X(\omega)$ to each
outcome $\omega$ in the sample space.
:::

The name is misleading on both counts: a random variable is neither random nor
a variable. It is a deterministic function whose input we do not know. Take a
training run. The outcome $\omega$ is everything that could vary: the random
seed, the order of the data, nondeterminism in the hardware. The validation
accuracy is a function of $\omega$. Once the run finishes, $\omega$ is fixed and
the accuracy is a number; before it finishes, all we can say is how our belief
is spread over the numbers it might be. A programmer can read $X$ as a pure
function applied to a hidden argument.

A word on notation, since later chapters rely on it. A capital letter $X$
names the random variable and a lowercase $x$ names a value it might take, so
"$X = x$" is the event that the variable takes the value $x$, and
$\Prob(X = x)$ is its probability. We write $p(x)$ for this probability when
the variable is clear from context. The symbol $\sim$ reads "is distributed
as": $X \sim \text{Bernoulli}(0.3)$ says that $X$ follows the distribution
named on the right. A vertical bar reads "given": $p(x \given y)$ is a
probability of $x$ computed as if $y$ were known, made precise in
@sec-joint-marginal-conditional. From @sec-linear-algebra onward the book
follows the custom of machine learning and drops the capitals, writing $p(y)$ and
$y \sim \N(0, 1)$ with one letter for both the variable and its value; the
context always tells which is meant.

## Discrete distributions {#sec-discrete}

A random variable is *discrete* when its possible values can be listed: a coin
lands heads or tails, a die shows one of six faces, a count is $0, 1, 2,
\dots$. For such a variable the books are a table, one entry per value.

::: {.definition #def-prob-pmf title="Probability mass function"}
The probability mass function of a discrete random variable $X$ is
$p(x) = \Prob(X = x)$. It satisfies $p(x) \ge 0$ for every $x$ and
$\sum_x p(x) = 1$.
:::

The simplest discrete distribution has two values. A *Bernoulli* random
variable is 1 ("heads", "success") with probability $\theta$ and 0 with
probability $1 - \theta$. Both cases fit in one formula,

$$
p(x \given \theta) = \theta^{x} (1 - \theta)^{1 - x}, \qquad x \in \{0, 1\},
$$ {#eq-prob-bernoulli}

which gives $\theta$ when $x = 1$ and $1 - \theta$ when $x = 0$. The bar in
$p(x \given \theta)$ separates the value whose probability we are stating from
the quantity it depends on. In the belief reading this is more than a notation
for a parameter: an unknown $\theta$ is just another uncertain quantity, and
$p(x \given \theta)$ is a conditional probability in the sense of
@sec-joint-marginal-conditional.

The Bernoulli distribution describes every yes-or-no outcome in this book. Does
a training run diverge? Does a test fail? Does a person, shown two designs,
prefer the first? @sec-part-preferences models each such comparison as a Bernoulli variable
whose $\theta$ depends on how much better the first option is than the second
(@sec-comparisons).

Count the heads in $n$ flips and you get a *binomial* random variable $K$. Any
particular sequence of flips with $k$ heads and $n - k$ tails has probability
$\theta^k (1 - \theta)^{n - k}$, because the flips do not influence each other
and probabilities of non-interacting events multiply (a fact made precise in
@sec-independence). There are $\binom{n}{k}$ such sequences, one for each way
of choosing which $k$ flips land heads, and no two of them can happen together,
so the third axiom adds their probabilities:

$$
p(k \given n, \theta) = \binom{n}{k} \theta^k (1 - \theta)^{n - k},
\qquad k = 0, 1, \dots, n.
$$ {#eq-prob-binomial}

For ten flips of a fair coin, the probability of exactly five heads is
$\binom{10}{5} / 2^{10} = 252 / 1024 \approx 0.246$. Even for a fair coin, an
even split happens less than one time in four.

A die has six values, each with its own probability $p_1, \dots, p_6$ summing
to one. This is a *categorical* distribution, the generalization of the
Bernoulli to any finite number of outcomes. It appears when a person picks one
favorite from several options at once, the setting of Luce's choice model in
@sec-bradley-terry.

::: {.example #ex-prob-random-search title="At least one good configuration"}
Suppose 5% of the configurations in a search space are "good", and we try
configurations drawn at random, each independently of the others. What is the
probability that 20 random tries find at least one good one?

The event "at least one" is awkward to count directly, but its complement,
"none", is easy. Each try misses with probability $0.95$, and independent
misses multiply, so all 20 miss with probability $0.95^{20} \approx 0.358$. By
the complement rule, at least one try succeeds with probability
$1 - 0.358 = 0.642$. This kind of calculation explains why random search is a
strong baseline when only a small fraction of a space needs to be found
(@sec-usual-tools): its chance of success depends on the fraction of good
configurations, however many dimensions the space has.
:::

## Continuous distributions {#sec-continuous}

Validation accuracy and the value of an objective at an input are real
numbers. A table cannot hold them. The values of a count can at least be
listed, $0, 1, 2, \dots$, with one probability for each; the real numbers in
an interval cannot be listed, and no assignment of positive probability to
every one of them sums to one. In fact, for a continuous quantity, every exact
value has probability zero. The probability that the accuracy is exactly
$0.9137000\dots$ is zero. What carries probability is an interval: the
accuracy lies between 0.91 and 0.92.

### Densities {#sec-prob-densities}

The fix is to record probability per unit length, the way physics records mass
per unit length. A steel rod has a density of so many grams per centimeter; no
single point of the rod has any mass, but every segment does, and its mass is
the integral of the density over the segment. A *probability density* works the
same way.

::: {.definition #def-prob-density title="Probability density"}
A continuous random variable $X$ has density $p(x)$ if, for every interval,

$$
\Prob(a \le X \le b) = \int_a^b p(x)\, \dd x.
$$

A density satisfies $p(x) \ge 0$ and $\int_{-\infty}^{\infty} p(x)\, \dd x = 1$.
:::

For a short interval of width $\Delta$ around $x$, the integral is close to the
height times the width, so $\Prob(x \le X \le x + \Delta) \approx p(x)\,
\Delta$. The density is the probability of a small interval divided by its
width.

::: {.pitfall title="A density is not a probability"}
A density can be larger than one. The uniform distribution on the interval
$[0, 0.1]$ has density $10$ there, because its one unit of probability is
packed into a length of $0.1$. A density also has units: one over the units of
$x$. A ratio of densities at two points is meaningful (it compares the
probabilities of small intervals around them), and so is the area under a
density, but a density value on its own is not the probability of anything.
This matters later, when the logarithm of a density evaluated at observed data
comes out as a large positive number: nothing has gone wrong.
:::

### The cumulative distribution function {#sec-prob-cdf}

A second way to describe any random variable, discrete or continuous, is its
*cumulative distribution function* (CDF),

$$
F(x) = \Prob(X \le x).
$$

The CDF rises from 0 on the far left to 1 on the far right and never
decreases. The probability of an interval is a difference of two CDF values,
$\Prob(a < X \le b) = F(b) - F(a)$, and for a continuous variable the density
is the slope of the CDF, $p(x) = F'(x)$. The CDF of the standard Gaussian,
written $\Phi$ and introduced in @sec-gaussian-1d, appears throughout the book:
in the probability of improvement and expected improvement (@sec-pi, @sec-ei)
and in the model of a person's comparisons (@sec-thurstone).

### Two examples {#sec-prob-continuous-examples}

The *uniform* distribution on $[a, b]$ has constant density $1 / (b - a)$
between $a$ and $b$ and zero elsewhere. It encodes "every value in the range is
equally plausible", and it is how random search and many initial designs choose
inputs (@sec-initial-design).

The *exponential* distribution describes the waiting time until an event that
occurs at a constant rate $\lambda$, such as the next failure of a long-running
job when failures strike at random. Its density and CDF for $x \ge 0$ are

$$
p(x) = \lambda e^{-\lambda x}, \qquad F(x) = 1 - e^{-\lambda x}.
$$

With one failure per day on average, $\lambda = 1$ per day, and the probability
that the next failure comes between one and two days from now is
$F(2) - F(1) = e^{-1} - e^{-2} \approx 0.233$. The same number comes from
integrating the density from 1 to 2.

::: {.aside title="Densities change when you change units"}
Rescaling a variable rescales its density. If $Y = cX$ for a constant $c > 0$,
then an interval of width $\Delta$ for $X$ becomes one of width $c\Delta$ for
$Y$ carrying the same probability, so $p_Y(y) = p_X(y / c) / c$. More generally,
for an increasing transformation $y = g(x)$, $p_Y(y) = p_X(x)\, \dd x / \dd y$.
One consequence surprises people: a distribution that is flat over a positive
quantity $\ell$ is not flat over $\log \ell$, and vice versa. "Uninformative"
depends on the units in which it is stated, which matters when a prior has to
be chosen for a model's own settings (@sec-kern-map).
:::

## Joint, marginal, and conditional {#sec-joint-marginal-conditional}

So far each distribution has described one quantity. Bayesian optimization is
about several quantities at once: the objective's values at two nearby inputs,
a person's answer and the preference behind it, a test result and whether the
code is broken. The point of a model is that learning one of these tells us
something about another. That requires a distribution over several quantities
together.

### A table of joint probabilities {#sec-prob-joint-table}

Take an example every software engineer knows. A commit is either *broken* or
*fine*, and the continuous integration test run on it either *fails* or
*passes*. Suppose the team's history says that 10% of commits are broken, that
the test fails on 90% of broken commits, and that it also fails on 5% of fine
commits, because of flaky tests and timeouts. In terms of counts: of 1,000
commits, 100 are broken and 90 of those fail the test; 900 are fine and 45 of
those fail anyway.

Dividing the counts by 1,000 gives the belief about the next commit, spread
over the four combinations. This is the *joint distribution* of the two
variables, and its table, with the sums of each row and column written in the
margins, holds everything there is to know about them.

::: {.table #tbl-prob-joint title="Joint probabilities of a commit's state and its test result, with row and column sums."}
|            | fail  | pass  | total |
|:-----------|------:|------:|------:|
| **broken** | 0.090 | 0.010 | 0.100 |
| **fine**   | 0.045 | 0.855 | 0.900 |
| **total**  | 0.135 | 0.865 | 1     |
:::

Three different questions can be read off this table, and each corresponds to
one operation.

### The sum rule: marginalizing {#sec-prob-sum-rule}

What is the probability that the next commit is broken, regardless of the test?
Add the broken row: $0.090 + 0.010 = 0.100$. What is the probability that the
test fails, regardless of the commit? Add the fail column:
$0.090 + 0.045 = 0.135$. Each total in the margin is the distribution of one
variable on its own, and it is called a *marginal distribution* for that reason.
Summing over a variable we no longer want to track is *marginalizing it out*. It
does not discard that variable's belief; it pools it.

### Conditioning {#sec-prob-conditioning}

Now suppose the test has failed. Which commits are still possible? Only those
in the fail column: the pass column is ruled out. The belief in the fail column
adds up to only $0.135$, so to make it a proper distribution again we divide
each entry by $0.135$:

$$
\Prob(\text{broken} \given \text{fail}) = \frac{0.090}{0.135} \approx 0.667,
\qquad
\Prob(\text{fine} \given \text{fail}) = \frac{0.045}{0.135} \approx 0.333.
$$

This is *conditioning*: strike out what the observation rules out, and rescale
what remains so it sums to one. A failing test raised the probability that the
commit is broken from 10% to about 67%. Every act of learning from data in this
book, from a Gaussian process absorbing an evaluation to a preference model
absorbing an answer, is this operation on a larger table.

### The product rule {#sec-prob-product-rule}

Written as a formula, conditioning divides a joint probability by a marginal:
$p(y \given x) = p(x, y) / p(x)$. Multiply both sides by $p(x)$ and read the
result the other way around. It says how to *build* a joint distribution: first
choose $x$ with probability $p(x)$, then choose $y$ with the probability that
applies once $x$ is known. This is how @tbl-prob-joint was made: the entry for
broken and fail is $\Prob(\text{broken}) \times \Prob(\text{fail} \given
\text{broken}) = 0.1 \times 0.9 = 0.09$. Together with the sum rule, this gives
the two rules the chapter's opening promised.

::: {.definition #def-prob-rules title="The sum rule and the product rule"}
For random variables $X$ and $Y$,

$$
p(x) = \sum_y p(x, y) \qquad \text{(sum rule)},
$$ {#eq-prob-sum-rule}

$$
p(x, y) = p(y \given x)\, p(x) = p(x \given y)\, p(y) \qquad \text{(product rule)}.
$$ {#eq-prob-product-rule}

The *conditional distribution* $p(y \given x) = p(x, y) / p(x)$ is defined
whenever $p(x) > 0$. For continuous variables, sums become integrals and
probabilities become densities.
:::

The product rule extends to any number of variables by applying it
repeatedly, peeling off one variable at a time:

$$
p(x_1, x_2, x_3) = p(x_1)\, p(x_2 \given x_1)\, p(x_3 \given x_1, x_2).
$$

This *chain rule* is the shape of every model in the book: a distribution over
the unknowns, then a distribution of the data given the unknowns.

Combining the two rules gives a formula used so often that it has a name of
its own. Marginalize $y$ with the sum rule, then expand each joint term with
the product rule:

$$
p(y) = \sum_x p(y \given x)\, p(x).
$$ {#eq-prob-total}

This is the *law of total probability*. In words: the probability of an
observation is its probability under each possibility, averaged with the
possibilities' own probabilities as weights. For the test,
$\Prob(\text{fail}) = 0.9 \times 0.1 + 0.05 \times 0.9 = 0.135$, the column
total of @tbl-prob-joint. This sentence returns as the denominator of Bayes'
rule.

### Seeing the rules {#sec-prob-mosaic}

@fig-prob-mosaic draws the same joint distribution as areas. The unit square
is cut into two columns whose widths are the probabilities of broken and fine;
each column is cut at a height given by the probability that the test fails
for that kind of commit. Width times height is area, so each region's area is
a joint probability: the product rule, drawn.

```{figure}
//| figure: prob-mosaic
//| label: fig-prob-mosaic
//| fig-cap: "A joint distribution of two yes-or-no quantities drawn as areas. Column widths are the probabilities that a commit is broken or fine; within each column, the solid region is the probability that the test fails given that state, and the faded region that it passes. Each region's area is a joint probability, listed with its row and column sums in the table. Choose what to condition on, or click a region, to keep only the matching regions and rescale them to sum to one. The rates are illustrative."
```

Some things to try:

- **Condition on fail, then on broken.** The first gives
  $\Prob(\text{broken} \given \text{fail}) \approx 0.667$, the second
  $\Prob(\text{fail} \given \text{broken}) = 0.9$. They are different numbers
  answering different questions: the first keeps the fail regions of both
  columns, the second keeps the broken column. Mistaking one for the other is
  the error that @sec-bayes-rule guards against.
- **Condition on pass.** A passing test lowers the probability of a broken
  commit from 0.1 to about 0.012. It does not make it zero, because 10% of broken
  commits pass.
- **Set $\Prob(\text{fail} \given \text{fine})$ to zero.** With no false
  alarms, every failure comes from a broken commit, and conditioning on fail
  gives certainty.
- **Lower $\Prob(\text{broken})$ to 0.01.** The broken column shrinks to a
  sliver, and so does its share of the fail regions. @sec-bayes-rule explains
  what this does to the meaning of a failure.

### The same rules for densities {#sec-prob-continuous-rules}

Everything above holds for continuous quantities with integrals in place of
sums. A joint density $p(x, y)$ over two real numbers has marginal
$p(x) = \int p(x, y)\, \dd y$ and conditional
$p(y \given x) = p(x, y) / p(x)$. One subtlety: conditioning on an exact value
$X = x$ conditions on an event of probability zero, which the definition of
conditional probability for events cannot handle. The density form handles it
naturally. Take the joint density along the line $X = x$, a one-dimensional
slice, and rescale the slice so that it integrates to one. Conditioning a
Gaussian on observed values is exactly this slicing (@sec-gaussian-conditioning),
and so is Gaussian process regression (@sec-gp-conditioning).

## Bayes' rule {#sec-bayes-rule}

In the test example, our knowledge naturally runs in one direction. We know how
the test behaves on broken and on fine commits, $\Prob(\text{fail} \given
\text{broken})$, because we can measure it by breaking code on purpose. The
question we face runs the other way: the test failed, so how likely is it that
the commit is broken? Bayes' rule turns one conditional into the other.

::: {.derivation title="Bayes' rule"}
1. By the product rule (@eq-prob-product-rule) written one way,
   $p(x, y) = p(y \given x)\, p(x)$.
2. By the same rule written the other way, $p(x, y) = p(x \given y)\, p(y)$.
3. Both right-hand sides equal $p(x, y)$, so
   $p(x \given y)\, p(y) = p(y \given x)\, p(x)$.
4. Divide both sides by $p(y)$, which is allowed whenever $p(y) > 0$:
   $p(x \given y) = p(y \given x)\, p(x) / p(y)$.
5. Expand the denominator with the law of total probability (@eq-prob-total),
   renaming the summation variable to $x'$ to keep it apart from the $x$ in the
   numerator: $p(y) = \sum_{x'} p(y \given x')\, p(x')$.
:::

The result is Bayes' rule. It earns names for its parts when the variable we
condition on is data, $\D$, and the other is an unknown, $\theta$, such as a
coin's bias, whether a commit is broken, or later a whole function:

$$
\underbrace{p(\theta \given \D)}_{\text{posterior}}
= \frac{\overbrace{p(\D \given \theta)}^{\text{likelihood}}\;
        \overbrace{p(\theta)}^{\text{prior}}}
       {\underbrace{p(\D)}_{\text{evidence}}},
\qquad
p(\D) = \sum_{\theta} p(\D \given \theta)\, p(\theta).
$$ {#eq-prob-bayes}

- The *prior* $p(\theta)$ is the belief about $\theta$ before seeing the data.
- The *likelihood* $p(\D \given \theta)$ is how probable the observed data would
  be if $\theta$ were the truth. It is read as a function of $\theta$ with the
  data held fixed, and it is not a distribution over $\theta$: its values need
  not sum to one.
- The *posterior* $p(\theta \given \D)$ is the revised belief.
- The *evidence* $p(\D)$, also called the *marginal likelihood*, is the
  probability the model gave to the data before seeing them. It does not depend
  on $\theta$; within one update it only rescales.

Because the evidence is the same for every $\theta$, Bayes' rule is often
written as a proportionality:

$$
p(\theta \given \D) \propto p(\D \given \theta)\, p(\theta).
$$ {#eq-prob-proportional}

The symbol $\propto$ means "equal up to a factor that does not depend on
$\theta$". As a procedure: multiply the prior by the likelihood, entry by
entry, then rescale so the result sums to one. The rescaling factor is one over
the evidence. @sec-prior-likelihood-posterior develops this vocabulary further,
and the evidence returns as the tool for choosing a model's own settings
(@sec-evidence, @sec-marginal-likelihood).

::: {.keyidea title="Posterior is prior times likelihood, rescaled"}
To learn from data, multiply what you believed by how well each possibility
predicted what you saw, then rescale so the total is one again. The amount of
rescaling is the probability you had given to what you saw.
:::

### When the prior is small {#sec-prob-base-rates}

The example below applies Bayes' rule in a setting where the result surprises
most people the first time.

::: {.example #ex-prob-flaky-test title="A failing test on a rarely broken codebase"}
Suppose only 1% of commits are broken, while the test keeps its rates: it fails
on 90% of broken commits and on 5% of fine ones. A commit fails the test. By
@eq-prob-bayes,

$$
\Prob(\text{broken} \given \text{fail})
= \frac{0.9 \times 0.01}{0.9 \times 0.01 + 0.05 \times 0.99}
= \frac{0.009}{0.0585} \approx 0.154.
$$

A failure that is eighteen times more likely from a broken commit than from a
fine one leaves the commit only about 15% likely to be broken. Counting makes the
reason plain: of 10,000 commits, 100 are broken and 90 of them fail, while
9,900 are fine and 495 of them fail. Of the 585 failures, only 90 come from
broken commits. Fine commits are so much more common that their rare false
alarms outnumber the real alarms.
:::

```{figure}
//| figure: prob-mosaic
//| label: fig-prob-mosaic-rare
//| fig-cap: "The setting of @ex-prob-flaky-test: broken commits are rare (1%), and the figure is conditioned on a failing test. The broken column is a sliver, so its fail region is small next to the fail region of the fine column, even though that one is only 5% of its column. Move the sliders to see how the answer depends on each rate; the rates are illustrative."
base: 0.01
given: fail
```

People reasoning informally tend to give the prior too little weight in
problems like this one, a pattern known as *base-rate neglect*
[@kahneman1973psychology; @barhillel1980base]. The same problems are solved
correctly more often when the numbers are stated as counts out of a population,
as in the last paragraph of @ex-prob-flaky-test, rather than as probabilities
[@gigerenzer1995improve]. A program applying @eq-prob-bayes has no such
difficulty, which is one reason to write the rule down rather than trust
intuition.

The same arithmetic applies to optimization. If good configurations are rare
and a single evaluation is noisy, then a configuration that beats the baseline
once is not, on that evidence alone, likely to be good: lucky draws from the
many mediocre configurations can outnumber the honest results of the few good
ones. A model that tracks both its prior and the noise accounts for this
automatically.

### Updating a belief flip by flip {#sec-prob-coin}

The test example had two possibilities and one observation. Optimization
gathers many observations about an unknown that can take many values. The
smallest problem with that shape is a coin with an unknown bias, the problem
Bayes himself considered.

Let $\theta$ be the probability that the coin lands heads. To keep the
bookkeeping visible, allow eleven possible values,
$\theta \in \{0, 0.1, 0.2, \dots, 1\}$. The belief about $\theta$ is a table of
eleven probabilities, and with no reason to favor any value, the prior puts
$1/11$ on each.

Flip the coin once and see heads. The likelihood of heads under hypothesis
$\theta$ is $\theta$ itself, by @eq-prob-bernoulli. By @eq-prob-proportional
the posterior is proportional to $\theta \times 1/11$, and rescaling divides by
the sum of all eleven products:

$$
p(\theta \given \text{H}) = \frac{\theta / 11}{\sum_{\theta'} \theta' / 11}
= \frac{\theta}{5.5}.
$$

So the hypothesis $\theta = 0$, a coin that never lands heads, now has
probability zero: one head rules it out for good. The hypothesis $\theta = 1$
has risen from $1/11 \approx 0.091$ to $1/5.5 \approx 0.182$. The evidence, the
sum before rescaling, is $\sum_{\theta} (\theta / 11) = 0.5$: before the flip,
the model gave heads a probability of one half, as it should for a flat prior.

The next flip uses the same rule with today's posterior as tomorrow's prior.
If it lands tails, the likelihood is $1 - \theta$, and the new belief is
proportional to $\theta (1 - \theta)$. After $h$ heads and $t$ tails, in any
order, the belief is

$$
p(\theta \given \text{flips}) \propto \theta^h (1 - \theta)^t\, p(\theta).
$$ {#eq-prob-coin-posterior}

::: {.derivation title="Updating one flip at a time equals updating all at once"}
Write the data as two flips $x_1, x_2$, and assume that the flips are
independent once $\theta$ is known (@sec-independence), so that
$p(x_1, x_2 \given \theta) = p(x_1 \given \theta)\, p(x_2 \given \theta)$.

1. Bayes' rule for both flips at once, @eq-prob-proportional, gives
   $p(\theta \given x_1, x_2) \propto p(x_1, x_2 \given \theta)\, p(\theta)$.
2. By the independence assumption, this is
   $p(x_2 \given \theta)\, p(x_1 \given \theta)\, p(\theta)$.
3. By @eq-prob-proportional for the first flip alone,
   $p(x_1 \given \theta)\, p(\theta) \propto p(\theta \given x_1)$.
4. So $p(\theta \given x_1, x_2) \propto p(x_2 \given \theta)\,
   p(\theta \given x_1)$: Bayes' rule for the second flip, with the posterior
   after the first flip as its prior.
5. Multiplication does not depend on order, so the result is the same whichever
   flip is processed first. Repeating the argument for $n$ flips gives
   @eq-prob-coin-posterior.
:::

@fig-prob-coin-update lets you run these updates. Each flip is drawn as one
application of @eq-prob-proportional: the belief before the flip, times the
likelihood of the flip, rescaled.

```{figure}
//| figure: prob-coin-update
//| label: fig-prob-coin-update
//| fig-cap: "Bayes' rule on a grid of eleven hypotheses about a coin's bias. The first panel shows the belief before the most recent flip, the second the likelihood of that flip under each hypothesis, and the third their product (dashed outline), rescaled to sum to one. Call flips yourself, or flip the mystery coin, whose bias is one of 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, and 0.9, hidden until you reveal it. The readouts give the probability that the next flip lands heads and the probability that the bias exceeds one half. The four priors are illustrative."
```

Some things to try:

- **Start over and call heads three times.** Press *New coin* to clear the
  flips. The bar at $\theta = 0$ vanishes after the first head and never comes
  back. A hypothesis that the data rule out stays ruled out, because zero times
  anything is zero.
- **Watch the normalizer.** Before each flip, the readout gives the probability
  that the next flip lands heads, $\sum_\theta \theta\, p(\theta \given
  \text{flips})$. Call the flip, and the line beneath reports the same number
  (or one minus it, for tails) as the amount the product was rescaled by. The
  evidence is the probability the model had given to what happened. A
  surprising flip, one with small evidence, moves the belief the most.
- **Change the prior, keep the flips.** With *Probably fair*, a handful of
  heads barely moves the belief away from 0.5; with *Maybe a trick coin*, a few
  flips of each side quickly eliminate both trick hypotheses. With *Certain it
  is fair*, nothing ever changes: a prior of zero on every other value cannot be
  overturned by any amount of data. A prior should give some probability to
  everything the data might need to reveal.
- **Play against the mystery coin.** Press *New coin*, flip the mystery coin
  ten times, guess its bias, and reveal it. Keep flipping: the belief concentrates around the true value, but
  slowly. A bias of 0.6 takes on the order of a hundred flips to tell apart
  from 0.5 with any confidence; the standard deviation (@sec-prob-variance) of
  the fraction of heads shrinks only like $1/\sqrt{n}$, for reasons
  @sec-independence makes precise.

The coin is not as far from this book's subject as it looks. Replace the flip
by a person asked whether design A is better than design B, and $\theta$ by the
probability that they say yes. Each answer is a Bernoulli observation, and the
update is the same multiply-and-rescale. @sec-part-preferences does this, with
an unknown utility function in place of the single number $\theta$
(@sec-gp-preference). And in @sec-beta-binomial the eleven hypotheses become a
continuum of values in $[0, 1]$ and the table becomes a density; the update is
the same, with integrals in place of sums.

### Computing it {#sec-prob-computing}

A grid of hypotheses is the most direct implementation of Bayes' rule, and two
practical points apply to it and to everything later.

First, products of many probabilities underflow. After a few hundred
observations, a likelihood such as $\theta^h (1 - \theta)^t$ is smaller than
the smallest positive floating-point number and rounds to zero. Implementations
therefore work with logarithms: the log of a product is a sum of logs, and the
rescaling step uses the *log-sum-exp* function,
$\log \sum_i e^{a_i}$, computed stably by factoring out the largest $a_i$.

::: {.code title="NumPy"}
```python
import numpy as np
from scipy.special import logsumexp

theta = np.linspace(0, 1, 11)         # the eleven hypotheses
log_prior = np.full(11, -np.log(11))  # flat prior

def update(log_belief, flip):
    with np.errstate(divide="ignore"):  # allow log(0) = -inf
        log_lik = np.log(theta if flip == "H" else 1 - theta)
    log_post = log_belief + log_lik     # prior times likelihood
    return log_post - logsumexp(log_post)  # divide by evidence

b = log_prior
for f in "HHTH":
    b = update(b, f)
print(np.round(np.exp(b), 3))
# [0.    0.002 0.013 0.038 0.078 0.127 0.176 0.209 0.208 0.148 0.   ]
```
:::

Second, grids do not scale. One unknown on eleven values needs eleven numbers;
$d$ unknowns need $11^d$. A belief about where the best setting of six
hyperparameters lies, with eleven candidate values for each, already needs
$11^6 \approx 1.8 \times 10^6$ entries, and ten hyperparameters need about 26
billion. The objective
of a Bayesian optimizer is an unknown function, with a value at every input, so
no grid can hold the belief about it. The way out is to choose distributions
whose updates have a closed form, so that the whole table is summarized by a
few numbers that Bayes' rule changes in a predictable way. The Gaussian
distribution of @sec-gaussian is the central example, and @sec-approx-inference
turns to approximations when no closed form exists.

## Expectation and variance {#sec-expectation}

A distribution is a whole table or curve. Decisions usually need a few numbers
from it: what value should we expect, and how sure are we? Every acquisition
function in @sec-part-bo is a summary of this kind; expected improvement, for
instance, is literally an expectation (@sec-ei).

### Expectation {#sec-prob-expectation-def}

::: {.definition #def-prob-expectation title="Expectation"}
The expectation, or mean, of a random variable $X$ is the average of its
values weighted by their probabilities:

$$
\E[X] = \sum_x x\, p(x)
\quad \text{(discrete)}, \qquad
\E[X] = \int x\, p(x)\, \dd x
\quad \text{(continuous)}.
$$
:::

The expectation is the center of mass of the belief: if the probabilities were
weights placed along a ruler, the ruler would balance at $\E[X]$. A fair die
has expectation $(1 + 2 + \dots + 6)/6 = 3.5$, which is not a value the die can
show; an expectation need not be a possible outcome. A Bernoulli variable has
expectation $1 \cdot \theta + 0 \cdot (1 - \theta) = \theta$. In
@fig-prob-coin-update, the probability that the next flip lands heads is the
expectation of $\theta$ under the current belief.

Often we need the expectation of a quantity computed from $X$ rather than of
$X$ itself. There is no need to work out the distribution of that quantity
first; weight its values directly:

$$
\E[g(X)] = \sum_x g(x)\, p(x).
$$ {#eq-prob-lotus}

In Bayesian optimization, $X$ is the unknown value of the objective at a
candidate input, and $g$ might be the improvement over the best value so far,
$\max(X - b, 0)$. Its expectation under the model's belief is the expected
improvement.

The most useful property of expectation is that it passes through sums and
constant multiples. For any random variables $X$ and $Y$ and constants $a$,
$b$, $c$,

$$
\E[aX + bY + c] = a\,\E[X] + b\,\E[Y] + c.
$$ {#eq-prob-linearity}

This *linearity* holds whether or not $X$ and $Y$ are related, which makes it
useful far beyond sums of unrelated quantities. The binomial count of @eq-prob-binomial is a sum of $n$
Bernoulli variables, one per flip, so its expectation is $n\theta$ without any
binomial coefficients.

::: {.pitfall title="The expectation of a function is not the function of the expectation"}
Linearity holds for sums and constant multiples only. For other functions,
$\E[g(X)]$ and $g(\E[X])$ differ in general. Expected improvement shows why the
difference matters. Suppose the model's belief about the objective at some
input has mean exactly equal to the best value found so far, $b$. Then the
improvement of the mean is $\max(\E[X] - b, 0) = 0$, but the expected
improvement $\E[\max(X - b, 0)]$ is positive, because the belief gives weight to
values above $b$ and the function ignores values below it. For a Gaussian
belief (@sec-gaussian) with standard deviation $s$ it is about $0.4\,s$
(@sec-ei). The gap is
the value of uncertainty, and it is what makes an optimizer explore.
:::

### Variance {#sec-prob-variance}

The expectation says where a belief is centered; the variance says how widely
it is spread.

::: {.definition #def-prob-variance title="Variance and standard deviation"}
The variance of $X$, with mean $\mu = \E[X]$, is the expected squared distance
from the mean,

$$
\Var[X] = \E\big[(X - \mu)^2\big] = \E[X^2] - \mu^2.
$$ {#eq-prob-variance}

The standard deviation is $\sqrt{\Var[X]}$, which has the same units as $X$.
:::

The second form of @eq-prob-variance follows from linearity: expanding the
square, $\E[X^2 - 2\mu X + \mu^2] = \E[X^2] - 2\mu\,\E[X] + \mu^2 = \E[X^2] -
\mu^2$. A Bernoulli variable has $\E[X^2] = \theta$ (since $X^2 = X$ when $X$ is
0 or 1), so its variance is $\theta - \theta^2 = \theta(1 - \theta)$, largest
for a fair coin and zero for a coin that always lands the same way. Shifting a
variable does not change its spread and scaling it scales the spread:
$\Var[aX + c] = a^2 \Var[X]$.

### Covariance {#sec-prob-covariance}

With two random variables we can ask whether they move together. The
*covariance*

$$
\Cov[X, Y] = \E\big[(X - \E[X])(Y - \E[Y])\big]
$$

is positive when $X$ tends to be above its mean whenever $Y$ is above its mean,
negative when they move in opposite directions, and zero when there is no such
linear tendency. Dividing by both standard deviations gives the *correlation*,
a number between $-1$ and $1$ that does not depend on units.

Covariance is what makes the variance of a sum differ from the sum of the
variances:

$$
\Var[X + Y] = \Var[X] + \Var[Y] + 2\,\Cov[X, Y].
$$ {#eq-prob-var-sum}

(Expand the square inside the expectation and apply linearity.) With many
variables, the covariances of all pairs form a table, the *covariance matrix*,
which is the central object of the next three chapters. @sec-positive-definite
explains which tables can be covariance matrices, and a Gaussian process kernel
is a rule for filling in such a table: $k(x, x')$ is the covariance between the
objective's values at $x$ and $x'$ (@sec-gp-definition). Two nearby inputs have
a large covariance, so observing one moves the belief about the other.

### Averaging over what you do not know {#sec-prob-total-variance}

The product rule let us build a joint distribution from a marginal and a
conditional. Expectation and variance can be built the same way, in two
stages. The *law of total expectation* says that the overall mean is the
average of the conditional means:

$$
\E[X] = \E\big[\E[X \given Y]\big].
$$ {#eq-prob-total-expectation}

Here $\E[X \given Y]$ is the mean of $X$ under the conditional distribution
$p(x \given y)$, a number that depends on $y$; the outer expectation averages
it over $Y$. (For discrete variables: $\sum_y p(y) \sum_x x\, p(x \given y) =
\sum_x x \sum_y p(x, y) = \sum_x x\, p(x)$, by the product rule and then the
sum rule.)

The variance version is more interesting, because it splits uncertainty into
two kinds. Picture one noisy evaluation of an objective. Part of its spread
would remain even if we knew the objective's value exactly: that is the noise.
The rest comes from not knowing that value. The law of total variance says
that the two parts add.

::: {.derivation title="The law of total variance"}
Write $m(Y) = \E[X \given Y]$ for the conditional mean.

1. By @eq-prob-variance, $\Var[X] = \E[X^2] - (\E[X])^2$.
2. By @eq-prob-total-expectation applied to $X^2$,
   $\E[X^2] = \E\big[\E[X^2 \given Y]\big]$.
3. By @eq-prob-variance applied to the conditional distribution,
   $\E[X^2 \given Y] = \Var[X \given Y] + m(Y)^2$.
4. Combining steps 2 and 3, $\E[X^2] = \E\big[\Var[X \given Y]\big] +
   \E\big[m(Y)^2\big]$.
5. By @eq-prob-total-expectation, $\E[X] = \E[m(Y)]$, so
   $(\E[X])^2 = (\E[m(Y)])^2$.
6. Substitute steps 4 and 5 into step 1. The two terms
   $\E\big[m(Y)^2\big] - (\E[m(Y)])^2$ form the variance of $m(Y)$ by
   @eq-prob-variance, which leaves the result below.
:::

$$
\Var[X] = \E\big[\Var[X \given Y]\big] + \Var\big[\E[X \given Y]\big].
$$ {#eq-prob-total-variance}

Now apply it to that noisy evaluation. Let $Y$ be the unknown value of the
objective at some input, $f$, and let $X$ be the result of the next noisy
evaluation there, $f$ plus independent noise of variance $\sigma_n^2$. Given
$f$, the evaluation has mean $f$ and variance $\sigma_n^2$. Then
@eq-prob-total-variance says

$$
\Var[\text{evaluation}] = \sigma_n^2 + \Var[f].
$$

The first term is noise: no amount of modeling removes it, and repeating the
evaluation would give a different number every time. The second term is the
model's own uncertainty about $f$, which evaluations reduce. The two are often
called *aleatoric* (from chance) and *epistemic* (from lack of knowledge)
uncertainty. @sec-gp-noise meets them as the difference between the posterior
variance of $f$ and the variance of a new measurement, and acquisition
functions target the second kind: there is no point in exploring to reduce
noise.

### Expectations by sampling {#sec-prob-monte-carlo}

When the sum or integral in an expectation has no closed form, the standard
remedy is to draw samples $x^{(1)}, \dots, x^{(S)}$ from $p(x)$ and average:
$\E[g(X)] \approx \frac{1}{S} \sum_{s} g(x^{(s)})$. This *Monte Carlo* estimate
is correct on average (unbiased), and its standard deviation shrinks like
$1/\sqrt{S}$, for the reason given in @sec-independence. Libraries such as BoTorch compute many
acquisition functions this way, by averaging over samples from the model's
posterior [@balandat2020botorch].

## Independence {#sec-independence}

Several steps in this chapter quietly assumed that observations do not
influence each other: the binomial formula, the sequential coin update, the
averaging of Monte Carlo samples. This section states the assumption, shows
where it fails, and explains why noise in Bayesian optimization is usually
modeled as independent.

::: {.definition #def-prob-independence title="Independence"}
Random variables $X$ and $Y$ are independent if $p(x, y) = p(x)\, p(y)$ for all
$x$ and $y$. Equivalently, whenever $p(x) > 0$, $p(y \given x) = p(y)$:
learning $X$ does not change the belief about $Y$.
:::

In the mosaic of @fig-prob-mosaic, independence means that both columns are cut
at the same height: the test fails equally often on broken and fine commits.
Set both sliders to 0.5 and condition on fail; the probability of a broken
commit does not move from its prior, because the test carries no information.

Independent variables have zero covariance, so by @eq-prob-var-sum the
variance of a sum of independent variables is the sum of their variances. The
converse is false. If $X$ is $-1$, $0$, or $1$ with equal probability and
$Y = X^2$, the covariance is zero, yet $Y$ is completely determined by $X$.
Covariance detects only linear relationships.

### Conditional independence {#sec-prob-conditional-independence}

The coin of @sec-prob-coin holds a subtlety that the rest of the book depends
on. Are two flips of a coin with unknown bias independent? Under the flat prior
on eleven values, the first flip lands heads with probability 0.5. If it does,
the belief about $\theta$ shifts toward heads, and the probability that the
second flip also lands heads becomes

$$
\Prob(\text{H}_2 \given \text{H}_1) = \sum_\theta \theta\, \frac{\theta}{5.5}
= \frac{3.85}{5.5} = 0.7.
$$

So $\Prob(\text{H}_1, \text{H}_2) = 0.5 \times 0.7 = 0.35$, not
$0.5 \times 0.5 = 0.25$. The flips are *not* independent: each one carries
information about the bias, and so about the next. You can watch this in
@fig-prob-coin-update, where the probability of heads on the next flip changes
with every flip.

Yet if the bias were known, say $\theta = 0.3$, each flip would land heads with
probability 0.3 regardless of the others. The flips are independent *given*
$\theta$.

::: {.definition #def-prob-conditional-independence title="Conditional independence"}
$X$ and $Y$ are conditionally independent given $Z$ if
$p(x, y \given z) = p(x \given z)\, p(y \given z)$ for all $x$, $y$, and $z$
with $p(z) > 0$.
:::

This is the structure of every model in the book. Observations are
conditionally independent given the unknown, which makes the likelihood a
product that is easy to write down. They are dependent once the unknown is
marginalized out, because they share it, and that dependence is exactly how one
observation informs predictions about another. A Gaussian process works the
same way, with a whole function in place of $\theta$.

Observations that are independent given the unknown and that all follow the
same distribution are called *independent and identically distributed*, or
i.i.d. Their likelihood is a product of identical factors,

$$
p(\D \given \theta) = \prod_{i=1}^n p(x_i \given \theta),
$$ {#eq-prob-iid}

which is how @eq-prob-coin-posterior arose.

### Why noise is modeled as independent {#sec-prob-independent-noise}

The standard observation model of Bayesian optimization, introduced in
@sec-gp-noise, writes each evaluation as the objective plus noise,
$y_i = f(x_i) + \varepsilon_i$, with the noise terms $\varepsilon_i$
independent of each other and of $f$. There are three reasons for this choice.

The first is that it is often physically plausible. Each training run draws a
fresh random seed; each stride of a person walking in an exoskeleton is a new
stride. Nothing ties the noise in one evaluation to the noise in the next. The
exoskeleton case also shows how much noise a method may have to absorb.
@ding2018human used Bayesian optimization to tune the peak and offset timing of
the hip assistance given by a soft exosuit, scoring each setting by the
walker's measured energy expenditure, a signal whose low signal-to-noise ratio
they name as a central practical challenge. The settings found, in an average of
21.4 ± 1.0 minutes, reduced metabolic cost by 17.4 ± 3.2% (mean ± standard
error) compared with walking without the device. @sec-cs-exoskeleton works
through this problem in detail.

The second is convenience. The likelihood becomes a product, as in
@eq-prob-iid, and with Gaussian noise the noise covariance matrix is diagonal,
which keeps the computations of @sec-gp-computation simple.

The third is that independence is what makes averaging work. If
$\varepsilon_1, \dots, \varepsilon_n$ are independent with variance $\sigma^2$
each, then by @eq-prob-var-sum (with every covariance zero) their sum has
variance $n\sigma^2$, and their mean, which divides the sum by $n$, has variance
$n\sigma^2 / n^2 = \sigma^2 / n$. The standard deviation of an average falls
like $1/\sqrt{n}$: four times as many evaluations halve the noise. This is the
rate at which the coin's belief concentrates and the rate at which a Monte Carlo
estimate converges.

::: {.pitfall title="Correlated errors counted as independent"}
When errors share a cause, treating them as independent makes a model
overconfident. Suppose $n$ evaluations share a common offset $c$ with variance
$\sigma_c^2$, such as a miscalibrated sensor or a person who is in a generous
mood for the whole session, on top of independent noise with variance
$\sigma^2$. Their mean has variance $\sigma_c^2 + \sigma^2 / n$, not
$\sigma^2 / n$: repetition
removes the independent part and leaves the shared offset untouched. A model
that assumes independence counts $n$ correlated measurements as $n$ separate
pieces of evidence and becomes more certain than the data justify. Evaluations
run in the same batch, and a person's answers within a session that drift or
anchor on the previous question, are the usual suspects. The book returns to
drifting human judgments in @sec-hci-unstable and @sec-theory-drift.
:::

## Exercises {#sec-prob-exercises}

::: {.exercise #exr-prob-two-failures}
Use the setting of @ex-prob-flaky-test: 1% of commits are broken, the test
fails on 90% of broken commits and on 5% of fine ones. A commit fails, the test
is run again, and it fails again. Assuming the two runs are conditionally
independent given the commit's state, compute
$\Prob(\text{broken} \given \text{two failures})$ in two ways: all at once,
and one failure at a time with the first posterior as the second prior. Then
explain why the assumption might fail for a flaky test.

::: {.solution}
All at once: the likelihoods are $0.9^2 = 0.81$ for a broken commit and
$0.05^2 = 0.0025$ for a fine one, so

$$
\Prob(\text{broken} \given \text{two failures})
= \frac{0.81 \times 0.01}{0.81 \times 0.01 + 0.0025 \times 0.99}
= \frac{0.0081}{0.010575} \approx 0.766.
$$

One at a time: the first failure gives $0.154$, as in @ex-prob-flaky-test.
Using it as the prior for the second failure,
$\frac{0.9 \times 0.154}{0.9 \times 0.154 + 0.05 \times 0.846} \approx 0.766$,
the same answer up to rounding, as the derivation in @sec-prob-coin promises.

The assumption fails if whatever made the test fail on a fine commit tends to
persist, for example a test that depends on a slow external service or on a
timing quirk of that particular change. Then
$\Prob(\text{second failure} \given \text{first failure}, \text{fine})$ is much
larger than 0.05, the second failure carries far less information, and the true
posterior is lower than 0.766.
:::
:::

::: {.exercise #exr-prob-three-coins}
A coin's bias is one of three values, $\theta \in \{0.25, 0.5, 0.75\}$, each
with prior probability $1/3$. (a) Find the posterior after one head. (b) Find
the probability that the second flip lands heads given that the first did, and
compare it with the probability that the first flip lands heads. (c) Find the
posterior after two heads.

::: {.solution}
(a) The posterior is proportional to $\theta \times 1/3$. The products are
$1/12, 2/12, 3/12$, summing to $1/2$, so the posterior is $1/6, 1/3, 1/2$.

(b) The first flip lands heads with probability $\frac{1}{3}(0.25 + 0.5 +
0.75) = 0.5$. After one head, the second lands heads with probability
$0.25 \cdot \frac{1}{6} + 0.5 \cdot \frac{1}{3} + 0.75 \cdot \frac{1}{2} =
0.583$. The flips are dependent, because they share the unknown bias, even
though they are independent given $\theta$.

(c) The posterior is proportional to $\theta^2$: $0.0625, 0.25, 0.5625$, summing
to $0.875$, so the posterior is about $0.071, 0.286, 0.643$. Applying the
update of (a) a second time gives the same result.
:::
:::

::: {.exercise #exr-prob-uniform}
Let $X$ be uniform on $[0, 0.25]$. What is its density? What is
$\Prob(0.1 \le X \le 0.2)$? Compute $\E[X]$ and $\Var[X]$.

::: {.solution}
The density is $1/0.25 = 4$ on the interval, a density larger than one. The
probability of the subinterval is its length times the density,
$0.1 \times 4 = 0.4$. The mean is the midpoint, $\E[X] = 0.125$, and
$\E[X^2] = \int_0^{0.25} 4x^2\, \dd x = \frac{4}{3} (0.25)^3 \approx 0.02083$,
so by @eq-prob-variance $\Var[X] = 0.02083 - 0.125^2 \approx 0.00521$, which is
$0.25^2 / 12$. The standard deviation is about $0.072$.
:::
:::

::: {.exercise #exr-prob-noise-split}
A model's belief about the objective at an input has mean $0.7$ and standard
deviation $0.1$. Each evaluation adds independent noise with standard
deviation $0.05$. What is the standard deviation of the next evaluation's
result? If the input is evaluated so many times that the model becomes certain
of $f$ there, what does it become?

::: {.solution}
By @eq-prob-total-variance, the variance is $0.05^2 + 0.1^2 = 0.0125$, so the
standard deviation is about $0.112$. Once the model is certain of $f$, the
second term vanishes and the standard deviation of a new evaluation is the noise
alone, $0.05$. Repeated evaluations remove the epistemic part of the
uncertainty, never the aleatoric part.
:::
:::

## Further reading {#further-reading .unnumbered}

- @blitzstein2019introduction, chapters 1 to 7, covers everything in this
  chapter in depth, with many worked problems and a gift for explaining
  conditional probability through stories.
- @mackay2003information, chapters 2 and 3, introduces probability and
  inference from the belief viewpoint, with coin and dice examples close to the
  ones here. The book is free to read online.
- @bishop2006pattern, section 1.2, presents the sum rule, the product rule,
  Bayes' rule, and expectations in the notation that machine learning papers
  use.
- @jaynes2003probability develops probability as an extension of logic,
  starting from the consistency argument of @cox1946probability; chapter 2
  derives the sum and product rules from it.
- @gigerenzer1995improve shows how the format in which numbers are presented
  changes how well people reason with Bayes' rule.
