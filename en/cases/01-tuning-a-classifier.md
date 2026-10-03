---
status: done
synopsis: "Bayesian optimization on measured outcomes: the validation-error landscape of a support vector machine over two hyperparameters, which you search yourself before grid, random, and Bayesian search replay it, then recorded searches over seven hyperparameters of a gradient-boosted tree model, with costs, noise, and held-out scores."
sources: ["real data: scikit-learn digits (UCI optdigits) and Spambase, measured with tools/figure-data/", "Hsu, Chang, and Lin 2003", "Snoek et al. 2012", "Bergstra and Bengio 2012"]
---

# Tuning a Classifier {#sec-cs-classifier}

The first four parts of this book worked on test functions: formulas chosen so
that one idea at a time would be easy to see. A formula has no cost, its noise
is whatever we add, and its domain is whatever box we draw. This part keeps the
details that formulas leave out. In this chapter the objective is the
validation error of a real classifier on real data, and every value an
optimizer sees in the figures was measured by training and scoring a model.

Tuning the settings of a machine learning model is the application that made
Bayesian optimization popular in machine learning [@snoek2012practical], and
it is the first of the four problems in @sec-cost-of-evaluation. It is also
where the strongest simple competitors live: a grid of settings recommended by
a practical guide, random search [@bergstra2012random], and the default values
a library ships with. A case study has to take all three seriously.

The chapter has two problems. The first has only two settings to choose, which
is few enough that we measured every combination on a fine lattice. That lets
you search the landscape yourself before anyone tells you its shape, then
watch three strategies search it with the same budget, and finally see all of
it. The second problem has seven settings and could not be measured
exhaustively, so we ran the searches offline, recorded every evaluation, and
replay them. The results are less one-sided than the test functions of
@fig-intro-race suggest, and the reasons are most of what the chapter teaches.

## The problem {#sec-cs-classifier-problem}

### Digits and a support vector machine {#sec-cs-classifier-svm}

The data are 1,797 handwritten digits, each a scanned image reduced to an
8 × 8 grid of ink counts from 0 to 16 [@alpaydin1998optical]. The task is to
say which digit, 0 to 9, an image shows. The set ships with scikit-learn
[@pedregosa2011scikit], so anyone can rerun every number in this chapter; we
scale each of the 64 pixel values to the range from 0 to 1, as the practical
guide we follow below recommends [@hsu2003practical].

The classifier is a **support vector machine** (SVM) with a radial basis
function kernel [@cortes1995support]. For this chapter it is enough to know
what it computes. To label a new image, it takes a weighted vote of training
images, and each training image's say is scaled by how similar it is to the
new one:

$$
k(\vx, \vx') = \exp\!\left(-\gamma\, \lVert \vx - \vx' \rVert^2\right).
$$ {#eq-cs-classifier-rbf}

This is the RBF kernel of @sec-function-space, written with
$\gamma = 1/(2\ell^2)$ in place of the lengthscale $\ell$. (With ten classes,
scikit-learn trains one such voter for each pair of digits and lets the
voters vote.) Training chooses the weights of the vote. Two numbers have to be
fixed before training starts, and training does not learn them:

- $\gamma$, the **kernel width**, sets how quickly similarity falls off with
  distance. A large $\gamma$ means only near-identical images count as
  similar.
- $C$, the **penalty**, sets how much training is charged for each training
  image that ends up on the wrong side of the decision boundary, or too close
  to it. A large $C$ makes training fit the training images closely; a small
  $C$ tolerates mistakes in exchange for a simpler boundary.

Settings of this kind are called **hyperparameters**, and choosing them is
**hyperparameter tuning**. The word is the same one @sec-kernels uses for a
kernel's lengthscale, and for the same reason: a quantity that shapes the
model but is not fitted along with the model's other parameters.

What $\gamma$ means becomes concrete with distances measured on these images.
After scaling, the median squared distance from an image to its nearest
neighbor is 1.05, to another image of the same digit 5.1, and to an image of a
different digit 9.9. At $\gamma = 2^{-2}$, @eq-cs-classifier-rbf turns these
into similarities of 0.77, 0.28, and 0.085: neighbors count a lot, other
examples of the same digit somewhat, other digits barely. At $\gamma = 2^{3}$
even the nearest neighbor has similarity 0.0002, so each training image is an
island and the classifier can do little more than memorize. At
$\gamma = 2^{-15}$ two typical images of different digits have similarity
0.9997, and the kernel can hardly tell images apart.

### Measuring an error {#sec-cs-classifier-cv}

A classifier is judged on images it was not trained on. The fraction of such
images it gets wrong is its **validation error**. With only 1,797 images we
cannot afford to set many aside, so we use **5-fold cross-validation**: split
the images at random into five parts of nearly equal size, each with the same
mix of digits, train on four, count the mistakes
on the fifth, rotate through all five, and divide the total number of mistakes
by 1,797. Every image is predicted exactly once, by a model that never saw it.
One evaluation of a setting is therefore five trainings.

On one CPU core an evaluation takes about 0.39 seconds. That is cheap enough
that we could measure the whole landscape: every setting on the lattice
described below, each under five different random splits, 37,925 trainings in
about four minutes on 16 cores. This is unusual. Training and validating one
configuration of a large model "often takes hours, days, or even weeks"
[@klein2017fast], and then nobody measures the whole landscape. The
cheapness is what lets this chapter show you the answer; it is not a property
of the method, and the figures treat each evaluation as if it were expensive.

Cross-validation is noisy because the split is random. The five folds of one
evaluation disagree with each other by a standard deviation of 0.65
percentage points where the error is below 3%. The average over five folds is
steadier: repeating the whole evaluation with a different split moves it by
a standard deviation of 0.16 points. That second number is the noise an
optimizer sees, and it is not small. The best setting on the lattice
misclassified 20, 14, 19, 15, and 15 images under the five splits we measured,
that is, between 0.78% and 1.11%.

### The search space {#sec-cs-classifier-space}

Before the first evaluation, someone has to decide where to look. A widely
used practical guide to SVMs recommends trying "exponentially growing
sequences" of the two hyperparameters, $C = 2^{-5}, 2^{-3}, \dots, 2^{15}$ and
$\gamma = 2^{-15}, 2^{-13}, \dots, 2^{3}$, then refining the grid around the
best cell [@hsu2003practical]. We adopt its ranges. They span 20 powers of two
in $C$ and 18 in $\gamma$, which is the first decision a practitioner makes:
these hyperparameters act by multiplication, so the search is done in
$\log_2 C$ and $\log_2 \gamma$, where a step means "twice as large" anywhere
in the range.

We measured the landscape on a lattice twice as fine as the guide's in each
direction, every half power of two: 41 values of $\log_2 C$ times 37 values of
$\log_2 \gamma$, 1,517 settings. The guide's own coarse grid of
$11 \times 10 = 110$ settings is a subset of it.

In the language of @sec-bo-problem, the input is
$\vx = (\log_2 C, \log_2 \gamma)$ in the box $[-5, 15] \times [-15, 3]$; the
objective $f(\vx)$ is the cross-validation error we would get on average over
random splits; and an evaluation returns $y = f(\vx) + \varepsilon$, where the
noise $\varepsilon$ comes from the split. We want the setting with the lowest
$f$. The rest of the book maximizes, so the Gaussian process below models the
negative logarithm of the error, which is largest where the error is smallest.

## The landscape we cannot see {#sec-cs-classifier-landscape}

You now know what the guide knows: the two ranges and nothing else. The
figure below hides the 1,517 measured errors. You have 15 evaluations. Click a
cell and the figure trains nothing; it looks up one of that setting's five
measured cross-validation runs and shows you the result, which is exactly what
you would have seen had you run it. Try to find the lowest error you can.

```{figure}
//| figure: cs-classifier-landscape
//| label: fig-cs-classifier-game
//| fig-cap: "Search a real landscape yourself. The grid is $\log_2 C$ (horizontal) by $\log_2 \gamma$ (vertical) for a support vector machine on 1,797 handwritten digits; a click evaluates one setting and colors its cell by the measured 5-fold cross-validation error (the stronger the color, the lower the error). When your budget is spent, or after *Skip to the race*, the timeline replays grid search (squares), random search (diamonds), and Bayesian optimization (circles) with the same budget, and the right panel scores the setting each of you would pick. Every value is measured, none is simulated: each evaluation returns one of five cross-validation runs with different random splits that we computed for every setting (tools/figure-data/cs-classifier-svm.py)."
```

A few things are worth trying before reading on.

**Spend your budget before revealing anything.** Count how many of your
evaluations came back above 50%. It is easy to lose several to a region you
could not have known was bad.

**Click the same cell twice.** The second value usually differs from the
first, because the second visit reads a different measured split. That is the noise
of @sec-cs-classifier-cv, and it means your best value so far is partly luck.

**Press play when the budget is spent.** The curve marked *you* on the right
is your own search, scored the same way as the three strategies. Then turn on
*Reveal the landscape*.

The revealed landscape has four parts, and each has a reason.

1. **A plateau at the lower left.** With small $C$ and small $\gamma$ the
   error is 84%, close to the 90% of guessing. The kernel sees all images as
   nearly the same, and the small penalty lets training give up on fitting
   them.
2. **A wall at the top.** From $\gamma = 2^{1}$ upward the error is at least
   11% for every $C$, and at $\gamma = 2^{3}$ it is between 78% and 89%.
   These are the islands of the previous section: the classifier memorizes
   each training image and has nothing to say about a new one.
3. **A broad shelf at the right.** For large $C$ and small $\gamma$ the error
   settles between 1.6% and 2.0%, rising to about 5% toward the edge of the
   plateau. As $\gamma$ shrinks with $C$ growing in
   proportion to $1/\gamma$, an RBF-kernel SVM behaves like a linear
   classifier [@keerthi2003asymptotic]. The same scaling explains why the edge
   of the plateau runs diagonally: along it, $C\gamma$ is roughly constant.
4. **A ridge.** For $\gamma$ between $2^{-4.5}$ and $2^{-1.5}$, and $C$ large
   enough (from $2^{-0.5}$ in the middle of the ridge to $2^{2}$ at its lower
   edge), the error is below 1.1%: 181 settings in all. The best setting,
   $C = 2^{0}$ and $\gamma = 2^{-2}$, has an average error of 0.92%, about
   16.6 of 1,797 images.

Two facts about this picture shape everything that follows. First, much of the
box is good: 687 of the 1,517 settings (45%) have an error below 2%. The prize
for searching well is the step from the shelf's 1.8% to the ridge's 0.92%,
which halves the number of mistakes but is less than one percentage point.
Second, in the good half of the box the error barely depends on $C$. At
$\gamma = 2^{-3}$, every $C$ from $2^{3}$ to $2^{15}$, a factor of 4,096,
gives the same error, 0.97%: once $C$ is large enough that training fits every
training image, raising it further changes nothing. Raising $\gamma$ by a
factor of four from the best setting, by contrast, more than doubles the
error, from 0.92% to 2.1%, and lowering it by the same factor raises the
error to 1.5% at the same $C$. This is what @bergstra2012random call low effective dimensionality: of
the two hyperparameters, mostly one matters here.

::: {.keyidea title="Most of the box is good; the last point is the hard part"}
On this landscape a careless search finds 2% error, and a careful one finds
1%. Every difference between strategies in the next section lives in that one
percentage point, and the evaluation noise is a sixth of it.
:::

## Grid, random, and Bayesian optimization {#sec-cs-classifier-race}

The three strategies in @fig-cs-classifier-game get the same budget and the
same measured values. Their rules are the ones a practitioner would use.

**Grid search** evaluates an $a \times b$ grid with $a$ values of $C$ and
$b$ values of $\gamma$, choosing the largest grid that fits the budget with
$a \ge b$ and $a - b \le 2$ (the guide also gives $C$ more values). Each range
is divided evenly including both ends, as in the guide, and the grid is
evaluated one value of $C$ at a time.

**Random search** draws settings uniformly from the 1,517 lattice points,
which is uniform in $\log_2 C$ and $\log_2 \gamma$.

**Bayesian optimization** starts from the same three settings random search
evaluates first, then runs the loop of @sec-bo-algorithm. The surrogate is a
Gaussian process on $-\ln y$, standardized to mean zero and standard deviation
one (@sec-gp-pitfalls), with a Matérn 5/2 kernel that has a separate
lengthscale for each axis (@sec-ard), the combination
@snoek2012practical recommended for hyperparameter tuning. The two
lengthscales, chosen from 0.06, 0.1, 0.16, 0.25, and 0.4 of each axis, and the
noise variance, chosen from 0.001, 0.01, and 0.05, are the combination with
the highest marginal likelihood (@sec-marginal-likelihood). The acquisition
function is expected improvement (@sec-ei) over the highest posterior mean at
an evaluated setting, the usual choice under noise (@sec-practice-noise), and
it is maximized over all 1,517 settings.

The logarithm deserves a sentence. The errors span two orders of magnitude,
from 0.9% to 89%. On the raw scale, the jump from the plateau to the shelf
would dominate everything the Gaussian process learns, and the difference
between 1.8% and 0.9% would look like rounding. On the log scale, halving the
error is the same step wherever it happens.

All three strategies recommend, at every step, the setting with the lowest
value they have measured so far. The race panel does not plot that measured
value. It plots the average of all five measured splits for the recommended
setting, the best estimate of its error the data allow, because the measured
value of a winner is biased low (@exr-cs-classifier-curse). The figure below
replays the race with a budget of 30 over the revealed landscape.

```{figure}
//| figure: cs-classifier-landscape
//| label: fig-cs-classifier-race
//| fig-cap: "The race with a budget of 30 on the revealed landscape. Grid search (squares) evaluates a 6 × 5 grid; random search (diamonds) and Bayesian optimization (circles) share their first three settings; the star marks the best setting. Right: the five-split average error of each strategy's current pick; values above 2.6% are pinned to the top edge. Press play to watch, *New random run* for other random draws and other measured splits, and change the budget to see the grid move. All errors are measured; the strategies' runs are replays on those measurements."
budget: 30
skip: true
reveal: true
```

One run shows one draw of luck. @tbl-cs-classifier-race summarizes 100 runs
of each strategy at four budgets, computed by the same code as the figure
(tools/figure-data/cs-classifier-race.ts).

::: {.table #tbl-cs-classifier-race title="Grid, random, and Bayesian search on the measured landscape, 100 runs each. Grid, Random, BO: median over runs of the five-split average error of the recommended setting. Columns marked < 1.0%: share of runs whose recommendation has an average error below 1.0%. The grid's settings are fixed; only the measured splits vary between its runs. The best setting has 0.92%."}
| Budget | Grid (shape) | Random | Random < 1.0% | BO | BO < 1.0% |
|---|---|---|---|---|---|
| 12 | 1.50% (4 × 3) | 1.02% | 33% | 1.01% | 43% |
| 15 | 1.31% (5 × 3) | 1.01% | 38% | 1.01% | 48% |
| 20 | 0.97% (5 × 4) | 1.01% | 42% | 0.97% | 53% |
| 30 | 1.02% (6 × 5) | 1.01% | 47% | 0.97% | 61% |
:::

Three readings of the table matter.

**A grid is as good as its alignment.** The $5 \times 3$ grid tests
$\gamma = 2^{-15}, 2^{-6}, 2^{3}$ and steps over the ridge entirely; the
$5 \times 4$ grid happens to test $\gamma = 2^{-3}$, lands on the ridge, and
beats the larger $6 \times 5$ grid. Nothing in advance tells you which grid
will be lucky, because the ridge's position is what you are searching for.
Random search tests a new value of $\gamma$ almost every time: 15 random
settings include about 12.5 distinct values of $\gamma$, while the
$5 \times 3$ grid spends 15 evaluations on 3. That is the argument of
@bergstra2012random, made with measured data. The guide's full coarse grid of
110 settings reaches 0.93%, the price of being systematic.

**Random search and Bayesian optimization are close here.** Their medians
differ by at most 0.04 points. On a landscape where 45% of settings are within
a factor of two of the best, a random draw lands somewhere decent almost
immediately, and both strategies get within a tenth of a point of the best in
15 to 30 evaluations. Bayesian optimization's advantage is in how often it
gets the last tenth: below 1.0% in 61% of runs at a budget of 30, against 47%
for random search, and its 90th-percentile run ends at 1.04% against 1.08%.
At budgets of 12 and 15 its worst runs are no better than random search's:
the 90th percentiles are 1.32% against 1.31% and 1.25% against 1.18%.

**Noise sets a floor.** Ninety-eight settings have an average error within 0.1
points of the best, and one evaluation's noise has a standard deviation of
0.16 points. With one measurement per setting, no strategy can reliably tell
those 98 apart. Past about 1.0%, a further evaluation of a new setting buys
less than a second measurement of the current favorites, which is the choice
@sec-practice-noise discusses (inference).

What the Gaussian process believes along the way can be shown directly. The
next figure replaces the landscape with the model's prediction of the error
after 12 evaluations of a run, and circles the setting expected improvement
picks next.

```{figure}
//| figure: cs-classifier-landscape
//| label: fig-cs-classifier-model
//| fig-cap: "Bayesian optimization's model of the landscape after 12 evaluations: the colors are the Gaussian process's posterior mean of the error, a prediction, not a measurement; the orange ring is the next query, where expected improvement is largest. Step the timeline to watch the model form; turn off *Show BO's model* and turn on *Reveal the landscape* to compare it with the measured errors."
budget: 30
skip: true
model: true
t: 12
```

After a dozen evaluations the model has the plateau, the shelf, and roughly
where the ridge is, while its details are wrong in every region it has not
visited. Its queries are drawn to the edges. Over 100 runs with a budget of
30, 37% of Bayesian optimization's queries after its random start lay on the
edge of the lattice, which holds 10% of the settings, and 98 of the 100 runs
queried at least one corner. Far from all data a stationary Gaussian process
returns to its prior, with the largest uncertainty in the box, and the
corners are farthest from everything. This over-exploration of the boundary
"is typically observed" in Bayesian optimization, although in tuning
problems the best setting rarely lies on the boundary
[@siivola2018correcting]. On this landscape the corners are bad, and those
evaluations are the price of the method's caution.

::: {.pitfall title="The best measured value is optimistic"}
When many settings are each measured once with noise, the lowest measurement
belongs disproportionately to settings whose noise happened to be favorable.
The winner's measured error is therefore biased low, and the more settings you
try, the larger the bias. @cawley2010over showed that this kind of
overfitting of the selection criterion can be as large as the differences
between learning algorithms that a study set out to compare. Report the error
of a chosen setting from fresh splits, or better from a test set that played
no part in the choice.
:::

Taken together, the two-dimensional race delivers a finding that a test
function would have hidden. On a real landscape with two hyperparameters,
much of it good and all of it measured with noise, Bayesian optimization and
random search end up with nearly the same median error; what the model buys is
reliability, a larger share of runs that reach the best tenth of a point. That
is not a failure of the method. It is
what the method should be expected to deliver when random draws are already
likely to land somewhere good, and the next section shows where its advantage
grows: with seven hyperparameters, where random draws rarely land somewhere
good (@sec-cs-classifier-many).

::: {.keyidea title="In two dimensions, Bayesian optimization buys reliability"}
On the measured SVM landscape the median result of Bayesian optimization and
random search differed by at most 0.04 points; the share of runs that reached
an error below 1.0% differed by up to 14 points. The gap in the median opens
only when the space is larger than random sampling can cover.
:::

## Many hyperparameters {#sec-cs-classifier-many}

Two hyperparameters let us measure everything. Most models have more, and
then nobody measures the landscape: the only data are the evaluations a
search made. The second problem is of that kind, and it is where the median
gap between the two methods opens.

The model is a **gradient-boosted tree ensemble**: a sum of small decision
trees, each trained to correct the mistakes of the trees before it
[@friedman2001greedy]. It is a common first choice for tabular data, and
scikit-learn's `HistGradientBoostingClassifier` [@pedregosa2011scikit]
implements it. The data are the Spambase collection: 4,601 e-mails, each
described by 57 numbers such as the frequencies of particular words and
characters and the lengths of runs of capital letters, 39.4% of them spam
[@hopkins1999spambase]. The task is to say which e-mails are spam.

We split the e-mails once, at random and keeping the share of spam equal, into
a tuning set of 3,450 and a test set of 1,151. The objective is the 5-fold
cross-validation error on the tuning set, with one fixed split, so the same
configuration always returns the same number. The test set is never shown to
an optimizer. For the analysis only, we also trained every evaluated
configuration on the whole tuning set and recorded its test error.
@tbl-cs-classifier-hparams lists the seven hyperparameters we tuned.

::: {.table #tbl-cs-classifier-hparams title="The seven hyperparameters of the gradient-boosted classifier, their ranges, and the scale on which the search moves. Integer hyperparameters are rounded after the scale is applied."}
| Hyperparameter | Range | Scale | What it controls |
|---|---|---|---|
| learning rate | 0.005 to 1 | log | how much each new tree corrects the previous ones |
| trees | 10 to 500 | log, integer | how many trees are added |
| leaves | 2 to 256 | log, integer | the largest number of leaves in one tree |
| min leaf | 1 to 200 | log, integer | the fewest e-mails a leaf may hold |
| L2 penalty | $10^{-4}$ to 10 | log | how strongly leaf values are shrunk toward zero |
| features | 0.1 to 1 | linear | the fraction of the 57 features each split may consider |
| bins | 4 to 255 | log, integer | how finely each feature is discretized |
:::

Each search maps the box to the unit cube $[0, 1]^7$ through these scales.
**Random search** draws points uniformly in the cube. **Bayesian
optimization** starts from the first ten points of the random search with the
same seed and then runs Gaussian-process expected improvement: a Matérn 5/2
kernel with one lengthscale per hyperparameter, an amplitude, and a noise
term, all fitted by maximum marginal likelihood with four restarts, on the
logarithm of the cross-validation error; expected improvement over the lowest
posterior mean at an evaluated point, maximized over 4,000 uniform candidates
and 2,000 small perturbations of the five best points (@sec-acq-optimization).
Each search gets 60 evaluations, and each runs with ten different seeds:
1,200 trained and scored configurations in all, every one of them recorded
(tools/figure-data/cs-classifier-gbm.py).

An evaluation here is not a fixed cost. On one CPU thread, the
cross-validation of one configuration took between 0.09 and 32.6 seconds, a
factor of 350, set mostly by the number of trees and leaves. Keep that in mind
when reading the next figure, whose horizontal axis can count either
evaluations or seconds.

```{figure}
//| figure: cs-classifier-many
//| label: fig-cs-classifier-many
//| fig-cap: "Ten recorded runs each of random search (violet) and Bayesian optimization (orange) over seven hyperparameters of a gradient-boosted classifier on Spambase. Top: best error so far, thin lines for single runs and thick lines for the median; switch *Score* to see the held-out test error of each run's pick, and *Spend* to count compute seconds instead of evaluations. Bottom: the evaluated configurations in parallel coordinates, one line per configuration across the seven hyperparameters and its cross-validation error; bold lines are in the best 10% of all 1,200 evaluations; the bars give each hyperparameter's main-effect share. Every configuration was trained and scored; nothing is simulated. The timeline replays the recorded order."
```

Some things to try in the figure:

**Watch one run.** Set *Run* to 1 and press play. In the bottom panel, the
thickest line is the configuration just evaluated. Random search's lines stay
spread over every axis. After its tenth evaluation, Bayesian optimization's
lines gather in a few bands, many of them at the ends of the axes.

**Change the score.** Switch to *Held-out test*. The tidy staircases of the
top panel turn into lines that wander up and down, and the gap between the two
methods narrows.

**Count seconds.** Switch *Spend* to *Compute*. The orange runs stretch to the
right: the configurations Bayesian optimization prefers are the expensive
ones.

@tbl-cs-classifier-gbm collects the results.

::: {.table #tbl-cs-classifier-gbm title="Results of the recorded searches: medians over ten runs, with the range over runs in parentheses. CV error is what the optimizers minimized; test error is the held-out error of each run's pick, which no optimizer saw. Compute is the cross-validation time of a whole run on one CPU thread."}
| | CV error | Test error of the pick | Compute per run |
|---|---|---|---|
| scikit-learn default | 4.78% | 3.82% | 1.2 s |
| Random search, 60 evaluations | 4.61% (4.46 to 4.78) | 3.91% (3.21 to 5.13) | 92 s |
| Bayesian optimization, 30 evaluations | 4.61% (4.41 to 4.90) | 3.74% (3.21 to 4.69) | |
| Bayesian optimization, 60 evaluations | 4.44% (4.29 to 4.64) | 3.65% (3.30 to 4.00) | 300 s |
:::

Read from top to bottom, the table tells four stories.

**On the number it was given, Bayesian optimization won.** After 60
evaluations its median cross-validation error was 4.44% against random
search's 4.61%, and it reached random search's 60-evaluation median after 26
evaluations. Nine of its ten runs ended below random search's median.

**Counting seconds instead of evaluations removes most of that lead.**
Bayesian optimization favored many trees and many leaves: after its random
start, its configurations had a median of 280 trees and 76 leaves, against 68
and 24 for random search. Those cost time, a median of 300 seconds per run
against 92. Within the first 92 seconds of
compute, both methods' median best was 4.61%. @snoek2012practical proposed
the remedy in the same paper that popularized the method for tuning: divide
expected improvement by the predicted cost of an evaluation and choose the
setting with the most improvement per second. Later work treats cost-aware
search as a decision problem of its own [@xie2024cost].

**On held-out data the differences shrink toward the test set's own noise.**
The picks of Bayesian optimization had a median test error of 3.65%, random
search's 3.91%, and the default 3.82%. An error rate near 3.8% measured on
1,151 e-mails has a binomial standard error of 0.56 points (the square root
of $0.038 \times 0.962 / 1{,}151$, the spread such a rate has from the luck of
which e-mails landed in the test set), larger than any of these gaps. Among the 361 configurations with a cross-validation error
below 5%, the correlation between cross-validation error and test error was
0.16. Once a configuration is good, most of what distinguishes it from other
good ones in this data set is the particular split (inference).

**The default was hard to beat.** Only 20 of the 600 random configurations
had a lower cross-validation error than the library's default settings. That
is a statement about this box as much as about the default: a box this wide
is mostly worse than a sensible default. A large study of six algorithms on
38 data sets turned this observation into a measure, the **tunability** of
each hyperparameter: how much tuning it improves on a good default
[@probst2019tunability].

### Which hyperparameters mattered {#sec-cs-classifier-importance}

The bars under the axes of @fig-cs-classifier-many answer a question every
practitioner asks after a search: which settings mattered? Because random
search samples the cube uniformly, its 600 evaluations allow a simple
estimate. Split the range of one hyperparameter into five equal bins, and ask
what share of the variance of the logarithm of the error is explained by which
bin a configuration falls in. That share is the hyperparameter's **main
effect**. It is the idea behind functional analysis of variance
[@hutter2014efficient], computed here directly from the evaluations instead
of from a fitted model of them.

The learning rate explains 36% of the variance and the number of trees 26%.
The number of leaves explains 5.5%, and the remaining four together less than
5%. The seven main effects sum to 72%, so about a quarter of the variance
comes from hyperparameters acting together. The largest interaction is visible
between the first two axes of the figure: good configurations pair a high
learning rate with few trees or a low learning rate with many, and their
lines cross between those axes. In the best 10% of all evaluations, the
product of learning rate and number of trees has a median of 40.5, against
12.5 over all evaluations; what matters is roughly how far the ensemble
moves in total, not either factor alone (inference).

This pattern, a few hyperparameters doing most of the work, is the common
finding. @hutter2014efficient report that "even in very high-dimensional
cases" most variation is attributable to a few hyperparameters, and
@bergstra2012random found that "only a few of the hyper-parameters really
matter, but that different hyper-parameters are important on different data
sets." The second half of that sentence is why one cannot simply tune the
two that mattered last time.

### Where the search went {#sec-cs-classifier-edges}

The parallel coordinates show one more thing: Bayesian optimization's lines
crowd the ends of the axes. Of its evaluations after the first ten, 84% had at
least one hyperparameter within 2% of the edge of its range, and all ten of
its final picks had one. For random search the share was 27%, close to the 31% that uniform sampling
produces by chance with seven coordinates, a 2% margin at each end, and
integer rounding; 2 of its 10 picks were at an edge.

The pile-up has two readings. The best values may lie outside the box, in
which case the range was too narrow. Or expected improvement is drawn to the
boundary for the reason seen in @fig-cs-classifier-model: the posterior is
most uncertain there [@siivola2018correcting]. The data cannot tell these
apart, but the practical response is the same in both cases: look at where
the best configurations sit, and widen any range they press against.

## What the practitioner decides {#sec-cs-classifier-decisions}

Every result in this chapter depended on choices made before the first
evaluation or after the last, and the optimizer made none of them. They are
the part of tuning that test functions hide.

**The ranges.** A published guide is a good start [@hsu2003practical]; the
library's documentation and earlier runs are others. After a search, check
whether the best configurations sit at an edge, as Bayesian optimization's did
in @sec-cs-classifier-edges, and widen the range if they do.

**The scales.** Hyperparameters that act by multiplication belong on a log
scale. The difference is not cosmetic: on a linear scale from $2^{-5}$ to
$2^{15}$, a uniform draw has a chance of 0.003% of landing below $C = 1$, and
the best setting of the digits problem has $C = 1$.

**What the surrogate models.** An error that spans orders of magnitude is
better modeled through its logarithm, standardized, as both searches in this
chapter did (@sec-gp-pitfalls).

**Noise and the final answer.** A single evaluation's noise was a sixth of the
whole prize on the digits problem, and on Spambase it was larger than every
gap between methods on held-out data. Re-measure the few best settings with
fresh splits before choosing; recommend by the posterior mean rather than the
lowest measurement (@sec-loop-recommend); and keep a test set that no part of
the search touches, scored once at the end [@cawley2010over].

**Budget and cost.** Count what an evaluation costs, not how many there are.
When the cost varies by a factor of 350, as it did here, cost-aware
acquisition [@snoek2012practical; @xie2024cost] or cheap partial evaluations
help more than a better surrogate. Hyperband trains many random
configurations briefly and continues only the promising ones; its authors
report "over an order-of-magnitude speedup" over competitors that included
popular Bayesian optimization methods, on deep-learning and kernel-method
problems [@li2018hyperband]; FABOLAS models
how the error changes with the size of the training subset and often finds
good configurations "10 to 100 times faster" [@klein2017fast].

**Stopping.** In the recorded runs, Bayesian optimization's median best
improved by 0.04 points over its last 20 evaluations, from 4.48% to 4.44%.
A rough standard error of one cross-validation estimate is its fold-to-fold
standard deviation, 0.59 points among the good configurations, divided by
$\sqrt{5}$: about 0.26 points (rough, because the folds share training
data). Those 20 evaluations bought an improvement about six times smaller
than the estimate's own uncertainty. A principled version of this judgment is
to continue only while some configuration's expected improvement exceeds the
cost of evaluating it. @xie2026cost prove that this rule, paired with
suitable acquisition functions, does no worse in expected cost-adjusted
regret than stopping immediately, which they describe as the first
theoretical guarantee of this type for an adaptive stopping rule in Bayesian
optimization.

**Baselines.** Evaluate the default first, then run random search with the
same budget as any cleverer method; both are cheap, and on Spambase the
default was at least as good as 580 of the 600 random configurations. When the evidence
is pooled over many problems, Bayesian optimization does beat random search:
in the 2020 black-box optimization challenge, which tuned standard models on
real data sets, 61 of 65 teams beat random search, and the best submissions
needed over 100 times fewer evaluations to match it [@turner2021bayesian].
The same paper cites surveys of authors in which only 7% of NeurIPS 2019 and
6% of ICLR 2020 papers tuned with such methods rather than by hand, grid, or
random search.

::: {.keyidea title="The optimizer optimizes the number you give it"}
Bayesian optimization found lower cross-validation errors than random search
on both problems. Whether that number was worth lowering, what an evaluation
cost, and how far the winner can be trusted were questions for the person who
set up the search, and on this evidence they mattered as much as the choice
of optimizer.
:::

The next chapter, @sec-cs-chemistry, moves from settings of software to
conditions of a chemical reaction, where an evaluation is an afternoon in a
laboratory, most choices are categorical, and experiments are run in batches.

## Exercises {#sec-cs-classifier-exercises}

::: {.exercise #exr-cs-classifier-gamma}
The SVM kernel @eq-cs-classifier-rbf and the RBF kernel of
@sec-function-space differ only in how the width is written. What lengthscale
$\ell$ corresponds to the best $\gamma = 2^{-2}$? Using the median squared
distances from @sec-cs-classifier-svm, explain in terms of $\ell$ why
$\gamma = 2^{3}$ makes every training image an island.

::: {.solution}
Matching $\exp(-\gamma r^2)$ with $\exp(-r^2 / 2\ell^2)$ gives
$\ell = 1/\sqrt{2\gamma}$. For $\gamma = 2^{-2}$, $\ell = 1/\sqrt{0.5} \approx
1.41$. The median distance to the nearest neighbor is $\sqrt{1.05} \approx
1.02$, less than one lengthscale, so neighbors are strongly similar; two
different digits are $\sqrt{9.9} \approx 3.1$ apart, more than two
lengthscales, so they barely interact. For $\gamma = 2^{3}$,
$\ell = 1/\sqrt{16} = 0.25$: even the nearest neighbor is about four
lengthscales away, where the kernel has fallen to
$\exp(-8 \times 1.05) \approx 0.0002$. Each training image then influences
only a tiny ball around itself, and a new image falls in nobody's ball.
:::
:::

::: {.exercise #exr-cs-classifier-curse}
Suppose 20 settings all have a true error of 1.0%, and each is measured once
with independent Gaussian noise of standard deviation 0.16 points, the noise
of one cross-validation run on the digits problem. The expected maximum of 20
independent standard normal variables is about 1.87. What is the expected
lowest measured error? How does the answer change if each setting is measured
four times and the measurements are averaged?

::: {.solution}
The lowest of 20 measurements is the mean minus the largest of 20 noise
draws, so its expectation is about $1.0 - 1.87 \times 0.16 \approx 0.70\%$.
The winner looks 0.3 points better than it is, although no setting is better
than any other. Averaging four measurements halves the standard deviation to
0.08 points, and the expected lowest average is about
$1.0 - 1.87 \times 0.08 \approx 0.85\%$. The bias shrinks but does not vanish,
which is why a final estimate should come from data the selection did not use.
:::
:::

::: {.exercise #exr-cs-classifier-top5}
Show that 60 independent uniform draws from a search box include at least one
point from the best 5% of the box with probability above 95%. On the digits
landscape, the best 5% of settings have an average error of at most 1.01%;
among the 600 random configurations on Spambase, the best 5% have a
cross-validation error of at most 4.81%. What does each number say about the
guarantee?

::: {.solution}
Each draw misses the best 5% with probability 0.95, so 60 independent draws
all miss with probability $0.95^{60} \approx 0.046$, and at least one hits
with probability about 0.954. The guarantee is relative to the box. On the
digits landscape the best 5% is a strong result, within 0.1 points of the
best setting. On Spambase it is not: the threshold of 4.81% is worse than the
library default of 4.78%, because most of that box is worse than the default.
"Top 5% of the box" says nothing about how good the box is, which is why the
range and the default matter as much as the search.
:::
:::

## Further reading {#further-reading .unnumbered}

- @hsu2003practical is the practical guide whose ranges and coarse-to-fine
  grid procedure this chapter adopts; it is short and still the best
  description of how SVMs are tuned by hand.
- @snoek2012practical brought Gaussian-process Bayesian optimization to
  machine-learning hyperparameters, with the ARD Matérn 5/2 kernel, the
  integration over kernel hyperparameters, and expected improvement per
  second.
- @bergstra2012random makes the case for random search over grids, with the
  low-effective-dimensionality argument illustrated in
  @sec-cs-classifier-landscape.
- @feurer2019hyperparameter surveys hyperparameter optimization, including the
  multi-fidelity methods this chapter only mentions.
- @hutter2014efficient and @probst2019tunability measure which
  hyperparameters matter and how much tuning improves on defaults, across many
  data sets.
- @cawley2010over explains why the best cross-validation score is optimistic
  and how much that can matter.
- @turner2021bayesian reports the 2020 black-box optimization challenge, the
  largest head-to-head comparison of tuning methods on real models.
- @keerthi2003asymptotic explains the shape of the SVM landscape: where it
  underfits, where it memorizes, and why it becomes linear.
- The data are @alpaydin1998optical (the digits, as shipped with scikit-learn)
  and @hopkins1999spambase (the e-mails), both under CC BY 4.0. The scripts
  that measured everything in this chapter are in `tools/figure-data/`
  (`cs-classifier-svm.py`, `cs-classifier-gbm.py`, `cs-classifier-race.ts`),
  with the exact commands in their headers.
