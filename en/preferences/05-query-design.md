---
status: done
synopsis: "A pair is not the only question a system can ask. Choices among several and rankings, a slider that searches along a line, galleries and projections, answers that say 'about the same', 'not sure', or 'it crashed', many people at once, and why the interface belongs to the model."
sources: ["Koyama et al. 2017", "Koyama and Igarashi 2018", "Koyama et al. 2020", "Mikkola et al. 2020", "Siivola et al. 2021", "Bıyık et al. 2019"]
---

# Designing the Question {#sec-query-design}

@sec-pbo built preferential Bayesian optimization (PBO) around one question: of these
two options, which do you prefer? The model chose the two options, the person
answered with one bit, and the loop repeated. That question is a design
decision, and it is not the only one available. A system can show four options
and ask for the best, or ask for an order. It can hand the person a slider and
let them search a whole line of designs in one gesture. It can let them say
that two options look about the same, that they are not sure, or that the
experiment failed.

Each of these choices changes three things at once: how much effort an answer
costs the person, how much the answer can tell the model, and what kind of
noise the answer carries. The likelihood, the function that says how probable
each answer is given the utility, has to change with it, because the
likelihood is the model's description of a person using a particular
interface. This chapter works through the main alternatives to the pair, each
with its likelihood, and ends with the argument that gives the chapter its
shape: the interface is part of the model.

## Pairs, sets, and rankings {#sec-query-forms}

A pair is the smallest question that reveals a preference, and that is both
its strength and its limit. It is easy to answer, but its answer carries at
most one bit, and in a design space of several dimensions one bit is a small
step. The obvious extension is to show more options at once. Showing $q$
options and asking for the best one can carry up to $\log_2 q$ bits; asking
for a complete order of $q$ options can carry up to $\log_2 q!$ bits, which is
about 4.6 bits for four options. These are upper bounds, reached only when
every answer is equally likely in advance (@sec-comparison-information), but
they show what is at stake.

### Choosing one from a set {#sec-query-choice}

The model for a choice among several options goes back to @luce1959individual.
Let $S$ be the set of options shown and $g$ the latent utility. The probability
that the person picks $\vx_i$ is

$$
\Prob(\vx_i \text{ chosen from } S) = \frac{\exp\!\big(g(\vx_i)/\tau\big)}{\sum_{\vx_j \in S} \exp\!\big(g(\vx_j)/\tau\big)},
$$ {#eq-query-luce}

where $\tau > 0$, the scale of the logistic link (@eq-cmp-logit), often
called a temperature, sets how noisy the choices are: a small $\tau$
makes the person pick the best option almost always, a large $\tau$ makes the
choice nearly uniform. A machine-learning reader will recognize the right-hand
side as the softmax function. With two options it reduces to
$1/(1 + e^{-(g(\vx_1) - g(\vx_2))/\tau})$, the logistic link of the
Bradley-Terry model [@bradley1952rank] that @sec-bradley-terry introduces.

The formula has a random-utility reading (@sec-random-utility). Suppose the
person perceives each option's utility with independent noise drawn from a
Gumbel distribution, the distribution that describes the largest of many
random values, and picks the option whose perceived utility is highest. Then
the probability that $\vx_i$ wins is exactly @eq-query-luce
[@mcfadden1974conditional]. The pairwise probit model of @sec-gp-preference
makes the same assumption with Gaussian noise instead; for two options the two
links differ only slightly in shape, but only the Gumbel version gives a
closed form for larger sets.

@eq-query-luce has a consequence that matters for interface design. It
inherits the independence of irrelevant alternatives that follows from Luce's
choice axiom (@sec-bradley-terry): the ratio of the probabilities of choosing $\vx_1$ and $\vx_2$ is
$\exp\!\big((g(\vx_1) - g(\vx_2))/\tau\big)$, whatever else is in the set.
Real choices sometimes violate it, as in the decoy or attraction effect
[@huber1982adding]. A pair has no third option, so it cannot show this effect;
a set of four can. The effect appears mostly when each attribute of the options
is shown as a number, and usually not when the attributes are perceived
directly [@frederick2014limits]. That is reassuring for visual design but not a
guarantee (inference). @sec-jdm reviews this evidence.

With @eq-query-luce as the likelihood, the rest of the machinery is unchanged.
The Laplace approximation of @sec-pref-laplace needs the gradient and the
curvature of the log-likelihood. For one choice with probabilities
$p_j$ over the options in $S$, the gradient with respect to $g(\vx_j)$ is
$(\mathbb{1}[j = i] - p_j)/\tau$, where $\mathbb{1}[j = i]$ is 1 for the
chosen option and 0 otherwise, and the negative Hessian is
$(\operatorname{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^\T)/\tau^2$ on those
options: the covariance matrix of a one-hot vector drawn with probabilities
$\mathbf{p}$. It is positive semidefinite, as the Laplace approximation
requires, and for two options it is the familiar pairwise term.

### Does a larger set help? {#sec-query-batch}

The evidence on larger sets points in two directions. @siivola2021preferential
derived a likelihood for any parallel feedback on two or more points and
argued that the *batch winner*, the best option of a set, is the most useful
form, because full rankings of large batches are laborious for people and
sometimes impossible, as in A/B testing. On six benchmark functions with
batches of four, they found that the differences between acquisition
functions were larger than the differences between feedback types; across
batch sizes from 2 to 6 they saw no clear difference. Their experiments used at
most four dimensions. The qEUBO experiments
(@sec-eubo), with an acquisition function designed for sets, found the opposite: queries of four
options reached a given simple regret (the gap between the best utility and
that of the recommended option, @sec-regret-definitions) with fewer queries
than pairs, while the
further gain from four to six was smaller and less stable; the authors note the
contrast with Siivola et al. [@astudillo2023qeubob]. The difference may lie in
the acquisition functions or the noise levels, but no paper has separated
these factors (inference).

Theory for finite sets of options gives a sharper answer, and it depends on
what the person reports. Under the Plackett-Luce model introduced below, and
for a finite set of options, consider the *sample complexity* of finding a
near-best option: the number of queries needed to find one with high
probability. Learning from the winners of $k$-option sets has the same sample
complexity, up to constant factors, as learning from pairs: it does not
improve with $k$. Reporting the top $m$ of each set instead reduces it by a
factor of $m$ [@saha2019pac]. A larger set helps only when the person tells more than which
option won. For linear utilities, a 2025 result shows that larger subsets
provably help under the Plackett-Luce model [@lee2025preference]; as of
September 2026 this has not been carried over to the Gaussian process models
of this book. @sec-dueling-bandits returns to these results.

### Rankings {#sec-query-rankings}

A ranking can be read as a sequence of choices: first the best of the set,
then the best of what remains, and so on. Under the choice axiom, each later
choice is again @eq-query-luce on the remaining options. That gives the
probability of a whole ranking.

::: {.derivation title="A ranking as a chain of choices"}
Let $\pi$ order the $q$ shown options from best to worst, so $\vx_{\pi(1)}$ is
ranked first.

1. By the chain rule of probability (@sec-joint-marginal-conditional), the
   probability of the ranking is the probability of the first place, times
   the probability of the second place given the first, and so on: a product
   over places $r = 1, \dots, q$ of the probability that $\vx_{\pi(r)}$ takes
   place $r$ given the places before it.
2. Given the first $r - 1$ places, the $r$-th place is a choice of the best
   option among the ones not yet placed, $S_r = \{\vx_{\pi(r)}, \dots, \vx_{\pi(q)}\}$.
3. Assume, in the spirit of Luce's choice axiom, that the options already
   placed do not affect that choice. Then it follows @eq-query-luce with the
   set $S_r$.
4. Substituting gives @eq-query-pl. The last factor, $r = q$, is a choice from
   a set of one and equals 1.
:::

$$
\Prob(\pi) = \prod_{r=1}^{q} \frac{\exp\!\big(g(\vx_{\pi(r)})/\tau\big)}{\sum_{s=r}^{q} \exp\!\big(g(\vx_{\pi(s)})/\tau\big)}.
$$ {#eq-query-pl}

This is the **Plackett-Luce** model [@plackett1975analysis;
@luce1959individual]. A *top-$k$ ranking*, in which the person orders only the
$k$ best options, keeps the first $k$ factors. The Gumbel reading carries over:
if every option's perceived utility has independent Gumbel noise and the person
sorts by perceived utility, the ranking follows @eq-query-pl.

@nguyen2021top built a Gaussian process surrogate inspired by the multinomial
logit and its ranking extension, which can be read as Gaussian process
regression with independent Gumbel noise; it handles top-$k$ rankings and ties,
is trained by variational inference, and was evaluated on synthetic
functions, CIFAR-10, and the SUSHI preference data. The tutorial of
@benavoli2026tutorial describes it as a Gaussian process generalization of the
Plackett-Luce model. @benavoli2023choice went further and let the person
choose a *subset* (of five options A to E, say, the acceptable ones are A, B,
and C), modeling the
choice with several latent utilities and choosing their number by how well
the model predicts held-out choices (cross-validation,
@sec-cs-classifier-cv); the abstract reports simulations only. As of September 2026
we found no newer Gaussian process PBO paper with a Plackett-Luce or
multinomial ranking likelihood.

::: {.keyidea title="More options per question help only if the answer says more"}
Showing more options raises what an answer can carry, but only if the person
reports more than the winner. A set of four with only its winner reported is,
in the worst case, no better than a pair; a ranking of the same four can be.
:::

## Searching along a line {#sec-line-search}

Sets and rankings still compare options the system chose. In a design space of
six or ten parameters, the system's few options are a sparse sample, and most
of the person's knowledge about what would look better goes unused. The
opposite extreme, handing the person every parameter, does not work either:
exploring a high-dimensional space directly is difficult even for designers
[@koyama2020computational], and absolute scores fail for the reason given in
@sec-cmp-ratings-hard: they require a familiarity with the whole design space
that a newcomer does not have.

@koyama2017sequentialb found a middle ground: give the person a single slider.
The slider maps to a line segment in the design space, the preview updates as
the person drags, and the person stops where the design looks best. One answer
is a one-dimensional optimization performed by the person, over a continuum of
designs, with no familiarity with the parameters required. The method is
called **sequential line search**.

### Which line {#sec-query-which-line}

The system chooses the segment. After $t$ answers, with a Gaussian process
posterior over the utility, let $\vx^+_t$ be the observed design with the
highest posterior mean and $\vx^{\EI}_t$ the design that maximizes expected
improvement (@sec-ei). The next slider runs between them:

$$
S_{t+1} = \big\{ (1 - s)\,\vx^+_t + s\,\vx^{\EI}_t \;:\; s \in [0, 1] \big\}.
$$ {#eq-query-slider}

One end is the best design so far, so the person can always keep it. The other
is the most promising place to look, so the slider spans the trade-off between
exploiting and exploring that every acquisition function balances. The first
slider, before any data, connects two random points [@koyama2020computational].

The answer is the chosen point $\vx^{\text{c}}_{t+1}$ on the segment. Koyama et
al. record it as a choice from a set of three: the chosen design beats both
ends, $\vx^{\text{c}} \succ \{\vx^+, \vx^{\EI}\}$, with the likelihood of
@eq-query-luce on those three designs [@koyama2020sequential]. The record
throws information away. The person preferred the chosen point to every other
point on the slider, a continuum of comparisons, not just to the two ends. The
authors note that more points could be added to the losing side at extra
computational cost. @mikkola2020projective take the continuum seriously: they
treat the answer as uncountably many pairwise comparisons, write its likelihood
as a limit of products over finer and finer partitions of the line, and
approximate it with a finite set of sampled comparisons.

::: {.algorithm #alg-query-line-search title="Sequential line search"}
Input: a design space $\X$, kernel $k$, choice scale $\tau$, budget $N$
sliders.

1. Set the first slider between two random designs.
2. Show the slider with a live preview. Record the position the person
   chooses, $\vx^{\text{c}}$.
3. Add the choice $\vx^{\text{c}} \succ \{\text{both ends}\}$ to the data and
   fit the utility posterior with the likelihood @eq-query-luce and the
   Laplace approximation.
4. Find $\vx^+$, the observed design with the highest posterior mean, and
   $\vx^{\EI}$, the design with the highest expected improvement over it.
5. Set the next slider by @eq-query-slider and return to step 2 until the
   budget is spent. Recommend $\vx^+$.
:::

### Try it {#sec-query-try-slider}

The figure below runs @alg-query-line-search with you on the slider. The design
is a small poster landscape with two parameters, warmth and vividness. Move the
slider, or drag along the strip of thumbnails, until the picture looks best to
you, then press *Choose this one*. The map on the right shows the model's
posterior mean over the two parameters (darker is better), the sliders you have
used, the designs you chose, and the next slider in orange, running from the
best design so far (circle) to the point of highest expected improvement
(diamond).

```{figure}
//| figure: query-line-search
//| label: fig-query-line-search
//| fig-cap: "Sequential line search [@koyama2017sequentialb] with you on the slider. Each answer is recorded as the method records it, a choice of the chosen design over both ends of the slider (@eq-query-luce with three options), and the posterior over the two design parameters is fitted with the Laplace approximation. The next slider runs from the best design so far to the design with the highest expected improvement. The design space has two parameters so that the map can show all of it; the photo enhancement task of the original paper had six. To keep the slider usable, its far end is kept at least 0.12 (in units of the map's side) from the near end. Switch to *Simulated person* to watch the loop against a hidden favorite; the simulated person picks a position on the whole slider, by a softmax over 41 positions with noise τ."
```

Some things to try. Answer five or six sliders honestly and watch the sliders
on the map: they get shorter and cluster as the model becomes sure where your
favorite region is, and the far end jumps to a new region when the
uncertainty elsewhere makes expected improvement large. Choose an end of the
slider without moving it, and the record becomes an ordinary pair, the best
design against the expected-improvement design. In the simulated mode, reveal
the hidden favorite and press *Simulate five* a few times: with the default
noise, the star usually sits close to the hidden favorite by the tenth
slider. Raise the noise to 0.2 and the chosen points
scatter along each slider; the model, which assumes a fixed $\tau$,
reads the scatter as weak preferences and converges more slowly.

The map makes the method look easy, because in two dimensions a dozen lines
already pass close to most of the space. The method was built for more. In six
dimensions, as in the photo task, a dozen lines pass close to only a small part
of the space, so the choice of line does the work: one end
anchors the search at the best design so far, and expected improvement points
the other end at the region where a better design is most likely. The
Sequential Gallery simulations below, run in 5 to 20 dimensions, show how much
the choice of subspace matters there.

### What the crowd did {#sec-query-crowd}

Sequential line search was designed for crowdsourcing, in which many paid
workers each perform a small task (a *microtask*) and the system combines
their answers. @koyama2020computational give the procedure in detail. Every
result used 15 iterations. In each iteration the system posted seven slider
microtasks and moved on once at least five answers had arrived, using the
median of the returned slider positions as the choice. A microtask paid 0.05
USD, so a result cost 5.25 USD, and the photo examples took about 68 minutes on
average.

For photo color enhancement with six parameters, crowd workers were then asked
which of four versions of each photo looked best: the original, the
crowd-optimized result, and the automatic enhancements of Adobe Photoshop and
Lightroom. Across three photos the crowd-optimized versions received 32, 26,
and 29 votes, against 0 to 3 for each alternative. Three runs started from
different initial conditions produced similar enhancements, and the
differences between them shrank rapidly within the first four or five
iterations [@koyama2020computational]. @sec-cs-photo works through a photo enhancement problem of
the same kind, with pairs and with a slider.

Two features of this setup recur in the rest of the chapter. Averaging over
many workers assumes, in the authors' words, that "a common 'general'
preference exists that is shared among crowds" (@sec-many-users). And a slider
answer has its own kind of noise: how precisely a person can position a
slider, and how finely they can see differences between nearby designs, are
properties of the person and the widget, not of the utility. The papers on
slider and projection queries fold it into the same single noise term as
everything else (inference).

## Galleries and projections {#sec-gallery-projective}

A line is one way to let the person search a subspace. Two other designs
generalize it: a plane of designs shown as a grid, and a projection along any
direction the system chooses.

### Sequential Gallery {#sec-query-gallery}

@koyama2020sequential replaced the slider with a two-dimensional plane and the
preview with a *zoomable grid*: a gallery of designs sampled from the plane,
5 by 5 for the photo task. The person clicks the best design; the grid zooms in
around it by a factor of two; after a fixed number of clicks (four in their
implementation) the selection is the answer. The method is called sequential
plane search, and the whole interactive framework Sequential Gallery.

The plane is chosen like the line. It is centered at the current best design
$\vx^+$, has the expected-improvement point as one of its vertices, and is
otherwise placed to maximize a new acquisition function, the average expected
improvement over the plane, approximated on a 5 by 5 lattice of points. The
answer is recorded as a choice over five representative points: the chosen
design beats the center and the four vertices.

In simulations with 50 trials per method, on test functions of 5 to 20
dimensions, plane search outperformed line search on every function at every
iteration count, and the acquisition-based plane outperformed a random plane
after the first few iterations. A preliminary study with five students and one
researcher (one participant described themselves as an expert, the other five
as novices) enhanced photos: participants pressed a button to say they were
satisfied after 5.36 iterations on average (standard deviation 2.69), and one
plane subtask took 14.8 seconds on average. The statement "I could get
inspiration for possible enhancement from the grid view" scored a mean of 6.50
(standard deviation 0.548) on a seven-point scale. The authors list the
limitations themselves: the grid suits only designs that can be recognized at a
glance, discrete parameters are not handled, and Bayesian optimization is
known to perform poorly beyond about 20 dimensions [@koyama2020sequential].

### Projective preferential queries {#sec-query-projective}

@mikkola2020projective asked the person for the best position along a
*projection*. A query is a direction $\boldsymbol{\xi}$ and a reference design
$\vx$; the person reports the best scalar $\alpha$ for the design
$\alpha\boldsymbol{\xi} + \vx$, with the coordinates that $\boldsymbol{\xi}$
leaves at zero held at the values in $\vx$. When $\boldsymbol{\xi}$ is a
coordinate direction, this is "set this one knob to its best value"; in their
user experiment, materials scientists set one coordinate at a time of the
position and orientation of a molecule above a surface.

Their likelihood is the continuum version of the slider record. They assume
Thurstone's Gaussian noise, modeled as a white-noise process along the
projection, write the probability that the reported point beats every other
point on the projection as a product integral, and approximate it with sampled
pseudo-comparisons inside the Laplace approximation. With a budget of 100
queries on four test functions of 2, 6, 10, and 20 dimensions, a preferential
coordinate descent strategy was best on three of the four, a projective
version of expected improvement was best on the 20-dimensional function, and
all projective variants clearly outperformed all pairwise variants. To show how
little a pair carries, they trained the dueling model of
@gonzalez2017preferentialb (@sec-dueling-formulation) on 2000 random duels of
the two-dimensional six-hump camel function; the point it found had value
0.1052 against the global minimum of −1.0316, and optimizing its
soft-Copeland score took 41 minutes, while the projective version with random
queries reached that accuracy within its first queries.

The paper also states two cautions that belong in this chapter. The more
nonzero coordinates a direction has, "the greater the 'cognitive burden' to a
human user", so the best query for a person is not the best query for a
perfect oracle. And a person may be unable to state a best value along some
direction at all; the authors suggest allowing the answer "I do not know" and
leave it for future research [@mikkola2020projective].

### Lines for the algorithm, not the person {#sec-query-linecospar}

A line can also restrict the algorithm rather than the person. LineCoSpar, for
exoskeleton gait tuning, takes at each iteration a random line through the
design with the highest posterior mean and runs Thompson sampling
(@sec-thompson) over that line and the designs already visited; the person
still answers pairwise comparisons of gaits. Tested with six able-bodied
participants tuning six gait parameters, it was introduced because running the
group's earlier method, CoSpar [@tucker2020preference], on a six-dimensional
space was infeasible [@tucker2020human]. The idea is the same as in
@sec-line-search, a one-dimensional subspace chosen to be promising, used for
the computation instead of the interaction. @sec-high-dimensions follows these
subspace methods further.

@tbl-query-forms collects the forms so far. The column of bits is the
logarithm of the number of possible answers, a ceiling that a noisy answer
never reaches; it shows how the forms differ in what they could carry, not what they
do carry.

::: {.table #tbl-query-forms title="Query forms, what the person does, how the answer enters the likelihood, and the most it could carry."}
| Form | The person | Recorded as | At most | Example |
|---|---|---|---|---|
| Pair | picks one of two | $\vx \succ \vx'$ | 1 bit | @brochu2007active; @sec-pbo |
| Best of $q$ | picks one of $q$ | @eq-query-luce | $\log_2 q$ bits | @siivola2021preferential; @astudillo2023qeubob |
| Top-$k$ of $q$ | orders the $k$ best | first $k$ factors of @eq-query-pl | $\log_2 \frac{q!}{(q-k)!}$ bits | @nguyen2021top |
| Slider | drags to the best point on a segment | chosen beats both ends, or a continuum | $\log_2$ of the slider's resolution | @koyama2017sequentialb |
| Gallery | clicks the best of a 5 by 5 grid, four times | chosen beats center and vertices | $4 \log_2 25$ bits | @koyama2020sequential |
| Projection | sets a direction to its best value | a continuum of comparisons | resolution-limited | @mikkola2020projective |
| With "about the same" | picks a, b, or neither | @eq-query-threshold | $\log_2 3$ bits | @byk2019asking |
| Graded | picks one of $R$ ordered levels, from "a, clearly" to "b, clearly" | @eq-query-ordinal | $\log_2 R$ bits | @li2021roial; @wu2025mixed |
:::

### What the evidence says about forms {#sec-query-form-evidence}

Read across papers, the evidence has a consistent direction and a consistent
gap. Forms that let one human action carry more (projections, planes, queries
of four) beat pairs in the papers that introduced them, and the only direct
comparison between batch winners and full rankings found little difference
[@siivola2021preferential]. But almost every comparison is a simulation, and
each paper uses its own acquisition function, budget, and simulated noise.
The first same-task comparisons with people appeared in 2026: in GimmBO, with
12 participants (all with computer science or machine learning backgrounds)
and 20 iterations each, ranking beat a slider on image similarity and success
rate; the slider and gallery baselines ended with redundant adapters switched
on, that is, adapters outside the set that had produced the target image (an
adapter is a small fine-tuned add-on to the image model, and the task was to
tune the weights with which adapters are merged); and users of Sequential
Gallery reported getting stuck in local minima. Ranking took longer per step,
50.5 seconds against 34.7 for the slider and 10.6 for the gallery
[@liu2026gimmbo]. As of September 2026 we found no
controlled study with people that compares pairs, batch winners, rankings, and
sliders under the same acquisition function and budget, and no acquisition
function derived for slider, plane, or projection queries beyond the adapted
expected improvement and random subspaces described here. @sec-hci-feedback-forms
reports the human-factors evidence in detail.

## Ties, indifference, and confidence {#sec-ties}

Every model so far forces an answer. A person who sees two designs that look
the same to them must still pick one, and what they pick is close to a coin
flip. The model cannot tell a coin flip from a weak preference, so it reads
the flip as evidence that one option is slightly better. This section looks at
answers that say more than "a" or "b": about the same, not sure, how sure, and
the experiment failed.

### About the same {#sec-query-indifference}

The simplest extension gives the person a third button. To model it, start
from Thurstone's picture (@sec-thurstone): the person perceives the utility
difference $\Delta = g(\vx_a) - g(\vx_b)$ with Gaussian noise of standard
deviation $s$. Add an *indifference threshold* $\delta \ge 0$, a
just-noticeable difference in the sense of psychophysics
(@sec-psychophysics): differences smaller than $\delta$ are not reported as
preferences. The three answers then have probabilities

$$
\begin{aligned}
\Prob(a) &= \Phi\!\left(\frac{\Delta - \delta}{s}\right), \\
\Prob(b) &= \Phi\!\left(\frac{-\Delta - \delta}{s}\right), \\
\Prob(\text{same}) &= 1 - \Prob(a) - \Prob(b).
\end{aligned}
$$ {#eq-query-threshold}

With $\delta = 0$ the middle answer never occurs and the model is the probit
pair of @sec-gp-preference. The tutorial of @benavoli2026tutorial lists a
just-noticeable-difference likelihood of this kind among its nine models, with
indistinguishability statements of the form
$\lvert g(\vx) - g(\vx')\rvert / \delta \le 1$, tracing the idea to Luce's 1956
notion of a discrimination threshold. @erarslan2025consecutive use such a
threshold in an extended Thurstone model and report clear gains when 10% to 20%
of comparisons are indifferent; theirs is a 2025 preprint.

The logistic version is due to @byk2019asking, who added an "About Equal"
option to active reward learning with a minimum perceivable difference
$\delta \ge 0$:

$$
\begin{aligned}
\Prob(a) &= \frac{1}{1 + \exp(\delta - \Delta)}, \\
\Prob(b) &= \frac{1}{1 + \exp(\delta + \Delta)}, \\
\Prob(\text{same}) &= \big(e^{2\delta} - 1\big)\,\Prob(a)\,\Prob(b).
\end{aligned}
$$ {#eq-query-biyik}

It reduces to the Bradley-Terry model at $\delta = 0$. Their reward was a
linear function of trajectory features rather than a Gaussian process, and in
their simulations the weak-preference queries consistently reduced the number
of wrong answers. The two treatments in use, a threshold in a Thurstone or
logistic model and ties in a multinomial logit as in @nguyen2021top, are the
ones the field has settled on (inference).

How much is the third button worth? Information theory answers this cleanly
(@sec-mutual-information). Compare two interfaces shown to the same person,
whose answers follow @eq-query-threshold. With three buttons, they report what
they perceive. With two, they report "a" or "b" when they perceive a
preference and flip a coin when they do not.

::: {.derivation title="A forced choice never carries more"}
Let $\Delta$ be the unknown utility difference, $Y$ the three-button answer,
and $Z$ the two-button answer.

1. $Z$ is computed from $Y$ alone: $Z = Y$ when $Y$ is "a" or "b", and $Z$ is
   a fair coin when $Y$ is "same". The coin does not depend on $\Delta$.
2. So $\Delta \to Y \to Z$ is a chain in which $Z$ depends on $\Delta$ only
   through $Y$.
3. The *data-processing inequality* says that processing an observation can
   only lose information about its cause: for such a chain,
   $I(\Delta; Z) \le I(\Delta; Y)$, where $I$ is mutual information
   [see @cover2006elements, ch. 2].
4. The inequality is usually strict, because "same" itself carries
   information (that $\lvert\Delta\rvert$ is probably small) and the coin
   erases it.
:::

The figure shows the size of the loss. The top panel plots the three answer
probabilities of @eq-query-threshold against the utility difference, with the
forced-choice probability of "a" dashed. The bottom panel plots how many bits
one answer carries about $\Delta$, for each interface, when the model's
current belief about $\Delta$ is Gaussian with mean $m$ and a given spread.

```{figure}
//| figure: query-ties
//| label: fig-query-ties
//| fig-cap: "What an about-the-same answer is worth, under the threshold model of @eq-query-threshold with unit comparison noise. Top: the probability of each answer against the utility difference Δ, with the indifference band shaded, and the probability of answering a when the same person must flip a coin instead of saying about the same (dashed). The gray bump is the model's belief about Δ. Bottom: the mutual information between one answer and Δ, in bits, for both interfaces; the shaded gap is what the coin flip destroys. The numbers are illustrative: they depend on the assumed noise and threshold."
```

Some things to try. Set the threshold to zero: the two curves in the bottom
panel coincide, because nobody is ever indifferent. Raise it to 1.5 with the
model's guess at $m = 0$, a close pair: the forced choice keeps only about a
third of what the three-button answer carries. Move $m$ to 3, a pair the
model already thinks is lopsided: both interfaces carry less than they do at
$m = 0$ (0.33 bits against 0.57 with three buttons, 0.14 against 0.19 with
two), because the answer is more predictable. Shrink the
uncertainty about $\Delta$: the bits fall toward zero for both, which is why acquisition functions avoid asking about
pairs the model is already sure of.

Two cautions keep the figure honest. A model that has no "same" outcome,
fitted to forced-choice data from a person who is often indifferent, reads the
coin flips as noise: if the noise level is fitted, it grows; if it is fixed,
the estimated utility differences shrink and the whole estimate flattens
(inference). And a third button may change behavior as well as recording it:
people may press it to avoid an effortful judgment, a possibility the threshold
model does not describe (inference).

### Not sure, and how sure {#sec-query-confidence}

"About the same" says the difference is small. "I do not know" says something
different: that the person cannot judge, perhaps because the options differ in
ways they cannot weigh. Mikkola et al. proposed that answer and left it for
future work (@sec-query-projective), and an IUI 2023 study let participants
rank four candidates while putting candidates they could not judge into a
separate "don't know" area, to express incomplete preferences [@ou2023impact].
As of September 2026 we found no Gaussian process PBO paper whose model has an
abstain or skip outcome distinct from a tie.

Confidence ratings go the other way and say more. A graded answer such as "a,
clearly" or "a, slightly" is an ordinal answer, and the threshold model extends
to it directly: place cut points $c_0 = -\infty < c_1 < \dots < c_{R-1} < c_R = \infty$
on the perceived difference and report level $r$ when it falls between
$c_{r-1}$ and $c_r$,

$$
\Prob(\text{level } r) = \Phi\!\left(\frac{c_r - \Delta}{s}\right) - \Phi\!\left(\frac{c_{r-1} - \Delta}{s}\right).
$$ {#eq-query-ordinal}

@eq-query-threshold is the case of three levels with cut points $-\delta$ and
$\delta$. A model can also add a *lapse rate* $\lambda_{\text{lapse}}$, the
probability that an answer is a random slip, so that each level has
probability $\lambda_{\text{lapse}}/R + (1 - \lambda_{\text{lapse}})$ times
the expression above; a single careless click then cannot drag the posterior
far.

Three studies show the range of what has been done. ROIAL combined pairwise
preferences with ordinal labels (very bad, bad, neutral, good) for exoskeleton gaits,
arguing that an $r$-level ordinal query yields at most $\log_2 r$ bits against
one bit for a preference; it was tested with 3 participants tuning 4 gait
parameters [@li2021roial]. @wu2025mixed, a preprint, combined the probit
preference likelihood with a Likert confidence likelihood (a Likert item is a
rating on a fixed scale of labeled levels) with learnable cut points and a
lapse rate, and on human comparisons of robot gaits report consistently lower
Brier scores and higher F1 scores, two measures of predictive accuracy. In a
study of vibrotactile feedback, 13 participants made 40 rounds of pairwise
comparisons with a five-level confidence rating that set the noise scale of
each comparison, and the learned model reached a held-out accuracy of 92.3%
(range 85% to 100%) [@zhang2026vibrotactile]. Response times, which can carry
similar information without asking anything extra, are covered with the rest
of these extensions in @sec-obs-extensions.

### It crashed {#sec-query-crash}

Some evaluations produce no design to judge. A controller makes a robot fall,
a simulation diverges, a recipe is inedible. Forcing such an outcome into a
comparison ("the crash is worse than anything") puts it on the utility scale
where it does not belong. The extensions in use give it a second outcome
instead. @benavoli2021preferentialb attach a valid or invalid label to each
evaluation alongside the preferences, for experiments that sometimes cannot
produce an output; C-GLISp learns the probability that a design is feasible
and satisfactory from judgments made one design at a time [@zhu2022c]; and
CrashPBO treats a crash report as a second kind of outcome, reducing crashes by
63% on synthetic benchmarks and validated on three robot platforms
[@menn2026preferential]. The common pattern is two models side by side, one
for the utility of designs that work and one for the probability that a design
works, combined in the acquisition function much as constrained Bayesian
optimization combines an objective with a feasibility model
(@sec-constrained-bo).

## Many people {#sec-many-users}

The crowdsourced line search of @sec-query-crowd took the median of several
workers' slider positions, which treats them as noisy measurements of one
utility. @koyama2020computational state the assumption plainly: "a common
'general' preference exists that is shared among crowds". They also name its
limits: in some domains "crowds from different backgrounds can have clearly
different preferences", and "the user's personal preference is not reflected
in computation when crowdsourcing is the only source of data". Whenever more
than one person answers, the model has to say how their utilities relate.

### Three positions and a kernel that spans them {#sec-query-population-model}

There are three basic positions. *Pool* everyone: one utility, and differences
between people become noise. *Separate* everyone: one independent utility per
person, which wastes everything the others' answers say. Or *share structure*:
each person's utility is the population's plus a personal deviation. A
Gaussian process expresses the third position with one kernel over pairs of
(person, design):

$$
k\big((u, \vx), (u', \vx')\big) = k_0(\vx, \vx') + \mathbb{1}[u = u']\, k_1(\vx, \vx').
$$ {#eq-query-hier-kernel}

The first term is the covariance of a utility $g_0$ shared by everyone; the
second is the covariance of an independent deviation $h_u$ for each person
$u$, so person $u$'s utility is $g_u = g_0 + h_u$. If both kernels have the same shape with signal variances
$v_0$ and $v_1$, two people's utilities at the same design have correlation
$v_0/(v_0 + v_1)$. Setting $v_1 = 0$ pools; setting $v_0 = 0$ separates. A new
person's model then starts from the population's posterior instead of from
the prior, and their own answers gradually override it.

Sharing $g_0$ also puts everyone on one scale, the question
@sec-pref-identifiability left open about comparing utilities across people.
Comparisons fix each person's utility only up to a shift and only in units of
that person's noise, so a model that adds personal deviations to a common
$g_0$ assumes that people are about equally noisy, unless it gives each person
a noise scale of their own (inference).

Richer versions share structure without assuming one common utility. The
collaborative model of @houlsby2012collaborative combines a preference kernel
with low-dimensional structure shared across users; crowdGPPL writes each
person's utility as a weighted combination of a few shared latent functions, a
matrix factorization with Gaussian process factors, and scales to thousands of
users and items with stochastic variational inference [@simpson2020scalable].
A 2026 preprint replaces the single utility with a mixture of latent
preference archetypes [@dubey2026active].

### What population priors buy {#sec-query-population-evidence}

Interactive systems since 2024 have used the population mostly as a prior that
gets a new user started. Meta-PO combines PBO
with meta-learning, which here means using the Gaussian process models fitted
to earlier users to start the search for a new one, and uses the Sequential
Gallery interface; in a study of 36 participants in three groups of
12, the iterations needed to reach a satisfactory result fell from 9.54
(standard deviation 2.19) without transfer to 5.86 (standard deviation 1.20)
when the earlier users had pursued the same theme, and to 7.41 (standard
deviation 1.28) across themes [@li2025efficient]. The more the new user's goal
differs, the less the population helps. HOMI trains an acquisition function represented by a
neural network on simulated users before any real user arrives; with 12
participants and a performance objective, it was better than its baselines
only at the second and third iterations, and from the sixth iteration on all
methods performed at the same level [@liao2026efficient].

Against this stands consistent evidence that people differ a great deal.
When 20 participants with varying design experience judged the same 600 pairs
of generated interfaces, they agreed at only 0.25 on a chance-corrected scale
where 1 is perfect agreement and 0 is what chance would give (Krippendorff's
α; Cohen's κ is the same to two decimals), in a 2026 preprint
[@peng2026efficient].
LineCoSpar found that the utilities behind different users' gait preferences
differ [@tucker2020human]. A population prior speeds up the first iterations,
but no study has shown that it does no harm to a person who differs from the
population (inference). @sec-hci-population collects the human-factors
evidence.

### What comparisons can and cannot recover {#sec-query-aggregation}

Theory adds a warning about pooling. Suppose many people, or one person in
varying unobserved situations, answer pairwise comparisons, and a single
Bradley-Terry utility is fitted to all of it. @siththaranjan2024distributional
proved that, with infinite data on a finite set of options, the fitted
utility ranks options by their
*Borda count*, the average probability that an option beats a random opponent,
and not in general by their expected utility; no method using infinite
pairwise data can always recover the expected-utility order. @sec-dueling-bandits
meets the Borda count again as one of the definitions of a best option. And
@chidambaram2026direct showed that if each person makes only one binary
comparison, the distribution of preferences in the population cannot be
identified at all, while comparisons among three or more options, even
incomplete rankings, can identify it under further conditions. For
heterogeneous populations, pairs are the weakest form of feedback;
@sec-theory-identifiability states these results precisely.

## The interface is part of the model {#sec-interface-model}

Every likelihood in this chapter is a model of a person doing something with an
interface. @eq-query-luce models clicking the best of a set; the slider record
models dragging and stopping; @eq-query-threshold models a third button. Change
the interface and the noise changes, the information per answer changes, the
effort changes, and sometimes the preference being measured changes too.

The evidence on how people use these interfaces makes the point concrete. With
one slider, an iteration of generative image search took 17.2 seconds; with
four sliders, 53.4 seconds, though the four-slider version converged in fewer
iterations and participants preferred its flexibility; that study had three
participants [@chong2021interactive]. In virtual-reality color grading, users
preferred to compare two options rather than four [@yuan2025personalized]. A
prompt-optimization system that asked only for binary preferences converged
faster and with lower workload than its baselines, but was less expressive than
baselines that accepted text feedback or manual edits [@li2026preference].
Critics of gallery interfaces point out that every option is still chosen by the
optimizer and the person can express nothing beyond liking or disliking it
[@mo2024cooperative].

The answers also depend on what the person has already seen: in the field
deployment described in @sec-cmp-ratings-hard, the experts' ratings were
anchored on meshes seen earlier and grew noisier as results improved
[@ou2022human]. A system that learns from ordinary slider edits
instead of explicit queries rests on the assumption that each edit seeks a
better design, and its authors note that they have not evaluated when that
holds [@koyama2022bo]. Even the parameterization is part of the interface: a
2026 preprint argues that the representation shown to the user is a core part
of interaction design rather than a preprocessing step, and with 40
participants found a five-dimensional learned design space better than the
original nine procedural parameters [@owaki2026feasibility].

These findings suggest a short list of questions to answer before choosing a
query form (inference, from the sections above):

- **What does the person do, and what does it cost them?** Seconds per answer,
  and how many answers before fatigue.
- **What is recorded?** Which likelihood turns the action into evidence, and
  what information does the record discard?
- **What noise does the action add?** Perceptual resolution, motor precision,
  and the effort that pushes people toward an answer that is merely good
  enough.
- **What happens when the person cannot answer?** A tie, a skip, a crash, or a
  forced guess that the model will misread.
- **Whose preference is it?** One person, a population, or one person in
  changing situations.

@sec-hci follows the human side of these questions through the HCI literature,
and @sec-rec-query turns them into recommendations.

::: {.frontier title="Settled, contested, missing"}
**Settled.** Choices from sets and rankings have standard likelihoods, Luce's
choice model and Plackett-Luce, with Gaussian process versions. Sliders,
galleries, and projections beat pairs in their own papers' simulations. In
finite-option theory, reporting only the winner of a larger set does not
improve on pairs in the worst case, while reporting the top $m$ does.

**Contested.** Whether queries of more than two options help in practice: the
two direct experiments disagree, with different acquisition functions. Whether
population priors are safe for people unlike the population.

**Missing.** Controlled human studies comparing query forms under the same
acquisition function and budget; acquisition functions derived for slider,
plane, and projection queries; an abstain answer distinct from a tie; models of
how the interface's own noise (motor, perceptual, effort) enters the
likelihood.
:::

## Exercises {#sec-query-exercises}

::: {.exercise #exr-query-pl}
(a) Show that the Plackett-Luce model @eq-query-pl with $q = 2$ is the
Bradley-Terry model. (b) For four options with utilities $g = (2, 1, 0, 0)$ and
$\tau = 1$, compute the probability that the person ranks them in the order
listed, first to fourth, and the probability that they pick the first as the
best.
(c) Why is the ranking probability so much smaller, and why does that not make
the ranking less informative?

::: {.solution}
(a) With $q = 2$ the product has two factors. The second is a choice from a set
of one and equals 1, so

$$
\Prob(\vx_1 \succ \vx_2) = \frac{e^{g_1/\tau}}{e^{g_1/\tau} + e^{g_2/\tau}} = \frac{1}{1 + e^{-(g_1 - g_2)/\tau}},
$$

the Bradley-Terry probability.

(b) The exponentials are $e^2 \approx 7.39$, $e^1 \approx 2.72$, $1$, $1$, with
sum 12.11. The first place has probability $7.39/12.11 \approx 0.61$, which is
also the probability of picking the first as best. Given that, the second place
has probability $2.72/(2.72 + 1 + 1) \approx 0.58$, and the third
$1/(1 + 1) = 0.5$. The ranking has probability
$0.61 \times 0.58 \times 0.5 \approx 0.18$.

(c) A ranking is one of $4! = 24$ possible answers, a winner one of 4, so each
particular ranking is less likely. Low probability of each answer is what lets
an answer carry more bits: the observed ranking rules out many more
alternatives than the observed winner does.
:::
:::

::: {.exercise #exr-query-biyik}
Show that the three probabilities in @eq-query-biyik sum to one, and that
$\Prob(\text{same})$ is largest when $\Delta = 0$. What is its value there for
$\delta = 1$, the setting @byk2019asking used in their simulations?

::: {.solution}
Write $A = e^{\delta - \Delta}$ and $B = e^{\delta + \Delta}$, so
$\Prob(a) = 1/(1 + A)$, $\Prob(b) = 1/(1 + B)$, and $AB = e^{2\delta}$. Then
$\Prob(a)\Prob(b) = 1/(1 + A + B + e^{2\delta})$ and
$\Prob(a) + \Prob(b) = (2 + A + B)\,\Prob(a)\Prob(b)$. Adding
$(e^{2\delta} - 1)\Prob(a)\Prob(b)$ gives
$(1 + A + B + e^{2\delta})\,\Prob(a)\Prob(b) = 1$.

For the maximum, $\Prob(\text{same})$ is proportional to
$1/(1 + A + B + e^{2\delta})$, and $A + B = 2e^{\delta}\cosh\Delta$ is smallest
at $\Delta = 0$. There $A = B = e^{\delta}$ and

$$
\Prob(\text{same}) = \frac{e^{2\delta} - 1}{(1 + e^{\delta})^2} = \frac{e^{\delta} - 1}{e^{\delta} + 1}.
$$

For $\delta = 1$ this is $(e - 1)/(e + 1) \approx 0.46$: when the two options are
in fact equal, the modeled person says "about the same" a little under half the
time.
:::
:::

::: {.exercise #exr-query-hier}
Under the kernel @eq-query-hier-kernel with $k_0 = v_0\,\kappa$ and
$k_1 = v_1\,\kappa$ for a common correlation function $\kappa$ with
$\kappa(\vx, \vx) = 1$, show that the correlation between two different
people's utilities at the same design is $v_0/(v_0 + v_1)$. A new person
arrives and answers nothing. What does the model predict for their utility,
and what is its variance compared with that of the population utility $g_0$?

::: {.solution}
For $u \ne u'$ the covariance is $k_0(\vx, \vx) = v_0$ and each variance is
$v_0 + v_1$, so the correlation is $v_0/(v_0 + v_1)$. For a new person $u$,
$g_u = g_0 + h_u$ where $h_u$ is independent of all data. Its posterior mean is
the posterior mean of $g_0$, the population's estimate, and its posterior
variance is the posterior variance of $g_0$ plus $v_1$: however much data the
population provides, the new person's utility stays uncertain by at least their
personal variance until they answer themselves.
:::
:::

::: {.exercise #exr-query-slider-record}
In @fig-query-line-search, choose an end of the slider without moving it. What
does the model record? Then argue that the record "chosen beats both ends"
discards information that the slider answer contains, and describe one way to
keep more of it within @eq-query-luce.

::: {.solution}
If the person keeps the end $\vx^+$, the chosen point coincides with it, the set
of three collapses to two, and the record is the pair $\vx^+ \succ \vx^{\EI}$;
keeping the other end records $\vx^{\EI} \succ \vx^+$. In general the person
preferred the chosen point to every point on the slider, a continuum of
comparisons, while the record keeps two. One way to keep more is to add $m$
further points of the segment to the losing set, for example evenly spaced
ones, so the choice becomes a choice from $m + 3$ options in @eq-query-luce.
This costs computation, since the latent vector grows with every point, and the
added points sit close together on one line, so their comparisons are strongly
correlated and add less than their number suggests. @mikkola2020projective
take the limit of many points with Gaussian noise instead.
:::
:::

## Further reading {#further-reading .unnumbered}

- @koyama2017sequentialb introduce sequential line search; the book chapter
  @koyama2020computational explains the crowdsourcing design, the microtask
  choices, and the "whose preference" question in plain terms.
- @koyama2020sequential extend the slider to a zoomable gallery on a plane and
  report both simulations and a small user study.
- @mikkola2020projective derive the likelihood of a projective answer as a
  continuum of comparisons and test it with materials scientists.
- @siivola2021preferential and @nguyen2021top give likelihoods for batch
  winners, rankings, and ties; @benavoli2026tutorial surveys nine likelihoods
  for preferences and choices with Gaussian processes.
- @byk2019asking add an "About Equal" answer and a stopping rule to active
  preference learning.
- @houlsby2012collaborative and @simpson2020scalable model many users' preferences
  with shared structure; @siththaranjan2024distributional show what a single
  Bradley-Terry utility recovers from a mixed population.
- @luce1959individual and @plackett1975analysis are the classical sources for
  choice from sets and for rankings.
