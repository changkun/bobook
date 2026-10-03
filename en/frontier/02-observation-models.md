---
status: done
synopsis: "What the likelihood assumes about a human answer, which surrogates replace the Gaussian process and why, how much the inference approximation matters, and what the default implementation actually does."
---

# Observation Models, Surrogates, and Inference {#sec-observation-models}

Every preferential Bayesian optimization (PBO) system rests on three modeling
choices, which @sec-part-preferences introduced one at a time. The **observation
model** says how a comparison, a ranking, or a choice arises from the latent
utility: in @sec-comparisons, a probit or logistic link applied to a utility
difference. The **surrogate** is the prior over the utility: in
@sec-gp-preference, a Gaussian process. **Posterior inference** combines the
two and hands the result to the acquisition function: in @sec-approx-inference,
usually the Laplace approximation.

From 2017 to September 2026 the default combination barely changed. A Gaussian
process prior, a probit link, and the Laplace approximation, which is the model
of @chu2005preference, remain what BoTorch's `PairwiseGP` implements
[@botorch2026pairwisegp]. What changed is the scrutiny each layer received, and
this chapter follows the three layers in order, ending with what the default
implementation decides for you.

## The baseline models {#sec-obs-baseline}

@chu2005preference placed a Gaussian process prior on
the latent utility $f$ and used the likelihood

$$
\Prob(\vx \succ \vx') = \Phi\!\left(\frac{f(\vx) - f(\vx')}{\sqrt{2}\,\sigma}\right),
$$ {#eq-obs-probit}

which is Thurstone's comparative judgment with Gaussian noise of scale $\sigma$
on each option (@sec-thurstone). They fitted the posterior with the Laplace
approximation and chose hyperparameters by the approximate evidence. The
pairwise model of @brochu2007active
is the same Thurstone model, as @koyama2020sequential point out.
@houlsby2011bayesian took another route to the same place: a
*preference kernel* that turns pairwise preference learning into Gaussian
process classification over pairs.

The paper that named the field used neither the probit link nor, as far as its
text says, the Laplace approximation. @gonzalez2017preferentialb modeled the
probability that $\vx$ wins a duel against $\vx'$ with a Gaussian process
classifier on the dueling space $\X \times \X$, with a Bernoulli likelihood and
the latent joint reward squashed by a **logistic** function, assuming the
latent function has the form $f([\vx, \vx']) = g(\vx') - g(\vx)$; their text
says only that the posterior is intractable and needs approximations, without
naming one. They listed the doubled input dimension as a limitation, and
@nguyen2021top later criticized the model for working in twice the dimension of
the objective, for not modeling the objective directly, and for not modeling
ties. Because the skew Gaussian process theorem of @sec-obs-inference concerns
the probit likelihood, it does not apply directly to this original model
(inference).

Since then the two links have been used side by side. BoTorch offers both:
`PairwiseProbitLikelihood` with $\Prob(v \succ u) = \Phi((f(v) - f(u))/\sqrt{2})$
and `PairwiseLogitLikelihood` with $\operatorname{sigmoid}(f(v) - f(u))$
[@botorch2026likelihood].
The logistic link appears in the noise analysis of qEUBO
[@astudillo2023qeubob], in POP-BO
[@xu2024principledb], and in MR-LPF
[@kayal2025bayesian]. For two options, a
softmax over utilities is exactly the logistic Bradley-Terry model. Both links
are random utility models (@sec-random-utility): the probit one comes from
Gaussian noise on each option, and the logistic one from Gumbel noise, a
correspondence Kayal et al. use when they discuss lower bounds.

::: {.keyidea title="The link was inherited, not chosen"}
We found no PBO paper that compares the probit and logistic links on human
data, and no study of robustness when the link is wrong. The only data-driven
alternative comes from @shvartsman2024response: a drift-diffusion model of the
decision process induces its own choice link, but the gains they measured
appeared after adding response-time data and cannot be credited to the link
itself (inference). The choice of link is a matter of lineage and convenience,
not of evidence (inference). Until a comparison exists, a practitioner can
treat the link as a checkable choice: fit both to the answers collected and
compare how well each predicts held-out comparisons (inference).
:::

## Extending the observation model {#sec-obs-extensions}

A person's answer to "which is better?" can carry more, or different,
information than one bit. Since 2017, researchers have extended the likelihood
in eight directions. @tbl-obs-extensions summarizes them; the paragraphs after
it give the details that matter for judging each one.

::: {.table #tbl-obs-extensions title="Extensions of the observation model since 2017, and the human evidence for each."}
| Extension | Representative work | Likelihood | Human data |
|---|---|---|---|
| Choice from a set, batch winner, ranking | [@koyama2017sequentialb; @koyama2020sequential; @siivola2021preferential; @nguyen2021top; @benavoli2023choice] | Bradley-Terry for $m$ options; batch-winner likelihood; multinomial logit and top-$k$ rankings; choice functions with several utilities | crowdsourcing and a 6-person pilot; benchmarks built from rating data; simulations only for Benavoli et al. |
| Ties and indifference | [@byk2019asking] (linear reward); [@nguyen2021top; @benavoli2026tutorial; @erarslan2025consecutive] | just-noticeable-difference threshold; multinomial logit with ties; indistinguishability $\lvert u(\vx) - u(\vx')\rvert / \delta \le 1$ | only Bıyık et al.'s user study, with a linear model |
| Ordinal labels and confidence | ROIAL [@li2021roial]; [@dao2025experience; @wu2025mixed; @peng2025uncertainty; @zhang2026vibrotactile] | ordered thresholds; 5-level Likert strength; Likert confidence with cut points and lapse rates | 3 participants (ROIAL); offline fits (Wu et al.); a user study (Peng et al.); $N = 13$ (Zhang et al.) |
| Response times | [@shvartsman2024response]; [@li2024enhancing] (linear) | differentiable approximation to the drift-diffusion likelihood | three data sets, offline |
| Failure, validity, crashes | [@benavoli2021preferentialb]; C-GLISp [@zhu2022c]; CrashPBO [@menn2026preferential] | valid/invalid label joint with preference; feasibility surrogate; crash as a second outcome | controller calibration; three robot platforms |
| Intransitivity | [@chau2022inconsistent] | Gaussian process on a skew-symmetric preference function | animal contests, NFL games, citations; no design preferences |
| Heteroscedastic noise | [@sinaga2024anchor] | input-dependent noise from a kernel density over anchors | Sushi and Candy benchmarks, no live users |
| Several users, several utilities | [@houlsby2012collaborative; @simpson2020scalable; @benavoli2023choice; @astudillo2025preferential; @dubey2026active] | collaborative low-rank GP; matrix factorization with GPs; several latent utilities; Dirichlet process mixture | real multi-user data (Houlsby, Simpson); simulations otherwise |
| Utility drifting over time | none | none | none |
:::

**Choices from sets, and rankings.** The sequential line search of
@koyama2017sequentialb records the person's pick along a slider as a
preference for the chosen point over both the current best and the point of
highest expected improvement, using the Bradley-Terry-Luce model, the extension
of Bradley-Terry to a choice among $m$ options; the Sequential Gallery models
pairwise data with the Thurstone model and choices among $m$ options with the
Bradley-Terry-Luce form [@koyama2020sequential]. @siivola2021preferential give
a likelihood for arbitrary parallel feedback on two or more points, focus on
the batch winner, and argue that full rankings are laborious for people.
@nguyen2021top build a surrogate inspired by the multinomial logit and its
ranking extension, interpretable as Gaussian process regression with i.i.d.
Gumbel noise, which handles top-$k$ rankings and ties. @benavoli2023choice
handle set-valued choices (picking, say, three of five options shown, when none
of the three is clearly better than the others) with several latent utilities;
their abstract reports only simulations.

**Ties and indifference.** @byk2019asking added an "About Equal" answer with a
minimum perceivable difference $\delta \ge 0$:
$\Prob(\text{About Equal}) = (e^{2\delta} - 1)\,\Prob(\text{choose }a)\,\Prob(\text{choose }b)$,
which reduces to the softmax model at $\delta = 0$. Their model is a linear
reward, not a Gaussian process, and comes with simulations and a user study.
C-GLISp encodes better, worse, or similar as constraints on a radial basis
function surrogate [@zhu2022c]. The tutorial of
Benavoli and Azzimonti lists nine likelihoods, among them a
just-noticeable-difference model that adds statements of indistinguishability,
$\lvert u(\vx) - u(\vx')\rvert/\delta \le 1$, going back to Luce's 1956 notion
of a discrimination threshold. @erarslan2025consecutive represent
indifference with a just-noticeable-difference threshold in an extended
Thurstone model and report clear gains when 10% to 20% of comparisons are
indifferent. The field has converged on two treatments, a threshold parameter
in a Thurstone or logit model, or ties in a multinomial logit (inference).
**We found no Gaussian process PBO paper with an "abstain" or "skip" outcome
distinct from a tie**, the answer a person gives when they cannot compare the
options at all.

**Ordinal labels, strength, and confidence.** ROIAL [@li2021roial] combines
pairwise preferences with ordinal labels (very bad, bad, neutral, good), arguing
that an $r$-level ordinal query yields at most $\log_2 r$ bits against one bit
for a preference; its participants were 3 non-disabled people tuning 4
exoskeleton gait parameters. @wu2025mixed add to the probit
preference likelihood a Likert confidence likelihood with learnable cut points
and a lapse rate; on human comparisons of robot gaits, they report that models
trained with the confidence ratings "consistently achieve lower Brier scores and
higher F1 scores", that is, smaller squared errors of the predicted
probabilities and more accurate predicted choices.

**Response times.** How long a person takes to answer says something about how
different the options felt. @shvartsman2024response approximated the
drift-diffusion likelihood with a moment-matched skewed three-parameter
distribution so that response times can enter a variational Gaussian process,
and describe this as the first Gaussian process model with a joint likelihood
of drift-diffusion response times and choices. On one participant's judgments
of 1,225 pairs of quadruped robot gaits and on a visual psychophysics data set,
the joint drift-diffusion models predicted held-out choices significantly
better than the choice-only model, especially with small training sets, while a
simpler stacking approach, which gives the choice model the logarithm of the
response time as an extra input, did not improve on choice alone; on a
recommender-system data set every joint model improved, stacking most.
@li2024enhancing
used the EZ-diffusion model for fixed-budget best-arm identification with
linear utilities and proved that, for strong preferences, response times add
information about preference strength.

**Failure, validity, and crashes.** Some experiments produce no output at all.
Validity labels and feasibility surrogates attach a second label to each
sample, and CrashPBO [@menn2026preferential] treats a crash report as a second
kind of outcome, reduces crashes by 63% on synthetic benchmarks, and was
validated on three robot platforms.

**Intransitivity.** @chau2022inconsistent placed a Gaussian process on a
skew-symmetric preference function $g(\vx, \vx')$ with a kernel they call the
generalized preferential kernel, which can represent cycles. On real data the
new model's accuracy against the Chu-Ghahramani model was 0.78 against 0.51 for
chameleon contests, 0.83 against 0.80 for flat lizards, 0.59 against 0.51 for
NFL games from 2000 to 2018 (where linear pairwise logistic regression did
best, at 0.65), and 0.74 against 0.66 for an arXiv citation graph. The authors
conclude that their findings support the conjecture that violations of
rankability, the assumption that one utility orders all items, are ubiquitous
in real preference data. These data sets are ones where intransitivity is
expected (contests and sports), the linear model won on the only human
competition data, and none is a single user's design preference, so carrying
the conclusion over to typical PBO settings lacks support (inference).

**Heteroscedastic noise.** Some comparisons are harder than others.
@sinaga2024anchor ask the user for reliable "anchor" points and turn them into
an input-dependent noise map with kernel density estimation, used in the probit
likelihood
$\Phi\big((f(\vx) - f(\vx'))/\sqrt{\sigma^2(\vx) + \sigma^2(\vx')}\big)$ or a
logistic likelihood with input-dependent temperature; their consistency
analysis holds only under an idealized i.i.d. anchor model, and the evaluation
uses the Sushi and Candy data without live users. @fauvel2021efficient argue
that acquisition functions for binary and preferential Bayesian optimization
must separate epistemic from aleatoric uncertainty, since only the former
should drive exploration.

**Several users, and drift.** Models for many users share low-dimensional
structure across them, and crowdGPPL [@simpson2020scalable] scales a
combination of matrix factorization and Gaussian processes to thousands of
users and items; @sec-many-users describes these models. We found no PBO or
Gaussian process preference paper that models a utility changing over time; the
finite-arm theory that does exist is in @sec-theory-drift.

**What the human evidence supports.** For ties there is one user study with a
linear model; for ordinal labels, three participants; for confidence, one
offline robot-gait data set and two small user studies; for response times,
three data sets, few participants, and offline fits; for crashes, three robot
platforms; for heteroscedastic noise, two benchmarks converted from rating
data; for intransitivity, no design preferences at all. Most evaluations
compare model fits offline rather than closing the loop with people
(inference). Of the signals that cost the person nothing extra, response times
and confidence are the two that have improved models on real human data, which
makes them the best-supported upgrades; recording both in every session costs
nothing and lets them be tested in closed-loop PBO later (inference).

## Surrogates beyond the Gaussian process {#sec-obs-surrogates}

The Gaussian process utility remains the workhorse, but four families of
alternatives appeared, each for a specific reason.

**Gaussian process variants.** @byk2020active fixed $f(\bar{\Psi}) = 0$ at a
reference point through the kernel, removing the additive non-identifiability
of @sec-pref-identifiability in the prior. The skew Gaussian process line was
built by Benavoli, Azzimonti, and Piga: a 2020 classification paper introduced
the skew Gaussian process and proved it conjugate to the probit likelihood
[@benavoli2020skew], a 2021 paper extended conjugacy to normal and affine
probit likelihoods and their products [@benavoli2021unified], and
@benavoli2024linearly showed that a Gaussian process with linear inequality
constraints is a skew Gaussian process, using it for monotone preference
learning.

**Radial basis functions, neural networks, and trees.** GLISp
[@bemporad2021global] fits a radial basis function surrogate by linear or
quadratic programming so that it satisfies the observed preferences as far as
possible, and reports that, with the same number of comparisons, it usually
gets closer to the optimum than PBO at lower computational cost; GLISp-r
[@previtali2023glisp] adds a proof of global convergence. These surrogates are
not probabilistic, so the acquisition functions of @sec-pbo do not apply to
them. Neural dueling bandits [@verma2025neural] estimate rewards from
preference feedback with a neural network and prove sublinear regret, on
synthetic data, and @wang2025bayesian replace the Gaussian process utility with
an ensemble of monotone neural networks in preference exploration. DT-PBO
[@leenders2025dt] builds shallow decision trees directly from comparisons, with
a Laplace approximation in each leaf, and its authors report convergence
competitive with Gaussian process PBO on 8 benchmark functions, particularly on
rugged ones.

**Amortized surrogates.** PABBO [@zhang2025pabbob] starts from the observation
that the non-conjugate likelihood makes every PBO step expensive. It
meta-learns surrogate and acquisition together in a transformer neural process,
trained with reinforcement learning plus an auxiliary preference-prediction
loss on synthetic tasks such as Gaussian process samples. Its abstract claims
it is "several orders of magnitude faster than the usual Gaussian process-based
strategies"; the paper reports a 12-fold average speedup over the fastest
Gaussian process strategy on five problems and about 10-fold over noisy batch
expected improvement on Gaussian process tasks. Its default evaluation
configuration uses noise-free comparisons [@zhang2025pabboc]. The authors list
as limitations that the query set grows with dimension, each dimension needs
its own model, how to build the pretraining data is open, and the person making
comparisons is not modeled.

**Priors across users, and generative latent spaces.** Cross-user structure has
been implemented through collaborative models such as crowdGPPL; through
transfer at the acquisition level, as in Meta-PO [@li2025efficient], which
stores earlier users' sessions as separate Gaussian processes and transfers them
with a rank-weighted acquisition function (@sec-hci-population reports its
user study); and through guidance models trained on population data, such as
the deep encoder of @granley2023human. Several systems also optimize in the
latent space of a generative model, such as FontCraft [@tatsukawa2025fontcraft]
in a font-style space and GimmBO [@liu2026gimmbo] over the merging weights of
20 to 30 diffusion adapters; none of these papers models how the geometry of
the latent space relates to the utility's lengthscale (inference).

Overall, the debate about surrogates since 2020 has been mainly about the
quality of inference in the same model, Gaussian approximation against the exact
skew Gaussian process, not about the model class. Changes of model class were
driven by structure (monotonicity, intransitivity, several utilities) or speed
(PABBO, GLISp), not by evidence that a Gaussian process utility prior is wrong
for human preferences (inference).

## Posterior inference {#sec-obs-inference}

@sec-approx-inference introduced the approximations, and
@sec-approximation-matters reported the main evidence on how well they do; this
section adds the details behind that evidence and the choices the libraries
make. @tbl-obs-inference lists where each method is used.

::: {.table #tbl-obs-inference title="Inference methods for preference models: where they are used and the problems recorded."}
| Method | Used by | Recorded problems |
|---|---|---|
| Laplace approximation | Chu and Ghahramani 2005; BoTorch `PairwiseGP`; Bıyık et al. 2020; ROIAL 2021; projective PBO 2020 | the mode can be far from the mean, worse at low noise; rank-deficient likelihood Hessian with EUBO (preprint) |
| Expectation propagation | Houlsby et al. 2012 (with variational Bayes); Siivola et al. 2021; Fauvel and Chalk 2021; optuna-dashboard | accurate means and credible intervals; duel probabilities near 0 or 1 distorted, relative order not preserved |
| Variational inference | Simpson and Gurevych 2020; Nguyen et al. 2021; Benavoli et al. 2023; Shvartsman et al. 2024; Wu et al. 2025; qEUBO experiments | no systematic error assessment for preference likelihoods |
| Exact skew GP sampling | Benavoli et al. 2021 (LinESS); Takeno et al. 2023 (Gibbs) | every prediction needs truncated multivariate normal sampling; the evidence needs high-dimensional normal CDFs |
| Hallucination believer | Takeno et al. 2023 | a single truncated-posterior sample; stuck in local optima under logistic noise (reported by POP-BO) |
| Amortization | PABBO 2025 | fixed dimension; depends on pretraining data |
:::

**The Laplace approximation is the library default, mostly for speed.**
@byk2020active wrote of expectation propagation that "while it is more
accurate than Laplace approximation, it is slower in practice", and chose
Laplace; projective PBO [@mikkola2020projective] uses it "for the sake of
simplicity". Koyama et al. use maximum a posteriori estimates with a tight
log-normal prior on each hyperparameter, written $\mathrm{LN}(\mu, 0.01)$ in
the paper, with $\mu = 0.5$ for every lengthscale and $\mu = 0.2$ for the
amplitude, which nearly fixes the lengthscale [@koyama2020sequential].
Expectation propagation is the inference method in Siivola et al.'s
experiments on real data, and the preferential Gaussian process sampler of
optuna-dashboard uses it to fit its hyperparameters [@optuna2026sampler].

**The exact posterior is a skew Gaussian process.** @benavoli2021preferentialb
proved this in their PBO paper (@thm-theory-skewgp) and sampled the posterior
with LinESS, rejection-free elliptical slice sampling for linearly truncated
Gaussians, at a cost they give as $O(n^3)$ time and $O(n^2)$ memory, the same
bottleneck (a Cholesky factorization) as an ordinary Gaussian process; they
report consistently better convergence and computation time than Laplace-based
PBO.

**How wrong the Gaussian approximations are.** @takeno2023practicalc took Gibbs
sampling as ground truth (10,000 samples after 1,000 burn-in, thinned by 10),
with an RBF kernel, noise variance $10^{-4}$, and uniformly random duels. Beyond
the findings of @sec-approximation-matters, that the Laplace mode can be very
far from the mean at small noise and that expectation propagation is accurate
for means and credible intervals but distorts duel probabilities, they report
that on the Ackley function expectation propagation underestimates
probabilities near 0 or 1 and does not preserve the true ordering, which
affects acquisition functions that depend on the joint distribution of two
points, and that Gibbs sampling mixed faster than LinESS. They consider
unrealistic the setting in which Benavoli et al. compared expectation
propagation with the skew Gaussian process, where all inputs in one interval
lose and all inputs in another win; an earlier study of binary Gaussian process
classification had found expectation propagation accurate
[@kuss2005assessing]. And they reject Laplace for the posterior yet use the
Laplace evidence for hyperparameters, so the accuracy needed for
hyperparameters and for acquisition is being treated separately.

Takeno et al.'s strongest evidence comes from noise variance $10^{-4}$, where
the posterior is nearly a truncated Gaussian; with noisier comparisons, closer
to people, the skew should weaken. As of September 2026, no paper measures how
large the errors of Laplace and expectation propagation are at realistic human
noise levels, or replicates Takeno et al.'s calibration on real human
comparisons. The experiment is cheap to describe: take comparisons from real
people, run exact sampling as ground truth, and measure how far each
approximation's predicted duel probabilities fall from it.

## Inside the default implementation {#sec-obs-pairwisegp}

Most people who run PBO run BoTorch's `PairwiseGP`, so its defaults are, in
effect, the field's defaults. @tbl-pref-botorch listed the choices inside
version 0.18.1 [@botorch2026pairwisegp; @botorch2026likelihood]. Three of them
shape what the model can learn. The probit likelihood is
$\Phi((f(v) - f(u))/\sqrt{2})$, which fixes the noise at 1, so the output
scale (BoTorch's name for the kernel's amplitude $\sigma_f^2$) acts as the
inverse of the noise level: it and the lengthscale must be learned jointly from
comparisons alone, with no separate noise parameter, under a smoothed box prior
on $[0.01, 100]$ and a constraint of $[0.005, 200]$ that a source comment calls
a rule of thumb to keep estimates away from scales that saturate $\Phi$. The
probit argument is clipped to $[-3, 3]$, which caps the likelihood of any
single comparison at about 0.9987 during fitting, an implicit robustness to
mislabeled answers (@sec-pairwisegp). And the lengthscale prior is
Gamma(2.4, 2.7), initialized at its mode, in every dimension.

The changelog adds the history [@botorch2026changelog]. Version 0.9.0 (August
2023) added a pairwise Bayesian active learning by disagreement acquisition
function; 0.10.0 (February 2024) added qEUBO; 0.12.0 (September 2024) switched
most models to lengthscale priors that scale with dimension but explicitly
excluded `PairwiseGP`; 0.18.0 (June 2026) fixed `PairwiseGP`'s state handling in
evaluation and cross-validation; and in 0.18.1 (June 2026) `PairwiseGP` still
uses Gamma(2.4, 2.7). The dimension-scaled priors came from a study of scalar
Bayesian optimization by @hvarfner2024vanilla, the subject of
@sec-high-dimensions. The preferential sampler in optuna-dashboard, based on
Takeno et al., uses a Matérn 3/2 kernel with one lengthscale per dimension and a
Gamma(5, 10) prior (mode 0.4), also independent of dimension
[@optuna2026sampler].

What the fixed prior does in high dimension can be computed (inference). The
mode of Gamma(2.4, 2.7) is $(2.4 - 1)/2.7 \approx 0.52$ in every dimension,
while the dimension-scaled prior's mode on the unit cube grows from about 0.65
at 10 dimensions to about 2.05 at 100. Since random points in $[0, 1]^d$ lie
about $\sqrt{d/6}$ apart (@sec-hd-seeing derives this), the default preference
model in high dimension works where kernel values between typical points are
near zero, and every comparison is nearly uninformative about every other
point. This has not been verified directly on `PairwiseGP`.

## Comparison graphs and conditioning {#sec-obs-graphs}

The numerical trouble reported in 2026 has a linear-algebra core that
@sec-comparison-graph derived: the Hessian $\mW$ of the negative
log-likelihood is the weighted **graph Laplacian** of the comparison graph,
whose nodes are compared inputs and whose edges are answered pairs, and it has
one zero eigenvalue for every connected component. The Laplace posterior
precision $\mK^{-1} + \mW$ is always full rank, because the prior contributes
$\mK^{-1}$, so the practical problem is not singularity but poor conditioning
when the prior is weak along the shift directions, that is, when lengthscales
are long or the output scale is large. The same fact has a statistical face,
which @fig-comparison-graph shows: a shift direction that no comparison
constrains is a relative utility that no answer has measured, and only the
prior can say anything about it.

```{figure}
//| figure: comparison-graph
//| label: fig-comparison-graph
//| fig-cap: "What the comparison graph lets the answers determine. Nodes are compared inputs, edges are answered duels, colors are connected components. *Isolated pairs* mimics queries that share no input with earlier ones, as EUBO tends to choose; *Chain*, *Star*, and *Random pairs* spend the same number of comparisons. Below, the eigenvalues of the likelihood Hessian W (unit curvature per comparison). Every zero is a direction no answer constrains: the first is the global shift, which never matters, and each further one (red) is an offset between two groups of inputs that were never compared with each other, which only the prior can fill in. The designs are illustrative."
```

@shao2026adaptive, a 2026 preprint, report that in the pipeline of
`PairwiseGP`, the Laplace approximation, and EUBO, the new pairs EUBO selects
share no candidate with earlier queries, so each pair forms an isolated
component of the comparison graph and the likelihood Hessian becomes rank
deficient. They call the deficiency structural, one that "cannot be resolved by
changing the surrogate modeling approach", and note that existing remedies
either force comparisons to stay connected, wasting query budget, or apply
uniform regularization that perturbs well-constrained directions. Their
correction, whose size @sec-comparison-graph reports, adds to the Hessian's
diagonal a term scaled by the prior uncertainty, only during model fitting; an
adaptive version applies it only when the surrogate is confident, and the
benchmarks include a 16-dimensional plasma-medicine controller.
@pukdee2026preference give conditions under which the Bradley-Terry model still
recovers the conditional preference distribution when the data violate it, and
show that the margin and the connectivity of the comparison graph govern sample
efficiency. Lemma 1 of local PBO [@menn2026local] shows that gradient and
Hessian estimates taken from the Laplace posterior inherit the bias of its
mode; the authors add that the Hessian estimate can be especially sensitive to
the lengthscale and to the conditioning of the kernel matrix, and their
experiments bound the lengthscale to $[0.05, 0.5]$ on the unit cube.

**How hyperparameters are learned in practice.** The approaches we found range
from a weak Gamma prior with the Laplace evidence (BoTorch) and the exact skew
Gaussian process evidence (Benavoli et al. 2021) to a tight log-normal prior
that nearly fixes the lengthscale (Koyama et al.) and box constraints (local
PBO). Whether any of them can identify lengthscales from the tens to one or two
hundred comparisons a person gives is unstudied, and @sec-hd-learning-ls
states the gap and the simulation that would close it.

## Settled, contested, missing {#sec-obs-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** Under a probit preference likelihood and a Gaussian process
prior, the exact posterior is a skew Gaussian process
[@benavoli2021preferentialb]. The original model of González et al.
used a logistic link and did not state its inference method. The preference
models in BoTorch and optuna-dashboard use lengthscale priors that do not scale
with dimension. The Laplace likelihood Hessian is rank deficient whenever the
comparison graph is disconnected, a fact of linear algebra; in practice it shows
up as poor conditioning rather than singularity (inference).

**Contested.** Whether the error of Gaussian approximations at human noise
levels is large enough to change optimization outcomes: the evidence of
Benavoli et al. and Takeno et al. comes from low noise or specific settings, and
they disagree about which settings are realistic. Whether the ill-conditioning
that the 2026 preprint calls structural causes measurable loss on human data.
Whether intransitivity is common: Chau et al.'s data are not design
preferences, and a linear model did better on the NFL data.

**Missing.** A comparison of link functions on human data, and robustness to a
wrong link. An observation model with an abstain outcome. A model of drifting
utility. Evidence on whether lengthscales are identifiable at human-scale
budgets. A test of dimension-scaled priors with a pairwise likelihood. A
closed-loop human experiment comparing Laplace, expectation propagation, exact
skew Gaussian process sampling, and the hallucination believer. A prior-fitted
network trained on pairwise data. Beyond the probit link with constant noise,
models of how people answer remain an open design space of unvalidated defaults
(inference).
:::

## Further reading {#further-reading .unnumbered}

- @benavoli2026tutorial, the tutorial by Benavoli and Azzimonti,
  is the most complete single treatment of Gaussian process models for
  preferences and choices, including nine likelihoods.
- @benavoli2021preferentialb and
  @takeno2023practicalc are the two sides of the
  inference debate; read them together.
- The source of `PairwiseGP` [@botorch2026pairwisegp]
  is short and worth reading before trusting its defaults.
- @shvartsman2024response is the clearest
  example of adding a free behavioral signal, response time, to the likelihood.
