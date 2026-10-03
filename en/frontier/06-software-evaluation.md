---
status: done
synopsis: "The maintained software for PBO and the defaults it ships, why research code is hard to rerun, how methods are evaluated with simulated users and why the choice of metric decides the winner, what changes when real people answer, and who does this research, in which disciplines, and how much of it there is."
---

# Software, Evaluation, and the Research Community {#sec-software-evaluation}

The previous chapters reported what methods for preferential Bayesian
optimization (PBO) claim. This chapter turns to the conditions under which
those claims were made and can be used: the software that implements the
methods, the defaults it chooses on the user's behalf, the way papers evaluate
one method against another, and the people who do the work.

The picture as of September 2026 has three parts. There is one maintained,
general-purpose software stack for PBO, Meta's BoTorch and Ax; the rest is
mostly code released with single papers. Evaluation relies almost entirely on
simulated users answering comparisons about synthetic test functions inherited
from scalar optimization. And the research community is small and spread over
machine learning, control, robotics, and human-computer interaction, which
publish in different venues and, in the cross-citations we checked, rarely cite
one another.

## Software {#sec-sw-software}

A reader who wants to run PBO today has to choose software, and the choice is
narrower than the size of the Bayesian optimization ecosystem suggests. Dates
below are those in the projects' changelogs (the files in which a project lists
what changed in each version) or, where a project has no changelog entry,
upload dates on the Python Package Index (PyPI); the two can differ by a day.

### BoTorch {#sec-sw-botorch}

BoTorch is Meta's library of Gaussian process models and acquisition functions
built on PyTorch; @sec-pairwisegp and @sec-obs-pairwisegp described its
preference model, `PairwiseGP`. Its preference features arrived between 2020 and
2024 [@botorch2026changelog]:

- 0.2.3 (April 27, 2020) added `PairwiseGP` for pairwise comparison data.
- 0.3.2 (October 2020) removed its noise term and added a `ScaleKernel` by
  default.
- 0.6.3 (March 28, 2022) added `LearnedObjective`, the analytic form of EUBO
  (the expected utility of the best option, @sec-eubo), and a tutorial on
  Bayesian optimization with preference exploration (BOPE), in which a person
  states preferences over the outcomes of experiments rather than over designs.
- 0.6.5 (July 2022) added a logistic link likelihood and switched the PBO
  tutorial to EUBO.
- 0.9.0 (August 1, 2023) added pairwise BALD (Bayesian active learning by
  disagreement, which picks the query whose answer the model's plausible
  hypotheses disagree about most).
- 0.10.0 (February 26, 2024) added qEUBO, the form of EUBO for queries of $q$ options.

Since then the preference code has received only fixes: in 0.18.0 (June 3,
2026), a fix to how `PairwiseGP` manages its stored state and a numerically
stabler mixture entropy in BALD. Version 0.18.1 (June 8, 2026) is the latest
release, and the last commit to the main branch was on September 30, 2026
[@botorch2026pypi]. The preference acquisition functions live in one source file
[@botorch2026preference].

BoTorch's PBO tutorial is the first code most newcomers run, so its design is
worth knowing. It uses a 4-dimensional linear utility, adds Gaussian noise with
standard deviation 0.1 to the utilities before comparing, evaluates the model
with Kendall's rank correlation (the fraction of pairs that two rankings order
the same way, rescaled to run from $-1$ to $1$), and compares EUBO with random
queries over only 3 repetitions [@botorch2026pairwise].

### Ax {#sec-sw-ax}

Ax is Meta's platform for managing experiments, which calls BoTorch for its
models. A pairwise model bridge, `PairwiseModelBridge`, first appears in the
installed package of Ax 0.2.6 (August 17, 2022) [@ax2022platform]. Ax 1.0.0 was
uploaded to PyPI on May 8, 2025, with a companion paper by @olson2025ax (AutoML
2025). Ax's preference features arrived in 2026 [@ax2026changelog]:

- 1.2.2 (January 2026; PyPI has no files for this version) added a
  `PreferenceOptimizationConfig` with storage support and a Kendall rank
  correlation diagnostic.
- 1.2.3 (February 19, 2026) added BOPE utility traces based on `PairwiseGP`, and
  `LLMProvider` and `LLMMessage` abstractions for language models.
- 1.3.0 (June 4, 2026) added LILO labeling trials with a check that the labels
  are fresh, automatic dispatch of qEUBO for PBO, `PairwiseGP` inside
  `ModelList`, and a utility ranking plot. LILO turns free-text feedback into
  pairwise labels with a language model (@sec-llm-in-loop).
- 1.3.1 (June 9, 2026) is the latest release and requires Python 3.11 or later
  [@ax2026platform].

Ax's changelog begins only with 1.1.0 (August 2025), so the earlier history
comes from the installed packages. The main branch (last commit September 29,
2026) has no tutorial on preferences, BOPE, or LILO [@ax2026tutorials], and
the LILO logic exists only in code [@ax2026transition].

### Every package with preference support {#sec-sw-packages}

@tbl-sw-packages lists the software with preference support that we found;
the links behind each name are code repositories or software documentation.
HB-EI and HB-UCB are the hallucination believer of @sec-choosing-pairs with
expected improvement or an upper confidence bound.

::: {.table #tbl-sw-packages title="Software with preference support, with maintenance status at the end of September 2026."}
| Software | Surrogate and inference | Acquisition | Preference features | Latest release | Status |
|---|---|---|---|---|---|
| BoTorch [@botorch2026license] | `PairwiseGP`: probit or logistic likelihood, Laplace approximation, hyperparameters by the Laplace marginal likelihood | EUBO, qEUBO, pairwise BALD, pairwise posterior variance | models for pairwise data, learned objectives, PBO and BOPE tutorials | 0.18.1 (2026-06-08) | active; preference module only fixed since February 2024 |
| Ax | calls BoTorch's `PairwiseGP` | qEUBO dispatched automatically (from 1.3.0) | preference optimization config, BOPE utility traces, LILO labeling trials, utility ranking plot, Kendall rank correlation diagnostic | 1.3.1 (2026-06-09) | active; no preference tutorial |
| AEPsych [@meta2026aepsych] | `PairwiseProbitModel`, inheriting from `PairwiseGP` | we did not check | pairwise psychophysics experiments | 0.8.0 (2025-04-11); main branch 0.8.0+dev | commits on 2026-08-26 (UTC) |
| optuna-dashboard [@optuna2026dashboard] | after Takeno et al. 2023: expectation propagation to fit kernel hyperparameters, Gibbs sampling of the latent utility | log expected improvement maximized on a sampled Gaussian process | web interface; the user marks the worst of several candidates (4 in the tutorial) and every implied pair is recorded; available since 0.13.0b1 (September 2023) | 0.21.0 (2026-09-10) | active; no dynamic search spaces |
| OptunaHub plmbo [@ozaki2026plmbo] | GPy Gaussian processes with an RBF kernel for each objective; preference weights by NumPyro Markov chain Monte Carlo | we did not check | implements the multi-objective BO with active preference learning of @ozaki2024multi (AAAI 2024), with comparisons typed at a console | optunahub 0.5.0 (2026-09-09); plugin declares Optuna 3.6.1 | OptunaHub has no purely preferential sampler |
| pySequentialLineSearch [@koyama2025sequential] (C++ with Eigen and NLopt; experimental Python bindings) | `PreferenceRegressor`: Bradley-Terry-Luce likelihood, Matérn 5/2 kernel, maximum a posteriori hyperparameters | expected improvement (default), GP upper confidence bound | slider-based line search; pairwise comparison demo | tag v0.5 (2025-11-11), `setup.py` still says 0.4; not on PyPI | maintained at low frequency; tested only on macOS 10.15 |
| GLIS, GLISp, C-GLISp [@bemporad2023glis] (Python and MATLAB) | radial basis function surrogate (inverse quadratic by default) with inverse distance weighting; not probabilistic | surrogate minus an inverse-distance-weighting variance term and an exploration term, solved by particle swarm optimization | preferences in $\{-1, 0, 1\}$ (ties allowed); C-GLISp adds unknown constraints and a "satisfactory" label | glis 2.0.2 (2023-06-19) | no commits after 2023-03-03; mixed-variable successor PWAS/PWASp [@zhu2025pwas], last commit 2025-01-23 |
| prefGP [@benavoli2026prefgp] (JAX and PyTorch) | 9 preference and choice likelihoods; Laplace, variational inference (full and sparse), slice sampling | none | preference learning only, no optimization loop | no release tag; not on PyPI | last commit 2026-07-14; pins botorch 0.9.3, gpytorch 1.11, jax 0.4.23, torch 2.1.2 |
| SkewGP [@benavoli2025skewgp] | closed-form skew Gaussian process posterior, optional Laplace | Thompson sampling, upper confidence bound, an information-gain rule | preference and mixed data | no tag | last commit 2025-08-05 (README only) |
| preferentialBO [@lab2023preferentialbo] (Python 3.9, GPy 1.10.0) | Gibbs-sampled preference Gaussian process; expectation propagation, Laplace, and MCMC baselines | HB-EI, HB-UCB, and the paper's 6 baseline rules | reproduction code for Takeno et al. (ICML 2023) | no tag | no commits after 2023-07-26 |
| qEUBO paper code [@research2023qeubo; @astudillo2023qeuboc] | in the paper, a variational Gaussian process with inducing points | qEUBO and baselines | the official repository has 5 tasks; the author's repository has more, including the Animation data | no tag | no commits after March 2023; no dependency file |
| PABBO [@zhang2026pabbo] | transformer policy pretrained by reinforcement learning | the policy outputs the query pair | amortized preference optimization | no tag | last commit 2026-09-14; pins botorch 0.10.0 and a CUDA 11.8 build of torch; needs a Weights and Biases key |
| Emukit example [@emukit2026preferential] (GPy, Stan) | expectation propagation, variational inference, MCMC | batch comparison rules | batch PBO, in the examples directory only | emukit 0.5.1 (2026-02-22) | example unmaintained; still requires Python 3.7, scipy 1.1.0, and pystan |
:::

Several more methods exist only as code for a single paper: PPBO
[@pml2022ppbo] (MIT, last commit September 13, 2022); POLAR [@tucker2024polar]
(MATLAB, BSD-3-Clause, last commit June 17, 2024, supporting pairwise,
coactive, and ordinal feedback); the LILO research code [@research2026lilo]
(MIT, last commit May 12, 2026); and DT-PBO, PBO with decision-tree surrogates
[@thomasq992026dt] (MIT). Three repositories have no license file, so default
copyright applies and reuse is not permitted without asking: POP-BO
[@predictepfl2024pop], MR-LPF [@kayal2025bohf], and CrashPBO
[@dsme2026crashpbo].

### Libraries without preference support {#sec-sw-no-preference}

None of the major Bayesian optimization and Gaussian process libraries has a
preference likelihood or a preference acquisition function. We searched the
source trees of the latest versions at the end of September 2026, GPyTorch
1.15.2 [@gpytorch2026pypi], GPflow 2.11.1 [@gpflow2026pypi], Trieste 4.6.0
[@labs2026trieste], SMAC 2.4.1 [@automlorg2026smac], HEBO 0.3.6
[@lab2024hebo], Dragonfly 0.1.7 [@dragonfly2022opt], GPyOpt 1.2.6
[@sheffieldml2023gpyopt], and Optuna 5.0.0 [@optuna2026pypi], for *pairwise*,
*preference*, and *dueling*. `PairwiseGP` uses GPyTorch's kernels but
implements its own Laplace inference; "dueling" appears in HEBO only in an
unrelated reinforcement learning subproject; and Optuna's preference support
lives in optuna-dashboard and OptunaHub. GPyOpt was archived on January 17,
2023, and Dragonfly is effectively unmaintained.

### What the software landscape means {#sec-sw-landscape}

Three judgments follow (inference). Practical PBO depends
on one organization: BoTorch supplies the models and acquisition functions, Ax
manages experiments, BOPE, and LILO, and AEPsych serves psychophysics; since
qEUBO in February 2024, new preference features have landed in Ax rather than
BoTorch. Alternatives to the Gaussian process (GLISp, DT-PBO, PABBO) are niche,
and only GLIS can be installed with `pip`. And for a practitioner, Ax 1.3.x is
the most complete maintained software but lacks documentation for these
features; we did not verify whether Ax lets a user pass a dimension-scaled
prior to `PairwiseGP` through its model configuration.

## Default priors that ignore dimension {#sec-sw-priors}

A default prior is a modeling decision that most users never see.
@sec-obs-pairwisegp read the choices inside `PairwiseGP` (@tbl-pref-botorch
lists them), and @sec-hd-diagnosis explained why a lengthscale prior of fixed
scale fails as the number of inputs grows. @tbl-sw-priors collects the
defaults of every preference package we examined, with BoTorch's scalar models
as the point of comparison. No row scales with
dimension except the comparison row.

::: {.table #tbl-sw-priors title="Default lengthscale settings in preference software and papers. Only BoTorch's scalar models, included for comparison, scale with dimension."}
| Software or paper | Kernel | Default lengthscale setting | Mode or initial value | Scales with dimension |
|---|---|---|---|---|
| BoTorch `PairwiseGP` [@botorch2026pairwisegp] | RBF, inside a required `ScaleKernel` | Gamma(2.4, 2.7), initialized at its mode, lower bound $10^{-4}$ | about 0.52 | no |
| BoTorch's other models from 0.12.0 (comparison) [@botorch2026gpytorch] | depends on the model | log-normal with location $\sqrt{2} + \tfrac12\log d$ and scale $\sqrt{3}$, lower bound 0.025 | about 0.65 at $d = 10$, about 2.05 at $d = 100$ (inference) | yes |
| optuna-dashboard preferential sampler [@optuna2026sampler] | Matérn 3/2, one lengthscale per input | Gamma(5, 10); noise prior Gamma(5, 50) | 0.4 | no |
| sequential-line-search [@koyama2025preference] | Matérn 5/2, one lengthscale per input | default 0.5, maximum a posteriori prior with variance 0.25 | 0.5 | no |
| Koyama et al. 2017 and 2020 papers [@koyama2017sequentialb; @koyama2020sequential] | Matérn 5/2 in the 2020 paper | log-normal with parameters $\mu = 0.5$ and $\sigma^2 = 0.01$, nearly fixed | about 0.5 | no |
| PABBO synthetic prior, in training and evaluation [@zhang2025pabbob; @zhang2025pabboc] | RBF and Matérn 5/2, 3/2, 1/2 | truncated to $[0.05, 2]$, centered near 1/3 | not applicable | no |
:::

No PBO package documents how its default prior relates to dimension, so anyone
using these packages above about 10 dimensions inherits the fixed-scale priors
that scalar Bayesian optimization abandoned in 2024 (inference;
@fig-hd-lengthscale shows the consequence, and @sec-hd-practice what to pass
instead). This is a different question from whether a prior should be informed
by other users' data, a population prior, which @sec-hci-population takes up:
the defaults in @tbl-sw-priors are weak priors, but weak at the wrong scale once
the dimension grows (inference).

## Reproducing research code {#sec-sw-reproducibility}

A method that cannot be rerun cannot be checked, and a baseline that cannot be
rerun gets reimplemented, each time a little differently. Three habits make
research code rerunnable: a **release tag** (a named, frozen version of the
repository), a **dependency file** that lists the exact library versions the
code needs, and a **license** that says who may reuse it.

At the end of September 2026 we checked 11 research repositories, prefGP,
SkewGP, preferentialBO, the official qEUBO repository, PABBO, PPBO, POLAR, GLIS,
POP-BO, LILO, and BO_toolbox, for release tags: **all 11 had none.**
Dependencies are commonly pinned to versions from 2020 to 2024
(@tbl-sw-packages lists several), PPBO pins numpy 1.18.4 and the archived
GPyOpt 1.2.6 [@pml2022ppbo], and neither qEUBO repository has a dependency file
[@research2023qeubo; @astudillo2023qeuboc]. preferentialBO's scripts call
`np.int`, which NumPy 1.24 removed, so they run only with the pinned numpy
1.23.2; its README also warns that on Ubuntu 20.04 it has not been confirmed
that the paper's results can be fully reproduced (the experiments were run on
CentOS 6.9) [@lab2023preferentialbo].

Licenses are not uniform either. Most repositories use MIT, BSD, or Apache
licenses, but PABBO uses the AGPL-3.0, a **copyleft** license that requires
derived software, including software offered over a network, to be released
under the same license; AEPsych's Creative Commons Attribution-NonCommercial
4.0 license restricts commercial reuse [@meta2026aepsych]; and three
repositories from 2024 to 2026 have no license file at all. Each method also brings its own
Gaussian process implementation (GPy, BoTorch 0.9 or 0.10, JAX, MATLAB), so
most papers reimplement their baselines, which may be one reason the same
baseline looks strong in one paper and weak in another (inference).

::: {.pitfall title="Running PBO research code"}
Expect to pin old versions. A repository without a tag is a moving target, and
one without a dependency file leaves the versions to guesswork: record the
commit hash and the versions you got working. Check the license before building
on the code; three of the repositories above have none, and one is copyleft.
When a paper's baseline is a reimplementation, the baseline's settings are part
of the result (inference).
:::

## How methods are evaluated {#sec-sw-evaluation}

Comparing optimizers is harder than it looks. A single run depends on the
random initial design and on the noise in the answers, so a method can win one
run by luck, which is why comparisons average over many runs (@sec-bo-loop).
And a regret bound (@sec-regret) holds for a class of functions, not for the
function in front of you, so only benchmarks can say how methods rank in
practice, within the limits this section describes.

### Inherited test functions and simulated users {#sec-sw-test-functions}

Since @gonzalez2017preferentialb, almost every PBO paper has been evaluated with
a **simulated user**: a program that answers each comparison by evaluating a
known function at the two options and adding noise. The functions are standard
synthetic test functions, usually in 1 to 8 dimensions, which González et al.
took from the library of scalar test functions maintained by Surjanovic and
Bingham and later papers kept. These functions were designed to test scalar
optimizers; they have none of the properties of human utility that
@sec-part-preferences and @sec-part-perspectives describe, such as
indifference thresholds, drift, intransitivity, and anchoring (inference).
@tbl-sw-evals lists the test problems and metrics of representative papers;
@tbl-acqf-studies gives the noise, budget, and conclusion of most of them, and
the lists below collect the noise models and metrics.

::: {.table #tbl-sw-evals title="How representative PBO papers evaluated their methods: test problems and main metric."}
| Paper | Test problems (dimension) | Main metric |
|---|---|---|
| @gonzalez2017preferentialb, ICML 2017 | Forrester (1); Six-hump camel, Gold-Stein, Levy (2); 33-point grid per dimension; 5 initial duels, total budget of 200 duels | true value at the current Condorcet winner |
| @mikkola2020projective, ICML 2020 | Six-hump camel (2), Hartmann (6), Levy (10), Ackley (20); 100 queries | true objective value at the maximizer of the posterior mean |
| @siivola2021preferential, MLSP 2021 | several functions from the SigOpt library; 4 real data sets including Sushi and Candy ($d \le 4$) | we did not extract it |
| @fauvel2021efficient, arXiv 2021 (preprint) | 34 functions from the Surjanovic and Bingham library, standardized to mean 0 and variance 1 | final optimum; Mann-Whitney U tests with Borda scores |
| BOPE: @lin2022preferenceb, AISTATS 2022 | Vehicle safety ($d = 5$, $k = 3$ outcomes), DTLZ2, OSY ($d = 6$, $k = 8$), Car cab ($d = 7$, $k = 9$), with several utilities | true utility at the maximizer of the posterior mean |
| qEUBO: @astudillo2023qeubob, AISTATS 2023 | Ackley (6), Alpine1 (7), Hartmann (6), Car cab (7), Sushi (4), Animation (5); $4d$ initial queries, then 150 | log simple regret at the maximizer of the posterior mean |
| @takeno2023practicalc, ICML 2023 | 12 functions (8 in the main text, up to Hartmann in 6 dimensions) | regret at the recommended point |
| POP-BO: @xu2024principledb, ICML 2024 | Gaussian process samples, standard test functions including 6-dimensional Ackley, a thermal comfort problem | cumulative regret and suboptimality of the reported solution |
| MaxMinLCB: @pasztor2024bandits, NeurIPS 2024 | Ackley, Eggholder, and others; Yelp restaurant data | cumulative regret in preference probabilities |
| PABBO: @zhang2025pabbob, ICLR 2025 | Gaussian process samples (1, 2), Forrester, Beale, Branin; Ackley (6) and Hartmann (6) in an appendix; HPO-B, Candy, Sushi | simple regret of the best queried point |
| MR-LPF: @kayal2025bayesian, ICML 2025 | samples from a reproducing kernel Hilbert space, Ackley (1), Yelp (275 restaurants, 20 users) | cumulative regret in preference probabilities |
| PF-TS: @lazzaro2026finiteb, AISTATS 2026 | Ackley (1) with $T = 300$; hydrogen yield of 63 catalyst compositions of three metals, with $T = 800$ | cumulative regret in preference probabilities |
:::

In the table, MaxMinLCB is the max-min lower confidence bound algorithm of
Pásztor et al., PF-TS is Thompson sampling with preference feedback, and HPO-B is
a benchmark built from hyperparameter tuning tasks of the kind
@sec-cs-classifier works through. Papers from 2026 continue the same practice,
among them KappaSharp, a preprint, with 11 benchmarks in 5 to 20 dimensions
[@shao2026adaptive].

### Noise models chosen paper by paper {#sec-sw-noise-models}

Each paper picks its own model of how the simulated user errs. We found at least
six kinds:

1. Gaussian noise added to the utilities before comparing, with standard
   deviation 0.01 (Takeno et al.), 0.05 (Siivola et al.), 0.1 (the BoTorch
   tutorial), or 10% of the function's range (local PBO [@menn2026local]).
2. Logistic noise calibrated so that random pairs among the top 1% of points
   are answered wrongly 10%, 20%, or 30% of the time (qEUBO, in an appendix
   study); for 6-dimensional Ackley, these correspond to logistic scales of
   0.0575, 0.1416, and 0.2943 [@astudillo2023noise].
3. A fixed probability of flipping the answer (10% in BOPE).
4. No noise by default (PABBO).
5. Probit noise on a standardized function (Fauvel and Chalk).
6. A decision maker simulated by a language model (LILO
   [@kobalczyk2026lilo]).

Calibrating to an error rate is closer to reality than setting a scale directly,
because it fixes how often the simulated user is wrong on the comparisons that
matter, but the noise is still homoscedastic (the same everywhere) and
independent from one answer to the next (inference). Even two models matched
to the same error rate on one gap disagree on every other gap, as
@fig-sw-noise shows.

```{figure}
//| figure: sw-noise
//| label: fig-sw-noise
//| fig-cap: "Three ways a simulated user can err, each matched so that a comparison whose utility gap is 0.1 is answered wrongly 10% of the time: a fixed probability of flipping the answer, the kind BOPE uses; logistic noise on the utility difference (the Bradley-Terry link), the kind qEUBO calibrates; and Gaussian noise on each utility (the probit link), the kind of Takeno et al., Siivola et al., and BoTorch's tutorial. The vertical axis is logarithmic and stops at 0.0001%. The sliders move the gap and error rate at which the models are matched, and the gap at which they are read off (orange line). The curves illustrate the three models with these parameters, not the settings of any one paper, whose utility scales differ; PABBO's default, no noise, would lie below the plot."
```

Some things to try:

- With the defaults, read the three curves at a gap of 0.30. The flip model
  still errs 10% of the time, logistic noise 0.14%, and probit noise 0.006%:
  "10% noise" makes a clearly worse option about 70 times more likely to win
  under the flip model than under logistic noise, and over a thousand times
  more likely than under probit noise (@exr-sw-flip works the probit case by
  hand).
- Move the reading gap to 0.05, a closer call than the matched one. Now
  logistic and probit noise err 25% and 26% of the time, the flip model still
  10%: the flip model is harsher on clear choices and gentler on close ones.
- Raise the error rate at the matched gap to 30%. At a gap of 0.30 the logistic
  and probit users now err 7.3% and 5.8% of the time, close to each other,
  while the flip model errs 30%. With very noisy answers the two utility-noise
  models nearly agree; it is the flip model that stands apart.

### Regret, defined five ways {#sec-sw-regret}

**Regret** is the shortfall of a method's result from the best achievable value
(@sec-regret-definitions). In PBO the method never sees utility values, so a
paper must decide which point counts as "the result", and the papers decide
differently:

- the function value at the current **Condorcet winner**, the point that wins
  most often (González et al.);
- the **simple regret at the maximizer of the posterior mean**, the point the
  model would recommend (qEUBO, BOPE; Takeno et al. use their recommended
  point);
- the simple regret of the **best point queried so far** (PABBO);
- the **cumulative regret** summed over both points of every comparison, in
  preference probabilities (MaxMinLCB, MR-LPF, PF-TS);
- the cumulative regret in utility (POP-BO).

Papers also report the quality of the model's ranking (Kendall's rank
correlation in the BoTorch tutorial and in Ax), the computing time per
iteration (qEUBO, PABBO, Takeno et al.), and, in a user experiment, the number
of simulation steps needed to reach the nearest local minimum (Mikkola et al.).
No paper uses the number of comparisons needed to
reach a threshold as its main metric, although that is the quantity a person in
the loop pays for.

The choice of metric changes the ranking of methods. In the POP-BO paper, qEUBO
reports a slightly better solution than POP-BO on Gaussian process sample
instances, but its cumulative regret is more than 2.5 times higher, and on
6-dimensional Ackley the two metrics reverse again [@xu2024principledb]. A
method that queries near the optimum is favored by regret at the queried
points; a method whose model is well calibrated is favored by regret at the
posterior mean (inference). @fig-sw-metrics lets the reader watch this happen.

```{figure}
//| figure: sw-metrics
//| label: fig-sw-metrics
//| fig-cap: "One set of simulated runs, three verdicts. Three ways of choosing the next duel run against the same simulated user on the book's running objective: EUBO; a simple incumbent-versus-challenger rule (the model's current best guess against the point with the highest optimistic estimate, posterior mean plus two standard deviations), which is an illustration, not a published method; and random pairs. Every run starts from one random duel and asks 15 more. Curves are means over the runs, with bands of ±1 standard error. *Recommended* is the regret of the point with the highest posterior mean; *Best queried* is the regret of the best point shown to the user so far; *Cumulative* adds up, over every duel, the average regret of its two options. The setup is illustrative: one input, 41 candidate points, an RBF kernel with lengthscale 0.1, the probit likelihood with noise 0.1 and the Laplace approximation on the whole grid, and Thurstone noise of the chosen size in the simulated answers. The reversal shows the mechanism behind the POP-BO observation, not a reproduction of it."
```

Some things to try:

- With the defaults (30 runs, noise 0.1), *Recommended* puts EUBO first, the
  incumbent rule close behind with overlapping bands, and random pairs last.
- Switch to *Best queried*. Random pairs now win, with regret near 0.05,
  through coverage rather than learning: 16 random pairs make 32 draws from
  only 41 candidates, so on average about 23 of the candidates have been shown
  (@exr-sw-coverage). In a larger space, or in more dimensions, the same metric
  would not favor random pairs so strongly (inference).
- Switch to *Cumulative*. The incumbent rule wins clearly: every pair it asks
  contains the model's current best guess, which is soon near the optimum, so
  half of each comparison costs almost nothing (@exr-sw-incumbent).
- Set *Runs* to 5 and press *Another batch of runs* several times. Under
  *Recommended*, the winner changes from batch to batch, mostly between EUBO and
  the incumbent rule: with few repetitions, the ranking is partly a draw.
- Change the noise. At 0.2, random pairs overtake the incumbent rule under
  *Recommended*, while the other two metrics keep their order.

### Statistics and the size of the differences {#sec-sw-statistics}

Statistical reporting is thin. Papers use from 3 repetitions (the BoTorch
tutorial), 10 (Takeno et al.), 20, and 30, up to 50 to 100 (qEUBO), and show
uncertainty as ±1, ±1.96, or ±2 standard errors, 95% confidence intervals, or
interquartile ranges. Only Fauvel and Chalk test many hypotheses formally:
across 34 functions they compare every pair of methods with a Mann-Whitney U
test (a test of whether one method's results tend to be larger than another's,
without assuming a distribution) and aggregate the wins into Borda scores (each
method scores one point for every method it beats) [@fauvel2021efficient]. No
later paper adopted the approach. Where differences are measured at all, the
gap between elaborate acquisition functions and random queries is often small,
and @sec-acqf-empirical reports the evidence and why the existing comparisons
do not add up. The practical reading: repeat runs at least tens of times,
include random queries, and test the differences that a claim rests on
(inference).

## Real-data tasks and missing datasets {#sec-sw-datasets}

The tasks that papers call "real data" all turn the judgments of a group into a
single deterministic utility. @tbl-sw-datasets lists the four in use.

::: {.table #tbl-sw-datasets title="The real-data tasks used in PBO evaluations, and how each becomes a deterministic test function."}
| Task | Source | What it contains | How PBO papers use it |
|---|---|---|---|
| Sushi | Kamishima's sushi preference data [@kamishima2026sushi] | human rankings of sushi types | Siivola et al.: complete rankings of 100 sushi types with 4 continuous features; PABBO: five-point ratings averaged over users |
| Candy | FiveThirtyEight's online pairwise voting [@fivethirtyeight2017candy] | 2 features, sugar percentile and price percentile | both papers turn the votes into one complete ranking or win rate; Siivola et al. count 86 candies, PABBO 85 |
| Yelp | restaurant ratings | 275 restaurants, 20 users, 32-dimensional embeddings | MR-LPF and MaxMinLCB; built from ratings, not comparisons |
| Animation | qEUBO [@astudillo2023qeubob] | a fire-like particle effect with 5 parameters, from an AEPsych demo | the authors "collected 100 such pairwise comparisons from human users", fitted a model, and used it as the ground-truth test function |
:::

These tasks reward the method that finds the optimum of a population, and they
hide the inconsistency of individual judgments, which is the very thing PBO
exists to handle (inference). We found no shared benchmark suite or
leaderboard for PBO, and no public data set of individual-level pairwise
judgments designed for evaluating PBO. Such a data set, with timestamps,
presentation order, response times, and repeated pairs, is what would let
methods be compared offline on real answers (@sec-rec-evaluation).

## Reproduction problems found {#sec-sw-reproduction}

Checking papers against their code turned up the following problems.

- **qEUBO's Animation task.** The paper says that the fitted model is a support
  vector machine [@astudillo2023qeubob]. The author's personal repository has
  two versions of the ground truth: `animation_runner.py` uses a support vector
  machine classifier and `animation2_runner.py` loads a fitted `PairwiseGP`
  [@astudillo2023qeuboc]. The official repository the paper cites has no run
  script or data for the Animation task [@research2023qeubo], and neither
  repository lists its dependencies.
- **BOPE's dimensions.** The text gives "DTLZ2 (d = 4, k = 8)" while a figure is
  labeled "DTLZ2 (d=8, k=4)" [@lin2022preferenceb].
- **Candy's size.** Two papers report 86 and 85 candies
  [@siivola2021preferential; @zhang2025pabbob]; FiveThirtyEight's data file has
  85 rows [@fivethirtyeight2017candy].
- **Takeno et al.'s code** does not run as is in a current environment
  (@sec-sw-reproducibility) [@lab2023preferentialbo].

None of these is large on its own; together they mean that the benchmark
results of PBO cannot at present be reproduced one by one without contacting
the authors (inference).

## Simulated and real users {#sec-sw-simulated-users}

Every benchmark above assumes that a simulated user stands in for a person well
enough to rank methods. The few studies that compared the two directly all
found clear differences. @tbl-sw-human collects them.

::: {.table #tbl-sw-human title="Studies that put real people where a simulated user is usually assumed, and what differed."}
| Study | Setting | Participants | What differed from the simulated user |
|---|---|---|---|
| @schoinas2025evaluating, EMBC 2025 | retinal implant encoders, simulated prosthetic vision | 17 sighted | same choice as the simulated agent in only about 50% of trials; final loss 0.27 against 0.07 in simulation |
| @ou2022human, Mensch und Computer 2022 | 9-parameter polygon simplification for 3D models | 2 artists for 3 months; 20 in the lab | satisfactory results in 11.9% of sequences in the field, 48.5% in the lab; inconsistent judgments, loss aversion |
| @ou2023impact, IUI 2023 | text, photo, and 3D mesh tasks | 60 | stopping depends on expertise |
| @colella2020human, UMAP 2020 | 1-dimensional function, scalar feedback | 21 | users who understood the optimizer gave strategically biased answers |
| @chan2022investigating, CHI 2022 | 3D touch interaction design, multi-objective BO on measured completion time and spatial error | 40 novice designers | better results but lower sense of agency and expressiveness |
| @taddei2026bayesian, arXiv 2026 (preprint) | preference optimization of an active prosthesis | simulations, then trials with 4 people (3 with one method, 1 with another) | in one of three trials with one participant, the preference estimate kept fluctuating after iteration 15, possibly from fatigue |
:::

The table's numbers carry the main point, and three details add to it. In the
retinal implant study, which tested an optimization that @granley2023human
(NeurIPS 2023) had evaluated only in simulation, 16 of the 17 participants
still preferred the optimized result under the main condition, and the authors
stress "the importance of validating optimization strategies with human
participants" [@schoinas2025evaluating]. In Ou et al.'s three-month field
deployment, 415 of 549 evaluation sequences stopped at the first iteration, and
the authors conclude that "optimization using preferential choices lacks
mechanisms to deal with inconsistent and contradictory human judgments", and
that "machine outcomes, in turn, influence future user inputs via heuristic
biases and loss aversion" [@ou2022human] (@sec-hci-unstable,
@sec-hci-expertise). And users who understand how the optimizer works
"strategically provide biased answers" [@colella2020human], contrary to the
simulated user's assumption that feedback is faithful, while Mikkola et al.
report that their method could tell choices made by people from those made by a
computer program [@mikkola2020projective].

None of these behaviors appears in any simulated-user benchmark we found
(inference). Simulated evaluation can compare how algorithms behave under a
given noise model, but it cannot predict how they rank with people
(@sec-acqf-empirical describes the missing experiment). Until simulators are
validated on individual data, a result obtained with a simulated user should be
checked on at least a few real people before it is reported as a result about
people (inference); the norms that experimental economics adopted for such
comparisons are a useful model (@sec-econ-replication).

## The research community {#sec-sw-community}

PBO is the work of a small number of groups. This section maps them by
discipline, then reports the infrastructure of a field (surveys, theses,
workshops, venues) and how much is published.

### Groups by discipline {#sec-sw-groups}

@tbl-sw-groups arranges the groups by the discipline they publish in, which is
also, largely, the kind of question they ask.

::: {.table #tbl-sw-groups title="Research groups in PBO, 2017 to 2026, arranged by discipline."}
| Discipline (typical venues) | Group (institutions) | Representative work | Direction |
|---|---|---|---|
| Machine learning methods (ICML, AISTATS, NeurIPS, ICLR, TMLR) | Frazier, Astudillo, Bakshy, Lin (Cornell University, Meta, Caltech) | [@astudillo2020multib]; BOPE [@lin2022preferenceb]; qEUBO [@astudillo2023qeubob]; [@astudillo2025preferential]; LILO [@kobalczyk2026lilo] | decision-theoretic acquisition and preference exploration over outcomes, recently labels from language models |
| | Takeno, Nomura, Karasuyama (Nagoya Institute of Technology, CyberAgent AI Lab, RIKEN) | [@takeno2023practicalc; @ozaki2024multi] | skew Gaussian process inference and the hallucination believer; the optuna-dashboard sampler implements their method |
| | Benavoli, Azzimonti, Piga (Trinity College Dublin, IDSIA) | [@benavoli2021preferentialb; @benavoli2026tutorial] | preference and choice likelihoods, exact posteriors, the prefGP code |
| | Kaski (Aalto University) | PPBO [@mikkola2020projective]; batch PBO [@siivola2021preferential]; PABBO [@zhang2025pabbob]; [@sinaga2024anchor] | projective queries, batches, amortization, noise models |
| Machine learning theory (ICML, NeurIPS, AISTATS) | Krause (ETH Zurich) | [@kirschner2021bias]; MaxMinLCB [@pasztor2024bandits] | dueling-bandit theory |
| | Vakili and colleagues (MediaTek Research; collaborators at University College London and Imperial College) | MR-LPF [@kayal2025bayesian]; PF-TS [@lazzaro2026finiteb] | regret theory; no software releases, no human studies |
| | Jones (EPFL) | POP-BO [@xu2024principledb] | guarantees, with building thermal comfort as the application |
| Control (ECC, ACC, IEEE Transactions on Control Systems Technology) | Bemporad, Piga (IMT School for Advanced Studies Lucca; IDSIA) | GLISp [@bemporad2021global]; C-GLISp [@zhu2022c]; PWAS and PWASp (code [@zhu2025pwas]) | non-probabilistic surrogates and controller calibration |
| Robotics (ICRA, IROS, IEEE Robotics and Automation Letters) | Ames, Yue, Tucker, Novoseller (Caltech) | CoSpar [@tucker2020preference]; LineCoSpar [@tucker2020human]; POLAR [@tucker2022polar] | personalizing exoskeleton gaits (@sec-cs-exoskeleton) |
| | Trimpe (RWTH Aachen University) | CrashPBO [@menn2026preferential]; local PBO [@menn2026local] | entered the field in 2026; tuning robot controllers |
| Human-computer interaction and graphics (SIGGRAPH, UIST, CHI, IUI, Mensch und Computer, UMAP) | Koyama (University of Tokyo; National Institute of Advanced Industrial Science and Technology) | sequential line search [@koyama2017sequentialb]; Sequential Gallery [@koyama2020sequential]; constrained PBO [@iwai2025constrainedb] | interactive optimization of visual design (@sec-cs-photo), turning to constraints and language-model assistance |
| | Oulasvirta (Aalto University) | [@colella2020human; @chan2022investigating] | human studies of human-in-the-loop optimization, mostly with ratings or performance feedback rather than comparisons |
| | Butz, Buschek, Mayer (LMU Munich, media informatics) | [@ou2022human; @ou2023impact] | human-subject studies of PBO in creative tools |
| Biomedical engineering (EMBC) | Fauvel, Chalk, Beyeler | [@fauvel2021efficient; @granley2023human; @schoinas2025evaluating] | human-in-the-loop optimization of retinal prosthesis encoders |
:::

Other recurring contributors include Mesbah and colleagues (KappaSharp, a 2026
preprint) [@shao2026adaptive], Wu and Gardner (a knowledge gradient for
preference learning, a 2026 preprint) [@wu2026knowledge], and Theiner, Hirt,
Findeisen, and colleagues in control (ECC 2025 and ECC 2026)
[@theiner2025exploiting; @theiner2026efficient]; we did not verify their
institutions. Read across the table, the groups are moving from methods toward applications and toward
language models as intermediaries, and the newcomers (Trimpe, Mesbah) come from
control (inference). The groups that prove regret bounds do not run human
studies, the groups that run human studies rarely change the acquisition
function, and the groups that build the software sit closest to the
decision-theoretic line (inference).

### No survey, nine theses, no dedicated workshop {#sec-sw-infrastructure}

**There is no dedicated survey of PBO.** General references on Bayesian
optimization give preferences a page at most: Frazier's tutorial
[@frazier2018tutorial] does not contain the words "preference", "pairwise", or
"duel", Garnett's monograph [@garnett2023bayesian] mentions optimizing human
preferences in about one paragraph, and the survey of @wang2023recent (ACM
Computing Surveys 2023) has no section on PBO. The dueling bandit survey of
@bengs2021preference (JMLR 2021) covers preference probabilities modeled with
Gaussian processes but cites neither González et al. 2017 nor Brochu et al. [@brochu2010tutorial]. The closest reference work is the tutorial of @benavoli2026tutorial
(Foundations and Trends in Machine Learning 2026), which covers nine Gaussian
process models of preferences and choices but mentions PBO only briefly. The
remaining teaching material consists of software tutorials and the book
chapter *Computational Design with Crowds* by Koyama and Igarashi
[@koyama2020computational], a preprint whose formal publication details we did
not verify.

**Doctoral theses carry much of the field's synthesis.** We found at least nine
doctoral theses from 2017 to 2025 that center on PBO or devote a substantial part
to it (@tbl-sw-theses).

::: {.table #tbl-sw-theses title="Doctoral theses on, or substantially about, PBO, 2017 to 2025."}
| Author | Title | Institution | Year |
|---|---|---|---|
| Yuki Koyama | *Computational Design Driven by Visual Aesthetic Preference* [@koyama2017computational] | University of Tokyo | 2017 |
| Ellen Novoseller | *Online Learning from Human Feedback with Applications to Exoskeleton Gait Optimization* [@novoseller2021online] | Caltech | 2021 (record date) |
| Eero Siivola | *Applications of human feedback in Gaussian processes* [@siivola2021applications] | Aalto University | 2021 |
| Tristan Fauvel | *Human-in-the-loop optimization of retinal prostheses encoders* [@fauvel2021human] | Sorbonne University | 2021 |
| Raul Astudillo | *Exploiting Composite Functions in Bayesian Optimization* [@astudillo2022exploiting] | Cornell University | 2022 |
| Maegan Tucker | *Enabling Robust and User-Customized Bipedal Locomotion on Lower-Body Assistive Devices via Hybrid System Theory and Preference-Based Learning* [@tucker2023enabling] | Caltech | 2023 |
| Petrus Mikkola | *Humans as Information Sources in Bayesian Optimization* [@mikkola2024humans] | Aalto University | 2024 |
| Mengjia Zhu | *Global and preference-based optimization using surrogate-based methods* [@zhu2024global] | IMT School for Advanced Studies Lucca | 2024 |
| Wenjie Xu | *Bayesian Optimization with Constraints, Structure and Human Feedback* [@xu2025bayesian] | EPFL | 2025 |
:::

**The closest workshops covered preference learning as a whole.** The ICML
2023 workshop *The Many Facets of Preference-Based Learning* covered dueling
bandits, reinforcement learning from human feedback (RLHF, the method used to
align language models with human preferences), social choice, and
optimization, with two papers on Bayesian optimization from preferences
[@icml2023many]. PBO work has also appeared at the ICML 2019 Workshop on Human
in the Loop Learning [@mccourt2019sampling], at the NeurIPS 2022 Workshop on
Gaussian Processes, Spatiotemporal Modeling, and Decision-making Systems
[@takeno2022preferential], and at ProbML 2026, a symposium held alongside ICML
with archival proceedings [@probml2026symposium]. No workshop dedicated to PBO
appeared at NeurIPS or ICML from 2024 to 2026, as far as arXiv comments and
three web searches show.

### How much is published {#sec-sw-counts}

Publication volume is growing, from a small base. We ran four searches, two over
arXiv abstracts [@arxiv2026preferential; @arxiv2026bayesian] and two over
Semantic Scholar [@scholar2026preferential], each in a narrow form (the phrase)
and a broad form (preference words together with Bayesian optimization).
@fig-sw-publications shows the yearly counts.

```{figure}
//| figure: sw-publications
//| label: fig-sw-publications
//| fig-cap: "Entries per year in four searches for PBO papers, run in September 2026; the menu switches between them on a fixed scale. The 2026 bar (hatched) covers January to September. Totals: 46 (arXiv, phrase), 162 (arXiv, broad), 43 (Semantic Scholar, phrase), and 135 (Semantic Scholar, broad). Recall is imperfect in both directions: the phrase searches miss known PBO papers that do not use the phrase (MaxMinLCB and POLAR are in neither arXiv result), and the broad searches include unrelated results from plant science and epidemiology. The Semantic Scholar phrase search did not list 2018 or 2019."
search: arxivPhrase
```

All four searches show growth by a factor of about 2.4 to 4.3 from 2023 to 2025,
and the first nine months of 2026 nearly match all of 2025 (12 entries against
13, 38 against 39, 10 against 13, and 32 against 34). But even in 2025 and 2026
a year brings only 10 to 40 entries. The growth coincides with the attention
that language-model alignment has brought to preference feedback (inference). The cross-citations we checked are few: the 2021
dueling bandit survey does not cite González et al. 2017, and the 2026 local
PBO paper does not cite the work on high-dimensional lengthscale priors
(@sec-hd-local) (inference).

## Settled, contested, missing {#sec-sw-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** The only maintained general-purpose PBO software is BoTorch and Ax,
both from Meta; the other major Bayesian optimization and Gaussian process
libraries have no preference support. None of the preference software we checked
scales its default lengthscale prior with dimension. Research code has no release
tags and outdated dependencies, and some of it has no license. Evaluation relies
on simulated users and synthetic functions in 1 to 8 dimensions, with noise
models, regret definitions, and numbers of repetitions chosen paper by paper.
There is no dedicated survey of PBO and no workshop dedicated to it; the
closest, at ICML 2023, covered preference-based learning as a whole.
Publication volume has grown since 2023 from a small base.

**Contested.** Which acquisition function is best: the answer changes with
dimension, noise, and metric, as qEUBO's better reported solution and worse
cumulative regret against POP-BO show. How far results with simulated users
carry over to people: the available evidence points to clear differences, but
the studies are small.

**Missing.** A shared benchmark suite and leaderboard. A public data set of
individual-level pairwise judgments. A reproduction study that reruns many
methods under one protocol. A human-subject comparison with random assignment of
acquisition functions. Tutorials for Ax's preference features and documentation
of the default priors. A maintained PBO package for R or Julia (not searched
systematically).
:::

## Exercises {#sec-sw-exercises}

::: {.exercise #exr-sw-coverage title="Why random pairs win on the best queried point"}
In @fig-sw-metrics, each random query is a pair of two different points drawn
uniformly from 41 candidates, and a run asks 16 such pairs. (a) What is the
probability that a given candidate is never shown? (b) How many distinct
candidates are shown on average? (c) On the running objective, three candidates
have regret below 0.12: the optimum and its two neighbors. What is the
probability that at least one of them is shown?

::: {.solution}
(a) A given candidate is left out of one pair with probability $39/41$, so out
of all 16 independent pairs with probability $(39/41)^{16} \approx 0.449$.
(b) By linearity of expectation, $41 \times (1 - 0.449) \approx 22.6$ distinct
candidates. (c) Three given candidates are all left out of one pair with
probability $\binom{38}{2}/\binom{41}{2} = (38 \cdot 37)/(41 \cdot 40) \approx
0.857$, and out of all 16 pairs with probability $0.857^{16} \approx 0.085$, so
at least one is shown with probability about 0.915. Random pairs nearly
exhaust a 41-point space; the best-queried metric rewards that coverage, and
says nothing about whether the model learned where the optimum is.
:::
:::

::: {.exercise #exr-sw-incumbent title="Why the incumbent rule wins on cumulative regret"}
The cumulative regret in @fig-sw-metrics adds, for every duel, the average
regret of its two options, $\tfrac12[r(\vx) + r(\vx')]$. Suppose that from some
step on, the incumbent is the true optimum. What is the smallest possible
per-duel regret of a rule that always includes the incumbent, and of a rule
that never shows the same point twice? What does the metric reward?

::: {.solution}
A duel containing the optimum costs $\tfrac12[0 + r(\vx')] = \tfrac12 r(\vx')$,
at most half of the challenger's regret, and zero if the challenger is the
optimum too. A rule that spreads its queries pays the full average regret of
two new points, which on the running objective averages about 0.7 per point
for a random candidate. Cumulative regret therefore rewards keeping the
incumbent in every query, even when the challenger teaches the model little: it
measures what the person was shown along the way, not how good the final
recommendation is. That is why a method can be better on one metric and worse
on another, as in the POP-BO comparison.
:::
:::

::: {.exercise #exr-sw-flip title="A fixed flip rate is not a noise model of utility"}
BOPE's simulated user picks the worse option 10% of the time, whatever the two
options are. Under probit noise of size $\sigma$ on each utility, the error
probability for a utility gap $\Delta > 0$ is
$\Phi(-\Delta/(\sqrt{2}\,\sigma))$. (a) Find the $\sigma$ for which a gap of
$\Delta = 0.1$ is answered wrongly 10% of the time. (b) With that $\sigma$,
how often is a gap of $0.3$ answered wrongly? (c) What does this say about
comparing papers that use the two noise models?

::: {.solution}
(a) We need $\Delta/(\sqrt{2}\,\sigma) = \Phi^{-1}(0.9) \approx 1.2816$, so
$\sigma = 0.1/(\sqrt{2} \times 1.2816) \approx 0.055$. (b) Then
$\Delta/(\sqrt{2}\,\sigma) \approx 3.84$, and $\Phi(-3.84) \approx 6 \times
10^{-5}$: almost never, against 10% under the flip model. (c) Under probit
noise, errors concentrate on close calls; under a fixed flip rate, a clearly
worse option is chosen as often as a marginally worse one. The second is
harsher on methods that rely on a few decisive comparisons, so the same
"10% noise" can mean very different difficulty, and results under the two
models cannot be pooled (inference).
:::
:::

## Further reading {#further-reading .unnumbered}

- The source of `PairwiseGP` [@botorch2026pairwisegp] and BoTorch's PBO
  tutorial [@botorch2026pairwise] are the de facto definition of PBO in
  practice; read them before trusting a default.
- @fauvel2021efficient is the one paper that evaluates many acquisition
  functions with formal multiple comparisons; its protocol is worth copying.
- @xu2024principledb shows within one paper how the metric decides which method
  looks better.
- @schoinas2025evaluating and @ou2022human are the clearest direct comparisons
  of simulated users and real people in preferential loops.
- @benavoli2026tutorial is the closest thing to a reference work on the models,
  though not on the optimization.
