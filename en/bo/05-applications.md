---
status: done
synopsis: "Hyperparameters, laboratories, robots, engineering designs, online experiments, and people: where Bayesian optimization has paid off and why, and how it relates to its neighbors in active learning, experimental design, bandits, reinforcement learning, and evolution strategies."
---

# Where Bayesian Optimization Works {#sec-bo-applications}

The last four chapters built a method. This one asks where it earns its keep.
Bayesian optimization is not the best optimizer for most problems: if you can
evaluate the objective a million times, or compute its gradient, simpler tools
do better. It pays off under a particular combination of circumstances, and
the fields that adopted it are the ones where that combination is common.

The chapter is a tour, not a set of worked examples. Four problems are worked
end to end in @sec-part-cases: tuning a classifier (@sec-cs-classifier),
optimizing a chemical reaction (@sec-cs-chemistry), tuning an exoskeleton
with a person in the loop (@sec-cs-exoskeleton), and enhancing a photo by
comparison (@sec-cs-photo). Here each field gets a few paragraphs: what an
evaluation is, what makes the field a good or awkward fit, and what the
published results show. The tour ends with people, where the evaluation is a
human judgment and the next part of the book begins. The last section places
Bayesian optimization among the methods it is most often confused with.

## When it pays off {#sec-app-when}

Four conditions make Bayesian optimization worth its overhead.

1. **Evaluations are expensive.** Each one costs minutes, hours, money,
   material, or a person's patience, so that the budget is tens or hundreds of
   evaluations and not millions. The cost of fitting a model and maximizing an
   acquisition function, a few seconds, is then negligible.
2. **There is no gradient and no formula.** The objective is a black box: a
   training run, an experiment, a simulation, a judgment.
3. **There are few inputs, or few that matter.** Up to about twenty with
   standard methods, more with the techniques of @sec-high-dim-practice.
4. **Noise is tolerable, and the function has some regularity.** Nearby
   inputs give similar outputs, so a model can generalize from a few
   evaluations.

When one of these fails, a neighbor usually takes over (@sec-bo-neighbors).
@tbl-app-domains lists applications where they hold, with what one evaluation
is and what the cited study reported.

::: {.table #tbl-app-domains title="Applications of Bayesian optimization discussed in this chapter"}
| Problem | One evaluation is | What the study reported | Source |
|---|---|---|---|
| Tuning the Go program AlphaGo | a set of self-play games | win rate in self-play up from 50% to 66.5% before a match | @chen2018bayesian (preprint) |
| Conditions of a chemical reaction | one reaction, in batches of five | better average efficiency and consistency than 50 expert chemists and engineers | @shields2021bayesian |
| Photocatalyst mixtures | one robot-run experiment | 688 experiments over eight days in a ten-variable space; mixtures six times more active | @burger2020mobile |
| Fast-charging protocols for batteries | a cycling test, shortened by early prediction | high-cycle-life protocols among 224 candidates in 16 days | @attia2020closed |
| Gait of a quadruped robot | a walking trial | dramatically fewer gait evaluations than local gradient methods | @lizotte2007automatic |
| Controller of a quadrotor | a flight | safe, automatic tuning without human intervention | @berkenkamp2016safe |
| Magnets of a free-electron laser | a beam measurement | significantly better than the facility's existing optimizers | @duris2020bayesian |
| A ranking system at Facebook | a randomized online experiment | demonstrated on live experiments; outperformed existing methods on noisy, constrained synthetic problems | @letham2019constrained |
| Timing of a soft exosuit | minutes of walking with measured energy cost | optimum found in 21.4 ± 1.0 minutes; metabolic cost down 17.4 ± 3.2% | @ding2018human |
| A cookie recipe | a batch baked and rated by tasters | "The cookies improved significantly over time" | @golovin2017google |
:::

## Hyperparameter tuning and AutoML {#sec-app-automl}

Machine learning models have settings that training does not choose: learning
rates, regularization strengths, the depth of a tree, the number of layers.
Each candidate setting must be evaluated by training a model and measuring its
error on held-out data, which takes minutes to days. This is the application
that made Bayesian optimization widely known. @snoek2012practical showed that
with a suitable kernel and a careful treatment of the model's own
hyperparameters, it could reach or surpass human experts in tuning models such
as convolutional networks. Before that, @bergstra2012random had shown that
random search beats grid search when only a few hyperparameters matter, and
@bergstra2011algorithms had proposed sequential model-based alternatives; the
question since then has been how much a model of the objective adds to random
sampling.

The evidence that it adds something is broad. In the black-box optimization
challenge held at NeurIPS 2020, which tuned standard machine learning models
on real data sets with held-out objective functions, 61 of 65 teams beat
random search, and the best submissions needed over 100 times fewer
evaluations to match it [@turner2021bayesian]. During the development of the
Go program AlphaGo, a 2018 preprint by its developers reports, its
hyperparameters were tuned with Bayesian optimization many times; one such
tuning before the match with Lee Sedol raised the win rate in self-play games
from 50% to 66.5% [@chen2018bayesian]. Google's internal service Vizier,
described by its developers as "the de facto parameter tuning engine at
Google", offers Bayesian optimization among its algorithms
[@golovin2017google].

Tuning hyperparameters is one step of a larger automation. *Automated machine
learning* (AutoML) treats the choice of algorithm as one more hyperparameter:
Auto-WEKA searched jointly over 27 base classifiers, their settings, and
feature selection methods, a space with many categorical and conditional
choices, with Bayesian optimization methods built for such spaces
[@thornton2013auto]. Conditional spaces (the number of trees matters only if
the algorithm is a forest) are a reason such systems often prefer tree-based
surrogates to Gaussian processes (@sec-practice-kernel).

Two features of this field shaped the methods of @sec-bo-practice. Evaluations
vary enormously in cost, and cheaper approximations are available, so
multi-fidelity methods matter (@sec-multi-fidelity). And evaluations run in
parallel on clusters, so batches matter (@sec-batch-bo).

A caution belongs here too. On a small problem, the gain over random search
can be modest. @sec-cs-classifier measures the full landscape of a
two-hyperparameter problem and finds that Bayesian optimization and random
search reach nearly the same median error; what the model buys is
reliability, and the gap in the median opens only with seven hyperparameters
(@sec-cs-classifier-many).

## Experimental science {#sec-app-science}

A laboratory experiment satisfies the four conditions almost by definition.
It is slow, it consumes material, its outcome cannot be differentiated, and a
chemist can vary only a handful of things at once: a catalyst, a solvent, a
temperature, a concentration.

**Chemistry.** @shields2021bayesian built a framework and an open-source tool
for optimizing reaction conditions, and tested it in a way few methods are
tested. They measured a large benchmark data set for a palladium-catalyzed
reaction, then had 50 expert chemists and engineers play a game in which each
chose batches of experiments, receiving the measured yields. Bayesian
optimization outperformed human decision-making in both average optimization
efficiency, the number of experiments needed, and consistency, the variance
of the outcome. @sec-cs-chemistry replays that optimization on the published
data and looks closely at the comparison with the chemists, who started
better and were overtaken (@sec-cs-chem-experts).

**Self-driving laboratories.** When robots run the experiments and the
optimizer chooses the next batch without waiting for a person, the setup is
called a self-driving laboratory. @burger2020mobile used a mobile robot that
"operated autonomously over eight days, performing 688 experiments within a
ten-variable experimental space", driven by a batched Bayesian search, and
found photocatalyst mixtures six times more active than the initial
formulations. @attia2020closed combined Bayesian optimization with a model
that predicts a battery's final cycle life from its first few cycles, and
identified high-cycle-life fast-charging protocols among 224 candidates in 16
days, compared with over 500 days for exhaustive search without early
prediction (@exr-app-attia asks what that ratio does and does not show).

How large is the gain in general? @adesiji2026benchmarking define the
*acceleration factor*, the number of experiments a reference strategy needs
to reach a target divided by the number the optimizer needs. Across 42 studies
and 63 benchmarks of self-driving laboratories, the median reported
acceleration factor was 6, with a range from 1.3 to 100. The same review found
that the benefit after a fixed number of experiments first grows with the
number of experiments per dimension and peaks at about 10 to 20 experiments
per dimension. @sec-adj-sdl discusses these measures and the role of people
in automated experiments, which is mostly to supervise
[@kalinin2023human].

**Where human experts fit.** Experts know things the model does not, and
models are patient where experts are not. In a study of semiconductor process
development, built as a controlled virtual game, human engineers excelled in
the early stages, while the algorithms were far more cost-efficient near the
tight tolerances of the target; a strategy with experts first and the
algorithm last cut the cost of reaching the target by half compared with
experts alone [@kanarik2023human]. Expert input is not always a gain: a 2025
preprint documents an industrial case in which adding expert knowledge made
Bayesian optimization fail [@weichert2025less].

## Robotics and control {#sec-app-robotics}

A robot's controller has parameters, gains, timings, and trajectory shapes,
that determine how well it walks, flies, or grasps. A simulator can suggest
values, but the final tuning happens on the hardware, where each trial takes
time, wears the machine, and occasionally breaks it. Trials are noisy, and
there are a few to a few dozen parameters.

Gait optimization was an early success. @lizotte2007automatic tuned the gait
of a quadruped robot for speed and for smoothness with Gaussian process
regression and needed dramatically fewer gait evaluations than the local
gradient methods then in use. They named the three drawbacks of local methods
that a global model removes: they get stuck in local optima, they discard
earlier evaluations after each step, and they do not model noise.
@calandra2016bayesian compared automatic gait optimization methods on
simulated problems and real robots and concluded that Bayesian optimization
is particularly suited to robotics, where good parameters must be found in a
small number of experiments.

Bayesian optimization also serves as a component of larger systems.
@cully2015robots let a six-legged robot adapt to damage, such as a broken or
missing leg, in less than two minutes. Before deployment, the robot built a
map of about 13,000 high-performing gaits in simulation; after damage, it ran
Bayesian optimization over that map, with the simulated performance as the
prior. The search space became a low-dimensional space of behaviors instead
of the high-dimensional space of controller parameters, and the authors note
that standard Bayesian optimization in the original parameter space did not
find working behaviors.

Hardware imposes the constraint of @sec-safe-bo: some parameter settings
crash the robot. @berkenkamp2016safe applied SafeOpt to tuning a quadrotor's
controller, starting from a safe but poorly performing controller and
exploring only parameters whose performance stays above a safety threshold
with high probability; the tuning ran safely and automatically, without
human intervention. For problems with more parameters and
larger budgets, trust region methods apply (@sec-hd-practice-local).

When the quality of a robot's behavior is a matter of judgment (does this
gait look natural, is this exoskeleton comfortable), the objective is a
person's preference, and the methods are those of @sec-part-preferences;
@sec-health-robots surveys that work.

## Engineering design {#sec-app-engineering}

Bayesian optimization's modern form came from engineering. The efficient
global optimization algorithm of @jones1998efficient was written for
engineering problems in which the number of evaluations is severely limited
by time or cost, typically because each one is a long computer simulation,
and designing with surrogate models has its own engineering textbook
[@forrester2008engineering]. A simulation of the airflow over a wing may take
hours; the design may have a dozen shape parameters; and there are
constraints, such as a maximum stress, that come out of the same simulation
as the objective (@sec-constrained-bo).
Simulations also come in cheaper and coarser versions, which is the setting of
multi-fidelity methods (@sec-multi-fidelity).

Physical machines are tuned the same way. The Linac Coherent Light Source, a
free-electron laser, changes configuration several times a day and has to be
retuned each time. @duris2020bayesian tuned groups of its quadrupole magnets
with Bayesian optimization, using a Gaussian process whose parameters were
fitted from archived scans and whose correlations between magnets came from a
simple physical model of the beam; the routine significantly outperformed the
facility's existing optimizers. The example shows a pattern common in
engineering: the kernel is not a default but encodes what is known about the
machine.

## Online experiments {#sec-app-online}

Internet companies tune their products by randomized experiments: some users
see variant A, others variant B, and the outcomes are compared. When the
variants differ in continuous parameters, such as the weights of a ranking
function, the experiment becomes an optimization, and each evaluation is an
A/B test that runs for days on live traffic.

This setting stretches the method in three ways. The noise is large, because
user behavior varies far more than the differences between variants. The
evaluations come in batches, since several variants run at once. And there
are constraints: a variant that improves one metric must not degrade others.
@letham2019constrained developed noisy expected improvement for this setting
(@sec-practice-noisy-ei) and demonstrated it at Facebook on a ranking system
and on the flags of a server compiler. The Ax platform packages the approach
for general adaptive experimentation [@olson2025ax].

Online experiments are also where the scores of @sec-regret-definitions
part ways. Every variant is shown to real users, so a bad variant has a cost
while it runs: cumulative regret matters, and with a small, fixed set of
variants the problem is a bandit (@sec-bandits). And the outcomes are several
metrics, not one, so someone must say how they trade off. Preference
exploration, which learns a decision maker's utility over predicted outcomes
from comparisons, was motivated by this problem [@lin2022preferenceb]. As of
September 2026, we found no public report that quantifies preferential
Bayesian optimization (PBO) in production A/B tests; @sec-sci-industry reviews
what is known.

## People in the loop {#sec-app-people}

In the applications so far, an instrument produced the number. In a growing
set of applications a person is part of the evaluation, in one of two ways.

**The person is the system being optimized, and an instrument measures the
result.** A wearable robot must be tuned to its wearer, and the objective can
be physiological. @ding2018human tuned the peak and offset timing of the hip
assistance of a soft exosuit by Bayesian optimization of the metabolic cost of
walking. The optimum was found in 21.4 ± 1.0
minutes on average, and metabolic cost fell by 17.4 ± 3.2% compared with
walking without the device. The evaluation is slow and noisy: several minutes
of walking yield one estimate of energy cost. @kim2017human had earlier
compared Bayesian optimization with a gradient descent method on a simpler
task, finding the step frequency that minimizes metabolic cost, and reported
faster convergence (12 minutes) with less variability between participants.
The person also adapts to the device during the session, so the function
being optimized moves; @sec-cs-exoskeleton takes up this problem.

**The person is the measuring instrument.** Some objectives exist only as
judgments. To demonstrate their tuning service, Google engineers optimized a
chocolate chip cookie recipe, with parameters that included the amounts of
sugar, butter, salt, and cayenne, and the baking time and temperature. Batches were baked, tasters in the company's cafes filled in a
survey, and the aggregated ratings went back to the optimizer. "The cookies
improved significantly over time", the authors report, and the exercise
needed features that real experiments need: marking infeasible recipes (too
little butter makes a dough that will not hold together), accepting that a
chef changed a suggested recipe, and transferring what was learned at small
scale to large-scale baking [@golovin2017google].

Ratings such as these are the simplest way to bring human judgment into the
loop, and they have known weaknesses. A rating scale has no fixed zero and
no fixed unit; people use it differently from one another and differently at
the end of a session than at the beginning (@sec-cmp-ratings-hard). Asking
which of two options is better avoids the scale, at the price of less
information per answer and a model that no longer has Gaussian observations.
That trade is the subject of @sec-part-preferences, and @sec-cs-photo works
through an example in which the only instrument is the eye.

Three things change when a person is in the loop, whatever the form of the
feedback (inference, from the studies cited in this section and in
@sec-part-humans). The budget is a person's time and patience, tens of
evaluations rather than hundreds. The noise is human: it depends on fatigue,
on what was shown before, and on how the question is asked. And the person
experiences every option tried, so the journey matters as well as the
destination, which is the difference between cumulative and simple regret.

## Neighboring methods {#sec-bo-neighbors}

Bayesian optimization shares its machinery with several other fields, and
the names are easy to confuse. The differences are mostly differences of
goal: what counts as success determines where to evaluate next.

### Five neighbors {#sec-app-five-neighbors}

**Active learning** wants an accurate model everywhere, with as few measured
inputs as possible. The learner chooses which inputs to have measured (labeled,
in the field's vocabulary), and a standard rule is uncertainty sampling
[@settles2009active], which @sec-info-sequential met as the greedy rule for
gathering information: ask about the input the model is least sure of, which
with a Gaussian process means evaluating where the posterior variance is
largest. The machinery is that of
Bayesian optimization with the acquisition function changed, and the outcome
is different: evaluations spread evenly, including over regions where the
function is low and of no interest to an optimizer.

**Bayesian experimental design** is the general theory behind both. It
chooses an experiment to maximize the expected utility of its outcome, where
the utility encodes the purpose of the experiment [@chaloner1995bayesian].
When the purpose is to learn, the utility is the information gained, a
criterion that goes back to @lindley1956measure and that @mackay1992information
turned into rules for selecting data (@sec-expected-information-gain).
Bayesian optimization is experimental design with a different utility: the
value of the best point found. Entropy search (@sec-entropy-search) makes the
link explicit by seeking information about the location of the maximum.

**Bandits** score every evaluation, not only the final answer
(@sec-regret). A bandit algorithm on a continuous domain with a Gaussian
process model is Bayesian optimization judged by cumulative regret
(@sec-gp-bandits), and the upper confidence bound serves both. The fields
differ in emphasis: bandit research proves guarantees, usually for many cheap
rounds; Bayesian optimization research builds methods for few expensive ones.

**Reinforcement learning** chooses sequences of actions in an environment
whose state changes in response, to maximize reward over time
[@sutton2018reinforcement]. A bandit is the special case with a single state:
the action affects the reward but not the situation the learner faces next.
Bayesian optimization therefore solves a much simpler problem, and it is
often used around reinforcement learning rather than instead of it: to tune
the hyperparameters of a learning system, as in AlphaGo, or to search
directly over the few parameters of a robot's controller, as in gait
optimization, where each evaluation runs the whole controller and returns
one number.

**Evolution strategies** search without a model. They keep a population of
candidates, or a distribution over candidates, evaluate samples from it, and
shift the distribution toward the better samples. CMA-ES, the standard
method, adapts a full covariance matrix for its sampling distribution
[@hansen2001completely]. Each step is cheap to compute, and nothing limits
the number of evaluations, so evolution strategies are the natural choice
when evaluations are cheap and plentiful. On a robot control benchmark with a
budget of 10,000 evaluations, CMA-ES outperformed every Bayesian optimization
method except the trust region method it was compared against
[@eriksson2019scalable]. With only tens of evaluations a model-based method
has the advantage, because a population needs many evaluations just to
estimate a direction (inference; @fig-app-goals shows a small case).
One well-known study optimized exoskeleton assistance during walking with an
evolution strategy, reducing metabolic energy consumption by 24.2 ± 7.4%
compared with no torque [@zhang2017human]. When people rate or choose among
the candidates of each generation, the method is called interactive
evolutionary computation [@takagi2001interactive]; @sec-adj-iec compares it
with PBO.

### Same data, different goals {#sec-app-goals}

The figure below runs four rules on the running objective with the same
budget: uncertainty sampling, Bayesian optimization with an upper confidence
bound, a simple evolution strategy that keeps one current point and mutates
it, and random search. It then scores all four in three ways.

```{figure}
//| figure: app-goals
//| label: fig-app-goals
//| fig-cap: "Four rules for choosing evaluations, on the running objective with noise sd 0.05. Top: where each rule evaluated in one run (ticks; taller for repeated inputs, a dot on the newest). Bottom: a score averaged over 20 runs that differ in the starting point and the noise: the error of a Gaussian process fitted to the rule's evaluations (root mean squared error against the true function), the simple regret of the best input evaluated, or the cumulative regret. Choose the score, and step through the evaluations. The objective, the kernel, and the budget are illustrative."
```

Some things to try.

**Start with the model error.** Uncertainty sampling ends with the most
accurate model: after 30 evaluations its error is about 0.04, against about
0.06 for both Bayesian optimization and random search. Its ticks are spread
evenly across the domain. Bayesian optimization's ticks pile up on the tall
peak, and its model of the rest of the function stops improving after about
fifteen evaluations.

**Switch to simple regret.** The ranking reverses. Bayesian optimization has
found the maximum in essentially every run by 15 evaluations; after 30,
uncertainty sampling and random search are still about 0.03 and 0.05 short on
average, because they never concentrate on the peak. The evolution strategy
is worst here: in many runs it climbs whichever bump is nearest its starting
point and stays, since it has no model to tell it that the rest of the domain
is unexplored.

**Switch to cumulative regret.** Bayesian optimization's curve flattens after
about ten evaluations, because nearly every later evaluation is near the
maximum. Uncertainty sampling and random search keep paying the same amount
per evaluation, and end near 22 and 21 against 7. The evolution strategy does
better than they do on this score, despite its poor final answer: it spends
its evaluations near a good point, although not the best one.

**Step back to five evaluations.** Early on the rules are hard to tell apart.
With little data, the upper confidence bound is dominated by uncertainty, so
Bayesian optimization starts as uncertainty sampling and departs from it only
once the model has something to exploit (@exr-app-ucb-limits).

::: {.keyidea title="The goal chooses the query"}
Active learning, Bayesian optimization, and bandits can share one model and
differ in a single line: the rule that turns the posterior into the next
evaluation. Before choosing a method, decide what will be scored: the model,
the final answer, or everything along the way.
:::

### Choosing among them {#sec-app-choosing}

@tbl-app-neighbors summarizes the comparison. The budgets are orders of
magnitude drawn from the examples in this chapter, not rules (inference).

::: {.table #tbl-app-neighbors title="Bayesian optimization and its neighbors"}
| Method | Goal | Uses a model of the objective | Typical budget | Reach for it when |
|---|---|---|---|---|
| Bayesian optimization | best input found | yes | tens to hundreds | evaluations are expensive and inputs are few |
| Active learning | accurate model everywhere | yes | tens to thousands | the model itself is the product |
| Bayesian experimental design | any stated utility, often information | yes | a few to hundreds | the purpose of the experiment can be written as a utility |
| Bandits | reward summed over all rounds | optional | thousands and more | every evaluation has consequences |
| Reinforcement learning | reward over sequences of actions | optional | very many | actions change the state of the world |
| Evolution strategies | best input found | no | thousands and more | evaluations are cheap, or the dimension is high |
:::

The boundaries are porous. TuRBO borrows the trust region from classical
local search (@sec-hd-practice-local); BOHB borrows early stopping from
bandits (@sec-multi-fidelity); the knowledge gradient and entropy search are
experimental design criteria aimed at the optimum
(@sec-knowledge-gradient). What Bayesian optimization contributes to the
family is the explicit probabilistic model of an expensive objective, and
the habit of asking, before each evaluation, what that evaluation is worth.
The next part keeps the model and the habit, and changes the observation: a
person's choice between two options instead of a number.

## Exercises {#sec-app-exercises}

::: {.exercise #exr-app-which}
For each problem, say which of the methods in @tbl-app-neighbors you would
try first, and why. (a) A compiler has 30 numeric flags; one benchmark run
takes 40 milliseconds. (b) A lab can run one polymer synthesis per day and
varies three process settings. (c) A news site wants to choose among five
headlines for an article that will be read for the next six hours. (d) An
engineering team needs a fast approximation of a slow simulator, accurate
over the whole range of four inputs, to use in later studies.

::: {.solution}
(a) An evolution strategy or random search: hundreds of thousands of
evaluations are affordable, and with evaluations this cheap the overhead of
fitting a model would exceed the cost of the evaluations it saves. (b)
Bayesian optimization: few inputs, a budget of a few dozen evaluations, each
one expensive. (c) A bandit algorithm such as Thompson sampling: there are
five arms, every reader shown a worse headline is a loss, and the score is
cumulative. (d) Active learning or a space-filling design (@sec-initial-design):
the goal is an accurate model everywhere, not a maximum, so evaluations
should go where the model is uncertain.
:::
:::

::: {.exercise #exr-app-attia}
@attia2020closed report finding good charging protocols in 16 days instead
of more than 500. Explain why 500/16 is not the acceleration factor of
Bayesian optimization as @adesiji2026benchmarking define it, and what
additional comparison would isolate the optimizer's contribution.

::: {.solution}
The study changed two things at once: an early-prediction model shortened
each experiment, and Bayesian optimization reduced the number of experiments.
The 500 days refer to exhaustive search without early prediction, so the
ratio of about 31 combines both effects and is measured in time, not in
experiments. The acceleration factor compares the *number of experiments*
needed to reach a target with and without the optimizer, everything else
equal. To isolate it, one would compare Bayesian optimization with a
reference strategy, such as random or exhaustive search, both using early
prediction, and count experiments to reach the same cycle life.
:::
:::

::: {.exercise #exr-app-ucb-limits}
The upper confidence bound rule evaluates at the maximizer of
$\mu_n(\vx) + \beta^{1/2}\sigma_n(\vx)$ (@eq-loop-ucb). Show that it becomes
uncertainty sampling as $\beta \to \infty$ and pure exploitation at
$\beta = 0$. Use this to explain why Bayesian optimization and uncertainty
sampling behave alike during the first few evaluations in @fig-app-goals.

::: {.solution}
Dividing the score by $\beta^{1/2}$ does not change its maximizer, and
$\mu_n(\vx)/\beta^{1/2} + \sigma_n(\vx) \to \sigma_n(\vx)$ as
$\beta \to \infty$, so the rule picks the input of largest posterior standard
deviation, which is uncertainty sampling. At $\beta = 0$ the score is
$\mu_n(\vx)$ alone, pure exploitation. For a fixed $\beta$, what matters is
the size of the variation in $\beta^{1/2}\sigma_n(\vx)$ across the domain
relative to the variation in $\mu_n(\vx)$. With few evaluations,
$\sigma_n(\vx)$ ranges from near zero at the data to the prior standard
deviation elsewhere, while $\mu_n(\vx)$ is still close to the constant prior
mean over most of the domain, so the uncertainty term decides and the rule
explores like uncertainty sampling. As data accumulate, $\sigma_n(\vx)$
shrinks everywhere and differences in $\mu_n(\vx)$ take over.
:::
:::

## Further reading {#further-reading .unnumbered}

- @shahriari2016taking is a survey of Bayesian optimization organized around
  its applications, with the title's promise of taking the human out of the
  loop; a good complement to this book, which puts the human back in.
- @shields2021bayesian is the chemistry study, with the game against 50
  chemists; @sec-cs-chemistry works through its data.
- @calandra2016bayesian compares gait optimization methods on real robots and
  is a careful example of evaluating Bayesian optimization on hardware.
- @letham2019constrained describes Bayesian optimization of online
  experiments, where noise and constraints dominate.
- @golovin2017google describes a tuning service used at scale, including
  transfer learning, early stopping, and the cookies.
- @settles2009active surveys active learning, and @chaloner1995bayesian
  reviews Bayesian experimental design; both are the right starting points
  for the neighbors of @sec-bo-neighbors.
- @lattimore2020bandit and @sutton2018reinforcement are the standard texts on
  bandits and reinforcement learning.
