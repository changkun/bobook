---
status: done
synopsis: "A re-enactment of human-in-the-loop exoskeleton tuning on a simulated walker calibrated to published studies: what a two-minute metabolic estimate is worth, a session in which you tune the device by feel against Bayesian optimization, an evolution strategy, and preferential optimization on the same walking time, and what a person who is still adapting does to all of them."
sources: ["Zhang et al. 2017", "Ding et al. 2018", "Poggensee and Collins 2021", "Kutulakos and Slade 2024", "Schäfer et al. 2026"]
---

# Tuning an Exoskeleton with a Person in the Loop {#sec-cs-exoskeleton}

In @sec-cs-chemistry people chose the search space and then watched an
optimizer work on measured yields. Here a person is part of the measurement.
An ankle exoskeleton pushes the foot down at the end of each step, and if the
push has the right size at the right moment, walking costs the wearer less
energy. The right size and moment differ from person to person, so the device
is tuned for each wearer, with the wearer walking in it and an optimizer
choosing what to try next. This is optimization with a person inside the
loop, and it is one of the best documented cases of it: the studies that
established the approach date from 2017, and @sec-health reviews what they
and their successors found.

This chapter takes the tuner's seat. It works one problem end to end: what is
being optimized and what one evaluation costs, what the published studies
did, a tuning session that you run yourself, and the two complications that
make the case instructive, a person who adapts while being measured and the
discovery that people tune such devices quite well by themselves.

One thing must be said plainly at the start. A reader cannot wear an
exoskeleton through a web page, and no published study released a person's
full metabolic landscape. So every interactive figure in this chapter runs on
a **simulated walker**, a model whose numbers were set to match published
measurements. The text says which values are measured and where they come
from, and which are assumptions. What the simulation shows is what the
published numbers imply when they are put together; it is not new evidence.

## The problem {#sec-cs-exo-problem}

### The device and its parameters {#sec-cs-exo-device}

An **exoskeleton** is a wearable robot that applies a torque, a turning force,
at a joint. The controller repeats a torque profile once per **gait cycle**,
the interval from one heel strike to the next heel strike of the same foot.
In the line of ankle exoskeleton experiments that began with
@zhang2017human, the profile is defined by four numbers: the peak torque,
the time in the gait cycle at which the peak occurs, and the times the torque
takes to rise and to fall [@slade2022personalizing]. Each is confined to a
safe range. The peak torque is normalized to body mass and allowed between 0
and 1 N·m per kilogram [@slade2022personalizing]; in the experiments of
@poggensee2021adaptation the peak, rise, and fall times were kept within 35%
to 55%, 10% to 40%, and 5% to 20% of the gait cycle, as
@kutulakos2024simulating report in the preprint that reuses those data.

To keep the whole problem visible on a map, this chapter tunes two of the
four: the **peak torque**, between 0 and 1 N·m per kilogram, and the **peak
time**, between 35% and 55% of the gait cycle. A setting is a point
$\vx = (x_1, x_2)$ in the unit square, with $x_1$ the peak torque and $x_2$
the peak time rescaled to $[0, 1]$. Two parameters is not a toy
size for this field: @ding2018human tuned exactly two, the peak and offset
timing of hip assistance.

### What is optimized {#sec-cs-exo-objective}

The classical objective is **metabolic cost**, the rate at which the body
uses energy while walking. It is estimated by indirect calorimetry: the
wearer breathes through a mask that measures the oxygen consumed and the
carbon dioxide produced. A good setting lowers the metabolic rate below that
of walking in the same device with the motors off. Using this measurement to
drive an optimizer is called **human-in-the-loop optimization**.

The other objective is what the wearer prefers. It needs no mask, it includes
comfort and the feeling of stability that a metabolic number leaves out, and
it can be read much faster. It is also a different quantity: around the
settings that participants chose by feel for a hip exoskeleton, the metabolic
rate showed no clear minimum, and the authors assume that the participants
were weighing more than effort, comfort for one [@schafer2026user]. Both
objectives appear below, the first as a measurement with noise, the second
as a comparison the wearer feels.

### What one evaluation costs {#sec-cs-exo-cost}

The metabolic rate cannot be read off instantly. After a change of setting,
the rate measured at the mouth moves toward its new level gradually, because
the body's oxygen stores and transport delay the response. @selinger2014estimating
showed that the response during walking is well described as a first-order
system, one that closes a fixed fraction of the remaining gap per unit time,
with a time constant of 42 ± 12 seconds (mean ± standard deviation across
subjects), and that individual breaths scatter widely around it. Waiting for
the rate to settle takes several minutes per setting. The laboratory
protocol that followed @zhang2017human instead has the wearer walk each
setting for two minutes while the breaths are recorded, and estimates from
them the steady level the response is heading for; the two minutes are a
compromise between the time per setting and the accuracy of the estimate
[@slade2022personalizing].

The estimate is noisy. @kutulakos2024simulating put its standard deviation at
4.6% of the metabolic rate, for a first-order fit to two minutes of data, and
cite the experiments of @zhang2017human for it; the supplement of that study
gives an average error of 4% for two-minute estimates. The figure below
simulates one such bout.

```{figure}
//| figure: cs-exo-measure
//| label: fig-cs-exo-measure
//| fig-cap: "One evaluation, simulated. The dots are breath-by-breath readings of the metabolic rate after a change of setting, the dashed curve the true first-order response toward the new level, and the solid curve the fit whose end level is the estimate. Measured values used: the time constant of 42 seconds [@selinger2014estimating] and the 4.6% standard deviation of a two-minute estimate [@kutulakos2024simulating]. Simulated or assumed: everything else, including one breath every three seconds and a breath-to-breath scatter of about 18%, which is the value that reproduces the 4.6%. The strip below shows the estimates from 60 repeated bouts of the chosen length. Change the length of the bout and press *Walk another bout*."
```

Some things to try. At two minutes the estimate has a standard deviation of
4.6 points around the true value, by construction; the figure prints this
value as "theory" next to the spread of the 60 bouts it happened to draw,
which is 4.4 points. Shorten the bout to one minute and the standard
deviation more than doubles, to 10.6 points, because the response has barely
left its starting level and the fit must extrapolate. Lengthen it to six
minutes and it falls to 1.9 points, at three times the cost. Then set the
true change to −3% and look at the strip at two minutes: an improvement of
that size is invisible in a single bout.

That last observation shapes the whole problem. The benefit of tuning for the
individual, over a good generic setting, is a matter of a few points: in
@poggensee2021adaptation a generic controller reduced the metabolic rate of
trained users by 31%, and customized assistance by 39%. Near the optimum,
neighboring settings differ by less than the noise of one estimate, so an
optimizer has to average, through its model, over many evaluations
(@exr-cs-exo-bouts). At two minutes each, a one-hour session buys thirty.
The metabolic optimization that @slade2022personalizing ran as their
laboratory reference took 128 minutes of walking.

### The person changes {#sec-cs-exo-moving}

The last ingredient is that the walker is not a fixed function. People learn
to use an exoskeleton. In @poggensee2021adaptation, naive users needed about
109 minutes of assisted walking to become expert; training contributed about
half of the final 39% reduction and customization about one quarter; the
generic controller that gave 31% after training gave only 10% before it; and
the best peak torque kept growing slowly over the whole study, which the
authors read as adaptation on a longer time scale. A tuning session of twenty
minutes is therefore run on a person who will not exist an hour later.
@sec-cs-exo-adaptation returns to what that does to an optimizer.

## What the studies did {#sec-cs-exo-studies}

Three ways of tuning have been tried on people, and they differ in who
judges a setting. @tbl-cs-exo-studies lists one or two studies of each kind
with the numbers this chapter uses; @sec-health-table has the full set, with
participants and validation, and @sec-health-manual discusses how far the
results can be compared.

::: {.table #tbl-cs-exo-studies title="Three ways of tuning an exoskeleton, with the published protocols and results used in this chapter. Reductions are relative to the baseline each study names; the baselines differ, so the rows are not a ranking."}
| Who judges | Study | Device and parameters | Tuner | Time and result |
|---|---|---|---|---|
| metabolic estimate | @zhang2017human | one ankle, 4 parameters, 11 participants | evolution strategy (CMA-ES) | 64 min of walking for 9 of the 11; 24.2 ± 7.4% below zero torque |
| metabolic estimate | @ding2018human | hip exosuit, 2 timing parameters, 8 participants | Bayesian optimization | 40 min of optimization, converged after 21.4 ± 1.0 min; 17.4 ± 3.2% below walking without the device |
| metabolic estimate | @poggensee2021adaptation | both ankles, 4 parameters, naive users in three training groups | CMA-ES, as @kutulakos2024simulating describe it, for the group with customized assistance | 39% below the device turned off, after about 109 min of training |
| wearable sensors | @slade2022personalizing | ankles, 4 parameters | model that ranks settings from ankle motion, 30 s per setting | 32 min in the laboratory, where metabolic estimates took 128 min |
| the wearer's comparisons | @tucker2020human | walking gaits, 6 parameters, 6 participants | preference learning along random lines | 30 gait trials and 6 validation trials each |
| the wearer's comparisons | @lee2023user | ankle, 4 parameters | evolutionary algorithm with a learned ranker | settings stable after 43 ± 7 comparisons |
| the wearer, by hand | @ingraham2022role | ankle, torque size and timing, 24 participants | self-tuning, blind to the values | converged in 105 seconds per trial |
| the wearer, by hand | @schafer2026user | hip, 4 timing parameters, 11 participants | self-tuning with a thumbstick | 10.9 ± 0.9 min, 30.5 settings; 16.6 ± 1.1% below zero torque |
:::

Two optimizers recur in the first rows. Bayesian optimization is the loop of
@sec-bo-loop: a Gaussian process models the metabolic rate over the
settings, and an acquisition function picks the next setting to walk.
@ding2018human chose it because it is suited to noisy signals and very few
evaluations. The other is the **covariance matrix adaptation evolution
strategy**, CMA-ES [@hansen2001completely], introduced in
@sec-app-five-neighbors. It keeps no model of the landscape. It draws a small
*generation* of settings from a Gaussian distribution, has the wearer walk
each one, moves the distribution's mean toward the better half, reshapes its
covariance along the directions that helped, and repeats. Its estimate of the
best setting is the mean. It forgets every generation after using it, which
makes it wasteful when evaluations are scarce and, as we will see, forgiving
when the person changes.

The preference studies replace the mask with the wearer's judgment. CoSpar
and LineCoSpar asked wearers which of two gaits they preferred and let them
suggest improvements [@tucker2020preference; @tucker2020human]. The authors of
the first note that users had difficulty remembering more than two trials,
a reason to compare each trial only with the one just before it. A 2026
preprint tuned six parameters of a hip exoskeleton from pairwise
comparisons in sessions of 20.6 ± 4.6 minutes, validation included, for five
participants [@liu2026personalized].

The self-tuning studies remove the optimizer too. The wearer holds a control,
changes a parameter, feels the result, and stops when satisfied. In
@schafer2026user the participants spent 18.7 seconds per setting on average
and changed only one parameter at a time in 97.5% of their adjustments.

## A session, re-enacted {#sec-cs-exo-session}

### The simulated walker {#sec-cs-exo-walker}

The simulated walker has a **reduction** $R(\vx, t)$: the fraction by which
setting $\vx$ lowers the metabolic rate below zero-torque walking, after $t$
minutes of assisted walking. It is the product of an amplitude, a torque
term, and a timing term,

$$
R(\vx, t) = A(t)\; s\big(x_1;\, T^\star(t)\big)\; \exp\!\left(-\frac{\big(x_2 - P^\star(t)\big)^2}{2 \cdot 0.45^2}\right),
$$ {#eq-cs-exo-walker}

where $T^\star$ and $P^\star$ are the walker's best torque and best peak
time. The torque term $s$ is zero at zero torque, rises to one at $T^\star$
as $1 - \big((x_1 - T^\star)/T^\star\big)^2$, and falls beyond it as
$1 - \big((x_1 - T^\star)/0.5\big)^2$, so too much torque can make walking
cost more than no torque at all. The timing term is wide: a peak that is 4%
of the gait cycle away from the best time costs an adapted walker about 3.7
points, less than the noise of one estimate.

Adaptation moves all three quantities. The walker's level of adaptation is
$a(t) = 1 - e^{-t/36}$, which reaches 95% at 108 minutes. The amplitude grows
from $A = 0.135$ for a novice to $0.39$ for an expert, the best peak time
shifts by 2.4% to 4.8% of the gait cycle, and the best torque rises from
roughly 0.45 to roughly 0.85 N·m per kilogram, half as fast as the rest. Each
walker has their own $T^\star$ and $P^\star$, drawn at random. A *generic
setting*, the same for everyone, sits at a torque of 0.6 and a peak time of
45%.

@tbl-cs-exo-calibration lists what the numbers were matched to.

::: {.table #tbl-cs-exo-calibration title="What the simulated walker and the tuners were calibrated to. The upper rows are measurements from the cited studies; the last rows are assumptions with no published value."}
| Quantity | Published value | In the simulation |
|---|---|---|
| reduction with a generic setting, before training | 10% [@poggensee2021adaptation] | 10.0% on average over all walkers the figure can draw, at $t = 0$ |
| reduction with a generic setting, after training | 31% [@poggensee2021adaptation] | 31.2% on average over all adapted walkers the figure can draw (for the 40 walkers of @sec-cs-exo-many, the mean is 32.1% and the median 32.9%) |
| reduction with customized assistance, after training | 39% [@poggensee2021adaptation] | 39% at every adapted walker's optimum |
| time to become an expert user | about 109 minutes; best peak torque grows more slowly [@poggensee2021adaptation] | 95% adapted at 108 minutes; best torque adapts half as fast |
| noise of a two-minute metabolic estimate | standard deviation 4.6% [@kutulakos2024simulating] | Gaussian, standard deviation 0.046 |
| ranges of the two parameters | peak torque 0 to 1 N·m per kilogram [@slade2022personalizing]; peak time 35% to 55% of the gait cycle [@kutulakos2024simulating] | the same |
| time per self-tuned setting | 18.7 seconds [@schafer2026user] | the same |
| CMA-ES settings | $4 + \lfloor 3 \ln N \rfloor$ per generation, initial step 30% of the range [@kutulakos2024simulating] | 6 per generation for $N = 2$, step 0.3 |
| Bayesian optimization settings | Matérn kernel, confidence bound with exploration constant 2.6, or 0.93 for a less exploring variant [@kutulakos2024simulating] | the same |
| a novice's best possible reduction | none found | 13.5% at $t = 0$ (set so that the generic setting gives 10%) |
| what the wearer feels | none found | see below |
| time per felt comparison of two settings | none found | 37.4 seconds, two settings at 18.7 seconds |
:::

The felt signal is the least certain part of the model. The wearer's felt
effort is the metabolic cost $1 - R$ plus a dislike of high torque,
$0.1\,(1 - a_T)\,x_1^2$, which fades as the walker adapts (with $a_T$ the
slower adaptation level of the torque). The dislike reflects a measured
tendency, that naive users of an ankle exoskeleton preferred more torque as
the experiment went on [@ingraham2022role], but its size is invented. Each
reading of the felt effort carries Gaussian noise of standard deviation 0.03,
and a difference between two readings smaller than 0.025 is reported as
"about the same". Those two numbers are assumptions; a perception floor of
this kind is real, and for the stiffness of an ankle exoskeleton during
walking the smallest noticeable change has been measured at about 42%
[@maberry2026just].

### Four tuners {#sec-cs-exo-tuners}

Four tuners work on the walker, each on a fresh copy, for the same minutes
of walking.

- **You**, by hand. You choose a setting, the walker tries it for 18.7
  seconds, and you are told whether it feels easier, harder, or about the
  same as the previous setting. You never see a number.
- **Bayesian optimization** on two-minute metabolic estimates. A Gaussian
  process with a Matérn 5/2 kernel models the measured rate; after four
  spread-out starting settings, the next setting minimizes the lower
  confidence bound $\mu(\vx) - 2.6\,\sigma(\vx)$, the minimizing counterpart
  of the upper confidence bound of @sec-ucb; the recommendation is the
  setting with the lowest posterior mean.
- **CMA-ES** on the same estimates, six settings per generation, starting at
  the center of the square. Its recommendation is its current mean.
- **Preferential Bayesian optimization** (PBO, @sec-pbo) on what the walker
  feels. Each query is a pair of settings chosen by EUBO (@sec-eubo); the
  walker tries both and says which feels easier; a comparison takes 37.4
  seconds. The recommendation is the compared setting with the highest
  posterior mean utility.

In 24 minutes that is 77 settings by hand, 38 comparisons, or 12 metabolic
estimates, which for CMA-ES is two generations.

```{figure}
//| figure: cs-exo-tune
//| label: fig-cs-exo-tune
//| fig-cap: "Tune a simulated walker yourself. This is a calibrated simulation, not data: the walker's response follows @eq-cs-exo-walker with the published numbers of @tbl-cs-exo-calibration (reductions of 10%, 31%, and 39%, about 109 minutes to adapt, 4.6% measurement noise, 18.7 seconds per setting), and what the walker feels is an assumption. Set the peak torque and the peak time with the sliders or by clicking the map, press *Walk*, and read how the setting felt compared with the previous one (green: easier; red: harder; gray: about the same). When you press *I'm done*, the setting on the sliders is kept, the true landscape at that moment is revealed (darker is a larger metabolic reduction), and three optimizers are run on fresh copies of the same walker for the same walking time. The table gives the true metabolic reduction of each final setting, now and, for a walker new to the device, once the walker has fully adapted. *Another walker* draws a new person."
```

Some things to try. Tune the default walker, who is new to the device, for
five to ten minutes of walking time (about fifteen to thirty settings), and
keep the setting you trust. Most readers find the same thing the simulated
person in the next figure finds: large mistakes are easy to feel and to
undo, and near the end almost every change feels "about the same". Then
compare your row of the table with the others, and use *Show trail of* to see
where each optimizer spent its evaluations. Switch to *Already adapted* and
tune again: the landscape is larger and steeper, and the feel is more
decisive. Finally try the 48-minute session on a novice and tune for all of
it; the walker's optimum moves while you work, and the best torque at the
end is not the best torque at the start.

For readers without a pointer, or without patience, the button *Let a
simulated person tune* runs a simple self-tuner: it starts at the generic
setting, changes one parameter at a time, keeps a change only if it feels
easier, and shrinks its step after repeated failures. The figure below shows
its session on the default walker.

```{figure}
//| figure: cs-exo-tune
//| label: fig-cs-exo-tuned
//| fig-cap: "A finished session, entirely simulated. The simulated person of the text tuned the default walker (new to the device) for 10.9 minutes, the mean duration in @schafer2026user, trying 35 settings; the dots are colored by how each felt against the one before. The star is the setting kept; the square, triangle, and diamond are the final settings of Bayesian optimization (5 metabolic estimates), CMA-ES (5 estimates, less than one generation, so still at its starting point in the center), and preferential Bayesian optimization (17 felt comparisons) after the same 10.9 minutes; the dashed circle is the generic setting. The cross is the walker's metabolic optimum at that moment, and the faint cross the optimum once adapted. The published numbers behind the walker are those of @tbl-cs-exo-calibration."
log: "0.6:0.5,0.8:0.5,0.6:0.5,0.4:0.5,0.6:0.5,0.6:0.7,0.6:0.5,0.6:0.3,0.6:0.5,0.7:0.5,0.6:0.5,0.5:0.5,0.6:0.5,0.6:0.6,0.6:0.5,0.6:0.4,0.6:0.5,0.65:0.5,0.6:0.5,0.55:0.5,0.5:0.5,0.45:0.5,0.5:0.5,0.55:0.5,0.5:0.5,0.5:0.45,0.5:0.5,0.5:0.55,0.5:0.5,0.45:0.5,0.5:0.5,0.55:0.5,0.5:0.5,0.5:0.45,0.5:0.4"
torque: 0.5
time: 44
done: true
sim: true
reveal: true
```

Read the table in the figure. After 10.9 minutes the walker's best possible
reduction is 20.2%, and the four tuners reach 20.0% (by hand), 18.3%
(Bayesian optimization), 19.6% (CMA-ES, which has not finished a generation
and still recommends its starting point), and 18.9% (preferences), against
18.9% for the generic setting that nobody tuned. The differences between the
tuners, and between them and no tuning at all, are less than two points. With
noise of 4.6 points per metabolic estimate, a real experiment would need
dozens of evaluation bouts per setting to tell them apart
(@exr-cs-exo-bouts).

The last column is the uncomfortable one. Judged on the walker as they will
be once adapted, the settings found in this session give 24% to 30%, while
the untuned generic setting gives 31.2%. The session tuned the device for a
novice who cannot yet use much torque, and the adapted walker wants far
more: the faint cross in the map sits at a torque of 0.88, the settings of
the four tuners between 0.35 and 0.5.

### Across many walkers {#sec-cs-exo-many}

One walker is an anecdote. Running the same four tuners on 40 simulated
walkers gives the following medians for walkers who are already adapted, so
that the landscape stands still. After 12 minutes, Bayesian optimization
reaches a reduction of 38.2%, PBO 37.9%, the simulated
self-tuner 37.8%, and CMA-ES 35.9%; the best possible is 39.0% and the
generic setting gives 32.9%. The self-tuner and the preference optimizer are
already at 37.5% after six minutes, when Bayesian optimization has had three
estimates and stands at 32.7%. CMA-ES reaches 37% after about 48 minutes and
stays near it.

So on a walker who holds still, everything except the evolution strategy
recovers most of the roughly six points between the generic setting and the
optimum within a quarter of an hour, and the two tuners that use the wearer's
feeling get there first, because they test six settings in the time one
metabolic estimate takes. This agrees in kind with what the studies report:
self-tuning in about eleven minutes [@schafer2026user], Bayesian optimization
of two parameters in about twenty-one [@ding2018human]. It also depends on an
assumption the simulation makes and the studies do not guarantee, that what
the adapted wearer feels points at the metabolic optimum.

## When the person adapts {#sec-cs-exo-adaptation}

A Gaussian process posterior treats every observation as a reading of one
fixed function (@sec-gp-regression). A walker who is learning the device is a
different function every few minutes. Measurements from the first ten minutes
describe a person who has since changed, yet they stay in the data with full
weight.

The evidence that this matters comes from several directions. The 109 minutes
of @poggensee2021adaptation are far longer than a tuning session. When people
meet a new exoskeleton behavior, the variability of their step frequency,
ankle angle, and muscle activity first rises and then falls, on different
time scales for different variables [@abram2022general]. Preferences move as
well: naive users preferred higher torque as the experiment progressed
[@ingraham2022role]. And in the preprint of @kutulakos2024simulating, which
simulated a novice by blending one subject's fitted landscape into another's
over 80 evaluations, Bayesian optimization, which had converged in about 60
evaluations on a fixed landscape, was slowed by the change, while CMA-ES
reached the optimum at a similar rate with and without it.

The figure below repeats that experiment on the walkers of this chapter.

```{figure}
//| figure: cs-exo-adapt
//| label: fig-cs-exo-adapt
//| fig-cap: "What adaptation does to the tuners, in simulation. Nothing here is measured: 40 simulated walkers following @eq-cs-exo-walker, calibrated as in @tbl-cs-exo-calibration, each tuned once by every method (sessions precomputed with tools/figure-data/cs-exo-race.ts). Each line is the median over walkers of the true metabolic reduction that a tuner's current recommendation would give, minute by minute; the bands are the interquartile ranges for Bayesian optimization and the self-tuner. *Judge each setting on* chooses the walker the recommendation is tested on: as they are at that minute, or as they will be once fully adapted, which is the walker who will use the device. The dashed line is the best possible reduction and the dotted line the generic setting. The preferential optimizer was run for 60 minutes only."
```

Three things can be read from it.

**Adaptation is larger than tuning.** With novices, judged minute by minute,
all the lines rise together and, after the first ten minutes, stay within
about two points of each other.
At 24 minutes the best possible reduction is 25.9%, the tuners are at 23.2%
to 23.6%, and the generic setting, which nobody tuned, is at 23.3%. At 60
minutes the tuners are at 31.4% to 32.5% and the generic setting at 31.3%.
What moves the lines from 15% to 35% is the walker's own learning. This is
the simulation's version of the finding it was calibrated to, that training
contributed about half of the benefit and customization about a quarter
[@poggensee2021adaptation].

**A setting tuned early is stale later.** Switch the judge to *The walker
once adapted*. The settings recommended after 24 minutes would give the
adapted walker 26.3% to 31.1%, below the 32.9% of the generic setting: the
optimizers have faithfully found low-torque settings for a novice. The
medians cross the generic setting only after 56 minutes for Bayesian
optimization, 60 for CMA-ES, and 62 for the self-tuner, and within the hour
it ran, PBO did not cross it. A short tuning session
on a new user can leave them worse off, later, than no tuning at all.

**How the tuner handles old data matters, and not always as expected.** Turn
on *Show Bayesian optimization variants*. Giving the Gaussian process the
time of each measurement as a third input, so that old measurements count
less for predictions about now, is the simplest model of drift
(@sec-rec-nonstationary). It costs a little on adapted walkers and helps on
novices: at 120 minutes the reduction on the adapted walker is 36.9% against
35.2% for the plain model. Exploring less, with the exploration constant
lowered from 2.6 to 0.93, hurts here: that variant ends at 31.8%, below the
generic setting, because it keeps returning to a region that was best for
the novice. @kutulakos2024simulating found the opposite on their landscapes:
with a simulated novice, the less exploring variant reached the optimum in
about 100 evaluations and the default variant more slowly. The authors
suggest, without testing it, that its frequent evaluations near the
estimated optimum let it follow the optimum as it moves; on fixed landscapes
with 12 and 20 parameters the same variant settled on a worse setting. The
two simulations differ in how the landscape moves, and neither is an
experiment; the disagreement is a reason to test the exploration constant on
the problem at hand, which those authors also advise, by simulation first
and then in pilot sessions. CMA-ES and the self-tuner, which keep no long
memory, track the change without any adjustment.

What should a practitioner do? The recommendations in @sec-rec-nonstationary
follow from this picture: let the person walk with the device before tuning
begins, give early measurements less weight or model time explicitly, and
tune again later. A preprint on a hip exoskeleton with 16 participants and
three parameters, which optimized walking speed, reports that a Bayesian
optimizer built for a changing response did better than the standard one in
effectiveness, model accuracy, and personalization [@kim2026validation]. For
preferences, the same problem has no tested solution: as of September 2026 we
found no preferential optimization method with a model of a drifting utility
(@sec-theory-drift), and the preference optimizer in the figure is the
slowest to recover.

## Against self-tuning {#sec-cs-exo-self-tuning}

The simulated self-tuner is a dozen lines of code with no model, and in the
figures above it keeps pace with the optimizers. That mirrors the most
informative comparison in the literature. In @schafer2026user, 11 people
without prior exoskeleton experience tuned four hip timing parameters by
thumbstick in 10.9 ± 0.9 minutes, and the resulting assistance lowered their
metabolic rate by 16.6 ± 1.1% compared with zero torque. By the authors' own
comparison, that is a quarter of the time of the optimization that
@ding2018human ran for two parameters, and half of the time after which that
optimization had converged. In @ingraham2022role, 24 people tuned ankle
torque and timing without seeing the values, converged in 105 seconds per
trial, and repeated their own choice with a standard deviation of 1.7 N·m and
1.5% of the gait cycle. @sec-health-manual discusses these studies and the caveats on
comparing them with optimizer studies, which used other devices and
baselines.

The simulation makes the reasons visible, and each is a property of the
problem more than of any algorithm.

**The wearer's sensor is fast.** A felt comparison takes seconds; a metabolic
estimate takes two minutes and is still noisy. In the same walking time the
hand tuner sees six times as many settings.

**The optimum is flat.** Near the best setting, the reduction changes by less
than the noise of a measurement and less than a person can feel. In
@schafer2026user, shifting a timing by up to ±8% of the stride time did not
change the metabolic reduction significantly. A coarse, fast search loses
almost nothing to a precise one.

**There are few parameters.** With two to four parameters, changing one at a
time works. Nothing here suggests it would with twenty.

The simulation also shows where self-tuning goes wrong, and these are the
same places where preferential optimization goes wrong, since both
listen to the same signal.

**What is felt is not what is measured.** The simulated novice dislikes
torque, so tuning by feel, by hand or by PBO, ends at
less torque than the metabolic optimum. For real wearers the gap between
preference and metabolic cost is documented in both directions: the hip
tuners of @schafer2026user did not sit at a metabolic minimum, and prosthesis
users preferred an ankle stiffness that made the motion of the prosthetic
and the intact joint symmetric, with no significant relation to metabolic
rate [@clites2021understanding]. Which of the two the device should serve is a
decision about the objective, to be made before any tuning.

**Below the perception floor, comparisons are noise.** Once every change
feels "about the same", further tuning by feel is a random walk. The floor
can be measured, about 42% for exoskeleton stiffness during walking
[@maberry2026just], and it gives a principled point to stop asking
(inference; the stopping rules of @sec-rec-stopping do not yet include one).

**People are not consistent with themselves.** Two of three people with an
amputation who explored the settings of a powered prosthesis chose different
settings in different trials on the same day [@diaz2026user].

What does this mean for the method? First, hand tuning is the baseline that a
preferential optimizer has to beat, and @sec-rec-three-questions makes
checking it the third question to ask before building one. No study has yet
run the two on the same people with the same time budget and an endpoint
fixed in advance; @sec-open-decisive describes that experiment, and the
figures of this chapter are its simulation, with the simulation's caveat that
the felt signal was written by us. Second, the simulation leaves out much of
what matters to a wearer. It has no comfort beyond one penalty term, no
safety, no fatigue, and no sense of agency, which in @schafer2026user fell
from 0.80 to 0.49 on a scale from 0 to 1 when assistance was switched on,
even though the participants had chosen the assistance themselves. And its
walkers are healthy adults, like nearly all participants of the studies it
was calibrated to (@sec-health-table).

::: {.keyidea title="What the case adds to the algorithm"}
In this problem the choice of optimizer is the smallest of the decisions.
What an evaluation is (two noisy minutes of breathing, or a few seconds of
feeling), when the person is tuned (before or after they have learned the
device), and which objective the device should serve (measured effort or
preference) each move the result by more than the difference between
Bayesian optimization, an evolution strategy, and a careful hand.
:::

The next chapter, @sec-cs-photo, removes the instrument altogether: the only
measurement is a person's choice between two versions of a photograph.

## Exercises {#sec-cs-exo-exercises}

::: {.exercise #exr-cs-exo-bouts}
A two-minute metabolic estimate has a standard deviation of 4.6 points. Two
settings in fact differ by 3 points. How many bouts per setting does it take
to detect the difference by the usual standard of a two-sided test at the 5%
level with 80% power, that is, a test that reports a difference in only 5% of
experiments when there is none and in 80% of experiments when the difference
is real? How many minutes of walking is that? Repeat for a difference of 8
points, the gap between generic and customized assistance in
@poggensee2021adaptation.

::: {.solution}
For two groups of $n$ independent estimates each, the difference of the
means has standard deviation $\sigma\sqrt{2/n}$. Detecting a difference
$\Delta$ at the 5% level with 80% power needs
$\Delta \ge (1.96 + 0.84)\,\sigma\sqrt{2/n}$, where 1.96 is the standard
normal value exceeded in size with probability 5% and 0.84 the value
exceeded with probability 20%, so
$n \ge 2 \cdot 2.8^2 \cdot (\sigma/\Delta)^2 = 15.7\,(\sigma/\Delta)^2$. With
$\sigma = 4.6$ and $\Delta = 3$: $n \ge 36.9$, so 37 bouts per setting, 74
bouts, 148 minutes of walking. With $\Delta = 8$: $n \ge 5.2$, so 6 bouts per
setting, 24 minutes. A gap of 8 points can be confirmed in one session; the
gaps of less than two points between tuners in @fig-cs-exo-tuned cannot, and
during those 148 minutes a new user would have adapted (@sec-cs-exo-moving).
:::
:::

::: {.exercise #exr-cs-exo-budget}
A session lasts 24 minutes. Count what each of the four tuners of
@sec-cs-exo-tuners gets to see. How many times does CMA-ES update its
distribution? Why does that explain its slow start in @fig-cs-exo-adapt, and
why does the same property protect it when the walker adapts?

::: {.solution}
By hand: $24 \cdot 60 / 18.7 = 77$ settings. Preferences:
$24 \cdot 60 / 37.4 = 38$ comparisons. Metabolic estimates: $24 / 2 = 12$,
for Bayesian optimization and for CMA-ES alike. CMA-ES with six settings per
generation completes two generations, so it updates its mean twice, the
first time after 12 minutes. Until then its recommendation is its starting
point. Bayesian optimization refits its model after every estimate. The
property that makes CMA-ES slow, that it uses a generation once and then
discards it, also means that measurements of the walker as a novice cannot
mislead it an hour later: nothing older than one generation is in its
state except through the mean and covariance they produced.
:::
:::

::: {.exercise #exr-cs-exo-feel}
In the simulation, a felt reading has Gaussian noise of standard deviation
0.03, and a change is reported as "easier" when the previous reading exceeds
the new one by more than 0.025. A change in fact lowers the felt effort by
0.02 (two points). With what probabilities is it reported as easier, about
the same, and harder? What does the simple self-tuner of the text do in each
case?

::: {.solution}
The difference of two readings is Gaussian with mean 0.02 and standard
deviation $0.03\sqrt{2} = 0.0424$. Easier: $\Prob(d > 0.025) =
1 - \Phi\big((0.025 - 0.02)/0.0424\big) = 1 - \Phi(0.118) \approx 0.45$.
Harder: $\Prob(d < -0.025) = \Phi\big((-0.025 - 0.02)/0.0424\big) =
\Phi(-1.06) \approx 0.14$. About the same: the remaining $0.40$. The
self-tuner keeps the change only in the first case, so it throws away a real
two-point improvement more often than it accepts it, and each rejection
costs two settings (the trial and the return). A two-point improvement is
within reach of neither the wearer's perception nor a single metabolic
estimate, which is the flat optimum seen from both sides.
:::
:::

::: {.exercise #exr-cs-exo-adapt}
Using $a(t) = 1 - e^{-t/36}$ for the amplitude and peak time and
$a_T(t) = 1 - e^{-t/72}$ for the torque, compute the adaptation levels after
a 24-minute session. A walker's best torque is 0.45 as a novice and 0.86 as
an expert. Where is it after 24 minutes, and what does the answer imply for
a setting tuned perfectly in that session?

::: {.solution}
$a(24) = 1 - e^{-2/3} = 0.49$ and $a_T(24) = 1 - e^{-1/3} = 0.28$. The best
torque is $0.45 + 0.28 \cdot (0.86 - 0.45) = 0.57$. A setting tuned perfectly
at 24 minutes has torque 0.57, while the adapted walker's best is 0.86. On
the adapted walker its torque term is
$1 - \big((0.57 - 0.86)/0.86\big)^2 = 0.89$, so even with perfect timing it
gives about $0.39 \cdot 0.89 = 34.6\%$ of the possible 39%. The session
optimized for a person who was halfway through learning the device.
:::
:::

## Further reading {#further-reading .unnumbered}

- @zhang2017human is the study that established human-in-the-loop
  optimization of exoskeletons, with an evolution strategy on two-minute
  metabolic estimates; @ding2018human is the Bayesian optimization
  counterpart on a hip exosuit.
- @selinger2014estimating explain why a metabolic rate can be estimated
  before it settles, and measure the time constant of the response.
- @poggensee2021adaptation separate what training, adaptation, and
  customization each contribute, with naive users followed until they became
  expert.
- @kutulakos2024simulating, a preprint, fit landscapes to those data and
  simulate optimizers on them, including a novice who adapts; the idea of
  this chapter's simulation comes from it.
- @schafer2026user and @ingraham2022role are the self-tuning studies:
  what people choose by feel, how fast, and how repeatably.
- @tucker2020preference and @tucker2020human tune exoskeleton gaits from the
  wearer's comparisons; @sec-health reviews the whole literature, including
  prostheses and the comparison with manual tuning.
