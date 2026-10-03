---
status: done
synopsis: "Exoskeletons, prostheses, hearing aids, neural implants, and robot controllers tuned from a person's preferences: who took part, what was measured, how the results were validated, and how algorithm tuning compares with letting people tune devices themselves."
---

# Wearable Robots, Health, and Assistive Technology {#sec-health}

In the design studies of @sec-hci, a person judges a picture on a screen. In
this chapter the thing being judged is felt through the body: the push of an
exoskeleton, the swing of a prosthetic knee, the sound of a hearing aid, the
light pattern of a retinal implant, the gait of a walking robot. Each
evaluation takes seconds to minutes, sessions are limited by fatigue and
safety, and the people who would benefit most are the hardest to recruit.
These are the settings @sec-optimizing-the-unknown used to motivate the book:
an expensive, noisy objective that only the person can report.

The chapter covers wearable robots, hearing aids, neural prostheses, and robot
control, from 2017 to September 2026, and counts other methods aimed at the
same problem (radial basis function surrogates, linear reward models, neural
rankers) alongside Gaussian process models, because several of the most cited
studies use them. The thread is measurement. Studies are small, and
validation is almost always internal: the same person's later choices agree
with the model. Where an objective outcome or a manual baseline was measured,
the picture is mixed: people tuning a hip exoskeleton themselves reached a
benefit of the same order as algorithm tuning, and preference tuning of hearing
aids improved sound quality but not speech clarity.

## Exoskeletons {#sec-health-exo}

An **exoskeleton** is a wearable robot that applies torque at a joint (hip,
knee, or ankle) to assist walking; a soft **exosuit** does the same through
textile straps and cables. Its controller is a curve of torque over the **gait
cycle**, one stride from heel strike to the next heel strike of the same foot,
shaped by a few numbers: peak torque, the timing of the peak, and when
assistance starts and stops. The right values differ from person to person.

### Before preferences: optimizing measured effort {#sec-health-hilo}

The field's starting point optimized **metabolic cost**, the energy the body
spends, estimated by **indirect calorimetry**: the wearer breathes through a
mask while oxygen and carbon dioxide are measured, and each setting must be
walked for minutes before the reading settles. This is called
**human-in-the-loop optimization**. In @zhang2017human, torque patterns
optimized by an evolution strategy (CMA-ES, @sec-app-five-neighbors) reduced
metabolic energy consumption by 24.2 ± 7.4% compared with no torque, for 11
participants with an ankle exoskeleton. @ding2018human used Bayesian
optimization to tune two timing parameters of hip assistance from a soft
exosuit for 8 participants; the optimum was found in 21.4 ± 1.0 minutes on
average, a convergence time computed afterward from a 40-minute optimization,
and reduced metabolic cost by 17.4 ± 3.2% compared with walking without the
device (mean ± standard error). @slade2022personalizing replaced the mask with
a model that estimates the metabolic effect from wearable sensors: it reached
the same parameters as metabolic optimization, within 5%, after 32 minutes of
walking instead of 128 (9 participants), and after one hour of walking
outdoors the optimized assistance reduced metabolic cost on a treadmill at
1.5 m/s by 23 ± 8% compared with normal shoes (10 participants).

Metabolic measurement is slow and noisy and leaves out comfort and the feeling
of safety [@schafer2026user]; a 2024 perspective [@slade2024human] and a
2023 review [@ingraham2023leveraging] discuss subjective objectives and user
preference as part of the design space. Adaptation is the other complication:
in
@poggensee2021adaptation, customized assistance after training reduced
metabolic rate by 39% compared with the exoskeleton turned off, but training
contributed about half of that benefit and customization about a quarter, and
becoming an expert user took about 109 minutes of assisted walking. A person
whose response is still changing is a moving target for any optimizer
(@sec-neuro-motor).

### The Caltech line: comparisons, suggestions, and ordinal labels {#sec-health-caltech}

A group at Caltech, with Maegan Tucker as first author of its main papers,
built the main Gaussian process line of preference-based gait tuning on
**Atalante**, a self-balancing lower-body exoskeleton that walks for its user.
**CoSpar** combines pairwise preferences with **coactive feedback**,
improvements the user volunteers ("a slightly longer step"), and picks the
next gait by **Thompson sampling**, drawing possible utility functions from the
posterior and letting them compete (@sec-thompson) [@tucker2020preference].
With three able-bodied participants and 20 gait trials, each participant's
blind ranking of three gaits agreed with the posterior; because "users have
difficulty remembering more than two trials at once", each trial was compared
only with the one before.

CoSpar became infeasible at 5 or more parameters, so **LineCoSpar** restricts
each iteration to a random line through the gait with the highest posterior
mean, the idea of
@sec-line-search [@tucker2020human]. Six able-bodied newcomers optimized 6 gait
parameters in 30 trials and agreed with the model on 4 validation preferences
each at **75%, 100%, 100%, 25%, 100%, and 100%**; utility functions differed
between people. **ROIAL** adds **ordinal labels**, a rating of each gait from
"very bad" to "good", and learns the landscape only inside a region of
interest that excludes uncomfortable gaits [@li2021roial]. With 3
participants, 4 parameters, and 40 trials, most ordinal predictions were within
one level of the reported label, and the search explored less than 2% of the
action space: comfortable for the wearer, and a landscape that covers only a
small corner.

**The only study with complete paraplegia.** Maegan Tucker's doctoral thesis
reports the only preference-based exoskeleton study we found with participants
who have complete motor paraplegia [@tucker2023enabling]: two people
classified ASIA A (no motor or sensory function below the injury), one with
more than 300 hours of experience with Atalante and one with fewer than 30.
Three gait parameters certified for use in the European Union were tuned, on
day one with ROIAL for 15 iterations and on day two with LineCoSpar for 15 and
25, with a 5-minute break every 20 minutes to prevent pressure sores. Each
person could complete only one evaluation trial, and both rated the
posterior's best gait as "good". The thesis also reports 8 able-bodied
participants.

**Several objectives.** @astudillo2025preferential proposed an
asymptotically consistent method for multi-objective optimization when every objective is
observed only through preferences, dueling scalarized Thompson sampling,
evaluated on simulated exoskeleton and driving tasks only (arXiv 2024, TMLR
2025). In a September 2026 preprint, MO-HILBO optimized metabolic cost and
ordinal comfort feedback together over 3 controller parameters of a hip-knee
exoskeleton, with 3 participants who each spent 3 to 4 hours; the surrogate
predicted **94% (17 of 18)** of the pairwise rankings among Pareto-front
controllers correctly [@janwani2026multi].

### Other groups {#sec-health-exo-others}

**A neural ranker.** @lee2023user used a neural network ranker, pretrained on
earlier preference data, to score ankle-assistance settings (4 parameters)
proposed by an evolutionary algorithm, while the wearer made forced choices
between pairs. It converged on the wearer's preferred parameters with an
accuracy of 88% on average when compared with randomly generated parameters,
and the preferred setting stabilized after 43 ± 7 queries. It is not a
Gaussian process method. We could read only the abstract: it does not name the
evolutionary algorithm (a later preprint calls it CMA-ES
[@liu2026personalized]), and the participant count of 14 given in a
third-party summary could not be checked against the paper.

**A radial basis function surrogate.** WANDER, an omnidirectional walking-aid
robot, tuned 2 **admittance parameters** (how heavy and how damped the robot
feels when pushed) with GLISp, the active preference learning algorithm of
Bemporad and Piga, which fits a radial basis function surrogate instead of a
Gaussian process [@fortuna2024personalizable; @bemporad2021global]. Against
two settings from the literature, with 12 healthy adults and up to 15
iterations, linear energy fell by 17.61% and 13.93% (both significant);
angular energy fell by 13.26% against the first (significant) and rose by 3.57%
against the second (not significant); and jerk fell by 1.12% (not significant)
and 5.95% (significant). The preferred virtual mass correlated with body
weight ($r = 0.59$, $p = 0.042$).

**A linear reward.** @ramella2025rapid modeled preference over hip torque
curves as a linear reward over 6 features, with a posterior sampled by the
Metropolis-Hastings algorithm (@sec-mcmc). Eight healthy participants made 12
pairwise comparisons, walking 20 seconds with each curve. Against perturbed
versions (±2 N·m, ±7% of the gait cycle) most kept their original preference,
though for four of them the check failed when the rise time was perturbed.
Preferred torque was synchronized with the wearer's movement and had lower
negative device power: the device absorbed less energy from the wearer.

**Perception thresholds in the acquisition function.** @arens2025preference
optimized lifting and lowering assistance from a soft back exosuit with 15
healthy participants, in 3 blocks of 15 iterations (each with 3 hidden
validation checks) over 2 parameters. The upper confidence bound (@sec-ucb)
accounted for the **just noticeable difference** (JND), the smallest change a
person can reliably detect (@sec-psychophysics), measured separately at 13.1 N
for lifting and 9.8 N for lowering. The **intraclass correlation coefficient**
(ICC), how consistently a person lands on the same setting (1 is perfect), was
0.80 for lowering and 0.91 for lifting. Preferred assistance rose by 15% and
29% from block 1 to block 3, a trend that was not significant ($p = 0.181$).

**Pairwise preferences with metabolic follow-up.** In a 2026 preprint,
@liu2026personalized optimized 6 parameters of hip assistance from pairwise
comparisons with five healthy adults who knew the device. After 20 iterations
they chose the optimized torque profile over a random one in 90.7 ± 1.3% of
validation comparisons, and in follow-up measurements on 2 to 3 of them
metabolic rate fell by 14.5% to 15.4% across three speeds relative to
assistance switched off, and muscle activation by 6.7% to 31.5% relative to
walking without the exoskeleton (@sec-health-manual discusses the baselines).

**Context, inverse models, and perception.** In a retrospective analysis of
data from 9 healthy adults, a 2026 preprint found that preference landscapes
in neighboring operating conditions tended to be more similar, and in
simulation sharing data across contexts helped when that continuity held and
caused negative transfer when it was weak [@baek2026context];
@park2026simultaneous infer an individual reward while learning a forward
model of the person, in simulation only. @maberry2026just found that a walking
person notices a change in ankle
exoskeleton stiffness only at about 42%, and in the rate of ankle angle
control at about 49%, more than three times the values measured standing
still. A 16-person preprint on dynamic Bayesian optimization handles a
non-stationary objective, but that objective is a measured hip angle, not a
preference
[@kim2025validation].

### The wearable studies at a glance {#sec-health-table}

@tbl-health-wearables lists the wearable studies. Most "validation" in the
table is internal: the same person's later choices are compared with the
model's predictions.

::: {.table #tbl-health-wearables title="Preference-based and self-tuning studies of exoskeletons and prostheses, 2020 to 2026. Sources are cited in the text."}
| Study | Venue | Participants | Parameters | Comparisons or trials | Validation | Objective outcome | Compared with manual or expert settings |
|---|---|---|---|---|---|---|---|
| CoSpar | ICRA 2020 | 3 able-bodied | 1 (also 2) | 20 gait trials | blind ranking of 3 gaits matched the posterior | not measured | no |
| LineCoSpar | IROS 2020 | 6 able-bodied | 6 | 30 plus 6 validation | 25% to 100% per person | preference correlated with gait dynamism | no |
| ROIAL | ICRA 2021 | 3 able-bodied | 4 | 40 (30 training, 10 validation) | most ordinal predictions within one level | not measured | no |
| Tucker thesis | Caltech 2023 | 2 with complete paraplegia, plus 8 able-bodied | 3 | 15; next day 15 and 25 | one evaluation each, both rated "good" | not measured | no |
| Lee et al. (neural ranker) | Science Robotics 2023 | not checked against the paper | 4 | stable after 43 ± 7 queries | 88% on random parameters | not covered | no |
| WANDER (GLISp) | ICRA 2024 | 12 healthy adults | 2 | up to 15 iterations | not reported | linear energy 13.93% to 17.61% lower; other measures partly significant | settings from the literature; partly better |
| Ramella et al. (linear reward) | ICRA 2025 | 8 healthy | 6 features | 12 pairwise comparisons | most kept their preference against perturbed versions | lower negative device power | no |
| Arens et al. | Science Advances 2025 | 15 healthy | 2 | 3 blocks of 15 | ICC 0.80 and 0.91 | preferred assistance rose 15% and 29% (not significant) | no |
| Liu et al. | preprint 2026 | 5 healthy adults | 6 | 20 | 90.7% | metabolic rate 14.5% to 15.4% lower than with assistance off (2 to 3 participants) | no |
| Taddei et al. | preprint 2026 | 2 amputees, 2 able-bodied | 4 | discrete 10.3 ± 2.5 (3 participants); continuous 35.0 ± 6.2 (1 participant) | 93% and 67% | gait asymmetry improved in one amputee | no |
| MO-HILBO | preprint 2026 | 3 | 3 | 3 to 4 hours each | 94% of Pareto rankings predicted | metabolic cost was one objective | no |
| Ingraham et al. (self-tuning) | Science Robotics 2022 | 24 able-bodied | 2 | about 105 s per tuning | repeatability SD 1.7 N·m and 1.5% of gait cycle | naive users drifted toward more torque | is itself self-tuning |
| Schäfer et al. (self-tuning) | npj Biomedical Innovations 2026 | 11 healthy adults | 4 | 30.5 settings in 10.9 min | not reported | metabolic cost 16.6% lower than with zero torque | is itself self-tuning |
| Díaz et al. (self-exploration) | JNER 2026 | 3 transfemoral amputees | 2 | 8 to 14 settings per trial | 2 of 3 inconsistent across trials | less intact-side muscle activity | is itself self-exploration |
:::

## Prostheses {#sec-health-prostheses}

A powered prosthesis adds a further constraint: the wearer depends on the
device for every step, so a poor setting is risky, and few people can take
part. @taddei2026bayesian, a 2026 preprint, tuned 4 parameters of an active
prosthesis with 2 **transfemoral** amputees (amputation above the knee) and 2
able-bodied people walking with an adapter. A discrete version, which combined
the expected utility of the best option (EUBO, @sec-eubo) with LineCoSpar and
was run with one amputee and the two able-bodied participants, converged in
10.3 ± 2.5 iterations (8.5 ± 3.6 minutes) with a validation identification
rate of 93%; for the amputee, swing-phase asymmetry improved from -13.4% to
-7.6% and stance-phase asymmetry from 6.2% to 3.9% compared with the person's
own prosthesis. A continuous version, run with the other amputee alone, needed
35.0 ± 6.2 iterations and reached a validation rate of only 67%. The authors
attribute growing variability after the 15th iteration of one trial to
fatigue, and write that "it remains unclear which biomechanical variables drive
these preferences".

@diaz2026user let 3 transfemoral amputees explore the timing and magnitude of a
powered knee-ankle prosthesis themselves, blind to the values, on a
touchscreen grid (8 to 14 settings per trial). Preferred settings went with
lower muscle activity on the intact side and less variable muscle synergies,
but for 2 of the 3 people the preferred setting was not consistent across
trials of the same day. Two neighboring studies are not PBO: in
@alili2023novel, amputees choose a preferred knee kinematics curve and a
reinforcement learning tuner fits 12 control parameters to it, and a
preliminary look at gait found no clear difference between the preferred
control and control tuned to a normative gait; @sun2024individual validated individualized prosthesis control
only on a gait data set, with no online test with amputees.

## Against manual tuning {#sec-health-manual}

Every method in this chapter is meant to save effort compared with tuning by
hand, which makes manual tuning the comparison that matters most. It is also
the one the field has run least.

**How reliable are people at tuning themselves?** In @ingraham2022role, twelve
naive and twelve knowledgeable able-bodied participants adjusted the magnitude
and timing of ankle exoskeleton torque themselves, blind to the values.
Preferred settings ranged from 7.9 to 19.4 N·m and from 54.1% to 59.2% of the
gait cycle; the repeatability, the standard deviation of a person's settings
across repeated tunings, was 1.7 N·m and 1.5% of the gait cycle; each tuning
took about 105 seconds; naive users' preferences shifted toward more torque
over the experiment; and knowledgeable users preferred more torque than naive
ones. Exposure and knowledge change the preferred setting itself, not only the
noise around it.

**The most informative direct comparison.** In @schafer2026user, 11 healthy
adults tuned 4 timing parameters of a bilateral hip exoskeleton themselves
with a thumbstick while walking on a treadmill. They took **10.9 ± 0.9
minutes** on average (at most 16.2), testing a median of 30.5 settings (range
16 to 111), almost always changing one parameter at a time (97.5% of
adjustments). Walking with the preferred assistance cost **16.6 ± 1.1%** less
metabolic energy than with the exoskeleton at zero torque. Shifting the timing
by up to ±8% of the stride time did not change the reduction significantly,
and preferred timings differed between people by up to 22.5% of the stride.
A sense-of-agency score (0 to 1) fell from 0.80 ± 0.07 at zero torque to
0.49 ± 0.07 with the self-tuned assistance ($p = 0.002$): even assistance the
wearers chose reduced their sense of controlling their own movement. The
authors note that self-tuning took a quarter of the time @ding2018human needed
for 2 parameters, and half of Ding et al.'s convergence time extracted after
the fact.

**Reading the two 2026 results side by side.** @liu2026personalized tuned 6
parameters by pairwise Bayesian optimization in sessions of 20.6 ± 4.6 minutes
including validation, with reductions of 14.5% to 15.4%. Its abstract calls
the baseline "unassisted walking", but its results section shows that it is the
exoskeleton worn with assistance off, the same kind of baseline as Schäfer et
al.'s zero torque; relative to walking without the exoskeleton, the
reductions at the three speeds were 8.6%, 9.3%, and 13.0%. Devices,
parameters, and speeds differ, so the studies cannot be compared directly;
what they show is that for a problem with few parameters, tuning by hand
already reaches the order of benefit that algorithm tuning reports.
@fig-health-tuning places these results, and the metabolic optimizations of
@sec-health-hilo, on one chart.

```{figure}
//| figure: health-tuning
//| label: fig-health-tuning
//| fig-cap: "Exoskeleton and prosthesis tuning studies side by side. *Time and benefit* plots the minutes of walking spent tuning against the metabolic reduction, with error bars as reported, colored by what judged each setting: the wearer by hand, the wearer's pairwise choices, measured metabolic cost, or a wearable-sensor model. Baselines (zero torque or assistance off, no device, normal shoes), devices, and speeds differ, so the chart is one frame, not a head-to-head comparison; Liu et al.'s time includes validation. *Who took part* plots participants against evaluations per person, with hollow markers for studies that included clinical participants. Choose a study to read its protocol and result. Studies with a count or time we could not verify are left out of the view that needs it."
```

Things to look for:

- In *Time and benefit*, the self-tuning result sits at the left, inside the
  range of the optimizer-tuned results, not below it.
- The one result clearly above the others, Slade et al.'s 23%, is measured
  against normal shoes and comes after an hour of walking.
- In *Who took part*, almost every marker sits below 15 participants, and the
  hollow markers for clinical participants sit at 2 to 4.

**What it means.** The ±8% tolerance around the optimum, the inconsistency of
self-chosen optima in Díaz et al., and the drop in validation rate when
candidates are close (67% for the continuous version of Taddei et al., with
one participant) all suggest that the preference landscape is flat near the
optimum, so a fast coarse search fits these problems better than a precise one
(inference). The perception data agree: if a walking person cannot feel a
stiffness change smaller than about 42% [@maberry2026just], comparisons
between closer candidates carry almost no information, which gives a
principled stopping rule for that parameter (inference; the rules in use,
listed in @sec-rec-stopping, ignore perception).

No study has compared preferential Bayesian optimization with expert manual
tuning on the same patients with preregistered endpoints. The experiment that
would settle it is simple: the same participants and time budget, self-tuning
against pairwise preference optimization in balanced order, with an objective
endpoint and a delayed check of whether the person still prefers the result.
Until it is run, a team tuning a device with 4 to 6 parameters should treat
self-tuning as the baseline to beat (inference). @sec-sci-evidence carries this into the
book's recommendations, and @sec-cs-exoskeleton lets you take both roles,
tuning a simulated walker by hand and letting an optimizer tune it.

## Hearing aids and audio {#sec-health-audio}

Hearing aids are one of the first areas in which preference learning reached a
commercial product. A hearing aid amplifies frequency bands differently, and
the **gain** in each band and the **compression ratio**, how strongly loud
sounds are turned down relative to soft ones, are normally set by an
audiologist from a **prescription**, a formula such as NAL-NL2 that maps a
person's audiogram to settings. In 2015 @nielsen2015perception already used a
Gaussian process with active learning for perception-based personalization,
and @baltzell2018efficient, at the research center of the maker Starkey, found
with preference-based Bayesian optimization that listeners differed in the
compression ratios they preferred and tended to prefer the learned ratio over
linear gain and over the NAL-NL2 prescription.

**SoundSense Learn.** @jensen2019perceptual, whose authors are from the maker
Widex and from FORCE Technology, evaluated Widex SoundSense Learn, which models
the user's utility over the gain in 3 frequency bands with a Gaussian process
prior and learns from paired comparisons in which the user marks a degree of
preference on a slider (20 comparisons in the study). Twenty hearing-impaired
participants each tuned 12 sound scenes, four for each of three attributes
(basic audio quality, listening comfort, speech clarity), with the hearing
aids on an acoustic manikin, and recordings of the tuned settings were rated
double-blind against two settings from Widex's own prescription. Basic
audio quality improved generally; listening comfort improved only in traffic
noise, not in babble from several talkers; and **speech clarity did not
improve significantly.** The size of the gain adjustments varied widely
between people and did not predict who would benefit.
@balling2021collaboration, also from the manufacturer, describe a Bayesian
optimization mechanism that runs continuously on user input and summarize laboratory and field results.

**Presets and dueling bandits.** For over-the-counter hearing aids,
@vyas2022personalizing discretized a 24-dimensional configuration space into
15 presets and treated the choice as a **dueling bandit** problem, a bandit
that learns from pairwise wins and losses (@sec-dueling-bandits). Thirty-five
people with mild-to-moderate hearing loss each compared every pair of presets
four times; the algorithms were then compared in simulations that drew each
answer from a person's recorded preferences, where the authors' algorithm
found the best preset in a median of 25 comparisons, half as many as the best
baseline.

**Other work.** @ignatenko2025preference proposed an information-theoretic
query design, validated in simulation and on one "real-life example" whose
domain the abstract does not name (we could not read the full text), and @tasnim2024review review machine learning for hearing-aid
personalization. The closest cochlear-implant study, @gilbert2022cochlear, had
14 MED-EL users rate music under fixed settings, not adaptive optimization.
The HearClip trial of Bayesian preference elicitation for hearing aids was
registered in Amsterdam in May 2008, and its record, last updated on 1
September 2025, still lists recruitment as "pending" with no results
[@umc2025personalization]. A preprint review of preference learning in audio
cites one music-generation study with 60% annotator agreement, against 75% for
text summarization; these are that study's figures, not a pooled estimate
[@broukhim2025preference].

The hearing-aid evidence is the strongest in this chapter by design, with a
double-blind comparison against prescribed settings, and it shows a recurring
pattern: preference tuning improves what the person judges directly (overall
sound quality) and does not reliably improve the outcome that motivated the
device, understanding speech, which was measured only as rated clarity and not
with an intelligibility test (inference). The evaluations of the commercial
system were run or co-authored by the manufacturer, and the compression-ratio
study came from a manufacturer's research center; none of the controlled
evaluations we found is independent of a manufacturer (inference, from the
authors' affiliations). An independent evaluation with a speech
intelligibility test is the study this area needs (inference).

## Neuroprostheses and neurostimulation {#sec-health-neuro}

A **neuroprosthesis** stimulates the nervous system to replace a lost
function. In a **retinal implant**, electrodes on the retina produce spots of
light called phosphenes, and an **encoder** turns a camera image into
stimulation patterns whose quality depends on parameters that differ between
patients. In **spinal cord stimulation**, an electrode array over the spinal
cord can restore some standing or grasping after injury, depending on which
electrodes are active and at what frequency and amplitude.

**Visual prostheses.** @fauvel2022human tuned the encoder of simulated
prosthetic vision for sighted viewers with PBO and report significant, robust
improvements in perceived image quality that transferred to other tasks; the
bioRxiv preprint reports 24 participants and runs of 60 duels, and we could not
read the journal version to confirm them. @granley2023human tuned 13 patient-specific
parameters of a deep encoder with duels on simulated patients only, reaching
high-quality percepts after about 20 duels and close to an ideal encoder after
about 75, robust to simulated response noise and to thresholds misspecified by
up to 300%, where it converged to slightly worse encodings.

The follow-up of @schoinas2025evaluating, at the IEEE Engineering in Medicine
and Biology Society conference (EMBC) in 2025, ran 60 duels per condition with
17 sighted undergraduates. Only about **50%** of the human choices agreed with
the simulated agent's, and the final loss in the main condition was 0.27 for
humans against 0.07 in the earlier simulations; yet 16 of 17 participants
preferred the optimized result in the main condition, all 17 with
misspecified thresholds, and 14 in the out-of-distribution condition. All
three studies used sighted people or simulated patients.

**Spinal cord stimulation.** CorrDuel, a correlated dueling bandit, chose
electrode configurations by having clinicians compare a patient's standing
under two configurations, in a live trial with two patients and 414
comparisons [@sui2017correlational]. Clinical knowledge first cut the
16-channel array's roughly $4.3 \times 10^7$ configurations to the order of
$10^3$ to $10^4$, and the comparison with the doctors' own choices was
qualitative: their selections were a subset of what the algorithm found.
**StageOpt**, stage-wise safe Bayesian optimization, explores only settings it
is confident are safe before optimizing within them (@sec-constrained-bo), with
preference feedback in its clinical part [@sui2018stagewise]. Over 10 weeks it
ran 564 grasping therapy trials with one patient with tetraplegia; it never
sampled an unsafe pattern, and after around 400 iterations a Gaussian process
fit to its chosen patterns exceeded the physicians' best choice. The authors
note that it assumes the patient's response does not change over time.
@zhao2021optimization built a Bayesian preference model for the 5 participants
of the E-STAND trial, with accuracy averaged over participants of 71.5% in
cross-validation and 65.6% in prospective validation, both significantly above
chance. A later paper with overlapping authors adds meta-learning; its
abstract reports validation on synthetic data and mentions no patients, and we
could not read the full text [@farooqi2025augmented].

Two lessons stand out (inference). The gap between simulated and real people
is large even for a simple task, so results on simulated patients cannot stand
in for clinical evidence (@sec-sw-simulated-users). And the clinical
successes, StageOpt and CorrDuel, are measured against physicians' choices
only qualitatively or on a single patient.

## Robot control and human-robot interaction {#sec-health-robots}

Robots are tuned by people, too: an engineer adjusts gains and constraints
until the robot moves "well", a judgment easier to make than to write down as
a cost function. @sec-app-robotics covers Bayesian optimization in robotics
with measured objectives; this section covers the cases where the objective is
a person's judgment.

**Legged robots.** @tucker2021preference used LineCoSpar to tune the
constraints of a gait optimization based on hybrid zero dynamics (a standard
method for stable biped gaits) on the AMBER-3M robot, producing robust walking
in 30 iterations with rigid feet and 50 with springs the gait model ignored.
@csomayshanklin2022controller tuned controller gains on AMBER, where none of
the initially sampled gains could walk and after 50 iterations 3 sets were
rated very good, and on Cassie, with a domain expert and with a naive user.
@cosner2022safety tuned a safety filter on a Unitree A1 quadruped from
pairwise preferences and safety labels, in 30 iterations in simulation and 7
on hardware indoors and 3 outdoors. In these studies the judge was one
operator at a time.
The same group released POLAR, a MATLAB toolbox [@tucker2022polar], a
preprint.

**Controller calibration, two schools.** The GLISp family uses radial basis
function surrogates: GLISp [@bemporad2021global], GLISp-r with a convergence
guarantee [@previtali2023glisp], a robotic sealing planner whose abstract
reports better deposition quality within 20 trials than programming by
demonstration and manual tuning [@roveda2021pairwise] (we could read only the
abstract), and collaborative spray painting validated with 15 PhD or master's
students [@cella2026adaptive]. The Gaussian process school includes a
simulated proportional-integral controller tuned from real users' pairwise
feedback, which came closer to the response users wanted than multi-objective
alternatives [@coutinho2024human], with a follow-up in Control Engineering
Practice [@coutinho2026efficient]; coactive multi-objective optimization for
personalized plasma medicine [@shao2024coactive]; CrashPBO, which lets the user
report a crash as worse than every successful experiment, on a backflipping
quadcopter (three lab members as judges, 15 trials each), a Furuta pendulum,
and a unicycle robot [@menn2026preferential], the earliest preference tuning
of a quadrotor we found; and a reaching demonstration on a Franka Panda arm at
the 2024 International Conference on Ubiquitous Robots, which reports no user
study [@feith2024integrating].

**An expert's cost function against the expert's choices.** A 2025 preprint by
@witte2025capture gives the most important negative finding in this area. A
robot arm pushed a block to a target, and 4 time-scale parameters of a cascade
of controllers were tuned; three preferential methods were run on hardware,
where an expert chose between two pushes (2 random and 13 algorithm-driven
comparisons, 30 experiments per
trial). PBO tuned the system to the expert's satisfaction, but **the cost
function the expert designed to describe their own priorities was not in full
agreement with their choices**, even after its weights were refitted to the
pushes the expert had rated highest. The expert rarely disagreed with it among
the 10% of experiments with the lowest cost; beyond that, agreements and
disagreements on its goal-reaching term were roughly balanced. Across
iterations the expert's choices followed the total cost relatively
consistently, but no repeatable pattern of shifting priorities could be
modeled. The Gaussian
process utility learned from the duels was more consistent with the expert's
decisions than the expert's own cost function. The experiments refer to one
expert; the paper gives no count.

**Active preference-based reward learning.** Robot learning offers better
controlled user studies, mostly with linear rewards. @sadigh2017active
introduced active preference-based reward learning from a person's choices
between pairs of trajectories (@sec-adj-pbrl), and @byk2018batch extended it
to batch queries. DemPref combined demonstrations with preferences and was
rated more successful than inverse reinforcement learning (learning a reward
from demonstrations alone) by 15 users of a Fetch robot [@palan2019reward],
with more user studies in its journal version [@biyik2022learning]; "Asking
Easy Questions" accounts for how hard a question is for the person
[@byk2019asking]. With a Gaussian process reward, 10 users taught a Fetch
robot a variant of minigolf and rated the learned behavior, on a 9-point
scale, **6.9 ± 0.6** (mean ± standard error) for the active Gaussian process,
**3.4 ± 0.7** for the active linear model, and 5.1 ± 0.7 for a Gaussian
process with random queries ($p < 0.05$) [@byk2020active].
@myers2021multimodal learned rewards from rankings, the APReL library collects
these algorithms [@byk2021aprel], and @kwon2022physically plated Japanese food
from comparisons of rendered dishes, with simulated users and 4 real users.

**What it means.** Control papers present PBO as an alternative to
trial-and-error tuning, but their baselines are a scoring function the
engineers could not write (as in @sec-sci-built) or an expert-written cost
function, not measured expert tuning time and quality, so the claimed saving in
tuning effort is not yet quantified (inference), with the sealing study as a
possible exception. De Witte et al. also suggest that synthetic decision
makers defined by a hidden cost function may overstate how well a method
works on real people (inference).

## Settled, contested, missing {#sec-health-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** Preference-based tuning of exoskeletons works in the sense its
studies test: people's later choices agree with the learned model at about 65%
to 100% when averaged over a study's participants, within sessions of 12 to 50
comparisons. Self-tuning by the wearer is repeatable within a person but varies
widely between people [@ingraham2022role], and a thumb-controlled self-tuning
of 4 hip parameters took about 11 minutes and reduced metabolic cost by 16.6%
against zero torque [@schafer2026user]. Hearing-aid preference tuning improved
overall sound quality without improving speech clarity [@jensen2019perceptual].
Simulated users can differ sharply from real ones [@schoinas2025evaluating].

**Contested.** Whether algorithm tuning beats tuning by hand for problems with
4 to 6 parameters; the two 2026 results are of the same order, on different
devices. How much the preferred setting is the person's and how much it is a
product of adaptation and of the optimizer's own dynamics: preferences rose
across blocks in Arens et al., naive users drifted toward more torque in
Ingraham et al., and customization contributed only a quarter of the benefit in
Poggensee and Collins. Whether an expert's judgment can be written down at all,
on the evidence of one expert [@witte2025capture].

**Missing.** A preregistered comparison of preferential optimization with
expert or self-tuning on the same patients and time budget. Clinical
populations beyond 2 people with paraplegia, a few with spinal cord injury, and
a few amputees. Retinal implant studies with blind users. Any study of deep
brain stimulation, cochlear implants, or functional electrical stimulation in
rehabilitation, and any new peer-reviewed user study of preferential Bayesian
optimization for hearing aids or cochlear implants from 2025 to September
2026. An independent evaluation of hearing-aid preference tuning. Measured
expert tuning time and quality as the baseline for controller calibration; one
abstract reports a comparison with manual tuning, and we could not read its
measurements [@roveda2021pairwise]. Preference tuning of quadrotors and
quadrupeds judged by more than a few people: we found one quadruped study with
a single user [@cosner2022safety] and one quadcopter study with three lab
members [@menn2026preferential] (inference from the searches reported above).
:::

## Exercises {#sec-health-exercises}

::: {.exercise #exr-health-baselines}
Liu et al. report metabolic reductions of 14.5% to 15.4% against wearing the
exoskeleton with assistance off, and 8.6% to 13.0% against walking without it.
Suppose wearing the exoskeleton without assistance costs 5% more energy than
walking without it. Starting from a 15% reduction against assistance off, what
reduction would you expect against no exoskeleton? What does this say about
comparing Schäfer et al.'s 16.6% (against zero torque) with Ding et al.'s
17.4% (against no device), and with Liu et al.'s 14.5% to 15.4%?

::: {.solution}
Let $E_0$ be the energy of walking without the device. With the device worn
but not assisting, $E_{\text{off}} = 1.05\,E_0$. A 15% reduction against that
gives $E = 0.85 \times 1.05\,E_0 \approx 0.89\,E_0$, an 11% reduction
against no device: carrying a device that does not assist costs energy, so the
same assistance looks better against "assistance off" than against "no
device".

Schäfer et al.'s 16.6% and Liu et al.'s 14.5% to 15.4% are both measured
against the device worn but not assisting, so they can be set side by side
(devices, parameters, and speeds still differ). Ding et al.'s 17.4% is
measured against no device; restated against the device worn without
assistance it would be larger, so the optimizer-tuned exosuit's advantage over
the self-tuned hip exoskeleton is somewhat bigger than the raw numbers
suggest, by an amount that depends on each device's penalty when worn without
assistance. The 5% is an assumption for the exercise.
:::
:::

::: {.exercise #exr-health-validation}
LineCoSpar's six participants agreed with the model on 4 validation preferences
each, at 75%, 100%, 100%, 25%, 100%, and 100%. If a participant answered at
random, what is the probability of agreeing on all 4? Of agreeing on at least 3?
What does this say about how much a single participant's 100% tells you?

::: {.solution}
At random each answer agrees with probability 1/2, so all four agree with
probability $(1/2)^4 = 1/16 \approx 6\%$, and at least three agree with
probability $(\binom{4}{3} + \binom{4}{4})/16 = 5/16 \approx 31\%$. A single
participant's 100% on 4 checks is weak evidence on its own, roughly a one in
sixteen event under pure guessing. Across six participants, four perfect
scores are much less likely by chance, so the study as a whole is informative,
but per-person validation with 4 checks cannot distinguish a person whose
preferences the model captured from one who was lucky. This is one reason
validation with a few internal checks is the weakest part of the evidence in
@tbl-health-wearables.
:::
:::

::: {.exercise #exr-health-jnd}
Arens et al. measured just-noticeable differences of 13.1 N for lifting and
9.8 N for lowering assistance. Suppose the search range for lifting assistance
is 0 to 80 N. If comparisons between settings closer than one JND carry no
information, roughly how many distinguishable levels does the lifting range
have? What does that suggest about how many pairwise comparisons the
optimization of this one parameter can usefully use?

::: {.solution}
Roughly $80 / 13.1 \approx 6$ distinguishable levels. With about six levels, a
search can locate the preferred level by something like bisection in about
$\log_2 6 \approx 3$ informative comparisons once noise is small, or a few
times that with realistic noise; comparisons between candidates less than one
JND apart add little. The perception threshold, not the optimizer, sets the
useful resolution, which is why Arens et al. built the JND into the
acquisition function and why the chapter suggests it as a stopping rule. The
range of 0 to 80 N is an assumption for the exercise.
:::
:::

## Further reading {#further-reading .unnumbered}

- @schafer2026user is the clearest test of whether an algorithm is needed at
  all, and its discussion compares durations with metabolic optimization.
- @tucker2020human and @li2021roial show how the query was adapted to walking:
  one-dimensional lines for scale, ordinal labels and a region of interest for
  comfort.
- @jensen2019perceptual is the best-controlled evaluation in the chapter, a
  double-blind comparison against the manufacturer's own prescribed settings,
  with both positive and null outcomes.
- @schoinas2025evaluating measures directly how far simulated users are from
  real ones.
- @witte2025capture shows, with one expert, why a person's choices may be
  easier to learn than their own description of what they want.
- @ingraham2022role is the reference point for how stable a wearer's own
  preference is.
