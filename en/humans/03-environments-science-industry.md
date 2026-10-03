---
status: done
synopsis: "Buildings, vehicles, expert-guided science, and industry: where preferential optimization has tools and methods but few real people, how the evidence compares across every application domain, and which common claims about it hold up."
---

# Built Environments, Science, and Industry {#sec-science-industry}

@sec-hci and @sec-health followed preferential optimization where people were
in the loop by necessity: a design only its user can judge, a device only its
wearer can feel. This chapter covers the remaining applications, where the case
for asking a person is less obvious and the evidence takes a different shape.
In buildings and vehicles, comfort and driving style are personal, but the
studies mostly use simulated occupants and drivers. In science and
engineering, preferences inject an expert's judgment into an optimization that
otherwise has a measured objective, and the experts are usually one to four
people. In industry, the software is ready, and the public record consists of
benchmarks and motivating statements. The chapter then compares the human
evidence across every application domain of this part, and ends by checking
claims about these applications that circulate in secondary accounts.

## Buildings and vehicles {#sec-sci-built}

### Thermal comfort and daylight {#sec-sci-buildings}

Comfort differs between people, and occupants answer comparisons about it
easily, which makes buildings a natural target. The evidence is almost
entirely simulated. A Purdue University group learned individual visual
satisfaction with office daylighting from comparative preferences between 2017
and 2020 [@xiong2017personalized; @xiong2018inferring; @xiong2020efficient];
we could not obtain the sample sizes or the number of queries.
@awalgaonkar2019personalized, a preprint, treated an occupant's answer ("I
would like it warmer", "cooler", or "I am satisfied") as the sign of the
derivative of their utility, used a Gaussian process constrained to a single
peak, and validated it with synthetic occupants.

POP-BO, an optimistic method of preferential Bayesian optimization (PBO) with a
bound on cumulative regret [@xu2024principledb], includes a thermal comfort
task in simulation, with preferences generated from the **predicted mean
vote** (PMV), a standard formula for a group's average comfort rating, over 2
or 4 variables such as air temperature and air speed. qEUBO
(EUBO for queries of $q$ options, @sec-eubo) found a slightly better final
solution, but its cumulative regret was almost twice that of POP-BO, and the
authors argue that online control of heating, ventilation, and air
conditioning (HVAC) should care about cumulative regret, the total discomfort
over the whole run (@sec-regret). Three more recent preprints are simulations
too: contextual PBO raised occupant utility by up to 23% over two simulated
months on BOPTEST, a building simulation platform [@wang2025personalized]; a
real-time preference-optimizing controller was evaluated on simulated thermal
comfort [@wang2025human]; and a consensus Bayesian optimization robust to
social influence uses thermal comfort as one application
[@adachi2025bayesian].

**Every building study of PBO from 2024 to 2026 used simulated occupants.**
That is a gap, not a negative result: the next useful study is a trial of
preference-based HVAC control with real occupants, reporting cumulative
discomfort as well as the final setting (inference).

### Vehicles and controller calibration {#sec-sci-vehicles}

**People misjudge their own driving.** In @basu2017you, not an optimization
study, users preferred a driving style clearly more defensive than how they
drove, preferred the style they believed was their own over their actual one,
and preferred different styles in different scenarios, so a system that
learns from how people drive and one that learns from what they say they
prefer reach different places.

**One study with real drivers.** @ran2023online personalized lane-centering trajectories online from
preferences, modeling the uncertainty in each answer; in a fixed-base driving
simulator, 29 drivers (15 inexperienced, 14 experienced) converged after
**11.1 ± 4.6 queries** on average, almost all in fewer than 20, and the
estimated utility agreed with their own rating scores.

**Controller calibration in simulation.** A **model predictive controller**
(MPC) chooses each action by optimizing over a short prediction of the future,
with weights an engineer usually calibrates by trial and error.
@zhu2021preference calibrated MPC controllers with GLISp, the radial basis
function method of @sec-health-robots, on a simulated reactor and a simulated
lane-keeping car (50 experiments each); for the car, the authors could not
write a scoring function usable for automatic calibration even after much trial
and error. In the later studies the decision maker was synthetic or an author.
C-GLISp adds feasibility labels for unknown constraints, with a synthetic
decision maker in its benchmarks and the first author as calibrator in its
driving case [@zhu2022c]; @theiner2025exploiting guided PBO with a virtual
decision maker built from real driving data, again with a synthetic main
decision maker, and converged faster than standard PBO, which needed about 70
iterations, with a multi-fidelity follow-up [@theiner2026efficient]; and two
preprints tuned a vehicle suspension with a synthetic user
[@cercola2025regularized] and the slip control of an electric race car, with
simulation results only [@vries2024human]. The in-car
interface studies optimized from ratings are in @sec-hci-ulm.

## Science and engineering with expert preferences {#sec-sci-experts}

In the sciences, preferences mostly bring an expert's judgment into a Bayesian
optimization whose objective would otherwise be a measured scalar, and in each
study the real experts number one to four. @sec-app-science covers Bayesian
optimization in experimental science with measured objectives, and
@sec-cs-chemistry works through a chemical reaction end to end.

### Materials and physical sciences {#sec-sci-materials}

**Expertise visibly changes the outcome.** The projective preferential
Bayesian optimization of @mikkola2020projective asks the user to pick the best
point along a one-dimensional projection with a slider
(@sec-gallery-projective). It was used to find the position and orientation
of a camphor molecule on a copper surface, Cu(111), with each candidate's
energy checked by density functional theory, a quantum-mechanical calculation
of the energy of a molecular configuration. Every user answered 24 queries.
The structures preferred by two materials-science experts relaxed to energy
minima of about -1.007 to -1.030 eV, while those of a non-expert relaxed to
shallower local minima of about -0.762 to -0.771 eV. The authors describe this
as a clear split by expertise, and the method could also tell a person's
choices from those of a random robot.

**Votes on spectra.** BOARS turns an operator's up or down votes on measured
spectra into an objective by Thurstone-Mosteller scaling (@sec-thurstone),
with the human leading early and the algorithm later, demonstrated on a live
atomic force microscope [@biswas2024dynamic]. A 2026 preprint, px-BO, fits a
Bradley-Terry model (@sec-bradley-terry) to the votes and then lets a
surrogate vote in the human's place, with periodic human checks
[@biswas2026human].

**Explanations, and the risk of over-trust.** In CoExBO (@sec-hci-trust), an
expert chooses between two candidates shown with explanations
[@adachi2024looping]. On battery electrolyte design with 4 participants, the
explanations improved the experts' accuracy; the authors note that the gains
assume the expert's comparative knowledge is accurate, and that experts tended
to expect the surrogate to understand the problem like an oracle, a form of
over-trust.

**Simulated experts.** BOAP models an expert's pairwise preferences over
"abstract properties" that cannot be measured, for lithium-ion electrode
manufacturing, but the preferences were simulated from published data sets
[@v2024enhanced]. In the multi-objective optimization of scanning probe
microscopy by @liu2025pareto, the human steers by objective weights and
reference points, not comparisons. @deneault2025preferential used PBO to
optimize subjective qualities of 3-D prints, judged by a person rather than
measured by sensors.

### Chemistry and drug discovery {#sec-sci-chemistry}

Chemistry holds the only large data set of expert preferences in this chapter.
**MolSkill** collected more than 5,000 pairwise comparisons of molecules from
35 chemists at Novartis (in wet-lab, computational, and analytical roles) over
several months, choosing each round by active learning
[@choung2023extracting]. Agreement was moderate. In preliminary rounds, the
inter-rater agreement on 200 pairs, measured by **Fleiss' kappa**, a
chance-corrected agreement among several raters where 0 is chance and 1 is
perfect, was **0.40 and 0.32**; the intra-rater agreement on 20 repeated
pairs, measured by Cohen's kappa, was 0.60 and 0.59, and for some chemists as
low as 0.16 to 0.27. The area under the ROC curve for classifying pairs, the
probability that the model ranks a random preferred molecule above a random
non-preferred one, rose from about 0.6 with 1,000 pairs to above 0.74 with
5,000, without reaching a plateau. The learned score captured drug-likeness
better than QED, the quantitative estimate of drug-likeness chemists commonly
use. MolSkill is a neural network that learns to rank, not Bayesian
optimization.

The other chemistry studies use simulated experts, a single expert, or none.
@sundin2022human learned a chemist's scoring function actively, improving
significantly in fewer than 200 queries in two cases with simulated ground
truth, with one medicinal chemist in a demonstration who judged one molecule at
a time, not pairs. **CheapVS**, a workshop paper, guided screening of 100,000
molecules with chemists' pairwise preferences over trade-offs among
properties: screening 6% of the library recovered 16 of the 37 known drugs for
one target (EGFR) and 37 of the 58 for another (DRD2), but with a synthetic
utility function, preliminary human data (expert rankings for EGFR about 80%
accurate), and a number of experts we could not find [@dang2025preferential].
@kristiadi2024useful, a workshop paper, simulated experts' pairwise labels with
a scoring function; the "preferential Bayesian optimization" for protein design
of @hawkinshooker2023preferential, also a workshop paper, takes its comparisons
from measured fitness; and @haltia2026elicitation, a preprint, choose between
an expensive evaluation and a query to an expert by their costs.

### Dividing the work between people and algorithms {#sec-sci-division}

@kanarik2023human compared people and Bayesian optimization in a virtual
process game for designing a semiconductor plasma etching process. Engineers
did well early, the algorithm was far more cost-efficient close to tight
tolerances, and a "human first-computer last" strategy halved the cost of
reaching the target compared with engineers alone. The objective was a
measured scalar, so how far this carries over to preference-driven design is
uncertain. And @weichert2025less, a preprint, document an industrial case in
which adding expert knowledge made the optimization fail.

**What it means.** Agreement of about 0.3 to 0.4 between chemists and about 0.6
within one says that experts' pairwise intuition can be learned but is only
moderately shared, which favors models per expert or mixture models over pooled
data (inference; @sec-obs-extensions). Mikkola et al.'s split between experts
and a non-expert, and CoExBO's assumption that expert knowledge is accurate,
point to an untested failure mode: the expert who is wrong or overconfident.
CoExBO's no-harm guarantee is the main design response so far (inference),
and a method that relies on expert preferences could be tested with at least
one deliberately misinformed expert (inference).

## Industry {#sec-sci-industry}

**The main industrial line: preference exploration at Meta.** BOPE (preference
exploration for Bayesian optimization with multiple outcomes) alternates
between running experiments and asking a decision maker to compare predicted
outcome vectors [@lin2022preferenceb]. The paper is motivated by A/B testing
workflows, but its experiments use a vehicle safety problem (5 inputs, 3
outputs), a car cab design problem (7 inputs, 9 outputs), and the DTLZ2 and
OSY test problems, with a noisy synthetic utility standing in for a human
decision maker. qEUBO extended the expected utility of the best option to
queries of $q$ options [@astudillo2023qeubob]. Both are in BoTorch
[@botorch2026bope], and between January and June 2026 Ax versions 1.2 to 1.3
added preference optimization configuration, BOPE utility tracking, qEUBO, and
pairwise Gaussian processes [@ax2026changelog] (@sec-sw-ax). LILO, which turns natural-language feedback into preference
signals with a language model, was evaluated with a language model standing in
for the decision maker [@kobalczyk2026lilo] (@sec-llms). Other industry-facing applications include
banner ads (@sec-hci-koyama), RankTuner for electronic design automation
tools [@xu2026ranktuner], contextual dueling bandits for recommendation
[@sankagiri2026recycling], multi-criteria decision support
[@huber2025bayesian], and hearing-aid presets [@vyas2022personalizing].

**Public evidence of deployment.** Two web searches turned up no report from
Meta or any other company that quantifies the effect of BOPE in production.
In the peer-reviewed literature, hearing aids are the one area with both a
described commercial mechanism and user evaluations, and those evaluations
involve the manufacturer (inference, from the authors' affiliations;
@sec-health-audio). So the toolchain supports production A/B testing, but the
public evidence of such use consists of benchmarks and motivating statements. Dueling bandits that compare search rankers by interleaving results
were outside the scope of our searches.

## The evidence base, domain by domain {#sec-sci-evidence}

@tbl-sci-evidence compares the human evidence across every application domain
of @sec-part-humans. It counts the studies cited in this part, not the result
of a systematic search, and "typical sample" means the number of real people
who gave judgments. Its last row collects the domains in which our searches,
as of September 2026, turned up no preferential optimization study.

::: {.table #tbl-sci-evidence title="The human evidence for preferential optimization, by application domain. Counts are of the studies cited in this part of the book."}
| Domain | Studies cited | Typical sample (range) | Real or simulated people | Main limitation |
|---|---|---|---|---|
| Visual, interface, and generative design (preference feedback) | about 24 | mostly 10 to 40 (3 to 60, excluding crowdsourcing) | mostly real | mostly weak comparisons and novices; no test of drift within a session |
| Rating or performance-based human-in-the-loop optimization | about 19 | 12 to 40 (8 to 200) | real | not preference feedback; against strong comparisons final quality often no different |
| Lower-limb exoskeletons (algorithm tuning) | 13 (10 with real people), plus 2 self-tuning baselines | 2 to 15, median about 5 | real, mostly young able-bodied adults | only 2 participants with paraplegia; internal validation; almost no comparison with manual tuning |
| Prostheses | 1 PBO (plus 3 related) | 2 to 3 amputees | real | tiny samples; preferences inconsistent across trials |
| Hearing aids and audio | 5 (plus 3 related) | 20 to 35 | both | no improvement in speech clarity; evaluations mostly involve the manufacturer |
| Visual prostheses | 3 | 17 sighted people; 1 study with unknown sample | 1 simulation only, 2 with sighted people | no blind users; about 50% agreement between people and the simulated agent |
| Spinal cord stimulation | 3 | 1 to 5 patients | real | comparison with physicians qualitative; assumes responses do not change |
| Thermal comfort and daylight in buildings | 8 | Purdue samples not obtained; no real people otherwise | all simulated since 2024 | no trial with real occupants |
| Driving style and controller calibration | 7 (plus 1 non-optimization study) | 1 study with 29 drivers; otherwise authors or synthetic decision makers | mostly simulated | no study on real roads; preferences vary with the scenario |
| Legged robots and controller tuning | about 14 (with method and tool papers) | 1 expert or a few lab members | few real people | an expert's own cost function disagrees with their choices; effort savings not quantified |
| Active preference-based reward learning | 8 | about 10 users | both | mostly linear rewards; Gaussian process versions costly in high dimension |
| Materials and physical sciences | 7 | 1 to 4 experts | more than half simulated | experts and non-experts diverge; over-trust in the model |
| Chemistry and drug discovery | 4 | 35 chemists (MolSkill, not BO); otherwise 1 or simulated | mostly simulated | only moderate agreement between chemists (kappa 0.32 to 0.40) |
| Protein design | 1 | none | no human preferences | "preference" comes from measured fitness |
| Industrial platforms and A/B testing | 6 | no public deployment data | synthetic utilities or language-model simulation | no public report quantifying production deployment |
| Domains with no study found | 0 | not applicable | not applicable | agriculture; food and flavor with sensory panels; motion sickness and motion-simulator cueing; cochlear implants; deep brain stimulation; functional electrical stimulation in rehabilitation; comfort of prosthetic sockets, seats, and clothing; drones before 2026; driving style on real roads; daylight after 2020; protein design from human experts' preferences; production A/B testing; lasers, welding, and machining (scalar Bayesian optimization only) |
:::

@fig-sci-evidence draws the same table on a common scale.

```{figure}
//| figure: sci-evidence
//| label: fig-sci-evidence
//| fig-cap: "The human evidence by application domain. Each row is a domain of @tbl-sci-evidence; the thin line spans the number of real people per study, the thick segment the typical range, and a dot marks a single value, median, or study (hollow: not preferential optimization, such as MolSkill). Color says where the judgments came from. Choose a domain to read its study count and main limitation. Values are from the table and approximate where it says so; on the log scale a 35-person study looks close to a 200-person one, so read the numbers in the panel."
```

**The pattern across domains.** Outside interactive design, each study with
human participants had 1 to 35 people who gave judgments, with a median below
10, and used 12 to 50 comparisons, with one clinical case reaching 564 trials
(@sec-health-neuro). Validation agreement was about 65% to 100% and nearly
always internal, and comparisons with manual or expert tuning were mostly
qualitative. The most rigorously designed studies, the double-blind comparison
of @jensen2019perceptual and the repeated blocks of @ingraham2022role and
@arens2025preference, are also the ones that report limits: partial benefits,
preferences that drift with exposure, a preferred setting that differs from
the physiologically best one (inference). Applications also drove the methods:
JND-aware acquisition, ordinal and crash labels, multi-objective preference
models, and contextual PBO all came out of applications rather than benchmarks
(inference).

**Three recommendations follow from this evidence** (inference), and none of
them depends on whether the surrogate is a Gaussian process.

1. For personalization problems with few parameters (about 4 to 6), where each
   evaluation is felt with the body and the optimum may be flat, start with
   self-tuning or a coarse grid search as the baseline, and bring in
   preferential Bayesian optimization only when that baseline falls short: a
   thumb-controlled self-tuning reached a 16.6% metabolic reduction in about 11
   minutes, with a ±8% tolerance around the preferred timing
   (@sec-health-manual).
2. New studies should include manual tuning, random or coarse grid search, and
   a linear utility model as comparisons, and measure an objective outcome;
   comparing only with another preferential optimizer or a slider cannot show
   whether preferential optimization is needed at all (@sec-hci-baselines).
3. Method papers should test their simulated users on at least a few real
   people: people agreed with the simulated agent only about 50% of the time in
   the retinal implant study, and an expert's own cost function disagreed with
   the expert's choices over most of the range in the robot study
   (@sec-health-neuro, @sec-health-robots).

@sec-recommendations turns these into practical guidance, and @sec-rec-simpler
asks when a simpler method should win.

## Common claims, checked {#sec-sci-claims}

Secondary accounts of these applications repeat several claims that the
primary sources do not support as stated. @tbl-sci-claims checks them.

::: {.table #tbl-sci-claims title="Claims about the applications of preferential optimization, checked against the primary sources."}
| Claim | Verdict | What the sources show |
|---|---|---|
| The exoskeleton gaits of Tucker et al. are among the benchmarks of the qEUBO paper. | wrong | qEUBO's benchmarks are Ackley, Alpine1, Hartmann, car cab design, Sushi, and an animation task [@astudillo2023qeubob]. Simulated exoskeleton personalization appears in the preferential multi-objective paper of Astudillo et al. [@astudillo2025preferential]. |
| ROIAL characterizes the whole preference landscape. | partly right | ROIAL learns only within a region of interest that excludes uncomfortable gaits, exploring less than 2% of the action space [@li2021roial]. |
| Abdelrahman and Miller (2022) optimize the indoor thermal environment from preference feedback. | wrong | The paper uses a graph neural network to decide when to collect occupants' thermal preference feedback for personal comfort models; it is not preferential BO and does not optimize the environment [@abdelrahman2022targeting]. |
| Hiranaka et al. (IROS 2023) learn primitive skills from human evaluative feedback. | partly right | The paper applies reinforcement learning from human feedback over parameterized primitive skills; it does not learn the skills themselves and is not preferential BO [@hiranaka2023primitive]. |
| BOARS appeared in npj Computational Materials in 2023, CoExBO is a 2023 paper, and BOAP is used for automated science. | partly right | BOARS was published in 2024 [@biswas2024dynamic]; CoExBO appeared at AISTATS 2024 [@adachi2024looping]; BOAP's expert preferences were simulated from published data [@v2024enhanced]. |
| Dueling scalarized Thompson sampling is the first provably convergent method for preferential multi-objective optimization, applied to autonomous driving and exoskeletons. | partly right | Published in TMLR in 2025 (arXiv 2024); both applications are simulated; "first" overlooks the earlier choice-function work on multi-objective optimization of Benavoli et al. [@astudillo2025preferential; @benavoli2021choice]. |
| PBO works well in experiential domains and where people have expertise, and poorly where preferences must be constructed. | an untested synthesis | Partial support comes from the split between experts and a non-expert in Mikkola et al. [@mikkola2020projective] and from the non-convergence in the field deployment of Ou et al. [@ou2022human]. @sec-what-comparisons-measure takes up the question. |
:::

## Settled, contested, missing {#sec-sci-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** Every building study of preferential Bayesian optimization since
2024 used simulated occupants. The only large set of expert pairwise
judgments, more than 5,000 pairs from 35 chemists, shows moderate agreement
between experts (kappa 0.32 to 0.40) and somewhat higher agreement within one
expert (about 0.6) [@choung2023extracting]. The software for preference
exploration in multi-output experiments exists in BoTorch and Ax. Across
domains, studies with real people are small, typically fewer than 10 people
outside interactive design, and validated internally.

**Contested.** Whether expert preferences improve scientific optimization
beyond what the measured objective achieves: the positive cases involve one to
four experts or simulated ones, and one industrial case found that expert
knowledge hurt. How to divide the work between people and algorithms: "human
first-computer last" halved costs in one scalar-objective game
[@kanarik2023human], but whether the same holds when the objective is a
preference is open. Whether experts who are wrong or overconfident can be
detected or safely ignored; CoExBO's no-harm guarantee is a partial answer.

**Missing.** A trial of preference-based HVAC control with real occupants. A
driving-style study on real roads. A public, quantified report of preferential
optimization in production A/B testing. Any study in agriculture, food and
flavor with sensory panels, or protein design from human preferences. A
comparison of preferential optimization with measured expert tuning time and
quality in controller calibration. Tests of simulated decision makers against
real people in method papers.
:::

## Exercises {#sec-sci-exercises}

::: {.exercise #exr-sci-kappa}
In each of the two preliminary rounds of MolSkill, the chemists' intra-rater
Cohen's kappa on repeated pairs was about 0.6 on average. For a binary choice
where each option is chosen half the time, chance agreement is 0.5, and kappa
is $(p_o - 0.5) / (1 - 0.5)$, where $p_o$ is the observed agreement. With that
chance level, what fraction of repeated pairs does a chemist with a kappa of
0.6 answer the same way? If you modeled that chemist with the probit
likelihood of @sec-comparisons and assumed every repeated pair had the same
utility difference $\Delta$ (in units of the noise), what $\Delta$ would
produce that repeat agreement?

::: {.solution}
From $0.6 = (p_o - 0.5)/0.5$, $p_o = 0.8$: such a chemist gives the same answer
to 80% of repeated pairs. Under a probit model each answer favors the better
molecule with probability $q = \Phi(\Delta)$, and two independent answers
agree with probability $q^2 + (1 - q)^2$. Setting $q^2 + (1-q)^2 = 0.8$ gives
$2q^2 - 2q + 0.2 = 0$, so $q = (1 + \sqrt{0.6})/2 \approx 0.887$, and
$\Delta = \Phi^{-1}(0.887) \approx 1.21$. In the model's units, a typical
repeated pair sits about 1.2 noise standard deviations apart, which is far
from a deterministic judge. (The paper's table lists raw repeat agreements of 78.9% to 100%, for
example 89.5% beside a kappa of 0.58, which a chance level of 0.5 does not
reproduce; with $p_o = 0.895$ the same steps give $\Delta \approx 1.59$,
still a noisy judge.) Real pairs
differ in $\Delta$, so this is an average picture, but it shows why a
noise-free simulated expert overstates what a real one provides.
:::
:::

::: {.exercise #exr-sci-regret}
POP-BO and qEUBO were compared on a simulated thermal comfort task; qEUBO
found a slightly better final setting, while its cumulative regret was almost
twice POP-BO's. Explain in one paragraph why a building operator might prefer
POP-BO anyway, and describe a deployment in which qEUBO would be the better
choice.

::: {.solution}
In online HVAC control every query is a room condition that real occupants
live in, so each poor query is experienced discomfort. Cumulative regret adds
up that discomfort over the whole run, while the final setting matters only
after learning ends. An operator who tunes comfort while people work in the
building should weigh the cumulative cost, which favors POP-BO. qEUBO would be
the better choice when exploration is cheap or happens offline, for example a
commissioning phase in an empty test room or with occupants who volunteer for
a short calibration session, after which the found setting runs for months.
The distinction is the one between simple and cumulative regret in
@sec-regret.
:::
:::

## Further reading {#further-reading .unnumbered}

- @choung2023extracting is the largest data set of expert pairwise judgments
  in this chapter and the clearest measurement of how much experts agree.
- @mikkola2020projective shows on a real physics problem how much the person in
  the loop matters, with experts and a non-expert reaching different minima.
- @lin2022preferenceb sets out preference exploration for multi-output
  experiments, the method behind the industrial tooling.
- @xu2024principledb includes the thermal comfort simulation and the argument
  for cumulative regret in online control.
- @kanarik2023human is the best evidence on dividing work between people and
  algorithms, though with a measured objective.
