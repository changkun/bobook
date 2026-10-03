---
status: done
synopsis: "What interactive design studies from 2017 to 2026 found when the person in the loop was real: six research lines, a consistent trade-off between agency and performance, unstable feedback in real sessions, evidence on feedback forms, population priors, expertise, explanations, and what happens against strong baselines."
---

# Interactive Design and Human-Computer Interaction {#sec-hci}

@sec-part-preferences built preferential Bayesian optimization (PBO) around a
simulated person: a fixed utility, constant noise, and unlimited patience, and
@sec-pbo-people warned that real people are none of these things. This
chapter collects what human-computer interaction (HCI), computer graphics, and
interactive design learned between 2017 and September 2026 by putting real
people in the loop: designers tuning photographs, artists simplifying 3D
meshes, crowd workers moving sliders, drivers rating dashboards.

The chapter reads "preferential" broadly: besides pairwise duels, it includes
sliders, galleries, and rankings, as the authors of one main system do when
they use the term PBO "in a broader sense" [@koyama2020sequential], and it
includes optimization driven by ratings or by measured performance, because
most of what is known about agency and population priors comes from those
studies. Each study is labeled with its **feedback type**, *preference*
(choices, rankings, comparisons), *rating* (a score on a scale), or
*performance* (a measurement of what the person did, such as task time),
because a result about one type does not automatically transfer to another.
$N$ is the number of participants.

Three findings carry the chapter, and each bears on the book's argument that
the hard part of PBO is now measurement, including what asking does to the
person (@sec-syn-bottleneck). When the optimizer leads, outcomes improve but
the person's sense of agency drops, and most of it returns when the person can
steer. Feedback in real sessions is unstable in ways the standard noise model
does not describe. And against strong comparisons, the usual result is the
same outcome at lower cost, not a better outcome, from samples that in pure
preference studies are mostly 6 to 60 people.

## Research lines {#sec-hci-lines}

Two papers mark the start: @brochu2007active used gallery-style active
preference learning to design materials, and @gonzalez2017preferentialb gave
the formal definition based on duels in 2017. Since then, HCI systems have
mostly changed the *form of the query* and rarely the acquisition function,
the rule that picks the next question (@sec-acq-definition), which is usually
expected improvement (@sec-ei) or the upper confidence bound (@sec-ucb)
rewritten to fit the query (inference, from the system descriptions below).
After 2024 the emphasis moved toward priors borrowed from other users and
toward language and generative models. The subsections follow the research
groups; @tbl-hci-studies collects the user studies.

### Koyama and collaborators: the query interface as the variable {#sec-hci-koyama}

Yuki Koyama and colleagues (University of Tokyo, AIST, and OMRON SINIC X)
treat the query interface as the main thing to study: from 2017 to 2025 each
paper added one channel for feedback, from a single slider to a gallery,
several sliders, implicit slider logs, references, constraints, and natural
language (inference, from the designs below).

**Sequential line search** turns each query into a single-slider microtask
along a line through the design space [@koyama2017sequentialb] (@sec-line-search;
@sec-cs-photo runs it on a photograph). Its crowdsourced pipeline cost about
\$5.25 and 68 minutes per result (15 iterations of 7 microtasks at \$0.05
each), and against the automatic enhancement of Photoshop and Lightroom the
optimized versions of three photos received **32, 26, and 29 crowd votes, and
each alternative 0 to 3** [@koyama2020computational]. The same book chapter
argues that absolute ratings generally work poorly because a rater must know the design
space, whereas a pairwise comparison "is easy to answer even for non-experts".

**Sequential Gallery** replaces the slider with a zoomable grid (5 by 5 for
photos) from which the person picks the best item [@koyama2020sequential]
(@sec-gallery-projective). Its plane search beat line search in simulation,
and in a preliminary study with $N = 6$, participants were satisfied after
**5.36 iterations** on average and rated the grid as a source of inspiration
at 6.50 on a 7-point scale. The authors list as limitations that the method
suits only visual designs, handles neither discrete parameters nor prior
knowledge, performs poorly above about 20 dimensions, and assumes that the
user's perceptual function "does not change over time".

Systems from the same period search the **latent space** of a generative model, the compact
input vector that a trained network decodes into an image, a melody, or a
font. @chiu2020human built slider subspaces in it without Bayesian
optimization; @chong2021interactive let people blend image candidates with
several sliders and guide the search by painting; and Zhou, Koyama, Goto, and
Igarashi had people pick the best of several generated melodies, first in a
pilot study [@zhou2020generative], then with a slider that sets the
exploration weight of the upper confidence bound [@zhou2021interactive].

**BO as Assistant** removes the explicit query loop: it extracts preference
pairs from ordinary slider edits and offers suggestions asynchronously, so the
designer can "take the full initiative"; it had no formal user study, and its
authors state that usability, usefulness, and agency were not evaluated
[@koyama2022bo]. **Photographer-in-the-loop** lighting is PBO
with a gallery of four photographs, which in the authors' demonstrations found
pleasing lighting in about 10 iterations [@yamamoto2022photographic].
**FontCraft** explores font styles from text, image, and font references, with
a history whose choices can be withdrawn [@tatsukawa2025fontcraft]. And
**constrained PBO** asks banner-ad designers only for pairwise preferences
while a model of the click-through rate (the fraction of viewers who click an
ad) acts as a constraint; its acquisition function, EUBOC, is the expected
utility of the best option (EUBO, @sec-eubo) with the constraint added
[@iwai2025constrainedb]. The line then turned to language: designers
intervening in a system-led optimization in natural language
[@niwa2025cooperative], a language model turning vague preferences into a
combinatorial problem [@kuroki2026lappi], and, in a 2026 preprint, a
feasibility-aware latent space of car-wheel designs [@owaki2026feasibility].

### Cambridge, Aalto, and NYCU: multi-objective optimization with designers {#sec-hci-cambridge-aalto}

The second line, around Per Ola Kristensson, John Dudley, Antti Oulasvirta,
Liwei Chan, George Mo, and Yi-Chi Liao (Cambridge, Aalto, and National Yang
Ming Chiao Tung University), mostly uses **multi-objective Bayesian
optimization** with performance objectives, not preference feedback.
Multi-objective optimization (@sec-multi-objective) measures several objectives
at once, such as speed and accuracy, and returns the **Pareto front**, the
designs that cannot be improved on one objective without getting worse on
another.

@dudley2019crowdsourcing optimized a map search interface for task time, with
200 crowd workers in its first experiment. @chan2022investigating compared optimizer-led with
designer-led design of a virtual reality touch technique (4 parameters) in a
**between-subjects** experiment, in which each participant saw only one
condition, with 40 novices; its results anchor @sec-hci-agency.
@shen2022personalization let 12 users pick their final gesture-keyboard design
from their own Pareto front; @liao2023interaction had 8 designers work with the
optimizer and with their own strategy; @mo2024cooperative proposed
**cooperative** optimization, in which designers mark designs and regions to
guide the optimizer; and @liao2024practical pooled sessions with a "global
Gaussian process" (the surrogate of @sec-gp-regression) and started new users
from a "warm-start Gaussian process". Later papers color-graded 360-degree
images with two-level PBO [@yuan2025personalized], built the cost of each
prototype into the acquisition function [@langerak2026cost], and separated
expert opinion, empirical studies, and simulators as feedback sources in MUSE
[@junior2026integrating], accepted at ACM Transactions on Interactive
Intelligent Systems (TiiS). A review by Dudley, Oulasvirta, Chan, and
Kristensson appeared online in ACM Computing Surveys in August 2026; we could not
obtain its text [@dudley2026putting].

### ETH Zurich and Saarland: priors learned from other people {#sec-hci-eth-saarland}

Since 2024 a third line, around Yi-Chi Liao, Zhipeng Li, Christoph Gebhardt,
Christian Holz, and Anna Maria Feit (ETH Zurich and Saarland University), has
studied **population priors**: data from earlier users, so that a new user
needs to be asked as little as possible. @liao2024meta, a collaboration of
Aalto University and Meta, built a population
model from 14 participants and used it as the prior of a transfer acquisition
function for 11 new ones, and continual human-in-the-loop optimization
represents the whole population with a Bayesian neural network, one that
keeps a distribution over its weights [@liao2025continual]; both use
performance feedback. **Meta-PO** meta-learns over the Gaussian process models
of earlier users with the Sequential Gallery interface [@li2025efficient], and
@song2025preference inferred objective weights from users' manual edits of
mixed-reality layouts. Three CHI 2026 papers followed: **HOMI** trained a
neural acquisition function on simulated users before any real user arrived
[@liao2026efficient]; **APPO** hides an image generator's text prompt and asks
only for binary preferences while a language model rewrites the prompt
[@li2026preference]; and **AutoOptimization** turns spoken preferences into the
objectives of a layout optimizer [@li2026automating].

### Ulm: automotive interfaces rated by drivers and passengers {#sec-hci-ulm}

The group of Pascal Jansen, Mark Colley, and Enrico Rukzio at Ulm University
optimizes automotive interfaces on questionnaire **ratings** (trust, perceived
safety, mental demand, predictability), with some of the largest samples in
the field: the visualizations of an automated vehicle in OptiCarVis
($N = 117$, online, between-subjects) [@jansen2025opticarvis], external
vehicle displays ($N = 37$) [@colley2025improving], an air-taxi simulation with
and without a motion seat ($N = 40$) [@meinhardt2025fly], a proactive in-car
assistant in ProVoice ($N = 19$, within-subjects, meaning every participant
experienced every condition) [@susak2026provoice], blur in virtual-reality
driving in BlurDriving (IMWUT, September 2026) [@li2026blurdriving], and a
three-day study accepted at AutomotiveUI 2026 ($N = 74$) that compared
continued optimization with keeping the day-one design [@colley2026multi].

### LMU Munich: expertise and field deployment {#sec-hci-lmu}

Ou, Buschek, Mayer, and Butz built, with an industry partner, a PBO system for
simplifying 3D meshes (9 parameters), in which artists' ratings were converted
into pairwise relations; it ran for 3 months with two full-time technical
artists and then in a lab study with $N = 20$ [@ou2022human]. @ou2023impact
compared 60 users of different expertise on three tasks with an interface that
asked for a ranking of 4 candidates, offered a "don't know" area, and ended
when the user pressed a satisfaction button. These two studies supply much of
@sec-hci-unstable and @sec-hci-expertise.

### Guiding generative models {#sec-hci-generative}

Four more systems put preference queries inside a generative model, often a
diffusion model, the kind of generator behind most current text-to-image
systems. **BOgen** builds 3D shapes part by
part with PBO and a variational autoencoder (a network that decodes a short
latent vector into an output), and was compared with a baseline by 30
designers [@lee2026part]. **ROMBO** personalizes music generation, evaluated in
simulation and with 16 volunteers who scored each piece from 0 to 10
[@marcos2025random]. **GimmBO** runs PBO over the merging weights of 20 to 30
adapters, small add-on networks that each push a diffusion model toward a
style [@liu2026gimmbo]. **MultiBO** (ICML 2026) has the user pick, from 4
generated images, the one closest to the image they have in mind, and steers
edits in the model's attention space for up to 50 rounds
[@rajagopalan2026personalized]. It reports an image similarity of 0.9364
against 0.8365 for DiffusionDPO and won 70.82% of comparisons rated by 30
users, and its authors observe that the preference signal weakens as the
images approach the target.

### Other domains: touch, text entry, reading, architecture, engineering {#sec-hci-other}

@catkin2023preference optimized the perceived realism of rendered springs and
friction from pairwise comparisons, validated against a model tuned by
experts, and in @zhang2026vibrotactile 13 participants made 40 rounds of
comparisons with a 5-level confidence rating, for a held-out accuracy of 92.3%
(range 85% to 100%). AdaptiFont optimized a generative font space for reading
speed (performance, $N = 11$ in the main study) [@kadner2021adaptifont]; @tanaka2026human had 40
design students design a pavilion with 6 parameters, by optimization driven by
ratings from 0 to 100 or by sliders; and @nandy2025exploring adapted a
parametric mug to the attribute "comfortable" ($N = 31$ from scratch, $N = 25$
from the previous group's data) and improved the perceived match between
intent and design over a non-interactive model. @mccourt2019sampling, a
workshop paper, had to extend PBO to handle ties when coloring artwork, and
@peng2026efficient, a preprint, personalized generative user interfaces from
pairwise judgments. For text entry the only study we found is the gesture keyboard of
Shen et al. In visualization design, the applications of Bayesian optimization we
found tune particular displays (OptiCarVis, a case in MUSE), and none of them
is a PBO study.

### The studies at a glance {#sec-hci-table}

@tbl-hci-studies lists the main user studies by year. The questionnaires named
in it are explained in @sec-hci-agency. **Hypervolume** measures how much of
the objective space a Pareto front covers; larger is better.

::: {.table #tbl-hci-studies title="Main user studies of interactive Bayesian optimization in HCI and design, 2017 to 2026. Sources are cited in the text of this section."}
| System or study | Venue | Participants | Feedback | Comparison | Main result |
|---|---|---|---|---|---|
| Sequential line search | SIGGRAPH 2017 | crowd workers, about 105 microtasks per result | preference: slider | Photoshop and Lightroom auto-enhancement | crowd votes 32, 26, 29 against 0 to 3 |
| Dudley et al. | CHI 2019 | 200 crowd workers (experiment 1) | performance: task time | baseline interface | no difference in batch 1 ($p = 0.88$); from batch 2, median time from about 34 s to about 21 s |
| Sequential Gallery | SIGGRAPH 2020 | 6, plus simulation | preference: best of a 2D gallery | line search, in simulation | plane search better in simulation; users satisfied after 5.36 iterations |
| Zhou et al. | IUI 2021 | 12 novice composers | preference: best of several | automatic expected improvement | Creativity Support Index higher; 11 of 12 preferred manual balancing |
| Chan et al. | CHI 2022 | 40 novices, between-subjects | performance | designer-led design | better spatial error; lower agency, ownership, expressiveness |
| Ou et al. | Mensch und Computer 2022 | 2 artists (3-month field), 20 (lab) | rating converted to pairs | none (observational) | 415 of 549 field sequences stopped at iteration 1; satisfied 16/134 (field), 97/200 (lab) |
| Photographer-in-the-loop | UIST 2022 | 12 | preference: best of 4 | manual lighting workflow | experience rated better on all six items; satisfaction with the lighting not significantly different |
| Shen et al. | ISMAR 2022 | 12 | performance; final pick by preference | baseline keyboard | speed +14.4%, accuracy +13.8% |
| Ou et al. | IUI 2023 | 60, three tasks | preference: rank 4, with "don't know" | levels of expertise | novices reach expert-level quality; experts iterate more, less satisfied |
| Liao et al. | CHI 2024 | 14 for the model, 11 for evaluation | performance | standard Bayesian optimization; manual calibration | absolute pointing improved 22.92% and 21.35% |
| Mo et al. | ACM TiiS 2024 | 18, within-subjects | performance; designers can mark | designer-led; optimizer-led | sense-of-control medians 6.0, 5.5, 1.5; 12 of 18 preferred cooperative |
| FontCraft | CHI 2025 | 10 non-experts | preference: slider, references, retractable history | single-slider Bayesian optimization | closer to target fonts (no inferential statistics seen) |
| Constrained PBO | IJCAI 2025 | 11 professional ad designers | preference: pairs | no control system | 4.2 s per choice; positive attitudes; in a pre-study, designer preference not positively correlated with click-through |
| OptiCarVis | CHI 2025 | 117, online | rating | no visualization; expert design; user-customized | personalized designs rated better on perceived safety, predictability, trust |
| Meta-PO | UIST 2025 | 36 (three groups of 12) | preference: best of a gallery | no transfer | iterations to satisfaction from 9.54 to 5.86 (same theme) and 7.41 (other theme) |
| Niwa et al. | UIST 2025 | 18 (study 1), 12 (study 2) | performance, plus natural language | designer-led; optimizer-led; explicit constraints | optimizer-led highest hypervolume, lowest agency; language lowers workload, gives less agency than constraints |
| Song et al. | UIST 2025 | 12 | weights inferred from manual edits | manual; ParetoSelect | fewer elements moved; layouts ranked above ParetoSelect's ($p = .02$); no difference in hypervolume or overall quality |
| GimmBO | SIGGRAPH North America 2026 | 12 (computing or machine-learning background) | preference: top-$k$ ranking; slider; gallery | slider; Sequential Gallery | ranking better on similarity and success rate; 50.5 s against 34.7 s per step |
| APPO | CHI 2026 | 16 | preference: binary, prompt hidden | PromptCharm; DSPy; clarifying questions | satisfied in fewer than 4 iterations (baselines more than 6); less expressive |
| Cost-aware Bayesian optimization | CHI 2026 | 12 | performance plus a comfort rating | Bayesian optimization ignoring cost | same performance at about 67% of the cost; final quality no different ($p = .77$) |
| HOMI | CHI 2026 | 12 | performance | transfer acquisition; continual Bayesian optimization | better only at iterations 2 and 3; equal from iteration 6 |
| Tanaka et al. | CAADRIA 2026 | 40 design students | rating, 0 to 100 | sliders | 62.5% preferred the optimized results ($p = 0.025$) |
| Owaki et al. | preprint 2026 | 40 | preference exploration | raw 9-dimensional parameters | higher shape similarity and more feasible suggestions in a 5-dimensional latent space |
:::

## Agency versus performance {#sec-hci-agency}

The most consistent result of the period concerns who leads the search. When
the optimizer leads, outcome measures are better and mental demand is lower,
but the person's **agency** (the sense of being in control of the process),
**ownership** (the sense that the result is one's own), and
**expressiveness** (the sense of being able to put one's own ideas into the
result) fall clearly. When designers get ways to steer, most of the agency
comes back at a small cost in outcome quality. Almost all of this evidence
comes from multi-objective optimization with performance objectives.

Two questionnaires recur. The **Creativity Support Index** scores support for
creative work on a 0 to 100 scale, from factors such as exploration,
expressiveness, and enjoyment; the **NASA Task Load Index** (NASA-TLX)
measures workload from six subscales, among them mental demand and effort.
$p$ is the probability of a difference at least this large if the conditions
truly did not differ, and $t(38)$ is a $t$ statistic with 38 degrees of
freedom whose sign says which group scored lower.

**The only study large enough to isolate the effect.** In the
between-subjects experiment of @chan2022investigating, with 40 novice
designers and performance objectives, the optimizer-led group reported lower
agency ($t(38) = -5.523$, $p < 0.001$) and ownership ($t(38) = -3.892$,
$p < 0.001$), while satisfaction and confidence did not differ. The Creativity
Support Index was 75.3 for designer-led and 65.4 for optimizer-led design
($p = .011$), a difference that came mainly from expressiveness (44.9 against
23.0, $p = .001$). The NASA-TLX total did not differ (57.6 against 49.6,
$p = .758$), but the optimizer-led group reported lower mental demand (14.9
against 8.4, $p = .011$) and effort (24.9 against 13.7, $p = .040$).
Optimizer-led sessions were longer, 78.0 minutes against 51.8, their designs
had better spatial error ($t(38) = 2.237$, $p < 0.05$), and the group explored
a larger share of the design space. In the words of the abstract, designers
guided by an optimizer "reported lower mental effort but also felt less
creative and less in charge of the progress", and the authors conclude that
such optimization can support novice designers "in cases where agency is not
critical".

**Three levels of control in one study.** @mo2024cooperative compared
designer-led, cooperative, and optimizer-led design within subjects ($N = 18$).
For "I felt that I had control over searching different areas of the design
space", the median ratings were **6.0, 5.5, and 1.5** on a 7-point scale.
Twelve participants preferred the cooperative condition, 4 the designer-led
one, and 2 the optimizer-led one, with medians for willingness to use the
method again of 6.0, 5.0, and 3.0 in the same order. The cooperative
condition's relative hypervolume did not differ significantly from the
optimizer-led one, and it needed fewer formal evaluations but found fewer
Pareto-optimal designs.

**Smaller studies point the same way.** Among the 8 designers of
@liao2023interaction, optimizer-found designs showed "very similar
performance" to the designers' own at significantly lower workload, while
"designers may become detached from the design process when typical aspects of
their role are subsumed" by the optimizer. In the melody study of
@zhou2021interactive (preference feedback, 12 novice composers), letting users
set the balance between exploration and exploitation themselves raised the
Creativity Support Index total ($p < 0.01$) and satisfaction ($p = 0.041$),
and 11 of 12 preferred balancing by hand. In study 1 of @niwa2025cooperative
($N = 18$), the optimizer-led condition reached a higher relative hypervolume
than designer-led design ($p < 0.001$) and than cooperation in natural
language ($p = 0.001$) but had significantly lower agency than both (corrected
$p = 0.002$); in study 2 ($N = 12$), language and explicit constraints
performed the same ($p = 0.204$), language gave lower task load
($p = 0.034$), and constraints gave more agency ($p = 0.032$). Some
participants were unsure "how much their instructions were actually being
followed".

**Weaker evidence.** FontCraft's participants (preference feedback, $N = 10$)
reported more agency with multimodal references than with the baseline, and 7
of the 10 mentioned the history view [@tatsukawa2025fontcraft]. In
@colley2025improving, "I felt in control of the design process" averaged 5.81
and "the final design is mine" 5.30 on 7-point scales. @tanaka2026human found
"process disclosure" (parameter plots, the range of evaluations, the direction
of optimization) essential for agency and trust, even when participants'
understanding stayed superficial, and @song2025preference note that an
unconstrained optimizer can move elements the user had placed by hand.

You can feel the trade-off in @fig-hci-agency: run the same budget three
times, choosing every design yourself, letting the optimizer choose, and
letting it suggest while you decide, and rate after each run how much you
controlled where the search went.

```{figure}
//| figure: hci-agency
//| label: fig-hci-agency
//| fig-cap: "Who leads the search? A hidden score over a two-parameter design stands in for the measured outcomes of the studies in this section; each run has 10 tries. With *Who leads* set to *You*, click the map or set the sliders and press *Try this design*; with *Optimizer*, a Gaussian process with expected improvement chooses every try; with *Cooperative*, the optimizer suggests (orange ring) and you accept or try your own design, a simplification of Mo et al.'s cooperative condition. After each run, rate how much you controlled the search; the dark ticks are Mo et al.'s medians for a similar statement (6.0, 5.5, 1.5; N = 18). The landscape and scores are illustrative: in a two-parameter toy you may match the optimizer, and the point is the gap the studies report, a small difference in outcome against a large one in control."
```

Things to try:

- Do the *You* run first, without revealing the landscape. You will probably
  find the broad hill first; notice whether you spend the remaining tries
  polishing it.
- Switch to *Optimizer* and press *Run to the end*. Compare its best score
  with yours, then compare your two control ratings. Which gap is larger?
- Press *New landscape* and repeat. On the first landscape the optimizer's
  exploration finds the narrow, taller peak; on most of the next ones it does
  not reach that peak within its 10 tries, and your own search can do as well.

The map shows the whole design space at a glance. The design spaces in the
studies were larger (4 parameters in Chan et al., 6 in the pavilion study, 9 in
the mesh simplification of Ou et al.), and a person moving 9 sliders can no
longer see which regions they have not tried, which is where an optimizer's
systematic exploration should count for more (inference); @fig-pbo-dims shows
from the other side how much less forty comparisons accomplish as parameters
are added.

**What it means for preferential optimization.** Agency falls in the order
designer-led, cooperative, user-guided, optimizer-led, and the differences in
agency (sense-of-control medians of 6.0 against 1.5) are much larger than the
differences in outcome between cooperative and optimizer-led design. For
design tasks, this favors adding channels through which the person can steer
over making the optimizer more autonomous (inference). Plain PBO, in which the
person only answers the optimizer's questions, is the optimizer-led condition
by construction (inference). The effect was isolated with performance
objectives; among preference-feedback systems only Zhou et al. and FontCraft
measured anything close to agency, with 12 and 10 participants (inference). A
team building a preference tool should therefore measure agency in its own
study, with an item such as Mo et al.'s, rather than assume the effect
transfers (inference).

## Unstable feedback and non-convergence {#sec-hci-unstable}

A PBO model assumes that each answer is a noisy reading of one fixed utility,
with independent, identically distributed (i.i.d.) noise (@sec-pbo). The
longest observation of real use shows how far practice can be from that. In
the deployment of @ou2022human with two professional 3D artists, **415 of 549
evaluation sequences stopped at the first iteration**, without any
optimization being requested. The remaining 134 sequences ran 4.1 iterations
on average (standard deviation 4.2, range 1 to 23), and only 16 of them
(11.9%) produced a satisfactory result. In the lab study, 97 of 200 sequences
(48.5%) ended satisfied. @fig-hci-field-funnel draws every sequence.

```{figure}
//| figure: hci-field-funnel
//| label: fig-hci-field-funnel
//| fig-cap: "Every evaluation sequence of the mesh-simplification study of Ou et al. (2022), one square per sequence. The 3-month field deployment with two professional 3D artists had 549 sequences, of which 415 stopped at the first iteration (gray), 134 asked for optimization, and 16 of those ended satisfied (green). In the lab study with 20 participants, 97 of 200 sequences ended satisfied; the lab's other 103 sequences are not split by where they stopped. *Count against* sets the denominator of the field percentage. All counts are as reported by the authors [@ou2022human]."
```

Things to try:

- With *Count against* set to *All sequences*, compare the two percentages:
  2.9% in the field, 48.5% in the lab.
- Switch to *Sequences that asked for optimization*. The 415 gray squares drop
  out of the count and the field figure rises to 11.9%. Both numbers are
  correct; @exr-hci-denominator asks which one describes the tool in practice.

The authors write that participants, if not rating at random, "at least behave
highly unstable and inconsistent in the rating process". Asked about specific
inconsistent ratings, the experts pointed to anchoring on grids they had seen
before, the availability and representativeness heuristics, loss aversion, and
diminishing returns, the judgment biases of @sec-jdm. The authors concluded
that the assumptions of a stable and complete preference and of i.i.d. noise
do not hold: "human judgment is a fragile function to optimize for". Reading
this as direct evidence that preferences are constructed during the session
is an interpretation: the paper documents instability and non-convergence, for
which construction is one explanation (inference).

**Scattered but consistent signs elsewhere.** A Sequential Gallery participant
selected designs "based on criteria that I didn't have at the beginning"
[@koyama2020sequential]; AdaptiFont finds the best font at the time of use
rather than a stable optimum, since the font that maximizes reading speed may
depend on the text, fatigue, and display [@kadner2021adaptifont]; and
@dudley2019crowdsourcing had to cope with noise from differences between users
and tasks and from learning effects.

**Real people against simulated ones.** In personalizing a retinal implant
with 17 sighted participants, only about 50% of choices agreed with the
simulated agent's
[@schoinas2025evaluating] (@sec-health-neuro). In the preprint of
@peng2026efficient, 20 participants with varying interface-design experience
judged the same 600 pairs of generated interfaces and agreed with each other
at only **0.25** on a chance-corrected agreement statistic where 0 means chance
and 1 perfect agreement (Krippendorff's $\alpha$; Cohen's $\kappa$ is the same
to two decimals). The vibrotactile study notes that comparisons between nearby
candidates are "dominated by response noise" and that its model assumes
stationarity, with no account of adaptation, fatigue, or drift
[@zhang2026vibrotactile]. Meta-PO, Song et al., and LAPPI list static
preferences as a limitation without measuring change [@li2025efficient;
@song2025preference; @kuroki2026lappi].

::: {.keyidea title="Drift was named, never tested"}
From 2017 to 2026 no design study tested preference drift within a session
with a repeated-measures design, for example by showing early pairs again at
the end of a session; most papers list drift as a limitation and leave it
there. A
Gaussian process model built on a latent utility and i.i.d. noise treats
anchoring, diminishing returns, and criteria that form during the session all
as noise (inference). The test costs a few comparisons per session
(@exr-hci-drift-design), so a team running PBO with people can add it to every
study (inference).
:::

The models that could absorb such effects are discussed in
@sec-obs-extensions and @sec-theory-drift, and the experiment that would
separate noise from drift in @sec-open-decisive.

## Forms of feedback {#sec-hci-feedback-forms}

What should the system ask? @sec-query-design treats the query forms as
methods; this section collects what was learned with people. Before 2026 each
study varied one aspect of the query. @koyama2020computational argue that a
query over a continuous space carries much more information than a comparison
but is hard for crowd workers, so the single slider is their compromise.
@chong2021interactive measured 17.2 seconds per iteration with one slider and
53.4 with four; four converged faster per iteration and were preferred, but
$N = 3$. In virtual-reality color grading, users preferred comparing two
options to four [@yuan2025personalized]. A "don't know" area let people express
incomplete preferences [@ou2023impact]; in the field study, rating 4 models
instead of picking 1 did not remove inconsistency [@ou2022human] (@sec-ties
covers ties and abstention). And
@mikkola2020projective argue that slider-like projective queries greatly
reduce user workload.

**The first comparisons on the same task, in 2026.** GimmBO ($N = 12$,
within-subjects, 20 iterations) found that ranking beat sliders in similarity
to the target, in success rate, and in recovering the right sparse set of
adapters; sliders switched on superfluous adapters and Sequential Gallery
users got stuck in local minima, but ranking took longer per step (50.5 against
34.7 seconds) [@liu2026gimmbo]. APPO, with binary preferences only, converged
faster and with lower load but was less expressive than baselines that allowed
text feedback or manual edits [@li2026preference]. MultiBO argues that choosing
one of 4 candidates per round balances information against load better than a
pair, from a design analysis rather than a controlled experiment
[@rajagopalan2026personalized]. In an online study with 12 new users,
@peng2026efficient (a preprint) found that personalization from 8 pairwise
judgments beat every baseline, including shared design guidelines and the
users' own written preferences, with an aggregate win rate of 60.35% against
50.9% for a judge conditioned on a persona. And @mo2024cooperative criticize galleries because all options are still
supplied by the optimizer, leaving the user nothing to express beyond liking
or disliking them.

These results point the same way: lower the burden of each query or raise the
information it carries. No study yet compares pairs, galleries, sliders,
rankings, and ratings on one task with real users, so advice on format rests
mainly on simulation, very small samples, and the authors' arguments
(inference). Until it exists, the cheapest safeguard is a short pilot of two
candidate formats on the real task, timing each answer and repeating a few
queries (inference).
@sec-interface-model takes up why the interface is part of the model.

## Individual differences and population priors {#sec-hci-population}

A crowdsourcing framework assumes one preference shared by the crowd, though
crowds from different backgrounds "can have clearly different preferences"
[@koyama2020computational]. Population priors borrow other people's data to
help each new person. They work in experiments, but their gains are
concentrated in the first iterations.

**The evidence for population priors.** The global Gaussian process of
@liao2024practical (performance) produced a design that cut completion time by
5.5% and spatial error by 48%, and the warm-start Gaussian process raised
hypervolume by 38% for experienced and 18% for novice users, though a global
design oriented toward speed was not better than the baseline in spatial
error. Meta-Bayesian optimization of wrist interaction improved absolute
pointing by 22.92% over standard Bayesian optimization and 21.35% over manual
calibration, and relative pointing by 25.43% and 13.60% [@liao2024meta].
Meta-PO (preference feedback) cut the iterations to satisfaction from 9.54
without transfer (standard deviation 2.19) to 5.86 when transferring between
users with the same theme (standard deviation 1.20, $t(11) = 5.11$,
$p < 0.001$) and to 7.41 across themes (standard deviation 1.28, $p < 0.01$);
across themes needed more iterations than within a theme ($p < 0.01$), so the
more the goals differ, the less transfer helps [@li2025efficient]. HOMI
was significantly better only at iterations 2 and 3 ($p = .040$ and
$p = .002$); from iteration 6 all methods converged to the same level, and
NASA-TLX did not differ [@liao2026efficient].

**The evidence for individual differences.** Every study that measured
individual differences found them large: hearing-aid gain adjustments
($N = 20$) [@jensen2019perceptual] and preferred compression ratios
[@baltzell2018efficient] (@sec-health-audio), the font that maximized reading
speed [@kadner2021adaptifont], the trade-off between typing speed and accuracy
[@shen2022personalization], design preferences for air-taxi interfaces
[@meinhardt2025fly], and preferred blur in driving [@li2026blurdriving]. Even
expert preference and measured outcome can disagree: in a pre-study,
professional ad designers' preferences showed no positive correlation with
actual click-through rates, which is why constrained PBO hands the click-through rate
to the machine [@iwai2025constrainedb].

A population prior shifts the unit of analysis from the person to the
population, which sits uneasily with these individual differences. The
studies show that population priors speed up early convergence; none shows
that they do no harm when an individual departs from the population
(inference). A system using such a prior should therefore let the person's own
answers outweigh it quickly, and report results separately for the people
farthest from the population (inference).
@sec-many-users and @sec-obs-surrogates describe the models.

## Novices and experts {#sec-hci-expertise}

Who is being optimized for matters as much as how. In the study of
@ou2023impact ($N = 60$), "novices can achieve an expert level of quality
performance, but participants with higher expertise led to more optimization
iteration with more explicit preference while keeping satisfaction low. In
contrast, novices were more easily satisfied and terminated faster." The
authors' explanation is that "experts seek more diverse outcomes while the
machine reaches optimal results". In the field study, too, lab participants
were more easily satisfied than the expert artists, who applied quality
criteria the lab participants ignored [@ou2022human].

Experts repeatedly asked for direct control: to manipulate parameters outside
the gallery [@koyama2020sequential], or to edit glyphs, switch off style
propagation, and see a fuller history [@tatsukawa2025fontcraft]. They were
positive when the optimizer took over what they could not judge themselves:
the 11 ad designers of constrained PBO, with 5.82 years of experience on
average, endorsed letting the system manage click-through rate
[@iwai2025constrainedb]. Materials-science experiments also found experts and
non-experts behaving differently (@sec-sci-experts).

Most samples, however, are novices or students: 40 novices in Chan et al., 18
participants aged 20 to 36 in Mo et al., 10 non-experts in FontCraft,
participants all with a computing or machine-learning background in GimmBO,
and participants aged 24 to 28 in APPO [@chan2022investigating;
@mo2024cooperative; @liu2026gimmbo; @li2026preference]. What is known about
professional designers using PBO tools comes mainly from two technical artists
and two interviews (inference).

## Explanation and trust {#sec-hci-trust}

Explanations improved task performance in experiments. In a between-subjects
experiment with $N = 213$ tuning 6 parameters of a simulated egg-boiling task,
any explanation (a bar chart, rules, or text) improved task success, reduced
the trials needed, and improved understanding and confidence, without adding
workload [@chakraborty2025explanation]. MOLONE's authors report that comparative
explanations over inputs and outcomes gave significantly faster convergence in
a user study whose size we could not obtain [@chakraborty2025comparative].
ShapleyBO, a preprint, explains each proposal with Shapley values, which
divide the proposal's expected value among the input parameters; in
personalizing an exosuit, teams that saw the explanations had lower cumulative
regret, the total shortfall from the best design over the session
(@sec-regret) [@rodemann2024explaining]. CoExBO explains each candidate to
foster trust and comes with a no-harm guarantee: even under adversarial input
from the user, it converges like ordinary Bayesian optimization
[@adachi2024looping]. Participants of @niwa2025cooperative found that
explanations made unexpected suggestions more acceptable, while some found
them too generic or too long.

Understanding the optimizer can also change the feedback. @colella2020human
found that users who understood the optimizer gave strategically biased
answers (21 people, a one-dimensional task, scalar feedback), unlike the
truthful simulated users of most papers. @sandholtz2021inverse (*Bayesian
Analysis*, 2023) found that many participants explored in ways standard
acquisition functions do not capture, and a preprint by @weichert2025less
documents an industrial case in which adding data and expert knowledge made
Bayesian optimization worse.

The "trust" in the Ulm studies is trust in the vehicle, an objective being
optimized, not trust in the optimizer [@jansen2025opticarvis], and trust in
the optimizer itself has almost never been measured directly in HCI studies of
PBO (inference); a study that wants to claim it should ask about the optimizer
by name. Making the search visible
and editable (explicit constraints, process disclosure, a withdrawable
history) appears to sustain agency better than text explanations alone
(inference).

## Against strong baselines {#sec-hci-baselines}

A method can look good by beating a weak competitor. @tbl-hci-baselines sorts
the main comparisons by how strong the comparison condition was. The grading
is this chapter's judgment (inference): random queries, a single slider, and
no visualization count as weak; designers' own design, manual parameter
tuning, and settings from the literature as medium; similar optimizers,
Bayesian optimization that ignores cost, and expert designs as strong.

::: {.table #tbl-hci-baselines title="Main comparisons, sorted by the strength of the comparison condition. The grading of strength is this chapter's judgment (inference). Sources: ROMBO [@marcos2025random], the multi-session study [@colley2026multi], and the studies cited in the text."}
| Study | Compared with | Strength | Result |
|---|---|---|---|
| ROMBO (2025) | random queries | weak | new favorite tracks found 40% more often and 16% faster, 18% less time on disliked tracks ($N = 16$) |
| FontCraft (2025) | single-slider Bayesian optimization | weak | closer to target; no inferential statistics seen ($N = 10$) |
| OptiCarVis (2025) | no visualization; user-customized; expert design | weak to strong | personalized designs beat expert and customized designs on several ratings |
| Sequential line search (2017) | commercial auto-enhancement | medium | large lead in crowd votes |
| Dudley et al. (2019) | baseline interface | medium | no difference in batch 1; shorter task times from batch 2 |
| Shen et al. (2022) | baseline keyboard | medium | speed +14.4% ($p = 0.0059$), accuracy +13.8% ($p = 0.0046$), learning effects ruled out |
| Liao et al. (2024) | standard Bayesian optimization; manual calibration | strong; medium | absolute pointing improved 22.92% and 21.35%, relative pointing 25.43% and 13.60% |
| Tanaka et al. (2026) | sliders | medium | 62.5% preferred the optimized results; 85% endorsed their diversity |
| GimmBO (2026) | sliders; gallery | medium | ranking better, but slower per step |
| Liao et al. (IEEE Pervasive Computing 2023) | designers' own strategies | medium | "very similar" design performance |
| Song et al. (2025) | manual; ParetoSelect | medium to strong | layouts ranked above ParetoSelect's ($p = .02$); no difference in hypervolume, overall quality, or experience |
| Multi-session vehicle visualization (AutomotiveUI 2026) | the day-one design kept on days two and three | medium to strong | continued optimization rated better on cognitive load, trust, predictability, and perceived safety ($N = 74$); the authors report shortcomings for subjective measures |
| Mo et al. (2024) | optimizer-led | strong | no difference in hypervolume; fewer Pareto designs |
| Niwa et al., study 2 (2025) | cooperation by explicit constraints | strong | no difference in performance ($p = 0.204$) |
| Cost-aware Bayesian optimization (2026) | Bayesian optimization ignoring cost | strong | about 67% of the cost; final quality no different ($p = .77$) |
| HOMI (2026) | transfer acquisition; continual Bayesian optimization | strong | no difference from iteration 6 on |
| BlurDriving (IMWUT 2026) | no blur | medium | no significant improvement in objective driving performance |
| Ou et al., field deployment (2022) | none | not applicable | 415 of 549 sequences never reached optimization |
:::

@fig-hci-evidence places the studies on two axes, sample size and the
strength of the comparison, colored by feedback type.

```{figure}
//| figure: hci-evidence
//| label: fig-hci-evidence
//| fig-cap: "An evidence map of the HCI studies in this chapter: participants (log scale) against the strength of the comparison condition (rows), colored by feedback type, with the marker showing what the comparison found. Choose a study, or click its marker, to read its comparison and result. The grading and the coding of results are this chapter's judgment (inference). Not plotted: sequential line search and BlurDriving (no participant count in the tables), photographer-in-the-loop (its comparison measured the experience of using the tool), and Owaki et al. (a comparison of representations). In the top row, against strong comparisons, six of the nine markers are diamonds, the same outcome at lower cost; along the x axis, no preference-feedback study (blue) has more than 60 participants."
```

The pattern is clear (inference). The stronger positive results come from
performance objectives: task time,
typing speed, pointing error. Studies of pure preference feedback more often
report satisfaction or iteration counts, or a win over a weak comparison such
as a single slider (ROMBO's win over random queries was obtained with
ratings). Against skilled manual work or a similar optimizer, final quality
usually does not differ. What interactive Bayesian optimization delivered in
design from 2017 to 2026 is best summarized as the same result at lower cost,
in fewer iterations, evaluations, or minutes of effort, rather than a better
result (inference). @sec-rec-evaluation turns this into advice on how to
evaluate a new system.

## Critiques and alternatives from design research {#sec-hci-critiques}

Between 2017 and 2025 the critiques came mainly from the authors of
optimization papers and from neighboring HCI research, and they are of two
kinds. **The first concerns agency**, with the evidence of @sec-hci-agency:
Chan et al. propose "to push the optimizer to the background, making its
suggestions recommendations and not dictations" [@chan2022investigating], and
@mo2024cooperative add that an optimizer typically cannot "leverage the
designer's expertise in quickly identifying that a given 'bad' design is not
worth" evaluating.

**The second concerns the utility model itself.** @ou2022human conclude that
"we need to generally rethink basic assumptions and approaches in the design
of HITL systems". @koyama2020computational ask "Whose preference?": crowd
frameworks assume "a 'general' (or universal) preference shared among crowds",
while in some domains "only experts can adequately assess the quality of
designs". @koyama2022bo leave open whether BO-generated suggestions are
creative, and @dudley2019crowdsourcing could not tell whether their optimizer
truly optimized or only excluded poorly performing regions. In 2026,
@langerak2026cost wrote that a predefined parameter space limits early
exploration while the design space is still evolving, and
@owaki2026feasibility argue that the representation shown to the user is part
of interaction design; in their study ($N = 40$) a 5-dimensional
feasibility-aware latent space beat the raw 9-dimensional parameters.

**Fixation.** **Design fixation** is the tendency to stay close to examples
one has seen, or to one's first idea. An AI image generator used during
ideation led to more fixation, fewer ideas, and lower variety and originality
($N = 60$) [@wadinambiarachchi2024effects], and students can fixate more on
their own first idea than on given examples [@leahy2020design]. A PBO gallery
is a stream of system-supplied examples with the current best as the first
idea, so the same fixation can be expected when PBO is used for ideation
(inference). @gmeiner2023exploring watched 14 trained designers struggle to
understand and steer generative design tools.

**Preferences formed in the interaction.** Research on co-creation with
generative AI in 2025 and 2026 argued that preferences form during the
interaction, that premature convergence and fixation are the main failures,
and that some friction may help. Switchable divergent and convergent modes
(HAICo, a preprint) scored higher than ChatGPT on every factor of the
Creativity Support Index ($N = 24$, $p < 0.002$) [@wen2026exploration];
IdeaBlocks led designers to explore 2.13 times as many images, with 12.5%
higher visual diversity [@choi2026ideablocks]; a workshop paper argues for
keeping reflective friction [@avelino2026creativity]; a preprint found that
elicitation surfaces preferences users had not yet formed
[@kim2026elicitive]; and @saracay2026expert (COLM 2026) argue that agents
should help users construct preferences, with a simulated-user benchmark whose
user model a study with 25 people supports. @sec-what-comparisons-measure
returns to what a comparison measures if preferences are partly built while
answering.

**The alternatives move toward mixed initiative**, in which either side can
take the lead: the systems of @sec-hci-agency, a withdrawable history
[@tatsukawa2025fontcraft], constraints handled by the machine
[@iwai2025constrainedb], and the designer as a curator of constraints rather
than a maker of prototypes [@jansen2025human], a workshop paper. For ideation,
@koch2019may used cooperative contextual bandits, a sequential recommender
that adapts to feedback, to suggest inspirational material for mood boards
instead of converging on an optimum; 14 of 16 professional designers preferred
the tool.

**Counter-evidence: optimization can support exploration.** Optimizer-led
designers explored more of the design space [@chan2022investigating],
cooperative designers moved farther between evaluations than designer-led ones
[@mo2024cooperative],
participants of Niwa et al. described suggestions that broke their fixation as
"Oh, I see!" moments [@niwa2025cooperative], and 85% of the architecture
students endorsed the diversity the optimizer brought ($p < 0.001$)
[@tanaka2026human].

**Where the two critiques stand.** The agency critique already has tested
remedies (cooperative control, explicit constraints, an exploration slider)
that restored most of the agency in experiments. The utility-model critique
has only proposed remedies inside PBO, such as discarding history, decaying
old data, and withdrawing choices, and a PBO tool has yet to be compared with
an ideation tool built for divergence on the same creative task (inference).
Critics and optimization authors largely agree on the diagnosis (fixation,
unclear goals, preferences that form during the interaction) and disagree on
the remedy.

The evidence suggests a working rule for design tools (inference): put the
optimizer in the role of an advisor; give the machine the measurable
sub-goals, such as click-through rate, feasibility, or task time; leave
judgments of appearance to comparisons or rankings; make the state of the
search visible and editable; and evaluate against manual tuning or skilled
designers, not only against a single slider or random queries. The rule
comes from the empirical results of this chapter and does not depend on the
Gaussian process framework.

## Common claims, checked {#sec-hci-claims}

@tbl-hci-claims checks claims that circulate in secondary accounts against
the primary sources.

::: {.table #tbl-hci-claims title="Claims about the HCI literature on interactive Bayesian optimization, checked against the primary sources."}
| Claim | Verdict | What the sources show |
|---|---|---|
| Sequential line search converges in about 15 to 20 iterations on a 6-dimensional problem. | partly right | The photo task has 6 parameters, and the crowdsourced runs were fixed at 15 iterations; runs from different starts came together within the first 4 to 5 iterations. Nothing supports "15 to 20" [@koyama2020computational]. |
| AdaptiFont, continual human-in-the-loop optimization, and Chan et al. are evidence about PBO. | wrong scope | AdaptiFont optimizes reading speed, Chan et al. speed and accuracy, and continual human-in-the-loop optimization uses performance feedback; none uses preference feedback [@kadner2021adaptifont; @chan2022investigating; @liao2025continual]. |
| In Niwa et al., a language model turns design intent into constraints on the search space. | partly right | The language model lets designers intervene in a system-led optimization and explains its reasoning; the study compared it with a method that uses explicit constraints [@niwa2025cooperative]. |
| Constrained PBO (Iwai et al. 2025) is the first PBO with inequality constraints. | the paper's own claim, too strong | The authors are Iwai, Kumagae, Koyama, Hamasaki, and Goto. "First" overlooks C-GLISp (IEEE TCST 2022) and StageOpt (ICML 2018) [@iwai2025constrainedb; @zhu2022c; @sui2018stagewise]. |
:::

Two citation details are easy to get wrong. The 2020 SIGGRAPH paper is called
*Sequential Gallery*; "sequential plane search" is the method inside it
[@koyama2020sequential]. And the melody work is two papers, a 2020 pilot
[@zhou2020generative] and the IUI 2021 study with 12 participants
[@zhou2021interactive].

## Settled, contested, missing {#sec-hci-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** In design tasks with performance objectives, letting the
optimizer lead lowers agency, ownership, and expressiveness by a large margin
while improving outcome measures modestly, and channels for steering restore
most of the lost agency at a small cost in outcome [@chan2022investigating;
@mo2024cooperative; @niwa2025cooperative]. Population priors cut the
iterations people need [@li2025efficient], and in the one study that tracked
the whole session their advantage was gone by the sixth iteration
[@liao2026efficient]. Individual differences are large wherever they were
measured. Real sessions often end early: in the one long field deployment,
three quarters of sequences never reached optimization [@ou2022human].

**Contested.** Whether optimizers help or hinder creative exploration: the
fixation literature points one way, measures of design-space coverage the
other. Whether the gains of interactive Bayesian optimization survive strong
comparisons, where final quality usually does not differ. Which query format
is best: the first same-task comparisons appeared only in 2026, with 12 to 16
participants. Whether explanations build trust in the optimizer, or only
improve task performance.

**Missing.** A repeated-measures test of preference drift within a session.
A comparison of pairs, galleries, sliders, rankings, and ratings on one task
with real users. A measurement of fixation inside a PBO session. A direct
measure of trust in the optimizer. A study of agency with pure preference
feedback and a sample comparable to Chan et al.'s. Studies of professional
designers beyond two artists and two interviews. A comparison of a PBO tool
with an ideation tool built for divergence on the same task. Evidence that
population priors do no harm to people far from the population (inference).
:::

## Exercises {#sec-hci-exercises}

::: {.exercise #exr-hci-denominator}
@fig-hci-field-funnel gives three success rates for the same tool: 2.9% of all
field sequences, 11.9% of the field sequences that asked for optimization, and
48.5% of the lab sequences. Which of them describes how the tool performs in
practice? Why might a paper that reports only the 134 sequences, or only the
lab study, give a reader the wrong idea?

::: {.solution}
The 2.9% ($16/549$) does, against $16/134 \approx 11.9\%$ of the sequences that
requested optimization and $97/200 = 48.5\%$ in the lab. The 415
sequences that stopped at the first iteration are part of how the tool was
used: the artists looked at the first grid and stopped, perhaps because a
candidate was good enough, perhaps because they gave up. Reporting only the
sequences that entered optimization conditions on people having already
chosen to keep going, and reporting only the lab study replaces professional
users with participants who, as the authors found, are more easily satisfied.
Both choices make the method look better than its field use.
:::
:::

::: {.exercise #exr-hci-drift-design}
No design study between 2017 and 2026 tested preference drift within a session.
Design the simplest test you can add to an existing PBO session of 30
comparisons. What would you repeat, when, and what pattern in the answers
would distinguish drift from ordinary answer noise?

::: {.solution}
Repeat some early pairs at two later points: for example, show pairs 2, 4, and
6 again right after they were first answered (short lag), and again at the end
of the session (long lag), in random order among the regular queries. Answer
noise alone predicts the same agreement rate at both lags, because each
answer is an independent noisy reading of a fixed utility. Drift predicts
lower agreement at the long lag than at the short lag, and in a consistent
direction: if the person's criteria moved, the late answers should agree with
the model fitted to late data better than with the model fitted to early
data. With a handful of repeats per person the test is weak for one person
and useful across a group; the cost is a few extra comparisons per session.
@sec-open-decisive describes a fuller version of this experiment.
:::
:::

::: {.exercise #exr-hci-baseline}
A new paper reports that its preference-based design tool reached a
satisfactory result in 6 iterations against 11 for a single-slider baseline,
with 12 participants. Using the grading of @tbl-hci-baselines, how strong is
this comparison? Name two comparison conditions that would make the result
more informative, and say what each would rule out.

::: {.solution}
A single slider is a weak comparison in the chapter's grading. Two stronger
ones: manual tuning of the same parameters by the same participants with the
same time budget, which tests whether people would do as well on their own;
and a similar optimizer with a different query form or acquisition function,
which tests whether the gain comes from the new component rather than from
optimization in general. Measuring final quality with judges who did not take
part, not only iterations to satisfaction, would also separate "faster" from
"better".
:::
:::

## Further reading {#further-reading .unnumbered}

- @chan2022investigating is the clearest controlled study of agency against
  performance; read its discussion of mixed initiative alongside the results.
- @mo2024cooperative show what a cooperative middle ground looks like and
  measure it against both extremes in one within-subjects study.
- @ou2022human is the only long field deployment of PBO with professionals,
  and the most detailed record of how real feedback departs from the model.
- @koyama2020sequential and the book chapter @koyama2020computational give the
  interface line's reasoning about query forms and its frank list of
  limitations.
- @niwa2025cooperative measure outcome and agency together when the designer
  steers in natural language, and are the bridge to the language-model systems
  of @sec-llms.
