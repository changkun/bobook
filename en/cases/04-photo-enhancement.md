---
status: done
synopsis: "Preferential Bayesian optimization on a real photograph with six adjustments: you enhance it by picking one of two versions or by sliding along a line, the model learns your taste, and the chapter sets what a session reveals (consistency, drift, dependence on the order of questions) against the published studies."
sources: ["Koyama et al. 2017", "Koyama and Igarashi 2020", "Koyama et al. 2020", "Ou et al. 2023"]
---

# Enhancing a Photo by Comparison {#sec-cs-photo}

The preference chapters taught their methods on designs small enough to see
whole: a hue on a color wheel in @sec-pbo, a flat poster landscape with two
knobs in @sec-line-search. Real design tasks are not like that. A photograph
has a dozen adjustments that interact, nobody can write down the function
that scores them, and the only instrument that can measure how good a version
looks is a person, who gets tired, changes their mind, and sees a different
picture on every screen.

In this chapter the problem is real because you are part of it. The figures
show a real photograph, and you enhance it by comparing versions: either by
choosing the better of two, or by sliding along a line through the space of
adjustments, the two query forms that @koyama2017sequentialb and
@koyama2020sequential studied for photo color enhancement. The algorithm is
the one built in @sec-gp-preference and @sec-pbo, unchanged, running on six
parameters instead of one or two. We first describe the problem and what a
single judgment costs, then the decisions that have to be made before the
first question, then the two query forms on the photograph, and finally what
a session reveals about the person in it, set against what the published
studies found.

## The problem {#sec-cs-photo-problem}

### A photograph and six adjustments {#sec-cs-photo-adjustments}

The photograph was taken by Aleš Krivec on 19 October 2014 at the Zelenci
nature reserve in Slovenia and dedicated to the public domain under CC0 1.0
([Wikimedia Commons, "Zelenci 2014 (2).jpg"](https://commons.wikimedia.org/wiki/File:Zelenci_2014_%282%29.jpg),
first published on Unsplash). It was chosen because its look depends on
every adjustment we will offer. It has a bright sky and a lit
mountain face that clip if pushed, a dark forest whose detail lives in the
shadows, golden grass that turns orange or olive with the white balance, and
turquoise water that a warm setting dulls. Most changes that improve one
region hurt another, so there is no setting that every viewer will call best.

A *design* is a setting of six adjustments, each a number $u_i \in [0, 1]$,
with $u_i = 0.5$ the untouched photograph. @tbl-cs-photo-knobs lists what
each one does and the range it spans. Mapping every adjustment to the unit
interval is the usual convention, also used by the photo enhancer in the study
of @ou2023impact; it lets one kernel lengthscale mean roughly the same thing
in every direction.

::: {.table #tbl-cs-photo-knobs title="The six adjustments. Each maps $u \in [0, 1]$ to a setting, with $u = 0.5$ the original photograph. EV is an exposure value, or stop: one stop doubles the light."}
| Adjustment | What it does | At $u = 0$ | At $u = 1$ |
|---|---|---|---|
| exposure | brightens or darkens; dark tones scale by $2^{\text{EV}}$, white stays white | $-1$ EV | $+1$ EV |
| contrast | stretches or flattens tones about mid-gray | $\times 0.66$ | $\times 1.52$ |
| saturation | moves each color away from or toward its gray | $\times 0.5$ | $\times 2$ |
| temperature | trades red against blue | cool | warm |
| tint | trades green against magenta | green | magenta |
| shadows | lifts or deepens the darker third of the tones | deeper | lifted |
:::

Before any optimization, try the problem the way a photo editor presents it:
six sliders.

```{figure}
//| figure: cs-photo-knobs
//| label: fig-cs-photo-knobs
//| fig-cap: "The six adjustments of @tbl-cs-photo-knobs by hand, on a real photograph (Zelenci, Slovenia, by Aleš Krivec, CC0 1.0, [via Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Zelenci_2014_%282%29.jpg)). Move the sliders to edit the picture; the original stays beside it. The small pairs show each adjustment at the two ends of its range with the other five where you left them, so they change as you edit. Nothing here is simulated or learned. Find the version you like best, note its six settings, and compare them later with what the comparison sessions find."
```

Spend a minute with it. The sliders are easy to move and the result is hard
to settle: improving the water can make the grass worse, more exposure calls
for less shadow lift, and soon it is unclear whether the last change helped.
@koyama2020computational describe the same experience: sliders affect the
picture "in complex ways", so a designer explores by trial and error while
building a mental model of the design space, and as parameters are added the
exploration becomes "very tedious". That is the difficulty the rest of the
chapter hands to an optimizer.

Three of the adjustments are multiplicative, so their ranges are symmetric on
a logarithmic scale: halving the saturation is as large a change as doubling
it. The hyperparameters of the classifier in @sec-cs-classifier-space are
searched on a log scale for the same reason. The adjustments also interact. Exposure and shadow lift both
brighten the picture, so too much of one can be undone by less of the other.
Saturation and contrast both add punch, and temperature and tint both move
the white balance. The utility of a design is therefore not a sum of six
separate preferences, which is one reason a Gaussian process over all six
together is a sensible model.

The figures apply the adjustments in the browser, with an SVG filter on the
image: a lookup table for the tone curve (exposure, shadows, contrast), then a
color matrix (saturation, temperature, tint). The same computation on an
array of pixels takes a dozen lines.

::: {.code title="The six adjustments on an image array"}
```python
import numpy as np

def enhance(img, u):
    """img: H x W x 3 sRGB values in [0, 1]; u: six numbers in [0, 1]."""
    ev, c, s, t, m, a = (np.asarray(u) - 0.5) * 2   # each in [-1, 1]
    k = 2.0 ** ev                                    # exposure in stops
    y = k * img / (1 + (k - 1) * img)                # scale dark tones, keep 0 and 1
    y = y + 0.14 * a * y * (1 - y) ** 2 / (4 / 27)   # lift or deepen the shadows
    y = np.clip(0.5 + 2.0 ** (0.6 * c) * (y - 0.5), 0, 1)   # contrast
    luma = (y @ np.array([0.2126, 0.7152, 0.0722]))[..., None]
    y = np.clip(luma + 2.0 ** s * (y - luma), 0, None)      # saturation
    gain = np.array([1 + 0.11 * t + 0.03 * m,        # temperature: red up, blue down
                     1 + 0.015 * t - 0.07 * m,       # tint: green down
                     1 - 0.13 * t + 0.03 * m])
    return np.clip(y * gain, 0, 1)
```
:::

The filter is deliberately simple. A full photo editor offers far more, for
instance separate color balance for shadows, midtones, and highlights;
Sequential Gallery's photo enhancer had 12 parameters, including that
three-way color balance [@koyama2020sequential]. Six is the number the 2017
sequential line search used, with brightness, contrast, saturation, and a
red, green, and blue color balance [@koyama2020computational].

### Why the eye is the only objective {#sec-cs-photo-eye}

An engineer's first instinct is to find a formula. Image-quality metrics and
automatic enhancement exist, and every photo editor has an *auto* button. But
an automatic enhancement encodes somebody's taste, learned from somebody's
data, and the question here is what *this* viewer wants. When
@koyama2020computational asked crowd workers which of four versions of a
photograph looked best (the original, the result of their crowd-driven
optimization, and the automatic enhancements of Adobe Photoshop and
Lightroom), the optimized versions of three photographs received 32, 26, and
29 votes; the originals received 0, 2, and 0, Photoshop's versions 0, 2, and
1, and Lightroom's 1, 3, and 3.

The objective also depends on the purpose, which is set by the instruction,
and the published studies set it differently. The crowd workers of sequential
line search were told simply to adjust the slider until the image looked best
[@koyama2017sequentialb]. In a pairwise-comparison task from the same line of
work, workers chose the photograph "that would be better to use in a magazine
or product advertisement" [@koyama2020computational]. The participants of
Sequential Gallery imagined uploading the photographs to their Facebook or
Instagram accounts and making them appealing to their friends
[@koyama2020sequential]. These instructions need not share an optimum
(inference). The figures below ask only which version you prefer; decide for
yourself what the picture is for, and keep that fixed.

Asking a person for a number would not work well either. A score on a scale
from 1 to 10 requires knowing what the rest of the design space looks like,
which a newcomer does not, while a comparison "is easy to answer even for
non-experts" [@koyama2020computational]; @sec-cmp-ratings-hard collects the
evidence. So the measurement is a comparison, the person is the instrument,
and every property of that instrument (its noise, its drift, its patience)
becomes a property of the optimization problem.

### What a judgment costs {#sec-cs-photo-cost}

A judgment costs seconds, not hours. In the closest published analogue, 11
professional designers compared color variations of banner ads, 50 pairs per
task on spaces of 12 and 6 parameters, and took 4.2 seconds per choice on
average, ranging from 1.0 to 23.1 seconds [@iwai2025constrainedb]. A richer
query takes longer: one plane of Sequential Gallery took 14.8 seconds on
average [@koyama2020sequential]. For a single slider, the nearest measurement
comes from a different task: when three participants steered a generative
image model toward a reference image, five reference images each, one
iteration took 17.2 seconds on average with one slider and 53.4 seconds with
four [@chong2021interactive]. Neither the 4.2 seconds nor the 17.2 seconds
was measured on photo enhancement. For a slider against a pair on a photo
task we found only a relative measurement: in crowdsourced color enhancement
toward a reference image, a slider task took longer than a task of choosing
between two versions, but less than twice as long, with the time for reading
the instructions and for repeated quality-control questions included
[@koyama2017sequentialb].

The binding constraint is attention. Participants in the Sequential Gallery
study pressed a "satisfied" button after 5.36 iterations on average (standard
deviation 2.69) [@koyama2020sequential], and in the three-month field
deployment described in @sec-hci-unstable most evaluation sequences never
reached the optimizer at all [@ou2022human]. The crowd version of sequential
line search bought its patience: 15 iterations of seven slider tasks each,
paid 0.05 USD per task (@sec-query-crowd). A personal session has perhaps ten
to thirty answers to spend, the same order as the comparisons in
@fig-pbo-oracle, but now in six dimensions.

Two further costs never appear in a test function. Every answer is noisy:
the same person, shown the same pair twice, will sometimes answer
differently, which @sec-thurstone models as noise on each option's perceived
utility. And the instrument includes the display. Sequential Gallery's study
fixed it, a 13-inch MacBook Pro with a mouse [@koyama2020sequential]; you
are reading this on a screen of unknown brightness and color, in light we
cannot see, and a version that looks best on a phone in daylight may look
garish on a monitor at night (inference).

## Before the first question {#sec-cs-photo-setup}

Every choice in this section is made before any data arrives, and each one
changes what the optimizer can find. @tbl-cs-photo-setups puts the choices
of three published photo-enhancement systems next to those of the figures in
this chapter.

::: {.table #tbl-cs-photo-setups title="Choices made before the first question, in three published photo-enhancement systems and in this chapter's figures."}
| | Sequential line search (2017) | Sequential Gallery (2020) | Ou et al. (2023) | This chapter |
|---|---|---|---|---|
| Parameters | 6: brightness, contrast, saturation, red, green, blue balance | 12: brightness, contrast, saturation, color balance for shadows, midtones, highlights | 5 in $[0, 1]$: brightness, contrast, saturation, temperature, tint | 6 in $[0, 1]$: @tbl-cs-photo-knobs |
| Instruction | adjust the slider until the image looks best | make the photo appealing to friends on Facebook or Instagram | improve and enhance the color of the photo | which version do you prefer? |
| Who answers | crowd workers; median of at least 5 slider answers per iteration | one person | one person (20 per task) | you, or a simulated taste |
| Question | one slider from the best design to the expected-improvement design, starting at a random position | a 5 by 5 zoomable grid on a plane | rank 4 versions, with an "I don't know" region | a pair, or a slider along a full line, starting at a random position |
| Answer recorded as | chosen beats both slider ends (3-option choice) | chosen beats the plane's center and 4 vertices | the ranking of the 4 | pairwise comparisons |
| Model | Gaussian process; hyperparameters by MAP with log-normal priors | ARD Matérn 5/2 kernel, log-normal priors with $\sigma^2 = 0.01$ | BoTorch, with EUBO extended to rankings; kernel not stated | Matérn 5/2, one fixed lengthscale, EUBO or expected improvement |
| Start | a slider between two random designs | a random plane | 4 rounds of quasi-random (Sobol) designs | the original against a random edit |
| Stop | 15 iterations | 15 iterations; a "satisfied" button recorded when | "I'm satisfied" button, at most 20 iterations | when you stop, at most 24 answers |
:::

The sources for the table are @koyama2017sequentialb and
@koyama2020computational for the first column, @koyama2020sequential for the
second, and @ou2023impact for the third. The rest of this section explains
the choices in the last column, and where they differ from the published
ones, why.

**The box.** The ranges in @tbl-cs-photo-knobs decide what the optimizer can
reach. A box that is too narrow excludes the look the person wants; a box
that is too wide fills most of its volume with absurd pictures, and in six
dimensions volume is everything. A design drawn uniformly from our box is
almost always worse than the original, because the original sits at the
center and most random settings push at least one adjustment to an extreme.
@fig-cs-photo-race below shows the consequence: twenty random pairs leave a
simulated viewer with a recommendation no better than the untouched photo.
The ranges here were set by eye to span from "clearly too little" to "clearly
too much" for this photograph.

**The kernel.** The utility model is the Gaussian process of
@sec-gp-preference with a Matérn 5/2 kernel, a lengthscale of 0.4 in every
direction, prior variance 1, and probit noise $\sigma = 0.5$, fitted with the
Laplace approximation (@sec-pref-laplace). The values are fixed, not learned
from the answers. Learning six lengthscales (@sec-ard) from one-bit answers
at a budget of twenty is fragile, and @sec-hd-learning-ls reports that no
study has checked whether it is even possible at such budgets. Koyama and
colleagues reached the same practical answer by another route: their
log-normal priors on the kernel hyperparameters, with $\sigma^2 = 0.01$, leave
the lengthscales almost no room to move [@koyama2020sequential]. We chose
0.4 and 0.5 by running simulated sessions, which is the honest way to set
such values before a real person sits down.

**A visible difference.** The acquisition function of @sec-eubo knows nothing
about eyes. Left alone, EUBO often proposes two versions so close that a
person cannot tell them apart, a symptom of the collapse toward the current
best described in @sec-pbo-failure-modes. The authors of a study of
vibrotactile preferences name the same problem for touch among their
limitations: very similar signals "may be perceptually indistinguishable", and
comparisons between nearby points "are dominated by response noise"
[@zhang2026vibrotactile]. The figures therefore only consider pairs
whose two settings are at least 0.2 apart in Euclidean distance, a crude
stand-in for a just-noticeable difference in this space. Answers of "about
the same" (@sec-query-indifference) are the principled alternative; we kept
the interface to two buttons.

**The candidate pool.** In one dimension, @fig-pbo-oracle scored every pair
of 36 hues. In six dimensions there is no grid to enumerate, so before each
question the figure draws 160 settings uniformly from the box, 60 small
random perturbations of the current best guess, and every setting compared
so far, then scores every pair in this pool of up to about 250 settings with
the closed form @eq-eubo-closed, roughly 30,000 pairs per question. The local
perturbations matter: a uniform point lands within 0.1 of the best guess in
every coordinate with probability $0.2^6 \approx 6 \times 10^{-5}$, so
without them the pool would almost never contain the small refinements that
the end of a session needs.

**The recommendation.** The best guess is the compared setting with the
highest posterior mean, the rule sequential line search uses for its $\vx^+$
[@koyama2020computational]. Restricting it to settings the person has
actually seen means the recommendation is never a picture nobody looked at.

**The start.** The first question shows the original against a random edit.
The original stays in the data, so every later answer is connected to it
through the comparison graph (@sec-comparison-graph), and the figure always
shows the original beside the best guess. One participant in the Sequential
Gallery study asked for exactly that, to "have the original photo alongside
to compare with during the enhancement", and the same participant found the
initial photograph of one task already satisfactory and was unsure whether to
press the "satisfied" button at the very start [@koyama2020sequential]. A system that
cannot recommend the original is biased toward change (inference).

## Comparing versions {#sec-cs-photo-pairs}

Here is the loop with you as the oracle. Pick the version you prefer, a dozen
times or more. Take each choice at face value; there is no right answer.

```{figure}
//| figure: cs-photo-enhance
//| label: fig-cs-photo-pairs
//| fig-cap: "Enhancing a real photograph by comparison. The photograph is real (Zelenci, Slovenia, by Aleš Krivec, CC0 1.0, [via Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Zelenci_2014_%282%29.jpg)) and so are your answers; in the *You* mode nothing about your taste is simulated. Click the version you prefer. A Gaussian process utility over the six adjustments of @tbl-cs-photo-knobs is learned from your answers with the probit likelihood and the Laplace approximation, and the next pair is the pair with the highest EUBO among about 250 candidate settings, at least 0.2 apart. Below the pair: the best guess (yellow ring), the compared setting with the highest posterior mean, beside the original; and the learned utility along each adjustment with the other five held at the best guess (line: posterior mean; band: 95% credible interval; star: the best guess; A and B: the current pair). *Ask an earlier pair again* repeats one of your earlier pairs with the sides swapped, without saying so; the history marks it afterwards. In the *Simulated taste* mode a hidden utility answers instead, with simulated noise σ; the model's own noise is fixed and does not adapt to it."
```

The two large pictures are the question. Below them, the figure shows the
model's current best guess beside the original, so you can always check
whether the session is improving on doing nothing. The six small panels show
what the model has learned: each is a slice of the posterior
utility (@sec-gp-preference) along one adjustment, with the other five held at
the best guess, so a peaked curve says "this adjustment matters to you, and
here is where you like it", and a flat curve with a wide band says "no
information yet". The letters A and B mark where the two versions of the
current question sit on each adjustment, which shows what the question is
probing.

Some things to try. Answer ten or fifteen pairs and watch which panels sharpen
first: they are the adjustments your answers have been most decisive about.
Watch the letters A and B in the panels. The first question changes every
adjustment at once; later questions tend to hold most adjustments nearly
equal and vary a few, so each answer probes a smaller part of the space (in
simulated sessions, the two versions differ by more than 0.1 in all six
adjustments at the first question and in two or three by the tenth). After a
dozen answers, compare the best guess with the original and ask yourself
honestly which you prefer. Then press *Ask an earlier pair again* a few
times and answer as you normally would; we will come back to what it
measures in @sec-cs-photo-consistency.

Two cautions about reading the panels. A slice is not the whole story: it
shows how the utility changes along one adjustment *at the best guess*, and
with interactions between adjustments the best exposure at a different
shadow setting may differ. And the vertical axis has no units. Comparisons
identify utility differences only up to a shift and a scale
(@sec-pref-identifiability), so what can be read off is the shape of each
curve and the relative heights of the six, not a number.

Switch to *Simulated taste* to watch the same loop with a hidden utility
answering. The simulated taste has a favorite setting and a utility that falls
off quadratically from it, with different weights per adjustment (exposure
and temperature matter most, tint least) and the exposure and shadow
interaction described above. Press *Simulate five* a few times, then reveal
the favorite: its picture replaces the original beside the best guess, the
dashed lines mark its settings in the panels, and the status line reports the
*utility gap* of the best guess, the simulated utility at the favorite minus
the utility at the best guess, next to the gap of the original. A gap of zero
is perfect.

## Searching along a line {#sec-cs-photo-line}

A pair gives the model one bit. A slider gives it a whole one-dimensional
search, performed by the person: they move along a line through the design
space, watch the picture change, and stop where it looks best. This is
sequential line search [@koyama2017sequentialb], described in
@sec-line-search, now on the real photograph.

```{figure}
//| figure: cs-photo-enhance
//| label: fig-cs-photo-slider
//| fig-cap: "The same photograph and model with a slider. The photograph and your answers are real. The slider runs along a straight line through the best version so far (star on the track) and the version with the highest expected improvement over it (diamond), extended to the edges of the adjustment box in both directions; before the first answer it runs through the original and a random edit. Drag along the strip of thumbnails or use the buttons, then press *Choose this one*. Each choice is recorded as up to four pairwise comparisons: the chosen version beats both ends of the slider and the points one third and two thirds along it, skipping any within 0.15 of the choice. In the panels, the orange bar shows the slider's extent along each adjustment and the circle its current position. The *Simulated taste* mode is simulated, as in @fig-cs-photo-pairs; the simulated person picks the best of 41 noisy positions along the slider."
form: slider
```

The figure departs from the published method in two ways, and both are
choices a practitioner faces.

**Which line.** @koyama2017sequentialb run the slider from the best design so
far, $\vx^+$, to the design with the highest expected improvement over it,
$\vx^{\EI}$ (@sec-query-which-line), and mention enlarging the segment by a
fixed factor such as 1.25 as an option. We use the same two points but extend
the segment all the way to the edges of the box in both directions. The person can then
ask for *less* of the proposed change as well as more, and the slider covers
enough range for the differences to be visible. Before any answer, the line
runs through the original and the same random edit that opens the pairs
figure. Whether the extension helps depends on the noise, as the comparison
below shows. One published detail we kept: the slider starts at a random
position for every question, which @koyama2017sequentialb did "to reduce
cognitive bias", the pull of wherever the knob happens to rest (an anchoring
effect, @sec-jdm-context).

**How a choice is recorded.** Koyama and colleagues record a slider answer as
a choice from three options, the chosen design against both ends, with the
choice likelihood of @eq-query-luce [@koyama2017sequentialb;
@koyama2020sequential]. Our model has
only the pairwise probit likelihood of @sec-gp-preference, so the figure
records the choice as up to four separate comparisons: the chosen point
beats both ends and the points at one third and two thirds of the slider.
This overstates the evidence. The four comparisons come from one act of
judgment, so their errors are not independent, yet the model treats them as
four independent answers and becomes more confident than it should
(@exr-cs-photo-overcount). @mikkola2020projective take the opposite view,
that a slider answer is uncountably many comparisons, and approximate the
resulting likelihood with a finite sample; both records discard the fact that
the person saw the whole continuum at once.

Some things to try. Drag slowly along the strip and notice how much of the
line is clearly wrong; most of the information in your answer is in which
region you avoided. After a few answers the line gets shorter in the panels,
because the expected-improvement point moves closer to the best guess.
Move the knob to the star and choose it, and the answer says that the best
design so far beats everything on the line, which is a strong statement.

### Pairs or a slider? {#sec-cs-photo-race}

The two forms trade information per answer against time per answer. The
published comparison comes from the crowd. @koyama2017sequentialb ran their
slider against a choice between two versions and a choice among four, in
crowdsourced photo color enhancement toward a reference image, three runs per
condition. The workers matched a target picture; they did not state a
preference. The slider's error fell faster over the iterations, and a slider
task took longer than a comparison task, but less than twice as long. As of
September 2026 we found no study that compares the two forms for one person's
own preference on a photo task, with the same model and budget. The
precomputed simulations below run four query strategies (EUBO pairs, random pairs, the published slider segment,
and the full-line slider) sixteen times each against the simulated taste of
@fig-cs-photo-pairs.

```{figure}
//| figure: cs-photo-race
//| label: fig-cs-photo-race
//| fig-cap: "Four ways of asking, replayed in simulation. Everything in this figure is simulated: the taste (the same hidden favorite as in @fig-cs-photo-pairs), its noise σ, and, with the toggle, its drift; only the model and the question rules are those of the interactive figures. Top: the utility gap of the recommendation after each answer (0 is the hidden favorite; the dotted line is the untouched photograph), median (line) and interquartile range (band) over 16 sessions that differ only in their candidate pools, first edit, and answer noise. With *Seconds* on the horizontal axis, each pair costs 4.2 s and each slider answer 17.2 s; these times come from two different studies with different tasks, professional designers choosing between two color variations of a banner ad [@iwai2025constrainedb] and three participants moving one slider of a generative image model toward a reference image [@chong2021interactive]; they are assumptions here, not measurements of this task. Bottom: the 16 final recommendations (after 20 answers) for each adjustment, for EUBO pairs (upper dots) and the full-line slider (lower dots), with the hidden favorite; with drift on, where the favorite started (solid) and where it ended (dashed). Sessions precomputed with tools/figure-data/cs-photo-race.ts."
```

Per answer, the slider is ahead early and the forms converge later. At the
default noise ($\sigma = 0.1$), after five answers the median gap is 0.37 for
EUBO pairs and 0.34 and 0.30 for the published and the full-line slider;
after twenty answers it is 0.14 for EUBO pairs, 0.08 for the published
slider, and 0.14 for the full line, against 0.49 for the untouched photo. At
low noise ($\sigma = 0.05$) the three end within 0.02 of each other. At high
noise ($\sigma = 0.25$) the full-line slider holds up best, with a median gap
of 0.16 after twenty answers against 0.31 for EUBO pairs and 0.34 for the
published segment. Random pairs are a warning: after twenty answers at the
default noise, the recommendation beat the untouched photograph in only 7 of
16 sessions, so here EUBO pairs are far ahead of random ones. The recorded
runs of @sec-pbo-dims show the opposite ordering,
with EUBO no better than random pairs after about twenty answers, under a
different setup: generated designs with a different simulated taste, no
minimum distance between the two options of a pair, and a best guess taken
from the whole candidate pool rather than from the compared settings; we have
not separated which of these differences accounts for the reversal
(inference).

Per second, the picture reverses, if the borrowed timings of
@sec-cs-photo-cost hold for this task. Switch the axis to *Seconds*, which
charges each pair 4.2 seconds and each slider answer 17.2 seconds: in the 84
seconds that twenty pairs then take, a slider session gets through about five
answers, and its median gap is still about 0.3 while the
pairs are at 0.14. The simulated slider is also generous to the slider: the
simulated person evaluates 41 positions with independent noise and keeps the
best, a careful search that a real person may not perform in 17 seconds. The
reversal also rests on the assumed ratio of about four between the two times.
If a slider answer took twice as long as a pair, the upper bound that
@koyama2017sequentialb report for crowd tasks, a slider session would give
ten answers in those 84 seconds and reach a median gap of 0.17 to 0.18, close
to the pairs' 0.14 (inference). So the simulation does not settle the
question. It shows instead which numbers would settle it, the time per answer and the noise of a slider answer
compared with a pairwise one, measured on the same people and the same task
(inference). A 2026 study that compared interfaces on one task, GimmBO, had
12 participants match a target image by merging image-model adapters: its
ranking interface, driven by Bayesian optimization, reached a higher
similarity to the target and a higher success rate than a panel of sliders
adjusted by hand, at 50.5 against 34.7 seconds of interaction per step
[@liu2026gimmbo]; @sec-hci-feedback-forms reports it with the other
comparisons of forms.

## What a session reveals {#sec-cs-photo-session}

Run a session on yourself and three things become visible that a test
function hides: you do not always give the same answer twice, your taste may
move while you answer, and the result depends on which questions you
happened to be asked. Each has a counterpart in the published studies, and
for each the studies say less than one would like.

### Consistency {#sec-cs-photo-consistency}

*Ask an earlier pair again* shows you a pair from earlier in the session with
the sides swapped, and the history then marks your answer *same* or
*flipped*. Under the model of @sec-thurstone the probability of a flip has a
simple form. If the person prefers A to B with probability
$p = \Phi\big(\Delta / (\sqrt{2}\,\sigma)\big)$, where $\Delta$ is the
utility difference and $\sigma$ the noise on each option, then two
independent answers agree with probability

$$
a = p^2 + (1 - p)^2 = \tfrac12 + \tfrac12 (2p - 1)^2 .
$$ {#eq-cs-photo-agree}

Agreement is never below one half, it reaches one only for pairs whose
difference is large compared with the noise, and for a pair the person finds
nearly equal it is close to a coin flip. With $p = 0.8$, a fairly clear
preference, two answers agree only 68% of the time. So a flipped answer is
not by itself a sign of carelessness. A high rate of flips on pairs the model
considered clear is a sign that the model's noise scale is too small for this
person, and that its posterior is overconfident (@exr-cs-photo-agree).

@eq-cs-photo-agree assumes that the second answer is independent of the
first, and memory can break that in either direction: a remembered answer
can simply be repeated, and the effort to recall it can add noise of its own.
In experiments on repeated beauty ratings, recall memory contributed half of
the measured variance when a single stimulus was rated, but added only about
10% to the variance when the set had nine or more stimuli
[@pombo2023intrinsic]. The figure therefore swaps the sides,
never announces a repeat, and only repeats pairs that are at least two
answers old; a careful study would space the repeats further apart.

Repeating pairs within a session is cheap, yet as of September 2026 we
found no study of preference-based design optimization that did it,
re-asking early comparisons later in the session to test consistency or
drift (@sec-hci-unstable). The closest evidence is indirect. When 17 sighted
participants personalized simulated retinal-implant vision, only
about 50% of their choices agreed with the predictions of a simulated agent
[@schoinas2025evaluating], so a simulated person can be a poor stand-in for a
real one. In the photo task of @ou2023impact, participants with more
photo-editing experience expressed indifference, placing versions in the same
rating region, significantly more often than novices ($p = 0.015$): the
experienced eye declared more ties, not fewer.

### Drift {#sec-cs-photo-drift}

A taste can move during a session: by adapting to vivid versions, so that the
original starts to look dull; by noticing something, such as the color of the
water, that was not a criterion at the start; or by fatigue.
@koyama2020sequential assume that the user's perceptual function "does not
change over time", yet one of their six participants reported selecting
designs "based on criteria that I didn't have at the beginning". The model of
this chapter has no notion of time. It treats an answer given in the first
minute and one given in the tenth as two noisy readings of the same utility.

Turn on *Taste drifts during the session* in @fig-cs-photo-race. The
simulated favorite now moves steadily over the twenty answers, toward more
saturation and warmth and less contrast, by about 0.3 in all. The gap curves
flatten, because the target is moving, and the bottom panel shows where the
sessions ended relative to where the favorite started and where it ended. At
the two lower noise levels they land between the two: projected onto the
drift, the final recommendations sit on average 13% to 56% of the way from
the starting favorite to the final one, depending on the query strategy. At the
highest noise, the EUBO sessions show no movement toward the new favorite at
all. A model that assumes a fixed utility averages over the session, so it
recommends what the person liked on average, not what they like now
(inference, from these simulations).
The remedies, a utility that changes with time or a forgetting factor on old
answers, are discussed in @sec-rec-nonstationary; none has been tested in a
photo task.

The drift in the figure is invented for illustration. We do not know how fast,
or even whether, a viewer's preferred saturation moves in a session of
twenty comparisons, and the experiment that would tell, re-asking early pairs
at the end with the order of questions randomized, is the one described in
@sec-open-decisive.

### Order {#sec-cs-photo-order}

Even with a fixed taste, the questions a person happens to be asked change
where the session ends. The bottom panel of @fig-cs-photo-race shows the
final recommendations of sixteen sessions with the same simulated taste and
the same noise; they differ only in their random candidate pools, their first
random edit, and the noise in individual answers. At the default noise the
final settings scatter with a standard deviation of 0.05 to 0.12 per
adjustment, on the unit scale of @tbl-cs-photo-knobs; 0.07 in exposure is
about 0.14 EV. The scatter is smallest for exposure, the adjustment the
simulated taste weighs most, and among the widest for tint, which it weighs
least. That is the posterior working as intended: where the utility is flat,
the answers cannot pin down a setting, and it costs little to be wrong.

The published evidence on order comes from the crowd. @koyama2017sequentialb
repeated sequential line search three times on the same photograph from
different random starting conditions and measured the differences between
the three runs over the iterations, both as a distance between parameter
settings and as a mean perceptual color difference per pixel: the differences
shrank rapidly in the first four or five iterations, and the three runs
approached similar enhancements. The authors took this as support for their
assumption of a goodness function shared by the crowd, and added that it may
not hold where some people prefer one style and others another. Each
iteration's answer, however, was the median of at least five workers' slider
positions, which averages away much
of any one person's noise. How much a single person's result depends on the
order of their questions has not, as far as we found, been measured in a
photo-enhancement session. Studies of judgment give reasons to expect an
effect. With artistic photographs and faces, a neutral picture was preferred
more after a preferable one than after a less preferable one, and a similar
trend remained when response bias was controlled [@chang2017sequential].
In seven experiments on beauty ratings, sequential dependence did not affect
the variance of the ratings as long as the stimuli were diverse, drawn from
different object categories [@pombo2023intrinsic]. Versions of one
photograph are the opposite of diverse, so the reassurance does not carry
over to this task (inference); @sec-jdm-sequential reviews these sequential
effects.

### Stopping, and whose result it is {#sec-cs-photo-stopping}

A session ends when the person stops, and the published numbers show how
early that is. Sequential Gallery's participants were satisfied after 5.36
planes on average [@koyama2020sequential]. In @ou2023impact, with five photo
adjustments in $[0, 1]$, four rounds of quasi-random versions before the
optimizer took over, and at most 20 iterations, participants with more
photo-editing experience ran significantly more iterations than novices
($p = 0.008$), and the authors summarize the study as finding that novices were
more easily satisfied and finished sooner, while experts explored further
and reported lower satisfaction. Satisfaction is not convergence:
@sec-rec-stopping explains why a person's "good enough" and a model's
flattened posterior are different stopping signals.

The other question a session raises is whose result it is. In the crowd
version, the answer is a population's: the authors note that the individual
user's preference "is not reflected in computation when crowdsourcing is the
only source of data", and that crowds from different backgrounds can have
clearly different preferences [@koyama2020computational]. In the figures of
this chapter, it is one person's, on one screen, on one day, with the noise
and drift that came with them. @sec-hci-population follows the line of work
that borrows strength from other users' sessions to start a new user's
session from a better prior.

::: {.keyidea title="What the case adds to the algorithm"}
The algorithm of @sec-pbo runs unchanged on a real photograph in six
dimensions. What the case adds is everything around it: ranges and units
whose edges are clearly too much but not absurd, fixed kernel settings chosen
by simulation, a minimum visible difference between the two versions, the
original always in view, a few repeated pairs to measure the person's noise,
and a record of the time each answer takes. None of these appears in a test
function, and each one changes the answer.
:::

## Exercises {#sec-cs-photo-exercises}

::: {.exercise #exr-cs-photo-agree}
In a session you repeat five earlier pairs and give the same answer to three
of them. Treat the five pairs as equally difficult. Using
@eq-cs-photo-agree, what value of $p$ does an agreement rate of $3/5$
suggest, and what ratio $\Delta / \sigma$ of utility difference to noise?
What would you conclude if the model had predicted $p = 0.95$ for those
pairs?

::: {.solution}
From $a = \tfrac12 + \tfrac12(2p - 1)^2$ with $a = 0.6$: $(2p - 1)^2 = 0.2$,
so $2p - 1 = 0.447$ and $p \approx 0.72$. Then
$\Delta / (\sqrt{2}\sigma) = \Phi^{-1}(0.72) \approx 0.59$, so
$\Delta / \sigma \approx 0.84$: the pairs differ by less than one noise
standard deviation for this person. A model predicting $p = 0.95$ would
expect agreement $0.95^2 + 0.05^2 = 0.905$; observing 3 of 5 is unlikely
under that prediction (the probability of 3 or fewer agreements out of 5 at
0.905 is about 0.07), so the model's noise scale is probably too small and
its posterior too confident. Five repeats are few, though; the estimate of
$a$ has a standard error of about 0.2.
:::
:::

::: {.exercise #exr-cs-photo-overcount}
The slider figure records one choice as up to four pairwise comparisons.
Suppose all four losers had the same utility, so that the four likelihood
terms are identical, $\Phi(z)^4$ with $z = (g_c - g_\ell)/(\sqrt{2}\sigma)$.
How does this compare with the likelihood of a single comparison, and with
four independent comparisons? Propose one simple correction.

::: {.solution}
$\Phi(z)^4$ is exactly the likelihood of four independent answers that all
favored the chosen point, so the posterior becomes as confident as if the
person had been asked four separate questions and agreed with themselves
every time, although they made one judgment. One correction is to temper the
slider-derived terms, raising each of the $k$ terms to the power $1/k$, which
restores the weight of one answer when the losers are similar. Another is to
use a larger noise scale for slider-derived comparisons. A third is the
choice likelihood of @eq-query-luce, which contributes one term per slider
answer, as Koyama and colleagues do.
:::
:::

::: {.exercise #exr-cs-photo-range}
Suppose the exposure range were $\pm 3$ EV instead of $\pm 1$ EV, with
everything else unchanged, including the lengthscale of 0.4 on the unit
scale. What does a lengthscale of 0.4 now mean in stops, and what happens to
random designs and to the candidate pool?

::: {.solution}
The unit interval now spans 6 EV instead of 2, so a lengthscale of 0.4 means
2.4 EV instead of 0.8 EV: the model assumes the utility changes three times
more slowly in stops, and will smooth over differences a viewer can see.
Most of the exposure range is now absurdly dark or bright, so a random design
is even more likely to be poor, and most of the uniform part of the
candidate pool is wasted on settings nobody would choose. The fixes are a
narrower range, a shorter lengthscale for exposure alone (one lengthscale per
input, @sec-ard), or both.
:::
:::

::: {.exercise #exr-cs-photo-pool}
The pool has 160 uniform settings, 60 perturbations of the best guess, and
the compared settings. How many pairs does EUBO score when the pool has 250
settings? Why does the figure not simply use 220 uniform settings? Compute
the probability that a uniform setting lies within 0.1 of a given setting in
every one of the six coordinates, and compare with the same probability in
one dimension.

::: {.solution}
$250 \cdot 249 / 2 = 31{,}125$ pairs. In one dimension, a uniform point is
within 0.1 of a given point (away from the edges) with probability 0.2; in
six dimensions the probability is $0.2^6 = 6.4 \times 10^{-5}$, so 220
uniform settings would contain a setting that close to the best guess with
probability of about $1.4\%$. Late in a session the useful questions are
small refinements of the best guess, and only the local perturbations
provide them. The same observation motivates the trust regions of
high-dimensional Bayesian optimization (@sec-high-dim-practice): in many
dimensions, good candidates have to be sought near the best point, because
uniform samples almost never land there.
:::
:::

## Further reading {#further-reading .unnumbered}

- @koyama2017sequentialb introduce sequential line search and apply it to
  photo color enhancement with six parameters, answered by crowds.
- @koyama2020computational is a book chapter on computational design with
  crowds; it gives the crowd pipeline, the vote against Photoshop and
  Lightroom, and the three runs from different starting conditions.
- @koyama2020sequential replace the slider with a zoomable grid on a plane
  (Sequential Gallery) and report a small user study on photo enhancement
  with 12 parameters, including the participants' own words.
- @ou2023impact compare novices and experts in three tasks, one of them photo
  color enhancement with an EUBO-based optimizer and a ranking interface.
- @iwai2025constrainedb run pairwise preferential optimization with
  professional designers on color variations of banner ads, 50 comparisons
  per task, with a click-through constraint handled by the machine.
- @mikkola2020projective treat a slider-like answer as infinitely many
  comparisons and derive its likelihood.
