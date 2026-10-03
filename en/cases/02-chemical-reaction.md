---
status: done
synopsis: "Bayesian optimization replayed on a published, fully measured reaction data set: 1,728 combinations of ligand, base, solvent, concentration, and temperature, each with a measured yield. You play the chemists' optimization game on the same data, then watch the optimizer with one-hot or descriptor encodings and different batch sizes, against random selection, 50 recorded chemists, and the paper's own runs."
sources: ["Shields et al. 2021 and its repositories edbo and EvML (MIT License), replayed with tools/figure-data/"]
---

# Optimizing a Chemical Reaction {#sec-cs-chemistry}

@sec-cs-classifier tuned the settings of software, where an evaluation is a
few seconds of computing and the settings are numbers. This chapter moves to a
laboratory. The settings are now which chemicals to use and at what
temperature, an evaluation is a chemical reaction that takes hours to run and
analyze, and several reactions are usually run side by side.

The study at the center of the chapter is @shields2021bayesian, published in
Nature in 2021. Its authors, chemists at Princeton University and
Bristol-Myers Squibb together with computer scientists, ran every one of
1,728 possible versions of one reaction and measured how much product each
gave. Because every outcome is known, the optimization can be replayed: an
"experiment" in this chapter's figures looks up the yield the laboratory
measured. The authors also had 50 chemists and engineers optimize the same
reaction through a game that returned those real results, and they published
the players' choices. Both data sets are in public repositories under the MIT
License [@shields2021edbo; @shields2020evml], which permits redistribution, so
the figures below use the measured data directly.

The chapter first describes the reaction and what it costs to try one
version, then the question this problem raises that the classifier did not:
how a Gaussian process can measure similarity between chemicals. You then
play the chemists' game yourself and watch the optimizer play it, and the last
two sections report what the study found against human experts and what
changes when a robot runs the experiments.

## The problem {#sec-cs-chem-problem}

### A reaction and its yield {#sec-cs-chem-reaction}

A chemical reaction turns starting materials into a product. The reaction in
this study is a **direct arylation**: it attaches a ring of carbon atoms (an
aryl group, here a fluorinated benzene ring) to a small nitrogen-containing
ring called an imidazole, at a position where the imidazole had only a
hydrogen atom. The product is
5-(2-fluorophenyl)-1-methyl-1H-imidazole-4-carbonitrile, and the reaction is
related to a key step in the commercial synthesis of BMS-911543, a drug
candidate that inhibits the enzyme JAK2 [@shields2021bayesian].

The reaction needs help. A palladium **catalyst**, a substance that speeds a
reaction without being used up, does the joining, and a **ligand**, a
molecule that binds to the palladium, shapes how well it works. A **base**
removes the acid the reaction releases, and a **solvent** is the liquid
everything is dissolved in. Two continuous conditions complete the recipe:
the **concentration** of the starting material and the **temperature**. The
outcome is the **yield**, the percentage of the starting material that ends
up as the desired product. A yield of 100% means none was wasted; 0% means
the reaction did not happen.

@tbl-cs-chem-space lists the choices the study allowed. Every combination is
one possible experiment: $12 \times 4 \times 4 \times 3 \times 3 = 1{,}728$.

::: {.table #tbl-cs-chem-space title="The search space of the direct arylation benchmark [@shields2021bayesian]. The 12 ligands were chosen from 70 candidate phosphines by expert judgment."}
| Choice | Options |
|---|---|
| Ligand (12) | BrettPhos, PPhtBu2, tBPh-CPhos, PCy3 HBF4, PPh3, X-Phos, P(fur)3, PPh2Me, GorlosPhos HBF4, JackiePhos, CgMe-PPh, PPhMe2 |
| Base (4) | KOAc, KOPiv, CsOAc, CsOPiv (potassium or cesium acetate or pivalate) |
| Solvent (4) | BuOAc (butyl acetate), p-xylene, BuCN (butyronitrile), DMAc (dimethylacetamide) |
| Concentration (3) | 0.057, 0.1, 0.153 M (moles per liter) |
| Temperature (3) | 90, 105, 120 °C |
:::

### What an experiment costs {#sec-cs-chem-cost}

A version of the reaction is tried by mixing the chemicals in a vial, heating
it, and measuring how much product formed. The authors write that "many
reactions take hours or days to run to completion", which is why experiments
are run in parallel batches rather than one at a time
[@shields2021bayesian]. The game described below gave its players bench room
for five experiments per working day and a "month" of 20 days.

To build the benchmark, the authors used **high-throughput experimentation**
(HTE), in which many miniaturized reactions are run in parallel, to run all
1,728 combinations once. Each reaction was measured
"without replication" [@shields2021bayesian], so each combination has exactly
one measured yield and the data say nothing about how much a repeat would
differ. This is the opposite of the classifier chapter, where every setting
had five measured repeats; here the replay is deterministic, and the
measurement noise is real but unseen.

### The landscape {#sec-cs-chem-landscape}

Most of the 1,728 reactions do poorly. The median yield is 8.1% and the mean
19.4%; 494 reactions (29%) gave no product at all. Only 67 reached 80%, 18
reached 90%, and 5 reached 99%. All five of those use the same ligand,
CgMe-PPh, and the two that reached 100% are CgMe-PPh with cesium acetate or
cesium pivalate in DMAc at 0.153 M and 105 °C (computed from the published
data; tools/figure-data/cs-chem-arylation.py prints these summaries).

The ligand matters most. Averaged over everything else, X-Phos (52.8%) and
CgMe-PPh (50.2%) give the highest yields, and three small phosphines (PPhMe2,
PPhtBu2, PPh2Me) give almost nothing (0.3%, 0.4%, 2.0%). Measured the way
@sec-cs-classifier-importance measured hyperparameters, as the share of the
variance of the yield explained by one choice alone, the ligand explains 48%,
the solvent 8%, the temperature 3%, the base 1%, and the concentration 0.2%.
These main effects sum to 60%; the other 40% comes from choices acting
together, such as a ligand that works only in some solvents.

### Choices made before the first experiment {#sec-cs-chem-choices}

"Reaction optimization truly begins by defining the search space," the
authors write [@shields2021bayesian], and three decisions were made before any
optimizer or player saw the problem. First, people chose the 12 ligands out of
70 candidates; the paper is explicit that this selection "was based on
valuable expert knowledge rather than machine learning". Second, the two
continuous conditions were reduced to three levels each, which made the space
finite and the exhaustive measurement possible. Third, the objective is yield
alone: nothing in the data says what a reagent costs, how safe a solvent is,
or how pure the product came out, all of which a process chemist weighs
(inference). Every result below is about this space, as defined.

## Encoding reagents {#sec-cs-chem-encoding}

A Gaussian process predicts a reaction's yield from the yields of reactions
it considers similar, and its kernel defines similar through a distance
between inputs (@sec-kernels). Temperature has a natural distance: 105 °C is
between 90 and 120. A ligand does not. "BrettPhos minus X-Phos" has no
meaning, and numbering the ligands 1 to 12 would make the model believe that
ligand 3 lies between ligands 2 and 4. Before the loop can start, each
reagent needs to become a vector. The study compared three ways, and the two
extremes show what is at stake.

### One-hot encoding {#sec-cs-chem-onehot}

The simplest encoding gives each option its own coordinate. A ligand becomes
a vector of 12 numbers, all zero except a single one at that ligand's
position: BrettPhos is $(1, 0, \dots, 0)$, PPhtBu2 is $(0, 1, 0, \dots, 0)$,
and so on. This is a **one-hot encoding**. A reaction is the concatenation
of its ligand, base, and solvent vectors with its concentration and
temperature levels.

::: {.derivation title="What a kernel over one-hot vectors assumes"}
Let $\mathbf{e}_a$ and $\mathbf{e}_b$ be the one-hot vectors of ligands $a$ and $b$.

1. If $a \ne b$, the two vectors differ in exactly two coordinates, each by
   1, so $\lVert \mathbf{e}_a - \mathbf{e}_b \rVert^2 = 2$. If $a = b$, the distance is 0.
2. For two reactions that differ only in their ligand, the squared distance
   is therefore $2 / \ell_{\text{lig}}^2$ for every pair of different
   ligands, with $\ell_{\text{lig}}$ the ligand's lengthscale (one per
   choice, as in @sec-ard).
3. A stationary kernel depends on the inputs only through this distance
   (@sec-kernel-family), so the prior correlation between such reactions is
   one number, the same for BrettPhos and X-Phos as for BrettPhos and PPh3.
:::

A one-hot model can therefore learn *that* the ligand matters, by shrinking
$\ell_{\text{lig}}$, but not *which ligands resemble each other*. Measuring
BrettPhos teaches it the same about every other ligand. This is a reasonable
assumption when nothing is known about the options, and a wasteful one when
chemistry knows a great deal.

### Descriptor encoding {#sec-cs-chem-descriptors}

The alternative describes each molecule by computed properties, called
**descriptors**. @shields2021bayesian computed theirs with density functional
theory (DFT), a quantum-mechanical calculation of a molecule's electrons, and
recorded quantities such as the energies of the outermost electron orbitals,
the dipole moment (how unevenly charge is spread), and the charges on
individual atoms. For the ligands that came to 1,358 numbers per molecule that
vary between ligands, 231 for the bases, and 116 for the solvents. With
descriptors, two ligands whose electronic and steric properties are close are
close to the kernel, and a yield measured with one says something about the
other. The assumption has changed from "all ligands are equally different" to
"yield varies smoothly with these properties", which is plausible and not
guaranteed.

A thousand coordinates per ligand is more than a Gaussian process needs, and
here there is an exact shortcut. Twelve points always lie in a space of at
most 11 dimensions, so after standardizing the descriptors and rotating them
onto their **principal components** (the directions of largest spread, found
by a singular value decomposition), 11 coordinates describe the 12 ligands
with no loss; likewise 3 coordinates for the 4 bases and 3 for the 4
solvents. The replays below use these rotated descriptors, which preserve
every distance between molecules up to one scale factor per choice.

### What the study found {#sec-cs-chem-encoding-evidence}

The paper tuned its optimizer on six other reactions with published data, a
Suzuki-Miyaura coupling and five Buchwald-Hartwig couplings, and compared
DFT descriptors, descriptors from the open-source cheminformatics library
Mordred, and one-hot encodings [@shields2021bayesian]. The average **loss**,
the shortfall between the best yield in the data set and the best yield the
optimizer found, was "largely indistinguishable" across the three encodings
($p > 0.05$ by Welch's $t$-test, which compares two means without assuming
equal variances). The difference was in the worst case: over many runs from
different random starts, the largest loss was at most 5% of yield for every
reaction with DFT descriptors, against at most 15% for Mordred and 8% for
one-hot. The authors kept DFT descriptors and noted that "acceptable
performance can be achieved in the wild with a number of reaction encodings."

::: {.keyidea title="An encoding is a prior"}
The encoding decides which reactions the model treats as similar before it
has measured anything. One-hot says the options are unrelated; descriptors
say they resemble each other as their computed properties do. Neither is
learned from the yields, so the choice is as much a modeling assumption as
the kernel.
:::

## Replaying the optimization {#sec-cs-chem-replay}

### Your turn {#sec-cs-chem-your-turn}

Before the optimizer, try the problem as the 50 players did. The figure below
is the game with the paper's rules, except that the budget is 50 experiments
instead of 100: by 50 experiments, 44 of the 50 players had already stopped.
Choose up to five reactions, run the batch, read the yields, and plan the next
batch.

```{figure}
//| figure: cs-chem-game
//| label: fig-cs-chem-game
//| fig-cap: "The reaction optimization game of @shields2021bayesian on its real data. The map shows all 1,728 reactions: each band of four rows is one ligand (its four bases), and each block of nine columns is one solvent (three temperatures, each at three concentrations). Click cells to queue a batch of up to five, or set the five conditions and press *Add*, then *Run batch*; each experiment reveals the yield the laboratory measured, and nothing is simulated. Right: your best yield so far (magenta) against the 50 players of the original game (thin lines; the thick neutral line is their mean, where a player who stopped keeps their last best) and the mean of the paper's 50 optimizer runs (dashed). The players' choices are from the EvML repository [@shields2020evml]."
```

Two things are worth noticing as you play. The first batch is the hardest,
because nothing on the map is known; the players used their chemistry there,
and @sec-cs-chem-experts shows that it helped. Later batches are a trade
between confirming a promising region and testing a ligand you have not yet
tried. If you finish without finding a yield above 99%, turn on *Reveal all
yields* and look at the CgMe-PPh band.

### The optimizer {#sec-cs-chem-optimizer}

The optimizer in the figures re-implements the method of the paper in
simplified form; it is not the paper's code. Its surrogate is a Gaussian
process on the standardized yields with a Matérn 5/2 kernel over the
encoding, with one lengthscale per choice (ligand, base, solvent,
concentration, temperature). The paper fitted its lengthscales by maximizing
the marginal likelihood with gamma priors that "assume that most of the
dimensions are irrelevant" by favoring long lengthscales
[@shields2021bayesian]. Ours picks each lengthscale from a short list of
values and the noise variance from another, by the marginal likelihood plus a
weak preference for long lengthscales (@sec-marginal-likelihood). The lists
and the center of that preference were settled by trying a few on this same
data set, a small amount of tuning on the test problem, which the paper
avoided by tuning on other reactions. The
acquisition function is expected improvement (@sec-ei) with the paper's
exploration offset of 0.01, maximized over every reaction not yet run.

The new element is the batch. Expected improvement scores one experiment, but
the bench runs five at once, and the five best-scoring reactions are usually
near-copies of each other. The paper used the **kriging believer**
[@ginsbourger2010kriging], which picks a batch one reaction at a time and
pretends, after each pick, that the model's prediction there has already been
measured.

::: {.algorithm #alg-cs-chem-believer title="Kriging believer batch selection"}
Input: a fitted Gaussian process, the set $R$ of reactions not yet run, a
batch size $q$.

1. For $j = 1, \dots, q$:
   1. Choose $\vx_j \leftarrow \argmax_{\vx \in R} \EI(\vx)$ under the current
      model.
   2. Remove $\vx_j$ from $R$.
   3. Add the pseudo-observation $(\vx_j, \mu(\vx_j))$, the model's own mean
      prediction, to the data, and update the posterior with the same kernel
      hyperparameters.
2. Run the $q$ reactions and replace the pseudo-observations with the
   measured yields.
:::

Step 1.3 leaves the posterior mean where it was but collapses the
uncertainty at $\vx_j$ and shrinks it nearby, so the expected improvement of
reactions similar to $\vx_j$ drops and the next pick moves elsewhere
(@exr-practice-believer shows why; @exr-cs-chem-believer applies it to the
reaction data). @sec-batch-bo treats batch acquisition more
generally. The paper reported that batches of five did as well on average as
one experiment at a time with a budget of 50 ($p > 0.05$), on its six
development reactions [@shields2021bayesian].

```{figure}
//| figure: cs-chem-replay
//| label: fig-cs-chem-replay
//| fig-cap: "Bayesian optimization replayed on the 1,728 measured reactions. Left: the reactions this run has tried, colored by their measured yield; rings mark the latest batch, listed below with the model's weight on each choice (the inverse of its fitted lengthscale; a taller bar means the yield changes faster with that choice). Right: best yield so far for this run (orange), for random selection with the same seed (violet), for the 50 recorded chemists (thin lines; the thick neutral line is their mean), and for the paper's own 50 runs (dashed, mean). Change the encoding, the batch size, and whether the first batch is random or a recorded chemist's first batch; *New run* draws another random start. Every yield is measured; the optimizer's choices are computed in your browser by a simplified re-implementation (lib/cs-chem.ts), not the paper's code."
```

Things to try:

**Step through the batches.** In the first batches the weight bars are all
short: with a dozen yields, most of them near zero, the model cannot tell
which choice matters and keeps the long lengthscales its prior prefers. By
about the fourth batch the ligand bar usually stands out. Over 50 runs the
median fitted lengthscale after 20 experiments was 1 for the ligand, 2 for the
solvent, and 4, the value its prior favors, for base, concentration, and
temperature
(tools/figure-data/cs-chem-race.ts). That ordering matches the main effects
of @sec-cs-chem-landscape, learned from 20 of 1,728 yields.

**Switch to one-hot.** The early batches change, because the model no longer
believes that similar ligands give similar yields. Watch whether the run still
finds the CgMe-PPh band, and how soon.

**Use a chemist's first batch.** The first five experiments become those of
one of the recorded players (a different player for each run). On average
the orange curve then starts higher, at 64.8% against 50.0% after five
experiments, but not in every run: the run the figure opens with drew a lucky
random batch (76.7%) and a player whose first batch reached only 20.6%. Press
*New run* a few times, and watch what happens to the curve after the first
batch.

**Reveal all yields** to see where the run looked and what it missed.

### Fifty runs of each {#sec-cs-chem-results}

One run is one draw of luck. @tbl-cs-chem-race collects 50 runs of each
variant, alongside the paper's own runs and the recorded chemists.

::: {.table #tbl-cs-chem-race title="Mean best yield found after 10, 20, 30, and 50 experiments on the measured direct arylation data, over 50 runs (players: the 50 recorded chemists, where a player who stopped keeps their last best). Last column: runs that found a yield of at least 99% within their first 50 experiments and, in parentheses, the median number of experiments those runs needed; the chemists and the paper's runs went on for up to 100 experiments, and what they found after the 50th is not counted. The paper's runs are from its EvML repository; our runs, random selection, and the chemists' curves are computed by tools/figure-data/cs-chem-race.ts. Batches are of five unless noted."}
| Strategy | 10 | 20 | 30 | 50 | ≥ 99% (median) |
|---|---|---|---|---|---|
| Random selection | 65.1% | 75.5% | 81.8% | 88.8% | 4 of 50 (40) |
| The 50 chemists | 79.8% | 88.4% | 92.9% | 94.2% | 24 of 50 (22) |
| Paper's optimizer, random start | 74.1% | 91.6% | 97.5% | 99.8% | 46 of 50 (19.5) |
| Paper's optimizer, chemist's start | 74.0% | 91.5% | 98.9% | 99.9% | 50 of 50 (22.5) |
| Ours, descriptors | 64.0% | 82.7% | 97.1% | 99.9% | 50 of 50 (26) |
| Ours, one-hot | 66.3% | 85.4% | 95.2% | 99.6% | 48 of 50 (27) |
| Ours, descriptors, batch of 1 | 54.8% | 87.4% | 99.3% | 100.0% | 50 of 50 (22) |
| Ours, descriptors, batch of 10 | 65.1% | 82.5% | 91.5% | 98.5% | 44 of 50 (31) |
| Ours, descriptors, chemist's start | 67.4% | 79.3% | 93.1% | 99.9% | 49 of 50 (31) |
:::

Five readings of the table follow.

**Random selection rarely finds the best.** Five of 1,728 reactions reach
99%, and a random order of experiments finds one of them in its first 50 only
4 times in 50. On average it would need about 288 (@exr-cs-chem-random). Every
version of Bayesian optimization found one in at least 44 of 50 runs.

**Our simplified optimizer is slower early and catches up.** After 10 and 20
experiments the paper's optimizer is ahead of ours by 9 to 10 points of
yield; by 30 experiments the two are within half a point. The paper tuned its
priors on six other reactions, and its runs encode the same chemistry with
the full set of descriptors after removing highly correlated ones
[@shields2021bayesian]; ours uses a coarse search over lengthscales. The
end result is not sensitive to those details, and the early phase is
(inference).

**One-hot is not much worse here.** Over 50 runs, one-hot was ahead of
descriptors after 10 and 20 experiments and behind at 30 and 50, and it
missed a yield of 99% in 2 of 50 runs where descriptors missed none. That
matches the paper's finding: similar averages, a better worst case with
descriptors. On this data set the ligand dominates, and a model that learns
"the ligand matters, and CgMe-PPh is good" needs no notion of which ligands
are alike once it has tried CgMe-PPh (inference).

**Batch size trades experiments for days.** Counted in experiments, smaller
batches are more efficient: one at a time reached 99% in a median of 22
experiments, batches of five in 26, batches of ten in 31, and batches of ten
missed it in 6 of 50 runs. Counted in rounds, the order reverses: a median of
22 rounds one at a time, 6 rounds in batches of five, and 4 in batches of
ten. When a round is a day at the bench, larger batches finish sooner at the
price of more reactions.

**A chemist's first batch helps the first batch only.** Starting from the
players' own first five experiments raised the best yield after five
experiments from 50.0% to 64.8%, for the paper's optimizer and ours alike. It
did not carry over. The paper's optimizer was at 91.5% after 20 experiments
from either start, and ours did worse from the chemists' start than from a
random one (79.3% against 82.7% after 20, and a median of 31 experiments to
99% against 26). The authors' own analysis compares the two starts batch by
batch with Welch's $t$-test: the difference is significant in the first
batch ($p = 0.004$) and not in the next five ($p$ between 0.13 and 0.98)
[@shields2020evml]. One
plausible reason is diversity: 7 of the 50 players tested a single ligand in
their whole first batch, and on average a player's first batch covered 3.8
ligands, against 4.2 for five random reactions. A start that is good but
narrow teaches the model less about the other ligands than a worse but more
varied one (inference; compare @sec-initial-design).

## Against human experts {#sec-cs-chem-experts}

The comparison with chemists is the part of @shields2021bayesian that drew
the most attention, and it rests on a carefully built game. The players
received the reaction, the 12 ligands, 4 bases, 4 solvents, and the levels of
concentration and temperature, and had "one month" to find the best
conditions, running one batch of five experiments "per workday". The game
allowed up to 20 batches, 100 experiments in all, "around 6% of the
experimental space". "Although the game was intended to simulate reaction
optimization on a fixed experimental budget, the data were real": each
experiment returned the measured yield from the high-throughput data. Fifty
"expert chemists and engineers from academia and industry" played, and the
optimizer played the same game 50 times from different random starts
[@shields2021bayesian]. According to the published records, 30 players came
from the pharmaceutical industry, 19 from academia, and 1 from elsewhere;
by job title there were 15 process chemists, 12 graduate students, 11
engineers, 6 postdoctoral researchers, 3 faculty members, and 3 medicinal
chemists [@shields2020evml].

The paper reports four findings [@shields2021bayesian].

1. **The chemists started better.** "Humans made significantly ($p < 0.05$)
   better initial choices than random selection, on average discovering
   conditions that had 15% higher yield in their first batch of experiments."
   In the published records, the mean best yield after the first batch was
   64.8% for the players and 50.0% for the optimizer's random starts.
2. **The optimizer overtook them within three batches.** In the authors'
   words, "even with random initialization, within three batches of five
   experiments the average performance of the optimizer surpassed that of the
   humans."
3. **The optimizer was more consistent.** It "achieved >99% yield 100% of
   the time within the experimental budget"; in the records, 28 of the 50
   players found a yield of at least 99% before they stopped.
4. **The optimizer found a ligand the experts did not expect.** The best
   conditions use CgMe-PPh, which, "to the best of our knowledge", had "not
   been used as a ligand for direct arylation of imidazoles. Thus, experienced
   chemists tended to not investigate this ligand initially." In the records,
   3 of the 50 players began with it and 24 began with BrettPhos; 46 of 50
   tried CgMe-PPh eventually, half of them by their 10th experiment.

A complication is that players could stop when they believed they had found
the optimum, and most did: the median player ran 25 experiments, and only one
ran all 100 [@shields2020evml]. Comparing averages after the 15th experiment
means comparing the optimizer with the players who were still playing. The
authors therefore bounded the human average. If players who stopped would
have found nothing better (the lower bound), the human average stays close to
the raw one, and for both the raw data and this bound a Welch $t$-test at
each batch finds the optimizer better on average after the fifth batch. If every player who stopped
would have reached 100% in the very next batch, an upper bound the authors
call "unrealistic", the human average "closely follows" the optimizer and the
difference is not significant [@shields2021bayesian].

The comparison has limits worth naming, and they are the kind every
human-versus-machine study has (inference). It is one reaction, in a space
whose ligands were chosen by experts, so expert knowledge entered before the
game began. The players had no material costs, no failed analyses, and no
competing projects. And "better on average" hides a spread: two players
found a yield above 99% within their first two batches, and the optimizer's
own runs needed between 8 and 70 experiments to do so [@shields2020evml].

The paper then applied the method to two reactions too large to measure
exhaustively [@shields2021bayesian]. For a Mitsunobu reaction, which couples
an alcohol to another molecule, with 180,000 possible configurations, the
standard conditions used at Bristol-Myers Squibb gave an average yield of
60% (59% and 60% in two replicates); Bayesian optimization with batches of
ten found "three distinct sets of reaction conditions" giving 99% "in only
four rounds of ten experiments". For a deoxyfluorination, which replaces an
alcohol group with a fluorine atom, with 312,500 configurations, standard
conditions gave 36% (35% and 36%); the optimizer, in batches of five,
surpassed that within three rounds and reached 69% in ten. For the Mitsunobu
reaction the authors note that the best conditions lay "in areas of reaction
space that would not typically be searched", and in both reactions they were
largely distinct from the standard ones.

::: {.keyidea title="Where the expert helped"}
In this study the chemists' knowledge paid off in the first batch and in the
choice of the search space, and the optimizer's bookkeeping paid off after
that. The most useful human contribution was the one made before the game
started.
:::

## From one reaction to a self-driving lab {#sec-cs-chem-sdl}

A replay hides the logistics that a real campaign cannot. When the
experiments are run by robots and the optimizer chooses the next plate
without waiting for a person, the setup is called a **self-driving lab**, and
Bayesian optimization is its usual decision maker (@sec-adj-sdl). Three
features of the arylation study become central there.

**Batches are set by the hardware.** A plate has a fixed number of wells, and
an analysis instrument processes one plate at a time, so the batch size is
chosen by the equipment, not by the optimizer. The trade in
@sec-cs-chem-results, more reactions in exchange for fewer rounds, is then
decided by what a round costs (@sec-batch-bo).

**Constraints and multiple objectives arrive with real chemistry.** Some
combinations precipitate, some are unsafe at high temperature, and yield is
rarely the only goal: cost, purity, and the amount of waste matter as well.
These become constraints on which experiments may be run and additional
objectives to trade off (@sec-constrained-bo; @sec-multi-objective). The
benchmark data set has none of them, which is part of why it is a clean test
and a simplified one.

**Gains are measured against a reference.** The **acceleration factor** of
@adesiji2026benchmarking (@sec-app-science) is the number of experiments a
reference strategy needs to reach a target divided by the number the
optimizer needs; across the studies they surveyed, its median was 6. On the
arylation data the same calculation can be done exactly
(@exr-cs-chem-random): random selection needs about 288 experiments on
average to find a yield of 99%, and the paper's optimizer needed a median of
20 over its 50 runs of up to 100 experiments, an acceleration factor of about
14 (inference, from the published runs).
For a target of 90%, which 18 reactions reach, random selection needs about
91 and the optimizer a median of 17.5, a factor of about 5.

People do not leave the loop in these labs; their role changes. In automated
microscopy, @kalinin2023human, in a 2023 preprint later published in *Microscopy Today*, argue that "the likely
strategy for the next several years will be human-in-the-loop automated
experiments", with a person monitoring progress and adjusting the agent's
policy, and @pratiush2024building, a 2024 preprint later published in *Digital Discovery*, found that the experiment
path can be "trapped in the local minima" for some settings, which is what
the monitoring is for. When the quality of a result is subjective, the
person's judgment becomes the measurement itself: @deneault2025preferential
tuned a 3-D printer by preferential Bayesian optimization, with human
judgment of the prints as the only measurement, and report that it improved
the efficiency of printing objects with such subjective qualities. That is the setting of @sec-part-preferences, where the
person is the objective rather than its supervisor.

Language models have entered the same loop with mixed results. As
autonomous optimizers they have done poorly in controlled tests:
@gupta2025llms found that replacing the true experimental outcomes with
randomly permuted labels had no effect on a language-model agent's
performance on gene-perturbation and molecular-property tasks, while classical
methods such as Gaussian-process optimization consistently did better. As a
source of encodings, the subject of @sec-cs-chem-encoding, they have helped:
GOLLuM trains language-model embeddings of reactions jointly with a Gaussian
process through the marginal likelihood, and across 23 chemistry and
materials tasks reached a top-5% coverage of 36.3% after 50 experiments,
against 26.5% for fixed embeddings with a Gaussian process, matching
conventional Bayesian optimization with 41% fewer experiments
[@rankovic2026large]. In both cases the Gaussian process keeps the
uncertainty and makes the decision.

::: {.keyidea title="What the case adds to the method"}
Nothing in the optimizer was specific to chemistry except the encoding.
Everything around it was: which reagents to allow, how many experiments fit
on the bench, what the experts already knew, and what "best" means beyond
yield.
:::

The next chapter, @sec-cs-exoskeleton, keeps a person in the loop in a
different way: the objective is measured on a person's body, and the person
changes while the optimizer is learning.

## Exercises {#sec-cs-chem-exercises}

::: {.exercise #exr-cs-chem-random}
Experiments are run in a uniformly random order, without repetition, from $N$
reactions of which $k$ are "good". Show that the expected number of
experiments up to and including the first good one is $(N + 1)/(k + 1)$.
Evaluate it for the arylation data with targets of 99% ($k = 5$) and 90%
($k = 18$), and compare with the paper's optimizer, which needed a median of
20 and 17.5 experiments.

::: {.solution}
The $k$ good reactions split the $N - k$ others into $k + 1$ gaps (before the
first good one, between consecutive good ones, and after the last). In a
random order every bad reaction is equally likely to fall in any gap, so each
gap holds $(N - k)/(k + 1)$ bad reactions on average. The first good one is
found after the bad reactions in the first gap plus itself:
$(N - k)/(k + 1) + 1 = (N + 1)/(k + 1)$. With $N = 1{,}728$: $1{,}729 / 6
\approx 288$ for $k = 5$ and $1{,}729 / 19 = 91$ for $k = 18$. Against the
optimizer's medians, the acceleration factors are about $288 / 20 \approx 14$
and $91 / 17.5 \approx 5$. The rarer the target, the more a model helps,
because random search pays for every reaction it wastes.
:::
:::

::: {.exercise #exr-cs-chem-onehot}
Using the derivation in @sec-cs-chem-onehot, suppose a one-hot model has
measured CgMe-PPh in one base, solvent, concentration, and temperature, and
found a high yield. Compare what its posterior mean says about (a) CgMe-PPh
with a different base, and (b) X-Phos with the same base. Which lengthscales
decide the answers? What would change with descriptors?

::: {.solution}
Reaction (a) differs from the measured one only in the base, so its
correlation with it is set by $\ell_{\text{base}}$; with a long base
lengthscale it is high, and the posterior mean of (a) is pulled up toward the
measured yield. Reaction (b) differs only in the ligand, so its correlation
is set by $\ell_{\text{lig}}$ through the fixed squared distance
$2 / \ell_{\text{lig}}^2$; if the model has learned a short ligand
lengthscale, the correlation is low and (b) is barely pulled up, and the same
is true of every other ligand alike. With descriptors the pull on (b) depends
on how close X-Phos is to CgMe-PPh in descriptor space: strong if their
computed properties are similar, weak if not, and different for each ligand.
:::
:::

::: {.exercise #exr-cs-chem-believer}
@exr-practice-believer showed that a kriging believer's pseudo-observation at
$\vx_1$ leaves the posterior mean unchanged and sets the posterior variance at
$\vx_1$ to zero. Apply this to the reaction data. The first pick of a batch is
CgMe-PPh with one base, solvent, concentration, and temperature. Under the
one-hot encoding of @sec-cs-chem-onehot, which untried reactions lose the most
expected improvement after the pseudo-observation, and which lose almost
none? What changes with descriptors, and what does that mean for how varied a
batch of five is under each encoding?

::: {.solution}
The variance at $\vx$ falls by
$\Cov[f(\vx), f(\vx_1)]^2 / \Var[f(\vx_1)]$, so a reaction loses expected
improvement in proportion to how strongly the kernel ties it to the pick.
Under one-hot, that correlation depends only on which of the five choices
differ and on their lengthscales. Reactions that share the ligand and differ
in a choice with a long lengthscale (base, concentration, or temperature,
which the fits of @sec-cs-chem-replay left long) are almost copies of the
pick and lose nearly all their expected improvement. Reactions with any other
ligand are all at the same squared distance $2/\ell_{\text{lig}}^2$ along the
ligand, so with a short ligand lengthscale they lose almost none, and all
eleven other ligands are treated alike. The next pick therefore changes the
ligand, but the model has no reason to prefer one new ligand over another
beyond their posterior means. With descriptors, ligands whose computed
properties are close to those of CgMe-PPh lose expected improvement too, and
distant ones keep it, so the believer spreads a batch over chemically
different ligands and not merely over different labels.
:::
:::

## Further reading {#further-reading .unnumbered}

- @shields2021bayesian is the study this chapter replays: the benchmark, the
  encodings, the batch method, the game against 50 chemists, and the two
  applications. Its Methods section gives the surrogate and acquisition
  settings.
- @shields2021edbo is the authors' software and the source of the 1,728
  measured yields and descriptors; @shields2020evml holds the game's records
  and the authors' statistical analysis. Both are MIT-licensed. The script
  tools/figure-data/cs-chem-arylation.py downloads them at pinned commits and
  writes the compact copy the figures use.
- @ginsbourger2010kriging introduced the kriging believer and related
  heuristics for choosing batches of points for a Gaussian process.
- @adesiji2026benchmarking defines acceleration and enhancement factors and
  collects them across self-driving lab studies.
- @rankovic2026large shows how learned language-model embeddings can serve as
  a reaction encoding inside a Gaussian process.
- @sec-adj-sdl and @sec-sci-chemistry review self-driving labs and the
  studies in which chemists and other experts take part in the optimization
  loop.
