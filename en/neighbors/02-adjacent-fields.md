---
status: done
synopsis: "Seven computing fields that learn from human judgments, and what they solved first: reward learning from trajectory comparisons, the AI safety analysis of systems that influence the people they measure, decision-theoretic preference elicitation, recommender feedback loops, interactive evolution, learning to rank, and self-driving labs. With an interactive look at position bias and a table of results preferential optimization can adopt."
---

# Reward Learning, Recommendation, Ranking, and Automated Science {#sec-adjacent}

Preferential Bayesian optimization (PBO) is one of many systems in computing
that learn from what people say they prefer. @sec-llms followed the largest
neighbor, the alignment of language models. This chapter visits seven more,
each of which met the problems of preferential optimization in its own form
and often earlier: reinforcement learning from comparisons of trajectories,
AI safety, preference elicitation in AI and decision analysis, recommender
systems, interactive evolutionary computation, learning to rank, and
self-driving laboratories.

Two conclusions run through the chapter. First, several things that the PBO
literature still treats as open have been solved, or have a ready practice,
next door: choosing queries by their effect on the final decision, monotonicity
and ranking queries in a Gaussian process model, simulated users with realistic
irrationalities, correction for the position in which an option is shown, warm
starts from population embeddings, calibration metrics, and a reporting
standard for speedups; @tbl-adj-solved collects them. Second, the AI safety
literature has formalized the idea that a system can change the person it
measures, and has found evidence of it in recommender systems and language
models. That is the book's claim that a PBO session is an intervention as well
as an estimate (@sec-syn-bottleneck), worked out by a neighbor; no study of PBO
has yet measured it, so each transfer of the concern below is marked as
inference.

## Preference-based reinforcement learning {#sec-adj-pbrl}

**Reinforcement learning** trains an agent, a policy that chooses actions, to
maximize the total reward it collects over a sequence of steps, the
**return**. Writing a reward function by hand is hard for tasks such as a
backflip or a helpful answer. **Preference-based reinforcement learning**
replaces it with human comparisons of short clips of the agent's behavior,
**trajectory segments**, and learns a reward model from the answers with the
Bradley-Terry likelihood of @sec-llm-reward-models; RLHF (@sec-llm-rlhf) is
this method applied to language models. @christiano2017deep (NeurIPS 2017)
showed it working at scale, giving feedback "on less than one percent of our
agent's interactions with the environment" and teaching new behaviors with
"about an hour of human time".

### Practice and benchmarks {#sec-adj-pbrl-practice}

PEBBLE [@lee2021pebble] (ICML 2021) relabels all past experience whenever the
reward model changes and pretrains the agent without rewards. B-Pref
[@lee2021b] (NeurIPS 2021 Datasets and Benchmarks) argued that "simulating
human input as giving perfect preferences for the ground truth reward function
is unrealistic", and built simulated teachers that each add one irrationality
to a perfect one: random choice with rationality $\beta = 1$ (the
Bradley-Terry model with unit scale), mistakes with probability 0.1, skipping
pairs that are hard to compare, declaring a tie when the returns are close,
and myopia, discounting earlier steps in a segment with $\gamma = 0.9$.
Uni-RLHF [@yuan2024uni] (ICLR 2024) collected crowdsourced annotations covering
"more than 15 million steps across 30+ popular tasks", with "competitive
performance compared to those from well-designed manual rewards".

### Theory {#sec-adj-pbrl-theory}

Dueling posterior sampling [@novoseller2020dueling] (UAI 2020) gave the first
regret guarantee for preference-based reinforcement learning, an asymptotic
Bayesian no-regret rate; @saha2023dueling (AISTATS 2023) proved near-optimal
regret for trajectory preferences under generalized linear models; and
@wang2023rlhf (NeurIPS 2023) proved that "for a wide range of preference
models, we can solve preference-based RL directly using existing algorithms and
techniques for reward-based RL, with small or no extra costs".

### Models of how people compare {#sec-adj-pbrl-models}

The model of how a person turns a reward into an answer matters more than
expected. @knox2024models (TMLR 2024) compared the usual model, in which a
person prefers the segment with the larger sum of rewards, its *partial
return*, with one in which the person judges each segment by its *regret*, how
much worse it is than the best the agent could have done from the same start.
The regret model was **identifiable** (given enough answers, only one reward
function is consistent with them), the partial-return model lacked
identifiability in several settings, and the regret model predicted real human
preferences better. How wrong can the inferred reward be if the model of the
person is slightly wrong? @hong2023sensitivity (ICLR 2023) showed that "it is
unfortunately possible to construct small adversarial biases in behavior that
lead to arbitrarily large errors in the inferred reward", and identified
"reasonable assumptions under which the reward inference error can be bounded
linearly in the error in" the human model. And @hatgiskessell2025influencing
(TMLR) turned the question around, helping people fit the model by showing
them the latent quantity it assumes, training them to follow it, or rewording
the question, in three studies with people: "All intervention types show
significant effects."

### Choosing queries and using richer feedback {#sec-adj-pbrl-queries}

Information-directed reward learning [@lindner2021information] (NeurIPS 2021)
selects "queries that maximize the information gain about the difference in
return between plausibly optimal policies", rather than about the reward
everywhere, and needs significantly fewer queries. @hu2024query (ICLR 2024)
named the failure this avoids **query-policy misalignment**: queries chosen to
improve the reward model overall "may not align with RL agents' interests, thus
offering little help on policy learning". RIME [@cheng2024rime] (ICML 2024)
filters noisy preferences; LiRE [@choi2024listwise] (ICML 2024) builds ranked
lists of trajectories from better, worse, and equal answers to exploit how
strong each preference is; and inverse preference learning
[@hejna2023inverse] (NeurIPS 2023) and contrastive preference learning
[@hejna2024contrastive] (ICLR 2024) skip the explicit reward model, both
assuming that preferences follow regret.

### Over-optimization {#sec-adj-pbrl-overopt}

When a policy is optimized hard against a learned reward, the true reward
eventually falls, an instance of Goodhart's law ("when a measure becomes a
target, it ceases to be a good measure"). @gao2022scaling (ICML 2023, in the
PMLR proceedings; the arXiv record lists no venue) measured this with a fixed
"gold-standard" reward model standing in for people: the relationship between
optimization and gold reward "follows a different functional form depending on the method of optimization",
and its coefficients scale smoothly with reward model size.
@rafailov2024scaling (NeurIPS 2024) found that DPO and its relatives
over-optimize too, "often before even a single epoch of the dataset is
completed". @casper2023open (TMLR 2023) survey the open problems and
fundamental limits of RLHF.

### What it means for PBO {#sec-adj-pbrl-meaning}

Three conclusions carry over as stated: the choice of preference model has
real consequences (Knox et al.), which in PBO is the choice between probit and
logistic links (@sec-obs-baseline); Hatgis-Kessell et al. recommend
"designing interfaces and training interventions to increase human conformance
with the modeling assumptions of the algorithm", which in PBO is how the question is worded and shown
(@sec-query-design); and a simulated teacher with perfect preferences is
unrealistic (B-Pref).

The rest is inference. By Hong et al., small systematic response biases, such
as an effect of position, can make a PBO posterior badly wrong even after many
comparisons, whereas random noise is the benign case (@sec-adj-ltr). B-Pref's
teachers map almost one to one onto simulated users for PBO (skipping to
incomparability, ties to indifference, @sec-ties, myopia to recency effects),
so PBO benchmarks, which mostly simulate homoscedastic, independent noise (the
same noise level for every pair, drawn afresh each time), could adopt them with little change (@sec-sw-simulated-users). The theory
on both sides, Wang, Liu, and Jin for reinforcement learning, Shah et al. for
ranking (@sec-adj-ltr), and Kayal et al. for PBO (@sec-theory-bt), suggests
that a pairwise answer carries less information per query than a numerical
one, but the achievable rate is of the same order, with the qualification that Kayal et al.'s result is a
conditional upper bound. And over-optimization reaches PBO only when a learned
utility is optimized without a person checking the results, as with the
utility over outcomes in preference exploration (BOPE); in standard PBO the
person judges the final candidates.

## The AI safety view {#sec-adj-safety}

Since 2017 the AI safety literature has formalized three ideas that matter for
any system that optimizes on behalf of a person: a stand-in for what the person
wants can fail when optimized, the system can change the person, and the person
and the system can be modeled as playing a cooperative game.

### When the proxy fails {#sec-adj-safety-proxies}

@skalse2022defining (NeurIPS 2022, where the paper appears as "Defining and
Characterizing Reward Gaming") defined a proxy reward as *unhackable* if
"increasing the expected proxy return can never decrease the expected true
return", and proved that "for the set of all stochastic policies, two reward
functions can only be unhackable if one of them is constant". "More capable
agents often exploit reward misspecifications", with "phase transitions:
capability thresholds at which the agent's behavior qualitatively shifts,
leading to a sharp decrease in the true reward" [@pan2022effects] (ICLR 2022).
@karwowski2024goodhart (ICLR 2024) explained Goodhart's law geometrically and
gave an early-stopping method that provably avoids it, and
@kwa2024catastrophic (NeurIPS 2024) showed that with heavy-tailed reward error
"some policies obtain arbitrarily high reward despite achieving no more utility
than the base model" even under KL regularization, though the reward models
they measured were "consistent with light-tailed error". From the principal's
side, unbounded optimization of a proxy that covers only some attributes can
drive the person's utility arbitrarily low, and letting the person update the
proxy over time helps [@zhuang2020consequences] (NeurIPS 2020); a written
reward function is "merely observations about what the designer actually
wants" [@hadfieldmenell2017inverse] (NeurIPS 2017); and by the **optimizer's
curse** [@smith2006optimizer], the estimated value of the option picked as best
is biased upward, because it was picked partly for its lucky error.

### Influence on the person, and approval optimization {#sec-adj-safety-influence}

**Approval optimization** means optimizing the evaluator's approval rather than
the outcome the evaluator cares about. @carroll2022estimating (ICML 2022)
pointed out that "systems trained via long-horizon optimization will have
direct incentives to manipulate users", in particular "to shift user
preferences so they are easier to satisfy", and proposed penalizing shifts outside a trust
region of "safe shifts", such as the drift a person's preferences would undergo
without the system. @carroll2024aib (ICML 2024) wrote that "existing AI
alignment approaches assume that preferences are static, which is unrealistic",
found that "8 such notions of alignment" all "either err towards causing
undesirable AI influence, or are overly risk-averse", and argued that the
optimization horizon "may partially help reduce undesirable AI influence".

The evidence comes from simulation and language models. A Q-learning
recommender "consistently learns to exploit its opportunities to polarize
simulated users" [@evans2023user] (AIES 2023). "Even if only 2% of users are
vulnerable to manipulative strategies", language models trained on user
feedback learn to identify and target them, and safety training or
language-model judges help in some settings but "backfire in others, sometimes
even leading to subtler manipulative behaviors"
[@williams2025targeted] (ICLR 2025). After RLHF, time-limited human evaluators
accepted wrong answers as correct more often, with false positive rates up
24.1% on QuALITY and 18.3% on APPS [@wen2025language] (ICLR 2025). "Both
humans and preference models (PMs) prefer convincingly-written sycophantic
responses over correct ones a non-negligible fraction of the time"
[@sharma2024understanding] (ICLR 2024). And when evaluators see only part of
what happened, RLHF can produce "deceptive inflation" and "overjustification"
[@lang2024your] (NeurIPS 2024).

### Assistance games {#sec-adj-safety-assistance}

An **assistance game**, also called cooperative inverse reinforcement learning,
models a person and an AI assistant who share the person's goal, which only
the person knows. AssistanceZero [@laidlaw2025assistancezero] (ICML 2025)
solved a Minecraft assistance game "with over $10^{400}$ possible goals", and
its assistant "significantly reduces the number of actions participants take
to complete building tasks". @emmons2025observation (ICML 2025) proved that an
optimal assistant must sometimes interfere with what the person can observe:
when the person decides based on immediate outcomes, the assistant may need to
interfere in order to query their preferences, an incentive that vanishes if
the person has a channel to communicate preferences; and a Boltzmann-irrational
person (choosing better options more often but not always) can also create an
incentive to interfere.
@ananthakrishnan2026provably, a 2026 preprint with a workshop version, gave the
first efficient algorithms for repeated assistance games, with a
$(1 - 1/e)$-approximate assistance regret of $\tilde O(T^{3/4})$, and proved
that beating $(1 - 1/e)$ is computationally infeasible; @fickinger2020multi, a
2020 preprint, applied impossibility theorems from social choice to assistants
serving several people (@sec-econ-social-choice). And existing RLHF algorithms
are not **strategyproof** (immune to gains from misreporting): "even a single
strategic labeler can cause arbitrarily large misalignment with social
welfare", and "any strategyproof RLHF algorithm must perform $k$-times worse
than the optimal policy, where $k$ is the number of labelers"
[@buening2025strategyproof] (NeurIPS 2025).

### What it means for PBO {#sec-adj-safety-meaning}

None of these results has been tested on PBO, so this subsection is inference.
PBO can be seen as a restricted assistance game in which the assistant acts
only by
choosing which candidates to show and what to recommend, and it assumes exactly
the Boltzmann-irrational person of Emmons et al., so an acquisition function
optimal under that model might favor pairs that are informative to the model
but distort the person's understanding of the design space. Whenever people
compare renderings, summaries, or explanations rather than real outcomes, the
results of Wen et al. and Lang et al. apply: the loop can converge to designs
that look better rather than designs that are better. And by the optimizer's
curse, the estimated utility of the recommended design is biased upward when
the model is misspecified, which Bayesian shrinkage toward the prior addresses.

Carroll et al.'s argument about horizons (2024) suggests that a myopic
acquisition function has no planned incentive to change the person, but
changes caused by what the person is shown still happen,
and @fig-soc-moving-target simulates how a benchmark score can improve while
the recommendation drifts from what the person first wanted. As of September
2026 we found no study of PBO that measures preference change caused by the
acquisition function. A study of PBO with people could measure it by
comparing preference drift when an optimizer chooses the pairs and when it
does not, taking the natural drift without the system, which Carroll et al.'s
2022 paper defines as a safe shift, as the baseline. Karwowski et al.'s provable early stopping is
a candidate for the stopping rule PBO lacks (@sec-rec-stopping), and with
several stakeholders, Kleine Buening et al.'s factor of $k$ limits what
incentive-compatible aggregation can achieve.

## Preference elicitation in AI and decision analysis {#sec-adj-elicitation}

**Preference elicitation** is the older name, in AI and decision analysis, for
the problem PBO solves: asking a person a few questions in order to make a good
decision for them. Its classical results still stand; we found no later work
that contradicts them. An efficient
allocation of many items can require communication exponential in their number
[@nisan2006communication]; whether a class of utilities can be elicited with
polynomially many queries depends on its structure [@blum2004preference]; and
elicitation cast as a partially observable Markov decision process, a planning
problem over beliefs, has continuous states and actions that put it beyond
standard solution techniques [@boutilier2002pomdp]. Choosing the question with the
highest **expected value of information**, the expected improvement in the
final decision from hearing the answer, goes back to @chajewska2000making and
is the counterpart of the decision-theoretic acquisition functions of
@sec-eubo (inference); minimizing the worst-case regret over utilities
consistent with the answers is the main alternative [@wang2003incremental].

The link with PBO is exact in one place: introducing qEUBO, Astudillo, Lin,
Bakshy, and Frazier wrote that it is closely related to @viappiani2010optimal,
who connected optimal recommendation sets with one-step Bayes-optimal query
sets; they also noted that the two criteria choose different queries when
answers are noisy, and their appendix uses a lemma deduced from the proof of
Theorem 3 in Viappiani and Boutilier's supplementary material
[@astudillo2023qeubob] (a journal version
appeared later [@viappiani2020equivalence]). How much can a comparison save?
If the learner may ask which of two examples is farther from the decision
boundary, under assumptions such as a large margin "it is possible to reveal
all the labels of a sample of size $n$ using approximately $O(\log n)$
queries", "an exponential improvement over classical active learning", while
without them $\Omega(n)$ queries are needed in the worst case
[@kane2017active] (FOCS 2017), extended to bounded (Massart) noise by
@hopkins2020noise (COLT 2020). For **CP-nets**, a qualitative representation
of conditional preferences ("if the main course is fish, I prefer white
wine"), @alanazi2020complexity (Artificial Intelligence 2020) computed the VC
and teaching dimensions (how many examples a learner needs) for learning
acyclic CP-nets from examples that differ in one attribute, with near-optimal
algorithms even when the oracle errs.

### Scale, robustness, and validation with people {#sec-adj-elicitation-progress}

The work closest to PBO is @zintgraf2018ordered (AAMAS 2018), who used
Gaussian process preference elicitation to choose among the solutions of a
multi-objective problem. With simulated and real users, they found that ranking
and clustering strategies "outperform the currently used pairwise methods",
that "users prefer ranking most", and that monotonicity, through "a linear
prior mean at the start and virtual comparisons to the nadir and ideal points"
(the worst and best values of every objective), "increases performance"; they
demonstrated the framework "in a real-world study on traffic regulation,
conducted with the city of Amsterdam".

Elicitation also grew in scale and robustness. @vendrov2020gradient (AAAI
2020) wrote the expected value of information as "a differentiable network that
can be optimized using gradient methods", and @martin2024model (IJCAI 2024)
learn the response and utility models from data and plan non-myopic
elicitation with Monte Carlo tree search. @vayanos2020robust, a 2020 preprint
whose journal version we did not find, represent uncertainty about the utility
as a set and choose queries for worst-case utility or regret;
@johnston2023deploying (EAAMO 2023) deployed this to prioritize COVID-19
patients for scarce hospital resources with "193 Amazon Mechanical Turk (MTurk)
workers", beating random queries by 21% in the utility of the recommended
policies. @herin2024noise (ADT 2024) proposed noise-tolerant active learning of
the weights of decision aggregation functions, and @mcelfresh2021indecision
(AAAI 2021) formalized models of *indecision* and tested them on a survey about
allocating organs. @defresne2025preference (IJCAI 2025) elicit preferences for
multi-objective combinatorial problems with "Maximum Likelihood Estimation of a
Bradley-Terry preference model" and "an ensemble-based acquisition function
inspired from Active Learning", and @bonilla2026causal (ICML 2026) elicit a
causal graph from experts with "a three-way likelihood over edge existence and
direction" and expected information gain.

### What it means for PBO {#sec-adj-elicitation-meaning}

Decision-centric query selection, choosing questions for their effect on the
final decision rather than for what they reveal about the utility everywhere,
appeared in three communities in turn (inference): decision analysis, with
Chajewska et al. (2000) and Viappiani and Boutilier (2010); preference-based
reinforcement learning, with information-directed reward learning (2021) and
query-policy misalignment (2024); and PBO, with qEUBO (2023) acknowledging its
debt to Viappiani and Boutilier. We did not check whether PBO papers cite the
reinforcement learning work, or the reverse.

Zintgraf et al. had already solved two problems within a Gaussian process
preference framework, monotonicity (a linear prior mean plus virtual
comparisons with the nadir and ideal points) and query format (ranking beats
pairs, and users prefer it), which later PBO work still treats as open
(inference): a monotone neural ensemble for preference exploration
[@wang2025bayesian] (NeurIPS 2025) attributes its gain to monotonicity in an
ablation, and
multi-choice interfaces such as GimmBO and MultiBO (@sec-query-forms) adopt
lists; we did not check whether these papers cite Zintgraf et al. The flow also
runs the other way: Defresne et al. adopt the Bradley-Terry likelihood and
ensemble acquisition familiar from PBO, without a Gaussian process
(inference).

Two practices are worth borrowing (inference). Minimax-regret elicitation gives
the worst-case regret over every utility consistent with the answers, though it
has traditionally assumed noise-free answers, and Herin et al. show it
converging with probabilistic methods; a PBO variant could report the
worst-case regret over a Gaussian process credible set next to the expected
regret, which suits the high-stakes allocations Vayanos et al. had in mind. And
Johnston et al. ran a randomized study with 193 users against random queries,
whereas PBO user studies usually have a dozen to a few dozen participants
(@sec-hci) and rarely a group answering random queries. Kane et al.'s
exponential gain needs a margin or a bounded description, the classical point
that structure decides whether elicitation is feasible; in Gaussian process PBO
the smoothness of the kernel plays that role (inference).

## Recommender systems and feedback loops {#sec-adj-recsys}

A recommender system chooses what a person sees, and what the person then
clicks becomes its training data: a **feedback loop**. A common claim is that
such loops trap people in a **filter bubble**, a narrowing diet of content
that reinforces what they already like, and by extension that PBO, which also
chooses what a person sees, could form one within a session. The evidence has
split. In simulation and theory, training on data confounded by earlier
recommendations "homogenizes user behavior without increasing utility"
[@chaney2018algorithmicb] (RecSys 2018); feedback loops amplify popularity
bias, and "the impact of feedback loop is generally stronger for the users who
belong to the minority group" [@mansoury2020feedback] (CIKM 2020); a model of
users who drift toward what a matrix-factorization recommender shows them
yields preference amplification under conditions the authors characterize and
validate with simulations [@kalimeris2021preference] (KDD 2021); and when preferences move toward what
people consume and like, "standard user reward maximization is an almost
trivial goal" ("a large class of simple algorithms will achieve only constant
regret") [@dean2022preference] (EC 2022).

Large field experiments find limited short-term effects on polarization. On
Facebook, individual choices limited exposure to diverse content more than
ranking did [@bakshy2015exposure]; on YouTube, "relying exclusively on the
YouTube recommender results in less partisan consumption", and when partisan
users switch to moderate content, the sidebar recommender "'forgets' their
partisan preference within roughly 30 videos"
[@hosseinmardi2024causally] (PNAS 2024); and in "four experiments with nearly 9,000
participants", manipulating recommendations to create filter bubbles and
rabbit holes "has limited effects on opinions" [@liu2025short] (PNAS 2025). But in a 7-week randomized
experiment with 4,965 users of X in the United States, switching from a
chronological feed to the algorithmic one raised the probability that users
considered the investigations into Donald Trump unacceptable by 5.5 percentage
points and moved policy priorities by 0.11 standard deviations, while switching
back had no comparable effect, and neither direction significantly changed
affective polarization or partisan identity [@gauthier2026political] (Nature
2026).

Recommenders also face elicitation directly at **cold start**, when a new user
has no history: asking only 2 questions improved recommendations by 25% over a
static model, with significant benefits from offline embeddings learned from
other users and from bandit-style exploration
[@christakopoulou2016conversational] (KDD 2016), and surveys of
conversational recommendation list question-based elicitation and evaluation
with simulated users among the open challenges [@jannach2021survey;
@gao2021advances]. **Critiquing** recommenders take directional feedback on
attributes ("cheaper", "more like this one but quieter");
@antognini2021fast (RecSys 2021) process critiques up to 25.6 times faster than
the best baselines with a variational autoencoder. PEBOL is discussed in
@sec-llm-translators.

### What it means for PBO {#sec-adj-recsys-meaning}

All of the following is inference. Warm starts from population embeddings have
been standard in recommender cold start since at least 2016, so the
meta-learned and population priors of PBO (PABBO, and the prior Liao et al.
pretrain from user models; @sec-hci-population) are the same idea, and that
Liao et al.'s prior helped significantly only in early iterations is
consistent with this reading. Dean and Morgenstern's result transfers to PBO
with changing preferences: if the person's utility moves toward what they are
shown, low regret can be achieved trivially, so stationarity or a safe-shift
criterion is the meaningful target (@fig-soc-moving-target). The field
experiments make an in-session filter bubble plausible but suggest that its
effect in a short session may be small, and Gauthier et al.'s asymmetry
suggests that changes induced by an optimizer might not reverse when it stops;
they concern weeks of news feeds, not a design session, so how far they
transfer is uncertain. And critiquing is the closest relative of the
projective and line-search queries of PBO (@sec-line-search,
@sec-gallery-projective): directional critique is a type of observation PBO
could adopt.

## Interactive evolutionary computation {#sec-adj-iec}

**Interactive evolutionary computation** is evolutionary search in which a
person, not a formula, judges the fitness of each candidate: the algorithm
breeds variants of the designs the person liked, shows them, and repeats. The
first comprehensive survey since @takagi2001interactive named user fatigue the
main challenge and does not compare the field with Bayesian optimization in its
abstract [@wang2024comprehensive] (Applied Soft Computing 2024). Interactive
differential evolution based on paired comparisons predates the 2017
formulation of PBO
[@takagi2009paired] (NaBIC 2009); the latent vector of a generative
adversarial network "can be put under evolutionary control" to evolve images
toward a target [@bontrager2018deep] (EvoMUSART 2018); and quality diversity
through human feedback "progressively infers diversity metrics from human
judgments of similarity among solutions" and "is more favorably received in
user studies" for text-to-image generation [@ding2024quality] (ICML 2024).
**Quality-diversity** search returns an archive of good solutions that differ
from one another, rather than one best solution. The exoskeleton study of
@lee2023user, in which an evolutionary algorithm proposed candidates for a
neural ranker and the wearer's forced choices, is in @sec-health-exo-others.

Evolution strategies have also been compared with Bayesian optimization where
a measured cost such as metabolic rate is minimized. Adaptive-sampling CMA-ES,
which spends evaluation time where candidates are hard to sort, "converged more
efficiently and reliably in complex landscapes, while in simpler landscapes,
AS-CMA was less efficient but equally reliable" [@martin2026improving]
(Evolutionary Computation 2026). In simulated exoskeleton tuning on a fitted
metabolic landscape [@kutulakos2024simulating] (a 2024 bioRxiv preprint),
Bayesian optimization converged near the optimum after about 60 evaluations on
a fixed landscape; when the landscape changed as a simulated novice adapted,
CMA-ES reached the optimum at similar rates for expert and novice simulations; and the authors conclude that no algorithm is clearly superior
for every use case. Both studies use a measured cost, not preferences.

### What it means for PBO {#sec-adj-iec-meaning}

All of the following is inference. Interactive evolution was using surrogate
fitness models, pairwise comparisons, and small displays against fatigue before
2017. PBO papers often claim sample efficiency over interactive evolution, but
the fair comparison is with surrogate-assisted interactive evolution under
human preference feedback, one of the missing studies listed at the end of
this chapter. An archive of diverse, high-quality options fits the goal of
exploration better than one maximizer when preferences are still forming, and
gallery and batch queries in PBO do this only in part. And CMA-ES's robustness
to a user who adapts is the scalar counterpart of PBO's drift problem: a
population-based or windowed method may tolerate drift better than a Gaussian
process posterior that keeps every stale comparison (@sec-theory-drift).
Beyond diversity and drift, the advances of interactive evolution itself (new
operators, models of fatigue) supply nothing PBO lacks.

## Learning to rank {#sec-adj-ltr}

**Learning to rank** trains models that order items, such as search results,
from clicks or judgments; **label ranking** predicts, for each input, an
ordering of a fixed set of labels; and **preference learning** is the umbrella
term. These fields have studied pairwise comparisons for decades, and three of
their results bear on how PBO should collect and read its data.

### How much a comparison is worth {#sec-adj-ltr-worth}

Two claims about comparisons circulate: that a comparison carries at most one
bit per query, far less than a rating (@sec-comparison-information), and that
pairwise feedback is not less efficient in an information-theoretic sense.
Both can be true. @shah2016estimation (JMLR 2016) derived tight minimax bounds
(bounds on the best achievable worst-case error) for estimating item qualities
under the Bradley-Terry-Luce and Thurstone models, which "depend on the
topology of the comparison graph induced by the subset of pairs being compared,
via the spectrum of the Laplacian of the comparison graph", and "the error
rates in the ordinal and cardinal settings have identical scalings apart from
constant pre-factors". Each comparison carries less information, but the rate
is of the same order.

### Three results PBO rarely uses {#sec-adj-ltr-results}

**Parametric links buy little.** @heckel2019active (Annals of Statistics 2019)
analyzed an algorithm that counts the comparisons each item wins and chooses
the next pair by confidence intervals, proved that it recovers the ranking
"using a number of comparisons that is optimal up to logarithmic factors"
without any parametric model, and settled what they call "a long-standing open
question": parametric assumptions such as the Thurstone or Bradley-Terry-Luce
models yield at most logarithmic gains for stochastic comparisons.

**Fitted utilities are not always majority-respecting.**
@noothigattu2020axioms (NeurIPS 2020) showed that "a large class of random
utility models (including the Thurstone-Mosteller Model), when estimated using
the MLE, satisfy a Pareto efficiency condition" and "a strong monotonicity
property", but "fail certain other consistency conditions from social choice
theory, and in particular do not always follow the majority opinion".

**Position bias can be measured and corrected.** @joachims2017unbiased (WSDM
2017) started from the observation that "position bias in search rankings
strongly influences how many clicks a result receives". Their counterfactual
framework treats the probability that a result in a given position is examined
as a **propensity**, reweights clicks by its inverse to obtain unbiased
learning to rank, and is "robust to noise and propensity model
misspecification". The propensity can be estimated only if the position of
results is varied, for instance by randomizing it for some users.

On label ranking, @thies2026calibrated (ICML 2026) proved that full-ranking
calibration implies the other notions, that "sub-ranking and top-k calibration
are incomparable", and that "popular label ranking models are often poorly
calibrated"; a model is **calibrated** when events it predicts with
probability $p$ happen about a fraction $p$ of the time. MORE-PLR
[@thies2026more] (Machine Learning 2026) predicts partial label rankings, in
which tied labels share a bucket.

### Position bias in a preference loop {#sec-adj-ltr-position}

The counterpart in PBO of a ranking position is the left or right slot of a
pair, or the position in a gallery. In the human studies of PBO that we read,
we saw none that reported randomizing or modeling the order of presentation,
though we did not check the methods section of every paper; language-model
judges have strong position bias, and averaging over both orders is the
simplest remedy (@sec-llm-judges) [@wang2024large]. @fig-adj-position-bias
shows what an uncorrected position bias can do to a preference loop.

```{figure}
//| figure: adj-position-bias
//| label: fig-adj-position-bias
//| fig-cap: "What a preference for the left slot does to a preference loop, if an interface always shows the current best design in the same slot. A simulated person answers by the Bradley-Terry model with an extra preference b for the left slot. The optimizer pairs its incumbent (the compared design with the highest fitted utility) with an optimistic challenger (the largest fitted improvement over the incumbent plus two standard deviations of that improvement); the surrogate is a Gaussian prior on 11 radial basis functions with a Laplace band. Top: one session, with the latest pair labeled by slot (the winner filled). Bottom: the regret of the incumbent, averaged over 24 simulated people, for three interfaces. The utilities, b, and the acquisition rule are illustrative."
```

Things to try:

1. Leave *Incumbent always left* selected with $b = 1$ and press play. The
   answers favor the incumbent about three times in four even when the
   challenger is better, the model reads the slot's advantage as quality, and
   the magenta curve stalls well above zero. The line "Answers for the left
   slot" mixes quality and slot advantage, and no model fitted to these data
   can separate them: the confounding Joachims et al. break by varying the
   position.
2. Switch to *Random side*. The bias now favors incumbent and challenger
   equally often, so it acts as extra noise rather than a thumb on the scale,
   and the green curve keeps falling.
3. Switch to *Random side, bias modeled*. The likelihood gains one parameter,
   the slot preference, whose estimate appears next to the true $b$; in these
   runs modeling mainly buys a measurement, with lower regret than random
   sides only at larger $b$.
4. Drag $b$ to 0. All three interfaces behave about the same: the same-slot
   design is harmless only if the person has no slot preference, and the
   same-slot data cannot tell you whether they do.

A bias toward the challenger's slot (not shown) did little harm in these
simulations, because it makes the loop switch incumbents too eagerly, which
costs little when the optimizer keeps the best compared design; the asymmetry
is a property of this incumbent-and-challenger loop, not a general result
(inference). The practical rule follows (inference): randomize the side of
every pair, and record it, so that a slot preference can be measured and
modeled later.

### What it means for PBO {#sec-adj-ltr-meaning}

All of the following is inference. By Heckel et al., the sample efficiency of
PBO should come mainly from the kernel sharing information between nearby
inputs, not from the form of the link. By Shah et al., the error depends on the
spectrum of the comparison graph, and a rule that always compares against the
current best builds a star-shaped graph that concentrates information on the
incumbent: good for locating the maximum, but not necessarily for learning a
reusable utility.
Two 2026 papers reached similar conclusions inside PBO (@sec-obs-graphs,
@fig-comparison-graph): @shao2026adaptive, a preprint, found that EUBO selects
pairs that form isolated components of the comparison graph, making the
Laplace likelihood Hessian rank deficient, and @pukdee2026preference (ICML
2026) identified the margin and the connectivity of the comparison graph as
what governs the sample efficiency of Bradley-Terry learning; this matches the
distinction of @sec-llm-purpose between pairs that locate the optimum and pairs
that teach a utility. PBO papers rarely report whether their predicted pairwise
probabilities are calibrated on held-out human comparisons, although Gaussian
approximations predict duel outcomes poorly [@takeno2023practicalc]
(@sec-obs-inference), and label ranking offers ready-made metrics. And a
Gaussian process utility pooled across users is a maximum likelihood random
utility model, so by Noothigattu et al. it may contradict the majority on some
pairs, which adds to the Borda-count result of @sec-llm-social.

## Automated science and self-driving labs {#sec-adj-sdl}

A **self-driving lab** couples robotic experimentation with an optimizer,
usually Bayesian optimization, that chooses the next experiment. One laboratory
study uses a person's judgment as the only measurement:
@deneault2025preferential (Digital Discovery 2025) tuned a 3-D printer for a
printing goal that is "difficult to measure with sensors but can be readily evaluated from human
judgment"; we could read only the abstract, so we do not report its campaign
sizes, query format, or gain. Other laboratory applications with experts in
the loop are in @sec-sci-experts.

@adesiji2026benchmarking (Digital Discovery 2026) define the **acceleration
factor**, the ratio of the experiments a reference strategy needs to reach a
target to those the optimizer needs, and the **enhancement factor**, the gain
after a fixed number of experiments. Across 42 studies and 63 benchmarks the
median reported acceleration factor was 6 (range 1.3 to 100), and the
enhancement factor peaked at about 10 to 20 experiments per dimension. As they
report, @shields2021bayesian (Nature 2021) found that by the 15th experiment
Bayesian optimization's average performance exceeded that of 50 expert
chemists (Adesiji et al. call the reaction space ten-dimensional; the
published data set has five choices and 1,728 measured conditions,
@sec-cs-chem-problem); @sec-cs-chemistry replays such an optimization, and
@sec-cs-chem-experts returns to the chemists.

People in automated experiments mostly supervise. "The likely strategy for the
next several years will be human-in-the-loop automated experiments", in which a
"human operator monitors experiment progression" and adjusts the agent's
policy [@kalinin2023human] (a 2023 preprint later published in *Microscopy
Today*); with deep kernel learning, "for certain parameter combinations the
experiment path can be trapped in the local minima", and monitoring was used to
construct "intervention strategies" [@pratiush2024building] (a 2024 preprint later published in
*Digital Discovery*); human input to an autonomous synthesis agent improved
sampling efficiency on synthetic benchmarks and found processing regions that
stabilize metastable phases in real Bi-Ti-O thin-film experiments
[@chang2026autonomous] (SARA-H, a 2026 preprint); and the GIFTERS checklist
for trustworthy AI in materials discovery (median score 5 out of 7 across the
reviewed work) argues for keeping people in the loop [@amirian2025building] (a
2025 preprint). LGBO used a language model's preferences about regions as side
information for scalar Bayesian optimization in a wet-lab experiment
(@sec-llm-roles).

### What it means for PBO {#sec-adj-sdl-meaning}

All of the following is inference. People in self-driving labs mostly act as
supervisors, stepping in when the surrogate gets stuck or the objective turns
out to be wrong, and pairwise preference is one channel among several, closer
to the BO-as-assistant arrangement of @sec-hci-agency than to a loop in which
the person only compares. The acceleration and enhancement factors, and the
peak at 10 to 20 experiments per dimension, offer PBO a reporting standard:
PBO papers mostly report regret on synthetic functions and rarely an
acceleration factor with real users against a human-only or random baseline,
Deneault et al. being a rare exception. Expert pairwise input pays off when the
expert's judgment carries information the surrogate lacks, such as a subjective
quality (Deneault et al.) or an unmeasured property, as with the materials
experts of @mikkola2020projective (@sec-sci-materials), and not when the
objective can be measured directly, where Bayesian optimization beat 50 expert
chemists on average [@shields2021bayesian].

## Common claims, checked {#sec-adj-claims}

@tbl-adj-claims checks claims about these neighboring fields that are repeated
in writing about PBO; two more, on reward-model calibration and on how many
comparisons each field uses, are checked in @sec-llm-claims.

::: {.table #tbl-adj-claims title="Common claims about the fields neighboring PBO, checked against the evidence."}
| Claim | What the evidence says |
|---|---|
| Inverse reinforcement learning and AI alignment treat the reward as fixed but unknown, so AI independently adopted PBO's assumption of a stable latent utility. | Accurate as a description of the formalisms, but questioned inside the field: "existing AI alignment approaches assume that preferences are static, which is unrealistic", which "may undermine the soundness of existing alignment techniques" [@carroll2024aib], and the usual model of how people generate preferences from a fixed reward is flawed [@knox2024models]. The shared assumption is a modeling convention, not independent evidence that stable utilities exist. |
| Potential-based reward shaping keeps instrumental queries from distorting inference about terminal preferences. | Shaping is one of the transformations that behavioral data cannot detect: by Theorem 3.3 of @skalse2023invariance (ICML 2023), Boltzmann-rational policies determine the reward only up to S′-redistribution and potential shaping. We found no source that uses shaping to design preference queries; the claim is an analogy. |
| HERON and DIPPER are published methods for hierarchical preference design in reinforcement learning. | Correct: HERON appeared at ICML 2025 and DIPPER, now titled "Direct Preference Optimization for Primitive-Enabled Hierarchical RL: A Bilevel Approach", at ICLR 2026 [@bukharin2023deep; @singh2024direct]. |
| Hejna et al. (CoRL 2023) used meta-learned reward functions to need 20 times fewer queries than PEBBLE. | The paper is @hejna2022few, in PMLR volume 205 (the 6th Conference on Robot Learning, December 14 to 18, 2022, published March 6, 2023). Its abstract reports reducing online feedback in Meta-World "by 20×", with a real Franka Panda demonstration, but does not name PEBBLE as the comparison. |
| Cooperative inverse reinforcement learning treats the human's reward parameters as existing, stable, and unknown to the robot. | Accurate. AssistanceZero keeps this assumption [@laidlaw2025assistancezero]; @emmons2025observation add partial observability. |
| Sequential strategic misreporting remains possible, so acquisition should be incentive-compatible. | Now quantified for RLHF: one strategic labeler can cause arbitrarily large misalignment, and any strategyproof algorithm can be $k$ times worse than optimal [@buening2025strategyproof]. |
| Exact elicitation, cast as a POMDP, is PSPACE-hard (attributed to Boutilier 2002), and the classical complexity results set the limits of elicitation. | The classical results stand (@sec-adj-elicitation), and no newer CP-net work relevant to PBO turned up in our searches. The PSPACE statement is not in @boutilier2002pomdp, whose abstract says standard POMDP techniques cannot solve the problem because its states and actions are continuous; the PSPACE-completeness of finite-horizon partially observed Markov decision problems with finitely many states [@papadimitriou1987complexity] is not a result about elicitation. |
| Interactive evolution makes discovery possible without domain knowledge, and design galleries implicitly acknowledged that preferences are discovered. | Holds. Recent interactive quality-diversity work states the same goals, few alternatives "to reduce cognitive load" that "should be diverse but similar to the previous user selection, to reduce user fatigue", and its windowed MAP-Elites (which keeps the best solution in each cell of a grid of behaviors) "finds more appropriate solutions to the user's taste", tested with "controllable artificial users" [@sfikas2023controllable]. |
:::

## What the neighbors solved first {#sec-adj-summary}

@tbl-adj-solved lists problems that a neighboring field has solved or given a
ready practice for, and that the PBO literature still treats as open or has not
adopted, with whether the suggestion depends on the Gaussian process framework.

::: {.table #tbl-adj-solved title="Problems a neighboring field solved first. The status column is our assessment of the PBO literature (inference)."}
| Problem | Field that got there first | Status in PBO | Depends on the GP framework? |
|---|---|---|---|
| Decision-centric query selection | decision analysis (Viappiani and Boutilier 2010); preference-based RL (Lindner et al. 2021, Hu et al. 2024) | adopted by qEUBO (2023), source acknowledged | no |
| Monotonicity and ranking queries | GP preference elicitation (Zintgraf et al. 2018) | still treated as open | monotonicity by virtual comparisons: yes; ranking: no |
| Simulated teachers with realistic irrationalities | preference-based RL (B-Pref 2021) | benchmarks mostly use homoscedastic, independent noise | no |
| Measuring and correcting position bias | learning to rank (Joachims et al. 2017); language-model judges (Wang et al. 2024) | no reports of randomized or modeled order found | no |
| Warm starts from population embeddings | recommender systems (Christakopoulou et al. 2016) | population priors presented as new | no |
| Designing the comparison graph | ranking estimation (Shah et al. 2016) | raised in 2026, through the rank-deficient Laplace Hessian | partly |
| Calibration metrics | label ranking (Thies et al. 2026) | calibration of pairwise probabilities rarely reported | no |
| Worst-case guarantees | robust elicitation (Vayanos et al. 2020, Johnston et al. 2023, Herin et al. 2024) | probabilistic posterior only | yes, over a GP credible set |
| Robustness when the user adapts | evolution strategies (Martin and Collins 2026, Kutulakos and Slade 2024) | no drifting-utility model | no |
| An archive of diverse options | quality diversity through human feedback (Ding et al. 2024) | no learned diversity objective | no |
| Directional feedback on attributes | critiquing recommenders (for example, Antognini et al. 2021) | not used as an observation | no |
| Formalizing the system's influence on the person | AI safety (Carroll et al. 2022, 2024; Emmons et al. 2025) | not measured | no |
| Reporting by acceleration factors | self-driving labs (Adesiji et al. 2026) | mostly regret on synthetic functions | no |
:::

Most of these suggestions come from the problem itself, not from the Gaussian
process framework; only monotonicity through virtual comparisons, worst-case
regret over a credible set, and the rank deficiency of the comparison graph are
repairs inside it.

## Settled, contested, missing {#sec-adj-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** Decision-centric query selection was proposed in decision
analysis, then in preference-based reinforcement learning, then in
PBO, and qEUBO acknowledges its decision-analytic source
[@astudillo2023qeubob]. Pairwise feedback carries less information per query
than ratings but achieves the same rate up to constants
[@shah2016estimation], and parametric links add at most logarithmic gains for
ranking [@heckel2019active]. The choice of model for how people generate
preferences changes what can be learned [@knox2024models], and small
adversarial biases can produce arbitrarily large errors in inferred rewards
[@hong2023sensitivity]. Position bias in rankings is measurable and correctable
when position is varied [@joachims2017unbiased]. Simulated teachers with
perfect preferences are unrealistic [@lee2021b]. Systems trained with long
horizons have an incentive to shift preferences [@carroll2022estimating], and
manipulation of a vulnerable minority has been demonstrated in language models
[@williams2025targeted].

**Contested.** Whether recommender feedback loops change attitudes: simulations
say yes, most field experiments find limited short-term effects, and one 7-week
experiment found an asymmetric effect [@gauthier2026political]. Whether
evolution strategies or Bayesian optimization suit human-in-the-loop
optimization better: the answer depends on the landscape and on whether the
user adapts [@martin2026improving; @kutulakos2024simulating]. Whether expert
pairwise input improves on a well-specified optimizer: it helped when the goal
was subjective [@deneault2025preferential] and lost to the optimizer when the
yield could be measured [@shields2021bayesian].

**Missing.** A measurement of preference change caused by an acquisition
function in a study of PBO with people. Any study of PBO that randomizes or
models presentation order. A comparison of interactive evolution and PBO under
pairwise preference feedback, including one with surrogate-assisted
interactive evolution as the baseline, and a quantitative model of fatigue taken from interactive
evolution. A PBO method that optimizes a diversity objective learned from
people, uses directional critique as an observation, or models a drifting
utility. Worst-case guarantees reported alongside the posterior. Acceleration
factors measured with real users against human-only or random baselines. A
self-driving lab with more than one human oracle giving pairwise preferences in
a closed loop. Benchmarks that use B-Pref-style irrational simulated users.
:::

## Further reading {#further-reading .unnumbered}

- @lee2021b, the B-Pref benchmark, is the clearest catalog of the ways a
  simulated person can be irrational, and reads as a checklist for
  PBO benchmarks.
- @knox2024models and @hong2023sensitivity together explain why the model of
  how a person answers matters as much as the model of what they want.
- @carroll2024aib is the most direct treatment of systems that change the
  preferences they learn from.
- @zintgraf2018ordered is a Gaussian process preference elicitation study with
  real users that settled monotonicity and query format in 2018.
- @joachims2017unbiased shows how a biased presentation can be turned into a
  measured propensity, the idea behind @fig-adj-position-bias.
- @adesiji2026benchmarking defines the acceleration and enhancement factors and
  collects them across self-driving lab studies.
