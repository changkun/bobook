---
status: done
synopsis: "How language-model alignment learns from comparisons, what language models do inside preference loops, which PBO and dueling-bandit methods have been applied to language-model problems and with what results, the Bradley-Terry likelihood both fields share and the seven analogies between them that break, and what has flowed back."
---

# Preferences and Large Language Models {#sec-llms}

The most visible use of pairwise comparisons in computing today is not
preferential Bayesian optimization (PBO). It is the alignment of large language
models, where people (and increasingly other models) read two responses to the
same prompt and say which is better, millions of times, and the answers are
used to tune the model. The question asked of the annotator is the question
@sec-pbo asks of a person tuning a design: *which of these two do you prefer?*
The likelihood used to learn from the answer is, in most alignment work, the
Bradley-Terry model of @sec-bradley-terry.

From 2023 to September 2026 the two fields met in two directions. Language
models moved into preference loops, as priors, simulated users, translators
from text to preference labels, feature extractors, and generators of
candidates. In the other direction, tools from dueling bandits and Bayesian
active learning were applied to prompt optimization, to choosing among outputs
and models, and to choosing which comparisons to collect for training. For
this book's argument, alignment is where the measurement question was asked at
scale: how far a model judge departs from a person, and which comparisons are
worth asking for (@sec-syn-bottleneck). The reported gains range from 1% to 6%
in win rate to roughly an order of magnitude in labels saved, but most results
replace people with a reward-model simulator or a language-model judge, and
some careful studies find that choosing comparisons cleverly barely beats
choosing them at random. The chapter explains how alignment learns from
comparisons, follows the two directions, separates the likelihood the fields
share from seven analogies that break, and ends with what has flowed back.

## How language models learn from comparisons {#sec-llm-primer}

A reader who knows Gaussian process preference learning already knows most of
the machinery. This section introduces the vocabulary of alignment in that
reader's terms.

### Policies, rewards, and the Bradley-Terry reward model {#sec-llm-reward-models}

A language model, given a prompt $x$, produces a response $y$ by sampling
words one at a time, which defines a probability distribution over responses,
$\pi(y \given x)$. Alignment research borrows the word **policy** from
reinforcement learning for this distribution: a rule for choosing actions,
where the action is the entire response. The model before preference tuning,
usually already fine-tuned on written demonstrations, is the **reference
policy** $\pi_{\text{ref}}$.

A **reward model** is a separate network that maps a prompt and a response to
a single number $r(x, y)$, often a copy of the language model with its output
layer replaced by one scalar [@ouyang2022training]. It plays exactly the role
of the latent utility $f$ in @sec-gp-preference: nobody observes it, and it is
learned only from comparisons. If an annotator is shown responses $y$ and $y'$
to prompt $x$, the Bradley-Terry model says

$$
\Prob(y \succ y' \given x) = \operatorname{sigmoid}\big(r(x, y) - r(x, y')\big),
\qquad \operatorname{sigmoid}(z) = \frac{1}{1 + e^{-z}},
$$ {#eq-llm-bt}

where $\operatorname{sigmoid}$ is the logistic function (language-model papers
write it $\sigma$, a letter this book reserves for standard deviations). A
reward model is trained by maximizing the log-likelihood of comparisons, each a
prompt with a preferred response $y_w$ and a rejected one $y_l$:

$$
\mathcal{L}(r) = -\sum_{(x, y_w, y_l)} \log \operatorname{sigmoid}\big(r(x, y_w) - r(x, y_l)\big).
$$ {#eq-llm-rm-loss}

Set beside @eq-pbo-probit, the only differences are the link (logistic rather
than probit, two noise distributions of the same random utility family,
@sec-random-utility), the absence of a prior on $r$, and the scale: a large
neural network fitted to tens of thousands to millions of comparisons instead
of a Gaussian process fitted to a few dozen. The ranking extension, in which
the probability of an ordering of $K$ responses is a product of softmax choices
from the remaining items, is the **Plackett-Luce** model.

### RLHF and the KL-regularized objective {#sec-llm-rlhf}

**Reinforcement learning from human feedback (RLHF)** uses such a reward model
to tune a policy. @christiano2017deep introduced it for games and simulated
robots, and @ouyang2022training applied it to instruction-following language
models. It samples responses and collects human comparisons, fits a reward
model with @eq-llm-rm-loss, and then fine-tunes the policy to maximize, for
each prompt $x$ and on average over prompts,

$$
\E_{y \sim \pi}\big[r(x, y)\big] - \beta\,\KL\big(\pi \,\|\, \pi_{\text{ref}}\big),
$$ {#eq-llm-rlhf}

where $\pi$ and $\pi_{\text{ref}}$ are short for $\pi(\cdot \given x)$ and
$\pi_{\text{ref}}(\cdot \given x)$. The second term is the **KL divergence**
of @sec-kl, how far the tuned policy has moved from the reference, and
$\beta > 0$ sets how much movement the reward must pay for. Without it, the
policy would drift toward whatever text the reward model overrates, a failure
called **reward over-optimization**: the learned reward keeps rising while
true quality falls [@coste2024reward]. Notice what @eq-llm-rlhf asks for: not
the single best response, but a *distribution* over responses that scores well
on average while staying close to the reference, the first analogy that breaks
in @sec-llm-breaks.

### DPO and the implicit reward {#sec-llm-dpo}

**Direct preference optimization (DPO)** removes the separate reward model and
the reinforcement learning step. @rafailov2023direct observed that
@eq-llm-rlhf has a closed-form optimum, and that the Bradley-Terry likelihood
can then be written directly in terms of the policy.

::: {.derivation title="From the KL-regularized objective to the DPO loss"}
Fix one prompt $x$ and drop it from the notation. Write $\pi(y)$ for the
policy and sums over all responses $y$.

1. The objective of @eq-llm-rlhf for this prompt is
   $J(\pi) = \sum_y \pi(y)\, r(y) - \beta \sum_y \pi(y) \log \frac{\pi(y)}{\pi_{\text{ref}}(y)}$,
   by the definition of the expectation and of the KL divergence.
2. Define $Z = \sum_y \pi_{\text{ref}}(y)\, e^{r(y)/\beta}$ and the
   distribution $\pi^\star(y) = \pi_{\text{ref}}(y)\, e^{r(y)/\beta} / Z$.
   Factoring $-\beta$ out of both terms of step 1 gives
   $J(\pi) = -\beta \sum_y \pi(y) \log \frac{\pi(y)}{\pi_{\text{ref}}(y)\, e^{r(y)/\beta}}$.
   Writing $\pi_{\text{ref}}(y)\, e^{r(y)/\beta} = Z\, \pi^\star(y)$ inside the
   logarithm splits it into $\log \frac{\pi(y)}{\pi^\star(y)} - \log Z$, and
   because $\sum_y \pi(y) = 1$ the constant contributes $-\log Z$ once:
   $J(\pi) = -\beta\, \KL(\pi \,\|\, \pi^\star) + \beta \log Z$.
3. A KL divergence is never negative and is zero only when the two
   distributions are equal, so $J$ is maximized by $\pi = \pi^\star$:
   $$
   \pi^\star(y) = \frac{1}{Z}\, \pi_{\text{ref}}(y)\, e^{r(y)/\beta}.
   $$ {#eq-llm-tilt}
4. Take logarithms of @eq-llm-tilt and solve for the reward:
   $$
   r(y) = \beta \log \frac{\pi^\star(y)}{\pi_{\text{ref}}(y)} + \beta \log Z.
   $$ {#eq-llm-implicit}
5. Restore the prompt and substitute @eq-llm-implicit into the Bradley-Terry
   model @eq-llm-bt. The difference $r(x, y_w) - r(x, y_l)$ contains
   $\beta \log Z(x)$ twice with opposite signs, because $Z$ depends on the
   prompt but not on the response, so it cancels. The intractable sum over all
   possible responses disappears.
6. What remains is a likelihood in which the policy itself plays the role of
   the reward. Write
   $\hat r_\pi(x, y) = \beta \log \pi(y \given x) - \beta \log \pi_{\text{ref}}(y \given x)$
   for this policy-defined reward. Maximizing the Bradley-Terry likelihood over
   a data set of comparisons is then the DPO loss:
   $$
   \mathcal{L}_{\text{DPO}}(\pi) = -\sum_{(x, y_w, y_l)} \log \operatorname{sigmoid}\big(\hat r_\pi(x, y_w) - \hat r_\pi(x, y_l)\big).
   $$ {#eq-llm-dpo}
:::

In the words of its title, the policy "is secretly a reward model": the
quantity $\hat r_\pi(x, y)$ is called the **implicit reward**, and
@eq-llm-dpo is @eq-llm-rm-loss with $\hat r_\pi$ in place of $r$, with a
Plackett-Luce version for rankings in the paper. Two relatives recur below,
both members of one family of convex losses [@tang2024generalized]: **IPO**
[@azar2024general] uses a squared loss, and **SLiC** a hinge loss. And
@eq-llm-tilt says that the optimal policy is the reference policy reweighted by
$e^{r/\beta}$, a tilt of what the model already does rather than a jump to the
best response.

### Where comparisons come from {#sec-llm-data}

**Offline** preference learning uses a fixed data set gathered in advance;
**online** learning samples new responses from the current policy during
training and sends them for labeling; **active** learning also chooses which
prompts and pairs are worth labeling. That choice is an acquisition function
in the sense of @sec-acquisition, and it is where tools from PBO and dueling
bandits enter alignment. A **dueling bandit** (@sec-dueling-bandits-intro)
chooses two arms per round and observes which one wins; a **contextual dueling
bandit** first observes a context, here the prompt. Much of the evidence below
uses an **LLM judge**, a language model prompted to compare two responses in
place of a person, and reports a **win rate**, the fraction of head-to-head
comparisons against a fixed baseline model that the tuned model wins.

::: {.keyidea title="Same question, different job"}
Alignment and PBO both learn a latent score from answers to
"which is better?" with a pairwise random utility likelihood. Alignment then
uses the score to reshape a distribution over text near a reference model;
PBO uses it to find one best design. Most of what follows comes
back to that difference.
:::

## Language models inside the preference loop {#sec-llm-in-loop}

Between 2023 and 2026 language models joined preference loops in five jobs: a
prior or warm start, a simulated user, a translator from text to preference
labels, a feature extractor, and a generator of candidates. The systems that
worked best kept a classical probabilistic model in charge of uncertainty and
of choosing queries, and let the language model work only at the interface;
systems in which the language model itself acted as the optimizer fell behind
classical dueling-bandit algorithms in strong regret.

### Translators: from words to preference signals {#sec-llm-translators}

**PEBOL** [@austin2024bayesian] elicits preferences in conversational
recommendation with an independent Beta-distributed utility per item, the
conjugate model of @sec-beta-binomial: a natural language inference model
turns the user's "yes" or "no" about an aspect into a likelihood, and Thompson
sampling or an upper confidence bound (@sec-thompson, @sec-ucb) decides what
GPT-3.5 should ask next. After 10 turns, its mean reciprocal rank at 10 (the
average of one over the rank of the user's target item in the top-10 list) was
0.27 on Yelp against 0.12 for a monolithic GPT-3.5 elicitor, 0.18 against 0.09
on MovieLens, and 0.17 against 0.11 on Recipe-MPR; the best monolithic
baseline, Gemini-Pro, reached 0.17. The authors attribute the monolithic
models' failure to over-exploitation, in severe cases asking the same question
again. The users were 100 GPT-3.5 simulations per data set, told in advance
which item they liked.

**OPEN** [@handa2024bayesian], a 2024 preprint, has a language model extract
and rank features of a domain to initialize the prior of a linear
Bradley-Terry utility, chooses pairwise queries by the expected information
gain of @sec-expected-information-gain over a particle-filter posterior (a
cloud of weighted samples), and has the language model rewrite each abstract
comparison as a natural question. With people on the Prolific platform,
recommending New York Times articles, it beat elicitation by the language model
alone and by experimental design alone; without the rewriting, users found it
markedly harder to express their preferences accurately, and mental demand was similar
across methods. **MAPLE** [@mahmud2025maple] (AAAI 2025) turns natural-language
feedback into samples or ranges for the linear weights of abstract concepts,
with a Bradley-Terry likelihood over pairwise rankings of trajectories and Markov chain Monte
Carlo (@sec-mcmc), evaluated with modeled humans.

**LILO** [@kobalczyk2026lilo] (ICML 2026) is the closest to PBO in the strict
sense. A language model translates free-text feedback and textual priors into
pairwise labels for a probit pairwise Gaussian process over a composite
utility $u(\vx) = g(f(\vx))$, where $f$ gives an experiment's measured outcomes
and $g$ the decision maker's utility over them, the setting of preference
exploration [@lin2022preferenceb], and qEUBO, the expected utility of the best
option for queries of $q$ options (@sec-eubo), selects which outcomes to have
compared. Across 10 environments it beat pure language-model optimizers and PBO
baselines, and in some settings one message from the decision maker matched or
exceeded 8 to 16 pairwise comparisons; the problems had 5 to 8 dimensions, and
the decision maker was simulated by Llama-3.3-70B. Its precursors include
preregistered experiments with people in which questions generated by a
language model elicited answers often more informative than prompts or labels
users wrote themselves, with less reported effort [@li2025eliciting]; a
NeurIPS 2023 workshop paper that chose questions by expected entropy reduction
[@piriyakulkij2023active]; and, by LILO's first author, Bayesian experimental
design over candidate solutions sampled from a language model
[@kobalczyk2025active] (ICLR 2025).

### Priors, features, candidates, and the optimizer itself {#sec-llm-roles}

**Priors and warm starts.** Besides OPEN and MAPLE, @eichelbeck2026supporting
(ICLR 2026) run PBO in an autoencoder's latent space with a prior a language
model generates from interviews, with simulated users only. On the scalar side
(@sec-hd-llm), LLAMBO [@liu2024large] (ICLR 2024) uses a language model for
warm starts, as a surrogate, and to sample candidates; a 2025 reproduction
preprint found that warm starting "substantially improves early regret
behaviour and reduces variance across runs" but that the surrogate is "weaker
than GP or SMAC as a pure single task regressor" (SMAC uses a random forest),
and its abstract attributes LLAMBO to "Daxberger et al. (2024)", which does not
match the original authors [@rychert2025reproducibility]. LGBO
[@yuan2026unleashing] (ICLR 2026) adds a language model's preferences about
regions to the surrogate mean, with a worst-case guarantee, faster convergence
when the preferences agree with the objective, and one wet-lab experiment, for
a scalar objective. The evidence against is pointed: language-model agents in
scalar BO performed no differently when their observed outcomes were replaced
by randomly permuted labels, while linear bandits and Gaussian process
optimization consistently won [@gupta2025llms] (EMNLP 2025 Findings), and the
better starting point a language-model adviser suggested was a default
configuration [@rodrigues2026llm], a 2026 preprint.

**Feature extractors and surrogates.** Language models help BO over molecules
"only if they have been pretrained or finetuned with domain-specific" data
[@kristiadi2024sober] (ICML 2024); BO-ICL [@ramos2023bayesian], first posted
in 2023 and published in 2026, found a near-optimal catalyst among 3,700
candidates within 6 iterations by in-context regression; and
@rankovic2026large (Nature Machine Intelligence 2026) train embeddings and a
Gaussian process jointly, matching conventional BO with a median of 41% fewer
iterations across 23 chemistry and materials tasks. All three use scalar
feedback; with preference feedback, the instances of "language model
embedding plus a probabilistic head" are APOHF and the method of Dwaracherla et al. (@sec-llm-methods),
neither with a Gaussian process.

**Candidates and the optimizer itself.** APPO [@li2026preference] (CHI 2026)
has a language model rewrite text-to-image prompts from a user's binary
preferences, and PDO and Duel-Evolve (@sec-llm-methods) let one language model
both generate candidates and judge them. @xia2025numeric (ACL 2025 Findings)
asked top language models to act as dueling-bandit algorithms in context. They
quickly brought the best arm into the duels, giving low short-term *weak*
regret (which counts the better of the two arms), but in *strong* regret
(which counts both) "an optimality gap still exists" with classical
algorithms; they "struggle to converge and consistently exploit even when
explicitly prompted to do so". A hybrid, LEAD, wraps a classical algorithm
around the language model and inherits its guarantees.

@tbl-llm-systems sorts the main systems by whether they are PBO in the strict
sense used in this book: a probabilistic surrogate that learns a latent utility
over a continuous or structured design space from pairwise or ranked feedback,
with an acquisition function choosing the queries.

::: {.table #tbl-llm-systems title="Systems that put a language model in a preference loop, and whether each is PBO in the strict sense."}
| System | Role of the language model | Probabilistic model and query rule | Feedback | Strict PBO? | Who answered in the evaluation |
|---|---|---|---|---|---|
| PEBOL (RecSys 2024) | asks questions; an inference model turns answers into a likelihood | independent Beta-Bernoulli item utilities; Thompson sampling or UCB | yes or no | no: independent-arm bandit | 100 GPT-3.5 simulated users per data set |
| OPEN (2024 preprint) | extracts features, initializes the prior, rewrites questions | linear Bradley-Terry utility; particle filter; expected information gain | pairwise | partly: no GP | people on Prolific |
| MAPLE (AAAI 2025) | supplies weight priors, interprets language feedback | linear weights over concepts; Bradley-Terry; MCMC | pairwise rankings plus language | partly: as for OPEN | modeled humans |
| LILO (ICML 2026) | translates free text into pairwise labels | probit pairwise GP on a composite utility; qEUBO | language turned into pairwise labels | yes | Llama-3.3-70B simulated decision maker, 5 to 8 dimensions |
| Eichelbeck et al. (ICLR 2026) | generates the prior from interviews | PBO in an autoencoder latent space | pairwise | yes | simulated users only |
| LGBO (ICLR 2026) | states regional preferences | shift of the GP mean | scalar plus language-model preferences | no: scalar BO | benchmarks and one wet-lab experiment |
| APOHF (workshop paper) | frozen embeddings as features | neural network, Bradley-Terry; greedy plus UCB | pairwise | no: neural dueling bandit | simulated (validation accuracy, image similarity) |
| Duel-Evolve (workshop paper) | generates candidates and judges them | Bayesian Bradley-Terry; double Thompson sampling | the model's own pairwise judgments | no: test-time search | ground-truth benchmark accuracy |
:::

Only LILO and the method of Eichelbeck et al. meet the strict definition, so
calling all of these "PBO with language models" would overstate how much of
this intersection Gaussian process preference models occupy.

### Language for goals, comparisons for judgments {#sec-llm-division}

If a language model can carry a person's words into the loop, should the
person still be asked to compare? In the design studies of @sec-hci, natural
language and explicit constraints gave the same optimization performance, with
lower workload for language and a stronger sense of agency for constraints,
and 90.9% of 187
natural-language requests described a desired outcome rather than a parameter
value [@niwa2025cooperative]. In a preprint, @peng2026efficient found that 20
people judging the same 600 pairs of generated interfaces agreed with one
another at a Krippendorff's $\alpha$ of only 0.25 (two of them made the same
choice on 62.4% of pairs on average), yet for 12 new users, personalization
from just 8 pairwise judgments beat every baseline, including the users' own
written preferences, with an aggregate win rate of 60.35%
(@sec-hci-feedback-forms).

The model side points the same way. Language models inferring a user's
preferences over several turns fall "far short" of normative Bayesian
updating, and training them to imitate a Bayesian model improves this and
generalizes [@qiu2025bayesian] (Nature Communications, January 2026); and on a
simulated-user benchmark for helping users *construct* preferences, no
frontier model exceeded 56% accuracy within 5 turns [@saracay2026expert]
(COLM 2026). Together these results support a division of labor: language for
goals, constraints, and priors; comparisons for fine judgments; and posterior
updating left to an explicit probabilistic model (inference), as in the case
for comparisons over ratings in @sec-people-objective.

### Language models as simulated users and judges {#sec-llm-judges}

Many systems in this chapter were evaluated with a language model standing in
for the person. The evidence from 2023 to 2026 points one way: close to people
in aggregate, unreliable for individuals, and biased in systematic
directions. In @tbl-llm-judges, **Spearman** correlation and **Kendall's
$\tau$** both measure how similarly two lists are ordered (1 for identical
order, 0 for no relation), and agreement rates count how often two judges pick
the same option.

::: {.table #tbl-llm-judges title="How faithful language models are as judges and simulated users, in aggregate and for individuals."}
| Study | Setting | In aggregate | For individuals, or biases |
|---|---|---|---|
| Zheng et al., NeurIPS 2023 Datasets and Benchmarks | GPT-4 judging MT-Bench and Chatbot Arena | agreement with people "over 80%", the same as between people | position bias, verbosity bias, self-enhancement bias |
| Dubois et al., NeurIPS 2023, AlpacaFarm | GPT-4 annotator with a single prompt | 65% agreement with people against 66% between people; method rankings correlate with those from human data at Spearman 0.98, at about 50 times lower cost | does not reproduce the variability of human labels or reward over-optimization; reproducing it needed a pool of simulated annotators and labels flipped at random with probability 0.25 |
| Wang et al., ACL 2024 | ChatGPT as evaluator | not reported | after swapping the order of responses, Vicuna-13B beat ChatGPT on 66 of 80 queries |
| Panickssery, Bowman, and Feng, NeurIPS 2024 | fine-tuning changes self-recognition | not reported | self-recognition ability correlates linearly with the strength of self-preference |
| Muldrew et al., ICML 2024 | 50 prompts labeled twice | GPT-4 agrees with itself more than 90% of the time | GPT-3.5-turbo about 60% |
| Kirk et al., 2026 preprint, PRISM-X | 530 people, each ranking four models; GPT-4o as simulated user, either ranking the person's transcripts or also holding the conversations | model scores fitted to simulated and to human rankings correlate at r = 0.99 and 0.98 | mean Kendall's τ with the person's own ranking of 0.22 (ranking only) and 0.11 (conversing), against 0.57 for people's own consistency; strong position effects (@fig-llm-judge-fidelity); more sycophantic than people |
| Kuric, Demcak, and Krajcovic, 2026 preprint | 29 real design preference tests (n = 2073), 78 tasks | top-choice agreement 53%; GPT 5.2 reaches 65% with no significant gain in fidelity | simulated and real distributions differ significantly in 44% of tasks; single-persona simulation deviates significantly in 91%; temperature and top-p have no significant effect |
| Xu et al., ICML 2025 | judges on AlpacaEval | round-robin plus Bradley-Terry raises Spearman correlation with Chatbot Arena from 95.0% to 96.4%, Kendall from 82.1% to 86.3% | judgments are intransitive; rankings depend on the chosen baseline model |
| Shi et al., IJCNLP-AACL 2025 | 15 judges, over 150,000 instances | not reported | position bias correlates weakly with prompt length and is "strongly affected by the quality gap between solutions" |
| Yang et al., 2026 preprint | 20 language models, pairs of responses of equal quality | not reported | capability and low self-preference bias are "often uncorrelated, or even negatively correlated"; structured multi-dimensional evaluation cuts the bias by 31.5% on average |
| Chawla, Thompson, and Young, 2026 preprint | 7 models, 4 tasks | not reported | intransitivity cannot be explained by a single ordering under any monotone link; a mixture of several latent orderings fits better |
| Seshadri et al., 2026 preprint | simulated users in agent evaluation | not reported | agent success rates differ by up to 9 percentage points between simulators; fidelity is worse for some dialect groups |
| Zhu, Huang, and Sang, WWW 2024 workshop | simulated users in conversational recommendation | not reported | data leakage in the dialogue history and the simulator's replies inflates results; PEBOL relies on exactly this kind of evaluation |
:::

The sources are, in order, @zheng2023judging, @dubois2023alpacafarm,
@wang2024large, @panickssery2024llm, @muldrew2024active, @kirk2026prism,
@kuric2026distorted, @xu2025investigating, @shi2025judging,
@yang2026quantifying, @chawla2026multiple, @seshadri2026lost, and
@zhu2024reliable. *Position bias* is a preference for whichever response is
shown first (or second); *verbosity bias* a preference for longer responses;
*self-enhancement* or *self-preference bias* a judge's preference for text it
generated itself;
*sycophancy* a tendency to tell the user what they appear to want to hear.
@fig-llm-judge-fidelity draws the clearest case, PRISM-X.

```{figure}
//| figure: llm-judge-fidelity
//| label: fig-llm-judge-fidelity
//| fig-cap: "Right about the population, wrong about the person, in PRISM-X (Kirk et al., a 2026 preprint): 530 people each ranked four models, and GPT-4o, as a simulated user, ranked the same models either from the person's transcripts only or after holding the conversations itself. *Agreement with people*: model scores fitted to simulated rankings correlate with those fitted to human rankings at r = 0.98 to 0.99 (the two conditions), while each simulated ranking agrees with that person's own at a mean Kendall's τ of 0.22 or 0.11, against 0.57 for the consistency of people's own ratings and rankings. *Ranked best*: the share of trials in which the model shown first, or fourth, of four was ranked best; with no position effect each would be 25%. Choose a condition to highlight it. All numbers as reported [@kirk2026prism]."
```

Things to try:

- With *Simulated user* set to *Ranks transcripts only*, compare the top bar
  of *Agreement with people* with the bar below it: near-perfect agreement about which model
  is better overall, and a per-person agreement of 0.22.
- Switch to *Also holds the conversations*. Per-person agreement halves to
  0.11, and the share of trials in which the first-shown model wins rises from
  33.8% to 44.9%, against 24.1% for people.

For PBO that uses a language model as its preference oracle, this evidence
means the observation model is misspecified (inference). Language-model labels
are faithful in aggregate and wrong for individuals (PRISM-X, Kuric et al.),
too consistent (AlpacaFarm needed 25% random flips to reproduce human-like
over-optimization), dependent on presentation order (Wang et al.), and biased
toward the judge's own text (Panickssery et al.). A probit or Bradley-Terry
likelihood treats every deviation as independent noise with a fixed variance,
but most of these deviations are bias, and a position bias that varies with
the quality gap (Shi et al.) makes the noise variance depend on the utility
difference, which a fixed noise scale cannot express (@sec-obs-extensions lists
heteroscedastic models). The most direct remedy is the balanced position
calibration of Wang et al.: average the judgments from both orders. NAOD
[@du2026optimal], a preprint posted on September 30, 2026, is the first method
we found that puts judge bias into the design of queries: arguing that active
acquisition exposes judge bias that remains after calibration, it lowered
average proxy policy regret by 29.1% on Chatbot Arena data with 17 judges,
against a matched design that targets information alone, and showed that
representation error "can reverse an oracle design advantage". LILO does not
model judge bias.

As of September 2026 we found no study that runs the same PBO loop with human
comparisons and with language-model comparisons and reports regret or sample
efficiency for both, and none that measures how position bias propagates into
acquisition decisions (for example, whether averaging over both orders changes
the sequence of selected queries). Until such a study exists, a PBO result
obtained with a language-model oracle is a result about that oracle; the
minimal check is to rerun a few sessions with people and with both
presentation orders, and to report how the selected queries change
(inference). The gap is not peculiar to language models: in a retinal-implant
study within PBO, only about 50% of human choices agreed with a simulated agent
that was not a language model, yet 16 of 17 participants preferred the
optimized result [@schoinas2025evaluating] (@sec-health-neuro,
@sec-sw-simulated-users).

### The pattern {#sec-llm-pattern}

From PEBOL in 2023 to LILO in 2026, the model of uncertainty stayed classical
and small (PEBOL's Beta-Bernoulli model, the linear Bradley-Terry models of
OPEN and MAPLE, LILO's pairwise Gaussian process), and the language model
worked only at the interfaces: features, prior initialization, wording of
questions, and translation of labels. Every approach that handed query
selection to the language model fell behind classical algorithms (inference).

## Preference methods for language-model problems {#sec-llm-methods}

The second direction applies tools built for dueling bandits and PBO to
problems of language models: optimizing prompts, choosing among outputs and
models, choosing which comparisons to collect for training, and exploring
online during alignment. The evidence is mixed in a way that turns out to be
informative, and it resolves once one asks what each query is *for*.

### Prompt optimization {#sec-llm-prompts}

**APOHF** [@lin2024prompt] trains a neural network on frozen prompt embeddings
with a Bradley-Terry likelihood and pairs the greedy maximizer with the
maximizer of an upper confidence bound, the recipe of linear dueling bandits,
against baselines that included an ensemble with **double Thompson
sampling**, a dueling-bandit rule that draws two independent posterior samples
and pairs their maximizers (@sec-dueling-algorithms). It handles only pairs,
and all its "human" feedback was simulated: validation accuracy on 30
instruction tasks and similarity to a target image for text-to-image tasks. It
was an ICML 2024 workshop oral, was not accepted at ICLR 2025, and has an ACL
Rolling Review submission record from May 2026 with no formal acceptance.
@kayal2025bayesian (ICML 2025) use it as the motivating application of
"Bayesian optimization from human feedback". **PDO** [@wu2026llm] (ACL 2026
Findings) schedules duels between prompts with double Thompson sampling under
a fixed budget of language-model judgments and uses the winners to guide
mutation, finding prompts better than label-free baselines on BIG-bench Hard
and MS MARCO. Among the prompt optimizers for text we found, none was
evaluated with people; the study of APPO with people concerns images. (Ji, He, and Gu also call an unrelated
active-query algorithm APPO.)

### Choosing among outputs and among models {#sec-llm-selection}

@zhang2024generating (ICML 2024) replace pointwise scoring of reasoning steps
with pairwise comparisons by a language model, handling the noise with
ensembles and dueling-bandit variants. **Chatbot Arena** [@chiang2024chatbot],
the public leaderboard built from crowd votes between anonymous models, chooses
the next pair in proportion to the expected reduction in the width of the
Bradley-Terry confidence intervals; in simulations fitted to 213,576 held-out
votes, random sampling needed 54% more data to reach a precision of 0.2 on the
win-rate matrix, but only 5% more to reach 0.3 on the scores.
**Duel-Evolve** [@karlekar2026duel], an ICLR 2026 workshop poster, fits a
Bayesian Bradley-Terry model to a language model's judgments of its own
candidates, allocates comparisons by double Thompson sampling, and picks
parents for the next generation in evolutionary fashion, with no reward model
or labels; it scored 20
percentage points above comparable iterative methods on MathBench and 12 or
more on LiveCodeBench, against ground truth. Model routing has been cast as a
contextual dueling bandit solved with Feel-Good Thompson sampling, a variant
with an optimism term, with lower cumulative regret on RouterBench and
MixInstruct [@chiang2025llm] (a preprint not accepted at ICLR 2026);
@gharat2026cost (NeurIPS 2026) study best-arm identification under dueling
feedback with heterogeneous query costs, assuming a **Condorcet winner** (an
option that beats every other with probability above one half) and checking
that assumption on real data; CUPID [@nguyen2026cupid] (ICML 2026) chooses
which two models a user should compare, with a study with people; and T-POP
[@qu2026t] (ICML 2026) learns one user's reward online with a dueling bandit
to steer the decoding of a frozen model.

### Active preference data for reward models and DPO {#sec-llm-active}

Choosing which comparisons to collect for training has the most evidence.
@muldrew2024active (ICML 2024) favor pairs on which DPO's implicit preference
model is confident but wrong, and with GPT-4 as the oracle improved win rate by
1% to 6% on average. **BAL-PM** [@melo2024deep] (NeurIPS 2024) observes that
naive estimates of epistemic uncertainty (uncertainty from lack of data, as
opposed to noise in the answers) select redundant samples, adds the entropy of
the already-acquired prompts in the model's feature space, and needed 33% to
68% fewer labels on two human preference data sets. @mehta2025sample (COLM
2025) reduce dueling feedback to a contextual **Borda function**, the
probability that an action beats a uniformly random one, prove a polynomial
regret bound, and report that their methods can improve performance by over
13% relative to baselines under a
limited budget, with uncertainty from Monte Carlo dropout (keeping dropout on
at prediction time). @das2025active (ECML-PKDD 2025) prove that sampling
contexts uniformly can leave a constant suboptimality gap under a small budget,
give a lower bound of $\Omega(d/\sqrt{T})$ for $d$ dimensions and $T$ rounds,
and show that APO, which picks the most uncertain context, matches it up to
logarithmic and nonlinear factors.
@ji2024reinforcement (TMLR) prove a regret of $\tilde O(d^2/\Delta)$ and a
query complexity of $\tilde O(d^2/\Delta^2)$, where $\Delta$ is the gap between
the best and second-best actions; their ADPO matches DPO with about half as
many queries.

A branch based on optimal experimental design uses the **Fisher information**,
the expected curvature of the log-likelihood, whose inverse approximates the
covariance of the fitted parameters; a **D-optimal** design maximizes its
determinant, shrinking the uncertainty ellipsoid. It includes an offline design
with a lower bound matching up to constant and logarithmic factors
[@scheid2024optimal], D-optimal feedback for a DPO
linearized at the last layer [@kveton2025active] (both preprints), a Fisher
criterion at the reward model's last layer, with the finding that comparisons
across prompts raise labeling efficiency [@shen2025active] (ICML 2025), a log-determinant choice of negative
examples for the Plackett-Luce model [@surana2026mass] (a 2026 preprint), and
PBO-style acquisition in RLHF with a last-layer Laplace approximation
(@sec-laplace) [@cercola2025efficient] (published in 2026).

**ActiveUltraFeedback** [@melikidze2026activeultrafeedback] (ICML 2026) is the
most systematic comparison so far: a contextual dueling bandit over responses
from 12 families of open models, an ensemble on a frozen backbone, and a
language-model judge's Likert ratings standing in for people. It compared
random selection, two heuristics, three classical dueling-bandit rules
(**infomax**, the pair whose choice probability the ensemble disagrees on
most; double Thompson sampling; and MaxMinLCB), and two new rules that seek a
large quality gap: *double reverse Thompson sampling*, which pairs the
maximizer and minimizer of one posterior sample, and *DeltaUCB*. Models
fine-tuned on 5,000 to 10,000 samples chosen by the new rules outperformed
models trained on 60,000 samples chosen by random selection, by UltraFeedback,
or by the dueling-bandit rules. For training a reward model, though, random
selection was strong, and the authors conclude that there diversity is
preferable to a large quality gap.

### Thompson sampling for online alignment {#sec-llm-online}

@dwaracherla2024efficient (ICML 2024) brought double Thompson sampling and
**epistemic neural networks**, which output a family of predictions indexed by
a random input so that its spread expresses what the network does not know,
into feedback collection for language models. Queries were a prompt and two
Gemini Nano responses; the "human" was a simulator choosing by the
Bradley-Terry model from a reward model built on Gemini Pro and fitted to
Anthropic's helpfulness and harmlessness data; and the policy was approximated
by best-of-$N$ (keep the highest-scoring of $N$ responses). Double Thompson
sampling beat passive querying, Boltzmann exploration, and infomax, which did
well early but then fell far behind, which the authors attribute to its seeking
information whether or not it is useful; it reached passive querying's performance with an order of magnitude
less data. @asghari2026efficient, a 2026 preprint, scale this to a 9B Gemma
policy, with an epistemic reward network adding fewer than 5% to the
parameters, pairs chosen by the variance of the choice probability across
ensemble particles (information-directed selection), and a small positive bias
on each reinforcement signal, which the authors call an affirmative nudge. With fewer than 20,000
labels it matched what offline RLHF reached with 200,000. The factor of 1,000
in the paper is an extrapolation of curves, the ablations do not separate the
selection rule from the nudge, and there is no independent replication.

The theory is well developed: a near-optimal trade-off between regret and
queries for trajectory preferences [@wu2024making] (ICLR 2024; @sec-adj-pbrl);
FGTS.CDB [@li2024feel] (ICML 2024), the first posterior sampling algorithm for
linear contextual dueling bandits, with near minimax optimal regret
$\tilde O(d\sqrt{T})$; neural-tangent-kernel versions of the upper confidence
bound and Thompson sampling [@verma2025neural] (ICLR 2025); and an
$O(\sqrt{T})$ bound for Thompson sampling in online RLHF with general function
approximation [@feng2025thompson] (a 2025 preprint, without experiments).

Two applied versions followed. **SEA** [@liu2024sample] frames alignment as a contextual dueling bandit solved
with Thompson sampling on models of 1B to 6.9B parameters with DPO, IPO, and
SLiC, its preferences coming from a scalar reward model
(Skywork-Reward-Llama-3.1-8B) plus one experiment with GPT-4o-mini as judge;
it is a NeurIPS 2024 workshop poster, not accepted at ICLR 2025, NeurIPS 2025,
or ICLR 2026. warmPref-PS [@agnihotri2024online], a preprint not accepted at
TMLR, warm-starts posterior sampling with offline preferences from an expert
of unknown competence.

A parallel branch replaces posterior sampling with exploration bonuses: XPO
[@xie2025exploratory] (ICLR 2025), value-incentivized preference optimization
[@cen2025value] (ICLR 2025), self-exploring language models [@zhang2024self]
(TMLR), and a count-based bonus added to the DPO objective [@bai2025online]
(ICLR 2025). Under KL or $\alpha$-divergence regularization (a family that
contains the KL) these bonuses "unintentionally" bias exploration toward the
reference model's high-probability regions [@li2026general] (ICLR 2026); the
sample complexity of all existing online RLHF algorithms grows exponentially
with the scale of the reward [@chen2025avoiding] (NeurIPS 2025); and iterative
Nash preference optimization (@sec-llm-social) without explicit exploration can
depend exponentially on the KL parameter [@nan2026efficient] (a 2026
preprint).

### Random is hard to beat {#sec-llm-random}

@oh2026random, an ICLR 2026 workshop paper (I Can't Believe It's Not Better),
compared uncertainty-based active preference learning with random selection in
online DPO, across harmlessness, helpfulness, and instruction following, with a
reward model and a language-model judge as proxies. Active selection "yields
negligible improvements in proxy win-rates compared to Random"; while proxy win
rate rose, capability on standard benchmarks fell, and active selection neither
prevented that nor clearly reduced variance. The authors point to the strong
prior from web-scale pretraining and the "cheap diversity" of random on-policy
samples. Other results qualify the positive ones: naive uncertainty selects
redundant samples (BAL-PM), infomax fell behind after the early phase
(Dwaracherla et al.), quality-gap rules beat the classical dueling rules
(ActiveUltraFeedback), and Chatbot Arena's savings were 54% for one target and
5% for another. And the positive results simulate people with a reward model
(Dwaracherla et al., Asghari et al.), use accuracy or similarity as the utility
(APOHF), or use language-model judges (Muldrew et al., ActiveUltraFeedback,
PDO, Duel-Evolve); BAL-PM's savings come from human preference data sets, and
Chatbot Arena's from simulations fitted to real votes.

### What the query is for {#sec-llm-purpose}

The results can be reconciled once the purpose of a query is separated
(inference). The acquisition rules of dueling bandits and PBO (double Thompson
sampling, MaxMinLCB, qEUBO, infomax) were designed to *find or confirm the best
option*. Alignment data need pairs that *teach* a policy or a reward model: DPO
training benefits from a large quality gap (ActiveUltraFeedback), reward-model
training from diversity (the random-selection results of BAL-PM and
ActiveUltraFeedback), and offline methods from coverage of the policy's own
distribution (Oh et al., and Song et al. in @sec-llm-sample). Using a
find-the-best rule unchanged answers a different question from the one being
asked. Two further differences matter: alignment picks from a small,
policy-dependent candidate set rather than a whole design space, so much of
the disagreement between "active" and "random" may come from how diverse that
set already is (inference); and Asghari et al. have an explicit posterior over
rewards, whereas Oh et al. have only the implicit reward of a directly
optimized policy.

As of September 2026 we found no study that compares double Thompson sampling,
information-directed selection, quality-gap selection, and random selection
under the same budget, with human labels, for both reward-model RLHF and DPO.
Until one does, a team collecting preference data should keep a random arm as
its baseline and choose the active rule by what the data are for (inference).

## A shared likelihood and seven broken analogies {#sec-llm-analogies}

The two fields share an observation model and, for the online problem, a
decision-theoretic framing, so an acquisition rule from one field can run in
the other almost unchanged. That shared core is easy to over-read; this
section separates it from seven places where the analogy breaks.

### What is shared {#sec-llm-shared}

**The likelihood.** @zhu2023principled (ICML 2023) prove that with linear
rewards the maximum likelihood estimate converges under both the
Bradley-Terry-Luce and Plackett-Luce models but that a policy trained on it can
fail, whereas a pessimistic estimate works under a coverage assumption; that
for rankings of $K$ items the full Plackett-Luce estimate is "asymptotically
more efficient" than splitting the ranking into pairs; and that RLHF and
maximum-entropy inverse reinforcement learning share one analysis.
@sun2025rethinking (ICLR 2025) give convergence rates for Bradley-Terry reward
models on deep embeddings and argue that Bradley-Terry is "not a necessary
choice", since downstream optimization needs only an order-consistent reward.
@tang2024generalized (ICML 2024) show that DPO, IPO, and SLiC differ only in
their convex loss, which corresponds to the choice of link in PBO. On the PBO
side, BoTorch's `PairwiseGP` [@botorch2026pairwisegp] (software documentation)
and LILO use the probit link of Thurstone (@sec-thurstone, @sec-obs-baseline),
while Kayal et al. and PF-TS [@lazzaro2026finiteb] (AISTATS 2026) analyze the
Bradley-Terry-Luce model.

**The online decision problem.** @xiong2024iterative (ICML 2024) formalize RLHF
as a "reverse-KL regularized contextual bandit" with finite-sample guarantees,
and Ji et al., Mehta et al., and SEA adopt the contextual dueling bandit form.
Regret guarantees exist on both sides (Kayal et al. and PF-TS for PBO,
@sec-theory-bt; Wu and Sun, Ji et al., and APO for RLHF), and acquisition
rules transfer almost unchanged.

### Where the analogy breaks {#sec-llm-breaks}

**The objective.** By @eq-llm-implicit, the optimum of the KL-regularized
objective lets the policy network represent both the language model and the
implicit reward [@rafailov2023direct]. @korbak2022rl (EMNLP 2022 Findings)
observe that plain reinforcement learning fine-tuning leads to "distribution
collapse", whereas KL-regularized reinforcement learning is "equivalent to
variational inference", approximating a Bayesian posterior that updates the
prior language model with the reward as evidence. PBO returns the maximizer of
a latent utility; alignment returns the reference policy tilted exponentially
by the reward, and measures regret against that tilted policy.
@won2025differential, a 2025 preprint, argue that when preferences encode the
information needed to update a reference policy into a target, the log-ratio
reward is the only reasonable choice.

**The implicit reward as a utility estimate.** Reading DPO's implicit reward
like a Gaussian process posterior mean is not reliable: it fits the training
data as well as an explicit reward model but is 3% less accurate on average,
and up to 7% less, across five out-of-distribution settings
[@lin2024limited] (EMNLP 2024 Findings), and most preference-tuned models rank
the pairs of common preference data sets correctly "less than 60%" of the
time, with the DPO objective ill-suited to correcting even mild ranking errors
of the reference model [@chen2024preference] (NeurIPS 2024).

**Strong preferences.** When preferences are close to deterministic,
Bradley-Terry reward differences diverge and the KL regularization grows ever
weaker, so DPO overfits, especially when each pair appears only a few times
[@azar2024general] (AISTATS 2024). In PBO the Gaussian process prior
constrains the latent utility directly, so the problem does not arise in the
same form (inference). @tbl-llm-analogies lays out all nine aspects.

::: {.table #tbl-llm-analogies title="PBO and preference-based alignment, aspect by aspect: two analogies that hold and seven that break."}
| Aspect | PBO assumes | Preference-based alignment assumes | Holds? | Why, and the evidence |
|---|---|---|---|---|
| Observation model | probit link (`PairwiseGP`, LILO) or logistic Bradley-Terry (Kayal et al., PF-TS) | logistic Bradley-Terry; Plackett-Luce for lists; DPO, IPO, and SLiC correspond to logistic, squared, and hinge losses | holds | both are pairwise random utility likelihoods; the links differ only in the noise distribution (Zhu et al. 2023; Tang et al. 2024; Rafailov et al. 2023) |
| Online decision problem | a dueling bandit with cumulative or simple regret | a contextual dueling bandit with reverse-KL regularization | holds | the same acquisition rules apply (Xiong et al. 2024; Ji et al.; Mehta et al. 2025) |
| 1. Objective | the maximizer of the latent utility | the reference policy tilted exponentially by the reward | breaks | regret is measured against a tilted reference policy (Rafailov et al. 2023; Korbak et al. 2022) |
| 2. Sample complexity | conditional upper bounds: preference feedback of the same order as order-optimal scalar feedback (Kayal et al.) | $O(1/\varepsilon)$ rather than $O(1/\varepsilon^2)$ under KL regularization; existing online algorithms exponential in the reward scale; iterative Nash methods can be exponential in the KL parameter | breaks | the KL term to a reference policy, not the likelihood, drives the rates (Zhao et al. 2025; Chen et al. 2025; Nan et al. 2026) |
| 3. Strong preferences | the GP prior constrains the latent utility | near-deterministic preferences make reward differences diverge and the KL constraint loses force | breaks | nothing like a prior on the reward holds the fit in place (Azar et al. 2024) |
| 4. Utility estimate | a posterior mean and variance | DPO's implicit reward has no uncertainty; 3% less accurate out of distribution on average; ranking accuracy mostly below 60% | breaks | the implicit reward is a by-product of fitting a policy (Lin et al. 2024; Chen et al. 2024) |
| 5. Query distribution | any pair anywhere in the design space | candidates sampled from the policy or a model pool; offline methods need global coverage; exploration bonuses drift toward the reference model's high-probability region | breaks | what can be compared is limited to what the policy generates (Song et al. 2024; Li et al. 2026) |
| 6. Scale | tens to hundreds of comparisons, about 2 to 20 dimensions, Laplace or expectation propagation | $10^4$ to $10^6$ comparisons over sequences; ensembles, dropout, or Laplace | breaks | a summary across the studies in this chapter (inference) |
| 7. Respondents | one decision maker with a consistent utility | many annotators or language-model judges | breaks | a single reward fitted to many people raises problems of social choice (Gölz et al. 2025; Siththaranjan et al. 2024; Chidambaram et al. 2026) |
:::

Three of these differences, sample size, heterogeneity of the respondents, and
where the queries come from, can be seen by fitting the same likelihood in both
worlds, and @fig-llm-two-worlds also shows the first, what each world returns.

```{figure}
//| figure: llm-two-worlds
//| label: fig-llm-two-worlds
//| fig-cap: "One Bradley-Terry likelihood, P(a ≻ b) = sigmoid(r(a) − r(b)), fitted to simulated comparisons in two worlds, with options on one axis. The fit uses a Gaussian prior on 15 radial basis functions (a small stand-in for a Gaussian process) and a Laplace band. *One person*: 50 comparisons from one utility, pairs drawn uniformly anywhere, standing in for an acquisition function free to query anywhere. *Many annotators*: a million comparisons from two groups with different favorites, pairs drawn from a reference policy that never produces options in the shaded regions. The strip shows where comparisons fall and what each world returns: one design (the best fitted option among those that can be shown) or the reference policy tilted by exp(r/β), the optimum of KL-regularized alignment (@eq-llm-tilt), here with β = 1. Utilities, policy, and β are illustrative; PBO's most-used implementation uses the closely related probit link (@sec-obs-pairwisegp)."
```

Things to try:

1. Start in the one-person world and press *New draw* a few times. With 50
   comparisons the band is wide and the returned design moves around the
   person's favorite near 0.75. Drag $N$ to 1,000,000: the band collapses onto
   the person's utility.
2. Raise the second group's share to 0.4. The band stays thin, but the fitted
   reward matches neither group, and the error no longer shrinks as $N$ grows:
   one reward fitted to two groups is a compromise (@sec-llm-social).
3. Switch *Where pairs come from* to *From the policy*. Outside the region the
   policy covers, the band stays wide however large $N$ grows: no amount of
   data reaches options the policy never generates.
4. Switch the world to *Many annotators*. The orange curve, the reference
   policy reweighted by $e^{r/\beta}$, can only reweight options the policy
   produces and so never reaches group A's favorite near 0.75. Back in *One
   person* with pairs from the policy and $N = 50$, PBO restricted to someone
   else's candidates stops at the edge of what the policy produces
   (@sec-llm-purpose).

### Sample complexity and coverage {#sec-llm-sample}

Write $\varepsilon$ for how close to optimal the learned policy must be. A
sample complexity of $O(1/\varepsilon^2)$ means that halving $\varepsilon$
needs four times the data; $O(1/\varepsilon)$ means twice. @zhao2025sharp
(NeurIPS 2025) point out that earlier analyses of KL-regularized RLHF gave the
same $O(1/\varepsilon^2)$ as the unregularized problem, and that a sharper
analysis gives $O(1/\varepsilon)$. @song2024importance (NeurIPS 2024) prove
that a *global coverage* condition, roughly that the data contain every
response the optimal policy might produce, is necessary and sufficient for
DPO-like offline contrastive methods to reach the optimal policy, while online
reinforcement learning needs only partial coverage; step 3 above shows what a
lack of coverage looks like. The main difference between the theories of PBO
and RLHF therefore lies in the KL constraint to a reference policy and in the
policy parameterization, not in the preference likelihood (inference).

### Many annotators: from optimization to social choice {#sec-llm-social}

The difference in respondents pushes alignment toward **social choice**, the
study of how to combine many people's preferences into one decision
(@sec-econ-social-choice), which PBO has barely started on (@sec-many-users).
With cyclic preferences there may be no Condorcet winner. The **Borda count**
scores each option by how often it beats an opponent drawn at random; a **von
Neumann winner** is a probability distribution over options that beats or ties
every single option in expectation, and exists even when preferences cycle; a
**Nash equilibrium** of a two-player game is a pair of strategies neither
player can improve on alone. Nash learning from human feedback
[@munos2024nash] (ICML 2024) seeks a policy preferred to any opponent, the
Nash equilibrium of a two-player constant-sum game, and with arbitrary
preferences the target of preference-based reinforcement learning becomes the
von Neumann winner [@wang2023rlhf] (NeurIPS 2023). The von Neumann winner is
also a solution concept of the dueling-bandit literature, so dueling-bandit
results on intransitive preferences (@sec-dueling-bandits-intro) are the
natural counterpart of Nash learning on the PBO side (inference); we did not
check whether a paper from 2025 or 2026 makes this connection formally.

@golz2025distortion (NeurIPS 2025) measure alignment methods by their
**distortion**, the worst-case ratio between the best achievable average
utility and that of the learned policy, and prove that Nash learning achieves
minimax-optimal distortion while RLHF and DPO can have exponential or unbounded
distortion in the full setting; @oko2026distortion (ICML 2026) then prove that
under reward clipping the exponential degradation comes from a mismatch
between the preference data and the reference policy, not from the algorithm.
The dispute is open. What a single fitted reward does with many people has
also been characterized: with unobserved context, such as which annotator
answered, a single learned utility implicitly aggregates by Borda count
[@siththaranjan2024distributional] (ICLR 2024), as does the
Bradley-Terry-Luce loss [@an2026differential] (a 2026 preprint), which is the
compromise curve of step 2 above; with limited data per user, binary
comparisons cannot identify latent user types whereas rankings of three or
more items can [@chidambaram2026direct] (AISTATS 2026); and under Bayesian
marginalization or KL-robust optimization the effective reward of the
KL-regularized objective has a closed form, optimistic in the Bayesian branch
and pessimistic in the robust one [@hahami2026unifying] (a 2026 preprint). The
same reward posterior should then be used optimistically when choosing queries
and pessimistically when choosing what to deploy, and the acquisition
functions of PBO handle only the first (inference). @sec-phil-alignment takes
up what alignment to many people can mean.

## Common claims, checked {#sec-llm-claims}

@tbl-llm-claims checks common claims about how PBO relates to alignment.

::: {.table #tbl-llm-claims title="Common claims about PBO and language-model alignment, checked against the evidence."}
| Claim | What the evidence says |
|---|---|
| PBO and RLHF share the Bradley-Terry model. | Partly. Both use pairwise random utility likelihoods, but the default PBO implementation, `PairwiseGP`, uses the probit link. Logistic forms appear in the original PBO paper of González et al. [@gonzalez2017preferentialb], in POP-BO [@xu2024principledb] (ICML 2024), in MaxMinLCB, and in MR-LPF (the algorithm of Kayal et al.). |
| DPO shares a probit or Bradley-Terry likelihood with PBO. | "Probit" is wrong. The DPO loss is the logistic Bradley-Terry log-likelihood, with a Plackett-Luce version in the appendix. |
| Bradley-Terry is the Rosetta stone connecting BO, RLHF, and DPO; DPO's implicit reward is conceptually like a Gaussian process posterior. | True at the level of the likelihood, and only an explanatory analogy for the implicit reward, which carries no uncertainty (@tbl-llm-analogies). |
| RLHF scales but is query-inefficient; PBO is query-efficient but does not scale; a hybrid gets the best of both. | The direction holds, with qualifications. Hybrids exist (an ensemble posterior with double Thompson sampling or information-directed selection), but their posteriors are neural ensembles, not Gaussian processes, and the same kind of method gave negligible gains over random selection in online DPO. We found no work that uses a Gaussian process or kernel surrogate as the reward model in an RLHF loop at language-model scale. |
| Bringing PBO's uncertainty into RLHF, and warm-starting PBO with priors from language models, are open challenges. | Both have been pursued since 2024: the first by Dwaracherla et al., BAL-PM, and Cercola et al., with positive and negative results; the second by OPEN, MAPLE, and Eichelbeck et al., all but OPEN with simulated users. |
| Ji et al.'s RLHF with active queries appeared at NeurIPS. | The content is right (ADPO reaches DPO's level with about half the queries), but the venue is TMLR. |
| RLHF reward models are "notoriously poorly calibrated". | We could not substantiate this as stated, and it usually comes without a citation. @thies2026calibrated (ICML 2026) found that the calibration of RLHF reward models "correlates strongly but not perfectly with benchmark accuracy". |
| PBO uses about 10 to 1,000 comparisons and RLHF about 50,000 to 500,000 (or: tens to one or two hundred, against thousands to millions). | The contrast in order of magnitude holds, but neither range has a source. Studies of PBO with people use tens to one or two hundred comparisons; simulated high-dimensional PBO uses about 10 to 15 per dimension, about 1,500 at 102 dimensions [@menn2026local] (a 2026 preprint); alignment uses $10^4$ to $10^6$. |
| LiPO shows that the gains from listwise preference data grow monotonically with list length. | Partly. Only the LiPO-λ loss benefits monotonically from longer lists; a Plackett-Luce form of DPO does not [@liu2025lipo] (NAACL 2025). |
| PEBOL raises MAP@10 by up to 131% after 10 turns. | That figure (mean average precision at 10) appears only in the first arXiv version [@austin2024bayesianb]. The RecSys version reports mean reciprocal rank at 10 instead: at most 0.27, against 0.17 for the best monolithic baseline. |
:::

## What flows back {#sec-llm-flows-back}

The main direction of transfer has been from bandits and PBO into alignment:
double Thompson sampling, epistemic neural networks, information-directed
sampling, kernelized dueling bandits, and uncertainty-aware reward models.
Transfer back has so far been mostly framing, applications, and evaluation.
MaxMinLCB [@pasztor2024bandits] (NeurIPS 2024), a kernelized preference
bandit that casts the choice of a pair as a zero-sum Stackelberg game (one
player commits first and the other responds), motivates itself by noting that
such a model "has been employed in systems for fine-tuning large language
models", and its group then published RewardUQ [@yang2026rewarduq], a EurIPS
2025 workshop paper, and ActiveUltraFeedback. Kayal et al. use prompt
optimization as their application; PF-TS draws its two competitors
symmetrically, which its authors argue is often desirable, and sometimes
necessary, with human evaluators or language-model judges; and the multi-user
dueling bandit of @ahmed2026multi (TMLR) opens with the unfairness to minority
groups of training on average preferences in language-model fine-tuning, and
proves a lower bound on regret for fairness across
$D$ users of $\Omega(T^{2/3}\min(K, D)^{1/3})$ with $K$ arms. Ax 1.2.3 added an
abstraction for language-model messages, and 1.3.0 language-in-the-loop
labeling trials and qEUBO scheduling for PBO [@ax2026changelog] (software
documentation). Not everything that looks like a transfer is one: LILO extends BOPE
[@lin2022preferenceb] (AISTATS 2022), which predates the alignment wave, and
the abstracts of PABBO [@zhang2025pabbob] (ICLR 2025) and POP-BO do not frame
their problems in terms of language models.

### Three results ready to transfer {#sec-llm-transfer}

Three results from alignment could be used in PBO directly. We found no PBO
paper that takes DPO, LiPO, PAL, or an uncertainty-aware reward model as the
source of a method component, though we did not systematically scan the
reference lists of PBO papers from 2024 to 2026 (inference).

**Rankings carry more than pairs.** Zhu et al. proved that the full
Plackett-Luce estimate beats splitting rankings into pairs, and Chidambaram et
al. that rankings of three items identify latent user types that pairs cannot.
@katkuri2026pairwise, a 2026 preprint, learn rewards with Plackett-Luce from a
vision-language model's rankings and match or beat pairwise Bradley-Terry and
RL-VLM-F [@wang2024rl] (ICML 2024) on Meta-World; LiPO [@liu2025lipo] applies
learning-to-rank losses (@sec-adj-ltr) to listwise preference optimization;
and GraphDPO [@liu2026pairs], a 2026 preprint, generalizes preferences to a
graph. The PBO interfaces that ask for choices among several options, GimmBO
[@liu2026gimmbo] (SIGGRAPH North America 2026) and MultiBO
[@rajagopalan2026personalized] (ICML 2026), adopted lists for reasons of
interface design (@sec-query-forms) and cite none of these formal arguments.

**Separating noise from ignorance.** Several alignment papers split reward
uncertainty into an *aleatoric* part, the irreducible noise in the answers, and
an *epistemic* part, uncertainty from lack of data; @lou2024uncertainty, a 2024
preprint, use a probabilistic value head for the first and ensemble
disagreement for the second. Others quantify reward uncertainty to mitigate
over-optimization, with reward-model ensembles [@coste2024reward] (ICLR 2024)
or a Laplace approximation on LoRA weights, a small set of low-rank adapter
weights [@yang2024bayesian] (a workshop paper), and BNRM adds non-negative
factor analysis to the Bradley-Terry model [@duan2026mitigating] (ICML 2026);
RewardUQ found that model size and initialization matter more than the choice
of uncertainty method. Human comparison noise
is aleatoric, while acquisition by disagreement needs the epistemic part.

**Modeling the oracle's bias.** NAOD puts the judge's bias into acquisition
(@sec-llm-judges); LILO, which uses a language-model oracle, does not.

### Population priors from pluralistic reward models {#sec-llm-population}

Pluralistic and personalized reward models could serve as population priors for
a new user of PBO. PAL [@chen2025pal] (ICLR 2025) uses an ideal-point model, in
which each user and option is a point in a shared latent space and preference
falls with distance, with mixture modeling, and generalizes to new users from a
few samples. The variational preference learning of @poddar2024personalizing
(NeurIPS 2024) infers a latent vector per user and suffers posterior collapse
(the per-user latent stops carrying information) when each user has little data
[@kim2026swap] (ICLR 2026). LoRe [@bose2025lore] (COLM 2025) and @cai2026one
(SIGIR 2026) represent each user's reward as a weighted combination of shared
basis functions, and on a benchmark of
personalized reward models the best reached 75.94% accuracy
[@ma2026personalized] (COLM 2026). None chooses queries to learn a user's
weights, so active per-user elicitation on a meta-learned low-rank prior is the
natural next step for PBO to take from them (inference).

ActiveUltraFeedback's result also points to a distinction PBO has not yet drawn
explicitly: the pairs that help locate the optimum are not the pairs that help
learn a reusable utility model (inference). PBO work that builds population
priors from earlier users is in the second situation; HOMI, for example,
pretrains a prior from user models and was significantly better only at the
second and third iterations [@liao2026efficient] (CHI 2026).

## Settled, contested, missing {#sec-llm-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** PBO and preference-based alignment share pairwise
random utility likelihoods (Bradley-Terry, Thurstone, Plackett-Luce) and the
contextual dueling-bandit framing, and acquisition rules move between them
almost unchanged. The optimum of the KL-regularized objective is the reference
policy tilted by $e^{r/\beta}$, and DPO follows because the normalizer cancels
in Bradley-Terry [@rafailov2023direct]. The default PBO implementation uses the
probit link, DPO the logistic one. Language models used as simulated users and
judges are close to people in aggregate but unreliable for individuals, and
biased by position, length, and self-preference. Language models acting
directly as dueling-bandit optimizers trail classical algorithms in strong
regret [@xia2025numeric], and in scalar BO, agents performed no differently
when their observations were replaced by random labels [@gupta2025llms]. Only
two of the eight systems in @tbl-llm-systems are PBO in the strict
sense.

**Contested.** Whether active selection of comparisons beats random selection
for alignment: gains of 1% to 6% in win rate, 33% to 68% fewer labels, and an
order of magnitude in simulated settings, against negligible gains in online
DPO [@oh2026random]; the reconciliation by the purpose of the query is our
inference. Whether the exponential distortion of RLHF and DPO belongs to the
algorithms [@golz2025distortion] or to a mismatch between data and reference
policy [@oko2026distortion]. Whether the large label savings of Asghari et al.
hold, given ablations that do not separate their two components and no
replication. Whether the Bradley-Terry form is needed at all
[@sun2025rethinking].

**Missing.** A PBO loop run with human comparisons and with
language-model comparisons, reporting regret or sample efficiency for both. A
measurement of how a judge's position bias propagates into acquisition
decisions. A comparison, with human labels and a common budget, of double
Thompson sampling, information-directed selection, quality-gap selection, and
random selection for both reward-model RLHF and DPO. An evaluation with people
of any preference-based prompt optimizer for text. A Gaussian process or
kernel surrogate as the reward model of an RLHF loop at language-model scale.
PBO methods built from alignment components (Plackett-Luce
rankings, the aleatoric and epistemic split, oracle-bias models, pluralistic
priors with active per-user elicitation), and an acquisition function that
distinguishes pairs for finding the optimum from pairs for learning a reusable
utility.
:::

## Further reading {#further-reading .unnumbered}

- @rafailov2023direct derive DPO from the KL-regularized objective in a few
  pages; the derivation in @sec-llm-dpo follows theirs.
- @ouyang2022training describe the RLHF pipeline that made comparisons central
  to language models, with the details of how the comparisons were collected.
- @dwaracherla2024efficient and @oh2026random are the two sides of active
  preference collection for alignment; read them together.
- @melikidze2026activeultrafeedback is the most systematic comparison of
  acquisition rules for alignment data, and the clearest evidence that the
  purpose of a query changes which rule wins.
- @kobalczyk2026lilo is the cleanest example of a language model working only
  at the interface of a Gaussian process preference loop.
- @kirk2026prism, a preprint, measures in one experiment how a simulated user
  can be right about the population and wrong about each person.
- @siththaranjan2024distributional and @golz2025distortion explain what a
  single reward does when many people answer.
