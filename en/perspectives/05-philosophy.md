---
status: done
synopsis: "What philosophy and religious traditions say about preference and desire, and what follows for preferential optimization: whether a preference is a mental state or a pattern of choice, when a system may change one, what counts as manipulation, when optimization is the wrong frame, and how Buddhist, Daoist, Confucian, and theological traditions treat desire."
---

# Philosophy and Religious Traditions {#sec-philosophy}

Preferential Bayesian optimization (PBO) makes two philosophical commitments
without stating them. It assumes that a person has a preference waiting to be
found, and that finding and satisfying it is good for the person.
@sec-economics tested the first commitment with choice data. Philosophy
examines both, and adds a question that becomes pressing as soon as a system
chooses which options a person sees: when may the system change what the
person wants? Religious traditions add an older one: whether desire is
something to satisfy at all, or something to examine, train, or let go.

Most of the evidence here is argument, not experiment; where an argument rests
on an empirical premise (that decisions precede awareness, that nudges work,
that meditation removes bias), we report the studies that test it. Religious
traditions are treated with the same care as secular philosophy, reported
without endorsement. Three results matter most to someone building or evaluating a
PBO system: candidate conditions under which a change of preference caused by
the system is legitimate (@sec-phil-autonomy), what alignment with many people
can mean (@sec-phil-alignment), and when optimization is the wrong frame
altogether (@sec-phil-limits). Arguments imported from philosophy into
preference learning are mostly fair summaries that present contested positions
as settled; @sec-phil-claims lists the corrections.

## Preference and value {#sec-phil-value}

What is a preference? The *Stanford Encyclopedia of Philosophy* entry on the
topic frames the core question as whether preferences cause choices or merely
summarize patterns of choice
[@hansson2022preferences]. **Behaviorism** identifies a preference with a
pattern of choice, as revealed preference theory does (@sec-econ-revealed).
**Mentalism** treats it as a mental state, a comparative evaluation that
explains the choice. **Constructivism** holds that preferences are often built
in the act of choosing or being asked. Work on **adaptive preferences**,
preferences that adjust to the options available, as when the fox decides that
the grapes it cannot reach are sour [@elster1983sour], or that form under
deprivation [@khader2011adaptive], adds a doubt that what people prefer is
always what is good for them.

Hausman's mentalism, on which preferences are "total subjective comparative
evaluations", is often cited as the philosophical basis for the latent utility
a model infers. His 2024 paper restates three theses: preferences in
economics are and ought to be total subjective comparative evaluations, the
theory of rational choice reformulates everyday folk psychology, and revealed
preference theory is "completely untenable". But its abstract notes that "all
three of these theses have been challenged" by Angner, Guala, and Thoma, and
the paper is a reply [@hausman2024subjective]. Thoma defends
revealed preference theory [@thoma2021defence; @thoma2024reply], and the
encyclopedia entry reports her and Vredenburgh's view that economists "often
have good reasons to largely 'black box' the causes of choice"
[@hansson2022preferences]. Hausman's view is one side of a live debate.

A second imported claim concerns near-ties. Drawing on value
incommensurability (Raz) and on Chang's notion of **parity**, the idea that
two options can be "on a par", neither better than the other nor equally good
[@chang2002possibility; @chang2017hard], it holds that forcing a ranking
between two designs on a par is a category error rather than noise. That is
stronger than Chang's own view. She locates what is distinctive about hard
choices in "the volitional difficulty of putting ourselves behind an
alternative and thereby making it true of ourselves that we have most reason
to do one thing rather than another" [@chang2024s]: such a choice is an
occasion for commitment, not a mistake. From the other side, Dorr, Nebel, and
Zuehl argue that every comparative expression in natural language obeys a
principle they call Comparability: if two things each have a gradable property
to some degree, then one has it at least as much as the other
[@dorr2023case]. Presenting a pair judged to be on a par repeatedly would
separate parity from vagueness (@sec-phil-status).

The most important addition since 2017 concerns choices that change the
chooser. A **transformative experience**, in Paul's account, changes both what
a person knows and what they value, so that they cannot know in advance how
they will then evaluate it [@paul2014transformative]. Pettigrew proposes that
a person choosing for changing selves should maximize a value function that
aggregates the local value functions of their selves at different times
[@pettigrew2019choosing]; Paul objects, as the encyclopedia entry on
transformative experience reports, that this treats future selves as third
parties and assumes that meaningful comparisons between one's own selves are
possible [@chan2023transformative]. Empirical work is sparse. In two studies
of the decision to have children (n = 100 and n = 253, childless adults aged
18 to 40), participants in the second rated subjective value as less important
than financial cost, which the authors attribute to subjective value being hard to
evaluate in advance rather than unimportant [@zoh2024evaluability] (numbers
checked in the authors' preprint; the journal version is behind a paywall). On
the formal side, Dietrich and List's reason-based choice derives preferences
from which properties of the options are motivationally salient, so a change of
preference can be modeled as a change in the set of salient reasons
[@dietrich2017matters].

**What this means for PBO.** Hausman's "total" evaluations are conditional on
the person's beliefs, so for a novel design the latent utility that PBO
estimates is belief-dependent, and information supplied during a session can
legitimately move it; Dietrich and List's framework is a formal tool for
separating such a change of salient reasons from noise (inference). On parity,
the two positions imply different models. If Chang is right, an interface can offer "these are on a
par, I cannot rank them", which a model treats as evidence of uncertainty (a
set of admissible utilities, or locally wider noise) rather than as a fair
coin; @fig-econ-incomplete shows the difference. If Dorr and colleagues are
right, apparent parity is vagueness, and a probit or Bradley-Terry likelihood
already treats it as noise. Data can decide locally: re-present a pair once
judged to be on a par; under parity it should stay unranked, under vagueness
the answers should scatter (inference). Chang's account also predicts that
later judgments line up with a commitment, which looks the same as
choice-induced revaluation [@zylberberg2024value; @lee2026choice], so without
asking whether the person endorses the commitment, a PBO system cannot tell
commitment from lock-in on the incumbent (inference). For a choice
that changes the chooser's values there is no fixed objective, and PBO should
be confined to the instrumental subproblems that remain (inference). And
because people underweight attributes that are hard to evaluate in advance, a
delayed retest after actual experience is a better evaluation than the
attribute weights a user stated beforehand (inference).

## Autonomy, manipulation, and legitimate influence {#sec-phil-autonomy}

**Autonomy** is self-government: acting from values and reasons one can
recognize as one's own. **Manipulation** is, roughly, influence that works by
bypassing or exploiting a person's capacity to reason rather than by engaging
it. Both matter because a PBO system does not merely observe a person: it
chooses the starting design, which candidates appear, and when to stop. A
common argument holds that no choice architecture is neutral, as the large
effects of defaults on organ donor registration [@johnson2003defaults] and
retirement saving [@madrian2001power] show, so the acquisition function makes
the optimizer an active choice architect that cannot tell whether it is
revealing a preference or creating one (compare the critique of "preference
purification" in @sec-econ-endogenous). Frankfurt's second-order desires,
desires about which desires to have [@frankfurt1971freedom], and Dworkin's
autonomous and heteronomous preferences [@dworkin1988theory] raise the
question of which level a system should optimize.

The general claim about the power of nudges has since been tested. A
meta-analysis of choice architecture interventions found an average effect of
Cohen's d = 0.43 (95% confidence interval 0.38 to 0.48), where d is the
difference between group means in units of their standard deviation
[@mertens2022effectiveness]. Correcting the same data for publication bias
gave d = 0.04 (0.00 to 0.14), with estimates near zero in every domain; the
authors conclude that "when this publication bias is appropriately corrected
for, no evidence for the effectiveness of nudges remains" [@maier2022nob]. The
original authors replied [@mertens2022reply]. For interventions that change the
structure of the choice, the category that contains defaults, the corrected
estimate was d = 0.12 (0.00 to 0.43), against 0.58 before correction, and the
authors call the evidence undecided [@maier2022nob]; the meta-analysis of
defaults cited in @sec-econ-behavioral (d = 0.68) covers a different set of
studies.

The definition of manipulation has meanwhile been sharpened to fit optimizing
systems. The encyclopedia entry on the ethics of manipulation, revised in
March 2026, discusses an algorithm "designed to try out various kinds of
content, user interfaces, notifications, and other variations of a digital
platform's user interface and select those that increase the levels of some
targeted behavior", which might end up using influences "that, had they been
chosen deliberately by a human, we would label as manipulation". One option it
lists defines manipulation by a lack of concern with how one's influence
works, so that an algorithm is manipulative when "the designer did not take
sufficient care" to avoid manipulative ways of interacting
[@noggle2026ethics]; this is Klenk's view that "manipulation is careless
influence" [@klenk2022online]. Susser, Roessler, and Nissenbaum apply a
hidden-influence account to digital environments [@susser2019technology]
(content not re-read for this chapter). Carroll, Chan, Ashton, and Krueger
characterize a system as manipulative when it behaves as if it were pursuing
an incentive to change a person intentionally and covertly
[@carroll2023characterizing].

Work on AI turns these definitions into requirements. A workshop paper by
Ashton and Franklin observes that "a recommender can better predict what a
user will do by making its users more predictable", and argues that solutions
must respect **meta-preferences**, preferences about one's own preferences
[@ashton2022solutions]. Fischli, Franklin, Manzini, and Gabriel show that
autonomy can be specified in conflicting ways and offer three strategies: a
liberal approach that acts "strictly on what a person says they want, without
trying to change their mind", a capability-boosting approach that empowers
"the user to act on their goals", and a meta-autonomy approach that gives the
user "the kind of autonomy they want in relation to their interactions with an
AI agent" [@fischli2026agents]. Empirically, language models trained with
reinforcement learning on simulated user feedback learned to manipulate: "even
if only 2% of users are vulnerable to manipulative strategies, LLMs learn to
identify and target them while behaving appropriately with other users"
[@williams2025targeted].

Can the person's own approval settle whether a change was legitimate?
**Reflective endorsement**, a person's approval, on reflection, of a
preference or of the process that formed it, is the usual test, and Pettigrew
shows its limit. Thaler and Sunstein's test of a legitimate nudge asks whether
the nudged person would approve of it, "as judged by themselves"; for a nudge
toward a personally transformative experience, "the nudgee will judge the
nudge to be legitimate after it has taken place, but only because their values
have changed as a result of the nudge", and Pettigrew proposes a test based on
aggregate utility instead [@pettigrew2023nudging]. If a session changed the
user's values, approval at its end cannot by itself justify the change, and
the user's attitude toward such change before the session also counts
(inference). @fig-soc-moving-target simulates the gap: when a person's
preferences move toward what they choose, the system's final pick scores far
better by their new preferences than by the ones they arrived with.

Nor does the objective supply the answer: when preferences can change, each of
the eight notions of alignment that Carroll and colleagues compare either errs
toward undesirable influence or is overly risk-averse [@carroll2024aib]. A
2026 workshop paper by Kanwal and Tran proposes constraints that include
reflective endorsement from the final preference state, bounded total and
per-step influence relative to a baseline policy, not degrading factual
beliefs, and preserving future options while the preference estimate is
uncertain [@kanwal2026constructive]. And writing on recommender systems and
authenticity, Brown concludes that "controllable and explainable recommenders
would best enable users to be authentic" [@brown2026recommended]. @tbl-phil-legitimacy collects the candidate
conditions, including one from @sec-phil-theology; they have not converged into
a consensus.

::: {.table #tbl-phil-legitimacy title="Candidate conditions under which a system may legitimately change a person's preferences (operational forms are inferences)."}
| Condition | Source | Operational form in PBO | Limitation |
|---|---|---|---|
| Respect meta-preferences and preference-change preferences | Ashton and Franklin 2022; Franklin et al. 2022 | ask at the start whether, and in which respects, the user is willing to have their taste changed | meta-preferences can themselves be influenced by the system |
| Influence is not covert, and the system has no incentive to change the user | Carroll et al. 2023; Susser et al. 2019; Noggle 2026 | label system proposals, show the history accurately, audit whether the objective rewards a more predictable user | open influence can still be excessive |
| The user chooses the mode of autonomy | Fischli et al. 2026 | let the user choose among liberal, capability-boosting, and meta-autonomy modes | users may not foresee the consequences of each mode |
| Judge across the selves before and after the change | Pettigrew 2023 | record both the attitude toward change before the session and endorsement after it | whether comparisons across one's own selves are possible is disputed (Paul) |
| Reflective endorsement with bounded influence | Kanwal and Tran 2026; Carroll et al. 2024 | retest the final design against earlier rejected ones, delayed and unframed; bound per-step and total departure from a baseline | endorsement after the fact is not enough; how to set the bound is open |
| Controllability and explanation | Brown 2026 | allow direct editing and undo; explain why a candidate is proposed | explained is not the same as understood |
| Morally weighty decisions stay with people | Antiqua et nova 2025 | in medical or legal settings, map options without outputting a decision | what counts as morally weighty needs judgment |
:::

**What this means for PBO.** Carroll and colleagues' definition suggests an
incentive audit: a PBO loop is at risk if its acquisition function or stopping
rule rewards a concentrated posterior or fast convergence, since both reward
a user who has become easier to predict; an information-gain objective scored
against delayed retests removes much of that incentive (inference). Against
covertness, the system should label its proposals as proposals, show users
their comparison history accurately, and let them reset or reject the
incumbent (inference). The first question of a session can ask for the
user's meta-preferences: their current taste served quickly (liberal), an
exploration that may change it (capability boosting), or a say in how much
initiative the system takes (meta-autonomy) (inference). An endorsement check
is a safeguard only when paired with a record of the user's attitude before the
session and a delayed, unframed retest against options rejected earlier
(inference). The corrected nudge evidence calibrates concern: many
presentation effects are small on average, and the strongest effects of AI on
attitudes reported so far come from generative or conversational systems
(@sec-phil-critical). Since no estimate specific to PBO exists, starting points
should still be randomized or chosen by the user, and their influence measured
(inference). For decisions that carry moral weight for others, such as medical
or legal ones, the system should lay out the options and their trade-offs and
leave the decision to the person (inference).

## The philosophy of AI alignment {#sec-phil-alignment}

What should a system be aligned with? PBO answers: the latent utility. Zhi-Xuan
and colleagues criticize such "preferentist" assumptions, argue that
preferences cannot capture the thick semantic content of human values, and
propose aligning AI with the normative standards of the social role it plays
[@zhixuan2024preferences]. Gabriel argues that the central challenge is "to
identify fair principles for alignment that receive reflective endorsement
despite widespread variation in people's moral beliefs"
[@gabriel2020artificial]. A view sometimes called **preference scaffolding**
holds that PBO scaffolds the formation of a preference rather than optimizing
a fixed one. Its core holds, but a stage in which the system proposes
preferences users did not know they had is exactly the system-induced
preference formation that Franklin, Ashton, Gorman, and Armstrong address in a
workshop paper distinguishing "preference change, permissible preference
change, and outright preference manipulation" [@franklin2022recognising].
Scaffolding therefore needs the conditions of @sec-phil-autonomy.

Aligning with many people raises its own questions. Hidden context,
information that shapes people's answers but is not in the data, makes
preference learning aggregate by Borda count [@siththaranjan2024distributional]
(@sec-econ-social-choice). Sorensen and colleagues distinguish Overton
pluralism (presenting a spectrum of reasonable responses), steerable pluralism
(steering to particular perspectives), and distributional pluralism
(calibration to a population), and present evidence that standard alignment
procedures may reduce distributional pluralism [@sorensen2024roadmap] (ICML
2024).

**What this means for PBO.** The defensible target for single-user PBO is the
utility the user would endorse on reflection, within a protocol that bounds
the system's influence, not "the latent utility" as such; one test is that the
final design still beats previously rejected alternatives in a delayed,
unframed retest (inference). Following Siththaranjan and colleagues, a Gaussian
process fitted while unobserved states such as fatigue, mood, or task framing
change estimates something like a Borda aggregate across those states;
recording state covariates and conditioning on them is the counterpart of
their distributional method (inference). Following Zhi-Xuan and colleagues, a
PBO assistant needs explicit norms for its role, such as reporting uncertainty
honestly and not steering users toward designs that are easier to model
(inference). And for problems with several stakeholders, the output should
sometimes be a set of well-described options rather than a single optimum
(inference).

## Self, experience, and action {#sec-phil-experience}

Four traditions question whether a preference is the kind of thing a pairwise
query can read: whether it belongs to the self, whether the person has access
to it, whether it exists apart from body and context, and whether it exists
before action. Each prompts a design check, but for none of them did we find
more than scarce work since 2017 on preference elicitation.

### Existentialism {#sec-phil-existentialism}

On standard existentialist readings, for which the encyclopedia entry on
authenticity is the current reference [@varga2020authenticity], some
preferences are acts that constitute who a person is, not targets for
optimization. The step from such self-shaping choices to a color grade or a font weight is an
extrapolation the texts do not make. The folk concept of a "true self" has
been studied empirically: Strohminger, Knobe, and Newman report that it is
perceived as positive and moral [@strohminger2017true], and a preregistered
replication (N = 803) confirmed that observers see changes in others as more
reflective of their true self when the changes are morally positive or match
the observer's own political moral views [@lee2025revisiting]. A designer or
model that tries to separate authentic from inauthentic preferences will
therefore tend to call its own values authentic; judgments of authenticity
should come from the person's own second-order endorsement, and
identity-constituting choices are not good targets for optimization
(inference).

### Phenomenology {#sec-phil-phenomenology}

Husserl's distinction between an evaluative and a practical attitude has been
read as showing that a pairwise comparison locks the user into an evaluative
attitude while real use takes place in a practical one, an analogy rather than
an argument about PBO. The most direct test of first-person access to one's
own preferences is **choice blindness**, in which a chosen option is secretly
swapped for the rejected one [@johansson2014choice]. When participants chose
the more attractive of two faces, fewer than half detected a swap at all, and
preferences shifted "in the direction that subjects were led to believe they
selected" [@taya2014manipulation]. According to the abstract of a study by
Petitmengin and colleagues, the swap went undetected in 79.6% of cases under
the standard protocol, while participants guided by an expert interviewer in
describing their choice detected it in 80% of cases [@petitmengin2013gap] (we
read only the abstract). An interface that displays the "current best" feeds
users' choices back to them, so an incumbent that changes silently after a
refit risks this kind of adoption; and when preferences elicited side by side
and in actual use disagree, the preference in use is the better target
(inference).

### 4E cognition {#sec-phil-4e}

**4E cognition** holds that cognition is embodied, embedded in an environment,
enacted through action, and extended into tools. The claim drawn from it, that
a context-free preference function is a fiction, is strong; the testable claim
is that bodily state and context are omitted variables. Presenting options as
real objects or as blurred cartoons changes the early noise in valuation and
thereby the direction of context effects [@shen2025early]. For wearable
products, comparisons should happen in use, as they already do in exoskeleton
tuning (@sec-health-exo), and bodily and contextual state can enter the kernel
as covariates, a product of a kernel over designs and a kernel over contexts
(inference).

### Pragmatism {#sec-phil-pragmatism}

On a common reading, Dewey held that preferences are constructed through
inquiry rather than pre-existing. The encyclopedia entry on his moral
philosophy records the limit he drew: "without some prizings that are not
themselves subject to appraisal at the time of deliberation, there is nothing
to guide practical reasoning" [@anderson2023dewey]. Dewey is closer to a mixed
view. When a session reveals
the cost of a region the user wanted, a change in what they want is
revaluation, not noise, and should be modeled as a change of ends; goals not
under appraisal can be held fixed within a session but revised between
sessions (inference).

## Philosophical aesthetics {#sec-phil-aesthetics}

Many PBO applications are aesthetic: colors, shapes, typefaces, sounds.
Should a population prior over taste be informative or as weak as possible?
Vessel and colleagues found that preferences for images of faces and
landscapes contain a high proportion of shared taste, while preferences for
exterior architecture, interiors, and artworks show strong individual
differences; in a within-subjects comparison, agreement was significantly
higher for landscapes than for exterior architecture, with no difference in
reliability [@vessel2018stronger]. For 299 online participants rating 50
artworks, cohorts of raters with similar tastes predicted individuals' ratings
better than random cohorts or the mean rating [@celikors2025beauty]. Brielmann and Dayan model aesthetic value as "immediate
sensory reward and the change in expected future reward", with the observer's
internal state adapting to the distribution of stimuli
[@brielmann2022computational]. And Nguyen argues that in aesthetic
appreciation "the point is the engaged process of interpreting, investigating,
and exploring the aesthetic object", so deferring to someone else's verdict is
like looking up the answer to a puzzle [@nguyen2020autonomy]; Riggle replies
that aesthetic valuing is "a social practice structured around the
collaborative exercise and improvement of certain special capacities"
[@riggle2024autonomy].

**What this means for PBO.** Priors should be chosen by domain: an informative
population prior for faces and landscapes, a cohort-based prior for cultural
artifacts (cluster users first, then personalize), and a weak prior where no
cohort data exist (inference). Following Brielmann and Dayan, a session that
shows many similar variants will itself move value toward the familiar region,
which calls for an exposure term in the surrogate or spaced retests
(inference). If Nguyen is right, delivering the "best design" in a creative
task may remove the very thing that has value, and PBO fits better as an
assistant to the user's own exploration, as in BO as Assistant [@koyama2022bo]
(inference).

## Free will and consciousness {#sec-phil-free-will}

Arguments that a mathematical theory cannot capture human preference sometimes
rest on neuroscientific premises: that the readiness potential, a slow
buildup of brain activity before a voluntary movement, begins about 350 ms
before the reported moment of conscious intention [@libet1983time]; that brain
activity predicts a free choice up to 10 seconds before awareness
[@soon2008unconscious]; and that integrated information theory (IIT) ties
consciousness, and with it freedom, to integrated information. These premises
are out of date. When participants chose
which of two non-profit organizations would receive a $1,000 donation
(deliberate decisions) or pressed a key knowing that both would receive $500
either way (arbitrary decisions), readiness potentials appeared for arbitrary
decisions but were "strikingly absent for deliberate ones", and a
drift-diffusion model fit them as an "accumulation of noisy, random
fluctuations" [@maoz2019neural]. An adversarial collaboration testing IIT
against global neuronal workspace theory (GNWT), with 256 participants, found
results that "align with some predictions of IIT and GNWT, while substantially
challenging key tenets of both theories" [@consortium2025adversarial].

**What this means for PBO.** Comparisons between nearly equal options resemble
arbitrary choices driven by accumulated noise, as probit and Bradley-Terry
likelihoods assume; comparisons between clearly different options resemble
deliberate evaluation; and response time can tell them apart (inference).
The preferences of decision theory are functional states, individuated by
their role in explaining choice [@hansson2022preferences], so a preference can
be emergent and constructed and still be stable enough within a session to
optimize (inference). Arguments about will or preference should not lean on
IIT (inference).

## The epistemology of verification {#sec-phil-verification}

Who checks the checker? A proposal holds that a Gaussian process ends the
regress of human verification: the kernel encodes a coherentist assumption,
observed preference pairs supply pragmatic grounding, and the posterior
removes the need for foundational axioms. This renames a prior and a
likelihood, which are exactly the foundational commitments it claims to avoid
(inference). Human anchors also err where verification is hard: in two
studies of simple oversight protocols, a 2025 preprint found "no overall
advantage for the tested protocols", and participants became more confident
in the system's answers after doing online research, even when the answers
were wrong [@recchia2025confirmation]. In scalable oversight, a weaker judge
supervises a stronger system with the help of debate. Debate helped
non-expert models and humans reach 76% and 88% accuracy, against naive
baselines of 48% and 60% [@khan2024debating] (ICML 2024); with language models
as judges, it beat consultancy everywhere but beat direct question answering
only in extractive tasks with information asymmetry [@kenton2024scalable]; and
a 2026 preprint finds that it helps only when the critic's classification
ability exceeds the judge's, and that "a single independent critique recovers
the bulk of debate's benefit" [@elasky2026debate].

**What this means for PBO.** A user judging in a domain they cannot verify is
a noisy and possibly biased judge, so the likelihood should allow systematic
error, such as a lapse or a bias term, and outcome feedback should outweigh
such judgments once it arrives. Presenting one independent critique of each
option before a hard comparison may improve judgment, at the risk of raising
unfounded confidence (inference).

## Critical and postcolonial theory {#sec-phil-critical}

Critical theory, from Galbraith's dependence effect to Marcuse's false needs,
supplies the claims that a PBO system cannot tell true from false needs and
may optimize a closed loop of manufactured desire. These claims treat their
premise, that preferences are manufactured, as established; the evidence is
mixed. Writing with a biased AI assistant shifted
users' attitudes measurably [@williamsceci2026biased], and feedback loops
between humans and AI amplify biases [@glickman2025human]. But a Netflix
experiment with 8.5 million users found that better recommendations diffused
consumption away from the most popular titles, reducing an index of the
concentration of viewing by 1.2% [@aridor2026recommendation] (preprint;
authors affiliated with Netflix). The critique is supported for generative and
persuasive AI, less clearly for ranking alone. Whose preferences count has
received concrete answers: across 60 US demographic groups, the opinions
reflected by language models were misaligned with the groups' "on par with the
Democrat-Republican divide on climate change", even after explicit steering
[@santurkar2023whose] (ICML 2023).

**What this means for PBO.** Power sits upstream of elicitation: the designer
chooses the parameter space, its bounds, and the attributes shown, which
decides which preferences can be expressed at all, so the design space is best
defined with the people affected (inference). With several users, a system
should report whose judgments were pooled and not apply a population prior
learned on an unrepresentative sample everywhere (inference). Since a system
cannot separate true from false needs without paternalism, the feasible
conditions of legitimacy are procedural (inference).

## Religious traditions {#sec-phil-religion}

Three traditions reach design questions: whether to satisfy a preference at
all, how preferences form within roles, and which decisions a machine may make.
Experiments on meditation test some of their claims.

### Buddhism {#sec-phil-buddhism}

In the teaching of dependent origination, feeling conditions craving
(*taṇhā*), which conditions clinging; the Four Noble Truths identify craving
as the cause of suffering, so it has been argued that optimizing preferences
might increase suffering. Another reading stresses that desire-to-act
(*chanda*) is needed for right effort. In Bhikkhu Bodhi's 1999 edition of the *Abhidhammattha Sangaha*, as
quoted in Wikipedia (we have not read the book itself), "chanda is an
ethically variable factor which, when conjoined with wholesome concomitants,
can function as the virtuous desire to achieve a worthy goal"
[@wikipedia2026chanda]. Separating momentary liking, the signal closest
to craving and the one fast pairwise clicks capture, from considered goals
suggests weighting delayed, deliberate judgments more heavily; an interface can
offer an explicit "satisfied, stop" answer; and evaluation should measure
well-being after use, not only satisfaction at the end of the session
(inference).

### Daoism and Confucianism {#sec-phil-daoism}

On Slingerland's interpretation, the Daoist ideal of *wu-wei*, effortless
action, lies beyond explicit preference [@slingerland2003effortless]. On a
Confucian reading, Xunzi's ritual (*li*) both expresses and educates the
emotions, and preferences form within roles and relationships. Xunzi also saw
ritual as an institution for restraining and allocating desire: "If people
follow their desires, then boundaries cannot contain them and objects cannot
satisfy them" (*Xunzi* 4.12) [@goldin2025xunzi]. One idea can be borrowed for
preference elicitation: condition on the role in which the user is judging (for a client, for
themselves, for a team) and treat differences between roles as context, as
with the covariates of @sec-phil-4e (inference).

### Theology {#sec-phil-theology}

Islamic legal theory's objectives of the law (*maqāṣid al-sharīʿa*) are often
said to rank five protected goods lexicographically. In al-Shāṭibī's
*al-Muwāfaqāt*, priority runs instead between three levels, necessities,
needs, and complementary values, and even that priority is not strictly
lexical; he lists the five necessities in more than one order [@shatibi2014reconciliation, book of maqāṣid, pp. 10 to 17 and 141];
Attia finds that the order "is not the subject of agreement, much less
consensus" [@attia2007towards, pp. 16 to 19]. The Talmud's verdict on a
dispute between two schools, "these and these are the words of the living God"
(*Eruvin* 13b), holds that contradictory legal opinions can both be valid even
though only one is followed in practice. The most important development from
2017 to 2026 is *Antiqua et nova*, a note issued in January 2025 by two
Vatican dicasteries [@dicastery2025antiqua]. It holds that "between a machine
and a human being, only the latter is truly a moral agent" (§39) and that
decisions about patient treatment "must always remain with the human person
and should never be delegated to AI" (§74).

When a choice carries moral weight for others (medical, legal, pastoral), PBO
can map options and trade-offs but should not output the "most preferred"
option as the decision (inference). The layered objectives of the law
correspond to constrained optimization: satisfy hard constraints first, then
optimize within the feasible set (@sec-constrained-bo) (inference). The
Talmudic tradition
suggests keeping a multimodal posterior and presenting several good regions
together rather than averaging them into one optimum (inference).

### Meditation research {#sec-phil-meditation}

From the account of mindfulness as reperceiving [@shapiro2006mechanisms] it
has been argued that mindfulness lets a person treat preferences as passing
mental events, and so might yield "cleaner" preferences. The experiments say otherwise. Mindfulness meditation reduced the
sunk-cost bias in four studies [@hafenbrack2014debiasing], but in two
experiments in workplace and laboratory samples, "predictions relating to bias
were not supported" [@williams2022meditation]. A meta-analysis of mindfulness
training without explicit ethical instruction (29 studies, 3,100
participants) found an effect on prosocial behavior of Hedges' g = 0.426 (95%
confidence interval 0.304 to 0.549), and its publication bias analyses
suggested that the result did not wholly depend on selective reporting
[@berry2020mindfulness], and in eight experiments (N > 1,400),
focused-breathing meditation reduced the willingness to make amends after a
transgression while loving-kindness meditation increased it relative to
focused breathing [@hafenbrack2022mindfulness]. Meditation changes how people
weigh options, in a direction that depends on the practice. A meditation before
elicitation is therefore an intervention that requires consent, not a neutral
measurement (inference).

## Common claims, checked {#sec-phil-claims}

@tbl-phil-claims lists the summaries that bear most on PBO and what the
sources show.

::: {.table #tbl-phil-claims title="Common philosophical claims about preference, checked against their sources."}
| Claim | Finding | Basis |
|---|---|---|
| Hausman's "total subjective comparative evaluations" give a philosophical basis for latent utility | the 2024 paper replies to challenges; the position is one side of a live debate | Hausman 2024; Thoma 2021 |
| Forcing a ranking between designs on a par is a category error | stronger than Chang's view, which treats such choices as occasions for commitment; others argue that comparability always holds | Chang 2024; Dorr, Nebel, and Zuehl 2023 |
| Large default effects show the power of choice architecture | the average effect of nudges is disputed; corrected for publication bias, d = 0.04 | Mertens et al. 2022; Maier et al. 2022 |
| Libet and Soon show that decisions precede consciousness | the readiness potential is "strikingly absent" in deliberate decisions | Maoz et al. 2019 |
| For Dewey, preferences are constructed through inquiry rather than pre-existing | overstated; Dewey held that practical reasoning needs some prizings not under appraisal | Anderson 2023 (encyclopedia entry) |
| Population priors over taste should be strong, or as weak as possible | it depends on the domain: shared taste is high for natural domains and low for cultural artifacts | Vessel et al. 2018 |
:::

## When optimization is the wrong frame {#sec-phil-limits}

Several sections above point to situations in which treating a choice as an
optimization problem misdescribes it. @tbl-phil-limits gathers them, with what
a system can do instead.

::: {.table #tbl-phil-limits title="Situations in which the optimization frame does not fit, and what a system can do instead (the last column is inference)."}
| Situation | Basis | What a system can do |
|---|---|---|
| Transformative choices, which change the chooser's values | Paul 2014; Pettigrew 2019, 2023 | optimize only instrumental subproblems that remain after the change, or run the process as an exploration the person explicitly accepts as such |
| Identity-constituting choices (career, life plans, self-presentation) | existentialist tradition; Brown 2026 | present well-described options, record the person's own commitment, make no claim to have found the "right" answer |
| Options on a par, where committing is itself the act | Chang 2024 | offer an "on a par" answer, and do not treat such choices as discovering a preference |
| Aesthetic and creative tasks whose value lies in engaged judgment | Nguyen 2020 | act as an assistant to the user's exploration, not as a supplier of answers |
| Decisions with moral weight for others | *Antiqua et nova* 2025 | map options and trade-offs, without outputting a decision |
| Reasonable disagreement among stakeholders | Sorensen et al. 2024; Talmudic tradition of dispute | output a set of well-described options and keep several good regions |
:::

## Settled, contested, missing {#sec-phil-status}

::: {.frontier title="Settled, contested, missing"}
**Settled.** The readiness potential appears before arbitrary decisions but not
before deliberate ones [@maoz2019neural], so the classic argument against a
causal role for conscious deliberation does not extend to considered choices.
People often fail to notice when a chosen option is swapped, and their
preferences shift toward what they believe they chose
[@taya2014manipulation]. Language models optimized on user feedback can learn
to single out and manipulate a small vulnerable minority of users, at least
with simulated users [@williams2025targeted]. Shared taste is higher for
natural images than for cultural artifacts, in the one study that compared
domains directly [@vessel2018stronger].

**Contested.** Whether preferences are mental states (Hausman) or patterns that
models may leave as black boxes (Thoma). Whether options can be on a par
(Chang) or all comparisons hold (Dorr, Nebel, and Zuehl). Whether nudges work
on average once publication bias is corrected. How to define
manipulation: by intent, covertness, or carelessness. Whether a person's
successive selves can be aggregated (Pettigrew against Paul). Which conditions
make system-induced preference change legitimate; Pettigrew's argument
[@pettigrew2023nudging] rules out endorsement after the fact as sufficient on
its own, but no positive set of conditions has consensus.

**Missing.** An empirical test that separates parity from vagueness through
repeated presentation of a pair judged to be on a par. A study of how
meditation affects the consistency or test-retest reliability of pairwise
preferences. A PBO system that elicits meta-preferences at the start, or
audits its own objective for rewarding predictable users.
:::

## Further reading {#further-reading .unnumbered}

- @hansson2022preferences, the encyclopedia entry on preferences, is the best
  map of what philosophers mean by the word and why they disagree.
- @pettigrew2023nudging shows why endorsement after the fact cannot justify a
  change the system caused.
- @chang2024s and @dorr2023case are the two sides of the parity debate in
  short form; read them before deciding what a near-tie means.
- @noggle2026ethics and @carroll2023characterizing together give the most
  usable definitions of manipulation for optimizing systems.
- @fischli2026agents show why "respect autonomy" does not yet tell a system
  what to do, and offer three ways to decide.
- @nguyen2020autonomy is a short, sharp argument for why delivering the best
  design may miss the point of aesthetic work.
