---
synopsis: "When the objective lives in a person's head, the most reliable measurement is often a comparison: this one or that one. This part models comparisons, extends the Gaussian process to learn from them, and builds preferential Bayesian optimization on top, with the theory of dueling bandits behind it."
---

# Learning from Comparisons {#sec-part-preferences}

When the objective lives in a person's head, the most reliable measurement is often a comparison: this one or that one. This part rebuilds Bayesian optimization around that measurement. It starts with why comparisons work and the century-old models that turn a choice into evidence about a hidden utility, then extends the Gaussian process to learn from them, which makes the posterior non-Gaussian and calls for approximate inference.

With a preference model in hand, the part builds preferential Bayesian optimization itself: how to choose the next pair, why the expected utility of the best option is a principled answer, and what a person in the loop changes. It closes with the other questions a system can ask besides "which of two", and with the theory of dueling bandits behind all of it. In the middle of the part, you become the person being optimized.

*The part assumes @sec-part-gp and @sec-part-bo; @sec-comparisons can be read on its own.*
