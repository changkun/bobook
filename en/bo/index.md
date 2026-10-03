---
synopsis: "With a surrogate that knows what it does not know, optimization becomes a sequence of decisions: where to evaluate next. This part builds the loop, derives the classic acquisition functions, defines how optimizers are scored, and surveys how the method is used in practice."
---

# Bayesian Optimization {#sec-part-bo}

With a surrogate that knows what it does not know, optimization becomes a sequence of decisions: given everything observed so far, where should the next, expensive evaluation go? This part builds the loop that makes those decisions, derives the classic answers to the question, and asks how to tell whether an optimizer is any good.

Acquisition functions are the heart of the part: probability of improvement, expected improvement, upper confidence bounds, Thompson sampling, knowledge gradient, and entropy search, each a different answer to "what is the next observation worth?". Regret and bandits give the theory a vocabulary. The last two chapters leave the textbook for practice: noise, batches, constraints, many dimensions, software, and the fields where Bayesian optimization has paid off.

*The part assumes Gaussian process regression (@sec-gp-regression).*
